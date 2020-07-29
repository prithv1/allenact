"""Defines the reinforcement learning `OnPolicyRLEngine`."""
import os
import random
import time
import traceback
import typing
from multiprocessing.context import BaseContext
from typing import Optional, Any, Dict, Union, List, Tuple, Sequence

import torch
import torch.distributions
import torch.multiprocessing as mp
import torch.optim
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler

from onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from onpolicy_sync.storage import RolloutStorage
from onpolicy_sync.vector_sampled_tasks import VectorSampledTasks
from rl_base.experiment_config import ExperimentConfig
from utils.experiment_utils import (
    ScalarMeanTracker,
    LinearDecay,
    set_deterministic_cudnn,
    set_seed,
    Builder,
    TrainingPipeline,
    PipelineStage,
)
from utils.system import get_logger
from utils.tensor_utils import batch_observations


class OnPolicyRLEngine(object):
    """The reinforcement learning primary controller.

    This `OnPolicyRLEngine` class handles all training, validation, and
    testing as well as logging and checkpointing. You are not expected
    to instantiate this class yourself, instead you should define an
    experiment which will then be used to instantiate an
    `OnPolicyRLEngine` and perform any desired tasks.
    """

    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        results_queue: mp.Queue,  # to output aggregated results
        checkpoints_queue: mp.Queue,  # to write/read (trainer/evaluator) ready checkpoints
        checkpoints_dir: str,
        mode: str = "train",
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        device: Union[str, torch.device, int] = "cpu",
        distributed_port: int = 0,
        **kwargs,
    ):
        """Initializer.

        # Parameters

        config : The ExperimentConfig defining the experiment to run.
        output_dir : Root directory at which checkpoints and logs should be saved.
        loaded_config_src_files : Paths to source config files used to create the experiment.
        seed : Seed used to encourage deterministic behavior (it is difficult to ensure
            completely deterministic behavior due to CUDA issues and nondeterminism
            in environments).
        mode : "train", "valid", or "test".
        deterministic_cudnn : Whether or not to use deterministic cudnn. If `True` this may lower
            training performance this is necessary (but not sufficient) if you desire
            deterministic behavior.
        extra_tag : An additional label to add to the experiment when saving tensorboard logs.
        """
        self.config = config
        self.results_queue = results_queue
        self.checkpoints_queue = checkpoints_queue
        self.mp_ctx = mp_ctx
        self.checkpoints_dir = checkpoints_dir
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.device = device
        self.distributed_port = distributed_port

        self.mode = mode.lower()
        assert self.mode in [
            "train",
            "valid",
            "test",
        ], "Only train, valid, test modes supported"

        self.deterministic_cudnn = deterministic_cudnn
        if self.deterministic_cudnn:
            set_deterministic_cudnn()

        self.seed = seed
        set_seed(self.seed)

        self.experiment_name = experiment_name

        self.machine_params = config.machine_params(self.mode)
        if self.num_workers > 1:
            self.all_samplers = self.machine_params["nprocesses"]
            self.num_samplers = self.all_samplers[self.worker_id]
        else:
            if isinstance(self.machine_params["nprocesses"], Sequence):
                self.num_samplers = self.machine_params["nprocesses"][self.worker_id]
            elif isinstance(self.machine_params["nprocesses"], int):
                self.num_samplers = self.machine_params["nprocesses"]
            else:
                raise Exception("Only list and int are valid nprocesses")
        self._vector_tasks: Optional[VectorSampledTasks] = None

        self.observation_set = None
        self.actor_critic = None
        if self.num_samplers > 0:
            if (
                "make_preprocessors_fns" in self.machine_params
                and self.machine_params["make_preprocessors_fns"] is not None
                and len(self.machine_params["make_preprocessors_fns"]) > 0
            ):
                # distributed observation sets, here we just need the observation space
                observation_set = self.machine_params["make_preprocessors_fns"][0]()
                set_seed(self.seed)
                self.actor_critic = typing.cast(
                    torch.nn.Module,
                    self.config.create_model(observation_set=observation_set),
                ).to(self.device)
                del observation_set
            elif (
                "observation_set" in self.machine_params
                and self.machine_params["observation_set"] is not None
            ):
                # centralized observation set,
                self.observation_set = self.machine_params["observation_set"]().to(
                    self.device
                )
                set_seed(self.seed)
                self.actor_critic = typing.cast(
                    torch.nn.Module,
                    self.config.create_model(observation_set=self.observation_set),
                ).to(self.device)
            else:
                # no observation set
                set_seed(self.seed)
                self.actor_critic = typing.cast(
                    torch.nn.Module, self.config.create_model()
                ).to(self.device)

        self.is_distributed = False
        self.store: Optional[torch.distributed.TCPStore] = None
        if self.num_workers > 1:
            if self.mode == "train":
                self.store = torch.distributed.TCPStore(
                    "localhost",
                    self.distributed_port,
                    self.num_workers,
                    self.worker_id == 0,
                )
                torch.distributed.init_process_group(
                    backend="nccl",
                    store=self.store,
                    rank=self.worker_id,
                    world_size=self.num_workers,
                )
                self.actor_critic = DistributedDataParallel(
                    self.actor_critic,
                    device_ids=[self.device],
                    output_device=self.device,
                )
            self.is_distributed = True  # for testing, this only means we need to synchronize after each checkpoint
        self.deterministic_agent = False

        self.step_count: int = 0
        self.total_steps: int = 0

        self.scalars = ScalarMeanTracker()

        set_seed(self.advance_seed(self.seed))

        self._is_closed: bool = False

    def advance_seed(self, seed: Optional[int]) -> Optional[int]:
        if seed is None:
            return seed
        seed = (seed ^ (self.total_steps + self.step_count + 1)) % (
            2 ** 31 - 1
        )  # same seed for all workers
        # TODO: fix for distributed test
        if self.mode == "train":
            return self.worker_seeds(self.num_workers, seed)[
                self.worker_id
            ]  # doesn't modify the current rng state
        else:
            return self.worker_seeds(1, seed)[0]  # doesn't modify the current rng state

    @property
    def vector_tasks(self, debug=False):  # TODO debug
        if self._vector_tasks is None and self.num_samplers > 0:
            if self.is_distributed:
                total_processes = sum(
                    self.all_samplers
                )  # TODO this will break the fixed seed for multi-device test
            else:
                total_processes = self.num_samplers

            seeds = self.worker_seeds(
                total_processes,
                initial_seed=self.seed,  # do not update the RNG state (creation might happen after seed resetting)
            )

            vector_class = VectorSampledTasks
            self._vector_tasks = vector_class(
                make_sampler_fn=self.config.make_sampler_fn,
                sampler_fn_args=self.get_sampler_fn_args(seeds),
                multiprocessing_start_method="forkserver"
                if self.mp_ctx is None
                else None,
                mp_ctx=self.mp_ctx,
            )
        return self._vector_tasks

    @staticmethod
    def worker_seeds(nprocesses: int, initial_seed: Optional[int]) -> List[int]:
        """Create a collection of seeds for workers without modifying the RNG state."""
        if initial_seed is not None:
            rstate = random.getstate()
            random.seed(initial_seed)
        seeds = [random.randint(0, 2 ** (31) - 1) for _ in range(nprocesses)]
        if initial_seed is not None:
            random.setstate(rstate)
        return seeds

    def get_sampler_fn_args(self, seeds: Optional[List[int]] = None):
        devices = (
            self.machine_params["sampler_devices"]
            if "sampler_devices" in self.machine_params
            else self.machine_params["gpu_ids"]
        )

        if self.mode == "train":
            fn = self.config.train_task_sampler_args
        elif self.mode == "valid":
            fn = self.config.valid_task_sampler_args
        elif self.mode == "test":
            fn = self.config.test_task_sampler_args
        else:
            raise NotImplementedError(
                "self.mode must be one of `train`, `valid` or `test`."
            )

        if self.is_distributed:
            total_processes = sum(self.all_samplers)
            process_offset = sum(self.all_samplers[: self.worker_id])
        else:
            total_processes = self.num_samplers
            process_offset = 0

        return [
            fn(
                process_ind=process_offset + it,
                total_processes=total_processes,
                devices=[self.device]
                if self.is_distributed or self.mode == "test"
                else devices,  # TODO is this ever used?!
                seeds=seeds,
            )
            for it in range(self.num_samplers)
        ]

    def checkpoint_load(
        self, ckpt: Union[str, Dict[str, Any]]
    ) -> Dict[str, Union[Dict[str, Any], torch.Tensor, float, int, str, typing.List]]:
        if isinstance(ckpt, str):
            get_logger().info(
                "{} worker {} loading checkpoint from {}".format(
                    self.mode, self.worker_id, ckpt
                )
            )
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt = torch.load(ckpt, map_location="cpu")

        ckpt = typing.cast(
            Dict[
                str, Union[Dict[str, Any], torch.Tensor, float, int, str, typing.List]
            ],
            ckpt,
        )

        target = (
            self.actor_critic
            if not self.mode == "train" or not self.is_distributed
            else self.actor_critic.module
        )
        target.load_state_dict(ckpt["model_state_dict"])

        self.step_count = ckpt["step_count"]  # type: ignore
        self.total_steps = ckpt["total_steps"]  # type: ignore

        return ckpt

    # aggregates task metrics currently in queue
    def aggregate_task_metrics(
        self, num_tasks: int = -1
    ) -> Tuple[Tuple[str, Dict[str, float], int], List[Dict[str, Any]]]:
        assert self.scalars.empty, "found non-empty scalars {}".format(
            self.scalars._counts
        )

        if num_tasks < 0:
            sentinel = ("aggregate.AUTO.sentinel", time.time())
            self.vector_tasks.metrics_out_queue.put(
                sentinel
            )  # valid since a single training/testing process is the only consumer

        task_outputs = []

        done = num_tasks == 0
        while not done:
            item = (
                self.vector_tasks.metrics_out_queue.get()
            )  # at least, there'll be a sentinel
            if isinstance(item, tuple) and item[0] == "aggregate.AUTO.sentinel":
                assert item[1] == sentinel[1], "wrong sentinel found: {} vs {}".format(
                    item[1], sentinel[1]
                )
                done = True
            else:
                task_outputs.append(item)
                if num_tasks > 0:
                    num_tasks -= 1
                done = num_tasks == 0

        # get_logger().debug(
        #     "worker {} got {} tasks".format(self.worker_id, len(task_outputs))
        # )
        #
        # get_logger().debug("worker {} sleeping for 10 s".format(self.worker_id))
        # time.sleep(10)
        # get_logger().debug(
        #     "worker {} empty {}".format(
        #         self.worker_id, self.vector_tasks.metrics_out_queue.empty()
        #     )
        # )

        nsamples = 0
        for task_output in task_outputs:
            if (
                len(task_output) == 0
                or (len(task_output) == 1 and "task_info" in task_output)
                or ("success" in task_output and task_output["success"] is None)
            ):
                continue
            self.scalars.add_scalars(
                {k: v for k, v in task_output.items() if k != "task_info"}
            )
            nsamples += 1

        if nsamples < len(task_outputs):
            get_logger().warning(
                "Discarded {} empty task metrics".format(len(task_outputs) - nsamples)
            )

        pkg_type = "task_metrics_package"
        payload = self.scalars.pop_and_reset() if len(task_outputs) > 0 else None

        return (pkg_type, payload, nsamples), task_outputs

    def _preprocess_observations(self, batched_observations):
        if self.observation_set is None:
            return batched_observations
        return self.observation_set.get_observations(batched_observations)

    def remove_paused(self, observations):
        paused, keep, running = [], [], []
        for it, obs in enumerate(observations):
            if obs is None:
                paused.append(it)
            else:
                keep.append(it)
                running.append(obs)

        for p in reversed(paused):
            self.vector_tasks.pause_at(p)

        batch = batch_observations(running, device=self.device)

        return len(paused), keep, batch

    def initialize_rollouts(self, rollouts, visualizer=None):
        observations = self.vector_tasks.get_observations()
        npaused, keep, batch = self.remove_paused(observations)
        if npaused > 0:
            rollouts.reshape(keep)
        rollouts.to(self.device)
        rollouts.insert_initial_observations(
            self._preprocess_observations(batch) if len(keep) > 0 else batch
        )
        if visualizer is not None and len(keep) > 0:
            visualizer.collect(vector_task=self.vector_tasks, alive=keep)
        return npaused

    def act(self, rollouts: RolloutStorage):
        with torch.no_grad():
            step_observation = rollouts.pick_observation_step(rollouts.step)
            memory = rollouts.pick_memory_step(rollouts.step)
            actor_critic_output, memory = self.actor_critic(
                step_observation,
                memory,
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        actions = (
            actor_critic_output.distributions.sample()
            if not self.deterministic_agent
            else actor_critic_output.distributions.mode()
        )

        return actions, actor_critic_output, memory, step_observation

    @staticmethod
    def _active_memory(memory, keep):
        if isinstance(memory, torch.Tensor):  # rnn hidden state
            return memory[:, keep] if memory.shape[1] > len(keep) else memory
        return memory.index_select(keep)  # arbitrary memory

    def collect_rollout_step(self, rollouts: RolloutStorage, visualizer=None):
        actions, actor_critic_output, memory, _ = self.act(rollouts)

        outputs = self.vector_tasks.step([a[0].item() for a in actions])

        rewards: Union[List, torch.Tensor]
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        rewards = rewards.unsqueeze(1)

        # If done then clean the history of observations.
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float32,
            device=self.device,
        )

        npaused, keep, batch = self.remove_paused(observations)

        if npaused > 0:
            rollouts.reshape(keep)

        rollouts.insert(
            observations=self._preprocess_observations(batch)
            if len(keep) > 0
            else batch,
            memory=self._active_memory(memory, keep),
            actions=actions[keep],
            action_log_probs=actor_critic_output.distributions.log_probs(actions)[keep],
            value_preds=actor_critic_output.values[keep],
            rewards=rewards[keep],
            masks=masks[keep],
        )

        # TODO we always miss tensors for the last action in the last episode of each worker
        if visualizer is not None:
            if len(keep) > 0:
                visualizer.collect(
                    rollout=rollouts,
                    vector_task=self.vector_tasks,
                    alive=keep,
                    actor_critic=actor_critic_output,
                )
            else:
                visualizer.collect(actor_critic=actor_critic_output)

        return npaused

    def close(self, verbose=True):
        if "_is_closed" in self.__dict__ and self._is_closed:
            return

        def logif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    get_logger().info(s)
                elif isinstance(s, Exception):
                    get_logger().exception(traceback.format_exc())
                else:
                    raise NotImplementedError()

        if "_vector_tasks" in self.__dict__ and self._vector_tasks is not None:
            try:
                logif(
                    "{} worker {} Closing OnPolicyRLEngine.vector_tasks.".format(
                        self.mode, self.worker_id
                    )
                )
                self._vector_tasks.close()
                logif("{} worker {} Closed.".format(self.mode, self.worker_id))
            except Exception as e:
                logif(
                    "{} worker {} Exception raised when closing OnPolicyRLEngine.vector_tasks:".format(
                        self.mode, self.worker_id
                    )
                )
                logif(e)
                pass

        self._is_closed = True

    def __del__(self):
        self.close(verbose=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=False)


class OnPolicyTrainer(OnPolicyRLEngine):
    class TrainState:
        def __init__(
            self,
            losses: Optional[Dict[str, AbstractActorCriticLoss]] = None,
            loss_weights: Optional[Dict[str, float]] = None,
            steps_in_rollout: int = -1,
            stage_task_steps: int = -1,
            update_epochs: int = -1,
            update_mini_batches: int = -1,
            gamma: float = 0.99,
            use_gae: bool = True,
            gae_lambda: float = 0.95,
            max_grad_norm: float = 0.5,
            pipeline_stage: int = 0,
            advance_scene_rollout_period: Optional[int] = None,
            teacher_forcing: Optional[LinearDecay] = None,
            total_updates: int = 0,
            rollout_count: int = 0,
            backprop_count: int = 0,
            log_interval: int = 0,
            save_interval: int = 0,
            last_log: int = 0,
            last_save: int = 0,
            tracking_types: Tuple[str, ...] = ("update", "teacher",),
            former_steps: int = 0,
        ):
            self.losses = losses
            self.loss_weights = loss_weights
            self.stage_task_steps = stage_task_steps
            self.steps_in_rollout = steps_in_rollout
            self.update_epochs = update_epochs
            self.update_mini_batches = update_mini_batches
            self.gamma = gamma
            self.use_gae = use_gae
            self.gae_lambda = gae_lambda
            self.max_grad_norm = max_grad_norm
            self.pipeline_stage = pipeline_stage
            self.advance_scene_rollout_period = advance_scene_rollout_period
            self.teacher_forcing = teacher_forcing
            self.total_updates = total_updates
            self.rollout_count = rollout_count
            self.backprop_count = backprop_count
            self.log_interval = log_interval
            self.save_interval = save_interval
            self.last_log = last_log
            self.last_save = last_save
            self.tracking_types = tracking_types
            self.tracking_info = {type: [] for type in self.tracking_types}
            self.former_steps = former_steps

            if self.steps_in_rollout > 0:
                get_logger().info("tstate {}".format(self.__dict__))

    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        results_queue: mp.Queue,
        checkpoints_queue: mp.Queue,
        checkpoints_dir: str = "",
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        device: Union[str, torch.device, int] = "cpu",
        distributed_port: int = 0,
        deterministic_agent: bool = False,
        distributed_preemption_threshold: float = 0.7,
        distributed_barrier: Optional[mp.Barrier] = None,
        **kwargs,
    ):
        kwargs["mode"] = "train"
        super().__init__(
            experiment_name=experiment_name,
            config=config,
            results_queue=results_queue,
            checkpoints_queue=checkpoints_queue,
            checkpoints_dir=checkpoints_dir,
            seed=seed,
            deterministic_cudnn=deterministic_cudnn,
            mp_ctx=mp_ctx,
            worker_id=worker_id,
            num_workers=num_workers,
            device=device,
            distributed_port=distributed_port,
            deterministic_agent=deterministic_agent,
            **kwargs,
        )

        self.actor_critic.train()

        self.training_pipeline: TrainingPipeline = config.training_pipeline()

        self.optimizer: optim.Optimizer = self.training_pipeline.optimizer_builder(
            params=[p for p in self.actor_critic.parameters() if p.requires_grad]
        )

        self.lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if self.training_pipeline.lr_scheduler_builder is not None:
            self.lr_scheduler = self.training_pipeline.lr_scheduler_builder(
                optimizer=self.optimizer
            )

        self.tstate: OnPolicyTrainer.TrainState = OnPolicyTrainer.TrainState(
            save_interval=self.training_pipeline.save_interval,
            log_interval=self.training_pipeline.metric_accumulate_interval,
        )

        self.distributed_barrier = distributed_barrier
        if self.is_distributed:
            # Tracks how many workers have finished their rollout
            self.num_workers_done = torch.distributed.PrefixStore(
                "num_workers_done", self.store
            )
            # Tracks the number of steps taken by each worker in current rollout
            self.num_workers_steps = torch.distributed.PrefixStore(
                "num_workers_steps", self.store
            )
            self.distributed_preemption_threshold = distributed_preemption_threshold
        else:
            self.num_workers_done = None
            self.num_workers_steps = None
            self.distributed_preemption_threshold = 1.0

    def deterministic_seeds(self) -> None:
        if self.seed is not None:
            set_seed(self.advance_seed(self.seed))  # known state for all workers
            seeds = self.worker_seeds(
                self.num_samplers, None
            )  # use latest seed for workers and update rng state
            self.vector_tasks.set_seeds(seeds)

    def checkpoint_save(self) -> str:
        self.deterministic_seeds()

        model_path = os.path.join(
            self.checkpoints_dir,
            "exp_{}__stage_{:02d}__steps_{:012d}.pt".format(
                self.experiment_name,
                self.tstate.pipeline_stage,
                self.total_steps + self.step_count,
            ),
        )

        target = (
            self.actor_critic if not self.is_distributed else self.actor_critic.module
        )

        save_dict = {
            "model_state_dict": target.state_dict(),
            "total_steps": self.total_steps,  # before current stage
            "step_count": self.step_count,  # current stage
            "optimizer_state_dict": self.optimizer.state_dict(),  # type: ignore
            "total_updates": self.tstate.total_updates,  # whole training
            "pipeline_stage": self.tstate.pipeline_stage,
            "rollout_count": self.tstate.rollout_count,  # current stage
            "backprop_count": self.tstate.backprop_count,  # whole training
            "trainer_seed": self.seed,
        }

        if self.lr_scheduler is not None:
            save_dict["scheduler_state"] = typing.cast(
                _LRScheduler, self.lr_scheduler
            ).state_dict()

        torch.save(save_dict, model_path)
        return model_path

    def checkpoint_load(
        self, ckpt: Union[str, Dict[str, Any]], restart: bool = False
    ) -> Dict[str, Union[Dict[str, Any], torch.Tensor, float, int, str, typing.List]]:
        if restart:
            step_count, total_steps = self.step_count, self.total_steps

        ckpt = super().checkpoint_load(ckpt)  # loads model, total_steps, step_count

        if not restart:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore
            self.tstate.total_updates = ckpt["total_updates"]  # type: ignore
            self.tstate.pipeline_stage = ckpt["pipeline_stage"]  # type: ignore
            self.tstate.rollout_count = ckpt["rollout_count"]  # type: ignore
            self.tstate.backprop_count = ckpt["backprop_count"]  # type: ignore
            self.seed = typing.cast(int, ckpt["trainer_seed"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])  # type: ignore
        else:
            self.step_count, self.total_steps = step_count, total_steps

        self.deterministic_seeds()

        return ckpt

    def _get_loss(self, loss_name) -> AbstractActorCriticLoss:
        assert (
            loss_name in self.training_pipeline.named_losses
        ), "undefined referenced loss {}".format(loss_name)
        if isinstance(self.training_pipeline.named_losses[loss_name], Builder):
            return typing.cast(
                Builder[AbstractActorCriticLoss],
                self.training_pipeline.named_losses[loss_name],
            )()
        else:
            return typing.cast(
                AbstractActorCriticLoss, self.training_pipeline.named_losses[loss_name]
            )

    def _load_losses(self, stage: PipelineStage):
        stage_losses: Dict[str, AbstractActorCriticLoss] = {}
        for loss_name in stage.loss_names:
            stage_losses[loss_name] = self._get_loss(loss_name)

        loss_weights_list = (
            stage.loss_weights
            if stage.loss_weights is not None
            else [1.0] * len(stage.loss_names)
        )
        stage_loss_weights = {
            name: weight for name, weight in zip(stage.loss_names, loss_weights_list)
        }

        return stage_losses, stage_loss_weights

    def _stage_value(self, stage: PipelineStage, field: str, allow_none: bool = False):
        if hasattr(stage, field) and getattr(stage, field) is not None:
            return getattr(stage, field)

        if (
            hasattr(self.training_pipeline, field)
            and getattr(self.training_pipeline, field) is not None
        ):
            return getattr(self.training_pipeline, field)

        if field in self.machine_params:
            return self.machine_params[field]

        if allow_none:
            return None
        else:
            raise RuntimeError("missing value for {}".format(field))

    def act(self, rollouts: RolloutStorage):
        actions, actor_critic_output, memory, step_observation = super().act(rollouts)

        if self.is_distributed:
            # TODO this is inaccurate/hacky, but gets synchronized after each rollout
            approx_steps = (
                self.step_count - self.tstate.former_steps
            ) * self.num_workers + self.tstate.former_steps
        else:
            approx_steps = self.step_count  # this is actually accurate

        if self.tstate.teacher_forcing is not None:
            if self.tstate.teacher_forcing(approx_steps) > 0:
                actions, enforce_info = self.apply_teacher_forcing(
                    actions, step_observation, approx_steps
                )
                num_enforced = (
                    enforce_info["teacher_forcing_mask"].sum().item()
                    / actions.nelement()
                )
            else:
                num_enforced = 0

            teacher_force_info = {
                "teacher_ratio/sampled": num_enforced,
                "teacher_ratio/enforced": self.tstate.teacher_forcing(approx_steps),
            }
            self.tstate.tracking_info["teacher"].append(
                ("teacher_package", teacher_force_info, actions.nelement())
            )

        self.step_count += actions.nelement()

        return actions, actor_critic_output, memory, step_observation

    # aggregates info of specific type from TrainState list
    def aggregate_info(self, type: str = "update") -> Tuple[str, Dict[str, float], int]:
        assert (
            type in self.tstate.tracking_types
        ), "Only {} types are accepted for aggregation".format(
            self.tstate.tracking_types
        )

        assert self.scalars.empty, "Found non-empty scalars {}".format(
            self.scalars._counts
        )

        infos = self.tstate.tracking_info[type]
        nsamples = sum(info[2] for info in infos)
        valid_infos = sum(
            info[2] > 0 for info in infos
        )  # used to cancel the averaging in self.scalars

        # assert nsamples != 0, "Attempting to aggregate type {} with 0 samples".format(type)

        for name, payload, nsamps in infos:
            if nsamps > 0:
                self.scalars.add_scalars(
                    {k: valid_infos * payload[k] * nsamps / nsamples for k in payload}
                )

        pkg_type = name
        payload = self.scalars.pop_and_reset() if nsamples > 0 else None

        self.tstate.tracking_info[type] = []  # reset tracking info for current type

        return pkg_type, payload, nsamples

    def update(self, rollouts: RolloutStorage):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        for e in range(self.tstate.update_epochs):
            data_generator = rollouts.recurrent_generator(
                advantages, self.tstate.update_mini_batches
            )

            for bit, batch in enumerate(data_generator):
                # TODO: check recursively within batch
                bsize = None
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        bsize = batch[key].shape[0]
                        break
                assert bsize is not None, "TODO check recursively for batch size"

                actor_critic_output, memory = self.actor_critic(
                    batch["observations"],
                    batch["memory"],
                    batch["prev_actions"],
                    batch["masks"],
                )

                info: Dict[str, float] = {}

                if self.lr_scheduler is not None:
                    info["lr"] = self.optimizer.param_groups[0]["lr"]  # type: ignore

                total_loss: Optional[torch.Tensor] = None
                for loss_name in self.tstate.losses:
                    loss, loss_weight = (
                        self.tstate.losses[loss_name],
                        self.tstate.loss_weights[loss_name],
                    )

                    current_loss, current_info = loss.loss(
                        step_count=self.step_count,
                        batch=batch,
                        actor_critic_output=actor_critic_output,
                    )
                    if total_loss is None:
                        total_loss = loss_weight * current_loss
                    else:
                        total_loss = total_loss + loss_weight * current_loss

                    for key in current_info:
                        info[loss_name + "/" + key] = current_info[key]

                assert total_loss is not None, "No losses specified?"

                info["total_loss"] = total_loss.item()
                self.tstate.tracking_info["update"].append(
                    ("update_package", info, bsize)
                )

                if isinstance(total_loss, torch.Tensor):
                    self.optimizer.zero_grad()  # type: ignore
                    total_loss.backward()  # synchronize
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.tstate.max_grad_norm,  # type: ignore
                    )
                    self.optimizer.step()  # type: ignore
                    self.tstate.backprop_count += 1
                else:
                    get_logger().warning(
                        "{} worker {}"
                        "Total loss ({}) is not a FloatTensor, it is a {}. This can happen when using teacher"
                        "enforcing alone if the expert does not know the optimal action".format(
                            self.mode, self.worker_id, total_loss, type(total_loss)
                        )
                    )
                    if self.is_distributed:
                        # TODO test the hack actually works
                        zero_loss = (
                            (
                                torch.zeros_like(actor_critic_output.distributions)
                                * actor_critic_output.distributions
                            ).sum()
                            + (
                                torch.zeros_like(actor_critic_output.values)
                                * actor_critic_output.values
                            ).sum()
                        )
                        self.optimizer.zero_grad()  # type: ignore
                        zero_loss.backward()  # synchronize
                        nn.utils.clip_grad_norm_(
                            self.actor_critic.parameters(), self.tstate.max_grad_norm,  # type: ignore
                        )
                        self.optimizer.step()  # type: ignore
                        self.tstate.backprop_count += 1

        # # TODO Useful for ensuring correctness of distributed infrastructure
        # target = self.actor_critic.module if self.is_distributed else self.actor_critic
        # state_dict = target.state_dict()
        # keys = sorted(list(state_dict.keys()))
        # get_logger().debug("worker {} param 0 {} param -1 {}".format(
        #     self.worker_id,
        #     state_dict[keys[0]].flatten()[0],
        #     state_dict[keys[-1]].flatten()[-1],
        # ))

    def apply_teacher_forcing(self, actions, step_observation, step_count):
        tf_mask_shape = step_observation["expert_action"].shape[:-1] + (1,)
        expert_actions = step_observation["expert_action"][..., 0:1]
        expert_action_exists_mask = step_observation["expert_action"][..., 1:2]

        teacher_forcing_mask = (
            torch.distributions.bernoulli.Bernoulli(
                torch.tensor(self.tstate.teacher_forcing(step_count))
            )
            .sample(tf_mask_shape)
            .long()
            .to(self.device)
        ) * expert_action_exists_mask

        actions = torch.where(teacher_forcing_mask, expert_actions, actions)

        return (
            actions,
            {"teacher_forcing_mask": teacher_forcing_mask},
        )

    def send_package(self):
        package_type = "train_package"

        task_pkg, task_outputs = self.aggregate_task_metrics()

        payload = (task_pkg,) + tuple(
            self.aggregate_info(type) for type in self.tstate.tracking_types
        )

        nsteps = self.total_steps + self.step_count

        self.results_queue.put((package_type, payload, nsteps))

    def train(self, rollouts: RolloutStorage):
        self.initialize_rollouts(rollouts)

        while self.step_count < self.tstate.stage_task_steps:
            if self.is_distributed:
                self.num_workers_done.set("done", str(0))
                self.num_workers_steps.set("steps", str(0))

                # Ensure all workers are done before incrementing num_workers_{steps, done}
                idx = self.distributed_barrier.wait()  # here we synchronize
                if idx == 0:
                    self.distributed_barrier.reset()

            self.tstate.former_steps = self.step_count
            for step in range(self.tstate.steps_in_rollout):
                self.collect_rollout_step(rollouts)
                if self.is_distributed:
                    # Preempt stragglers
                    num_done = int(self.num_workers_done.get("done"))
                    if (
                        num_done
                        > self.distributed_preemption_threshold * self.num_workers
                        and self.tstate.steps_in_rollout / 4
                        <= step
                        < 0.95 * (self.tstate.steps_in_rollout - 1)
                    ):
                        get_logger().debug(
                            "{} worker {} narrowed rollouts at step {} ({}) with {} done".format(
                                self.mode, self.worker_id, rollouts.step, step, num_done
                            )
                        )
                        rollouts.narrow()
                        break

            with torch.no_grad():
                actor_critic_output, _ = self.actor_critic(
                    rollouts.pick_observation_step(-1),
                    rollouts.pick_memory_step(-1),
                    rollouts.prev_actions[-1],
                    rollouts.masks[-1],
                )

            if self.is_distributed:
                # Mark that a worker is done collecting experience
                self.num_workers_done.add("done", 1)
                self.num_workers_steps.add(
                    "steps", self.step_count - self.tstate.former_steps
                )

                # Ensure all workers are done before updating step counter
                idx = self.distributed_barrier.wait()  # here we synchronize
                if idx == 0:
                    self.distributed_barrier.reset()

                ndone = int(self.num_workers_done.get("done"))
                assert (
                    ndone == self.num_workers
                ), "# workers done {} <> # workers {}".format(ndone, self.num_workers)

                # get the actual step_count
                new_worker_steps = self.step_count - self.tstate.former_steps
                all_new_steps = int(self.num_workers_steps.get("steps"))
                self.step_count += all_new_steps - new_worker_steps
                assert (
                    self.step_count == self.tstate.former_steps + all_new_steps
                ), "num steps {} doesn't match {}".format(
                    self.step_count, self.tstate.former_steps + all_new_steps
                )

            rollouts.compute_returns(
                actor_critic_output.values.detach(),
                self.tstate.use_gae,
                self.tstate.gamma,
                self.tstate.gae_lambda,
            )

            self.update(rollouts)  # here we synchronize

            rollouts.after_update()
            self.tstate.rollout_count += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch=self.step_count + self.total_steps)

            if (
                self.step_count - self.tstate.last_log >= self.tstate.log_interval
                or self.step_count >= self.tstate.stage_task_steps
            ):
                self.send_package()
                self.tstate.last_log = self.step_count

            # save for every interval-th episode or for the last epoch
            if (
                (
                    self.step_count - self.tstate.last_save >= self.tstate.save_interval
                    or self.step_count >= self.tstate.stage_task_steps
                )
                and self.checkpoints_dir != ""
                and self.tstate.save_interval > 0
            ):
                if self.worker_id == 0:
                    model_path = self.checkpoint_save()
                    self.checkpoints_queue.put(("eval", model_path))
                self.tstate.last_save = self.step_count

            if (self.tstate.advance_scene_rollout_period is not None) and (
                self.tstate.rollout_count % self.tstate.advance_scene_rollout_period
                == 0
            ):
                get_logger().info(
                    "{} worker {} Force advance tasks with {} rollouts".format(
                        self.mode, self.worker_id, self.tstate.rollout_count
                    )
                )
                self.vector_tasks.next_task(force_advance_scene=True)
                self.initialize_rollouts(rollouts)

    def run_pipeline(
        self, checkpoint_file_name: Optional[str] = None, restart: bool = False
    ):
        assert (
            self.mode == "train"
        ), "run_pipeline only to be called from a train instance"

        finalized = False
        try:
            if checkpoint_file_name is not None:
                self.checkpoint_load(checkpoint_file_name, restart)

            for stage_num, stage in self.training_pipeline.iterator_starting_at(
                self.tstate.pipeline_stage
            ):
                assert (
                    stage_num == self.tstate.pipeline_stage
                ), "stage_num {} differs from pipeline_stage {}".format(
                    stage_num, self.tstate.pipeline_stage
                )

                stage_losses, stage_weights = self._load_losses(stage)

                self.tstate = OnPolicyTrainer.TrainState(
                    losses=stage_losses,
                    loss_weights=stage_weights,
                    steps_in_rollout=self._stage_value(stage, "num_steps"),
                    stage_task_steps=self._stage_value(stage, "max_stage_steps"),
                    update_epochs=self._stage_value(stage, "update_repeats"),
                    update_mini_batches=self._stage_value(stage, "num_mini_batch"),
                    gamma=self._stage_value(stage, "gamma"),
                    use_gae=self._stage_value(stage, "use_gae"),
                    gae_lambda=self._stage_value(stage, "gae_lambda"),
                    max_grad_norm=self._stage_value(stage, "max_grad_norm"),
                    pipeline_stage=stage_num,
                    advance_scene_rollout_period=self._stage_value(
                        stage, "advance_scene_rollout_period", allow_none=True
                    ),
                    teacher_forcing=stage.teacher_forcing,
                    total_updates=self.tstate.total_updates,
                    rollout_count=0,
                    backprop_count=self.tstate.backprop_count,
                    log_interval=self.tstate.log_interval,
                    save_interval=self.tstate.save_interval,
                    last_log=self.step_count - self.tstate.log_interval,
                    last_save=self.step_count
                    - (
                        self.tstate.log_interval if stage_num == 0 else 0
                    ),  # enforce early val
                    tracking_types=("update",)
                    if stage.teacher_forcing is None
                    else ("update", "teacher"),
                    former_steps=self.step_count,
                )

                self.train(
                    RolloutStorage(
                        self.tstate.steps_in_rollout,
                        self.num_samplers,
                        self.actor_critic
                        if not self.is_distributed
                        else self.actor_critic.module,
                    )
                )

                self.tstate.total_updates += self.tstate.rollout_count
                self.tstate.pipeline_stage += 1

                self.total_steps += self.step_count
                self.step_count = 0
            finalized = True
        except KeyboardInterrupt:
            get_logger().info(
                "KeyboardInterrupt. Terminating {} worker {}".format(
                    self.mode, self.worker_id
                )
            )
        except Exception:
            get_logger().error(
                "Encountered Exception. Terminating {} worker {}".format(
                    self.mode, self.worker_id
                )
            )
            get_logger().exception(traceback.format_exc())
        finally:
            if finalized:
                if self.worker_id == 0:
                    self.results_queue.put(("train_stopped", 0))
                get_logger().info(
                    "{} worker {} COMPLETE".format(self.mode, self.worker_id)
                )
            else:
                self.results_queue.put(("train_stopped", 1 + self.worker_id))
            self.close()


class OnPolicyInference(OnPolicyRLEngine):
    def __init__(
        self,
        config: ExperimentConfig,
        results_queue: mp.Queue,  # to output aggregated results
        checkpoints_queue: mp.Queue,  # to write/read (trainer/evaluator) ready checkpoints
        checkpoints_dir: str = "",
        mode: str = "valid",  # or "test"
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        device: Union[str, torch.device, int] = "cpu",
        deterministic_agent: bool = True,
        worker_id: int = 0,
        num_workers: int = 1,
        distributed_barrier: Optional[mp.Barrier] = None,
        **kwargs,
    ):
        super().__init__(
            experiment_name="",
            config=config,
            results_queue=results_queue,
            checkpoints_queue=checkpoints_queue,
            checkpoints_dir=checkpoints_dir,
            mode=mode,
            seed=seed,
            deterministic_cudnn=deterministic_cudnn,
            mp_ctx=mp_ctx,
            deterministic_agent=deterministic_agent,
            device=device,
            worker_id=worker_id,
            num_workers=num_workers,
            **kwargs,
        )
        if self.actor_critic is not None:
            self.actor_critic.eval()

        # get_logger().debug("{} worker {} using device {}".format(self.mode, self.worker_id, self.device))

        self.deterministic_agent = deterministic_agent

        self.distributed_barrier = distributed_barrier

    def run_eval(
        self,
        checkpoint_file_name: str,
        rollout_steps=100,
        visualizer=None,
        update_secs=20,
    ):
        assert self.actor_critic is not None, "called run_eval with no actor_critic"

        self.checkpoint_load(checkpoint_file_name)

        rollouts = RolloutStorage(rollout_steps, self.num_samplers, self.actor_critic,)

        if visualizer is not None:
            assert visualizer.empty()

        num_paused = self.initialize_rollouts(rollouts, visualizer=visualizer)
        num_tasks = sum(
            self.vector_tasks.command(
                "sampler_attr", ["total_unique"] * (self.num_samplers - num_paused)
            )
        )
        # get_logger().debug(
        #     "worker {} number of tasks {}".format(self.worker_id, num_tasks)
        # )
        steps = 0
        if self.mode == "test":
            lengths = self.vector_tasks.command(
                "sampler_attr", ["length"] * (self.num_samplers - num_paused)
            )
            get_logger().info(
                "worker {}: {} tasks pending ({})".format(
                    self.worker_id, sum(lengths), lengths
                )
            )
            last_time = time.time()
            init_time = last_time
            frames = self.num_samplers - num_paused
        while num_paused < self.num_samplers:
            num_paused += self.collect_rollout_step(rollouts, visualizer=visualizer)
            steps += 1
            if steps % rollout_steps == 0:
                rollouts.after_update()
            if self.mode == "test":
                new_time = time.time()
                if new_time - last_time >= update_secs:
                    lengths = self.vector_tasks.command(
                        "sampler_attr", ["length"] * (self.num_samplers - num_paused)
                    )
                    get_logger().info(
                        "worker {}: {:.1f} fps, {} tasks pending ({})".format(
                            self.worker_id,
                            frames / (new_time - init_time),
                            sum(lengths),
                            lengths,
                        )
                    )
                    last_time = new_time
                frames += self.num_samplers - num_paused

        self.vector_tasks.resume_all()
        self.vector_tasks.set_seeds(self.worker_seeds(self.num_samplers, self.seed))
        self.vector_tasks.reset_all()

        metrics_pkg, task_outputs = self.aggregate_task_metrics(num_tasks)

        pkg_type = "{}_package".format(self.mode)
        viz_package = visualizer.read_and_reset() if visualizer is not None else None
        payload = (metrics_pkg, task_outputs, viz_package, checkpoint_file_name)
        nsteps = self.total_steps + self.step_count

        return pkg_type, payload, nsteps

    def skip_to_latest(self, command, data):
        cond = True
        while cond:
            sentinel = ("skip.AUTO.sentinel", time.time())
            self.checkpoints_queue.put(
                sentinel
            )  # valid since a single valid process is the only consumer
            forwarded = False
            while not forwarded:
                new_command: Optional[str] = None
                new_data: Any = None
                (
                    new_command,
                    new_data,
                ) = self.checkpoints_queue.get()  # block until next command arrives
                if new_command == command:
                    data = new_data
                elif new_command == sentinel[0]:
                    assert (
                        new_data == sentinel[1]
                    ), "wrong sentinel found: {} vs {}".format(new_data, sentinel[1])
                    forwarded = True
                else:
                    raise ValueError(
                        "Unexpected command {} with data {}".format(
                            new_command, new_data
                        )
                    )
            time.sleep(1)
            cond = not self.checkpoints_queue.empty()
        return data

    def barrier(self):
        if self.is_distributed:
            idx = self.distributed_barrier.wait()  # here we synchronize
            if idx == 0:
                self.distributed_barrier.reset()

    def process_checkpoints(self):
        assert (
            self.mode != "train"
        ), "process_checkpoints only to be called from a valid or test instance"

        visualizer = None

        finalized = False
        try:
            while True:
                command: Optional[str] = None
                data: Any = None
                (
                    command,
                    data,
                ) = self.checkpoints_queue.get()  # block until first command arrives
                # get_logger().debug(
                #     "{} {} command {} data {}".format(
                #         self.mode, self.worker_id, command, data
                #     )
                # )

                if command == "eval":
                    if self.num_samplers > 0:
                        if self.mode == "valid":
                            # skip to latest using
                            # 1. there's only consumer in valid
                            # 2. there's no quit/exit/close message issued by runner nor trainer
                            data = self.skip_to_latest(command, data)

                        if (
                            visualizer is None
                            and "visualizer" in self.machine_params
                            and self.machine_params["visualizer"] is not None
                        ):
                            visualizer = self.machine_params[
                                "visualizer"
                            ]()  # builder object

                        eval_package = self.run_eval(
                            checkpoint_file_name=data, visualizer=visualizer
                        )

                        # get_logger().debug(
                        #     "queueing eval_package {} with {} tasks".format(
                        #         self.worker_id, eval_package[1][0][2]
                        #     )
                        # )

                        self.results_queue.put(eval_package)

                        self.barrier()
                    else:
                        self.results_queue.put(
                            ("{}_package".format(self.mode), None, -1)
                        )
                elif command in ["quit", "exit", "close"]:
                    finalized = True
                    break
                else:
                    raise NotImplementedError()
        except KeyboardInterrupt:
            get_logger().info(
                "KeyboardInterrupt. Terminating {} worker {}".format(
                    self.mode, self.worker_id
                )
            )
        except Exception:
            get_logger().error(
                "Encountered Exception. Terminating {} worker {}".format(
                    self.mode, self.worker_id
                )
            )
            get_logger().exception(traceback.format_exc())
        finally:
            if finalized:
                if self.mode == "test":
                    self.results_queue.put(("test_stopped", 0))
                get_logger().info(
                    "{} worker {} complete".format(self.mode, self.worker_id)
                )
            else:
                if self.mode == "test":
                    self.results_queue.put(("test_stopped", self.worker_id + 1))
            self.close(verbose=False)

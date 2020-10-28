"""
ObjectNav RGB-DDPPO Vanilla Experiment Config for Sim2Sim set of experiments
"""
import glob
import os
from abc import ABC
from math import ceil
from typing import Dict, Any, List, Optional, Sequence

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from constants import ABS_PATH_OF_TOP_LEVEL_DIR

from core.base_abstractions.preprocessor import ObservationSet
from core.base_abstractions.sensor import ExpertActionSensor
from core.base_abstractions.task import TaskSampler
from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from core.base_abstractions.experiment_config import ExperimentConfig

from plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler
from plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from plugins.habitat_plugin.habitat_preprocessors import ResnetPreProcessorHabitat
from plugins.robothor_plugin.robothor_tasks import ObjectNavTask


from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)

from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNavRoboThorRGBPPOExperimentConfig(ExperimentConfig):
    """
    Complete ObjectNav RoboThor RGB Navigation Experimental Config
    """

    def __init__(self):
        super().__init__()

        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        self.SCREEN_SIZE = 224
        self.MAX_STEPS = 300  # 500
        self.STEP_SIZE = 0.25
        self.ROTATION_DEGREES = 30.0
        self.VISIBILITY_DISTANCE = 1.0
        self.STOCHASTIC = True
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,
        }

        self.TARGET_TYPES = sorted(
            [
                "AlarmClock",
                "Apple",
                "BaseballBat",
                "BasketBall",
                "Bowl",
                "GarbageCan",
                "HousePlant",
                "Laptop",
                "Mug",
                "SprayBottle",
                "Television",
                "Vase",
            ]
        )
        self.ENV_ARGS = dict(
            width=self.CAMERA_WIDTH,
            height=self.CAMERA_HEIGHT,
            continuousMode=True,
            applyActionNoise=self.STOCHASTIC,
            agentType="stochastic",
            rotateStepDegrees=self.ROTATION_DEGREES,
            visibilityDistance=self.VISIBILITY_DISTANCE,
            gridSize=self.STEP_SIZE,
            snapToGrid=False,
            agentMode="bot",
            include_private_scenes=False,
        )

        self.NUM_PROCESSES = 60
        self.TRAIN_GPU_IDS = list(range(min(torch.cuda.device_count(), 8)))
        self.SAMPLER_GPU_IDS = self.TRAIN_GPU_IDS
        self.VALID_GPU_IDS = (
            [torch.cuda.device_count() - 1] if torch.cuda.is_available() else []
        )
        self.TEST_GPU_IDS = [0]

        self.ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None

        # self.TRAIN_DATASET_DIR = os.path.join(
        #     ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/train"
        # )
        # self.VAL_DATASET_DIR = os.path.join(
        #     ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/val"
        # )

        # Random crop specifications for data augmentations
        self.CROP_WIDTH = 512
        self.CROP_HEIGHT = 384

        # self.SENSORS = [
        #     RGBSensorThor(
        #         height=self.SCREEN_SIZE,
        #         width=self.SCREEN_SIZE,
        #         use_resnet_normalization=True,
        #         uuid="rgb_lowres",
        #     ),
        #     GoalObjectTypeThorSensor(object_types=self.TARGET_TYPES,),
        # ]

        self.PREPROCESSORS = [
            Builder(
                ResnetPreProcessorHabitat,
                {
                    "input_height": self.SCREEN_SIZE,
                    "input_width": self.SCREEN_SIZE,
                    "output_width": 7,
                    "output_height": 7,
                    "output_dims": 512,
                    "pool": False,
                    "torchvision_resnet_model": models.resnet18,
                    "input_uuids": ["rgb_lowres"],
                    "output_uuid": "rgb_resnet",
                    "parallel": False,
                },
            ),
        ]

        self.OBSERVATIONS = [
            "rgb_resnet",
            "goal_object_type_ind",
        ]

    @classmethod
    def tag(cls):
        return "Objectnav-S2S-RoboTHOR-RGB-ResNetGRU-DDPPO"

    # @classmethod
    def training_pipeline(self, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 5000000
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def monkey_patch_sensor(
        self, corruptions=None, severities=None, random_crop=False, color_jitter=False
    ):
        self.SENSORS = [
            RGBSensorThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb_lowres",
                corruptions=corruptions,
                severities=severities,
                random_crop=random_crop,
                crop_height=self.CROP_HEIGHT,
                crop_width=self.CROP_WIDTH,
                color_jitter=color_jitter,
            ),
            GoalObjectTypeThorSensor(object_types=self.TARGET_TYPES,),
        ]
        self.CORRUPTIONS = corruptions
        self.SEVERITIES = severities

    def monkey_patch_datasets(self, train_dataset, val_dataset, test_dataset):
        if train_dataset is not None:
            self.TRAIN_DATASET_DIR = train_dataset
        else:
            self.TRAIN_DATASET_DIR = os.path.join(
                ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/train"
            )

        if val_dataset is not None:
            self.VAL_DATASET_DIR = val_dataset
        else:
            self.VAL_DATASET_DIR = os.path.join(
                ABS_PATH_OF_TOP_LEVEL_DIR,
                "datasets/robothor-objectnav/eval_val_0.4_per_sc",
            )

        if test_dataset is not None:
            self.TEST_DATASET_DIR = test_dataset
        else:
            self.TEST_DATASET_DIR = os.path.join(
                ABS_PATH_OF_TOP_LEVEL_DIR,
                "datasets/robothor-objectnav/eval_val_0.4_per_sc",
            )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            rgb_resnet_preprocessor_uuid="rgb_resnet",
            hidden_size=512,
            goal_dims=32,
        )

    @staticmethod
    def split_num_processes(nprocesses: int, ndevices: int):
        assert nprocesses >= ndevices, "NUM_PROCESSES {} < ndevices {}".format(
            nprocesses, ndevices
        )
        res = [0] * ndevices
        for it in range(nprocesses):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else self.TRAIN_GPU_IDS * workers_per_device
            )
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else self.split_num_processes(self.NUM_PROCESSES, ndevices=len(gpu_ids))
            )
            sampler_devices = self.SAMPLER_GPU_IDS
        elif mode == "valid":
            nprocesses = 15 if torch.cuda.is_available() else 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALID_GPU_IDS
        elif mode == "test":
            nprocesses = 15 if torch.cuda.is_available() else 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TEST_GPU_IDS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensors = [*self.SENSORS]
        observations = [*self.OBSERVATIONS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]
            observations = [o for o in observations if "expert_action" not in o]

        # Disable parallelization for validation process
        if mode == "valid":
            for prep in self.PREPROCESSORS:
                prep.kwargs["parallel"] = False

        observation_set = (
            Builder(
                ObservationSet,
                kwargs=dict(
                    source_ids=observations,
                    all_preprocessors=self.PREPROCESSORS,
                    all_sensors=sensors,
                ),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "sampler_devices": sampler_devices
            if mode == "train"
            else gpu_ids,  # ignored with > 1 gpu_ids
            "observation_set": observation_set,
        }

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavDatasetTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes_dir: str,
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        include_expert_sensor: bool = True,
    ) -> Dict[str, Any]:
        path = os.path.join(scenes_dir, "*.json.gz")
        scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
        if len(scenes) == 0:
            raise RuntimeError(
                (
                    "Could find no scene dataset information in directory {}."
                    " Are you sure you've downloaded them? "
                    " If not, see https://allenact.org/installation/download-datasets/ information"
                    " on how this can be done."
                ).format(scenes_dir)
            )
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": [
                s
                for s in self.SENSORS
                if (include_expert_sensor or not isinstance(s, ExpertActionSensor))
            ],
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.TRAIN_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = self.TRAIN_DATASET_DIR
        res["loop_dataset"] = True
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        res["allow_flipping"] = True
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            include_expert_sensor=False,
        )
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = False
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.TEST_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            include_expert_sensor=False,
        )
        res["scene_directory"] = self.TEST_DATASET_DIR
        res["loop_dataset"] = False
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res

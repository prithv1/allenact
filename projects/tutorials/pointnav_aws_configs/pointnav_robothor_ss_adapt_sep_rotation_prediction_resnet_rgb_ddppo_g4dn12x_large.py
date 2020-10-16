"""
A general pointnav robothor experiment config that can support multiple
sensor variations, etc. for pre-training
"""
import glob
import os
from math import ceil
from typing import Dict, Any, List, Optional

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses import RotationPred
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from core.algorithms.onpolicy_sync.losses.rotation_pred import RotPredConfig
from projects.pointnav_baselines.models.point_nav_models import (
    ResnetTensorPointNavActorCritic,
    ResnetModelTensorPointNavActorCritic,
)

from core.base_abstractions.experiment_config import ExperimentConfig
from core.base_abstractions.preprocessor import ObservationSet
from core.base_abstractions.task import TaskSampler
from core.base_abstractions.sensor import RotationSensor, SeparateRotatedVisionSensor
from plugins.habitat_plugin.habitat_preprocessors import (
    ResnetPreProcessorHabitat,
    IdentityPreProcessorHabitat,
)
from plugins.robothor_plugin.robothor_sensors import (
    GPSCompassSensorRoboThor,
    RGBSensorRoboThor,
)
from plugins.robothor_plugin.robothor_task_samplers import PointNavDatasetTaskSampler
from plugins.robothor_plugin.robothor_tasks import PointNavTask
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class PointNavRoboThorRGBPPOExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuration in RoboThor."""

    # Task Parameters
    MAX_STEPS = 500
    REWARD_CONFIG = {
        "step_penalty": -0.01,
        "goal_success_reward": 10.0,
        "failed_stop_reward": 0.0,
        "shaping_weight": 1.0,
    }

    # Simulator Parameters
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    SCREEN_SIZE = 224

    # Random crop specifications for data augmentations
    CROP_WIDTH = 512
    CROP_HEIGHT = 384

    # Training Engine Parameters
    ADVANCE_SCENE_ROLLOUT_PERIOD = 10 ** 13
    NUM_PROCESSES = 15
    TRAINING_GPUS = [0, 1, 2]
    VALIDATION_GPUS = [3]
    TESTING_GPUS = [3]
    # NUM_PROCESSES = 1
    # TRAINING_GPUS = [0]
    # VALIDATION_GPUS = [1]
    # TESTING_GPUS = [1]

    # # Dataset Parameters
    # TRAIN_DATASET_DIR = os.path.join(
    #     ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-pointnav/debug"
    # )
    # VAL_DATASET_DIR = os.path.join(
    #     ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-pointnav/debug"
    # )

    PREPROCESSORS = [
        Builder(
            IdentityPreProcessorHabitat,
            {
                "input_height": SCREEN_SIZE,
                "input_width": SCREEN_SIZE,
                "output_width": 7,
                "output_height": 7,
                "output_dims": 512,
                "input_uuids": ["rgb_lowres"],
                "output_uuid": "rgb_resnet",
                "parallel": False,
            },
        ),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "target_coordinates_ind",
        "rot_label",
        "sep_rot_rgb",
    ]

    ENV_ARGS = dict(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        rotateStepDegrees=30.0,
        visibilityDistance=1.0,
        gridSize=0.25,
    )

    @classmethod
    def tag(cls):
        return "PointNavRobothorResNetRGBPPOSeparateRotationPredictionAdapt"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(250000000)
        lr = 3e-4
        # lr = 2.8197e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        # save_interval = 5000000
        save_interval = 5000
        log_interval = 1000
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
            named_losses={
                "ppo_loss": Builder(PPO, kwargs={}, default=PPOConfig,),
                "rotation_pred_loss": Builder(
                    RotationPred, kwargs={}, default=RotPredConfig
                ),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss", "rotation_pred_loss"],
                    max_stage_steps=ppo_steps,
                    loss_weights=[0.0, 0.01],
                )
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def monkey_patch_sensor(
        self,
        corruptions,
        severities,
        random_crop,
        color_jitter,
        rotate=True,
        rotate_prob=0.7,
    ):
        self.SENSORS = [
            RGBSensorRoboThor(
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
                sep_rotate=rotate,
                # rotate_prob=rotate_prob,
            ),
            GPSCompassSensorRoboThor(),
            RotationSensor(uuid="rot_label"),
            SeparateRotatedVisionSensor(
                height=self.SCREEN_SIZE, width=self.SCREEN_SIZE, uuid="sep_rot_rgb",
            ),
        ]
        self.CORRUPTIONS = corruptions
        self.SEVERITIES = severities

    def monkey_patch_datasets(self, train_dataset, val_dataset, test_dataset):
        if train_dataset is not None:
            self.TRAIN_DATASET_DIR = train_dataset
        else:
            self.TRAIN_DATASET_DIR = os.path.join(
                ABS_PATH_OF_TOP_LEVEL_DIR,
                "datasets/robothor-pointnav/minival_60_per_sc",
            )

        if val_dataset is not None:
            self.VAL_DATASET_DIR = val_dataset
        else:
            self.VAL_DATASET_DIR = os.path.join(
                ABS_PATH_OF_TOP_LEVEL_DIR,
                "datasets/robothor-pointnav/minival_60_per_sc",
            )

        if test_dataset is not None:
            self.TEST_DATASET_DIR = test_dataset
        else:
            self.TEST_DATASET_DIR = os.path.join(
                ABS_PATH_OF_TOP_LEVEL_DIR,
                "datasets/robothor-pointnav/minival_60_per_sc",
            )

    def split_num_processes(self, ndevices):
        assert self.NUM_PROCESSES >= ndevices, "NUM_PROCESSES {} < ndevices {}".format(
            self.NUM_PROCESSES, ndevices
        )
        res = [0] * ndevices
        for it in range(self.NUM_PROCESSES):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else self.TRAINING_GPUS * workers_per_device
            )
            nprocesses = (
                8
                if not torch.cuda.is_available()
                else self.split_num_processes(len(gpu_ids))
            )
            sampler_devices = self.TRAINING_GPUS
            render_video = False
        elif mode == "valid":
            nprocesses = 15
            gpu_ids = [] if not torch.cuda.is_available() else self.VALIDATION_GPUS
            render_video = False
        elif mode == "test":
            nprocesses = 15
            gpu_ids = [] if not torch.cuda.is_available() else self.TESTING_GPUS
            render_video = False
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Disable parallelization for validation process
        if mode == "valid":
            for prep in self.PREPROCESSORS:
                prep.kwargs["parallel"] = False

        observation_set = (
            Builder(
                ObservationSet,
                kwargs=dict(
                    source_ids=self.OBSERVATIONS,
                    all_preprocessors=self.PREPROCESSORS,
                    all_sensors=self.SENSORS,
                ),
            )
            if mode == "train" or mode == "test" or nprocesses > 0
            else None
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "sampler_devices": sampler_devices if mode == "train" else gpu_ids,
            "observation_set": observation_set,
            "render_video": render_video,
        }

    # Define Model
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetModelTensorPointNavActorCritic(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            rgb_resnet_preprocessor_uuid="rgb_resnet",
            rot_rgb_resnet_preprocessor_uuid="sep_rot_rgb",
            hidden_size=512,
            goal_dims=32,
            aux_mode=True,
            sep_rot_mode=True,
            rot_train_mode=False,
            resnet_embed_grad_mode=False,
            visual_encoder_grad_mode=True,
            recurrent_controller_grad_mode=True,
        )

    # Define Task Sampler
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavDatasetTaskSampler(**kwargs)

    # Utility Functions for distributing scenes between GPUs
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
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
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

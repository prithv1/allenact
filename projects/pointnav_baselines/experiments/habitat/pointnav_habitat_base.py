import os
from abc import ABC
from typing import Dict, Any, List, Optional

import gym
import torch

from core.base_abstractions.experiment_config import MachineParams
from core.base_abstractions.preprocessor import ObservationSet
from core.base_abstractions.task import TaskSampler
from plugins.habitat_plugin.habitat_constants import (
    HABITAT_DATASETS_DIR,
    HABITAT_CONFIGS_DIR,
    HABITAT_SCENE_DATASETS_DIR,
)
from plugins.habitat_plugin.habitat_task_samplers import PointNavTaskSampler
from plugins.habitat_plugin.habitat_tasks import PointNavTask
from plugins.habitat_plugin.habitat_utils import get_habitat_config
from projects.pointnav_baselines.experiments.pointnav_base import PointNavBaseConfig
from utils.experiment_utils import Builder


class PointNavHabitatBaseConfig(PointNavBaseConfig, ABC):
    """The base config for all Habitat PointNav experiments."""

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    NUM_STEPS = 128
    LR = 2.5e-4
    FAILED_END_REWARD = -1.0

    def __init__(self):
        super().__init__()
        self.CAMERA_WIDTH = 300
        self.CAMERA_HEIGHT = 225

        task_data_dir_template = os.path.join(
            HABITAT_DATASETS_DIR, "pointnav/gibson/v1/{}/{}.json.gz"
        )
        self.TRAIN_SCENES = task_data_dir_template.format(*(["train"] * 2))
        self.VALID_SCENES = task_data_dir_template.format(*(["val"] * 2))
        self.TEST_SCENES = task_data_dir_template.format(*(["test"] * 2))

        self.NUM_PROCESSES = max(5 * torch.cuda.device_count() - 1, 4)
        self.CONFIG = get_habitat_config(
            os.path.join(HABITAT_CONFIGS_DIR, "tasks/pointnav_gibson.yaml")
        )
        self.CONFIG.defrost()
        self.CONFIG.NUM_PROCESSES = self.NUM_PROCESSES
        self.CONFIG.SIMULATOR_GPU_IDS = list(range(torch.cuda.device_count()))
        self.CONFIG.DATASET.SCENES_DIR = HABITAT_SCENE_DATASETS_DIR

        self.CONFIG.DATASET.DATA_PATH = self.TRAIN_SCENES
        self.CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
        self.CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = self.CAMERA_WIDTH
        self.CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = self.CAMERA_HEIGHT
        self.CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = self.CAMERA_WIDTH
        self.CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = self.CAMERA_HEIGHT
        self.CONFIG.SIMULATOR.TURN_ANGLE = self.ROTATION_DEGREES
        self.CONFIG.SIMULATOR.FORWARD_STEP_SIZE = self.STEP_SIZE
        self.CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = self.MAX_STEPS

        self.CONFIG.TASK.TYPE = "Nav-v0"
        self.CONFIG.TASK.SUCCESS_DISTANCE = self.DISTANCE_TO_GOAL
        self.CONFIG.TASK.SENSORS = ["POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
        self.CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
        self.CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
        self.CONFIG.TASK.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
        self.CONFIG.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL"]
        self.CONFIG.TASK.SPL.TYPE = "SPL"
        self.CONFIG.TASK.SPL.SUCCESS_DISTANCE = self.DISTANCE_TO_GOAL
        self.CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE = self.DISTANCE_TO_GOAL

        self.CONFIG.MODE = "train"

        self.TRAINING_GPUS = list(range(torch.cuda.device_count()))
        self.VALIDATION_GPUS = [torch.cuda.device_count() - 1]
        self.TESTING_GPUS = [torch.cuda.device_count() - 1]

        self.TRAIN_CONFIGS = None
        self.TEST_CONFIGS = None
        self.SENSORS = None

    @classmethod
    def tag(cls):
        return "PointNav"

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
                1
                if not torch.cuda.is_available()
                else self.split_num_processes(len(gpu_ids))
            )
        elif mode == "valid":
            nprocesses = 1
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = self.VALIDATION_GPUS
        elif mode == "test":
            nprocesses = 1
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = self.TESTING_GPUS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        observation_set = (
            Builder(
                ObservationSet,
                kwargs=dict(
                    source_ids=self.OBSERVATIONS,
                    all_preprocessors=self.PREPROCESSORS,
                    all_sensors=self.SENSORS,
                ),
            )
            if mode == "train" or nprocesses > 0
            else None
        )

        return MachineParams(
            nprocesses=nprocesses, devices=gpu_ids, observation_set=observation_set,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(
            **{"failed_end_reward": cls.FAILED_END_REWARD, **kwargs} # type: ignore
        )

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TRAIN_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if total_processes != 1:
            raise NotImplementedError(
                "In validation, `total_processes` must equal 1 for habitat tasks"
            )
        config = self.CONFIG.clone()
        config.defrost()
        config.DATASET.DATA_PATH = self.VALID_SCENES
        config.MODE = "validate"
        config.freeze()
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TEST_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

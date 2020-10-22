from typing import Optional, List, Any, Dict

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from core.algorithms.onpolicy_sync.losses.imitation import Imitation
from core.base_abstractions.sensor import ExpertActionSensor
from plugins.habitat_plugin.habitat_preprocessors import ResnetPreProcessorHabitat
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNaviThorRGBDAggerExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG["shaping_weight"] = 0

        self.SENSORS = [
            RGBSensorThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb_lowres",
            ),
            GoalObjectTypeThorSensor(object_types=self.TARGET_TYPES,),
            ExpertActionSensor(nactions=len(ObjectNavTask.class_action_names()),),
        ]

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
            "expert_action",
        ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNetGRU-DAgger"

    def training_pipeline(self, **kwargs):
        training_steps = int(300000000)
        tf_steps = int(5e6)
        anneal_steps = int(5e6)
        il_no_tf_steps = training_steps - tf_steps - anneal_steps
        assert il_no_tf_steps > 0

        lr = 3e-4
        num_mini_batch = 2 if torch.cuda.is_available() else 1
        update_repeats = 4
        num_steps = 30
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
            named_losses={"imitation_loss": Imitation(),},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=tf_steps,
                    teacher_forcing=LinearDecay(startp=1.0, endp=1.0, steps=tf_steps,),
                ),
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=anneal_steps + il_no_tf_steps,
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=anneal_steps,
                    ),
                ),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=training_steps)},
            ),
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

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        kwargs = super(
            ObjectNaviThorRGBDAggerExperimentConfig, self
        ).test_task_sampler_args(
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        kwargs["rewards_config"]["shaping_weight"] = 0

        return kwargs

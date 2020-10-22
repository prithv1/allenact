import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from plugins.habitat_plugin.habitat_sensors import (
    DepthSensorHabitat,
    TargetCoordinatesSensorHabitat,
)
from plugins.habitat_plugin.habitat_tasks import PointNavTask
from plugins.habitat_plugin.habitat_utils import construct_env_configs
from projects.pointnav_baselines.experiments.habitat.pointnav_habitat_base import (
    PointNavHabitatBaseConfig,
)
from projects.pointnav_baselines.models.point_nav_models import (
    PointNavActorCriticSimpleConvRNN,
)
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class PointNavHabitatDepthDeterministiSimpleConvGRUDDPPOExperimentConfig(
    PointNavHabitatBaseConfig
):
    """An Point Navigation experiment configuration in Habitat with Depth
    input."""

    def __init__(self):
        super().__init__()

        self.SENSORS = [
            DepthSensorHabitat(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_normalization=True,
            ),
            TargetCoordinatesSensorHabitat(coordinate_dims=2),
        ]

        self.PREPROCESSORS = []

        self.OBSERVATIONS = [
            "depth",
            "target_coordinates_ind",
        ]

        self.CONFIG = self.CONFIG.clone()
        self.CONFIG.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR"]

        self.TRAIN_CONFIGS = construct_env_configs(config=self.CONFIG)

    @classmethod
    def tag(cls):
        return "Pointnav-Habitat-Depth-SimpleConv-DDPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(75000000)
        lr = cls.LR
        num_mini_batch = 1
        update_repeats = 3
        num_steps = cls.NUM_STEPS
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
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
        )

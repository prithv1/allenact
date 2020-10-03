"""
-- Add temporal distance as an auxiliary task
-- Add CPC|A as an auxiliary task
"""
import typing
from typing import Tuple, Dict, Optional, Union, List

import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict

from core.models.basic_models import SimpleCNN, RNNStateEncoder
from core.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from core.base_abstractions.misc import ActorCriticOutput, Memory
from core.base_abstractions.distributions import CategoricalDistr

import math
import copy
from torch.distributions import Bernoulli


class PointNavActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        embed_coordinates=False,
        coordinate_embedding_dim=8,
        coordinate_dims=2,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class ResnetTensorPointNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        rgb_resnet_preprocessor_uuid: Optional[str] = None,
        depth_resnet_preprocessor_uuid: Optional[str] = None,
        hidden_size: int = 512,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        aux_mode: bool = False,
        inv_mode: bool = False,
        inv_visual_mode: bool = False,
        inv_belief_mode: bool = False,
        fwd_mode: bool = False,
        fwd_visual_mode: bool = False,
        fwd_belief_mode: bool = False,
        rot_mode: bool = False,
        td_mode: bool = False,
        feat_random_mode: bool = False,
        atc_mode: bool = False,
    ):
        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        # Intrinsic Curiousity Module Arguments
        self.aux_mode = aux_mode
        self.inv_mode = inv_mode
        self.inv_vis = inv_visual_mode
        self.inv_bel = inv_belief_mode
        self.fwd_mode = fwd_mode
        self.fwd_vis = fwd_visual_mode
        self.fwd_bel = fwd_belief_mode
        self.rot_mode = rot_mode
        self.td_mode = td_mode
        self.feat_random_mode = feat_random_mode
        self.atc_mode = atc_mode
        self.rgb_resnet_preprocessor_uuid = rgb_resnet_preprocessor_uuid
        self.observation_space = observation_space
        self._hidden_size = hidden_size
        if (
            rgb_resnet_preprocessor_uuid is None
            or depth_resnet_preprocessor_uuid is None
        ):
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid
                if rgb_resnet_preprocessor_uuid is not None
                else depth_resnet_preprocessor_uuid
            )
            self.goal_visual_encoder = ResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        else:
            self.goal_visual_encoder = ResnetDualTensorGoalEncoder(  # type:ignore
                self.observation_space,
                goal_sensor_uuid,
                rgb_resnet_preprocessor_uuid,
                depth_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        self.state_encoder = RNNStateEncoder(
            self.goal_visual_encoder.output_dims, self._hidden_size,
        )
        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.inv_mode and self.aux_mode:
            if self.inv_vis:
                self.inverse_model = InverseModel(
                    action_space,
                    2 * self.goal_visual_encoder.output_dims,
                    self.inv_vis,
                    self.inv_bel,
                )
            elif self.inv_bel:
                self.inverse_model = InverseModel(
                    action_space,
                    2 * self.goal_visual_encoder.output_dims + self._hidden_size,
                    self.inv_vis,
                    self.inv_bel,
                )

        if self.fwd_mode and self.aux_mode:
            if self.fwd_vis:
                self.forward_model = ForwardModel(
                    action_space,
                    10,
                    self.goal_visual_encoder.output_dims,
                    self.goal_visual_encoder.output_dims,
                    self.fwd_vis,
                    self.fwd_bel,
                )
            elif self.fwd_bel:
                self.forward_model = ForwardModel(
                    action_space,
                    10,
                    self.goal_visual_encoder.output_dims + self._hidden_size,
                    self.goal_visual_encoder.output_dims,
                    self.fwd_vis,
                    self.fwd_bel,
                )

        if self.rot_mode and self.aux_mode:
            self.rotation_model = RotationModel(self.goal_visual_encoder.output_dims)

        if self.td_mode and self.aux_mode:
            self.temporal_model = TemporalDistancePredictor(
                2 * self.goal_visual_encoder.output_dims + self._hidden_size,
            )

        if self.feat_random_mode and self.aux_mode:
            self.feat_randomizer = nn.Sequential(
                nn.Conv2d(
                    observation_space.spaces[rgb_resnet_preprocessor_uuid].shape[0],
                    observation_space.spaces[rgb_resnet_preprocessor_uuid].shape[0],
                    3,
                    padding=1,
                ),
                # nn.Conv2d(
                #     observation_space.spaces[rgb_resnet_preprocessor_uuid].shape[0],
                #     observation_space.spaces[rgb_resnet_preprocessor_uuid].shape[0],
                #     1,
                # ),
                # nn.Dropout2d(0.2),
                # nn.Conv2d(
                #     observation_space.spaces[rgb_resnet_preprocessor_uuid].shape[0],
                #     observation_space.spaces[rgb_resnet_preprocessor_uuid].shape[0],
                #     1,
                # ),
            )
            self.identity = nn.Sequential(nn.Identity())
            self.random_prob = (
                0.9  # Feed clean observations with prob. 0.7; perturbed with prob 0.3
            )
            for param in self.feat_randomizer.parameters():
                param.requires_grad = False

        # if self.atc_mode and self.aux_mode:
        #     if (
        #         rgb_resnet_preprocessor_uuid is None
        #         or depth_resnet_preprocessor_uuid is None
        #     ):
        #         resnet_preprocessor_uuid = (
        #             rgb_resnet_preprocessor_uuid
        #             if rgb_resnet_preprocessor_uuid is not None
        #             else depth_resnet_preprocessor_uuid
        #         )
        #         self.momentum_goal_visual_encoder = ResnetTensorGoalEncoder(
        #             self.observation_space,
        #             goal_sensor_uuid,
        #             resnet_preprocessor_uuid,
        #             goal_dims,
        #             resnet_compressor_hidden_out_dims,
        #             combiner_hidden_out_dims,
        #         )
        #     else:
        #         self.momentum_goal_visual_encoder = ResnetDualTensorGoalEncoder(  # type:ignore
        #             self.observation_space,
        #             goal_sensor_uuid,
        #             rgb_resnet_preprocessor_uuid,
        #             depth_resnet_preprocessor_uuid,
        #             goal_dims,
        #             resnet_compressor_hidden_out_dims,
        #             combiner_hidden_out_dims,
        #         )

        #     self.code_size = 128
        #     self.momentum = 0.01
        #     self.ATCModel = AugmentedTemporalContrastModel(
        #         self.goal_visual_encoder,
        #         self.momentum_goal_visual_encoder,
        #         self.goal_visual_encoder.output_dims,
        #         self.code_size,
        #         self.momentum,
        #     )

        self.train()
        self.memory_key = "rnn"

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        }

    # def forward(  # type:ignore
    #     self,
    #     observations: ObservationType,
    #     memory: Memory,
    #     prev_actions: torch.Tensor,
    #     masks: torch.FloatTensor,
    #     **kwargs,
    # ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
    #     x = self.goal_visual_encoder(observations)
    #     x, rnn_hidden_states = self.state_encoder(
    #         x, memory.tensor(self.memory_key), masks
    #     )
    #     return (
    #         ActorCriticOutput(
    #             distributions=self.actor(x), values=self.critic(x), extras={}
    #         ),
    #         memory.set_tensor(self.memory_key, rnn_hidden_states),
    #     )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
        actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:

        if self.aux_mode:
            orig_observations = copy.deepcopy(observations)

        obs_encoding = self.goal_visual_encoder(observations)
        obs_encoding, rnn_hidden_states = self.state_encoder(
            obs_encoding, memory.tensor(self.memory_key), masks
        )

        # Action-Prediction
        action_logits = None
        rotation_logits = None
        next_state = None
        next_state_pred = None
        td_preds = None
        td_pair_target = None
        curr_obs_encoding = obs_encoding
        if self.inv_vis:
            obs_input = self.goal_visual_encoder.visual_compressor(orig_observations)
            action_logits = self.inverse_model(obs_input)
            # action_logits = self.inverse_model(curr_obs_encoding) # Bug -- Avoid this

        if self.fwd_vis:
            if actions is not None:
                next_state_pred, next_state = self.forward_model(
                    curr_obs_encoding, actions
                )

        if self.rot_mode:
            obs_input = self.goal_visual_encoder.visual_compressor(orig_observations)
            rotation_logits = self.rotation_model(obs_input)
            # rotation_logits = self.rotation_model(curr_obs_encoding)

        # fwd_rnn_hidden_states = None
        if self.feat_random_mode:
            # Bug here
            # Use the randomized observations to collect rollout and update
            # based on advantage estimates
            # Add the feature matching loss to match things to the clean
            # observation

            probs = torch.tensor([self.random_prob]).repeat(curr_obs_encoding.shape[1])
            randomizer_dist = Bernoulli(probs)
            mask = randomizer_dist.sample()
            mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            mask = mask.repeat(
                orig_observations[self.rgb_resnet_preprocessor_uuid].shape[0],
                1,
                orig_observations[self.rgb_resnet_preprocessor_uuid].shape[2],
                orig_observations[self.rgb_resnet_preprocessor_uuid].shape[3],
                orig_observations[self.rgb_resnet_preprocessor_uuid].shape[4],
            )
            mask = mask.to(curr_obs_encoding.device)
            std = math.sqrt(
                1
                / (
                    self.observation_space.spaces[
                        self.rgb_resnet_preprocessor_uuid
                    ].shape[0]
                )
            )
            for m in self.feat_randomizer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal(m.weight, mean=0, std=std)
                    nn.init.constant(m.bias, 0)

            rgb_observations = orig_observations[self.rgb_resnet_preprocessor_uuid]
            nstep, nsampler = rgb_observations.shape[:2]
            feat_dims = rgb_observations.shape[-3:]

            rgb_observations = mask * rgb_observations + (
                1 - mask
            ) * self.feat_randomizer(rgb_observations.view(-1, *feat_dims)).view(
                nstep, nsampler, *feat_dims
            )

            orig_observations[self.rgb_resnet_preprocessor_uuid] = rgb_observations

            fwd_obs_encoding = self.goal_visual_encoder(orig_observations)
            fwd_obs_encoding, fwd_rnn_hidden_states = self.state_encoder(
                fwd_obs_encoding, memory.tensor(self.memory_key), masks
            )
        else:
            fwd_obs_encoding = obs_encoding
            fwd_rnn_hidden_states = rnn_hidden_states

        if self.inv_bel:
            obs_input = self.goal_visual_encoder.visual_compressor(orig_observations)
            action_logits = self.inverse_model(obs_input, rnn_hidden_states)
            # action_logits = self.inverse_model(curr_obs_encoding, rnn_hidden_states)  # Bug -- Avoid this

        if self.fwd_bel:
            if actions is not None:
                next_state_pred, next_state = self.forward_model(
                    curr_obs_encoding, actions, rnn_hidden_states
                )

        if self.td_mode:
            obs_input = self.goal_visual_encoder.visual_compressor(orig_observations)
            td_preds, td_pair_target = self.temporal_model(
                obs_input, rnn_hidden_states,
            )
            # td_preds, td_pair_target = self.temporal_model(
            #     curr_obs_encoding, rnn_hidden_states,
            # )

        # return (
        #     ActorCriticOutput(
        #         distributions=self.actor(obs_encoding),
        #         values=self.critic(obs_encoding),
        #         extras={
        #             "pred_action_logits": action_logits,
        #             "next_state": next_state,
        #             "next_state_pred": next_state_pred,
        #             "td_preds": td_preds,
        #             "td_pair_target": td_pair_target,
        #             "fwd_rnn_hidden_states": fwd_rnn_hidden_states,
        #             "rnn_hidden_states": rnn_hidden_states,
        #         },
        #     ),
        #     memory.set_tensor(self.memory_key, rnn_hidden_states),
        # )

        return (
            ActorCriticOutput(
                distributions=self.actor(fwd_obs_encoding),
                values=self.critic(fwd_obs_encoding),
                extras={
                    "pred_action_logits": action_logits,
                    "pred_rotation_logits": rotation_logits,
                    "next_state": next_state,
                    "next_state_pred": next_state_pred,
                    "td_preds": td_preds,
                    "td_pair_target": td_pair_target,
                    "fwd_rnn_hidden_states": rnn_hidden_states,
                    "rnn_hidden_states": fwd_rnn_hidden_states,
                },
            ),
            memory.set_tensor(self.memory_key, fwd_rnn_hidden_states),
        )


class InverseModel(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        input_size: int,
        inv_vis: bool,
        inv_bel: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = action_space.n - 1
        self.inv_vis = inv_vis
        self.inv_bel = inv_bel

        # self.action_predictor = nn.Sequential(
        #     nn.Linear(self.input_size, 256), nn.ReLU(), nn.Linear(256, self.output_size),
        # )

        # self.action_predictor = nn.Sequential(
        #     nn.Linear(self.input_size, 500),
        #     nn.LayerNorm(500),
        #     nn.Tanh(),
        #     nn.Linear(500, 500),
        #     nn.LayerNorm(500),
        #     nn.Tanh(),
        #     nn.Linear(500, self.output_size),
        # )

        self.action_predictor = nn.Linear(self.input_size, self.output_size)

    def forward(self, obs_encoding, rnn_hidden_states=None):
        if self.inv_vis:
            curr_state = obs_encoding
            next_state = obs_encoding.roll(-1, 0)
            next_state[obs_encoding.size(0) - 1, :, :] = obs_encoding[
                obs_encoding.size(0) - 1, :, :
            ]
            action_logits = self.action_predictor(
                torch.cat([curr_state, next_state], 2)
            )
        elif self.inv_bel:
            curr_state = obs_encoding
            next_state = obs_encoding.roll(-1, 0)
            next_state[obs_encoding.size(0) - 1, :, :] = obs_encoding[
                obs_encoding.size(0) - 1, :, :
            ]
            action_logits = self.action_predictor(
                torch.cat(
                    [
                        curr_state,
                        next_state,
                        rnn_hidden_states.repeat(curr_state.shape[0], 1, 1),
                    ],
                    2,
                )
            )
        return action_logits


class ForwardModel(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        action_embed: int,
        state_size: int,
        output_size: int,
        fwd_vis: bool,
        fwd_bel: bool,
    ):
        super().__init__()
        self.action_space = action_space.n - 1  # Ignore end action
        self.action_embed = action_embed
        self.state_size = state_size
        self.output_size = output_size
        self.fwd_vis = fwd_vis
        self.fwd_bel = fwd_bel
        self.embed_action = nn.Embedding(self.action_space, self.action_embed)
        self.state_predictor = nn.Sequential(
            nn.Linear(self.state_size + self.action_embed, 500),
            nn.Tanh(),
            nn.Linear(500, self.output_size),
        )

    def forward(self, obs_encoding, taken_actions, rnn_hidden_states=None):
        curr_state = obs_encoding
        next_state = obs_encoding.roll(-1, 0)
        next_state[obs_encoding.size(0) - 1, :, :] = obs_encoding[
            obs_encoding.size(0) - 1, :, :
        ]

        if self.fwd_bel:
            bel_states = rnn_hidden_states
            bel_states = bel_states.repeat(curr_state.shape[0], 1, 1)
            bel_states = bel_states.view(-1, bel_states.size(2))

        # Compress and change shape
        curr_state = curr_state.view(-1, curr_state.size(2))
        next_state = next_state.view(-1, next_state.size(2))
        actions = taken_actions.view(-1)

        # Remove END actions
        nend_bin = actions != 3  # Indices where the action is not END
        nend_ind = nend_bin.nonzero().long()
        actions = actions[nend_ind].squeeze()
        actions = torch.where(actions > 3, actions - 1, actions)
        curr_state = curr_state[nend_ind, :].squeeze()
        next_state = next_state[nend_ind, :].squeeze()

        if self.fwd_bel:
            bel_states = bel_states[nend_ind, :]
            bel_states = bel_states.squeeze()

        action_embedding = self.embed_action(actions)

        if curr_state.nelement() != 0:
            if len(curr_state.shape) == 1:
                curr_state = curr_state.unsqueeze(0)
                action_embedding = action_embedding.unsqueeze(0)

            if self.fwd_vis:
                next_state_prediction = self.state_predictor(
                    torch.cat([curr_state, action_embedding], 1)
                )
            elif self.fwd_bel:
                if len(bel_states.shape) == 1:
                    bel_states = bel_states.unsqueeze(0)
                next_state_prediction = self.state_predictor(
                    torch.cat([curr_state, action_embedding, bel_states], 1)
                )
        else:
            next_state_prediction = None
            next_state = None
        return next_state_prediction, next_state


class RotationModel(nn.Module):
    def __init__(self, state_size: int = 1568):
        super().__init__()
        self.state_size = state_size

        # self.rotation_predictor = nn.Sequential(
        #     nn.Linear(state_size, 256), nn.ReLU(), nn.Linear(256, 4),
        # )

        self.rotation_predictor = nn.Sequential(nn.Linear(state_size, 4))

    def forward(self, state):
        rotation_logits = self.rotation_predictor(state)
        return rotation_logits


class TemporalDistancePredictor(nn.Module):
    def __init__(self, input_size: int, num_pairs: int = 5, close_thresh: float = 5.0):
        super().__init__()
        self.input_size = input_size
        self.num_pairs = num_pairs
        self.close_thresh = close_thresh
        self.td_predictor = nn.Sequential(nn.Linear(self.input_size, 1))

    def forward(self, state, rnn_hidden_states):
        # Sample negative indices
        pair_indices_per_step = []
        pair_indices_target = []
        # print("State shape", state.shape)
        for i in range(state.shape[0]):
            curr_ind = (
                torch.randint(i, state.shape[0], (self.num_pairs,)).unique().long()
            )
            ind_diff = torch.abs(i - curr_ind.float())
            ind_diff = torch.where(
                ind_diff > self.close_thresh, ind_diff, torch.tensor(0).float()
            )
            curr_target = ind_diff / state.shape[0]
            pair_indices_per_step.append(curr_ind)
            pair_indices_target.append(curr_target)

        pair_start = [
            state[i].repeat(x.shape[0], 1, 1)
            for i, x in enumerate(pair_indices_per_step)
        ]
        pair_end = [state[x] for x in pair_indices_per_step]
        if state.shape[0] == 1:
            pair_target = [
                x.unsqueeze(1).unsqueeze(2).repeat(1, state.shape[1], 1)
                for i, x in enumerate(pair_indices_target)
            ]
            # print([x.shape for x in pair_target])
        else:
            pair_target = [
                x.unsqueeze(1).unsqueeze(2).repeat(1, state.shape[1], 1)
                for i, x in enumerate(pair_indices_target)
            ]
            # print([x.shape for x in pair_target])
        pair_start = torch.cat(pair_start, 0)
        pair_end = torch.cat(pair_end, 0)
        pair_target = torch.cat(pair_target, 0)
        pair_bels = rnn_hidden_states.repeat(pair_start.shape[0], 1, 1)

        pair_start = pair_start.view(-1, pair_start.size(2))
        pair_end = pair_end.view(-1, pair_end.size(2))
        pair_target = pair_target.view(-1, pair_target.size(2))
        pair_bels = pair_bels.view(-1, pair_bels.size(2))

        preds = self.td_predictor(torch.cat([pair_start, pair_end, pair_bels], 1))
        # if state.shape[0] > 1:
        #     print("Pair Start", pair_start.shape)
        #     print("Pair End", pair_end.shape)
        #     print("Pair Target", pair_target.shape)
        #     print("Pair Bels", pair_bels.shape)
        #     print("Prediction Shape", preds.shape)
        #     print("Target Shape", pair_target.shape)

        return preds, pair_target


class AugmentedTemporalContrastModel(nn.Module):
    def __init__(
        self,
        visual_encoder,
        momentum_visual_encoder,
        global_compressor_input_size: int = 1568,
        code_size: int = 128,
        momentum: float = 0.01,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.momentum_visual_encoder = momentum_visual_encoder
        self.global_compressor_input_size = global_compressor_input_size
        self.c_dim = code_size
        self.m = self.momentum

        # Define global compressor and momentum global compressor
        self.global_compressor = nn.Sequential(
            nn.Linear(self.global_compressor_input_size, self.c_dim)
        )
        self.momentum_global_compressor = nn.Sequential(
            nn.Linear(self.global_compressor_input_size, self.c_dim)
        )

        # Residual Predictor
        self.residual_predictor = nn.Sequential(
            nn.Linear(self.c_dim, 100), nn.ReLU(), nn.Linear(100, self.c_dim),
        )

        # Contrastive Transformation Matrix
        self.W = nn.Bilinear(self.c_dim, self.c_im, 1, bias=False)

        # Momentum encoder parameter initialization
        for param_v, param_m in zip(
            self.visual_encoder.parameters(), self.momentum_visual_encoder.parameters()
        ):
            param_m.copy_(param_v.data)
            param_m.requires_grad = False

        for param_v, param_m in zip(
            self.global_compressor.parameters(),
            self.momentum_global_compressor.parameters(),
        ):
            param_m.copy_(param_v.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_v, param_m in zip(
            self.visual_encoder.parameters(), self.momentum_visual_encoder.parameters()
        ):
            param_m.data = param_m.data * self.m + param_v.data * (1.0 - self.m)

        for param_v, param_m in zip(
            self.global_compressor.parameters(),
            self.momentum_global_compressor.parameters(),
        ):
            param_m.data = param_m.data * self.m + param_v.data * (1.0 - self.m)

    def forward(self, observations):
        """
        What all do we need to do?
        - Anchor is the current time-step
        - Positive is from t to t+k
        - Negative are other episodes
        - Sample each of the above and perform a forward pass
        """
        # Parameter momentum update
        self._momentum_update_key_encoder()

        anchor_observations = copy.deepcopy(observations)
        rest_observations = copy.deepcopy(observations)
        anchor_encodings = self.visual_encoder(anchor_observations)
        rest_encodings = self.momentum_visual_encoder(rest_observations)

        # Get codes for all samples
        anchor_codes = self.global_compressor(anchor_encodings)
        anchor_codes += self.residual_predictor(anchor_codes)
        rest_codes = self.momentum_global_compressor(rest_encodings)

        # Sample positives and negatives
        num_steps = anchor_codes.shape[0]
        anchor_tensors, positive_tensors, negative_tensors, positive_indices = (
            [],
            [],
            [],
            [],
        )

        # # Positives
        # for i in range(num_steps):
        #     curr_ind = torch.randint(i, num_steps, (1,)).unique().long()
        #     positive_indices.append(curr_ind)

        # # Negatives
        # for i in range(num_steps):
        #     positive_tensors.append()
        pass


class ResnetTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuid: str,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid
        self.goal_dims = goal_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.embed_goal = nn.Linear(2, self.goal_dims)
        self.blind = self.resnet_uuid not in observation_spaces.spaces
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape
            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_dims
        else:
            return (
                self.combine_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return typing.cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 2)

        return observations, use_agent, nstep, nsampler, nagent

    def adapt_output(self, x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        embs = [
            self.compress_resnet(observations),
            self.distribute_target(observations),
        ]
        x = self.target_obs_combiner(torch.cat(embs, dim=1,))
        x = x.view(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

    def visual_compressor(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        x = self.compress_resnet(observations)
        x = x.view(x.size(0), -1)  # flatten
        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetDualTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        rgb_resnet_preprocessor_uuid: str,
        depth_resnet_preprocessor_uuid: str,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.rgb_resnet_uuid = rgb_resnet_preprocessor_uuid
        self.depth_resnet_uuid = depth_resnet_preprocessor_uuid
        self.goal_dims = goal_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.embed_goal = nn.Linear(2, self.goal_dims)
        self.blind = (
            self.rgb_resnet_uuid not in observation_spaces.spaces
            or self.depth_resnet_uuid not in observation_spaces.spaces
        )
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[
                self.rgb_resnet_uuid
            ].shape
            self.rgb_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.depth_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.rgb_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )
            self.depth_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_dims
        else:
            return (
                2
                * self.combine_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return typing.cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_rgb_resnet(self, observations):
        return self.rgb_resnet_compressor(observations[self.rgb_resnet_uuid])

    def compress_depth_resnet(self, observations):
        return self.depth_resnet_compressor(observations[self.depth_resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        rgb = observations[self.rgb_resnet_uuid]
        depth = observations[self.depth_resnet_uuid]

        use_agent = False
        nagent = 1

        if len(rgb.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = rgb.shape[:3]
        else:
            nstep, nsampler = rgb.shape[:2]

        observations[self.rgb_resnet_uuid] = rgb.view(-1, *rgb.shape[-3:])
        observations[self.depth_resnet_uuid] = depth.view(-1, *depth.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 2)

        return observations, use_agent, nstep, nsampler, nagent

    def adapt_output(self, x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        rgb_embs = [
            self.compress_rgb_resnet(observations),
            self.distribute_target(observations),
        ]
        rgb_x = self.rgb_target_obs_combiner(torch.cat(rgb_embs, dim=1,))
        depth_embs = [
            self.compress_depth_resnet(observations),
            self.distribute_target(observations),
        ]
        depth_x = self.depth_target_obs_combiner(torch.cat(depth_embs, dim=1,))
        x = torch.cat([rgb_x, depth_x], dim=1)
        x = x.view(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

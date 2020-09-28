"""Defining the PPO loss for actor critic type models."""

import typing
from typing import Dict, Union
from typing import Optional, Callable

import torch

from core.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from core.base_abstractions.misc import ActorCriticOutput
from core.base_abstractions.distributions import CategoricalDistr


class RandomPred(AbstractActorCriticLoss):
    """Implementation of the random consistency loss
    """

    def __init__(self, loss_coeff, *args, **kwargs):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.loss_coeff = loss_coeff

    def loss(
        self,
        rnn_hidden_states: torch.FloatTensor,
        fwd_rnn_hidden_states: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        vec_dim = rnn_hidden_states.size(2)
        rnn_hidden_states = rnn_hidden_states.view(-1, vec_dim)
        fwd_rnn_hidden_states = fwd_rnn_hidden_states.view(-1, vec_dim)
        fwd_loss = torch.nn.MSELoss()
        random_pred_loss = fwd_loss(rnn_hidden_states, fwd_rnn_hidden_states)
        total_loss = self.loss_coeff * random_pred_loss
        return (
            total_loss,
            {
                "random_pred_loss": total_loss.item(),
                "raw_random_pred_loss": random_pred_loss.item(),
            },
        )


RandomPredConfig = dict(loss_coeff=1.0)

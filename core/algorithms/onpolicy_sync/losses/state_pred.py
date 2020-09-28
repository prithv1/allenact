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


class StatePred(AbstractActorCriticLoss):
    """Implementation of the forward-dynamics loss
    
    - We have already ignored the <END> action
    """

    def __init__(self, loss_coeff, *args, **kwargs):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.loss_coeff = loss_coeff

    def loss(
        self,
        next_state_pred: torch.FloatTensor,
        next_state: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        # print(next_state_pred)
        # print(next_state)
        fwd_loss = torch.nn.SmoothL1Loss()
        if next_state is not None:
            state_pred_loss = fwd_loss(next_state_pred, next_state.detach())
            total_loss = self.loss_coeff * state_pred_loss
        else:
            state_pred_loss = torch.tensor(0).float()
            total_loss = torch.tensor(0).float()

        return (
            total_loss,
            {
                "state_pred_loss": total_loss.item(),
                "raw_state_pred_loss": state_pred_loss.item(),
            },
        )


StatePredConfig = dict(loss_coeff=1.0)

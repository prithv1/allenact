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


class TDPred(AbstractActorCriticLoss):
    """Implementation of the temporal-distance loss
    """

    def __init__(self, loss_coeff, *args, **kwargs):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.loss_coeff = loss_coeff

    def loss(
        self,
        td_preds: torch.FloatTensor,
        td_pair_target: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        # fwd_loss = torch.nn.SmoothL1Loss()
        fwd_loss = torch.nn.MSELoss()
        td_pred_loss = fwd_loss(
            td_preds, td_pair_target.cuda(td_preds.get_device()).detach()
        )
        total_loss = self.loss_coeff * td_pred_loss
        return (
            total_loss,
            {
                "td_pred_loss": total_loss.item(),
                "raw_td_pred_loss": td_pred_loss.item(),
            },
        )


TDPredConfig = dict(loss_coeff=1.0)

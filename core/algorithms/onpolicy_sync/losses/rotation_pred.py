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


class RotationPred(AbstractActorCriticLoss):
    """Implementation of the rotation-prediction loss    
    """

    def __init__(self, loss_coeff, *args, **kwargs):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.loss_coeff = loss_coeff

    def loss(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        pred_rotation_logits: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        ce_loss = torch.nn.CrossEntropyLoss()  # Define the loss expression
        rotation_targets = typing.cast(
            torch.LongTensor, batch["observations"]["rot_label"]
        )  # Labels
        rotation_targets = rotation_targets.view(-1)
        pred_rotation_logits = pred_rotation_logits.view(
            -1, pred_rotation_logits.shape[2]
        )
        rotation_pred_loss = ce_loss(pred_rotation_logits, rotation_targets.squeeze())
        total_loss = self.loss_coeff * rotation_pred_loss

        _, rotation_preds = torch.max(pred_rotation_logits.data, 1)
        correct = (rotation_preds == rotation_targets).sum().float()
        accuracy = 100 * correct / len(rotation_targets)

        return (
            total_loss,
            {
                "rotation_pred_loss": total_loss.item(),
                "raw_rotation_pred_loss": rotation_pred_loss.item(),
                "raw_rotation_pred_acc": accuracy.item(),
            },
        )


RotPredConfig = dict(loss_coeff=1.0)

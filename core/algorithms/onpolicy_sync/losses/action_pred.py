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


class ActionPred(AbstractActorCriticLoss):
    """Implementation of the action-prediction loss
    
    - Can we ignore predicting the END action?
    - END has no consequences in terms of movement / translation,
        so might as well avoid predicting the same.
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
        pred_action_logits: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        ce_loss = torch.nn.CrossEntropyLoss()  # Define the loss expression

        # reshape relevant output and target
        actions = typing.cast(torch.LongTensor, batch["actions"])
        actions = actions.view(-1)
        pred_action_logits = pred_action_logits.view(-1, pred_action_logits.size(2))

        nend_bin = actions != 3  # Indices where the action is not END
        nend_ind = nend_bin.nonzero().long()

        actions = actions[nend_ind]
        actions = torch.where(actions > 3, actions - 1, actions)
        pred_action_logits = pred_action_logits[nend_ind, :]
        actions = actions.squeeze()
        pred_action_logits = pred_action_logits.squeeze()

        if actions.nelement() == 0:
            action_pred_loss = torch.tensor(0).float()
            accuracy = torch.tensor(0).float()
        else:
            if len(pred_action_logits.size()) == 1:
                # print(pred_action_logits.shape)
                # print(actions.shape)
                # print(actions)
                actions = actions.unsqueeze(0)
                pred_action_logits = pred_action_logits.unsqueeze(0)
            action_pred_loss = ce_loss(pred_action_logits, actions)
            # print(action_pred_loss.dtype)
            _, predicted_actions = torch.max(pred_action_logits.data, 1)
            correct = (predicted_actions == actions).sum().float()
            accuracy = 100 * correct / len(actions)
            # print(accuracy.dtype)

        total_loss = self.loss_coeff * action_pred_loss

        return (
            total_loss,
            {
                "action_pred_loss": total_loss.item(),
                "raw_action_pred_loss": action_pred_loss.item(),
                "raw_action_pred_acc": accuracy.item(),
            },
        )


ActPredConfig = dict(loss_coeff=1.0)

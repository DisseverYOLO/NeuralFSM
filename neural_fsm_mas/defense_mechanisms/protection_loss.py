"""
Protection Constrained Loss

Joint training objective:
    L_total = L_task + λ_protect · L_protect + λ_reg · L_reg

Author: Neural FSM Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ProtectionConstrainedLoss(nn.Module):
    """Protection-constrained loss function."""

    def __init__(
        self,
        lambda_protect: float = 0.1,
        lambda_reg: float = 0.01,
        task_loss_type: str = "cross_entropy",
    ):
        """Initialize the loss function."""
        super().__init__()
        self.lambda_protect = lambda_protect
        self.lambda_reg = lambda_reg
        self.task_loss_type = task_loss_type

    def compute_task_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the main task loss."""
        if self.task_loss_type == "cross_entropy":
            return F.cross_entropy(predictions, labels, reduction="mean")
        if self.task_loss_type == "mse":
            return F.mse_loss(predictions, labels, reduction="mean")
        raise ValueError(f"Unknown task loss type: {self.task_loss_type}")

    def compute_protection_loss(
        self,
        messages: torch.Tensor,
        anomaly_scores: torch.Tensor,
        priorities: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the protection loss."""
        source_ids = edge_index[0]
        target_ids = edge_index[1]

        source_anomaly = anomaly_scores[source_ids]
        target_priority = priorities[target_ids]
        risk_scores = source_anomaly * target_priority

        message_norm = torch.norm(messages, dim=1)
        message_dim = messages.size(1) if messages.dim() > 1 and messages.size(0) > 0 else 1
        normalized_strength = message_norm / (message_dim ** 0.5 + 1e-8)
        normalized_strength = torch.clamp(normalized_strength, 0.0, 2.0)

        return torch.mean(risk_scores * normalized_strength)

    def compute_regularization_loss(
        self,
        model_params: List[torch.nn.Parameter],
    ) -> torch.Tensor:
        """Compute L2 regularization loss."""
        if not model_params:
            return torch.tensor(0.0)

        l2_reg = sum(torch.norm(p) ** 2 for p in model_params if p.requires_grad)
        return l2_reg / len([p for p in model_params if p.requires_grad])

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        messages: torch.Tensor,
        anomaly_scores: torch.Tensor,
        priorities: torch.Tensor,
        edge_index: torch.Tensor,
        model_params: Optional[List[torch.nn.Parameter]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass: compute the total loss."""
        l_task = self.compute_task_loss(predictions, labels)
        l_protect = self.compute_protection_loss(messages, anomaly_scores, priorities, edge_index)

        if model_params is not None:
            l_reg = self.compute_regularization_loss(model_params)
        else:
            l_reg = torch.tensor(0.0, device=l_task.device)

        l_total = l_task + self.lambda_protect * l_protect + self.lambda_reg * l_reg

        loss_dict = {
            "total": l_total.item(),
            "task": l_task.item(),
            "protection": l_protect.item(),
            "regularization": l_reg.item(),
        }
        return l_total, loss_dict


class AdaptiveProtectionLoss(ProtectionConstrainedLoss):
    """Adaptive protection loss with learnable loss weights."""

    def __init__(
        self,
        lambda_protect_init: float = 0.1,
        lambda_reg_init: float = 0.01,
        task_loss_type: str = "cross_entropy",
    ):
        """Initialize the adaptive loss."""
        nn.Module.__init__(self)
        self.task_loss_type = task_loss_type
        self.lambda_protect = nn.Parameter(torch.tensor(lambda_protect_init))
        self.lambda_reg = nn.Parameter(torch.tensor(lambda_reg_init))

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        messages: torch.Tensor,
        anomaly_scores: torch.Tensor,
        priorities: torch.Tensor,
        edge_index: torch.Tensor,
        model_params: Optional[List[torch.nn.Parameter]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass using learnable weights."""
        l_task = self.compute_task_loss(predictions, labels)
        l_protect = self.compute_protection_loss(messages, anomaly_scores, priorities, edge_index)

        if model_params is not None:
            l_reg = self.compute_regularization_loss(model_params)
        else:
            l_reg = torch.tensor(0.0, device=l_task.device)

        lambda_p = torch.sigmoid(self.lambda_protect) * 0.5
        lambda_r = torch.sigmoid(self.lambda_reg) * 0.1
        l_total = l_task + lambda_p * l_protect + lambda_r * l_reg

        loss_dict = {
            "total": l_total.item(),
            "task": l_task.item(),
            "protection": l_protect.item(),
            "regularization": l_reg.item(),
            "lambda_protect": lambda_p.item(),
            "lambda_reg": lambda_r.item(),
        }
        return l_total, loss_dict


__all__ = [
    "ProtectionConstrainedLoss",
    "AdaptiveProtectionLoss",
]

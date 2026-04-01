"""
Trust Calculator

Core formula:
    trust(i, t) = (1 - α(i, t)) · (1 + π(i)) · (1 - β · pollution(i))

This module computes trust scores from node anomaly scores, protection
priorities, and message-pollution levels.

Author: Neural FSM Team
Date: 2025-10-28
Version: 2.0 (extended)
"""

import torch
import torch.nn as nn
from typing import Union, Optional


class TrustCalculator(nn.Module):
    """
    Trust score calculator.

    Formula:
        trust(i) = (1 - α(i)) · (1 + π(i)) · (1 - β · pollution(i))
    """

    def __init__(self, pollution_penalty_weight: float = 0.5):
        """Initialize the trust calculator."""
        super().__init__()
        self.pollution_penalty_weight = pollution_penalty_weight

    def forward(
        self,
        anomaly_scores: torch.Tensor,
        priorities: torch.Tensor,
        pollution_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute trust scores, including optional pollution penalties.
        """
        trust = (1.0 - anomaly_scores) * (1.0 + priorities)

        if pollution_scores is not None:
            pollution_penalty = 1.0 - self.pollution_penalty_weight * pollution_scores
            pollution_penalty = torch.clamp(pollution_penalty, 0.1, 1.0)
            trust = trust * pollution_penalty

        return torch.clamp(trust, 0.0, 2.0)

    def compute_with_details(
        self,
        anomaly_scores: torch.Tensor,
        priorities: torch.Tensor,
        pollution_scores: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute trust scores and return detailed outputs."""
        base_trust = (1.0 - anomaly_scores) * (1.0 + priorities)

        if pollution_scores is not None:
            pollution_penalty = 1.0 - self.pollution_penalty_weight * pollution_scores
            pollution_penalty = torch.clamp(pollution_penalty, 0.1, 1.0)
        else:
            pollution_penalty = torch.ones_like(anomaly_scores)

        trust = torch.clamp(base_trust * pollution_penalty, 0.0, 2.0)

        return {
            "trust_scores": trust,
            "base_trust": base_trust,
            "pollution_penalty": pollution_penalty,
            "most_trusted": int(torch.argmax(trust).item()),
            "least_trusted": int(torch.argmin(trust).item()),
        }


def compute_trust_scores(
    anomaly_scores: Union[torch.Tensor, list],
    priorities: Union[torch.Tensor, list],
    pollution_scores: Optional[Union[torch.Tensor, list]] = None,
) -> torch.Tensor:
    """Convenience helper for quickly computing trust scores."""
    if not isinstance(anomaly_scores, torch.Tensor):
        anomaly_scores = torch.tensor(anomaly_scores, dtype=torch.float32)
    if not isinstance(priorities, torch.Tensor):
        priorities = torch.tensor(priorities, dtype=torch.float32)
    if pollution_scores is not None and not isinstance(pollution_scores, torch.Tensor):
        pollution_scores = torch.tensor(pollution_scores, dtype=torch.float32)

    calculator = TrustCalculator()
    return calculator(anomaly_scores, priorities, pollution_scores)


__all__ = [
    "TrustCalculator",
    "compute_trust_scores",
]

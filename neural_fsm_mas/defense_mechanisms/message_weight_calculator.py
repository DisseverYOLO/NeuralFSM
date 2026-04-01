"""
Message Weight Calculator

Compute message weights based on the source node's trust score and the
target node's protection priority. This is used inside the TGN message
function to attenuate messages from low-trust nodes.

Author: Neural FSM Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
from typing import Union


class MessageWeightCalculator(nn.Module):
    """
    Learnable message weight calculator.

    Function: compute message weights w_{i→j} = f(trust(i), π(j))

    Design ideas:
    1. Lower trust in the source node leads to smaller weights.
    2. More important target nodes are more sensitive to low-trust sources.
    3. A small MLP learns the optimal weighting strategy.
    """

    def __init__(self, hidden_dim: int = 64):
        """
        Initialize the calculator.

        Args:
            hidden_dim: Hidden layer dimension (default: 64)
        """
        super().__init__()

        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        trust_source: torch.Tensor,
        priority_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute message weights.

        Args:
            trust_source: Source-node trust scores [num_edges], range [0, 2]
            priority_target: Target-node protection priorities [num_edges], range [0, 1]

        Returns:
            Message weights [num_edges], range [0, 1]
        """
        trust_normalized = trust_source / 2.0
        features = torch.stack([trust_normalized, priority_target], dim=-1)
        return self.weight_net(features).squeeze(-1)


class SimpleMessageWeightCalculator(nn.Module):
    """
    Simple message weight calculator without learning.

    Uses the fixed formula: w = sigmoid(10 * (1 - risk))
    where risk = (1 - trust / 2) * priority.
    """

    def __init__(self, temperature: float = 10.0):
        """
        Initialize the calculator.

        Args:
            temperature: Sigmoid temperature parameter; larger means steeper
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        trust_source: torch.Tensor,
        priority_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute message weights using the fixed formula.

        Args:
            trust_source: Source-node trust scores [num_edges], range [0, 2]
            priority_target: Target-node protection priorities [num_edges], range [0, 1]

        Returns:
            Message weights [num_edges], range [0, 1]
        """
        risk = (1.0 - trust_source / 2.0) * priority_target
        return torch.sigmoid(self.temperature * (1.0 - risk))


def compute_message_weights(
    trust_source: Union[torch.Tensor, list],
    priority_target: Union[torch.Tensor, list],
    learnable: bool = True,
    hidden_dim: int = 64,
) -> torch.Tensor:
    """
    Convenience helper for quickly computing message weights.

    Args:
        trust_source: Source-node trust scores
        priority_target: Target-node protection priorities
        learnable: Whether to use the learnable calculator
        hidden_dim: Hidden layer dimension, only used when learnable=True

    Returns:
        Message weights
    """
    if not isinstance(trust_source, torch.Tensor):
        trust_source = torch.tensor(trust_source, dtype=torch.float32)
    if not isinstance(priority_target, torch.Tensor):
        priority_target = torch.tensor(priority_target, dtype=torch.float32)

    if learnable:
        calculator = MessageWeightCalculator(hidden_dim=hidden_dim)
    else:
        calculator = SimpleMessageWeightCalculator()

    return calculator(trust_source, priority_target)


__all__ = [
    "MessageWeightCalculator",
    "SimpleMessageWeightCalculator",
    "compute_message_weights",
]

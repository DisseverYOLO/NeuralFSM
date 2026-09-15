"""
LLM Cost Loss Function

Features:
1. Compute the LLM call cost for an episode.
2. Define a cost loss that encourages fewer LLM calls.
3. Combine cost loss with other loss functions.
4. Normalize and scale cost values.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple


class CostLoss(nn.Module):
    """LLM cost loss."""
    
    def __init__(self, 
                 baseline_cost: float = 0.005,
                 cost_scale: float = 30.0,
                 reduction: str = 'mean'):
        """
        Initialize the cost loss.
        
        Args:
            baseline_cost: Baseline cost in USD per question, used for normalization.
                - For `gpt-5-nano`, the real cost is about $0.0016 per question.
                - The default `$0.005` makes the normalized value about 0.32.
            cost_scale: Cost scaling factor.
                - The default `30` keeps cost loss milder so it does not dominate training.
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.baseline_cost = baseline_cost
        self.cost_scale = cost_scale
        self.reduction = reduction
    
    def forward(self, 
                episode_costs: torch.Tensor,
                target_costs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the cost loss.
        
        Args:
            episode_costs: `[batch_size]` actual cost in USD for each episode.
            target_costs: `[batch_size]` target costs, optional.
        
        Returns:
            cost_loss: Scalar or `[batch_size]`.
        """
        # Normalize costs relative to the baseline.
        normalized_costs = episode_costs / self.baseline_cost
        
        if target_costs is not None:
            # If target costs exist, compute deviation from the target.
            normalized_targets = target_costs / self.baseline_cost
            loss = torch.abs(normalized_costs - normalized_targets)
        else:
            # Otherwise, directly penalize high cost.
            loss = normalized_costs
        
        # Scale to a suitable numerical range.
        loss = loss * self.cost_scale
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveCostLoss(nn.Module):
    """Adaptive cost loss that considers task difficulty."""
    
    def __init__(self, 
                 baseline_cost: float = 0.005,
                 cost_scale: float = 30.0,
                 difficulty_aware: bool = True):
        """
        Initialize the adaptive cost loss.
        
        Args:
            baseline_cost: Baseline cost in USD per question.
            cost_scale: Cost scaling factor.
            difficulty_aware: Whether to consider task difficulty.
        """
        super().__init__()
        self.baseline_cost = baseline_cost
        self.cost_scale = cost_scale
        self.difficulty_aware = difficulty_aware
    
    def forward(self, 
                episode_costs: torch.Tensor,
                task_difficulties: Optional[torch.Tensor] = None,
                accuracy_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute adaptive cost loss.
        
        Args:
            episode_costs: `[batch_size]` episode costs.
            task_difficulties: `[batch_size]` task difficulty in `[0, 1]`, where `1` is hard.
            accuracy_scores: `[batch_size]` accuracy in `[0, 1]`.
        
        Returns:
            cost_loss: Scalar.
        """
        normalized_costs = episode_costs / self.baseline_cost
        
        if self.difficulty_aware and task_difficulties is not None:
            # Allow higher cost for difficult tasks.
            # Penalize high cost more strictly for easier tasks.
            difficulty_weight = 1.0 - task_difficulties * 0.5  # [0.5, 1.0]
            loss = normalized_costs * difficulty_weight
        else:
            loss = normalized_costs
        
        if accuracy_scores is not None:
            # Apply extra penalty when accuracy is low but cost is high.
            # If accuracy is high, high cost is more acceptable.
            cost_efficiency = accuracy_scores / (normalized_costs + 1e-8)
            # Inefficiency means high cost with low accuracy.
            efficiency_penalty = torch.clamp(1.0 - cost_efficiency, min=0.0, max=2.0)
            loss = loss + efficiency_penalty
        
        loss = loss * self.cost_scale
        return loss.mean()


class CostRegularizedLoss(nn.Module):
    """Cost-regularized combined loss."""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 delta: float = 0.1,
                 baseline_cost: float = 0.005,
                 cost_scale: float = 30.0):
        """
        Initialize the four-objective combined loss.
        
        Args:
            alpha: Weight for policy-gradient loss.
            beta: Weight for state-transition loss.
            gamma: Weight for listener-path loss.
            delta: Weight for cost loss.
            baseline_cost: Baseline cost in USD per question.
                - For `gpt-5-nano`, the real cost is about $0.0016 per question.
                - The default `$0.005` gives a moderate penalty for high cost.
            cost_scale: Cost scaling factor.
                - The default `30` keeps cost loss milder so it does not dominate training.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        self.cost_loss_fn = CostLoss(baseline_cost=baseline_cost, cost_scale=cost_scale)
    
    def forward(self,
                policy_loss: torch.Tensor,
                transition_loss: torch.Tensor,
                listener_loss: torch.Tensor,
                episode_costs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined loss.
        
        Args:
            policy_loss: Policy-gradient loss.
            transition_loss: State-transition loss.
            listener_loss: Listener-path loss.
            episode_costs: Episode costs.
        
        Returns:
            (total_loss, loss_dict)
        """
        # Compute the cost loss.
        cost_loss = self.cost_loss_fn(episode_costs)
        
        # Combine all loss terms.
        total_loss = (self.alpha * policy_loss + 
                     self.beta * transition_loss + 
                     self.gamma * listener_loss + 
                     self.delta * cost_loss)
        
        # Return detailed loss values.
        loss_dict = {
            'total': total_loss.item(),
            'policy': policy_loss.item(),
            'transition': transition_loss.item(),
            'listener': listener_loss.item(),
            'cost': cost_loss.item(),
            'policy_weighted': (self.alpha * policy_loss).item(),
            'transition_weighted': (self.beta * transition_loss).item(),
            'listener_weighted': (self.gamma * listener_loss).item(),
            'cost_weighted': (self.delta * cost_loss).item(),
        }
        
        return total_loss, loss_dict


# ========== Helper Functions ==========

def calculate_episode_cost(cost_tracker, episode_start_idx: int) -> float:
    """
    Compute the episode cost from a `cost_tracker`.
    
    Args:
        cost_tracker: `LLMCostTracker` instance.
        episode_start_idx: Call index where the episode starts.
    
    Returns:
        Episode cost in USD.
    """
    episode_calls = cost_tracker.call_history[episode_start_idx:]
    return sum(call.cost_usd for call in episode_calls)


def batch_episode_costs(costs_list: List[float], device='cpu') -> torch.Tensor:
    """
    Convert a list of episode costs to a tensor.
    
    Args:
        costs_list: List of costs.
        device: Device.
    
    Returns:
        Cost tensor.
    """
    return torch.tensor(costs_list, dtype=torch.float32, device=device)


def estimate_baseline_cost(model_name: str, 
                          avg_input_tokens: int = 500,
                          avg_output_tokens: int = 200,
                          avg_calls_per_episode: int = 5) -> float:
    """
    Estimate the baseline cost.
    
    Args:
        model_name: Model name.
        avg_input_tokens: Average number of input tokens.
        avg_output_tokens: Average number of output tokens.
        avg_calls_per_episode: Average number of calls per episode.
    
    Returns:
        Baseline cost in USD.
    """
    from neural_fsm_mas.utils.llm_cost_tracker import LLMPricingRegistry
    
    pricing = LLMPricingRegistry.get_pricing(model_name)
    single_call_cost = pricing.calculate_cost(avg_input_tokens, avg_output_tokens)
    baseline = single_call_cost * avg_calls_per_episode
    
    return baseline


# ========== Test Example ==========

if __name__ == "__main__":
    print("Testing Cost Loss Functions...\n")
    
    # 1. Basic cost loss.
    print("="*60)
    print("1. Basic Cost Loss")
    print("="*60)
    
    cost_loss = CostLoss(baseline_cost=0.005, cost_scale=100.0)
    
    # Simulate a batch of episode costs based on the real `gpt-5-nano` cost of about $0.0016 per question.
    episode_costs = torch.tensor([0.0012, 0.0018, 0.0015, 0.0020])
    print(f"Episode costs: {episode_costs.tolist()}")
    print(f"Baseline: $0.005, Scale: 100")
    
    loss = cost_loss(episode_costs)
    print(f"Cost loss: {loss.item():.4f}\n")
    
    # 2. Adaptive cost loss.
    print("="*60)
    print("2. Adaptive Cost Loss")
    print("="*60)
    
    adaptive_cost_loss = AdaptiveCostLoss(baseline_cost=0.005, cost_scale=100.0)
    
    # Simulate task difficulty and accuracy.
    task_difficulties = torch.tensor([0.2, 0.8, 0.3, 0.9])  # Difficulty.
    accuracy_scores = torch.tensor([0.9, 0.7, 0.8, 0.6])    # Accuracy.
    
    print(f"Episode costs: {episode_costs.tolist()}")
    print(f"Task difficulties: {task_difficulties.tolist()}")
    print(f"Accuracy scores: {accuracy_scores.tolist()}")
    
    loss = adaptive_cost_loss(episode_costs, task_difficulties, accuracy_scores)
    print(f"Adaptive cost loss: {loss.item():.4f}\n")
    
    # 3. Combined loss.
    print("="*60)
    print("3. Cost-Regularized Combined Loss")
    print("="*60)
    
    combined_loss = CostRegularizedLoss(
        alpha=1.0, beta=0.3, gamma=0.2, delta=0.1,
        baseline_cost=0.005, cost_scale=100.0
    )
    
    # Simulate the other loss terms based on practical experiment scales.
    policy_loss = torch.tensor(32.0)    # Typically around 31-35.
    transition_loss = torch.tensor(7.5)  # Typically around 7-8.
    listener_loss = torch.tensor(3.2)    # Typically around 3.2.
    
    total_loss, loss_dict = combined_loss(
        policy_loss, transition_loss, listener_loss, episode_costs
    )
    
    print("Loss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTotal Loss: {total_loss.item():.4f}\n")
    
    # 4. Baseline cost estimation.
    print("="*60)
    print("4. Baseline Cost Estimation")
    print("="*60)
    
    baseline = estimate_baseline_cost(
        model_name='gpt-5-nano',
        avg_input_tokens=500,
        avg_output_tokens=200,
        avg_calls_per_episode=5
    )
    print(f"Estimated baseline cost for gpt-5-nano: ${baseline:.6f}")
    
    baseline = estimate_baseline_cost(
        model_name='gpt-4o',
        avg_input_tokens=500,
        avg_output_tokens=200,
        avg_calls_per_episode=5
    )
    print(f"Estimated baseline cost for gpt-4o: ${baseline:.6f}\n")

"""
LLM Cost Loss Function
LLM成本损失函数

功能:
1. 计算episode的LLM调用成本
2. 成本损失函数（鼓励减少LLM调用）
3. 与其他损失函数组合
4. 成本归一化和缩放
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple


class CostLoss(nn.Module):
    """LLM成本损失函数"""
    
    def __init__(self, 
                 baseline_cost: float = 0.005,
                 cost_scale: float = 30.0,
                 reduction: str = 'mean'):
        """
        初始化成本损失
        
        Args:
            baseline_cost: 基准成本（美元/问题），用于归一化
                - gpt-5-nano 每问题实际成本约 $0.0016
                - 默认 $0.005 使归一化后的值 ≈ 0.32
            cost_scale: 成本缩放因子
                - 默认 30，使成本损失更温和（避免主导训练）
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
        计算成本损失
        
        Args:
            episode_costs: [batch_size] 每个episode的实际成本（美元）
            target_costs: [batch_size] 目标成本（可选）
        
        Returns:
            cost_loss: 标量或[batch_size]
        """
        # 归一化成本（相对于baseline）
        normalized_costs = episode_costs / self.baseline_cost
        
        if target_costs is not None:
            # 如果有目标成本，计算与目标的偏差
            normalized_targets = target_costs / self.baseline_cost
            loss = torch.abs(normalized_costs - normalized_targets)
        else:
            # 否则，直接惩罚高成本
            loss = normalized_costs
        
        # 缩放到合适的数值范围
        loss = loss * self.cost_scale
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveCostLoss(nn.Module):
    """自适应成本损失（考虑任务难度）"""
    
    def __init__(self, 
                 baseline_cost: float = 0.005,
                 cost_scale: float = 30.0,
                 difficulty_aware: bool = True):
        """
        初始化自适应成本损失
        
        Args:
            baseline_cost: 基准成本（美元/问题）
            cost_scale: 成本缩放因子
            difficulty_aware: 是否考虑任务难度
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
        计算自适应成本损失
        
        Args:
            episode_costs: [batch_size] episode成本
            task_difficulties: [batch_size] 任务难度 (0-1, 1=困难)
            accuracy_scores: [batch_size] 准确率 (0-1)
        
        Returns:
            cost_loss: 标量
        """
        normalized_costs = episode_costs / self.baseline_cost
        
        if self.difficulty_aware and task_difficulties is not None:
            # 对于困难任务，允许更高的成本
            # 对于简单任务，更严格地惩罚高成本
            difficulty_weight = 1.0 - task_difficulties * 0.5  # [0.5, 1.0]
            loss = normalized_costs * difficulty_weight
        else:
            loss = normalized_costs
        
        if accuracy_scores is not None:
            # 如果准确率低但成本高，额外惩罚
            # 如果准确率高，成本高可以接受
            cost_efficiency = accuracy_scores / (normalized_costs + 1e-8)
            # 低效率（高成本低准确率）增加损失
            efficiency_penalty = torch.clamp(1.0 - cost_efficiency, min=0.0, max=2.0)
            loss = loss + efficiency_penalty
        
        loss = loss * self.cost_scale
        return loss.mean()


class CostRegularizedLoss(nn.Module):
    """成本正则化损失（组合损失）"""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 delta: float = 0.1,
                 baseline_cost: float = 0.005,
                 cost_scale: float = 30.0):
        """
        初始化四目标组合损失
        
        Args:
            alpha: 策略梯度损失权重
            beta: 状态转移损失权重
            gamma: 监听路径损失权重
            delta: 成本损失权重
            baseline_cost: 基准成本（美元/问题）
                - gpt-5-nano 每问题实际成本约 $0.0016
                - 默认 $0.005 使成本损失对高成本有适度惩罚
            cost_scale: 成本缩放因子
                - 默认 30，使成本损失更温和（避免主导训练）
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
        计算组合损失
        
        Args:
            policy_loss: 策略梯度损失
            transition_loss: 状态转移损失
            listener_loss: 监听路径损失
            episode_costs: episode成本
        
        Returns:
            (total_loss, loss_dict)
        """
        # 计算成本损失
        cost_loss = self.cost_loss_fn(episode_costs)
        
        # 组合
        total_loss = (self.alpha * policy_loss + 
                     self.beta * transition_loss + 
                     self.gamma * listener_loss + 
                     self.delta * cost_loss)
        
        # 返回详细损失
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


# ========== 辅助函数 ==========

def calculate_episode_cost(cost_tracker, episode_start_idx: int) -> float:
    """
    从cost_tracker计算episode成本
    
    Args:
        cost_tracker: LLMCostTracker实例
        episode_start_idx: episode开始的调用索引
    
    Returns:
        episode成本（美元）
    """
    episode_calls = cost_tracker.call_history[episode_start_idx:]
    return sum(call.cost_usd for call in episode_calls)


def batch_episode_costs(costs_list: List[float], device='cpu') -> torch.Tensor:
    """
    将episode成本列表转换为tensor
    
    Args:
        costs_list: 成本列表
        device: 设备
    
    Returns:
        成本tensor
    """
    return torch.tensor(costs_list, dtype=torch.float32, device=device)


def estimate_baseline_cost(model_name: str, 
                          avg_input_tokens: int = 500,
                          avg_output_tokens: int = 200,
                          avg_calls_per_episode: int = 5) -> float:
    """
    估算基准成本
    
    Args:
        model_name: 模型名称
        avg_input_tokens: 平均输入token数
        avg_output_tokens: 平均输出token数
        avg_calls_per_episode: 每episode平均调用次数
    
    Returns:
        基准成本（美元）
    """
    from neural_fsm_mas.utils.llm_cost_tracker import LLMPricingRegistry
    
    pricing = LLMPricingRegistry.get_pricing(model_name)
    single_call_cost = pricing.calculate_cost(avg_input_tokens, avg_output_tokens)
    baseline = single_call_cost * avg_calls_per_episode
    
    return baseline


# ========== 测试示例 ==========

if __name__ == "__main__":
    print("Testing Cost Loss Functions...\n")
    
    # 1. 基础成本损失
    print("="*60)
    print("1. Basic Cost Loss")
    print("="*60)
    
    cost_loss = CostLoss(baseline_cost=0.005, cost_scale=100.0)
    
    # 模拟一批episode成本（基于gpt-5-nano实际成本 ~$0.0016/问题）
    episode_costs = torch.tensor([0.0012, 0.0018, 0.0015, 0.0020])
    print(f"Episode costs: {episode_costs.tolist()}")
    print(f"Baseline: $0.005, Scale: 100")
    
    loss = cost_loss(episode_costs)
    print(f"Cost loss: {loss.item():.4f}\n")
    
    # 2. 自适应成本损失
    print("="*60)
    print("2. Adaptive Cost Loss")
    print("="*60)
    
    adaptive_cost_loss = AdaptiveCostLoss(baseline_cost=0.005, cost_scale=100.0)
    
    # 模拟任务难度和准确率
    task_difficulties = torch.tensor([0.2, 0.8, 0.3, 0.9])  # 困难度
    accuracy_scores = torch.tensor([0.9, 0.7, 0.8, 0.6])    # 准确率
    
    print(f"Episode costs: {episode_costs.tolist()}")
    print(f"Task difficulties: {task_difficulties.tolist()}")
    print(f"Accuracy scores: {accuracy_scores.tolist()}")
    
    loss = adaptive_cost_loss(episode_costs, task_difficulties, accuracy_scores)
    print(f"Adaptive cost loss: {loss.item():.4f}\n")
    
    # 3. 组合损失
    print("="*60)
    print("3. Cost-Regularized Combined Loss")
    print("="*60)
    
    combined_loss = CostRegularizedLoss(
        alpha=1.0, beta=0.3, gamma=0.2, delta=0.1,
        baseline_cost=0.005, cost_scale=100.0
    )
    
    # 模拟其他损失（基于实际实验数据量级）
    policy_loss = torch.tensor(32.0)    # 实际约 31-35
    transition_loss = torch.tensor(7.5)  # 实际约 7-8
    listener_loss = torch.tensor(3.2)    # 实际约 3.2
    
    total_loss, loss_dict = combined_loss(
        policy_loss, transition_loss, listener_loss, episode_costs
    )
    
    print("Loss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTotal Loss: {total_loss.item():.4f}\n")
    
    # 4. 基准成本估算
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


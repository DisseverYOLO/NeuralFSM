"""
Trust Calculator
信任分数计算器（✨ 扩展版，支持消息污染惩罚）

核心公式: trust(i,t) = (1 - α(i,t)) · (1 + π(i)) · (1 - β · pollution(i))

结合节点的异常分数、保护优先级和消息污染程度,计算信任分数

作者: Neural FSM Team
日期: 2025-10-28
版本: 2.0 (扩展版)
"""

import torch
import torch.nn as nn
from typing import Union, Optional


class TrustCalculator(nn.Module):
    """
    信任分数计算器（✨ 扩展版）
    
    公式: trust(i) = (1 - α(i)) · (1 + π(i)) · (1 - β · pollution(i))
    
    含义:
    - (1 - α): 异常分数越低,信任度越高
    - (1 + π): 重要节点的信任基线更高
    - (1 - β · pollution): ✨ 消息污染惩罚因子
    - 乘积: 同时考虑行为、结构和消息内容
    
    取值范围: [0, 2]
    - 0: 完全不可信 (α=1 的普通节点 或 高污染节点)
    - 1: 中等可信 (α=0 的普通节点 或 α=0.5 的重要节点)
    - 2: 完全可信 (α=0 的最重要节点,π=1, pollution=0)
    """
    
    def __init__(self, pollution_penalty_weight: float = 0.5):
        """
        初始化信任计算器
        
        Args:
            pollution_penalty_weight: ✨ 消息污染惩罚权重 β (默认0.5)
        """
        super().__init__()
        self.pollution_penalty_weight = pollution_penalty_weight
    
    def forward(self,
                anomaly_scores: torch.Tensor,
                priorities: torch.Tensor,
                pollution_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算信任分数（✨ 扩展版，支持消息污染惩罚）
        
        Args:
            anomaly_scores: 异常分数 [num_nodes], 范围 [0, 1]
            priorities: 保护优先级 [num_nodes], 范围 [0, 1]
            pollution_scores: ✨ 消息污染分数 [num_nodes], 范围 [0, 1] (可选)
        
        Returns:
            trust_scores: 信任分数 [num_nodes], 范围 [0, 2]
        """
        # 基础信任分数: trust = (1 - α) × (1 + π)
        trust = (1.0 - anomaly_scores) * (1.0 + priorities)
        
        # ✨ 应用消息污染惩罚
        if pollution_scores is not None:
            # 污染惩罚因子: (1 - β · pollution)
            pollution_penalty = 1.0 - self.pollution_penalty_weight * pollution_scores
            pollution_penalty = torch.clamp(pollution_penalty, 0.1, 1.0)  # 最低保留10%信任
            trust = trust * pollution_penalty
        
        # 裁剪到合理范围 (理论上不需要,但为了数值稳定性)
        trust = torch.clamp(trust, 0.0, 2.0)
        
        return trust
    
    def compute_with_details(self,
                            anomaly_scores: torch.Tensor,
                            priorities: torch.Tensor,
                            pollution_scores: Optional[torch.Tensor] = None) -> dict:
        """
        计算信任分数并返回详细信息
        
        Args:
            anomaly_scores: 异常分数 [num_nodes], 范围 [0, 1]
            priorities: 保护优先级 [num_nodes], 范围 [0, 1]
            pollution_scores: ✨ 消息污染分数 [num_nodes], 范围 [0, 1] (可选)
        
        Returns:
            {
                'trust_scores': 信任分数 [num_nodes],
                'base_trust': 基础信任分数 (未应用污染惩罚),
                'pollution_penalty': 污染惩罚因子 [num_nodes],
                'most_trusted': 最可信节点ID,
                'least_trusted': 最不可信节点ID
            }
        """
        # 基础信任分数
        base_trust = (1.0 - anomaly_scores) * (1.0 + priorities)
        
        # 污染惩罚
        if pollution_scores is not None:
            pollution_penalty = 1.0 - self.pollution_penalty_weight * pollution_scores
            pollution_penalty = torch.clamp(pollution_penalty, 0.1, 1.0)
        else:
            pollution_penalty = torch.ones_like(anomaly_scores)
        
        # 最终信任分数
        trust = base_trust * pollution_penalty
        trust = torch.clamp(trust, 0.0, 2.0)
        
        return {
            'trust_scores': trust,
            'base_trust': base_trust,
            'pollution_penalty': pollution_penalty,
            'most_trusted': int(torch.argmax(trust).item()),
            'least_trusted': int(torch.argmin(trust).item())
        }


# 便捷函数
def compute_trust_scores(anomaly_scores: Union[torch.Tensor, list],
                        priorities: Union[torch.Tensor, list],
                        pollution_scores: Optional[Union[torch.Tensor, list]] = None) -> torch.Tensor:
    """
    便捷函数: 快速计算信任分数（✨ 扩展版，支持污染惩罚）
    
    Args:
        anomaly_scores: 异常分数
        priorities: 保护优先级
        pollution_scores: ✨ 消息污染分数 (可选)
    
    Returns:
        信任分数
    """
    # 转换为tensor
    if not isinstance(anomaly_scores, torch.Tensor):
        anomaly_scores = torch.tensor(anomaly_scores, dtype=torch.float32)
    if not isinstance(priorities, torch.Tensor):
        priorities = torch.tensor(priorities, dtype=torch.float32)
    if pollution_scores is not None and not isinstance(pollution_scores, torch.Tensor):
        pollution_scores = torch.tensor(pollution_scores, dtype=torch.float32)
    
    calculator = TrustCalculator()
    return calculator(anomaly_scores, priorities, pollution_scores)


__all__ = [
    'TrustCalculator',
    'compute_trust_scores'
]


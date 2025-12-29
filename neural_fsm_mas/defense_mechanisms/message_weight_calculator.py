"""
Message Weight Calculator
消息权重计算器

根据源节点的信任分数和目标节点的保护优先级,计算消息权重
用于在TGN消息函数中衰减来自低信任节点的消息

作者: Neural FSM Team
日期: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class MessageWeightCalculator(nn.Module):
    """
    消息权重计算器 (可学习)
    
    功能: 计算消息权重 w_{i→j} = f(trust(i), π(j))
    
    设计思想:
    1. 源节点信任度低 → 权重小 (衰减消息)
    2. 目标节点重要性高 → 对低信任源更敏感 (加强保护)
    3. 使用小型MLP学习最优权重计算策略
    """
    
    def __init__(self, hidden_dim: int = 64):
        """
        初始化
        
        Args:
            hidden_dim: 隐藏层维度 (默认64)
        """
        super().__init__()
        
        # 小型MLP学习权重计算策略
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入: [trust_source/2.0, priority_target]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出: [0, 1]
        )
    
    def forward(self,
                trust_source: torch.Tensor,
                priority_target: torch.Tensor) -> torch.Tensor:
        """
        计算消息权重
        
        Args:
            trust_source: 源节点信任分数 [num_edges], 范围 [0, 2]
            priority_target: 目标节点保护优先级 [num_edges], 范围 [0, 1]
        
        Returns:
            weights: 消息权重 [num_edges], 范围 [0, 1]
        """
        # 归一化trust到[0, 1]
        trust_normalized = trust_source / 2.0
        
        # 拼接特征 [num_edges, 2]
        features = torch.stack([trust_normalized, priority_target], dim=-1)
        
        # 计算权重
        weights = self.weight_net(features).squeeze(-1)
        
        return weights


class SimpleMessageWeightCalculator(nn.Module):
    """
    简单的消息权重计算器 (无需学习)
    
    使用固定公式: w = sigmoid(10 * (1 - risk))
    其中 risk = (1 - trust/2) * priority
    """
    
    def __init__(self, temperature: float = 10.0):
        """
        初始化
        
        Args:
            temperature: Sigmoid温度参数 (越大越陡峭)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self,
                trust_source: torch.Tensor,
                priority_target: torch.Tensor) -> torch.Tensor:
        """
        计算消息权重 (固定公式)
        
        Args:
            trust_source: 源节点信任分数 [num_edges], 范围 [0, 2]
            priority_target: 目标节点保护优先级 [num_edges], 范围 [0, 1]
        
        Returns:
            weights: 消息权重 [num_edges], 范围 [0, 1]
        """
        # 计算风险分数
        risk = (1.0 - trust_source / 2.0) * priority_target
        
        # 权重: 风险越高,权重越低
        weights = torch.sigmoid(self.temperature * (1.0 - risk))
        
        return weights


# 便捷函数
def compute_message_weights(trust_source: Union[torch.Tensor, list],
                           priority_target: Union[torch.Tensor, list],
                           learnable: bool = True,
                           hidden_dim: int = 64) -> torch.Tensor:
    """
    便捷函数: 快速计算消息权重
    
    Args:
        trust_source: 源节点信任分数
        priority_target: 目标节点保护优先级
        learnable: 是否使用可学习的计算器
        hidden_dim: 隐藏层维度 (仅learnable=True时有效)
    
    Returns:
        消息权重
    """
    # 转换为tensor
    if not isinstance(trust_source, torch.Tensor):
        trust_source = torch.tensor(trust_source, dtype=torch.float32)
    if not isinstance(priority_target, torch.Tensor):
        priority_target = torch.tensor(priority_target, dtype=torch.float32)
    
    # 选择计算器
    if learnable:
        calculator = MessageWeightCalculator(hidden_dim=hidden_dim)
    else:
        calculator = SimpleMessageWeightCalculator()
    
    return calculator(trust_source, priority_target)


__all__ = [
    'MessageWeightCalculator',
    'SimpleMessageWeightCalculator',
    'compute_message_weights'
]


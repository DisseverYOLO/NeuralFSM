"""
Protection Constrained Loss
保护约束损失函数

联合训练目标: L_total = L_task + λ_protect · L_protect + λ_reg · L_reg

作者: Neural FSM Team
日期: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple


class ProtectionConstrainedLoss(nn.Module):
    """
    保护约束损失函数
    
    L_total = L_task + λ_protect · L_protect + λ_reg · L_reg
    
    其中:
    - L_task: 主任务损失 (如交叉熵)
    - L_protect: 保护损失,惩罚高风险连接的强消息
    - L_reg: 参数正则化
    """
    
    def __init__(self,
                 lambda_protect: float = 0.1,
                 lambda_reg: float = 0.01,
                 task_loss_type: str = 'cross_entropy'):
        """
        初始化
        
        Args:
            lambda_protect: 保护损失权重 (默认0.1)
            lambda_reg: 正则化权重 (默认0.01)
            task_loss_type: 任务损失类型 ('cross_entropy' 或 'mse')
        """
        super().__init__()
        
        self.lambda_protect = lambda_protect
        self.lambda_reg = lambda_reg
        self.task_loss_type = task_loss_type
    
    def compute_task_loss(self,
                         predictions: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        """
        计算主任务损失
        
        Args:
            predictions: 模型预测 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
        
        Returns:
            任务损失
        """
        if self.task_loss_type == 'cross_entropy':
            return F.cross_entropy(predictions, labels, reduction='mean')
        elif self.task_loss_type == 'mse':
            return F.mse_loss(predictions, labels, reduction='mean')
        else:
            raise ValueError(f"Unknown task loss type: {self.task_loss_type}")
    
    def compute_protection_loss(self,
                               messages: torch.Tensor,
                               anomaly_scores: torch.Tensor,
                               priorities: torch.Tensor,
                               edge_index: torch.Tensor) -> torch.Tensor:
        """
        计算保护损失
        
        L_protect = Σ_{(i,j)∈E} risk(i,j) · ||m_{i→j}||²
        
        其中 risk(i,j) = α(i) · π(j)
        
        含义: 惩罚"异常源 → 关键目标"的强消息
        
        Args:
            messages: 消息向量 [num_edges, message_dim]
            anomaly_scores: 节点异常分数 [num_nodes]
            priorities: 节点保护优先级 [num_nodes]
            edge_index: 边索引 [2, num_edges]
        
        Returns:
            保护损失
        """
        # 获取源节点和目标节点ID
        source_ids = edge_index[0]  # [num_edges]
        target_ids = edge_index[1]  # [num_edges]
        
        # 计算每条边的风险分数
        source_anomaly = anomaly_scores[source_ids]  # [num_edges]
        target_priority = priorities[target_ids]      # [num_edges]
        risk_scores = source_anomaly * target_priority  # [num_edges]
        
        # ✨ 修正：归一化消息强度，避免保护损失过大
        # 计算消息强度 (L2范数)，然后归一化到 [0, 1]
        message_norm = torch.norm(messages, dim=1)  # [num_edges]
        # 使用消息维度进行归一化
        message_dim = messages.size(1) if messages.dim() > 1 and messages.size(0) > 0 else 1
        normalized_strength = message_norm / (message_dim ** 0.5 + 1e-8)  # 归一化
        # 裁剪到合理范围
        normalized_strength = torch.clamp(normalized_strength, 0.0, 2.0)
        
        # 保护损失: 高风险边的强消息会被惩罚
        # ✨ 使用归一化后的强度，损失值更加平衡
        protection_loss = torch.mean(risk_scores * normalized_strength)
        
        return protection_loss
    
    def compute_regularization_loss(self,
                                   model_params: List[torch.nn.Parameter]) -> torch.Tensor:
        """
        计算参数正则化损失 (L2)
        
        Args:
            model_params: 模型参数列表
        
        Returns:
            正则化损失
        """
        if not model_params:
            return torch.tensor(0.0)
        
        l2_reg = sum(torch.norm(p) ** 2 for p in model_params if p.requires_grad)
        
        # 归一化
        reg_loss = l2_reg / len([p for p in model_params if p.requires_grad])
        
        return reg_loss
    
    def forward(self,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                messages: torch.Tensor,
                anomaly_scores: torch.Tensor,
                priorities: torch.Tensor,
                edge_index: torch.Tensor,
                model_params: Optional[List[torch.nn.Parameter]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        前向传播: 计算总损失
        
        Args:
            predictions: 模型预测 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            messages: 消息向量 [num_edges, message_dim]
            anomaly_scores: 节点异常分数 [num_nodes]
            priorities: 节点保护优先级 [num_nodes]
            edge_index: 边索引 [2, num_edges]
            model_params: 模型参数列表 (可选)
        
        Returns:
            (总损失, 损失详情字典)
        """
        # 1. 任务损失
        L_task = self.compute_task_loss(predictions, labels)
        
        # 2. 保护损失
        L_protect = self.compute_protection_loss(
            messages, anomaly_scores, priorities, edge_index
        )
        
        # 3. 正则化损失
        if model_params is not None:
            L_reg = self.compute_regularization_loss(model_params)
        else:
            L_reg = torch.tensor(0.0, device=L_task.device)
        
        # 4. 总损失
        L_total = (L_task + 
                  self.lambda_protect * L_protect + 
                  self.lambda_reg * L_reg)
        
        # 5. 损失详情
        loss_dict = {
            'total': L_total.item(),
            'task': L_task.item(),
            'protection': L_protect.item(),
            'regularization': L_reg.item()
        }
        
        return L_total, loss_dict


class AdaptiveProtectionLoss(ProtectionConstrainedLoss):
    """
    自适应保护损失 (可学习的损失权重)
    
    lambda_protect 和 lambda_reg 作为可学习参数
    """
    
    def __init__(self,
                 lambda_protect_init: float = 0.1,
                 lambda_reg_init: float = 0.01,
                 task_loss_type: str = 'cross_entropy'):
        """
        初始化
        
        Args:
            lambda_protect_init: 保护损失权重初始值
            lambda_reg_init: 正则化权重初始值
            task_loss_type: 任务损失类型
        """
        # 不调用父类__init__,因为我们要用可学习参数
        nn.Module.__init__(self)
        
        self.task_loss_type = task_loss_type
        
        # 可学习的损失权重
        self.lambda_protect = nn.Parameter(torch.tensor(lambda_protect_init))
        self.lambda_reg = nn.Parameter(torch.tensor(lambda_reg_init))
    
    def forward(self,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                messages: torch.Tensor,
                anomaly_scores: torch.Tensor,
                priorities: torch.Tensor,
                edge_index: torch.Tensor,
                model_params: Optional[List[torch.nn.Parameter]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        前向传播 (使用可学习的权重)
        """
        # 计算各损失项
        L_task = self.compute_task_loss(predictions, labels)
        L_protect = self.compute_protection_loss(
            messages, anomaly_scores, priorities, edge_index
        )
        
        if model_params is not None:
            L_reg = self.compute_regularization_loss(model_params)
        else:
            L_reg = torch.tensor(0.0, device=L_task.device)
        
        # 使用可学习的权重 (sigmoid确保正值)
        λ_p = torch.sigmoid(self.lambda_protect) * 0.5  # 限制到[0, 0.5]
        λ_r = torch.sigmoid(self.lambda_reg) * 0.1     # 限制到[0, 0.1]
        
        # 总损失
        L_total = L_task + λ_p * L_protect + λ_r * L_reg
        
        # 损失详情
        loss_dict = {
            'total': L_total.item(),
            'task': L_task.item(),
            'protection': L_protect.item(),
            'regularization': L_reg.item(),
            'lambda_protect': λ_p.item(),
            'lambda_reg': λ_r.item()
        }
        
        return L_total, loss_dict


__all__ = [
    'ProtectionConstrainedLoss',
    'AdaptiveProtectionLoss'
]


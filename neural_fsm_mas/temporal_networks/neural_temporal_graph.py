"""
Neural Temporal Graph Networks for Multi-Agent System
神经时间图网络用于多智能体系统

适配自原始TGN实现，专门为MetaAgent项目优化
支持智能体通信网络的动态学习和状态转移规则学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from typing import Dict, List, Optional, Tuple
import numpy as np


class AgentMemoryBank(nn.Module):
    """
    智能体记忆库，用于维护智能体的历史交互状态和通信记录
    """
    def __init__(self, memory_dimension: int, agent_count: int):
        super().__init__()
        self.memory_dimension = memory_dimension
        self.agent_count = agent_count
        
        # 初始化智能体记忆存储
        self.agent_memories = nn.Parameter(torch.zeros(agent_count, memory_dimension), requires_grad=False)
        self.last_interaction_time = nn.Parameter(torch.zeros(agent_count), requires_grad=False)
        
        # 记忆更新网络 - 使用GRU进行记忆演化
        self.memory_evolution_cell = nn.GRUCell(memory_dimension, memory_dimension)
        
    def retrieve_agent_memory(self, agent_indices: torch.Tensor) -> torch.Tensor:
        """检索指定智能体的记忆"""
        return self.agent_memories[agent_indices]
    
    def evolve_agent_memory(self, agent_indices: torch.Tensor, 
                           interaction_messages: torch.Tensor, 
                           interaction_timestamps: torch.Tensor):
        """演化智能体记忆状态"""
        with torch.no_grad():
            for idx, agent_id in enumerate(agent_indices):
                # 使用GRU演化记忆状态
                evolved_memory = self.memory_evolution_cell(
                    interaction_messages[idx].unsqueeze(0), 
                    self.agent_memories[agent_id].unsqueeze(0)
                ).squeeze(0)
                self.agent_memories[agent_id] = evolved_memory
                self.last_interaction_time[agent_id] = interaction_timestamps[idx]
    
    def reset_all_memories(self):
        """重置所有智能体的记忆状态"""
        self.agent_memories.data.zero_()
        self.last_interaction_time.data.zero_()


class TemporalEncoder(nn.Module):
    """
    时间编码器，将时间序列信息编码为高维特征向量
    用于捕捉智能体交互的时间依赖性
    """
    def __init__(self, temporal_dimension: int):
        super().__init__()
        self.temporal_dimension = temporal_dimension
        self.linear_projection = nn.Linear(1, temporal_dimension)
        
    def forward(self, temporal_stamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_stamps: [batch_size] 时间戳序列
        Returns:
            temporal_features: [batch_size, temporal_dimension] 时间特征编码
        """
        # 将时间戳转换为浮点数并增加维度
        temporal_stamps = temporal_stamps.float().unsqueeze(-1)
        
        # 使用正弦余弦位置编码捕捉时间周期性
        temporal_features = torch.zeros(temporal_stamps.size(0), self.temporal_dimension, 
                                      device=temporal_stamps.device)
        
        # 计算频率分量
        frequency_components = torch.exp(torch.arange(0, self.temporal_dimension, 2, 
                                                    device=temporal_stamps.device).float() * 
                                       -(np.log(10000.0) / self.temporal_dimension))
        
        # 应用正弦余弦编码
        temporal_features[:, 0::2] = torch.sin(temporal_stamps * frequency_components)
        temporal_features[:, 1::2] = torch.cos(temporal_stamps * frequency_components)
        
        return temporal_features


class CommunicationAggregator(nn.Module):
    """
    通信聚合器，聚合智能体间的通信消息
    用于学习智能体通信网络的最优连接模式
    """
    def __init__(self, agent_feature_dim: int, communication_edge_dim: int, message_dim: int):
        super().__init__()
        self.communication_network = nn.Sequential(
            nn.Linear(2 * agent_feature_dim + communication_edge_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, source_agents: torch.Tensor, target_agents: torch.Tensor, 
                communication_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_agents: [num_communications, agent_feature_dim] 源智能体特征
            target_agents: [num_communications, agent_feature_dim] 目标智能体特征  
            communication_features: [num_communications, communication_edge_dim] 通信边特征
        Returns:
            aggregated_messages: [num_communications, message_dim] 聚合后的通信消息
        """
        # 拼接源智能体、目标智能体和通信特征
        combined_features = torch.cat([source_agents, target_agents, communication_features], dim=-1)
        aggregated_messages = self.communication_network(combined_features)
        return aggregated_messages


class NeuralTemporalGraph(nn.Module):
    """
    神经时间图网络 (Neural Temporal Graph Network)
    
    核心功能：
    1. 学习智能体通信网络的动态连接模式
    2. 学习有限状态机的状态转移规则
    3. 维护智能体的时间敏感记忆
    4. 支持多轮交互的策略优化
    """
    def __init__(self, 
                 agent_feature_dim: int,
                 communication_edge_dim: int = 16,
                 memory_dimension: int = 128,
                 temporal_dimension: int = 32,
                 message_dimension: int = 64,
                 agent_count: int = None,
                 network_layers: int = 2):
        super().__init__()
        
        self.agent_feature_dim = agent_feature_dim
        self.communication_edge_dim = communication_edge_dim
        self.memory_dimension = memory_dimension
        self.temporal_dimension = temporal_dimension
        self.message_dimension = message_dimension
        self.network_layers = network_layers
        
        # 智能体记忆库
        self.agent_memory_bank = AgentMemoryBank(memory_dimension, agent_count) if agent_count else None
        
        # 时间编码器
        self.temporal_encoder = TemporalEncoder(temporal_dimension)
        
        # 通信聚合器
        self.communication_aggregator = CommunicationAggregator(
            agent_feature_dim + memory_dimension + temporal_dimension, 
            communication_edge_dim, 
            message_dimension
        )
        
        # 智能体状态更新器
        self.agent_state_updater = nn.GRUCell(message_dimension, 
                                             agent_feature_dim + memory_dimension + temporal_dimension)
        
        # 多层图神经网络
        self.graph_neural_layers = nn.ModuleList([
            TransformerConv(agent_feature_dim + memory_dimension + temporal_dimension, 
                          agent_feature_dim + memory_dimension + temporal_dimension,
                          heads=4, concat=False, dropout=0.1)
            for _ in range(network_layers)
        ])
        
        # 输出特征投影
        self.feature_projector = nn.Linear(agent_feature_dim + memory_dimension + temporal_dimension, 
                                         agent_feature_dim)
        
        # 正则化层
        self.feature_dropout = nn.Dropout(0.1)
        self.layer_normalization = nn.LayerNorm(agent_feature_dim + memory_dimension + temporal_dimension)
        
    def forward(self, 
                agent_features: torch.Tensor,
                communication_topology: torch.Tensor,
                communication_attributes: Optional[torch.Tensor] = None,
                temporal_stamps: Optional[torch.Tensor] = None,
                agent_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 学习智能体通信网络和状态转移
        
        Args:
            agent_features: [num_agents, agent_feature_dim] 智能体特征
            communication_topology: [2, num_communications] 通信拓扑结构
            communication_attributes: [num_communications, communication_edge_dim] 通信边属性
            temporal_stamps: [num_agents] 时间戳
            agent_indices: [num_agents] 智能体索引
            
        Returns:
            evolved_agent_features: [num_agents, agent_feature_dim] 演化后的智能体特征
        """
        num_agents = agent_features.size(0)
        device = agent_features.device
        
        # 处理默认参数
        if temporal_stamps is None:
            temporal_stamps = torch.zeros(num_agents, device=device)
            
        if agent_indices is None:
            agent_indices = torch.arange(num_agents, device=device)
            
        if communication_attributes is None:
            communication_attributes = torch.zeros(communication_topology.size(1), 
                                                 self.communication_edge_dim, device=device)
        
        # 检索智能体记忆
        if self.agent_memory_bank is not None:
            agent_memories = self.agent_memory_bank.retrieve_agent_memory(agent_indices)
        else:
            agent_memories = torch.zeros(num_agents, self.memory_dimension, device=device)
        
        # 时间特征编码
        temporal_features = self.temporal_encoder(temporal_stamps)
        
        # 融合智能体特征、记忆和时间信息
        fused_agent_features = torch.cat([agent_features, agent_memories, temporal_features], dim=-1)
        
        # 通过多层图神经网络进行特征演化
        # 注意：TransformerConv 不支持 edge_attr 作为位置参数，只支持 edge_index
        # communication_attributes 用于 CommunicationAggregator，不用于 TransformerConv
        for graph_layer in self.graph_neural_layers:
            residual_features = fused_agent_features
            # TransformerConv 只接受 (x, edge_index) 参数，不支持 edge_attr
            fused_agent_features = graph_layer(fused_agent_features, communication_topology)
            fused_agent_features = F.relu(fused_agent_features)
            fused_agent_features = self.feature_dropout(fused_agent_features)
            
            # 残差连接和层归一化
            fused_agent_features = self.layer_normalization(fused_agent_features + residual_features)
        
        # 投影到输出特征空间
        evolved_agent_features = self.feature_projector(fused_agent_features)
        
        # 更新智能体记忆（如果启用记忆库）
        if self.agent_memory_bank is not None:
            # 计算通信消息用于记忆更新
            # 注意：CommunicationNetwork 期望 agent_feature_dim 维度的特征，而不是 fused_features
            # 所以使用原始的 agent_features 或者 evolved_agent_features
            source_agent_features = agent_features[communication_topology[0]]  # [num_edges, agent_feature_dim]
            target_agent_features = agent_features[communication_topology[1]]  # [num_edges, agent_feature_dim]
            communication_messages = self.communication_aggregator(source_agent_features, 
                                                                 target_agent_features, 
                                                                 communication_attributes)
            
            # 为每个智能体聚合接收到的消息
            aggregated_agent_messages = torch.zeros(num_agents, self.message_dimension, device=device)
            for agent_id in range(num_agents):
                incoming_message_mask = communication_topology[1] == agent_id
                if incoming_message_mask.sum() > 0:
                    aggregated_agent_messages[agent_id] = communication_messages[incoming_message_mask].mean(dim=0)
            
            # 演化智能体记忆
            self.agent_memory_bank.evolve_agent_memory(agent_indices, aggregated_agent_messages, temporal_stamps)
        
        return evolved_agent_features
    
    def reset_agent_memories(self):
        """重置所有智能体的记忆状态"""
        if self.agent_memory_bank is not None:
            self.agent_memory_bank.reset_all_memories()
    
    def get_memory_snapshot(self) -> Optional[torch.Tensor]:
        """获取当前记忆状态快照"""
        if self.agent_memory_bank is not None:
            return self.agent_memory_bank.agent_memories.clone()
        return None
    
    def compute_communication_probabilities(self, agent_features: torch.Tensor) -> torch.Tensor:
        """
        计算智能体间的通信概率矩阵
        用于学习最优的智能体通信网络拓扑
        """
        num_agents = agent_features.size(0)
        
        # 计算智能体特征相似度
        feature_similarities = torch.matmul(agent_features, agent_features.t())
        
        # 应用softmax得到通信概率
        communication_probabilities = F.softmax(feature_similarities, dim=-1)
        
        # 移除自连接
        mask = torch.eye(num_agents, device=agent_features.device).bool()
        communication_probabilities.masked_fill_(mask, 0.0)
        
        return communication_probabilities
    
    def sample_communication_topology(self, communication_probabilities: torch.Tensor, 
                                    sampling_temperature: float = 1.0) -> torch.Tensor:
        """
        基于通信概率采样通信拓扑结构
        用于策略梯度训练中的动态拓扑学习
        """
        num_agents = communication_probabilities.size(0)
        
        # 温度调节的概率分布
        adjusted_probabilities = communication_probabilities / sampling_temperature
        
        # 采样通信连接
        sampled_connections = torch.multinomial(adjusted_probabilities.view(-1), 
                                              num_samples=num_agents * 2, 
                                              replacement=True)
        
        # 转换为边索引格式
        source_agents = sampled_connections // num_agents
        target_agents = sampled_connections % num_agents
        
        # 移除自连接
        valid_connections = source_agents != target_agents
        source_agents = source_agents[valid_connections]
        target_agents = target_agents[valid_connections]
        
        communication_topology = torch.stack([source_agents, target_agents], dim=0)
        
        return communication_topology


class MultiLayerPerceptron(nn.Module):
    """
    多层感知机解码器
    用于将智能体特征解码为决策输出
    """
    def __init__(self, input_dimension: int, hidden_dimension: int, 
                 output_dimension: int, num_hidden_layers: int = 2):
        super().__init__()
        
        network_layers = []
        current_dimension = input_dimension
        
        # 构建隐藏层
        for layer_idx in range(num_hidden_layers - 1):
            network_layers.extend([
                nn.Linear(current_dimension, hidden_dimension),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dimension)
            ])
            current_dimension = hidden_dimension
            
        # 输出层
        network_layers.append(nn.Linear(current_dimension, output_dimension))
        
        self.decision_network = nn.Sequential(*network_layers)
        
    def forward(self, agent_features: torch.Tensor) -> torch.Tensor:
        return self.decision_network(agent_features)


class CompatibilityGraphNetwork(nn.Module):
    """
    兼容性图网络 - 保持与原有接口的兼容性
    内部使用神经时间图网络实现
    """
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int):
        super().__init__()
        
        # 使用神经时间图网络作为核心
        self.neural_temporal_graph = NeuralTemporalGraph(
            agent_feature_dim=output_channels,
            memory_dimension=hidden_channels,
            agent_count=None,  # 动态确定
            network_layers=2
        )
        
        # 输入特征投影
        self.input_feature_projector = nn.Linear(input_channels, output_channels)
        
    def reset_parameters(self):
        """重置网络参数"""
        self.input_feature_projector.reset_parameters()
        self.neural_temporal_graph.reset_agent_memories()
        
    def forward(self, agent_features: torch.Tensor, communication_topology: torch.Tensor, 
                temporal_stamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 保持与原接口的兼容性
        
        Args:
            agent_features: [num_agents, input_channels] 智能体特征
            communication_topology: [2, num_communications] 通信拓扑
            temporal_stamps: [num_agents] 可选的时间戳
            
        Returns:
            output_features: [num_agents, output_channels] 输出特征
        """
        # 投影输入特征
        projected_features = self.input_feature_projector(agent_features)
        
        # 通过神经时间图网络处理
        evolved_features = self.neural_temporal_graph(projected_features, communication_topology, 
                                                    temporal_stamps=temporal_stamps)
        
        # 应用log_softmax保持接口一致性
        return F.log_softmax(evolved_features, dim=1)


class StateTransitionLearner(nn.Module):
    """
    状态转移学习器
    专门用于学习有限状态机的状态转移规则
    """
    def __init__(self, state_feature_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_features: [batch_size, state_feature_dim] 状态特征
        Returns:
            action_logits: [batch_size, action_dim] 动作概率分布
            state_values: [batch_size, 1] 状态价值估计
        """
        encoded_states = self.state_encoder(state_features)
        action_logits = self.transition_predictor(encoded_states)
        state_values = self.value_estimator(encoded_states)
        
        return action_logits, state_values


# 导出主要类
__all__ = [
    'NeuralTemporalGraph', 
    'MultiLayerPerceptron', 
    'CompatibilityGraphNetwork',
    'StateTransitionLearner',
    'AgentMemoryBank',
    'TemporalEncoder',
    'CommunicationAggregator'
]

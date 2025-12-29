"""
Multi-Agent Topology Management System
多智能体拓扑管理系统

适配自原始Graph实现，专门为MetaAgent项目优化
支持动态智能体通信拓扑和状态转移学习
"""

import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import sys
import os
from pathlib import Path

# 添加MetaAgent路径
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from neural_fsm_mas.temporal_networks.neural_temporal_graph import (
    NeuralTemporalGraph, MultiLayerPerceptron, CompatibilityGraphNetwork
)
from neural_fsm_mas.agent_topology.agent_node import AgentExecutionNode, ConcreteAgentExecutionNode
from neural_fsm_mas.agent_topology.fsm_state_manager import FSMStateManager, FSMState
from neural_fsm_mas.reasoning_agents.agent_factory import ReasoningAgentFactory
from neural_fsm_mas.domain_prompts.prompt_manager import DomainPromptManager
from torch_geometric.utils import dense_to_sparse


class MultiAgentTopologyManager(ABC):
    """
    多智能体拓扑管理器
    
    核心功能：
    1. 管理智能体网络的拓扑结构
    2. 学习最优的智能体通信模式
    3. 支持动态状态转移规则学习
    4. 集成时间敏感的记忆管理
    
    这个类是MetaAgent项目中智能体协作的核心组件
    """

    def __init__(self, 
                task_domain: str,
                language_model_name: Optional[str],
                agent_role_names: List[str],
                decision_strategy: str,
                enable_spatial_optimization: bool = False,
                initial_spatial_connection_prob: float = 0.5,
                fixed_spatial_connection_masks: List[List[int]] = None,
                enable_temporal_optimization: bool = False,
                initial_temporal_connection_prob: float = 0.5,
                fixed_temporal_connection_masks: List[List[int]] = None,
                agent_configuration_params: List[Dict] = None,
                use_neural_temporal_graph: bool = True,
                memory_bank_dimension: int = 128,
                temporal_encoding_dimension: int = 32,
                ):
        
        # 处理默认连接掩码
        if fixed_spatial_connection_masks is None:
            fixed_spatial_connection_masks = [
                [1 if i != j else 0 for j in range(len(agent_role_names))] 
                for i in range(len(agent_role_names))
            ]
        if fixed_temporal_connection_masks is None:
            fixed_temporal_connection_masks = [
                [1 for j in range(len(agent_role_names))] 
                for i in range(len(agent_role_names))
            ]
        
        # 转换为张量格式
        fixed_spatial_connection_masks = torch.tensor(fixed_spatial_connection_masks).view(-1)
        fixed_temporal_connection_masks = torch.tensor(fixed_temporal_connection_masks).view(-1)
        
        # 验证掩码维度
        expected_mask_size = len(agent_role_names) * len(agent_role_names)
        assert len(fixed_spatial_connection_masks) == expected_mask_size, \
            f"Spatial connection masks size mismatch: expected {expected_mask_size}, got {len(fixed_spatial_connection_masks)}"
        assert len(fixed_temporal_connection_masks) == expected_mask_size, \
            f"Temporal connection masks size mismatch: expected {expected_mask_size}, got {len(fixed_temporal_connection_masks)}"
        
        # 初始化基本属性
        self.topology_id: str = shortuuid.ShortUUID().random(length=6)
        self.task_domain: str = task_domain
        self.language_model_name: str = language_model_name
        self.agent_role_names: List[str] = agent_role_names
        self.enable_spatial_optimization = enable_spatial_optimization
        self.enable_temporal_optimization = enable_temporal_optimization
        self.use_neural_temporal_graph = use_neural_temporal_graph
        self.memory_bank_dimension = memory_bank_dimension
        self.temporal_encoding_dimension = temporal_encoding_dimension
        self.current_interaction_round = 0
        
        # 智能体和决策相关
        self.decision_executor: AgentExecutionNode = ReasoningAgentFactory.create_agent(
            decision_strategy, 
            domain=self.task_domain, 
            llm_name=self.language_model_name
        )
        self.agent_execution_nodes: Dict[str, AgentExecutionNode] = {}
        
        # FSM状态管理器（新增）
        self.fsm_state_manager: Optional[FSMStateManager] = None
        self.use_fsm_mode: bool = False  # 默认兼容旧模式
        self.potential_spatial_connections: List[List[str, str]] = []
        self.potential_temporal_connections: List[List[str, str]] = []
        self.agent_configuration_params = agent_configuration_params if agent_configuration_params is not None else [{} for _ in agent_role_names]
        
        # 初始化智能体网络
        self._initialize_agent_nodes()
        self._initialize_potential_connections()
        
        # 初始化提示管理和特征
        self.domain_prompt_manager = DomainPromptManager.get_manager(task_domain)
        self.role_adjacency_matrix = self._construct_role_adjacency_matrix()
        self.agent_features = self._construct_agent_features()
        
        # 初始化神经网络组件
        self._initialize_neural_networks()
        
        # 初始化连接参数
        self._initialize_connection_parameters(
            initial_spatial_connection_prob, 
            fixed_spatial_connection_masks,
            initial_temporal_connection_prob,
            fixed_temporal_connection_masks
        )
    
    def _initialize_agent_nodes(self):
        """初始化智能体执行节点"""
        for i, agent_role in enumerate(self.agent_role_names):
            agent_config = self.agent_configuration_params[i]
            # 使用具体实现类而不是抽象基类，避免抽象方法实例化错误
            agent_node = ConcreteAgentExecutionNode(
                node_id=f"agent_{i}",
                agent_role=agent_role,
                domain=self.task_domain,
                llm_name=self.language_model_name,
                **agent_config
            )
            # ✨ 标记为FSM模式，用于成本优化（跳过冗余的spatial/temporal context）
            agent_node._use_fsm_mode = self.use_fsm_mode
            self.agent_execution_nodes[agent_node.node_id] = agent_node
    
    def _initialize_potential_connections(self):
        """初始化潜在连接"""
        agent_ids = list(self.agent_execution_nodes.keys())
        
        # 空间连接（智能体间通信）
        for source_id in agent_ids:
            for target_id in agent_ids:
                if source_id != target_id:
                    self.potential_spatial_connections.append([source_id, target_id])
        
        # 时间连接（跨轮次连接）
        for source_id in agent_ids:
            for target_id in agent_ids:
                self.potential_temporal_connections.append([source_id, target_id])
    
    def _construct_role_adjacency_matrix(self):
        """构建角色邻接矩阵"""
        role_connections: List[Tuple[str, str]] = self.domain_prompt_manager.get_role_connections()
        num_agents = self.num_agents
        role_adjacency = torch.zeros((num_agents, num_agents))
        role_to_indices = {}
        
        # 建立角色到索引的映射
        for connection in role_connections:
            input_role, output_role = connection
            role_to_indices[input_role] = []
            role_to_indices[output_role] = []
        
        for i, agent_id in enumerate(self.agent_execution_nodes):
            agent_role = self.agent_execution_nodes[agent_id].agent_role
            if agent_role in role_to_indices:
                role_to_indices[agent_role].append(i)
            
        # 构建邻接矩阵
        for connection in role_connections:
            input_role, output_role = connection
            input_indices = role_to_indices.get(input_role, [])
            output_indices = role_to_indices.get(output_role, [])
            
            for input_idx in input_indices:
                for output_idx in output_indices:
                    role_adjacency[input_idx][output_idx] = 1
        
        # 转换为稀疏格式
        edge_index, edge_weights = dense_to_sparse(role_adjacency)
        return edge_index
    
    def _construct_agent_features(self):
        """构建智能体特征"""
        agent_features = []
        for agent_id in self.agent_execution_nodes:
            agent_role = self.agent_execution_nodes[agent_id].agent_role
            role_description = self.domain_prompt_manager.get_role_description(agent_role)
            
            # 使用嵌入获取特征（这里需要实现嵌入函数）
            feature_vector = self._get_text_embedding(role_description)
            agent_features.append(feature_vector)
        
        return torch.tensor(np.array(agent_features))
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入（简化实现）"""
        # 这里应该使用实际的嵌入模型，暂时使用随机向量
        return np.random.randn(384)  # 假设使用384维嵌入
    
    def _initialize_neural_networks(self):
        """初始化神经网络组件"""
        if self.use_neural_temporal_graph:
            # 使用神经时间图网络
            self.neural_temporal_graph = NeuralTemporalGraph(
                agent_feature_dim=self.agent_features.size(1),
                memory_dimension=self.memory_bank_dimension,
                temporal_dimension=self.temporal_encoding_dimension,
                agent_count=len(self.agent_execution_nodes),
                network_layers=2
            )
            
            # 保持兼容性的图网络
            self.compatibility_graph_network = CompatibilityGraphNetwork(
                self.agent_features.size(1) * 2, 16, self.agent_features.size(1)
            )
        else:
            # 使用传统图网络
            self.compatibility_graph_network = CompatibilityGraphNetwork(
                self.agent_features.size(1) * 2, 16, self.agent_features.size(1)
            )
            self.neural_temporal_graph = None
            
        # 多层感知机解码器
        self.decision_decoder = MultiLayerPerceptron(384, 16, 16)
    
    def _initialize_connection_parameters(self, 
                                        initial_spatial_prob: float,
                                        fixed_spatial_masks: torch.Tensor,
                                        initial_temporal_prob: float,
                                        fixed_temporal_masks: torch.Tensor):
        """初始化连接参数"""
        # 空间连接参数
        if self.enable_spatial_optimization:
            initial_spatial_logit = torch.log(torch.tensor(initial_spatial_prob / (1 - initial_spatial_prob)))
        else:
            initial_spatial_logit = 10.0
            
        self.spatial_connection_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_spatial_connections), requires_grad=self.enable_spatial_optimization) * initial_spatial_logit,
            requires_grad=self.enable_spatial_optimization
        )
        self.spatial_connection_masks = torch.nn.Parameter(fixed_spatial_masks, requires_grad=False)

        # 时间连接参数
        if self.enable_temporal_optimization:
            initial_temporal_logit = torch.log(torch.tensor(initial_temporal_prob / (1 - initial_temporal_prob)))
        else:
            initial_temporal_logit = 10.0
            
        self.temporal_connection_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_temporal_connections), requires_grad=self.enable_temporal_optimization) * initial_temporal_logit,
            requires_grad=self.enable_temporal_optimization
        )
        self.temporal_connection_masks = torch.nn.Parameter(fixed_temporal_masks, requires_grad=False)
    
    def construct_enhanced_agent_features(self, task_query: str):
        """构建增强的智能体特征（融合任务信息）"""
        query_embedding = torch.tensor(self._get_text_embedding(task_query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_agents, 1))
        enhanced_features = torch.cat((self.agent_features, query_embedding), dim=1)
        return enhanced_features
        
    @property
    def spatial_adjacency_matrix(self):
        """获取空间邻接矩阵"""
        matrix = np.zeros((len(self.agent_execution_nodes), len(self.agent_execution_nodes)))
        for i, agent1_id in enumerate(self.agent_execution_nodes):
            for j, agent2_id in enumerate(self.agent_execution_nodes):
                if self.agent_execution_nodes[agent2_id] in self.agent_execution_nodes[agent1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adjacency_matrix(self):
        """获取时间邻接矩阵"""
        matrix = np.zeros((len(self.agent_execution_nodes), len(self.agent_execution_nodes)))
        for i, agent1_id in enumerate(self.agent_execution_nodes):
            for j, agent2_id in enumerate(self.agent_execution_nodes):
                if self.agent_execution_nodes[agent2_id] in self.agent_execution_nodes[agent1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_agents(self):
        """获取智能体数量"""
        return len(self.agent_execution_nodes)

    def construct_spatial_connections(self, sampling_temperature: float = 1.0, 
                                    connection_threshold: float = None) -> torch.Tensor:
        """构建空间连接（智能体通信网络）"""
        self.clear_spatial_connections()
        connection_log_probs = [torch.tensor(0.0, requires_grad=self.enable_spatial_optimization)]
        
        for potential_connection, edge_logit, edge_mask in zip(
            self.potential_spatial_connections, 
            self.spatial_connection_logits, 
            self.spatial_connection_masks
        ):
            source_agent: AgentExecutionNode = self.find_agent_node(potential_connection[0])
            target_agent: AgentExecutionNode = self.find_agent_node(potential_connection[1])
            
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and not self.enable_spatial_optimization:
                if not self._check_connection_cycle(target_agent, {source_agent}):
                    source_agent.add_successor_connection(target_agent, 'spatial')
                continue
            
            # 计算连接概率
            connection_probability = torch.sigmoid(edge_logit / sampling_temperature)
            if connection_threshold:
                connection_probability = torch.tensor(1 if connection_probability > connection_threshold else 0)
                
            # 采样连接
            if torch.rand(1) < connection_probability:
                source_agent.add_successor_connection(target_agent, 'spatial')
                connection_log_probs.append(torch.log(connection_probability))
            else:
                connection_log_probs.append(torch.log(1 - connection_probability))
                    
        return torch.sum(torch.stack(connection_log_probs))
    
    def construct_temporal_connections(self, interaction_round: int = 0, 
                                     sampling_temperature: float = 1.0, 
                                     connection_threshold: float = None) -> torch.Tensor:
        """构建时间连接（跨轮次连接）"""
        self.clear_temporal_connections()
        connection_log_probs = [torch.tensor(0.0, requires_grad=self.enable_temporal_optimization)]
        
        if interaction_round == 0:
            return torch.sum(torch.stack(connection_log_probs))
            
        for potential_connection, edge_logit, edge_mask in zip(
            self.potential_temporal_connections, 
            self.temporal_connection_logits, 
            self.temporal_connection_masks
        ):
            source_agent: AgentExecutionNode = self.find_agent_node(potential_connection[0])
            target_agent: AgentExecutionNode = self.find_agent_node(potential_connection[1])
            
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and not self.enable_temporal_optimization:
                if not self._check_connection_cycle(target_agent, {source_agent}):
                    source_agent.add_successor_connection(target_agent, 'temporal')
                continue
            
            # 计算连接概率
            connection_probability = torch.sigmoid(edge_logit / sampling_temperature)
            if connection_threshold:
                connection_probability = torch.tensor(1 if connection_probability > connection_threshold else 0)
                
            # 采样连接
            if torch.rand(1) < connection_probability:
                source_agent.add_successor_connection(target_agent, 'temporal')
                connection_log_probs.append(torch.log(connection_probability))
            else:
                connection_log_probs.append(torch.log(1 - connection_probability))
                    
        return torch.sum(torch.stack(connection_log_probs))

    def clear_spatial_connections(self):
        """清除空间连接"""
        for agent_node in self.agent_execution_nodes.values():
            agent_node.clear_spatial_connections()

    def clear_temporal_connections(self):
        """清除时间连接"""
        for agent_node in self.agent_execution_nodes.values():
            agent_node.clear_temporal_connections()

    def find_agent_node(self, agent_id: str) -> AgentExecutionNode:
        """查找智能体节点"""
        return self.agent_execution_nodes.get(agent_id)

    def _check_connection_cycle(self, new_agent: AgentExecutionNode, target_agents: set) -> bool:
        """检查连接是否会产生循环"""
        if new_agent in target_agents:
            return True
        for successor in new_agent.spatial_successors:
            if self._check_connection_cycle(successor, target_agents):
                return True
        return False

    async def execute_multi_agent_reasoning(self, 
                                          task_input: Dict[str, str], 
                                          num_interaction_rounds: int = 3, 
                                          max_retry_attempts: int = 3, 
                                          max_execution_time: int = 600) -> List[Any]:
        """
        执行多智能体推理过程
        
        这是核心的执行方法，整合了神经时间图网络的学习能力
        
        Returns:
            (final_reasoning_results, reasoning_log_probs):
                - final_reasoning_results: List[Any] - 推理结果
                - reasoning_log_probs: torch.Tensor - 对数概率(可微分)
        """
        # 初始化为tensor以支持梯度传播
        reasoning_log_probs = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        enhanced_features = self.construct_enhanced_agent_features(task_input['task'])
        
        if self.use_neural_temporal_graph and self.neural_temporal_graph is not None:
            # 使用神经时间图网络进行推理
            temporal_stamps = torch.tensor([self.current_interaction_round] * len(self.agent_execution_nodes), dtype=torch.float)
            agent_indices = torch.arange(len(self.agent_execution_nodes))
            
            # 通过神经时间图网络获取智能体嵌入
            reasoning_logits = self.neural_temporal_graph(
                enhanced_features, 
                self.role_adjacency_matrix, 
                temporal_stamps=temporal_stamps, 
                agent_indices=agent_indices
            )
        else:
            # 使用传统图网络
            reasoning_logits = self.compatibility_graph_network(enhanced_features, self.role_adjacency_matrix)
            
        # 解码决策特征
        reasoning_logits = self.decision_decoder(reasoning_logits)
        self.spatial_connection_logits = reasoning_logits @ reasoning_logits.t()
        self.spatial_connection_logits = self._min_max_normalize(torch.flatten(self.spatial_connection_logits))

        # 多轮交互推理
        for interaction_round in range(num_interaction_rounds):
            self.current_interaction_round = interaction_round
            reasoning_log_probs += self.construct_spatial_connections()
            reasoning_log_probs += self.construct_temporal_connections(interaction_round)
            
            # 计算智能体执行顺序（拓扑排序）
            agent_in_degrees = {agent_id: len(agent.spatial_predecessors) for agent_id, agent in self.agent_execution_nodes.items()}
            execution_queue = [agent_id for agent_id, degree in agent_in_degrees.items() if degree == 0]

            # 按拓扑顺序执行智能体
            while execution_queue:
                current_agent_id = execution_queue.pop(0)
                retry_attempts = 0
                
                while retry_attempts < max_retry_attempts:
                    try:
                        await asyncio.wait_for(
                            self.agent_execution_nodes[current_agent_id].async_execute_reasoning(task_input),
                            timeout=max_execution_time
                        )
                        break
                    except Exception as e:
                        print(f"Agent {current_agent_id} execution error: {e}")
                    retry_attempts += 1
                
                # 更新后续智能体的执行队列
                for successor_agent in self.agent_execution_nodes[current_agent_id].spatial_successors:
                    if successor_agent.node_id not in self.agent_execution_nodes.keys():
                        continue
                    agent_in_degrees[successor_agent.node_id] -= 1
                    if agent_in_degrees[successor_agent.node_id] == 0:
                        execution_queue.append(successor_agent.node_id)
            
            # 更新智能体记忆
            self.update_agent_memories()
            
        # 执行最终决策
        self.connect_to_decision_executor()
        await self.decision_executor.async_execute_reasoning(task_input)
        
        final_reasoning_results = self.decision_executor.execution_outputs
        if len(final_reasoning_results) == 0:
            final_reasoning_results.append("No reasoning result from decision executor")
            
        return final_reasoning_results, reasoning_log_probs
    
    def update_agent_memories(self):
        """更新智能体记忆"""
        # 同步所有智能体的轮次信息
        for agent_id, agent_node in self.agent_execution_nodes.items():
            agent_node.set_current_interaction_round(self.current_interaction_round)
            agent_node.update_interaction_memory()
        
        # 如果使用神经时间图网络，进行额外的记忆管理
        if self.use_neural_temporal_graph and self.neural_temporal_graph is not None:
            # 神经时间图网络的记忆更新在前向传播中自动完成
            pass
    
    def connect_to_decision_executor(self):
        """连接到决策执行器"""
        for agent_node in self.agent_execution_nodes.values():
            self.decision_executor.add_predecessor_connection(agent_node, 'spatial')
    
    def reset_neural_temporal_memories(self):
        """重置神经时间图网络的记忆"""
        if self.use_neural_temporal_graph and self.neural_temporal_graph is not None:
            self.neural_temporal_graph.reset_agent_memories()
            self.current_interaction_round = 0
    
    def get_neural_memory_snapshot(self):
        """获取神经网络的记忆状态快照"""
        if self.use_neural_temporal_graph and self.neural_temporal_graph is not None:
            return self.neural_temporal_graph.get_memory_snapshot()
        return None
    
    def optimize_connection_topology(self, pruning_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """优化连接拓扑结构"""
        if self.enable_spatial_optimization:
            active_spatial_connections = (self.spatial_connection_masks > 0).sum()
            inactive_spatial_connections = (self.spatial_connection_masks == 0).sum()
            prune_connection_count = torch.round(active_spatial_connections * pruning_ratio) if torch.round(active_spatial_connections * pruning_ratio) > 0 else 1
            
            connection_logits_copy = self.spatial_connection_logits.clone()
            min_logit_value = connection_logits_copy.min()
            connection_logits_copy[self.spatial_connection_masks == 0] = min_logit_value - 1.0
            
            sorted_connection_indices = torch.argsort(connection_logits_copy)
            prune_indices = sorted_connection_indices[:int(prune_connection_count + inactive_spatial_connections)]
            self.spatial_connection_masks[prune_indices] = 0
        
        if self.enable_temporal_optimization:
            active_temporal_connections = (self.temporal_connection_masks > 0).sum()
            inactive_temporal_connections = (self.temporal_connection_masks == 0).sum()
            prune_connection_count = torch.round(active_temporal_connections * pruning_ratio) if torch.round(active_temporal_connections * pruning_ratio) > 0 else 1
            
            connection_logits_copy = self.temporal_connection_logits.clone()
            min_logit_value = connection_logits_copy.min()
            connection_logits_copy[self.temporal_connection_masks == 0] = min_logit_value - 1.0
            
            sorted_connection_indices = torch.argsort(connection_logits_copy)
            prune_indices = sorted_connection_indices[:int(prune_connection_count + inactive_temporal_connections)]
            self.temporal_connection_masks[prune_indices] = 0
            
        return self.spatial_connection_masks, self.temporal_connection_masks

    def _min_max_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """最小-最大归一化"""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val == 0:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)
    
    # ========================================================================
    # FSM模式方法（新增）
    # ========================================================================
    
    def initialize_fsm_from_description(self, fsm_description: Dict[str, Any]):
        """
        从FSM描述初始化有限状态机
        
        Args:
            fsm_description: FSM描述字典，包含：
                - states: 状态列表
                - listeners: 监听关系
        
        示例:
            {
                'states': [
                    {'id': 0, 'name': 'Analyze', 'agent': 'agent_0', 'is_initial': True},
                    {'id': 1, 'name': 'Solve', 'agent': 'agent_1'},
                    {'id': 2, 'name': 'Verify', 'agent': 'agent_2', 'is_final': True}
                ],
                'listeners': {
                    0: ['agent_1', 'agent_2'],  # State 0输出传给agent_1和agent_2
                    1: ['agent_2'],              # State 1输出传给agent_2
                    2: []                        # State 2是最终状态，无需监听
                }
            }
        """
        agent_ids = list(self.agent_execution_nodes.keys())
        self.fsm_state_manager = FSMStateManager(agent_ids)
        
        # 添加所有状态
        for state_desc in fsm_description.get('states', []):
            self.fsm_state_manager.add_state(
                state_id=state_desc['id'],
                state_name=state_desc['name'],
                responsible_agent_id=state_desc['agent'],
                is_initial=state_desc.get('is_initial', False),
                is_final=state_desc.get('is_final', False),
                description=state_desc.get('description', '')
            )
        
        # 设置监听关系（通信路径）
        listeners_dict = fsm_description.get('listeners', {})
        for state_id, listener_ids in listeners_dict.items():
            self.fsm_state_manager.set_listeners(int(state_id), listener_ids)
        
        # 启用FSM模式
        self.use_fsm_mode = True
        
        print(f"✅ FSM initialized with {len(self.fsm_state_manager.states)} states")
        print(self.fsm_state_manager.get_state_summary())
    
    async def execute_fsm_reasoning(self,
                                    task_input: Dict[str, str],
                                    max_transitions: int = 10,
                                    max_retry_attempts: int = 3,
                                    max_execution_time: int = 600,
                                    fsm_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[str, torch.Tensor]:
        """
        执行FSM模式的推理
        
        核心逻辑：
        1. 从初始状态开始
        2. 执行当前状态负责的智能体
        3. 提取状态转移目标或最终答案
        4. 将输出传递给监听智能体
        5. 转移到下一状态
        6. 重复直到达到最终状态或最大转移次数
        
        Args:
            task_input: 任务输入
            max_transitions: 最大状态转移次数
            max_retry_attempts: 每个智能体的最大重试次数
            max_execution_time: 执行超时时间
        
        Returns:
            (final_answer, reasoning_log_probs)
        """
        if not self.use_fsm_mode or self.fsm_state_manager is None:
            raise ValueError("FSM mode is not enabled. Call initialize_fsm_from_description() first.")
        
        # 重置FSM到初始状态
        self.fsm_state_manager.reset()
        
        # 初始化推理对数概率（用于策略梯度）
        reasoning_log_probs = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        
        # 状态转移循环
        transition_count = 0
        last_state_output: Optional[str] = None  # 记录最后一个非终止状态的输出
        
        while transition_count < max_transitions:
            current_state = self.fsm_state_manager.get_current_state()
            
            if current_state is None:
                return "Error: No current state", reasoning_log_probs
            
            # 获取负责当前状态的智能体
            agent_node = self.agent_execution_nodes.get(current_state.responsible_agent_id)
            
            # 显示 agent 编号和名称
            if agent_node:
                agent_display = f"{current_state.responsible_agent_id} ({agent_node.agent_role})"
            else:
                agent_display = current_state.responsible_agent_id
            
            print(f"\n{'='*60}")
            print(f"State {current_state.state_id}: {current_state.state_name}")
            print(f"Responsible Agent: {agent_display}")
            print(f"{'='*60}")
            
            if agent_node is None:
                return f"Error: Agent {current_state.responsible_agent_id} not found", reasoning_log_probs
            
            # 执行智能体推理
            retry_attempts = 0
            output = None
            
            while retry_attempts < max_retry_attempts:
                try:
                    await asyncio.wait_for(
                        agent_node.async_execute_reasoning(
                            task_input,
                            is_final_state=current_state.is_final
                        ),
                        timeout=max_execution_time
                    )
                    
                    # 获取输出
                    if agent_node.execution_outputs:
                        output = agent_node.execution_outputs[-1]
                    break
                
                except Exception as e:
                    print(f"⚠️  Agent execution error (attempt {retry_attempts + 1}): {e}")
                    retry_attempts += 1
            
            if output is None:
                return "Error: Agent execution failed", reasoning_log_probs
            
            import re
            clean_output = re.sub(r'<\|[^>]+?\|>', '', output)
            print(f"\n🤖 Agent Output:\n{clean_output[:200]}...")
            last_state_output = output
            
            # 检查是否是最终状态
            if current_state.is_final:
                final_answer = self.fsm_state_manager.check_final_answer(output)
                if final_answer:
                    print(f"\n✅ Final Answer: {final_answer}")
                    return final_answer, reasoning_log_probs
                else:
                    # 最终状态但没有答案标记，直接返回输出
                    print(f"\n✅ Reached final state, returning output")
                    return output, reasoning_log_probs
            
            # 提取状态转移目标
            next_state_id = self.fsm_state_manager.extract_state_transition(output)
            
            # 如果从输出中无法提取状态转移标记，尝试使用 TGN 预测的概率来采样下一状态
            if next_state_id is None:
                print(f"\n⚠️  无法从 Agent 输出中提取状态转移标记 (<STATE_TRANS>: X)")
                print(f"   💡 将尝试使用 TGN 预测的状态转移概率进行采样...")
                
                # 尝试使用 TGN 预测的状态转移概率（如果可用）
                if fsm_outputs and 'transition_probs' in fsm_outputs:
                    transition_probs = fsm_outputs['transition_probs']
                    if transition_probs is not None and len(transition_probs) > 0:
                        # 从概率分布中采样下一状态
                        import torch.nn.functional as F
                        if isinstance(transition_probs, torch.Tensor):
                            # 确保概率分布有效
                            if transition_probs.dim() == 1:
                                # 使用当前状态的概率分布
                                sampled_idx = torch.multinomial(transition_probs, num_samples=1).item()
                                next_state_id = sampled_idx
                                print(f"   ✅ 使用 TGN 预测的状态转移概率采样")
                                print(f"   📊 采样结果: State {next_state_id} (概率: {transition_probs[sampled_idx]:.4f})")
                            else:
                                # 如果是矩阵，使用当前状态对应的行
                                if current_state.state_id < transition_probs.size(0):
                                    state_probs = transition_probs[current_state.state_id]
                                    sampled_idx = torch.multinomial(state_probs, num_samples=1).item()
                                    next_state_id = sampled_idx
                                    print(f"   ✅ 使用 TGN 预测的状态转移概率采样")
                                    print(f"   📊 采样结果: State {next_state_id} (概率: {state_probs[sampled_idx]:.4f})")
                                else:
                                    print(f"   ❌ TGN 预测的状态转移概率矩阵维度不匹配")
                    else:
                        print(f"   ❌ TGN 预测的状态转移概率为空或无效")
                else:
                    print(f"   ❌ 未找到 TGN 预测的状态转移概率 (fsm_outputs 中无 'transition_probs')")
                
                # 如果仍然无法确定下一状态，尝试转移到下一个顺序状态
                if next_state_id is None:
                    print(f"\n   ⚠️  无法使用 TGN 预测的状态转移概率")
                    print(f"   💡 将使用降级策略：转移到下一个顺序状态...")
                    
                    # 降级策略：转移到下一个状态（如果存在）
                    all_states = list(self.fsm_state_manager.states.keys())
                    current_idx = all_states.index(current_state.state_id) if current_state.state_id in all_states else -1
                    if current_idx >= 0 and current_idx < len(all_states) - 1:
                        next_state_id = all_states[current_idx + 1]
                        print(f"   ✅ 使用降级策略: State {current_state.state_id} → State {next_state_id}")
                    else:
                        print(f"   ❌ 降级策略失败：无法找到下一个顺序状态")
                        print(f"   ❌ 状态转移失败，终止推理")
                return "Error: No state transition found", reasoning_log_probs
            else:
                print(f"\n✅ 从 Agent 输出中成功提取状态转移标记")
                print(f"   📋 提取结果: State {current_state.state_id} → State {next_state_id}")
            
            print(f"\n→ Transitioning to State {next_state_id}")
            
            # 将输出传递给监听智能体（通信路径）
            listeners = self.fsm_state_manager.get_listeners(current_state.state_id)
            
            if listeners:
                print(f"📡 Broadcasting to listeners: {listeners}")
                
                for listener_id in listeners:
                    listener_node = self.agent_execution_nodes.get(listener_id)
                    if listener_node:
                        # 将当前状态的输出添加到监听者的上下文
                        message = f"\n[Message from {agent_node.agent_role} at State {current_state.state_id}]:\n{output}\n"
                        # 这里可以通过更新智能体的记忆或提示来传递信息
                        # 具体实现取决于AgentExecutionNode的接口
                        listener_node.add_predecessor_message(message)
            
            # 转移到下一状态
            success = self.fsm_state_manager.transition_to_state(next_state_id)
            
            if not success:
                return f"Error: Failed to transition to state {next_state_id}", reasoning_log_probs
            
            transition_count += 1
        
        # 达到最大转移次数
        print(f"\n⚠️  Reached maximum transitions ({max_transitions})")
        return "Error: Maximum transitions reached", reasoning_log_probs
    
    async def execute_fsm_reasoning_with_sampling(self,
                                                 task_input: str,
                                                 fsm_outputs: Dict[str, torch.Tensor],
                                                 fsm_manager: 'FSMStateManager',
                                                 fsm_tgn: Optional[Any] = None,
                                                 question_embedding: Optional[torch.Tensor] = None,
                                                 agent_features: Optional[torch.Tensor] = None,
                                                 context_features: Optional[torch.Tensor] = None,
                                                 enable_transition_prediction: bool = True,
                                                 enable_comm_sampling: bool = True,
                                                 max_transitions: int = 8,
                                                 max_retry_attempts: int = 3,
                                                 max_execution_time: int = 600,
                                                 temperature: float = 1.0,
                                                 verbose: bool = True,
                                                 phase: str = "train",
                                                 attack_injector: Optional[Any] = None) -> Tuple[str, torch.Tensor, bool, Dict[int, str]]:
        """
        ✨ 使用概率采样执行FSM推理（任务自适应）
        
        核心逻辑：
        1. 从TGN输出的transition_probs中采样下一状态（而不是依赖显式标记）
        2. 从TGN输出的listener_weights中采样通信路径
        3. 从总FSM中匹配选中的状态、转移、智能体
        
        Args:
            task_input: 任务输入（问题文本）
            fsm_outputs: TGN输出字典，包含：
                - 'transition_probs': [num_states] 状态转移概率分布
                - 'listener_weights': [num_states, num_agents] 监听关系权重
            fsm_manager: FSM状态管理器
            max_transitions: 最大状态转移次数
            max_retry_attempts: 每个智能体的最大重试次数
            max_execution_time: 执行超时时间
            temperature: 采样温度（1.0=原始分布，>1.0=更随机，<1.0=更确定）
            phase: 执行阶段 ("train", "val", "test")，用于日志区分
        
        Returns:
            (final_answer, reasoning_log_probs, reached_max_transitions, collected_agent_messages)
            - final_answer: 最终答案
            - reasoning_log_probs: 推理过程的对数概率（用于策略梯度）
            - reached_max_transitions: 是否达到最大转移次数（用于惩罚）
            - collected_agent_messages: ✨ 收集的智能体消息 {agent_id: message}（用于语义一致性检测）
        """
        if not self.use_fsm_mode or self.fsm_state_manager is None:
            raise ValueError("FSM mode is not enabled. Call initialize_fsm_from_description() first.")
        
        # 重置FSM到初始状态
        fsm_manager.reset()
        
        # ✨ 新增：收集智能体消息（用于语义一致性检测）
        collected_agent_messages = {}
        
        # ✨ 新增：收集实际采样的通信路径（用于动态图中心性计算）
        # 格式: [(source_agent_id, target_agent_id), ...]
        sampled_communication_edges = []

        # ✨ 每个新问题开始前，清空所有智能体在上一个问题中的记忆和监听消息
        for agent_node in self.agent_execution_nodes.values():
            # 清空前序消息（FSM监听得到的跨状态消息）
            try:
                agent_node.clear_predecessor_messages()
            except Exception:
                pass
            # 重置交互记忆与历史
            try:
                agent_node.reset_interaction_memory()
            except Exception:
                pass
            # 清空本地输入/输出缓存，避免被后续状态查询到旧结果
            agent_node.execution_inputs = []
            agent_node.execution_outputs = []
            agent_node.raw_task_inputs = []
            # 重置 LLM 会话历史，只保留系统提示，避免跨问题的对话累积
            reasoning_llm = getattr(agent_node, "reasoning_llm", None)
            reasoning_prompt = getattr(agent_node, "reasoning_prompt", None)
            if reasoning_llm is not None and reasoning_prompt is not None:
                try:
                    reasoning_llm.messages = [{"role": "system", "content": reasoning_prompt}]
                except Exception:
                    pass
        
        # 初始化推理对数概率（用于策略梯度）
        reasoning_log_probs = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        
        # ✨ 标记是否达到最大转移次数（用于惩罚）
        reached_max_transitions = False
        
        # 获取初始TGN输出（listener_weights是全局的，不需要重新计算）
        listener_weights = None
        if enable_comm_sampling:
            listener_weights = fsm_outputs.get("listener_weights")  # [num_states, num_agents]
            if listener_weights is None:
                raise ValueError("listener_weights not found in fsm_outputs")
        
        # 如果禁用状态转移预测，则不使用transition_probs/动态重算
        if enable_transition_prediction:
            # 如果提供了fsm_tgn，可以在每次状态转移时重新计算transition_probs
            # 否则使用初始的transition_probs（如果存在）
            use_dynamic_transition = (
                fsm_tgn is not None
                and question_embedding is not None
                and agent_features is not None
                and context_features is not None
            )
            
            # 初始transition_probs作为第一步/兜底分布
            transition_probs = fsm_outputs.get("transition_probs")  # [num_states] or [num_states, num_states]
            if transition_probs is None:
                raise ValueError("transition_probs not found in fsm_outputs and cannot compute dynamically")
        else:
            use_dynamic_transition = False
            transition_probs = None
        
        # 状态转移循环
        transition_count = 0
        last_state_output = None  # 记录最后一个非终止状态的输出，用于传递给最终状态
        
        while transition_count < max_transitions:
            current_state = fsm_manager.get_current_state()
            
            if current_state is None:
                return "Error: No current state", reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
            # 获取负责当前状态的智能体
            agent_node = self.agent_execution_nodes.get(current_state.responsible_agent_id)

            # 显示 agent 编号和名称
            if agent_node:
                agent_display = f"{current_state.responsible_agent_id} ({agent_node.agent_role})"
            else:
                agent_display = current_state.responsible_agent_id

            if verbose:
                print(f"\n{'='*60}")
                print(f"State {current_state.state_id}: {current_state.state_name}")
                print(f"Responsible Agent: {agent_display}")
            print(f"{'='*60}")
            
            if agent_node is None:
                return f"Error: Agent {current_state.responsible_agent_id} not found", reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
            # 执行智能体推理
            retry_attempts = 0
            output = None
            
            while retry_attempts < max_retry_attempts:
                try:
                    # ✨ 状态转移层面：传递上一个状态的直接输出（不附带历史）
                    # 与监听机制层面（predecessor_messages）独立运作
                    if last_state_output is not None:
                        # ✨ 最终状态需要完整的上一状态输出（用于代码精炼等任务）
                        # 非最终状态使用较短的截断以节省token
                        if current_state.is_final:
                            # 最终状态：不截断，保留完整输出（代码生成任务需要完整代码）
                            truncated_prev_output = str(last_state_output)
                        else:
                            # 中间状态：限制长度以节省token
                            MAX_PREVIOUS_OUTPUT_LENGTH = 500
                        truncated_prev_output = str(last_state_output)
                        if len(truncated_prev_output) > MAX_PREVIOUS_OUTPUT_LENGTH:
                            truncated_prev_output = truncated_prev_output[:MAX_PREVIOUS_OUTPUT_LENGTH] + "...[truncated]"
                        
                        current_input = (
                            f"{task_input}\n\n"
                            f"[Previous State Output]\n"
                            f"{truncated_prev_output}"
                        )
                    else:
                        current_input = task_input
                    
                    await asyncio.wait_for(
                        agent_node.async_execute_reasoning(
                            current_input,
                            is_final_state=current_state.is_final
                        ),
                        timeout=max_execution_time
                    )
                    
                    # 获取输出
                    if agent_node.execution_outputs:
                        output = agent_node.execution_outputs[-1]
                    break
                
                except Exception as e:
                    print(f"⚠️  Agent execution error (attempt {retry_attempts + 1}): {e}")
                    retry_attempts += 1
            
            if output is None:
                return "Error: Agent execution failed", reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
            # ✨ 应用消息污染攻击（如果启用）
            # 被攻击的智能体的输出会被污染，影响传递给其他智能体的消息
            if attack_injector is not None and hasattr(attack_injector, 'pollute_message'):
                # 获取当前智能体的ID（从 responsible_agent_id 中提取数字部分）
                agent_id_str = current_state.responsible_agent_id
                try:
                    agent_idx = int(agent_id_str.split('_')[-1]) if '_' in agent_id_str else int(agent_id_str)
                except (ValueError, IndexError):
                    agent_idx = 0
                
                # 检查是否需要污染消息
                if attack_injector.should_pollute_message(agent_idx):
                    original_output = output
                    output = attack_injector.pollute_message(
                        agent_id=agent_idx,
                        original_message=output,
                        pollution_type=attack_injector.attack_type
                    )
                    if verbose:
                        print(f"  🔴 消息已被污染 (Agent {agent_idx}, 攻击类型: {attack_injector.attack_type})")
            
            # 记录最后一个状态的输出（用于传递给最终状态）
            last_state_output = output
            
            # ✨ 收集智能体消息（用于语义一致性检测）
            try:
                agent_id_str = current_state.responsible_agent_id
                agent_idx = int(agent_id_str.split('_')[-1]) if '_' in agent_id_str else int(agent_id_str)
                collected_agent_messages[agent_idx] = output
            except (ValueError, IndexError):
                pass

            if verbose:
                print(f"\n🤖 Agent Output:\n{output[:200]}...")
            
            # 检查是否是最终状态
            if current_state.is_final:
                # ✨ 将采样的通信边添加到返回数据中（用于动态图中心性计算）
                collected_agent_messages['_communication_edges'] = sampled_communication_edges
                
                final_answer = fsm_manager.check_final_answer(output)
                if final_answer:
                    if verbose:
                        print(f"\n✅ Final Answer: {final_answer}")
                    return final_answer, reasoning_log_probs, reached_max_transitions, collected_agent_messages
                else:
                    # 最终状态但没有答案标记，直接返回输出
                    if verbose:
                        print(f"\n✅ Reached final state, returning output")
                    return output, reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
            # ✨ 选择下一状态
            current_state_id = current_state.state_id

            # --- 1) 状态转移选择 ---
            if enable_transition_prediction:
                # 动态转移：每次状态转移时重新计算 transition_probs
                if use_dynamic_transition:
                    num_agents = agent_features.size(0)
                    if num_agents > 1:
                        edge_indices = [(i, j) for i in range(num_agents) for j in range(num_agents) if i != j]
                        edge_index = (
                            torch.tensor(edge_indices, dtype=torch.long, device=agent_features.device).t()
                            if edge_indices
                            else torch.zeros((2, 0), dtype=torch.long, device=agent_features.device)
                        )
                    else:
                        edge_index = torch.zeros((2, 0), dtype=torch.long, device=agent_features.device)

                    current_fsm_outputs = fsm_tgn(
                        agent_features=agent_features,
                        communication_topology=edge_index,
                        context_features=context_features,
                        current_state_id=current_state_id,
                        question_embedding=question_embedding,
                    )
                    transition_probs = current_fsm_outputs.get("transition_probs")
                    if transition_probs is None:
                        raise ValueError("Failed to compute transition_probs dynamically")

                if transition_probs is None:
                    raise ValueError("transition_probs is None (enable_transition_prediction=True)")

                # 获取当前状态的转移概率分布
                if transition_probs.dim() == 1:
                    state_transition_probs = transition_probs
                elif transition_probs.dim() == 2:
                    state_transition_probs = transition_probs[current_state_id]
                else:
                    raise ValueError(f"Unexpected transition_probs shape: {transition_probs.shape}")

                # --- 2) 初始状态约束 + 采样 ---
                initial_state = fsm_manager.get_initial_state()
                if initial_state is not None and current_state_id == initial_state.state_id:
                    final_state_ids = [
                        s.state_id for s in fsm_manager.states.values()
                        if s.is_final and s.state_id != current_state_id
                    ]
                    if final_state_ids:
                        mask = torch.ones_like(state_transition_probs)
                        for fid in final_state_ids:
                            if 0 <= fid < mask.size(0):
                                mask[fid] = 0.0
                        masked_probs = state_transition_probs * mask
                        if masked_probs.sum() > 0:
                            state_transition_probs = masked_probs / masked_probs.sum()
                            if verbose:
                                print("\n⚖️  初始状态约束：已暂时屏蔽直达最终状态的转移概率")
                        else:
                            if verbose:
                                print("\n⚠️  初始状态约束导致所有概率为0，回退到原始分布")

                # 温度调节
                if temperature != 1.0:
                    logits = torch.log(state_transition_probs + 1e-8) / temperature
                    state_transition_probs = torch.nn.functional.softmax(logits, dim=-1)

                # 采样下一状态（默认不采样自环）
                probs = state_transition_probs.clone()
                if current_state_id < probs.size(0):
                    probs[current_state_id] = 0.0
                if probs.sum() <= 0:
                    probs = state_transition_probs

                sampled_next_state_id = torch.multinomial(probs.unsqueeze(0), num_samples=1).item()

                # 计算采样动作的对数概率，用于策略梯度损失
                sampled_prob = probs[sampled_next_state_id]
                sampled_log_prob = torch.log(sampled_prob + 1e-8)
                reasoning_log_probs = reasoning_log_probs + sampled_log_prob
            else:
                # 消融：不使用TGN转移预测，改为固定状态推进 i -> i+1
                all_state_ids = sorted(list(fsm_manager.states.keys()))
                if not all_state_ids:
                    return (
                        "Error: No FSM states defined",
                        reasoning_log_probs,
                        reached_max_transitions,
                        collected_agent_messages,
                    )
                if current_state_id in all_state_ids:
                    idx = all_state_ids.index(current_state_id)
                    sampled_next_state_id = all_state_ids[min(idx + 1, len(all_state_ids) - 1)]
                else:
                    sampled_next_state_id = all_state_ids[-1]

                if verbose:
                    print("\n📊 消融：禁用转移预测，使用固定状态推进")
                    print(f"\n→ 固定转移: State {current_state_id} → State {sampled_next_state_id}")

            # ✨ 仅记录采样结果（状态名称）
            current_state_name = current_state.state_name if current_state else f"State_{current_state_id}"
            sampled_state_obj = fsm_manager.states.get(sampled_next_state_id)
            sampled_state_name = sampled_state_obj.state_name if sampled_state_obj else f"State_{sampled_next_state_id}"

            if phase == "val":
                log_prefix = "✅ 验证"
            elif phase == "test":
                log_prefix = "🧪 测试"
            else:
                log_prefix = "🎯 训练"

            if verbose:
                if enable_transition_prediction:
                    print("\n📊 使用 TGN 预测的状态转移概率进行采样")
                    print(f"   📈 当前状态 {current_state_id} ({current_state_name}) 的转移概率分布:")
                    for state_id, prob in enumerate(state_transition_probs):
                        marker = "👉" if state_id == sampled_next_state_id else "  "
                        state_obj = fsm_manager.states.get(state_id)
                        state_name = state_obj.state_name if state_obj else f"State_{state_id}"
                        print(f"   {marker} State {state_id} ({state_name}): {prob:.4f}")
                    print(
                        f"\n→ 采样结果: State {current_state_id} ({current_state_name}) "
                        f"→ State {sampled_next_state_id} ({sampled_state_name}) "
                        f"(概率: {state_transition_probs[sampled_next_state_id]:.4f})"
                    )
                else:
                    print(
                        f"\n→ 固定转移: State {current_state_id} ({current_state_name}) "
                        f"→ State {sampled_next_state_id} ({sampled_state_name})"
                    )
            else:
                print(
                    f"{log_prefix} 状态转移: State {current_state_id} ({current_state_name}) "
                    f"→ State {sampled_next_state_id} ({sampled_state_name})"
                )

            # ✨ 选择通信路径（监听者）
            sampled_listeners: List[str] = []
            if enable_comm_sampling:
                if listener_weights is None:
                    raise ValueError("listener_weights is None while enable_comm_sampling=True")

                if listener_weights.dim() == 2:
                    current_listener_weights = listener_weights[current_state_id]  # [num_agents]
                elif listener_weights.dim() == 1:
                    current_listener_weights = listener_weights
                else:
                    raise ValueError(f"Unexpected listener_weights shape: {listener_weights.shape}")

                if temperature != 1.0:
                    logits = torch.log(current_listener_weights + 1e-8) / temperature
                    current_listener_weights = torch.nn.functional.softmax(logits, dim=-1)

                num_agents = len(fsm_manager.agent_ids)
                sampled_agent_indices = torch.multinomial(
                    current_listener_weights.unsqueeze(0),
                    num_samples=min(num_agents, 3),
                    replacement=False,
                ).squeeze(0)

                sampled_listener_indices = sampled_agent_indices.detach().cpu().numpy().tolist()
                for idx in sampled_listener_indices:
                    if idx < len(fsm_manager.agent_ids):
                        sampled_listeners.append(fsm_manager.agent_ids[idx])
            else:
                sampled_listeners = []
                if verbose or phase in ["val", "test"]:
                    print("   📡 消融：禁用通信采样，关闭监听广播（0个监听者）")
            
            # ✨ 更新FSM管理器中的监听关系（用于损失函数计算）
            # 注意：即便没有监听者，也要显式写入空列表，避免残留上一题/上一步的listeners
            fsm_manager.set_listeners(current_state_id, sampled_listeners)
            
            if sampled_listeners:
                # 显示监听者的编号和名称
                listener_display = []
                for listener_id in sampled_listeners:
                    listener_node = self.agent_execution_nodes.get(listener_id)
                    if listener_node:
                        listener_display.append(f"{listener_id} ({listener_node.agent_role})")
                    else:
                        listener_display.append(listener_id)
                
                # 根据verbose和phase决定是否显示监听者
                if verbose or phase in ["val", "test"]:
                    print(f"   📡 监听者: {listener_display}")
                
                # ✨ 记录实际采样的通信边（用于动态图中心性计算）
                # 当前状态的agent → 所有监听者
                current_agent_id = current_state.responsible_agent_id
                for listener_id in sampled_listeners:
                    if listener_id != current_agent_id:  # 避免自环
                        sampled_communication_edges.append((current_agent_id, listener_id))
                
                for listener_id in sampled_listeners:
                    listener_node = self.agent_execution_nodes.get(listener_id)
                    if listener_node:
                        # ✨ 成本优化：截断消息长度，避免监听消息过长导致token爆炸
                        MAX_MESSAGE_LENGTH = 400  # 每条监听消息最多400字符（避免上下文爆炸）
                        truncated_output = str(output)
                        if len(truncated_output) > MAX_MESSAGE_LENGTH:
                            truncated_output = truncated_output[:MAX_MESSAGE_LENGTH] + "...[truncated]"
                        message = f"\n[Message from {agent_node.agent_role} at State {current_state.state_id}]:\n{truncated_output}\n"
                        listener_node.add_predecessor_message(message)
            
            # 执行状态转移（完全由TGN概率控制路径，不再用条件阻塞）
            success = fsm_manager.transition_to_state(sampled_next_state_id)
            
            if not success:
                sampled_state_obj = fsm_manager.states.get(sampled_next_state_id)
                sampled_state_name = sampled_state_obj.state_name if sampled_state_obj else f"State_{sampled_next_state_id}"
                if verbose:
                    print(f"⚠️  无法转移到状态 {sampled_next_state_id} ({sampled_state_name})，尝试使用概率最高的状态")
                # 如果采样失败：
                # - 正常模式：使用概率最高的状态
                # - 消融模式（固定推进）：使用“最后一个状态”作为兜底
                if enable_transition_prediction:
                    fallback_state_id = torch.argmax(state_transition_probs).item()
                else:
                    all_state_ids = sorted(list(fsm_manager.states.keys()))
                    fallback_state_id = all_state_ids[-1] if all_state_ids else sampled_next_state_id
                success = fsm_manager.transition_to_state(fallback_state_id)
                if not success:
                    return f"Error: Failed to transition to state {sampled_next_state_id}", reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
            transition_count += 1
        
        # ✨ 达到最大转移次数：如果还没到最终状态，强制跳到最终状态并给出结果
        # 标记为达到最大转移次数，用于后续惩罚
        reached_max_transitions = True
        
        current_state = fsm_manager.get_current_state()
        current_state_name = current_state.state_name if current_state else "Unknown"
        if verbose:
            print(f"\n⚠️  Reached maximum transitions ({max_transitions})")
            print(f"  📍 当前位置: State {current_state.state_id} ({current_state_name})")

        # 查找一个最终状态
        final_states = [s for s in fsm_manager.states.values() if s.is_final]
        if not final_states:
            if verbose:
                print("  ❌ 没有定义最终状态，返回错误")
            return "Error: Maximum transitions reached and no final state defined", reasoning_log_probs, reached_max_transitions, collected_agent_messages
        
        final_state = final_states[0]
        print(f"  ⚠️  未在限制步数内到达最终状态，强制跳转到最终状态 State {final_state.state_id} ({final_state.state_name})")
        fsm_manager.transition_to_state(final_state.state_id)
        
        # 执行最终状态的智能体一次，给出最后结果
        agent_node = self.agent_execution_nodes.get(final_state.responsible_agent_id)
        if agent_node is None:
            return f"Error: Final agent {final_state.responsible_agent_id} not found", reasoning_log_probs, reached_max_transitions, collected_agent_messages
        
        retry_attempts = 0
        output = None
        while retry_attempts < max_retry_attempts:
            try:
                # ✨ 最终状态：传递上一个状态的输出作为推理依据
                # ✨ 不截断，保留完整输出（代码生成任务需要完整代码）
                if last_state_output is not None:
                    final_input = (
                        f"{task_input}\n\n"
                        f"[Previous Reasoning]\n"
                        f"{last_state_output}"
                    )
                else:
                    final_input = task_input
                
                await asyncio.wait_for(
                    agent_node.async_execute_reasoning(final_input, is_final_state=True),
                    timeout=max_execution_time
                )
                if agent_node.execution_outputs:
                    output = agent_node.execution_outputs[-1]
                break
            except Exception as e:
                print(f"⚠️  Final agent execution error (attempt {retry_attempts + 1}): {e}")
                retry_attempts += 1
        
        if output is None:
            collected_agent_messages['_communication_edges'] = sampled_communication_edges
            return "Error: Final agent execution failed", reasoning_log_probs, reached_max_transitions, collected_agent_messages
        
        import re
        clean_output = re.sub(r'<\|[^>]+?\|>', '', output)
        print(f"\n🤖 Final Agent Output:\n{clean_output[:200]}...")
        
        # ✨ 将采样的通信边添加到返回数据中
        collected_agent_messages['_communication_edges'] = sampled_communication_edges
        
        final_answer = fsm_manager.check_final_answer(output)
        if final_answer:
            print(f"\n✅ Forced Final Answer: {final_answer}")
            return final_answer, reasoning_log_probs, reached_max_transitions, collected_agent_messages
        else:
            print(f"\n✅ Forced final state reached, returning raw output")
            return output, reasoning_log_probs, reached_max_transitions, collected_agent_messages
    
    def learn_fsm_from_data(self, 
                           training_samples: List[Dict[str, Any]],
                           num_states: int = 4) -> Dict[str, Any]:
        """
        从训练数据中学习FSM结构
        
        使用TGN学习：
        1. 最优状态数量
        2. 状态转移概率
        3. 监听关系（通信路径）
        
        Args:
            training_samples: 训练样本列表
            num_states: 状态数量
        
        Returns:
            学习到的FSM描述
        """
        # TODO: 实现基于TGN的FSM学习
        # 这里先返回一个默认的FSM结构
        
        agent_ids = list(self.agent_execution_nodes.keys())
        
        # 默认FSM：状态数 = 智能体数
        fsm_description = {
            'states': [],
            'listeners': {}
        }
        
        for i, agent_id in enumerate(agent_ids[:num_states]):
            state = {
                'id': i,
                'name': f"State_{i}",
                'agent': agent_id,
                'is_initial': (i == 0),
                'is_final': (i == num_states - 1)
            }
            fsm_description['states'].append(state)
            
            # 默认监听关系：当前状态输出传给下一个状态的智能体
            if i < num_states - 1:
                fsm_description['listeners'][i] = [agent_ids[i + 1]]
            else:
                fsm_description['listeners'][i] = []
        
        return fsm_description


# 导出主要类
__all__ = ['MultiAgentTopologyManager']

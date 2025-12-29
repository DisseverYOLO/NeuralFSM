"""
FSM-TGN: Temporal Graph Network for Finite State Machine Learning
FSM时间图网络：用于学习有限状态机的状态转移和监听关系

扩展NeuralTemporalGraph以支持FSM特定的学习任务：
1. 学习状态转移概率矩阵
2. 学习监听关系权重矩阵
3. 学习状态-智能体最优匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph


class FSMTransitionPredictor(nn.Module):
    """
    FSM状态转移预测器
    
    学习状态转移概率：P(State_j | State_i, context, question)
    """
    
    def __init__(self, 
                 state_feature_dim: int,
                 question_embedding_dim: int = 384,  # Sentence Transformer默认维度
                 hidden_dim: int = 128,
                 num_states: int = 4):
        super().__init__()
        
        self.num_states = num_states
        self.state_feature_dim = state_feature_dim
        self.question_embedding_dim = question_embedding_dim
        
        # 状态特征编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 问题嵌入编码器（将问题嵌入映射到hidden_dim）
        self.question_encoder = nn.Sequential(
            nn.Linear(question_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 状态转移预测网络
        # 输入：当前状态特征 + 任务上下文 + 问题嵌入
        # 输出：下一个状态的概率分布
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # state + context + question
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_states)
        )
        
    def forward(self, 
                current_state_features: torch.Tensor,
                context_features: torch.Tensor,
                question_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测状态转移概率
        
        Args:
            current_state_features: [batch_size, state_feature_dim] 当前状态特征
            context_features: [batch_size, state_feature_dim] 任务上下文特征
            question_embedding: [batch_size, question_embedding_dim] 问题嵌入（可选）
        
        Returns:
            transition_probs: [batch_size, num_states] 转移到各状态的概率
        """
        # 编码状态特征
        state_encoded = self.state_encoder(current_state_features)
        context_encoded = self.state_encoder(context_features)
        
        # 编码问题嵌入（如果提供）
        if question_embedding is not None:
            # 确保 question_embedding 是正确的维度 [batch_size, question_embedding_dim]
            if question_embedding.dim() == 1:
                # 如果是 1D，添加 batch 维度
                question_embedding = question_embedding.unsqueeze(0)
            elif question_embedding.dim() > 2:
                # 如果是 3D 或更高，展平或取第一个
                question_embedding = question_embedding.view(-1, self.question_embedding_dim)[:1]
            
            # 确保维度匹配
            if question_embedding.size(-1) != self.question_embedding_dim:
                # 如果维度不匹配，使用线性层进行适配
                if not hasattr(self, '_question_dim_adapter'):
                    self._question_dim_adapter = nn.Linear(question_embedding.size(-1), self.question_embedding_dim).to(question_embedding.device)
                question_embedding = self._question_dim_adapter(question_embedding)
            
            question_encoded = self.question_encoder(question_embedding)
            # 拼接：状态 + 上下文 + 问题
            combined = torch.cat([state_encoded, context_encoded, question_encoded], dim=-1)
        else:
            # 兼容旧代码：如果没有问题嵌入，只使用状态和上下文
            combined = torch.cat([state_encoded, context_encoded], dim=-1)
            # 需要调整predictor的输入维度，但为了兼容性，我们使用一个适配层
            if not hasattr(self, '_no_question_adapter'):
                self._no_question_adapter = nn.Linear(hidden_dim * 2, hidden_dim * 3).to(combined.device)
            combined = self._no_question_adapter(combined)
        
        logits = self.transition_predictor(combined)
        
        # Softmax得到概率分布
        transition_probs = F.softmax(logits, dim=-1)
        
        return transition_probs
    
    def sample_next_state(self,
                          transition_probs: torch.Tensor,
                          temperature: float = 1.0) -> torch.Tensor:
        """
        从转移概率分布中采样下一个状态
        
        Args:
            transition_probs: [batch_size, num_states] 转移概率分布
            temperature: 采样温度（1.0=原始分布，>1.0=更随机，<1.0=更确定）
        
        Returns:
            sampled_states: [batch_size] 采样的状态索引
        """
        # 温度调节
        if temperature != 1.0:
            logits = torch.log(transition_probs + 1e-8) / temperature
            transition_probs = F.softmax(logits, dim=-1)
        
        # 采样
        sampled_states = torch.multinomial(transition_probs, num_samples=1).squeeze(-1)
        return sampled_states
    
    def compute_transition_matrix(self,
                                  state_features: torch.Tensor,
                                  context_features: torch.Tensor,
                                  question_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算完整的状态转移概率矩阵
        
        Args:
            state_features: [num_states, state_feature_dim] 所有状态的特征
            context_features: [state_feature_dim] 任务上下文
            question_embedding: [question_embedding_dim] 问题嵌入（可选）
        
        Returns:
            transition_matrix: [num_states, num_states] 状态转移概率矩阵
        """
        num_states = state_features.size(0)
        transition_matrix = torch.zeros(num_states, num_states, device=state_features.device)
        
        # 扩展context到所有状态
        context_expanded = context_features.unsqueeze(0).expand(num_states, -1)
        
        # 扩展question_embedding到所有状态（如果提供）
        if question_embedding is not None:
            question_expanded = question_embedding.unsqueeze(0).expand(num_states, -1)
        else:
            question_expanded = None
        
        # 对每个状态计算转移概率
        for i in range(num_states):
            current_state = state_features[i].unsqueeze(0)
            q_emb = question_expanded[i].unsqueeze(0) if question_expanded is not None else None
            transition_probs = self.forward(current_state, context_expanded[i].unsqueeze(0), q_emb)
            transition_matrix[i] = transition_probs.squeeze(0)
        
        return transition_matrix


class FSMListenerPredictor(nn.Module):
    """
    FSM监听关系预测器
    
    学习监听权重：W(State_i -> Agent_j | question)
    """
    
    def __init__(self,
                 state_feature_dim: int,
                 agent_feature_dim: int,
                 question_embedding_dim: int = 384,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.state_feature_dim = state_feature_dim
        self.agent_feature_dim = agent_feature_dim
        self.question_embedding_dim = question_embedding_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 智能体编码器
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 问题嵌入编码器
        self.question_encoder = nn.Sequential(
            nn.Linear(question_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 监听关系预测（基于注意力机制，融合问题信息）
        self.attention_query = nn.Linear(hidden_dim * 2, hidden_dim)  # state + question
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self,
                state_features: torch.Tensor,
                agent_features: torch.Tensor,
                question_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测监听关系权重矩阵
        
        Args:
            state_features: [num_states, state_feature_dim] 状态特征
            agent_features: [num_agents, agent_feature_dim] 智能体特征
            question_embedding: [question_embedding_dim] 问题嵌入（可选）
        
        Returns:
            listener_weights: [num_states, num_agents] 监听关系权重
        """
        num_states = state_features.size(0)
        num_agents = agent_features.size(0)
        
        # 编码状态和智能体特征
        state_encoded = self.state_encoder(state_features)  # [num_states, hidden_dim]
        agent_encoded = self.agent_encoder(agent_features)  # [num_agents, hidden_dim]
        
        # 融合问题信息到状态特征
        if question_embedding is not None:
            question_encoded = self.question_encoder(question_embedding)  # [hidden_dim]
            # 将问题嵌入扩展到所有状态
            question_expanded = question_encoded.unsqueeze(0).expand(num_states, -1)  # [num_states, hidden_dim]
            # 拼接状态和问题特征
            state_question_combined = torch.cat([state_encoded, question_expanded], dim=-1)  # [num_states, hidden_dim*2]
            queries = self.attention_query(state_question_combined)  # [num_states, hidden_dim]
        else:
            # 兼容旧代码：如果没有问题嵌入，只使用状态特征
            if not hasattr(self, '_no_question_query'):
                self._no_question_query = nn.Linear(hidden_dim, hidden_dim).to(state_encoded.device)
            queries = self._no_question_query(state_encoded)
        
        keys = self.attention_key(agent_encoded)  # [num_agents, hidden_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.t())  # [num_states, num_agents]
        attention_scores = attention_scores / (queries.size(-1) ** 0.5)  # scaled dot-product
        
        # Softmax归一化（每个状态的监听权重和为1）
        listener_weights = F.softmax(attention_scores, dim=-1)
        
        return listener_weights
    
    def sample_listeners(self,
                        listener_weights: torch.Tensor,
                        temperature: float = 1.0,
                        num_samples: Optional[int] = None) -> torch.Tensor:
        """
        从监听权重分布中采样监听关系
        
        Args:
            listener_weights: [num_states, num_agents] 监听权重分布
            temperature: 采样温度（1.0=原始分布，>1.0=更随机，<1.0=更确定）
            num_samples: 每个状态采样的监听者数量（None表示采样所有可能的监听者）
        
        Returns:
            listener_mask: [num_states, num_agents] 二进制监听掩码（1=监听，0=不监听）
        """
        num_states, num_agents = listener_weights.shape
        listener_mask = torch.zeros_like(listener_weights)
        
        # 温度调节
        if temperature != 1.0:
            logits = torch.log(listener_weights + 1e-8) / temperature
            listener_weights = F.softmax(logits, dim=-1)
        
        # 对每个状态采样监听者
        for state_idx in range(num_states):
            if num_samples is None:
                # 采样所有可能的监听者（根据概率）
                sampled_agents = torch.multinomial(listener_weights[state_idx], 
                                                   num_samples=num_agents, 
                                                   replacement=True)
                # 去重并设置掩码
                unique_agents = torch.unique(sampled_agents)
                listener_mask[state_idx, unique_agents] = 1.0
            else:
                # 采样固定数量的监听者
                sampled_agents = torch.multinomial(listener_weights[state_idx], 
                                                   num_samples=min(num_samples, num_agents), 
                                                   replacement=False)
                listener_mask[state_idx, sampled_agents] = 1.0
        
        return listener_mask
    
    def compute_binary_listener_mask(self,
                                     listener_weights: torch.Tensor,
                                     threshold: float = 0.1) -> torch.Tensor:
        """
        将监听权重转换为二进制掩码
        
        Args:
            listener_weights: [num_states, num_agents] 监听权重
            threshold: 阈值，权重大于此值的才认为是监听关系
        
        Returns:
            listener_mask: [num_states, num_agents] 二进制监听掩码
        """
        return (listener_weights > threshold).float()


class FSMStateAgentMatcher(nn.Module):
    """
    FSM状态-智能体匹配器
    
    学习最优的状态-智能体分配
    """
    
    def __init__(self,
                 state_feature_dim: int,
                 agent_feature_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 智能体编码器
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 兼容性评分网络
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self,
                state_features: torch.Tensor,
                agent_features: torch.Tensor) -> torch.Tensor:
        """
        计算状态-智能体兼容性矩阵
        
        Args:
            state_features: [num_states, state_feature_dim] 状态特征
            agent_features: [num_agents, agent_feature_dim] 智能体特征
        
        Returns:
            compatibility_matrix: [num_states, num_agents] 兼容性分数
        """
        num_states = state_features.size(0)
        num_agents = agent_features.size(0)
        
        # 编码
        state_encoded = self.state_encoder(state_features)  # [num_states, hidden_dim]
        agent_encoded = self.agent_encoder(agent_features)  # [num_agents, hidden_dim]
        
        # 计算所有状态-智能体配对的兼容性
        compatibility_matrix = torch.zeros(num_states, num_agents, device=state_features.device)
        
        for i in range(num_states):
            for j in range(num_agents):
                paired_features = torch.cat([state_encoded[i], agent_encoded[j]], dim=-1)
                compatibility_matrix[i, j] = self.compatibility_scorer(paired_features.unsqueeze(0)).squeeze()
        
        return compatibility_matrix
    
    def get_optimal_assignment(self,
                              compatibility_matrix: torch.Tensor) -> List[int]:
        """
        基于兼容性矩阵得到最优分配
        
        Args:
            compatibility_matrix: [num_states, num_agents] 兼容性矩阵
        
        Returns:
            assignment: [num_states] 每个状态分配的智能体索引
        """
        # 贪心分配：每个状态选择兼容性最高的智能体
        assignment = torch.argmax(compatibility_matrix, dim=-1).tolist()
        return assignment


class FSMTemporalGraph(nn.Module):
    """
    FSM时间图网络（整合模块）
    
    集成三个预测器：
    1. FSMTransitionPredictor - 状态转移
    2. FSMListenerPredictor - 监听关系
    3. FSMStateAgentMatcher - 状态-智能体匹配
    """
    
    def __init__(self,
                 agent_feature_dim: int = 256,
                 state_feature_dim: int = 256,
                 question_embedding_dim: int = 384,
                 num_states: int = 4,
                 num_agents: int = 4,
                 hidden_dim: int = 128,
                 use_base_tgn: bool = True,
                 **tgn_kwargs):
        super().__init__()
        
        self.agent_feature_dim = agent_feature_dim
        self.state_feature_dim = state_feature_dim
        self.question_embedding_dim = question_embedding_dim
        self.num_states = num_states
        self.num_agents = num_agents
        
        # 基础TGN（可选）
        self.use_base_tgn = use_base_tgn
        if use_base_tgn:
            self.base_tgn = NeuralTemporalGraph(
                agent_feature_dim=agent_feature_dim,
                **tgn_kwargs
            )
        else:
            self.base_tgn = None
        
        # FSM特定的学习模块
        self.transition_predictor = FSMTransitionPredictor(
            state_feature_dim=state_feature_dim,
            question_embedding_dim=question_embedding_dim,
            hidden_dim=hidden_dim,
            num_states=num_states
        )
        
        self.listener_predictor = FSMListenerPredictor(
            state_feature_dim=state_feature_dim,
            agent_feature_dim=agent_feature_dim,
            question_embedding_dim=question_embedding_dim,
            hidden_dim=hidden_dim
        )
        
        self.state_agent_matcher = FSMStateAgentMatcher(
            state_feature_dim=state_feature_dim,
            agent_feature_dim=agent_feature_dim,
            hidden_dim=hidden_dim
        )
        
        # 状态特征表示（可学习）
        self.state_embeddings = nn.Parameter(torch.randn(num_states, state_feature_dim))
        
    def forward(self,
                agent_features: torch.Tensor,
                communication_topology: torch.Tensor,
                current_state_id: Optional[int] = None,
                context_features: Optional[torch.Tensor] = None,
                question_embedding: Optional[torch.Tensor] = None,
                **tgn_kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            agent_features: [num_agents, agent_feature_dim] 智能体特征
            communication_topology: [2, num_edges] 通信拓扑
            current_state_id: 当前状态ID
            context_features: [state_feature_dim] 任务上下文特征
            question_embedding: [question_embedding_dim] 问题嵌入（可选）
            **tgn_kwargs: 传递给基础TGN的其他参数
        
        Returns:
            outputs: {
                'agent_features': 演化后的智能体特征,
                'transition_probs': 状态转移概率,
                'listener_weights': 监听关系权重,
                'state_agent_compatibility': 状态-智能体兼容性
            }
        """
        outputs = {}
        
        # 1. 基础TGN演化智能体特征（可选）
        protection_data = None  # ✨ 用于存储保护数据
        
        if self.base_tgn is not None:
            # ✨ 检查 base_tgn 是否是 ProtectedTGN（支持额外参数）
            from neural_fsm_mas.defense_mechanisms import ProtectedTGN
            is_protected_tgn = isinstance(self.base_tgn, ProtectedTGN)
            
            if is_protected_tgn:
                # ProtectedTGN 支持所有参数
                base_tgn_output = self.base_tgn(
                agent_features,
                communication_topology,
                **tgn_kwargs
            )
                # ProtectedTGN 返回 Dict，需要提取 agent_features 和 protection_data
                if isinstance(base_tgn_output, dict):
                    evolved_agent_features = base_tgn_output.get('agent_features', agent_features)
                    protection_data = base_tgn_output.get('protection_data', None)  # ✨ 提取保护数据
                else:
                    evolved_agent_features = base_tgn_output
            else:
                # NeuralTemporalGraph 只支持特定参数，过滤掉不支持的参数
                valid_params = {'communication_attributes', 'temporal_stamps', 'agent_indices'}
                filtered_kwargs = {k: v for k, v in tgn_kwargs.items() if k in valid_params}
                evolved_agent_features = self.base_tgn(
                    agent_features,
                    communication_topology,
                    **filtered_kwargs
                )
        else:
            evolved_agent_features = agent_features
        
        outputs['agent_features'] = evolved_agent_features
        
        # ✨ 如果有保护数据，传递到输出
        if protection_data is not None:
            outputs['protection_data'] = protection_data
        
        # 2. 预测状态转移概率
        if current_state_id is not None and context_features is not None:
            current_state_features = self.state_embeddings[current_state_id].unsqueeze(0)
            q_emb = question_embedding.unsqueeze(0) if question_embedding is not None else None
            transition_probs = self.transition_predictor(
                current_state_features,
                context_features.unsqueeze(0),
                q_emb
            )
            outputs['transition_probs'] = transition_probs.squeeze(0)
        else:
            # 计算完整的转移矩阵
            if context_features is not None:
                transition_matrix = self.transition_predictor.compute_transition_matrix(
                    self.state_embeddings,
                    context_features,
                    question_embedding
                )
                outputs['transition_matrix'] = transition_matrix
        
        # 3. 预测监听关系权重
        listener_weights = self.listener_predictor(
            self.state_embeddings,
            evolved_agent_features,
            question_embedding
        )
        outputs['listener_weights'] = listener_weights
        
        # 4. 计算状态-智能体兼容性
        state_agent_compatibility = self.state_agent_matcher(
            self.state_embeddings,
            evolved_agent_features
        )
        outputs['state_agent_compatibility'] = state_agent_compatibility
        
        return outputs
    
    def get_fsm_structure(self,
                         agent_features: torch.Tensor,
                         context_features: torch.Tensor,
                         agent_ids: List[str],
                         listener_threshold: float = 0.15) -> Dict[str, Any]:
        """
        从学习的模型中提取FSM结构
        
        Args:
            agent_features: [num_agents, agent_feature_dim] 智能体特征
            context_features: [state_feature_dim] 任务上下文
            agent_ids: 智能体ID列表
            listener_threshold: 监听关系阈值
        
        Returns:
            fsm_structure: FSM结构描述（可用于initialize_fsm_from_description）
        """
        with torch.no_grad():
            # 获取预测结果
            outputs = self.forward(
                agent_features,
                communication_topology=torch.zeros(2, 0, dtype=torch.long),
                context_features=context_features
            )
            
            # 提取状态-智能体匹配
            compatibility = outputs['state_agent_compatibility']
            agent_assignment = self.state_agent_matcher.get_optimal_assignment(compatibility)
            
            # 提取监听关系
            listener_weights = outputs['listener_weights']
            listener_mask = self.listener_predictor.compute_binary_listener_mask(
                listener_weights,
                threshold=listener_threshold
            )
            
            # 构建FSM描述
            fsm_structure = {
                'states': [],
                'listeners': {}
            }
            
            for state_id in range(self.num_states):
                state_info = {
                    'id': state_id,
                    'name': f'State_{state_id}',
                    'agent': agent_ids[agent_assignment[state_id]],
                    'is_initial': (state_id == 0),
                    'is_final': (state_id == self.num_states - 1)
                }
                fsm_structure['states'].append(state_info)
                
                # 提取监听者
                listener_indices = torch.where(listener_mask[state_id] > 0)[0].tolist()
                fsm_structure['listeners'][state_id] = [agent_ids[idx] for idx in listener_indices]
            
            return fsm_structure
    
    def reset_agent_memories(self):
        """重置基础TGN的记忆"""
        if self.base_tgn is not None:
            self.base_tgn.reset_agent_memories()


# 导出
__all__ = [
    'FSMTemporalGraph',
    'FSMTransitionPredictor',
    'FSMListenerPredictor',
    'FSMStateAgentMatcher'
]


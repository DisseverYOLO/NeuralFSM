"""
Protected Temporal Graph Network
集成保护机制的时序图网络

将保护机制深度集成到TGN的消息传递过程中

作者: Neural FSM Team
日期: 2025-10-28
"""

import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union

from .simplified_centrality import SimplifiedCentralityAnalyzer
from .simplified_anomaly import SimplifiedAnomalyDetector
from .trust_calculator import TrustCalculator
from .message_weight_calculator import MessageWeightCalculator
from .protection_loss import ProtectionConstrainedLoss


class ProtectedTGN(nn.Module):
    """
    集成保护机制的时序图网络
    
    核心功能:
    1. 计算节点保护优先级 (静态,基于图中心性)
    2. 检测节点异常行为 (动态,基于历史)
    3. 计算信任分数 (结合优先级和异常)
    4. 在TGN消息函数中应用权重 (衰减低信任消息)
    5. 使用保护约束损失联合训练
    """
    
    def __init__(self,
                 tgn_model: nn.Module,
                 feature_dim: int,
                 graph: Optional[nx.Graph] = None,
                 w_betweenness: float = 0.6,
                 w_pagerank: float = 0.4,
                 lambda_freq: float = 0.3,
                 lambda_semantic: float = 0.7,
                 weight_hidden_dim: int = 64,
                 lambda_protect: float = 0.1,
                 lambda_reg: float = 0.01,
                 learnable_weights: bool = True):
        """
        初始化
        
        Args:
            tgn_model: 原始TGN模型
            feature_dim: 节点特征维度
            graph: 通信拓扑图 (可选,如果为None需要在forward时提供)
            w_betweenness: 介数中心性权重
            w_pagerank: PageRank权重
            lambda_freq: 频率异常权重
            lambda_semantic: 语义异常权重
            weight_hidden_dim: 权重计算网络隐藏层维度
            lambda_protect: 保护损失权重
            lambda_reg: 正则化权重
            learnable_weights: 所有权重是否可学习
        """
        super().__init__()
        
        # 原始TGN模型
        self.tgn = tgn_model
        self.feature_dim = feature_dim
        
        # 保护组件
        self.centrality_analyzer = SimplifiedCentralityAnalyzer(
            w_betweenness=w_betweenness,
            w_pagerank=w_pagerank,
            learnable=learnable_weights
        )
        
        self.anomaly_detector = SimplifiedAnomalyDetector(
            feature_dim=feature_dim,
            lambda_freq=lambda_freq,
            lambda_semantic=lambda_semantic,
            learnable=learnable_weights
        )
        
        self.trust_calculator = TrustCalculator()
        
        self.weight_calculator = MessageWeightCalculator(
            hidden_dim=weight_hidden_dim
        )
        
        self.protection_loss = ProtectionConstrainedLoss(
            lambda_protect=lambda_protect,
            lambda_reg=lambda_reg
        )
        
        # 缓存
        self._graph = graph
        self._priorities_cache = None
        self._priorities_tensor_cache = None
    
    def update_graph(self, graph: nx.Graph):
        """
        更新通信拓扑图
        
        Args:
            graph: 新的拓扑图
        """
        self._graph = graph
        self._priorities_cache = None
        self._priorities_tensor_cache = None
    
    def set_text_embedding_model(self, model):
        """
        ✨ 设置文本嵌入模型（用于语义一致性检测）
        
        这将启用基于BERT的语义一致性检测功能，
        大幅提升对语义攻击的检测能力。
        
        Args:
            model: 文本嵌入模型（需要有encode或encode_query方法）
        """
        if hasattr(self, 'anomaly_detector') and self.anomaly_detector is not None:
            self.anomaly_detector.set_text_embedding_model(model)
            print(f"  🧠 语义一致性检测已启用（基于BERT）")
    
    def compute_protection_priorities(self,
                                     graph: Optional[nx.Graph] = None,
                                     use_cache: bool = True,
                                     device: Optional[torch.device] = None) -> Tuple[Dict, torch.Tensor]:
        """
        计算节点保护优先级
        
        Args:
            graph: 拓扑图 (如果为None,使用self._graph)
            use_cache: 是否使用缓存
            device: 目标设备 (如果为None则使用CPU)
        
        Returns:
            (优先级字典, 优先级tensor)
        """
        if graph is None:
            graph = self._graph
        
        if graph is None:
            raise ValueError("Graph must be provided")
        
        # 检查缓存
        if use_cache and self._priorities_cache is not None:
            # ✨ 如果缓存的tensor在不同设备上，需要移动
            if device is not None and self._priorities_tensor_cache.device != device:
                self._priorities_tensor_cache = self._priorities_tensor_cache.to(device)
            return self._priorities_cache, self._priorities_tensor_cache
        
        # 计算优先级
        priorities = self.centrality_analyzer.compute_priority_scores(graph)
        
        # 转换为tensor (按节点ID排序)
        node_ids = sorted(graph.nodes())
        priority_list = [priorities[nid] for nid in node_ids]
        # ✨ 直接在目标设备上创建tensor
        priority_tensor = torch.tensor(priority_list, dtype=torch.float32, device=device)
        
        # 更新缓存
        if use_cache:
            self._priorities_cache = priorities
            self._priorities_tensor_cache = priority_tensor
        
        return priorities, priority_tensor
    
    def compute_anomaly_scores(self,
                               node_ids: List[Any],
                               message_counts: Optional[Dict] = None,
                               embeddings: Optional[Dict] = None,
                               agent_messages: Optional[Dict] = None,
                               device: Optional[torch.device] = None) -> torch.Tensor:
        """
        计算节点异常分数（✨ 扩展版，支持消息污染检测）
        
        Args:
            node_ids: 节点ID列表
            message_counts: 节点消息数量字典
            embeddings: 节点嵌入字典
            agent_messages: ✨ 智能体消息字典 {agent_id: message_text}
            device: 目标设备 (如果为None则使用CPU)
        
        Returns:
            异常分数tensor [num_nodes]
        """
        anomaly_dict = self.anomaly_detector.batch_compute_anomaly_scores(
            node_ids, message_counts, embeddings, agent_messages
        )
        
        # 转换为tensor (按node_ids顺序)
        anomaly_list = [anomaly_dict.get(nid, 0.0) for nid in node_ids]
        # ✨ 直接在目标设备上创建tensor
        anomaly_tensor = torch.tensor(anomaly_list, dtype=torch.float32, device=device)
        
        return anomaly_tensor
    
    def forward(self,
                agent_features: torch.Tensor,
                communication_topology: torch.Tensor,
                communication_attributes: Optional[torch.Tensor] = None,
                temporal_stamps: Optional[torch.Tensor] = None,
                agent_indices: Optional[torch.Tensor] = None,
                # ✨ FSM-TGN专用参数（新增）
                current_state_id: Optional[int] = None,
                context_features: Optional[torch.Tensor] = None,
                question_embedding: Optional[torch.Tensor] = None,
                transition_history: Optional[torch.Tensor] = None,
                # 保护机制专用参数
                graph: Optional[nx.Graph] = None,
                message_counts: Optional[Dict] = None,
                embeddings: Optional[Dict] = None,
                agent_messages: Optional[Dict] = None,  # ✨ 新增：智能体消息（用于污染检测）
                **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播 (带保护机制) - 完全兼容NeuralTemporalGraph和FSMTemporalGraph接口
        
        Args:
            agent_features: 智能体特征 [num_agents, feature_dim]
            communication_topology: 通信拓扑 [2, num_edges]
            communication_attributes: 通信边属性 (可选)
            temporal_stamps: 时间戳 [num_agents]
            agent_indices: 智能体索引 [num_agents]
            # ✨ FSM-TGN专用参数（新增）
            current_state_id: 当前状态ID（FSM-TGN使用）
            context_features: 任务上下文特征（FSM-TGN使用）
            question_embedding: 问题嵌入（FSM-TGN使用，任务自适应）
            transition_history: 状态转移历史（FSM-TGN使用）
            # 保护机制专用参数
            graph: NetworkX图 (用于计算中心性)
            message_counts: 消息计数 (用于异常检测)
            embeddings: 节点嵌入 (用于异常检测)
            **kwargs: 其他参数
        
        Returns:
            - 如果是FSMTemporalGraph: 返回字典 {
                'agent_features': 演化后的智能体特征,
                'transition_probs': 状态转移概率,
                'listener_weights': 监听关系权重,
                ...
            }
            - 如果是NeuralTemporalGraph: 返回输出特征 [num_agents, feature_dim]
        """
        device = agent_features.device
        num_nodes = agent_features.size(0)
        
        # 1. 计算保护优先级
        # ✨ 关键改进：优先使用实际采样的通信边（而不是全连接图）！
        # 实际采样边可以从 agent_messages 的特殊键 '_communication_edges' 获取
        sampled_edges = None
        if agent_messages is not None and '_communication_edges' in agent_messages:
            sampled_edges = agent_messages['_communication_edges']
        
        if sampled_edges and len(sampled_edges) > 0:
            # ✨ 使用实际采样的通信边构建图（最准确的方式！）
            dynamic_graph = self._build_graph_from_sampled_edges(sampled_edges, num_nodes)
            # 动态计算中心性（不缓存，因为每个问题的采样不同）
            _, priorities = self.compute_protection_priorities(dynamic_graph, use_cache=False, device=device)
            # print(f"  🎯 使用实际采样边计算中心性 ({len(sampled_edges)} 条边)")
        elif communication_topology is not None and communication_topology.size(1) > 0:
            # 回退：使用全连接通信拓扑（会导致所有优先级相同）
            dynamic_graph = self._build_graph_from_topology(communication_topology, num_nodes)
            _, priorities = self.compute_protection_priorities(dynamic_graph, use_cache=False, device=device)
        elif graph is not None:
            # 回退到静态图（可缓存）
            _, priorities = self.compute_protection_priorities(graph, use_cache=True, device=device)
        else:
            # 如果没有提供图,使用默认优先级
            priorities = torch.ones(num_nodes, dtype=torch.float32, device=device) * 0.5
        
        # priorities 已经在正确设备上，无需再次 .to(device)
        
        # 2. 计算异常分数 (动态) - ✨ 扩展版，支持消息污染检测
        if message_counts is not None or embeddings is not None or agent_messages is not None:
            node_ids = list(range(num_nodes))
            # ✨ 更新异常检测器的历史数据（用于后续检测）
            if message_counts is not None:
                for agent_id, count in message_counts.items():
                    if isinstance(agent_id, int) and 0 <= agent_id < num_nodes:
                        self.anomaly_detector.update_history(agent_id, message_count=count)
            if embeddings is not None:
                for agent_id, emb in embeddings.items():
                    if isinstance(agent_id, int) and 0 <= agent_id < num_nodes:
                        # 确保嵌入是tensor格式
                        if isinstance(emb, torch.Tensor):
                            self.anomaly_detector.update_history(agent_id, embedding=emb)
                        else:
                            # 如果是numpy或其他格式，转换为tensor
                            emb_tensor = torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb
                            self.anomaly_detector.update_history(agent_id, embedding=emb_tensor)
            
            # ✨ 计算异常分数（使用更新后的历史 + 消息污染检测）
            # ✨ 传递device参数，确保在正确设备上创建tensor
            anomaly_scores = self.compute_anomaly_scores(
                node_ids, message_counts, embeddings, agent_messages, device=device
            )
            
            # ✨ 如果检测到消息污染，打印警告日志
            if agent_messages:
                for agent_id, message in agent_messages.items():
                    if isinstance(agent_id, int) and 0 <= agent_id < num_nodes:
                        pollution_score, pollution_info = self.anomaly_detector.detect_message_pollution(message)
                        if pollution_score > 0.5:
                            pollution_type = pollution_info.get('pollution_type', 'unknown')
                            print(f"  🛡️ 检测到消息污染: Agent {agent_id}, 类型={pollution_type}, 分数={pollution_score:.3f}")
        else:
            # 如果没有提供历史数据,使用默认异常分数
            # ✨ 直接在目标设备上创建
            anomaly_scores = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        
        # anomaly_scores 已经在正确设备上
        
        # ✨ 2.5 计算消息污染分数（如果有消息）
        pollution_scores = None
        if agent_messages:
            pollution_list = []
            for agent_id in range(num_nodes):
                message = agent_messages.get(agent_id, "")
                if message:
                    pollution_score, _ = self.anomaly_detector.detect_message_pollution(message)
                    pollution_list.append(pollution_score)
                else:
                    pollution_list.append(0.0)
            pollution_scores = torch.tensor(pollution_list, dtype=torch.float32, device=device)
        
        # 3. 计算信任分数（✨ 传入污染分数）
        trust_scores = self.trust_calculator(anomaly_scores, priorities, pollution_scores)
        
        # 4. 计算消息权重
        source_trust = trust_scores[communication_topology[0]]      # [num_edges]
        target_priority = priorities[communication_topology[1]]     # [num_edges]
        message_weights = self.weight_calculator(source_trust, target_priority)
        
        # 5. 应用保护机制：修改通信拓扑，过滤低信任消息
        # 核心创新：基于trust和priority动态调整通信边
        protected_topology, protected_weights = self._apply_message_attenuation(
            communication_topology, message_weights, anomaly_scores, priorities
        )
        
        # 6. 调用原始TGN，使用受保护的拓扑
        # ✨ 检查是否是FSMTemporalGraph（需要特殊参数）
        is_fsm_tgn = hasattr(self.tgn, 'transition_predictor') and hasattr(self.tgn, 'listener_predictor')
        
        try:
            if is_fsm_tgn:
                # ✨ FSMTemporalGraph需要返回字典，包含transition_probs和listener_weights
                # 如果启用了激进的保护策略，修改拓扑
                if protected_topology.size(1) < communication_topology.size(1):
                    # 使用过滤后的拓扑
                    fsm_outputs = self.tgn(
                        agent_features=agent_features,
                        communication_topology=protected_topology,
                        current_state_id=current_state_id,
                        context_features=context_features,
                        question_embedding=question_embedding,  # ✨ 传递问题嵌入
                        transition_history=transition_history,
                        communication_attributes=None,  # 过滤后的边没有原始属性
                        temporal_stamps=temporal_stamps,
                        agent_indices=agent_indices,
                        **kwargs
                    )
                else:
                    # 使用原始拓扑，但会在后处理中应用衰减
                    fsm_outputs = self.tgn(
                        agent_features=agent_features,
                        communication_topology=communication_topology,
                        current_state_id=current_state_id,
                        context_features=context_features,
                        question_embedding=question_embedding,  # ✨ 传递问题嵌入
                        transition_history=transition_history,
                        communication_attributes=communication_attributes,
                        temporal_stamps=temporal_stamps,
                        agent_indices=agent_indices,
                        **kwargs
                    )
                    
                    # ✅ 实际消息衰减：对输出应用基于信任的加权
                    if 'agent_features' in fsm_outputs:
                        fsm_outputs['agent_features'] = self._apply_output_attenuation(
                            fsm_outputs['agent_features'], agent_features, communication_topology, 
                            message_weights, anomaly_scores
                        )
                
                # ✨ 返回FSM输出字典（包含transition_probs和listener_weights）
                # ✅ 添加保护相关数据（用于计算保护损失）
                fsm_outputs['protection_data'] = {
                    'anomaly_scores': anomaly_scores,
                    'priorities': priorities,
                    'trust_scores': trust_scores,
                    'message_weights': message_weights
                }
                return fsm_outputs
            else:
                # 普通TGN（NeuralTemporalGraph）
                if protected_topology.size(1) < communication_topology.size(1):
                    # 使用过滤后的拓扑
                    output = self.tgn(
                        agent_features,
                        protected_topology,
                        communication_attributes=None,  # 过滤后的边没有原始属性
                        temporal_stamps=temporal_stamps,
                        agent_indices=agent_indices
                    )
                else:
                    # 使用原始拓扑，但会在后处理中应用衰减
                    output = self.tgn(
                        agent_features,
                        communication_topology,
                        communication_attributes=communication_attributes,
                        temporal_stamps=temporal_stamps,
                        agent_indices=agent_indices
                    )
                    
                    # ✅ 实际消息衰减：对输出应用基于信任的加权
                    output = self._apply_output_attenuation(
                        output, agent_features, communication_topology, 
                        message_weights, anomaly_scores
                    )
                
                # ✨ 关键修复：返回字典格式，包含 protection_data
                # 这样 FSMTemporalGraph 才能正确提取 protection_data
                return {
                    'agent_features': output,
                    'protection_data': {
                        'anomaly_scores': anomaly_scores,
                        'priorities': priorities,
                        'trust_scores': trust_scores,
                        'message_weights': message_weights
                    }
                }
                
        except Exception as e:
            print(f"⚠️  TGN前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            # 降级: 返回输入特征或空字典
            if is_fsm_tgn:
                return {
                    'agent_features': agent_features,
                    'transition_probs': None,
                    'listener_weights': None
                }
            else:
                return agent_features
    
    def _build_graph_from_topology(self, 
                                   communication_topology: torch.Tensor,
                                   num_nodes: int) -> nx.DiGraph:
        """
        从通信拓扑张量构建 NetworkX 有向图
        
        ✨ 这是动态中心性计算的关键：每轮FSM执行时，
        根据实际采样出的通信边构建图，然后计算中心性
        
        Args:
            communication_topology: 通信边索引 [2, num_edges]
                - communication_topology[0]: 源节点索引
                - communication_topology[1]: 目标节点索引
            num_nodes: 节点总数
        
        Returns:
            NetworkX 有向图
        """
        graph = nx.DiGraph()
        
        # 添加所有节点（即使没有边也需要节点存在）
        graph.add_nodes_from(range(num_nodes))
        
        # 从通信拓扑添加边
        if communication_topology.size(1) > 0:
            edges = communication_topology.cpu().numpy()
            for i in range(edges.shape[1]):
                src = int(edges[0, i])
                dst = int(edges[1, i])
                if src != dst:  # 避免自环
                    graph.add_edge(src, dst)
        
        return graph
    
    def _build_graph_from_sampled_edges(self, 
                                       sampled_edges: List[Tuple],
                                       num_nodes: int) -> nx.DiGraph:
        """
        从实际采样的通信边构建 NetworkX 有向图
        
        ✨ 这是最准确的图构建方式，基于FSM执行过程中
        实际采样的监听者关系，而不是全连接假设！
        
        Args:
            sampled_edges: 采样的通信边列表 [(src_agent_id, dst_agent_id), ...]
                - agent_id 可以是字符串（如 "agent_0"）或整数
            num_nodes: 节点总数
        
        Returns:
            NetworkX 有向图（基于实际通信模式）
        """
        graph = nx.DiGraph()
        
        # 添加所有节点
        graph.add_nodes_from(range(num_nodes))
        
        # 添加实际采样的边
        for src, dst in sampled_edges:
            # 转换 agent_id 为整数索引
            try:
                if isinstance(src, str) and '_' in src:
                    src_idx = int(src.split('_')[-1])
                else:
                    src_idx = int(src)
                    
                if isinstance(dst, str) and '_' in dst:
                    dst_idx = int(dst.split('_')[-1])
                else:
                    dst_idx = int(dst)
                
                # 确保索引在有效范围内且不是自环
                if 0 <= src_idx < num_nodes and 0 <= dst_idx < num_nodes and src_idx != dst_idx:
                    graph.add_edge(src_idx, dst_idx)
            except (ValueError, AttributeError):
                continue
        
        return graph
    
    def _apply_message_attenuation(self,
                                   communication_topology: torch.Tensor,
                                   message_weights: torch.Tensor,
                                   anomaly_scores: torch.Tensor,
                                   priorities: torch.Tensor,
                                   threshold: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用消息衰减策略 - 过滤极低信任的边
        
        Args:
            communication_topology: 原始通信拓扑 [2, num_edges]
            message_weights: 消息权重 [num_edges]
            anomaly_scores: 异常分数 [num_nodes]
            priorities: 保护优先级 [num_nodes]
            threshold: 权重阈值，低于此值的边将被过滤（默认0.2，更强过滤）
        
        Returns:
            (受保护的拓扑, 受保护的权重)
        """
        # 策略1: 过滤极低信任的消息 (可选，较激进)
        # 只有当消息权重非常低时才过滤，避免过度保护
        if threshold > 0:
            valid_mask = message_weights > threshold
            
            if valid_mask.sum() > 0:  # 确保至少保留一些边
                protected_topology = communication_topology[:, valid_mask]
                protected_weights = message_weights[valid_mask]
                
                if protected_topology.size(1) < communication_topology.size(1):
                    num_filtered = communication_topology.size(1) - protected_topology.size(1)
                    print(f"  🛡️  过滤了 {num_filtered} 条低信任通信边")
                
                return protected_topology, protected_weights
        
        # 默认：返回原始拓扑
        return communication_topology, message_weights
    
    def _apply_output_attenuation(self,
                                  output: torch.Tensor,
                                  original_features: torch.Tensor,
                                  communication_topology: torch.Tensor,
                                  message_weights: torch.Tensor,
                                  anomaly_scores: torch.Tensor) -> torch.Tensor:
        """
        对输出应用消息衰减 - 核心保护机制
        
        ✅ 正确逻辑：惩罚发送者（攻击源），保护接收者（潜在受害者）
        
        策略：
        1. 对于每个节点，计算它接收到的消息的"发送者信任度"
        2. 发送者异常分数高 → 发送者信任度低 → 该消息被衰减
        3. 接收者本身不受惩罚（它是潜在的受害者）
        
        Args:
            output: TGN的原始输出 [num_nodes, feature_dim]
            original_features: 原始输入特征 [num_nodes, feature_dim]
            communication_topology: 通信拓扑 [2, num_edges]
            message_weights: 消息权重 [num_edges]
            anomaly_scores: 异常分数 [num_nodes]
        
        Returns:
            受保护的输出 [num_nodes, feature_dim]
        """
        device = output.device
        num_nodes = output.size(0)
        
        # ═══════════════════════════════════════════════════════════════════
        # 核心：计算每个节点收到的消息的"发送者信任度"
        # 发送者异常分数高 → 信任度低 → 消息影响被削弱
        # ═══════════════════════════════════════════════════════════════════
        sender_trust_for_receiver = torch.ones(num_nodes, device=device)
        
        for node_id in range(num_nodes):
            # 找到所有指向该节点的边（谁给我发消息了？）
            incoming_mask = communication_topology[1] == node_id
            
            if incoming_mask.sum() > 0:
                # 获取发送者的索引
                sender_indices = communication_topology[0][incoming_mask]
                
                # ✅ 关键：发送者的异常分数决定信任度
                sender_anomalies = anomaly_scores[sender_indices]
                sender_trust = 1.0 - sender_anomalies  # 发送者异常高 → 信任低
                
                # 获取消息权重（已经包含了信任信息）
                incoming_weights = message_weights[incoming_mask]
                
                # 综合发送者信任和消息权重
                # 双重惩罚：消息权重低 AND 发送者异常高 → 影响更小
                combined_sender_trust = incoming_weights * sender_trust
                
                # 加权平均：权重更高的边影响更大
                if incoming_weights.sum() > 0:
                    sender_trust_for_receiver[node_id] = (
                        (combined_sender_trust * incoming_weights).sum() / 
                        incoming_weights.sum()
                    )
                else:
                    sender_trust_for_receiver[node_id] = combined_sender_trust.mean()
        
        # ═══════════════════════════════════════════════════════════════════
        # 输出混合：发送者信任度决定使用多少TGN输出
        # - 发送者信任高 (≈1): 使用TGN输出（消息聚合结果）
        # - 发送者信任低 (≈0): 保留原始特征（不受污染消息影响）
        # ═══════════════════════════════════════════════════════════════════
        
        # 扩展维度以匹配特征维度
        trust_weights = sender_trust_for_receiver.unsqueeze(-1)  # [num_nodes, 1]
        
        # ✅ 不惩罚接收者！接收者是潜在受害者，不应该被惩罚
        # 只根据发送者的信任度来决定输出混合比例
        
        # 确保信任度在合理范围内
        trust_weights = torch.clamp(trust_weights, min=0.1, max=1.0)
        
        # 加权混合：
        # protected_output = sender_trust × TGN_output + (1 - sender_trust) × original_features
        #
        # 解释：
        # - 如果发送者可信 (trust≈1): 主要使用TGN聚合后的输出
        # - 如果发送者不可信 (trust≈0): 保留原始特征，忽略可能被污染的消息
        protected_output = trust_weights * output + (1.0 - trust_weights) * original_features
        
        return protected_output
    
    def get_protection_statistics(self) -> Dict[str, Any]:
        """
        获取保护机制的统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'centrality_weights': self.centrality_analyzer.get_fusion_weights(),
            'anomaly_weights': self.anomaly_detector.get_fusion_weights(),
            'graph_cached': self._graph is not None,
            'priorities_cached': self._priorities_cache is not None
        }
    
    def tgn_with_protection(self,
                           agent_features: torch.Tensor,
                           communication_topology: torch.Tensor,
                           message_weights: torch.Tensor,
                           temporal_stamps: Optional[torch.Tensor] = None,
                           agent_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带保护的TGN消息传递 (内部辅助方法,已不再直接使用)
        
        Args:
            agent_features: 智能体特征
            communication_topology: 通信拓扑
            message_weights: 消息权重 [num_edges]
            temporal_stamps: 时间戳
            agent_indices: 智能体索引
        
        Returns:
            (输出特征, 保护后的消息)
        """
        try:
            # 直接调用原始TGN,参数名已匹配
            output = self.tgn(
                agent_features,
                communication_topology,
                communication_attributes=None,
                temporal_stamps=temporal_stamps,
                agent_indices=agent_indices
            )
            
            # 计算原始消息
            raw_messages = self.compute_messages(agent_features, communication_topology)
            
            # 应用权重 - 保护机制的核心
            protected_messages = raw_messages * message_weights.unsqueeze(-1)
            
            return output, protected_messages
            
        except Exception as e:
            print(f"⚠️  TGN前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回安全默认值
            num_edges = communication_topology.size(1) if communication_topology.dim() > 1 else 0
            default_messages = torch.zeros(num_edges, self.feature_dim * 2, 
                                          dtype=agent_features.dtype, 
                                          device=agent_features.device)
            return agent_features, default_messages
    
    def compute_messages(self,
                        agent_features: torch.Tensor,
                        communication_topology: torch.Tensor) -> torch.Tensor:
        """
        计算原始消息
        
        Args:
            agent_features: 智能体特征 [num_agents, feature_dim]
            communication_topology: 通信拓扑 [2, num_edges]
        
        Returns:
            消息 [num_edges, message_dim]
        """
        source_features = agent_features[communication_topology[0]]  # [num_edges, feature_dim]
        target_features = agent_features[communication_topology[1]]  # [num_edges, feature_dim]
        
        # 简单拼接
        messages = torch.cat([source_features, target_features], dim=-1)
        
        return messages
    
    def compute_loss(self,
                    predictions: torch.Tensor,
                    labels: torch.Tensor,
                    messages: torch.Tensor,
                    anomaly_scores: torch.Tensor,
                    priorities: torch.Tensor,
                    edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算保护约束损失
        
        Args:
            predictions: 模型预测
            labels: 真实标签
            messages: 消息向量
            anomaly_scores: 异常分数
            priorities: 保护优先级
            edge_index: 边索引
        
        Returns:
            (总损失, 损失详情)
        """
        # 获取所有可训练参数
        model_params = [p for p in self.parameters() if p.requires_grad]
        
        # 计算损失
        loss, loss_dict = self.protection_loss(
            predictions, labels, messages,
            anomaly_scores, priorities, edge_index,
            model_params
        )
        
        return loss, loss_dict
    
    def get_learned_weights(self) -> Dict[str, Any]:
        """
        获取所有学习到的权重
        
        Returns:
            权重字典
        """
        return {
            'centrality': self.centrality_analyzer.get_fusion_weights(),
            'anomaly': self.anomaly_detector.get_fusion_weights()
        }
    
    def visualize_protection(self,
                            graph: nx.Graph,
                            save_path: Optional[str] = None):
        """
        可视化保护优先级
        
        Args:
            graph: 拓扑图
            save_path: 保存路径
        """
        priorities, _ = self.compute_protection_priorities(graph, use_cache=False)
        self.centrality_analyzer.visualize(graph, priorities, save_path)
    
    def update_anomaly_history(self,
                               agent_id: Any,
                               message_count: Optional[int] = None,
                               embedding: Optional[torch.Tensor] = None):
        """
        更新异常检测器的历史数据
        
        Args:
            agent_id: 智能体ID
            message_count: 消息数量
            embedding: 消息嵌入
        """
        self.anomaly_detector.update_history(agent_id, message_count, embedding)
    
    def reset_anomaly_history(self, agent_id: Optional[Any] = None):
        """
        重置异常检测器的历史数据
        
        Args:
            agent_id: 智能体ID (如果为None则重置所有)
        """
        self.anomaly_detector.reset_history(agent_id)


__all__ = [
    'ProtectedTGN'
]


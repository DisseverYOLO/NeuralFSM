"""
FSM Multi-Agent System Generator
FSM多智能体系统生成器

完整集成MetaAgent的智能体生成和FSM生成能力
支持随机采样拓扑图并用TGN学习最优路径

🎯 核心架构说明：
-----------------
1. 本模块负责生成FSM多智能体系统的初始结构
2. 随机采样状态转移和通信拓扑作为学习的起点
3. 提供MSE重构损失作为辅助正则化项，帮助TGN学习稳定的节点表示
4. ⚠️ TGN参数的实际优化由train_neural_mas.py中的组合损失完成

损失函数设计（多任务学习）：
- 主损失：策略梯度损失（基于MMLU准确率，任务导向）
- 辅助损失：MSE重构损失（稳定训练，正则化节点表示）
- 组合损失：total_loss = α * policy_gradient_loss + β * reconstruction_loss

正确的训练流程：
- 生成FSM-MAS系统（本模块）
- 使用MMLU数据训练，组合损失优化TGN参数（train_neural_mas.py）
- 策略梯度确保任务性能，MSE损失提供稳定梯度信号
"""

import json
import random
import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from baseclass.FSM_Gen import Generate_Agent_Description, Generate_FSM
from baseclass.MultiAgent import MultiAgentSystem
from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph
from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager
from neural_fsm_mas.embeddings import get_embedding_model


class FSMMultiAgentSystemGenerator:
    """
    FSM多智能体系统生成器
    
    核心功能：
    1. 使用MetaAgent生成智能体描述和FSM状态
    2. 随机采样状态转移拓扑图
    3. 随机采样Listening智能体通信拓扑图
    4. 使用TGN学习最优的状态转移路径和通信路径
    """
    
    def __init__(self, 
                 use_neural_learning: bool = True,
                 memory_dimension: int = 128,
                 temporal_dimension: int = 32,
                 random_seed: int = 42,
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        
        self.use_neural_learning = use_neural_learning
        self.memory_dimension = memory_dimension
        self.temporal_dimension = temporal_dimension
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # 初始化嵌入模型（参考GDesigner）
        print(f"🔤 初始化文本嵌入模型...")
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.embedding_dim = self.embedding_model.embedding_dim
        print(f"✅ 嵌入模型加载完成，维度: {self.embedding_dim}")
        
        # 存储生成的组件
        self.generated_agents = None
        self.generated_fsm = None
        self.state_topology_graph = None
        self.listening_topology_graph = None
        self.neural_state_learner = None
        self.neural_communication_learner = None
        
    def generate_complete_fsm_mas(self, 
                                 task_description: str,
                                 available_tools: List[str] = None) -> Dict[str, Any]:
        """
        生成完整的FSM多智能体系统
        
        Args:
            task_description: 任务描述
            available_tools: 可用工具列表
            
        Returns:
            完整的FSM-MAS系统配置
        """
        print("🚀 开始生成FSM多智能体系统...")
        
        # 默认工具列表
        if available_tools is None:
            available_tools = [
                "code_interpreter", "web_search", "calculator", 
                "knowledge_retrieval", "logical_reasoning", "analysis"
            ]
        
        # Step 1: 使用MetaAgent生成智能体描述
        print("🤖 Step 1: 生成智能体角色描述...")
        self.generated_agents, _ = Generate_Agent_Description(task_description, available_tools)
        print(f"✅ 生成了 {len(self.generated_agents)} 个智能体")
        
        # Step 2: 使用MetaAgent生成FSM状态
        print("🔄 Step 2: 生成有限状态机...")
        self.generated_fsm, _ = Generate_FSM(task_description, self.generated_agents)
        if self.generated_fsm is None:
            raise ValueError("FSM生成失败")
        print(f"✅ 生成了 {len(self.generated_fsm['states'])} 个状态")
        
        # Step 3: 随机采样状态转移拓扑图
        print("🕸️ Step 3: 随机采样状态转移拓扑图...")
        self.state_topology_graph = self._sample_state_transition_topology()
        print(f"✅ 采样了状态转移拓扑，包含 {len(self.state_topology_graph.edges)} 条边")
        
        # Step 4: 随机采样Listening智能体通信拓扑图
        print("📡 Step 4: 随机采样Listening智能体通信拓扑图...")
        self.listening_topology_graph = self._sample_listening_communication_topology()
        print(f"✅ 采样了通信拓扑，包含 {len(self.listening_topology_graph.edges)} 条边")
        
        # Step 5: 如果启用神经学习，创建TGN学习器
        if self.use_neural_learning:
            print("🧠 Step 5: 初始化TGN神经学习器...")
            self._initialize_neural_learners()
            print("✅ TGN学习器初始化完成")
        
        # 构建完整系统配置
        fsm_mas_config = {
            "task_description": task_description,
            "agents": self.generated_agents,
            "fsm": self.generated_fsm,
            "state_topology": self._graph_to_dict(self.state_topology_graph),
            "listening_topology": self._graph_to_dict(self.listening_topology_graph),
            "neural_learning_enabled": self.use_neural_learning,
            "system_metadata": {
                "num_agents": len(self.generated_agents),
                "num_states": len(self.generated_fsm['states']),
                "num_transitions": len(self.generated_fsm['transitions']),
                "state_topology_edges": len(self.state_topology_graph.edges),
                "listening_topology_edges": len(self.listening_topology_graph.edges)
            }
        }
        
        print("🎉 FSM多智能体系统生成完成！")
        return fsm_mas_config
    
    def _sample_state_transition_topology(self) -> nx.DiGraph:
        """
        随机采样状态转移拓扑图
        
        基于原始FSM的转移关系，添加随机的额外连接
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加所有状态作为节点
        states = self.generated_fsm['states']
        for state in states:
            G.add_node(state['state_id'], 
                      agent_id=state['agent_id'],
                      instruction=state['instruction'],
                      is_initial=state['is_initial'],
                      is_final=state['is_final'])
        
        # 添加原始转移关系
        for transition in self.generated_fsm['transitions']:
            G.add_edge(transition['from_state'], 
                      transition['to_state'],
                      condition=transition['condition'],
                      edge_type='original')
        
        # 随机添加额外的转移连接（用于学习）
        state_ids = [state['state_id'] for state in states]
        num_additional_edges = random.randint(1, len(state_ids) // 2)
        
        for _ in range(num_additional_edges):
            from_state = random.choice(state_ids)
            to_state = random.choice(state_ids)
            
            # 避免自环和重复边
            if from_state != to_state and not G.has_edge(from_state, to_state):
                G.add_edge(from_state, to_state,
                          condition="learnable_transition",
                          edge_type='sampled')
        
        return G
    
    def _sample_listening_communication_topology(self) -> nx.Graph:
        """
        随机采样Listening智能体通信拓扑图
        
        基于FSM中的listener关系，添加随机的通信连接
        """
        # 创建无向图（通信是双向的）
        G = nx.Graph()
        
        # 添加所有智能体作为节点
        for agent in self.generated_agents:
            G.add_node(agent['agent_id'], 
                      name=agent['name'],
                      system_prompt=agent['system_prompt'],
                      tools=agent['tools'])
        
        # 基于FSM中的listener关系添加通信边
        for state in self.generated_fsm['states']:
            current_agent = state['agent_id']
            listeners = state.get('listener', [])
            
            for listener_id in listeners:
                if listener_id != current_agent:  # 避免自连接
                    G.add_edge(current_agent, listener_id,
                             edge_type='listening',
                             state_context=state['state_id'])
        
        # 随机添加额外的通信连接
        agent_ids = [agent['agent_id'] for agent in self.generated_agents]
        num_additional_edges = random.randint(1, len(agent_ids))
        
        for _ in range(num_additional_edges):
            agent1 = random.choice(agent_ids)
            agent2 = random.choice(agent_ids)
            
            # 避免自环和重复边
            if agent1 != agent2 and not G.has_edge(agent1, agent2):
                G.add_edge(agent1, agent2,
                          edge_type='sampled_communication',
                          weight=random.uniform(0.1, 1.0))
        
        return G
    
    def _initialize_neural_learners(self):
        """
        初始化TGN神经学习器（使用嵌入维度）
        
        参考GDesigner，使用Sentence Transformer的嵌入维度作为节点特征维度
        """
        # 状态转移学习器
        num_states = len(self.generated_fsm['states'])
        print(f"🧠 初始化状态转移TGN (特征维度={self.embedding_dim})...")
        self.neural_state_learner = NeuralTemporalGraph(
            agent_feature_dim=self.embedding_dim,  # 使用嵌入维度
            memory_dimension=self.memory_dimension,
            temporal_dimension=self.temporal_dimension,
            agent_count=num_states,
            network_layers=2
        )
        
        # 智能体通信学习器
        num_agents = len(self.generated_agents)
        print(f"🧠 初始化智能体通信TGN (特征维度={self.embedding_dim})...")
        self.neural_communication_learner = NeuralTemporalGraph(
            agent_feature_dim=self.embedding_dim,  # 使用嵌入维度
            memory_dimension=self.memory_dimension,
            temporal_dimension=self.temporal_dimension,
            agent_count=num_agents,
            network_layers=2
        )
    
    def learn_optimal_topologies(self, 
                                training_episodes: int = 100,
                                learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        计算辅助的MSE重构损失（用于与策略梯度组合）
        
        ⚠️ 注意：这个方法仅计算MSE重构损失，不执行参数更新
        - MSE损失作为辅助正则化项，帮助TGN学习稳定的节点表示
        - 实际的参数更新在train_neural_mas.py中通过组合损失完成
        - 组合损失 = α * 策略梯度 + β * MSE重构
        
        Args:
            training_episodes: 训练轮数（当前仅计算单次损失）
            learning_rate: 学习率（未使用，仅为兼容性保留）
            
        Returns:
            包含MSE损失值和拓扑结构的字典
        """
        if not self.use_neural_learning:
            raise ValueError("Neural learning is not enabled")
        
        print(f"🎯 计算辅助MSE重构损失（用于组合损失）...")
        
        # 准备特征
        state_features = self._prepare_state_features()
        agent_features = self._prepare_agent_features()
        
        # 计算重构损失（不执行参数更新）
        state_reconstruction_loss = self._train_state_transitions(
            state_features, None, 0
        )
        communication_reconstruction_loss = self._train_communication_paths(
            agent_features, None, 0
        )
        
        # 生成优化后的拓扑结构
        optimized_state_topology = self._generate_optimized_state_topology()
        optimized_communication_topology = self._generate_optimized_communication_topology()
        
        learning_results = {
            "training_episodes": 1,
            "final_state_loss": state_reconstruction_loss,  # MSE辅助损失
            "final_communication_loss": communication_reconstruction_loss,  # MSE辅助损失
            "state_loss_history": [state_reconstruction_loss],
            "communication_loss_history": [communication_reconstruction_loss],
            "optimized_state_topology": optimized_state_topology,
            "optimized_communication_topology": optimized_communication_topology,
            "optimization_method": "combined_policy_gradient_and_reconstruction"
        }
        
        print(f"✅ MSE辅助损失计算完成: State={state_reconstruction_loss:.4f}, Comm={communication_reconstruction_loss:.4f}")
        return learning_results
    
    def _prepare_state_features(self) -> torch.Tensor:
        """
        准备状态特征（参考GDesigner的construct_features）
        
        使用Sentence Transformer将状态描述嵌入为向量
        """
        if self.generated_fsm is None:
            raise ValueError("FSM not generated yet")
        
        # 使用嵌入模型将状态描述转换为向量
        state_embeddings = self.embedding_model.encode_states(self.generated_fsm['states'])
        
        print(f"📊 状态特征准备完成: {state_embeddings.shape}")
        return state_embeddings
    
    def _prepare_agent_features(self) -> torch.Tensor:
        """
        准备智能体特征（参考GDesigner的construct_features）
        
        使用Sentence Transformer将智能体描述嵌入为向量
        """
        if self.generated_agents is None:
            raise ValueError("Agents not generated yet")
        
        # 使用嵌入模型将智能体描述转换为向量
        agent_embeddings = self.embedding_model.encode_agents(self.generated_agents)
        
        print(f"📊 智能体特征准备完成: {agent_embeddings.shape}")
        return agent_embeddings
    
    def _prepare_features_with_query(self, 
                                     node_features: torch.Tensor,
                                     query: str) -> torch.Tensor:
        """
        将节点特征与查询嵌入结合（参考GDesigner的construct_new_features）
        
        Args:
            node_features: 节点特征 [num_nodes, feature_dim]
            query: 任务查询文本
            
        Returns:
            组合特征 [num_nodes, feature_dim + embedding_dim]
        """
        combined_features = self.embedding_model.combine_features_with_query(
            node_features, query
        )
        
        print(f"📊 查询特征结合完成: {combined_features.shape}")
        return combined_features
    
    def compute_auxiliary_reconstruction_loss(self, 
                                            node_features: torch.Tensor,
                                            edge_index: torch.Tensor,
                                            timestamps: torch.Tensor,
                                            learner_network: torch.nn.Module) -> float:
        """
        计算辅助的MSE重构损失
        
        这个损失函数作为正则化项，帮助TGN学习稳定且有意义的节点表示
        与策略梯度损失组合使用，可以提高训练稳定性
        
        Args:
            node_features: 节点特征
            edge_index: 边索引
            timestamps: 时间戳
            learner_network: TGN网络
            
        Returns:
            重构损失值
        """
        # 前向传播
        evolved_features = learner_network(
            node_features, edge_index, timestamps=timestamps
        )
        
        # 计算重构损失
        reconstruction_loss = torch.nn.functional.mse_loss(evolved_features, node_features)
        
        return reconstruction_loss.item()
    
    def _train_state_transitions(self, 
                               state_features: torch.Tensor,
                               optimizer: torch.optim.Optimizer,
                               episode: int) -> float:
        """
        【辅助方法】计算状态转移的MSE重构损失
        
        注意：此方法仅计算损失值，不执行参数更新
        实际的参数更新由train_neural_mas.py中的组合损失完成
        """
        edge_index = self._build_state_edge_index()
        timestamps = torch.tensor([episode] * len(self.generated_fsm['states']), dtype=torch.float)
        
        return self.compute_auxiliary_reconstruction_loss(
            state_features, edge_index, timestamps, self.neural_state_learner
        )
    
    def _train_communication_paths(self, 
                                 agent_features: torch.Tensor,
                                 optimizer: torch.optim.Optimizer,
                                 episode: int) -> float:
        """
        【辅助方法】计算通信路径的MSE重构损失
        
        注意：此方法仅计算损失值，不执行参数更新
        实际的参数更新由train_neural_mas.py中的组合损失完成
        """
        edge_index = self._build_communication_edge_index()
        timestamps = torch.tensor([episode] * len(self.generated_agents), dtype=torch.float)
        
        return self.compute_auxiliary_reconstruction_loss(
            agent_features, edge_index, timestamps, self.neural_communication_learner
        )
    
    def _build_state_edge_index(self) -> torch.Tensor:
        """构建状态转移的边索引"""
        edges = []
        state_id_to_idx = {state['state_id']: i for i, state in enumerate(self.generated_fsm['states'])}
        
        for edge in self.state_topology_graph.edges():
            from_idx = state_id_to_idx[edge[0]]
            to_idx = state_id_to_idx[edge[1]]
            edges.append([from_idx, to_idx])
        
        if not edges:
            # 如果没有边，创建一个自环
            edges = [[0, 0]]
        
        return torch.tensor(edges).t().contiguous()
    
    def _build_communication_edge_index(self) -> torch.Tensor:
        """构建通信的边索引"""
        edges = []
        agent_id_to_idx = {agent['agent_id']: i for i, agent in enumerate(self.generated_agents)}
        
        for edge in self.listening_topology_graph.edges():
            from_idx = agent_id_to_idx[edge[0]]
            to_idx = agent_id_to_idx[edge[1]]
            edges.append([from_idx, to_idx])
            edges.append([to_idx, from_idx])  # 无向图，添加反向边
        
        if not edges:
            # 如果没有边，创建一个自环
            edges = [[0, 0]]
        
        return torch.tensor(edges).t().contiguous()
    
    def _generate_optimized_state_topology(self) -> Dict[str, Any]:
        """生成优化后的状态拓扑"""
        # 使用TGN学习到的连接概率
        with torch.no_grad():
            state_features = self._prepare_state_features()
            communication_probs = self.neural_state_learner.compute_communication_probabilities(state_features)
        
        # 基于概率构建优化拓扑
        optimized_edges = []
        state_ids = [state['state_id'] for state in self.generated_fsm['states']]
        
        for i, from_state in enumerate(state_ids):
            for j, to_state in enumerate(state_ids):
                if i != j and communication_probs[i, j] > 0.5:  # 阈值过滤
                    optimized_edges.append({
                        "from_state": from_state,
                        "to_state": to_state,
                        "probability": communication_probs[i, j].item(),
                        "learned": True
                    })
        
        return {"edges": optimized_edges, "type": "optimized_state_transitions"}
    
    def _generate_optimized_communication_topology(self) -> Dict[str, Any]:
        """生成优化后的通信拓扑"""
        # 使用TGN学习到的连接概率
        with torch.no_grad():
            agent_features = self._prepare_agent_features()
            communication_probs = self.neural_communication_learner.compute_communication_probabilities(agent_features)
        
        # 基于概率构建优化拓扑
        optimized_edges = []
        agent_ids = [agent['agent_id'] for agent in self.generated_agents]
        
        for i, from_agent in enumerate(agent_ids):
            for j, to_agent in enumerate(agent_ids):
                if i != j and communication_probs[i, j] > 0.5:  # 阈值过滤
                    optimized_edges.append({
                        "from_agent": from_agent,
                        "to_agent": to_agent,
                        "probability": communication_probs[i, j].item(),
                        "learned": True
                    })
        
        return {"edges": optimized_edges, "type": "optimized_communication"}
    
    def _graph_to_dict(self, graph: nx.Graph) -> Dict[str, Any]:
        """将NetworkX图转换为字典格式"""
        return {
            "nodes": list(graph.nodes(data=True)),
            "edges": list(graph.edges(data=True)),
            "type": "networkx_graph"
        }
    
    def save_fsm_mas_system(self, output_path: str):
        """保存完整的FSM-MAS系统"""
        if not all([self.generated_agents, self.generated_fsm, 
                   self.state_topology_graph, self.listening_topology_graph]):
            raise ValueError("FSM-MAS system not fully generated")
        
        system_data = {
            "agents": self.generated_agents,
            "fsm": self.generated_fsm,
            "state_topology": self._graph_to_dict(self.state_topology_graph),
            "listening_topology": self._graph_to_dict(self.listening_topology_graph),
            "neural_learning_enabled": self.use_neural_learning
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(system_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 FSM-MAS系统已保存到 {output_path}")
    
    def create_executable_system(self) -> MultiAgentSystem:
        """
        创建可执行的多智能体系统
        
        将生成和优化后的FSM-MAS转换为可执行的MultiAgentSystem
        
        Returns:
            MultiAgentSystem: 可执行的多智能体系统
        """
        if not all([self.generated_agents, self.generated_fsm]):
            raise ValueError("FSM-MAS system not fully generated. Please call generate_complete_fsm_mas first.")
        
        print("🔧 创建可执行的多智能体系统...")
        
        # 创建MultiAgentSystem实例
        executable_system = MultiAgentSystem(
            agents_json=self.generated_agents,
            states_json=self.generated_fsm
        )
        
        print("✅ 可执行系统创建完成")
        return executable_system
    
    def execute_task(self, task_input: str, max_transitions: int = 10) -> Tuple[str, float]:
        """
        执行任务
        
        Args:
            task_input: 任务输入
            max_transitions: 最大状态转移次数
            
        Returns:
            Tuple[str, float]: (执行结果, 总成本)
        """
        if not all([self.generated_agents, self.generated_fsm]):
            raise ValueError("FSM-MAS system not fully generated. Please call generate_complete_fsm_mas first.")
        
        print(f"🚀 开始执行任务: {task_input}")
        
        # 创建可执行系统
        executable_system = self.create_executable_system()
        
        # 执行任务
        result, cost = executable_system.start(task_input, max_transitions)
        
        print(f"✅ 任务执行完成，成本: {cost}")
        return result, cost
    
    def generate_learn_and_execute(self, 
                                 task_description: str,
                                 task_input: str,
                                 available_tools: List[str] = None,
                                 training_episodes: int = 50,
                                 max_transitions: int = 10) -> Dict[str, Any]:
        """
        完整的生成-学习-执行流程
        
        Args:
            task_description: 任务描述（用于生成FSM）
            task_input: 具体任务输入（用于执行）
            available_tools: 可用工具
            training_episodes: TGN训练轮数
            max_transitions: 执行时最大状态转移次数
            
        Returns:
            完整的结果包含生成、学习和执行的所有信息
        """
        print("🌟 开始完整的生成-学习-执行流程...")
        
        # Step 1: 生成FSM-MAS系统
        fsm_mas_config = self.generate_complete_fsm_mas(task_description, available_tools)
        
        # Step 2: TGN学习优化
        learning_results = self.learn_optimal_topologies(training_episodes)
        
        # Step 3: 执行任务
        execution_result, execution_cost = self.execute_task(task_input, max_transitions)
        
        # 构建完整结果
        complete_results = {
            'task_description': task_description,
            'task_input': task_input,
            'fsm_mas_config': fsm_mas_config,
            'learning_results': learning_results,
            'execution_result': execution_result,
            'execution_cost': execution_cost,
            'workflow': 'generate_learn_execute'
        }
        
        print("🎉 完整流程执行完成！")
        return complete_results
    
    def load_fsm_mas_system(self, input_path: str):
        """加载FSM-MAS系统"""
        with open(input_path, 'r', encoding='utf-8') as f:
            system_data = json.load(f)
        
        self.generated_agents = system_data['agents']
        self.generated_fsm = system_data['fsm']
        
        # 重建图结构
        self.state_topology_graph = nx.DiGraph()
        state_topo = system_data['state_topology']
        self.state_topology_graph.add_nodes_from(state_topo['nodes'])
        self.state_topology_graph.add_edges_from(state_topo['edges'])
        
        self.listening_topology_graph = nx.Graph()
        listening_topo = system_data['listening_topology']
        self.listening_topology_graph.add_nodes_from(listening_topo['nodes'])
        self.listening_topology_graph.add_edges_from(listening_topo['edges'])
        
        self.use_neural_learning = system_data.get('neural_learning_enabled', False)
        
        if self.use_neural_learning:
            self._initialize_neural_learners()
        
        print(f"📂 FSM-MAS系统已从 {input_path} 加载")


# 便捷函数
def create_fsm_mas_generator(use_neural_learning: bool = True,
                           memory_dimension: int = 128,
                           temporal_dimension: int = 32) -> FSMMultiAgentSystemGenerator:
    """创建FSM-MAS生成器"""
    return FSMMultiAgentSystemGenerator(
        use_neural_learning=use_neural_learning,
        memory_dimension=memory_dimension,
        temporal_dimension=temporal_dimension
    )


def generate_and_learn_fsm_mas(task_description: str,
                             available_tools: List[str] = None,
                             training_episodes: int = 100,
                             output_path: str = None) -> Dict[str, Any]:
    """一键生成并学习FSM-MAS系统"""
    
    # 创建生成器
    generator = create_fsm_mas_generator()
    
    # 生成完整系统
    fsm_mas_config = generator.generate_complete_fsm_mas(task_description, available_tools)
    
    # 学习最优拓扑
    learning_results = generator.learn_optimal_topologies(training_episodes)
    
    # 合并结果
    complete_results = {
        **fsm_mas_config,
        "learning_results": learning_results
    }
    
    # 保存结果
    if output_path:
        generator.save_fsm_mas_system(output_path)
    
    return complete_results


def generate_learn_and_execute_fsm_mas(task_description: str,
                                     task_input: str,
                                     available_tools: List[str] = None,
                                     training_episodes: int = 50,
                                     max_transitions: int = 10,
                                     output_path: str = None) -> Dict[str, Any]:
    """一键生成、学习并执行FSM-MAS系统"""
    
    # 创建生成器
    generator = create_fsm_mas_generator()
    
    # 执行完整流程
    complete_results = generator.generate_learn_and_execute(
        task_description=task_description,
        task_input=task_input,
        available_tools=available_tools,
        training_episodes=training_episodes,
        max_transitions=max_transitions
    )
    
    # 保存结果
    if output_path:
        generator.save_fsm_mas_system(output_path)
    
    return complete_results


# 导出主要类和函数
__all__ = [
    'FSMMultiAgentSystemGenerator',
    'create_fsm_mas_generator',
    'generate_and_learn_fsm_mas',
    'generate_learn_and_execute_fsm_mas'
]

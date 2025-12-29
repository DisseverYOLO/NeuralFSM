"""
FSM Multi-Agent System Training Script V2
FSM多智能体系统训练脚本 V2

✅ 完全适配新FSM架构：
1. 一个状态对应一个智能体 (State → Agent Mapping)
2. 同时优化状态转移概率 (State Transition Optimization)
3. 同时优化监听通信路径 (Listener Communication Path Optimization)
4. 使用FSMStateManager管理状态
5. 集成FSMTemporalGraph学习最优拓扑

🎯 损失函数设计（三目标优化）：
- 组合损失 = α * 策略梯度损失 + β * 状态转移损失 + γ * 监听路径损失
- α (策略梯度权重): 1.0 - 直接优化任务准确率
- β (状态转移权重): 0.3 - 学习最优状态转移序列
- γ (监听路径权重): 0.2 - 学习最优通信路径
"""

import asyncio
import argparse
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random

# 添加路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager
from neural_fsm_mas.agent_topology.fsm_state_manager import FSMStateManager
from neural_fsm_mas.temporal_networks.fsm_tgn import FSMTemporalGraph
from neural_fsm_mas.training_data.unified_data_processor import UnifiedDataProcessor
from neural_fsm_mas.domain_prompts.prompt_manager import DomainPromptManager
from baseclass.FSM_Gen import Generate_Agent_Description


class FSMMultiAgentSystemTrainerV2:
    """
    FSM多智能体系统训练器 V2
    
    核心功能：
    1. ✅ 一个状态对应一个智能体
    2. ✅ 使用FSMStateManager管理状态转移
    3. ✅ 使用FSMTemporalGraph同时学习：
       - 状态转移概率
       - 监听通信路径
    4. ✅ 三目标组合损失优化
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 dataset_root: str = "./datasets",
                 output_dir: str = "./fsm_mas_v2_outputs"):
        
        self.config = config
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化统一数据处理器
        self.data_processor = UnifiedDataProcessor(str(self.dataset_root))
        
        # ✨ 初始化文本嵌入模型（用于问题嵌入）
        from neural_fsm_mas.embeddings.text_embedding import get_embedding_model
        self.text_embedding_model = get_embedding_model()
        print(f"📦 文本嵌入模型已加载，嵌入维度: {self.text_embedding_model.embedding_dim}")
        
        # 训练状态
        self.training_history = []
        self.domain_topologies: Dict[str, MultiAgentTopologyManager] = {}
        self.domain_fsm_managers: Dict[str, FSMStateManager] = {}
        self.domain_fsm_tgns: Dict[str, FSMTemporalGraph] = {}
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 使用设备: {self.device}")
        
        # ✨ 初始化成本跟踪器
        from neural_fsm_mas.utils.llm_cost_tracker import create_cost_tracker, set_global_cost_tracker
        model_name = self.config.get('llm_name', 'gpt-5-nano')
        self.cost_tracker = create_cost_tracker(model_name)
        # ✨ 设置全局成本跟踪器，使LLM客户端能够自动记录成本
        set_global_cost_tracker(self.cost_tracker)
        print(f"💰 成本跟踪器已初始化（模型: {model_name}）")
        
        # ✨ 初始化四目标损失函数（成本损失核心）
        # 基准成本说明：
        #   - gpt-5-nano 每问题实际成本约 $0.0016
        #   - 设置 baseline_cost = $0.005 (约为实际3倍) 使得：
        #     - 成本低于基准 → 负向惩罚较小
        #     - 成本高于基准 → 正向惩罚增加
        #   - 设置 cost_scale = 100 使得成本损失量级与其他损失相近
        from neural_fsm_mas.losses.cost_loss import CostRegularizedLoss
        baseline_cost = self.config.get('baseline_cost', 0.005)  # $0.005/问题
        cost_scale = self.config.get('cost_scale', 100)  # 缩放因子
        self.cost_loss_fn = CostRegularizedLoss(
            alpha=self.config.get('policy_gradient_weight', 1.0),
            beta=self.config.get('transition_loss_weight', 0.3),
            gamma=self.config.get('listener_loss_weight', 0.2),
            delta=self.config.get('cost_loss_weight', 0.1),
            baseline_cost=baseline_cost,
            cost_scale=cost_scale
        )
        print(f"📊 四目标损失函数已初始化（基准成本: ${baseline_cost:.4f}, 缩放: {cost_scale}）")
        
        # ✨ 初始化状态描述优化器（新增）
        self.prompt_optimizers: Dict[str, Any] = {}
        if self.config.get('enable_prompt_optimization', False):
            print(f"✨ 状态描述优化: 已启用")
        
        # ✨ 初始化攻击注入器（新增）
        self.attack_injector = None
        self.attack_stats = {
            'total_attacks': 0,
            'frequency_attacks': 0,
            'semantic_attacks': 0,
            'selfish_attacks': 0,
            'attacked_agents': set()
        }
        
        # ✨ 初始化异常检测历史数据（用于ProtectedTGN的异常检测）
        # 这些数据会在每个问题开始时更新，用于检测频率和语义异常
        self.anomaly_detection_history = {
            'message_counts': {},  # {agent_id: [count1, count2, ...]} 用于频率异常检测
            'embeddings': {},      # {agent_id: [emb1, emb2, ...]} 用于语义异常检测
            'messages': {}         # ✨ {agent_id: str} 用于语义一致性检测（BERT）
        }
        if self.config.get('enable_attack', False):
            from neural_fsm_mas.defense_mechanisms import (
                create_frequency_attacker,
                create_semantic_attacker,
                create_selfish_attacker
            )
            
            attack_config = self.config.get('attack', {})
            attack_type = attack_config.get('attack_type', 'selfish')
            attack_ratio = attack_config.get('attack_ratio', 0.3)
            attack_strength = attack_config.get('attack_strength', 1.5)
            attack_seed = attack_config.get('attack_seed', self.config.get('random_seed', 42))
            
            # 创建对应类型的攻击器
            if attack_type == 'frequency':
                self.attack_injector = create_frequency_attacker(attack_ratio, attack_strength, random_seed=attack_seed)
            elif attack_type == 'semantic':
                self.attack_injector = create_semantic_attacker(attack_ratio, attack_strength, random_seed=attack_seed)
            elif attack_type == 'selfish':
                self.attack_injector = create_selfish_attacker(attack_ratio, attack_strength, random_seed=attack_seed)
            
            print(f"🔴 攻击注入器: 已启用 (类型: {attack_type}, 比例: {attack_ratio*100:.0f}%, 强度: {attack_strength})")
    
    def create_domain_fsm_system(self, domain: str, category: Optional[str] = None) -> Tuple[MultiAgentTopologyManager, FSMStateManager, FSMTemporalGraph]:
        """
        为领域创建完整的FSM系统
        
        Args:
            domain: 领域名称 (mmlu/gsm8k/humaneval/hotpotqa/alfworld/math)
            category: MMLU类别（仅MMLU需要，用于加载类别级FSM）
        
        Returns:
            (拓扑管理器, FSM状态管理器, FSM-TGN)
        """
        print(f"\n🏗️  为领域 {domain}" + (f" (类别: {category})" if category else "") + " 创建FSM系统...")
        
        force_regenerate = self.config.get('force_regenerate_fsm', False)
        
        # MMLU特殊处理：尝试加载类别级FSM（可被 force_regenerate 覆盖）
        if (not force_regenerate) and domain == 'mmlu' and category and hasattr(self, 'fsm_cache_manager'):
            cached = self._load_category_fsm_if_available(domain, category)
            if cached:
                print(f"  ✅ 使用类别级FSM: {category}")
                return self._create_fsm_from_cache(cached, domain)
        
        # 其他情况：使用默认方法或缓存（可被 force_regenerate 覆盖）
        if (not force_regenerate) and hasattr(self, 'fsm_cache_manager') and self.fsm_cache_manager.has_cache(domain):
            cached = self.fsm_cache_manager.load_fsm(domain)
            print(f"  ✅ 使用缓存的FSM（force_regenerate_fsm=False）")
            return self._create_fsm_from_cache(cached, domain)
        
        # 1. 获取智能体角色
        agent_roles = self._get_agent_roles_for_domain(domain)
        print(f"  智能体角色: {agent_roles}")
        
        # 2. 创建拓扑管理器
        topology = MultiAgentTopologyManager(
            task_domain=domain,
            language_model_name=self.config.get('llm_name', 'gpt-5-nano'),
            agent_role_names=agent_roles,
            decision_strategy='final_decision',
            enable_spatial_optimization=False,
            enable_temporal_optimization=False,
            use_neural_temporal_graph=True,
            memory_bank_dimension=self.config.get('memory_dim', 128),
            temporal_encoding_dimension=self.config.get('time_dim', 32)
        )
        
        # 3. 创建FSM状态管理器
        agent_ids = list(topology.agent_execution_nodes.keys())
        fsm_manager = FSMStateManager(agent_ids)
        
        # 4. 自动生成FSM状态（使用增强FSM生成器进行细粒度状态划分）
        print(f"  🧬 使用增强FSM生成器为领域 {domain} 生成细粒度状态...")
        if hasattr(self, 'fsm_generator') and hasattr(self, 'fsm_cache_manager'):
            try:
                complete_system, token_cost = self.fsm_generator.generate_complete_mas(dataset=domain)
                fsm_config = complete_system.get('fsm', {})
                agents = complete_system.get('agents', [])
                
                # 将完整FSM配置写入缓存，便于下次复用
                print(f"  💾 缓存增强FSM配置（domain={domain}）")
                self.fsm_cache_manager.save_fsm(
                    domain=domain,
                    fsm_config=fsm_config,
                    agents=agents
                )
                
                # 使用与 _create_fsm_from_cache 相同的恢复逻辑，确保状态/转移规则一致
                cached_wrapper = {
                    'fsm_config': fsm_config,
                    'agents': agents
                }
                topology, fsm_manager, fsm_tgn = self._create_fsm_from_cache(cached_wrapper, domain)
                
                # ✨ 打印所有状态的详细信息
                print(f"\n  📋 所有FSM状态:")
                initial_state = fsm_manager.get_initial_state()
                initial_state_id = initial_state.state_id if initial_state else None
                
                for state_id in sorted(fsm_manager.states.keys()):
                    state = fsm_manager.states[state_id]
                    marker = " [INITIAL]" if state_id == initial_state_id else ""
                    marker += " [FINAL]" if state.is_final else ""
                    print(f"    State {state_id}: {state.state_name}{marker}")
                    
                    # 获取智能体名称
                    agent_id = state.responsible_agent_id
                    agent_name = "Unknown"
                    if agent_id in topology.agent_execution_nodes:
                        agent_name = topology.agent_execution_nodes[agent_id].agent_role
                    print(f"      负责智能体: {agent_id} ({agent_name})")
                    
                    if state.description:
                        # 截断过长的描述
                        desc = state.description[:150] + "..." if len(state.description) > 150 else state.description
                        print(f"      状态描述: {desc}")
                print()  # 空行分隔
                
                # ✨ 如果启用状态描述优化，创建优化器（新增）
                if self.config.get('enable_prompt_optimization', False) and domain not in self.prompt_optimizers:
                    from neural_fsm_mas.prompt_optimization import create_prompt_optimizer
                    max_trans_threshold = self.config.get('max_transitions_threshold', 3)
                    accuracy_threshold = self.config.get('accuracy_enhancement_threshold', 5)
                    efficiency_threshold = self.config.get('efficiency_enhancement_threshold', 3)
                    self.prompt_optimizers[domain] = create_prompt_optimizer(
                        domain,
                        max_trans_threshold,
                        accuracy_threshold,
                        efficiency_threshold
                    )
                    print(f"  📝 状态描述优化器已创建")
                
                return topology, fsm_manager, fsm_tgn
            except Exception as e:
                print(f"  ⚠️  使用增强FSM生成器失败，回退到简单的一状态一智能体策略: {e}")
        
        # 回退策略：一个状态对应一个智能体（旧逻辑）
        self._generate_fsm_states(fsm_manager, agent_roles, agent_ids, domain)
        
        # 5. 创建FSM-TGN网络
        num_states = len(fsm_manager.states)
        num_agents = len(agent_ids)
        
        # 注意：NeuralTemporalGraph 使用参数名 temporal_dimension，
        # 这里统一映射，避免关键字不匹配错误
        fsm_tgn = FSMTemporalGraph(
            agent_feature_dim=self.config.get('agent_embedding_dim', 256),
            state_feature_dim=self.config.get('state_feature_dim', 256),
            num_states=num_states,
            num_agents=num_agents,
            hidden_dim=128,
            use_base_tgn=True,
            memory_dimension=self.config.get('memory_dim', 128),
            temporal_dimension=self.config.get('time_dim', 32),
            communication_edge_dim=64
        ).to(self.device)
        
        # ✨ 如果启用保护机制，包装FSM-TGN为ProtectedTGN（新增）
        use_protection = self.config.get('use_protection', False)
        if use_protection:
            try:
                from neural_fsm_mas.defense_mechanisms import ProtectedTGN
                import networkx as nx
                
                # 构建通信图（基于FSM的监听关系）
                communication_graph = nx.DiGraph()
                for state_id, state in fsm_manager.states.items():
                    # ✨ 监听关系在fsm_manager.listeners中，而不是state.listeners
                    for listener_id in fsm_manager.listeners.get(state_id, []):
                        communication_graph.add_edge(state_id, listener_id)
                
                # 如果图为空，创建全连接图
                if len(communication_graph.nodes()) == 0:
                    communication_graph = nx.complete_graph(num_agents, nx.DiGraph())
                
                # 获取保护配置
                protection_config = self.config.get('protection', {})
                
                # 包装FSM-TGN的base_tgn（如果存在）
                if hasattr(fsm_tgn, 'base_tgn') and fsm_tgn.base_tgn is not None:
                    protected_base_tgn = ProtectedTGN(
                        tgn_model=fsm_tgn.base_tgn,
                        feature_dim=self.config.get('agent_embedding_dim', 256),
                        graph=communication_graph,
                        w_betweenness=protection_config.get('w_betweenness', 0.6),
                        w_pagerank=protection_config.get('w_pagerank', 0.4),
                        lambda_freq=protection_config.get('lambda_freq', 0.3),
                        lambda_semantic=protection_config.get('lambda_semantic', 0.7),
                        weight_hidden_dim=protection_config.get('weight_hidden_dim', 64),
                        lambda_protect=protection_config.get('lambda_protect', 0.1),
                        lambda_reg=protection_config.get('lambda_reg', 0.01),
                        learnable_weights=protection_config.get('learnable_weights', True)
                    ).to(self.device)  # ✨ 关键修复：将ProtectedTGN移动到正确设备
                    
                    # ✨ 设置文本嵌入模型（用于语义一致性检测）
                    if hasattr(self, 'text_embedding_model') and self.text_embedding_model is not None:
                        protected_base_tgn.set_text_embedding_model(self.text_embedding_model)
                    
                    fsm_tgn.base_tgn = protected_base_tgn
                    print(f"  🛡️  保护机制已集成到FSM-TGN（设备: {self.device}）")
            except Exception as e:
                print(f"  ⚠️  保护机制集成失败: {e}")
                print(f"  将继续使用标准FSM-TGN")
        
        # 6. 集成到拓扑管理器
        topology.use_fsm_mode = True
        topology.fsm_state_manager = fsm_manager
        
        print(f"✅ FSM系统创建完成")
        print(fsm_manager.get_state_summary())
        
        return topology, fsm_manager, fsm_tgn
    
    def _load_category_fsm_if_available(self, domain: str, category: str) -> Optional[Dict]:
        """
        尝试加载类别级FSM（MMLU专用）
        
        Args:
            domain: 数据集名称
            category: 类别名称
        
        Returns:
            缓存的FSM配置，如果不存在则返回None
        """
        if not hasattr(self, 'fsm_cache_manager'):
            return None
        
        try:
            if self.fsm_cache_manager.has_cache(domain, category):
                return self.fsm_cache_manager.load_fsm(domain, category)
        except Exception as e:
            print(f"  ⚠️  加载类别FSM失败: {e}")
        
        return None
    
    def _create_fsm_from_cache(self, cached: Dict, domain: str) -> Tuple[MultiAgentTopologyManager, FSMStateManager, FSMTemporalGraph]:
        """
        从缓存创建FSM系统
        
        ✨ 正确恢复Enhanced_FSM_Gen.py生成的完整FSM配置
        
        Args:
            cached: 缓存的FSM配置（包含agents和fsm_config）
            domain: 数据集名称
        
        Returns:
            (拓扑管理器, FSM状态管理器, FSM-TGN)
        """
        fsm_config = cached.get('fsm_config', {})
        agents = cached.get('agents', [])
        
        if not agents or not fsm_config:
            raise ValueError(f"Invalid cached FSM: missing agents or fsm_config")
        
        # 提取智能体角色（使用缓存的智能体名称）
        agent_roles = [agent.get('name', f"Agent{i}") for i, agent in enumerate(agents)]
        
        # 创建拓扑管理器（使用缓存的智能体）
        topology = MultiAgentTopologyManager(
            task_domain=domain,
            language_model_name=self.config.get('llm_name', 'gpt-5-nano'),
            agent_role_names=agent_roles,
            decision_strategy='final_decision',
            enable_spatial_optimization=False,
            enable_temporal_optimization=False,
            use_neural_temporal_graph=True,
            memory_bank_dimension=self.config.get('memory_dim', 128),
            temporal_encoding_dimension=self.config.get('time_dim', 32)
        )
        
        # 创建FSM状态管理器
        agent_ids = list(topology.agent_execution_nodes.keys())
        fsm_manager = FSMStateManager(agent_ids)
        
        # 建立缓存中的agent_id到拓扑实际agent_id的映射
        # 缓存中的agent_id通常是 "0","1",...，而拓扑中的节点ID是 "agent_0","agent_1",...
        cached_id_to_topology_id: Dict[str, str] = {}
        for idx, agent in enumerate(agents):
            cached_agent_id = str(agent.get('agent_id', str(idx)))
            if idx < len(agent_ids):
                cached_id_to_topology_id[cached_agent_id] = agent_ids[idx]
        
        # ✨ 从缓存的FSM配置恢复状态（关键修复）
        states = fsm_config.get('states', [])
        if not states:
            raise ValueError(f"Cached FSM has no states")
        
        # 创建agent_id到agent_index的映射（基于拓扑ID）
        agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        
        # 恢复每个状态
        for state_data in states:
            state_id = int(state_data['state_id'])
            raw_agent_id = str(state_data['agent_id'])
            
            # 将缓存中的agent_id映射到拓扑中的实际agent_id
            if raw_agent_id in cached_id_to_topology_id:
                agent_id = cached_id_to_topology_id[raw_agent_id]
            else:
                print(f"  ⚠️  Warning: State {state_id} references unknown cached agent_id {raw_agent_id}, skipping")
                continue
            
            # 再次验证映射后的agent_id是否在当前拓扑中
            if agent_id not in agent_id_to_index:
                print(f"  ⚠️  Warning: State {state_id} mapped to invalid agent_id {agent_id}, skipping")
                continue
            
            # 添加状态（包含completion_condition）
            completion_condition = state_data.get('completion_condition', '')
            fsm_manager.add_state(
                state_id=state_id,
                state_name=state_data.get('state_name', f'State_{state_id}'),
                responsible_agent_id=agent_id,
                is_initial=state_data.get('is_initial', False),
                is_final=state_data.get('is_final', False),
                description=state_data.get('instruction', state_data.get('description', '')),
                completion_condition=completion_condition  # ✨ 直接传递completion_condition
            )
            
            # 恢复监听关系（同样从缓存ID映射到拓扑ID）
            listeners = state_data.get('listeners', [])
            for raw_listener_id in listeners:
                raw_listener_id = str(raw_listener_id)
                if raw_listener_id in cached_id_to_topology_id:
                    listener_id = cached_id_to_topology_id[raw_listener_id]
                if listener_id in agent_id_to_index:
                    fsm_manager.add_listener(
                        state_id=state_id,
                        listener_agent_id=listener_id
                    )
                else:
                        print(f"  ⚠️  Warning: Mapped listener_id {listener_id} not in topology for state {state_id}")
        
        # 恢复状态转移规则（存储到fsm_manager中）
        transitions = fsm_config.get('transitions', [])
        if transitions:
            # 存储转移规则（用于后续的状态转移判断）
            if not hasattr(fsm_manager, 'transition_rules'):
                fsm_manager.transition_rules = []
            # 确保转移规则中的状态ID是整数类型，并规范化字段名
            normalized_transitions = []
            for trans in transitions:
                normalized_trans = {
                    'from_state': int(trans.get('from_state', trans.get('from', 0))),
                    'to_state': int(trans.get('to_state', trans.get('to', 0))),
                    'condition': trans.get('condition', trans.get('description', '')),
                    'priority': trans.get('priority', 1.0)
                }
                normalized_transitions.append(normalized_trans)
            fsm_manager.transition_rules = normalized_transitions
            print(f"  ✅ 加载了 {len(normalized_transitions)} 个状态转移规则")

            # ✨ 在一开始打印出初始的状态转移情况
            print("  🔰 初始FSM状态转移表:")
            for rule in normalized_transitions:
                fs = rule['from_state']
                ts = rule['to_state']
                cond = rule.get('condition', '')
                print(f"    State {fs} → State {ts}: {cond}")
        else:
            print(f"  ⚠️  警告：FSM配置中没有找到状态转移规则")
        
        # 设置初始状态
        initial_states = [s for s in fsm_manager.states.values() if s.is_initial]
        if initial_states:
            fsm_manager.current_state_id = initial_states[0].state_id
        elif fsm_manager.states:
            # 如果没有初始状态，使用第一个状态
            first_state_id = min(fsm_manager.states.keys())
            fsm_manager.current_state_id = first_state_id
            fsm_manager.states[first_state_id].is_initial = True
        
        # ✨ 如果启用保护机制，包装FSM-TGN为ProtectedTGN（新增）
        use_protection = self.config.get('use_protection', False)
        
        # 创建FSM-TGN
        num_states = len(fsm_manager.states)
        num_agents = len(agent_ids)
        
        fsm_tgn = FSMTemporalGraph(
            agent_feature_dim=self.config.get('agent_embedding_dim', 256),
            state_feature_dim=self.config.get('state_feature_dim', 256),
            num_states=num_states,
            num_agents=num_agents,
            hidden_dim=128,
            use_base_tgn=True,
            memory_dimension=self.config.get('memory_dim', 128),
            # NeuralTemporalGraph 使用参数名 temporal_dimension，这里统一为 temporal_dimension
            temporal_dimension=self.config.get('time_dim', 32),
            communication_edge_dim=64
        ).to(self.device)
        
        # ✨ 如果启用保护机制，包装FSM-TGN为ProtectedTGN（新增）
        if use_protection:
            try:
                from neural_fsm_mas.defense_mechanisms import ProtectedTGN
                import networkx as nx
                
                # 构建通信图（基于FSM的监听关系）
                communication_graph = nx.DiGraph()
                for state_id, state in fsm_manager.states.items():
                    # ✨ 监听关系在fsm_manager.listeners中，而不是state.listeners
                    for listener_id in fsm_manager.listeners.get(state_id, []):
                        communication_graph.add_edge(state_id, listener_id)
                
                # 如果图为空，创建全连接图
                if len(communication_graph.nodes()) == 0:
                    communication_graph = nx.complete_graph(num_agents, nx.DiGraph())
                
                # 获取保护配置
                protection_config = self.config.get('protection', {})
                
                # 包装FSM-TGN的base_tgn（如果存在）
                if hasattr(fsm_tgn, 'base_tgn') and fsm_tgn.base_tgn is not None:
                    protected_base_tgn = ProtectedTGN(
                        tgn_model=fsm_tgn.base_tgn,
                        feature_dim=self.config.get('agent_embedding_dim', 256),
                        graph=communication_graph,
                        w_betweenness=protection_config.get('w_betweenness', 0.6),
                        w_pagerank=protection_config.get('w_pagerank', 0.4),
                        lambda_freq=protection_config.get('lambda_freq', 0.3),
                        lambda_semantic=protection_config.get('lambda_semantic', 0.7),
                        weight_hidden_dim=protection_config.get('weight_hidden_dim', 64),
                        lambda_protect=protection_config.get('lambda_protect', 0.1),
                        lambda_reg=protection_config.get('lambda_reg', 0.01),
                        learnable_weights=protection_config.get('learnable_weights', True)
                    ).to(self.device)  # ✨ 关键修复：将ProtectedTGN移动到正确设备
                    
                    # ✨ 设置文本嵌入模型（用于语义一致性检测）
                    if hasattr(self, 'text_embedding_model') and self.text_embedding_model is not None:
                        protected_base_tgn.set_text_embedding_model(self.text_embedding_model)
                    
                    fsm_tgn.base_tgn = protected_base_tgn
                    print(f"  🛡️  保护机制已集成到FSM-TGN（从缓存，设备: {self.device}）")
            except Exception as e:
                print(f"  ⚠️  保护机制集成失败: {e}")
        
        topology.use_fsm_mode = True
        topology.fsm_state_manager = fsm_manager
        
        print(f"  ✅ 从缓存恢复FSM: {num_states}个状态, {num_agents}个智能体, {len(transitions)}个转移规则")
        
        # ✨ 打印所有状态的详细信息
        print(f"\n  📋 所有FSM状态:")
        initial_state = fsm_manager.get_initial_state()
        initial_state_id = initial_state.state_id if initial_state else None
        
        for state_id in sorted(fsm_manager.states.keys()):
            state = fsm_manager.states[state_id]
            marker = " [INITIAL]" if state_id == initial_state_id else ""
            marker += " [FINAL]" if state.is_final else ""
            print(f"    State {state_id}: {state.state_name}{marker}")
            
            # 获取智能体名称
            agent_id = state.responsible_agent_id
            agent_name = "Unknown"
            if agent_id in topology.agent_execution_nodes:
                agent_name = topology.agent_execution_nodes[agent_id].agent_role
            print(f"      负责智能体: {agent_id} ({agent_name})")
            
            if state.description:
                # 截断过长的描述
                desc = state.description[:150] + "..." if len(state.description) > 150 else state.description
                print(f"      状态描述: {desc}")
        print()  # 空行分隔
        
        return topology, fsm_manager, fsm_tgn
    
    def _get_agent_roles_for_domain(self, domain: str) -> List[str]:
        """
        获取领域的智能体角色
        
        ✨ 优先级：
        1. 缓存的FSM中的智能体（Enhanced_FSM_Gen.py生成）
        2. prompt_manager.py的预定义角色（fallback）
        3. 默认角色（最后fallback）
        """
        # 优先级1: 优先使用缓存的FSM中的智能体
        if hasattr(self, 'fsm_cache_manager') and self.fsm_cache_manager:
            try:
                if self.fsm_cache_manager.has_cache(domain):
                    cached = self.fsm_cache_manager.load_fsm(domain)
                    agents = cached.get('agents', [])
                    if agents:
                        agent_roles = [agent.get('name', f"Agent{i}") for i, agent in enumerate(agents)]
                        print(f"  ✅ 使用缓存的FSM智能体 ({len(agent_roles)}个)")
                        return agent_roles
            except Exception as e:
                print(f"  ⚠️  加载缓存FSM失败: {e}")
        
        # 优先级2: 使用prompt_manager.py的预定义角色（fallback）
        try:
            prompt_set = DomainPromptManager.get_manager(domain)
            agent_roles = prompt_set.get_available_roles()
            print(f"  ⚠️  使用prompt_manager预定义角色 ({len(agent_roles)}个)")
            return agent_roles
        except Exception as e:
            print(f"  ⚠️  prompt_manager加载失败: {e}")
        
        # 优先级3: 使用默认角色（最后fallback）
        print(f"  ⚠️  使用默认角色")
        role_mapping = {
            "mmlu": ["Knowledge Expert", "Subject Specialist", "Critical Analyzer", "Decision Maker"],
            "gsm8k": ["Problem Analyzer", "Math Solver", "Calculation Verifier", "Solution Critic"],
            "humaneval": ["Code Designer", "Code Writer", "Code Reviewer", "Test Engineer"],
            "hotpotqa": ["Question Analyzer", "Document Retriever", "Fact Extractor", "Multi-hop Reasoner", "Answer Synthesizer"],
            "alfworld": ["Task Parser", "Environment Explorer", "Action Planner", "Action Executor", "State Monitor"],
            "math": ["Problem Analyzer", "Math Concept Expert", "Strategy Designer", "Step-by-step Solver", "Symbolic Calculator"],
            "gaia": ["Task Interpreter", "Tool Planner", "Information Extractor", "Answer Verifier"]
        }
        agent_roles = role_mapping.get(domain, ["Agent1", "Agent2", "Agent3", "Agent4"])
        return agent_roles
    
    def _generate_fsm_states(self, 
                            fsm_manager: FSMStateManager,
                            agent_roles: List[str],
                            agent_ids: List[str],
                            domain: str):
        """
        自动生成FSM状态（一个状态对应一个智能体）
        
        Args:
            fsm_manager: FSM状态管理器
            agent_roles: 智能体角色列表
            agent_ids: 智能体ID列表
            domain: 领域名称
        """
        # 状态名称映射
        state_name_templates = {
            "mmlu": [
                "Knowledge Retrieval",
                "Subject Analysis",
                "Critical Thinking",
                "Final Decision"
            ],
            "gsm8k": [
                "Problem Understanding",
                "Mathematical Reasoning",
                "Calculation Verification",
                "Solution Finalization"
            ],
            "humaneval": [
                "Code Planning",
                "Code Implementation",
                "Code Review",
                "Testing & Validation"
            ],
            "gaia": [
                "Task Understanding",
                "Planning & Tooling",
                "Evidence Extraction",
                "Final Answer"
            ]
        }
        
        state_names = state_name_templates.get(domain, [f"State{i}" for i in range(len(agent_roles))])
        
        # 确保状态数与智能体数相同
        if len(state_names) < len(agent_roles):
            state_names.extend([f"State{i}" for i in range(len(state_names), len(agent_roles))])
        
        # 添加状态（一个状态对应一个智能体）
        for i, (role, agent_id) in enumerate(zip(agent_roles, agent_ids)):
            fsm_manager.add_state(
                state_id=i,
                state_name=state_names[i],
                responsible_agent_id=agent_id,
                is_initial=(i == 0),  # 第一个状态为初始状态
                is_final=(i == len(agent_roles) - 1),  # 最后一个状态为最终状态
                description=f"{state_names[i]} handled by {role}"
            )
        
        # 初始化监听关系（默认每个状态的输出传给下一个状态的智能体）
        for i in range(len(agent_roles) - 1):
            fsm_manager.add_listener(state_id=i, listener_agent_id=agent_ids[i + 1])
    
    async def train_domain(self, 
                          domain: str,
                          num_epochs: int = 50) -> Dict[str, Any]:
        """
        训练单个领域的FSM系统
        
        Args:
            domain: 领域名称
            num_epochs: 训练轮数
        
        Returns:
            训练结果
        """
        print(f"\n{'='*80}")
        print(f"🎯 开始训练领域: {domain}")
        print(f"{'='*80}")
        
        # 1. 创建FSM系统
        # MMLU特殊处理：如果启用类别级FSM，不在这里创建，而是在训练时动态加载
        use_category_fsm = (domain == 'mmlu' and 
                           self.config.get('mmlu_use_category_fsm', False))  # ✨ 默认False，使用统一FSM
        
        if use_category_fsm:
            print(f"  ℹ️  MMLU使用类别级FSM，将在训练时动态加载")
            # 创建一个占位符，实际FSM在训练时按类别加载
            topology, fsm_manager, fsm_tgn = self.create_domain_fsm_system(domain)
        else:
            topology, fsm_manager, fsm_tgn = self.create_domain_fsm_system(domain)
        
        # 保存引用
        self.domain_topologies[domain] = topology
        self.domain_fsm_managers[domain] = fsm_manager
        self.domain_fsm_tgns[domain] = fsm_tgn
        
        # MMLU类别级FSM缓存
        if use_category_fsm:
            self.mmlu_category_fsms = {}  # {category: (topology, fsm_manager, fsm_tgn)}
        
        # 2. 准备训练数据（统一使用 train/val/test 三集）
        # ✨ GAIA：按 level(1/2/3) 过滤（由外部实验脚本写入 config['gaia_level']）
        try:
            if domain == "gaia" and hasattr(self, "data_processor") and self.data_processor is not None:
                self.data_processor.gaia_level = self.config.get("gaia_level", None)
        except Exception:
            pass
        self.data_processor.create_domain_splits(
            domain=domain,
            train_ratio=self.config.get('train_ratio', 0.7),
            val_ratio=self.config.get('val_ratio', 0.15),
            test_ratio=self.config.get('test_ratio', 0.15),
            random_seed=self.config.get('random_seed', 42)
        )
        
        training_batches = self.data_processor.prepare_multi_agent_training_data(
            domain=domain,
            split="train",
            batch_size=self.config.get('batch_size', 16)
        )
        
        # ✨ 支持随机采样指定数量的训练样本
        train_samples = self.config.get('train_samples', None)
        if train_samples is not None and train_samples > 0:
            import random
            # 将所有批次的样本合并
            all_train_samples = []
            for batch in training_batches:
                all_train_samples.extend(batch)
            
            # 随机采样
            if len(all_train_samples) > train_samples:
                random.seed(self.config.get('random_seed', 42))
                all_train_samples = random.sample(all_train_samples, train_samples)
                print(f"  🎲 随机采样 {train_samples} 条训练样本（原始 {sum(len(b) for b in training_batches)} 条）")
            
            # 重新构建批次
            batch_size = self.config.get('batch_size', 16)
            training_batches = [
                all_train_samples[i:i+batch_size] 
                for i in range(0, len(all_train_samples), batch_size)
            ]
        
        # ✨ 支持只训练模式（跳过验证和测试）
        train_only = self.config.get('train_only', False)
        
        if train_only:
            validation_batches = []
            test_batches = []
            print(f"  ⚡ 只训练模式：跳过验证和测试阶段")
        else:
            validation_batches = self.data_processor.prepare_multi_agent_validation_data(
            domain=domain,
            split="val",
            batch_size=self.config.get('batch_size', 16)
        )
        
        test_batches = self.data_processor.prepare_multi_agent_training_data(
            domain=domain,
            split="test",
            batch_size=self.config.get('batch_size', 16)
        )
        
        print(f"📊 训练批次: {len(training_batches)}, 验证批次: {len(validation_batches)}, 测试批次: {len(test_batches)}")
        
        # 3. 设置优化器
        optimizer = optim.Adam(fsm_tgn.parameters(), lr=self.config.get('learning_rate', 0.001))
        
        # 4. 训练循环
        training_results = {
            'domain': domain,
            'epoch_losses': [],
            'epoch_policy_losses': [],
            'epoch_transition_losses': [],
            'epoch_listener_losses': [],
            'epoch_cost_losses': [],  # ✨ 新增：LLM成本损失
            'epoch_protection_losses': [],  # ✨ 新增：保护损失
            'epoch_accuracies': [],
            # ✨ 每个epoch内每个batch的准确率列表，方便事后排查问题
            # 结构: epoch_batch_accuracies[epoch_idx][batch_idx] = accuracy
            'epoch_batch_accuracies': [],
            'validation_accuracies': [],
            'epoch_training_costs': [],  # 每个epoch的训练成本
            'epoch_validation_costs': [],  # 每个epoch的验证成本
            'best_accuracy': 0.0,
            'best_epoch': 0,
            'test_accuracy': 0.0,
            'test_cost': 0.0,  # 测试集总成本
            'total_training_cost': 0.0,  # 总训练成本
            'total_validation_cost': 0.0,  # 总验证成本
            'total_cost': 0.0,  # 总成本（训练+验证+测试）
            'cost_statistics': {},  # 详细成本统计
            # ✨ 新增：保护数据统计（仅实验2）
            'protection_statistics': {
                'epoch_avg_anomaly_scores': [],
                'epoch_avg_trust_scores': [],
                'epoch_avg_priorities': [],
                'epoch_high_anomaly_agents': []  # 每个epoch中高异常分数的agent统计
            }
        }
        
        # 记录训练开始前的成本（用于计算训练成本）
        initial_cost = 0.0
        if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
            initial_cost = self.cost_tracker.get_total_cost()
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
            
            # 记录epoch开始前的成本
            epoch_start_cost = 0.0
            if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                epoch_start_cost = self.cost_tracker.get_total_cost()
            
            # 训练阶段
            epoch_metrics = await self._train_epoch(
                domain, topology, fsm_manager, fsm_tgn, 
                training_batches, optimizer
            )
            
            # 计算该epoch的训练成本
            epoch_training_cost = 0.0
            if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                epoch_end_cost = self.cost_tracker.get_total_cost()
                epoch_training_cost = epoch_end_cost - epoch_start_cost
            
            # 验证阶段（内部会计算成本）
            # ✨ 如果验证集不为空，才执行验证
            if len(validation_batches) > 0:
                val_accuracy, epoch_validation_cost = await self._validate_epoch(
                    domain, topology, fsm_manager, validation_batches, is_test=False
            )
            else:
                # 验证集为空，跳过验证阶段
                val_accuracy = 0.0
                epoch_validation_cost = 0.0
                print(f"  ⚠️  跳过验证阶段（验证集为空）")
            
            # 统一四舍五入保留4位小数的准确率
            train_accuracy_rounded = round(epoch_metrics['accuracy'], 4)
            val_accuracy_rounded = round(val_accuracy, 4)
            
            # 记录结果
            training_results['epoch_losses'].append(epoch_metrics['total_loss'])
            training_results['epoch_policy_losses'].append(epoch_metrics['policy_loss'])
            training_results['epoch_transition_losses'].append(epoch_metrics['transition_loss'])
            training_results['epoch_listener_losses'].append(epoch_metrics['listener_loss'])
            # ✨ 新增：记录LLM成本损失
            training_results['epoch_cost_losses'].append(epoch_metrics.get('cost_loss', 0.0))
            # ✨ 新增：记录保护损失
            training_results['epoch_protection_losses'].append(epoch_metrics.get('protection_loss', 0.0))
            training_results['epoch_accuracies'].append(train_accuracy_rounded)
            training_results['epoch_batch_accuracies'].append(epoch_metrics.get('batch_accuracies', []))
            training_results['validation_accuracies'].append(val_accuracy_rounded)
            training_results['epoch_training_costs'].append(epoch_training_cost)
            training_results['epoch_validation_costs'].append(epoch_validation_cost)
            
            # ✨ 新增：记录保护数据统计（如果启用了保护机制）
            if self.config.get('use_protection', False):
                protection_stats = epoch_metrics.get('protection_statistics', {})
                training_results['protection_statistics']['epoch_avg_anomaly_scores'].append(
                    protection_stats.get('avg_anomaly_score', 0.0))
                training_results['protection_statistics']['epoch_avg_trust_scores'].append(
                    protection_stats.get('avg_trust_score', 0.0))
                training_results['protection_statistics']['epoch_avg_priorities'].append(
                    protection_stats.get('avg_priority', 0.0))
                training_results['protection_statistics']['epoch_high_anomaly_agents'].append(
                    protection_stats.get('high_anomaly_agents', []))
            
            # ✨ 更新最佳模型：如果有验证集，用验证准确率；否则用训练准确率
            if len(validation_batches) > 0:
                # 有验证集，使用验证准确率
                current_metric = val_accuracy
                current_metric_rounded = val_accuracy_rounded
                metric_name = "验证"
            else:
                # 无验证集，使用训练准确率
                current_metric = epoch_metrics['accuracy']
                current_metric_rounded = train_accuracy_rounded
                metric_name = "训练"
            
            if current_metric > training_results['best_accuracy']:
                training_results['best_accuracy'] = current_metric_rounded
                training_results['best_epoch'] = epoch + 1
                self._save_best_model(domain, fsm_tgn, epoch, current_metric_rounded)
                print(f"  💾 保存最佳模型（{metric_name}准确率: {current_metric_rounded:.4f}）")
            
            # 打印训练和验证准确率
            if len(validation_batches) > 0:
                print(f"  📊 训练准确率: {train_accuracy_rounded:.4f}, 验证准确率: {val_accuracy_rounded:.4f}")
            else:
                print(f"  📊 训练准确率: {train_accuracy_rounded:.4f}")
            # ✨ 打印所有损失（包括保护损失）
            protection_loss_str = ""
            if self.config.get('use_protection', False):
                protection_loss_str = f", 保护={epoch_metrics.get('protection_loss', 0.0):.4f}"
            print(f"  💰 损失: 总={epoch_metrics['total_loss']:.4f}, "
                  f"策略={epoch_metrics['policy_loss']:.4f}, "
                  f"转移={epoch_metrics['transition_loss']:.4f}, "
                  f"监听={epoch_metrics['listener_loss']:.4f}{protection_loss_str}, "
                  f"最大转移惩罚={epoch_metrics.get('max_transitions_penalty', 0.0):.4f}")
            print(f"  💵 成本: 训练=${epoch_training_cost:.6f}, 验证=${epoch_validation_cost:.6f}")
        
        # 计算总训练成本和总验证成本
        training_results['total_training_cost'] = sum(training_results['epoch_training_costs'])
        training_results['total_validation_cost'] = sum(training_results['epoch_validation_costs'])
        
        # 训练结束后，在测试集上进行最终评估（内部会计算成本）
        # ✨ 如果是只训练模式，跳过测试阶段
        if len(test_batches) > 0:
            test_accuracy, test_cost = await self._validate_epoch(
            domain, topology, fsm_manager, test_batches, is_test=True
        )
            test_accuracy = round(test_accuracy, 4)
            training_results['test_cost'] = test_cost
            training_results['test_accuracy'] = test_accuracy
        else:
            test_accuracy = 0.0
            test_cost = 0.0
            training_results['test_cost'] = 0.0
            training_results['test_accuracy'] = 0.0
            print(f"  ⚠️  跳过测试阶段（只训练模式）")
        training_results['total_cost'] = (
            training_results['total_training_cost'] + 
            training_results['total_validation_cost'] + 
            training_results['test_cost']
        )
        
        # 获取详细的成本统计信息
        if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
            cost_stats = self.cost_tracker.get_statistics()
            training_results['cost_statistics'] = {
                'total_calls': cost_stats['total_calls'],
                'total_input_tokens': cost_stats['total_input_tokens'],
                'total_output_tokens': cost_stats['total_output_tokens'],
                'total_cost_usd': cost_stats['total_cost_usd'],
                'avg_cost_per_episode': cost_stats['avg_cost_per_episode'],
                'cost_by_agent': cost_stats['cost_by_agent'],
                'cost_by_state': cost_stats['cost_by_state']
            }
        
        # 获取最终训练准确率（最后一个epoch的准确率）
        final_train_accuracy = training_results['epoch_accuracies'][-1] if training_results['epoch_accuracies'] else 0.0
        final_train_accuracy = round(final_train_accuracy, 4)
        
        # ✨ 计算总训练准确率（所有epoch的平均值）
        if training_results['epoch_accuracies']:
            total_train_accuracy = sum(training_results['epoch_accuracies']) / len(training_results['epoch_accuracies'])
        else:
            total_train_accuracy = 0.0
        
        # ✨ 保存总训练准确率到结果中
        total_train_accuracy = round(total_train_accuracy, 4)
        training_results['total_train_accuracy'] = total_train_accuracy
        
        print(f"\n✅ 领域 {domain} 训练完成")
        print(f"📊 最终训练准确率（最后一个epoch）: {final_train_accuracy:.4f}")
        print(f"📊 总训练准确率（所有epoch平均）: {total_train_accuracy:.4f}")
        print(f"🧪 最终测试集准确率: {test_accuracy:.4f}")
        print(f"💰 成本统计:")
        print(f"  训练成本: ${training_results['total_training_cost']:.6f} USD")
        print(f"  验证成本: ${training_results['total_validation_cost']:.6f} USD")
        print(f"  测试成本: ${training_results['test_cost']:.6f} USD")
        print(f"  总成本: ${training_results['total_cost']:.6f} USD")
        
        # ✨ 打印攻击统计（如果启用了攻击）
        if self.config.get('enable_attack', False):
            print(f"\n🔴 攻击统计:")
            print(f"  总攻击次数: {self.attack_stats['total_attacks']}")
            print(f"  频率攻击: {self.attack_stats['frequency_attacks']}")
            print(f"  语义攻击: {self.attack_stats['semantic_attacks']}")
            print(f"  自私攻击: {self.attack_stats['selfish_attacks']}")
            print(f"  被攻击节点数: {len(self.attack_stats['attacked_agents'])}")
            
            # 将攻击统计添加到结果中
            training_results['attack_stats'] = {
                'total_attacks': self.attack_stats['total_attacks'],
                'frequency_attacks': self.attack_stats['frequency_attacks'],
                'semantic_attacks': self.attack_stats['semantic_attacks'],
                'selfish_attacks': self.attack_stats['selfish_attacks'],
                'attacked_agents_count': len(self.attack_stats['attacked_agents'])
            }
        
        return training_results
    
    async def _train_epoch(self,
                          domain: str,
                          topology: MultiAgentTopologyManager,
                          fsm_manager: FSMStateManager,
                          fsm_tgn: FSMTemporalGraph,
                          training_batches: List[Dict],
                          optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """训练一个epoch"""
        
        fsm_tgn.train()

        # ✨ 防御性检查：避免与“提示优化器”变量名冲突导致 optimizer 被覆盖
        if optimizer is None or not hasattr(optimizer, "zero_grad"):
            raise ValueError("Torch optimizer is invalid (None or missing zero_grad).")
        
        total_policy_loss = 0.0
        total_transition_loss = 0.0
        total_listener_loss = 0.0
        total_cost_loss = 0.0  # ✨ 新增：LLM成本损失累加器
        total_max_transitions_penalty = 0.0
        total_protection_loss = 0.0  # ✨ 新增：保护损失累加器
        total_correct = 0
        total_questions = 0
        batch_accuracies: List[float] = []
        
        # ==================== 损失函数和惩罚项权重配置 ====================
        # 实验1: 4个损失函数 + 1个惩罚项 = 五目标优化
        # 实验2: 4个损失函数 + 1个专属损失 + 1个惩罚项 = 六目标优化
        # ==================================================================
        
        # 【4个基础损失函数】（实验1和实验2共享）
        alpha = self.config.get('policy_gradient_weight', 1.0)      # L_policy: 策略梯度损失
        beta = self.config.get('transition_loss_weight', 0.3)       # L_transition: 状态转移损失
        gamma = self.config.get('listener_loss_weight', 0.2)        # L_listener: 监听路径损失
        delta = self.config.get('cost_loss_weight', 0.1)            # L_cost: LLM成本损失（集成在四目标框架中）
        
        # 【实验2专属损失函数】
        # ✨ 降低默认权重，避免保护损失主导总损失
        epsilon = self.config.get('protection_loss_weight', 0.05)    # L_protection: 保护损失（仅实验2）
        
        # 【惩罚项】（实验1和实验2共享）
        zeta = self.config.get('max_transitions_penalty_weight', 0.5)  # P_max_trans: 最大转移惩罚（二值：0或1）
        
        print(f"  📦 开始训练，共 {len(training_batches)} 个批次")
        
        # ✨ 新增：收集保护数据统计（用于写入结果文件）
        epoch_anomaly_scores: List[float] = []
        epoch_trust_scores: List[float] = []
        epoch_priorities: List[float] = []
        epoch_high_anomaly_agents: List[List[int]] = []
        
        # 遍历所有训练批次（完整训练）
        for batch_idx, batch in enumerate(training_batches):
            # 这些必须是 Tensor；否则一旦执行失败返回 float log_prob，会导致下游 loss_fn 调用 .item() 崩溃
            batch_policy_loss = torch.tensor(0.0, device=self.device)
            batch_transition_loss = torch.tensor(0.0, device=self.device)
            batch_listener_loss = torch.tensor(0.0, device=self.device)
            batch_max_transitions_penalty = torch.tensor(0.0, device=self.device)
            batch_protection_loss = torch.tensor(0.0, device=self.device)  # ✨ 新增
            batch_correct = 0
            batch_episode_costs: List[float] = []
            
            # 当前实现中，batch 是一个由若干 question_dict 组成的列表
            # 这里对一个batch中的所有问题进行训练
            for question_data in batch:
                try:
                    # MMLU特殊处理：根据问题类别加载对应的FSM
                    current_topology = topology
                    current_fsm_manager = fsm_manager
                    current_fsm_tgn = fsm_tgn
                    
                    if domain == 'mmlu' and self.config.get('mmlu_use_category_fsm', False):  # ✨ 默认False
                        category = question_data.get('subject', 'unknown')
                        current_topology, current_fsm_manager, current_fsm_tgn = self._get_or_create_category_fsm(domain, category)
                    
                    # 格式化问题
                    formatted_question = self.data_processor.format_question_for_agents(question_data, domain)
                    
                    # ✨ 动态优化状态描述（新增）
                    if self.config.get('enable_prompt_optimization', False) and domain in self.prompt_optimizers:
                        prompt_optimizer = self.prompt_optimizers[domain]
                        
                        # 遍历所有状态，增强描述
                        for state_id, state in current_fsm_manager.states.items():
                            if state.description:
                                enhanced_desc, was_enhanced = prompt_optimizer.get_enhanced_description(
                                    state.state_name,
                                    state.description
                                )
                                # 更新状态描述
                                if was_enhanced:
                                    state.description = enhanced_desc
                    
                    # ✨ 生成问题嵌入（任务自适应）：将问题 + 简化的状态和智能体信息一起编码
                    # ✨ 成本优化：简化状态描述，只保留ID和名称，不包含完整描述（描述已在system prompt中）
                    state_summaries = []
                    for s in current_fsm_manager.states.values():
                        # 只保留状态ID和名称，不包含完整描述以节省token
                        state_summaries.append(f"State {s.state_id}: {s.state_name}")
                    state_block = "\n".join(state_summaries)
                    
                    # 只保留智能体ID和角色名称
                    agent_summaries = []
                    for agent_id, agent_node in current_topology.agent_execution_nodes.items():
                        agent_summaries.append(f"{agent_id}: {agent_node.agent_role}")
                    agent_block = "\n".join(agent_summaries)
                    
                    # ✨ 简化enriched_text，减少不必要的描述文本
                    enriched_text = (
                        f"{formatted_question}\n\n"
                        f"[FSM States: {len(state_summaries)} states]\n{state_block}\n\n"
                        f"[Agents: {len(agent_summaries)} agents]\n{agent_block}"
                    )
                    
                    question_embedding = self.text_embedding_model.encode_query(enriched_text)
                    question_embedding = question_embedding.to(self.device)
                    
                    # 执行FSM推理
                    current_fsm_manager.reset()
                    agent_features = self._get_agent_features(current_topology)
                    context_features = self._get_context_features(domain)
                    
                    # ✨ 构建通信拓扑：当前实现使用全连接（去除自环）拓扑，供TGN学习优化
                    num_agents = agent_features.size(0)
                    if num_agents > 1:
                        edge_indices = [
                            (i, j)
                            for i in range(num_agents)
                            for j in range(num_agents)
                            if i != j
                        ]
                        if edge_indices:
                            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
                        else:
                            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    else:
                        # 只有一个智能体时，通信图为空
                        edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    
                    # ✨ 攻击注入（如果启用）
                    attacked_communication_counts = None  # 用于频率攻击检测
                    if self.attack_injector is not None and self.config.get('enable_attack', False):
                        attack_config = self.config.get('attack', {})
                        attack_type = attack_config.get('attack_type', 'selfish')
                        attack_frequency = attack_config.get('attack_frequency', 'always')
                        attack_start_epoch = attack_config.get('attack_start_epoch', 0)
                        current_epoch = getattr(self, '_current_epoch', 0)

                        # ✨ 保护最终状态负责agent：不允许被选为攻击目标
                        try:
                            final_states = [s for s in current_fsm_manager.states.values() if getattr(s, 'is_final', False)]
                            if final_states and hasattr(self.attack_injector, 'set_excluded_agents'):
                                final_agent_id = final_states[0].responsible_agent_id
                                final_idx = int(final_agent_id.split('_')[-1]) if isinstance(final_agent_id, str) and '_' in final_agent_id else int(final_agent_id)
                                self.attack_injector.set_excluded_agents([final_idx], clear_if_conflict=True)
                        except Exception:
                            pass
                    
                        # 根据attack_frequency决定是否执行攻击
                        should_attack = False
                        
                        if current_epoch < attack_start_epoch:
                            should_attack = False
                        elif attack_frequency == 'always':
                            should_attack = True
                        elif attack_frequency == 'periodic':
                            period = 2
                            if not hasattr(self, '_attack_counter'):
                                self._attack_counter = 0
                            should_attack = (self._attack_counter % period == 0)
                            self._attack_counter += 1
                        elif attack_frequency == 'random':
                            import random
                            should_attack = random.random() < 0.5
                        else:
                            should_attack = True
                        
                        if should_attack:
                            # 创建默认通信计数（用于频率攻击）
                            communication_counts = {i: 10 for i in range(num_agents)}
                            
                            # 注入攻击
                            attack_result = self.attack_injector.inject_attack(
                                agent_features=agent_features.clone(),
                                communication_graph=torch.ones(num_agents, num_agents),
                                communication_counts=communication_counts  # ✨ 传递通信计数（用于频率攻击）
                            )
                            
                            # 使用被攻击后的特征
                            agent_features = attack_result['agent_features']
                            
                            # ✨ 提取被攻击后的通信计数（用于异常检测）
                            if 'communication_counts' in attack_result:
                                attacked_communication_counts = attack_result['communication_counts']
                            
                            # ✨ 如果自私攻击修改了通信图，应用到edge_index
                            if attack_type == 'selfish' and 'communication_graph' in attack_result:
                                attacked_graph = attack_result['communication_graph']
                                # 将密集图转换为稀疏edge_index格式
                                # 只保留attacked_graph中值为1的边
                                edge_list = []
                                for i in range(num_agents):
                                    for j in range(num_agents):
                                        if i != j and attacked_graph[i, j].item() > 0.5:
                                            edge_list.append([i, j])
                                if edge_list:
                                    edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
                                else:
                                    # 如果没有边，创建空edge_index
                                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                            
                            # 更新攻击统计
                            self.attack_stats['total_attacks'] += 1
                            if attack_type == 'frequency':
                                self.attack_stats['frequency_attacks'] += 1
                            elif attack_type == 'semantic':
                                self.attack_stats['semantic_attacks'] += 1
                            elif attack_type == 'selfish':
                                self.attack_stats['selfish_attacks'] += 1
                            
                            # 记录被攻击的节点
                            if hasattr(self.attack_injector, 'attacked_agents'):
                                self.attack_stats['attacked_agents'].update(self.attack_injector.attacked_agents)
                    
                    # ✨ 准备异常检测数据（用于ProtectedTGN）
                    message_counts_for_detection = None
                    embeddings_for_detection = None
                    if self.config.get('use_protection', False):
                        # 1. 更新消息计数历史（用于频率异常检测）
                        if attacked_communication_counts is not None:
                            message_counts_for_detection = attacked_communication_counts
                        else:
                            message_counts_for_detection = {i: 10 for i in range(num_agents)}
                        
                        # 更新历史记录（用于异常检测器计算Z-score）
                        for agent_id in range(num_agents):
                            if agent_id not in self.anomaly_detection_history['message_counts']:
                                self.anomaly_detection_history['message_counts'][agent_id] = []
                            self.anomaly_detection_history['message_counts'][agent_id].append(
                                message_counts_for_detection.get(agent_id, 10)
                            )
                            # 限制历史长度（只保留最近20条）
                            if len(self.anomaly_detection_history['message_counts'][agent_id]) > 20:
                                self.anomaly_detection_history['message_counts'][agent_id].pop(0)
                        
                        # 2. 准备嵌入数据（用于语义异常检测）
                        embeddings_for_detection = {}
                        for agent_id in range(num_agents):
                            if agent_id not in self.anomaly_detection_history['embeddings']:
                                self.anomaly_detection_history['embeddings'][agent_id] = []
                            # 保存当前特征（detach以避免梯度问题）
                            self.anomaly_detection_history['embeddings'][agent_id].append(
                                agent_features[agent_id].detach().cpu()
                            )
                            # 限制历史长度
                            if len(self.anomaly_detection_history['embeddings'][agent_id]) > 20:
                                self.anomaly_detection_history['embeddings'][agent_id].pop(0)
                            # 使用历史中的最后一个作为当前嵌入
                            embeddings_for_detection[agent_id] = self.anomaly_detection_history['embeddings'][agent_id][-1]
                    
                    # ✨ FSM-TGN前向传播：学习状态转移和监听关系（传入问题嵌入和通信拓扑）
                    # 注意：transition_history 不应该传递给 TGN，它只在计算损失时使用
                    # ✨ 如果使用ProtectedTGN，传递message_counts和embeddings用于异常检测
                    # 检查是否真正集成了保护机制（通过检查base_tgn是否是ProtectedTGN类型）
                    from neural_fsm_mas.defense_mechanisms import ProtectedTGN
                    protection_active = (
                        self.config.get('use_protection', False) and 
                        hasattr(current_fsm_tgn, 'base_tgn') and 
                        isinstance(current_fsm_tgn.base_tgn, ProtectedTGN)
                    )
                    
                    if protection_active:
                        # 实验2：使用 ProtectedTGN，传递异常检测参数
                        # ✨ 使用历史消息进行语义一致性检测
                        agent_messages_for_detection = dict(self.anomaly_detection_history.get('messages', {}))
                        
                        # ✨ 关键：传递上一个问题采样的通信边（用于动态中心性计算）
                        # 这样优先级不再是全0.5，而是基于实际通信模式动态计算！
                        if 'communication_edges' in self.anomaly_detection_history:
                            agent_messages_for_detection['_communication_edges'] = \
                                self.anomaly_detection_history['communication_edges']
                        
                        fsm_outputs = current_fsm_tgn(
                            agent_features=agent_features,
                            communication_topology=edge_index,
                            context_features=context_features,
                            current_state_id=current_fsm_manager.current_state_id,
                            question_embedding=question_embedding,  # ✨ 传入问题嵌入
                            message_counts=message_counts_for_detection,  # ✨ 传入消息计数（用于频率异常检测）
                            embeddings=embeddings_for_detection,  # ✨ 传入嵌入（用于语义异常检测）
                            agent_messages=agent_messages_for_detection  # ✨ 传入消息文本 + 通信边
                        )
                    else:
                        # 实验1：使用普通 FSMTemporalGraph，不传递异常检测参数
                        fsm_outputs = current_fsm_tgn(
                            agent_features=agent_features,
                            communication_topology=edge_index,
                            context_features=context_features,
                            current_state_id=current_fsm_manager.current_state_id,
                            question_embedding=question_embedding  # ✨ 传入问题嵌入
                        )
                    
                    # 开始记录本episode的LLM成本
                    episode_cost = 0.0
                    if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                        # 使用call_history长度作为episode起点索引
                        from neural_fsm_mas.losses.cost_loss import calculate_episode_cost
                        episode_start_idx = len(self.cost_tracker.call_history)
                    else:
                        episode_start_idx = None
                    
                    # 模拟执行并获取奖励
                    # ✨ 如果启用攻击，传递攻击注入器用于消息污染
                    final_answer, execution_log = await self._execute_fsm_with_learned_structure(
                        current_topology, current_fsm_manager, fsm_outputs, formatted_question,
                        fsm_tgn=current_fsm_tgn,  # 传入TGN模型
                        question_embedding=question_embedding,  # 传入问题嵌入
                        agent_features=agent_features,  # 传入智能体特征
                        context_features=context_features,  # 传入上下文特征
                        attack_injector=self.attack_injector if self.config.get('enable_attack', False) else None
                    )
                    
                    # 结束episode成本统计
                    if episode_start_idx is not None:
                        from neural_fsm_mas.losses.cost_loss import calculate_episode_cost
                        episode_cost = calculate_episode_cost(self.cost_tracker, episode_start_idx)
                        batch_episode_costs.append(episode_cost)
                    
                    # ✨ 存储收集的智能体消息和通信边（用于异常检测和动态中心性）
                    if 'collected_agent_messages' in execution_log:
                        collected_messages = execution_log['collected_agent_messages']
                        if collected_messages:
                            # 更新消息历史（用于语义一致性检测）
                            # 提取纯消息（排除特殊键）
                            pure_messages = {k: v for k, v in collected_messages.items() 
                                           if not isinstance(k, str) or not k.startswith('_')}
                            self.anomaly_detection_history['messages'] = pure_messages
                            
                            # ✨ 提取采样的通信边（用于动态图中心性计算）
                            if '_communication_edges' in collected_messages:
                                sampled_edges = collected_messages['_communication_edges']
                                # 存储到异常检测历史中
                                self.anomaly_detection_history['communication_edges'] = sampled_edges
                                
                                # ✨ 基于实际通信更新异常检测器的消息计数
                                # 每个发送消息的agent增加计数
                                if protection_active and hasattr(current_fsm_tgn, 'base_tgn'):
                                    base_tgn = current_fsm_tgn.base_tgn
                                    if hasattr(base_tgn, 'anomaly_detector'):
                                        # 统计每个agent发送的消息数
                                        agent_msg_counts = {}
                                        for src, dst in sampled_edges:
                                            # 将agent_id转换为整数索引
                                            try:
                                                src_idx = int(src.split('_')[-1]) if isinstance(src, str) and '_' in src else int(src)
                                                agent_msg_counts[src_idx] = agent_msg_counts.get(src_idx, 0) + 1
                                            except (ValueError, AttributeError):
                                                pass
                                        # 更新异常检测器的历史（这会累积！）
                                        for agent_id, count in agent_msg_counts.items():
                                            base_tgn.anomaly_detector.update_history(agent_id, message_count=count)
                    
                    # 评估准确性
                    ground_truth = self._extract_ground_truth(question_data, domain)
                    # 传递数据集特定的验证参数
                    validation_kwargs = self._get_validation_kwargs(question_data, domain)
                    is_correct = self._check_correctness(final_answer, ground_truth, domain, **validation_kwargs)
                    reward = 1.0 if is_correct else 0.0
                    
                    # ✨ 打印每个问题的最终答案及对错
                    try:
                        short_q = formatted_question.replace("\n", " ")[:200]
                    except Exception:
                        short_q = str(formatted_question)[:200]
                    short_pred = str(final_answer).replace("\n", " ")[:200]
                    short_gt = str(ground_truth).replace("\n", " ")[:200]
                    result_flag = "✅ 正确" if is_correct else "❌ 错误"
                    print("\n  ── 单题结果 ──")
                    print(f"  Q: {short_q}")
                    print(f"  Pred: {short_pred}")
                    print(f"  GT:   {short_gt}")
                    # 对 ALFWorld 额外展示数据集中要求的 subgoals，便于对照评估
                    if domain == 'alfworld':
                        subgoals = question_data.get('subgoals', [])
                        if subgoals:
                            print(f"  Subgoals (from dataset): {subgoals}")
                    print(f"  Result: {result_flag}")
                    
                    # ✨ 显示保护数据（如果启用了保护机制）
                    if self.config.get('use_protection', False) and 'protection_data' in fsm_outputs:
                        protection_data = fsm_outputs['protection_data']
                        print(f"\n  🛡️  Protection Data (每个Agent详情):")
                        
                        num_agents = 0
                        # 获取agent数量
                        if 'anomaly_scores' in protection_data:
                            num_agents = len(protection_data['anomaly_scores'])
                        
                        # 打印表头
                        print(f"    {'Agent':<8} {'异常分数':<12} {'优先级':<12} {'信任分数':<12}")
                        print(f"    {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
                        
                        # 打印每个Agent的详细数据
                        for agent_id in range(num_agents):
                            anomaly_val = protection_data.get('anomaly_scores', torch.zeros(num_agents))[agent_id].item()
                            priority_val = protection_data.get('priorities', torch.zeros(num_agents))[agent_id].item()
                            trust_val = protection_data.get('trust_scores', torch.zeros(num_agents))[agent_id].item()
                            
                            # 标记异常Agent
                            marker = "⚠️" if anomaly_val > 0.5 else "  "
                            print(f"    {marker}Agent {agent_id:<3} {anomaly_val:<12.4f} {priority_val:<12.4f} {trust_val:<12.4f}")
                        
                        # 打印汇总统计
                        print(f"\n    📊 汇总:")
                        if 'anomaly_scores' in protection_data:
                            anomaly = protection_data['anomaly_scores']
                            print(f"      异常分数: mean={anomaly.mean().item():.4f}, max={anomaly.max().item():.4f}")
                            # ✨ 收集统计数据
                            epoch_anomaly_scores.append(anomaly.mean().item())
                            # 找出高异常分数的agents
                            high_anomaly_agents = [i for i in range(len(anomaly)) if anomaly[i].item() > 0.5]
                            if high_anomaly_agents:
                                epoch_high_anomaly_agents.append(high_anomaly_agents)
                        if 'trust_scores' in protection_data:
                            trust = protection_data['trust_scores']
                            print(f"      信任分数: mean={trust.mean().item():.4f}, min={trust.min().item():.4f}")
                            epoch_trust_scores.append(trust.mean().item())
                        if 'priorities' in protection_data:
                            priorities = protection_data['priorities']
                            epoch_priorities.append(priorities.mean().item())
                        if 'message_weights' in protection_data and protection_data['message_weights'].numel() > 0:
                            msg_weights = protection_data['message_weights']
                            print(f"      消息权重: mean={msg_weights.mean().item():.4f}, "
                                  f"range=[{msg_weights.min().item():.4f}, {msg_weights.max().item():.4f}]")
                        
                        # ✨✨ 显示累积异常分数（关键：展示异常是否在累积）
                        if protection_active and hasattr(current_fsm_tgn, 'base_tgn'):
                            base_tgn = current_fsm_tgn.base_tgn
                            if hasattr(base_tgn, 'anomaly_detector'):
                                cum_scores = base_tgn.anomaly_detector.cumulative_anomaly_score
                                poll_history = base_tgn.anomaly_detector.pollution_history
                                if cum_scores:
                                    print(f"\n    📈 累积异常统计（跨问题累积）:")
                                    for agent_id in sorted(cum_scores.keys()):
                                        cum_val = cum_scores[agent_id]
                                        poll_count = len(poll_history.get(agent_id, []))
                                        if cum_val > 0.1 or poll_count > 0:
                                            print(f"      Agent {agent_id}: 累积基准={cum_val:.4f}, 污染次数={poll_count}")
                    
                    # ✨ 记录执行结果（新增 - 用于优化状态描述）
                    if self.config.get('enable_prompt_optimization', False):
                        prompt_optimizer = self.prompt_optimizers.get(domain)
                        if prompt_optimizer:
                            # ✨ 提取访问过的状态名称序列（使用List，保留重复访问）
                            visited_state_names = []
                            if 'state_transitions' in execution_log:
                                for transition in execution_log['state_transitions']:
                                    if len(transition) >= 2:
                                        from_state_id, to_state_id = transition[0], transition[1]
                                        # 添加from状态（只在第一次转移时）
                                        from_state = current_fsm_manager.states.get(from_state_id)
                                        if from_state and (not visited_state_names or visited_state_names[-1] != from_state.state_name):
                                            visited_state_names.append(from_state.state_name)
                                        # 添加to状态（每次都添加，保留重复）
                                        to_state = current_fsm_manager.states.get(to_state_id)
                                        if to_state:
                                            visited_state_names.append(to_state.state_name)
                            
                            # 检查是否达到最大转移次数
                            reached_max_transitions = execution_log.get('reached_max_transitions', False)
                            
                            # 记录执行结果（传入状态序列，保留重复访问）
                            prompt_optimizer.record_state_execution(
                                visited_states=visited_state_names,
                                is_correct=is_correct,
                                reached_max_transitions=reached_max_transitions
                            )
                    
                    # 1. 策略梯度损失
                    log_prob = execution_log.get('log_prob', 0.0)
                    policy_loss = -log_prob * reward if isinstance(log_prob, (int, float)) else -log_prob.mean() * reward
                    if not isinstance(policy_loss, torch.Tensor):
                        policy_loss = torch.tensor(float(policy_loss), device=self.device)
                    
                    # 2. 状态转移损失（学习最优转移序列）
                    if 'transition_probs' in fsm_outputs:
                        transition_probs = fsm_outputs['transition_probs']  # [num_states]
                        # 鼓励高概率转移到正确的下一状态
                        transition_targets = self._get_transition_targets(execution_log, current_fsm_manager)
                        if transition_targets is not None and len(transition_targets) > 0:
                            # transition_probs是当前状态的转移概率分布
                            # 对于每个实际转移，计算交叉熵损失
                            # 需要为每个转移创建一个概率分布
                            num_states = transition_probs.size(0)
                            losses = []
                            for target_state in transition_targets:
                                if 0 <= target_state < num_states:
                                    # 使用负对数似然：-log(P(target_state))
                                    loss = -torch.log(transition_probs[target_state] + 1e-8)
                                    losses.append(loss)
                            if losses:
                                transition_loss = torch.stack(losses).mean()
                            else:
                                transition_loss = torch.tensor(0.0, device=self.device)
                        else:
                            transition_loss = torch.tensor(0.0, device=self.device)
                    else:
                        transition_loss = torch.tensor(0.0, device=self.device)
                    
                    # 3. 监听路径损失（学习最优通信路径）
                    if 'listener_weights' in fsm_outputs:
                        listener_weights = fsm_outputs['listener_weights']  # [num_states, num_agents]
                        # 鼓励向实际监听的智能体发送消息
                        listener_targets = self._get_listener_targets(current_fsm_manager)
                        if listener_targets is not None:
                            # listener_weights已经是概率分布（经过softmax），使用BCE损失
                            listener_loss = nn.functional.binary_cross_entropy(
                                listener_weights,
                                listener_targets,
                                reduction='mean'
                            )
                        else:
                            listener_loss = torch.tensor(0.0, device=self.device)
                    else:
                        listener_loss = torch.tensor(0.0, device=self.device)
                    
                    # 4. ✨ 最大转移次数惩罚（鼓励更高效的状态转移路径）
                    reached_max_transitions = execution_log.get('reached_max_transitions', False)
                    if reached_max_transitions:
                        # 如果达到最大转移次数，给予惩罚（鼓励更短的路径）
                        max_transitions_penalty = torch.tensor(1.0, device=self.device, requires_grad=True)
                    else:
                        max_transitions_penalty = torch.tensor(0.0, device=self.device)
                    
                    # 5. ✨ 保护损失（实验2核心：防御异常攻击）
                    protection_loss = torch.tensor(0.0, device=self.device)
                    # ✅ 检查是否使用 ProtectedTGN（通过检查 base_tgn）
                    has_protection = (
                        self.config.get('use_protection', False) and 
                        hasattr(current_fsm_tgn, 'base_tgn') and 
                        current_fsm_tgn.base_tgn is not None and
                        hasattr(current_fsm_tgn.base_tgn, 'protection_loss')
                    )
                    if has_protection:
                        # ✅ 只有实验2（use_protection=True）且使用ProtectedTGN时才计算
                        # 实验1（use_protection=False）不会执行此分支
                        try:
                            # 从 fsm_outputs 中提取保护相关数据
                            protection_data = fsm_outputs.get('protection_data', {})
                            
                            if protection_data:
                                anomaly_scores = protection_data['anomaly_scores']
                                priorities = protection_data['priorities']
                                
                                # 计算消息向量（从通信拓扑和智能体特征）
                                if edge_index.size(1) > 0:
                                    source_features = agent_features[edge_index[0]]  # [num_edges, feature_dim]
                                    target_features = agent_features[edge_index[1]]  # [num_edges, feature_dim]
                                    messages = torch.cat([source_features, target_features], dim=-1)  # [num_edges, 2*feature_dim]
                                else:
                                    # 没有边时，创建空消息
                                    messages = torch.zeros((0, agent_features.size(1) * 2), device=self.device)
                                
                                # ✅ 通过 base_tgn 访问 ProtectedTGN 的 protection_loss
                                protection_loss = current_fsm_tgn.base_tgn.protection_loss.compute_protection_loss(
                                    messages=messages,
                                    anomaly_scores=anomaly_scores,
                                    priorities=priorities,
                                    edge_index=edge_index
                                )
                                
                                # 打印保护损失（用于调试）
                                if protection_loss.item() > 0:
                                    print(f"    🛡️  保护损失: {protection_loss.item():.6f}")
                        except Exception as e:
                            print(f"    ⚠️  计算保护损失时出错: {e}")
                            import traceback
                            traceback.print_exc()
                            protection_loss = torch.tensor(0.0, device=self.device)
                    
                    # 累加批次损失
                    batch_policy_loss = batch_policy_loss + policy_loss
                    batch_transition_loss += transition_loss
                    batch_listener_loss += listener_loss
                    batch_max_transitions_penalty += max_transitions_penalty
                    batch_protection_loss += protection_loss  # ✨ 新增
                    
                    if is_correct:
                        batch_correct += 1
                    # ✨ 注意：不在循环中累加total_questions，避免重复计算
                    # total_questions 将在batch结束后统一累加 len(batch)
                    
                except Exception as e:
                    print(f"  ⚠️  训练错误: {e}")
                    continue
            
            # ==================== 组合损失并反向传播 ====================
            # 实验1: L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + ζ·P_max_trans (五目标)
            # 实验2: L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + ε·L_protection + ζ·P_max_trans (六目标)
            # ===============================================================
            batch_cost_loss = 0.0  # ✨ 新增：批次成本损失
            if batch_correct > 0 or len(batch) > 0:
                if hasattr(self, 'compute_four_objective_loss') and batch_episode_costs:
                    # 使用集成的四目标损失函数（包含成本正则 δ·L_cost）
                    total_loss, loss_detail = self.compute_four_objective_loss(
                        batch_policy_loss,
                        batch_transition_loss,
                        batch_listener_loss,
                        batch_episode_costs
                    )
                    # ✨ 从 loss_detail 中提取成本损失
                    batch_cost_loss = loss_detail.get('cost', 0.0)
                    # ✨ 添加保护损失（实验2核心）和最大转移惩罚项
                    total_loss = total_loss + epsilon * batch_protection_loss + zeta * batch_max_transitions_penalty
                else:
                    # 回退到显式损失计算（无成本损失）
                    batch_cost_loss = 0.0
                    total_loss = (alpha * batch_policy_loss + 
                            beta * batch_transition_loss + 
                            gamma * batch_listener_loss +
                            epsilon * batch_protection_loss +  # ✨ 保护损失（仅实验2，实验1时epsilon=0）
                            zeta * batch_max_transitions_penalty)  # 惩罚项
                
                optimizer.zero_grad()
                if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                    total_loss.backward()
                    # 对所有FSM-TGN应用梯度裁剪（包括类别级FSM）
                    torch.nn.utils.clip_grad_norm_(fsm_tgn.parameters(), max_norm=1.0)
                    if hasattr(self, 'mmlu_category_fsms'):
                        for _, _, cat_fsm_tgn in self.mmlu_category_fsms.values():
                            if cat_fsm_tgn is not fsm_tgn:
                                torch.nn.utils.clip_grad_norm_(cat_fsm_tgn.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_policy_loss += batch_policy_loss.item() if isinstance(batch_policy_loss, torch.Tensor) else batch_policy_loss
                total_transition_loss += batch_transition_loss.item() if isinstance(batch_transition_loss, torch.Tensor) else batch_transition_loss
                total_listener_loss += batch_listener_loss.item() if isinstance(batch_listener_loss, torch.Tensor) else batch_listener_loss
                total_cost_loss += batch_cost_loss  # ✨ 新增：累积成本损失
                total_protection_loss += batch_protection_loss.item() if isinstance(batch_protection_loss, torch.Tensor) else 0.0
                total_max_transitions_penalty += batch_max_transitions_penalty.item() if isinstance(batch_max_transitions_penalty, torch.Tensor) else batch_max_transitions_penalty
                total_correct += batch_correct
                total_questions += len(batch)
            
            # 显示每个batch的统计信息
            batch_accuracy = batch_correct / len(batch) if len(batch) > 0 else 0.0
            print(f"    Batch {batch_idx + 1}/{len(training_batches)}, "
                  f"Acc: {batch_correct}/{len(batch)} ({batch_accuracy:.4f})")
            batch_accuracies.append(batch_accuracy)
        
        # 计算平均指标
        num_batches = len(training_batches) if len(training_batches) > 0 else 1
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_transition_loss = total_transition_loss / num_batches if num_batches > 0 else 0.0
        avg_listener_loss = total_listener_loss / num_batches if num_batches > 0 else 0.0
        avg_cost_loss = total_cost_loss / num_batches if num_batches > 0 else 0.0  # ✨ 新增：成本损失
        avg_protection_loss = total_protection_loss / num_batches if num_batches > 0 else 0.0
        avg_max_transitions_penalty = total_max_transitions_penalty / num_batches if num_batches > 0 else 0.0
        # 计算总损失（4个损失函数 + 保护损失(仅实验2) + 1个惩罚项）
        avg_total_loss = (alpha * avg_policy_loss + 
                         beta * avg_transition_loss + 
                         gamma * avg_listener_loss +
                         delta * avg_cost_loss +  # ✨ 成本损失
                         epsilon * avg_protection_loss +  # ✨ 保护损失（仅实验2有效）
                         zeta * avg_max_transitions_penalty)  # 惩罚项
        avg_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # ✨ 计算保护数据的epoch级统计
        protection_statistics = {}
        if self.config.get('use_protection', False):
            protection_statistics = {
                'avg_anomaly_score': sum(epoch_anomaly_scores) / len(epoch_anomaly_scores) if epoch_anomaly_scores else 0.0,
                'avg_trust_score': sum(epoch_trust_scores) / len(epoch_trust_scores) if epoch_trust_scores else 0.0,
                'avg_priority': sum(epoch_priorities) / len(epoch_priorities) if epoch_priorities else 0.0,
                'high_anomaly_agents': epoch_high_anomaly_agents,
                'num_high_anomaly_questions': len(epoch_high_anomaly_agents)
            }
        
        return {
            'total_loss': avg_total_loss,
            'policy_loss': avg_policy_loss,
            'transition_loss': avg_transition_loss,
            'listener_loss': avg_listener_loss,
            'cost_loss': avg_cost_loss,  # ✨ 新增：LLM成本损失
            'protection_loss': avg_protection_loss,
            'max_transitions_penalty': avg_max_transitions_penalty,
            'accuracy': avg_accuracy,
            # ✨ 记录该epoch内每个batch的准确率，便于排查问题
            'batch_accuracies': batch_accuracies,
            # ✨ 记录保护数据统计（用于写入结果文件）
            'protection_statistics': protection_statistics,
        }
    
    async def _validate_epoch(self,
                             domain: str,
                             topology: MultiAgentTopologyManager,
                             fsm_manager: FSMStateManager,
                             validation_batches: List[Dict],
                             is_test: bool = False) -> Tuple[float, float]:
        """
        验证一个epoch
        
        Args:
            domain: 领域名称
            topology: 多智能体拓扑管理器
            fsm_manager: FSM状态管理器
            validation_batches: 验证批次列表
            is_test: 是否为测试集（用于区分验证和测试）
        
        Returns:
            (accuracy, cost): 准确率和成本
        """
        
        total_correct = 0
        total_questions = 0
        total_cost = 0.0
        
        # 使用全部验证批次和全部样本
        for batch in validation_batches:
            for question_data in batch:
                try:
                    formatted_question = self.data_processor.format_question_for_agents(question_data, domain)
                    
                    # 记录问题开始前的成本
                    question_start_cost = 0.0
                    if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                        question_start_cost = self.cost_tracker.get_total_cost()
                    
                    # 执行FSM推理
                    fsm_manager.reset()
                    
                    # 为了支持状态转移，需要先计算 fsm_outputs
                    # 获取智能体特征和上下文特征
                    agent_features = self._get_agent_features(topology)
                    context_features = self._get_context_features(domain)
                    
                    # ✨ 成本优化：生成问题嵌入时简化FSM状态和智能体描述，避免重复的长文本
                    # 只保留状态ID和名称，不包含完整描述（描述在智能体的system prompt中已有）
                    state_summaries = []
                    for s in fsm_manager.states.values():
                        # 只保留状态ID和名称，不包含完整描述以节省token
                        state_summaries.append(f"State {s.state_id}: {s.state_name}")
                    state_block = "\n".join(state_summaries)
                    
                    # 只保留智能体ID和角色名称
                    agent_summaries = []
                    for agent_id, agent_node in topology.agent_execution_nodes.items():
                        agent_summaries.append(f"{agent_id}: {agent_node.agent_role}")
                    agent_block = "\n".join(agent_summaries)
                    
                    # ✨ 简化enriched_text，减少不必要的描述文本
                    enriched_text = (
                        f"{formatted_question}\n\n"
                        f"[FSM States: {len(state_summaries)} states]\n{state_block}\n\n"
                        f"[Agents: {len(agent_summaries)} agents]\n{agent_block}"
                    )
                    
                    question_emb = self.text_embedding_model.encode_query(enriched_text)
                    question_emb = question_emb.to(self.device)
                    
                    # ✨ 构建通信拓扑（与训练阶段一致）
                    num_agents = agent_features.size(0)
                    if num_agents > 1:
                        edge_indices = [
                            (i, j)
                            for i in range(num_agents)
                            for j in range(num_agents)
                            if i != j
                        ]
                        if edge_indices:
                            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
                        else:
                            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    else:
                        edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    
                    # ✨ 攻击注入（验证/测试阶段）- 评估防御效果
                    attacked_communication_counts = None
                    if self.attack_injector is not None and self.config.get('enable_attack', False):
                        # ✅ 验证/测试阶段也应该有攻击注入，以评估保护机制的防御效果
                        # 创建默认通信计数（用于频率攻击）
                        communication_counts = {i: 10 for i in range(num_agents)}
                        
                        # 注入攻击
                        attack_result = self.attack_injector.inject_attack(
                            agent_features=agent_features.clone(),
                            communication_graph=torch.ones(num_agents, num_agents),
                            communication_counts=communication_counts
                        )
                        
                        # 使用被攻击后的特征
                        agent_features = attack_result['agent_features']
                        
                        # 提取被攻击后的通信计数
                        if 'communication_counts' in attack_result:
                            attacked_communication_counts = attack_result['communication_counts']
                        
                        # 如果是自私攻击，应用修改后的通信图
                        attack_config = self.config.get('attack', {})
                        attack_type = attack_config.get('attack_type', 'selfish')
                        if attack_type == 'selfish' and 'communication_graph' in attack_result:
                            attacked_graph = attack_result['communication_graph']
                            # 将密集图转换为稀疏edge_index格式
                            edge_list = []
                            for i in range(num_agents):
                                for j in range(num_agents):
                                    if i != j and attacked_graph[i, j].item() > 0.5:
                                        edge_list.append([i, j])
                            if edge_list:
                                edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
                            else:
                                edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    
                    # ✨ 准备异常检测数据（验证/测试阶段也需要，以确保保护机制正常工作）
                    message_counts_for_detection = None
                    embeddings_for_detection = None
                    if self.config.get('use_protection', False):
                        # 使用攻击后的消息计数（如果有攻击）或默认值
                        if attacked_communication_counts is not None:
                            message_counts_for_detection = attacked_communication_counts
                        else:
                            message_counts_for_detection = {i: 10 for i in range(num_agents)}
                        
                        # 准备嵌入数据（使用当前的智能体特征）
                        embeddings_for_detection = {}
                        for agent_id in range(num_agents):
                            # 使用当前特征作为嵌入（detach以避免梯度问题）
                            embeddings_for_detection[agent_id] = agent_features[agent_id].detach().cpu()
                    
                    # 获取 FSM-TGN（如果可用）
                    fsm_tgn = self.domain_fsm_tgns.get(domain)
                    if fsm_tgn is not None:
                        # 检查是否真正集成了保护机制
                        from neural_fsm_mas.defense_mechanisms import ProtectedTGN
                        protection_active = (
                            self.config.get('use_protection', False) and 
                            hasattr(fsm_tgn, 'base_tgn') and 
                            isinstance(fsm_tgn.base_tgn, ProtectedTGN)
                        )
                        
                        # 计算 fsm_outputs
                        if protection_active:
                            # 实验2：使用 ProtectedTGN，传递异常检测参数
                            fsm_outputs = fsm_tgn(
                                agent_features=agent_features,
                                communication_topology=edge_index,
                                context_features=context_features,
                                current_state_id=fsm_manager.current_state_id,
                                question_embedding=question_emb,
                                message_counts=message_counts_for_detection,  # ✨ 传入消息计数（用于频率异常检测）
                                embeddings=embeddings_for_detection  # ✨ 传入嵌入（用于语义异常检测）
                            )
                        else:
                            # 实验1：使用普通 FSMTemporalGraph，不传递异常检测参数
                            fsm_outputs = fsm_tgn(
                                agent_features=agent_features,
                                communication_topology=edge_index,
                                context_features=context_features,
                                current_state_id=fsm_manager.current_state_id,
                                question_embedding=question_emb
                            )
                    else:
                        fsm_outputs = None
                    
                    # ✨ 验证/测试阶段也使用纯TGN采样方法（与训练阶段一致）
                    final_answer, _, _ = await topology.execute_fsm_reasoning_with_sampling(
                        task_input=formatted_question,
                        fsm_outputs=fsm_outputs,
                        fsm_manager=fsm_manager,
                        fsm_tgn=fsm_tgn,  # 传入TGN模型，用于动态重新计算
                        question_embedding=question_emb,  # 传入问题嵌入
                        agent_features=agent_features,  # 传入智能体特征
                        context_features=context_features,  # 传入上下文特征
                        max_transitions=self.config.get('max_transitions', 8),
                        verbose=False,  # 验证/测试阶段减少详细日志
                        phase="test" if is_test else "val"  # 区分验证和测试阶段
                    )
                    
                    # 计算该问题的成本
                    question_cost = 0.0
                    if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                        question_end_cost = self.cost_tracker.get_total_cost()
                        question_cost = question_end_cost - question_start_cost
                    total_cost += question_cost
                    
                    # 评估
                    ground_truth = self._extract_ground_truth(question_data, domain)
                    validation_kwargs = self._get_validation_kwargs(question_data, domain)
                    is_correct = self._check_correctness(final_answer, ground_truth, domain, **validation_kwargs)

                    # ✨ 调试信息：检查验证阶段的答案
                    if is_test:
                        debug_prefix = "🧪 测试"
                    else:
                        debug_prefix = "✅ 验证"
                    print(f"{debug_prefix} Q{total_questions+1}: 预测='{final_answer}' vs 正确='{ground_truth}' -> {'✓' if is_correct else '✗'}")
                    
                    if is_correct:
                        total_correct += 1
                    total_questions += 1
                    
                except Exception as e:
                    continue
        
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        return accuracy, total_cost
    
    # ========== 辅助方法 ==========
    
    def _get_agent_features(self, topology: MultiAgentTopologyManager) -> torch.Tensor:
        """获取智能体特征"""
        if hasattr(topology, '_agent_embeddings') and topology._agent_embeddings is not None:
            return topology._agent_embeddings.to(self.device)
        else:
            num_agents = len(topology.agent_execution_nodes)
            return torch.randn(num_agents, 256, device=self.device)
    
    def _get_context_features(self, domain: str) -> torch.Tensor:
        """获取上下文特征"""
        # 为每个数据集分配一个独立的domain_id，用作简单的“领域 one-hot”特征
        domain_ids = {
            'mmlu': 0,
            'gsm8k': 1,
            'humaneval': 2,
            'mbpp': 3,
            'hotpotqa': 4,
            'alfworld': 5,
            'math': 6,
            'gaia': 7,
        }
        domain_id = domain_ids.get(domain, 0)
        context = torch.zeros(256, device=self.device)
        context[domain_id] = 1.0
        return context
    
    async def _execute_fsm_with_learned_structure(self,
                                                  topology: MultiAgentTopologyManager,
                                                  fsm_manager: FSMStateManager,
                                                  fsm_outputs: Dict,
                                                  question: str,
                                                  fsm_tgn: Optional[FSMTemporalGraph] = None,
                                                  question_embedding: Optional[torch.Tensor] = None,
                                                  agent_features: Optional[torch.Tensor] = None,
                                                  context_features: Optional[torch.Tensor] = None,
                                                  attack_injector: Optional[Any] = None) -> Tuple[str, Dict]:
        """
        使用学习到的FSM结构执行推理（任务自适应）
        
        核心逻辑：
        1. 从TGN输出的transition_probs中采样下一状态
        2. 从TGN输出的listener_weights中采样通信路径
        3. 从总FSM中匹配选中的状态、转移、智能体
        
        Args:
            topology: 多智能体拓扑管理器
            fsm_manager: FSM状态管理器
            fsm_outputs: TGN初始输出（包含transition_matrix和listener_weights）
            question: 问题文本
            fsm_tgn: TGN模型（用于动态重新计算转移概率）
            question_embedding: 问题嵌入（用于动态重新计算）
            agent_features: 智能体特征（用于动态重新计算）
            context_features: 上下文特征（用于动态重新计算）
            attack_injector: ✨ 攻击注入器（用于消息污染攻击）
        """
        try:
            # ✨ 使用概率采样执行FSM推理
            final_answer, log_probs, reached_max_transitions, collected_agent_messages = await topology.execute_fsm_reasoning_with_sampling(
                task_input=question,
                fsm_outputs=fsm_outputs,  # 传入TGN输出
                fsm_manager=fsm_manager,
                fsm_tgn=fsm_tgn,  # 传入TGN模型，用于动态重新计算
                question_embedding=question_embedding,  # 传入问题嵌入
                agent_features=agent_features,  # 传入智能体特征
                context_features=context_features,  # 传入上下文特征
                enable_transition_prediction=self.config.get('enable_transition_prediction', True),
                enable_comm_sampling=self.config.get('enable_comm_sampling', True),
                max_transitions=self.config.get('max_transitions', 8),
                phase="train",  # 训练阶段
                attack_injector=attack_injector  # ✨ 传入攻击注入器
            )
            return final_answer, {
                'log_prob': log_probs, 
                'reached_max_transitions': reached_max_transitions,
                'collected_agent_messages': collected_agent_messages  # ✨ 新增：收集的智能体消息
            }
        except Exception as e:
            print(f"⚠️  执行FSM推理错误: {e}")
            import traceback
            traceback.print_exc()
            return "", {'log_prob': 0.0, 'reached_max_transitions': False}
    
    def _get_transition_targets(self, execution_log: Dict, fsm_manager: FSMStateManager) -> Optional[torch.Tensor]:
        """
        ✨ 获取状态转移目标（基于实际采样的状态转移）
        
        从transition_history中获取实际执行的状态转移序列，
        用于计算transition_loss，鼓励TGN预测高概率给实际转移到的状态。
        """
        if fsm_manager.transition_history:
            num_states = len(fsm_manager.states)
            targets = []
            # 使用所有转移历史（不仅仅是最近5次）
            for from_state, to_state in fsm_manager.transition_history:
                # 确保目标状态ID在有效范围内
                if 0 <= to_state < num_states:
                    targets.append(to_state)
            if targets:
                return torch.tensor(targets, dtype=torch.long, device=self.device)
        return None
    
    def _get_listener_targets(self, fsm_manager: FSMStateManager) -> Optional[torch.Tensor]:
        """
        ✨ 获取监听目标（基于实际采样的监听关系）
        
        从fsm_manager.listeners中获取实际执行的监听关系，
        用于计算listener_loss，鼓励TGN预测高权重给实际监听的智能体。
        """
        num_states = len(fsm_manager.states)
        num_agents = fsm_manager.num_agents
        targets = torch.zeros(num_states, num_agents, device=self.device)
        
        for state_id, listeners in fsm_manager.listeners.items():
            for listener_id in listeners:
                if listener_id in fsm_manager.agent_ids:
                    agent_idx = fsm_manager.agent_ids.index(listener_id)
                    targets[state_id, agent_idx] = 1.0
        
        return targets if targets.sum() > 0 else None
    
    def _extract_ground_truth(self, question_data: Dict, domain: str) -> str:
        """提取正确答案"""
        if domain == 'mmlu':
            return question_data.get('answer', '')
        elif domain == 'gpqa':
            return question_data.get('answer', '')
        elif domain == 'gaia':
            return str(question_data.get('answer', ''))
        elif domain == 'gsm8k':
            return str(question_data.get('answer', ''))
        elif domain == 'humaneval':
            return question_data.get('canonical_solution', '') or question_data.get('answer', '')
        elif domain == 'hotpotqa':
            return question_data.get('answer', '')
        elif domain == 'alfworld':
            return question_data.get('goal', '')
        elif domain == 'math':
            return question_data.get('answer', '')
        elif domain == 'mbpp':
            # MBPP 的“正确答案”主要用于日志展示，真实判定由 test_list 执行断言
            return question_data.get('code', '')
        return ''
    
    def _get_validation_kwargs(self, question_data: Dict, domain: str) -> Dict[str, Any]:
        """
        获取数据集特定的验证参数
        
        Args:
            question_data: 问题数据
            domain: 数据集名称
        
        Returns:
            验证参数字典
        """
        kwargs = {}
        
        if domain == 'humaneval':
            kwargs['test_code'] = question_data.get('test_code', '')
            kwargs['entry_point'] = question_data.get('entry_point', '') or question_data.get('answer', '')
        elif domain == 'mbpp':
            # 传递 MBPP 的测试用例列表，用于代码执行验证
            kwargs['test_list'] = question_data.get('test_list', [])
        elif domain == 'alfworld':
            kwargs['subgoals'] = question_data.get('subgoals', [])
        
        return kwargs
    
    def _get_or_create_category_fsm(self, domain: str, category: str) -> Tuple[MultiAgentTopologyManager, FSMStateManager, FSMTemporalGraph]:
        """
        获取或创建类别级FSM（MMLU专用）
        
        Args:
            domain: 数据集名称
            category: 类别名称
        
        Returns:
            (topology, fsm_manager, fsm_tgn)
        """
        # 检查缓存
        if hasattr(self, 'mmlu_category_fsms') and category in self.mmlu_category_fsms:
            return self.mmlu_category_fsms[category]
        
        # 尝试从缓存加载
        if hasattr(self, 'fsm_cache_manager'):
            cached = self._load_category_fsm_if_available(domain, category)
            if cached:
                topology, fsm_manager, fsm_tgn = self._create_fsm_from_cache(cached, domain)
                if not hasattr(self, 'mmlu_category_fsms'):
                    self.mmlu_category_fsms = {}
                self.mmlu_category_fsms[category] = (topology, fsm_manager, fsm_tgn)
                return topology, fsm_manager, fsm_tgn
        
        # ✨ 如果缓存不存在，尝试自动生成（如果启用）
        if self.config.get('generate_fsm_if_missing', False):
            print(f"  🔨 类别 {category} 的FSM不存在，尝试自动生成...")
            try:
                from baseclass.Enhanced_FSM_Gen import EnhancedFSMGenerator
                # 使用注入的fsm_generator，如果不存在则创建
                if hasattr(self, 'fsm_generator') and self.fsm_generator is not None:
                    fsm_generator = self.fsm_generator
                else:
                    fsm_generator = EnhancedFSMGenerator(use_azure=False)
                
                mas_config, cost = fsm_generator.generate_complete_mas(
                    dataset='mmlu',
                    task_description=f"MMLU {category} category questions",
                    save_path=None
                )
                
                # 保存到缓存
                if hasattr(self, 'fsm_cache_manager') and self.fsm_cache_manager:
                    self.fsm_cache_manager.save_fsm(
                        dataset='mmlu',
                        category=category,
                        fsm_config=mas_config.get('fsm', {}),
                        agents=mas_config.get('agents', []),
                        metadata={'generation_cost': cost, 'category': category, 'auto_generated': True}
                    )
                else:
                    # 如果没有缓存管理器，创建临时缓存管理器
                    from neural_fsm_mas.fsm_cache_manager import create_cache_manager
                    cache_dir = self.config.get('fsm_cache_dir', './fsm_cache')
                    temp_cache_manager = create_cache_manager(cache_dir)
                    temp_cache_manager.save_fsm(
                        dataset='mmlu',
                        category=category,
                        fsm_config=mas_config.get('fsm', {}),
                        agents=mas_config.get('agents', []),
                        metadata={'generation_cost': cost, 'category': category, 'auto_generated': True}
                    )
                
                # 从生成的配置创建FSM（转换为缓存格式后使用_create_fsm_from_cache）
                cached_format = {
                    'fsm_config': mas_config.get('fsm', {}),
                    'agents': mas_config.get('agents', []),
                    'metadata': {'generation_cost': cost, 'category': category, 'auto_generated': True}
                }
                topology, fsm_manager, fsm_tgn = self._create_fsm_from_cache(cached_format, domain)
                
                if not hasattr(self, 'mmlu_category_fsms'):
                    self.mmlu_category_fsms = {}
                self.mmlu_category_fsms[category] = (topology, fsm_manager, fsm_tgn)
                print(f"  ✅ 类别 {category} 的FSM生成完成 (成本: ${cost:.4f})")
                return topology, fsm_manager, fsm_tgn
            except Exception as e:
                print(f"  ⚠️  自动生成失败: {e}")
                print(f"  ⚠️  将使用默认FSM")
        
        # 如果自动生成失败或未启用，使用默认FSM
        print(f"  ⚠️  类别 {category} 的FSM不存在，使用默认FSM")
        if not hasattr(self, 'mmlu_category_fsms'):
            self.mmlu_category_fsms = {}
        
        # 使用默认的MMLU FSM
        if domain not in self.domain_topologies:
            topology, fsm_manager, fsm_tgn = self.create_domain_fsm_system(domain)
            self.domain_topologies[domain] = topology
            self.domain_fsm_managers[domain] = fsm_manager
            self.domain_fsm_tgns[domain] = fsm_tgn
        else:
            topology = self.domain_topologies[domain]
            fsm_manager = self.domain_fsm_managers[domain]
            fsm_tgn = self.domain_fsm_tgns[domain]
        
        self.mmlu_category_fsms[category] = (topology, fsm_manager, fsm_tgn)
        return topology, fsm_manager, fsm_tgn
    
    def _check_correctness(self, prediction: str, ground_truth: str, domain: str, **kwargs) -> bool:
        """
        检查答案正确性
        
        使用专门的答案验证器，针对不同数据集使用不同的验证方法
        """
        from neural_fsm_mas.training_data.answer_validator import create_answer_validator
        
        validator = create_answer_validator()
        is_correct, details = validator.validate(
            prediction=prediction,
            ground_truth=ground_truth,
            domain=domain,
            **kwargs
        )
        # 对代码类数据集，如果验证失败，输出部分错误信息便于调试
        if not is_correct and domain in ("humaneval", "mbpp"):
            error_msg = details.get("error")
            if error_msg:
                print(f"  ⚠️ 验证阶段错误信息（{domain}）: {str(error_msg)[:300]}")
        return is_correct
    
    def _save_best_model(self, domain: str, fsm_tgn: FSMTemporalGraph, epoch: int, accuracy: float):
        """保存最佳模型"""
        model_dir = self.output_dir / "best_models" / domain
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"fsm_tgn_epoch_{epoch}_acc_{accuracy:.4f}.pt"
        torch.save({
            'epoch': epoch,
            'accuracy': accuracy,
            'model_state_dict': fsm_tgn.state_dict(),
            'config': self.config
        }, model_path)
        
        print(f"  💾 保存最佳模型到 {model_path}")
    
    async def train_all_domains(self, domains: List[str] = None) -> Dict[str, Dict]:
        """训练所有领域"""
        if domains is None:
            domains = ['mmlu', 'gsm8k', 'humaneval']
        
        all_results = {}
        for domain in domains:
            results = await self.train_domain(domain, num_epochs=self.config.get('num_epochs', 10))
            all_results[domain] = results
        
        # 保存总结
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        return all_results
    
    def compute_four_objective_loss(self,
                                    policy_loss: torch.Tensor,
                                    transition_loss: torch.Tensor,
                                    listener_loss: torch.Tensor,
                                    episode_costs: List[float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算四目标组合损失（成本损失核心方法）
        
        L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost
        
        Args:
            policy_loss: 策略梯度损失（准确率）
            transition_loss: 状态转移损失
            listener_loss: 监听路径损失
            episode_costs: episode成本列表（美元）
        
        Returns:
            (total_loss, loss_dict)
        """
        # 将成本列表转换为张量
        cost_tensor = torch.tensor(episode_costs, dtype=torch.float32, device=self.device)
        
        # 使用预初始化的四目标损失函数计算
        total_loss, loss_dict = self.cost_loss_fn(
            policy_loss,
            transition_loss,
            listener_loss,
            cost_tensor
        )
        
        return total_loss, loss_dict


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FSM MAS Training V2")
    
    parser.add_argument("--domains", type=str, nargs='+', default=['gsm8k'],
                       help="训练领域")
    parser.add_argument("--dataset_root", type=str, default="./datasets",
                       help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="./fsm_mas_v2_outputs",
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="学习率")
    
    # 损失权重
    parser.add_argument("--policy_gradient_weight", type=float, default=1.0,
                       help="策略梯度损失权重 (α)")
    parser.add_argument("--transition_loss_weight", type=float, default=0.3,
                       help="状态转移损失权重 (β)")
    parser.add_argument("--listener_loss_weight", type=float, default=0.2,
                       help="监听路径损失权重 (γ)")
    
    # 模型参数
    parser.add_argument("--memory_dim", type=int, default=128,
                       help="TGN记忆维度")
    parser.add_argument("--time_dim", type=int, default=32,
                       help="时间编码维度")
    parser.add_argument("--llm_name", type=str, default="gpt-5-nano",
                       help="LLM模型名称")
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'policy_gradient_weight': args.policy_gradient_weight,
        'transition_loss_weight': args.transition_loss_weight,
        'listener_loss_weight': args.listener_loss_weight,
        'memory_dim': args.memory_dim,
        'time_dim': args.time_dim,
        'llm_name': args.llm_name,
        'agent_embedding_dim': 256,
        'state_feature_dim': 256,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42
    }
    
    print("="*80)
    print("🚀 FSM多智能体系统训练 V2")
    print("="*80)
    print("\n✅ 新FSM架构特性:")
    print("  1. 一个状态对应一个智能体")
    print("  2. 同时优化状态转移概率")
    print("  3. 同时优化监听通信路径")
    print(f"\n📊 配置: {json.dumps(config, indent=2)}")
    print("="*80 + "\n")
    
    trainer = FSMMultiAgentSystemTrainerV2(
        config=config,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir
    )
    
    try:
        results = await trainer.train_all_domains(args.domains)
        
        print("\n" + "="*80)
        print("🎉 训练完成！")
        print("="*80)
        
        for domain, result in results.items():
            print(f"\n📊 {domain}:")
            print(f"  最佳准确率: {result['best_accuracy']:.4f} (Epoch {result['best_epoch']})")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


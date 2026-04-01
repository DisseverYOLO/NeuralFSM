"""
FSM Multi-Agent System Training Script V2

Fully adapted to the new FSM architecture:
1. One state corresponds to one agent (State -> Agent mapping)
2. Simultaneously optimize state-transition probabilities
3. Simultaneously optimize listener communication paths
4. Use FSMStateManager to manage states
5. Integrate FSMTemporalGraph to learn optimal topologies

Loss design for three-objective optimization:
- Combined loss = alpha * policy-gradient loss + beta * state-transition loss + gamma * listener-path loss
- alpha (policy-gradient weight): 1.0, directly optimizes task accuracy
- beta (state-transition weight): 0.3, learns optimal state-transition sequences
- gamma (listener-path weight): 0.2, learns optimal communication paths
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

# Add paths
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
    Trainer for the FSM multi-agent system, version 2.

    Core capabilities:
    1. One state corresponds to one agent.
    2. Use FSMStateManager to manage state transitions.
    3. Use FSMTemporalGraph to jointly learn:
       - State transition probabilities
       - Listener communication paths
    4. Optimize a three-objective combined loss.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 dataset_root: str = "./datasets",
                 output_dir: str = "./fsm_mas_v2_outputs"):
        
        self.config = config
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the unified data processor
        self.data_processor = UnifiedDataProcessor(str(self.dataset_root))
        
        # Initialize the text embedding model for question embeddings
        from neural_fsm_mas.embeddings.text_embedding import get_embedding_model
        self.text_embedding_model = get_embedding_model()
        print(f"📦 Text embedding model loaded, embedding dimension: {self.text_embedding_model.embedding_dim}")
        
        # Training state
        self.training_history = []
        self.domain_topologies: Dict[str, MultiAgentTopologyManager] = {}
        self.domain_fsm_managers: Dict[str, FSMStateManager] = {}
        self.domain_fsm_tgns: Dict[str, FSMTemporalGraph] = {}
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 Using device: {self.device}")
        
        # Initialize the cost tracker
        from neural_fsm_mas.utils.llm_cost_tracker import create_cost_tracker, set_global_cost_tracker
        model_name = self.config.get('llm_name', 'gpt-5-nano')
        self.cost_tracker = create_cost_tracker(model_name)
        # Set a global cost tracker so the LLM client can log costs automatically
        set_global_cost_tracker(self.cost_tracker)
        print(f"💰 Cost tracker initialized (model: {model_name})")
        
        # Initialize the four-objective loss, centered on cost regularization
        # Baseline cost notes:
        #   - gpt-5-nano costs about $0.0016 per question in practice
        #   - baseline_cost = $0.005 (about 3x the actual cost), so that:
        #     - Cost below the baseline leads to a smaller negative penalty
        #     - Cost above the baseline increases the positive penalty
        #   - cost_scale = 100 keeps the cost-loss magnitude close to other losses
        from neural_fsm_mas.losses.cost_loss import CostRegularizedLoss
        baseline_cost = self.config.get('baseline_cost', 0.005)  # $0.005 per question
        cost_scale = self.config.get('cost_scale', 100)  # Scaling factor
        self.cost_loss_fn = CostRegularizedLoss(
            alpha=self.config.get('policy_gradient_weight', 1.0),
            beta=self.config.get('transition_loss_weight', 0.3),
            gamma=self.config.get('listener_loss_weight', 0.2),
            delta=self.config.get('cost_loss_weight', 0.1),
            baseline_cost=baseline_cost,
            cost_scale=cost_scale
        )
        print(f"📊 Four-objective loss initialized (baseline cost: ${baseline_cost:.4f}, scale: {cost_scale})")
        
        # Initialize state-description optimizers
        self.prompt_optimizers: Dict[str, Any] = {}
        if self.config.get('enable_prompt_optimization', False):
            print(f"✨ State-description optimization: enabled")
        
        # Initialize the attack injector
        self.attack_injector = None
        self.attack_stats = {
            'total_attacks': 0,
            'frequency_attacks': 0,
            'semantic_attacks': 0,
            'selfish_attacks': 0,
            'attacked_agents': set()
        }
        
        # Initialize anomaly-detection history for ProtectedTGN
        # These values are updated at the start of each question to detect frequency and semantic anomalies
        self.anomaly_detection_history = {
            'message_counts': {},  # {agent_id: [count1, count2, ...]} for frequency anomaly detection
            'embeddings': {},      # {agent_id: [emb1, emb2, ...]} for semantic anomaly detection
            'messages': {}         # {agent_id: str} for semantic consistency checks (BERT)
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
            
            # Create an attacker of the requested type
            if attack_type == 'frequency':
                self.attack_injector = create_frequency_attacker(attack_ratio, attack_strength, random_seed=attack_seed)
            elif attack_type == 'semantic':
                self.attack_injector = create_semantic_attacker(attack_ratio, attack_strength, random_seed=attack_seed)
            elif attack_type == 'selfish':
                self.attack_injector = create_selfish_attacker(attack_ratio, attack_strength, random_seed=attack_seed)
            
            print(f"🔴 Attack injector: enabled (type: {attack_type}, ratio: {attack_ratio*100:.0f}%, strength: {attack_strength})")
    
    def create_domain_fsm_system(self, domain: str, category: Optional[str] = None) -> Tuple[MultiAgentTopologyManager, FSMStateManager, FSMTemporalGraph]:
        """
        Create a complete FSM system for a domain.
        
        Args:
            domain: Domain name (mmlu/gsm8k/humaneval/hotpotqa/alfworld/math)
            category: MMLU category, only needed for MMLU category-level FSM loading
        
        Returns:
            (topology manager, FSM state manager, FSM-TGN)
        """
        print(f"\n🏗️  Creating an FSM system for domain {domain}" + (f" (category: {category})" if category else "") + "...")
        
        force_regenerate = self.config.get('force_regenerate_fsm', False)
        
        if (not force_regenerate) and domain == 'mmlu' and category and hasattr(self, 'fsm_cache_manager'):
            cached = self._load_category_fsm_if_available(domain, category)
            if cached:
                print(f"  ✅ translatedFSM: {category}")
                return self._create_fsm_from_cache(cached, domain)
        
        if (not force_regenerate) and hasattr(self, 'fsm_cache_manager') and self.fsm_cache_manager.has_cache(domain):
            cached = self.fsm_cache_manager.load_fsm(domain)
            print(f"  ✅ translatedFSM（force_regenerate_fsm=False）")
            return self._create_fsm_from_cache(cached, domain)
        
        agent_roles = self._get_agent_roles_for_domain(domain)
        print(f"  translated: {agent_roles}")
        
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
        
        agent_ids = list(topology.agent_execution_nodes.keys())
        fsm_manager = FSMStateManager(agent_ids)
        
        print(f"  🧬 translatedFSMtranslated {domain} translated...")
        if hasattr(self, 'fsm_generator') and hasattr(self, 'fsm_cache_manager'):
            try:
                complete_system, token_cost = self.fsm_generator.generate_complete_mas(dataset=domain)
                fsm_config = complete_system.get('fsm', {})
                agents = complete_system.get('agents', [])
                
                print(f"  💾 translatedFSMtranslated（domain={domain}）")
                self.fsm_cache_manager.save_fsm(
                    domain=domain,
                    fsm_config=fsm_config,
                    agents=agents
                )
                
                cached_wrapper = {
                    'fsm_config': fsm_config,
                    'agents': agents
                }
                topology, fsm_manager, fsm_tgn = self._create_fsm_from_cache(cached_wrapper, domain)
                
                print(f"\n  📋 translatedFSMtranslated:")
                initial_state = fsm_manager.get_initial_state()
                initial_state_id = initial_state.state_id if initial_state else None
                
                for state_id in sorted(fsm_manager.states.keys()):
                    state = fsm_manager.states[state_id]
                    marker = " [INITIAL]" if state_id == initial_state_id else ""
                    marker += " [FINAL]" if state.is_final else ""
                    print(f"    State {state_id}: {state.state_name}{marker}")
                    
                    agent_id = state.responsible_agent_id
                    agent_name = "Unknown"
                    if agent_id in topology.agent_execution_nodes:
                        agent_name = topology.agent_execution_nodes[agent_id].agent_role
                    print(f"      translated: {agent_id} ({agent_name})")
                    
                    if state.description:
                        desc = state.description[:150] + "..." if len(state.description) > 150 else state.description
                        print(f"      translated: {desc}")
                print()
                
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
                    print(f"  📝 translated")
                
                return topology, fsm_manager, fsm_tgn
            except Exception as e:
                print(f"  ⚠️  translatedFSMtranslated，translated: {e}")
        
        self._generate_fsm_states(fsm_manager, agent_roles, agent_ids, domain)
        
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
            temporal_dimension=self.config.get('time_dim', 32),
            communication_edge_dim=64
        ).to(self.device)
        
        use_protection = self.config.get('use_protection', False)
        if use_protection:
            try:
                from neural_fsm_mas.defense_mechanisms import ProtectedTGN
                import networkx as nx
                
                communication_graph = nx.DiGraph()
                for state_id, state in fsm_manager.states.items():
                    for listener_id in fsm_manager.listeners.get(state_id, []):
                        communication_graph.add_edge(state_id, listener_id)
                
                if len(communication_graph.nodes()) == 0:
                    communication_graph = nx.complete_graph(num_agents, nx.DiGraph())
                
                protection_config = self.config.get('protection', {})
                
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
                    ).to(self.device)
                    
                    if hasattr(self, 'text_embedding_model') and self.text_embedding_model is not None:
                        protected_base_tgn.set_text_embedding_model(self.text_embedding_model)
                    
                    fsm_tgn.base_tgn = protected_base_tgn
                    print(f"  🛡️  translatedFSM-TGN（translated: {self.device}）")
            except Exception as e:
                print(f"  ⚠️  translated: {e}")
                print(f"  translatedFSM-TGN")
        
        topology.use_fsm_mode = True
        topology.fsm_state_manager = fsm_manager
        
        print(f"✅ FSMtranslated")
        print(fsm_manager.get_state_summary())
        
        return topology, fsm_manager, fsm_tgn
    
    def _load_category_fsm_if_available(self, domain: str, category: str) -> Optional[Dict]:
        if not hasattr(self, 'fsm_cache_manager'):
            return None
        
        try:
            if self.fsm_cache_manager.has_cache(domain, category):
                return self.fsm_cache_manager.load_fsm(domain, category)
        except Exception as e:
            print(f"  ⚠️  translatedFSMtranslated: {e}")
        
        return None
    
    def _create_fsm_from_cache(self, cached: Dict, domain: str) -> Tuple[MultiAgentTopologyManager, FSMStateManager, FSMTemporalGraph]:
        fsm_config = cached.get('fsm_config', {})
        agents = cached.get('agents', [])
        
        if not agents or not fsm_config:
            raise ValueError(f"Invalid cached FSM: missing agents or fsm_config")
        
        agent_roles = [agent.get('name', f"Agent{i}") for i, agent in enumerate(agents)]
        
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
        
        agent_ids = list(topology.agent_execution_nodes.keys())
        fsm_manager = FSMStateManager(agent_ids)
        
        cached_id_to_topology_id: Dict[str, str] = {}
        for idx, agent in enumerate(agents):
            cached_agent_id = str(agent.get('agent_id', str(idx)))
            if idx < len(agent_ids):
                cached_id_to_topology_id[cached_agent_id] = agent_ids[idx]
        
        states = fsm_config.get('states', [])
        if not states:
            raise ValueError(f"Cached FSM has no states")
        
        agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        
        for state_data in states:
            state_id = int(state_data['state_id'])
            raw_agent_id = str(state_data['agent_id'])
            
            if raw_agent_id in cached_id_to_topology_id:
                agent_id = cached_id_to_topology_id[raw_agent_id]
            else:
                print(f"  ⚠️  Warning: State {state_id} references unknown cached agent_id {raw_agent_id}, skipping")
                continue
            
            if agent_id not in agent_id_to_index:
                print(f"  ⚠️  Warning: State {state_id} mapped to invalid agent_id {agent_id}, skipping")
                continue
            
            completion_condition = state_data.get('completion_condition', '')
            fsm_manager.add_state(
                state_id=state_id,
                state_name=state_data.get('state_name', f'State_{state_id}'),
                responsible_agent_id=agent_id,
                is_initial=state_data.get('is_initial', False),
                is_final=state_data.get('is_final', False),
                description=state_data.get('instruction', state_data.get('description', '')),
                completion_condition=completion_condition
            )
            
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
        
        transitions = fsm_config.get('transitions', [])
        if transitions:
            if not hasattr(fsm_manager, 'transition_rules'):
                fsm_manager.transition_rules = []
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
            print(f"  ✅ translated {len(normalized_transitions)} translated")

            print("  🔰 translatedFSMtranslated:")
            for rule in normalized_transitions:
                fs = rule['from_state']
                ts = rule['to_state']
                cond = rule.get('condition', '')
                print(f"    State {fs} → State {ts}: {cond}")
        else:
            print(f"  ⚠️  translated：FSMtranslated")
        
        initial_states = [s for s in fsm_manager.states.values() if s.is_initial]
        if initial_states:
            fsm_manager.current_state_id = initial_states[0].state_id
        elif fsm_manager.states:
            first_state_id = min(fsm_manager.states.keys())
            fsm_manager.current_state_id = first_state_id
            fsm_manager.states[first_state_id].is_initial = True
        
        use_protection = self.config.get('use_protection', False)
        
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
            temporal_dimension=self.config.get('time_dim', 32),
            communication_edge_dim=64
        ).to(self.device)
        
        if use_protection:
            try:
                from neural_fsm_mas.defense_mechanisms import ProtectedTGN
                import networkx as nx
                
                communication_graph = nx.DiGraph()
                for state_id, state in fsm_manager.states.items():
                    for listener_id in fsm_manager.listeners.get(state_id, []):
                        communication_graph.add_edge(state_id, listener_id)
                
                if len(communication_graph.nodes()) == 0:
                    communication_graph = nx.complete_graph(num_agents, nx.DiGraph())
                
                protection_config = self.config.get('protection', {})
                
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
                    ).to(self.device)
                    
                    if hasattr(self, 'text_embedding_model') and self.text_embedding_model is not None:
                        protected_base_tgn.set_text_embedding_model(self.text_embedding_model)
                    
                    fsm_tgn.base_tgn = protected_base_tgn
                    print(f"  🛡️  translatedFSM-TGN（translated，translated: {self.device}）")
            except Exception as e:
                print(f"  ⚠️  translated: {e}")
        
        topology.use_fsm_mode = True
        topology.fsm_state_manager = fsm_manager
        
        print(f"  ✅ translatedFSM: {num_states}translated, {num_agents}translated, {len(transitions)}translated")
        
        print(f"\n  📋 translatedFSMtranslated:")
        initial_state = fsm_manager.get_initial_state()
        initial_state_id = initial_state.state_id if initial_state else None
        
        for state_id in sorted(fsm_manager.states.keys()):
            state = fsm_manager.states[state_id]
            marker = " [INITIAL]" if state_id == initial_state_id else ""
            marker += " [FINAL]" if state.is_final else ""
            print(f"    State {state_id}: {state.state_name}{marker}")
            
            agent_id = state.responsible_agent_id
            agent_name = "Unknown"
            if agent_id in topology.agent_execution_nodes:
                agent_name = topology.agent_execution_nodes[agent_id].agent_role
            print(f"      translated: {agent_id} ({agent_name})")
            
            if state.description:
                desc = state.description[:150] + "..." if len(state.description) > 150 else state.description
                print(f"      translated: {desc}")
        print()
        
        return topology, fsm_manager, fsm_tgn
    
    def _get_agent_roles_for_domain(self, domain: str) -> List[str]:
        if hasattr(self, 'fsm_cache_manager') and self.fsm_cache_manager:
            try:
                if self.fsm_cache_manager.has_cache(domain):
                    cached = self.fsm_cache_manager.load_fsm(domain)
                    agents = cached.get('agents', [])
                    if agents:
                        agent_roles = [agent.get('name', f"Agent{i}") for i, agent in enumerate(agents)]
                        print(f"  ✅ translatedFSMtranslated ({len(agent_roles)}translated)")
                        return agent_roles
            except Exception as e:
                print(f"  ⚠️  translatedFSMtranslated: {e}")
        
        try:
            prompt_set = DomainPromptManager.get_manager(domain)
            agent_roles = prompt_set.get_available_roles()
            print(f"  ⚠️  translatedprompt_managertranslated ({len(agent_roles)}translated)")
            return agent_roles
        except Exception as e:
            print(f"  ⚠️  prompt_managertranslated: {e}")
        
        print(f"  ⚠️  translated")
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
        
        if len(state_names) < len(agent_roles):
            state_names.extend([f"State{i}" for i in range(len(state_names), len(agent_roles))])
        
        for i, (role, agent_id) in enumerate(zip(agent_roles, agent_ids)):
            fsm_manager.add_state(
                state_id=i,
                state_name=state_names[i],
                responsible_agent_id=agent_id,
                is_initial=(i == 0),
                is_final=(i == len(agent_roles) - 1),
                description=f"{state_names[i]} handled by {role}"
            )
        
        for i in range(len(agent_roles) - 1):
            fsm_manager.add_listener(state_id=i, listener_agent_id=agent_ids[i + 1])
    
    async def train_domain(self, 
                          domain: str,
                          num_epochs: int = 50) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"🎯 translated: {domain}")
        print(f"{'='*80}")
        
        use_category_fsm = (domain == 'mmlu' and 
                           self.config.get('mmlu_use_category_fsm', False))
        
        if use_category_fsm:
            print(f"  ℹ️  MMLUtranslatedFSM，translated")
            topology, fsm_manager, fsm_tgn = self.create_domain_fsm_system(domain)
        else:
            topology, fsm_manager, fsm_tgn = self.create_domain_fsm_system(domain)
        
        self.domain_topologies[domain] = topology
        self.domain_fsm_managers[domain] = fsm_manager
        self.domain_fsm_tgns[domain] = fsm_tgn
        
        if use_category_fsm:
            self.mmlu_category_fsms = {}  # {category: (topology, fsm_manager, fsm_tgn)}
        
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
        
        train_samples = self.config.get('train_samples', None)
        if train_samples is not None and train_samples > 0:
            import random
            all_train_samples = []
            for batch in training_batches:
                all_train_samples.extend(batch)
            
            if len(all_train_samples) > train_samples:
                random.seed(self.config.get('random_seed', 42))
                all_train_samples = random.sample(all_train_samples, train_samples)
                print(f"  🎲 translated {train_samples} translated（translated {sum(len(b) for b in training_batches)} translated）")
            
            batch_size = self.config.get('batch_size', 16)
            training_batches = [
                all_train_samples[i:i+batch_size] 
                for i in range(0, len(all_train_samples), batch_size)
            ]
        
        train_only = self.config.get('train_only', False)
        
        if train_only:
            validation_batches = []
            test_batches = []
            print(f"  ⚡ translated：translated")
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
        
        print(f"📊 translated: {len(training_batches)}, translated: {len(validation_batches)}, translated: {len(test_batches)}")
        
        optimizer = optim.Adam(fsm_tgn.parameters(), lr=self.config.get('learning_rate', 0.001))
        
        training_results = {
            'domain': domain,
            'epoch_losses': [],
            'epoch_policy_losses': [],
            'epoch_transition_losses': [],
            'epoch_listener_losses': [],
            'epoch_cost_losses': [],
            'epoch_protection_losses': [],
            'epoch_accuracies': [],
            'epoch_batch_accuracies': [],
            'validation_accuracies': [],
            'epoch_training_costs': [],
            'epoch_validation_costs': [],
            'best_accuracy': 0.0,
            'best_epoch': 0,
            'test_accuracy': 0.0,
            'test_cost': 0.0,
            'total_training_cost': 0.0,
            'total_validation_cost': 0.0,
            'total_cost': 0.0,
            'cost_statistics': {},
            'protection_statistics': {
                'epoch_avg_anomaly_scores': [],
                'epoch_avg_trust_scores': [],
                'epoch_avg_priorities': [],
                'epoch_high_anomaly_agents': []
            }
        }
        
        initial_cost = 0.0
        if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
            initial_cost = self.cost_tracker.get_total_cost()
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
            
            epoch_start_cost = 0.0
            if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                epoch_start_cost = self.cost_tracker.get_total_cost()
            
            epoch_metrics = await self._train_epoch(
                domain, topology, fsm_manager, fsm_tgn, 
                training_batches, optimizer
            )
            
            epoch_training_cost = 0.0
            if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                epoch_end_cost = self.cost_tracker.get_total_cost()
                epoch_training_cost = epoch_end_cost - epoch_start_cost
            
            if len(validation_batches) > 0:
                val_accuracy, epoch_validation_cost = await self._validate_epoch(
                    domain, topology, fsm_manager, validation_batches, is_test=False
            )
            else:
                val_accuracy = 0.0
                epoch_validation_cost = 0.0
                print(f"  ⚠️  translated（translated）")
            
            train_accuracy_rounded = round(epoch_metrics['accuracy'], 4)
            val_accuracy_rounded = round(val_accuracy, 4)
            
            training_results['epoch_losses'].append(epoch_metrics['total_loss'])
            training_results['epoch_policy_losses'].append(epoch_metrics['policy_loss'])
            training_results['epoch_transition_losses'].append(epoch_metrics['transition_loss'])
            training_results['epoch_listener_losses'].append(epoch_metrics['listener_loss'])
            training_results['epoch_cost_losses'].append(epoch_metrics.get('cost_loss', 0.0))
            training_results['epoch_protection_losses'].append(epoch_metrics.get('protection_loss', 0.0))
            training_results['epoch_accuracies'].append(train_accuracy_rounded)
            training_results['epoch_batch_accuracies'].append(epoch_metrics.get('batch_accuracies', []))
            training_results['validation_accuracies'].append(val_accuracy_rounded)
            training_results['epoch_training_costs'].append(epoch_training_cost)
            training_results['epoch_validation_costs'].append(epoch_validation_cost)
            
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
            
            if len(validation_batches) > 0:
                current_metric = val_accuracy
                current_metric_rounded = val_accuracy_rounded
                metric_name = "translated"
            else:
                current_metric = epoch_metrics['accuracy']
                current_metric_rounded = train_accuracy_rounded
                metric_name = "translated"
            
            if current_metric > training_results['best_accuracy']:
                training_results['best_accuracy'] = current_metric_rounded
                training_results['best_epoch'] = epoch + 1
                self._save_best_model(domain, fsm_tgn, epoch, current_metric_rounded)
                print(f"  💾 translated（{metric_name}translated: {current_metric_rounded:.4f}）")
            
            if len(validation_batches) > 0:
                print(f"  📊 translated: {train_accuracy_rounded:.4f}, translated: {val_accuracy_rounded:.4f}")
            else:
                print(f"  📊 translated: {train_accuracy_rounded:.4f}")
            protection_loss_str = ""
            if self.config.get('use_protection', False):
                protection_loss_str = f", translated={epoch_metrics.get('protection_loss', 0.0):.4f}"
            print(f"  💰 translated: translated={epoch_metrics['total_loss']:.4f}, "
                  f"translated={epoch_metrics['policy_loss']:.4f}, "
                  f"translated={epoch_metrics['transition_loss']:.4f}, "
                  f"translated={epoch_metrics['listener_loss']:.4f}{protection_loss_str}, "
                  f"translated={epoch_metrics.get('max_transitions_penalty', 0.0):.4f}")
            print(f"  💵 translated: translated=${epoch_training_cost:.6f}, translated=${epoch_validation_cost:.6f}")
        
        training_results['total_training_cost'] = sum(training_results['epoch_training_costs'])
        training_results['total_validation_cost'] = sum(training_results['epoch_validation_costs'])
        
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
            print(f"  ⚠️  translated（translated）")
        training_results['total_cost'] = (
            training_results['total_training_cost'] + 
            training_results['total_validation_cost'] + 
            training_results['test_cost']
        )
        
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
        
        final_train_accuracy = training_results['epoch_accuracies'][-1] if training_results['epoch_accuracies'] else 0.0
        final_train_accuracy = round(final_train_accuracy, 4)
        
        if training_results['epoch_accuracies']:
            total_train_accuracy = sum(training_results['epoch_accuracies']) / len(training_results['epoch_accuracies'])
        else:
            total_train_accuracy = 0.0
        
        total_train_accuracy = round(total_train_accuracy, 4)
        training_results['total_train_accuracy'] = total_train_accuracy
        
        print(f"\n✅ translated {domain} translated")
        print(f"📊 translated（translatedepoch）: {final_train_accuracy:.4f}")
        print(f"📊 translated（translatedepochtranslated）: {total_train_accuracy:.4f}")
        print(f"🧪 translated: {test_accuracy:.4f}")
        print(f"💰 translated:")
        print(f"  translated: ${training_results['total_training_cost']:.6f} USD")
        print(f"  translated: ${training_results['total_validation_cost']:.6f} USD")
        print(f"  translated: ${training_results['test_cost']:.6f} USD")
        print(f"  translated: ${training_results['total_cost']:.6f} USD")
        
        if self.config.get('enable_attack', False):
            print(f"\n🔴 translated:")
            print(f"  translated: {self.attack_stats['total_attacks']}")
            print(f"  translated: {self.attack_stats['frequency_attacks']}")
            print(f"  translated: {self.attack_stats['semantic_attacks']}")
            print(f"  translated: {self.attack_stats['selfish_attacks']}")
            print(f"  translated: {len(self.attack_stats['attacked_agents'])}")
            
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
        
        fsm_tgn.train()

        if optimizer is None or not hasattr(optimizer, "zero_grad"):
            raise ValueError("Torch optimizer is invalid (None or missing zero_grad).")
        
        total_policy_loss = 0.0
        total_transition_loss = 0.0
        total_listener_loss = 0.0
        total_cost_loss = 0.0
        total_max_transitions_penalty = 0.0
        total_protection_loss = 0.0
        total_correct = 0
        total_questions = 0
        batch_accuracies: List[float] = []
        
        # ==================================================================
        
        alpha = self.config.get('policy_gradient_weight', 1.0)
        beta = self.config.get('transition_loss_weight', 0.3)
        gamma = self.config.get('listener_loss_weight', 0.2)
        delta = self.config.get('cost_loss_weight', 0.1)
        
        epsilon = self.config.get('protection_loss_weight', 0.05)
        
        zeta = self.config.get('max_transitions_penalty_weight', 0.5)
        
        print(f"  📦 translated，translated {len(training_batches)} translated")
        
        epoch_anomaly_scores: List[float] = []
        epoch_trust_scores: List[float] = []
        epoch_priorities: List[float] = []
        epoch_high_anomaly_agents: List[List[int]] = []
        
        for batch_idx, batch in enumerate(training_batches):
            batch_policy_loss = torch.tensor(0.0, device=self.device)
            batch_transition_loss = torch.tensor(0.0, device=self.device)
            batch_listener_loss = torch.tensor(0.0, device=self.device)
            batch_max_transitions_penalty = torch.tensor(0.0, device=self.device)
            batch_protection_loss = torch.tensor(0.0, device=self.device)
            batch_correct = 0
            batch_episode_costs: List[float] = []
            
            for question_data in batch:
                try:
                    current_topology = topology
                    current_fsm_manager = fsm_manager
                    current_fsm_tgn = fsm_tgn
                    
                    if domain == 'mmlu' and self.config.get('mmlu_use_category_fsm', False):
                        category = question_data.get('subject', 'unknown')
                        current_topology, current_fsm_manager, current_fsm_tgn = self._get_or_create_category_fsm(domain, category)
                    
                    formatted_question = self.data_processor.format_question_for_agents(question_data, domain)
                    
                    if self.config.get('enable_prompt_optimization', False) and domain in self.prompt_optimizers:
                        prompt_optimizer = self.prompt_optimizers[domain]
                        
                        for state_id, state in current_fsm_manager.states.items():
                            if state.description:
                                enhanced_desc, was_enhanced = prompt_optimizer.get_enhanced_description(
                                    state.state_name,
                                    state.description
                                )
                                if was_enhanced:
                                    state.description = enhanced_desc
                    
                    state_summaries = []
                    for s in current_fsm_manager.states.values():
                        state_summaries.append(f"State {s.state_id}: {s.state_name}")
                    state_block = "\n".join(state_summaries)
                    
                    agent_summaries = []
                    for agent_id, agent_node in current_topology.agent_execution_nodes.items():
                        agent_summaries.append(f"{agent_id}: {agent_node.agent_role}")
                    agent_block = "\n".join(agent_summaries)
                    
                    enriched_text = (
                        f"{formatted_question}\n\n"
                        f"[FSM States: {len(state_summaries)} states]\n{state_block}\n\n"
                        f"[Agents: {len(agent_summaries)} agents]\n{agent_block}"
                    )
                    
                    question_embedding = self.text_embedding_model.encode_query(enriched_text)
                    question_embedding = question_embedding.to(self.device)
                    
                    current_fsm_manager.reset()
                    agent_features = self._get_agent_features(current_topology)
                    context_features = self._get_context_features(domain)
                    
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
                    
                    attacked_communication_counts = None
                    if self.attack_injector is not None and self.config.get('enable_attack', False):
                        attack_config = self.config.get('attack', {})
                        attack_type = attack_config.get('attack_type', 'selfish')
                        attack_frequency = attack_config.get('attack_frequency', 'always')
                        attack_start_epoch = attack_config.get('attack_start_epoch', 0)
                        current_epoch = getattr(self, '_current_epoch', 0)

                        try:
                            final_states = [s for s in current_fsm_manager.states.values() if getattr(s, 'is_final', False)]
                            if final_states and hasattr(self.attack_injector, 'set_excluded_agents'):
                                final_agent_id = final_states[0].responsible_agent_id
                                final_idx = int(final_agent_id.split('_')[-1]) if isinstance(final_agent_id, str) and '_' in final_agent_id else int(final_agent_id)
                                self.attack_injector.set_excluded_agents([final_idx], clear_if_conflict=True)
                        except Exception:
                            pass
                    
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
                            communication_counts = {i: 10 for i in range(num_agents)}
                            
                            attack_result = self.attack_injector.inject_attack(
                                agent_features=agent_features.clone(),
                                communication_graph=torch.ones(num_agents, num_agents),
                                communication_counts=communication_counts
                            )
                            
                            agent_features = attack_result['agent_features']
                            
                            if 'communication_counts' in attack_result:
                                attacked_communication_counts = attack_result['communication_counts']
                            
                            if attack_type == 'selfish' and 'communication_graph' in attack_result:
                                attacked_graph = attack_result['communication_graph']
                                edge_list = []
                                for i in range(num_agents):
                                    for j in range(num_agents):
                                        if i != j and attacked_graph[i, j].item() > 0.5:
                                            edge_list.append([i, j])
                                if edge_list:
                                    edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
                                else:
                                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                            
                            self.attack_stats['total_attacks'] += 1
                            if attack_type == 'frequency':
                                self.attack_stats['frequency_attacks'] += 1
                            elif attack_type == 'semantic':
                                self.attack_stats['semantic_attacks'] += 1
                            elif attack_type == 'selfish':
                                self.attack_stats['selfish_attacks'] += 1
                            
                            if hasattr(self.attack_injector, 'attacked_agents'):
                                self.attack_stats['attacked_agents'].update(self.attack_injector.attacked_agents)
                    
                    message_counts_for_detection = None
                    embeddings_for_detection = None
                    if self.config.get('use_protection', False):
                        if attacked_communication_counts is not None:
                            message_counts_for_detection = attacked_communication_counts
                        else:
                            message_counts_for_detection = {i: 10 for i in range(num_agents)}
                        
                        for agent_id in range(num_agents):
                            if agent_id not in self.anomaly_detection_history['message_counts']:
                                self.anomaly_detection_history['message_counts'][agent_id] = []
                            self.anomaly_detection_history['message_counts'][agent_id].append(
                                message_counts_for_detection.get(agent_id, 10)
                            )
                            if len(self.anomaly_detection_history['message_counts'][agent_id]) > 20:
                                self.anomaly_detection_history['message_counts'][agent_id].pop(0)
                        
                        embeddings_for_detection = {}
                        for agent_id in range(num_agents):
                            if agent_id not in self.anomaly_detection_history['embeddings']:
                                self.anomaly_detection_history['embeddings'][agent_id] = []
                            self.anomaly_detection_history['embeddings'][agent_id].append(
                                agent_features[agent_id].detach().cpu()
                            )
                            if len(self.anomaly_detection_history['embeddings'][agent_id]) > 20:
                                self.anomaly_detection_history['embeddings'][agent_id].pop(0)
                            embeddings_for_detection[agent_id] = self.anomaly_detection_history['embeddings'][agent_id][-1]
                    
                    from neural_fsm_mas.defense_mechanisms import ProtectedTGN
                    protection_active = (
                        self.config.get('use_protection', False) and 
                        hasattr(current_fsm_tgn, 'base_tgn') and 
                        isinstance(current_fsm_tgn.base_tgn, ProtectedTGN)
                    )
                    
                    if protection_active:
                        agent_messages_for_detection = dict(self.anomaly_detection_history.get('messages', {}))
                        
                        if 'communication_edges' in self.anomaly_detection_history:
                            agent_messages_for_detection['_communication_edges'] =\
                                self.anomaly_detection_history['communication_edges']
                        
                        fsm_outputs = current_fsm_tgn(
                            agent_features=agent_features,
                            communication_topology=edge_index,
                            context_features=context_features,
                            current_state_id=current_fsm_manager.current_state_id,
                            question_embedding=question_embedding,
                            message_counts=message_counts_for_detection,
                            embeddings=embeddings_for_detection,
                            agent_messages=agent_messages_for_detection
                        )
                    else:
                        fsm_outputs = current_fsm_tgn(
                            agent_features=agent_features,
                            communication_topology=edge_index,
                            context_features=context_features,
                            current_state_id=current_fsm_manager.current_state_id,
                            question_embedding=question_embedding
                        )
                    
                    episode_cost = 0.0
                    if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                        from neural_fsm_mas.losses.cost_loss import calculate_episode_cost
                        episode_start_idx = len(self.cost_tracker.call_history)
                    else:
                        episode_start_idx = None
                    
                    final_answer, execution_log = await self._execute_fsm_with_learned_structure(
                        current_topology, current_fsm_manager, fsm_outputs, formatted_question,
                        fsm_tgn=current_fsm_tgn,
                        question_embedding=question_embedding,
                        agent_features=agent_features,
                        context_features=context_features,
                        attack_injector=self.attack_injector if self.config.get('enable_attack', False) else None
                    )
                    
                    if episode_start_idx is not None:
                        from neural_fsm_mas.losses.cost_loss import calculate_episode_cost
                        episode_cost = calculate_episode_cost(self.cost_tracker, episode_start_idx)
                        batch_episode_costs.append(episode_cost)
                    
                    if 'collected_agent_messages' in execution_log:
                        collected_messages = execution_log['collected_agent_messages']
                        if collected_messages:
                            pure_messages = {k: v for k, v in collected_messages.items() 
                                           if not isinstance(k, str) or not k.startswith('_')}
                            self.anomaly_detection_history['messages'] = pure_messages
                            
                            if '_communication_edges' in collected_messages:
                                sampled_edges = collected_messages['_communication_edges']
                                self.anomaly_detection_history['communication_edges'] = sampled_edges
                                
                                if protection_active and hasattr(current_fsm_tgn, 'base_tgn'):
                                    base_tgn = current_fsm_tgn.base_tgn
                                    if hasattr(base_tgn, 'anomaly_detector'):
                                        agent_msg_counts = {}
                                        for src, dst in sampled_edges:
                                            try:
                                                src_idx = int(src.split('_')[-1]) if isinstance(src, str) and '_' in src else int(src)
                                                agent_msg_counts[src_idx] = agent_msg_counts.get(src_idx, 0) + 1
                                            except (ValueError, AttributeError):
                                                pass
                                        for agent_id, count in agent_msg_counts.items():
                                            base_tgn.anomaly_detector.update_history(agent_id, message_count=count)
                    
                    ground_truth = self._extract_ground_truth(question_data, domain)
                    validation_kwargs = self._get_validation_kwargs(question_data, domain)
                    is_correct = self._check_correctness(final_answer, ground_truth, domain, **validation_kwargs)
                    reward = 1.0 if is_correct else 0.0
                    
                    try:
                        short_q = formatted_question.replace("\n", " ")[:200]
                    except Exception:
                        short_q = str(formatted_question)[:200]
                    short_pred = str(final_answer).replace("\n", " ")[:200]
                    short_gt = str(ground_truth).replace("\n", " ")[:200]
                    result_flag = "✅ translated" if is_correct else "❌ translated"
                    print("\n  ── translated ──")
                    print(f"  Q: {short_q}")
                    print(f"  Pred: {short_pred}")
                    print(f"  GT:   {short_gt}")
                    if domain == 'alfworld':
                        subgoals = question_data.get('subgoals', [])
                        if subgoals:
                            print(f"  Subgoals (from dataset): {subgoals}")
                    print(f"  Result: {result_flag}")
                    
                    if self.config.get('use_protection', False) and 'protection_data' in fsm_outputs:
                        protection_data = fsm_outputs['protection_data']
                        print(f"\n  🛡️  Protection Data (translatedAgenttranslated):")
                        
                        num_agents = 0
                        if 'anomaly_scores' in protection_data:
                            num_agents = len(protection_data['anomaly_scores'])
                        
                        print(f"    {'Agent':<8} {'translated':<12} {'translated':<12} {'translated':<12}")
                        print(f"    {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
                        
                        for agent_id in range(num_agents):
                            anomaly_val = protection_data.get('anomaly_scores', torch.zeros(num_agents))[agent_id].item()
                            priority_val = protection_data.get('priorities', torch.zeros(num_agents))[agent_id].item()
                            trust_val = protection_data.get('trust_scores', torch.zeros(num_agents))[agent_id].item()
                            
                            marker = "⚠️" if anomaly_val > 0.5 else "  "
                            print(f"    {marker}Agent {agent_id:<3} {anomaly_val:<12.4f} {priority_val:<12.4f} {trust_val:<12.4f}")
                        
                        print(f"\n    📊 translated:")
                        if 'anomaly_scores' in protection_data:
                            anomaly = protection_data['anomaly_scores']
                            print(f"      translated: mean={anomaly.mean().item():.4f}, max={anomaly.max().item():.4f}")
                            epoch_anomaly_scores.append(anomaly.mean().item())
                            high_anomaly_agents = [i for i in range(len(anomaly)) if anomaly[i].item() > 0.5]
                            if high_anomaly_agents:
                                epoch_high_anomaly_agents.append(high_anomaly_agents)
                        if 'trust_scores' in protection_data:
                            trust = protection_data['trust_scores']
                            print(f"      translated: mean={trust.mean().item():.4f}, min={trust.min().item():.4f}")
                            epoch_trust_scores.append(trust.mean().item())
                        if 'priorities' in protection_data:
                            priorities = protection_data['priorities']
                            epoch_priorities.append(priorities.mean().item())
                        if 'message_weights' in protection_data and protection_data['message_weights'].numel() > 0:
                            msg_weights = protection_data['message_weights']
                            print(f"      translated: mean={msg_weights.mean().item():.4f}, "
                                  f"range=[{msg_weights.min().item():.4f}, {msg_weights.max().item():.4f}]")
                        
                        if protection_active and hasattr(current_fsm_tgn, 'base_tgn'):
                            base_tgn = current_fsm_tgn.base_tgn
                            if hasattr(base_tgn, 'anomaly_detector'):
                                cum_scores = base_tgn.anomaly_detector.cumulative_anomaly_score
                                poll_history = base_tgn.anomaly_detector.pollution_history
                                if cum_scores:
                                    print(f"\n    📈 translated（translated）:")
                                    for agent_id in sorted(cum_scores.keys()):
                                        cum_val = cum_scores[agent_id]
                                        poll_count = len(poll_history.get(agent_id, []))
                                        if cum_val > 0.1 or poll_count > 0:
                                            print(f"      Agent {agent_id}: translated={cum_val:.4f}, translated={poll_count}")
                    
                    if self.config.get('enable_prompt_optimization', False):
                        prompt_optimizer = self.prompt_optimizers.get(domain)
                        if prompt_optimizer:
                            visited_state_names = []
                            if 'state_transitions' in execution_log:
                                for transition in execution_log['state_transitions']:
                                    if len(transition) >= 2:
                                        from_state_id, to_state_id = transition[0], transition[1]
                                        from_state = current_fsm_manager.states.get(from_state_id)
                                        if from_state and (not visited_state_names or visited_state_names[-1] != from_state.state_name):
                                            visited_state_names.append(from_state.state_name)
                                        to_state = current_fsm_manager.states.get(to_state_id)
                                        if to_state:
                                            visited_state_names.append(to_state.state_name)
                            
                            reached_max_transitions = execution_log.get('reached_max_transitions', False)
                            
                            prompt_optimizer.record_state_execution(
                                visited_states=visited_state_names,
                                is_correct=is_correct,
                                reached_max_transitions=reached_max_transitions
                            )
                    
                    log_prob = execution_log.get('log_prob', 0.0)
                    policy_loss = -log_prob * reward if isinstance(log_prob, (int, float)) else -log_prob.mean() * reward
                    if not isinstance(policy_loss, torch.Tensor):
                        policy_loss = torch.tensor(float(policy_loss), device=self.device)
                    
                    if 'transition_probs' in fsm_outputs:
                        transition_probs = fsm_outputs['transition_probs']  # [num_states]
                        transition_targets = self._get_transition_targets(execution_log, current_fsm_manager)
                        if transition_targets is not None and len(transition_targets) > 0:
                            num_states = transition_probs.size(0)
                            losses = []
                            for target_state in transition_targets:
                                if 0 <= target_state < num_states:
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
                    
                    if 'listener_weights' in fsm_outputs:
                        listener_weights = fsm_outputs['listener_weights']  # [num_states, num_agents]
                        listener_targets = self._get_listener_targets(current_fsm_manager)
                        if listener_targets is not None:
                            listener_loss = nn.functional.binary_cross_entropy(
                                listener_weights,
                                listener_targets,
                                reduction='mean'
                            )
                        else:
                            listener_loss = torch.tensor(0.0, device=self.device)
                    else:
                        listener_loss = torch.tensor(0.0, device=self.device)
                    
                    reached_max_transitions = execution_log.get('reached_max_transitions', False)
                    if reached_max_transitions:
                        max_transitions_penalty = torch.tensor(1.0, device=self.device, requires_grad=True)
                    else:
                        max_transitions_penalty = torch.tensor(0.0, device=self.device)
                    
                    protection_loss = torch.tensor(0.0, device=self.device)
                    has_protection = (
                        self.config.get('use_protection', False) and 
                        hasattr(current_fsm_tgn, 'base_tgn') and 
                        current_fsm_tgn.base_tgn is not None and
                        hasattr(current_fsm_tgn.base_tgn, 'protection_loss')
                    )
                    if has_protection:
                        try:
                            protection_data = fsm_outputs.get('protection_data', {})
                            
                            if protection_data:
                                anomaly_scores = protection_data['anomaly_scores']
                                priorities = protection_data['priorities']
                                
                                if edge_index.size(1) > 0:
                                    source_features = agent_features[edge_index[0]]  # [num_edges, feature_dim]
                                    target_features = agent_features[edge_index[1]]  # [num_edges, feature_dim]
                                    messages = torch.cat([source_features, target_features], dim=-1)  # [num_edges, 2*feature_dim]
                                else:
                                    messages = torch.zeros((0, agent_features.size(1) * 2), device=self.device)
                                
                                protection_loss = current_fsm_tgn.base_tgn.protection_loss.compute_protection_loss(
                                    messages=messages,
                                    anomaly_scores=anomaly_scores,
                                    priorities=priorities,
                                    edge_index=edge_index
                                )
                                
                                if protection_loss.item() > 0:
                                    print(f"    🛡️  translated: {protection_loss.item():.6f}")
                        except Exception as e:
                            print(f"    ⚠️  translated: {e}")
                            import traceback
                            traceback.print_exc()
                            protection_loss = torch.tensor(0.0, device=self.device)
                    
                    batch_policy_loss = batch_policy_loss + policy_loss
                    batch_transition_loss += transition_loss
                    batch_listener_loss += listener_loss
                    batch_max_transitions_penalty += max_transitions_penalty
                    batch_protection_loss += protection_loss
                    
                    if is_correct:
                        batch_correct += 1
                    
                except Exception as e:
                    print(f"  ⚠️  translated: {e}")
                    continue
            
            # ===============================================================
            batch_cost_loss = 0.0
            if batch_correct > 0 or len(batch) > 0:
                if hasattr(self, 'compute_four_objective_loss') and batch_episode_costs:
                    total_loss, loss_detail = self.compute_four_objective_loss(
                        batch_policy_loss,
                        batch_transition_loss,
                        batch_listener_loss,
                        batch_episode_costs
                    )
                    batch_cost_loss = loss_detail.get('cost', 0.0)
                    total_loss = total_loss + epsilon * batch_protection_loss + zeta * batch_max_transitions_penalty
                else:
                    batch_cost_loss = 0.0
                    total_loss = (alpha * batch_policy_loss + 
                            beta * batch_transition_loss + 
                            gamma * batch_listener_loss +
                            epsilon * batch_protection_loss +
                            zeta * batch_max_transitions_penalty)
                
                optimizer.zero_grad()
                if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(fsm_tgn.parameters(), max_norm=1.0)
                    if hasattr(self, 'mmlu_category_fsms'):
                        for _, _, cat_fsm_tgn in self.mmlu_category_fsms.values():
                            if cat_fsm_tgn is not fsm_tgn:
                                torch.nn.utils.clip_grad_norm_(cat_fsm_tgn.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_policy_loss += batch_policy_loss.item() if isinstance(batch_policy_loss, torch.Tensor) else batch_policy_loss
                total_transition_loss += batch_transition_loss.item() if isinstance(batch_transition_loss, torch.Tensor) else batch_transition_loss
                total_listener_loss += batch_listener_loss.item() if isinstance(batch_listener_loss, torch.Tensor) else batch_listener_loss
                total_cost_loss += batch_cost_loss
                total_protection_loss += batch_protection_loss.item() if isinstance(batch_protection_loss, torch.Tensor) else 0.0
                total_max_transitions_penalty += batch_max_transitions_penalty.item() if isinstance(batch_max_transitions_penalty, torch.Tensor) else batch_max_transitions_penalty
                total_correct += batch_correct
                total_questions += len(batch)
            
            batch_accuracy = batch_correct / len(batch) if len(batch) > 0 else 0.0
            print(f"    Batch {batch_idx + 1}/{len(training_batches)}, "
                  f"Acc: {batch_correct}/{len(batch)} ({batch_accuracy:.4f})")
            batch_accuracies.append(batch_accuracy)
        
        num_batches = len(training_batches) if len(training_batches) > 0 else 1
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_transition_loss = total_transition_loss / num_batches if num_batches > 0 else 0.0
        avg_listener_loss = total_listener_loss / num_batches if num_batches > 0 else 0.0
        avg_cost_loss = total_cost_loss / num_batches if num_batches > 0 else 0.0
        avg_protection_loss = total_protection_loss / num_batches if num_batches > 0 else 0.0
        avg_max_transitions_penalty = total_max_transitions_penalty / num_batches if num_batches > 0 else 0.0
        avg_total_loss = (alpha * avg_policy_loss + 
                         beta * avg_transition_loss + 
                         gamma * avg_listener_loss +
                         delta * avg_cost_loss +
                         epsilon * avg_protection_loss +
                         zeta * avg_max_transitions_penalty)
        avg_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
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
            'cost_loss': avg_cost_loss,
            'protection_loss': avg_protection_loss,
            'max_transitions_penalty': avg_max_transitions_penalty,
            'accuracy': avg_accuracy,
            'batch_accuracies': batch_accuracies,
            'protection_statistics': protection_statistics,
        }
    
    async def _validate_epoch(self,
                             domain: str,
                             topology: MultiAgentTopologyManager,
                             fsm_manager: FSMStateManager,
                             validation_batches: List[Dict],
                             is_test: bool = False) -> Tuple[float, float]:
        
        total_correct = 0
        total_questions = 0
        total_cost = 0.0
        
        for batch in validation_batches:
            for question_data in batch:
                try:
                    formatted_question = self.data_processor.format_question_for_agents(question_data, domain)
                    
                    question_start_cost = 0.0
                    if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                        question_start_cost = self.cost_tracker.get_total_cost()
                    
                    fsm_manager.reset()
                    
                    agent_features = self._get_agent_features(topology)
                    context_features = self._get_context_features(domain)
                    
                    state_summaries = []
                    for s in fsm_manager.states.values():
                        state_summaries.append(f"State {s.state_id}: {s.state_name}")
                    state_block = "\n".join(state_summaries)
                    
                    agent_summaries = []
                    for agent_id, agent_node in topology.agent_execution_nodes.items():
                        agent_summaries.append(f"{agent_id}: {agent_node.agent_role}")
                    agent_block = "\n".join(agent_summaries)
                    
                    enriched_text = (
                        f"{formatted_question}\n\n"
                        f"[FSM States: {len(state_summaries)} states]\n{state_block}\n\n"
                        f"[Agents: {len(agent_summaries)} agents]\n{agent_block}"
                    )
                    
                    question_emb = self.text_embedding_model.encode_query(enriched_text)
                    question_emb = question_emb.to(self.device)
                    
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
                    
                    attacked_communication_counts = None
                    if self.attack_injector is not None and self.config.get('enable_attack', False):
                        communication_counts = {i: 10 for i in range(num_agents)}
                        
                        attack_result = self.attack_injector.inject_attack(
                            agent_features=agent_features.clone(),
                            communication_graph=torch.ones(num_agents, num_agents),
                            communication_counts=communication_counts
                        )
                        
                        agent_features = attack_result['agent_features']
                        
                        if 'communication_counts' in attack_result:
                            attacked_communication_counts = attack_result['communication_counts']
                        
                        attack_config = self.config.get('attack', {})
                        attack_type = attack_config.get('attack_type', 'selfish')
                        if attack_type == 'selfish' and 'communication_graph' in attack_result:
                            attacked_graph = attack_result['communication_graph']
                            edge_list = []
                            for i in range(num_agents):
                                for j in range(num_agents):
                                    if i != j and attacked_graph[i, j].item() > 0.5:
                                        edge_list.append([i, j])
                            if edge_list:
                                edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
                            else:
                                edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    
                    message_counts_for_detection = None
                    embeddings_for_detection = None
                    if self.config.get('use_protection', False):
                        if attacked_communication_counts is not None:
                            message_counts_for_detection = attacked_communication_counts
                        else:
                            message_counts_for_detection = {i: 10 for i in range(num_agents)}
                        
                        embeddings_for_detection = {}
                        for agent_id in range(num_agents):
                            embeddings_for_detection[agent_id] = agent_features[agent_id].detach().cpu()
                    
                    fsm_tgn = self.domain_fsm_tgns.get(domain)
                    if fsm_tgn is not None:
                        from neural_fsm_mas.defense_mechanisms import ProtectedTGN
                        protection_active = (
                            self.config.get('use_protection', False) and 
                            hasattr(fsm_tgn, 'base_tgn') and 
                            isinstance(fsm_tgn.base_tgn, ProtectedTGN)
                        )
                        
                        if protection_active:
                            fsm_outputs = fsm_tgn(
                                agent_features=agent_features,
                                communication_topology=edge_index,
                                context_features=context_features,
                                current_state_id=fsm_manager.current_state_id,
                                question_embedding=question_emb,
                                message_counts=message_counts_for_detection,
                                embeddings=embeddings_for_detection
                            )
                        else:
                            fsm_outputs = fsm_tgn(
                                agent_features=agent_features,
                                communication_topology=edge_index,
                                context_features=context_features,
                                current_state_id=fsm_manager.current_state_id,
                                question_embedding=question_emb
                            )
                    else:
                        fsm_outputs = None
                    
                    final_answer, _, _ = await topology.execute_fsm_reasoning_with_sampling(
                        task_input=formatted_question,
                        fsm_outputs=fsm_outputs,
                        fsm_manager=fsm_manager,
                        fsm_tgn=fsm_tgn,
                        question_embedding=question_emb,
                        agent_features=agent_features,
                        context_features=context_features,
                        max_transitions=self.config.get('max_transitions', 8),
                        verbose=False,
                        phase="test" if is_test else "val"
                    )
                    
                    question_cost = 0.0
                    if hasattr(self, 'cost_tracker') and self.cost_tracker is not None:
                        question_end_cost = self.cost_tracker.get_total_cost()
                        question_cost = question_end_cost - question_start_cost
                    total_cost += question_cost
                    
                    ground_truth = self._extract_ground_truth(question_data, domain)
                    validation_kwargs = self._get_validation_kwargs(question_data, domain)
                    is_correct = self._check_correctness(final_answer, ground_truth, domain, **validation_kwargs)

                    if is_test:
                        debug_prefix = "🧪 translated"
                    else:
                        debug_prefix = "✅ translated"
                    print(f"{debug_prefix} Q{total_questions+1}: translated='{final_answer}' vs translated='{ground_truth}' -> {'✓' if is_correct else '✗'}")
                    
                    if is_correct:
                        total_correct += 1
                    total_questions += 1
                    
                except Exception as e:
                    continue
        
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        return accuracy, total_cost
    
    
    def _get_agent_features(self, topology: MultiAgentTopologyManager) -> torch.Tensor:
        if hasattr(topology, '_agent_embeddings') and topology._agent_embeddings is not None:
            return topology._agent_embeddings.to(self.device)
        else:
            num_agents = len(topology.agent_execution_nodes)
            return torch.randn(num_agents, 256, device=self.device)
    
    def _get_context_features(self, domain: str) -> torch.Tensor:
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
        try:
            final_answer, log_probs, reached_max_transitions, collected_agent_messages = await topology.execute_fsm_reasoning_with_sampling(
                task_input=question,
                fsm_outputs=fsm_outputs,
                fsm_manager=fsm_manager,
                fsm_tgn=fsm_tgn,
                question_embedding=question_embedding,
                agent_features=agent_features,
                context_features=context_features,
                enable_transition_prediction=self.config.get('enable_transition_prediction', True),
                enable_comm_sampling=self.config.get('enable_comm_sampling', True),
                max_transitions=self.config.get('max_transitions', 8),
                phase="train",
                attack_injector=attack_injector
            )
            return final_answer, {
                'log_prob': log_probs, 
                'reached_max_transitions': reached_max_transitions,
                'collected_agent_messages': collected_agent_messages
            }
        except Exception as e:
            print(f"⚠️  translatedFSMtranslated: {e}")
            import traceback
            traceback.print_exc()
            return "", {'log_prob': 0.0, 'reached_max_transitions': False}
    
    def _get_transition_targets(self, execution_log: Dict, fsm_manager: FSMStateManager) -> Optional[torch.Tensor]:
        if fsm_manager.transition_history:
            num_states = len(fsm_manager.states)
            targets = []
            for from_state, to_state in fsm_manager.transition_history:
                if 0 <= to_state < num_states:
                    targets.append(to_state)
            if targets:
                return torch.tensor(targets, dtype=torch.long, device=self.device)
        return None
    
    def _get_listener_targets(self, fsm_manager: FSMStateManager) -> Optional[torch.Tensor]:
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
            return question_data.get('code', '')
        return ''
    
    def _get_validation_kwargs(self, question_data: Dict, domain: str) -> Dict[str, Any]:
        kwargs = {}
        
        if domain == 'humaneval':
            kwargs['test_code'] = question_data.get('test_code', '')
            kwargs['entry_point'] = question_data.get('entry_point', '') or question_data.get('answer', '')
        elif domain == 'mbpp':
            kwargs['test_list'] = question_data.get('test_list', [])
        elif domain == 'alfworld':
            kwargs['subgoals'] = question_data.get('subgoals', [])
        
        return kwargs
    
    def _get_or_create_category_fsm(self, domain: str, category: str) -> Tuple[MultiAgentTopologyManager, FSMStateManager, FSMTemporalGraph]:
        if hasattr(self, 'mmlu_category_fsms') and category in self.mmlu_category_fsms:
            return self.mmlu_category_fsms[category]
        
        if hasattr(self, 'fsm_cache_manager'):
            cached = self._load_category_fsm_if_available(domain, category)
            if cached:
                topology, fsm_manager, fsm_tgn = self._create_fsm_from_cache(cached, domain)
                if not hasattr(self, 'mmlu_category_fsms'):
                    self.mmlu_category_fsms = {}
                self.mmlu_category_fsms[category] = (topology, fsm_manager, fsm_tgn)
                return topology, fsm_manager, fsm_tgn
        
        if self.config.get('generate_fsm_if_missing', False):
            print(f"  🔨 translated {category} translatedFSMtranslated，translated...")
            try:
                from baseclass.Enhanced_FSM_Gen import EnhancedFSMGenerator
                if hasattr(self, 'fsm_generator') and self.fsm_generator is not None:
                    fsm_generator = self.fsm_generator
                else:
                    fsm_generator = EnhancedFSMGenerator(use_azure=False)
                
                mas_config, cost = fsm_generator.generate_complete_mas(
                    dataset='mmlu',
                    task_description=f"MMLU {category} category questions",
                    save_path=None
                )
                
                if hasattr(self, 'fsm_cache_manager') and self.fsm_cache_manager:
                    self.fsm_cache_manager.save_fsm(
                        dataset='mmlu',
                        category=category,
                        fsm_config=mas_config.get('fsm', {}),
                        agents=mas_config.get('agents', []),
                        metadata={'generation_cost': cost, 'category': category, 'auto_generated': True}
                    )
                else:
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
                
                cached_format = {
                    'fsm_config': mas_config.get('fsm', {}),
                    'agents': mas_config.get('agents', []),
                    'metadata': {'generation_cost': cost, 'category': category, 'auto_generated': True}
                }
                topology, fsm_manager, fsm_tgn = self._create_fsm_from_cache(cached_format, domain)
                
                if not hasattr(self, 'mmlu_category_fsms'):
                    self.mmlu_category_fsms = {}
                self.mmlu_category_fsms[category] = (topology, fsm_manager, fsm_tgn)
                print(f"  ✅ translated {category} translatedFSMtranslated (translated: ${cost:.4f})")
                return topology, fsm_manager, fsm_tgn
            except Exception as e:
                print(f"  ⚠️  translated: {e}")
                print(f"  ⚠️  translatedFSM")
        
        print(f"  ⚠️  translated {category} translatedFSMtranslated，translatedFSM")
        if not hasattr(self, 'mmlu_category_fsms'):
            self.mmlu_category_fsms = {}
        
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
        from neural_fsm_mas.training_data.answer_validator import create_answer_validator
        
        validator = create_answer_validator()
        is_correct, details = validator.validate(
            prediction=prediction,
            ground_truth=ground_truth,
            domain=domain,
            **kwargs
        )
        if not is_correct and domain in ("humaneval", "mbpp"):
            error_msg = details.get("error")
            if error_msg:
                print(f"  ⚠️ translated（{domain}）: {str(error_msg)[:300]}")
        return is_correct
    
    def _save_best_model(self, domain: str, fsm_tgn: FSMTemporalGraph, epoch: int, accuracy: float):
        model_dir = self.output_dir / "best_models" / domain
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"fsm_tgn_epoch_{epoch}_acc_{accuracy:.4f}.pt"
        torch.save({
            'epoch': epoch,
            'accuracy': accuracy,
            'model_state_dict': fsm_tgn.state_dict(),
            'config': self.config
        }, model_path)
        
        print(f"  💾 translated {model_path}")
    
    async def train_all_domains(self, domains: List[str] = None) -> Dict[str, Dict]:
        if domains is None:
            domains = ['mmlu', 'gsm8k', 'humaneval']
        
        all_results = {}
        for domain in domains:
            results = await self.train_domain(domain, num_epochs=self.config.get('num_epochs', 10))
            all_results[domain] = results
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        return all_results
    
    def compute_four_objective_loss(self,
                                    policy_loss: torch.Tensor,
                                    transition_loss: torch.Tensor,
                                    listener_loss: torch.Tensor,
                                    episode_costs: List[float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        cost_tensor = torch.tensor(episode_costs, dtype=torch.float32, device=self.device)
        
        total_loss, loss_dict = self.cost_loss_fn(
            policy_loss,
            transition_loss,
            listener_loss,
            cost_tensor
        )
        
        return total_loss, loss_dict


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FSM MAS Training V2")
    
    parser.add_argument("--domains", type=str, nargs='+', default=['gsm8k'],
                       help="Training domains")
    parser.add_argument("--dataset_root", type=str, default="./datasets",
                       help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./fsm_mas_v2_outputs",
                       help="Output directory")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    
    # Loss weights
    parser.add_argument("--policy_gradient_weight", type=float, default=1.0,
                       help="Policy gradient loss weight (alpha)")
    parser.add_argument("--transition_loss_weight", type=float, default=0.3,
                       help="State transition loss weight (beta)")
    parser.add_argument("--listener_loss_weight", type=float, default=0.2,
                       help="Listener-path loss weight (gamma)")
    
    # Model parameters
    parser.add_argument("--memory_dim", type=int, default=128,
                       help="TGN memory dimension")
    parser.add_argument("--time_dim", type=int, default=32,
                       help="Time encoding dimension")
    parser.add_argument("--llm_name", type=str, default="gpt-5-nano",
                       help="LLM model name")
    
    return parser.parse_args()


async def main():
    """Main entry point."""
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
    print("🚀 FSM multi-agent system training V2")
    print("="*80)
    print("\n✅ New FSM architecture features:")
    print("  1. One state corresponds to one agent")
    print("  2. Jointly optimize state transition probabilities")
    print("  3. Jointly optimize listener communication paths")
    print(f"\n📊 Configuration: {json.dumps(config, indent=2)}")
    print("="*80 + "\n")
    
    trainer = FSMMultiAgentSystemTrainerV2(
        config=config,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir
    )
    
    try:
        results = await trainer.train_all_domains(args.domains)
        
        print("\n" + "="*80)
        print("🎉 Training completed!")
        print("="*80)
        
        for domain, result in results.items():
            print(f"\n📊 {domain}:")
            print(f"  Best accuracy: {result['best_accuracy']:.4f} (Epoch {result['best_epoch']})")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import sys
import os
from pathlib import Path

# Translated comment
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
    """Translated class documentation."""

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
        
                # Translated comment
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
        
                # Translated comment
        fixed_spatial_connection_masks = torch.tensor(fixed_spatial_connection_masks).view(-1)
        fixed_temporal_connection_masks = torch.tensor(fixed_temporal_connection_masks).view(-1)
        
                # Translated comment
        expected_mask_size = len(agent_role_names) * len(agent_role_names)
        assert len(fixed_spatial_connection_masks) == expected_mask_size,\
            f"Spatial connection masks size mismatch: expected {expected_mask_size}, got {len(fixed_spatial_connection_masks)}"
        assert len(fixed_temporal_connection_masks) == expected_mask_size,\
            f"Temporal connection masks size mismatch: expected {expected_mask_size}, got {len(fixed_temporal_connection_masks)}"
        
                # Translated comment
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
        
                # Translated comment
        self.decision_executor: AgentExecutionNode = ReasoningAgentFactory.create_agent(
            decision_strategy, 
            domain=self.task_domain, 
            llm_name=self.language_model_name
        )
        self.agent_execution_nodes: Dict[str, AgentExecutionNode] = {}
        
                # Translated comment
        self.fsm_state_manager: Optional[FSMStateManager] = None
        self.use_fsm_mode: bool = False          # Translated comment
        self.potential_spatial_connections: List[List[str, str]] = []
        self.potential_temporal_connections: List[List[str, str]] = []
        self.agent_configuration_params = agent_configuration_params if agent_configuration_params is not None else [{} for _ in agent_role_names]
        
                # Translated comment
        self._initialize_agent_nodes()
        self._initialize_potential_connections()
        
                # Translated comment
        self.domain_prompt_manager = DomainPromptManager.get_manager(task_domain)
        self.role_adjacency_matrix = self._construct_role_adjacency_matrix()
        self.agent_features = self._construct_agent_features()
        
                # Translated comment
        self._initialize_neural_networks()
        
                # Translated comment
        self._initialize_connection_parameters(
            initial_spatial_connection_prob, 
            fixed_spatial_connection_masks,
            initial_temporal_connection_prob,
            fixed_temporal_connection_masks
        )
    
    def _initialize_agent_nodes(self):
        """Translated function documentation."""
        for i, agent_role in enumerate(self.agent_role_names):
            agent_config = self.agent_configuration_params[i]
                        # Translated comment
            agent_node = ConcreteAgentExecutionNode(
                node_id=f"agent_{i}",
                agent_role=agent_role,
                domain=self.task_domain,
                llm_name=self.language_model_name,
                **agent_config
            )
                        # Translated comment
            agent_node._use_fsm_mode = self.use_fsm_mode
            self.agent_execution_nodes[agent_node.node_id] = agent_node
    
    def _initialize_potential_connections(self):
        """Translated function documentation."""
        agent_ids = list(self.agent_execution_nodes.keys())
        
                # Translated comment
        for source_id in agent_ids:
            for target_id in agent_ids:
                if source_id != target_id:
                    self.potential_spatial_connections.append([source_id, target_id])
        
                # Translated comment
        for source_id in agent_ids:
            for target_id in agent_ids:
                self.potential_temporal_connections.append([source_id, target_id])
    
    def _construct_role_adjacency_matrix(self):
        """Translated function documentation."""
        role_connections: List[Tuple[str, str]] = self.domain_prompt_manager.get_role_connections()
        num_agents = self.num_agents
        role_adjacency = torch.zeros((num_agents, num_agents))
        role_to_indices = {}
        
                # Translated comment
        for connection in role_connections:
            input_role, output_role = connection
            role_to_indices[input_role] = []
            role_to_indices[output_role] = []
        
        for i, agent_id in enumerate(self.agent_execution_nodes):
            agent_role = self.agent_execution_nodes[agent_id].agent_role
            if agent_role in role_to_indices:
                role_to_indices[agent_role].append(i)
            
                # Translated comment
        for connection in role_connections:
            input_role, output_role = connection
            input_indices = role_to_indices.get(input_role, [])
            output_indices = role_to_indices.get(output_role, [])
            
            for input_idx in input_indices:
                for output_idx in output_indices:
                    role_adjacency[input_idx][output_idx] = 1
        
                # Translated comment
        edge_index, edge_weights = dense_to_sparse(role_adjacency)
        return edge_index
    
    def _construct_agent_features(self):
        """Translated function documentation."""
        agent_features = []
        for agent_id in self.agent_execution_nodes:
            agent_role = self.agent_execution_nodes[agent_id].agent_role
            role_description = self.domain_prompt_manager.get_role_description(agent_role)
            
                        # Translated comment
            feature_vector = self._get_text_embedding(role_description)
            agent_features.append(feature_vector)
        
        return torch.tensor(np.array(agent_features))
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Translated function documentation."""
                # Translated comment
        return np.random.randn(384)          # Translated comment
    
    def _initialize_neural_networks(self):
        """Translated function documentation."""
        if self.use_neural_temporal_graph:
                        # Translated comment
            self.neural_temporal_graph = NeuralTemporalGraph(
                agent_feature_dim=self.agent_features.size(1),
                memory_dimension=self.memory_bank_dimension,
                temporal_dimension=self.temporal_encoding_dimension,
                agent_count=len(self.agent_execution_nodes),
                network_layers=2
            )
            
                        # Translated comment
            self.compatibility_graph_network = CompatibilityGraphNetwork(
                self.agent_features.size(1) * 2, 16, self.agent_features.size(1)
            )
        else:
                        # Translated comment
            self.compatibility_graph_network = CompatibilityGraphNetwork(
                self.agent_features.size(1) * 2, 16, self.agent_features.size(1)
            )
            self.neural_temporal_graph = None
            
                # Translated comment
        self.decision_decoder = MultiLayerPerceptron(384, 16, 16)
    
    def _initialize_connection_parameters(self, 
                                        initial_spatial_prob: float,
                                        fixed_spatial_masks: torch.Tensor,
                                        initial_temporal_prob: float,
                                        fixed_temporal_masks: torch.Tensor):
        """Translated function documentation."""
                # Translated comment
        if self.enable_spatial_optimization:
            initial_spatial_logit = torch.log(torch.tensor(initial_spatial_prob / (1 - initial_spatial_prob)))
        else:
            initial_spatial_logit = 10.0
            
        self.spatial_connection_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_spatial_connections), requires_grad=self.enable_spatial_optimization) * initial_spatial_logit,
            requires_grad=self.enable_spatial_optimization
        )
        self.spatial_connection_masks = torch.nn.Parameter(fixed_spatial_masks, requires_grad=False)

                # Translated comment
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
        """Translated function documentation."""
        query_embedding = torch.tensor(self._get_text_embedding(task_query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_agents, 1))
        enhanced_features = torch.cat((self.agent_features, query_embedding), dim=1)
        return enhanced_features
        
    @property
    def spatial_adjacency_matrix(self):
        """Translated function documentation."""
        matrix = np.zeros((len(self.agent_execution_nodes), len(self.agent_execution_nodes)))
        for i, agent1_id in enumerate(self.agent_execution_nodes):
            for j, agent2_id in enumerate(self.agent_execution_nodes):
                if self.agent_execution_nodes[agent2_id] in self.agent_execution_nodes[agent1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adjacency_matrix(self):
        """Translated function documentation."""
        matrix = np.zeros((len(self.agent_execution_nodes), len(self.agent_execution_nodes)))
        for i, agent1_id in enumerate(self.agent_execution_nodes):
            for j, agent2_id in enumerate(self.agent_execution_nodes):
                if self.agent_execution_nodes[agent2_id] in self.agent_execution_nodes[agent1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_agents(self):
        """Translated function documentation."""
        return len(self.agent_execution_nodes)

    def construct_spatial_connections(self, sampling_temperature: float = 1.0, 
                                    connection_threshold: float = None) -> torch.Tensor:
        """Translated function documentation."""
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
            
                        # Translated comment
            connection_probability = torch.sigmoid(edge_logit / sampling_temperature)
            if connection_threshold:
                connection_probability = torch.tensor(1 if connection_probability > connection_threshold else 0)
                
                        # Translated comment
            if torch.rand(1) < connection_probability:
                source_agent.add_successor_connection(target_agent, 'spatial')
                connection_log_probs.append(torch.log(connection_probability))
            else:
                connection_log_probs.append(torch.log(1 - connection_probability))
                    
        return torch.sum(torch.stack(connection_log_probs))
    
    def construct_temporal_connections(self, interaction_round: int = 0, 
                                     sampling_temperature: float = 1.0, 
                                     connection_threshold: float = None) -> torch.Tensor:
        """Translated function documentation."""
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
            
                        # Translated comment
            connection_probability = torch.sigmoid(edge_logit / sampling_temperature)
            if connection_threshold:
                connection_probability = torch.tensor(1 if connection_probability > connection_threshold else 0)
                
                        # Translated comment
            if torch.rand(1) < connection_probability:
                source_agent.add_successor_connection(target_agent, 'temporal')
                connection_log_probs.append(torch.log(connection_probability))
            else:
                connection_log_probs.append(torch.log(1 - connection_probability))
                    
        return torch.sum(torch.stack(connection_log_probs))

    def clear_spatial_connections(self):
        """Translated function documentation."""
        for agent_node in self.agent_execution_nodes.values():
            agent_node.clear_spatial_connections()

    def clear_temporal_connections(self):
        """Translated function documentation."""
        for agent_node in self.agent_execution_nodes.values():
            agent_node.clear_temporal_connections()

    def find_agent_node(self, agent_id: str) -> AgentExecutionNode:
        """Translated function documentation."""
        return self.agent_execution_nodes.get(agent_id)

    def _check_connection_cycle(self, new_agent: AgentExecutionNode, target_agents: set) -> bool:
        """Translated function documentation."""
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
        """Translated function documentation."""
                # Translated comment
        reasoning_log_probs = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        enhanced_features = self.construct_enhanced_agent_features(task_input['task'])
        
        if self.use_neural_temporal_graph and self.neural_temporal_graph is not None:
                        # Translated comment
            temporal_stamps = torch.tensor([self.current_interaction_round] * len(self.agent_execution_nodes), dtype=torch.float)
            agent_indices = torch.arange(len(self.agent_execution_nodes))
            
                        # Translated comment
            reasoning_logits = self.neural_temporal_graph(
                enhanced_features, 
                self.role_adjacency_matrix, 
                temporal_stamps=temporal_stamps, 
                agent_indices=agent_indices
            )
        else:
                        # Translated comment
            reasoning_logits = self.compatibility_graph_network(enhanced_features, self.role_adjacency_matrix)
            
                # Translated comment
        reasoning_logits = self.decision_decoder(reasoning_logits)
        self.spatial_connection_logits = reasoning_logits @ reasoning_logits.t()
        self.spatial_connection_logits = self._min_max_normalize(torch.flatten(self.spatial_connection_logits))

                # Translated comment
        for interaction_round in range(num_interaction_rounds):
            self.current_interaction_round = interaction_round
            reasoning_log_probs += self.construct_spatial_connections()
            reasoning_log_probs += self.construct_temporal_connections(interaction_round)
            
                        # Translated comment
            agent_in_degrees = {agent_id: len(agent.spatial_predecessors) for agent_id, agent in self.agent_execution_nodes.items()}
            execution_queue = [agent_id for agent_id, degree in agent_in_degrees.items() if degree == 0]

                        # Translated comment
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
                
                                # Translated comment
                for successor_agent in self.agent_execution_nodes[current_agent_id].spatial_successors:
                    if successor_agent.node_id not in self.agent_execution_nodes.keys():
                        continue
                    agent_in_degrees[successor_agent.node_id] -= 1
                    if agent_in_degrees[successor_agent.node_id] == 0:
                        execution_queue.append(successor_agent.node_id)
            
                        # Translated comment
            self.update_agent_memories()
            
                # Translated comment
        self.connect_to_decision_executor()
        await self.decision_executor.async_execute_reasoning(task_input)
        
        final_reasoning_results = self.decision_executor.execution_outputs
        if len(final_reasoning_results) == 0:
            final_reasoning_results.append("No reasoning result from decision executor")
            
        return final_reasoning_results, reasoning_log_probs
    
    def update_agent_memories(self):
        """Translated function documentation."""
                # Translated comment
        for agent_id, agent_node in self.agent_execution_nodes.items():
            agent_node.set_current_interaction_round(self.current_interaction_round)
            agent_node.update_interaction_memory()
        
                # Translated comment
        if self.use_neural_temporal_graph and self.neural_temporal_graph is not None:
                        # Translated comment
            pass
    
    def connect_to_decision_executor(self):
        """Translated function documentation."""
        for agent_node in self.agent_execution_nodes.values():
            self.decision_executor.add_predecessor_connection(agent_node, 'spatial')
    
    def reset_neural_temporal_memories(self):
        """Translated function documentation."""
        if self.use_neural_temporal_graph and self.neural_temporal_graph is not None:
            self.neural_temporal_graph.reset_agent_memories()
            self.current_interaction_round = 0
    
    def get_neural_memory_snapshot(self):
        """Translated function documentation."""
        if self.use_neural_temporal_graph and self.neural_temporal_graph is not None:
            return self.neural_temporal_graph.get_memory_snapshot()
        return None
    
    def optimize_connection_topology(self, pruning_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Translated function documentation."""
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
        """Translated function documentation."""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val == 0:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)
    
    # ========================================================================
        # Translated comment
    # ========================================================================
    
    def initialize_fsm_from_description(self, fsm_description: Dict[str, Any]):
        """Translated function documentation."""
        agent_ids = list(self.agent_execution_nodes.keys())
        self.fsm_state_manager = FSMStateManager(agent_ids)
        
                # Translated comment
        for state_desc in fsm_description.get('states', []):
            self.fsm_state_manager.add_state(
                state_id=state_desc['id'],
                state_name=state_desc['name'],
                responsible_agent_id=state_desc['agent'],
                is_initial=state_desc.get('is_initial', False),
                is_final=state_desc.get('is_final', False),
                description=state_desc.get('description', '')
            )
        
                # Translated comment
        listeners_dict = fsm_description.get('listeners', {})
        for state_id, listener_ids in listeners_dict.items():
            self.fsm_state_manager.set_listeners(int(state_id), listener_ids)
        
                # Translated comment
        self.use_fsm_mode = True
        
        print(f"✅ FSM initialized with {len(self.fsm_state_manager.states)} states")
        print(self.fsm_state_manager.get_state_summary())
    
    async def execute_fsm_reasoning(self,
                                    task_input: Dict[str, str],
                                    max_transitions: int = 10,
                                    max_retry_attempts: int = 3,
                                    max_execution_time: int = 600,
                                    fsm_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[str, torch.Tensor]:
        """Translated function documentation."""
        if not self.use_fsm_mode or self.fsm_state_manager is None:
            raise ValueError("FSM mode is not enabled. Call initialize_fsm_from_description() first.")
        
                # Translated comment
        self.fsm_state_manager.reset()
        
                # Translated comment
        reasoning_log_probs = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        
                # Translated comment
        transition_count = 0
        last_state_output: Optional[str] = None          # Translated comment
        
        while transition_count < max_transitions:
            current_state = self.fsm_state_manager.get_current_state()
            
            if current_state is None:
                return "Error: No current state", reasoning_log_probs
            
                        # Translated comment
            agent_node = self.agent_execution_nodes.get(current_state.responsible_agent_id)
            
                        # Translated comment
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
            
                        # Translated comment
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
                    
                                        # Translated comment
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
            
                        # Translated comment
            if current_state.is_final:
                final_answer = self.fsm_state_manager.check_final_answer(output)
                if final_answer:
                    print(f"\n✅ Final Answer: {final_answer}")
                    return final_answer, reasoning_log_probs
                else:
                                        # Translated comment
                    print(f"\n✅ Reached final state, returning output")
                    return output, reasoning_log_probs
            
                        # Translated comment
            next_state_id = self.fsm_state_manager.extract_state_transition(output)
            
                        # Translated comment
            if next_state_id is None:
                print(f"\n⚠️  Failed to extract a state-transition tag from the agent output (<STATE_TRANS>: X)")
                print(f"   💡 Will try sampling from the TGN-predicted state-transition probabilities...")
                
                                # Translated comment
                if fsm_outputs and 'transition_probs' in fsm_outputs:
                    transition_probs = fsm_outputs['transition_probs']
                    if transition_probs is not None and len(transition_probs) > 0:
                                                # Translated comment
                        import torch.nn.functional as F
                        if isinstance(transition_probs, torch.Tensor):
                                                        # Translated comment
                            if transition_probs.dim() == 1:
                                                                # Translated comment
                                sampled_idx = torch.multinomial(transition_probs, num_samples=1).item()
                                next_state_id = sampled_idx
                                print(f"   ✅ Sampled using TGN-predicted state-transition probabilities")
                                print(f"   📊 Sampled result: State {next_state_id} (probability: {transition_probs[sampled_idx]:.4f})")
                            else:
                                                                # Translated comment
                                if current_state.state_id < transition_probs.size(0):
                                    state_probs = transition_probs[current_state.state_id]
                                    sampled_idx = torch.multinomial(state_probs, num_samples=1).item()
                                    next_state_id = sampled_idx
                                    print(f"   ✅ Sampled using TGN-predicted state-transition probabilities")
                                    print(f"   📊 Sampled result: State {next_state_id} (probability: {state_probs[sampled_idx]:.4f})")
                                else:
                                    print(f"   ❌ TGN-predicted state-transition probability matrix has mismatched dimensions")
                    else:
                        print(f"   ❌ TGN-predicted state-transition probabilities are empty or invalid")
                else:
                    print(f"   ❌ Could not find TGN-predicted state-transition probabilities ('transition_probs' missing in fsm_outputs)")
                
                                # Translated comment
                if next_state_id is None:
                    print(f"\n   ⚠️  Could not use TGN-predicted state-transition probabilities")
                    print(f"   💡 Falling back to the next sequential state...")
                    
                                        # Translated comment
                    all_states = list(self.fsm_state_manager.states.keys())
                    current_idx = all_states.index(current_state.state_id) if current_state.state_id in all_states else -1
                    if current_idx >= 0 and current_idx < len(all_states) - 1:
                        next_state_id = all_states[current_idx + 1]
                        print(f"   ✅ Fallback strategy used: State {current_state.state_id} -> State {next_state_id}")
                    else:
                        print(f"   ❌ Fallback strategy failed: unable to find the next sequential state")
                        print(f"   ❌ State transition failed; terminating reasoning")
                return "Error: No state transition found", reasoning_log_probs
            else:
                print(f"\n✅ Successfully extracted a state-transition tag from the agent output")
                print(f"   📋 Extracted result: State {current_state.state_id} -> State {next_state_id}")
            
            print(f"\n→ Transitioning to State {next_state_id}")
            
                        # Translated comment
            listeners = self.fsm_state_manager.get_listeners(current_state.state_id)
            
            if listeners:
                print(f"📡 Broadcasting to listeners: {listeners}")
                
                for listener_id in listeners:
                    listener_node = self.agent_execution_nodes.get(listener_id)
                    if listener_node:
                                                # Translated comment
                        message = f"\n[Message from {agent_node.agent_role} at State {current_state.state_id}]:\n{output}\n"
                                                # Translated comment
                                                # Translated comment
                        listener_node.add_predecessor_message(message)
            
                        # Translated comment
            success = self.fsm_state_manager.transition_to_state(next_state_id)
            
            if not success:
                return f"Error: Failed to transition to state {next_state_id}", reasoning_log_probs
            
            transition_count += 1
        
                # Translated comment
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
        """Translated function documentation."""
        if not self.use_fsm_mode or self.fsm_state_manager is None:
            raise ValueError("FSM mode is not enabled. Call initialize_fsm_from_description() first.")
        
                # Translated comment
        fsm_manager.reset()
        
                # Translated comment
        collected_agent_messages = {}
        
                # Translated comment
                # Translated comment
        sampled_communication_edges = []

                # Translated comment
        for agent_node in self.agent_execution_nodes.values():
                        # Translated comment
            try:
                agent_node.clear_predecessor_messages()
            except Exception:
                pass
                        # Translated comment
            try:
                agent_node.reset_interaction_memory()
            except Exception:
                pass
                        # Translated comment
            agent_node.execution_inputs = []
            agent_node.execution_outputs = []
            agent_node.raw_task_inputs = []
                        # Translated comment
            reasoning_llm = getattr(agent_node, "reasoning_llm", None)
            reasoning_prompt = getattr(agent_node, "reasoning_prompt", None)
            if reasoning_llm is not None and reasoning_prompt is not None:
                try:
                    reasoning_llm.messages = [{"role": "system", "content": reasoning_prompt}]
                except Exception:
                    pass
        
                # Translated comment
        reasoning_log_probs = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        
                # Translated comment
        reached_max_transitions = False
        
                # Translated comment
        listener_weights = None
        if enable_comm_sampling:
            listener_weights = fsm_outputs.get("listener_weights")  # [num_states, num_agents]
            if listener_weights is None:
                raise ValueError("listener_weights not found in fsm_outputs")
        
                # Translated comment
        if enable_transition_prediction:
                        # Translated comment
                        # Translated comment
            use_dynamic_transition = (
                fsm_tgn is not None
                and question_embedding is not None
                and agent_features is not None
                and context_features is not None
            )
            
                        # Translated comment
            transition_probs = fsm_outputs.get("transition_probs")  # [num_states] or [num_states, num_states]
            if transition_probs is None:
                raise ValueError("transition_probs not found in fsm_outputs and cannot compute dynamically")
        else:
            use_dynamic_transition = False
            transition_probs = None
        
                # Translated comment
        transition_count = 0
        last_state_output = None          # Translated comment
        
        while transition_count < max_transitions:
            current_state = fsm_manager.get_current_state()
            
            if current_state is None:
                return "Error: No current state", reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
                        # Translated comment
            agent_node = self.agent_execution_nodes.get(current_state.responsible_agent_id)

                        # Translated comment
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
            
                        # Translated comment
            retry_attempts = 0
            output = None
            
            while retry_attempts < max_retry_attempts:
                try:
                                        # Translated comment
                                        # Translated comment
                    if last_state_output is not None:
                                                # Translated comment
                                                # Translated comment
                        if current_state.is_final:
                                                        # Translated comment
                            truncated_prev_output = str(last_state_output)
                        else:
                                                        # Translated comment
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
                    
                                        # Translated comment
                    if agent_node.execution_outputs:
                        output = agent_node.execution_outputs[-1]
                    break
                
                except Exception as e:
                    print(f"⚠️  Agent execution error (attempt {retry_attempts + 1}): {e}")
                    retry_attempts += 1
            
            if output is None:
                return "Error: Agent execution failed", reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
                        # Translated comment
                        # Translated comment
            if attack_injector is not None and hasattr(attack_injector, 'pollute_message'):
                                # Translated comment
                agent_id_str = current_state.responsible_agent_id
                try:
                    agent_idx = int(agent_id_str.split('_')[-1]) if '_' in agent_id_str else int(agent_id_str)
                except (ValueError, IndexError):
                    agent_idx = 0
                
                                # Translated comment
                if attack_injector.should_pollute_message(agent_idx):
                    original_output = output
                    output = attack_injector.pollute_message(
                        agent_id=agent_idx,
                        original_message=output,
                        pollution_type=attack_injector.attack_type
                    )
                    if verbose:
                        print(f"  🔴 Message was polluted (Agent {agent_idx}, attack type: {attack_injector.attack_type})")
            
                        # Translated comment
            last_state_output = output
            
                        # Translated comment
            try:
                agent_id_str = current_state.responsible_agent_id
                agent_idx = int(agent_id_str.split('_')[-1]) if '_' in agent_id_str else int(agent_id_str)
                collected_agent_messages[agent_idx] = output
            except (ValueError, IndexError):
                pass

            if verbose:
                print(f"\n🤖 Agent Output:\n{output[:200]}...")
            
                        # Translated comment
            if current_state.is_final:
                                # Translated comment
                collected_agent_messages['_communication_edges'] = sampled_communication_edges
                
                final_answer = fsm_manager.check_final_answer(output)
                if final_answer:
                    if verbose:
                        print(f"\n✅ Final Answer: {final_answer}")
                    return final_answer, reasoning_log_probs, reached_max_transitions, collected_agent_messages
                else:
                                        # Translated comment
                    if verbose:
                        print(f"\n✅ Reached final state, returning output")
                    return output, reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
                        # Translated comment
            current_state_id = current_state.state_id

                        # Translated comment
            if enable_transition_prediction:
                                # Translated comment
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

                                # Translated comment
                if transition_probs.dim() == 1:
                    state_transition_probs = transition_probs
                elif transition_probs.dim() == 2:
                    state_transition_probs = transition_probs[current_state_id]
                else:
                    raise ValueError(f"Unexpected transition_probs shape: {transition_probs.shape}")

                                # Translated comment
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
                                print("\n⚖️  Initial-state constraint: temporarily masked transitions that jump directly to the final state")
                        else:
                            if verbose:
                                print("\n⚠️  Initial-state constraint made all probabilities zero; falling back to the original distribution")

                                # Translated comment
                if temperature != 1.0:
                    logits = torch.log(state_transition_probs + 1e-8) / temperature
                    state_transition_probs = torch.nn.functional.softmax(logits, dim=-1)

                                # Translated comment
                probs = state_transition_probs.clone()
                if current_state_id < probs.size(0):
                    probs[current_state_id] = 0.0
                if probs.sum() <= 0:
                    probs = state_transition_probs

                sampled_next_state_id = torch.multinomial(probs.unsqueeze(0), num_samples=1).item()

                                # Translated comment
                sampled_prob = probs[sampled_next_state_id]
                sampled_log_prob = torch.log(sampled_prob + 1e-8)
                reasoning_log_probs = reasoning_log_probs + sampled_log_prob
            else:
                                # Translated comment
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
                    print("\n📊 Ablation: transition prediction disabled; using fixed state progression")
                    print(f"\n-> Fixed transition: State {current_state_id} -> State {sampled_next_state_id}")

                        # Translated comment
            current_state_name = current_state.state_name if current_state else f"State_{current_state_id}"
            sampled_state_obj = fsm_manager.states.get(sampled_next_state_id)
            sampled_state_name = sampled_state_obj.state_name if sampled_state_obj else f"State_{sampled_next_state_id}"

            if phase == "val":
                log_prefix = "✅ Validation"
            elif phase == "test":
                log_prefix = "🧪 Test"
            else:
                log_prefix = "🎯 Train"

            if verbose:
                if enable_transition_prediction:
                    print("\n📊 Sampling with TGN-predicted state-transition probabilities")
                    print(f"   📈 Transition probability distribution for current state {current_state_id} ({current_state_name}):")
                    for state_id, prob in enumerate(state_transition_probs):
                        marker = "👉" if state_id == sampled_next_state_id else "  "
                        state_obj = fsm_manager.states.get(state_id)
                        state_name = state_obj.state_name if state_obj else f"State_{state_id}"
                        print(f"   {marker} State {state_id} ({state_name}): {prob:.4f}")
                    print(
                        f"\n-> Sampled result: State {current_state_id} ({current_state_name}) "
                        f"→ State {sampled_next_state_id} ({sampled_state_name}) "
                        f"(probability: {state_transition_probs[sampled_next_state_id]:.4f})"
                    )
                else:
                    print(
                        f"\n-> Fixed transition: State {current_state_id} ({current_state_name}) "
                        f"→ State {sampled_next_state_id} ({sampled_state_name})"
                    )
            else:
                print(
                    f"{log_prefix} State transition: State {current_state_id} ({current_state_name}) "
                    f"→ State {sampled_next_state_id} ({sampled_state_name})"
                )

                        # Translated comment
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
                    print("   📡 Ablation: communication sampling disabled; listener broadcast turned off (0 listeners)")
            
                        # Translated comment
                        # Translated comment
            fsm_manager.set_listeners(current_state_id, sampled_listeners)
            
            if sampled_listeners:
                                # Translated comment
                listener_display = []
                for listener_id in sampled_listeners:
                    listener_node = self.agent_execution_nodes.get(listener_id)
                    if listener_node:
                        listener_display.append(f"{listener_id} ({listener_node.agent_role})")
                    else:
                        listener_display.append(listener_id)
                
                                # Translated comment
                if verbose or phase in ["val", "test"]:
                    print(f"   📡 Listeners: {listener_display}")
                
                                # Translated comment
                                # Translated comment
                current_agent_id = current_state.responsible_agent_id
                for listener_id in sampled_listeners:
                    if listener_id != current_agent_id:                      # Translated comment
                        sampled_communication_edges.append((current_agent_id, listener_id))
                
                for listener_id in sampled_listeners:
                    listener_node = self.agent_execution_nodes.get(listener_id)
                    if listener_node:
                                                # Translated comment
                        MAX_MESSAGE_LENGTH = 400                          # Translated comment
                        truncated_output = str(output)
                        if len(truncated_output) > MAX_MESSAGE_LENGTH:
                            truncated_output = truncated_output[:MAX_MESSAGE_LENGTH] + "...[truncated]"
                        message = f"\n[Message from {agent_node.agent_role} at State {current_state.state_id}]:\n{truncated_output}\n"
                        listener_node.add_predecessor_message(message)
            
                        # Translated comment
            success = fsm_manager.transition_to_state(sampled_next_state_id)
            
            if not success:
                sampled_state_obj = fsm_manager.states.get(sampled_next_state_id)
                sampled_state_name = sampled_state_obj.state_name if sampled_state_obj else f"State_{sampled_next_state_id}"
                if verbose:
                    print(f"⚠️  Unable to transition to state {sampled_next_state_id} ({sampled_state_name}); trying the highest-probability state instead")
                                # Translated comment
                                # Translated comment
                                # Translated comment
                if enable_transition_prediction:
                    fallback_state_id = torch.argmax(state_transition_probs).item()
                else:
                    all_state_ids = sorted(list(fsm_manager.states.keys()))
                    fallback_state_id = all_state_ids[-1] if all_state_ids else sampled_next_state_id
                success = fsm_manager.transition_to_state(fallback_state_id)
                if not success:
                    return f"Error: Failed to transition to state {sampled_next_state_id}", reasoning_log_probs, reached_max_transitions, collected_agent_messages
            
            transition_count += 1
        
                # Translated comment
                # Translated comment
        reached_max_transitions = True
        
        current_state = fsm_manager.get_current_state()
        current_state_name = current_state.state_name if current_state else "Unknown"
        if verbose:
            print(f"\n⚠️  Reached maximum transitions ({max_transitions})")
            print(f"  📍 Current position: State {current_state.state_id} ({current_state_name})")

                # Translated comment
        final_states = [s for s in fsm_manager.states.values() if s.is_final]
        if not final_states:
            if verbose:
                print("  ❌ No final state is defined; returning an error")
            return "Error: Maximum transitions reached and no final state defined", reasoning_log_probs, reached_max_transitions, collected_agent_messages
        
        final_state = final_states[0]
        print(f"  ⚠️  The final state was not reached within the step limit; forcing a jump to final state {final_state.state_id} ({final_state.state_name})")
        fsm_manager.transition_to_state(final_state.state_id)
        
                # Translated comment
        agent_node = self.agent_execution_nodes.get(final_state.responsible_agent_id)
        if agent_node is None:
            return f"Error: Final agent {final_state.responsible_agent_id} not found", reasoning_log_probs, reached_max_transitions, collected_agent_messages
        
        retry_attempts = 0
        output = None
        while retry_attempts < max_retry_attempts:
            try:
                                # Translated comment
                                # Translated comment
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
        
                # Translated comment
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
        """Translated function documentation."""
                # Translated comment
                # Translated comment
        
        agent_ids = list(self.agent_execution_nodes.keys())
        
                # Translated comment
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
            
                        # Translated comment
            if i < num_states - 1:
                fsm_description['listeners'][i] = [agent_ids[i + 1]]
            else:
                fsm_description['listeners'][i] = []
        
        return fsm_description


# Translated comment
__all__ = ['MultiAgentTopologyManager']

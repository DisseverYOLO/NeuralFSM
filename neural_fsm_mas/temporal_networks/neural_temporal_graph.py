
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from typing import Dict, List, Optional, Tuple
import numpy as np


class AgentMemoryBank(nn.Module):
    def __init__(self, memory_dimension: int, agent_count: int):
        super().__init__()
        self.memory_dimension = memory_dimension
        self.agent_count = agent_count
        
        self.agent_memories = nn.Parameter(torch.zeros(agent_count, memory_dimension), requires_grad=False)
        self.last_interaction_time = nn.Parameter(torch.zeros(agent_count), requires_grad=False)
        
        self.memory_evolution_cell = nn.GRUCell(memory_dimension, memory_dimension)
        
    def retrieve_agent_memory(self, agent_indices: torch.Tensor) -> torch.Tensor:
        return self.agent_memories[agent_indices]
    
    def evolve_agent_memory(self, agent_indices: torch.Tensor, 
                           interaction_messages: torch.Tensor, 
                           interaction_timestamps: torch.Tensor):
        with torch.no_grad():
            for idx, agent_id in enumerate(agent_indices):
                evolved_memory = self.memory_evolution_cell(
                    interaction_messages[idx].unsqueeze(0), 
                    self.agent_memories[agent_id].unsqueeze(0)
                ).squeeze(0)
                self.agent_memories[agent_id] = evolved_memory
                self.last_interaction_time[agent_id] = interaction_timestamps[idx]
    
    def reset_all_memories(self):
        self.agent_memories.data.zero_()
        self.last_interaction_time.data.zero_()


class TemporalEncoder(nn.Module):
    def __init__(self, temporal_dimension: int):
        super().__init__()
        self.temporal_dimension = temporal_dimension
        self.linear_projection = nn.Linear(1, temporal_dimension)
        
    def forward(self, temporal_stamps: torch.Tensor) -> torch.Tensor:
        temporal_stamps = temporal_stamps.float().unsqueeze(-1)
        
        temporal_features = torch.zeros(temporal_stamps.size(0), self.temporal_dimension, 
                                      device=temporal_stamps.device)
        
        frequency_components = torch.exp(torch.arange(0, self.temporal_dimension, 2, 
                                                    device=temporal_stamps.device).float() * 
                                       -(np.log(10000.0) / self.temporal_dimension))
        
        temporal_features[:, 0::2] = torch.sin(temporal_stamps * frequency_components)
        temporal_features[:, 1::2] = torch.cos(temporal_stamps * frequency_components)
        
        return temporal_features


class CommunicationAggregator(nn.Module):
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
        combined_features = torch.cat([source_agents, target_agents, communication_features], dim=-1)
        aggregated_messages = self.communication_network(combined_features)
        return aggregated_messages


class NeuralTemporalGraph(nn.Module):
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
        
        self.agent_memory_bank = AgentMemoryBank(memory_dimension, agent_count) if agent_count else None
        
        self.temporal_encoder = TemporalEncoder(temporal_dimension)
        
        self.communication_aggregator = CommunicationAggregator(
            agent_feature_dim + memory_dimension + temporal_dimension, 
            communication_edge_dim, 
            message_dimension
        )
        
        self.agent_state_updater = nn.GRUCell(message_dimension, 
                                             agent_feature_dim + memory_dimension + temporal_dimension)
        
        self.graph_neural_layers = nn.ModuleList([
            TransformerConv(agent_feature_dim + memory_dimension + temporal_dimension, 
                          agent_feature_dim + memory_dimension + temporal_dimension,
                          heads=4, concat=False, dropout=0.1)
            for _ in range(network_layers)
        ])
        
        self.feature_projector = nn.Linear(agent_feature_dim + memory_dimension + temporal_dimension, 
                                         agent_feature_dim)
        
        self.feature_dropout = nn.Dropout(0.1)
        self.layer_normalization = nn.LayerNorm(agent_feature_dim + memory_dimension + temporal_dimension)
        
    def forward(self, 
                agent_features: torch.Tensor,
                communication_topology: torch.Tensor,
                communication_attributes: Optional[torch.Tensor] = None,
                temporal_stamps: Optional[torch.Tensor] = None,
                agent_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_agents = agent_features.size(0)
        device = agent_features.device
        
        if temporal_stamps is None:
            temporal_stamps = torch.zeros(num_agents, device=device)
            
        if agent_indices is None:
            agent_indices = torch.arange(num_agents, device=device)
            
        if communication_attributes is None:
            communication_attributes = torch.zeros(communication_topology.size(1), 
                                                 self.communication_edge_dim, device=device)
        
        if self.agent_memory_bank is not None:
            agent_memories = self.agent_memory_bank.retrieve_agent_memory(agent_indices)
        else:
            agent_memories = torch.zeros(num_agents, self.memory_dimension, device=device)
        
        temporal_features = self.temporal_encoder(temporal_stamps)
        
        fused_agent_features = torch.cat([agent_features, agent_memories, temporal_features], dim=-1)
        
        for graph_layer in self.graph_neural_layers:
            residual_features = fused_agent_features
            fused_agent_features = graph_layer(fused_agent_features, communication_topology)
            fused_agent_features = F.relu(fused_agent_features)
            fused_agent_features = self.feature_dropout(fused_agent_features)
            
            fused_agent_features = self.layer_normalization(fused_agent_features + residual_features)
        
        evolved_agent_features = self.feature_projector(fused_agent_features)
        
        if self.agent_memory_bank is not None:
            source_agent_features = agent_features[communication_topology[0]]  # [num_edges, agent_feature_dim]
            target_agent_features = agent_features[communication_topology[1]]  # [num_edges, agent_feature_dim]
            communication_messages = self.communication_aggregator(source_agent_features, 
                                                                 target_agent_features, 
                                                                 communication_attributes)
            
            aggregated_agent_messages = torch.zeros(num_agents, self.message_dimension, device=device)
            for agent_id in range(num_agents):
                incoming_message_mask = communication_topology[1] == agent_id
                if incoming_message_mask.sum() > 0:
                    aggregated_agent_messages[agent_id] = communication_messages[incoming_message_mask].mean(dim=0)
            
            self.agent_memory_bank.evolve_agent_memory(agent_indices, aggregated_agent_messages, temporal_stamps)
        
        return evolved_agent_features
    
    def reset_agent_memories(self):
        if self.agent_memory_bank is not None:
            self.agent_memory_bank.reset_all_memories()
    
    def get_memory_snapshot(self) -> Optional[torch.Tensor]:
        if self.agent_memory_bank is not None:
            return self.agent_memory_bank.agent_memories.clone()
        return None
    
    def compute_communication_probabilities(self, agent_features: torch.Tensor) -> torch.Tensor:
        num_agents = agent_features.size(0)
        
        feature_similarities = torch.matmul(agent_features, agent_features.t())
        
        communication_probabilities = F.softmax(feature_similarities, dim=-1)
        
        mask = torch.eye(num_agents, device=agent_features.device).bool()
        communication_probabilities.masked_fill_(mask, 0.0)
        
        return communication_probabilities
    
    def sample_communication_topology(self, communication_probabilities: torch.Tensor, 
                                    sampling_temperature: float = 1.0) -> torch.Tensor:
        num_agents = communication_probabilities.size(0)
        
        adjusted_probabilities = communication_probabilities / sampling_temperature
        
        sampled_connections = torch.multinomial(adjusted_probabilities.view(-1), 
                                              num_samples=num_agents * 2, 
                                              replacement=True)
        
        source_agents = sampled_connections // num_agents
        target_agents = sampled_connections % num_agents
        
        valid_connections = source_agents != target_agents
        source_agents = source_agents[valid_connections]
        target_agents = target_agents[valid_connections]
        
        communication_topology = torch.stack([source_agents, target_agents], dim=0)
        
        return communication_topology


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, 
                 output_dimension: int, num_hidden_layers: int = 2):
        super().__init__()
        
        network_layers = []
        current_dimension = input_dimension
        
        for layer_idx in range(num_hidden_layers - 1):
            network_layers.extend([
                nn.Linear(current_dimension, hidden_dimension),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dimension)
            ])
            current_dimension = hidden_dimension
            
        network_layers.append(nn.Linear(current_dimension, output_dimension))
        
        self.decision_network = nn.Sequential(*network_layers)
        
    def forward(self, agent_features: torch.Tensor) -> torch.Tensor:
        return self.decision_network(agent_features)


class CompatibilityGraphNetwork(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int):
        super().__init__()
        
        self.neural_temporal_graph = NeuralTemporalGraph(
            agent_feature_dim=output_channels,
            memory_dimension=hidden_channels,
            agent_count=None,
            network_layers=2
        )
        
        self.input_feature_projector = nn.Linear(input_channels, output_channels)
        
    def reset_parameters(self):
        self.input_feature_projector.reset_parameters()
        self.neural_temporal_graph.reset_agent_memories()
        
    def forward(self, agent_features: torch.Tensor, communication_topology: torch.Tensor, 
                temporal_stamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        projected_features = self.input_feature_projector(agent_features)
        
        evolved_features = self.neural_temporal_graph(projected_features, communication_topology, 
                                                    temporal_stamps=temporal_stamps)
        
        return F.log_softmax(evolved_features, dim=1)


class StateTransitionLearner(nn.Module):
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
        encoded_states = self.state_encoder(state_features)
        action_logits = self.transition_predictor(encoded_states)
        state_values = self.value_estimator(encoded_states)
        
        return action_logits, state_values


__all__ = [
    'NeuralTemporalGraph', 
    'MultiLayerPerceptron', 
    'CompatibilityGraphNetwork',
    'StateTransitionLearner',
    'AgentMemoryBank',
    'TemporalEncoder',
    'CommunicationAggregator'
]

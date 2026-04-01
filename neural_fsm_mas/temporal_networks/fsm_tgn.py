
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph


class FSMTransitionPredictor(nn.Module):
    
    def __init__(self, 
                 state_feature_dim: int,
                 question_embedding_dim: int = 384,
                 hidden_dim: int = 128,
                 num_states: int = 4):
        super().__init__()
        
        self.num_states = num_states
        self.state_feature_dim = state_feature_dim
        self.question_embedding_dim = question_embedding_dim
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.question_encoder = nn.Sequential(
            nn.Linear(question_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
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
        state_encoded = self.state_encoder(current_state_features)
        context_encoded = self.state_encoder(context_features)
        
        if question_embedding is not None:
            if question_embedding.dim() == 1:
                question_embedding = question_embedding.unsqueeze(0)
            elif question_embedding.dim() > 2:
                question_embedding = question_embedding.view(-1, self.question_embedding_dim)[:1]
            
            if question_embedding.size(-1) != self.question_embedding_dim:
                if not hasattr(self, '_question_dim_adapter'):
                    self._question_dim_adapter = nn.Linear(question_embedding.size(-1), self.question_embedding_dim).to(question_embedding.device)
                question_embedding = self._question_dim_adapter(question_embedding)
            
            question_encoded = self.question_encoder(question_embedding)
            combined = torch.cat([state_encoded, context_encoded, question_encoded], dim=-1)
        else:
            combined = torch.cat([state_encoded, context_encoded], dim=-1)
            if not hasattr(self, '_no_question_adapter'):
                self._no_question_adapter = nn.Linear(hidden_dim * 2, hidden_dim * 3).to(combined.device)
            combined = self._no_question_adapter(combined)
        
        logits = self.transition_predictor(combined)
        
        transition_probs = F.softmax(logits, dim=-1)
        
        return transition_probs
    
    def sample_next_state(self,
                          transition_probs: torch.Tensor,
                          temperature: float = 1.0) -> torch.Tensor:
        if temperature != 1.0:
            logits = torch.log(transition_probs + 1e-8) / temperature
            transition_probs = F.softmax(logits, dim=-1)
        
        sampled_states = torch.multinomial(transition_probs, num_samples=1).squeeze(-1)
        return sampled_states
    
    def compute_transition_matrix(self,
                                  state_features: torch.Tensor,
                                  context_features: torch.Tensor,
                                  question_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_states = state_features.size(0)
        transition_matrix = torch.zeros(num_states, num_states, device=state_features.device)
        
        context_expanded = context_features.unsqueeze(0).expand(num_states, -1)
        
        if question_embedding is not None:
            question_expanded = question_embedding.unsqueeze(0).expand(num_states, -1)
        else:
            question_expanded = None
        
        for i in range(num_states):
            current_state = state_features[i].unsqueeze(0)
            q_emb = question_expanded[i].unsqueeze(0) if question_expanded is not None else None
            transition_probs = self.forward(current_state, context_expanded[i].unsqueeze(0), q_emb)
            transition_matrix[i] = transition_probs.squeeze(0)
        
        return transition_matrix


class FSMListenerPredictor(nn.Module):
    
    def __init__(self,
                 state_feature_dim: int,
                 agent_feature_dim: int,
                 question_embedding_dim: int = 384,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.state_feature_dim = state_feature_dim
        self.agent_feature_dim = agent_feature_dim
        self.question_embedding_dim = question_embedding_dim
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.question_encoder = nn.Sequential(
            nn.Linear(question_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attention_query = nn.Linear(hidden_dim * 2, hidden_dim)  # state + question
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self,
                state_features: torch.Tensor,
                agent_features: torch.Tensor,
                question_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_states = state_features.size(0)
        num_agents = agent_features.size(0)
        
        state_encoded = self.state_encoder(state_features)  # [num_states, hidden_dim]
        agent_encoded = self.agent_encoder(agent_features)  # [num_agents, hidden_dim]
        
        if question_embedding is not None:
            question_encoded = self.question_encoder(question_embedding)  # [hidden_dim]
            question_expanded = question_encoded.unsqueeze(0).expand(num_states, -1)  # [num_states, hidden_dim]
            state_question_combined = torch.cat([state_encoded, question_expanded], dim=-1)  # [num_states, hidden_dim*2]
            queries = self.attention_query(state_question_combined)  # [num_states, hidden_dim]
        else:
            if not hasattr(self, '_no_question_query'):
                self._no_question_query = nn.Linear(hidden_dim, hidden_dim).to(state_encoded.device)
            queries = self._no_question_query(state_encoded)
        
        keys = self.attention_key(agent_encoded)  # [num_agents, hidden_dim]
        
        attention_scores = torch.matmul(queries, keys.t())  # [num_states, num_agents]
        attention_scores = attention_scores / (queries.size(-1) ** 0.5)  # scaled dot-product
        
        listener_weights = F.softmax(attention_scores, dim=-1)
        
        return listener_weights
    
    def sample_listeners(self,
                        listener_weights: torch.Tensor,
                        temperature: float = 1.0,
                        num_samples: Optional[int] = None) -> torch.Tensor:
        num_states, num_agents = listener_weights.shape
        listener_mask = torch.zeros_like(listener_weights)
        
        if temperature != 1.0:
            logits = torch.log(listener_weights + 1e-8) / temperature
            listener_weights = F.softmax(logits, dim=-1)
        
        for state_idx in range(num_states):
            if num_samples is None:
                sampled_agents = torch.multinomial(listener_weights[state_idx], 
                                                   num_samples=num_agents, 
                                                   replacement=True)
                unique_agents = torch.unique(sampled_agents)
                listener_mask[state_idx, unique_agents] = 1.0
            else:
                sampled_agents = torch.multinomial(listener_weights[state_idx], 
                                                   num_samples=min(num_samples, num_agents), 
                                                   replacement=False)
                listener_mask[state_idx, sampled_agents] = 1.0
        
        return listener_mask
    
    def compute_binary_listener_mask(self,
                                     listener_weights: torch.Tensor,
                                     threshold: float = 0.1) -> torch.Tensor:
        return (listener_weights > threshold).float()


class FSMStateAgentMatcher(nn.Module):
    
    def __init__(self,
                 state_feature_dim: int,
                 agent_feature_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self,
                state_features: torch.Tensor,
                agent_features: torch.Tensor) -> torch.Tensor:
        num_states = state_features.size(0)
        num_agents = agent_features.size(0)
        
        state_encoded = self.state_encoder(state_features)  # [num_states, hidden_dim]
        agent_encoded = self.agent_encoder(agent_features)  # [num_agents, hidden_dim]
        
        compatibility_matrix = torch.zeros(num_states, num_agents, device=state_features.device)
        
        for i in range(num_states):
            for j in range(num_agents):
                paired_features = torch.cat([state_encoded[i], agent_encoded[j]], dim=-1)
                compatibility_matrix[i, j] = self.compatibility_scorer(paired_features.unsqueeze(0)).squeeze()
        
        return compatibility_matrix
    
    def get_optimal_assignment(self,
                              compatibility_matrix: torch.Tensor) -> List[int]:
        assignment = torch.argmax(compatibility_matrix, dim=-1).tolist()
        return assignment


class FSMTemporalGraph(nn.Module):
    
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
        
        self.use_base_tgn = use_base_tgn
        if use_base_tgn:
            self.base_tgn = NeuralTemporalGraph(
                agent_feature_dim=agent_feature_dim,
                **tgn_kwargs
            )
        else:
            self.base_tgn = None
        
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
        
        self.state_embeddings = nn.Parameter(torch.randn(num_states, state_feature_dim))
        
    def forward(self,
                agent_features: torch.Tensor,
                communication_topology: torch.Tensor,
                current_state_id: Optional[int] = None,
                context_features: Optional[torch.Tensor] = None,
                question_embedding: Optional[torch.Tensor] = None,
                **tgn_kwargs) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        protection_data = None
        
        if self.base_tgn is not None:
            from neural_fsm_mas.defense_mechanisms import ProtectedTGN
            is_protected_tgn = isinstance(self.base_tgn, ProtectedTGN)
            
            if is_protected_tgn:
                base_tgn_output = self.base_tgn(
                agent_features,
                communication_topology,
                **tgn_kwargs
            )
                if isinstance(base_tgn_output, dict):
                    evolved_agent_features = base_tgn_output.get('agent_features', agent_features)
                    protection_data = base_tgn_output.get('protection_data', None)
                else:
                    evolved_agent_features = base_tgn_output
            else:
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
        
        if protection_data is not None:
            outputs['protection_data'] = protection_data
        
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
            if context_features is not None:
                transition_matrix = self.transition_predictor.compute_transition_matrix(
                    self.state_embeddings,
                    context_features,
                    question_embedding
                )
                outputs['transition_matrix'] = transition_matrix
        
        listener_weights = self.listener_predictor(
            self.state_embeddings,
            evolved_agent_features,
            question_embedding
        )
        outputs['listener_weights'] = listener_weights
        
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
        with torch.no_grad():
            outputs = self.forward(
                agent_features,
                communication_topology=torch.zeros(2, 0, dtype=torch.long),
                context_features=context_features
            )
            
            compatibility = outputs['state_agent_compatibility']
            agent_assignment = self.state_agent_matcher.get_optimal_assignment(compatibility)
            
            listener_weights = outputs['listener_weights']
            listener_mask = self.listener_predictor.compute_binary_listener_mask(
                listener_weights,
                threshold=listener_threshold
            )
            
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
                
                listener_indices = torch.where(listener_mask[state_id] > 0)[0].tolist()
                fsm_structure['listeners'][state_id] = [agent_ids[idx] for idx in listener_indices]
            
            return fsm_structure
    
    def reset_agent_memories(self):
        if self.base_tgn is not None:
            self.base_tgn.reset_agent_memories()


__all__ = [
    'FSMTemporalGraph',
    'FSMTransitionPredictor',
    'FSMListenerPredictor',
    'FSMStateAgentMatcher'
]


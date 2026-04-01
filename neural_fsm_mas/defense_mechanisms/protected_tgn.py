
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
        super().__init__()
        
        self.tgn = tgn_model
        self.feature_dim = feature_dim
        
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
        
        self._graph = graph
        self._priorities_cache = None
        self._priorities_tensor_cache = None
    
    def update_graph(self, graph: nx.Graph):
        self._graph = graph
        self._priorities_cache = None
        self._priorities_tensor_cache = None
    
    def set_text_embedding_model(self, model):
        if hasattr(self, 'anomaly_detector') and self.anomaly_detector is not None:
            self.anomaly_detector.set_text_embedding_model(model)
            print(f"  🧠 Semantic consistency detection enabled (BERT-based)")
    
    def compute_protection_priorities(self,
                                     graph: Optional[nx.Graph] = None,
                                     use_cache: bool = True,
                                     device: Optional[torch.device] = None) -> Tuple[Dict, torch.Tensor]:
        if graph is None:
            graph = self._graph
        
        if graph is None:
            raise ValueError("Graph must be provided")
        
        if use_cache and self._priorities_cache is not None:
            if device is not None and self._priorities_tensor_cache.device != device:
                self._priorities_tensor_cache = self._priorities_tensor_cache.to(device)
            return self._priorities_cache, self._priorities_tensor_cache
        
        priorities = self.centrality_analyzer.compute_priority_scores(graph)
        
        node_ids = sorted(graph.nodes())
        priority_list = [priorities[nid] for nid in node_ids]
        priority_tensor = torch.tensor(priority_list, dtype=torch.float32, device=device)
        
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
        anomaly_dict = self.anomaly_detector.batch_compute_anomaly_scores(
            node_ids, message_counts, embeddings, agent_messages
        )
        
        anomaly_list = [anomaly_dict.get(nid, 0.0) for nid in node_ids]
        anomaly_tensor = torch.tensor(anomaly_list, dtype=torch.float32, device=device)
        
        return anomaly_tensor
    
    def forward(self,
                agent_features: torch.Tensor,
                communication_topology: torch.Tensor,
                communication_attributes: Optional[torch.Tensor] = None,
                temporal_stamps: Optional[torch.Tensor] = None,
                agent_indices: Optional[torch.Tensor] = None,
                current_state_id: Optional[int] = None,
                context_features: Optional[torch.Tensor] = None,
                question_embedding: Optional[torch.Tensor] = None,
                transition_history: Optional[torch.Tensor] = None,
                graph: Optional[nx.Graph] = None,
                message_counts: Optional[Dict] = None,
                embeddings: Optional[Dict] = None,
                agent_messages: Optional[Dict] = None,
                **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        device = agent_features.device
        num_nodes = agent_features.size(0)
        
        sampled_edges = None
        if agent_messages is not None and '_communication_edges' in agent_messages:
            sampled_edges = agent_messages['_communication_edges']
        
        if sampled_edges and len(sampled_edges) > 0:
            dynamic_graph = self._build_graph_from_sampled_edges(sampled_edges, num_nodes)
            _, priorities = self.compute_protection_priorities(dynamic_graph, use_cache=False, device=device)
        elif communication_topology is not None and communication_topology.size(1) > 0:
            dynamic_graph = self._build_graph_from_topology(communication_topology, num_nodes)
            _, priorities = self.compute_protection_priorities(dynamic_graph, use_cache=False, device=device)
        elif graph is not None:
            _, priorities = self.compute_protection_priorities(graph, use_cache=True, device=device)
        else:
            priorities = torch.ones(num_nodes, dtype=torch.float32, device=device) * 0.5
        
        
        if message_counts is not None or embeddings is not None or agent_messages is not None:
            node_ids = list(range(num_nodes))
            if message_counts is not None:
                for agent_id, count in message_counts.items():
                    if isinstance(agent_id, int) and 0 <= agent_id < num_nodes:
                        self.anomaly_detector.update_history(agent_id, message_count=count)
            if embeddings is not None:
                for agent_id, emb in embeddings.items():
                    if isinstance(agent_id, int) and 0 <= agent_id < num_nodes:
                        if isinstance(emb, torch.Tensor):
                            self.anomaly_detector.update_history(agent_id, embedding=emb)
                        else:
                            emb_tensor = torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb
                            self.anomaly_detector.update_history(agent_id, embedding=emb_tensor)
            
            anomaly_scores = self.compute_anomaly_scores(
                node_ids, message_counts, embeddings, agent_messages, device=device
            )
            
            if agent_messages:
                for agent_id, message in agent_messages.items():
                    if isinstance(agent_id, int) and 0 <= agent_id < num_nodes:
                        pollution_score, pollution_info = self.anomaly_detector.detect_message_pollution(message)
                        if pollution_score > 0.5:
                            pollution_type = pollution_info.get('pollution_type', 'unknown')
                            print(f"  🛡️ Detected message pollution: Agent {agent_id}, type={pollution_type}, score={pollution_score:.3f}")
        else:
            anomaly_scores = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        
        
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
        
        trust_scores = self.trust_calculator(anomaly_scores, priorities, pollution_scores)
        
        source_trust = trust_scores[communication_topology[0]]      # [num_edges]
        target_priority = priorities[communication_topology[1]]     # [num_edges]
        message_weights = self.weight_calculator(source_trust, target_priority)
        
        protected_topology, protected_weights = self._apply_message_attenuation(
            communication_topology, message_weights, anomaly_scores, priorities
        )
        
        is_fsm_tgn = hasattr(self.tgn, 'transition_predictor') and hasattr(self.tgn, 'listener_predictor')
        
        try:
            if is_fsm_tgn:
                if protected_topology.size(1) < communication_topology.size(1):
                    fsm_outputs = self.tgn(
                        agent_features=agent_features,
                        communication_topology=protected_topology,
                        current_state_id=current_state_id,
                        context_features=context_features,
                        question_embedding=question_embedding,
                        transition_history=transition_history,
                        communication_attributes=None,
                        temporal_stamps=temporal_stamps,
                        agent_indices=agent_indices,
                        **kwargs
                    )
                else:
                    fsm_outputs = self.tgn(
                        agent_features=agent_features,
                        communication_topology=communication_topology,
                        current_state_id=current_state_id,
                        context_features=context_features,
                        question_embedding=question_embedding,
                        transition_history=transition_history,
                        communication_attributes=communication_attributes,
                        temporal_stamps=temporal_stamps,
                        agent_indices=agent_indices,
                        **kwargs
                    )
                    
                    if 'agent_features' in fsm_outputs:
                        fsm_outputs['agent_features'] = self._apply_output_attenuation(
                            fsm_outputs['agent_features'], agent_features, communication_topology, 
                            message_weights, anomaly_scores
                        )
                
                fsm_outputs['protection_data'] = {
                    'anomaly_scores': anomaly_scores,
                    'priorities': priorities,
                    'trust_scores': trust_scores,
                    'message_weights': message_weights
                }
                return fsm_outputs
            else:
                if protected_topology.size(1) < communication_topology.size(1):
                    output = self.tgn(
                        agent_features,
                        protected_topology,
                        communication_attributes=None,
                        temporal_stamps=temporal_stamps,
                        agent_indices=agent_indices
                    )
                else:
                    output = self.tgn(
                        agent_features,
                        communication_topology,
                        communication_attributes=communication_attributes,
                        temporal_stamps=temporal_stamps,
                        agent_indices=agent_indices
                    )
                    
                    output = self._apply_output_attenuation(
                        output, agent_features, communication_topology, 
                        message_weights, anomaly_scores
                    )
                
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
            print(f"⚠️  TGN forward pass failed: {e}")
            import traceback
            traceback.print_exc()
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
        graph = nx.DiGraph()
        
        graph.add_nodes_from(range(num_nodes))
        
        if communication_topology.size(1) > 0:
            edges = communication_topology.cpu().numpy()
            for i in range(edges.shape[1]):
                src = int(edges[0, i])
                dst = int(edges[1, i])
                if src != dst:
                    graph.add_edge(src, dst)
        
        return graph
    
    def _build_graph_from_sampled_edges(self, 
                                       sampled_edges: List[Tuple],
                                       num_nodes: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        
        graph.add_nodes_from(range(num_nodes))
        
        for src, dst in sampled_edges:
            try:
                if isinstance(src, str) and '_' in src:
                    src_idx = int(src.split('_')[-1])
                else:
                    src_idx = int(src)
                    
                if isinstance(dst, str) and '_' in dst:
                    dst_idx = int(dst.split('_')[-1])
                else:
                    dst_idx = int(dst)
                
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
        if threshold > 0:
            valid_mask = message_weights > threshold
            
            if valid_mask.sum() > 0:
                protected_topology = communication_topology[:, valid_mask]
                protected_weights = message_weights[valid_mask]
                
                if protected_topology.size(1) < communication_topology.size(1):
                    num_filtered = communication_topology.size(1) - protected_topology.size(1)
                    print(f"  🛡️  Filtered out {num_filtered} low-trust communication edges")
                
                return protected_topology, protected_weights
        
        return communication_topology, message_weights
    
    def _apply_output_attenuation(self,
                                  output: torch.Tensor,
                                  original_features: torch.Tensor,
                                  communication_topology: torch.Tensor,
                                  message_weights: torch.Tensor,
                                  anomaly_scores: torch.Tensor) -> torch.Tensor:
        device = output.device
        num_nodes = output.size(0)
        
        # ═══════════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════════
        sender_trust_for_receiver = torch.ones(num_nodes, device=device)
        
        for node_id in range(num_nodes):
            incoming_mask = communication_topology[1] == node_id
            
            if incoming_mask.sum() > 0:
                sender_indices = communication_topology[0][incoming_mask]
                
                sender_anomalies = anomaly_scores[sender_indices]
                sender_trust = 1.0 - sender_anomalies
                
                incoming_weights = message_weights[incoming_mask]
                
                combined_sender_trust = incoming_weights * sender_trust
                
                if incoming_weights.sum() > 0:
                    sender_trust_for_receiver[node_id] = (
                        (combined_sender_trust * incoming_weights).sum() / 
                        incoming_weights.sum()
                    )
                else:
                    sender_trust_for_receiver[node_id] = combined_sender_trust.mean()
        
        # ═══════════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════════
        
        trust_weights = sender_trust_for_receiver.unsqueeze(-1)  # [num_nodes, 1]
        
        
        trust_weights = torch.clamp(trust_weights, min=0.1, max=1.0)
        
        # protected_output = sender_trust × TGN_output + (1 - sender_trust) × original_features
        #
        protected_output = trust_weights * output + (1.0 - trust_weights) * original_features
        
        return protected_output
    
    def get_protection_statistics(self) -> Dict[str, Any]:
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
        try:
            output = self.tgn(
                agent_features,
                communication_topology,
                communication_attributes=None,
                temporal_stamps=temporal_stamps,
                agent_indices=agent_indices
            )
            
            raw_messages = self.compute_messages(agent_features, communication_topology)
            
            protected_messages = raw_messages * message_weights.unsqueeze(-1)
            
            return output, protected_messages
            
        except Exception as e:
            print(f"⚠️  TGN forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            
            num_edges = communication_topology.size(1) if communication_topology.dim() > 1 else 0
            default_messages = torch.zeros(num_edges, self.feature_dim * 2, 
                                          dtype=agent_features.dtype, 
                                          device=agent_features.device)
            return agent_features, default_messages
    
    def compute_messages(self,
                        agent_features: torch.Tensor,
                        communication_topology: torch.Tensor) -> torch.Tensor:
        source_features = agent_features[communication_topology[0]]  # [num_edges, feature_dim]
        target_features = agent_features[communication_topology[1]]  # [num_edges, feature_dim]
        
        messages = torch.cat([source_features, target_features], dim=-1)
        
        return messages
    
    def compute_loss(self,
                    predictions: torch.Tensor,
                    labels: torch.Tensor,
                    messages: torch.Tensor,
                    anomaly_scores: torch.Tensor,
                    priorities: torch.Tensor,
                    edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        model_params = [p for p in self.parameters() if p.requires_grad]
        
        loss, loss_dict = self.protection_loss(
            predictions, labels, messages,
            anomaly_scores, priorities, edge_index,
            model_params
        )
        
        return loss, loss_dict
    
    def get_learned_weights(self) -> Dict[str, Any]:
        return {
            'centrality': self.centrality_analyzer.get_fusion_weights(),
            'anomaly': self.anomaly_detector.get_fusion_weights()
        }
    
    def visualize_protection(self,
                            graph: nx.Graph,
                            save_path: Optional[str] = None):
        priorities, _ = self.compute_protection_priorities(graph, use_cache=False)
        self.centrality_analyzer.visualize(graph, priorities, save_path)
    
    def update_anomaly_history(self,
                               agent_id: Any,
                               message_count: Optional[int] = None,
                               embedding: Optional[torch.Tensor] = None):
        self.anomaly_detector.update_history(agent_id, message_count, embedding)
    
    def reset_anomaly_history(self, agent_id: Optional[Any] = None):
        self.anomaly_detector.reset_history(agent_id)


__all__ = [
    'ProtectedTGN'
]


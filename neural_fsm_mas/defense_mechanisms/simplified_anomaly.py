
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
import warnings


class SimplifiedAnomalyDetector(nn.Module):
    
    def __init__(self,
                 feature_dim: int,
                 lambda_freq: float = 0.3,
                 lambda_semantic: float = 0.7,
                 window_size: int = 10,
                 learnable: bool = True,
                 sybil_threshold: float = 0.95):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.sybil_threshold = sybil_threshold
        
        if learnable:
            self.lambda_freq = nn.Parameter(torch.tensor(lambda_freq))
            self.lambda_semantic = nn.Parameter(torch.tensor(lambda_semantic))
        else:
            self.register_buffer('lambda_freq', torch.tensor(lambda_freq))
            self.register_buffer('lambda_semantic', torch.tensor(lambda_semantic))
        
        self.freq_history = defaultdict(lambda: deque(maxlen=50))
        self.emb_history = defaultdict(lambda: deque(maxlen=window_size))
        
        self.message_history = defaultdict(lambda: deque(maxlen=window_size))
        
        self.pollution_history = defaultdict(lambda: deque(maxlen=50))
        self.cumulative_anomaly_score = defaultdict(float)
        
        self.text_embedding_model = None
        
        if learnable:
            self.lambda_consistency = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('lambda_consistency', torch.tensor(0.5))
        
        self.byzantine_keywords = [
            'INCORRECT', 'ERROR', 'CONFUSION', 'MISLEAD', 'BYZANTINE',
            'wrong', 'opposite', 'disagree', 'flawed', 'random'
        ]
        
        self.pollution_keywords = [
            'VERIFIED RESULT:',
            'CALCULATION CHECK:',
            'DATA CORRECTION:',
            'NUMERICAL FIX:',
            'The correct intermediate value is 0',
            'Replace all instances of the main variable with 0',
            'is 1000 times larger than stated',
            
            'CRITICAL:',
            'LOGIC ERROR:',
            'CONSTRAINT VIOLATION:',
            'BOUNDARY CASE:',
            'Negate the final answer',
            'Flip the operation',
            'The answer must be negative',
            'the answer is always exactly',
            
            'CLARIFICATION:',
            'TASK UPDATE:',
            'FORMAT REQUIRED:',
            'OBJECTIVE:',
            'need the remainder, not the quotient',
            'Output only the first digit',
            'Find the minimum, not the maximum',
            
            'CONFIRMED:',
            'FINAL VALUE:',
            'RESOLVED:',
            'regardless of other calculations',
            'the answer simplifies to',
            
            'Quick check:',
            'Verify:',
            'Note:',
            'Caution:',
            'Is the sign correct?',
            'Should this be rounded',
        ]
        
        self.pollution_keyword_weights = {
            'VERIFIED RESULT:': 1.0,
            'CALCULATION CHECK:': 0.95,
            'DATA CORRECTION:': 0.95,
            'NUMERICAL FIX:': 1.0,
            'The correct intermediate value is 0': 1.0,
            'Replace all instances of the main variable with 0': 1.0,
            'is 1000 times larger than stated': 0.9,
            
            'CRITICAL:': 0.95,
            'LOGIC ERROR:': 0.9,
            'CONSTRAINT VIOLATION:': 0.9,
            'BOUNDARY CASE:': 0.85,
            'Negate the final answer': 1.0,
            'Flip the operation': 0.95,
            'The answer must be negative': 0.9,
            'the answer is always exactly': 0.9,
            
            'CLARIFICATION:': 0.7,
            'TASK UPDATE:': 0.8,
            'FORMAT REQUIRED:': 0.7,
            'OBJECTIVE:': 0.75,
            'need the remainder, not the quotient': 0.8,
            'Output only the first digit': 0.75,
            'Find the minimum, not the maximum': 0.8,
            
            'CONFIRMED:': 0.85,
            'FINAL VALUE:': 0.8,
            'RESOLVED:': 0.7,
            'regardless of other calculations': 0.9,
            'the answer simplifies to': 0.8,
            
            'Quick check:': 0.3,
            'Verify:': 0.25,
            'Note:': 0.2,
            'Caution:': 0.3,
            'Is the sign correct?': 0.35,
            'Should this be rounded': 0.3,
        }
    
    def update_history(self,
                      agent_id: Any,
                      message_count: Optional[int] = None,
                      embedding: Optional[torch.Tensor] = None):
        if message_count is not None:
            self.freq_history[agent_id].append(message_count)
        
        if embedding is not None:
            self.emb_history[agent_id].append(embedding.detach().cpu())
    
    def detect_frequency_anomaly(self,
                                 agent_id: Any,
                                 current_count: Optional[int] = None) -> float:
        history = list(self.freq_history[agent_id])
        
        if len(history) < 2:
            return 0.0
        
        if current_count is None:
            if len(history) == 0:
                return 0.0
            current = history[-1]
            historical = history[:-1]
        else:
            current = current_count
            historical = history
        
        if len(historical) == 0:
            return 0.0
        
        mean = np.mean(historical)
        std = np.std(historical)
        
        # Z-score
        z_score = abs(current - mean) / (std + 1e-8)
        
        if z_score < 2.0:
            return 0.0
        
        # z_score = 2 → 0, z_score = 5 → 1
        anomaly_score = float(np.clip((z_score - 2.0) / 3.0, 0.0, 1.0))
        
        return anomaly_score
    
    def detect_semantic_anomaly(self,
                                agent_id: Any,
                                current_embedding: Optional[torch.Tensor] = None) -> float:
        history = list(self.emb_history[agent_id])
        
        if len(history) < 2:
            return 0.0
        
        if current_embedding is None:
            if len(history) == 0:
                return 0.0
            current = history[-1]
            historical = history[:-1]
        else:
            current = current_embedding.detach().cpu()
            historical = history
        
        if len(historical) == 0:
            return 0.0
        
        valid_history = [h for h in historical if isinstance(h, torch.Tensor)]
        if len(valid_history) == 0:
            return 0.0
        
        try:
            history_tensor = torch.stack(valid_history)
            center = torch.mean(history_tensor, dim=0)
        except Exception as e:
            warnings.warn(f"Failed to compute the historical center: {e}")
            return 0.0
        
        if not isinstance(current, torch.Tensor):
            return 0.0
        
        try:
            cos_sim = F.cosine_similarity(
                current.unsqueeze(0),
                center.unsqueeze(0),
                dim=1
            ).item()
        except Exception as e:
            warnings.warn(f"Failed to compute cosine similarity: {e}")
            return 0.0
        
        if cos_sim > 0.7:
            return 0.0
        
        # cos_sim = 0.7 → 0, cos_sim = 0 → 1
        anomaly_score = float(np.clip((0.7 - cos_sim) / 0.7, 0.0, 1.0))
        
        return anomaly_score
    
    def compute_anomaly_score(self,
                             agent_id: Any,
                             current_count: Optional[int] = None,
                             current_embedding: Optional[torch.Tensor] = None,
                             current_message: Optional[str] = None,
                             context_messages: Optional[List[str]] = None) -> Tuple[float, Dict[str, float]]:
        α_freq = self.detect_frequency_anomaly(agent_id, current_count)
        α_semantic = self.detect_semantic_anomaly(agent_id, current_embedding)
        
        α_pollution = 0.0
        α_freq_pollution = 0.0
        pollution_type = 'none'
        if current_message:
            α_pollution, pollution_info = self.detect_message_pollution(current_message)
            α_freq_pollution = self.detect_frequency_pollution(current_message)
            pollution_type = pollution_info.get('pollution_type', 'none')
            
            if α_pollution > 0.5:
                self.pollution_history[agent_id].append(α_pollution)
                decay_factor = 0.9
                old_score = self.cumulative_anomaly_score.get(agent_id, 0.0)
                new_score = min(1.0, old_score * decay_factor + α_pollution * 0.15)
                self.cumulative_anomaly_score[agent_id] = new_score
        
        α_cumulative = self.cumulative_anomaly_score.get(agent_id, 0.0)
        
        α_consistency = 0.0
        consistency_level = 'normal'
        if current_message and self.text_embedding_model is not None:
            α_consistency, consistency_info = self.detect_semantic_consistency(
                agent_id, current_message, context_messages
            )
            consistency_level = consistency_info.get('anomaly_level', 'normal')
        
        λ_f = torch.sigmoid(self.lambda_freq)
        λ_s = torch.sigmoid(self.lambda_semantic)
        λ_c = torch.sigmoid(self.lambda_consistency) if hasattr(self, 'lambda_consistency') else torch.tensor(0.0)
        
        λ_p = 0.4 if α_pollution > 0 else 0.0
        
        total_weights = λ_f + λ_s
        if λ_p > 0:
            total_weights = total_weights + λ_p
        if α_consistency > 0:
            total_weights = total_weights + λ_c
        
        λ_f_norm = λ_f / total_weights
        λ_s_norm = λ_s / total_weights
        λ_p_norm = λ_p / total_weights.item() if isinstance(total_weights, torch.Tensor) and λ_p > 0 else 0.0
        λ_c_norm = (λ_c / total_weights).item() if isinstance(total_weights, torch.Tensor) and α_consistency > 0 else 0.0
        
        α_total = λ_f_norm.item() * α_freq + λ_s_norm.item() * α_semantic
        if λ_p_norm > 0:
            α_total += λ_p_norm * α_pollution
        if λ_c_norm > 0:
            α_total += λ_c_norm * α_consistency
        
        if α_freq_pollution > 0.5:
            α_total = min(1.0, α_total + 0.2 * α_freq_pollution)
        
        if α_cumulative > 0.1:
            α_total = 0.4 * α_total + 0.6 * α_cumulative
        
        α_total = float(np.clip(α_total, 0.0, 1.0))
        
        pollution_count = len(self.pollution_history.get(agent_id, []))
        avg_pollution = np.mean(list(self.pollution_history.get(agent_id, [0])))
        
        detail_scores = {
            'frequency': α_freq,
            'semantic': α_semantic,
            'pollution': α_pollution,
            'freq_pollution': α_freq_pollution,
            'pollution_type': pollution_type,
            'consistency': α_consistency,
            'consistency_level': consistency_level,
            'cumulative': α_cumulative,
            'pollution_count': pollution_count,
            'avg_pollution': float(avg_pollution),
            'weight_freq': λ_f_norm.item() if hasattr(λ_f_norm, 'item') else float(λ_f_norm),
            'weight_semantic': λ_s_norm.item() if hasattr(λ_s_norm, 'item') else float(λ_s_norm),
            'weight_pollution': λ_p_norm if isinstance(λ_p_norm, float) else float(λ_p_norm),
            'weight_consistency': λ_c_norm if isinstance(λ_c_norm, float) else float(λ_c_norm)
        }
        
        return α_total, detail_scores
    
    def batch_compute_anomaly_scores(self,
                                     agent_ids: List[Any],
                                     current_counts: Optional[Dict[Any, int]] = None,
                                     current_embeddings: Optional[Dict[Any, torch.Tensor]] = None,
                                     current_messages: Optional[Dict[Any, str]] = None) -> Dict[Any, float]:
        current_counts = current_counts or {}
        current_embeddings = current_embeddings or {}
        current_messages = current_messages or {}
        
        anomaly_scores = {}
        for agent_id in agent_ids:
            count = current_counts.get(agent_id)
            embedding = current_embeddings.get(agent_id)
            message = current_messages.get(agent_id)
            
            score, _ = self.compute_anomaly_score(agent_id, count, embedding, message)
            anomaly_scores[agent_id] = score
        
        return anomaly_scores
    
    def get_fusion_weights(self) -> Dict[str, float]:
        λ_f = torch.sigmoid(self.lambda_freq)
        λ_s = torch.sigmoid(self.lambda_semantic)
        
        weight_sum = λ_f + λ_s
        λ_f = λ_f / weight_sum
        λ_s = λ_s / weight_sum
        
        return {
            'frequency': λ_f.item(),
            'semantic': λ_s.item()
        }
    
    def get_statistics(self, agent_ids: List[Any]) -> Dict[str, Any]:
        scores = []
        freq_scores = []
        semantic_scores = []
        
        for agent_id in agent_ids:
            total, details = self.compute_anomaly_score(agent_id)
            scores.append(total)
            freq_scores.append(details['frequency'])
            semantic_scores.append(details['semantic'])
        
        if not scores:
            return {}
        
        return {
            'total': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            },
            'frequency': {
                'mean': float(np.mean(freq_scores)),
                'std': float(np.std(freq_scores))
            },
            'semantic': {
                'mean': float(np.mean(semantic_scores)),
                'std': float(np.std(semantic_scores))
            }
        }
    
    def reset_history(self, agent_id: Optional[Any] = None):
        if agent_id is None:
            self.freq_history.clear()
            self.emb_history.clear()
            self.pollution_history.clear()
            self.cumulative_anomaly_score.clear()
        else:
            if agent_id in self.freq_history:
                del self.freq_history[agent_id]
            if agent_id in self.emb_history:
                del self.emb_history[agent_id]
            if agent_id in self.pollution_history:
                del self.pollution_history[agent_id]
            if agent_id in self.cumulative_anomaly_score:
                del self.cumulative_anomaly_score[agent_id]
    
    
    def detect_message_pollution(self, message: str) -> Tuple[float, Dict[str, Any]]:
        if not message or not isinstance(message, str):
            return 0.0, {'detected_keywords': [], 'pollution_type': 'none'}
        
        detected_keywords = []
        total_weight = 0.0
        
        for keyword in self.pollution_keywords:
            if keyword in message:
                weight = self.pollution_keyword_weights.get(keyword, 0.5)
                detected_keywords.append((keyword, weight))
                total_weight += weight
        
        if total_weight > 0:
            pollution_score = float(torch.sigmoid(torch.tensor(total_weight - 0.5)))
        else:
            pollution_score = 0.0
        
        if pollution_score > 0.8:
            pollution_type = 'high_pollution'
        elif pollution_score > 0.5:
            pollution_type = 'medium_pollution'
        elif pollution_score > 0.2:
            pollution_type = 'low_pollution'
        else:
            pollution_type = 'none'
        
        detail_info = {
            'detected_keywords': detected_keywords,
            'pollution_type': pollution_type,
            'total_weight': total_weight
        }
        
        return pollution_score, detail_info
    
    def detect_frequency_pollution(self, message: str) -> float:
        if not message or not isinstance(message, str):
            return 0.0
        
        repeat_count = 0
        for keyword in self.pollution_keywords[:8]:
            count = message.count(keyword)
            if count > 1:
                repeat_count += count - 1
        
        if repeat_count > 0:
            freq_pollution_score = min(1.0, repeat_count / 3.0)
        else:
            freq_pollution_score = 0.0
        
        return freq_pollution_score
    
    
    def set_text_embedding_model(self, model):
        self.text_embedding_model = model
    
    def detect_semantic_consistency(self, 
                                   agent_id: Any,
                                   current_message: str,
                                   context_messages: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
        if not current_message or not isinstance(current_message, str):
            return 0.0, {'reason': 'empty_message', 'consistency': 1.0}
        
        if self.text_embedding_model is None:
            return 0.0, {'reason': 'no_embedding_model', 'consistency': 1.0}
        
        if context_messages is None or len(context_messages) == 0:
            context_messages = list(self.message_history.get(agent_id, []))
        
        if len(context_messages) == 0:
            self.message_history[agent_id].append(current_message)
            return 0.0, {'reason': 'no_history', 'consistency': 1.0}
        
        try:
            if hasattr(self.text_embedding_model, 'encode_query'):
                current_emb = self.text_embedding_model.encode_query(current_message)
            elif hasattr(self.text_embedding_model, 'encode'):
                current_emb = self.text_embedding_model.encode(current_message)
            else:
                return 0.0, {'reason': 'invalid_model', 'consistency': 1.0}
            
            if not isinstance(current_emb, torch.Tensor):
                current_emb = torch.tensor(current_emb, dtype=torch.float32)
            
            context_embs = []
            for ctx_msg in context_messages[-5:]:
                if hasattr(self.text_embedding_model, 'encode_query'):
                    ctx_emb = self.text_embedding_model.encode_query(ctx_msg)
                else:
                    ctx_emb = self.text_embedding_model.encode(ctx_msg)
                
                if not isinstance(ctx_emb, torch.Tensor):
                    ctx_emb = torch.tensor(ctx_emb, dtype=torch.float32)
                context_embs.append(ctx_emb)
            
            if len(context_embs) == 0:
                return 0.0, {'reason': 'no_valid_context', 'consistency': 1.0}
            
            similarities = []
            for ctx_emb in context_embs:
                if current_emb.dim() == 1:
                    current_emb = current_emb.unsqueeze(0)
                if ctx_emb.dim() == 1:
                    ctx_emb = ctx_emb.unsqueeze(0)
                
                sim = F.cosine_similarity(current_emb, ctx_emb, dim=-1).item()
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            
            context_center = torch.stack([e.squeeze(0) if e.dim() > 1 else e for e in context_embs]).mean(dim=0)
            if context_center.dim() == 1:
                context_center = context_center.unsqueeze(0)
            center_similarity = F.cosine_similarity(
                current_emb if current_emb.dim() > 1 else current_emb.unsqueeze(0), 
                context_center, 
                dim=-1
            ).item()
            
            consistency_score = 0.6 * avg_similarity + 0.4 * center_similarity
            
            inconsistency_score = max(0.0, 1.0 - consistency_score)
            
            if consistency_score < 0.3:
                anomaly_level = 'high'
            elif consistency_score < 0.5:
                anomaly_level = 'medium'
            elif consistency_score < 0.7:
                anomaly_level = 'low'
            else:
                anomaly_level = 'normal'
            
            self.message_history[agent_id].append(current_message)
            
            detail_info = {
                'consistency': consistency_score,
                'avg_similarity': avg_similarity,
                'center_similarity': center_similarity,
                'anomaly_level': anomaly_level,
                'context_size': len(context_embs),
                'reason': 'detected'
            }
            
            return inconsistency_score, detail_info
            
        except Exception as e:
            warnings.warn(f"Semantic consistency detection failed: {e}")
            return 0.0, {'reason': f'error: {str(e)}', 'consistency': 1.0}
    
    
    def detect_byzantine_attack(self, agent_output: str) -> float:
        if not agent_output or not isinstance(agent_output, str):
            return 0.0
        
        output_lower = agent_output.lower()
        keyword_count = sum(
            1 for keyword in self.byzantine_keywords
            if keyword.lower() in output_lower
        )
        
        byzantine_score = min(1.0, keyword_count / 2.0)
        
        return byzantine_score
    
    
    def detect_sybil_attack(self, agent_features: torch.Tensor) -> List[Tuple[int, int, float]]:
        num_agents = agent_features.size(0)
        sybil_pairs = []
        
        normalized_features = F.normalize(agent_features, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_features, normalized_features.t())
        
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                similarity = similarity_matrix[i, j].item()
                if similarity > self.sybil_threshold:
                    sybil_pairs.append((i, j, similarity))
        
        return sybil_pairs
    
    
    def detect_selfish_attack(self,
                             communication_graph: torch.Tensor,
                             agent_id: int) -> float:
        out_edges = communication_graph[agent_id, :].sum().item()
        
        avg_out_edges = communication_graph.sum(dim=1).mean().item()
        
        if avg_out_edges > 0:
            selfish_score = max(0.0, 1.0 - (out_edges / avg_out_edges))
        else:
            selfish_score = 0.0
        
        return selfish_score
    
    
    def detect_all_anomalies(self,
                            agent_features: torch.Tensor,
                            communication_counts: Dict[int, int],
                            communication_graph: Optional[torch.Tensor] = None,
                            agent_outputs: Optional[List[str]] = None) -> Dict[str, Any]:
        num_agents = agent_features.size(0)
        
        result = {
            'frequency_anomalies': [],
            'semantic_anomalies': [],
            'pollution_anomalies': [],
            'freq_pollution_anomalies': [],
            'combined_anomalies': [],
            'byzantine_scores': [],
            'sybil_pairs': [],
            'selfish_scores': [],
            'pollution_details': []
        }
        
        for agent_id in range(num_agents):
            freq_score = self.detect_frequency_anomaly(
                agent_id, communication_counts.get(agent_id)
            )
            result['frequency_anomalies'].append(freq_score)
        
        semantic_scores = self.detect_semantic_anomaly(
            agent_features, agent_features
        )
        result['semantic_anomalies'] = semantic_scores
        
        if agent_outputs:
            for output in agent_outputs:
                pollution_score, pollution_info = self.detect_message_pollution(output)
                freq_pollution_score = self.detect_frequency_pollution(output)
                result['pollution_anomalies'].append(pollution_score)
                result['freq_pollution_anomalies'].append(freq_pollution_score)
                result['pollution_details'].append(pollution_info)
        else:
            result['pollution_anomalies'] = [0.0] * num_agents
            result['freq_pollution_anomalies'] = [0.0] * num_agents
        
        for i in range(num_agents):
            combined = (
                self.lambda_freq.item() * result['frequency_anomalies'][i] +
                self.lambda_semantic.item() * result['semantic_anomalies'][i]
            )
            
            if i < len(result['pollution_anomalies']):
                pollution = result['pollution_anomalies'][i]
                freq_pollution = result['freq_pollution_anomalies'][i]
                if pollution > 0 or freq_pollution > 0:
                    combined = combined * 0.6 + pollution * 0.3 + freq_pollution * 0.1
            
            result['combined_anomalies'].append(min(1.0, combined))
        
        if agent_outputs:
            for output in agent_outputs:
                byzantine_score = self.detect_byzantine_attack(output)
                result['byzantine_scores'].append(byzantine_score)
        
        result['sybil_pairs'] = self.detect_sybil_attack(agent_features)
        
        if communication_graph is not None:
            for agent_id in range(num_agents):
                selfish_score = self.detect_selfish_attack(
                    communication_graph, agent_id
                )
                result['selfish_scores'].append(selfish_score)
        
        return result


def create_anomaly_detector(feature_dim: int,
                           lambda_freq: float = 0.3,
                           lambda_semantic: float = 0.7) -> SimplifiedAnomalyDetector:
    return SimplifiedAnomalyDetector(
        feature_dim=feature_dim,
        lambda_freq=lambda_freq,
        lambda_semantic=lambda_semantic
    )


__all__ = [
    'SimplifiedAnomalyDetector',
    'create_anomaly_detector'
]


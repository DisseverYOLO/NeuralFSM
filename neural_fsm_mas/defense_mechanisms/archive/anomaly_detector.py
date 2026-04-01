
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import warnings


class HistoricalBehaviorAnomalyDetector(nn.Module):
    
    def __init__(self, 
                 feature_dim: int,
                 hidden_dim: int = 128,
                 window_size: int = 10,
                 num_lstm_layers: int = 2,
                 fusion_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        self.fusion_weights = fusion_weights or {
            'frequency': 0.2,
            'semantic': 0.3,
            'transition': 0.0,
            'temporal': 0.5
        }
        
        self.lstm_detector = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=0.1 if num_lstm_layers > 1 else 0.0,
            batch_first=True
        )
        
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.semantic_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        self.message_history = defaultdict(lambda: deque(maxlen=window_size))
        self.feature_history = defaultdict(lambda: deque(maxlen=window_size))
        self.embedding_history = defaultdict(lambda: deque(maxlen=window_size))
        self.frequency_history = defaultdict(lambda: deque(maxlen=50))
        
    def update_history(self, 
                      agent_id: Any,
                      message_count: int = None,
                      feature: torch.Tensor = None,
                      embedding: torch.Tensor = None):
        if message_count is not None:
            self.frequency_history[agent_id].append(message_count)
        
        if feature is not None:
            self.feature_history[agent_id].append(feature.detach().cpu())
        
        if embedding is not None:
            self.embedding_history[agent_id].append(embedding.detach().cpu())
    
    def detect_frequency_anomaly(self, 
                                agent_id: Any,
                                current_count: int = None) -> float:
        freq_history = list(self.frequency_history[agent_id])
        
        if len(freq_history) < 2:
            return 0.0
        
        if current_count is None:
            if len(freq_history) == 0:
                return 0.0
            current_count = freq_history[-1]
            historical_counts = freq_history[:-1]
        else:
            historical_counts = freq_history
        
        if len(historical_counts) == 0:
            return 0.0
        
        mean_freq = np.mean(historical_counts)
        std_freq = np.std(historical_counts)
        
        # Z-score
        z_score = abs(current_count - mean_freq) / (std_freq + 1e-8)
        
        anomaly_score = float(torch.sigmoid(torch.tensor(z_score / 2.0)))
        
        return anomaly_score
    
    def detect_semantic_anomaly(self, 
                               agent_id: Any,
                               current_embedding: torch.Tensor = None) -> float:
        emb_history = list(self.embedding_history[agent_id])
        
        if len(emb_history) < 2:
            return 0.0
        
        if current_embedding is None:
            if len(emb_history) == 0:
                return 0.0
            current = emb_history[-1]
            history = emb_history[:-1]
        else:
            current = current_embedding.detach().cpu()
            history = emb_history
        
        if len(history) == 0:
            return 0.0
        
        history_tensor = torch.stack([h for h in history if isinstance(h, torch.Tensor)])
        if len(history_tensor) == 0:
            return 0.0
        
        center = torch.mean(history_tensor, dim=0)
        
        if not isinstance(current, torch.Tensor):
            return 0.0
        
        cos_sim = F.cosine_similarity(
            current.unsqueeze(0), 
            center.unsqueeze(0),
            dim=1
        ).item()
        
        anomaly_score = max(0.0, 1.0 - cos_sim)
        
        return anomaly_score
    
    def detect_temporal_anomaly_lstm(self, 
                                    agent_id: Any,
                                    current_feature: torch.Tensor = None) -> float:
        feat_history = list(self.feature_history[agent_id])
        
        if len(feat_history) < self.window_size:
            return 0.0
        
        if current_feature is None:
            window = feat_history[-self.window_size:]
        else:
            window = feat_history[-(self.window_size-1):] + [current_feature.detach().cpu()]
        
        try:
            x = torch.stack([f for f in window if isinstance(f, torch.Tensor)])
            if len(x) < self.window_size:
                return 0.0
            x = x.unsqueeze(0)  # [1, window_size, feature_dim]
        except Exception as e:
            warnings.warn(f"Failed to prepare data for temporal anomaly detection: {e}")
            return 0.0
        
        try:
            with torch.no_grad():
                lstm_out, (h_n, c_n) = self.lstm_detector(x)
                
                reconstructed = self.reconstruction_decoder(lstm_out)
                
                reconstruction_error = F.mse_loss(reconstructed, x)
                
                anomaly_score = torch.sigmoid(reconstruction_error * 5.0).item()
        
        except Exception as e:
            warnings.warn(f"LSTM anomaly detection failed: {e}")
            return 0.0
        
        return anomaly_score
    
    def compute_comprehensive_anomaly_score(self, 
                                  agent_id: Any,
                                  current_data: Optional[Dict] = None) -> Tuple[float, Dict[str, float]]:
        current_data = current_data or {}
        
        α_freq = self.detect_frequency_anomaly(
            agent_id, 
            current_data.get('message_count')
        )
        
        α_semantic = self.detect_semantic_anomaly(
            agent_id, 
            current_data.get('embedding')
        )
        
        α_temporal = self.detect_temporal_anomaly_lstm(
            agent_id, 
            current_data.get('feature')
        )
        
        λ = self.fusion_weights
        
        α_total = (λ['frequency'] * α_freq + 
                  λ['semantic'] * α_semantic + 
                  λ['temporal'] * α_temporal)
        
        weight_sum = λ['frequency'] + λ['semantic'] + λ['temporal']
        if weight_sum > 0:
            α_total = α_total / weight_sum
        
        α_total = float(np.clip(α_total, 0.0, 1.0))
        
        detail_scores = {
            'frequency': α_freq,
            'semantic': α_semantic,
            'temporal': α_temporal
        }
        
        return α_total, detail_scores
    
    def batch_compute_anomaly_scores(self, 
                                    agent_ids: List[Any],
                                    current_data_dict: Optional[Dict[Any, Dict]] = None) -> Dict[Any, float]:
        current_data_dict = current_data_dict or {}
        
        anomaly_scores = {}
        for agent_id in agent_ids:
            score, _ = self.compute_comprehensive_anomaly_score(
                agent_id, 
                current_data_dict.get(agent_id)
            )
            anomaly_scores[agent_id] = score
        
        return anomaly_scores
    
    def get_anomaly_statistics(self, agent_ids: List[Any]) -> Dict[str, Any]:
        scores = []
        detail_stats = defaultdict(list)
        
        for agent_id in agent_ids:
            total_score, details = self.compute_comprehensive_anomaly_score(agent_id)
            scores.append(total_score)
            
            for key, value in details.items():
                detail_stats[key].append(value)
        
        if not scores:
            return {}
        
        stats = {
            'total': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            }
        }
        
        for key, values in detail_stats.items():
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return stats
    
    def reset_history(self, agent_id: Optional[Any] = None):
        if agent_id is None:
            self.message_history.clear()
            self.feature_history.clear()
            self.embedding_history.clear()
            self.frequency_history.clear()
        else:
            if agent_id in self.message_history:
                del self.message_history[agent_id]
            if agent_id in self.feature_history:
                del self.feature_history[agent_id]
            if agent_id in self.embedding_history:
                del self.embedding_history[agent_id]
            if agent_id in self.frequency_history:
                del self.frequency_history[agent_id]


def create_anomaly_detector(feature_dim: int, 
                           hidden_dim: int = 128,
                           window_size: int = 10) -> HistoricalBehaviorAnomalyDetector:
    return HistoricalBehaviorAnomalyDetector(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        window_size=window_size
    )


__all__ = [
    'HistoricalBehaviorAnomalyDetector',
    'create_anomaly_detector'
]



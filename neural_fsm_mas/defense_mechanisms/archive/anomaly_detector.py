"""
Historical Behavior Anomaly Detector
历史行为异常检测器

功能:
1. 消息频率异常检测 (统计方法)
2. 语义偏离异常检测 (基于嵌入)
3. 状态转移模式异常检测
4. 时序异常检测 (LSTM-based)
5. 综合异常评分

作者: Neural FSM Team
日期: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import warnings


class HistoricalBehaviorAnomalyDetector(nn.Module):
    """
    历史行为异常检测器
    
    使用多种方法检测智能体的异常行为:
    1. 统计方法: Z-score异常检测
    2. 语义方法: 余弦相似度偏离
    3. 时序方法: LSTM重构误差
    4. 图结构方法: 状态转移模式
    """
    
    def __init__(self, 
                 feature_dim: int,
                 hidden_dim: int = 128,
                 window_size: int = 10,
                 num_lstm_layers: int = 2,
                 fusion_weights: Optional[Dict[str, float]] = None):
        """
        初始化异常检测器
        
        Args:
            feature_dim: 节点特征维度
            hidden_dim: LSTM隐藏层维度
            window_size: 时间窗口大小
            num_lstm_layers: LSTM层数
            fusion_weights: 各检测方法的融合权重
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        # 融合权重
        self.fusion_weights = fusion_weights or {
            'frequency': 0.2,
            'semantic': 0.3,
            'transition': 0.0,  # 暂未实现
            'temporal': 0.5
        }
        
        # LSTM时序异常检测器
        self.lstm_detector = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=0.1 if num_lstm_layers > 1 else 0.0,
            batch_first=True
        )
        
        # 重构解码器(用于异常检测)
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 语义编码器
        self.semantic_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # 历史数据缓存
        self.message_history = defaultdict(lambda: deque(maxlen=window_size))
        self.feature_history = defaultdict(lambda: deque(maxlen=window_size))
        self.embedding_history = defaultdict(lambda: deque(maxlen=window_size))
        self.frequency_history = defaultdict(lambda: deque(maxlen=50))  # 更长的历史
        
    def update_history(self, 
                      agent_id: Any,
                      message_count: int = None,
                      feature: torch.Tensor = None,
                      embedding: torch.Tensor = None):
        """
        更新智能体的历史数据
        
        Args:
            agent_id: 智能体ID
            message_count: 当前时间步的消息数量
            feature: 节点特征向量
            embedding: 消息嵌入向量
        """
        if message_count is not None:
            self.frequency_history[agent_id].append(message_count)
        
        if feature is not None:
            self.feature_history[agent_id].append(feature.detach().cpu())
        
        if embedding is not None:
            self.embedding_history[agent_id].append(embedding.detach().cpu())
    
    def detect_frequency_anomaly(self, 
                                agent_id: Any,
                                current_count: int = None) -> float:
        """
        消息频率异常检测
        
        方法: Z-score异常检测
        α_freq(i,t) = sigmoid(|freq(i,t) - μ| / (σ + ε))
        
        Args:
            agent_id: 智能体ID
            current_count: 当前消息数量(可选,如果未提供则使用历史最后一个)
        
        Returns:
            异常分数 ∈ [0, 1]
        """
        freq_history = list(self.frequency_history[agent_id])
        
        if len(freq_history) < 2:
            return 0.0  # 数据不足,无法判断
        
        # 使用历史数据计算统计量
        if current_count is None:
            if len(freq_history) == 0:
                return 0.0
            current_count = freq_history[-1]
            historical_counts = freq_history[:-1]
        else:
            historical_counts = freq_history
        
        if len(historical_counts) == 0:
            return 0.0
        
        # 计算均值和标准差
        mean_freq = np.mean(historical_counts)
        std_freq = np.std(historical_counts)
        
        # Z-score
        z_score = abs(current_count - mean_freq) / (std_freq + 1e-8)
        
        # 转换为[0,1]的异常分数
        anomaly_score = float(torch.sigmoid(torch.tensor(z_score / 2.0)))
        
        return anomaly_score
    
    def detect_semantic_anomaly(self, 
                               agent_id: Any,
                               current_embedding: torch.Tensor = None) -> float:
        """
        语义偏离异常检测
        
        方法: 计算当前消息与历史消息中心的余弦距离
        α_semantic(i,t) = 1 - cos_sim(m_t, center(M_history))
        
        Args:
            agent_id: 智能体ID
            current_embedding: 当前消息嵌入(可选)
        
        Returns:
            异常分数 ∈ [0, 1]
        """
        emb_history = list(self.embedding_history[agent_id])
        
        if len(emb_history) < 2:
            return 0.0
        
        # 获取当前嵌入
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
        
        # 计算历史中心
        history_tensor = torch.stack([h for h in history if isinstance(h, torch.Tensor)])
        if len(history_tensor) == 0:
            return 0.0
        
        center = torch.mean(history_tensor, dim=0)
        
        # 余弦相似度
        if not isinstance(current, torch.Tensor):
            return 0.0
        
        cos_sim = F.cosine_similarity(
            current.unsqueeze(0), 
            center.unsqueeze(0),
            dim=1
        ).item()
        
        # 转为异常分数 (相似度越低,异常分数越高)
        anomaly_score = max(0.0, 1.0 - cos_sim)
        
        return anomaly_score
    
    def detect_temporal_anomaly_lstm(self, 
                                    agent_id: Any,
                                    current_feature: torch.Tensor = None) -> float:
        """
        时序异常检测 (基于LSTM重构误差)
        
        方法:
        1. 用LSTM学习正常行为模式
        2. 计算重构误差作为异常分数
        α_temporal(i,t) = sigmoid(||x_t - x̂_t||²)
        
        Args:
            agent_id: 智能体ID
            current_feature: 当前特征向量(可选)
        
        Returns:
            异常分数 ∈ [0, 1]
        """
        feat_history = list(self.feature_history[agent_id])
        
        if len(feat_history) < self.window_size:
            return 0.0  # 数据不足
        
        # 准备时间窗口数据
        if current_feature is None:
            window = feat_history[-self.window_size:]
        else:
            window = feat_history[-(self.window_size-1):] + [current_feature.detach().cpu()]
        
        # 转换为tensor
        try:
            x = torch.stack([f for f in window if isinstance(f, torch.Tensor)])
            if len(x) < self.window_size:
                return 0.0
            x = x.unsqueeze(0)  # [1, window_size, feature_dim]
        except Exception as e:
            warnings.warn(f"时序异常检测数据准备失败: {e}")
            return 0.0
        
        # LSTM编码
        try:
            with torch.no_grad():
                lstm_out, (h_n, c_n) = self.lstm_detector(x)
                
                # 重构
                reconstructed = self.reconstruction_decoder(lstm_out)
                
                # 计算重构误差
                reconstruction_error = F.mse_loss(reconstructed, x)
                
                # 归一化为[0,1]
                anomaly_score = torch.sigmoid(reconstruction_error * 5.0).item()
        
        except Exception as e:
            warnings.warn(f"LSTM异常检测失败: {e}")
            return 0.0
        
        return anomaly_score
    
    def compute_综合_anomaly_score(self, 
                                  agent_id: Any,
                                  current_data: Optional[Dict] = None) -> Tuple[float, Dict[str, float]]:
        """
        计算综合异常分数
        
        融合多种检测方法的结果
        α_total(i,t) = Σ_k λ_k · α_k(i,t)
        
        Args:
            agent_id: 智能体ID
            current_data: 当前数据字典,包含:
                - 'message_count': 当前消息数
                - 'embedding': 当前消息嵌入
                - 'feature': 当前特征向量
        
        Returns:
            (总异常分数, 各维度异常分数字典)
        """
        current_data = current_data or {}
        
        # 检测各维度异常
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
        
        # 融合权重
        λ = self.fusion_weights
        
        # 计算加权和
        α_total = (λ['frequency'] * α_freq + 
                  λ['semantic'] * α_semantic + 
                  λ['temporal'] * α_temporal)
        
        # 归一化
        weight_sum = λ['frequency'] + λ['semantic'] + λ['temporal']
        if weight_sum > 0:
            α_total = α_total / weight_sum
        
        # 裁剪到[0,1]
        α_total = float(np.clip(α_total, 0.0, 1.0))
        
        # 详细分数
        detail_scores = {
            'frequency': α_freq,
            'semantic': α_semantic,
            'temporal': α_temporal
        }
        
        return α_total, detail_scores
    
    def batch_compute_anomaly_scores(self, 
                                    agent_ids: List[Any],
                                    current_data_dict: Optional[Dict[Any, Dict]] = None) -> Dict[Any, float]:
        """
        批量计算异常分数
        
        Args:
            agent_ids: 智能体ID列表
            current_data_dict: 智能体ID到当前数据的映射
        
        Returns:
            智能体ID到异常分数的映射
        """
        current_data_dict = current_data_dict or {}
        
        anomaly_scores = {}
        for agent_id in agent_ids:
            score, _ = self.compute_综合_anomaly_score(
                agent_id, 
                current_data_dict.get(agent_id)
            )
            anomaly_scores[agent_id] = score
        
        return anomaly_scores
    
    def get_anomaly_statistics(self, agent_ids: List[Any]) -> Dict[str, Any]:
        """
        获取异常统计信息
        
        Args:
            agent_ids: 智能体ID列表
        
        Returns:
            统计信息字典
        """
        scores = []
        detail_stats = defaultdict(list)
        
        for agent_id in agent_ids:
            total_score, details = self.compute_综合_anomaly_score(agent_id)
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
        
        # 各维度统计
        for key, values in detail_stats.items():
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return stats
    
    def reset_history(self, agent_id: Optional[Any] = None):
        """
        重置历史数据
        
        Args:
            agent_id: 智能体ID(如果为None则重置所有)
        """
        if agent_id is None:
            # 重置所有
            self.message_history.clear()
            self.feature_history.clear()
            self.embedding_history.clear()
            self.frequency_history.clear()
        else:
            # 重置特定智能体
            if agent_id in self.message_history:
                del self.message_history[agent_id]
            if agent_id in self.feature_history:
                del self.feature_history[agent_id]
            if agent_id in self.embedding_history:
                del self.embedding_history[agent_id]
            if agent_id in self.frequency_history:
                del self.frequency_history[agent_id]


# 便捷函数
def create_anomaly_detector(feature_dim: int, 
                           hidden_dim: int = 128,
                           window_size: int = 10) -> HistoricalBehaviorAnomalyDetector:
    """
    创建异常检测器
    
    Args:
        feature_dim: 特征维度
        hidden_dim: 隐藏层维度
        window_size: 时间窗口大小
    
    Returns:
        异常检测器实例
    """
    return HistoricalBehaviorAnomalyDetector(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        window_size=window_size
    )


__all__ = [
    'HistoricalBehaviorAnomalyDetector',
    'create_anomaly_detector'
]



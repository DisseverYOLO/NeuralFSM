"""
Graph Centrality Analyzer
图中心性分析器

功能:
1. 计算多种图中心性指标(度、介数、接近性、PageRank)
2. 融合中心性指标为保护优先级分数
3. 支持动态图的增量更新
4. 识别关键节点以进行重点保护

作者: Neural FSM Team
日期: 2025-10-28
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class GraphCentralityAnalyzer:
    """
    图中心性分析器
    
    计算多维度的中心性指标,并融合为统一的保护优先级分数
    支持动态图的高效更新
    """
    
    def __init__(self, 
                 fusion_weights: Optional[Dict[str, float]] = None,
                 normalize: bool = True,
                 enable_caching: bool = True):
        """
        初始化图中心性分析器
        
        Args:
            fusion_weights: 各中心性指标的融合权重
                默认: {'degree': 0.25, 'betweenness': 0.35, 
                       'closeness': 0.2, 'pagerank': 0.2}
            normalize: 是否归一化到[0,1]区间
            enable_caching: 是否启用计算缓存
        """
        self.fusion_weights = fusion_weights or {
            'degree': 0.25,        # 度中心性
            'betweenness': 0.35,   # 介数中心性(最重要)
            'closeness': 0.2,      # 接近中心性
            'pagerank': 0.2        # PageRank中心性
        }
        
        # 验证权重和为1
        weight_sum = sum(self.fusion_weights.values())
        if not np.isclose(weight_sum, 1.0):
            print(f"警告: 融合权重和为{weight_sum:.4f},将自动归一化")
            self.fusion_weights = {
                k: v / weight_sum for k, v in self.fusion_weights.items()
            }
        
        self.normalize = normalize
        self.enable_caching = enable_caching
        
        # 缓存
        self._centrality_cache = {}
        self._priority_cache = {}
        self._graph_hash = None
    
    def _compute_graph_hash(self, graph: nx.Graph) -> int:
        """
        计算图的哈希值,用于缓存判断
        
        Args:
            graph: NetworkX图对象
        
        Returns:
            图的哈希值
        """
        # 使用节点数、边数和边列表的哈希作为图的标识
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        edge_tuple = tuple(sorted(graph.edges()))
        
        return hash((num_nodes, num_edges, edge_tuple))
    
    def _check_cache(self, graph: nx.Graph) -> bool:
        """检查缓存是否有效"""
        if not self.enable_caching:
            return False
        
        current_hash = self._compute_graph_hash(graph)
        return current_hash == self._graph_hash
    
    def _update_cache(self, graph: nx.Graph, centralities: Dict, priorities: Dict):
        """更新缓存"""
        if self.enable_caching:
            self._graph_hash = self._compute_graph_hash(graph)
            self._centrality_cache = centralities
            self._priority_cache = priorities
    
    def compute_degree_centrality(self, graph: nx.Graph) -> Dict[Any, float]:
        """
        计算度中心性
        
        度中心性衡量节点的直接连接数量
        DC(i) = deg(i) / (n-1)
        
        Args:
            graph: NetworkX图对象
        
        Returns:
            节点到度中心性的映射
        """
        return nx.degree_centrality(graph)
    
    def compute_betweenness_centrality(self, 
                                      graph: nx.Graph,
                                      normalized: bool = True) -> Dict[Any, float]:
        """
        计算介数中心性
        
        介数中心性衡量节点在网络中的"桥梁"作用
        BC(i) = Σ_{s≠i≠t} σ_st(i) / σ_st
        
        其中σ_st是s到t的最短路径数, σ_st(i)是经过i的最短路径数
        
        Args:
            graph: NetworkX图对象
            normalized: 是否归一化
        
        Returns:
            节点到介数中心性的映射
        """
        return nx.betweenness_centrality(graph, normalized=normalized)
    
    def compute_closeness_centrality(self, graph: nx.Graph) -> Dict[Any, float]:
        """
        计算接近中心性
        
        接近中心性衡量节点到其他所有节点的平均距离
        CC(i) = (n-1) / Σ_{j≠i} d(i,j)
        
        Args:
            graph: NetworkX图对象
        
        Returns:
            节点到接近中心性的映射
        """
        # 处理不连通图
        if not nx.is_connected(graph.to_undirected()):
            print("警告: 图不连通,使用强连通分量计算接近中心性")
            return self._compute_closeness_for_disconnected_graph(graph)
        
        return nx.closeness_centrality(graph)
    
    def _compute_closeness_for_disconnected_graph(self, 
                                                 graph: nx.Graph) -> Dict[Any, float]:
        """为不连通图计算接近中心性"""
        closeness = {}
        
        for node in graph.nodes():
            # 计算到可达节点的平均距离
            lengths = nx.single_source_shortest_path_length(graph, node)
            
            if len(lengths) > 1:
                avg_length = sum(lengths.values()) / (len(lengths) - 1)
                closeness[node] = 1.0 / avg_length if avg_length > 0 else 0.0
            else:
                closeness[node] = 0.0
        
        return closeness
    
    def compute_pagerank_centrality(self, 
                                   graph: nx.Graph,
                                   alpha: float = 0.85,
                                   max_iter: int = 100) -> Dict[Any, float]:
        """
        计算PageRank中心性
        
        PageRank递归定义节点重要性,考虑邻居节点的重要性
        PR(i) = (1-α)/n + α·Σ_{j→i} PR(j)/deg_out(j)
        
        Args:
            graph: NetworkX图对象
            alpha: 阻尼系数(通常为0.85)
            max_iter: 最大迭代次数
        
        Returns:
            节点到PageRank值的映射
        """
        return nx.pagerank(graph, alpha=alpha, max_iter=max_iter)
    
    def compute_all_centralities(self, graph: nx.Graph) -> Dict[Any, Dict[str, float]]:
        """
        计算所有中心性指标
        
        Args:
            graph: NetworkX图对象
        
        Returns:
            嵌套字典: {node_id: {metric_name: score}}
        """
        # 检查缓存
        if self._check_cache(graph):
            print("✓ 使用缓存的中心性计算结果")
            return self._centrality_cache
        
        print("⏳ 计算图中心性指标...")
        
        # 计算各中心性
        dc = self.compute_degree_centrality(graph)
        bc = self.compute_betweenness_centrality(graph)
        cc = self.compute_closeness_centrality(graph)
        pr = self.compute_pagerank_centrality(graph)
        
        # 组织为嵌套字典
        centralities = {}
        for node in graph.nodes():
            centralities[node] = {
                'degree': dc.get(node, 0.0),
                'betweenness': bc.get(node, 0.0),
                'closeness': cc.get(node, 0.0),
                'pagerank': pr.get(node, 0.0)
            }
        
        print(f"✓ 完成{len(centralities)}个节点的中心性计算")
        
        return centralities
    
    def compute_protection_priority(self, 
                                   centralities: Dict[Any, Dict[str, float]]) -> Dict[Any, float]:
        """
        融合中心性指标为保护优先级
        
        公式: π(i) = Σ_k w_k · C_k(i)
        
        Args:
            centralities: 各节点的中心性指标
        
        Returns:
            节点到保护优先级的映射
        """
        priorities = {}
        
        for node, metrics in centralities.items():
            # 加权融合
            priority = sum(
                self.fusion_weights.get(metric, 0.0) * score
                for metric, score in metrics.items()
                if metric in self.fusion_weights
            )
            priorities[node] = priority
        
        # 归一化
        if self.normalize and priorities:
            max_priority = max(priorities.values())
            min_priority = min(priorities.values())
            
            if max_priority > min_priority:
                for node in priorities:
                    priorities[node] = (priorities[node] - min_priority) / \
                                      (max_priority - min_priority)
            else:
                # 所有节点优先级相同
                for node in priorities:
                    priorities[node] = 0.5
        
        return priorities
    
    def analyze_graph(self, graph: nx.Graph) -> Tuple[Dict, Dict]:
        """
        完整的图分析流程
        
        Args:
            graph: NetworkX图对象
        
        Returns:
            (centralities, priorities): 中心性指标和保护优先级
        """
        # 计算中心性
        centralities = self.compute_all_centralities(graph)
        
        # 计算保护优先级
        priorities = self.compute_protection_priority(centralities)
        
        # 更新缓存
        self._update_cache(graph, centralities, priorities)
        
        return centralities, priorities
    
    def identify_critical_nodes(self, 
                               priorities: Dict[Any, float],
                               top_k: int = None,
                               threshold: float = None) -> List[Any]:
        """
        识别关键节点
        
        Args:
            priorities: 节点保护优先级
            top_k: 返回前k个关键节点
            threshold: 优先级阈值,高于此值的节点被认为是关键节点
        
        Returns:
            关键节点列表
        """
        if top_k is not None:
            # 返回top-k节点
            sorted_nodes = sorted(priorities.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
            return [node for node, _ in sorted_nodes[:top_k]]
        
        elif threshold is not None:
            # 返回高于阈值的节点
            return [node for node, priority in priorities.items() 
                   if priority >= threshold]
        
        else:
            # 默认: 返回高于均值的节点
            mean_priority = np.mean(list(priorities.values()))
            return [node for node, priority in priorities.items() 
                   if priority > mean_priority]
    
    def get_priority_statistics(self, priorities: Dict[Any, float]) -> Dict[str, float]:
        """
        获取保护优先级的统计信息
        
        Args:
            priorities: 节点保护优先级
        
        Returns:
            统计信息字典
        """
        values = list(priorities.values())
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
    
    def visualize_priorities(self, 
                           graph: nx.Graph,
                           priorities: Dict[Any, float],
                           save_path: Optional[str] = None):
        """
        可视化保护优先级
        
        Args:
            graph: NetworkX图对象
            priorities: 节点保护优先级
            save_path: 保存路径(可选)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("警告: matplotlib未安装,无法可视化")
            return
        
        # 创建布局
        pos = nx.spring_layout(graph, seed=42)
        
        # 获取节点颜色(基于优先级)
        node_colors = [priorities.get(node, 0.0) for node in graph.nodes()]
        
        # 绘制
        plt.figure(figsize=(12, 8))
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=500,
            cmap=plt.cm.RdYlGn,
            vmin=0, vmax=1,
            alpha=0.8
        )
        
        # 绘制边
        nx.draw_networkx_edges(graph, pos, alpha=0.3, width=1.5)
        
        # 绘制标签
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        # 添加颜色条
        plt.colorbar(nodes, label='Protection Priority')
        
        plt.title('Node Protection Priority Visualization', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 可视化已保存到 {save_path}")
        else:
            plt.show()
        
        plt.close()


class LearnableCentralityFusion(nn.Module):
    """
    可学习的中心性融合模块
    
    使用神经网络学习最优的中心性指标融合权重
    """
    
    def __init__(self, num_centrality_metrics: int = 4):
        """
        Args:
            num_centrality_metrics: 中心性指标数量(默认4: degree, betweenness, closeness, pagerank)
        """
        super().__init__()
        
        # 可学习的融合权重
        self.fusion_weights = nn.Parameter(
            torch.ones(num_centrality_metrics) / num_centrality_metrics
        )
        
        # 注意力机制(可选)
        self.attention_net = nn.Sequential(
            nn.Linear(num_centrality_metrics, num_centrality_metrics * 2),
            nn.ReLU(),
            nn.Linear(num_centrality_metrics * 2, num_centrality_metrics),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, centrality_scores: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 融合中心性指标
        
        Args:
            centrality_scores: [num_nodes, num_metrics] 中心性分数
        
        Returns:
            priorities: [num_nodes] 保护优先级
        """
        # 方法1: 简单加权
        # weights = F.softmax(self.fusion_weights, dim=0)
        # priorities = torch.matmul(centrality_scores, weights)
        
        # 方法2: 注意力加权(更灵活)
        attention_weights = self.attention_net(centrality_scores)  # [num_nodes, num_metrics]
        priorities = torch.sum(centrality_scores * attention_weights, dim=-1)  # [num_nodes]
        
        return priorities
    
    def get_fusion_weights(self) -> Dict[str, float]:
        """获取当前的融合权重"""
        with torch.no_grad():
            weights = torch.softmax(self.fusion_weights, dim=0).cpu().numpy()
        
        metric_names = ['degree', 'betweenness', 'closeness', 'pagerank']
        return {name: float(weight) for name, weight in zip(metric_names, weights)}


# 便捷函数
def analyze_topology_protection(graph: nx.Graph, 
                               top_k: int = 5) -> Tuple[Dict, List]:
    """
    便捷函数: 分析拓扑并识别关键节点
    
    Args:
        graph: NetworkX图对象
        top_k: 返回前k个关键节点
    
    Returns:
        (priorities, critical_nodes): 优先级和关键节点列表
    """
    analyzer = GraphCentralityAnalyzer()
    centralities, priorities = analyzer.analyze_graph(graph)
    critical_nodes = analyzer.identify_critical_nodes(priorities, top_k=top_k)
    
    return priorities, critical_nodes


# 导出
__all__ = [
    'GraphCentralityAnalyzer',
    'LearnableCentralityFusion',
    'analyze_topology_protection'
]



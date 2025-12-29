"""
Simplified Centrality Analyzer
简化版图中心性分析器

只使用2种中心性指标:
1. 介数中心性 (Betweenness Centrality) - 控制信息流动
2. PageRank中心性 - 全局重要性

作者: Neural FSM Team  
日期: 2025-10-28
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict


class SimplifiedCentralityAnalyzer(nn.Module):
    """
    简化版图中心性分析器
    
    只计算介数中心性和PageRank,并通过可学习的权重融合
    """
    
    def __init__(self, 
                 w_betweenness: float = 0.6,
                 w_pagerank: float = 0.4,
                 learnable: bool = True,
                 normalize: bool = True):
        """
        初始化
        
        Args:
            w_betweenness: 介数中心性初始权重 (默认0.6,更重要)
            w_pagerank: PageRank初始权重 (默认0.4)
            learnable: 权重是否可学习
            normalize: 是否归一化到[0,1]
        """
        super().__init__()
        
        self.normalize = normalize
        
        # 可学习的融合权重
        if learnable:
            self.w_betweenness = nn.Parameter(torch.tensor(w_betweenness))
            self.w_pagerank = nn.Parameter(torch.tensor(w_pagerank))
        else:
            self.register_buffer('w_betweenness', torch.tensor(w_betweenness))
            self.register_buffer('w_pagerank', torch.tensor(w_pagerank))
        
        # 缓存
        self._cache = {}
        self._graph_hash = None
    
    def _compute_graph_hash(self, graph: nx.Graph) -> int:
        """计算图的哈希值"""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        edge_tuple = tuple(sorted(graph.edges()))
        return hash((num_nodes, num_edges, edge_tuple))
    
    def compute_betweenness_centrality(self, graph: nx.Graph) -> Dict[Any, float]:
        """
        计算介数中心性
        
        BC(i) = Σ_{s≠i≠t} σ_st(i) / σ_st
        
        衡量节点作为"桥梁"的重要性
        """
        return nx.betweenness_centrality(graph, normalized=True)
    
    def compute_pagerank_centrality(self, 
                                   graph: nx.Graph, 
                                   alpha: float = 0.85) -> Dict[Any, float]:
        """
        计算PageRank中心性
        
        PR(i) = (1-α)/n + α·Σ_{j→i} PR(j)/deg_out(j)
        
        递归定义的全局重要性
        """
        return nx.pagerank(graph, alpha=alpha, max_iter=100)
    
    def compute_priority_scores(self, 
                                graph: nx.Graph,
                                return_components: bool = False) -> Dict[Any, float]:
        """
        计算保护优先级分数
        
        π(i) = w_BC · BC(i) + w_PR · PR(i)
        
        Args:
            graph: NetworkX图对象
            return_components: 是否返回各组件分数
        
        Returns:
            priorities: 节点保护优先级字典
        """
        # 检查缓存
        graph_hash = self._compute_graph_hash(graph)
        if graph_hash == self._graph_hash and 'priorities' in self._cache:
            print("✓ 使用缓存的中心性计算结果")
            if return_components:
                return self._cache['priorities'], self._cache['components']
            return self._cache['priorities']
        
        print("⏳ 计算图中心性指标...")
        
        # 计算两种中心性
        BC = self.compute_betweenness_centrality(graph)
        PR = self.compute_pagerank_centrality(graph)
        
        # 获取权重 (使用sigmoid归一化)
        w_bc = torch.sigmoid(self.w_betweenness)
        w_pr = torch.sigmoid(self.w_pagerank)
        
        # 归一化权重使和为1
        weight_sum = w_bc + w_pr
        w_bc = w_bc / weight_sum
        w_pr = w_pr / weight_sum
        
        # 转换为numpy
        w_bc_np = w_bc.detach().cpu().numpy()
        w_pr_np = w_pr.detach().cpu().numpy()
        
        # 计算加权优先级
        priorities = {}
        components = {}
        
        for node in graph.nodes():
            bc_score = BC.get(node, 0.0)
            pr_score = PR.get(node, 0.0)
            
            priority = w_bc_np * bc_score + w_pr_np * pr_score
            priorities[node] = float(priority)
            
            if return_components:
                components[node] = {
                    'betweenness': bc_score,
                    'pagerank': pr_score
                }
        
        # 归一化到[0,1]
        if self.normalize and priorities:
            max_p = max(priorities.values())
            min_p = min(priorities.values())
            
            if max_p > min_p:
                for node in priorities:
                    priorities[node] = (priorities[node] - min_p) / (max_p - min_p)
            else:
                for node in priorities:
                    priorities[node] = 0.5
        
        # 更新缓存
        self._graph_hash = graph_hash
        self._cache['priorities'] = priorities
        self._cache['components'] = components
        
        print(f"✓ 完成{len(priorities)}个节点的优先级计算")
        
        if return_components:
            return priorities, components
        return priorities
    
    def priority_to_tensor(self, 
                          priorities: Dict[Any, float],
                          node_order: list = None) -> torch.Tensor:
        """
        将优先级字典转换为tensor
        
        Args:
            priorities: 节点优先级字典
            node_order: 节点顺序列表 (如果为None,使用sorted(priorities.keys()))
        
        Returns:
            priority_tensor: [num_nodes]
        """
        if node_order is None:
            node_order = sorted(priorities.keys())
        
        priority_list = [priorities[node] for node in node_order]
        return torch.tensor(priority_list, dtype=torch.float32)
    
    def get_fusion_weights(self) -> Dict[str, float]:
        """
        获取当前的融合权重
        
        Returns:
            权重字典 {'betweenness': w_bc, 'pagerank': w_pr}
        """
        w_bc = torch.sigmoid(self.w_betweenness)
        w_pr = torch.sigmoid(self.w_pagerank)
        
        weight_sum = w_bc + w_pr
        w_bc = w_bc / weight_sum
        w_pr = w_pr / weight_sum
        
        return {
            'betweenness': w_bc.item(),
            'pagerank': w_pr.item()
        }
    
    def identify_critical_nodes(self,
                               priorities: Dict[Any, float],
                               top_k: int = None,
                               threshold: float = None) -> list:
        """
        识别关键节点
        
        Args:
            priorities: 节点优先级
            top_k: 返回前k个节点
            threshold: 阈值(优先级>threshold的节点)
        
        Returns:
            关键节点列表
        """
        if top_k is not None:
            sorted_nodes = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:top_k]]
        
        elif threshold is not None:
            return [node for node, priority in priorities.items() 
                   if priority >= threshold]
        
        else:
            # 默认: 返回高于均值的节点
            mean_priority = np.mean(list(priorities.values()))
            return [node for node, priority in priorities.items() 
                   if priority > mean_priority]
    
    def get_statistics(self, priorities: Dict[Any, float]) -> Dict[str, float]:
        """获取优先级统计信息"""
        values = list(priorities.values())
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    def visualize(self, 
                 graph: nx.Graph,
                 priorities: Dict[Any, float],
                 save_path: Optional[str] = None,
                 figsize: Tuple[int, int] = (12, 8)):
        """
        可视化保护优先级
        
        Args:
            graph: NetworkX图
            priorities: 节点优先级
            save_path: 保存路径(可选)
            figsize: 图像大小
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("警告: matplotlib未安装,无法可视化")
            return
        
        pos = nx.spring_layout(graph, seed=42)
        node_colors = [priorities.get(node, 0.0) for node in graph.nodes()]
        
        plt.figure(figsize=figsize)
        
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
        
        # 颜色条
        plt.colorbar(nodes, label='Protection Priority (π)')
        
        # 标题 (显示当前权重)
        weights = self.get_fusion_weights()
        plt.title(f'Node Protection Priority\n'
                 f'(w_BC={weights["betweenness"]:.3f}, '
                 f'w_PR={weights["pagerank"]:.3f})',
                 fontsize=14)
        
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 可视化已保存到 {save_path}")
        else:
            plt.show()
        
        plt.close()


# 便捷函数
def compute_node_priorities(graph: nx.Graph, 
                           w_betweenness: float = 0.6,
                           w_pagerank: float = 0.4) -> Dict[Any, float]:
    """
    便捷函数: 快速计算节点保护优先级
    
    Args:
        graph: NetworkX图
        w_betweenness: 介数中心性权重
        w_pagerank: PageRank权重
    
    Returns:
        节点保护优先级字典
    """
    analyzer = SimplifiedCentralityAnalyzer(
        w_betweenness=w_betweenness,
        w_pagerank=w_pagerank,
        learnable=False
    )
    
    return analyzer.compute_priority_scores(graph)


__all__ = [
    'SimplifiedCentralityAnalyzer',
    'compute_node_priorities'
]



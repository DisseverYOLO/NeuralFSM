
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class GraphCentralityAnalyzer:
    
    def __init__(self, 
                 fusion_weights: Optional[Dict[str, float]] = None,
                 normalize: bool = True,
                 enable_caching: bool = True):
        self.fusion_weights = fusion_weights or {
            'degree': 0.25,
            'betweenness': 0.35,
            'closeness': 0.2,
            'pagerank': 0.2
        }
        
        weight_sum = sum(self.fusion_weights.values())
        if not np.isclose(weight_sum, 1.0):
            print(f"Warning: fusion weights sum to {weight_sum:.4f}; they will be normalized automatically")
            self.fusion_weights = {
                k: v / weight_sum for k, v in self.fusion_weights.items()
            }
        
        self.normalize = normalize
        self.enable_caching = enable_caching
        
        self._centrality_cache = {}
        self._priority_cache = {}
        self._graph_hash = None
    
    def _compute_graph_hash(self, graph: nx.Graph) -> int:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        edge_tuple = tuple(sorted(graph.edges()))
        
        return hash((num_nodes, num_edges, edge_tuple))
    
    def _check_cache(self, graph: nx.Graph) -> bool:
        if not self.enable_caching:
            return False
        
        current_hash = self._compute_graph_hash(graph)
        return current_hash == self._graph_hash
    
    def _update_cache(self, graph: nx.Graph, centralities: Dict, priorities: Dict):
        if self.enable_caching:
            self._graph_hash = self._compute_graph_hash(graph)
            self._centrality_cache = centralities
            self._priority_cache = priorities
    
    def compute_degree_centrality(self, graph: nx.Graph) -> Dict[Any, float]:
        return nx.degree_centrality(graph)
    
    def compute_betweenness_centrality(self, 
                                      graph: nx.Graph,
                                      normalized: bool = True) -> Dict[Any, float]:
        return nx.betweenness_centrality(graph, normalized=normalized)
    
    def compute_closeness_centrality(self, graph: nx.Graph) -> Dict[Any, float]:
        if not nx.is_connected(graph.to_undirected()):
            print("Warning: the graph is disconnected; closeness centrality will be computed with connected components")
            return self._compute_closeness_for_disconnected_graph(graph)
        
        return nx.closeness_centrality(graph)
    
    def _compute_closeness_for_disconnected_graph(self, 
                                                 graph: nx.Graph) -> Dict[Any, float]:
        closeness = {}
        
        for node in graph.nodes():
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
        return nx.pagerank(graph, alpha=alpha, max_iter=max_iter)
    
    def compute_all_centralities(self, graph: nx.Graph) -> Dict[Any, Dict[str, float]]:
        if self._check_cache(graph):
            print("✓ Using cached centrality results")
            return self._centrality_cache
        
        print("⏳ Computing graph centrality metrics...")
        
        dc = self.compute_degree_centrality(graph)
        bc = self.compute_betweenness_centrality(graph)
        cc = self.compute_closeness_centrality(graph)
        pr = self.compute_pagerank_centrality(graph)
        
        centralities = {}
        for node in graph.nodes():
            centralities[node] = {
                'degree': dc.get(node, 0.0),
                'betweenness': bc.get(node, 0.0),
                'closeness': cc.get(node, 0.0),
                'pagerank': pr.get(node, 0.0)
            }
        
        print(f"✓ Finished computing centralities for {len(centralities)} nodes")
        
        return centralities
    
    def compute_protection_priority(self, 
                                   centralities: Dict[Any, Dict[str, float]]) -> Dict[Any, float]:
        priorities = {}
        
        for node, metrics in centralities.items():
            priority = sum(
                self.fusion_weights.get(metric, 0.0) * score
                for metric, score in metrics.items()
                if metric in self.fusion_weights
            )
            priorities[node] = priority
        
        if self.normalize and priorities:
            max_priority = max(priorities.values())
            min_priority = min(priorities.values())
            
            if max_priority > min_priority:
                for node in priorities:
                    priorities[node] = (priorities[node] - min_priority) /\
                                      (max_priority - min_priority)
            else:
                for node in priorities:
                    priorities[node] = 0.5
        
        return priorities
    
    def analyze_graph(self, graph: nx.Graph) -> Tuple[Dict, Dict]:
        centralities = self.compute_all_centralities(graph)
        
        priorities = self.compute_protection_priority(centralities)
        
        self._update_cache(graph, centralities, priorities)
        
        return centralities, priorities
    
    def identify_critical_nodes(self, 
                               priorities: Dict[Any, float],
                               top_k: int = None,
                               threshold: float = None) -> List[Any]:
        if top_k is not None:
            sorted_nodes = sorted(priorities.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
            return [node for node, _ in sorted_nodes[:top_k]]
        
        elif threshold is not None:
            return [node for node, priority in priorities.items() 
                   if priority >= threshold]
        
        else:
            mean_priority = np.mean(list(priorities.values()))
            return [node for node, priority in priorities.items() 
                   if priority > mean_priority]
    
    def get_priority_statistics(self, priorities: Dict[Any, float]) -> Dict[str, float]:
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
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("Warning: matplotlib is not installed, so visualization is unavailable")
            return
        
        pos = nx.spring_layout(graph, seed=42)
        
        node_colors = [priorities.get(node, 0.0) for node in graph.nodes()]
        
        plt.figure(figsize=(12, 8))
        
        nodes = nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=500,
            cmap=plt.cm.RdYlGn,
            vmin=0, vmax=1,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(graph, pos, alpha=0.3, width=1.5)
        
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        plt.colorbar(nodes, label='Protection Priority')
        
        plt.title('Node Protection Priority Visualization', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class LearnableCentralityFusion(nn.Module):
    
    def __init__(self, num_centrality_metrics: int = 4):
        super().__init__()
        
        self.fusion_weights = nn.Parameter(
            torch.ones(num_centrality_metrics) / num_centrality_metrics
        )
        
        self.attention_net = nn.Sequential(
            nn.Linear(num_centrality_metrics, num_centrality_metrics * 2),
            nn.ReLU(),
            nn.Linear(num_centrality_metrics * 2, num_centrality_metrics),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, centrality_scores: torch.Tensor) -> torch.Tensor:
        # weights = F.softmax(self.fusion_weights, dim=0)
        # priorities = torch.matmul(centrality_scores, weights)
        
        attention_weights = self.attention_net(centrality_scores)  # [num_nodes, num_metrics]
        priorities = torch.sum(centrality_scores * attention_weights, dim=-1)  # [num_nodes]
        
        return priorities
    
    def get_fusion_weights(self) -> Dict[str, float]:
        with torch.no_grad():
            weights = torch.softmax(self.fusion_weights, dim=0).cpu().numpy()
        
        metric_names = ['degree', 'betweenness', 'closeness', 'pagerank']
        return {name: float(weight) for name, weight in zip(metric_names, weights)}


def analyze_topology_protection(graph: nx.Graph, 
                               top_k: int = 5) -> Tuple[Dict, List]:
    analyzer = GraphCentralityAnalyzer()
    centralities, priorities = analyzer.analyze_graph(graph)
    critical_nodes = analyzer.identify_critical_nodes(priorities, top_k=top_k)
    
    return priorities, critical_nodes


__all__ = [
    'GraphCentralityAnalyzer',
    'LearnableCentralityFusion',
    'analyze_topology_protection'
]



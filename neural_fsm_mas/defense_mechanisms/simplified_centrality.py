"""
Simplified Centrality Analyzer

Uses only two centrality metrics:
1. Betweenness centrality, which controls information flow
2. PageRank centrality, which captures global importance

Author: Neural FSM Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional, Tuple


class SimplifiedCentralityAnalyzer(nn.Module):
    """Simplified graph centrality analyzer."""

    def __init__(
        self,
        w_betweenness: float = 0.6,
        w_pagerank: float = 0.4,
        learnable: bool = True,
        normalize: bool = True,
    ):
        """Initialize the analyzer."""
        super().__init__()
        self.normalize = normalize

        if learnable:
            self.w_betweenness = nn.Parameter(torch.tensor(w_betweenness))
            self.w_pagerank = nn.Parameter(torch.tensor(w_pagerank))
        else:
            self.register_buffer("w_betweenness", torch.tensor(w_betweenness))
            self.register_buffer("w_pagerank", torch.tensor(w_pagerank))

        self._cache = {}
        self._graph_hash = None

    def _compute_graph_hash(self, graph: nx.Graph) -> int:
        """Compute a hash for the graph."""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        edge_tuple = tuple(sorted(graph.edges()))
        return hash((num_nodes, num_edges, edge_tuple))

    def compute_betweenness_centrality(self, graph: nx.Graph) -> Dict[Any, float]:
        """Compute betweenness centrality."""
        return nx.betweenness_centrality(graph, normalized=True)

    def compute_pagerank_centrality(
        self,
        graph: nx.Graph,
        alpha: float = 0.85,
    ) -> Dict[Any, float]:
        """Compute PageRank centrality."""
        return nx.pagerank(graph, alpha=alpha, max_iter=100)

    def compute_priority_scores(
        self,
        graph: nx.Graph,
        return_components: bool = False,
    ) -> Dict[Any, float]:
        """Compute protection priority scores."""
        graph_hash = self._compute_graph_hash(graph)
        if graph_hash == self._graph_hash and "priorities" in self._cache:
            print("✓ Using cached centrality results")
            if return_components:
                return self._cache["priorities"], self._cache["components"]
            return self._cache["priorities"]

        print("⏳ Computing graph centrality metrics...")

        bc_scores = self.compute_betweenness_centrality(graph)
        pr_scores = self.compute_pagerank_centrality(graph)

        w_bc = torch.sigmoid(self.w_betweenness)
        w_pr = torch.sigmoid(self.w_pagerank)
        weight_sum = w_bc + w_pr
        w_bc = w_bc / weight_sum
        w_pr = w_pr / weight_sum

        w_bc_np = w_bc.detach().cpu().numpy()
        w_pr_np = w_pr.detach().cpu().numpy()

        priorities = {}
        components = {}

        for node in graph.nodes():
            bc_score = bc_scores.get(node, 0.0)
            pr_score = pr_scores.get(node, 0.0)
            priorities[node] = float(w_bc_np * bc_score + w_pr_np * pr_score)

            if return_components:
                components[node] = {
                    "betweenness": bc_score,
                    "pagerank": pr_score,
                }

        if self.normalize and priorities:
            max_p = max(priorities.values())
            min_p = min(priorities.values())
            if max_p > min_p:
                for node in priorities:
                    priorities[node] = (priorities[node] - min_p) / (max_p - min_p)
            else:
                for node in priorities:
                    priorities[node] = 0.5

        self._graph_hash = graph_hash
        self._cache["priorities"] = priorities
        self._cache["components"] = components

        print(f"✓ Finished computing priorities for {len(priorities)} nodes")

        if return_components:
            return priorities, components
        return priorities

    def priority_to_tensor(
        self,
        priorities: Dict[Any, float],
        node_order: list = None,
    ) -> torch.Tensor:
        """Convert the priority dictionary to a tensor."""
        if node_order is None:
            node_order = sorted(priorities.keys())

        priority_list = [priorities[node] for node in node_order]
        return torch.tensor(priority_list, dtype=torch.float32)

    def get_fusion_weights(self) -> Dict[str, float]:
        """Get the current fusion weights."""
        w_bc = torch.sigmoid(self.w_betweenness)
        w_pr = torch.sigmoid(self.w_pagerank)
        weight_sum = w_bc + w_pr
        w_bc = w_bc / weight_sum
        w_pr = w_pr / weight_sum

        return {
            "betweenness": w_bc.item(),
            "pagerank": w_pr.item(),
        }

    def identify_critical_nodes(
        self,
        priorities: Dict[Any, float],
        top_k: int = None,
        threshold: float = None,
    ) -> list:
        """Identify critical nodes."""
        if top_k is not None:
            sorted_nodes = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:top_k]]

        if threshold is not None:
            return [node for node, priority in priorities.items() if priority >= threshold]

        mean_priority = np.mean(list(priorities.values()))
        return [node for node, priority in priorities.items() if priority > mean_priority]

    def get_statistics(self, priorities: Dict[Any, float]) -> Dict[str, float]:
        """Get priority statistics."""
        values = list(priorities.values())
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    def visualize(
        self,
        graph: nx.Graph,
        priorities: Dict[Any, float],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """Visualize protection priorities."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib is not installed, so visualization is unavailable")
            return

        pos = nx.spring_layout(graph, seed=42)
        node_colors = [priorities.get(node, 0.0) for node in graph.nodes()]

        plt.figure(figsize=figsize)
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=500,
            cmap=plt.cm.RdYlGn,
            vmin=0,
            vmax=1,
            alpha=0.8,
        )
        nx.draw_networkx_edges(graph, pos, alpha=0.3, width=1.5)
        nx.draw_networkx_labels(graph, pos, font_size=10)
        plt.colorbar(nodes, label="Protection Priority (π)")

        weights = self.get_fusion_weights()
        plt.title(
            "Node Protection Priority\n"
            f'(w_BC={weights["betweenness"]:.3f}, '
            f'w_PR={weights["pagerank"]:.3f})',
            fontsize=14,
        )

        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


def compute_node_priorities(
    graph: nx.Graph,
    w_betweenness: float = 0.6,
    w_pagerank: float = 0.4,
) -> Dict[Any, float]:
    """Convenience helper for quickly computing node protection priorities."""
    analyzer = SimplifiedCentralityAnalyzer(
        w_betweenness=w_betweenness,
        w_pagerank=w_pagerank,
        learnable=False,
    )
    return analyzer.compute_priority_scores(graph)


__all__ = [
    "SimplifiedCentralityAnalyzer",
    "compute_node_priorities",
]

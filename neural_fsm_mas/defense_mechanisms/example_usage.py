"""
Defense Mechanisms Usage Example

Demonstrates how to use the simplified protection mechanisms.

Author: Neural FSM Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np

from neural_fsm_mas.defense_mechanisms import (
    SimplifiedCentralityAnalyzer,
    SimplifiedAnomalyDetector,
    TrustCalculator,
    MessageWeightCalculator,
    ProtectionConstrainedLoss,
    ProtectedTGN,
)

try:
    from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph
except ImportError:
    print("Warning: NeuralTemporalGraph not found, using the simplified version")

    class NeuralTemporalGraph(nn.Module):
        def __init__(self, feature_dim, hidden_dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),
            )

        def forward(self, x, edge_index, timestamps=None):
            return self.mlp(x)


def example_1_basic_usage():
    """Example 1: basic usage."""
    print("=" * 60)
    print("Example 1: basic usage")
    print("=" * 60)

    G = nx.karate_club_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[Step 1] Computing protection priorities...")
    centrality_analyzer = SimplifiedCentralityAnalyzer(
        w_betweenness=0.6,
        w_pagerank=0.4,
        learnable=True,
    )
    priorities = centrality_analyzer.compute_priority_scores(G)

    critical_nodes = centrality_analyzer.identify_critical_nodes(priorities, top_k=5)
    print(f"Critical nodes (Top-5): {critical_nodes}")

    stats = centrality_analyzer.get_statistics(priorities)
    print(f"Priority statistics: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    print("\n[Step 2] Anomaly detection...")
    anomaly_detector = SimplifiedAnomalyDetector(
        feature_dim=384,
        lambda_freq=0.3,
        lambda_semantic=0.7,
    )

    for node in G.nodes():
        for _ in range(10):
            anomaly_detector.update_history(
                agent_id=node,
                message_count=np.random.randint(3, 8),
                embedding=torch.randn(384),
            )

    anomaly_detector.update_history(0, message_count=50)
    anomaly_detector.update_history(1, embedding=torch.randn(384) * 10)

    anomaly_scores = anomaly_detector.batch_compute_anomaly_scores(list(G.nodes()))
    print(f"Anomaly scores: node0={anomaly_scores[0]:.3f}, node1={anomaly_scores[1]:.3f}")

    print("\n[Step 3] Computing trust scores...")
    trust_calculator = TrustCalculator()
    priority_tensor = torch.tensor([priorities[i] for i in sorted(G.nodes())])
    anomaly_tensor = torch.tensor([anomaly_scores[i] for i in sorted(G.nodes())])
    trust_scores = trust_calculator(anomaly_tensor, priority_tensor)
    print(f"Trust scores: node0={trust_scores[0]:.3f}, node1={trust_scores[1]:.3f}")
    print(f"Trust score range: [{trust_scores.min():.3f}, {trust_scores.max():.3f}]")

    print("\n[Step 4] Inspecting weights...")
    cent_weights = centrality_analyzer.get_fusion_weights()
    print(f"Centrality weights: BC={cent_weights['betweenness']:.3f}, PR={cent_weights['pagerank']:.3f}")

    anomaly_weights = anomaly_detector.get_fusion_weights()
    print(
        f"Anomaly-detection weights: "
        f"Freq={anomaly_weights['frequency']:.3f}, Semantic={anomaly_weights['semantic']:.3f}"
    )

    print("\n✅ Example 1 completed\n")


def example_2_integrated_training():
    """Example 2: integrated training."""
    print("=" * 60)
    print("Example 2: integrated training")
    print("=" * 60)

    num_nodes = 10
    feature_dim = 64
    hidden_dim = 128
    batch_size = 4
    num_classes = 3

    G = nx.erdos_renyi_graph(num_nodes, 0.3)
    print(f"Graph: {num_nodes} nodes, {G.number_of_edges()} edges")

    tgn = NeuralTemporalGraph(feature_dim=feature_dim, hidden_dim=hidden_dim)

    print("\n[Step 1] Creating ProtectedTGN...")
    protected_tgn = ProtectedTGN(
        tgn_model=tgn,
        feature_dim=feature_dim,
        graph=G,
        w_betweenness=0.6,
        w_pagerank=0.4,
        lambda_freq=0.3,
        lambda_semantic=0.7,
        weight_hidden_dim=64,
        lambda_protect=0.1,
        lambda_reg=0.01,
        learnable_weights=True,
    )

    node_features = torch.randn(num_nodes, feature_dim)
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    labels = torch.randint(0, num_classes, (batch_size,))

    print("\n[Step 2] Training...")
    optimizer = optim.Adam(protected_tgn.parameters(), lr=0.001)

    for epoch in range(5):
        output, messages, anomalies, priorities, weights = protected_tgn(
            node_features, edge_index, graph=G
        )
        predictions = output[:batch_size]
        loss, loss_dict = protected_tgn.compute_loss(
            predictions, labels, messages, anomalies, priorities, edge_index
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(protected_tgn.parameters(), 1.0)
        optimizer.step()

        print(f"Epoch {epoch + 1}/5:")
        print(f"  Total Loss: {loss_dict['total']:.4f}")
        print(f"  Task Loss: {loss_dict['task']:.4f}")
        print(f"  Protection Loss: {loss_dict['protection']:.4f}")

        if epoch == 4:
            weights_dict = protected_tgn.get_learned_weights()
            print(f"  Learned centrality weights: {weights_dict['centrality']}")
            print(f"  Learned anomaly weights: {weights_dict['anomaly']}")

    print("\n✅ Example 2 completed\n")


def example_3_message_filtering():
    """Example 3: message-filtering behavior."""
    print("=" * 60)
    print("Example 3: message-filtering behavior")
    print("=" * 60)

    weight_calc = MessageWeightCalculator(hidden_dim=64)

    trust_high = torch.tensor([1.8])
    priority_high = torch.tensor([0.9])
    weight1 = weight_calc(trust_high, priority_high)
    print(
        f"Scenario 1 (high trust -> high priority): "
        f"trust={trust_high.item():.2f}, priority={priority_high.item():.2f}"
    )
    print(f"  -> Message weight: {weight1.item():.3f} (should be close to 1, pass through)")

    trust_low = torch.tensor([0.3])
    priority_high = torch.tensor([0.9])
    weight2 = weight_calc(trust_low, priority_high)
    print(
        f"\nScenario 2 (low trust -> high priority): "
        f"trust={trust_low.item():.2f}, priority={priority_high.item():.2f}"
    )
    print(f"  -> Message weight: {weight2.item():.3f} (should be small, attenuated)")

    trust_low = torch.tensor([0.3])
    priority_low = torch.tensor([0.2])
    weight3 = weight_calc(trust_low, priority_low)
    print(
        f"\nScenario 3 (low trust -> low priority): "
        f"trust={trust_low.item():.2f}, priority={priority_low.item():.2f}"
    )
    print(f"  -> Message weight: {weight3.item():.3f} (moderate)")

    trust_high = torch.tensor([1.8])
    priority_low = torch.tensor([0.2])
    weight4 = weight_calc(trust_high, priority_low)
    print(
        f"\nScenario 4 (high trust -> low priority): "
        f"trust={trust_high.item():.2f}, priority={priority_low.item():.2f}"
    )
    print(f"  -> Message weight: {weight4.item():.3f} (should be close to 1, pass through)")

    print("\nAnalysis:")
    print("- For high-priority targets, messages from low-trust sources are strongly attenuated")
    print("- For low-priority targets, protection is relatively looser")
    print("- Messages from high-trust sources usually pass through normally")
    print("\n✅ Example 3 completed\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  NeuralFSM protection mechanism examples")
    print("=" * 60 + "\n")

    torch.manual_seed(42)
    np.random.seed(42)

    example_1_basic_usage()
    example_2_integrated_training()
    example_3_message_filtering()

    print("=" * 60)
    print("  All examples completed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()

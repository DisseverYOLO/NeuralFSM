"""
Defense Mechanisms Usage Example
保护机制使用示例

展示如何使用简化版保护机制

作者: Neural FSM Team
日期: 2025-10-28
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
    ProtectedTGN
)

# 导入TGN (假设已经实现)
try:
    from neural_fsm_mas.temporal_networks.neural_temporal_graph import NeuralTemporalGraph
except ImportError:
    print("警告: NeuralTemporalGraph未找到,使用简化版")
    # 简化的TGN模拟
    class NeuralTemporalGraph(nn.Module):
        def __init__(self, feature_dim, hidden_dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
        
        def forward(self, x, edge_index, timestamps=None):
            return self.mlp(x)


def example_1_basic_usage():
    """
    示例1: 基础使用
    """
    print("=" * 60)
    print("示例1: 基础使用")
    print("=" * 60)
    
    # 1. 创建图
    G = nx.karate_club_graph()
    print(f"图: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
    
    # 2. 计算保护优先级
    print("\n[步骤1] 计算保护优先级...")
    centrality_analyzer = SimplifiedCentralityAnalyzer(
        w_betweenness=0.6,
        w_pagerank=0.4,
        learnable=True
    )
    
    priorities = centrality_analyzer.compute_priority_scores(G)
    
    # 识别关键节点
    critical_nodes = centrality_analyzer.identify_critical_nodes(priorities, top_k=5)
    print(f"关键节点 (Top-5): {critical_nodes}")
    
    # 查看统计
    stats = centrality_analyzer.get_statistics(priorities)
    print(f"优先级统计: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    # 3. 异常检测
    print("\n[步骤2] 异常检测...")
    anomaly_detector = SimplifiedAnomalyDetector(
        feature_dim=384,
        lambda_freq=0.3,
        lambda_semantic=0.7
    )
    
    # 模拟历史数据
    for node in G.nodes():
        for _ in range(10):
            anomaly_detector.update_history(
                agent_id=node,
                message_count=np.random.randint(3, 8),
                embedding=torch.randn(384)
            )
    
    # 注入异常
    anomaly_detector.update_history(0, message_count=50)  # 频率异常
    anomaly_detector.update_history(1, embedding=torch.randn(384) * 10)  # 语义异常
    
    # 计算异常分数
    anomaly_scores = anomaly_detector.batch_compute_anomaly_scores(list(G.nodes()))
    print(f"异常分数: 节点0={anomaly_scores[0]:.3f}, 节点1={anomaly_scores[1]:.3f}")
    
    # 4. 计算信任分数
    print("\n[步骤3] 计算信任分数...")
    trust_calculator = TrustCalculator()
    
    priority_tensor = torch.tensor([priorities[i] for i in sorted(G.nodes())])
    anomaly_tensor = torch.tensor([anomaly_scores[i] for i in sorted(G.nodes())])
    
    trust_scores = trust_calculator(anomaly_tensor, priority_tensor)
    print(f"信任分数: 节点0={trust_scores[0]:.3f}, 节点1={trust_scores[1]:.3f}")
    print(f"信任分数范围: [{trust_scores.min():.3f}, {trust_scores.max():.3f}]")
    
    # 5. 查看学习到的权重
    print("\n[步骤4] 查看权重...")
    cent_weights = centrality_analyzer.get_fusion_weights()
    print(f"中心性权重: BC={cent_weights['betweenness']:.3f}, PR={cent_weights['pagerank']:.3f}")
    
    anomaly_weights = anomaly_detector.get_fusion_weights()
    print(f"异常检测权重: Freq={anomaly_weights['frequency']:.3f}, Semantic={anomaly_weights['semantic']:.3f}")
    
    print("\n✅ 示例1完成\n")


def example_2_integrated_training():
    """
    示例2: 集成训练
    """
    print("=" * 60)
    print("示例2: 集成训练")
    print("=" * 60)
    
    # 配置
    num_nodes = 10
    num_edges = 20
    feature_dim = 64
    hidden_dim = 128
    batch_size = 4
    num_classes = 3
    
    # 1. 创建图
    G = nx.erdos_renyi_graph(num_nodes, 0.3)
    print(f"图: {num_nodes}个节点, {G.number_of_edges()}条边")
    
    # 2. 创建TGN
    tgn = NeuralTemporalGraph(feature_dim=feature_dim, hidden_dim=hidden_dim)
    
    # 3. 创建ProtectedTGN
    print("\n[步骤1] 创建ProtectedTGN...")
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
        learnable_weights=True
    )
    
    # 4. 准备数据
    node_features = torch.randn(num_nodes, feature_dim)
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 5. 训练
    print("\n[步骤2] 训练...")
    optimizer = optim.Adam(protected_tgn.parameters(), lr=0.001)
    
    for epoch in range(5):
        # 前向传播
        output, messages, anomalies, priorities, weights = protected_tgn(
            node_features, edge_index, graph=G
        )
        
        # 模拟预测 (取前batch_size个节点)
        predictions = output[:batch_size]
        
        # 计算损失
        loss, loss_dict = protected_tgn.compute_loss(
            predictions, labels, messages, anomalies, priorities, edge_index
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(protected_tgn.parameters(), 1.0)
        optimizer.step()
        
        # 打印
        print(f"Epoch {epoch+1}/5:")
        print(f"  Total Loss: {loss_dict['total']:.4f}")
        print(f"  Task Loss: {loss_dict['task']:.4f}")
        print(f"  Protection Loss: {loss_dict['protection']:.4f}")
        
        # 查看学习到的权重
        if epoch == 4:
            weights_dict = protected_tgn.get_learned_weights()
            print(f"  学习到的中心性权重: {weights_dict['centrality']}")
            print(f"  学习到的异常权重: {weights_dict['anomaly']}")
    
    print("\n✅ 示例2完成\n")


def example_3_message_filtering():
    """
    示例3: 消息过滤效果演示
    """
    print("=" * 60)
    print("示例3: 消息过滤效果演示")
    print("=" * 60)
    
    # 创建权重计算器
    weight_calc = MessageWeightCalculator(hidden_dim=64)
    
    # 场景1: 高信任源 → 高优先级目标
    trust_high = torch.tensor([1.8])  # 信任分数高
    priority_high = torch.tensor([0.9])  # 优先级高
    weight1 = weight_calc(trust_high, priority_high)
    print(f"场景1 (高信任→高优先级): trust={trust_high.item():.2f}, priority={priority_high.item():.2f}")
    print(f"  → 消息权重: {weight1.item():.3f} (应该接近1,放行)")
    
    # 场景2: 低信任源 → 高优先级目标
    trust_low = torch.tensor([0.3])  # 信任分数低
    priority_high = torch.tensor([0.9])  # 优先级高
    weight2 = weight_calc(trust_low, priority_high)
    print(f"\n场景2 (低信任→高优先级): trust={trust_low.item():.2f}, priority={priority_high.item():.2f}")
    print(f"  → 消息权重: {weight2.item():.3f} (应该较小,衰减)")
    
    # 场景3: 低信任源 → 低优先级目标
    trust_low = torch.tensor([0.3])  # 信任分数低
    priority_low = torch.tensor([0.2])  # 优先级低
    weight3 = weight_calc(trust_low, priority_low)
    print(f"\n场景3 (低信任→低优先级): trust={trust_low.item():.2f}, priority={priority_low.item():.2f}")
    print(f"  → 消息权重: {weight3.item():.3f} (中等)")
    
    # 场景4: 高信任源 → 低优先级目标
    trust_high = torch.tensor([1.8])  # 信任分数高
    priority_low = torch.tensor([0.2])  # 优先级低
    weight4 = weight_calc(trust_high, priority_low)
    print(f"\n场景4 (高信任→低优先级): trust={trust_high.item():.2f}, priority={priority_low.item():.2f}")
    print(f"  → 消息权重: {weight4.item():.3f} (应该接近1,放行)")
    
    print("\n分析:")
    print("- 对于高优先级目标,低信任源的消息被显著衰减")
    print("- 对于低优先级目标,保护相对宽松")
    print("- 高信任源的消息通常能正常传递")
    
    print("\n✅ 示例3完成\n")


def main():
    """
    运行所有示例
    """
    print("\n" + "=" * 60)
    print("  NeuralFSM 保护机制使用示例")
    print("=" * 60 + "\n")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行示例
    example_1_basic_usage()
    example_2_integrated_training()
    example_3_message_filtering()
    
    print("=" * 60)
    print("  所有示例完成! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()


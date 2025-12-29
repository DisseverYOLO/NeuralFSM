"""
Defense Mechanisms for Neural FSM Multi-Agent System
神经FSM多智能体系统防御机制 - 简化版

核心组件:
1. SimplifiedCentralityAnalyzer - 图中心性分析 (BC + PageRank)
2. SimplifiedAnomalyDetector - 异常检测 (频率 + 语义)
3. TrustCalculator - 信任分数计算
4. MessageWeightCalculator - 消息权重计算
5. ProtectionConstrainedLoss - 保护约束损失
6. ProtectedTGN - 集成保护的TGN

核心公式:
- 保护优先级: π(i) = w_BC · BC(i) + w_PR · PR(i)
- 异常分数: α(i,t) = λ_freq · α_freq + λ_semantic · α_semantic
- 信任分数: trust(i,t) = (1 - α(i,t)) · (1 + π(i))
- 消息权重: w_{i→j} = MLP([trust(i), π(j)])
- 总损失: L = L_task + λ_protect · L_protect + λ_reg · L_reg

作者: Neural FSM Team
日期: 2025-10-28
版本: 1.0 (简化版)
"""

from .simplified_centrality import (
    SimplifiedCentralityAnalyzer,
    compute_node_priorities
)

from .simplified_anomaly import (
    SimplifiedAnomalyDetector,
    create_anomaly_detector
)

from .trust_calculator import (
    TrustCalculator,
    compute_trust_scores
)

from .message_weight_calculator import (
    MessageWeightCalculator,
    SimpleMessageWeightCalculator,
    compute_message_weights
)

from .protection_loss import (
    ProtectionConstrainedLoss,
    AdaptiveProtectionLoss
)

from .protected_tgn import (
    ProtectedTGN
)

from .advanced_attack_injector import (
    AdvancedAttackInjector,
    AttackConfig,
    create_frequency_attacker,
    create_semantic_attacker,
    create_byzantine_attacker,
    create_sybil_attacker,
    create_selfish_attacker,
    create_mixed_attacker
)


__all__ = [
    # 核心组件
    'SimplifiedCentralityAnalyzer',
    'SimplifiedAnomalyDetector',
    'TrustCalculator',
    'MessageWeightCalculator',
    'SimpleMessageWeightCalculator',
    'ProtectionConstrainedLoss',
    'AdaptiveProtectionLoss',
    'ProtectedTGN',
    
    # 攻击模拟（高级版）
    'AdvancedAttackInjector',
    'AttackConfig',
    'create_frequency_attacker',
    'create_semantic_attacker',
    'create_byzantine_attacker',
    'create_sybil_attacker',
    'create_selfish_attacker',
    'create_mixed_attacker',
    
    # 便捷函数
    'compute_node_priorities',
    'create_anomaly_detector',
    'compute_trust_scores',
    'compute_message_weights',
]


__version__ = '1.0.0'
__author__ = 'Neural FSM Team'

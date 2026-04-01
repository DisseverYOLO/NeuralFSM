"""
Defense Mechanisms for Neural FSM Multi-Agent System
Defense mechanisms for the Neural FSM multi-agent system - simplified edition

Core components:
1. SimplifiedCentralityAnalyzer - graph centrality analysis (BC + PageRank)
2. SimplifiedAnomalyDetector - anomaly detection (frequency + semantics)
3. TrustCalculator - trust score computation
4. MessageWeightCalculator - message weight computation
5. ProtectionConstrainedLoss - protection-constrained loss
6. ProtectedTGN - protection-integrated TGN

Core formulas:
- Protection priority: π(i) = w_BC · BC(i) + w_PR · PR(i)
- Anomaly score: α(i,t) = λ_freq · α_freq + λ_semantic · α_semantic
- Trust score: trust(i,t) = (1 - α(i,t)) · (1 + π(i))
- Message weight: w_{i→j} = MLP([trust(i), π(j)])
- Total loss: L = L_task + λ_protect · L_protect + λ_reg · L_reg

Author: Neural FSM Team
Date: 2025-10-28
Version: 1.0 (simplified edition)
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
    # Core components
    'SimplifiedCentralityAnalyzer',
    'SimplifiedAnomalyDetector',
    'TrustCalculator',
    'MessageWeightCalculator',
    'SimpleMessageWeightCalculator',
    'ProtectionConstrainedLoss',
    'AdaptiveProtectionLoss',
    'ProtectedTGN',
    
    # Attack simulation (advanced edition)
    'AdvancedAttackInjector',
    'AttackConfig',
    'create_frequency_attacker',
    'create_semantic_attacker',
    'create_byzantine_attacker',
    'create_sybil_attacker',
    'create_selfish_attacker',
    'create_mixed_attacker',
    
    # Convenience functions
    'compute_node_priorities',
    'create_anomaly_detector',
    'compute_trust_scores',
    'compute_message_weights',
]


__version__ = '1.0.0'
__author__ = 'Neural FSM Team'

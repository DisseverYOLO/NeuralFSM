"""
Neural FSM Multi-Agent System (Neural FSM-MAS)
神经有限状态机多智能体系统

完整集成MetaAgent的FSM生成能力和GDesigner的TGN学习能力
支持：
1. 自动生成智能体角色和状态描述
2. 随机采样状态转移拓扑和Listening通信拓扑  
3. 使用TGN学习最优路径
4. 支持多数据集训练
"""

# FSM集成模块
from neural_fsm_mas.fsm_integration.fsm_mas_generator import (
    FSMMultiAgentSystemGenerator,
    create_fsm_mas_generator,
    generate_and_learn_fsm_mas,
    generate_learn_and_execute_fsm_mas
)

# 时间图网络
from neural_fsm_mas.temporal_networks.neural_temporal_graph import (
    NeuralTemporalGraph, 
    MultiLayerPerceptron, 
    CompatibilityGraphNetwork,
    StateTransitionLearner
)

# 智能体拓扑
from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager
from neural_fsm_mas.agent_topology.agent_node import AgentExecutionNode, ConcreteAgentExecutionNode
from neural_fsm_mas.agent_topology.fsm_state_manager import FSMStateManager, FSMState
from neural_fsm_mas.agent_topology.fsm_validator import FSMValidator, FSMTransitionConditionGenerator, validate_fsm
from neural_fsm_mas.agent_topology.fsm_executor import FSMExecutor, create_fsm_executor

# 推理智能体
from neural_fsm_mas.reasoning_agents.agent_factory import ReasoningAgentFactory, ReasoningAgentRegistry
from neural_fsm_mas.reasoning_agents.mathematical_reasoning_agent import MathematicalReasoningAgent
from neural_fsm_mas.reasoning_agents.analytical_reasoning_agent import AnalyticalReasoningAgent
from neural_fsm_mas.reasoning_agents.decision_making_agent import DecisionMakingAgent
from neural_fsm_mas.reasoning_agents.code_generation_agent import CodeGenerationAgent
from neural_fsm_mas.reasoning_agents.adversarial_reasoning_agent import AdversarialReasoningAgent

# 领域提示
from neural_fsm_mas.domain_prompts.prompt_manager import (
    DomainPromptManager, 
    DomainPromptRegistry,
    MMLUDomainPromptSet,
    GSM8KDomainPromptSet
)

# 数据处理
from neural_fsm_mas.training_data.mmlu_data_processor import (
    MMLUDataProcessor,
    create_mmlu_processor,
    prepare_mmlu_training_data
)
from neural_fsm_mas.training_data.unified_data_processor import UnifiedDataProcessor
from neural_fsm_mas.training_data.answer_validator import AnswerValidator, create_answer_validator

# 成本追踪
from neural_fsm_mas.utils.llm_cost_tracker import (
    LLMCostTracker,
    LLMPricingRegistry,
    create_cost_tracker,
    print_pricing_table
)

# 损失函数
from neural_fsm_mas.losses.cost_loss import (
    CostLoss,
    AdaptiveCostLoss,
    CostRegularizedLoss,
    estimate_baseline_cost
)

# ✨ 核心集成模块（新增）
from neural_fsm_mas.core_integration import (
    EnhancedFSMIntegration,
    create_enhanced_integration,
    integrate_four_objective_loss,
    StateAgentCorrespondenceLearner
)

# ✨ FSM缓存管理（新增）
from neural_fsm_mas.fsm_cache_manager import (
    FSMCacheManager,
    create_cache_manager
)

# ✨ 提示词优化（新增）- 实验1专用
from neural_fsm_mas.prompt_optimization import (
    StateDescriptionOptimizer,
    create_prompt_optimizer
)

__version__ = "2.0.0"
__author__ = "Neural FSM-MAS Team"

__all__ = [
    # FSM集成核心
    'FSMMultiAgentSystemGenerator',
    'create_fsm_mas_generator',
    'generate_and_learn_fsm_mas',
    'generate_learn_and_execute_fsm_mas',
    
    # 时间图网络
    'NeuralTemporalGraph',
    'MultiLayerPerceptron', 
    'CompatibilityGraphNetwork',
    'StateTransitionLearner',
    
    # 智能体拓扑
    'MultiAgentTopologyManager',
    'AgentExecutionNode',
    'ConcreteAgentExecutionNode',
    'FSMStateManager',
    'FSMState',
    'FSMValidator',
    'FSMTransitionConditionGenerator',
    'FSMExecutor',
    'validate_fsm',
    'create_fsm_executor',
    
    # 推理智能体
    'ReasoningAgentFactory',
    'ReasoningAgentRegistry',
    'MathematicalReasoningAgent',
    'AnalyticalReasoningAgent',
    'DecisionMakingAgent',
    'CodeGenerationAgent',
    'AdversarialReasoningAgent',
    
    # 领域提示
    'DomainPromptManager',
    'DomainPromptRegistry',
    'MMLUDomainPromptSet',
    'GSM8KDomainPromptSet',
    
    # 数据处理
    'MMLUDataProcessor',
    'create_mmlu_processor',
    'prepare_mmlu_training_data',
    'UnifiedDataProcessor',  # ✨ 统一数据处理器（支持所有6个数据集）
    'AnswerValidator',  # ✨ 答案验证器
    'create_answer_validator',  # ✨ 答案验证器工厂函数
    
    # 成本追踪
    'LLMCostTracker',
    'LLMPricingRegistry',
    'create_cost_tracker',
    'print_pricing_table',
    
    # 损失函数
    'CostLoss',
    'AdaptiveCostLoss',
    'CostRegularizedLoss',
    'estimate_baseline_cost',
    
    # ✨ 核心集成（新增）
    'EnhancedFSMIntegration',
    'create_enhanced_integration',
    'integrate_four_objective_loss',
    'StateAgentCorrespondenceLearner',
    
    # ✨ FSM缓存管理（新增）
    'FSMCacheManager',
    'create_cache_manager',
    
    # ✨ 提示词优化（新增）- 实验1专用
    'StateDescriptionOptimizer',
    'create_prompt_optimizer',
]
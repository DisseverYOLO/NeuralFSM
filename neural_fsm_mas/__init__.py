"""
Neural finite-state-machine multi-agent system


Supports:
1. Automatically generating agent roles and state descriptions
2. Randomly sampling state-transition topology and listening communication topology
3. Using TGN to learn optimal paths
4. Supporting multi-dataset training
"""

# FSM integration modules
from neural_fsm_mas.fsm_integration.fsm_mas_generator import (
    FSMMultiAgentSystemGenerator,
    create_fsm_mas_generator,
    generate_and_learn_fsm_mas,
    generate_learn_and_execute_fsm_mas
)

# Temporal graph networks
from neural_fsm_mas.temporal_networks.neural_temporal_graph import (
    NeuralTemporalGraph, 
    MultiLayerPerceptron, 
    CompatibilityGraphNetwork,
    StateTransitionLearner
)

# Agent topology
from neural_fsm_mas.agent_topology.multi_agent_topology import MultiAgentTopologyManager
from neural_fsm_mas.agent_topology.agent_node import AgentExecutionNode, ConcreteAgentExecutionNode
from neural_fsm_mas.agent_topology.fsm_state_manager import FSMStateManager, FSMState
from neural_fsm_mas.agent_topology.fsm_validator import FSMValidator, FSMTransitionConditionGenerator, validate_fsm
from neural_fsm_mas.agent_topology.fsm_executor import FSMExecutor, create_fsm_executor

# Reasoning agents
from neural_fsm_mas.reasoning_agents.agent_factory import ReasoningAgentFactory, ReasoningAgentRegistry
from neural_fsm_mas.reasoning_agents.mathematical_reasoning_agent import MathematicalReasoningAgent
from neural_fsm_mas.reasoning_agents.analytical_reasoning_agent import AnalyticalReasoningAgent
from neural_fsm_mas.reasoning_agents.decision_making_agent import DecisionMakingAgent
from neural_fsm_mas.reasoning_agents.code_generation_agent import CodeGenerationAgent
from neural_fsm_mas.reasoning_agents.adversarial_reasoning_agent import AdversarialReasoningAgent

# Domain prompts
from neural_fsm_mas.domain_prompts.prompt_manager import (
    DomainPromptManager, 
    DomainPromptRegistry,
    MMLUDomainPromptSet,
    GSM8KDomainPromptSet
)

# Data processing
from neural_fsm_mas.training_data.mmlu_data_processor import (
    MMLUDataProcessor,
    create_mmlu_processor,
    prepare_mmlu_training_data
)
from neural_fsm_mas.training_data.unified_data_processor import UnifiedDataProcessor
from neural_fsm_mas.training_data.answer_validator import AnswerValidator, create_answer_validator

# Cost tracking
from neural_fsm_mas.utils.llm_cost_tracker import (
    LLMCostTracker,
    LLMPricingRegistry,
    create_cost_tracker,
    print_pricing_table
)

# Loss functions
from neural_fsm_mas.losses.cost_loss import (
    CostLoss,
    AdaptiveCostLoss,
    CostRegularizedLoss,
    estimate_baseline_cost
)

#  Core integration module 
from neural_fsm_mas.core_integration import (
    EnhancedFSMIntegration,
    create_enhanced_integration,
    integrate_four_objective_loss,
    StateAgentCorrespondenceLearner
)

#  FSM cache management 
from neural_fsm_mas.fsm_cache_manager import (
    FSMCacheManager,
    create_cache_manager
)

#  Prompt optimization 
from neural_fsm_mas.prompt_optimization import (
    StateDescriptionOptimizer,
    create_prompt_optimizer
)

__version__ = "2.0.0"
__author__ = "Neural FSM-MAS Team"

__all__ = [
    # FSM integration core
    'FSMMultiAgentSystemGenerator',
    'create_fsm_mas_generator',
    'generate_and_learn_fsm_mas',
    'generate_learn_and_execute_fsm_mas',
    
    # Temporal graph networks
    'NeuralTemporalGraph',
    'MultiLayerPerceptron', 
    'CompatibilityGraphNetwork',
    'StateTransitionLearner',
    
    # Agent topology
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
    
    # Reasoning agents
    'ReasoningAgentFactory',
    'ReasoningAgentRegistry',
    'MathematicalReasoningAgent',
    'AnalyticalReasoningAgent',
    'DecisionMakingAgent',
    'CodeGenerationAgent',
    'AdversarialReasoningAgent',
    
    # Domain prompts
    'DomainPromptManager',
    'DomainPromptRegistry',
    'MMLUDomainPromptSet',
    'GSM8KDomainPromptSet',
    
    # Data processing
    'MMLUDataProcessor',
    'create_mmlu_processor',
    'prepare_mmlu_training_data',
    'UnifiedDataProcessor',  
    'AnswerValidator',  
    'create_answer_validator', 
    
    # Cost tracking
    'LLMCostTracker',
    'LLMPricingRegistry',
    'create_cost_tracker',
    'print_pricing_table',
    
    # Loss functions
    'CostLoss',
    'AdaptiveCostLoss',
    'CostRegularizedLoss',
    'estimate_baseline_cost',
    
    #  Core integration
    'EnhancedFSMIntegration',
    'create_enhanced_integration',
    'integrate_four_objective_loss',
    'StateAgentCorrespondenceLearner',
    
    # FSM cache management 
    'FSMCacheManager',
    'create_cache_manager',
    
    #  Prompt optimization 
    'StateDescriptionOptimizer',
    'create_prompt_optimizer',
]

"""
Reasoning Agents Module
Reasoning agents module

Contains implementations of multiple reasoning agent types
"""

from .agent_factory import ReasoningAgentFactory, ReasoningAgentRegistry
from .mathematical_reasoning_agent import MathematicalReasoningAgent
from .analytical_reasoning_agent import AnalyticalReasoningAgent
from .decision_making_agent import DecisionMakingAgent  # Imported only from decision_making_agent.py
from .code_generation_agent import CodeGenerationAgent
from .adversarial_reasoning_agent import AdversarialReasoningAgent

__all__ = [
    'ReasoningAgentFactory',
    'ReasoningAgentRegistry',
    'MathematicalReasoningAgent',
    'AnalyticalReasoningAgent',
    'DecisionMakingAgent',
    'CodeGenerationAgent',
    'AdversarialReasoningAgent'
]

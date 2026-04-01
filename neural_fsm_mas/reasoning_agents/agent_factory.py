"""
Reasoning Agent Factory for Multi-Agent System
Reasoning agent factory

Adapted from the original AgentRegistry implementation and optimized for the MetaAgent project
Supports dynamic creation and management of different reasoning agent types
"""

from typing import Type, Dict, Any, Optional
import sys
from pathlib import Path

# Add MetaAgent path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from neural_fsm_mas.agent_topology.agent_node import AgentExecutionNode


class ReasoningAgentRegistry:
    """
    Reasoning agent registry

    Manages registration and creation of different reasoning agent types
    """
    _agent_registry: Dict[str, Type[AgentExecutionNode]] = {}
    
    @classmethod
    def register_agent_type(cls, agent_type_name: str):
        """Decorator for registering agent types."""
        def decorator(agent_class: Type[AgentExecutionNode]):
            cls._agent_registry[agent_type_name] = agent_class
            return agent_class
        return decorator
    
    @classmethod
    def get_registered_agent_types(cls):
        """Get all registered agent types."""
        return list(cls._agent_registry.keys())
    
    @classmethod
    def create_agent(cls, agent_type_name: str, *args, **kwargs) -> AgentExecutionNode:
        """Create an agent of the specified type."""
        if agent_type_name not in cls._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type_name}. Available types: {list(cls._agent_registry.keys())}")
        
        agent_class = cls._agent_registry[agent_type_name]
        return agent_class(*args, **kwargs)
    
    @classmethod
    def get_agent_class(cls, agent_type_name: str) -> Type[AgentExecutionNode]:
        """Get the agent class."""
        if agent_type_name not in cls._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type_name}")
        return cls._agent_registry[agent_type_name]


class ReasoningAgentFactory:
    """
    Reasoning agent factory

    Provides convenient interfaces for creating agents
    """
    
    @staticmethod
    def create_agent(agent_type: str, **config_params) -> AgentExecutionNode:
        """
        Create a reasoning agent.
        
        Args:
            agent_type: Agent type name
            **config_params: Agent configuration parameters
            
        Returns:
            Created agent instance
        """
        return ReasoningAgentRegistry.create_agent(agent_type, **config_params)
    
    @staticmethod
    def get_available_agent_types() -> list:
        """Get available agent types."""
        return ReasoningAgentRegistry.get_registered_agent_types()
    
    @staticmethod
    def register_custom_agent(agent_type_name: str, agent_class: Type[AgentExecutionNode]):
        """Register a custom agent type."""
        ReasoningAgentRegistry._agent_registry[agent_type_name] = agent_class


# Import and register concrete agent implementations
from neural_fsm_mas.reasoning_agents.mathematical_reasoning_agent import MathematicalReasoningAgent
from neural_fsm_mas.reasoning_agents.analytical_reasoning_agent import AnalyticalReasoningAgent
from neural_fsm_mas.reasoning_agents.decision_making_agent import DecisionMakingAgent
from neural_fsm_mas.reasoning_agents.code_generation_agent import CodeGenerationAgent
from neural_fsm_mas.reasoning_agents.adversarial_reasoning_agent import AdversarialReasoningAgent


# Export main classes and functions
__all__ = [
    'ReasoningAgentRegistry', 
    'ReasoningAgentFactory'
]

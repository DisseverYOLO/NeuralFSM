"""
Reasoning Agent Factory for Multi-Agent System
推理智能体工厂

适配自原始AgentRegistry实现，专门为MetaAgent项目优化
支持动态创建和管理不同类型的推理智能体
"""

from typing import Type, Dict, Any, Optional
import sys
from pathlib import Path

# 添加MetaAgent路径
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from neural_fsm_mas.agent_topology.agent_node import AgentExecutionNode


class ReasoningAgentRegistry:
    """
    推理智能体注册表
    
    管理不同类型推理智能体的注册和创建
    """
    _agent_registry: Dict[str, Type[AgentExecutionNode]] = {}
    
    @classmethod
    def register_agent_type(cls, agent_type_name: str):
        """注册智能体类型的装饰器"""
        def decorator(agent_class: Type[AgentExecutionNode]):
            cls._agent_registry[agent_type_name] = agent_class
            return agent_class
        return decorator
    
    @classmethod
    def get_registered_agent_types(cls):
        """获取所有已注册的智能体类型"""
        return list(cls._agent_registry.keys())
    
    @classmethod
    def create_agent(cls, agent_type_name: str, *args, **kwargs) -> AgentExecutionNode:
        """创建指定类型的智能体"""
        if agent_type_name not in cls._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type_name}. Available types: {list(cls._agent_registry.keys())}")
        
        agent_class = cls._agent_registry[agent_type_name]
        return agent_class(*args, **kwargs)
    
    @classmethod
    def get_agent_class(cls, agent_type_name: str) -> Type[AgentExecutionNode]:
        """获取智能体类"""
        if agent_type_name not in cls._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type_name}")
        return cls._agent_registry[agent_type_name]


class ReasoningAgentFactory:
    """
    推理智能体工厂
    
    提供便捷的智能体创建接口
    """
    
    @staticmethod
    def create_agent(agent_type: str, **config_params) -> AgentExecutionNode:
        """
        创建推理智能体
        
        Args:
            agent_type: 智能体类型名称
            **config_params: 智能体配置参数
            
        Returns:
            创建的智能体实例
        """
        return ReasoningAgentRegistry.create_agent(agent_type, **config_params)
    
    @staticmethod
    def get_available_agent_types() -> list:
        """获取可用的智能体类型"""
        return ReasoningAgentRegistry.get_registered_agent_types()
    
    @staticmethod
    def register_custom_agent(agent_type_name: str, agent_class: Type[AgentExecutionNode]):
        """注册自定义智能体类型"""
        ReasoningAgentRegistry._agent_registry[agent_type_name] = agent_class


# 导入并注册具体的智能体实现
from neural_fsm_mas.reasoning_agents.mathematical_reasoning_agent import MathematicalReasoningAgent
from neural_fsm_mas.reasoning_agents.analytical_reasoning_agent import AnalyticalReasoningAgent
from neural_fsm_mas.reasoning_agents.decision_making_agent import DecisionMakingAgent
from neural_fsm_mas.reasoning_agents.code_generation_agent import CodeGenerationAgent
from neural_fsm_mas.reasoning_agents.adversarial_reasoning_agent import AdversarialReasoningAgent


# 导出主要类和函数
__all__ = [
    'ReasoningAgentRegistry', 
    'ReasoningAgentFactory'
]

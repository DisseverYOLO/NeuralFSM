"""
Agent Execution Node for Multi-Agent System
智能体执行节点

适配自原始Node实现，专门为MetaAgent项目优化
支持时间敏感的记忆管理和智能体交互
"""

import shortuuid
from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import warnings
import time
import torch
import asyncio
import sys
from pathlib import Path

from baseclass.LLM import LLM

# 添加MetaAgent路径
sys.path.append(str(Path(__file__).parent.parent.parent))


class AgentExecutionNode(ABC):
    """
    智能体执行节点
    
    这个类封装了多智能体系统中单个智能体的执行逻辑，管理智能体间的连接、
    输入输出处理以及执行分配的操作。支持个体和聚合处理模式。
    
    核心功能：
    1. 管理智能体间的空间和时间连接
    2. 处理智能体的输入输出和执行逻辑
    3. 维护时间敏感的交互记忆
    4. 支持异步执行和推理
    
    属性:
        node_id (str): 节点的唯一标识符
        agent_role (str): 智能体角色，用于特定操作
        spatial_predecessors (List[AgentExecutionNode]): 空间前驱节点
        spatial_successors (List[AgentExecutionNode]): 空间后继节点
        temporal_predecessors (List[AgentExecutionNode]): 时间前驱节点
        temporal_successors (List[AgentExecutionNode]): 时间后继节点
        execution_inputs (List[Any]): 待处理的输入
        execution_outputs (List[Any]): 执行后产生的结果
        raw_task_inputs (List[Any]): 原始输入，包含问题或任务
        interaction_memory (Dict[str,List[Any]]): 上一时间戳的输入输出记忆
    """

    def __init__(self, 
                 node_id: Optional[str] = None,
                 agent_role: str = "",
                 domain: str = "", 
                 llm_name: str = "",
                 **kwargs):
        """
        初始化智能体执行节点
        """
        self.node_id: str = node_id if node_id is not None else shortuuid.ShortUUID().random(length=6)
        self.agent_role: str = agent_role
        self.task_domain: str = domain
        self.language_model_name: str = llm_name
        
        # 连接管理
        self.spatial_predecessors: List['AgentExecutionNode'] = []
        self.spatial_successors: List['AgentExecutionNode'] = []
        self.temporal_predecessors: List['AgentExecutionNode'] = []
        self.temporal_successors: List['AgentExecutionNode'] = []
        
        # 执行相关
        self.execution_inputs: List[Any] = []
        self.execution_outputs: List[Any] = []
        self.raw_task_inputs: List[Any] = []
        self.agent_role_description = ""
        self.interaction_memory: Dict[str, List[Any]] = {
            'inputs': [], 'outputs': [], 'raw_inputs': []
        }
        
        # 时间敏感的记忆管理
        self.interaction_memory_history: List[Dict[str, Any]] = []
        self.interaction_timestamps: List[float] = []
        self.current_interaction_round: int = 0
        self.memory_storage_capacity: int = 100
        
        # 额外配置参数
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def execution_node_name(self):
        """获取执行节点名称"""
        return self.__class__.__name__
    
    def add_predecessor_connection(self, predecessor_node: 'AgentExecutionNode', connection_type='spatial'):
        """添加前驱连接"""
        if connection_type == 'spatial' and predecessor_node not in self.spatial_predecessors:
            self.spatial_predecessors.append(predecessor_node)
            predecessor_node.spatial_successors.append(self)
        elif connection_type == 'temporal' and predecessor_node not in self.temporal_predecessors:
            self.temporal_predecessors.append(predecessor_node)
            predecessor_node.temporal_successors.append(self)

    def add_successor_connection(self, successor_node: 'AgentExecutionNode', connection_type='spatial'):
        """添加后继连接"""
        if connection_type == 'spatial' and successor_node not in self.spatial_successors:
            self.spatial_successors.append(successor_node)
            successor_node.spatial_predecessors.append(self)
        elif connection_type == 'temporal' and successor_node not in self.temporal_successors:
            self.temporal_successors.append(successor_node)
            successor_node.temporal_predecessors.append(self)

    def remove_predecessor_connection(self, predecessor_node: 'AgentExecutionNode', connection_type='spatial'):
        """移除前驱连接"""
        if connection_type == 'spatial' and predecessor_node in self.spatial_predecessors:
            self.spatial_predecessors.remove(predecessor_node)
            predecessor_node.spatial_successors.remove(self)
        elif connection_type == 'temporal' and predecessor_node in self.temporal_predecessors:
            self.temporal_predecessors.remove(predecessor_node)
            predecessor_node.temporal_successors.remove(self)

    def remove_successor_connection(self, successor_node: 'AgentExecutionNode', connection_type='spatial'):
        """移除后继连接"""
        if connection_type == 'spatial' and successor_node in self.spatial_successors:
            self.spatial_successors.remove(successor_node)
            successor_node.spatial_predecessors.remove(self)
        elif connection_type == 'temporal' and successor_node in self.temporal_successors:
            self.temporal_successors.remove(successor_node)
            successor_node.temporal_predecessors.remove(self)

    def clear_spatial_connections(self):
        """清除空间连接"""
        self.spatial_predecessors: List['AgentExecutionNode'] = []
        self.spatial_successors: List['AgentExecutionNode'] = []

    def clear_temporal_connections(self):
        """清除时间连接"""
        self.temporal_predecessors: List['AgentExecutionNode'] = []
        self.temporal_successors: List['AgentExecutionNode'] = []        
    
    def update_interaction_memory(self):
        """更新交互记忆，支持时间敏感的记忆管理"""
        # 更新基础记忆
        self.interaction_memory['inputs'] = self.execution_inputs.copy() if self.execution_inputs else []
        self.interaction_memory['outputs'] = self.execution_outputs.copy() if self.execution_outputs else []
        self.interaction_memory['raw_inputs'] = self.raw_task_inputs.copy() if self.raw_task_inputs else []
        
        # 创建时间敏感的记忆快照
        current_timestamp = time.time()
        memory_snapshot = {
            'interaction_round': self.current_interaction_round,
            'timestamp': current_timestamp,
            'inputs': self.execution_inputs.copy() if self.execution_inputs else [],
            'outputs': self.execution_outputs.copy() if self.execution_outputs else [],
            'raw_inputs': self.raw_task_inputs.copy() if self.raw_task_inputs else [],
            'spatial_connection_count': len(self.spatial_predecessors),
            'temporal_connection_count': len(self.temporal_predecessors)
        }
        
        # 添加到历史记忆
        self.interaction_memory_history.append(memory_snapshot)
        self.interaction_timestamps.append(current_timestamp)
        
        # 记忆容量管理：如果超过容量限制，移除最旧的记忆
        if len(self.interaction_memory_history) > self.memory_storage_capacity:
            self.interaction_memory_history.pop(0)
            self.interaction_timestamps.pop(0)
    
    def get_recent_interaction_memory(self, num_recent_interactions: int = 5) -> List[Dict[str, Any]]:
        """获取最近的交互记忆"""
        return self.interaction_memory_history[-num_recent_interactions:] if self.interaction_memory_history else []
    
    def get_memory_by_interaction_round(self, round_number: int) -> Optional[Dict[str, Any]]:
        """根据交互轮次获取记忆"""
        for memory in reversed(self.interaction_memory_history):
            if memory['interaction_round'] == round_number:
                return memory
        return None
    
    def get_temporal_interaction_context(self, time_window_seconds: float = 60.0) -> List[Dict[str, Any]]:
        """获取指定时间窗口内的交互记忆上下文"""
        current_timestamp = time.time()
        temporal_memories = []
        
        for i, timestamp in enumerate(self.interaction_timestamps):
            if current_timestamp - timestamp <= time_window_seconds:
                temporal_memories.append(self.interaction_memory_history[i])
        
        return temporal_memories
    
    def reset_interaction_memory(self):
        """重置交互记忆"""
        self.interaction_memory_history.clear()
        self.interaction_timestamps.clear()
        self.current_interaction_round = 0
        self.interaction_memory = {'inputs': [], 'outputs': [], 'raw_inputs': []}
    
    def set_current_interaction_round(self, round_number: int):
        """设置当前交互轮次"""
        self.current_interaction_round = round_number
    
    def add_predecessor_message(self, message: str):
        """
        添加前序智能体的消息（用于FSM监听机制）
        
        Args:
            message: 前序智能体的输出消息
        """
        # 将消息添加到执行输入中
        if not hasattr(self, 'predecessor_messages'):
            self.predecessor_messages = []
        
        self.predecessor_messages.append(message)
    
    def get_predecessor_messages(self) -> List[str]:
        """获取所有前序智能体的消息"""
        if not hasattr(self, 'predecessor_messages'):
            self.predecessor_messages = []
        return self.predecessor_messages
    
    def clear_predecessor_messages(self):
        """清空前序消息"""
        self.predecessor_messages = []

    def get_spatial_interaction_info(self) -> Dict[str, Dict]:
        """获取空间交互信息"""
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                spatial_info[predecessor.node_id] = {
                    'agent_role': predecessor.agent_role,
                    'execution_outputs': predecessor.execution_outputs,
                    'interaction_memory': predecessor.interaction_memory
                }
        return spatial_info

    def get_temporal_interaction_info(self) -> Dict[str, Dict]:
        """获取时间交互信息"""
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                temporal_info[predecessor.node_id] = {
                    'agent_role': predecessor.agent_role,
                    'execution_outputs': predecessor.execution_outputs,
                    'interaction_memory': predecessor.interaction_memory
                }
        return temporal_info

    def execute_reasoning(self, task_inputs: Any, **execution_kwargs) -> List[Any]:
        """
        执行推理过程
        
        处理输入并通过节点的操作进行推理，支持个体处理模式。
        """
        if not isinstance(task_inputs, list):
            task_inputs = [task_inputs]
        
        self.raw_task_inputs = task_inputs.copy()
        
        # 获取空间和时间交互信息
        spatial_interaction_info = self.get_spatial_interaction_info()
        temporal_interaction_info = self.get_temporal_interaction_info()
        
        # 处理输入
        processed_inputs = self._process_reasoning_inputs(
            task_inputs, spatial_interaction_info, temporal_interaction_info, **execution_kwargs
        )
        
        self.execution_inputs = processed_inputs
        execution_results = []
        
        # 执行推理
        for single_input in processed_inputs:
            try:
                result = self._execute_single_reasoning(single_input, **execution_kwargs)
                execution_results.append(result)
            except Exception as e:
                warnings.warn(f"Reasoning execution failed for input {single_input}: {str(e)}")
                execution_results.append(None)
        
        self.execution_outputs = execution_results
        return execution_results

    async def async_execute_reasoning(self, task_inputs: Any, **execution_kwargs) -> List[Any]:
        """
        异步执行推理过程
        """
        if not isinstance(task_inputs, list):
            task_inputs = [task_inputs]
        
        self.raw_task_inputs = task_inputs.copy()
        
        # 获取交互信息
        spatial_interaction_info = self.get_spatial_interaction_info()
        temporal_interaction_info = self.get_temporal_interaction_info()
        
        # 处理输入
        processed_inputs = self._process_reasoning_inputs(
            task_inputs, spatial_interaction_info, temporal_interaction_info, **execution_kwargs
        )
        
        self.execution_inputs = processed_inputs
        execution_results = []
        
        # 异步执行推理
        for single_input in processed_inputs:
            try:
                result = await self._async_execute_single_reasoning(single_input, **execution_kwargs)
                execution_results.append(result)
            except Exception as e:
                warnings.warn(f"Async reasoning execution failed for input {single_input}: {str(e)}")
                execution_results.append(None)
        
        self.execution_outputs = execution_results
        return execution_results

    @abstractmethod
    def _execute_single_reasoning(self, single_input: Any, **execution_kwargs) -> Any:
        """
        执行单个输入的推理 - 抽象方法
        
        这个方法应该在具体的智能体节点类型中实现，定义如何处理单个输入。
        """
        pass

    async def _async_execute_single_reasoning(self, single_input: Any, **execution_kwargs) -> Any:
        """
        异步执行单个输入的推理
        
        默认实现调用同步版本，子类可以重写以提供真正的异步实现。
        """
        return self._execute_single_reasoning(single_input, **execution_kwargs)

    def _process_reasoning_inputs(self, 
                                raw_task_inputs: List[Any], 
                                spatial_interaction_info: Dict[str, Dict], 
                                temporal_interaction_info: Dict[str, Dict], 
                                **execution_kwargs) -> List[Any]:
        """
        处理推理输入
        
        这个方法整合原始输入、空间交互信息和时间交互信息，生成最终的处理输入。
        子类可以重写此方法以实现特定的输入处理逻辑。
        """
        # 默认实现：简单返回原始输入
        # 子类可以重写以实现更复杂的输入融合逻辑
        return raw_task_inputs

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'node_id': self.node_id,
            'agent_role': self.agent_role,
            'spatial_predecessors_count': len(self.spatial_predecessors),
            'spatial_successors_count': len(self.spatial_successors),
            'temporal_predecessors_count': len(self.temporal_predecessors),
            'temporal_successors_count': len(self.temporal_successors),
            'execution_inputs_count': len(self.execution_inputs),
            'execution_outputs_count': len(self.execution_outputs),
            'memory_history_length': len(self.interaction_memory_history),
            'current_interaction_round': self.current_interaction_round
        }

    def validate_connections(self) -> Dict[str, bool]:
        """验证连接的有效性"""
        validation_results = {
            'spatial_connections_valid': True,
            'temporal_connections_valid': True,
            'no_self_connections': True,
            'no_duplicate_connections': True
        }
        
        # 检查是否存在自连接
        all_connections = (self.spatial_predecessors + self.spatial_successors + 
                         self.temporal_predecessors + self.temporal_successors)
        if self in all_connections:
            validation_results['no_self_connections'] = False
        
        # 检查是否存在重复连接
        if (len(set(self.spatial_predecessors)) != len(self.spatial_predecessors) or
            len(set(self.spatial_successors)) != len(self.spatial_successors) or
            len(set(self.temporal_predecessors)) != len(self.temporal_predecessors) or
            len(set(self.temporal_successors)) != len(self.temporal_successors)):
            validation_results['no_duplicate_connections'] = False
        
        return validation_results

    def __repr__(self):
        return f"AgentExecutionNode(id={self.node_id}, role={self.agent_role}, domain={self.task_domain})"

    def __str__(self):
        return f"Agent[{self.node_id}]({self.agent_role})"


class ConcreteAgentExecutionNode(AgentExecutionNode):
    """
    具体的智能体执行节点实现
    
    这是一个可以直接使用的智能体节点实现，提供了基本的推理执行逻辑。
    """
    
    def __init__(self, 
                 node_id: Optional[str] = None,
                 agent_role: str = "",
                 domain: str = "", 
                 llm_name: str = "",
                 reasoning_prompt: str = "",
                 **kwargs):
        """
        具体智能体节点，集成 baseclass.LLM 进行真实推理。
        """
        super().__init__(node_id, agent_role, domain, llm_name, **kwargs)
        # 基础系统提示
        base_prompt = reasoning_prompt or f"You are a {agent_role} agent working on {domain} tasks."
        self.reasoning_prompt = base_prompt
        # 初始化 LLM（模型名称通过 NEURALFSM_LLM_MODEL 环境变量控制）
        use_azure = kwargs.get('use_azure', False)
        try:
            self.reasoning_llm = LLM(system_prompt=base_prompt, use_azure=use_azure)
        except Exception as e:
            self.reasoning_llm = None
            warnings.warn(f"[ConcreteAgentExecutionNode] 初始化 LLM 失败 (role={agent_role}, domain={domain}): {e}")
    
    def _build_user_message(self, single_input: Any, is_final_state: bool = False) -> str:
        """
        构建发给 LLM 的 user message，统一适配所有数据集。
        """
        problem_text = str(single_input)

        # ✨ 成本优化：在FSM模式下，spatial_context和temporal_context通常为空或冗余
        # 因为FSM已经通过listener_messages机制传递跨状态消息，避免重复传递历史信息
        spatial_context = ""
        temporal_context = ""
        # 只在非FSM模式下使用spatial/temporal context（向后兼容）
        if not hasattr(self, '_use_fsm_mode') or not getattr(self, '_use_fsm_mode', False):
            spatial_info = self.get_spatial_interaction_info()
            temporal_info = self.get_temporal_interaction_info()
            if spatial_info:
                spatial_context = f"\n[Spatial context]: {spatial_info}\n"
            if temporal_info:
                temporal_context = f"\n[Temporal context]: {temporal_info}\n"
        
        # 动态监听得到的前序智能体消息（基于FSM监听关系）
        try:
            listener_messages = self.get_predecessor_messages()
        except Exception:
            listener_messages = []
        
        # ✨ 成本优化：限制监听消息的长度和数量，避免token爆炸
        MAX_LISTENER_MESSAGES = 2  # 最多保留最近2条监听消息（避免上下文过长）
        MAX_MESSAGE_LENGTH = 400  # 每条消息最多400字符（对于代码生成，保留关键部分）
        
        listener_context = ""
        if listener_messages:
            # 只保留最近的N条消息，并截断过长消息
            recent_messages = listener_messages[-MAX_LISTENER_MESSAGES:]
            truncated_messages = []
            for msg in recent_messages:
                msg_str = str(msg)
                if len(msg_str) > MAX_MESSAGE_LENGTH:
                    msg_str = msg_str[:MAX_MESSAGE_LENGTH] + "...[truncated]"
                truncated_messages.append(msg_str)
            
            listener_context = "\n[Listener messages from previous FSM states]\n" + "\n".join(truncated_messages)
        
        domain = (self.task_domain or "").lower()
        
        if is_final_state:
            # 针对不同数据集定制最终答案格式，所有情况都必须包含 <|submit|> ANSWER
            if domain == "gsm8k":
                user_msg = (
                    f"You are the final '{self.agent_role}' agent in a multi-agent FSM solving a GSM8K math word problem.\n\n"
                    f"Problem:\n{problem_text}\n\n"
                    f"Reasoning context from previous agents (may be empty):\n{spatial_context}\n{temporal_context}{listener_context}\n\n"
                    "Now output ONLY the final numeric answer in the following format (no explanation):\n\n"
                    "<|submit|> 1234\n\n"
                    "Replace 1234 with the correct final number. Do NOT include units, commas, currency symbols, "
                    "or any other text. The line must start with <|submit|> followed by a single space and the number."
                )
            elif domain == "mmlu":
                user_msg = (
                    f"You are the final '{self.agent_role}' agent in a multi-agent FSM solving an MMLU multiple-choice question.\n\n"
                    f"Question and options:\n{problem_text}\n\n"
                    f"Reasoning context from previous agents (may be empty):\n{spatial_context}\n{temporal_context}{listener_context}\n\n"
                    "Decide which option (A, B, C, or D) is correct, then output ONLY one line in this format:\n\n"
                    "<|submit|> A\n\n"
                    "Replace A with the correct option letter (A/B/C/D). Do NOT add any explanation or extra text."
                )
            elif domain == "humaneval":
                user_msg = (
                    f"You are the final '{self.agent_role}' agent in a multi-agent FSM solving a HumanEval code generation task.\n\n"
                    f"Task description and function signature:\n{problem_text}\n\n"
                    f"Context from previous agents (may be empty):\n{spatial_context}\n{temporal_context}{listener_context}\n\n"
                    "Now output ONLY the complete Python function implementation that satisfies the specification.\n"
                    "Return it on a single block starting immediately after the token below:\n\n"
                    "<|submit|> YOUR_CODE\n\n"
                    "Replace YOUR_CODE with valid Python code (function definition and body). Do not explain or add "
                    "anything outside the code. The line must start with <|submit|> followed by a space and then the code."
                )
            elif domain == "hotpotqa":
                user_msg = (
                    f"You are the final '{self.agent_role}' agent in a multi-agent FSM solving a HotpotQA multi-hop QA task.\n\n"
                    f"Question and context:\n{problem_text}\n\n"
                    f"Reasoning context from previous agents (may be empty):\n{spatial_context}\n{temporal_context}{listener_context}\n\n"
                    "After reasoning, output ONLY the final short answer string in the following format:\n\n"
                    "<|submit|> ANSWER\n\n"
                    "ANSWER should be a concise phrase or short sentence that directly answers the question. "
                    "Do NOT include explanations or any additional text."
                )
            elif domain == "alfworld":
                user_msg = (
                    f"You are the final '{self.agent_role}' agent in a multi-agent FSM solving an ALFWorld instruction-following task.\n\n"
                    f"Task description and environment information:\n{problem_text}\n\n"
                    f"Context from previous agents (may be empty):\n{spatial_context}\n{temporal_context}{listener_context}\n\n"
                    "Output ONLY the final action sequence that completes the task, in the following format:\n\n"
                    "<|submit|> action1 ; action2 ; action3\n\n"
                    "Use a semicolon and a space to separate actions. Do NOT include explanations or extra text."
                )
            elif domain == "math":
                user_msg = (
                    f"You are the final '{self.agent_role}' agent in a multi-agent FSM solving a MATH dataset problem.\n\n"
                    f"Problem:\n{problem_text}\n\n"
                    f"Reasoning context from previous agents (may be empty):\n{spatial_context}\n{temporal_context}{listener_context}\n\n"
                    "After doing all necessary reasoning, output ONLY the final LaTeX-formatted answer in the "
                    "following format:\n\n"
                    "<|submit|> \\boxed{ANSWER}\n\n"
                    "Replace ANSWER with the correct expression. Do NOT add any other text or explanation."
                )
            else:
                # 默认：短文本/数值答案
                user_msg = (
                    f"You are the final '{self.agent_role}' agent in a multi-agent FSM for domain '{self.task_domain}'.\n\n"
                    f"Task (problem):\n{problem_text}\n\n"
                    f"Context from previous agents (may be empty):\n{spatial_context}\n{temporal_context}{listener_context}\n\n"
                    "Now provide ONLY the final concise answer in the following format:\n\n"
                    "<|submit|> ANSWER\n\n"
                    "Where ANSWER is the final answer string (numeric or short text). "
                    "Do NOT add any extra explanation or text before or after this line."
                )
        else:
            # 中间状态：鼓励链式推理，但不提交最终答案
            user_msg = (
                f"You are '{self.agent_role}' in a multi-agent FSM for domain '{self.task_domain}'.\n\n"
                f"Task (problem):\n{problem_text}\n\n"
                f"Context from other agents (may be empty):\n{spatial_context}\n{temporal_context}{listener_context}\n\n"
                "Please think step by step and provide helpful intermediate reasoning, analysis, or calculations.\n"
                "Do NOT include the token <|submit|> or a final submitted answer in this step."
            )
        return user_msg
    
    def _execute_single_reasoning(self, single_input: Any, **execution_kwargs) -> Any:
        """
        执行单个输入的推理：真实调用 LLM。
        """
        is_final_state: bool = execution_kwargs.get("is_final_state", False)
        
        # 如果 LLM 初始化失败，退回到占位实现
        if self.reasoning_llm is None:
            return f"Processed by {self.agent_role}: {single_input}"
        
        user_msg = self._build_user_message(single_input, is_final_state=is_final_state)
        try:
            response = self.reasoning_llm.chat(message=user_msg)
            # ✨ 清理输出中的污染前缀（如 "Processed by"）
            if response.startswith(f"Processed by {self.agent_role}:"):
                response = response[len(f"Processed by {self.agent_role}:"):].strip()
            return response
        except Exception as e:
            warnings.warn(f"[ConcreteAgentExecutionNode] LLM 推理失败 (role={self.agent_role}): {e}")
            # ✨ 打印更详细的错误信息以便调试
            print(f"  ⚠️ LLM 推理错误详情: {str(e)}")
            return f"Processed by {self.agent_role}: {single_input}"
    
    async def _async_execute_single_reasoning(self, single_input: Any, **execution_kwargs) -> Any:
        """
        异步执行单个输入的推理
        """
        # 模拟异步处理
        await asyncio.sleep(0.1)  # 模拟处理时间
        return self._execute_single_reasoning(single_input, **execution_kwargs)


# 导出主要类
__all__ = ['AgentExecutionNode', 'ConcreteAgentExecutionNode']

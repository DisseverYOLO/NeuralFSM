"""
Agent Execution Node for Multi-Agent System
Agent execution node

Adapted from the original Node implementation and optimized for the MetaAgent project.
Supports time-sensitive memory management and agent interaction.
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

# Add the MetaAgent path.
sys.path.append(str(Path(__file__).parent.parent.parent))


class AgentExecutionNode(ABC):
    """
    Agent execution node.
    
    This class encapsulates the execution logic of a single agent in a
    multi-agent system, including inter-agent connections, input/output
    handling, and assigned operations. It supports both individual and
    aggregated processing modes.
    
    Core features:
    1. Manage spatial and temporal connections between agents.
    2. Handle agent inputs, outputs, and execution logic.
    3. Maintain time-sensitive interaction memory.
    4. Support asynchronous execution and reasoning.
    
    Attributes:
        node_id (str): Unique identifier of the node.
        agent_role (str): Agent role used for specialized operations.
        spatial_predecessors (List[AgentExecutionNode]): Spatial predecessor nodes.
        spatial_successors (List[AgentExecutionNode]): Spatial successor nodes.
        temporal_predecessors (List[AgentExecutionNode]): Temporal predecessor nodes.
        temporal_successors (List[AgentExecutionNode]): Temporal successor nodes.
        execution_inputs (List[Any]): Inputs waiting to be processed.
        execution_outputs (List[Any]): Outputs produced after execution.
        raw_task_inputs (List[Any]): Raw inputs containing questions or tasks.
        interaction_memory (Dict[str,List[Any]]): Input/output memory from the previous timestamp.
    """

    def __init__(self, 
                 node_id: Optional[str] = None,
                 agent_role: str = "",
                 domain: str = "", 
                 llm_name: str = "",
                 **kwargs):
        """
        Initialize the agent execution node.
        """
        self.node_id: str = node_id if node_id is not None else shortuuid.ShortUUID().random(length=6)
        self.agent_role: str = agent_role
        self.task_domain: str = domain
        self.language_model_name: str = llm_name
        
        # Connection management.
        self.spatial_predecessors: List['AgentExecutionNode'] = []
        self.spatial_successors: List['AgentExecutionNode'] = []
        self.temporal_predecessors: List['AgentExecutionNode'] = []
        self.temporal_successors: List['AgentExecutionNode'] = []
        
        # Execution-related state.
        self.execution_inputs: List[Any] = []
        self.execution_outputs: List[Any] = []
        self.raw_task_inputs: List[Any] = []
        self.agent_role_description = ""
        self.interaction_memory: Dict[str, List[Any]] = {
            'inputs': [], 'outputs': [], 'raw_inputs': []
        }
        
        # Time-sensitive memory management.
        self.interaction_memory_history: List[Dict[str, Any]] = []
        self.interaction_timestamps: List[float] = []
        self.current_interaction_round: int = 0
        self.memory_storage_capacity: int = 100
        
        # Additional configuration parameters.
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def execution_node_name(self):
        """Get the execution node name."""
        return self.__class__.__name__
    
    def add_predecessor_connection(self, predecessor_node: 'AgentExecutionNode', connection_type='spatial'):
        """Add a predecessor connection."""
        if connection_type == 'spatial' and predecessor_node not in self.spatial_predecessors:
            self.spatial_predecessors.append(predecessor_node)
            predecessor_node.spatial_successors.append(self)
        elif connection_type == 'temporal' and predecessor_node not in self.temporal_predecessors:
            self.temporal_predecessors.append(predecessor_node)
            predecessor_node.temporal_successors.append(self)

    def add_successor_connection(self, successor_node: 'AgentExecutionNode', connection_type='spatial'):
        """Add a successor connection."""
        if connection_type == 'spatial' and successor_node not in self.spatial_successors:
            self.spatial_successors.append(successor_node)
            successor_node.spatial_predecessors.append(self)
        elif connection_type == 'temporal' and successor_node not in self.temporal_successors:
            self.temporal_successors.append(successor_node)
            successor_node.temporal_predecessors.append(self)

    def remove_predecessor_connection(self, predecessor_node: 'AgentExecutionNode', connection_type='spatial'):
        """Remove a predecessor connection."""
        if connection_type == 'spatial' and predecessor_node in self.spatial_predecessors:
            self.spatial_predecessors.remove(predecessor_node)
            predecessor_node.spatial_successors.remove(self)
        elif connection_type == 'temporal' and predecessor_node in self.temporal_predecessors:
            self.temporal_predecessors.remove(predecessor_node)
            predecessor_node.temporal_successors.remove(self)

    def remove_successor_connection(self, successor_node: 'AgentExecutionNode', connection_type='spatial'):
        """Remove a successor connection."""
        if connection_type == 'spatial' and successor_node in self.spatial_successors:
            self.spatial_successors.remove(successor_node)
            successor_node.spatial_predecessors.remove(self)
        elif connection_type == 'temporal' and successor_node in self.temporal_successors:
            self.temporal_successors.remove(successor_node)
            successor_node.temporal_predecessors.remove(self)

    def clear_spatial_connections(self):
        """Clear spatial connections."""
        self.spatial_predecessors: List['AgentExecutionNode'] = []
        self.spatial_successors: List['AgentExecutionNode'] = []

    def clear_temporal_connections(self):
        """Clear temporal connections."""
        self.temporal_predecessors: List['AgentExecutionNode'] = []
        self.temporal_successors: List['AgentExecutionNode'] = []        
    
    def update_interaction_memory(self):
        """Update interaction memory with time-sensitive tracking."""
        # Update base memory.
        self.interaction_memory['inputs'] = self.execution_inputs.copy() if self.execution_inputs else []
        self.interaction_memory['outputs'] = self.execution_outputs.copy() if self.execution_outputs else []
        self.interaction_memory['raw_inputs'] = self.raw_task_inputs.copy() if self.raw_task_inputs else []
        
        # Create a time-sensitive memory snapshot.
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
        
        # Add to memory history.
        self.interaction_memory_history.append(memory_snapshot)
        self.interaction_timestamps.append(current_timestamp)
        
        # Capacity management: remove the oldest memory when the limit is exceeded.
        if len(self.interaction_memory_history) > self.memory_storage_capacity:
            self.interaction_memory_history.pop(0)
            self.interaction_timestamps.pop(0)
    
    def get_recent_interaction_memory(self, num_recent_interactions: int = 5) -> List[Dict[str, Any]]:
        """Get recent interaction memory."""
        return self.interaction_memory_history[-num_recent_interactions:] if self.interaction_memory_history else []
    
    def get_memory_by_interaction_round(self, round_number: int) -> Optional[Dict[str, Any]]:
        """Get memory by interaction round."""
        for memory in reversed(self.interaction_memory_history):
            if memory['interaction_round'] == round_number:
                return memory
        return None
    
    def get_temporal_interaction_context(self, time_window_seconds: float = 60.0) -> List[Dict[str, Any]]:
        """Get interaction memory context within a given time window."""
        current_timestamp = time.time()
        temporal_memories = []
        
        for i, timestamp in enumerate(self.interaction_timestamps):
            if current_timestamp - timestamp <= time_window_seconds:
                temporal_memories.append(self.interaction_memory_history[i])
        
        return temporal_memories
    
    def reset_interaction_memory(self):
        """Reset interaction memory."""
        self.interaction_memory_history.clear()
        self.interaction_timestamps.clear()
        self.current_interaction_round = 0
        self.interaction_memory = {'inputs': [], 'outputs': [], 'raw_inputs': []}
    
    def set_current_interaction_round(self, round_number: int):
        """Set the current interaction round."""
        self.current_interaction_round = round_number
    
    def add_predecessor_message(self, message: str):
        """
        Add a message from an upstream agent for the FSM listener mechanism.
        
        Args:
            message: Output message from an upstream agent.
        """
        # Add the message to execution inputs.
        if not hasattr(self, 'predecessor_messages'):
            self.predecessor_messages = []
        
        self.predecessor_messages.append(message)
    
    def get_predecessor_messages(self) -> List[str]:
        """Get all messages from upstream agents."""
        if not hasattr(self, 'predecessor_messages'):
            self.predecessor_messages = []
        return self.predecessor_messages
    
    def clear_predecessor_messages(self):
        """Clear upstream messages."""
        self.predecessor_messages = []

    def get_spatial_interaction_info(self) -> Dict[str, Dict]:
        """Get spatial interaction information."""
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
        """Get temporal interaction information."""
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
        Execute the reasoning process.
        
        Process inputs and perform reasoning through node operations,
        supporting individual processing mode.
        """
        if not isinstance(task_inputs, list):
            task_inputs = [task_inputs]
        
        self.raw_task_inputs = task_inputs.copy()
        
        # Get spatial and temporal interaction information.
        spatial_interaction_info = self.get_spatial_interaction_info()
        temporal_interaction_info = self.get_temporal_interaction_info()
        
        # Process inputs.
        processed_inputs = self._process_reasoning_inputs(
            task_inputs, spatial_interaction_info, temporal_interaction_info, **execution_kwargs
        )
        
        self.execution_inputs = processed_inputs
        execution_results = []
        
        # Execute reasoning.
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
        Execute the reasoning process asynchronously.
        """
        if not isinstance(task_inputs, list):
            task_inputs = [task_inputs]
        
        self.raw_task_inputs = task_inputs.copy()
        
        # Get interaction information.
        spatial_interaction_info = self.get_spatial_interaction_info()
        temporal_interaction_info = self.get_temporal_interaction_info()
        
        # Process inputs.
        processed_inputs = self._process_reasoning_inputs(
            task_inputs, spatial_interaction_info, temporal_interaction_info, **execution_kwargs
        )
        
        self.execution_inputs = processed_inputs
        execution_results = []
        
        # Execute reasoning asynchronously.
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
        Execute reasoning for a single input - abstract method.
        
        This method should be implemented by concrete agent node types to
        define how a single input is handled.
        """
        pass

    async def _async_execute_single_reasoning(self, single_input: Any, **execution_kwargs) -> Any:
        """
        Execute reasoning for a single input asynchronously.
        
        The default implementation calls the synchronous version. Subclasses
        can override it to provide a truly asynchronous implementation.
        """
        return self._execute_single_reasoning(single_input, **execution_kwargs)

    def _process_reasoning_inputs(self, 
                                raw_task_inputs: List[Any], 
                                spatial_interaction_info: Dict[str, Dict], 
                                temporal_interaction_info: Dict[str, Dict], 
                                **execution_kwargs) -> List[Any]:
        """
        Process reasoning inputs.
        
        This method combines raw inputs, spatial interaction information,
        and temporal interaction information to produce the final processed
        inputs. Subclasses can override it to implement task-specific logic.
        """
        # Default implementation: simply return the raw inputs.
        # Subclasses can override this with more complex fusion logic.
        return raw_task_inputs

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get an execution summary."""
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
        """Validate connection integrity."""
        validation_results = {
            'spatial_connections_valid': True,
            'temporal_connections_valid': True,
            'no_self_connections': True,
            'no_duplicate_connections': True
        }
        
        # Check for self-connections.
        all_connections = (self.spatial_predecessors + self.spatial_successors + 
                         self.temporal_predecessors + self.temporal_successors)
        if self in all_connections:
            validation_results['no_self_connections'] = False
        
        # Check for duplicate connections.
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
    Concrete implementation of an agent execution node.
    
    This is a directly usable agent node implementation that provides
    basic reasoning execution logic.
    """
    
    def __init__(self, 
                 node_id: Optional[str] = None,
                 agent_role: str = "",
                 domain: str = "", 
                 llm_name: str = "",
                 reasoning_prompt: str = "",
                 **kwargs):
        """
        Concrete agent node that integrates `baseclass.LLM` for real reasoning.
        """
        super().__init__(node_id, agent_role, domain, llm_name, **kwargs)
        # Base system prompt.
        base_prompt = reasoning_prompt or f"You are a {agent_role} agent working on {domain} tasks."
        self.reasoning_prompt = base_prompt
        # Initialize the LLM. The model name is controlled by the NEURALFSM_LLM_MODEL environment variable.
        use_azure = kwargs.get('use_azure', False)
        try:
            self.reasoning_llm = LLM(system_prompt=base_prompt, use_azure=use_azure)
        except Exception as e:
            self.reasoning_llm = None
            warnings.warn(f"[ConcreteAgentExecutionNode] Failed to initialize LLM (role={agent_role}, domain={domain}): {e}")
    
    def _build_user_message(self, single_input: Any, is_final_state: bool = False) -> str:
        """
        Build the user message sent to the LLM in a unified format for all datasets.
        """
        problem_text = str(single_input)

        # Cost optimization: in FSM mode, spatial and temporal context are
        # usually empty or redundant because cross-state messages are already
        # passed through the listener_messages mechanism.
        spatial_context = ""
        temporal_context = ""
        # Use spatial/temporal context only outside FSM mode for backward compatibility.
        if not hasattr(self, '_use_fsm_mode') or not getattr(self, '_use_fsm_mode', False):
            spatial_info = self.get_spatial_interaction_info()
            temporal_info = self.get_temporal_interaction_info()
            if spatial_info:
                spatial_context = f"\n[Spatial context]: {spatial_info}\n"
            if temporal_info:
                temporal_context = f"\n[Temporal context]: {temporal_info}\n"
        
        # Dynamically collected upstream agent messages based on FSM listener relationships.
        try:
            listener_messages = self.get_predecessor_messages()
        except Exception:
            listener_messages = []
        
        # Cost optimization: limit listener message length and count to avoid token explosion.
        MAX_LISTENER_MESSAGES = 2  # Keep at most the 2 most recent listener messages.
        MAX_MESSAGE_LENGTH = 400  # Keep at most 400 characters per message.
        
        listener_context = ""
        if listener_messages:
            # Keep only the most recent N messages and truncate long ones.
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
            # Tailor final-answer formatting to each dataset.
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
                # Default: short text or numeric answer.
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
            # Intermediate state: encourage step-by-step reasoning without submitting a final answer.
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
        Execute reasoning for a single input by calling the LLM.
        """
        is_final_state: bool = execution_kwargs.get("is_final_state", False)
        
        # If LLM initialization failed, fall back to a placeholder implementation.
        if self.reasoning_llm is None:
            return f"Processed by {self.agent_role}: {single_input}"
        
        user_msg = self._build_user_message(single_input, is_final_state=is_final_state)
        try:
            response = self.reasoning_llm.chat(message=user_msg)
            # Clean polluted output prefixes such as "Processed by".
            if response.startswith(f"Processed by {self.agent_role}:"):
                response = response[len(f"Processed by {self.agent_role}:"):].strip()
            return response
        except Exception as e:
            warnings.warn(f"[ConcreteAgentExecutionNode] LLM reasoning failed (role={self.agent_role}): {e}")
            # Print more detailed error information for debugging.
            print(f"  ⚠️ LLM reasoning error details: {str(e)}")
            return f"Processed by {self.agent_role}: {single_input}"
    
    async def _async_execute_single_reasoning(self, single_input: Any, **execution_kwargs) -> Any:
        """
        Execute reasoning for a single input asynchronously.
        """
        # Simulate asynchronous processing.
        await asyncio.sleep(0.1)  # Simulated processing time.
        return self._execute_single_reasoning(single_input, **execution_kwargs)


# Export main classes
__all__ = ['AgentExecutionNode', 'ConcreteAgentExecutionNode']

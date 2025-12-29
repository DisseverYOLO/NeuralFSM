"""
FSM State Manager for Neural Multi-Agent System
有限状态机状态管理器

负责管理FSM状态、状态转移规则、监听关系
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import re
from dataclasses import dataclass


@dataclass
class FSMState:
    """FSM状态定义"""
    state_id: int
    state_name: str
    responsible_agent_id: str  # 负责该状态的智能体ID
    is_initial: bool = False
    is_final: bool = False
    description: str = ""
    completion_condition: str = ""  # ✨ 状态完成条件（Enhanced_FSM_Gen.py生成）
    
    def __repr__(self):
        return f"State({self.state_id}, {self.state_name}, agent={self.responsible_agent_id})"


class FSMStateManager:
    """
    有限状态机状态管理器
    
    核心功能：
    1. 管理FSM状态定义
    2. 管理状态→智能体映射
    3. 管理监听关系（通信路径图）
    4. 判断状态转移条件
    5. 提取状态转移目标
    """
    
    def __init__(self, agent_ids: List[str]):
        """
        初始化FSM状态管理器
        
        Args:
            agent_ids: 智能体ID列表
        """
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        
        # FSM状态
        self.states: Dict[int, FSMState] = {}
        self.current_state_id: Optional[int] = None
        
        # 监听关系（通信路径图）
        # listeners[state_id] = [agent_id1, agent_id2, ...]
        # 表示state_id状态的输出会传给这些智能体
        self.listeners: Dict[int, List[str]] = {}
        
        # 状态转移历史
        self.transition_history: List[Tuple[int, int]] = []  # [(from_state, to_state), ...]
        
        # ✨ 状态转移规则（从总FSM中加载，包含转移条件）
        # transition_rules: List[Dict] = [{'from_state': int, 'to_state': int, 'condition': str}, ...]
        self.transition_rules: List[Dict[str, Any]] = []
        
        # 状态转移概率（由TGN学习）
        # transition_probs[from_state][to_state] = probability
        self.transition_probs: Optional[torch.Tensor] = None
        
        # 监听关系权重（由TGN学习）
        # listener_weights[state_id][agent_idx] = weight
        self.listener_weights: Optional[torch.Tensor] = None
    
    def add_state(self, 
                  state_id: int,
                  state_name: str,
                  responsible_agent_id: str,
                  is_initial: bool = False,
                  is_final: bool = False,
                  description: str = "",
                  completion_condition: str = ""):
        """
        添加FSM状态
        
        Args:
            state_id: 状态ID
            state_name: 状态名称
            responsible_agent_id: 负责该状态的智能体ID
            is_initial: 是否为初始状态
            is_final: 是否为最终状态
            description: 状态描述
            completion_condition: 状态完成条件（Enhanced_FSM_Gen.py生成）
        """
        state = FSMState(
            state_id=state_id,
            state_name=state_name,
            responsible_agent_id=responsible_agent_id,
            is_initial=is_initial,
            is_final=is_final,
            description=description,
            completion_condition=completion_condition
        )
        
        self.states[state_id] = state
        
        # 初始化监听列表
        if state_id not in self.listeners:
            self.listeners[state_id] = []
        
        # 如果是初始状态，设置为当前状态
        if is_initial:
            self.current_state_id = state_id
    
    def add_listener(self, state_id: int, listener_agent_id: str):
        """
        为状态添加监听智能体（建立通信路径）
        
        Args:
            state_id: 状态ID
            listener_agent_id: 监听该状态输出的智能体ID
        """
        if state_id not in self.listeners:
            self.listeners[state_id] = []
        
        if listener_agent_id not in self.listeners[state_id]:
            self.listeners[state_id].append(listener_agent_id)
    
    def set_listeners(self, state_id: int, listener_agent_ids: List[str]):
        """
        设置状态的监听智能体列表
        
        Args:
            state_id: 状态ID
            listener_agent_ids: 监听智能体ID列表
        """
        self.listeners[state_id] = listener_agent_ids.copy()
    
    def get_state(self, state_id: int) -> Optional[FSMState]:
        """获取状态信息"""
        return self.states.get(state_id)
    
    def get_current_state(self) -> Optional[FSMState]:
        """获取当前状态"""
        if self.current_state_id is not None:
            return self.states.get(self.current_state_id)
        return None
    
    def get_initial_state(self) -> Optional[FSMState]:
        """获取初始状态"""
        for state in self.states.values():
            if state.is_initial:
                return state
        return None
    
    def get_responsible_agent(self, state_id: int) -> Optional[str]:
        """获取负责指定状态的智能体ID"""
        state = self.states.get(state_id)
        if state:
            return state.responsible_agent_id
        return None
    
    def get_listeners(self, state_id: int) -> List[str]:
        """获取状态的监听智能体列表"""
        return self.listeners.get(state_id, [])
    
    def extract_state_transition(self, output: str) -> Optional[int]:
        """
        从智能体输出中提取状态转移目标
        
        Args:
            output: 智能体输出文本
        
        Returns:
            目标状态ID，如果没有找到则返回None
        """
        # 匹配 <STATE_TRANS>: <state_id> 或 <STATE_TRANS>:<state_id>
        patterns = [
            r'<STATE_TRANS>:\s*(\d+)',
            r'<STATE_TRANS>\s*:\s*(\d+)',
            r'STATE_TRANS:\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                state_id = int(match.group(1))
                if state_id in self.states:
                    return state_id
        
        return None
    
    def check_final_answer(self, output: str) -> Optional[str]:
        """
        检查是否包含最终答案标记
        
        Args:
            output: 智能体输出文本
        
        Returns:
            提取的答案，如果没有找到则返回None
        """
        if "<|submit|>" in output:
            parts = output.split("<|submit|>")
            if len(parts) > 1:
                answer = parts[1].strip()
                # ✨ 清理所有控制标记（如 </|submit|>、<|end|> 等）
                import re
                answer = re.sub(r'</?[|][^|]+[|]>', '', answer)
                return answer.strip()
        
        return None
    
    def transition_to_state(self, target_state_id: int) -> bool:
        """
        转移到目标状态
        
        Args:
            target_state_id: 目标状态ID
        
        Returns:
            是否转移成功
        """
        if target_state_id not in self.states:
            return False
        
        # 记录转移历史
        if self.current_state_id is not None:
            self.transition_history.append((self.current_state_id, target_state_id))
        
        # 更新当前状态
        self.current_state_id = target_state_id
        return True
    
    def get_transition_condition(self, from_state_id: int, to_state_id: int) -> Optional[str]:
        """
        ✨ 从总FSM中获取状态转移条件描述
        
        Args:
            from_state_id: 源状态ID
            to_state_id: 目标状态ID
        
        Returns:
            转移条件描述，如果不存在则返回None
        """
        # 确保类型匹配（可能是int或str）
        for rule in self.transition_rules:
            rule_from = rule.get('from_state')
            rule_to = rule.get('to_state')
            # 尝试类型转换匹配
            if (int(rule_from) == int(from_state_id) and 
                int(rule_to) == int(to_state_id)):
                condition = rule.get('condition', '')
                if condition:
                    return condition
                # 如果没有condition，尝试使用其他字段
                return rule.get('description', rule.get('priority', ''))
        return None
    
    def add_transition_rule(self, from_state_id: int, to_state_id: int, condition: str = ""):
        """
        ✨ 添加状态转移规则（从总FSM中加载）
        
        Args:
            from_state_id: 源状态ID
            to_state_id: 目标状态ID
            condition: 转移条件描述
        """
        # 检查是否已存在
        for rule in self.transition_rules:
            if rule.get('from_state') == from_state_id and rule.get('to_state') == to_state_id:
                # 更新条件
                rule['condition'] = condition
                return
        
        # 添加新规则
        self.transition_rules.append({
            'from_state': from_state_id,
            'to_state': to_state_id,
            'condition': condition
        })
    
    def is_final_state(self, state_id: Optional[int] = None) -> bool:
        """
        检查是否为最终状态
        
        Args:
            state_id: 状态ID，如果为None则检查当前状态
        
        Returns:
            是否为最终状态
        """
        if state_id is None:
            state_id = self.current_state_id
        
        if state_id is None:
            return False
        
        state = self.states.get(state_id)
        return state.is_final if state else False
    
    def reset(self):
        """重置FSM到初始状态"""
        initial_state = self.get_initial_state()
        if initial_state:
            self.current_state_id = initial_state.state_id
        else:
            self.current_state_id = None
        
        self.transition_history.clear()
    
    def get_transition_graph(self) -> torch.Tensor:
        """
        获取状态转移图的邻接矩阵表示
        
        Returns:
            [num_states, num_states] 的邻接矩阵
        """
        num_states = len(self.states)
        adj_matrix = torch.zeros(num_states, num_states)
        
        # 基于转移历史构建
        for from_state, to_state in self.transition_history:
            adj_matrix[from_state, to_state] = 1.0
        
        return adj_matrix
    
    def get_listener_graph(self) -> torch.Tensor:
        """
        获取监听关系图（通信路径图）
        
        Returns:
            [num_states, num_agents] 的监听权重矩阵
        """
        num_states = len(self.states)
        listener_matrix = torch.zeros(num_states, self.num_agents)
        
        # 为每个状态填充监听关系
        for state_id, listener_ids in self.listeners.items():
            if state_id < num_states:
                for listener_id in listener_ids:
                    if listener_id in self.agent_ids:
                        agent_idx = self.agent_ids.index(listener_id)
                        listener_matrix[state_id, agent_idx] = 1.0
        
        return listener_matrix
    
    def update_transition_probs(self, probs: torch.Tensor):
        """
        更新状态转移概率（由TGN学习）
        
        Args:
            probs: [num_states, num_states] 状态转移概率矩阵
        """
        self.transition_probs = probs
    
    def update_listener_weights(self, weights: torch.Tensor):
        """
        更新监听关系权重（由TGN学习）
        
        Args:
            weights: [num_states, num_agents] 监听权重矩阵
        """
        self.listener_weights = weights
    
    def get_state_summary(self) -> str:
        """获取FSM状态摘要"""
        summary = f"FSM State Manager Summary:\n"
        summary += f"  Total States: {len(self.states)}\n"
        summary += f"  Current State: {self.current_state_id}\n"
        summary += f"  States:\n"
        
        for state_id, state in sorted(self.states.items()):
            markers = []
            if state.is_initial:
                markers.append("INITIAL")
            if state.is_final:
                markers.append("FINAL")
            
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            listeners_str = ', '.join(self.listeners.get(state_id, []))

            summary += f"    State {state_id}: {state.state_name}{marker_str}\n"
            if state.description:
                # 将描述单独一行打印，便于阅读
                summary += f"      Description: {state.description}\n"
            summary += f"      Agent: {state.responsible_agent_id}\n"
            summary += f"      Listeners: {listeners_str}\n"
        
        return summary
    
    def __repr__(self):
        return f"FSMStateManager({len(self.states)} states, current={self.current_state_id})"



"""
FSM Executor with Loop Prevention
FSM执行器（支持循环避免）

功能:
1. 执行FSM状态转移
2. 状态访问次数限制（避免无限循环）
3. 状态转移条件匹配
4. 与TGN集成的下一状态预测
5. 执行历史追踪
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import re


class FSMExecutor:
    """FSM执行器（支持循环避免）"""
    
    def __init__(self, 
                 fsm: Dict[str, Any],
                 max_visits_per_state: int = 3,
                 max_total_steps: int = 20):
        """
        初始化FSM执行器
        
        Args:
            fsm: FSM结构
            max_visits_per_state: 每个状态最大访问次数（避免局部循环）
            max_total_steps: 最大总步数（避免整体循环）
        """
        self.fsm = fsm
        self.max_visits = max_visits_per_state
        self.max_steps = max_total_steps
        
        # 状态字典
        self.states = {s['state_id']: s for s in fsm['states']}
        self.transitions = fsm['transitions']
        
        # 执行状态
        self.current_state_id = None
        self.visit_counts = defaultdict(int)
        self.step_count = 0
        self.state_history = []
        self.transition_history = []
        
        # 找到初始状态
        initial_states = [s for s in fsm['states'] if s.get('is_initial', False)]
        if initial_states:
            self.current_state_id = initial_states[0]['state_id']
    
    def reset(self):
        """重置执行器到初始状态"""
        self.visit_counts.clear()
        self.step_count = 0
        self.state_history.clear()
        self.transition_history.clear()
        
        initial_states = [s for s in self.fsm['states'] if s.get('is_initial', False)]
        if initial_states:
            self.current_state_id = initial_states[0]['state_id']
    
    def get_current_state(self) -> Optional[Dict]:
        """获取当前状态"""
        if self.current_state_id:
            return self.states.get(self.current_state_id)
        return None
    
    def is_final_state(self) -> bool:
        """检查当前是否为最终状态"""
        current = self.get_current_state()
        if current:
            return current.get('is_final', False)
        return False
    
    def can_continue(self) -> bool:
        """检查是否可以继续执行"""
        # 检查是否超过总步数
        if self.step_count >= self.max_steps:
            return False
        
        # 检查是否在最终状态
        if self.is_final_state():
            return False
        
        return True
    
    def get_valid_transitions(self, agent_output: str) -> List[Dict]:
        """
        获取当前状态的有效转移
        
        Args:
            agent_output: Agent的输出
        
        Returns:
            有效转移列表（条件满足且未超过访问次数）
        """
        if not self.current_state_id:
            return []
        
        # 找到所有可能的转移
        possible_transitions = [
            t for t in self.transitions
            if t['from_state'] == self.current_state_id
        ]
        
        # 过滤：检查条件和访问次数
        valid_transitions = []
        for trans in possible_transitions:
            # 检查目标状态是否已超过访问次数
            target_state = trans['to_state']
            if self.visit_counts[target_state] >= self.max_visits:
                # 除非目标是最终状态（最终状态不受限制）
                target_state_obj = self.states.get(target_state)
                if not target_state_obj or not target_state_obj.get('is_final', False):
                    continue
            
            # 检查转移条件
            if self._check_transition_condition(trans, agent_output):
                valid_transitions.append(trans)
        
        # 按优先级排序
        valid_transitions.sort(key=lambda t: t.get('priority', 1))
        
        return valid_transitions
    
    def _check_transition_condition(self, transition: Dict, agent_output: str) -> bool:
        """
        检查转移条件是否满足
        
        Args:
            transition: 转移对象
            agent_output: Agent输出
        
        Returns:
            条件是否满足
        """
        condition = transition.get('condition', '')
        if not condition:
            return True  # 无条件转移
        
        condition_lower = condition.lower()
        
        # 1. 检查标记存在
        if 'contains' in condition_lower or 'has' in condition_lower:
            # 提取标记 例如: "contains <DONE>"
            markers = re.findall(r'<([^>]+)>', condition)
            if markers:
                for marker in markers:
                    if f'<{marker}>' in agent_output:
                        return True
                return False
        
        # 2. 检查关键词
        keywords = ['complete', 'success', 'done', 'finish', 'ready']
        for keyword in keywords:
            if keyword in condition_lower and keyword in agent_output.lower():
                return True
        
        # 3. 检查失败/错误条件
        fail_keywords = ['fail', 'error', 'issue', 'problem']
        for keyword in fail_keywords:
            if keyword in condition_lower and keyword in agent_output.lower():
                return True
        
        # 4. 默认：检查是否有DONE或COMPLETE标记
        if '<DONE>' in agent_output or '<COMPLETE>' in agent_output:
            return True
        
        # 5. 如果条件很简单，默认满足
        if len(condition.split()) <= 5:
            return True
        
        return False
    
    def select_next_state(self, 
                         agent_output: str, 
                         tgn_predictor=None) -> Optional[str]:
        """
        选择下一个状态
        
        Args:
            agent_output: Agent输出
            tgn_predictor: TGN预测器（可选，用于多个有效转移时）
        
        Returns:
            下一个状态ID，如果无法转移则返回None
        """
        valid_transitions = self.get_valid_transitions(agent_output)
        
        if not valid_transitions:
            # 没有有效转移：尝试救援转移
            rescue_transitions = [
                t for t in self.transitions
                if t['from_state'] == self.current_state_id 
                and t.get('is_rescue', False)
            ]
            if rescue_transitions:
                return rescue_transitions[0]['to_state']
            
            return None  # 真的卡住了
        
        # 如果只有一个有效转移，直接使用
        if len(valid_transitions) == 1:
            return valid_transitions[0]['to_state']
        
        # 多个有效转移：使用TGN预测（如果可用）
        if tgn_predictor:
            try:
                next_state = tgn_predictor.predict_next_state(
                    self.current_state_id,
                    valid_transitions
                )
                return next_state
            except:
                pass
        
        # 降级：选择优先级最高的
        return valid_transitions[0]['to_state']
    
    def execute_transition(self, next_state_id: str) -> bool:
        """
        执行状态转移
        
        Args:
            next_state_id: 目标状态ID
        
        Returns:
            是否转移成功
        """
        if next_state_id not in self.states:
            return False
        
        # 记录历史
        if self.current_state_id:
            self.transition_history.append((self.current_state_id, next_state_id))
        
        # 执行转移
        self.current_state_id = next_state_id
        self.visit_counts[next_state_id] += 1
        self.step_count += 1
        self.state_history.append(next_state_id)
        
        return True
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'current_state': self.current_state_id,
            'step_count': self.step_count,
            'is_final': self.is_final_state(),
            'state_history': self.state_history.copy(),
            'transition_history': self.transition_history.copy(),
            'visit_counts': dict(self.visit_counts),
            'states_visited': len(set(self.state_history)),
            'most_visited_state': max(self.visit_counts.items(), key=lambda x: x[1])[0] if self.visit_counts else None
        }
    
    def print_execution_trace(self):
        """打印执行轨迹"""
        print("\n" + "=" * 80)
        print("FSM Execution Trace")
        print("=" * 80)
        
        print(f"Total Steps: {self.step_count}/{self.max_steps}")
        print(f"Current State: {self.current_state_id}")
        print(f"Is Final: {'Yes' if self.is_final_state() else 'No'}")
        
        print(f"\nState Sequence ({len(self.state_history)} states):")
        for i, state_id in enumerate(self.state_history, 1):
            state = self.states.get(state_id, {})
            state_name = state.get('state_name', state_id)
            visit_num = sum(1 for s in self.state_history[:i] if s == state_id)
            marker = "🎯" if state.get('is_final') else ("🚀" if state.get('is_initial') else "📍")
            print(f"  {i:2d}. {marker} {state_name} (visit #{visit_num})")
        
        print(f"\nVisit Counts:")
        for state_id, count in sorted(self.visit_counts.items(), key=lambda x: -x[1]):
            state_name = self.states.get(state_id, {}).get('state_name', state_id)
            limit_marker = "⚠️ " if count >= self.max_visits else "   "
            print(f"  {limit_marker}{state_name}: {count}/{self.max_visits}")
        
        print("=" * 80 + "\n")


# ========== 便捷函数 ==========

def create_fsm_executor(fsm: Dict, 
                       max_visits_per_state: int = 3,
                       max_total_steps: int = 20) -> FSMExecutor:
    """创建FSM执行器"""
    return FSMExecutor(fsm, max_visits_per_state, max_total_steps)


# ========== 测试示例 ==========

if __name__ == "__main__":
    # 创建测试FSM（包含循环）
    test_fsm = {
        'states': [
            {'state_id': '0', 'state_name': 'Start', 'is_initial': True, 'is_final': False},
            {'state_id': '1', 'state_name': 'Process', 'is_initial': False, 'is_final': False},
            {'state_id': '2', 'state_name': 'Verify', 'is_initial': False, 'is_final': False},
            {'state_id': '3', 'state_name': 'Final', 'is_initial': False, 'is_final': True},
        ],
        'transitions': [
            {'from_state': '0', 'to_state': '1', 'condition': 'contains <DONE>', 'priority': 1},
            {'from_state': '1', 'to_state': '2', 'condition': 'contains <DONE>', 'priority': 1},
            {'from_state': '2', 'to_state': '1', 'condition': 'contains <RETRY>', 'priority': 1},  # 回退
            {'from_state': '2', 'to_state': '3', 'condition': 'contains <DONE>', 'priority': 1},
            {'from_state': '2', 'to_state': '3', 'condition': '', 'priority': 99, 'is_rescue': True},
        ]
    }
    
    # 创建执行器
    executor = create_fsm_executor(test_fsm, max_visits_per_state=3, max_total_steps=10)
    
    # 模拟执行
    print("Simulating FSM Execution...")
    
    # Step 1: Start → Process
    print("\nStep 1: Start state")
    next_state = executor.select_next_state("<DONE>")
    print(f"  Next state: {next_state}")
    executor.execute_transition(next_state)
    
    # Step 2: Process → Verify
    print("\nStep 2: Process state")
    next_state = executor.select_next_state("<DONE>")
    print(f"  Next state: {next_state}")
    executor.execute_transition(next_state)
    
    # Step 3: Verify → Process (回退)
    print("\nStep 3: Verify state (retry)")
    next_state = executor.select_next_state("<RETRY>")
    print(f"  Next state: {next_state} (backtrack)")
    executor.execute_transition(next_state)
    
    # Step 4: Process → Verify (再次)
    print("\nStep 4: Process state (again)")
    next_state = executor.select_next_state("<DONE>")
    print(f"  Next state: {next_state}")
    executor.execute_transition(next_state)
    
    # Step 5: Verify → Final
    print("\nStep 5: Verify state (final)")
    next_state = executor.select_next_state("<DONE>")
    print(f"  Next state: {next_state}")
    executor.execute_transition(next_state)
    
    # 打印执行轨迹
    executor.print_execution_trace()
    
    # 获取摘要
    summary = executor.get_execution_summary()
    print("\nExecution Summary:")
    import json
    print(json.dumps(summary, indent=2))


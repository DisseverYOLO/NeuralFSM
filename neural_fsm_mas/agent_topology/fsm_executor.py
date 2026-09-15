"""
FSM executor with loop prevention

Features:
1. Execute FSM state transitions.
2. Limit state visit counts to avoid infinite loops.
3. Match state transition conditions.
4. Support next-state prediction integrated with TGN.
5. Track execution history.
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import re


class FSMExecutor:
    """FSM executor with loop prevention."""
    
    def __init__(self, 
                 fsm: Dict[str, Any],
                 max_visits_per_state: int = 3,
                 max_total_steps: int = 20):
        """
        Initialize the FSM executor.
        
        Args:
            fsm: FSM structure.
            max_visits_per_state: Maximum visits per state to avoid local loops.
            max_total_steps: Maximum total steps to avoid global loops.
        """
        self.fsm = fsm
        self.max_visits = max_visits_per_state
        self.max_steps = max_total_steps
        
        # State dictionary.
        self.states = {s['state_id']: s for s in fsm['states']}
        self.transitions = fsm['transitions']
        
        # Execution state.
        self.current_state_id = None
        self.visit_counts = defaultdict(int)
        self.step_count = 0
        self.state_history = []
        self.transition_history = []
        
        # Find the initial state.
        initial_states = [s for s in fsm['states'] if s.get('is_initial', False)]
        if initial_states:
            self.current_state_id = initial_states[0]['state_id']
    
    def reset(self):
        """Reset the executor to the initial state."""
        self.visit_counts.clear()
        self.step_count = 0
        self.state_history.clear()
        self.transition_history.clear()
        
        initial_states = [s for s in self.fsm['states'] if s.get('is_initial', False)]
        if initial_states:
            self.current_state_id = initial_states[0]['state_id']
    
    def get_current_state(self) -> Optional[Dict]:
        """Get the current state."""
        if self.current_state_id:
            return self.states.get(self.current_state_id)
        return None
    
    def is_final_state(self) -> bool:
        """Check whether the current state is final."""
        current = self.get_current_state()
        if current:
            return current.get('is_final', False)
        return False
    
    def can_continue(self) -> bool:
        """Check whether execution can continue."""
        # Check whether the total step limit has been exceeded.
        if self.step_count >= self.max_steps:
            return False
        
        # Check whether the current state is final.
        if self.is_final_state():
            return False
        
        return True
    
    def get_valid_transitions(self, agent_output: str) -> List[Dict]:
        """
        Get valid transitions for the current state.
        
        Args:
            agent_output: Agent output.
        
        Returns:
            Valid transitions whose conditions are satisfied and visit limits are not exceeded.
        """
        if not self.current_state_id:
            return []
        
        # Find all possible transitions.
        possible_transitions = [
            t for t in self.transitions
            if t['from_state'] == self.current_state_id
        ]
        
        # Filter by condition and visit count.
        valid_transitions = []
        for trans in possible_transitions:
            # Check whether the target state exceeded the visit limit.
            target_state = trans['to_state']
            if self.visit_counts[target_state] >= self.max_visits:
                # Final states are exempt from the visit limit.
                target_state_obj = self.states.get(target_state)
                if not target_state_obj or not target_state_obj.get('is_final', False):
                    continue
            
            # Check the transition condition.
            if self._check_transition_condition(trans, agent_output):
                valid_transitions.append(trans)
        
        # Sort by priority.
        valid_transitions.sort(key=lambda t: t.get('priority', 1))
        
        return valid_transitions
    
    def _check_transition_condition(self, transition: Dict, agent_output: str) -> bool:
        """
        Check whether a transition condition is satisfied.
        
        Args:
            transition: Transition object.
            agent_output: Agent output.
        
        Returns:
            Whether the condition is satisfied.
        """
        condition = transition.get('condition', '')
        if not condition:
            return True  # Unconditional transition.
        
        condition_lower = condition.lower()
        
        # 1. Check for required markers.
        if 'contains' in condition_lower or 'has' in condition_lower:
            # Extract markers, for example: "contains <DONE>".
            markers = re.findall(r'<([^>]+)>', condition)
            if markers:
                for marker in markers:
                    if f'<{marker}>' in agent_output:
                        return True
                return False
        
        # 2. Check keywords.
        keywords = ['complete', 'success', 'done', 'finish', 'ready']
        for keyword in keywords:
            if keyword in condition_lower and keyword in agent_output.lower():
                return True
        
        # 3. Check failure/error conditions.
        fail_keywords = ['fail', 'error', 'issue', 'problem']
        for keyword in fail_keywords:
            if keyword in condition_lower and keyword in agent_output.lower():
                return True
        
        # 4. Default: check for DONE or COMPLETE markers.
        if '<DONE>' in agent_output or '<COMPLETE>' in agent_output:
            return True
        
        # 5. If the condition is very simple, treat it as satisfied by default.
        if len(condition.split()) <= 5:
            return True
        
        return False
    
    def select_next_state(self, 
                         agent_output: str, 
                         tgn_predictor=None) -> Optional[str]:
        """
        Select the next state.
        
        Args:
            agent_output: Agent output.
            tgn_predictor: Optional TGN predictor used when multiple valid transitions exist.
        
        Returns:
            Next state ID, or `None` if no transition is possible.
        """
        valid_transitions = self.get_valid_transitions(agent_output)
        
        if not valid_transitions:
            # No valid transition: try a rescue transition.
            rescue_transitions = [
                t for t in self.transitions
                if t['from_state'] == self.current_state_id 
                and t.get('is_rescue', False)
            ]
            if rescue_transitions:
                return rescue_transitions[0]['to_state']
            
            return None  # Truly stuck.
        
        # If there is only one valid transition, use it directly.
        if len(valid_transitions) == 1:
            return valid_transitions[0]['to_state']
        
        # Multiple valid transitions: use TGN prediction if available.
        if tgn_predictor:
            try:
                next_state = tgn_predictor.predict_next_state(
                    self.current_state_id,
                    valid_transitions
                )
                return next_state
            except:
                pass
        
        # Fallback: choose the highest-priority transition.
        return valid_transitions[0]['to_state']
    
    def execute_transition(self, next_state_id: str) -> bool:
        """
        Execute a state transition.
        
        Args:
            next_state_id: Target state ID.
        
        Returns:
            Whether the transition succeeded.
        """
        if next_state_id not in self.states:
            return False
        
        # Record history.
        if self.current_state_id:
            self.transition_history.append((self.current_state_id, next_state_id))
        
        # Execute the transition.
        self.current_state_id = next_state_id
        self.visit_counts[next_state_id] += 1
        self.step_count += 1
        self.state_history.append(next_state_id)
        
        return True
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get an execution summary."""
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
        """Print the execution trace."""
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


# ========== Convenience Function ==========

def create_fsm_executor(fsm: Dict, 
                       max_visits_per_state: int = 3,
                       max_total_steps: int = 20) -> FSMExecutor:
    """Create an FSM executor."""
    return FSMExecutor(fsm, max_visits_per_state, max_total_steps)


# ========== Test Example ==========

if __name__ == "__main__":
    # Create a test FSM that contains a loop.
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
            {'from_state': '2', 'to_state': '1', 'condition': 'contains <RETRY>', 'priority': 1},  # Backtrack
            {'from_state': '2', 'to_state': '3', 'condition': 'contains <DONE>', 'priority': 1},
            {'from_state': '2', 'to_state': '3', 'condition': '', 'priority': 99, 'is_rescue': True},
        ]
    }
    
    # Create the executor.
    executor = create_fsm_executor(test_fsm, max_visits_per_state=3, max_total_steps=10)
    
    # Simulate execution.
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
    
    # Step 3: Verify → Process (backtrack)
    print("\nStep 3: Verify state (retry)")
    next_state = executor.select_next_state("<RETRY>")
    print(f"  Next state: {next_state} (backtrack)")
    executor.execute_transition(next_state)
    
    # Step 4: Process → Verify (again)
    print("\nStep 4: Process state (again)")
    next_state = executor.select_next_state("<DONE>")
    print(f"  Next state: {next_state}")
    executor.execute_transition(next_state)
    
    # Step 5: Verify → Final
    print("\nStep 5: Verify state (final)")
    next_state = executor.select_next_state("<DONE>")
    print(f"  Next state: {next_state}")
    executor.execute_transition(next_state)
    
    # Print the execution trace.
    executor.print_execution_trace()
    
    # Get the summary.
    summary = executor.get_execution_summary()
    print("\nExecution Summary:")
    import json
    print(json.dumps(summary, indent=2))

"""
FSM Validator and Optimizer

Features:
1. Check FSM reachability to ensure final states can be reached.
2. Detect loops to avoid infinite cycling.
3. Automatically add rescue transitions.
4. Generate state transition conditions.
5. Optimize FSM structure.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque, defaultdict
import json


class FSMValidator:
    """FSM validator."""
    
    def __init__(self, fsm: Dict[str, Any]):
        """
        Initialize the validator.
        
        Args:
            fsm: FSM structure dictionary containing `states` and `transitions`.
        """
        self.fsm = fsm
        self.states = {state['state_id']: state for state in fsm['states']}
        self.transitions = fsm['transitions']
        
        # Build the adjacency list.
        self.graph = defaultdict(list)
        for trans in self.transitions:
            self.graph[trans['from_state']].append(trans['to_state'])
    
    def check_reachability(self, verbose: bool = False) -> Tuple[bool, List[str]]:
        """
        Check FSM reachability from an initial state to a final state.
        
        Args:
            verbose: Whether to print detailed information.
        
        Returns:
            (is_reachable, unreachable_final_states)
        """
        # Find initial states and final states.
        initial_states = [s['state_id'] for s in self.fsm['states'] if s.get('is_initial', False)]
        final_states = [s['state_id'] for s in self.fsm['states'] if s.get('is_final', False)]
        
        if not initial_states:
            if verbose:
                print("❌ No initial state found!")
            return False, final_states
        
        if not final_states:
            if verbose:
                print("❌ No final state found!")
            return False, []
        
        # Run BFS from the initial states.
        visited = set()
        queue = deque(initial_states)
        reachable_final_states = []
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            # Check whether the current state is final.
            if current in final_states:
                reachable_final_states.append(current)
            
            # Add neighbors.
            for next_state in self.graph.get(current, []):
                if next_state not in visited:
                    queue.append(next_state)
        
        # Check whether all final states are reachable.
        unreachable_final_states = [fs for fs in final_states if fs not in reachable_final_states]
        
        if verbose:
            print(f"\n🔍 FSM Reachability Check:")
            print(f"  Initial States: {initial_states}")
            print(f"  Final States: {final_states}")
            print(f"  Reachable Final States: {reachable_final_states}")
            if unreachable_final_states:
                print(f"  ⚠️  Unreachable Final States: {unreachable_final_states}")
        
        return len(reachable_final_states) > 0, unreachable_final_states
    
    def detect_strongly_connected_components(self) -> List[List[str]]:
        """
        Detect strongly connected components for loop detection.
        
        Uses Tarjan's algorithm.
        
        Returns:
            List of strongly connected components.
        """
        index_counter = [0]
        stack = []
        lowlink = {}
        index = {}
        on_stack = defaultdict(lambda: False)
        sccs = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            for successor in self.graph.get(node, []):
                if successor not in index:
                    strongconnect(successor)
                    lowlink[node] = min(lowlink[node], lowlink[successor])
                elif on_stack[successor]:
                    lowlink[node] = min(lowlink[node], index[successor])
            
            if lowlink[node] == index[node]:
                scc = []
                while True:
                    successor = stack.pop()
                    on_stack[successor] = False
                    scc.append(successor)
                    if successor == node:
                        break
                sccs.append(scc)
        
        for node in self.states.keys():
            if node not in index:
                strongconnect(node)
        
        return sccs
    
    def check_infinite_loops(self, verbose: bool = False) -> Tuple[bool, List[List[str]]]:
        """
        Check whether possible infinite loops exist.
        
        Args:
            verbose: Whether to print detailed information.
        
        Returns:
            (has_loops, loop_components)
        """
        sccs = self.detect_strongly_connected_components()
        
        # Find SCCs with size > 1, which may indicate loops.
        loops = [scc for scc in sccs if len(scc) > 1]
        
        if verbose:
            print(f"\n🔄 Loop Detection:")
            if loops:
                print(f"  ⚠️  Found {len(loops)} potential loop(s):")
                for i, loop in enumerate(loops, 1):
                    print(f"    Loop {i}: {' → '.join(loop)}")
            else:
                print(f"  ✅ No loops detected")
        
        return len(loops) > 0, loops
    
    def add_rescue_transitions(self, priority: int = 99, verbose: bool = False) -> int:
        """
        Add rescue transitions to ensure all states can reach a final state.
        
        Args:
            priority: Rescue transition priority, lowest by default.
            verbose: Whether to print detailed information.
        
        Returns:
            Number of transitions added.
        """
        is_reachable, unreachable_finals = self.check_reachability(verbose=False)
        
        if is_reachable and not unreachable_finals:
            if verbose:
                print("✅ FSM already has full reachability, no rescue transitions needed")
            return 0
        
        # Find all non-final states.
        final_state_ids = [s['state_id'] for s in self.fsm['states'] if s.get('is_final', False)]
        non_final_states = [s for s in self.fsm['states'] if not s.get('is_final', False)]
        
        if not final_state_ids:
            if verbose:
                print("❌ No final state exists, cannot add rescue transitions")
            return 0
        
        # Use the first final state as the rescue target.
        rescue_target = final_state_ids[0]
        rescue_target_name = self.states[rescue_target].get('state_name', rescue_target)
        
        added_count = 0
        for state in non_final_states:
            # Check whether a transition to a final state already exists.
            has_path_to_final = any(
                trans['from_state'] == state['state_id'] and trans['to_state'] in final_state_ids
                for trans in self.transitions
            )
            
            if not has_path_to_final:
                # Add a rescue transition.
                rescue_transition = {
                    'from_state': state['state_id'],
                    'to_state': rescue_target,
                    'condition': f"If maximum steps reached or all attempts exhausted, finalize from {state.get('state_name', state['state_id'])}",
                    'priority': priority,
                    'is_rescue': True  # Mark as a rescue transition.
                }
                self.transitions.append(rescue_transition)
                self.graph[state['state_id']].append(rescue_target)
                added_count += 1
                
                if verbose:
                    print(f"  ➕ Added rescue: {state.get('state_name', state['state_id'])} → {rescue_target_name}")
        
        if verbose:
            print(f"\n✅ Added {added_count} rescue transition(s)")
        
        return added_count
    
    def validate_all(self, auto_fix: bool = True, verbose: bool = True) -> Dict[str, Any]:
        """
        Run all validation checks and return the results.
        
        Args:
            auto_fix: Whether to automatically fix issues.
            verbose: Whether to print detailed information.
        
        Returns:
            Validation result dictionary.
        """
        if verbose:
            print("\n" + "=" * 80)
            print("FSM Validation Report")
            print("=" * 80)
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'fixes_applied': []
        }
        
        # 1. Check the basic structure.
        if not self.fsm.get('states'):
            results['is_valid'] = False
            results['errors'].append("No states defined")
        
        if not self.fsm.get('transitions'):
            results['warnings'].append("No transitions defined")
        
        # 2. Check initial and final states.
        initial_states = [s for s in self.fsm['states'] if s.get('is_initial', False)]
        final_states = [s for s in self.fsm['states'] if s.get('is_final', False)]
        
        if len(initial_states) == 0:
            results['is_valid'] = False
            results['errors'].append("No initial state defined")
        elif len(initial_states) > 1:
            results['warnings'].append(f"Multiple initial states: {[s['state_id'] for s in initial_states]}")
        
        if len(final_states) == 0:
            results['is_valid'] = False
            results['errors'].append("No final state defined")
        
        # 3. Check reachability.
        is_reachable, unreachable = self.check_reachability(verbose=verbose)
        if not is_reachable:
            results['is_valid'] = False
            results['errors'].append("Final states are not reachable from initial state")
            
            if auto_fix:
                added = self.add_rescue_transitions(verbose=verbose)
                results['fixes_applied'].append(f"Added {added} rescue transitions")
        
        # 4. Check loops.
        has_loops, loops = self.check_infinite_loops(verbose=verbose)
        if has_loops:
            results['warnings'].append(f"Found {len(loops)} potential loop(s) - ensure visit count limits are enforced")
        
        # 5. Check transition conditions.
        transitions_without_condition = [
            t for t in self.transitions 
            if not t.get('condition') or t['condition'].strip() == ''
        ]
        if transitions_without_condition:
            results['warnings'].append(f"{len(transitions_without_condition)} transitions missing conditions")
        
        if verbose:
            print("\n" + "=" * 80)
            print("Validation Summary:")
            print("=" * 80)
            print(f"Valid: {'✅ Yes' if results['is_valid'] else '❌ No'}")
            if results['errors']:
                print(f"Errors: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"  ❌ {error}")
            if results['warnings']:
                print(f"Warnings: {len(results['warnings'])}")
                for warning in results['warnings']:
                    print(f"  ⚠️  {warning}")
            if results['fixes_applied']:
                print(f"Fixes Applied: {len(results['fixes_applied'])}")
                for fix in results['fixes_applied']:
                    print(f"  🔧 {fix}")
            print("=" * 80 + "\n")
        
        return results


class FSMTransitionConditionGenerator:
    """FSM transition condition generator."""
    
    @staticmethod
    def generate_condition_from_llm(state_from: Dict, state_to: Dict, llm) -> str:
        """
        Use an LLM to generate a state transition condition.
        
        Args:
            state_from: Source state.
            state_to: Target state.
            llm: LLM instance.
        
        Returns:
            Transition condition string.
        """
        from_name = state_from.get('state_name', state_from['state_id'])
        from_desc = state_from.get('instruction', 'No description')
        to_name = state_to.get('state_name', state_to['state_id'])
        to_desc = state_to.get('instruction', 'No description')
        
        prompt = f"""Given two states in a problem-solving Finite State Machine:

FROM State: "{from_name}"
Task: {from_desc}

TO State: "{to_name}"
Task: {to_desc}

Generate a clear, testable transition condition that determines when the system should transition from FROM state to TO state.

Requirements:
1. Be based on observable outputs (e.g., "output contains <TAG>", "calculation result present")
2. Be specific and actionable
3. Handle both success and failure cases
4. Use structured markers when possible

Output ONLY the condition as a single sentence, no explanation:
"""
        
        try:
            condition = llm.chat(prompt).strip()
            # Remove possible surrounding quotes.
            condition = condition.strip('"').strip("'")
            return condition
        except Exception as e:
            # Fallback strategy: generate a default condition.
            return f"If {from_name} task completed, proceed to {to_name}"
    
    @staticmethod
    def generate_default_condition(state_from: Dict, state_to: Dict) -> str:
        """
        Generate a default transition condition without using an LLM.
        
        Args:
            state_from: Source state.
            state_to: Target state.
        
        Returns:
            Default transition condition.
        """
        from_name = state_from.get('state_name', state_from['state_id'])
        to_name = state_to.get('state_name', state_to['state_id'])
        
        # Check whether a completion_condition field exists.
        if 'completion_condition' in state_from:
            return f"When {from_name} completion condition is met: {state_from['completion_condition']}"
        
        # Check whether the target is a final state.
        if state_to.get('is_final', False):
            return f"When {from_name} completes and ready to submit final answer"
        
        # Default condition.
        return f"When {from_name} task completed successfully, proceed to {to_name}"
    
    @staticmethod
    def add_conditions_to_transitions(fsm: Dict, llm=None, use_llm: bool = False) -> int:
        """
        Add conditions to FSM transitions that are missing them.
        
        Args:
            fsm: FSM structure.
            llm: LLM instance when `use_llm=True`.
            use_llm: Whether to use an LLM to generate conditions.
        
        Returns:
            Number of conditions added.
        """
        states_dict = {s['state_id']: s for s in fsm['states']}
        added_count = 0
        
        for trans in fsm['transitions']:
            if not trans.get('condition') or trans['condition'].strip() == '':
                state_from = states_dict.get(trans['from_state'])
                state_to = states_dict.get(trans['to_state'])
                
                if not state_from or not state_to:
                    continue
                
                if use_llm and llm:
                    condition = FSMTransitionConditionGenerator.generate_condition_from_llm(
                        state_from, state_to, llm
                    )
                else:
                    condition = FSMTransitionConditionGenerator.generate_default_condition(
                        state_from, state_to
                    )
                
                trans['condition'] = condition
                if 'priority' not in trans:
                    trans['priority'] = 1  # Default priority.
                
                added_count += 1
        
        return added_count


# ========== Convenience Functions ==========

def validate_fsm(fsm: Dict, auto_fix: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Validate an FSM."""
    validator = FSMValidator(fsm)
    return validator.validate_all(auto_fix=auto_fix, verbose=verbose)


def check_fsm_reachability(fsm: Dict, verbose: bool = False) -> bool:
    """Check FSM reachability."""
    validator = FSMValidator(fsm)
    is_reachable, _ = validator.check_reachability(verbose=verbose)
    return is_reachable


def add_transition_conditions(fsm: Dict, llm=None, use_llm: bool = False) -> int:
    """Add conditions to transitions."""
    return FSMTransitionConditionGenerator.add_conditions_to_transitions(fsm, llm, use_llm)


# ========== Test Example ==========

if __name__ == "__main__":
    # Create a test FSM.
    test_fsm = {
        'states': [
            {'state_id': '0', 'state_name': 'Initial', 'is_initial': True, 'is_final': False, 'instruction': 'Start'},
            {'state_id': '1', 'state_name': 'Process', 'is_initial': False, 'is_final': False, 'instruction': 'Process data'},
            {'state_id': '2', 'state_name': 'Verify', 'is_initial': False, 'is_final': False, 'instruction': 'Verify result'},
            {'state_id': '3', 'state_name': 'Final', 'is_initial': False, 'is_final': True, 'instruction': 'Submit answer'},
        ],
        'transitions': [
            {'from_state': '0', 'to_state': '1', 'condition': 'Initial setup complete', 'priority': 1},
            {'from_state': '1', 'to_state': '2', 'condition': 'Processing complete', 'priority': 1},
            # The transition from 2 to 3 is missing and will be detected and fixed.
        ]
    }
    
    print("Testing FSM Validator...")
    print("\nOriginal FSM:")
    print(json.dumps(test_fsm, indent=2))
    
    # Validate and automatically fix it.
    results = validate_fsm(test_fsm, auto_fix=True, verbose=True)
    
    print("\nFixed FSM:")
    print(json.dumps(test_fsm, indent=2))

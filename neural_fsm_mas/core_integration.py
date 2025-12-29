"""
Core Integration Module for Enhanced FSM System
核心集成模块 - 整合所有新功能

This module integrates:
1. FSM Validation and Optimization
2. FSM Executor with Loop Prevention  
3. LLM Cost Tracking
4. Four-Objective Combined Loss
5. Transition Condition Matching
6. State-Agent Correspondence Learning via TGN

Usage in experiments:
    from neural_fsm_mas.core_integration import EnhancedFSMIntegration
    
    integration = EnhancedFSMIntegration(config)
    fsm = integration.generate_and_validate_fsm(dataset)
    integration.integrate_with_trainer(trainer)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

# Import all new modules
from neural_fsm_mas.agent_topology.fsm_validator import (
    FSMValidator, 
    FSMTransitionConditionGenerator,
    validate_fsm,
    add_transition_conditions
)
from neural_fsm_mas.agent_topology.fsm_executor import FSMExecutor, create_fsm_executor
from neural_fsm_mas.utils.llm_cost_tracker import (
    LLMCostTracker,
    create_cost_tracker,
    estimate_cost,
    set_global_cost_tracker,
)
from neural_fsm_mas.losses.cost_loss import (
    CostLoss,
    AdaptiveCostLoss,
    CostRegularizedLoss,
    estimate_baseline_cost
)


class EnhancedFSMIntegration:
    """
    Enhanced FSM System Integration
    
    Integrates all new components into the training pipeline:
    - FSM validation and optimization
    - Cost tracking and four-objective loss
    - Transition condition matching
    - Loop prevention via FSMExecutor
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize integration
        
        Args:
            config: Configuration dictionary with keys:
                - model_name: LLM model name (e.g., 'gpt-5-nano')
                - alpha, beta, gamma, delta: Loss weights
                - max_visits_per_state: Loop prevention threshold
                - max_total_steps: Maximum episode steps
                - use_azure: Whether to use Azure OpenAI
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cost tracker
        self.cost_tracker = create_cost_tracker(
            config.get('model_name', 'gpt-5-nano')
        )
        
        # Estimate baseline cost
        self.baseline_cost = estimate_baseline_cost(
            model_name=config.get('model_name', 'gpt-5-nano'),
            avg_input_tokens=config.get('avg_input_tokens', 500),
            avg_output_tokens=config.get('avg_output_tokens', 200),
            avg_calls_per_episode=config.get('avg_calls_per_episode', 5)
        )
        
        # Four-objective loss function ✨
        self.loss_fn = CostRegularizedLoss(
            alpha=config.get('alpha', 1.0),      # Policy gradient
            beta=config.get('beta', 0.3),        # State transition
            gamma=config.get('gamma', 0.2),      # Listener path
            delta=config.get('delta', 0.1),      # LLM cost ✨ NEW
            baseline_cost=self.baseline_cost
        )
        
        # FSM components
        self.fsm = None
        self.fsm_executor = None
        self.fsm_validator = None
        
        print(f"✅ EnhancedFSMIntegration initialized")
        print(f"   Device: {self.device}")
        print(f"   Baseline Cost: ${self.baseline_cost:.6f}")
        print(f"   Loss Weights: α={config.get('alpha', 1.0)}, "
              f"β={config.get('beta', 0.3)}, "
              f"γ={config.get('gamma', 0.2)}, "
              f"δ={config.get('delta', 0.1)} ✨")
    
    def generate_and_validate_fsm(self, 
                                  fsm: Dict[str, Any],
                                  auto_fix: bool = True,
                                  use_llm_for_conditions: bool = False) -> Dict[str, Any]:
        """
        Generate and validate FSM with all transitions having explicit conditions
        
        This addresses your question: "一开始就生成了FSM中的所有状态之间的转移条件"
        
        Args:
            fsm: FSM structure from Enhanced_FSM_Gen or other sources
            auto_fix: Whether to auto-fix issues (add rescue transitions)
            use_llm_for_conditions: Whether to use LLM to generate missing conditions
        
        Returns:
            Validated and enhanced FSM with all transition conditions defined
        """
        print(f"\n{'='*80}")
        print("FSM Validation and Enhancement")
        print(f"{'='*80}")
        
        # Step 1: Validate FSM structure
        validation_results = validate_fsm(fsm, auto_fix=auto_fix, verbose=True)
        
        if not validation_results['is_valid']:
            raise ValueError("FSM validation failed: " + str(validation_results['errors']))
        
        # Step 2: Ensure all transitions have explicit conditions
        # 关键点：确保每个转移都有明确的条件
        print("\n🔍 Ensuring all transitions have explicit conditions...")
        
        missing_conditions = 0
        for trans in fsm.get('transitions', []):
            if not trans.get('condition') or trans['condition'].strip() == '':
                missing_conditions += 1
        
        if missing_conditions > 0:
            print(f"⚠️  Found {missing_conditions} transitions without conditions")
            
            if use_llm_for_conditions:
                # Use LLM to generate conditions (more accurate but slower)
                print("   Using LLM to generate transition conditions...")
                from baseclass.LLM import LLM
                llm = LLM("", use_azure=self.config.get('use_azure', False))
                added = add_transition_conditions(fsm, llm=llm, use_llm=True)
            else:
                # Use rule-based generation (faster)
                print("   Using rule-based condition generation...")
                added = add_transition_conditions(fsm, llm=None, use_llm=False)
            
            print(f"✅ Added {added} transition conditions")
        else:
            print("✅ All transitions already have conditions")
        
        # Step 3: Create validator and executor
        self.fsm = fsm
        self.fsm_validator = FSMValidator(fsm)
        self.fsm_executor = create_fsm_executor(
            fsm,
            max_visits_per_state=self.config.get('max_visits_per_state', 3),
            max_total_steps=self.config.get('max_total_steps', 20)
        )
        
        print(f"\n✅ FSM ready with {len(fsm['states'])} states and {len(fsm['transitions'])} transitions")
        print(f"   All transitions have explicit matching conditions")
        
        return fsm
    
    def get_transition_for_sampled_path(self, 
                                        from_state_id: str, 
                                        to_state_id: str) -> Optional[Dict]:
        """
        Get transition condition for a sampled state path
        
        This addresses: "采样到哪些状态及他们之间的转移路径就匹配对应的转移条件"
        
        When TGN samples a path (e.g., state 0 → state 2),
        we retrieve the pre-defined transition condition between them.
        
        Args:
            from_state_id: Source state ID
            to_state_id: Target state ID
        
        Returns:
            Transition dict with condition, or None if no such transition exists
        """
        if not self.fsm:
            return None
        
        for trans in self.fsm.get('transitions', []):
            if trans['from_state'] == from_state_id and trans['to_state'] == to_state_id:
                return trans
        
        return None
    
    def match_transition_condition(self, 
                                   agent_output: str,
                                   from_state_id: str,
                                   candidate_next_states: List[str]) -> Optional[str]:
        """
        Match agent output against transition conditions to select next state
        
        This is the core condition matching logic mentioned in docs:
        "执行时自动匹配条件"
        
        Args:
            agent_output: Output from the agent
            from_state_id: Current state ID
            candidate_next_states: Possible next states (from TGN sampling or all transitions)
        
        Returns:
            Next state ID whose condition is matched, or None
        """
        if not self.fsm_executor:
            return None
        
        # Get valid transitions from current state
        valid_transitions = self.fsm_executor.get_valid_transitions(agent_output)
        
        # Filter by candidate states
        for trans in valid_transitions:
            if trans['to_state'] in candidate_next_states:
                return trans['to_state']
        
        return None
    
    def compute_four_objective_loss(self,
                                    policy_loss: torch.Tensor,
                                    transition_loss: torch.Tensor,
                                    listener_loss: torch.Tensor,
                                    episode_costs: List[float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute four-objective combined loss
        
        L_total = α·L_policy + β·L_transition + γ·L_listener + δ·L_cost
        
        Args:
            policy_loss: Policy gradient loss (accuracy)
            transition_loss: State transition loss
            listener_loss: Listener path loss
            episode_costs: List of episode costs in USD
        
        Returns:
            (total_loss, loss_dict)
        """
        # Convert costs to tensor
        cost_tensor = torch.tensor(episode_costs, dtype=torch.float32, device=self.device)
        
        # Compute combined loss
        total_loss, loss_dict = self.loss_fn(
            policy_loss,
            transition_loss,
            listener_loss,
            cost_tensor
        )
        
        return total_loss, loss_dict
    
    def integrate_with_trainer(self, trainer: Any):
        """
        Integrate all components with an existing trainer
        
        This method injects the new functionality into existing trainer classes.
        
        Args:
            trainer: A trainer instance (e.g., NeuralMASTrainer, FSMNeuralMASTrainer)
        """
        # Inject cost tracker
        if not hasattr(trainer, 'cost_tracker'):
            trainer.cost_tracker = self.cost_tracker
        # 同时将成本追踪器注册为全局追踪器，供LLM客户端在任意位置记录调用成本
        set_global_cost_tracker(trainer.cost_tracker)
        
        # Inject FSM executor
        if not hasattr(trainer, 'fsm_executor'):
            trainer.fsm_executor = self.fsm_executor
        
        # Inject loss function
        if not hasattr(trainer, 'cost_loss_fn'):
            trainer.cost_loss_fn = self.loss_fn
        
        # Inject matching function
        trainer.match_transition_condition = self.match_transition_condition
        
        # Inject cost computation
        trainer.compute_four_objective_loss = self.compute_four_objective_loss
        
        print("✅ Integrated with trainer:")
        print("   - Cost tracker injected")
        print("   - FSM executor injected")
        print("   - Four-objective loss function injected")
        print("   - Transition matching function injected")
    
    def save_fsm(self, save_path: str):
        """Save validated FSM to file"""
        if self.fsm:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.fsm, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved FSM to: {save_path}")


# ========== Convenience Functions ==========

def create_enhanced_integration(config: Dict[str, Any]) -> EnhancedFSMIntegration:
    """Create EnhancedFSMIntegration instance"""
    return EnhancedFSMIntegration(config)


def integrate_four_objective_loss(trainer: Any, config: Dict[str, Any]):
    """
    Quick integration of four-objective loss into existing trainer
    
    Usage:
        from neural_fsm_mas.core_integration import integrate_four_objective_loss
        
        trainer = MyTrainer(config)
        integrate_four_objective_loss(trainer, config)
    """
    integration = EnhancedFSMIntegration(config)
    integration.integrate_with_trainer(trainer)
    return integration


# ========== State-Agent Correspondence Learning ==========

class StateAgentCorrespondenceLearner:
    """
    Learn state-to-agent correspondence via TGN
    
    This addresses: "采样的状态对应的agent也是采样的，即这种对应关系可以通过TGN学习的？"
    
    YES! The state-agent correspondence can be learned by TGN.
    
    How it works:
    1. Initially, each state is assigned to an agent (can be random or by LLM)
    2. During training, TGN learns which agent is best for each state
    3. The assignment matrix is part of TGN's learnable parameters
    4. Gradient descent optimizes the assignment based on task performance
    """
    
    def __init__(self, num_states: int, num_agents: int, device='cpu'):
        """
        Initialize learnable state-agent correspondence
        
        Args:
            num_states: Number of states in FSM
            num_agents: Number of available agents
            device: torch device
        """
        self.num_states = num_states
        self.num_agents = num_agents
        self.device = device
        
        # Learnable assignment matrix: [num_states, num_agents]
        # assignment[i, j] = probability that state i uses agent j
        self.assignment_logits = nn.Parameter(
            torch.randn(num_states, num_agents, device=device)
        )
        
        print(f"✅ StateAgentCorrespondenceLearner initialized")
        print(f"   {num_states} states × {num_agents} agents")
        print(f"   Learnable parameters: {num_states * num_agents}")
    
    def get_agent_for_state(self, state_id: int, use_sampling: bool = False) -> int:
        """
        Get agent for a given state
        
        Args:
            state_id: State index
            use_sampling: If True, sample from distribution; if False, use argmax
        
        Returns:
            Agent index
        """
        probs = torch.softmax(self.assignment_logits[state_id], dim=0)
        
        if use_sampling:
            agent_id = torch.multinomial(probs, num_samples=1).item()
        else:
            agent_id = torch.argmax(probs).item()
        
        return agent_id
    
    def get_assignment_matrix(self) -> torch.Tensor:
        """Get soft assignment matrix (probabilities)"""
        return torch.softmax(self.assignment_logits, dim=-1)
    
    def compute_assignment_loss(self, 
                                state_trajectory: List[int],
                                agent_trajectory: List[int],
                                rewards: List[float]) -> torch.Tensor:
        """
        Compute loss for state-agent assignment
        
        This encourages assigning states to agents that yield high rewards
        
        Args:
            state_trajectory: List of state indices visited
            agent_trajectory: List of agents actually used
            rewards: Rewards for each step
        
        Returns:
            Assignment loss
        """
        loss = 0.0
        for state_id, agent_id, reward in zip(state_trajectory, agent_trajectory, rewards):
            # Get probability of this assignment
            probs = torch.softmax(self.assignment_logits[state_id], dim=0)
            log_prob = torch.log(probs[agent_id] + 1e-8)
            
            # Policy gradient: maximize log_prob * reward
            loss = loss - log_prob * reward
        
        return loss / len(state_trajectory)


if __name__ == "__main__":
    print("="*80)
    print("Core Integration Module Test")
    print("="*80)
    
    # Test configuration
    config = {
        'model_name': 'gpt-5-nano',
        'alpha': 1.0,
        'beta': 0.3,
        'gamma': 0.2,
        'delta': 0.1,
        'max_visits_per_state': 3,
        'max_total_steps': 20
    }
    
    # Create integration
    integration = create_enhanced_integration(config)
    
    # Test FSM structure
    test_fsm = {
        'states': [
            {'state_id': '0', 'state_name': 'Start', 'agent_id': '0', 'is_initial': True, 'is_final': False},
            {'state_id': '1', 'state_name': 'Process', 'agent_id': '1', 'is_initial': False, 'is_final': False},
            {'state_id': '2', 'state_name': 'Final', 'agent_id': '2', 'is_initial': False, 'is_final': True},
        ],
        'transitions': [
            {'from_state': '0', 'to_state': '1', 'condition': 'Start complete', 'priority': 1},
            {'from_state': '1', 'to_state': '2', 'condition': 'Process complete', 'priority': 1},
        ]
    }
    
    # Validate FSM
    validated_fsm = integration.generate_and_validate_fsm(test_fsm)
    
    print("\n✅ Test passed!")


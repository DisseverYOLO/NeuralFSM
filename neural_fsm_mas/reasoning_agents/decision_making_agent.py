"""
Decision Making Agent
Decision-making agent

Adapted from the original FinalDecision implementation and optimized for the MetaAgent project
"""

from typing import List, Any, Dict
import sys
from pathlib import Path

# Add MetaAgent path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from neural_fsm_mas.agent_topology.agent_node import AgentExecutionNode
from neural_fsm_mas.reasoning_agents.agent_factory import ReasoningAgentRegistry
from baseclass.LLM import LLM


@ReasoningAgentRegistry.register_agent_type('final_decision')
class DecisionMakingAgent(AgentExecutionNode):
    """
    Decision-making agent

    Responsible for synthesizing outputs from all agents and making the final decision
    """
    
    def __init__(self, 
                 node_id: str = None, 
                 agent_role: str = "Final Decision Maker",
                 domain: str = "decision", 
                 llm_name: str = "gpt-4o-mini",
                 **kwargs):
        super().__init__(node_id, agent_role, domain, llm_name, **kwargs)
        
        self.reasoning_llm = LLM(
            system_prompt=self._get_decision_making_prompt(),
            use_azure=kwargs.get('use_azure', False)
        )
    
    def _get_decision_making_prompt(self) -> str:
        """Get the system prompt for decision making."""
        return """You are the final decision-making agent. Your role is to:

1. Synthesize all available information from other agents
2. Evaluate different perspectives and solutions
3. Make the final, well-reasoned decision
4. Provide clear justification for your decision
5. Ensure the decision addresses the original problem effectively

Guidelines:
- Consider all agent inputs carefully
- Look for consensus and identify disagreements
- Weigh the credibility and reasoning quality of each input
- Make decisions based on the strongest evidence and logic
- Provide clear, actionable final answers"""
    
    def _execute_single_reasoning(self, reasoning_context: Dict[str, Any], **execution_kwargs) -> Any:
        """Execute decision making."""
        problem = reasoning_context['problem']
        spatial_context = reasoning_context.get('spatial_context', '')
        
        decision_prompt = f"""Original Problem: {problem}

Agent Inputs:
{spatial_context}

Based on all the above information, please make a final decision and provide your reasoning."""
        
        try:
            decision_result = self.reasoning_llm.chat(message=decision_prompt)
            return {
                'final_decision': decision_result,
                'problem': problem,
                'agent_role': self.agent_role
            }
        except Exception as e:
            return {"error": f"Decision making failed: {str(e)}", "problem": problem}


# Export main class
__all__ = ['DecisionMakingAgent']

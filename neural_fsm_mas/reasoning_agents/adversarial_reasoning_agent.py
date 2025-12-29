"""
Adversarial Reasoning Agent
对抗推理智能体

适配自原始AdversarialAgent实现，专门为MetaAgent项目优化
"""

from typing import List, Any, Dict
import sys
from pathlib import Path

# 添加MetaAgent路径
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from neural_fsm_mas.agent_topology.agent_node import AgentExecutionNode
from neural_fsm_mas.reasoning_agents.agent_factory import ReasoningAgentRegistry
from baseclass.LLM import LLM


@ReasoningAgentRegistry.register_agent_type('adversarial_reasoning')
class AdversarialReasoningAgent(AgentExecutionNode):
    """
    对抗推理智能体
    
    专门用于质疑和挑战其他智能体的推理，提供反向思考
    """
    
    def __init__(self, 
                 node_id: str = None, 
                 agent_role: str = "Adversarial Reasoner",
                 domain: str = "adversarial", 
                 llm_name: str = "gpt-4o-mini",
                 **kwargs):
        super().__init__(node_id, agent_role, domain, llm_name, **kwargs)
        
        self.reasoning_llm = LLM(
            system_prompt=self._get_adversarial_reasoning_prompt(),
            use_azure=kwargs.get('use_azure', False)
        )
    
    def _get_adversarial_reasoning_prompt(self) -> str:
        """获取对抗推理的系统提示"""
        return """You are an adversarial reasoning agent. Your role is to:

1. Challenge and question other agents' reasoning
2. Identify potential flaws, biases, or errors
3. Propose alternative perspectives and solutions
4. Test the robustness of proposed answers
5. Play devil's advocate to strengthen overall reasoning

Guidelines:
- Be constructively critical, not destructive
- Look for logical fallacies and weak assumptions
- Consider edge cases and alternative explanations
- Question the evidence and reasoning process
- Propose counter-arguments and alternative solutions
- Help improve the overall quality of reasoning"""
    
    def _execute_single_reasoning(self, reasoning_context: Dict[str, Any], **execution_kwargs) -> Any:
        """执行对抗推理"""
        problem = reasoning_context['problem']
        spatial_context = reasoning_context.get('spatial_context', '')
        
        adversarial_prompt = f"""Original Problem: {problem}

Other Agents' Responses:
{spatial_context}

As an adversarial reasoner, please:
1. Critically examine the above responses
2. Identify potential weaknesses or errors
3. Propose alternative perspectives or solutions
4. Challenge assumptions and reasoning
5. Provide constructive criticism to improve the overall solution quality"""
        
        try:
            adversarial_result = self.reasoning_llm.chat(message=adversarial_prompt)
            return {
                'adversarial_analysis': adversarial_result,
                'problem': problem,
                'agent_role': self.agent_role
            }
        except Exception as e:
            return {"error": f"Adversarial reasoning failed: {str(e)}", "problem": problem}


# 导出主要类
__all__ = ['AdversarialReasoningAgent']

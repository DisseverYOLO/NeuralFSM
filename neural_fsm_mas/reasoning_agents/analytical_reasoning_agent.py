"""
Code Generation Agent
Code generation agent

Adapted from the original CodeWriting implementation and optimized for the MetaAgent project
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


@ReasoningAgentRegistry.register_agent_type('code_generation')
class CodeGenerationAgent(AgentExecutionNode):
    """
    Code generation agent

    Specialized in generating and improving code
    """
    
    def __init__(self, 
                 node_id: str = None, 
                 agent_role: str = "Code Generator",
                 domain: str = "coding", 
                 llm_name: str = "gpt-4o-mini",
                 **kwargs):
        super().__init__(node_id, agent_role, domain, llm_name, **kwargs)
        
        self.reasoning_llm = LLM(
            system_prompt=self._get_code_generation_prompt(),
            use_azure=kwargs.get('use_azure', False)
        )
        
        self.programming_language = kwargs.get('programming_language', 'python')
    
    def _get_code_generation_prompt(self) -> str:
        """Get the system prompt for code generation."""
        return f"""You are an expert code generation agent specializing in {self.programming_language}. Your role is to:

1. Generate clean, efficient, and well-documented code
2. Follow best practices and coding standards
3. Implement proper error handling
4. Write comprehensive comments and documentation
5. Optimize code for readability and performance

Guidelines:
- Write production-quality code
- Include proper imports and dependencies
- Add meaningful variable and function names
- Implement error handling where appropriate
- Provide clear code comments and documentation"""
    
    def _execute_single_reasoning(self, reasoning_context: Dict[str, Any], **execution_kwargs) -> Any:
        """Execute code generation."""
        problem = reasoning_context['problem']
        spatial_context = reasoning_context.get('spatial_context', '')
        
        code_prompt = f"""Code Generation Task: {problem}

Programming Language: {self.programming_language}

Additional Context from Other Agents:
{spatial_context}

Please generate clean, well-documented code that solves the given task."""
        
        try:
            code_result = self.reasoning_llm.chat(message=code_prompt)
            return {
                'generated_code': code_result,
                'programming_language': self.programming_language,
                'problem': problem,
                'agent_role': self.agent_role
            }
        except Exception as e:
            return {"error": f"Code generation failed: {str(e)}", "problem": problem}


# Export main class
__all__ = ['CodeGenerationAgent']

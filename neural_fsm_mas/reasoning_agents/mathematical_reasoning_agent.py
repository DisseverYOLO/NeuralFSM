"""
Mathematical Reasoning Agent
Mathematical reasoning agent

Adapted from the original MathSolver implementation and optimized for the MetaAgent project.
Supports mathematical reasoning and problem solving.
"""

from typing import List, Any, Dict
import sys
from pathlib import Path

# Add the MetaAgent path.
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from neural_fsm_mas.agent_topology.agent_node import AgentExecutionNode
from neural_fsm_mas.reasoning_agents.agent_factory import ReasoningAgentRegistry
from baseclass.LLM import LLM
import json
import re


@ReasoningAgentRegistry.register_agent_type('mathematical_reasoning')
class MathematicalReasoningAgent(AgentExecutionNode):
    """
    Mathematical reasoning agent.
    
    Specialized for mathematical reasoning and problem solving, with support for:
    1. Parsing and computing mathematical expressions.
    2. Multi-step mathematical reasoning.
    3. Collaborative reasoning with other agents.
    4. Context-based mathematical problem solving.
    """
    
    def __init__(self, 
                 node_id: str = None, 
                 agent_role: str = "Mathematical Reasoner",
                 domain: str = "mathematics", 
                 llm_name: str = "gpt-4o-mini",
                 **kwargs):
        super().__init__(node_id, agent_role, domain, llm_name, **kwargs)
        
        # Initialize the LLM.
        self.reasoning_llm = LLM(
            system_prompt=self._get_mathematical_reasoning_prompt(),
            use_azure=kwargs.get('use_azure', False)
        )
        
        # Configuration related to mathematical reasoning.
        self.enable_step_by_step_reasoning = kwargs.get('enable_step_by_step_reasoning', True)
        self.enable_verification = kwargs.get('enable_verification', True)
        self.max_reasoning_steps = kwargs.get('max_reasoning_steps', 10)
        
    def _get_mathematical_reasoning_prompt(self) -> str:
        """Get the system prompt for mathematical reasoning."""
        return """You are an expert mathematical reasoning agent. Your role is to:

1. Analyze mathematical problems step by step
2. Show clear reasoning process for each step
3. Verify your calculations and logic
4. Collaborate with other agents when needed
5. Provide accurate and well-explained solutions

Guidelines:
- Break down complex problems into smaller, manageable steps
- Show all calculations explicitly
- Use proper mathematical notation
- Explain your reasoning clearly
- Double-check your work for accuracy
- Consider multiple approaches when appropriate

When working with other agents:
- Integrate their insights and suggestions
- Build upon their partial solutions
- Identify and resolve any inconsistencies
- Provide constructive feedback on their approaches"""
    
    def _process_reasoning_inputs(self, 
                                raw_task_inputs: List[Any], 
                                spatial_interaction_info: Dict[str, Dict], 
                                temporal_interaction_info: Dict[str, Dict], 
                                **execution_kwargs) -> List[Any]:
        """
        Process mathematical reasoning inputs.
        
        Combine the raw math problem, spatial interaction information
        from other agents, and temporal interaction information from past rounds.
        """
        processed_inputs = []
        
        for raw_input in raw_task_inputs:
            # Extract the math problem.
            if isinstance(raw_input, dict):
                math_problem = raw_input.get('task', str(raw_input))
            else:
                math_problem = str(raw_input)
            
            # Build the reasoning context.
            reasoning_context = {
                'problem': math_problem,
                'spatial_context': self._format_spatial_context(spatial_interaction_info),
                'temporal_context': self._format_temporal_context(temporal_interaction_info),
                'reasoning_mode': 'mathematical'
            }
            
            processed_inputs.append(reasoning_context)
        
        return processed_inputs
    
    def _format_spatial_context(self, spatial_info: Dict[str, Dict]) -> str:
        """Format the spatial interaction context from other agents' current solutions."""
        if not spatial_info:
            return ""
        
        context_parts = ["Current insights from other agents:"]
        
        for agent_id, info in spatial_info.items():
            agent_role = info.get('agent_role', 'Unknown Agent')
            agent_outputs = info.get('execution_outputs', [])
            
            if agent_outputs:
                latest_output = agent_outputs[-1] if agent_outputs else "No output"
                context_parts.append(f"\n{agent_role} (Agent {agent_id}):")
                context_parts.append(f"  Solution: {latest_output}")
                
                # Try to extract a numerical answer.
                extracted_answer = self._extract_numerical_answer(str(latest_output))
                if extracted_answer:
                    context_parts.append(f"  Extracted Answer: {extracted_answer}")
        
        return "\n".join(context_parts)
    
    def _format_temporal_context(self, temporal_info: Dict[str, Dict]) -> str:
        """Format the temporal interaction context from historical solutions."""
        if not temporal_info:
            return ""
        
        context_parts = ["Previous round insights:"]
        
        for agent_id, info in temporal_info.items():
            agent_role = info.get('agent_role', 'Unknown Agent')
            agent_outputs = info.get('execution_outputs', [])
            
            if agent_outputs:
                latest_output = agent_outputs[-1] if agent_outputs else "No output"
                context_parts.append(f"\n{agent_role} (Agent {agent_id}) previously said:")
                context_parts.append(f"  {latest_output}")
        
        return "\n".join(context_parts)
    
    def _extract_numerical_answer(self, text: str) -> str:
        """Extract a numerical answer from text."""
        # Search for common answer patterns.
        patterns = [
            r'(?:answer|result|solution)(?:\s*is\s*|\s*:\s*)([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*(?:is\s+the\s+answer|is\s+correct)',
            r'=\s*([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*$'  # Number at the end of the line.
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        
        return ""
    
    def _execute_single_reasoning(self, reasoning_context: Dict[str, Any], **execution_kwargs) -> Any:
        """
        Execute a single mathematical reasoning task.
        """
        problem = reasoning_context['problem']
        spatial_context = reasoning_context['spatial_context']
        temporal_context = reasoning_context['temporal_context']
        
        # Build the full reasoning prompt.
        reasoning_prompt = self._build_mathematical_reasoning_prompt(
            problem, spatial_context, temporal_context
        )
        
        try:
            # Call the LLM to perform reasoning.
            reasoning_result = self.reasoning_llm.chat(message=reasoning_prompt)
            
            # Post-process the reasoning result.
            processed_result = self._post_process_mathematical_result(reasoning_result)
            
            return processed_result
            
        except Exception as e:
            error_message = f"Mathematical reasoning failed: {str(e)}"
            print(f"Error in {self.node_id}: {error_message}")
            return {"error": error_message, "problem": problem}
    
    def _build_mathematical_reasoning_prompt(self, 
                                           problem: str, 
                                           spatial_context: str, 
                                           temporal_context: str) -> str:
        """Build the mathematical reasoning prompt."""
        prompt_parts = [f"Mathematical Problem: {problem}"]
        
        if spatial_context:
            prompt_parts.append(f"\n{spatial_context}")
            prompt_parts.append("\nPlease consider the above insights from other agents.")
        
        if temporal_context:
            prompt_parts.append(f"\n{temporal_context}")
            prompt_parts.append("\nPlease also consider the previous round's insights.")
        
        if self.enable_step_by_step_reasoning:
            prompt_parts.append("\nPlease solve this step by step:")
            prompt_parts.append("1. Understand the problem")
            prompt_parts.append("2. Identify the mathematical concepts involved")
            prompt_parts.append("3. Plan your solution approach")
            prompt_parts.append("4. Execute the solution with clear calculations")
            prompt_parts.append("5. Verify your answer")
        
        prompt_parts.append("\nProvide a clear, accurate solution with your reasoning process.")
        
        return "\n".join(prompt_parts)
    
    def _post_process_mathematical_result(self, raw_result: str) -> Dict[str, Any]:
        """Post-process the mathematical reasoning result."""
        # Extract the numerical answer.
        numerical_answer = self._extract_numerical_answer(raw_result)
        
        # Analyze the reasoning steps.
        reasoning_steps = self._extract_reasoning_steps(raw_result)
        
        # Build a structured result.
        processed_result = {
            'raw_solution': raw_result,
            'numerical_answer': numerical_answer,
            'reasoning_steps': reasoning_steps,
            'agent_role': self.agent_role,
            'confidence_score': self._estimate_confidence(raw_result)
        }
        
        return processed_result
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps."""
        # Look for numbered steps.
        step_patterns = [
            r'(?:Step\s+\d+|^\d+\.)\s*[:\-]?\s*(.+?)(?=(?:Step\s+\d+|^\d+\.)|$)',
            r'(?:First|Second|Third|Fourth|Fifth|Finally)[:\-,]\s*(.+?)(?=(?:First|Second|Third|Fourth|Fifth|Finally)|$)'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if matches:
                steps.extend([step.strip() for step in matches])
                break
        
        # If no explicit steps are found, split by sentence.
        if not steps:
            sentences = text.split('.')
            steps = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return steps[:self.max_reasoning_steps]  # Limit the number of steps.
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate the confidence of the reasoning result."""
        confidence_indicators = {
            'high': ['verified', 'confirmed', 'double-checked', 'certain', 'definitely'],
            'medium': ['likely', 'probably', 'should be', 'appears to be'],
            'low': ['might be', 'could be', 'uncertain', 'not sure', 'guess']
        }
        
        text_lower = text.lower()
        
        # Check high-confidence indicators.
        high_count = sum(1 for indicator in confidence_indicators['high'] if indicator in text_lower)
        medium_count = sum(1 for indicator in confidence_indicators['medium'] if indicator in text_lower)
        low_count = sum(1 for indicator in confidence_indicators['low'] if indicator in text_lower)
        
        # Compute confidence based on the indicators.
        if high_count > 0:
            base_confidence = 0.8
        elif medium_count > 0:
            base_confidence = 0.6
        elif low_count > 0:
            base_confidence = 0.4
        else:
            base_confidence = 0.7  # Default medium confidence.
        
        # Adjust based on reasoning length because more detailed reasoning is often more reliable.
        length_factor = min(len(text) / 500, 1.0) * 0.2
        
        final_confidence = min(base_confidence + length_factor, 1.0)
        return round(final_confidence, 2)
    
    async def _async_execute_single_reasoning(self, reasoning_context: Dict[str, Any], **execution_kwargs) -> Any:
        """
        Execute a single mathematical reasoning task asynchronously.
        """
        # Mathematical reasoning usually does not need true asynchronous execution,
        # but async LLM call logic could be added here.
        return self._execute_single_reasoning(reasoning_context, **execution_kwargs)
    
    def get_mathematical_capabilities(self) -> Dict[str, Any]:
        """Get a description of the mathematical reasoning capabilities."""
        return {
            'supported_topics': [
                'arithmetic', 'algebra', 'geometry', 'calculus', 
                'statistics', 'probability', 'word_problems'
            ],
            'reasoning_features': [
                'step_by_step_solution', 'verification', 'multi_approach',
                'collaboration_with_other_agents', 'error_detection'
            ],
            'output_formats': [
                'structured_solution', 'numerical_answer', 'reasoning_steps',
                'confidence_estimation'
            ]
        }


# Export main classes
__all__ = ['MathematicalReasoningAgent']

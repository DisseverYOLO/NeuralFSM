"""
Analytical Reasoning Agent
分析推理智能体

适配自原始AnalyzeAgent实现，专门为MetaAgent项目优化
支持复杂问题的分析和推理
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


@ReasoningAgentRegistry.register_agent_type('analytical_reasoning')
class AnalyticalReasoningAgent(AgentExecutionNode):
    """
    分析推理智能体
    
    专门用于复杂问题的分析和推理，支持：
    1. 问题分解和结构化分析
    2. 多角度思考和评估
    3. 逻辑推理和论证
    4. 批判性思维和质疑
    """
    
    def __init__(self, 
                 node_id: str = None, 
                 agent_role: str = "Analytical Reasoner",
                 domain: str = "analysis", 
                 llm_name: str = "gpt-4o-mini",
                 **kwargs):
        super().__init__(node_id, agent_role, domain, llm_name, **kwargs)
        
        self.reasoning_llm = LLM(
            system_prompt=self._get_analytical_reasoning_prompt(),
            use_azure=kwargs.get('use_azure', False)
        )
        
        self.analysis_depth = kwargs.get('analysis_depth', 'comprehensive')
        self.enable_critical_thinking = kwargs.get('enable_critical_thinking', True)
    
    def _get_analytical_reasoning_prompt(self) -> str:
        """获取分析推理的系统提示"""
        return """You are an expert analytical reasoning agent. Your role is to:

1. Break down complex problems into manageable components
2. Analyze problems from multiple perspectives
3. Apply logical reasoning and critical thinking
4. Identify assumptions, biases, and potential issues
5. Synthesize information from various sources
6. Provide well-structured and evidence-based conclusions

Guidelines:
- Use systematic analytical frameworks
- Consider both quantitative and qualitative aspects
- Question assumptions and challenge conventional thinking
- Look for patterns, relationships, and underlying principles
- Evaluate the strength of evidence and arguments
- Consider alternative explanations and solutions"""
    
    def _execute_single_reasoning(self, reasoning_context: Dict[str, Any], **execution_kwargs) -> Any:
        """执行单个分析推理任务"""
        problem = reasoning_context['problem']
        spatial_context = reasoning_context.get('spatial_context', '')
        temporal_context = reasoning_context.get('temporal_context', '')
        
        analysis_prompt = self._build_analytical_prompt(problem, spatial_context, temporal_context)
        
        try:
            analysis_result = self.reasoning_llm.chat(message=analysis_prompt)
            return self._structure_analysis_result(analysis_result, problem)
        except Exception as e:
            return {"error": f"Analytical reasoning failed: {str(e)}", "problem": problem}
    
    def _build_analytical_prompt(self, problem: str, spatial_context: str, temporal_context: str) -> str:
        """构建分析推理提示"""
        prompt_parts = [f"Problem for Analysis: {problem}"]
        
        if spatial_context:
            prompt_parts.append(f"\nCurrent perspectives from other agents:\n{spatial_context}")
        
        if temporal_context:
            prompt_parts.append(f"\nPrevious analysis:\n{temporal_context}")
        
        prompt_parts.extend([
            "\nPlease provide a comprehensive analysis including:",
            "1. Problem decomposition and key components",
            "2. Multiple perspectives and viewpoints",
            "3. Logical reasoning and evidence evaluation",
            "4. Potential assumptions and limitations",
            "5. Synthesis and conclusions"
        ])
        
        return "\n".join(prompt_parts)
    
    def _structure_analysis_result(self, raw_result: str, problem: str) -> Dict[str, Any]:
        """结构化分析结果"""
        return {
            'analysis': raw_result,
            'problem': problem,
            'agent_role': self.agent_role,
            'analysis_type': 'comprehensive_analytical_reasoning'
        }


# 导出主要类
__all__ = ['AnalyticalReasoningAgent']

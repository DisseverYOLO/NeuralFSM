"""
Mathematical Reasoning Agent
数学推理智能体

适配自原始MathSolver实现，专门为MetaAgent项目优化
支持数学问题的推理和求解
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
import json
import re


@ReasoningAgentRegistry.register_agent_type('mathematical_reasoning')
class MathematicalReasoningAgent(AgentExecutionNode):
    """
    数学推理智能体
    
    专门用于处理数学问题的推理和求解，支持：
    1. 数学表达式解析和计算
    2. 多步骤数学推理
    3. 与其他智能体的协作推理
    4. 基于上下文的数学问题求解
    """
    
    def __init__(self, 
                 node_id: str = None, 
                 agent_role: str = "Mathematical Reasoner",
                 domain: str = "mathematics", 
                 llm_name: str = "gpt-4o-mini",
                 **kwargs):
        super().__init__(node_id, agent_role, domain, llm_name, **kwargs)
        
        # 初始化LLM
        self.reasoning_llm = LLM(
            system_prompt=self._get_mathematical_reasoning_prompt(),
            use_azure=kwargs.get('use_azure', False)
        )
        
        # 数学推理相关配置
        self.enable_step_by_step_reasoning = kwargs.get('enable_step_by_step_reasoning', True)
        self.enable_verification = kwargs.get('enable_verification', True)
        self.max_reasoning_steps = kwargs.get('max_reasoning_steps', 10)
        
    def _get_mathematical_reasoning_prompt(self) -> str:
        """获取数学推理的系统提示"""
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
        处理数学推理输入
        
        整合原始数学问题、空间交互信息（其他智能体的解答）和时间交互信息（历史解答）
        """
        processed_inputs = []
        
        for raw_input in raw_task_inputs:
            # 提取数学问题
            if isinstance(raw_input, dict):
                math_problem = raw_input.get('task', str(raw_input))
            else:
                math_problem = str(raw_input)
            
            # 构建推理上下文
            reasoning_context = {
                'problem': math_problem,
                'spatial_context': self._format_spatial_context(spatial_interaction_info),
                'temporal_context': self._format_temporal_context(temporal_interaction_info),
                'reasoning_mode': 'mathematical'
            }
            
            processed_inputs.append(reasoning_context)
        
        return processed_inputs
    
    def _format_spatial_context(self, spatial_info: Dict[str, Dict]) -> str:
        """格式化空间交互上下文（其他智能体的当前解答）"""
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
                
                # 尝试提取数值答案
                extracted_answer = self._extract_numerical_answer(str(latest_output))
                if extracted_answer:
                    context_parts.append(f"  Extracted Answer: {extracted_answer}")
        
        return "\n".join(context_parts)
    
    def _format_temporal_context(self, temporal_info: Dict[str, Dict]) -> str:
        """格式化时间交互上下文（历史解答）"""
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
        """从文本中提取数值答案"""
        # 查找常见的答案模式
        patterns = [
            r'(?:answer|result|solution)(?:\s*is\s*|\s*:\s*)([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*(?:is\s+the\s+answer|is\s+correct)',
            r'=\s*([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*$'  # 行末的数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        
        return ""
    
    def _execute_single_reasoning(self, reasoning_context: Dict[str, Any], **execution_kwargs) -> Any:
        """
        执行单个数学推理任务
        """
        problem = reasoning_context['problem']
        spatial_context = reasoning_context['spatial_context']
        temporal_context = reasoning_context['temporal_context']
        
        # 构建完整的推理提示
        reasoning_prompt = self._build_mathematical_reasoning_prompt(
            problem, spatial_context, temporal_context
        )
        
        try:
            # 调用LLM进行推理
            reasoning_result = self.reasoning_llm.chat(message=reasoning_prompt)
            
            # 后处理推理结果
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
        """构建数学推理提示"""
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
        """后处理数学推理结果"""
        # 提取数值答案
        numerical_answer = self._extract_numerical_answer(raw_result)
        
        # 分析推理步骤
        reasoning_steps = self._extract_reasoning_steps(raw_result)
        
        # 构建结构化结果
        processed_result = {
            'raw_solution': raw_result,
            'numerical_answer': numerical_answer,
            'reasoning_steps': reasoning_steps,
            'agent_role': self.agent_role,
            'confidence_score': self._estimate_confidence(raw_result)
        }
        
        return processed_result
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """提取推理步骤"""
        # 查找编号的步骤
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
        
        # 如果没找到明确的步骤，按句子分割
        if not steps:
            sentences = text.split('.')
            steps = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return steps[:self.max_reasoning_steps]  # 限制步骤数量
    
    def _estimate_confidence(self, text: str) -> float:
        """估计推理结果的置信度"""
        confidence_indicators = {
            'high': ['verified', 'confirmed', 'double-checked', 'certain', 'definitely'],
            'medium': ['likely', 'probably', 'should be', 'appears to be'],
            'low': ['might be', 'could be', 'uncertain', 'not sure', 'guess']
        }
        
        text_lower = text.lower()
        
        # 检查高置信度指标
        high_count = sum(1 for indicator in confidence_indicators['high'] if indicator in text_lower)
        medium_count = sum(1 for indicator in confidence_indicators['medium'] if indicator in text_lower)
        low_count = sum(1 for indicator in confidence_indicators['low'] if indicator in text_lower)
        
        # 基于指标计算置信度
        if high_count > 0:
            base_confidence = 0.8
        elif medium_count > 0:
            base_confidence = 0.6
        elif low_count > 0:
            base_confidence = 0.4
        else:
            base_confidence = 0.7  # 默认中等置信度
        
        # 根据推理长度调整（更详细的推理通常更可靠）
        length_factor = min(len(text) / 500, 1.0) * 0.2
        
        final_confidence = min(base_confidence + length_factor, 1.0)
        return round(final_confidence, 2)
    
    async def _async_execute_single_reasoning(self, reasoning_context: Dict[str, Any], **execution_kwargs) -> Any:
        """
        异步执行单个数学推理任务
        """
        # 对于数学推理，通常不需要真正的异步处理
        # 但可以在这里添加异步LLM调用逻辑
        return self._execute_single_reasoning(reasoning_context, **execution_kwargs)
    
    def get_mathematical_capabilities(self) -> Dict[str, Any]:
        """获取数学推理能力描述"""
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


# 导出主要类
__all__ = ['MathematicalReasoningAgent']

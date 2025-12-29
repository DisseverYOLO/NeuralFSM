"""
提示词优化器
Prompt Optimizer

功能：
1. 集成提示词模板管理器和Few-shot选择器
2. 在训练过程中自动优化提示词
3. 支持A/B测试和性能追踪
4. 动态调整提示词策略
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path

from .prompt_template_manager import (
    PromptTemplateManager,
    create_prompt_template_manager
)
from .few_shot_selector import (
    FewShotSelector,
    create_few_shot_selector
)


@dataclass
class PromptOptimizationConfig:
    """提示词优化配置"""
    enable_ab_test: bool = True
    ab_test_ratio: float = 0.1
    auto_switch_best: bool = True
    switch_interval: int = 100  # 每100个样本检查一次
    few_shot_strategy: str = "hybrid"  # similarity, performance, hybrid
    max_few_shot_examples: int = 3
    performance_metric: str = "accuracy"  # accuracy, success_rate, cost


class PromptOptimizer:
    """
    提示词优化器
    
    集成提示词模板管理和Few-shot选择，在训练过程中自动优化
    """
    
    def __init__(self,
                 domain: str,
                 config: Optional[PromptOptimizationConfig] = None,
                 cache_dir: str = "./prompt_cache"):
        """
        初始化提示词优化器
        
        Args:
            domain: 数据集领域
            config: 优化配置
            cache_dir: 缓存目录
        """
        self.domain = domain
        self.config = config or PromptOptimizationConfig()
        self.cache_dir = cache_dir
        
        # 初始化组件
        self.template_manager = create_prompt_template_manager(domain, cache_dir)
        self.few_shot_selector = create_few_shot_selector(
            domain, 
            self.config.max_few_shot_examples,
            cache_dir
        )
        
        # 训练统计
        self.sample_count = 0
        self.last_switch_check = 0
    
    def get_prompt(self,
                  question: str,
                  question_embedding: Optional[Any] = None,
                  agent_role: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        获取优化后的提示词
        
        Args:
            question: 问题
            question_embedding: 问题嵌入向量
            agent_role: 智能体角色
        
        Returns:
            (完整提示词, 元数据)
        """
        # 选择模板（A/B测试）
        template, is_test = self.template_manager.select_template_for_ab_test(
            self.config.ab_test_ratio if self.config.enable_ab_test else 0.0
        )
        
        if template is None:
            # 如果没有模板，返回默认提示词
            default_prompt = f"You are a helpful assistant working on {self.domain} tasks."
            return default_prompt, {'template_id': None, 'is_test': False}
        
        # 选择Few-shot示例
        few_shot_examples = self.few_shot_selector.select_examples(
            question,
            question_embedding,
            strategy=self.config.few_shot_strategy
        )
        
        # 构建完整提示词
        system_prompt = template.system_prompt
        
        # 添加Few-shot示例
        if few_shot_examples:
            few_shot_text = "\n\nExamples:\n"
            for i, ex in enumerate(few_shot_examples, 1):
                few_shot_text += f"\nExample {i}:\n"
                few_shot_text += f"Question: {ex.question}\n"
                if ex.reasoning:
                    few_shot_text += f"Reasoning: {ex.reasoning}\n"
                few_shot_text += f"Answer: {ex.answer}\n"
            
            system_prompt += few_shot_text
        
        # 元数据
        metadata = {
            'template_id': template.template_id,
            'template_name': template.template_name,
            'is_test': is_test,
            'few_shot_count': len(few_shot_examples),
            'few_shot_example_ids': [ex.example_id for ex in few_shot_examples]
        }
        
        return system_prompt, metadata
    
    def record_result(self,
                     template_id: str,
                     is_correct: bool,
                     response_time: float = 0.0,
                     token_usage: int = 0,
                     cost: float = 0.0,
                     few_shot_example_ids: Optional[List[str]] = None):
        """
        记录结果并更新性能
        
        Args:
            template_id: 模板ID
            is_correct: 是否正确
            response_time: 响应时间
            token_usage: Token使用量
            cost: 成本
            few_shot_example_ids: 使用的Few-shot示例ID列表
        """
        # 更新模板性能
        self.template_manager.update_performance(
            template_id,
            is_correct,
            response_time,
            token_usage,
            cost
        )
        
        # 更新Few-shot示例性能
        if few_shot_example_ids:
            for example_id in few_shot_example_ids:
                self.few_shot_selector.update_example_performance(
                    example_id,
                    is_correct
                )
        
        # 更新样本计数
        self.sample_count += 1
        
        # 检查是否需要切换到最佳模板
        if (self.config.auto_switch_best and 
            self.sample_count - self.last_switch_check >= self.config.switch_interval):
            self._check_and_switch_best_template()
            self.last_switch_check = self.sample_count
    
    def _check_and_switch_best_template(self):
        """检查并切换到最佳模板"""
        current_template = self.template_manager.get_template()
        best_template = self.template_manager.get_best_template(
            self.config.performance_metric
        )
        
        if (best_template and current_template and 
            best_template.template_id != current_template.template_id):
            # 检查性能提升是否显著（至少5%）
            current_perf = self.template_manager.performance.get(
                current_template.template_id
            )
            best_perf = self.template_manager.performance.get(
                best_template.template_id
            )
            
            if (current_perf and best_perf and 
                best_perf.total_count >= 20 and  # 至少20个样本
                best_perf.success_rate > current_perf.success_rate * 1.05):
                self.template_manager.switch_to_best_template(
                    self.config.performance_metric
                )
                print(f"🔄 切换到最佳模板: {best_template.template_name} "
                      f"(准确率: {best_perf.success_rate:.2%})")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        template_summary = self.template_manager.get_performance_summary()
        
        return {
            'domain': self.domain,
            'sample_count': self.sample_count,
            'template_summary': template_summary,
            'few_shot_examples_count': len(self.few_shot_selector.examples)
        }
    
    def register_default_templates(self):
        """注册默认提示词模板"""
        # 基础模板
        self.template_manager.register_template(
            template_id="default",
            template_name="Default Template",
            system_prompt=f"You are an expert assistant specializing in {self.domain} tasks. "
                         f"Provide accurate and detailed responses."
        )
        
        # 详细模板
        self.template_manager.register_template(
            template_id="detailed",
            template_name="Detailed Template",
            system_prompt=f"You are an expert assistant specializing in {self.domain} tasks. "
                         f"Think step by step and provide detailed reasoning before giving your answer. "
                         f"Be thorough and accurate."
        )
        
        # 简洁模板
        self.template_manager.register_template(
            template_id="concise",
            template_name="Concise Template",
            system_prompt=f"You are an expert assistant specializing in {self.domain} tasks. "
                         f"Provide clear and concise answers."
        )


def create_prompt_optimizer(domain: str,
                           config: Optional[PromptOptimizationConfig] = None,
                           cache_dir: str = "./prompt_cache") -> PromptOptimizer:
    """
    创建提示词优化器
    
    Args:
        domain: 数据集领域
        config: 优化配置
        cache_dir: 缓存目录
    
    Returns:
        提示词优化器实例
    """
    return PromptOptimizer(domain, config, cache_dir)



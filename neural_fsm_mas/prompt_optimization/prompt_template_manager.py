"""
提示词模板管理器
Prompt Template Manager

功能：
1. 管理多个版本的提示词模板
2. 支持A/B测试不同提示词版本
3. 记录提示词性能指标
4. 动态选择最佳提示词
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import random


@dataclass
class PromptTemplate:
    """提示词模板"""
    template_id: str
    template_name: str
    system_prompt: str
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def format(self, **kwargs) -> str:
        """格式化提示词模板"""
        prompt = self.system_prompt
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt


@dataclass
class PromptPerformance:
    """提示词性能指标"""
    template_id: str
    accuracy: float = 0.0
    avg_response_time: float = 0.0
    avg_token_usage: int = 0
    success_count: int = 0
    total_count: int = 0
    cost: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0


class PromptTemplateManager:
    """
    提示词模板管理器
    
    管理多个提示词版本，支持性能追踪和动态选择
    """
    
    def __init__(self, domain: str, cache_dir: str = "./prompt_cache"):
        """
        初始化提示词模板管理器
        
        Args:
            domain: 数据集领域
            cache_dir: 缓存目录
        """
        self.domain = domain
        self.cache_dir = Path(cache_dir) / domain
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 模板存储
        self.templates: Dict[str, PromptTemplate] = {}
        self.performance: Dict[str, PromptPerformance] = {}
        
        # 当前使用的模板ID
        self.current_template_id: Optional[str] = None
        
        # 加载缓存的模板和性能数据
        self._load_from_cache()
    
    def register_template(self, 
                         template_id: str,
                         template_name: str,
                         system_prompt: str,
                         few_shot_examples: Optional[List[Dict[str, str]]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> PromptTemplate:
        """
        注册新的提示词模板
        
        Args:
            template_id: 模板ID
            template_name: 模板名称
            system_prompt: 系统提示词
            few_shot_examples: Few-shot示例列表
            metadata: 元数据
        
        Returns:
            创建的模板对象
        """
        template = PromptTemplate(
            template_id=template_id,
            template_name=template_name,
            system_prompt=system_prompt,
            few_shot_examples=few_shot_examples or [],
            metadata=metadata or {}
        )
        
        self.templates[template_id] = template
        
        # 初始化性能追踪
        if template_id not in self.performance:
            self.performance[template_id] = PromptPerformance(template_id=template_id)
        
        # 如果是第一个模板，设为当前模板
        if self.current_template_id is None:
            self.current_template_id = template_id
        
        # 保存到缓存
        self._save_to_cache()
        
        return template
    
    def get_template(self, template_id: Optional[str] = None) -> Optional[PromptTemplate]:
        """
        获取提示词模板
        
        Args:
            template_id: 模板ID，如果为None则返回当前模板
        
        Returns:
            提示词模板
        """
        if template_id is None:
            template_id = self.current_template_id
        
        return self.templates.get(template_id)
    
    def update_performance(self,
                          template_id: str,
                          is_correct: bool,
                          response_time: float = 0.0,
                          token_usage: int = 0,
                          cost: float = 0.0):
        """
        更新提示词性能指标
        
        Args:
            template_id: 模板ID
            is_correct: 是否正确
            response_time: 响应时间（秒）
            token_usage: Token使用量
            cost: 成本
        """
        if template_id not in self.performance:
            self.performance[template_id] = PromptPerformance(template_id=template_id)
        
        perf = self.performance[template_id]
        perf.total_count += 1
        if is_correct:
            perf.success_count += 1
        
        # 更新平均值
        perf.accuracy = perf.success_rate
        perf.avg_response_time = (
            (perf.avg_response_time * (perf.total_count - 1) + response_time) / perf.total_count
        )
        perf.avg_token_usage = (
            (perf.avg_token_usage * (perf.total_count - 1) + token_usage) / perf.total_count
        )
        perf.cost += cost
        perf.last_updated = datetime.now().isoformat()
        
        # 保存到缓存
        self._save_to_cache()
    
    def get_best_template(self, metric: str = "accuracy") -> Optional[PromptTemplate]:
        """
        根据性能指标获取最佳模板
        
        Args:
            metric: 性能指标（accuracy, success_rate, avg_response_time等）
        
        Returns:
            最佳模板
        """
        if not self.performance:
            return self.get_template()
        
        # 过滤有足够样本的模板（至少10个样本）
        valid_perfs = {
            tid: perf for tid, perf in self.performance.items()
            if perf.total_count >= 10
        }
        
        if not valid_perfs:
            return self.get_template()
        
        # 根据指标选择最佳模板
        if metric == "accuracy" or metric == "success_rate":
            best_id = max(valid_perfs.items(), key=lambda x: x[1].success_rate)[0]
        elif metric == "avg_response_time":
            best_id = min(valid_perfs.items(), key=lambda x: x[1].avg_response_time)[0]
        elif metric == "cost":
            best_id = min(valid_perfs.items(), key=lambda x: x[1].cost)[0]
        else:
            best_id = max(valid_perfs.items(), key=lambda x: x[1].success_rate)[0]
        
        return self.templates.get(best_id)
    
    def select_template_for_ab_test(self, 
                                    test_ratio: float = 0.1) -> Tuple[PromptTemplate, bool]:
        """
        为A/B测试选择模板
        
        Args:
            test_ratio: 测试比例（0.1表示10%使用测试模板）
        
        Returns:
            (模板, 是否为测试模板)
        """
        if len(self.templates) < 2:
            return self.get_template(), False
        
        # 随机选择是否使用测试模板
        if random.random() < test_ratio:
            # 选择非当前模板作为测试模板
            test_templates = [tid for tid in self.templates.keys() 
                            if tid != self.current_template_id]
            if test_templates:
                test_id = random.choice(test_templates)
                return self.templates[test_id], True
        
        # 使用当前模板
        return self.get_template(), False
    
    def switch_to_best_template(self, metric: str = "accuracy"):
        """
        切换到最佳模板
        
        Args:
            metric: 性能指标
        """
        best_template = self.get_best_template(metric)
        if best_template:
            self.current_template_id = best_template.template_id
            self._save_to_cache()
            return True
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'domain': self.domain,
            'total_templates': len(self.templates),
            'current_template': self.current_template_id,
            'templates': {}
        }
        
        for template_id, perf in self.performance.items():
            template = self.templates.get(template_id)
            summary['templates'][template_id] = {
                'template_name': template.template_name if template else 'Unknown',
                'accuracy': perf.accuracy,
                'success_rate': perf.success_rate,
                'total_count': perf.total_count,
                'avg_response_time': perf.avg_response_time,
                'avg_token_usage': perf.avg_token_usage,
                'cost': perf.cost,
                'last_updated': perf.last_updated
            }
        
        return summary
    
    def _save_to_cache(self):
        """保存到缓存"""
        cache_file = self.cache_dir / "templates.json"
        performance_file = self.cache_dir / "performance.json"
        
        # 保存模板
        templates_data = {
            tid: {
                'template_id': t.template_id,
                'template_name': t.template_name,
                'system_prompt': t.system_prompt,
                'few_shot_examples': t.few_shot_examples,
                'metadata': t.metadata,
                'created_at': t.created_at
            }
            for tid, t in self.templates.items()
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'current_template_id': self.current_template_id,
                'templates': templates_data
            }, f, indent=2, ensure_ascii=False)
        
        # 保存性能数据
        performance_data = {
            tid: {
                'template_id': perf.template_id,
                'accuracy': perf.accuracy,
                'avg_response_time': perf.avg_response_time,
                'avg_token_usage': perf.avg_token_usage,
                'success_count': perf.success_count,
                'total_count': perf.total_count,
                'cost': perf.cost,
                'last_updated': perf.last_updated
            }
            for tid, perf in self.performance.items()
        }
        
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)
    
    def _load_from_cache(self):
        """从缓存加载"""
        cache_file = self.cache_dir / "templates.json"
        performance_file = self.cache_dir / "performance.json"
        
        # 加载模板
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.current_template_id = data.get('current_template_id')
                    
                    for tid, t_data in data.get('templates', {}).items():
                        self.templates[tid] = PromptTemplate(
                            template_id=t_data['template_id'],
                            template_name=t_data['template_name'],
                            system_prompt=t_data['system_prompt'],
                            few_shot_examples=t_data.get('few_shot_examples', []),
                            metadata=t_data.get('metadata', {}),
                            created_at=t_data.get('created_at', datetime.now().isoformat())
                        )
            except Exception as e:
                print(f"⚠️  加载模板缓存失败: {e}")
        
        # 加载性能数据
        if performance_file.exists():
            try:
                with open(performance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for tid, perf_data in data.items():
                        self.performance[tid] = PromptPerformance(
                            template_id=perf_data['template_id'],
                            accuracy=perf_data.get('accuracy', 0.0),
                            avg_response_time=perf_data.get('avg_response_time', 0.0),
                            avg_token_usage=perf_data.get('avg_token_usage', 0),
                            success_count=perf_data.get('success_count', 0),
                            total_count=perf_data.get('total_count', 0),
                            cost=perf_data.get('cost', 0.0),
                            last_updated=perf_data.get('last_updated', datetime.now().isoformat())
                        )
            except Exception as e:
                print(f"⚠️  加载性能缓存失败: {e}")


def create_prompt_template_manager(domain: str, cache_dir: str = "./prompt_cache") -> PromptTemplateManager:
    """
    创建提示词模板管理器
    
    Args:
        domain: 数据集领域
        cache_dir: 缓存目录
    
    Returns:
        提示词模板管理器实例
    """
    return PromptTemplateManager(domain, cache_dir)



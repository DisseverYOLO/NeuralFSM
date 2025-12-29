"""
Few-shot示例选择器
Few-shot Example Selector

功能：
1. 根据问题类型动态选择最相关的Few-shot示例
2. 基于相似度选择示例
3. 支持示例性能追踪
4. 自动优化示例选择策略
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # 创建一个简单的numpy替代（用于相似度计算）
    class NumpyStub:
        @staticmethod
        def dot(a, b):
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return sum(x * y for x, y in zip(a, b))
            return 0
        
        @staticmethod
        def linalg():
            class LinalgStub:
                @staticmethod
                def norm(x):
                    if isinstance(x, (list, tuple)):
                        return sum(y * y for y in x) ** 0.5
                    return 1.0
            return LinalgStub()
    np = NumpyStub()


@dataclass
class FewShotExample:
    """Few-shot示例"""
    example_id: str
    question: str
    answer: str
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0  # 示例效果评分
    usage_count: int = 0
    success_count: int = 0


class FewShotSelector:
    """
    Few-shot示例选择器
    
    根据问题相似度和示例效果动态选择Few-shot示例
    """
    
    def __init__(self, 
                 domain: str,
                 max_examples: int = 3,
                 cache_dir: str = "./prompt_cache"):
        """
        初始化Few-shot选择器
        
        Args:
            domain: 数据集领域
            max_examples: 最大示例数量
            cache_dir: 缓存目录
        """
        self.domain = domain
        self.max_examples = max_examples
        self.cache_dir = Path(cache_dir) / domain / "few_shot"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 示例存储
        self.examples: Dict[str, FewShotExample] = {}
        self.example_embeddings: Dict[str, Any] = {}  # 示例嵌入向量（支持list或numpy array）
        
        # 加载缓存的示例
        self._load_from_cache()
    
    def add_example(self,
                   example_id: str,
                   question: str,
                   answer: str,
                   reasoning: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   embedding: Optional[Any] = None):
        """
        添加Few-shot示例
        
        Args:
            example_id: 示例ID
            question: 问题
            answer: 答案
            reasoning: 推理过程
            metadata: 元数据
            embedding: 问题嵌入向量（用于相似度计算）
        """
        example = FewShotExample(
            example_id=example_id,
            question=question,
            answer=answer,
            reasoning=reasoning,
            metadata=metadata or {}
        )
        
        self.examples[example_id] = example
        
        if embedding is not None:
            self.example_embeddings[example_id] = embedding
        
        self._save_to_cache()
    
    def select_examples(self,
                       question: str,
                       question_embedding: Optional[Any] = None,
                       strategy: str = "similarity") -> List[FewShotExample]:
        """
        选择Few-shot示例
        
        Args:
            question: 当前问题
            question_embedding: 问题嵌入向量
            strategy: 选择策略（similarity, performance, hybrid）
        
        Returns:
            选中的示例列表
        """
        if not self.examples:
            return []
        
        if strategy == "similarity":
            return self._select_by_similarity(question, question_embedding)
        elif strategy == "performance":
            return self._select_by_performance()
        elif strategy == "hybrid":
            return self._select_hybrid(question, question_embedding)
        else:
            return self._select_by_similarity(question, question_embedding)
    
    def _select_by_similarity(self,
                              question: str,
                              question_embedding: Optional[Any] = None) -> List[FewShotExample]:
        """根据相似度选择示例"""
        if not question_embedding or not self.example_embeddings or not HAS_NUMPY:
            # 如果没有嵌入向量或numpy，随机选择
            return list(self.examples.values())[:self.max_examples]
        
        # 计算相似度
        similarities = []
        for example_id, example_embedding in self.example_embeddings.items():
            similarity = np.dot(question_embedding, example_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(example_embedding) + 1e-8
            )
            similarities.append((similarity, example_id))
        
        # 按相似度排序，选择前max_examples个
        similarities.sort(reverse=True, key=lambda x: x[0])
        selected_ids = [eid for _, eid in similarities[:self.max_examples]]
        
        return [self.examples[eid] for eid in selected_ids if eid in self.examples]
    
    def _select_by_performance(self) -> List[FewShotExample]:
        """根据性能选择示例"""
        examples_list = list(self.examples.values())
        
        # 按性能评分排序
        examples_list.sort(reverse=True, key=lambda x: x.performance_score)
        
        return examples_list[:self.max_examples]
    
    def _select_hybrid(self,
                      question: str,
                      question_embedding: Optional[Any] = None) -> List[FewShotExample]:
        """混合策略：相似度 + 性能"""
        if not question_embedding or not self.example_embeddings or not HAS_NUMPY:
            return self._select_by_performance()
        
        # 计算综合得分：相似度 * 0.6 + 性能 * 0.4
        scores = []
        for example_id, example in self.examples.items():
            if example_id in self.example_embeddings:
                similarity = np.dot(question_embedding, self.example_embeddings[example_id]) / (
                    np.linalg.norm(question_embedding) * 
                    np.linalg.norm(self.example_embeddings[example_id]) + 1e-8
                )
                # 归一化性能评分到[0, 1]
                normalized_perf = example.performance_score / (max(example.performance_score, 1.0) + 1e-8)
                hybrid_score = 0.6 * similarity + 0.4 * normalized_perf
                scores.append((hybrid_score, example_id))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        selected_ids = [eid for _, eid in scores[:self.max_examples]]
        
        return [self.examples[eid] for eid in selected_ids if eid in self.examples]
    
    def update_example_performance(self,
                                  example_id: str,
                                  is_successful: bool):
        """
        更新示例性能
        
        Args:
            example_id: 示例ID
            is_successful: 是否成功
        """
        if example_id not in self.examples:
            return
        
        example = self.examples[example_id]
        example.usage_count += 1
        if is_successful:
            example.success_count += 1
        
        # 更新性能评分（成功率）
        example.performance_score = (
            example.success_count / example.usage_count 
            if example.usage_count > 0 else 0.0
        )
        
        self._save_to_cache()
    
    def _save_to_cache(self):
        """保存到缓存"""
        cache_file = self.cache_dir / "examples.json"
        
        examples_data = {
            eid: {
                'example_id': ex.example_id,
                'question': ex.question,
                'answer': ex.answer,
                'reasoning': ex.reasoning,
                'metadata': ex.metadata,
                'performance_score': ex.performance_score,
                'usage_count': ex.usage_count,
                'success_count': ex.success_count
            }
            for eid, ex in self.examples.items()
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(examples_data, f, indent=2, ensure_ascii=False)
    
    def _load_from_cache(self):
        """从缓存加载"""
        cache_file = self.cache_dir / "examples.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for eid, ex_data in data.items():
                        self.examples[eid] = FewShotExample(
                            example_id=ex_data['example_id'],
                            question=ex_data['question'],
                            answer=ex_data['answer'],
                            reasoning=ex_data.get('reasoning'),
                            metadata=ex_data.get('metadata', {}),
                            performance_score=ex_data.get('performance_score', 0.0),
                            usage_count=ex_data.get('usage_count', 0),
                            success_count=ex_data.get('success_count', 0)
                        )
            except Exception as e:
                print(f"⚠️  加载Few-shot示例缓存失败: {e}")


def create_few_shot_selector(domain: str,
                            max_examples: int = 3,
                            cache_dir: str = "./prompt_cache") -> FewShotSelector:
    """
    创建Few-shot选择器
    
    Args:
        domain: 数据集领域
        max_examples: 最大示例数量
        cache_dir: 缓存目录
    
    Returns:
        Few-shot选择器实例
    """
    return FewShotSelector(domain, max_examples, cache_dir)


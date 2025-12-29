"""
MMLU Data Processor for Multi-Agent System Training
MMLU数据处理器

专门为MetaAgent项目优化的MMLU数据集处理模块
支持按领域划分训练集和验证集，用于训练TGN网络
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import random
from collections import defaultdict


class MMLUDataProcessor:
    """
    MMLU数据处理器
    
    功能：
    1. 加载和处理MMLU数据集
    2. 按学科领域划分数据
    3. 创建训练集和验证集
    4. 为多智能体系统准备数据格式
    """
    
    def __init__(self, data_root_path: str):
        self.data_root_path = Path(data_root_path)
        self.subject_categories = self._define_subject_categories()
        self.processed_data = {}
        self.domain_splits = {}
        
    def _define_subject_categories(self) -> Dict[str, List[str]]:
        """定义学科分类"""
        return {
            "STEM": [
                "abstract_algebra", "anatomy", "astronomy", "college_biology",
                "college_chemistry", "college_computer_science", "college_mathematics",
                "college_physics", "computer_security", "conceptual_physics",
                "electrical_engineering", "elementary_mathematics", "high_school_biology",
                "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
                "high_school_physics", "high_school_statistics", "machine_learning"
            ],
            
            "Humanities": [
                "formal_logic", "high_school_european_history", "high_school_us_history",
                "high_school_world_history", "history", "human_sexuality", "philosophy",
                "prehistory", "professional_psychology", "world_religions"
            ],
            
            "Social_Sciences": [
                "econometrics", "high_school_geography", "high_school_government_and_politics",
                "high_school_macroeconomics", "high_school_microeconomics", "human_aging",
                "international_law", "jurisprudence", "logical_fallacies", "moral_disputes",
                "moral_scenarios", "political_science", "professional_accounting",
                "professional_law", "public_relations", "security_studies", "sociology",
                "us_foreign_policy"
            ],
            
            "Other": [
                "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
                "marketing", "medical_genetics", "miscellaneous", "nutrition", "virology"
            ]
        }
    
    def load_mmlu_data(self, split: str = "test") -> Dict[str, List[Dict]]:
        """
        加载MMLU数据
        
        Args:
            split: 数据集分割 ("test", "dev", "val")
            
        Returns:
            按学科组织的数据字典
        """
        data_by_subject = {}
        
        # 查找数据文件
        data_dir = self.data_root_path / split
        if not data_dir.exists():
            raise FileNotFoundError(f"MMLU data directory not found: {data_dir}")
        
        # 加载每个学科的数据
        for csv_file in data_dir.glob("*.csv"):
            subject_name = csv_file.stem
            
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file, header=None)
                
                # 转换为标准格式
                subject_data = []
                for _, row in df.iterrows():
                    question_data = {
                        "subject": subject_name,
                        "question": row[0],
                        "choices": [row[1], row[2], row[3], row[4]],
                        "answer": row[5] if len(row) > 5 else None,
                        "split": split
                    }
                    subject_data.append(question_data)
                
                data_by_subject[subject_name] = subject_data
                print(f"Loaded {len(subject_data)} questions for {subject_name}")
                
            except Exception as e:
                print(f"Error loading {subject_name}: {e}")
                continue
        
        self.processed_data[split] = data_by_subject
        return data_by_subject
    
    def organize_by_domain(self, split: str = "test") -> Dict[str, List[Dict]]:
        """
        按领域组织数据
        
        Args:
            split: 数据集分割
            
        Returns:
            按领域组织的数据字典
        """
        if split not in self.processed_data:
            self.load_mmlu_data(split)
        
        domain_data = defaultdict(list)
        
        for subject, questions in self.processed_data[split].items():
            # 找到学科所属的领域
            domain = self._get_subject_domain(subject)
            domain_data[domain].extend(questions)
        
        # 转换为普通字典并打印统计信息
        domain_data = dict(domain_data)
        for domain, questions in domain_data.items():
            print(f"Domain {domain}: {len(questions)} questions")
        
        return domain_data
    
    def _get_subject_domain(self, subject: str) -> str:
        """获取学科所属领域"""
        for domain, subjects in self.subject_categories.items():
            if subject in subjects:
                return domain
        return "Other"  # 默认分类
    
    def create_domain_splits(self, 
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           random_seed: int = 42) -> Dict[str, Dict[str, List[Dict]]]:
        """
        为每个领域创建训练/验证/测试分割
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            
        Returns:
            按领域和分割组织的数据字典
        """
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 确保比例和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # 加载所有可用的数据
        all_splits = ["test", "dev", "val"]
        all_domain_data = defaultdict(list)
        
        for split in all_splits:
            try:
                domain_data = self.organize_by_domain(split)
                for domain, questions in domain_data.items():
                    all_domain_data[domain].extend(questions)
            except FileNotFoundError:
                print(f"Split {split} not found, skipping...")
                continue
        
        # 为每个领域创建分割
        domain_splits = {}
        
        for domain, all_questions in all_domain_data.items():
            # 随机打乱数据
            random.shuffle(all_questions)
            
            total_questions = len(all_questions)
            train_size = int(total_questions * train_ratio)
            val_size = int(total_questions * val_ratio)
            
            # 分割数据
            train_data = all_questions[:train_size]
            val_data = all_questions[train_size:train_size + val_size]
            test_data = all_questions[train_size + val_size:]
            
            domain_splits[domain] = {
                "train": train_data,
                "validation": val_data,
                "test": test_data
            }
            
            print(f"Domain {domain}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        self.domain_splits = domain_splits
        return domain_splits
    
    def prepare_multi_agent_training_data(self, 
                                        domain: str, 
                                        split: str = "train",
                                        batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        为多智能体系统准备训练数据
        
        Args:
            domain: 领域名称
            split: 数据分割 ("train", "validation", "test")
            batch_size: 批次大小
            
        Returns:
            多智能体训练数据批次列表
        """
        if not self.domain_splits:
            self.create_domain_splits()
        
        if domain not in self.domain_splits:
            raise ValueError(f"Domain {domain} not found. Available domains: {list(self.domain_splits.keys())}")
        
        if split not in self.domain_splits[domain]:
            raise ValueError(f"Split {split} not found for domain {domain}")
        
        questions = self.domain_splits[domain][split]
        
        # 创建批次
        batches = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            
            batch_data = {
                "domain": domain,
                "split": split,
                "batch_id": i // batch_size,
                "questions": batch_questions,
                "batch_size": len(batch_questions)
            }
            
            batches.append(batch_data)
        
        return batches
    
    def format_question_for_agents(self, question_data: Dict[str, Any]) -> Dict[str, str]:
        """
        将问题格式化为智能体可处理的格式
        
        Args:
            question_data: 原始问题数据
            
        Returns:
            格式化后的问题数据
        """
        choices_text = "\n".join([
            f"A) {question_data['choices'][0]}",
            f"B) {question_data['choices'][1]}",
            f"C) {question_data['choices'][2]}",
            f"D) {question_data['choices'][3]}"
        ])
        
        formatted_question = f"""Subject: {question_data['subject']}

Question: {question_data['question']}

Choices:
{choices_text}

Please select the best answer (A, B, C, or D) and provide your reasoning."""
        
        return {
            "task": formatted_question,
            "subject": question_data["subject"],
            "correct_answer": question_data.get("answer", ""),
            "choices": question_data["choices"],
            "original_question": question_data["question"]
        }
    
    def evaluate_agent_response(self, 
                               agent_response: str, 
                               correct_answer: str) -> Dict[str, Any]:
        """
        评估智能体响应
        
        Args:
            agent_response: 智能体的响应
            correct_answer: 正确答案
            
        Returns:
            评估结果
        """
        # 从响应中提取答案
        predicted_answer = self._extract_answer_from_response(agent_response)
        
        # 计算准确性
        is_correct = predicted_answer.upper() == correct_answer.upper()
        
        return {
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "confidence_score": self._estimate_response_confidence(agent_response)
        }
    
    def _extract_answer_from_response(self, response: str) -> str:
        """从响应中提取答案"""
        import re
        
        # 查找答案模式
        patterns = [
            r'(?:answer|choice|select|option)(?:\s+is\s+|\s*:\s*)([A-D])',
            r'([A-D])\)',
            r'([A-D])\s*(?:is\s+correct|is\s+the\s+answer)',
            r'(?:^|\s)([A-D])(?:\s|$|\.)'
        ]
        
        response_upper = response.upper()
        
        for pattern in patterns:
            matches = re.findall(pattern, response_upper)
            if matches:
                return matches[0]
        
        # 如果没找到明确的答案，返回最常见的字母
        letter_counts = {letter: response_upper.count(letter) for letter in 'ABCD'}
        return max(letter_counts, key=letter_counts.get)
    
    def _estimate_response_confidence(self, response: str) -> float:
        """估计响应的置信度"""
        confidence_keywords = {
            'high': ['certain', 'definitely', 'clearly', 'obviously', 'undoubtedly'],
            'medium': ['likely', 'probably', 'seems', 'appears'],
            'low': ['might', 'could', 'possibly', 'uncertain', 'guess']
        }
        
        response_lower = response.lower()
        
        high_count = sum(1 for word in confidence_keywords['high'] if word in response_lower)
        medium_count = sum(1 for word in confidence_keywords['medium'] if word in response_lower)
        low_count = sum(1 for word in confidence_keywords['low'] if word in response_lower)
        
        if high_count > 0:
            return 0.9
        elif medium_count > 0:
            return 0.7
        elif low_count > 0:
            return 0.4
        else:
            return 0.6  # 默认中等置信度
    
    def save_processed_data(self, output_path: str):
        """保存处理后的数据"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存领域分割数据
        if self.domain_splits:
            for domain, splits in self.domain_splits.items():
                domain_file = output_path / f"{domain}_splits.json"
                with open(domain_file, 'w', encoding='utf-8') as f:
                    json.dump(splits, f, indent=2, ensure_ascii=False)
                print(f"Saved {domain} data to {domain_file}")
        
        # 保存配置信息
        config = {
            "data_root_path": str(self.data_root_path),
            "subject_categories": self.subject_categories,
            "available_domains": list(self.domain_splits.keys()) if self.domain_splits else []
        }
        
        config_file = output_path / "mmlu_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Saved configuration to {config_file}")
    
    def load_processed_data(self, input_path: str):
        """加载处理后的数据"""
        input_path = Path(input_path)
        
        # 加载配置
        config_file = input_path / "mmlu_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.subject_categories = config.get("subject_categories", {})
        
        # 加载领域分割数据
        self.domain_splits = {}
        for json_file in input_path.glob("*_splits.json"):
            domain = json_file.stem.replace("_splits", "")
            with open(json_file, 'r', encoding='utf-8') as f:
                self.domain_splits[domain] = json.load(f)
        
        print(f"Loaded data for domains: {list(self.domain_splits.keys())}")
    
    def get_domain_statistics(self) -> Dict[str, Dict[str, int]]:
        """获取领域统计信息"""
        if not self.domain_splits:
            return {}
        
        statistics = {}
        for domain, splits in self.domain_splits.items():
            statistics[domain] = {
                split: len(questions) 
                for split, questions in splits.items()
            }
        
        return statistics


# 便捷函数
def create_mmlu_processor(data_root_path: str) -> MMLUDataProcessor:
    """创建MMLU数据处理器"""
    return MMLUDataProcessor(data_root_path)


def prepare_mmlu_training_data(data_root_path: str, 
                              output_path: str,
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15,
                              test_ratio: float = 0.15) -> MMLUDataProcessor:
    """准备MMLU训练数据的便捷函数"""
    processor = MMLUDataProcessor(data_root_path)
    processor.create_domain_splits(train_ratio, val_ratio, test_ratio)
    processor.save_processed_data(output_path)
    return processor


# 导出主要类和函数
__all__ = [
    'MMLUDataProcessor',
    'create_mmlu_processor',
    'prepare_mmlu_training_data'
]

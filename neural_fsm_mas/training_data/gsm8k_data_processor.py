"""
GSM8K Data Processor for Multi-Agent System Training
GSM8K数据处理器

专门为NeuralFSM项目优化的GSM8K数据集处理模块
支持数学推理问题的处理，用于训练TGN网络
"""

import json
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import random


class GSM8KDataProcessor:
    """
    GSM8K数据处理器
    
    功能：
    1. 加载和处理GSM8K数据集
    2. 创建训练集和验证集
    3. 为多智能体系统准备数据格式
    """
    
    def __init__(self, data_file_path: str):
        """
        初始化GSM8K数据处理器
        
        Args:
            data_file_path: GSM8K数据文件路径 (JSONL格式)
        """
        self.data_file_path = Path(data_file_path)
        if not self.data_file_path.exists():
            raise FileNotFoundError(f"GSM8K数据文件不存在: {data_file_path}")
        
        self.raw_data = []
        self.processed_data = []
        self.train_data = []
        self.val_data = []
    
    def load_gsm8k_data(self) -> List[Dict]:
        """
        加载GSM8K数据
        
        Returns:
            原始数据列表
        """
        print(f"📂 加载GSM8K数据: {self.data_file_path}")
        
        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_item = json.loads(line.strip())
                self.raw_data.append(data_item)
        
        print(f"✅ 加载完成: {len(self.raw_data)} 条数据")
        return self.raw_data
    
    def process_gsm8k_data(self) -> List[Dict]:
        """
        处理GSM8K数据（提取问题、步骤和答案）
        
        参考GDesigner的gsm_data_process函数
        
        Returns:
            处理后的数据列表
        """
        if not self.raw_data:
            self.load_gsm8k_data()
        
        print(f"🔄 处理GSM8K数据...")
        
        for data in self.raw_data:
            item = {"task": data["question"]}
            raw_answer = data["answer"]
            
            # 分离推理步骤和最终答案
            raw_answer_list = raw_answer.split("\n####")
            item["step"] = raw_answer_list[0].strip()
            item["answer"] = raw_answer_list[-1].replace(",", "").strip()
            
            self.processed_data.append(item)
        
        print(f"✅ 处理完成: {len(self.processed_data)} 条数据")
        return self.processed_data
    
    def split_train_val(self, 
                       train_ratio: float = 0.8,
                       shuffle: bool = True,
                       random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        划分训练集和验证集
        
        Args:
            train_ratio: 训练集比例
            shuffle: 是否打乱数据
            random_seed: 随机种子
            
        Returns:
            (训练集, 验证集)
        """
        if not self.processed_data:
            self.process_gsm8k_data()
        
        data = self.processed_data.copy()
        
        if shuffle:
            random.seed(random_seed)
            random.shuffle(data)
        
        split_point = int(len(data) * train_ratio)
        self.train_data = data[:split_point]
        self.val_data = data[split_point:]
        
        print(f"📊 数据划分完成:")
        print(f"   训练集: {len(self.train_data)} 条")
        print(f"   验证集: {len(self.val_data)} 条")
        
        return self.train_data, self.val_data
    
    def get_train_data(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        获取训练数据
        
        Args:
            num_samples: 采样数量，None表示全部
            
        Returns:
            训练数据
        """
        if not self.train_data:
            self.split_train_val()
        
        if num_samples is None:
            return self.train_data
        else:
            return self.train_data[:num_samples]
    
    def get_val_data(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        获取验证数据
        
        Args:
            num_samples: 采样数量，None表示全部
            
        Returns:
            验证数据
        """
        if not self.val_data:
            self.split_train_val()
        
        if num_samples is None:
            return self.val_data
        else:
            return self.val_data[:num_samples]
    
    @staticmethod
    def extract_answer_from_response(response: str) -> str:
        """
        从模型响应中提取答案（参考GDesigner的gsm_get_predict函数）
        
        Args:
            response: 模型响应文本
            
        Returns:
            提取的答案
        """
        pred_str = response
        
        # 查找"The answer is"
        if 'The answer is ' in pred_str:
            pred = pred_str.split('The answer is ')[-1].strip()
        elif 'the answer is ' in pred_str:
            pred = pred_str.split('the answer is ')[-1].strip()
        # 查找LaTeX格式答案
        elif 'boxed' in pred_str:
            ans = pred_str.split('boxed')[-1]
            if ans[0] == '{':
                stack = 1
                a = ''
                for c in ans[1:]:
                    if c == '{':
                        stack += 1
                        a += c
                    elif c == '}':
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            a = GSM8KDataProcessor._strip_string(a)
            pred = a
        else:
            # 使用正则表达式提取数字
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str)
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ''
        
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
            if pred[-1] == "/":
                pred = pred[:-1]
        
        pred = GSM8KDataProcessor._strip_string(pred)
        
        # 提取数字
        if pred.isdigit():
            return pred
        else:
            matches = re.findall(r'\d+', pred)
            return matches[-1] if matches else '0'
    
    @staticmethod
    def _strip_string(string: str) -> str:
        """清理字符串"""
        string = string.replace("\n", "")
        string = string.replace("\\!", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        string = string.replace("\\$", "")
        string = string.replace("\\%", "")
        string = string.replace("\%", "")
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        
        if len(string) == 0:
            return string
        
        if string[0] == ".":
            string = "0" + string
        
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]
        
        string = string.replace(" ", "")
        
        return string
    
    def format_for_mas_training(self, data: List[Dict]) -> List[Dict]:
        """
        格式化数据用于多智能体系统训练
        
        Args:
            data: 原始数据
            
        Returns:
            格式化后的数据
        """
        formatted_data = []
        
        for item in data:
            formatted_item = {
                "id": len(formatted_data),
                "question": item["task"],
                "ground_truth": item["answer"],
                "reasoning_steps": item.get("step", ""),
                "task_type": "math_reasoning"
            }
            formatted_data.append(formatted_item)
        
        return formatted_data


# 导出函数
__all__ = [
    'GSM8KDataProcessor'
]


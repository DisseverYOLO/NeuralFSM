"""
GSM8K Data Processor for Multi-Agent System Training
GSM8K data processor

GSM8K dataset processing module optimized for the NeuralFSM project.
Supports math reasoning problem processing for TGN training.
"""

import json
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import random


class GSM8KDataProcessor:
    """
    GSM8K data processor
    
    Features:
    1. Load and process the GSM8K dataset.
    2. Create training and validation sets.
    3. Prepare data formats for the multi-agent system.
    """
    
    def __init__(self, data_file_path: str):
        """
        Initialize the GSM8K data processor.
        
        Args:
            data_file_path: Path to the GSM8K data file (JSONL format).
        """
        self.data_file_path = Path(data_file_path)
        if not self.data_file_path.exists():
            raise FileNotFoundError(f"GSM8K data file does not exist: {data_file_path}")
        
        self.raw_data = []
        self.processed_data = []
        self.train_data = []
        self.val_data = []
    
    def load_gsm8k_data(self) -> List[Dict]:
        """
        Load GSM8K data.
        
        Returns:
            List of raw data items.
        """
        print(f"📂 Loading GSM8K data: {self.data_file_path}")
        
        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_item = json.loads(line.strip())
                self.raw_data.append(data_item)
        
        print(f"✅ Loading complete: {len(self.raw_data)} records")
        return self.raw_data
    
    def process_gsm8k_data(self) -> List[Dict]:
        """
        Process GSM8K data by extracting questions, steps, and answers.
        
        Based on GDesigner's `gsm_data_process` function.
        
        Returns:
            List of processed data items.
        """
        if not self.raw_data:
            self.load_gsm8k_data()
        
        print("🔄 Processing GSM8K data...")
        
        for data in self.raw_data:
            item = {"task": data["question"]}
            raw_answer = data["answer"]
            
            # Separate reasoning steps and the final answer.
            raw_answer_list = raw_answer.split("\n####")
            item["step"] = raw_answer_list[0].strip()
            item["answer"] = raw_answer_list[-1].replace(",", "").strip()
            
            self.processed_data.append(item)
        
        print(f"✅ Processing complete: {len(self.processed_data)} records")
        return self.processed_data
    
    def split_train_val(self, 
                       train_ratio: float = 0.8,
                       shuffle: bool = True,
                       random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Split the dataset into training and validation sets.
        
        Args:
            train_ratio: Training set ratio.
            shuffle: Whether to shuffle the data.
            random_seed: Random seed.
            
        Returns:
            (training_set, validation_set)
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
        
        print("📊 Data split complete:")
        print(f"   Training set: {len(self.train_data)} records")
        print(f"   Validation set: {len(self.val_data)} records")
        
        return self.train_data, self.val_data
    
    def get_train_data(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Get training data.
        
        Args:
            num_samples: Number of samples to return; `None` means all.
            
        Returns:
            Training data.
        """
        if not self.train_data:
            self.split_train_val()
        
        if num_samples is None:
            return self.train_data
        else:
            return self.train_data[:num_samples]
    
    def get_val_data(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Get validation data.
        
        Args:
            num_samples: Number of samples to return; `None` means all.
            
        Returns:
            Validation data.
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
        Extract the answer from a model response.
        
        Based on GDesigner's `gsm_get_predict` function.
        
        Args:
            response: Model response text.
            
        Returns:
            Extracted answer.
        """
        pred_str = response
        
        # Look for "The answer is".
        if 'The answer is ' in pred_str:
            pred = pred_str.split('The answer is ')[-1].strip()
        elif 'the answer is ' in pred_str:
            pred = pred_str.split('the answer is ')[-1].strip()
        # Look for a LaTeX-formatted answer.
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
            # Use a regular expression to extract numbers.
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
        
        # Extract digits.
        if pred.isdigit():
            return pred
        else:
            matches = re.findall(r'\d+', pred)
            return matches[-1] if matches else '0'
    
    @staticmethod
    def _strip_string(string: str) -> str:
        """Clean a string."""
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
        Format data for multi-agent system training.
        
        Args:
            data: Raw data.
            
        Returns:
            Formatted data.
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


# Exported symbols
__all__ = [
    'GSM8KDataProcessor'
]

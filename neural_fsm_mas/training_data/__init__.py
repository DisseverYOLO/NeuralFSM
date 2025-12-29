"""
Training Data Module
训练数据模块

包含数据处理和准备功能
"""

from .mmlu_data_processor import (
    MMLUDataProcessor,
    create_mmlu_processor,
    prepare_mmlu_training_data
)

from .unified_data_processor import (
    UnifiedDataProcessor
)

from .answer_validator import (
    AnswerValidator,
    create_answer_validator
)

__all__ = [
    # MMLU处理器（兼容旧代码）
    'MMLUDataProcessor',
    'create_mmlu_processor',
    'prepare_mmlu_training_data',
    
    # ✨ 统一数据处理器（支持所有6个数据集）
    'UnifiedDataProcessor',
    
    # ✨ 答案验证器（支持所有6个数据集）
    'AnswerValidator',
    'create_answer_validator'
]

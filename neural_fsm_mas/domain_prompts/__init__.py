"""
Domain Prompts Module
领域提示模块

包含不同领域的提示管理和角色定义
"""

from .prompt_manager import (
    DomainPromptManager,
    DomainPromptRegistry,
    DomainPromptSet,
    MMLUDomainPromptSet,
    GSM8KDomainPromptSet,
    HumanEvalDomainPromptSet,
    HotpotQADomainPromptSet,  # ✨ NEW
    ALFWorldDomainPromptSet,  # ✨ NEW
    MATHDomainPromptSet       # ✨ NEW
)

__all__ = [
    'DomainPromptManager',
    'DomainPromptRegistry',
    'DomainPromptSet',
    'MMLUDomainPromptSet',
    'GSM8KDomainPromptSet',
    'HumanEvalDomainPromptSet',
    'HotpotQADomainPromptSet',  # ✨ NEW
    'ALFWorldDomainPromptSet',  # ✨ NEW
    'MATHDomainPromptSet'       # ✨ NEW
]

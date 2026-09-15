"""
Domain Prompts Module

Contains prompt management and role definitions for different domains
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

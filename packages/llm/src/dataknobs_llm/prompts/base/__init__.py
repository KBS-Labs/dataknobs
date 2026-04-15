"""Base classes and types for the prompt library system."""

from .types import (
    ValidationLevel,
    ValidationConfig,
    PromptTemplateDict,
    RAGConfig,
    MessageIndex,
    RenderResult,
    TemplateDict,
    MessageIndexDict,
    ParameterDict,
    AdapterDict,
)
from .abstract_prompt_library import AbstractPromptLibrary
from .base_prompt_library import BasePromptLibrary

__all__ = [
    # Validation types
    "ValidationLevel",
    "ValidationConfig",
    # Template types
    "PromptTemplateDict",
    "RAGConfig",
    "MessageIndex",
    "RenderResult",
    # Type aliases
    "TemplateDict",
    "MessageIndexDict",
    "ParameterDict",
    "AdapterDict",
    # Base classes
    "AbstractPromptLibrary",
    "BasePromptLibrary",
]

# Note: TemplateSyntax is exported from dataknobs_llm.prompts.syntax, not here,
# to keep the base package focused on library types.

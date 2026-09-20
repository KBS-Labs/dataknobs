# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Base classes and types for the prompt library system."""

from .types import (
    ValidationLevel,
    ValidationConfig,
    PromptTemplateDict,
    RAGConfig,
    rag_config_from_dict,
    MessageIndex,
    RenderResult,
    TemplateDict,
    MessageIndexDict,
    ParameterDict,
    AdapterDict,
)
from .abstract_prompt_library import AbstractPromptLibrary
from .async_prompt_library import AsyncPromptLibrary
from .base_prompt_library import BasePromptLibrary
from .views import (
    AsyncPromptLibraryView,
    SyncPromptLibraryView,
    as_async,
    as_sync,
)

__all__ = [
    # Validation types
    "ValidationLevel",
    "ValidationConfig",
    # Template types
    "PromptTemplateDict",
    "RAGConfig",
    "rag_config_from_dict",
    "MessageIndex",
    "RenderResult",
    # Type aliases
    "TemplateDict",
    "MessageIndexDict",
    "ParameterDict",
    "AdapterDict",
    # Base classes
    "AbstractPromptLibrary",
    "AsyncPromptLibrary",
    "BasePromptLibrary",
    # Flavour conversion
    "AsyncPromptLibraryView",
    "SyncPromptLibraryView",
    "as_async",
    "as_sync",
]

# Note: TemplateSyntax is exported from dataknobs_llm.prompts.syntax, not here,
# to keep the base package focused on library types.

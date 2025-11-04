"""Core type definitions for the prompt library system.

This module defines:
- Validation levels and configuration
- Prompt template structures
- RAG configuration types
- Message index types
- Render result types
"""

from enum import Enum
from typing import Any, Dict, List, TypedDict
from dataclasses import dataclass, field


class ValidationLevel(Enum):
    """Validation strictness levels for template rendering.

    Attributes:
        ERROR: Raise exception for missing required parameters
        WARN: Log warning for missing required parameters (default)
        IGNORE: Silently ignore missing required parameters
    """
    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


class TemplateMode(Enum):
    """Template rendering mode.

    Attributes:
        MIXED: Pre-process (( )) conditionals then render with Jinja2 (default)
        JINJA2: Pure Jinja2 rendering, skip (( )) preprocessing
    """
    MIXED = "mixed"
    JINJA2 = "jinja2"

    @classmethod
    def from_string(cls, value: str) -> "TemplateMode":
        """Parse mode from string.

        Args:
            value: Mode string ("mixed" or "jinja2")

        Returns:
            TemplateMode enum value

        Raises:
            ValueError: If value is not a valid mode
        """
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid template mode: {value}. "
                f"Valid modes: {', '.join(m.value for m in cls)}"
            ) from e


@dataclass
class ValidationConfig:
    """Configuration for template parameter validation.

    Attributes:
        level: Validation strictness level (None to inherit from context)
        required_params: Set of parameter names that must be provided
        optional_params: Set of parameter names that are optional
    """
    level: ValidationLevel | None = None
    required_params: set[str] = field(default_factory=set)
    optional_params: set[str] = field(default_factory=set)

    def __init__(
        self,
        level: ValidationLevel | None = None,
        required_params: List[str] | None = None,
        optional_params: List[str] | None = None
    ):
        """Initialize validation configuration.

        Args:
            level: Validation strictness level (None to inherit from context)
            required_params: List of required parameter names
            optional_params: List of optional parameter names
        """
        self.level = level
        self.required_params = set(required_params or [])
        self.optional_params = set(optional_params or [])


class PromptTemplateDict(TypedDict, total=False):
    """TypedDict structure for prompt template configuration.

    This defines the schema for prompt template definitions used throughout
    the prompt library system. It supports template inheritance, validation,
    RAG configuration, and flexible template composition.

    Attributes:
        template: The template string with {{variables}} and ((conditionals))
        defaults: Default values for template parameters
        validation: Validation configuration for this template
        metadata: Additional metadata (author, version, etc.)
        sections: Section definitions for template composition
        extends: Name of base template to inherit from
        rag_config_refs: References to standalone RAG configurations
        rag_configs: Inline RAG configurations
        template_mode: Template rendering mode ("mixed" or "jinja2")
    """
    template: str
    defaults: Dict[str, Any]
    validation: ValidationConfig
    metadata: Dict[str, Any]
    sections: Dict[str, str]
    extends: str
    rag_config_refs: List[str]
    rag_configs: List['RAGConfig']
    template_mode: str


class RAGConfig(TypedDict, total=False):
    """Configuration for RAG (Retrieval-Augmented Generation) searches.

    Attributes:
        adapter_name: Name of the adapter to use for RAG search
        query: RAG search query (may contain {{variables}})
        k: Number of results to retrieve
        filters: Additional filters for the search
        placeholder: Placeholder name in template (e.g., "RAG_CONTENT")
        header: Header text to prepend to RAG results
        item_template: Template for formatting individual RAG items
    """
    adapter_name: str
    query: str
    k: int
    filters: Dict[str, Any]
    placeholder: str
    header: str
    item_template: str


class MessageIndex(TypedDict, total=False):
    """Structure for a message index definition.

    Message indexes define sequences of messages with roles and content,
    including support for RAG content injection.

    Attributes:
        messages: List of message dictionaries with 'role' and 'content'
        rag_configs: RAG configurations for this message sequence
        metadata: Additional metadata for this message index
    """
    messages: List[Dict[str, str]]
    rag_configs: List[RAGConfig]
    metadata: Dict[str, Any]


@dataclass
class RenderResult:
    """Result of rendering a prompt template.

    Attributes:
        content: The final rendered content
        params_used: Parameters that were actually used in rendering
        params_missing: Required parameters that were missing
        validation_warnings: List of validation warnings (if level=WARN)
        metadata: Additional metadata about the rendering
        rag_metadata: Optional RAG metadata (if return_rag_metadata=True)
                    Contains details about RAG searches executed during rendering
    """
    content: str
    params_used: Dict[str, Any] = field(default_factory=dict)
    params_missing: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    rag_metadata: Dict[str, Any] | None = None


# Type aliases for convenience
TemplateDict = Dict[str, PromptTemplateDict]
MessageIndexDict = Dict[str, MessageIndex]
ParameterDict = Dict[str, Any]
AdapterDict = Dict[str, Any]  # Will be refined in adapter modules

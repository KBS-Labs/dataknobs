"""Base prompt builder with shared functionality for sync and async builders.

This module provides BasePromptBuilder, an abstract base class that contains
all the shared logic between PromptBuilder and AsyncPromptBuilder. This reduces
code duplication and ensures consistent behavior across both implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..base import (
    AbstractPromptLibrary,
    PromptTemplateDict,
    RAGConfig,
    ValidationLevel,
    ValidationConfig,
)
from ..rendering import TemplateRenderer

logger = logging.getLogger(__name__)


class BasePromptBuilder(ABC):
    """Abstract base class with shared functionality for prompt builders.

    This class provides common methods for:
    - Template parameter merging
    - RAG query rendering
    - RAG result formatting
    - Required parameter extraction
    - String representation

    Subclasses must implement the async/sync-specific methods for:
    - Rendering prompts (with I/O operations)
    - Executing RAG searches (with I/O operations)
    """

    def __init__(
        self,
        library: AbstractPromptLibrary,
        adapters: Dict[str, Any] | None = None,
        default_validation: ValidationLevel = ValidationLevel.WARN,
        raise_on_rag_error: bool = False
    ):
        """Initialize the base prompt builder.

        Args:
            library: Prompt library to retrieve templates from
            adapters: Dictionary of named resource adapters
            default_validation: Default validation level for templates
            raise_on_rag_error: If True, raise exceptions on RAG failures
        """
        self.library = library
        self.adapters = adapters or {}
        self._renderer = TemplateRenderer(default_validation=default_validation)
        self._raise_on_rag_error = raise_on_rag_error

    # ===== Shared Helper Methods =====

    def _extract_formatted_content_from_cache(
        self,
        cached_rag: Dict[str, Any]
    ) -> Dict[str, str]:
        r"""Extract formatted content from cached RAG metadata.

        This method extracts the pre-formatted RAG content from cache
        so it can be injected directly into templates without re-executing
        searches or re-formatting results.

        Args:
            cached_rag: Cached RAG metadata dict with structure:
                {
                    "placeholder_name": {
                        "formatted_content": "...",
                        "query": "...",
                        "results": [...],
                        ...
                    }
                }

        Returns:
            Dict mapping placeholder names to formatted content strings

        Example:
            >>> cache = {
            ...     "RAG_CONTENT": {
            ...         "formatted_content": "# Docs\n\n1. Python is...",
            ...         "query": "python docs",
            ...         ...
            ...     }
            ... }
            >>> content = builder._extract_formatted_content_from_cache(cache)
            >>> content
            {'RAG_CONTENT': '# Docs\n\n1. Python is...'}
        """
        rag_content = {}
        for placeholder, cache_entry in cached_rag.items():
            rag_content[placeholder] = cache_entry.get("formatted_content", "")
        return rag_content

    def _compute_rag_query_hash(
        self,
        adapter_name: str,
        query: str
    ) -> str:
        """Compute a hash for RAG query matching.

        This hash is used to match cached RAG results with new queries.
        Two queries with the same hash are considered equivalent and
        can reuse cached results.

        Args:
            adapter_name: Name of the adapter
            query: Rendered query string

        Returns:
            SHA256 hex digest of adapter_name:query

        Example:
            >>> hash1 = builder._compute_rag_query_hash("docs", "python decorators")
            >>> hash2 = builder._compute_rag_query_hash("docs", "python decorators")
            >>> hash1 == hash2
            True
        """
        import hashlib
        combined = f"{adapter_name}:{query}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _render_rag_query(self, query_template: str, params: Dict[str, Any]) -> str:
        """Render a RAG query template with parameters.

        Args:
            query_template: Query template string with {{variables}}
            params: Parameters for substitution

        Returns:
            Rendered query string
        """
        from dataknobs_llm.template_utils import render_conditional_template
        return render_conditional_template(query_template, params)

    def _format_rag_results(
        self,
        results: List[Dict[str, Any]],
        rag_config: RAGConfig,
        params: Dict[str, Any]
    ) -> str:
        """Format RAG search results according to configuration.

        Args:
            results: List of search results from adapter
            rag_config: RAG configuration with formatting options
            params: Parameters for template rendering

        Returns:
            Formatted RAG content string
        """
        if not results:
            return ""

        # Get formatting configuration
        header = rag_config.get("header", "")
        item_template = rag_config.get("item_template", "{{content}}")

        # Render header
        formatted_header = self._render_rag_query(header, params)

        # Format each result
        formatted_items = []
        for i, result in enumerate(results, start=1):
            # Prepare item parameters
            item_params = {
                **params,
                "index": i,
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "metadata": result.get("metadata", {}),
                **result.get("metadata", {})  # Also expose metadata fields directly
            }

            # Render item
            formatted_item = self._render_rag_query(item_template, item_params)
            formatted_items.append(formatted_item)

        # Combine header and items
        return formatted_header + "".join(formatted_items)

    def _merge_params_with_defaults(
        self,
        template_dict: PromptTemplateDict,
        runtime_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge template defaults with runtime parameters.

        Args:
            template_dict: Template dictionary with defaults
            runtime_params: Runtime parameters (higher priority)

        Returns:
            Merged parameters dictionary
        """
        defaults = template_dict.get("defaults", {})
        return {**defaults, **runtime_params}

    def _prepare_validation_config(
        self,
        template_dict: PromptTemplateDict,
        validation_override: ValidationLevel | None
    ) -> ValidationConfig | None:
        """Prepare validation configuration with override support.

        Args:
            template_dict: Template dictionary
            validation_override: Optional validation level override

        Returns:
            Validation configuration or None
        """
        validation_config = template_dict.get("validation")

        # Apply validation override if provided
        if validation_override is not None:
            if validation_config is None:
                validation_config = ValidationConfig()
            validation_config.level = validation_override

        return validation_config

    def get_required_parameters(
        self,
        name: str,
        prompt_type: str = "system",
        index: int = 0,
        **kwargs: Any
    ) -> List[str]:
        """Get list of required parameters for a prompt.

        Useful for validation before rendering.

        Args:
            name: Prompt identifier
            prompt_type: Type of prompt ("system" or "user")
            index: Prompt variant index (for user prompts)
            **kwargs: Additional parameters passed to library

        Returns:
            List of required parameter names

        Raises:
            ValueError: If prompt not found
        """
        # Retrieve template
        if prompt_type == "system":
            template_dict = self.library.get_system_prompt(name, **kwargs)
        else:
            template_dict = self.library.get_user_prompt(name, index=index, **kwargs)

        if template_dict is None:
            raise ValueError(f"Prompt not found: {name} (type={prompt_type})")

        # Extract required parameters from validation config
        validation_config = template_dict.get("validation")
        if validation_config:
            return list(validation_config.required_params)

        return []

    def __repr__(self) -> str:
        """Return a string representation of this builder."""
        return (
            f"{self.__class__.__name__}("
            f"library={self.library}, "
            f"adapters={list(self.adapters.keys())}"
            f")"
        )

    # ===== Abstract Methods (Must be implemented by subclasses) =====

    @abstractmethod
    def _validate_adapters(self) -> None:
        """Validate that all adapters are the correct type (sync or async).

        Raises:
            TypeError: If adapter types don't match builder type
        """
        pass

    @abstractmethod
    def _render_prompt_impl(
        self,
        prompt_name: str,
        prompt_type: str,
        template_dict: PromptTemplateDict,
        runtime_params: Dict[str, Any],
        include_rag: bool,
        validation_override: ValidationLevel | None,
        return_rag_metadata: bool = False,
        cached_rag: Dict[str, Any] | None = None,
        index: int = 0,
        **kwargs: Any
    ):
        """Internal method to render a prompt template.

        This is the core rendering logic that differs between sync/async.

        Args:
            prompt_name: Name of the prompt
            prompt_type: Type of prompt ("system" or "user")
            template_dict: Template dictionary from library
            runtime_params: Runtime parameters
            include_rag: Whether to include RAG content
            validation_override: Validation level override
            return_rag_metadata: If True, capture and return RAG metadata
            cached_rag: If provided, use these cached RAG results instead
                       of executing new searches
            index: Prompt index (for user prompts)
            **kwargs: Additional parameters

        Returns:
            RenderResult with rendered content and metadata (including
            rag_metadata if return_rag_metadata=True)
        """
        pass

    @abstractmethod
    def _execute_rag_searches_impl(
        self,
        prompt_name: str,
        prompt_type: str,
        index: int,
        params: Dict[str, Any],
        capture_metadata: bool = False,
        **kwargs: Any
    ):
        """Execute RAG searches and format results for injection.

        This method differs between sync (sequential) and async (parallel).

        Args:
            prompt_name: Name of the prompt
            prompt_type: Type of prompt ("system" or "user")
            index: Prompt index (for user prompts)
            params: Resolved parameters for query templating
            capture_metadata: If True, capture RAG metadata
            **kwargs: Additional parameters

        Returns:
            Tuple of (rag_content, rag_metadata):
            - rag_content: Dictionary mapping placeholder names to formatted content
            - rag_metadata: Optional dict with full RAG details (if capture_metadata=True)
        """
        pass

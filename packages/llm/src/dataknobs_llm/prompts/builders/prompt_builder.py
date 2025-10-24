"""Synchronous prompt builder for constructing prompts with parameter resolution and RAG.

This module provides the PromptBuilder class which coordinates between:
- Prompt libraries (template sources)
- Resource adapters (data sources)
- Template renderer (rendering engine)

The builder handles:
- Parameter resolution from multiple sources
- RAG content retrieval and injection
- Validation enforcement
- Template defaults merging
"""

import logging
from typing import Any, Dict, List, Optional

from ..base import (
    PromptTemplate,
    RAGConfig,
    ValidationLevel,
    RenderResult,
)
from ..adapters import ResourceAdapter
from .base_prompt_builder import BasePromptBuilder

logger = logging.getLogger(__name__)


class PromptBuilder(BasePromptBuilder):
    """Synchronous prompt builder for constructing prompts with RAG and validation.

    This class provides a high-level API for building prompts by:
    1. Retrieving prompt templates from a library
    2. Resolving parameters from adapters and runtime values
    3. Executing RAG searches via adapters
    4. Injecting RAG content into templates
    5. Rendering final prompts with validation

    Example:
        >>> library = ConfigPromptLibrary(config)
        >>> adapters = {
        ...     'config': DictResourceAdapter(config_data),
        ...     'docs': DataknobsBackendAdapter(docs_db)
        ... }
        >>> builder = PromptBuilder(library=library, adapters=adapters)
        >>>
        >>> # Render a system prompt
        >>> result = builder.render_system_prompt(
        ...     'analyze_code',
        ...     params={'code': code_snippet, 'language': 'python'}
        ... )
    """

    def __init__(
        self,
        library,
        adapters: Optional[Dict[str, ResourceAdapter]] = None,
        default_validation: ValidationLevel = ValidationLevel.WARN,
        raise_on_rag_error: bool = False
    ):
        """Initialize the synchronous prompt builder.

        Args:
            library: Prompt library to retrieve templates from
            adapters: Dictionary of named resource adapters for parameter
                     resolution and RAG searches
            default_validation: Default validation level for templates without
                              explicit validation configuration
            raise_on_rag_error: If True, raise exceptions on RAG failures;
                              if False (default), log warning and continue

        Raises:
            TypeError: If any adapter is async (use AsyncPromptBuilder instead)
        """
        super().__init__(library, adapters, default_validation, raise_on_rag_error)
        self._validate_adapters()

    def _validate_adapters(self) -> None:
        """Validate that all adapters are synchronous.

        Raises:
            TypeError: If any adapter is async
        """
        for name, adapter in self.adapters.items():
            if adapter.is_async():
                raise TypeError(
                    f"Adapter '{name}' is async. "
                    "Use AsyncPromptBuilder for async adapters."
                )

    def render_system_prompt(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        include_rag: bool = True,
        validation_override: Optional[ValidationLevel] = None,
        **kwargs: Any
    ) -> RenderResult:
        """Render a system prompt with parameters and optional RAG content.

        Args:
            name: System prompt identifier
            params: Runtime parameters to use in rendering
            include_rag: Whether to include RAG content (default: True)
            validation_override: Override validation level for this render
            **kwargs: Additional parameters passed to library

        Returns:
            RenderResult with rendered content and metadata

        Raises:
            ValueError: If prompt not found or validation fails
        """
        params = params or {}

        # Retrieve template from library
        template_dict = self.library.get_system_prompt(name, **kwargs)
        if template_dict is None:
            raise ValueError(f"System prompt not found: {name}")

        # Render the prompt
        return self._render_prompt_impl(
            prompt_name=name,
            prompt_type="system",
            template_dict=template_dict,
            runtime_params=params,
            include_rag=include_rag,
            validation_override=validation_override,
            **kwargs
        )

    def render_user_prompt(
        self,
        name: str,
        index: int = 0,
        params: Optional[Dict[str, Any]] = None,
        include_rag: bool = True,
        validation_override: Optional[ValidationLevel] = None,
        **kwargs: Any
    ) -> RenderResult:
        """Render a user prompt with parameters and optional RAG content.

        Args:
            name: User prompt identifier
            index: Prompt variant index (default: 0)
            params: Runtime parameters to use in rendering
            include_rag: Whether to include RAG content (default: True)
            validation_override: Override validation level for this render
            **kwargs: Additional parameters passed to library

        Returns:
            RenderResult with rendered content and metadata

        Raises:
            ValueError: If prompt not found or validation fails
        """
        params = params or {}

        # Retrieve template from library
        template_dict = self.library.get_user_prompt(name, index=index, **kwargs)
        if template_dict is None:
            raise ValueError(f"User prompt not found: {name} (index={index})")

        # Render the prompt
        return self._render_prompt_impl(
            prompt_name=name,
            prompt_type="user",
            template_dict=template_dict,
            runtime_params=params,
            include_rag=include_rag,
            validation_override=validation_override,
            index=index,
            **kwargs
        )

    def _render_prompt_impl(
        self,
        prompt_name: str,
        prompt_type: str,
        template_dict: PromptTemplate,
        runtime_params: Dict[str, Any],
        include_rag: bool,
        validation_override: Optional[ValidationLevel],
        index: int = 0,
        **kwargs: Any
    ) -> RenderResult:
        """Internal method to render a prompt template synchronously.

        Args:
            prompt_name: Name of the prompt
            prompt_type: Type of prompt ("system" or "user")
            template_dict: Template dictionary from library
            runtime_params: Runtime parameters
            include_rag: Whether to include RAG content
            validation_override: Validation level override
            index: Prompt index (for user prompts)
            **kwargs: Additional parameters

        Returns:
            RenderResult with rendered content and metadata
        """
        # Extract template components
        template = template_dict.get("template", "")
        template_metadata = template_dict.get("metadata", {})

        # Step 1: Merge defaults with runtime params
        all_params = self._merge_params_with_defaults(template_dict, runtime_params)

        # Step 2: Execute RAG searches and inject content
        if include_rag:
            rag_content = self._execute_rag_searches_impl(
                prompt_name=prompt_name,
                prompt_type=prompt_type,
                index=index,
                params=all_params,
                **kwargs
            )
            # Merge RAG content into parameters
            all_params.update(rag_content)

        # Step 3: Prepare validation config with override
        validation_config = self._prepare_validation_config(template_dict, validation_override)

        # Step 4: Render template with validation
        result = self._renderer.render(
            template=template,
            params=all_params,
            validation=validation_config,
            template_metadata=template_metadata
        )

        # Add builder metadata
        result.metadata.update({
            "prompt_name": prompt_name,
            "prompt_type": prompt_type,
            "include_rag": include_rag,
        })
        if prompt_type == "user":
            result.metadata["index"] = index

        return result

    def _execute_rag_searches_impl(
        self,
        prompt_name: str,
        prompt_type: str,
        index: int,
        params: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, str]:
        """Execute RAG searches and format results for injection.

        Args:
            prompt_name: Name of the prompt
            prompt_type: Type of prompt ("system" or "user")
            index: Prompt index (for user prompts)
            params: Resolved parameters for query templating
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping placeholder names to formatted RAG content
        """
        # Get RAG configurations for this prompt
        rag_configs = self.library.get_prompt_rag_configs(
            prompt_name=prompt_name,
            prompt_type=prompt_type,
            index=index,
            **kwargs
        )

        if not rag_configs:
            return {}

        rag_content = {}

        for rag_config in rag_configs:
            try:
                # Format this RAG config's content
                placeholder = rag_config.get("placeholder", "RAG_CONTENT")
                content = self._execute_single_rag_search(rag_config, params)
                rag_content[placeholder] = content

            except Exception as e:
                error_msg = f"RAG search failed for {prompt_name}: {e}"
                if self._raise_on_rag_error:
                    raise RuntimeError(error_msg) from e
                else:
                    logger.warning(error_msg)
                    # Use empty content on failure
                    placeholder = rag_config.get("placeholder", "RAG_CONTENT")
                    rag_content[placeholder] = ""

        return rag_content

    def _execute_single_rag_search(
        self,
        rag_config: RAGConfig,
        params: Dict[str, Any]
    ) -> str:
        """Execute a single RAG search and format results.

        Args:
            rag_config: RAG configuration
            params: Parameters for query templating

        Returns:
            Formatted RAG content string

        Raises:
            KeyError: If adapter not found
            Exception: If search fails
        """
        # Get adapter
        adapter_name = rag_config.get("adapter_name")
        if not adapter_name:
            raise ValueError("RAG config missing 'adapter_name'")

        if adapter_name not in self.adapters:
            raise KeyError(
                f"Adapter '{adapter_name}' not found. "
                f"Available adapters: {list(self.adapters.keys())}"
            )

        adapter = self.adapters[adapter_name]

        # Render query template
        query_template = rag_config.get("query", "")
        query = self._render_rag_query(query_template, params)

        # Execute search (synchronous)
        k = rag_config.get("k", 5)
        filters = rag_config.get("filters")
        search_results = adapter.search(query=query, k=k, filters=filters)

        # Format results
        formatted_content = self._format_rag_results(
            results=search_results,
            rag_config=rag_config,
            params=params
        )

        return formatted_content

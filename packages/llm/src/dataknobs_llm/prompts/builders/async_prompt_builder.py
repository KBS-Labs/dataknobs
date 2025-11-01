"""Asynchronous prompt builder for constructing prompts with parameter resolution and RAG.

This module provides the AsyncPromptBuilder class which coordinates between:
- Prompt libraries (template sources)
- Async resource adapters (data sources)
- Template renderer (rendering engine)

The async builder handles:
- Concurrent parameter resolution from multiple sources
- Parallel RAG content retrieval and injection
- Validation enforcement
- Template defaults merging

All async operations use asyncio.gather() for maximum parallelism.
"""

import asyncio
import logging
from typing import Any, Dict

from ..base import (
    PromptTemplate,
    RAGConfig,
    ValidationLevel,
    RenderResult,
)
from ..adapters import AsyncResourceAdapter
from .base_prompt_builder import BasePromptBuilder

logger = logging.getLogger(__name__)


class AsyncPromptBuilder(BasePromptBuilder):
    """Asynchronous prompt builder for constructing prompts with RAG and validation.

    This class provides a high-level async API for building prompts by:
    1. Retrieving prompt templates from a library
    2. Resolving parameters from adapters and runtime values (concurrently)
    3. Executing RAG searches via adapters (in parallel)
    4. Injecting RAG content into templates
    5. Rendering final prompts with validation

    Example:
        >>> library = ConfigPromptLibrary(config)
        >>> adapters = {
        ...     'config': AsyncDictResourceAdapter(config_data),
        ...     'docs': AsyncDataknobsBackendAdapter(docs_db)
        ... }
        >>> builder = AsyncPromptBuilder(library=library, adapters=adapters)
        >>>
        >>> # Render a system prompt
        >>> result = await builder.render_system_prompt(
        ...     'analyze_code',
        ...     params={'code': code_snippet, 'language': 'python'}
        ... )
    """

    def __init__(
        self,
        library,
        adapters: Dict[str, AsyncResourceAdapter] | None = None,
        default_validation: ValidationLevel = ValidationLevel.WARN,
        raise_on_rag_error: bool = False
    ):
        """Initialize the asynchronous prompt builder.

        Args:
            library: Prompt library to retrieve templates from
            adapters: Dictionary of named async resource adapters for parameter
                     resolution and RAG searches
            default_validation: Default validation level for templates without
                              explicit validation configuration
            raise_on_rag_error: If True, raise exceptions on RAG failures;
                              if False (default), log warning and continue

        Raises:
            TypeError: If any adapter is sync (use PromptBuilder instead)
        """
        super().__init__(library, adapters, default_validation, raise_on_rag_error)
        self._validate_adapters()

    def _validate_adapters(self) -> None:
        """Validate that all adapters are asynchronous.

        Raises:
            TypeError: If any adapter is sync
        """
        for name, adapter in self.adapters.items():
            if not adapter.is_async():
                raise TypeError(
                    f"Adapter '{name}' is synchronous. "
                    "Use PromptBuilder for sync adapters."
                )

    async def render_system_prompt(
        self,
        name: str,
        params: Dict[str, Any] | None = None,
        include_rag: bool = True,
        validation_override: ValidationLevel | None = None,
        return_rag_metadata: bool = False,
        cached_rag: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> RenderResult:
        """Render a system prompt with parameters and optional RAG content.

        Args:
            name: System prompt identifier
            params: Runtime parameters to use in rendering
            include_rag: Whether to include RAG content (default: True)
            validation_override: Override validation level for this render
            return_rag_metadata: If True, attach RAG metadata to result
            cached_rag: If provided, use these cached RAG results instead
                       of executing new searches
            **kwargs: Additional parameters passed to library

        Returns:
            RenderResult with rendered content and metadata

        Raises:
            ValueError: If prompt not found or validation fails

        Example:
            >>> # Capture RAG metadata
            >>> result = await builder.render_system_prompt(
            ...     'code_question',
            ...     params={'language': 'python'},
            ...     return_rag_metadata=True
            ... )
            >>> print(result.rag_metadata)
            >>>
            >>> # Reuse cached RAG
            >>> result2 = await builder.render_system_prompt(
            ...     'code_question',
            ...     params={'language': 'python'},
            ...     cached_rag=result.rag_metadata
            ... )
        """
        params = params or {}

        # Retrieve template from library
        template_dict = self.library.get_system_prompt(name, **kwargs)
        if template_dict is None:
            raise ValueError(f"System prompt not found: {name}")

        # Render the prompt
        return await self._render_prompt_impl(
            prompt_name=name,
            prompt_type="system",
            template_dict=template_dict,
            runtime_params=params,
            include_rag=include_rag,
            validation_override=validation_override,
            return_rag_metadata=return_rag_metadata,
            cached_rag=cached_rag,
            **kwargs
        )

    async def render_user_prompt(
        self,
        name: str,
        params: Dict[str, Any] | None = None,
        include_rag: bool = True,
        validation_override: ValidationLevel | None = None,
        return_rag_metadata: bool = False,
        cached_rag: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> RenderResult:
        """Render a user prompt with parameters and optional RAG content.

        Args:
            name: User prompt identifier
            params: Runtime parameters to use in rendering
            include_rag: Whether to include RAG content (default: True)
            validation_override: Override validation level for this render
            return_rag_metadata: If True, attach RAG metadata to result
            cached_rag: If provided, use these cached RAG results instead
                       of executing new searches
            **kwargs: Additional parameters passed to library

        Returns:
            RenderResult with rendered content and metadata

        Raises:
            ValueError: If prompt not found or validation fails
        """
        params = params or {}

        # Retrieve template from library
        template_dict = self.library.get_user_prompt(name, **kwargs)
        if template_dict is None:
            raise ValueError(f"User prompt not found: {name}")

        # Render the prompt
        return await self._render_prompt_impl(
            prompt_name=name,
            prompt_type="user",
            template_dict=template_dict,
            runtime_params=params,
            include_rag=include_rag,
            validation_override=validation_override,
            return_rag_metadata=return_rag_metadata,
            cached_rag=cached_rag,
            **kwargs
        )

    async def _render_prompt_impl(
        self,
        prompt_name: str,
        prompt_type: str,
        template_dict: PromptTemplate,
        runtime_params: Dict[str, Any],
        include_rag: bool,
        validation_override: ValidationLevel | None,
        return_rag_metadata: bool = False,
        cached_rag: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> RenderResult:
        """Internal method to render a prompt template asynchronously.

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
            **kwargs: Additional parameters

        Returns:
            RenderResult with rendered content and metadata
        """
        # Extract template components
        template = template_dict.get("template", "")
        template_metadata = template_dict.get("metadata", {})

        # Step 1: Merge defaults with runtime params
        all_params = self._merge_params_with_defaults(template_dict, runtime_params)

        # Step 2: Execute or reuse RAG searches
        rag_metadata = None
        if include_rag:
            if cached_rag:
                # Use cached RAG results
                rag_content = self._extract_formatted_content_from_cache(cached_rag)
                if return_rag_metadata:
                    rag_metadata = cached_rag  # Pass through cached metadata
            else:
                # Execute fresh RAG searches
                rag_content, rag_metadata = await self._execute_rag_searches_impl(
                    prompt_name=prompt_name,
                    prompt_type=prompt_type,
                    params=all_params,
                    capture_metadata=return_rag_metadata,
                    **kwargs
                )

            # Merge RAG content into parameters
            all_params.update(rag_content)

        # Step 3: Prepare validation config with override
        validation_config = self._prepare_validation_config(template_dict, validation_override)

        # Step 4: Render template with validation (synchronous)
        result = self._renderer.render(
            template=template,
            params=all_params,
            validation=validation_config,
            template_metadata=template_metadata
        )

        # Attach RAG metadata if requested
        if return_rag_metadata and rag_metadata:
            result.rag_metadata = rag_metadata

        # Add builder metadata
        result.metadata.update({
            "prompt_name": prompt_name,
            "prompt_type": prompt_type,
            "include_rag": include_rag,
            "used_cached_rag": cached_rag is not None,
        })

        return result

    async def _execute_rag_searches_impl(
        self,
        prompt_name: str,
        prompt_type: str,
        params: Dict[str, Any],
        capture_metadata: bool = False,
        **kwargs: Any
    ) -> tuple[Dict[str, str], Dict[str, Any] | None]:
        """Execute RAG searches in parallel and format results for injection.

        Args:
            prompt_name: Name of the prompt
            prompt_type: Type of prompt ("system" or "user")
            params: Resolved parameters for query templating
            capture_metadata: If True, capture RAG metadata
            **kwargs: Additional parameters

        Returns:
            Tuple of (rag_content, rag_metadata):
            - rag_content: Dictionary mapping placeholder names to formatted content
            - rag_metadata: Optional dict with full RAG details (if capture_metadata=True)
        """
        # Get RAG configurations for this prompt
        rag_configs = self.library.get_prompt_rag_configs(
            prompt_name=prompt_name,
            prompt_type=prompt_type,
            **kwargs
        )

        if not rag_configs:
            return {}, None

        # Execute all RAG searches in parallel
        if capture_metadata:
            tasks = [
                self._execute_single_rag_with_metadata(rag_config, params)
                for rag_config in rag_configs
            ]
        else:
            tasks = [
                self._execute_single_rag_search_safe(rag_config, params)
                for rag_config in rag_configs
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        rag_content = {}
        rag_metadata = {} if capture_metadata else None

        for rag_config, result in zip(rag_configs, results):
            placeholder = rag_config.get("placeholder", "RAG_CONTENT")

            if isinstance(result, Exception):
                error_msg = f"RAG search failed for {prompt_name}: {result}"
                if self._raise_on_rag_error:
                    raise RuntimeError(error_msg) from result
                else:
                    logger.warning(error_msg)
                    rag_content[placeholder] = ""
                    if capture_metadata:
                        from datetime import datetime
                        rag_metadata[placeholder] = {
                            "error": str(result),
                            "timestamp": datetime.now().isoformat()
                        }
            else:
                if capture_metadata:
                    formatted_content, metadata = result
                    rag_content[placeholder] = formatted_content
                    if metadata:
                        rag_metadata[placeholder] = metadata
                else:
                    rag_content[placeholder] = result

        return rag_content, rag_metadata

    async def _execute_single_rag_search_safe(
        self,
        rag_config: RAGConfig,
        params: Dict[str, Any]
    ) -> str:
        """Safely execute a single RAG search (for use with asyncio.gather).

        Args:
            rag_config: RAG configuration
            params: Parameters for query templating

        Returns:
            Formatted RAG content string

        Raises:
            Exception: Propagated from _execute_single_rag_search
        """
        return await self._execute_single_rag_search(rag_config, params)

    async def _execute_single_rag_search(
        self,
        rag_config: RAGConfig,
        params: Dict[str, Any]
    ) -> str:
        """Execute a single RAG search and format results asynchronously.

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

        # Execute search (async)
        k = rag_config.get("k", 5)
        filters = rag_config.get("filters")
        search_results = await adapter.search(query=query, k=k, filters=filters)

        # Format results
        formatted_content = self._format_rag_results(
            results=search_results,
            rag_config=rag_config,
            params=params
        )

        return formatted_content

    async def _execute_single_rag_with_metadata(
        self,
        rag_config: RAGConfig,
        params: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Execute a single RAG search with metadata capture.

        This method executes a RAG search and captures detailed metadata
        including the query, results, and query hash for caching.

        Args:
            rag_config: RAG configuration
            params: Parameters for query templating

        Returns:
            Tuple of (formatted_content, metadata):
            - formatted_content: Formatted RAG content string
            - metadata: Dictionary with RAG metadata including:
                - adapter_name: Name of the adapter used
                - query: Rendered query string
                - query_hash: SHA256 hash for cache matching
                - k: Number of results requested
                - filters: Filters applied to search
                - timestamp: ISO format timestamp
                - results: Raw search results
                - formatted_content: Formatted output
                - item_template: Template used for formatting
                - header: Header text used

        Raises:
            KeyError: If adapter not found
            Exception: If search fails
        """
        from datetime import datetime

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

        # Compute query hash for cache matching
        query_hash = self._compute_rag_query_hash(adapter_name, query)

        # Execute search (async)
        k = rag_config.get("k", 5)
        filters = rag_config.get("filters")
        search_results = await adapter.search(query=query, k=k, filters=filters)

        # Format results
        formatted_content = self._format_rag_results(
            results=search_results,
            rag_config=rag_config,
            params=params
        )

        # Build metadata
        metadata = {
            "adapter_name": adapter_name,
            "query": query,
            "query_hash": query_hash,
            "k": k,
            "filters": filters,
            "timestamp": datetime.now().isoformat(),
            "results": search_results,  # Store raw results
            "formatted_content": formatted_content,
            "item_template": rag_config.get("item_template"),
            "header": rag_config.get("header"),
        }

        return formatted_content, metadata

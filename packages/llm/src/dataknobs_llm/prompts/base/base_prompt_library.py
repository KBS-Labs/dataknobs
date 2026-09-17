# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Shared functionality for prompt libraries of either flavour.

This module provides BasePromptLibrary, a mixin carrying the caching and
parsing every library implementation needs and none of the interface itself.
"""

from __future__ import annotations

from typing import Any
import logging

from .types import (
    MessageIndex,
    PromptTemplateDict,
    RAGConfig,
    ValidationConfig,
    ValidationLevel,
    rag_config_from_dict,
)

logger = logging.getLogger(__name__)


class BasePromptLibrary:
    """Caching, parsing and metadata, shared by libraries of either flavour.

    This class provides:
    - Optional caching of loaded prompts and message indexes
    - Helper methods for cache management
    - Shared metadata handling
    - Shared parsing of templates, validation blocks and RAG configs

    It is a **mixin, not an interface**. It declares neither
    :class:`AbstractPromptLibrary` nor :class:`AsyncPromptLibrary`, so a
    library names its flavour itself and inherits the machinery here
    alongside it::

        class FileSystemPromptLibrary(BasePromptLibrary, AbstractPromptLibrary):

    That is the shape this package already runs one layer down --
    ``ResourceAdapterBase`` with ``ResourceAdapter`` and
    ``AsyncResourceAdapter`` -- and it is what lets an asynchronous library
    reuse this caching rather than copy it. A mixin that declared the
    synchronous interface could only be reused by making an async library
    answer ``True`` to both.

    Being flavour-neutral is a property of every member here, not just of the
    bases: the shared reloading is :meth:`_reload_caches`, and each library
    spells the public ``reload`` in its own flavour over it.

    It also used to *stub* that interface: eight ``NotImplementedError``
    bodies, one per abstract method. To :class:`abc.ABC` a stub is an
    implementation, so those eight switched off the construct-time check for
    every subclass -- a library missing a method built fine and raised at the
    first call instead, which is the one moment its author is no longer
    watching. They are gone; the ninth, :meth:`get_metadata`, was never a stub
    and stays.
    """

    def __init__(self, enable_cache: bool = True, metadata: dict[str, Any] | None = None):
        """Initialize the base prompt library.

        Args:
            enable_cache: Whether to cache loaded prompts (default: True)
            metadata: Optional metadata dictionary
        """
        self._enable_cache = enable_cache
        self._metadata = metadata or {}

        # Caches for loaded prompts and indexes
        self._system_prompt_cache: dict[str, PromptTemplateDict] = {}
        self._user_prompt_cache: dict[str, PromptTemplateDict] = {}
        self._message_index_cache: dict[str, MessageIndex] = {}
        self._rag_config_cache: dict[str, RAGConfig] = {}  # Standalone RAG configs
        self._prompt_rag_cache: dict[tuple, list[RAGConfig]] = {}  # (name, type)

    # ===== Cache Management =====

    def clear_cache(self) -> None:
        """Clear all cached prompts and indexes."""
        self._system_prompt_cache.clear()
        self._user_prompt_cache.clear()
        self._message_index_cache.clear()
        self._rag_config_cache.clear()
        self._prompt_rag_cache.clear()
        logger.debug(f"Cleared cache for {self.__class__.__name__}")

    def _reload_caches(self) -> None:
        """Drop everything cached here, so the next read goes to the source.

        The reloading a library of *either* flavour shares, and deliberately
        not spelled ``reload``. A public ``def reload`` here would be a flavour
        after all: it wins the MRO over :class:`AsyncPromptLibrary`'s
        ``async def`` default, so an asynchronous library reusing this mixin
        constructed fine, awaited every accessor correctly, and raised
        ``TypeError: object NoneType can't be used in 'await' expression`` on
        ``await library.reload()`` --- a mixin whose whole claim is that it
        declares no flavour, declaring one.

        Each library spells its own ``reload`` in its own flavour and calls
        this; what differs between them is only the ``async``.
        """
        if self._enable_cache:
            self.clear_cache()
        logger.info(f"Reloaded {self.__class__.__name__}")

    # ===== Metadata =====

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about this prompt library.

        Returns:
            Dictionary with library metadata
        """
        return {
            "class": self.__class__.__name__,
            "cache_enabled": self._enable_cache,
            **self._metadata,
        }

    # ===== Cache Helpers =====

    def _get_cached_system_prompt(self, name: str) -> PromptTemplateDict | None:
        """Get system prompt from cache if caching is enabled.

        Args:
            name: System prompt identifier

        Returns:
            Cached PromptTemplateDict if found, None otherwise
        """
        if not self._enable_cache:
            return None
        return self._system_prompt_cache.get(name)

    def _cache_system_prompt(self, name: str, template: PromptTemplateDict) -> None:
        """Cache a system prompt if caching is enabled.

        Args:
            name: System prompt identifier
            template: PromptTemplateDict to cache
        """
        if self._enable_cache:
            self._system_prompt_cache[name] = template

    def _get_cached_user_prompt(self, name: str) -> PromptTemplateDict | None:
        """Get user prompt from cache if caching is enabled.

        Args:
            name: User prompt identifier

        Returns:
            Cached PromptTemplateDict if found, None otherwise
        """
        if not self._enable_cache:
            return None
        return self._user_prompt_cache.get(name)

    def _cache_user_prompt(self, name: str, template: PromptTemplateDict) -> None:
        """Cache a user prompt if caching is enabled.

        Args:
            name: User prompt identifier
            template: PromptTemplateDict to cache
        """
        if self._enable_cache:
            self._user_prompt_cache[name] = template

    def _get_cached_message_index(self, name: str) -> MessageIndex | None:
        """Get message index from cache if caching is enabled.

        Args:
            name: Message index identifier

        Returns:
            Cached MessageIndex if found, None otherwise
        """
        if not self._enable_cache:
            return None
        return self._message_index_cache.get(name)

    def _cache_message_index(self, name: str, index: MessageIndex) -> None:
        """Cache a message index if caching is enabled.

        Args:
            name: Message index identifier
            index: MessageIndex to cache
        """
        if self._enable_cache:
            self._message_index_cache[name] = index

    def _get_cached_rag_config(self, name: str) -> RAGConfig | None:
        """Get standalone RAG config from cache if caching is enabled.

        Args:
            name: RAG config identifier

        Returns:
            Cached RAGConfig if found, None otherwise
        """
        if not self._enable_cache:
            return None
        return self._rag_config_cache.get(name)

    def _cache_rag_config(self, name: str, config: RAGConfig) -> None:
        """Cache a standalone RAG config if caching is enabled.

        Args:
            name: RAG config identifier
            config: RAGConfig to cache
        """
        if self._enable_cache:
            self._rag_config_cache[name] = config

    def _get_cached_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str
    ) -> list[RAGConfig] | None:
        """Get prompt RAG configs from cache if caching is enabled.

        Args:
            prompt_name: Prompt identifier
            prompt_type: Type of prompt ("user" or "system")

        Returns:
            Cached list of RAGConfig if found, None otherwise
        """
        if not self._enable_cache:
            return None
        return self._prompt_rag_cache.get((prompt_name, prompt_type))

    def _cache_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str, configs: list[RAGConfig]
    ) -> None:
        """Cache prompt RAG configurations if caching is enabled.

        Args:
            prompt_name: Prompt identifier
            prompt_type: Type of prompt ("user" or "system")
            configs: List of RAGConfig to cache
        """
        if self._enable_cache:
            self._prompt_rag_cache[(prompt_name, prompt_type)] = configs

    # ===== Common Parsing Methods =====

    def _parse_validation_config(self, data: dict | ValidationConfig) -> ValidationConfig:
        """Parse validation configuration from dict or ValidationConfig.

        This method is shared by all library implementations for consistent
        validation config parsing.

        Args:
            data: Validation data (dict or ValidationConfig instance)

        Returns:
            ValidationConfig instance

        Raises:
            ValueError: If data type is invalid
        """
        if isinstance(data, ValidationConfig):
            return data

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid validation config: expected dict or ValidationConfig, got {type(data)}"
            )

        # Parse level
        level = None
        if "level" in data:
            level_data = data["level"]
            if isinstance(level_data, str):
                level = ValidationLevel(level_data.lower())
            elif isinstance(level_data, ValidationLevel):
                level = level_data

        # Parse params
        required_params = data.get("required_params", [])
        optional_params = data.get("optional_params", [])

        return ValidationConfig(
            level=level, required_params=required_params, optional_params=optional_params
        )

    def _parse_rag_config(self, data: dict[str, Any]) -> RAGConfig:
        """Parse RAG configuration from dict.

        This method is shared by all library implementations for consistent
        RAG config parsing.

        Args:
            data: RAG config data dictionary

        Returns:
            RAGConfig dictionary
        """
        return rag_config_from_dict(data)

    def _parse_prompt_template(self, data: Any) -> PromptTemplateDict:
        """Parse prompt template from various formats.

        This method is shared by all library implementations for consistent
        template parsing. Supports:
        - String templates (converted to {"template": string})
        - Dict with "template" key
        - Dict with "extends" key but no "template" (template inherited)
        - Empty dict (treated as {"template": ""})

        Args:
            data: Prompt template data (string or dict)

        Returns:
            PromptTemplateDict dictionary

        Raises:
            ValueError: If data format is invalid
        """
        # If just a string, treat as template
        if isinstance(data, str):
            return {"template": data}

        # If empty dict, treat as empty template
        if isinstance(data, dict) and len(data) == 0:
            return {"template": ""}

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid prompt template data: expected dict with 'template' or 'extends' key, "
                f"or string, got {type(data)}"
            )

        # Must have either "template" or "extends" (or both)
        has_template = "template" in data
        has_extends = "extends" in data

        if not has_template and not has_extends:
            raise ValueError(
                f"Invalid prompt template data: expected dict with 'template' or 'extends' key, "
                f"or string, got dict with keys: {list(data.keys())}"
            )

        # Build the template dict — start with the required key(s)
        template: PromptTemplateDict = {}
        if has_template:
            template["template"] = data["template"]
        if has_extends:
            template["extends"] = data["extends"]

        # Copy optional fields shared by all template variants
        self._apply_optional_fields(template, data)

        return template

    def _apply_optional_fields(self, template: PromptTemplateDict, data: dict[str, Any]) -> None:
        """Copy optional fields from parsed data into a PromptTemplateDict.

        This is the single location where optional PromptTemplateDict fields
        are transferred from input data to the parsed template. Adding a new
        field to PromptTemplateDict requires only one change here.

        Args:
            template: The template dict being built (modified in place).
            data: The raw input data dict.
        """
        # Simple pass-through fields
        passthrough_fields = (
            "defaults",
            "metadata",
            "template_mode",
            "template_syntax",
            "sections",
            "rag_config_refs",
        )
        for field_name in passthrough_fields:
            if field_name in data:
                template[field_name] = data[field_name]  # type: ignore[literal-required]

        # Fields requiring parsing
        if "validation" in data:
            template["validation"] = self._parse_validation_config(data["validation"])

        if "rag_configs" in data:
            template["rag_configs"] = [
                self._parse_rag_config(rag_data) for rag_data in data["rag_configs"]
            ]

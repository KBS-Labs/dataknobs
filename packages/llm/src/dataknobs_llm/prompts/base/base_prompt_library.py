"""Base implementation of prompt library with shared functionality.

This module provides BasePromptLibrary, a concrete implementation of
AbstractPromptLibrary that includes caching and common utilities.
"""

from typing import Any, Dict, List, Union
import logging

from .abstract_prompt_library import AbstractPromptLibrary
from .types import PromptTemplateDict, MessageIndex, RAGConfig, ValidationConfig, ValidationLevel

logger = logging.getLogger(__name__)


class BasePromptLibrary(AbstractPromptLibrary):
    """Base implementation with caching and common functionality.

    This class provides:
    - Optional caching of loaded prompts and message indexes
    - Helper methods for cache management
    - Shared metadata handling
    - Default implementations of optional methods

    Subclasses should implement the abstract methods to provide
    the actual prompt loading logic.
    """

    def __init__(
        self,
        enable_cache: bool = True,
        metadata: Dict[str, Any] | None = None
    ):
        """Initialize the base prompt library.

        Args:
            enable_cache: Whether to cache loaded prompts (default: True)
            metadata: Optional metadata dictionary
        """
        self._enable_cache = enable_cache
        self._metadata = metadata or {}

        # Caches for loaded prompts and indexes
        self._system_prompt_cache: Dict[str, PromptTemplateDict] = {}
        self._user_prompt_cache: Dict[str, PromptTemplateDict] = {}
        self._message_index_cache: Dict[str, MessageIndex] = {}
        self._rag_config_cache: Dict[str, RAGConfig] = {}  # Standalone RAG configs
        self._prompt_rag_cache: Dict[tuple, List[RAGConfig]] = {}  # (name, type)

    # ===== Cache Management =====

    def clear_cache(self) -> None:
        """Clear all cached prompts and indexes."""
        self._system_prompt_cache.clear()
        self._user_prompt_cache.clear()
        self._message_index_cache.clear()
        self._rag_config_cache.clear()
        self._prompt_rag_cache.clear()
        logger.debug(f"Cleared cache for {self.__class__.__name__}")

    def reload(self) -> None:
        """Reload the library by clearing the cache.

        Subclasses can override to perform additional reload logic.
        """
        if self._enable_cache:
            self.clear_cache()
        logger.info(f"Reloaded {self.__class__.__name__}")

    # ===== Metadata =====

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this prompt library.

        Returns:
            Dictionary with library metadata
        """
        return {
            "class": self.__class__.__name__,
            "cache_enabled": self._enable_cache,
            **self._metadata
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
        self,
        prompt_name: str,
        prompt_type: str
    ) -> List[RAGConfig] | None:
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
        self,
        prompt_name: str,
        prompt_type: str,
        configs: List[RAGConfig]
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

    def _parse_validation_config(self, data: Union[Dict, ValidationConfig]) -> ValidationConfig:
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
                f"Invalid validation config: expected dict or ValidationConfig, "
                f"got {type(data)}"
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
            level=level,
            required_params=required_params,
            optional_params=optional_params
        )

    def _parse_rag_config(self, data: Dict[str, Any]) -> RAGConfig:
        """Parse RAG configuration from dict.

        This method is shared by all library implementations for consistent
        RAG config parsing.

        Args:
            data: RAG config data dictionary

        Returns:
            RAGConfig dictionary
        """
        rag_config: RAGConfig = {
            "adapter_name": data.get("adapter_name", ""),
            "query": data.get("query", ""),
        }

        # Add optional fields
        if "k" in data:
            rag_config["k"] = data["k"]

        if "filters" in data:
            rag_config["filters"] = data["filters"]

        if "placeholder" in data:
            rag_config["placeholder"] = data["placeholder"]

        if "header" in data:
            rag_config["header"] = data["header"]

        if "item_template" in data:
            rag_config["item_template"] = data["item_template"]

        return rag_config

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

        # Initialize template
        template: PromptTemplateDict = None

        # If dict with template field
        if isinstance(data, dict) and "template" in data:
            template = {
                "template": data["template"],
            }

            # Add optional fields
            if "defaults" in data:
                template["defaults"] = data["defaults"]

            if "validation" in data:
                template["validation"] = self._parse_validation_config(data["validation"])

            if "metadata" in data:
                template["metadata"] = data["metadata"]

            # Add template mode field
            if "template_mode" in data:
                template["template_mode"] = data["template_mode"]

            # Add composition fields
            if "sections" in data:
                template["sections"] = data["sections"]

            if "extends" in data:
                template["extends"] = data["extends"]

            # Add RAG configuration fields
            if "rag_config_refs" in data:
                template["rag_config_refs"] = data["rag_config_refs"]

            if "rag_configs" in data:
                template["rag_configs"] = [
                    self._parse_rag_config(rag_data)
                    for rag_data in data["rag_configs"]
                ]

            return template

        # If dict with extends field but no template (template will be inherited)
        elif isinstance(data, dict) and "extends" in data:
            template = {}

            # Template will be inherited from base
            template["extends"] = data["extends"]

            # Add optional override fields
            if "defaults" in data:
                template["defaults"] = data["defaults"]

            if "validation" in data:
                template["validation"] = self._parse_validation_config(data["validation"])

            if "metadata" in data:
                template["metadata"] = data["metadata"]

            # Add template mode field
            if "template_mode" in data:
                template["template_mode"] = data["template_mode"]

            if "sections" in data:
                template["sections"] = data["sections"]

            # Add RAG configuration fields
            if "rag_config_refs" in data:
                template["rag_config_refs"] = data["rag_config_refs"]

            if "rag_configs" in data:
                template["rag_configs"] = [
                    self._parse_rag_config(rag_data)
                    for rag_data in data["rag_configs"]
                ]

            return template

        else:
            raise ValueError(
                f"Invalid prompt template data: expected dict with 'template' or 'extends' key, "
                f"or string, got {type(data)}"
            )

    # ===== Abstract Methods (must be implemented by subclasses) =====

    def get_system_prompt(
        self,
        name: str,
        **kwargs: Any
    ) -> PromptTemplateDict | None:
        """Retrieve a system prompt template by name.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_system_prompt()"
        )

    def list_system_prompts(self) -> List[str]:
        """List all available system prompt names.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement list_system_prompts()"
        )

    def get_user_prompt(
        self,
        name: str,
        index: int = 0,
        **kwargs: Any
    ) -> PromptTemplateDict | None:
        """Retrieve a user prompt template by name and index.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_user_prompt()"
        )

    def list_user_prompts(self) -> List[str]:
        """List available user prompts.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement list_user_prompts()"
        )

    def get_message_index(
        self,
        name: str,
        **kwargs: Any
    ) -> MessageIndex | None:
        """Retrieve a message index by name.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_message_index()"
        )

    def list_message_indexes(self) -> List[str]:
        """List all available message index names.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement list_message_indexes()"
        )

    def get_rag_config(
        self,
        name: str,
        **kwargs: Any
    ) -> RAGConfig | None:
        """Retrieve a standalone RAG configuration by name.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_rag_config()"
        )

    def get_prompt_rag_configs(
        self,
        prompt_name: str,
        prompt_type: str = "user",
        index: int = 0,
        **kwargs: Any
    ) -> List[RAGConfig]:
        """Retrieve RAG configurations for a specific prompt.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_prompt_rag_configs()"
        )

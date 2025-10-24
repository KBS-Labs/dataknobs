"""Base implementation of prompt library with shared functionality.

This module provides BasePromptLibrary, a concrete implementation of
AbstractPromptLibrary that includes caching and common utilities.
"""

from typing import Any, Dict, List, Optional
import logging

from .abstract_prompt_library import AbstractPromptLibrary
from .types import PromptTemplate, MessageIndex, RAGConfig

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
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the base prompt library.

        Args:
            enable_cache: Whether to cache loaded prompts (default: True)
            metadata: Optional metadata dictionary
        """
        self._enable_cache = enable_cache
        self._metadata = metadata or {}

        # Caches for loaded prompts and indexes
        self._system_prompt_cache: Dict[str, PromptTemplate] = {}
        self._user_prompt_cache: Dict[tuple, PromptTemplate] = {}  # (name, index)
        self._message_index_cache: Dict[str, MessageIndex] = {}
        self._rag_config_cache: Dict[str, RAGConfig] = {}  # Standalone RAG configs
        self._prompt_rag_cache: Dict[tuple, List[RAGConfig]] = {}  # (name, type, index)

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

    def _get_cached_system_prompt(self, name: str) -> Optional[PromptTemplate]:
        """Get system prompt from cache if caching is enabled.

        Args:
            name: System prompt identifier

        Returns:
            Cached PromptTemplate if found, None otherwise
        """
        if not self._enable_cache:
            return None
        return self._system_prompt_cache.get(name)

    def _cache_system_prompt(self, name: str, template: PromptTemplate) -> None:
        """Cache a system prompt if caching is enabled.

        Args:
            name: System prompt identifier
            template: PromptTemplate to cache
        """
        if self._enable_cache:
            self._system_prompt_cache[name] = template

    def _get_cached_user_prompt(
        self,
        name: str,
        index: int
    ) -> Optional[PromptTemplate]:
        """Get user prompt from cache if caching is enabled.

        Args:
            name: User prompt identifier
            index: Prompt variant index

        Returns:
            Cached PromptTemplate if found, None otherwise
        """
        if not self._enable_cache:
            return None
        return self._user_prompt_cache.get((name, index))

    def _cache_user_prompt(
        self,
        name: str,
        index: int,
        template: PromptTemplate
    ) -> None:
        """Cache a user prompt if caching is enabled.

        Args:
            name: User prompt identifier
            index: Prompt variant index
            template: PromptTemplate to cache
        """
        if self._enable_cache:
            self._user_prompt_cache[(name, index)] = template

    def _get_cached_message_index(self, name: str) -> Optional[MessageIndex]:
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

    def _get_cached_rag_config(self, name: str) -> Optional[RAGConfig]:
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
        prompt_type: str,
        index: int
    ) -> Optional[List[RAGConfig]]:
        """Get prompt RAG configs from cache if caching is enabled.

        Args:
            prompt_name: Prompt identifier
            prompt_type: Type of prompt ("user" or "system")
            index: Prompt variant index

        Returns:
            Cached list of RAGConfig if found, None otherwise
        """
        if not self._enable_cache:
            return None
        return self._prompt_rag_cache.get((prompt_name, prompt_type, index))

    def _cache_prompt_rag_configs(
        self,
        prompt_name: str,
        prompt_type: str,
        index: int,
        configs: List[RAGConfig]
    ) -> None:
        """Cache prompt RAG configurations if caching is enabled.

        Args:
            prompt_name: Prompt identifier
            prompt_type: Type of prompt ("user" or "system")
            index: Prompt variant index
            configs: List of RAGConfig to cache
        """
        if self._enable_cache:
            self._prompt_rag_cache[(prompt_name, prompt_type, index)] = configs

    # ===== Abstract Methods (must be implemented by subclasses) =====

    def get_system_prompt(
        self,
        name: str,
        **kwargs: Any
    ) -> Optional[PromptTemplate]:
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
    ) -> Optional[PromptTemplate]:
        """Retrieve a user prompt template by name and index.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_user_prompt()"
        )

    def list_user_prompts(self, name: Optional[str] = None) -> List[str]:
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
    ) -> Optional[MessageIndex]:
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
    ) -> Optional[RAGConfig]:
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

"""Abstract base class for prompt libraries.

This module defines the interface that all prompt library implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .types import PromptTemplateDict, MessageIndex, RAGConfig


class AbstractPromptLibrary(ABC):
    """Abstract base class for prompt library implementations.

    All prompt libraries (filesystem, config, composite, etc.) must implement
    this interface to provide consistent access to prompts, message indexes,
    and RAG configurations.
    """

    # ===== System Prompts =====

    @abstractmethod
    def get_system_prompt(
        self,
        name: str,
        **kwargs: Any
    ) -> PromptTemplateDict | None:
        """Retrieve a system prompt template by name.

        Args:
            name: System prompt identifier
            **kwargs: Additional library-specific parameters

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        pass

    @abstractmethod
    def list_system_prompts(self) -> List[str]:
        """List all available system prompt names.

        Returns:
            List of system prompt identifiers
        """
        pass

    # ===== User Prompts =====

    @abstractmethod
    def get_user_prompt(
        self,
        name: str,
        **kwargs: Any
    ) -> PromptTemplateDict | None:
        """Retrieve a user prompt template by name.

        Args:
            name: User prompt identifier
            **kwargs: Additional library-specific parameters

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        pass

    @abstractmethod
    def list_user_prompts(self) -> List[str]:
        """List available user prompts.

        Returns:
            List of user prompt identifiers
        """
        pass

    # ===== Message Indexes =====

    @abstractmethod
    def get_message_index(
        self,
        name: str,
        **kwargs: Any
    ) -> MessageIndex | None:
        """Retrieve a message index by name.

        Args:
            name: Message index identifier
            **kwargs: Additional library-specific parameters

        Returns:
            MessageIndex if found, None otherwise
        """
        pass

    @abstractmethod
    def list_message_indexes(self) -> List[str]:
        """List all available message index names.

        Returns:
            List of message index identifiers
        """
        pass

    # ===== RAG Configurations =====

    @abstractmethod
    def get_rag_config(
        self,
        name: str,
        **kwargs: Any
    ) -> RAGConfig | None:
        """Retrieve a standalone RAG configuration by name.

        Standalone RAG configs can be referenced from prompts or used directly.

        Args:
            name: RAG configuration identifier
            **kwargs: Additional library-specific parameters

        Returns:
            RAGConfig if found, None otherwise
        """
        pass

    @abstractmethod
    def get_prompt_rag_configs(
        self,
        prompt_name: str,
        prompt_type: str = "user",
        **kwargs: Any
    ) -> List[RAGConfig]:
        """Retrieve RAG configurations for a specific prompt.

        This resolves both inline RAG configs and references to standalone configs.

        Args:
            prompt_name: Name of the prompt
            prompt_type: Type of prompt ("user" or "system")
            **kwargs: Additional library-specific parameters

        Returns:
            List of RAG configurations (empty if none defined)
        """
        pass

    # ===== Metadata & Lifecycle =====

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this prompt library.

        Returns:
            Dictionary with library metadata (source, version, etc.)
        """
        pass

    @abstractmethod
    def reload(self) -> None:
        """Reload the prompt library from its source.

        This is optional - implementations that support reloading should override.
        Default implementation does nothing.
        """
        pass

    # ===== Validation & Health Checks =====

    def validate(self) -> List[str]:
        """Validate the prompt library configuration.

        Returns:
            List of validation error messages (empty if valid)

        Note:
            Default implementation returns empty list. Implementations should
            override to provide specific validation logic.
        """
        return []

    def __repr__(self) -> str:
        """Return a string representation of this library."""
        metadata = self.get_metadata()
        return f"{self.__class__.__name__}({metadata.get('source', 'unknown')})"

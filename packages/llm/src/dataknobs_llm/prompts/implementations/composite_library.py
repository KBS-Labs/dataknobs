"""Composite prompt library implementation.

This module provides a library that combines multiple prompt libraries with
fallback behavior. Searches libraries in order and returns the first match.

Example:
    # Create layered configuration with fallbacks
    base_library = FileSystemPromptLibrary(Path("prompts/base/"))
    override_library = ConfigPromptLibrary({
        "system": {
            "analyze_code": {"template": "Custom analysis prompt..."}
        }
    })

    # Composite library searches override first, then base
    library = CompositePromptLibrary(
        libraries=[override_library, base_library],
        names=["override", "base"]
    )

    # Gets from override if available, otherwise from base
    template = library.get_system_prompt("analyze_code")
"""

import logging
from typing import Any, Dict, List

from ..base import (
    AbstractPromptLibrary,
    PromptTemplateDict,
    RAGConfig,
    MessageIndex,
)

logger = logging.getLogger(__name__)


class CompositePromptLibrary(AbstractPromptLibrary):
    """Composite library that combines multiple prompt libraries with fallback.

    Features:
    - Searches libraries in order (first match wins)
    - Useful for layered configurations (overrides + defaults)
    - Supports all prompt types
    - Optional library naming for debugging

    Example:
        >>> base = FileSystemPromptLibrary(Path("base/"))
        >>> custom = ConfigPromptLibrary(custom_config)
        >>> library = CompositePromptLibrary([custom, base])
        >>> # Searches custom first, then base
        >>> template = library.get_system_prompt("analyze")
    """

    def __init__(
        self,
        libraries: List[AbstractPromptLibrary] | None = None,
        names: List[str] | None = None
    ):
        """Initialize composite prompt library.

        Args:
            libraries: List of libraries to search (in order of priority)
            names: Optional names for libraries (for debugging/logging)
        """
        self._libraries = libraries or []
        self._names = names or [f"library_{i}" for i in range(len(self._libraries))]

        if len(self._names) != len(self._libraries):
            raise ValueError(
                f"Number of names ({len(self._names)}) must match "
                f"number of libraries ({len(self._libraries)})"
            )

    def add_library(
        self,
        library: AbstractPromptLibrary,
        name: str | None = None,
        priority: int = -1
    ) -> None:
        """Add a library to the composite.

        Args:
            library: Library to add
            name: Optional name for the library
            priority: Position to insert library (default: -1 = end)
                     0 = highest priority (searched first)
        """
        if name is None:
            name = f"library_{len(self._libraries)}"

        if priority < 0 or priority >= len(self._libraries):
            # Append to end
            self._libraries.append(library)
            self._names.append(name)
        else:
            # Insert at specific position
            self._libraries.insert(priority, library)
            self._names.insert(priority, name)

        logger.debug(
            f"Added library '{name}' at priority {priority if priority >= 0 else len(self._libraries) - 1}"
        )

    def remove_library(self, name: str) -> bool:
        """Remove a library by name.

        Args:
            name: Name of the library to remove

        Returns:
            True if library was removed, False if not found
        """
        try:
            index = self._names.index(name)
            self._libraries.pop(index)
            self._names.pop(index)
            logger.debug(f"Removed library '{name}'")
            return True
        except ValueError:
            logger.warning(f"Library '{name}' not found")
            return False

    def get_system_prompt(self, name: str, **kwargs) -> PromptTemplateDict | None:
        """Get a system prompt by name, searching libraries in order.

        Args:
            name: System prompt name
            **kwargs: Additional arguments passed to libraries

        Returns:
            PromptTemplateDict from first library that has it, or None
        """
        for lib, lib_name in zip(self._libraries, self._names, strict=True):
            template = lib.get_system_prompt(name, **kwargs)
            if template is not None:
                logger.debug(f"Found system prompt '{name}' in library '{lib_name}'")
                return template

        logger.debug(f"System prompt '{name}' not found in any library")
        return None

    def get_user_prompt(
        self,
        name: str,
        **kwargs
    ) -> PromptTemplateDict | None:
        """Get a user prompt by name, searching libraries in order.

        Args:
            name: User prompt name
            **kwargs: Additional arguments passed to libraries

        Returns:
            PromptTemplateDict from first library that has it, or None
        """
        for lib, lib_name in zip(self._libraries, self._names, strict=True):
            template = lib.get_user_prompt(name, **kwargs)
            if template is not None:
                logger.debug(
                    f"Found user prompt '{name}' in library '{lib_name}'"
                )
                return template

        logger.debug(f"User prompt '{name}' not found in any library")
        return None

    def get_message_index(self, name: str, **kwargs) -> MessageIndex | None:
        """Get a message index by name, searching libraries in order.

        Args:
            name: Message index name
            **kwargs: Additional arguments passed to libraries

        Returns:
            MessageIndex from first library that has it, or None
        """
        for lib, lib_name in zip(self._libraries, self._names, strict=True):
            message_index = lib.get_message_index(name, **kwargs)
            if message_index is not None:
                logger.debug(f"Found message index '{name}' in library '{lib_name}'")
                return message_index

        logger.debug(f"Message index '{name}' not found in any library")
        return None

    def get_rag_config(self, name: str, **kwargs) -> RAGConfig | None:
        """Get a standalone RAG configuration by name, searching libraries in order.

        Args:
            name: RAG config name
            **kwargs: Additional arguments passed to libraries

        Returns:
            RAGConfig from first library that has it, or None
        """
        for lib, lib_name in zip(self._libraries, self._names, strict=True):
            rag_config = lib.get_rag_config(name, **kwargs)
            if rag_config is not None:
                logger.debug(f"Found RAG config '{name}' in library '{lib_name}'")
                return rag_config

        logger.debug(f"RAG config '{name}' not found in any library")
        return None

    def get_prompt_rag_configs(
        self,
        prompt_name: str,
        prompt_type: str = "user",
        **kwargs
    ) -> List[RAGConfig]:
        """Get RAG configurations for a prompt, searching libraries in order.

        Args:
            prompt_name: Prompt name
            prompt_type: Type of prompt ("user" or "system")
            **kwargs: Additional arguments passed to libraries

        Returns:
            List of RAGConfig from first library that has the prompt
        """
        for lib, lib_name in zip(self._libraries, self._names, strict=True):
            configs = lib.get_prompt_rag_configs(prompt_name, prompt_type, **kwargs)
            if configs:
                logger.debug(
                    f"Found {len(configs)} RAG config(s) for prompt '{prompt_name}' "
                    f"in library '{lib_name}'"
                )
                return configs

        logger.debug(f"No RAG configs found for prompt '{prompt_name}' in any library")
        return []

    def list_system_prompts(self) -> List[str]:
        """List all available system prompt names from all libraries.

        Returns:
            Combined list of system prompt identifiers
        """
        prompts = set()
        for lib in self._libraries:
            prompts.update(lib.list_system_prompts())
        return sorted(prompts)

    def list_user_prompts(self) -> List[str]:
        """List available user prompts from all libraries.

        Returns:
            Combined list of user prompt names or indices
        """
        prompts = set()
        for lib in self._libraries:
            prompts.update(lib.list_user_prompts())
        return sorted(prompts)

    def list_message_indexes(self) -> List[str]:
        """List all available message index names from all libraries.

        Returns:
            Combined list of message index identifiers
        """
        indexes = set()
        for lib in self._libraries:
            indexes.update(lib.list_message_indexes())
        return sorted(indexes)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this composite library.

        Returns:
            Dictionary with library metadata
        """
        return {
            "class": self.__class__.__name__,
            "num_libraries": len(self._libraries),
            "library_names": self._names,
        }

    def reload(self) -> None:
        """Reload the library by clearing the cache.

        Subclasses can override to perform additional reload logic.
        """
        for lib in self._libraries:
            lib.reload()

    @property
    def libraries(self) -> List[AbstractPromptLibrary]:
        """Get list of libraries in priority order."""
        return self._libraries.copy()

    @property
    def library_names(self) -> List[str]:
        """Get list of library names in priority order."""
        return self._names.copy()

    def get_library_by_name(self, name: str) -> AbstractPromptLibrary | None:
        """Get a specific library by name.

        Args:
            name: Library name

        Returns:
            Library if found, None otherwise
        """
        try:
            index = self._names.index(name)
            return self._libraries[index]
        except ValueError:
            return None

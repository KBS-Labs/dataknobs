# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The asynchronous half of the prompt-library interface.

:class:`AbstractPromptLibrary` declares every accessor ``def``, which is right
for a library whose content is already in memory --- a directory read at
construction, a configuration dictionary --- and wrong for one that fronts a
store. A library backed by a database has to *await* to answer, and a
synchronous signature leaves it two ways out, both bad: block the caller's
thread on a private loop, or refuse when the caller is already on one.

So the flavour is declared rather than worked around. This module is
:class:`AbstractPromptLibrary` member for member with the reaching members made
coroutines; :mod:`dataknobs_llm.prompts.base.views` is the pair of doors
between the two, for a caller holding one flavour and needing the other.

The one member that does not flip is :meth:`AsyncPromptLibrary.get_metadata`.
It answers from the library's own configuration rather than from its content,
so it reaches nothing and an ``await`` on it would cost every caller a
suspension for a dictionary literal. That difference is *declared* to the
parity guard (``unflavoured_members=["get_metadata"]``) rather than discovered,
so a second divergence fails the guard instead of quietly joining the first.

Note the annotations here are evaluated rather than deferred --- no
``from __future__ import annotations``. The parity guard compares the two
halves' parameter annotations by equality through :func:`inspect.signature`,
which does not evaluate strings, so a deferred module would compare ``'str'``
against :class:`str` and report drift on a pair that agrees.
"""

from abc import ABC, abstractmethod
from typing import Any

from .types import MessageIndex, PromptTemplateDict, RAGConfig


class AsyncPromptLibrary(ABC):
    """Interface for prompt libraries that reach a store to answer.

    The asynchronous twin of :class:`AbstractPromptLibrary`. Implement this one
    when answering involves I/O --- a database, an object store, a network
    call; implement the synchronous one when the content is already in memory
    by the time the library exists.

    A library implementing this can still inherit
    :class:`~dataknobs_llm.prompts.base.base_prompt_library.BasePromptLibrary`
    for caching, parsing and metadata: that class is a mixin and declares
    neither flavour.

    Consumers written against the other flavour are reached through
    :func:`~dataknobs_llm.prompts.base.views.as_sync`, and a synchronous
    library is presented to an async consumer with
    :func:`~dataknobs_llm.prompts.base.views.as_async`.
    """

    # ===== System Prompts =====

    @abstractmethod
    async def get_system_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Retrieve a system prompt template by name.

        Args:
            name: System prompt identifier
            **kwargs: Additional library-specific parameters

        Returns:
            PromptTemplateDict if found, None otherwise
        """

    @abstractmethod
    async def list_system_prompts(self) -> list[str]:
        """List all available system prompt names.

        Returns:
            List of system prompt identifiers
        """

    # ===== User Prompts =====

    @abstractmethod
    async def get_user_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Retrieve a user prompt template by name.

        Args:
            name: User prompt identifier
            **kwargs: Additional library-specific parameters

        Returns:
            PromptTemplateDict if found, None otherwise
        """

    @abstractmethod
    async def list_user_prompts(self) -> list[str]:
        """List available user prompts.

        Returns:
            List of user prompt identifiers
        """

    # ===== Message Indexes =====

    @abstractmethod
    async def get_message_index(self, name: str, **kwargs: Any) -> MessageIndex | None:
        """Retrieve a message index by name.

        Args:
            name: Message index identifier
            **kwargs: Additional library-specific parameters

        Returns:
            MessageIndex if found, None otherwise
        """

    @abstractmethod
    async def list_message_indexes(self) -> list[str]:
        """List all available message index names.

        Returns:
            List of message index identifiers
        """

    # ===== RAG Configurations =====

    @abstractmethod
    async def get_rag_config(self, name: str, **kwargs: Any) -> RAGConfig | None:
        """Retrieve a standalone RAG configuration by name.

        Standalone RAG configs can be referenced from prompts or used directly.

        Args:
            name: RAG configuration identifier
            **kwargs: Additional library-specific parameters

        Returns:
            RAGConfig if found, None otherwise
        """

    @abstractmethod
    async def get_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str = "user", **kwargs: Any
    ) -> list[RAGConfig]:
        """Retrieve RAG configurations for a specific prompt.

        This resolves both inline RAG configs and references to standalone configs.

        Args:
            prompt_name: Name of the prompt
            prompt_type: Type of prompt ("user" or "system")
            **kwargs: Additional library-specific parameters

        Returns:
            List of RAG configurations (empty if none defined)
        """

    # ===== Metadata & Lifecycle =====

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about this prompt library.

        Synchronous on both halves: it answers from this object's own
        configuration, so there is nothing to await.

        Returns:
            Dictionary with library metadata (source, version, etc.)
        """

    async def reload(self) -> None:  # noqa: B027 - optional hook; no-op default is the contract
        """Reload the prompt library from its source.

        This is optional - implementations that support reloading should override.
        Default implementation does nothing.
        """

    # ===== Validation & Health Checks =====

    async def validate(self) -> list[str]:
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

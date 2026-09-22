# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Bot execution context."""

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BotContext:
    """Runtime context for bot execution.

    Supports dict-like access for dynamic attributes via request_metadata.
    Use `context["key"]` or `context.get("key")` for dynamic data.

    That dict-like access is complete: ``in``, iteration, ``len``,
    ``keys``/``values``/``items``, ``dict(context)`` and ``{**context}`` all
    answer about ``request_metadata``. It used to stop at ``__getitem__`` /
    ``__setitem__`` / ``__contains__``, and the gap was a trap rather than a
    limitation: Python answers iteration from the *type*, so with no
    ``__iter__`` the interpreter fell back to the protocol that predates it
    and asked for index ``0``. ``dict(context)`` and ``list(context)`` raised
    ``KeyError: 0`` --- a key nothing here is ever stored under. ``in``
    worked, which is what kept it hidden.

    **Deliberately not a** :class:`~collections.abc.Mapping`. The dict-like
    surface is a convenience over *one field*, while the context's identity
    is ``conversation_id`` / ``client_id`` / ``user_id``. Registering the ABC
    would make ``isinstance(context, Mapping)`` true for everything that
    reads a mapping as the whole object --- serializing it, merging it,
    walking it --- and each of those would silently drop the three identity
    fields. For the same reason :meth:`copy` keeps its own meaning (clone the
    context, with field overrides) rather than a mapping's, and ``bool`` stays
    what it was: a context always carries a conversation and a client, so it
    is never empty. Ask ``len(context)`` about the metadata.

    Attributes:
        conversation_id: Unique identifier for the conversation
        client_id: Identifier for the client/tenant
        user_id: Optional user identifier
        session_metadata: Metadata for the session
        request_metadata: Metadata for the current request (also used for dict-like access)
    """

    conversation_id: str
    client_id: str
    user_id: str | None = None
    session_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Get item from request_metadata using dict-like access.

        Args:
            key: Key to retrieve

        Returns:
            Value from request_metadata

        Raises:
            KeyError: If key not found in request_metadata
        """
        return self.request_metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in request_metadata using dict-like access.

        Args:
            key: Key to set
            value: Value to store
        """
        self.request_metadata[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in request_metadata.

        Args:
            key: Key to check

        Returns:
            True if key exists in request_metadata
        """
        return key in self.request_metadata

    def __iter__(self) -> Iterator[str]:
        """Iterate the request_metadata keys.

        Without this, iterating a context asked it for ``context[0]``.
        """
        return iter(self.request_metadata)

    def __len__(self) -> int:
        """How many request_metadata entries this context carries."""
        return len(self.request_metadata)

    def __bool__(self) -> bool:
        """A context always exists, whatever its metadata holds.

        Defined rather than left to ``__len__``: a context is not its request
        metadata, and every ``if context:`` written before ``__len__`` existed
        means "was a context supplied", not "does it carry metadata". Falling
        through to ``__len__`` would have silently reversed all of them.
        """
        return True

    def keys(self) -> KeysView[str]:
        """The request_metadata keys. Also what makes ``dict(context)`` work."""
        return self.request_metadata.keys()

    def values(self) -> ValuesView[Any]:
        """The request_metadata values."""
        return self.request_metadata.values()

    def items(self) -> ItemsView[str, Any]:
        """The request_metadata items."""
        return self.request_metadata.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from request_metadata with optional default.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value from request_metadata or default
        """
        return self.request_metadata.get(key, default)

    def copy(self, **overrides: Any) -> "BotContext":
        """Create a copy of this context with optional field overrides.

        Not a mapping's ``copy``: it clones the whole context, not the
        request metadata, which is one of the reasons this class does not
        claim the :class:`~collections.abc.Mapping` ABC.

        Creates shallow copies of session_metadata and request_metadata dicts
        to avoid mutation issues between the original and copy.

        Args:
            **overrides: Field values to override in the copy

        Returns:
            New BotContext instance with copied values

        Example:
            >>> ctx = BotContext(conversation_id="conv-1", client_id="client-1")
            >>> ctx2 = ctx.copy(conversation_id="conv-2")
            >>> ctx2.conversation_id
            'conv-2'
        """
        return BotContext(
            conversation_id=overrides.get("conversation_id", self.conversation_id),
            client_id=overrides.get("client_id", self.client_id),
            user_id=overrides.get("user_id", self.user_id),
            session_metadata=overrides.get("session_metadata", dict(self.session_metadata)),
            request_metadata=overrides.get("request_metadata", dict(self.request_metadata)),
        )

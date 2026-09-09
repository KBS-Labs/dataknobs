"""The four protocols a placement is written against.

Two pairs, twinned. A cascade over a vocabulary someone typed is dictionary
lookups all the way down, so the **synchronous** flavour is the native one
here and the asynchronous is the accommodation -- the reverse of the usual
case, and the reason both are declared rather than one being a wrapper over
the other.

All four carry ``@runtime_checkable``, and for this family that is
load-bearing rather than conventional: two of them are a
:class:`~dataknobs_common.registry.PluginRegistry`'s ``validate_type``, and a
registry refuses a Protocol without the decorator at registration -- because
``isinstance`` refuses such a base at creation too. The decorator is a
property of the protocol; a registry cannot add it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataknobs_common.entity_resolution.values import (
        EntityCandidate,
        ResolutionResult,
        Within,
    )

__all__ = [
    "AsyncEntityResolver",
    "AsyncMatchSignal",
    "EntityResolver",
    "MatchSignal",
]


@runtime_checkable
class MatchSignal(Protocol):
    """One rung of a cascade: a way of proposing entities for a query."""

    @property
    def name(self) -> str:
        """The key this rung is registered under.

        The same string a consumer writes as ``kind:``, and the same string
        the evidence this rung produces carries as its ``signal`` -- so a
        caller reading ``evidence.signal`` can correlate a hit back to the
        rung they configured.
        """
        ...

    def narrows(self) -> bool:
        """Whether this rung can honour a filter.

        A rung that cannot must say so rather than accepting a scope and
        ignoring it, which returns candidates from outside it while looking
        like it worked.
        """
        ...

    def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]: ...

    def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]:
        """The batch form, for a corpus rather than a turn.

        A rung with no real batch path loops :meth:`candidates`; one that can
        ask its backing a single question for many queries overrides this.
        """
        ...


@runtime_checkable
class AsyncMatchSignal(Protocol):
    """:class:`MatchSignal` for a rung that reaches for data.

    ``name`` stays a property and ``narrows()`` a plain ``def``: neither
    touches a backing, so making them awaitable would cost every caller an
    ``await`` and buy nothing. Those two are also what makes the twins
    separable in a registry -- the guard that tells the flavours apart skips a
    property, agrees on ``narrows()``, and separates the pair on the two
    ``candidates`` members.
    """

    @property
    def name(self) -> str: ...

    def narrows(self) -> bool: ...

    async def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]: ...

    async def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]: ...


@runtime_checkable
class EntityResolver(Protocol):
    """Turn a string into ranked entities, with the reason each won."""

    def resolve(self, name: str, *, k: int = 5, within: Within = None) -> ResolutionResult: ...

    def resolve_many(
        self, names: Sequence[str], *, k: int = 5, within: Within = None
    ) -> list[ResolutionResult]:
        """The bulk form, for a corpus rather than a turn."""
        ...


@runtime_checkable
class AsyncEntityResolver(Protocol):
    """:class:`EntityResolver` for a cascade whose rungs reach for data."""

    async def resolve(
        self, name: str, *, k: int = 5, within: Within = None
    ) -> ResolutionResult: ...

    async def resolve_many(
        self, names: Sequence[str], *, k: int = 5, within: Within = None
    ) -> list[ResolutionResult]: ...

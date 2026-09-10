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
    from collections.abc import Collection, Mapping, Sequence

    from dataknobs_common.entity_resolution.values import (
        EntityCandidate,
        ResolutionResult,
        Within,
    )
    from dataknobs_common.ontology.model import Entity

__all__ = [
    "AliasFormSource",
    "AsyncAliasFormSource",
    "AsyncEntityResolver",
    "AsyncMatchSignal",
    "EntityResolver",
    "MatchSignal",
    "MembershipOracle",
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

        A rung that cannot must say so rather than silently ignoring a scope
        it was handed -- it is offered a filter only if it answers ``True``.

        **Answering ``True`` is a promise in one direction only.** A rung that
        narrows may **over-admit freely and must never under-admit**: a
        superset filter, not a second reading of the scope. The cascade rules
        on every candidate a rung produces and can only *remove*, so returning
        too much is corrected and returning too little is not -- a candidate
        the rung withholds is one nothing downstream can recover. When in
        doubt, return it and let the cascade decide.

        Narrowing is therefore an optimisation: it keeps a rung's ``k`` meaning
        ``k`` under a scope, rather than handing over ``k`` unscoped candidates
        the cascade then thins.
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
        ask its backing a single question for many queries answers it directly.

        **This protocol carries no implementation of that loop** -- a Protocol
        member is a shape, so a rung written against this alone must write the
        loop itself.
        :class:`~dataknobs_common.entity_resolution.DeclaredSignal` is where the
        loop actually lives, along with the constructor, ``name``,
        ``narrows()`` and the rung-side narrowing; subclassing it is the
        shorter path to a correct rung, and this protocol is the escape hatch
        for a rung it cannot serve.
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


@runtime_checkable
class MembershipOracle(Protocol):
    """A source that answers what an entity **is**, per scope axis.

    The library derives membership from ``Entity.type`` -- see
    :func:`~dataknobs_common.entity_resolution.values.within_memberships`,
    which is the published default and stays the answer for a source that does
    not satisfy this. Satisfying it is how a source that knows more than a type
    about its entities -- a state, a tenant, a jurisdiction -- says so, and
    thereby makes a multi-axis ``within`` scope mean something.

    **A separate protocol rather than a member on**
    :class:`~dataknobs_common.ontology.sources.EntitySource`. That protocol is
    ``@runtime_checkable`` and consumers satisfy it structurally, so widening
    it would leave ``isinstance`` passing against implementations we never see
    while every call carrying the new member raised ``AttributeError``. A
    source that does not satisfy *this* protocol is simply never asked, so
    nothing migrates and no existing source changes.

    **One protocol, not a twin pair**, and deliberately so: like
    ``EntitySource.describe``, this reads an entity the caller already holds
    and touches no backing. An awaitable form would cost every caller an
    ``await`` and buy nothing, so the asynchronous cascade consults this same
    synchronous member.
    """

    def memberships(self, entity: Entity) -> Mapping[str, str]:
        """What this entity is, on every axis this source can answer for.

        Returns:
            Axis name to the one id the entity has on that axis. An axis left
            out is an axis the entity declares nothing on, which
            :func:`~dataknobs_common.entity_resolution.values.within_admits`
            reads as *excluded* rather than *unconstrained*.
        """
        ...

    def axes(self) -> Collection[str]:
        """Every axis name this source can answer on -- **the legal set**.

        A scope naming an axis outside this is refused at the cascade's
        boundary rather than answered. It has to be, because
        :meth:`memberships` reads a missing axis as *excluded*: a mistyped
        axis name matched nothing and returned an empty result, which is what
        a correctly spelled scope over an empty vocabulary returns too. The
        caller cannot tell those apart, and the reading they will reach for is
        the one that is not their fault.

        This is what
        :attr:`~dataknobs_common.ontology.sources.SourceDescription.declares`
        was described as being and is not: ``declares`` holds the type ids a
        source carries, which is one axis's *values* rather than the set of
        axis names.

        A source that does not satisfy this protocol publishes exactly
        ``{ENTITY_TYPE_KEY}``, which is what the bare ``within`` spellings
        mean and is why they can never be refused.

        Returns:
            The axis names, including ``ENTITY_TYPE_KEY`` where
            :meth:`memberships` still answers on it -- this is the whole set,
            not the additions.
        """
        ...


@runtime_checkable
class AliasFormSource(Protocol):
    """A source that can report which of its entities declare an **alias** form.

    Optional, and separate for the reason
    :class:`~dataknobs_common.ontology.sources.EntitySource` gives for not
    growing members: that protocol is ``@runtime_checkable`` and consumers
    satisfy it structurally, so a member added to it turns every
    implementation we never see from conforming into non-conforming, silently
    and at once. A member on a *separate* protocol costs those implementations
    nothing -- a source that does not satisfy this is simply not asked.

    **A default body would not have rescued them.** A Protocol's method body
    runs for an explicit subclass and for nothing else: ``isinstance`` and
    ``hasattr`` both stay ``False`` for a structural conformer, so a default
    fixes the type checker's complaint while leaving the runtime break exactly
    where it was -- half a fix that reads like a whole one.

    :class:`~dataknobs_common.entity_resolution.AliasSignal` checks for this
    and falls back to *declares no aliases*, so a source without it yields an
    empty rung rather than an ``AttributeError``.
    """

    def by_alias_form(self, form: str) -> frozenset[str]: ...


@runtime_checkable
class AsyncAliasFormSource(Protocol):
    """:class:`AliasFormSource` for a source that reaches for data.

    ``isinstance`` cannot tell this from its twin -- a runtime-checkable
    protocol compares member *names*, and both spell it ``by_alias_form`` --
    which costs nothing here: each flavour's rung holds a source of its own
    flavour already, and the check it makes is *does this source answer for
    alias forms at all*. The pair exists for the type checker, which does
    distinguish them, and for a reader looking for the asynchronous spelling.
    """

    async def by_alias_form(self, form: str) -> frozenset[str]: ...

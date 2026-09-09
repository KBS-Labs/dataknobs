"""The rungs an authored vocabulary can run with no dependency at all.

Two matchers, twinned. Both read an entity source and nothing else -- no
store, no embedder, no network -- which is what makes them the default a
cascade falls back to when a consumer configures nothing.

They differ in **which index they ask**, and that difference is the whole
reason the source publishes two members. ``ExactNormalizedSignal`` asks for
any form: an id, a name, or an alias, folded together, which is what an exact
lookup wants. ``AliasSignal`` asks for alias forms specifically, because the
folded map cannot report which of the three matched -- so a rung that
reconstructed the distinction by fetching each hit and re-folding its aliases
would be deciding, in the matcher, something the vocabulary already declares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataknobs_common.entity_resolution.values import (
    EntityCandidate,
    EvidenceKind,
    MatchEvidence,
    Scoring,
    within_admits,
    within_axes,
    within_memberships,
)
from dataknobs_common.text import default_normalizer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dataknobs_common.ontology.sources import AsyncEntitySource, EntitySource

__all__ = [
    "AliasSignal",
    "AsyncAliasSignal",
    "AsyncDeclaredSignal",
    "AsyncExactNormalizedSignal",
    "DeclaredSignal",
    "ExactNormalizedSignal",
]

#: A declared hit's score. 1.0 by fiat and carrying no information, which is
#: what ``Scoring.DECLARED`` exists to say: it is not a measurement, and a
#: caller must not average it against a cosine.
_DECLARED_SCORE = 1.0


def _candidate(entity_id: str, signal: str, query: str) -> EntityCandidate:
    """One declared hit, with the one piece of evidence its rung can give.

    ``span`` is ``None`` because these rungs match the **whole** query string
    rather than locating a mention inside it. A span is an offset *into* the
    query, and reporting ``(0, len(query))`` would be indistinguishable from a
    scan that found the whole string -- a different rung, with a gate of its
    own, that this one must not impersonate.
    """
    return EntityCandidate(
        entity_id=entity_id,
        score=_DECLARED_SCORE,
        evidence=(
            MatchEvidence(
                signal=signal,
                kind=EvidenceKind.DECLARED,
                score=_DECLARED_SCORE,
                scoring=Scoring.DECLARED,
                matched_text=query,
                span=None,
            ),
        ),
    )


def _ordered(hits: frozenset[str], k: int) -> list[str]:
    """At most ``k`` ids, in a stable order.

    The source answers with a frozenset, deliberately: two entities may
    legitimately share a form, and that ambiguity is the vocabulary's to
    declare. Iteration order over a set is not stable across processes, so a
    rung that passed one straight through would return the *same* candidates
    in a different order on a different run -- and the cascade positions by
    arrival. Sorting makes the ambiguity survive as an ambiguity rather than
    as an arbitrary winner that moves.
    """
    return sorted(hits)[:k]


class DeclaredSignal:
    """What every synchronous rung shares: everything but the index it asks.

    The two shipped rungs differ in one expression. Writing that as two classes
    with two copies of the constructor, the name, the filter handling and the
    candidate construction is how a fix to one of them stops being a fix to the
    other -- and the same argument reaches a consumer's rung, which is why this
    is published rather than private.

    **Subclass this to write a rung.** Set :attr:`key` and implement
    :meth:`_hits`; everything else -- the constructor, ``name``, ``narrows()``,
    the rung-side narrowing and the batch loop -- comes with it. In particular
    the narrowing comes with it *correct*: it is a superset filter, and
    :meth:`~dataknobs_common.entity_resolution.MatchSignal.narrows` explains
    why getting that direction wrong is the one mistake nothing downstream can
    recover. A rung written against the bare
    :class:`~dataknobs_common.entity_resolution.MatchSignal` protocol gets no
    such help, so prefer this base and keep the protocol for the case it cannot
    serve -- a rung whose backing is not a dictionary lookup, or one that
    already has a superclass.
    """

    #: The registered key, and the string the evidence carries as its
    #: ``signal`` -- one value, so a caller reading ``evidence.signal`` can
    #: correlate a hit back to the ``kind:`` they configured.
    key = ""

    def __init__(
        self,
        entities: EntitySource,
        *,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        """Args:
        entities: The vocabulary to match against.
        normalizer: How the query is folded before lookup. Defaults to
            strip-and-casefold. A source folds the forms it indexed with a
            normalizer of its own; handing a different one here is how a
            caller matches more loosely than the index was built for.
        """
        self._entities = entities
        self._normalizer = normalizer or default_normalizer

    @property
    def name(self) -> str:
        """The key this rung is registered under."""
        return self.key

    def narrows(self) -> bool:
        """True: a filter is honoured against the source's declared types."""
        return True

    def _hits(self, query: str) -> frozenset[str]:
        """The ids this rung proposes for an already-folded query -- **the hook**.

        The one member a subclass supplies. It stays underscored although the
        class is published: it is called by :meth:`candidates` and never by a
        consumer of the rung, so it is the extension point rather than part of
        the rung's surface.

        Args:
            query: Folded by this rung's normalizer already. Do not fold again.

        Returns:
            Every id that matches. Ordering and the cut to ``k`` are
            :meth:`candidates`'s job, and the scope is the cascade's -- return
            what matches and let the layers above decide.
        """
        raise NotImplementedError

    def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]:
        """At most ``k`` entities whose forms match this query."""
        hits = _admitted(self._entities, self._hits(self._normalizer(query)), filter)
        return [_candidate(entity_id, self.key, query) for entity_id in _ordered(hits, k)]

    def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]:
        """One answer per query, in the order asked.

        A loop, because the backing is a dictionary: there is no round trip to
        save, so a batch path here would be the same work behind a second
        spelling of it.
        """
        return [self.candidates(query, k, filter=filter) for query in queries]


class ExactNormalizedSignal(DeclaredSignal):
    """Match a folded query against any form the vocabulary carries."""

    key = "exact"

    def _hits(self, query: str) -> frozenset[str]:
        return self._entities.by_surface_form(query)


class AliasSignal(DeclaredSignal):
    """Match a folded query against **alias** forms specifically."""

    key = "alias"

    def _hits(self, query: str) -> frozenset[str]:
        return self._entities.by_alias_form(query)


class AsyncDeclaredSignal:
    """The asynchronous twin's shared half -- **subclass this** for an async rung.

    :class:`DeclaredSignal`'s argument applies unchanged: set :attr:`key`,
    implement ``_hits``, and the constructor, ``name``, ``narrows()``, the
    superset-filter narrowing and the batch loop come with it.

    ``name`` and ``narrows()`` stay synchronous: neither reaches for data, and
    making them awaitable would cost every caller an ``await`` for nothing.
    They are also what lets a registry tell the twins apart -- the guard that
    separates the flavours skips a property, agrees on a plain ``def``, and
    separates the pair on the two ``candidates`` members.
    """

    key = ""

    def __init__(
        self,
        entities: AsyncEntitySource,
        *,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        """Args:
        entities: The vocabulary to match against.
        normalizer: How the query is folded before lookup.
        """
        self._entities = entities
        self._normalizer = normalizer or default_normalizer

    @property
    def name(self) -> str:
        """The key this rung is registered under."""
        return self.key

    def narrows(self) -> bool:
        """True: a filter is honoured against the source's declared types."""
        return True

    async def _hits(self, query: str) -> frozenset[str]:
        raise NotImplementedError

    async def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]:
        """At most ``k`` entities whose forms match this query."""
        hits = await _async_admitted(
            self._entities, await self._hits(self._normalizer(query)), filter
        )
        return [_candidate(entity_id, self.key, query) for entity_id in _ordered(hits, k)]

    async def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]:
        """One answer per query, in the order asked."""
        return [await self.candidates(query, k, filter=filter) for query in queries]


class AsyncExactNormalizedSignal(AsyncDeclaredSignal):
    """:class:`ExactNormalizedSignal` over an asynchronous source."""

    key = "exact"

    async def _hits(self, query: str) -> frozenset[str]:
        return await self._entities.by_surface_form(query)


class AsyncAliasSignal(AsyncDeclaredSignal):
    """:class:`AliasSignal` over an asynchronous source."""

    key = "alias"

    async def _hits(self, query: str) -> frozenset[str]:
        return await self._entities.by_alias_form(query)


def _admitted(
    entities: EntitySource, hits: frozenset[str], filter: dict[str, Any] | None
) -> frozenset[str]:
    """The hits a filter admits -- a rung-side **narrowing**, not the ruling.

    The cascade decides what a scope admits, against its own source. This runs
    first, inside the rung, for one reason: a rung asked for ``k`` that returns
    ``k`` unscoped candidates hands the cascade a batch it will then thin, and
    the query comes back short of ``k`` when the vocabulary could have filled
    it. Narrowing here is what makes the rung's ``k`` mean ``k``.

    So it is an optimisation -- but **only in one direction**. It may
    over-admit freely and must never under-admit: a **superset filter**, not a
    second reading of the scope. ``_admits`` filters what a rung *produced*, so
    it can only remove; a rung that admits too much is overruled, while a rung
    that admits too little never hands the candidate over and nothing
    downstream recovers it.

    **The general form, since it is not specific to this pair:** a downstream
    filter cannot recover what an upstream one removed, so in any two-stage
    filter the upstream stage must be a superset filter. The asymmetry holds
    even when both stages call the same function -- which is why the rule
    itself is
    :func:`~dataknobs_common.entity_resolution.values.within_admits` over
    :func:`~dataknobs_common.entity_resolution.values.within_memberships`, the
    same two published functions the cascade applies, rather than a projection
    that agrees with the cascade's today.
    """
    axes = within_axes(filter)
    if not axes:
        return hits
    found = entities.get_many(sorted(hits))
    return frozenset(
        entity_id
        for entity_id, entity in found.items()
        if within_admits(axes, within_memberships(entity, entities))
    )


async def _async_admitted(
    entities: AsyncEntitySource, hits: frozenset[str], filter: dict[str, Any] | None
) -> frozenset[str]:
    """:func:`_admitted` over an asynchronous source."""
    axes = within_axes(filter)
    if not axes:
        return hits
    found = await entities.get_many(sorted(hits))
    return frozenset(
        entity_id
        for entity_id, entity in found.items()
        if within_admits(axes, within_memberships(entity, entities))
    )

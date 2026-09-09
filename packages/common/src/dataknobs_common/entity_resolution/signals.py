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
)
from dataknobs_common.text import default_normalizer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dataknobs_common.ontology.sources import AsyncEntitySource, EntitySource

__all__ = [
    "AliasSignal",
    "AsyncAliasSignal",
    "AsyncExactNormalizedSignal",
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


class _DeclaredSignal:
    """What both synchronous rungs share: everything but the index they ask.

    The two differ in one expression. Writing that as two classes with two
    copies of the constructor, the name, the filter handling and the candidate
    construction is how a fix to one of them stops being a fix to the other.
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


class ExactNormalizedSignal(_DeclaredSignal):
    """Match a folded query against any form the vocabulary carries."""

    key = "exact"

    def _hits(self, query: str) -> frozenset[str]:
        return self._entities.by_surface_form(query)


class AliasSignal(_DeclaredSignal):
    """Match a folded query against **alias** forms specifically."""

    key = "alias"

    def _hits(self, query: str) -> frozenset[str]:
        return self._entities.by_alias_form(query)


class _AsyncDeclaredSignal:
    """The asynchronous twin's shared half.

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


class AsyncExactNormalizedSignal(_AsyncDeclaredSignal):
    """:class:`ExactNormalizedSignal` over an asynchronous source."""

    key = "exact"

    async def _hits(self, query: str) -> frozenset[str]:
        return await self._entities.by_surface_form(query)


class AsyncAliasSignal(_AsyncDeclaredSignal):
    """:class:`AliasSignal` over an asynchronous source."""

    key = "alias"

    async def _hits(self, query: str) -> frozenset[str]:
        return await self._entities.by_alias_form(query)


def _admitted(
    entities: EntitySource, hits: frozenset[str], filter: dict[str, Any] | None
) -> frozenset[str]:
    """The hits a filter admits.

    **Union within one axis, conjunction across them** -- a filter naming two
    axes admits only what is in both, which is the only reading under which a
    consumer scoped to a kind *and* a state can say so.

    On this path an axis's values are entity types, because
    ``describe().declares`` is what an entity source publishes as its scope
    terms and it names types. An axis value naming nothing declared admits
    nothing, rather than being ignored: a mis-keyed filter that silently
    matched everything is the failure this refuses to have.
    """
    if not filter:
        return hits
    for values in filter.values():
        admitted: set[str] = set()
        for value in _as_values(values):
            admitted |= entities.by_type(value)
        hits = hits & admitted
    return hits


async def _async_admitted(
    entities: AsyncEntitySource, hits: frozenset[str], filter: dict[str, Any] | None
) -> frozenset[str]:
    """:func:`_admitted` over an asynchronous source."""
    if not filter:
        return hits
    for values in filter.values():
        admitted: set[str] = set()
        for value in _as_values(values):
            admitted |= await entities.by_type(value)
        hits = hits & admitted
    return hits


def _as_values(values: Any) -> tuple[str, ...]:
    """One axis's values, whether it was given one or several."""
    if isinstance(values, str):
        return (values,)
    return tuple(values)

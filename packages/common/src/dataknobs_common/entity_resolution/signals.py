# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The rungs an authored vocabulary can run with no dependency at all.

Three matchers, twinned. All of them read an entity source and nothing else --
no store, no embedder, no network -- which is what makes them the default a
cascade falls back to when a consumer configures nothing.

Two of them differ in **which index they ask**, and that difference is the
whole reason the source publishes two *index* members. ``ExactNormalizedSignal`` asks
for any form: an id, a name, or an alias, folded together, which is what an
exact lookup wants. ``AliasSignal`` asks for alias forms specifically, because
the folded map cannot report which of the three matched -- so a rung that
reconstructed the distinction by fetching each hit and re-folding its aliases
would be deciding, in the matcher, something the vocabulary already declares.

``ScanningSignal`` asks the same index as the first of those and differs in
**how much of the query it hands over**: every span of consecutive tokens,
rather than the whole string once. That is a difference in what the rung can
*find*, not merely in what it reports, and neither direction subsumes the
other -- see the class for the measurement, which is why it is a third kind
and not the first one configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataknobs_common.entity_resolution.protocols import AliasFormSource, AsyncAliasFormSource
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.entity_resolution.values import (
    EntityCandidate,
    EvidenceKind,
    FormHit,
    MatchEvidence,
    Scoring,
    within_admits,
    within_axes,
    within_memberships,
)
from dataknobs_common.text import content_span, token_spans

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from dataknobs_common.ontology.sources import AsyncEntitySource, EntitySource

__all__ = [
    "AliasSignal",
    "AsyncAliasSignal",
    "AsyncDeclaredSignal",
    "AsyncExactNormalizedSignal",
    "AsyncScanningSignal",
    "DeclaredSignal",
    "ExactNormalizedSignal",
    "ScanningSignal",
]

#: A declared hit's score. 1.0 by fiat and carrying no information, which is
#: what ``Scoring.DECLARED`` exists to say: it is not a measurement, and a
#: caller must not average it against a cosine.
_DECLARED_SCORE = 1.0


def _candidate(
    entity_id: str, signal: str, query: str, found: Sequence[FormHit]
) -> EntityCandidate:
    """One declared entity, with one piece of evidence per place it was found.

    **Every declared hit carries a span**, including the whole-string one.
    That was once refused here on the grounds that reporting the query's own
    extent would be indistinguishable from a scan that found the whole
    string -- true, and not an objection: being indistinguishable *there* is
    the point. A caller who already knows the phrase and one who hands over a
    whole sentence read the same field and need not know which rung answered.

    That is as far as the claim goes, and this docstring used to take it
    further: it said a scan **subsumes** whole-string matching. It does not,
    and :func:`~dataknobs_common.text.token_spans` is why -- see
    :class:`ScanningSignal` for the measured cases each rung reaches and the
    other does not.

    ``matched_text`` is the slice rather than the query, which is the same
    sentence read the other way: it is what the span points at, so the two
    fields agree by construction instead of by a caller's trust.
    """
    return EntityCandidate(
        entity_id=entity_id,
        score=_DECLARED_SCORE,
        evidence=tuple(
            MatchEvidence(
                signal=signal,
                kind=EvidenceKind.DECLARED,
                score=_DECLARED_SCORE,
                scoring=Scoring.DECLARED,
                matched_text=query[hit.span[0] : hit.span[1]],
                span=hit.span,
            )
            for hit in found
        ),
    )


def _folded(query: str, normalizer: Callable[[str], str] | None) -> str:
    """The query as the rung will ask it -- folded only if one was handed over.

    **No normalizer means no fold here.** The source folds every form it
    indexed and folds a lookup on the way in, so a rung that folded as well
    ran a second, different fold over an answer the vocabulary had already
    decided. That was invisible while both folds agreed -- which they do for
    any vocabulary spelled in lower case -- and unreachable the moment they
    did not: an index built with ``str.strip`` holds ``Beagles``, a rung
    pre-folding to ``beagles`` could never reach it, and no spelling of the
    query could.

    Handing one is therefore the opt-in this parameter always described: a
    caller who wants to match *differently* from the way the index was built
    prepends their own fold, and gets exactly one more fold than the source
    performs.
    """
    return query if normalizer is None else normalizer(query)


def _grouped(
    found: Sequence[FormHit],
    admitted: frozenset[str],
    k: int,
    *,
    signal: str,
    query: str,
) -> list[EntityCandidate]:
    """One candidate per admitted id, in the rung's own order, cut to ``k``.

    **``k`` counts entities, not hits.** A query naming one entity twice is
    one candidate carrying two pieces of evidence, which is what the cascade
    does with two *rungs* producing the same id -- so a rung that produced it
    twice itself answers the same shape rather than spending two of the
    caller's ``k`` on one entity.

    The order is whatever :meth:`DeclaredSignal._located` returned, preserved
    by the insertion order of the mapping below. That is the rung's to set:
    a declared score is ``1.0`` by fiat and carries none, so if the order is
    to mean anything -- the longer form before the shorter one it contains --
    the rung is the only layer that can say so.
    """
    hits: dict[str, list[FormHit]] = {}
    for hit in found:
        if hit.entity_id in admitted:
            hits.setdefault(hit.entity_id, []).append(hit)
    return [_candidate(entity_id, signal, query, at) for entity_id, at in list(hits.items())[:k]]


def _whole_string(query: str, ids: Sequence[str]) -> tuple[FormHit, ...]:
    """Every id, located at the extent of ``query`` the fold kept.

    What a rung that compared the **whole** query against the index reports.
    The span is :func:`~dataknobs_common.text.content_span`'s rather than
    ``(0, len(query))`` because the fold strips, so a query of ``"  beagle  "``
    matched ``beagle`` and did not match two spaces.
    """
    span = content_span(query)
    return tuple(FormHit(entity_id=entity_id, span=span) for entity_id in ids)


def _probe_spans(query: str, longest: int | None) -> Iterator[tuple[int, int]]:
    """Spans of up to ``longest`` consecutive tokens in ``query``, longest first.

    The enumeration a scan runs, and ``longest`` is what keeps it affordable.
    Unbounded it is *n(n+1)/2* spans for *n* tokens -- 21 for a six-token
    utterance, but 500,500 for a thousand-token paste, with each probe slicing
    a span that may be the whole query. Quadratic in the token count, and the
    characters copied grow with the cube of it, on text a caller hands over.

    **A derived bound costs no answer**, which is what makes it an arithmetic
    fix rather than a budget -- and ``longest`` is only derived when the
    vocabulary could support it. It is how many tokens the longest form the
    vocabulary *declares* occupies -- see
    :meth:`~dataknobs_common.ontology.EntitySource.longest_form_tokens` -- and
    a window wider than that is one no declared form could fill, so every
    probe removed is one that would have answered the empty set. What remains
    is *n x longest* spans: linear in the caller's input, with the constant
    set by the vocabulary rather than by the query.

    **``None`` means no bound could be derived, and then every window is
    enumerated.** A source whose fold merges token boundaries reports it, and
    a rung that folds for itself does not ask -- in both cases the implication
    the bound rests on has failed, and there is no wider number to fall back
    to. Quadratic and complete beats linear and lossy: the alternative is a
    declared form the scan silently stops finding.
    ``max_window`` on :class:`ScanningSignal` is where a caller who knows
    their own queries puts a number the vocabulary could not supply.

    **Longest first** because the rung publishes its own order and this is
    where the *enumeration*'s order is decided: the cascade positions by
    arrival, and a form containing another should be proposed before the form
    it contains. Ordering the ids found *within* one span is a different
    question and :meth:`DeclaredSignal._order`'s.

    Shared by both flavours rather than written twice, for
    :func:`_whole_string`'s reason one rung along: the arithmetic here is the
    half of a scan that has nothing to do with awaiting, so two copies of it
    could drift in a direction no twin-parity check over signatures would see.
    """
    tokens = token_spans(query)
    widest = len(tokens) if longest is None else min(longest, len(tokens))
    return (
        (tokens[first][0], tokens[first + length - 1][1])
        for length in range(widest, 0, -1)
        for first in range(len(tokens) - length + 1)
    )


def _window_bound(
    entities: EntitySource | AsyncEntitySource,
    normalizer: Callable[[str], str] | None,
    max_window: int | None,
) -> int | None:
    """How many tokens a probe may span, or ``None`` to enumerate every window.

    **Shared by both flavours**, for the reason :func:`_probe_spans` is: the
    three-way decision below has nothing to do with awaiting, and two copies
    of it could disagree in a direction no twin-parity check over signatures
    would see. The twins answering identically is the property, and one
    function is how it is held rather than asserted.

    The source's number describes the source's own fold. A rung handed a
    ``normalizer`` folds again before the source does, so the window that
    reaches a key may carry more tokens than the key does, and the source's
    number was never an answer to that question -- it measured its keys before
    this rung existed and cannot see the extra fold. No source-side
    measurement can, so such a rung does not ask.

    ``max_window`` caps whatever that leaves, including nothing. It is the one
    number here a caller chose rather than the vocabulary implied, so it is
    the only one that can cost an answer -- see :class:`ScanningSignal`.
    """
    derived = None if normalizer is not None else entities.longest_form_tokens()
    if max_window is None:
        return derived
    return max_window if derived is None else min(derived, max_window)


def _checked_max_window(max_window: int | None) -> int | None:
    """``max_window`` as given, having refused a value that means no scan.

    Shared so the twins refuse identically. Zero and negatives are refused
    rather than clamped: each would build a rung that probes nothing and
    reports no candidates, which is indistinguishable from a vocabulary that
    matches nothing and is the kind of silence this family refuses to ship.
    A caller who wants a rung that never fires leaves it out of the cascade.

    Raises:
        ValidationError: For a ``max_window`` below one.
    """
    if max_window is not None and max_window < 1:
        raise ValidationError(
            f"max_window must be at least 1, got {max_window}. A scan with a "
            f"window narrower than one token probes nothing and would report "
            f"an empty result for every query."
        )
    return max_window


def _hits_at(span: tuple[int, int], ids: Sequence[str]) -> list[FormHit]:
    """One :class:`FormHit` per id, all at the same span, in the order given.

    The other half of a probe, extracted for :func:`_probe_spans`'s reason:
    what is left in each flavour's ``_located`` is then the loop and the
    ``await``, and nothing a twin-parity check over signatures could not see.

    The order is the caller's -- :meth:`DeclaredSignal._order`'s answer, not a
    ``sorted`` spelled here. A rung that overrides that hook is overriding it
    for the scanning path too, which is what the hook's own docstring promises
    a rung "with a longer form and a shorter one inside it".
    """
    return [FormHit(entity_id=entity_id, span=span) for entity_id in ids]


class DeclaredSignal:
    """What every synchronous rung shares: everything but the index it asks.

    The two whole-string rungs differ in one expression, and the scanning one
    in one method. Writing those as three classes with three copies of the
    constructor, the name, the filter handling and the candidate construction
    is how a fix to one of them stops being a fix to the others -- and the same
    argument reaches a consumer's rung, which is why this is published rather
    than private.

    **Subclass this to write a rung.** Set :attr:`key` and implement
    :meth:`_hits`; everything else -- the constructor, ``name``, ``narrows()``,
    the rung-side narrowing and the batch loop -- comes with it. A rung that
    *locates* a form inside the query rather than comparing the whole of it
    overrides :meth:`_located` instead, as :class:`ScanningSignal` does, and a
    rung that wants its own order over :meth:`_hits`'s answer overrides
    :meth:`_order`. In particular the narrowing comes with it *correct*: it is a superset filter, and
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
        normalizer: An **extra** fold, applied before the source's own.
            ``None``, the default, means this rung does not fold at all and
            the source's fold is the only one -- see :func:`_folded` for why
            a second default fold could not be right. Handing one is how a
            caller matches differently from the way the index was built.
        """
        self._entities = entities
        self._normalizer = normalizer

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
            query: The query, folded by this rung's own normalizer if it was
                given one and otherwise untouched. It is **not** folded to
                the index's spelling -- a source that folds does that when
                asked, and a subclass reading a backing that does not fold is
                the one that must.

        Returns:
            Every id that matches. Ordering and the cut to ``k`` are
            :meth:`candidates`'s job, and the scope is the cascade's -- return
            what matches and let the layers above decide.
        """
        raise NotImplementedError

    def _fold(self, form: str) -> str:
        """This rung's own extra fold, or the form untouched.

        Published to subclasses because a scanning rung folds the *slices* it
        probes rather than the query, so it cannot reach the fold through
        :meth:`_located`'s caller. See :func:`_folded` for why the default is
        no fold at all.
        """
        return _folded(form, self._normalizer)

    def _order(self, hits: frozenset[str]) -> Sequence[str]:
        """The order :meth:`_hits`'s answer is proposed in -- **overridable**.

        The source answers with a frozenset, deliberately: two entities may
        legitimately share a form, and that ambiguity is the vocabulary's to
        declare. Iteration order over a set is not stable across processes, so
        a rung that passed one straight through would return the *same*
        candidates in a different order on a different run -- and the cascade
        positions by arrival. Sorting makes the ambiguity survive as an
        ambiguity rather than as an arbitrary winner that moves.

        It is a hook rather than a constant because a declared score is 1.0 by
        fiat and carries no order at all, so any order that is to *mean*
        something has to be published by the rung. Alphabetical means nothing
        beyond stability, and is the right default for a rung whose hits are
        all one form; a rung with a longer form and a shorter one inside it
        has something to say and says it here.
        """
        return sorted(hits)

    def _located(self, query: str) -> Sequence[FormHit]:
        """Where this rung's matches sit in the query -- **the span hook**.

        The default compares the **whole** query against the index and reports
        every hit at the extent the fold kept, which is what
        :meth:`_hits`-shaped rungs mean. Override this instead of
        :meth:`_hits` to scan: return one :class:`FormHit` per place a
        declared form was found, in the order the rung wants them proposed,
        and overlapping forms as the several hits they are.

        :meth:`_hits` remains the hook for the whole-string case rather than
        being folded into this one. It returns a ``frozenset[str]``, which has
        nowhere to put an offset -- so a scanning rung could never have been
        written through it, and the two shipped rungs would gain nothing from
        being made to answer in spans they compute identically.
        """
        return _whole_string(query, self._order(self._hits(self._fold(query))))

    def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]:
        """At most ``k`` entities whose forms match this query."""
        found = self._located(query)
        admitted = _admitted(self._entities, frozenset(hit.entity_id for hit in found), filter)
        return _grouped(found, admitted, k, signal=self.key, query=query)

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
    """Match a query against **alias** forms specifically.

    Reporting alias forms is an optional capability -- see
    :class:`~dataknobs_common.entity_resolution.AliasFormSource` for why it is
    a protocol of its own rather than a member every entity source owes. A
    source that does not answer for them yields a rung that matches nothing,
    which is what *this vocabulary declares no aliases* means, rather than an
    ``AttributeError`` from inside a cascade.
    """

    key = "alias"

    def _hits(self, query: str) -> frozenset[str]:
        if not isinstance(self._entities, AliasFormSource):
            return frozenset()
        return self._entities.by_alias_form(query)


class ScanningSignal(DeclaredSignal):
    """Find declared forms **inside** the query, and report where each one sat.

    Every span of consecutive tokens is looked up as the slice it covers, so
    the form reaches the index carrying whatever separated its tokens and
    matches whichever spelling the vocabulary declared -- ``golden retriever``
    and ``golden_retriever`` both, if that is how they were written. At most
    twenty-one lookups for a six-token utterance, over a dictionary, and fewer
    than that unless the vocabulary declares a six-token form: the enumeration
    is cut at the longest form there is, which is what keeps the cost linear in
    the query rather than quadratic in it. See
    :func:`_probe_spans` for the enumeration and its bound, and
    :func:`~dataknobs_common.text.token_spans` for the boundary policy that
    stops it finding ``beagle`` inside ``unbeagleable``.

    **Overlapping forms are all returned.** ``"my golden retriever has been
    limping"`` yields ``golden_retriever`` at ``(3, 19)`` and ``retriever`` at
    ``(10, 19)``; choosing one is a verdict, and this family does not make
    verdicts -- the offsets make the containment visible and the caller
    decides. Longest first, because a declared score carries no order.

    **It overrides** :meth:`~DeclaredSignal._located` **and not**
    :meth:`~DeclaredSignal._hits`: a scan reports *where*, and a
    ``frozenset[str]`` has nowhere to put an offset. Everything else -- the
    constructor, ``name``, ``narrows()``, the rung-side narrowing and the
    batch loop -- comes from the base unchanged.

    **It does not subsume** :class:`ExactNormalizedSignal`, although both read
    ``by_surface_form``, so the two are worth composing together rather than
    choosing between. A probe is a slice *between* token boundaries, so a
    declared form whose first or last character is not alphanumeric is never
    probed at all. Over a vocabulary declaring the aliases ``(beagle)`` and
    ``C.D.C.``, each query being the declared form itself:

    - ``"(beagle)"`` -- the whole-string rung finds it at ``(0, 8)``; this
      rung finds **nothing**, because ``token_spans`` reports ``((1, 7),)``
      and no probe reaches the parentheses.
    - ``"C.D.C."`` -- found at ``(0, 6)``; this rung finds **nothing**, for
      the trailing period.
    - ``"K-9"`` -- found at ``(0, 3)`` by both. An interior boundary character
      is fine; only the edges matter.
    - ``"Beagles!"`` -- the whole-string rung finds **nothing**, and this one
      finds ``beagle`` at ``(0, 7)``.

    The last of those is the other direction, and is why neither rung is the
    stronger one: a scan reaches a declared form sitting inside a punctuated
    query that a whole-string comparison misses. They ask the same index
    differently.
    """

    key = "scan"

    def __init__(
        self,
        entities: EntitySource,
        *,
        normalizer: Callable[[str], str] | None = None,
        max_window: int | None = None,
    ) -> None:
        """Args:
        entities: The vocabulary to match against.
        normalizer: An **extra** fold, applied before the source's own, as on
            every rung -- and here it also means this rung stops asking the
            source how wide a window can be. See :func:`_window_bound`.
        max_window: The widest probe, in tokens, or ``None`` to spend
            whatever the vocabulary implies. **The only number here that can
            cost an answer**: the derived bound removes probes that could not
            have matched, and this one removes probes that could. It exists
            for the vocabulary that cannot be bounded -- a source whose fold
            merges token boundaries reports no bound, and a scan over one
            enumerates every window unless a caller who knows their own
            queries says how wide is worth probing.

        Raises:
            ValidationError: For a ``max_window`` below one.
        """
        super().__init__(entities, normalizer=normalizer)
        self._max_window = _checked_max_window(max_window)

    def _located(self, query: str) -> Sequence[FormHit]:
        found: list[FormHit] = []
        bound = _window_bound(self._entities, self._normalizer, self._max_window)
        for span in _probe_spans(query, bound):
            hits = self._entities.by_surface_form(self._fold(query[span[0] : span[1]]))
            found += _hits_at(span, self._order(hits))
        return found


class AsyncDeclaredSignal:
    """The asynchronous twin's shared half -- **subclass this** for an async rung.

    :class:`DeclaredSignal`'s argument applies unchanged: set :attr:`key`,
    implement ``_hits``, and the constructor, ``name``, ``narrows()``, the
    superset-filter narrowing and the batch loop come with it. ``_located``
    and ``_order`` are the same two further hooks, and ``_order`` stays
    synchronous on this side as well -- ordering a set that has already
    arrived reaches for nothing.

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
        normalizer: An **extra** fold, applied before the source's own.
            ``None``, the default, means the source's fold is the only one.
        """
        self._entities = entities
        self._normalizer = normalizer

    @property
    def name(self) -> str:
        """The key this rung is registered under."""
        return self.key

    def narrows(self) -> bool:
        """True: a filter is honoured against the source's declared types."""
        return True

    async def _hits(self, query: str) -> frozenset[str]:
        raise NotImplementedError

    def _fold(self, form: str) -> str:
        """This rung's own extra fold, or the form untouched."""
        return _folded(form, self._normalizer)

    def _order(self, hits: frozenset[str]) -> Sequence[str]:
        """:meth:`DeclaredSignal._order`, and synchronous for its reasons."""
        return sorted(hits)

    async def _located(self, query: str) -> Sequence[FormHit]:
        """:meth:`DeclaredSignal._located` over an asynchronous index."""
        return _whole_string(query, self._order(await self._hits(self._fold(query))))

    async def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]:
        """At most ``k`` entities whose forms match this query."""
        found = await self._located(query)
        admitted = await _async_admitted(
            self._entities, frozenset(hit.entity_id for hit in found), filter
        )
        return _grouped(found, admitted, k, signal=self.key, query=query)

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
        if not isinstance(self._entities, AsyncAliasFormSource):
            return frozenset()
        return await self._entities.by_alias_form(query)


class AsyncScanningSignal(AsyncDeclaredSignal):
    """:class:`ScanningSignal` over an asynchronous source.

    The same hook one ``await`` further in, which is what makes the span seam
    a property of the family rather than of one flavour: a consumer scanning
    asynchronously does not have to drop to the bare
    :class:`~dataknobs_common.entity_resolution.AsyncMatchSignal` protocol to
    do it.

    **The probes are sequential, and the source protocol has no batch form.**
    Over :class:`~dataknobs_common.ontology.AsyncMappingEntitySource` that
    costs nothing -- the data is in memory and no ``await`` here suspends --
    which is the case this family is for: the rung's contract is a dictionary
    lookup, *no store, no embedder, no network*. A source that does reach for
    data on every ``by_surface_form`` pays one round trip per probe, and the
    bound :func:`_probe_spans` takes from the vocabulary is what keeps that a
    number proportional to the query rather than to its square. If a backing
    that genuinely round-trips is wanted here, the thing to add is a batch
    member on
    :class:`~dataknobs_common.ontology.AsyncEntitySource` -- not a rung that
    fans out, which would make the order it publishes an accident of
    completion.
    """

    key = "scan"

    def __init__(
        self,
        entities: AsyncEntitySource,
        *,
        normalizer: Callable[[str], str] | None = None,
        max_window: int | None = None,
    ) -> None:
        """The same constructor as the synchronous twin's, over an asynchronous source.

        ``max_window`` carries more weight on this flavour than on the other
        one: a probe here is whatever the backing does when awaited, so an
        enumeration nobody could bound is a round trip per window rather than
        a dictionary lookup per window. See :class:`ScanningSignal` for what
        each argument means and for which of them can cost an answer.

        Raises:
            ValidationError: For a ``max_window`` below one.
        """
        super().__init__(entities, normalizer=normalizer)
        self._max_window = _checked_max_window(max_window)

    async def _located(self, query: str) -> Sequence[FormHit]:
        found: list[FormHit] = []
        bound = _window_bound(self._entities, self._normalizer, self._max_window)
        for span in _probe_spans(query, bound):
            hits = await self._entities.by_surface_form(self._fold(query[span[0] : span[1]]))
            found += _hits_at(span, self._order(hits))
        return found


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

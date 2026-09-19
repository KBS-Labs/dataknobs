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

import asyncio
from difflib import SequenceMatcher
from math import ceil
from typing import TYPE_CHECKING, Any, NoReturn

from dataknobs_common.callbacks import is_async_callable
from dataknobs_common.capabilities import Capability

from dataknobs_common.entity_resolution.protocols import (
    AliasFormSource,
    AsyncAliasFormSource,
    AsyncSurfaceFormCatalog,
    SurfaceFormCatalog,
)
from dataknobs_common.hierarchy import K
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
from dataknobs_common.text import content_span, default_normalizer, token_spans

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence, Set as AbstractSet

    from dataknobs_common.ontology.sources import AsyncEntitySource, EntitySource

__all__ = [
    "AliasSignal",
    "AsyncAliasSignal",
    "AsyncDeclaredSignal",
    "AsyncExactNormalizedSignal",
    "AsyncLexicalSignal",
    "AsyncScanningSignal",
    "DeclaredSignal",
    "ExactNormalizedSignal",
    "LexicalSignal",
    "ScanningSignal",
    "declared_candidates",
]

#: A declared hit's score. 1.0 by fiat and carrying no information, which is
#: what ``Scoring.DECLARED`` exists to say: it is not a measurement, and a
#: caller must not average it against a cosine.
_DECLARED_SCORE = 1.0


def _hit_score(hit: FormHit[K]) -> float:
    """What a hit scored, or the declared 1.0 for a rung that did not measure.

    The fallback is the whole of what makes
    :attr:`~dataknobs_common.entity_resolution.values.FormHit.score` a
    widening rather than a change: a hit carrying ``None`` produces exactly
    the number this module wrote unconditionally before the field existed, so
    every rung that does not measure is unchanged by construction rather than
    by inspection.
    """
    return _DECLARED_SCORE if hit.score is None else hit.score


def _candidate(
    entity_id: K,
    signal: str,
    query: str,
    found: Sequence[FormHit[K]],
    *,
    kind: EvidenceKind,
    scoring: Scoring,
) -> EntityCandidate[K]:
    """One entity, with one piece of evidence per place it was found.

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

    **The candidate's own score is its best hit's.** An entity found at two
    places is one candidate carrying two pieces of evidence, and the number
    on the candidate answers *how good is this entity* rather than *how good
    is this place* -- so the best of them is the only reading that does not
    penalise an entity for having been found twice. For a rung that measures
    nothing every hit is 1.0 and so is the maximum, which is the constant
    this line used to write.
    """
    return EntityCandidate(
        entity_id=entity_id,
        score=max(_hit_score(hit) for hit in found),
        evidence=tuple(
            MatchEvidence(
                signal=signal,
                kind=kind,
                score=_hit_score(hit),
                scoring=scoring,
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


def declared_candidates(
    found: Sequence[FormHit[K]],
    k: int,
    *,
    signal: str,
    query: str,
    admitted: AbstractSet[K] | None = None,
    kind: EvidenceKind = EvidenceKind.DECLARED,
    scoring: Scoring = Scoring.DECLARED,
) -> list[EntityCandidate[K]]:
    """One candidate per id found, in the rung's own order, cut to ``k``.

    **What a rung over declared forms owes its cascade, once it knows where
    the forms are.** A rung reaching a backing this package cannot see --
    an authority stack, a gazetteer, a service holding a consumer's own
    vocabulary -- finds its own hits and has the same assembly left to do,
    and that assembly is not a detail of how the hits were found.

    Published for that reason rather than because a caller asked. It was
    private while :class:`DeclaredSignal` was the only way to reach it, and
    that class's own docstring names the rung it cannot serve: one whose
    backing is not a dictionary lookup, which must be written against the
    bare protocol instead. Leaving the assembly private meant such a rung
    reimplemented it, so a declared score of ``1.0`` and the shape of a
    :class:`MatchEvidence` existed in as many copies as there were rungs,
    with nothing comparing them.

    **``k`` counts entities, not hits.** A query naming one entity twice is
    one candidate carrying two pieces of evidence, which is what the cascade
    does with two *rungs* producing the same id -- so a rung that produced it
    twice itself answers the same shape rather than spending two of the
    caller's ``k`` on one entity.

    The order is whatever ``found`` is in, preserved by the insertion order
    of the mapping below. That is the rung's to set: a declared score is
    ``1.0`` by fiat and carries none, so if the order is to mean anything --
    the longer form before the shorter one it contains -- the rung is the
    only layer that can say so. A rung that *does* measure has a number that
    could be sorted on and still publishes its own order, because the cascade
    positions by arrival and this function is not the layer that ranks.

    **Still named for declared forms**, although two of its arguments now let
    a rung say its hits were inferred. The word names where the forms came
    from and not how sure the rung is: a near-spelling rung proposes an
    entity whose form the vocabulary *declares*, having found it spelled
    almost that way, which is the same relation
    :class:`~dataknobs_common.entity_resolution.values.FormHit` keeps its own
    name for. A rung with no declared form behind its hit -- a cosine over an
    embedded utterance, which has no span either -- cannot build a
    :class:`FormHit` at all and so was never a caller of this.

    Args:
        found: Where the rung found each declared form, in the order it wants
            them proposed. Overlapping forms are the several hits they are.
        k: How many **entities** to return at most.
        signal: What the evidence carries as its ``signal``, which is the
            rung's :attr:`~MatchSignal.name` and the string a consumer wrote
            as ``kind:``.
        query: The text the spans point into. ``matched_text`` is sliced from
            it rather than taken from the caller, so the two agree by
            construction instead of by trust.
        admitted: Ids a filter left standing, for a rung that narrows. The
            default admits every id found, which is what a rung answering
            ``narrows() is False`` wants -- it is offered no filter, and the
            cascade rules on its candidates itself.
        kind: What the evidence claims -- the closed half of a rung's
            identity, and the only half a consumer branches on. The default
            is what a rung matching a form the vocabulary carries means; a
            rung proposing an entity the query did not spell passes
            :attr:`EvidenceKind.INFERRED`.
        scoring: What kind of number the score is. The default marks it as
            carrying no information, which is the truth for a hit with no
            :attr:`~dataknobs_common.entity_resolution.values.FormHit.score`;
            a rung that measures passes the kind its scorer produces, and
            :attr:`Scoring.NATIVE` is what a scorer a caller chose produces.
            Passed separately from ``kind`` rather than derived from it
            because the two are independent: ``cascade.py``'s ``_SCORING_FOR``
            pairs them only for a candidate carrying no evidence at all.

    Returns:
        At most ``k`` candidates, each carrying one
        :class:`~dataknobs_common.entity_resolution.values.MatchEvidence` per
        place its form was found. A hit's score is its own where it measured
        one and ``1.0`` where it did not, and a candidate's is the best of
        its hits' -- so a rung that measures nothing produces exactly what
        this function produced before it could carry a measurement.

    Raises:
        ValidationError: For a negative ``k``. The cut is a list slice, where
            a negative counts back from the end -- so ``k=-1`` returned every
            entity **but the last one** rather than none, which is an answer
            no caller meant and none could distinguish from a real one.
            Nothing upstream validates ``k``: a resolver takes it as a keyword
            and hands it down, so the rungs are where a nonsensical one first
            becomes visible, and refusing it here refuses it for every rung at
            once. ``0`` is a real request and answers the empty list.
    """
    if k < 0:
        raise ValidationError(
            f"k must not be negative, got {k}: a rung cannot return fewer than "
            "no entities, and the cut is a slice that would otherwise read a "
            "negative as counting back from the end"
        )
    hits: dict[K, list[FormHit[K]]] = {}
    for hit in found:
        if admitted is None or hit.entity_id in admitted:
            hits.setdefault(hit.entity_id, []).append(hit)
    return [
        _candidate(entity_id, signal, query, at, kind=kind, scoring=scoring)
        for entity_id, at in list(hits.items())[:k]
    ]


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


def _checked_query_tokens(max_query_tokens: int | None) -> int | None:
    """``max_query_tokens`` as given, having refused a cap that probes nothing.

    Shared so the twins refuse identically, for :func:`_checked_max_window`'s
    reason and with its ruling: below one is refused rather than clamped,
    because it builds a rung that reports nothing for every query and is
    indistinguishable from a vocabulary that matches nothing.

    Raises:
        ValidationError: For a ``max_query_tokens`` below one.
    """
    if max_query_tokens is not None and max_query_tokens < 1:
        raise ValidationError(
            f"max_query_tokens must be at least 1, got {max_query_tokens}. A "
            f"rung that may probe no tokens reports an empty result for every "
            f"query, which is what a vocabulary matching nothing looks like."
        )
    return max_query_tokens


def _checked_query(query: str, max_query_tokens: int | None) -> None:
    """Refuse a query with more tokens than this rung agreed to probe.

    **Refused rather than truncated.** Probing the first *n* tokens and
    answering from them is a plausible-looking answer to a question nobody
    asked: the entity a caller is reaching for is as likely to be in the tail
    of a paste as in its head, so a truncating cap converts a cost problem
    into a correctness one and reports nothing about the swap. A caller who
    set this number wants to hear that the input was the wrong shape, and the
    repair -- chunk the text, or raise the cap -- is theirs either way.

    Raises:
        ValidationError: For a query carrying more tokens than the cap.
    """
    if max_query_tokens is None:
        return
    tokens = len(token_spans(query))
    if tokens > max_query_tokens:
        raise ValidationError(
            f"this query carries {tokens} tokens and max_query_tokens is "
            f"{max_query_tokens}. A near-spelling rung scores every window "
            f"against every declared form, so the work grows with the "
            f"caller's own text -- chunk the text, or raise the cap for a "
            f"vocabulary small enough to afford it."
        )


def _checked_callable(
    value: Callable[..., Any] | None, parameter: str
) -> Callable[..., Any] | None:
    """``value`` as given, having refused something that cannot be called.

    **A document is the reason.** Every rung here takes its ``normalizer``
    from a config dict, and the near-spelling rung takes its ``scorer`` the
    same way -- but a YAML or JSON document cannot express a callable, so the
    obvious thing to write is the dotted path, ``scorer: "rapidfuzz.fuzz.ratio"``.
    A string is truthy: it passes ``scorer or _RatioScorer(...)`` and every
    other guard, then raises ``TypeError: 'str' object is not callable`` at
    the first query, once per ``(window, form)`` pair, from inside a loop the
    caller did not write.

    Refused here instead, where the value was supplied and the parameter has
    a name. Shared by both bases and both flavours so that a rung cannot be
    reached by a route that skips it.

    This is the refusal and **not** a resolution: a dotted path is not
    resolved into the function it names. Doing that would make a config
    document able to name any importable callable, which is a wider decision
    than this one and belongs to the layer that already has a convention for
    it.

    Raises:
        ValidationError: For a value that is neither ``None`` nor callable.
    """
    if value is not None and not callable(value):
        raise ValidationError(
            f"{parameter} must be callable or None, got {type(value).__name__} "
            f"({value!r}). A configuration document cannot write a callable, so "
            f"a dotted path here names nothing this rung can call -- build the "
            f"rung in code, or resolve the path before handing it over."
        )
    return value


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


def _require_flavour(entities: object, member: str, *, asynchronous: bool) -> None:
    """Refuse a source publishing ``member`` in the **other** flavour.

    ``isinstance`` against a runtime-checkable protocol compares member
    *names* and nothing else, and every capability protocol in this package
    is twinned under one name -- ``surface_forms`` on both catalogues,
    ``by_alias_form`` on both alias sources. So the structural check alone
    answers ``True`` for either flavour, and the two protocol docstrings that
    say so rest the cost on *each flavour's rung holds a source of its own
    flavour already*. Nothing establishes that: a source arrives as
    ``entities:`` in a document, resolved before either registry sees it, so
    the flavours are exactly what a configuration gets wrong.

    What it costs unguarded is not a refusal anyone can read. A synchronous
    source in an asynchronous rung constructs cleanly and raises
    ``TypeError: object dict_keys can't be used in 'await' expression`` from
    inside the scan; the mirror case raises ``'coroutine' object is not
    iterable`` and warns that a coroutine was never awaited. Both are a
    misconfiguration wearing the costume of a bug in this package.

    So the flavour is asked separately, of the **member** rather than of the
    type. :func:`~dataknobs_common.callbacks.is_async_callable` is the
    judgement rather than :func:`inspect.iscoroutinefunction`, because a
    source may publish the member as a callable *object* -- a client holding
    a session -- whose async-ness lives on its ``__call__``.

    Whether the member is there **at all** is the caller's question and not
    this one's, because the two have different answers: a source lacking
    ``by_alias_form`` declares no aliases and a source lacking
    ``surface_forms`` is misconfigured. Only the flavour means the same thing
    in both places, which is why only the flavour is shared.

    Separate from :func:`_refuse_catalogue` for the same reason one layer
    down: the repair differs, and the reader needs to know which they have.
    That one is missing a member; this one has it and spelled it for the
    other twin, so a message naming only the protocol would send a reader to
    write a method they have already written.

    Raises:
        ValidationError: For a member of the other flavour.
    """
    if is_async_callable(getattr(entities, member)) is asynchronous:
        return
    wanted, got = (
        ("asynchronous", "synchronous") if asynchronous else ("synchronous", "asynchronous")
    )
    raise ValidationError(
        f"{type(entities).__name__} publishes {member}() as a {got} member, "
        f"and this rung needs the {wanted} one. The two protocols spell the "
        f"member identically, so a structural check cannot tell them apart -- "
        f"pass the {wanted} source, or the rung of the other flavour."
    )


def _refuse_catalogue(entities: object, protocol: str) -> NoReturn:
    """Refuse a source that cannot hand over its forms.

    Raised at **construction**, which is the whole of the difference between
    this rung and :class:`AliasSignal`. That one falls back to an empty
    answer, because a vocabulary declaring no aliases is a fact the fallback
    reports honestly. No vocabulary has no forms, so the same fallback here
    would report a misconfigured source as an empty one -- and an empty one
    is exactly what a near-spelling rung looks like when it is working and
    the query was simply not near anything.

    The message names the protocol and the member, because a source
    satisfies one of these **structurally**: there is nothing to inherit and
    nothing to register, so the repair is to add a method with that name and
    the reader needs the name.

    Raises:
        ValidationError: Always.
    """
    raise ValidationError(
        f"{type(entities).__name__} does not publish its surface forms, so a "
        f"near-spelling rung has nothing to compare a query against. It must "
        f"satisfy {protocol}, which asks for one member: surface_forms(), "
        f"answering with every form the vocabulary declares. The protocol is "
        f"structural -- a source satisfies it by having the member, with "
        f"nothing to subclass or register."
    )


def _refuse_unfolded(entities: object, rung: str) -> None:
    """Refuse a source that declines to fold, for a rung that reads its folds.

    :func:`_refuse_catalogue`'s argument against the member one layer over:
    ``by_surface_form`` answers ``frozenset()`` for *ran and matched nothing*,
    a cascade falls through to a guessing rung on exactly that reading, and a
    source that cannot fold has no way to produce any other answer. So it
    withholds
    :attr:`~dataknobs_common.capabilities.Capability.SURFACE_FORM_LOOKUP` and
    raises when asked anyway -- and a rung that would do the asking refuses to
    be built over it rather than carrying a call that can only fail.

    **At construction**, which is where every other refusal in this module
    happens and is what makes this one reach the composition nobody wrote: a
    document declaring no ``resolver:`` gets the default rungs, so no
    inspection of the document can see that two of them read the member. The
    door that builds them holds both halves.

    **Asked of** :meth:`~dataknobs_common.ontology.sources.EntitySource.describe`,
    which is where the contract says the answer lives. A source that publishes
    no ``describe`` is not asked and not refused: the check reads a published
    answer rather than inferring one from a missing member, and an object
    without that member does not satisfy the protocol this rung declares
    anyway.

    Args:
        entities: The source the rung would read.
        rung: The rung's registered key, so the message names what to remove
            from the composition as well as what to fix in the binding.

    Raises:
        ValidationError: For a source whose ``describe()`` withholds the
            capability.
    """
    describe = getattr(entities, "describe", None)
    if describe is None:
        return
    if Capability.SURFACE_FORM_LOOKUP in describe().capabilities:
        return
    raise ValidationError(
        f"{type(entities).__name__} withholds {Capability.SURFACE_FORM_LOOKUP.value}, "
        f"so the {rung!r} rung has nothing it may fold a query against. That "
        f"source holds its forms as they were written and no engine folds at "
        f"query time the way str.casefold does, so by_surface_form would raise "
        f"rather than answer -- and an empty answer, which is what this rung "
        f"would otherwise have to report, already means the vocabulary was "
        f"asked and matched nothing. Give the source a folded lookup to read, "
        f"or compose a cascade without the rungs that read one -- each kind "
        f"declares whether it does, as `reads_surface_forms` on the rung and "
        f"in the registry it is registered under."
    )


def _window_chars(forms: Sequence[str], threshold: float) -> int:
    """The widest window, **in characters**, that could still clear ``threshold``.

    A near-spelling rung's equivalent of :func:`_probe_spans`'s token bound,
    and it exists because that one is **not sound here**. The scan's bound
    rests on a window of *L* tokens needing a key of *L* tokens or more; a
    rung that scores does not need the key to be there at all, so a window
    one token wider than the longest declared form may still score above the
    threshold -- and the typo class *a space where the vocabulary has none*
    needs exactly that. Measured over a vocabulary declaring the one token
    ``labradorretriever``: the query ``labrador retriever`` scores ``0.97``
    at a two-token window and ``0.69`` at the best one-token window, so the
    token bound loses the entity outright rather than losing a span.

    **This bound loses nothing, and that is arithmetic rather than a
    budget.** :meth:`difflib.SequenceMatcher.ratio` is ``2M/T`` for ``M``
    matched characters out of ``T = len(a) + len(b)``, and ``M`` cannot
    exceed the shorter string -- so a window longer than the form can score
    at most ``2|F|/(|W|+|F|)``, which reaches ``threshold`` only while
    ``|W| <= |F|(2-t)/t``. Every window past that answers *below the
    threshold* for **every** form, so removing it removes no candidate.

    Rounded **up**, and deliberately: the equality case is a real match --
    a 17-character form at ``0.85`` admits a 23-character window scoring
    exactly ``0.85`` -- and the division that produces 23 produces
    ``22.999999999999996``. A bound that is one character generous costs a
    handful of probes; a bound that is one character short costs an answer
    the docstring above promises it does not.

    The claim is measured as well as argued, in
    ``packages/common/tests/test_lexical_signal.py``: no window past this
    bound clears its threshold, over pairs drawn to include a form with
    padding on either side, which is the shape that would break it.

    **It is a property of the scorer**, so a rung handed one that is not a
    ratio of matched characters to total length has a bound that does not
    describe it -- see :class:`LexicalSignal`'s ``scorer`` argument, where
    that is the stated contract rather than an assumption.
    """
    longest = max((len(form) for form in forms), default=0)
    return ceil(longest * (2.0 - threshold) / threshold)


def _scored_probe_spans(
    query: str, max_chars: int, fold: Callable[[str], str]
) -> Iterator[tuple[tuple[int, int], str]]:
    """Every span of consecutive tokens whose **folded** width fits ``max_chars``.

    :func:`_probe_spans`'s counterpart for a rung bounded by characters
    rather than by tokens, and shared by both flavours for that function's
    reason: the enumeration has nothing to do with awaiting.

    **It folds, and it measures what the fold left**, which is the unit the
    bound is stated in: :func:`_window_chars` derives it from the folded
    forms and :func:`_near_forms` scores folded windows against them, so a
    probe counting the raw slice would be spending a budget denominated in
    one unit against a quantity measured in another. With the default fold
    the two agree to within a conservative margin -- ``strip`` is a no-op
    inside a token span and ``casefold`` never shortens -- and a caller's own
    ``normalizer`` is free to *delete*, at which point the raw count exceeds
    the bound while the folded one is still under it and the window that
    would have matched is never probed.

    Folding here rather than in the caller also means each window is folded
    once: the admission test and the comparison read the same string.

    The inner loop **breaks** rather than filtering, which is what keeps the
    cost linear: spans grow monotonically from a fixed start, so the first
    one too wide means every longer one is too. That leaves *n x w* spans for
    *n* tokens and *w* the number that fit in the bound -- and *w* is set by
    the vocabulary's longest form, not by the query. The break is sound for a
    fold that does not shorten as its input grows, which is what
    :class:`LexicalSignal`'s ``normalizer`` argument asks for and what every
    character-wise fold is.

    Shortest first within each start, which is the opposite of
    :func:`_probe_spans`: that one publishes an order because a declared
    score carries none, and this rung's hits carry a score that does. The
    enumeration order is therefore free, and growing spans is what lets the
    loop break.

    Yields:
        ``(span, window)`` -- the half-open extent in the **query**, and what
        the fold made of the slice it points at.
    """
    tokens = token_spans(query)
    for first in range(len(tokens)):
        for last in range(first, len(tokens)):
            span = (tokens[first][0], tokens[last][1])
            window = fold(query[span[0] : span[1]])
            if len(window) > max_chars:
                break
            yield span, window


def _checked_threshold(threshold: float) -> float:
    """``threshold`` as given, having refused a value that means no rung.

    Shared so the twins refuse identically, for
    :func:`_checked_max_window`'s reason.

    Zero and below are refused rather than clamped: a threshold of zero
    admits every window against every form, which is a rung proposing the
    whole vocabulary for any query, and a negative one says the same thing
    less legibly. Zero is also the value that would divide by itself in
    :func:`_window_chars`, so the refusal is what keeps that arithmetic from
    having to have an opinion.

    Above one is refused because nothing can reach it: a ratio is in
    ``[0, 1]``, so such a rung matches nothing for every query, which is
    indistinguishable from a vocabulary that declares nothing and is the kind
    of silence this family refuses to ship. One exactly is a real request --
    *only an exact fold* -- and is kept.

    Raises:
        ValidationError: For a threshold outside ``(0, 1]``.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValidationError(
            f"threshold must be greater than 0 and at most 1, got {threshold}. "
            f"A threshold of 0 or less admits every form for every query, and "
            f"one above 1 is unreachable for a score that is a ratio, so "
            f"neither builds a rung that can answer."
        )
    return threshold


def _near_forms(
    query: str,
    forms: Sequence[str],
    *,
    threshold: float,
    scorer: Callable[[str, str], float],
    fold: Callable[[str], str],
) -> list[tuple[str, tuple[int, int], float]]:
    """Each form a window of ``query`` scored at or above ``threshold``.

    The half of a near-spelling probe that reaches for nothing, shared by the
    twins so the two cannot disagree about what *near* means.

    **Both sides are folded, with the same function**, which is this rung's
    one departure from the family's fold policy and the reason
    :class:`LexicalSignal` defaults its ``normalizer`` where every other rung
    defaults to no fold at all. The others hand a string to the index and the
    index folds it; this one *is* the comparison, so an unfolded window
    against a source-folded form would score a case difference as a
    misspelling. Folding both sides with one function makes the comparison
    well defined whatever the source's own fold was. The windows arrive
    already folded, because :func:`_scored_probe_spans` has to fold each one
    to decide whether it fits the bound -- the same string, folded once.

    **Forms outermost**, which is a performance property rather than a
    semantic one and is stated because :class:`_RatioScorer` depends on it:
    that scorer holds an index built from the *form*, so a loop holding the
    form still while the windows change builds it once per form rather than
    once per pair. The order the hits come out in is not affected --
    :func:`_ranked` sorts.

    Returns:
        ``(form, span, score)`` per form-and-window that cleared the
        threshold -- the **form** as the catalogue spelled it, so the caller
        can resolve it through ``by_surface_form`` without folding it twice.
    """
    folded = [(form, fold(form)) for form in forms]
    bound = _window_chars([shape for _form, shape in folded], threshold)
    windows = list(_scored_probe_spans(query, bound, fold))
    # The catalogue is folded and measured on every query rather than cached
    # against the source, and that is a ruling rather than an omission: the
    # asynchronous twin re-reads because a source's contents can change under
    # it, and caching the derived half alone would put mutable state back on
    # a rung whose scorer was just taken off it. Measured at 0.5% of a query
    # over both 1,000 and 10,000 entities -- the scan below is the other
    # 99.5%, so the cache would buy a rounding error and cost a hazard.
    found: list[tuple[str, tuple[int, int], float]] = []
    for form, shape in folded:
        for span, window in windows:
            score = scorer(window, shape)
            if score >= threshold:
                found.append((form, span, score))
    return found


def _ranked(found: Sequence[FormHit[K]]) -> list[FormHit[K]]:
    """One hit per entity per span, the best of them first.

    Two collapses and one order, none of which a declared rung ever needed.

    **One hit per (entity, span).** An entity's id, its name and its aliases
    all fold into the catalogue separately, so one window routinely clears
    the threshold against several forms of the same entity -- and reporting
    three pieces of evidence at one span for one entity would say the
    vocabulary was found three times where it was found once. The best score
    wins, because the others are the same finding measured against a worse
    spelling of it.

    **Best first**, which is the order
    :meth:`DeclaredSignal._order` describes and cannot supply: it says an
    order that is to *mean* something has to be published by the rung, and
    for a rung whose hits carry a measurement the meaningful order is the
    measurement. The cascade positions by arrival, so this is what decides
    which near-spelling a caller sees first.

    Ties break on the **longer span, then the earlier one, then the id** --
    entirely for determinism, since a score is the only thing here that
    carries meaning. Longer first for :func:`_probe_spans`'s reason: where a
    form contains another, the containing one is proposed before the form it
    contains.
    """
    best: dict[tuple[K, tuple[int, int]], FormHit[K]] = {}
    for hit in found:
        at = (hit.entity_id, hit.span)
        current = best.get(at)
        if current is None or _hit_score(hit) > _hit_score(current):
            best[at] = hit
    return sorted(
        best.values(),
        key=lambda hit: (
            -_hit_score(hit),
            -(hit.span[1] - hit.span[0]),
            hit.span[0],
            str(hit.entity_id),
        ),
    )


class _RatioScorer:
    """:meth:`difflib.SequenceMatcher.ratio`, refusing early what it can rule out.

    The default :class:`LexicalSignal` builds when a caller supplies no
    ``scorer``. It is a stateful object rather than a function for two
    reasons, both of them measured, and both **lossless** in the sense
    :func:`_window_chars` uses the word: they remove work that could not have
    changed an answer.

    **It skips a pair the cheap bounds already refuse.**
    :meth:`~difflib.SequenceMatcher.real_quick_ratio` and
    :meth:`~difflib.SequenceMatcher.quick_ratio` are documented *upper
    bounds* on ``ratio()`` -- the first from the lengths alone, the second
    from the multiset of characters -- so a pair either of them scores below
    the threshold cannot reach it, and computing the real ratio for it is
    work with a known answer. Measured over 12,000 pairs at ``0.85``: 80 ms
    becomes 17 ms.

    **It reuses one matcher across the windows of a form.**
    :meth:`~difflib.SequenceMatcher.set_seq2` builds the index this algorithm
    runs on and :meth:`~difflib.SequenceMatcher.set_seq1` does not, so a
    matcher whose *form* is held still while its *windows* change builds that
    index once instead of once per pair. :func:`_near_forms` iterates forms
    outermost so that this holds; if that loop were ever turned inside out,
    the cache would stop paying and nothing else would change. The two
    together: 80 ms becomes 6 ms.

    **Below the threshold it answers ``0.0`` rather than the true score**,
    which is the one thing here a reader should not take for a ratio. It is
    private and has exactly one caller, which keeps only what reaches the
    threshold -- so the substitution is unobservable, and saying so is
    cheaper than computing a number nobody reads.
    """

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._matcher = SequenceMatcher(None)
        self._form: str | None = None

    def __call__(self, window: str, form: str) -> float:
        if form != self._form:
            self._matcher.set_seq2(form)
            self._form = form
        self._matcher.set_seq1(window)
        if self._matcher.real_quick_ratio() < self._threshold:
            return 0.0
        if self._matcher.quick_ratio() < self._threshold:
            return 0.0
        return self._matcher.ratio()


def _scorer_for(
    scorer: Callable[[str, str], float] | None, threshold: float
) -> Callable[[str, str], float]:
    """The scorer one query will use: the caller's, or a fresh default.

    **A fresh one per query**, which is what makes :class:`_RatioScorer`'s
    cache safe. That object holds a matcher and the form it was last given,
    and skips rebuilding the index when the two agree -- so its bookkeeping
    and the matcher's state are two things that must stay in step, and sole
    ownership for the length of one scan is the only thing that keeps them
    there. Held on the rung instead, two callers interleave: one moves
    ``set_seq2`` between the other's cache check and its ``ratio()``, and the
    other scores against a form it never asked about. A wrong number, not a
    crash -- and a rung built once and served from a thread pool, or one
    whose scan has been moved off the event loop, is an ordinary shape rather
    than an exotic one.

    It costs one object per query and no index work: ``SequenceMatcher(None)``
    builds nothing until a sequence is set, and the cache it exists for is
    within a single scan, where the forms are the outer loop.

    A scorer the caller supplied is returned as it was given. Whether it is
    safe to share is then the caller's to know, which it has to be: they may
    have handed over a module-level function, and nothing here could make a
    copy of one.
    """
    return _RatioScorer(threshold) if scorer is None else scorer


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
    :meth:`_order`. A rung that *measures* -- one proposing an entity the
    query did not spell, rather than one the vocabulary spells exactly --
    sets :attr:`kind` and :attr:`scoring` as well and puts its number on each
    :class:`~dataknobs_common.entity_resolution.values.FormHit` it returns;
    :class:`LexicalSignal` is the one in this package that does.
    In particular the narrowing comes with it *correct*: it is a superset filter, and
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

    #: What this rung's evidence claims -- **overridable**, and the closed
    #: half of a rung's identity. The default is what a rung matching a form
    #: the vocabulary carries means, which is the three rungs below and every
    #: rung this base was written for; a rung proposing an entity the query
    #: did not spell sets :attr:`~EvidenceKind.INFERRED`.
    kind = EvidenceKind.DECLARED

    #: Whether this rung reads
    #: :meth:`~dataknobs_common.ontology.sources.EntitySource.by_surface_form`
    #: -- **overridable**, and the fact a composition can be refused against
    #: before anything is built. That member is *partial*: a source holding
    #: its forms as they were written withholds
    #: :attr:`~dataknobs_common.capabilities.Capability.SURFACE_FORM_LOOKUP`
    #: and raises rather than answering over an unfolded column. A rung
    #: setting this True is refused at construction over such a source
    #: instead of carrying a call that can only fail -- so a consumer's own
    #: rung reading the member declares the same thing and gets the same
    #: guard. The default is False, which is the safe direction for a rung
    #: nobody classified: a missed refusal surfaces as a loud error at the
    #: first query, and a wrong one blocks a composition that works.
    reads_surface_forms = False

    #: Whether this rung's enumeration is bounded by the vocabulary's longest
    #: declared form -- **overridable**, and the second fact a composition can
    #: be refused against before anything is built.
    #:
    #: A rung setting this True probes every window of a query and asks the
    #: source, through
    #: :meth:`~dataknobs_common.ontology.sources.EntitySource.longest_form_tokens`,
    #: how wide a window can be. A source answering ``None`` cannot bound it,
    #: and the enumeration is then *n(n+1)/2* probes for *n* tokens -- which
    #: over an in-memory source is a dictionary lookup per probe and over one
    #: that reaches for data is a round trip per probe. The two cases differ by
    #: orders of magnitude and not by anything readable from this flag, so what
    #: it enables is a *door's* refusal rather than a rule here: a door binding
    #: a source that round-trips can ask which of a declared composition's
    #: rungs depend on a number that source cannot supply.
    #:
    #: The default is False, matching :attr:`reads_surface_forms`' safe
    #: direction: a rung nobody classified is not refused, and a wrong True
    #: blocks a composition that works.
    bounded_by_longest_form = False

    #: What kind of number this rung's score is -- **overridable**. The
    #: default marks it as carrying none, which is the truth for a rung whose
    #: hits carry no
    #: :attr:`~dataknobs_common.entity_resolution.values.FormHit.score`.
    #:
    #: Separate from :attr:`kind` because the two are independent. A rung that
    #: infers may still have nothing to measure, and a rung that measures
    #: decides for itself whether its number is comparable with another
    #: rung's -- which is what
    #: :data:`~dataknobs_common.entity_resolution.values._NORMALIZING` admits
    #: and :attr:`~Scoring.NATIVE` declines.
    scoring = Scoring.DECLARED

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
        if self.reads_surface_forms:
            _refuse_unfolded(entities, self.key)
        self._entities = entities
        self._normalizer = _checked_callable(normalizer, "normalizer")

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
        return declared_candidates(
            found,
            k,
            signal=self.key,
            query=query,
            admitted=admitted,
            kind=self.kind,
            scoring=self.scoring,
        )

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
    reads_surface_forms = True

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

    def __init__(
        self,
        entities: EntitySource,
        *,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        """The base's constructor, having settled the source's flavour.

        **At construction rather than per query**, which is both where the
        mistake was made and off a path whose entire work is one dictionary
        lookup. A source *lacking* ``by_alias_form`` is not settled here at
        all -- that is the legitimate *declares no aliases* case, and
        :meth:`_hits` answers it with the empty set as it always has.

        Raises:
            ValidationError: For a source publishing ``by_alias_form``
                asynchronously, which no synchronous rung can call.
        """
        super().__init__(entities, normalizer=normalizer)
        if isinstance(entities, AliasFormSource):
            _require_flavour(entities, "by_alias_form", asynchronous=False)

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
    reads_surface_forms = True
    bounded_by_longest_form = True

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


class LexicalSignal(DeclaredSignal):
    """Propose an entity the query did **not** spell, and say how near it came.

    The first rung in this package whose evidence is a *measurement*. The
    three beside it answer a lookup -- the form is in the vocabulary or it is
    not -- so a query carrying a typo reaches none of them, and
    ``"my goldne retriver has been limping"`` returns nothing at all from a
    cascade of all three. This rung compares each window of the query against
    every form the vocabulary declares and proposes the ones that came close.

    **It needs the forms, not a lookup**, which is what makes it the one rung
    here that asks its source for something
    :class:`~dataknobs_common.ontology.EntitySource` does not publish: every
    member there takes a form and answers with ids, and a query spelling no
    form has nothing to hand them. So the source must also satisfy
    :class:`~dataknobs_common.entity_resolution.SurfaceFormCatalog`, and one
    that does not is refused **at construction** rather than yielding a rung
    that matches nothing.

    That refusal is where this differs from :class:`AliasSignal`, which falls
    back to *declares no aliases* for a source without
    :class:`~dataknobs_common.entity_resolution.AliasFormSource`. The
    difference is in the vocabulary rather than in the taste: a vocabulary
    may legitimately declare no aliases, and none has no forms -- so an empty
    answer there is a fact and here would be a misconfiguration reported as
    one.

    **Both sides of the comparison are folded**, and this rung's
    ``normalizer`` therefore defaults to
    :func:`~dataknobs_common.text.default_normalizer` where every other rung
    in this family defaults to no fold at all. Those hand a string to the
    index and the index folds it; this one performs the comparison itself, so
    an unfolded window scored against a source-folded form would read a
    capital letter as a misspelling. See :func:`_near_forms`.

    **The evidence is** :attr:`~EvidenceKind.INFERRED` **and**
    :attr:`~Scoring.NATIVE`. The first because the entity was proposed rather
    than found: the vocabulary does not carry what the query said. The second
    although a ratio is already in ``[0, 1]`` -- the number means whatever the
    configured ``scorer`` means, so two rungs with different scorers produce
    incomparable ``0.8``s, and
    :data:`~dataknobs_common.entity_resolution.values._NORMALIZING` admits
    :attr:`~Scoring.NORMALIZED` alone precisely so that a scorer-defined
    number cannot enter distribution arithmetic.

    **A near-spelling hit scoring 0.94 still sits behind a declared hit
    scoring 1.0**, and not because 0.94 is smaller: a cascade positions by
    the first rung that produced an id, so the composition decides it. Put
    this rung after the declared ones, which is where the guide puts it and
    what ``test_lexical_signal.py`` asserts over a real
    :class:`~dataknobs_common.entity_resolution.CascadingResolver` -- on the
    correctly spelled query, where both rungs score the same entities at
    ``1.0`` and an order by number would be a coin toss.

    **It is the first rung here whose evidence carries a span it did not
    declare**, and
    :class:`~dataknobs_common.entity_resolution.values.Coverage` reads
    ``DECLARED`` evidence because of it. Until this rung, *inferred* implied
    *unlocated* by construction and the two tests were one; a near-spelling
    proposal is located and inferred, so the implication had to become a
    condition. The effect is that adding this rung to a cascade adds
    candidates and never coverage: the phrase it resolved stays in
    ``unmatched_text()``, where a maintainer wants it, and the windows it
    overreaches into cannot widen ``matched``. Its own spans are on its
    evidence, which is where a caller reads them.
    """

    key = "lexical"
    reads_surface_forms = True

    kind = EvidenceKind.INFERRED
    scoring = Scoring.NATIVE

    def __init__(
        self,
        entities: EntitySource,
        *,
        threshold: float = 0.85,
        scorer: Callable[[str, str], float] | None = None,
        normalizer: Callable[[str], str] | None = None,
        max_query_tokens: int | None = None,
    ) -> None:
        """Args:
        entities: The vocabulary to match against. It must also satisfy
            :class:`~dataknobs_common.entity_resolution.SurfaceFormCatalog`,
            because this rung reads the forms out rather than looking one up.
        threshold: The score at or above which a window is proposed.
            **Measured rather than chosen**, over a vocabulary declaring
            ``golden retriever`` and ``retriever``:

            - ``0.60`` admits ``log`` against ``dog`` at ``0.67``, from an
              ordinary sentence naming no entity. That is
              ``structured_config.py``'s cutoff and is right *there*, where
              the candidates are long configuration keys; a vocabulary
              carries three-letter forms, and any three-letter word is within
              one edit of many others.
            - ``0.75`` admits the window ``retriver has`` at ``0.76`` -- a
              real entity at a span that overreaches into the next token.
            - ``0.85`` returns the two intended entities on the typo query
              and nothing else, and the same two at ``1.0`` on the clean one.

            It also sets how far the probe reaches: see
            :func:`_window_chars`, where a lower threshold buys wider windows
            and the windows it buys are the overreaching ones above.
        scorer: How near two strings are, in ``[0, 1]``. ``None`` means
            :meth:`difflib.SequenceMatcher.ratio` -- the standard library,
            which is why this rung ships here and needs no dependency.

            **A scorer must not exceed** ``2 * min(len(a), len(b)) /
            (len(a) + len(b))``, which is what *a ratio of matched characters
            to total length* means and what both the default and
            ``rapidfuzz.fuzz.ratio`` are. The probe's bound is derived from
            that inequality, so a scorer breaking it has a rung that stops
            probing before its scorer would have answered.

            ``rapidfuzz.fuzz.partial_ratio`` is the one to **not** pass. It
            scores any substring ``1.0``, so ``gold retriever`` against
            ``retriever`` is ``1.00`` and picks the wrong entity over a
            vocabulary whose forms contain one another -- and it breaks the
            inequality above, so it also silently shortens the probe.
        normalizer: The fold applied to **both** the window and the form
            before they are compared. Defaults to
            :func:`~dataknobs_common.text.default_normalizer`, which is the
            fold this package's own sources apply, rather than to ``None`` as
            it does on every other rung here -- see the class docstring.

            It must not **shorten** as its input grows: the probe's bound is
            spent in folded characters and stops widening at the first window
            over it, so a fold that deletes more from a longer slice than
            from a shorter one could stop the probe early. Every
            character-wise fold satisfies this, including the default and
            anything built from ``strip``, ``casefold`` or ``replace``.
        max_query_tokens: How many tokens a query may carry, or ``None`` --
            the default -- for no limit.

            **The one parameter here that can cost an answer**, which is why
            it is off unless asked for, and the reason it exists is that
            nothing else bounds the *number* of windows. ``threshold`` bounds
            how wide one may be; how many there are is the caller's token
            count, and each costs one scorer call per declared form where a
            scan's costs one dictionary lookup. Measured: linear in the
            token count, with a nine-hundred-token paste over a
            five-hundred-entity vocabulary taking two seconds of one CPU.

            A query over the cap is **refused rather than truncated** -- see
            :func:`_checked_query`.

        Raises:
            ValidationError: For a ``threshold`` outside ``(0, 1]``, for a
                ``max_query_tokens`` below one, for a ``scorer`` or
                ``normalizer`` that is not callable, or for a source that
                does not publish its forms in this flavour.
        """
        super().__init__(entities, normalizer=normalizer or default_normalizer)
        if not isinstance(entities, SurfaceFormCatalog):
            _refuse_catalogue(entities, "SurfaceFormCatalog")
        _require_flavour(entities, "surface_forms", asynchronous=False)
        self._catalogue = entities
        self._threshold = _checked_threshold(threshold)
        self._scorer = _checked_callable(scorer, "scorer")
        self._max_query_tokens = _checked_query_tokens(max_query_tokens)

    def _located(self, query: str) -> Sequence[FormHit]:
        """Every entity a window of the query came near, best first.

        :meth:`~DeclaredSignal._hits` is not the hook here for
        :class:`ScanningSignal`'s reason one step further on: a
        ``frozenset[str]`` has nowhere to put an offset, and this rung also
        has a score to carry.

        :meth:`~DeclaredSignal._order` is not reached either, and that is
        worth stating because a subclass overriding it would silently change
        nothing. That hook orders the ids found at one span, for a rung whose
        hits are all 1.0; here the order is the score's and
        :func:`_ranked` is where it is decided.
        """
        _checked_query(query, self._max_query_tokens)
        found = _near_forms(
            query,
            list(self._catalogue.surface_forms()),
            threshold=self._threshold,
            scorer=_scorer_for(self._scorer, self._threshold),
            fold=self._fold,
        )
        return _ranked(
            [
                FormHit(entity_id=entity_id, span=span, score=score)
                for form, span, score in found
                for entity_id in self._entities.by_surface_form(form)
            ]
        )


class AsyncDeclaredSignal:
    """The asynchronous twin's shared half -- **subclass this** for an async rung.

    :class:`DeclaredSignal`'s argument applies unchanged: set :attr:`key`,
    implement ``_hits``, and the constructor, ``name``, ``narrows()``, the
    superset-filter narrowing and the batch loop come with it. ``_located``
    and ``_order`` are the same two further hooks, and ``_order`` stays
    synchronous on this side as well -- ordering a set that has already
    arrived reaches for nothing.

    :attr:`kind` and :attr:`scoring` are declared here too, with the same
    defaults, rather than being reached through the synchronous base. The two
    bases share no superclass by design, so a widening applied to one and not
    the other is the per-flavour drift this family's twin-parity guard exists
    to refuse -- and a guard over the *rungs* could not see it, because both
    twins would simply be wrong in the same way.

    ``name`` and ``narrows()`` stay synchronous: neither reaches for data, and
    making them awaitable would cost every caller an ``await`` for nothing.
    They are also what lets a registry tell the twins apart -- the guard that
    separates the flavours skips a property, agrees on a plain ``def``, and
    separates the pair on the two ``candidates`` members.
    """

    key = ""

    #: What this rung's evidence claims -- **overridable**, and the closed
    #: half of a rung's identity. The default is what a rung matching a form
    #: the vocabulary carries means, which is the three rungs below and every
    #: rung this base was written for; a rung proposing an entity the query
    #: did not spell sets :attr:`~EvidenceKind.INFERRED`.
    kind = EvidenceKind.DECLARED

    #: Whether this rung reads
    #: :meth:`~dataknobs_common.ontology.sources.EntitySource.by_surface_form`
    #: -- **overridable**, and the fact a composition can be refused against
    #: before anything is built. That member is *partial*: a source holding
    #: its forms as they were written withholds
    #: :attr:`~dataknobs_common.capabilities.Capability.SURFACE_FORM_LOOKUP`
    #: and raises rather than answering over an unfolded column. A rung
    #: setting this True is refused at construction over such a source
    #: instead of carrying a call that can only fail -- so a consumer's own
    #: rung reading the member declares the same thing and gets the same
    #: guard. The default is False, which is the safe direction for a rung
    #: nobody classified: a missed refusal surfaces as a loud error at the
    #: first query, and a wrong one blocks a composition that works.
    reads_surface_forms = False

    #: Whether this rung's enumeration is bounded by the vocabulary's longest
    #: declared form -- **overridable**, and the second fact a composition can
    #: be refused against before anything is built.
    #:
    #: A rung setting this True probes every window of a query and asks the
    #: source, through
    #: :meth:`~dataknobs_common.ontology.sources.EntitySource.longest_form_tokens`,
    #: how wide a window can be. A source answering ``None`` cannot bound it,
    #: and the enumeration is then *n(n+1)/2* probes for *n* tokens -- which
    #: over an in-memory source is a dictionary lookup per probe and over one
    #: that reaches for data is a round trip per probe. The two cases differ by
    #: orders of magnitude and not by anything readable from this flag, so what
    #: it enables is a *door's* refusal rather than a rule here: a door binding
    #: a source that round-trips can ask which of a declared composition's
    #: rungs depend on a number that source cannot supply.
    #:
    #: The default is False, matching :attr:`reads_surface_forms`' safe
    #: direction: a rung nobody classified is not refused, and a wrong True
    #: blocks a composition that works.
    bounded_by_longest_form = False

    #: What kind of number this rung's score is -- **overridable**. The
    #: default marks it as carrying none, which is the truth for a rung whose
    #: hits carry no
    #: :attr:`~dataknobs_common.entity_resolution.values.FormHit.score`.
    #:
    #: Separate from :attr:`kind` because the two are independent. A rung that
    #: infers may still have nothing to measure, and a rung that measures
    #: decides for itself whether its number is comparable with another
    #: rung's -- which is what
    #: :data:`~dataknobs_common.entity_resolution.values._NORMALIZING` admits
    #: and :attr:`~Scoring.NATIVE` declines.
    scoring = Scoring.DECLARED

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
        if self.reads_surface_forms:
            _refuse_unfolded(entities, self.key)
        self._entities = entities
        self._normalizer = _checked_callable(normalizer, "normalizer")

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
        return declared_candidates(
            found,
            k,
            signal=self.key,
            query=query,
            admitted=admitted,
            kind=self.kind,
            scoring=self.scoring,
        )

    async def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]:
        """One answer per query, in the order asked."""
        return [await self.candidates(query, k, filter=filter) for query in queries]


class AsyncExactNormalizedSignal(AsyncDeclaredSignal):
    """:class:`ExactNormalizedSignal` over an asynchronous source."""

    key = "exact"
    reads_surface_forms = True

    async def _hits(self, query: str) -> frozenset[str]:
        return await self._entities.by_surface_form(query)


class AsyncAliasSignal(AsyncDeclaredSignal):
    """:class:`AliasSignal` over an asynchronous source."""

    key = "alias"

    def __init__(
        self,
        entities: AsyncEntitySource,
        *,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        """:meth:`AliasSignal.__init__` over an asynchronous source.

        Raises:
            ValidationError: For a source publishing ``by_alias_form``
                synchronously, which this rung would await.
        """
        super().__init__(entities, normalizer=normalizer)
        if isinstance(entities, AsyncAliasFormSource):
            _require_flavour(entities, "by_alias_form", asynchronous=True)

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
    reads_surface_forms = True
    bounded_by_longest_form = True

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


class AsyncLexicalSignal(AsyncDeclaredSignal):
    """:class:`LexicalSignal` over an asynchronous source.

    The same rung one ``await`` further in, and the source must satisfy
    :class:`~dataknobs_common.entity_resolution.AsyncSurfaceFormCatalog`.

    **Reading the forms is awaited on every query**, where the synchronous
    twin's is a dict view. That is the honest shape for a source that does
    reach for data: a catalogue is the source's whole contents, so a rung
    that read it once at construction would hold a vocabulary that had since
    changed and report matches against forms the source no longer carries. A
    source that can afford to answer from memory answers immediately, which
    is what :class:`~dataknobs_common.ontology.AsyncMappingEntitySource`
    does.

    The id resolution is one ``await`` per **winning form** rather than per
    window, which is what the threshold buys: the probe is characters of
    arithmetic and only the forms that cleared it are looked up.
    """

    key = "lexical"
    reads_surface_forms = True

    kind = EvidenceKind.INFERRED
    scoring = Scoring.NATIVE

    def __init__(
        self,
        entities: AsyncEntitySource,
        *,
        threshold: float = 0.85,
        scorer: Callable[[str, str], float] | None = None,
        normalizer: Callable[[str], str] | None = None,
        max_query_tokens: int | None = None,
    ) -> None:
        """The same constructor as the synchronous twin's, over an asynchronous source.

        See :class:`LexicalSignal` for what each argument means, for the
        measurement behind the default threshold, and for the inequality a
        replacement ``scorer`` must satisfy.

        ``max_query_tokens`` carries more weight on this flavour than on the
        other one, for the reason :class:`AsyncScanningSignal`'s
        ``max_window`` does and then one further: the scan is handed to a
        worker thread, so an unbounded one occupies a thread from the default
        executor for as long as it runs rather than merely occupying the
        caller.

        Raises:
            ValidationError: For a ``threshold`` outside ``(0, 1]``, for a
                ``max_query_tokens`` below one, for a ``scorer`` or
                ``normalizer`` that is not callable, or for a source that
                does not publish its forms in this flavour.
        """
        super().__init__(entities, normalizer=normalizer or default_normalizer)
        if not isinstance(entities, AsyncSurfaceFormCatalog):
            _refuse_catalogue(entities, "AsyncSurfaceFormCatalog")
        _require_flavour(entities, "surface_forms", asynchronous=True)
        self._catalogue = entities
        self._threshold = _checked_threshold(threshold)
        self._scorer = _checked_callable(scorer, "scorer")
        self._max_query_tokens = _checked_query_tokens(max_query_tokens)

    async def _located(self, query: str) -> Sequence[FormHit]:
        """:meth:`LexicalSignal._located` over an asynchronous source.

        **The scan is offloaded**, and it is the one place in this family
        that needs to be. Every other rung here awaits a lookup and does
        arithmetic on the answer; this one scores every window against every
        form, which is pure CPU proportional to the vocabulary -- ~315 ms per
        query over 10,000 entities, measured, with no ``await`` inside it to
        yield on. Left on the event loop that is 315 ms during which every
        other task sharing the loop makes no progress: the harm
        ``async-transport.md`` names, reached through arithmetic rather than
        through a syscall, so neither the ``ASYNC2xx`` lint nor
        ``assert_no_blocking`` reports it and a test asserting the answer
        cannot see it either.

        :func:`_near_forms` is a plain function taking only values, which is
        what makes it handable to a thread -- and
        :func:`_scorer_for` building a scorer per query is what makes it safe
        to, since the default one is stateful and would otherwise be shared
        across whatever else the loop is running.
        """
        _checked_query(query, self._max_query_tokens)
        found = await asyncio.to_thread(
            _near_forms,
            query,
            list(await self._catalogue.surface_forms()),
            threshold=self._threshold,
            scorer=_scorer_for(self._scorer, self._threshold),
            fold=self._fold,
        )
        hits: list[FormHit] = []
        for form, span, score in found:
            for entity_id in await self._entities.by_surface_form(form):
                hits.append(FormHit(entity_id=entity_id, span=span, score=score))
        return _ranked(hits)


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

"""The control flow, and the two resolvers that drive it.

A cascade asks its rungs in order and stops when it has enough. It does not
fuse, it does not re-score, and there is **no threshold above the rungs** --
the composition is the policy, and the composition is a list. That is what
lets a declared hit outrank a higher-scoring vector one without a weight
anywhere: the exact rung is simply asked first.

**Saturation, not short-circuit**, which has four consequences no signature
shows. A rung is asked for ``k`` rather than for the remainder, so it never
has to know what the ones before it found. A duplicate appends evidence and
moves nothing. The rung of record is the earliest one, and no later rung
re-scores what an earlier one placed. And a rung is not consulted at all once
``k`` is filled.

:class:`CascadeState`, :func:`merge_rung` and :func:`finish` are the shared
core, and the rule they follow is *share everything that is not an* ``await``.
Merging, de-duplication, ordering and saturation are all of the cascade except
the call into a rung -- which is why the two resolvers below are not two
implementations of one algorithm. Patch :func:`merge_rung` and both change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Self

from dataknobs_common.entity_resolution.values import (
    CompatibilityVerdict,
    Coverage,
    EntityCandidate,
    EvidenceKind,
    MatchEvidence,
    ResolutionResult,
    Scoring,
    Within,
    within_admits,
    within_axes,
)
from dataknobs_common.entity_resolution.values import TAXONOMY_ID_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from dataknobs_common.entity_resolution.protocols import (
        AsyncEntityResolver,
        AsyncMatchSignal,
        MatchSignal,
    )
    from dataknobs_common.ontology.sources import AsyncEntitySource, EntitySource

__all__ = [
    "AsyncCascadingResolver",
    "BridgedEntityResolver",
    "CascadeState",
    "CascadingResolver",
    "finish",
    "merge_rung",
]


@dataclass(frozen=True)
class CascadeState:
    """What the loop holds between rungs.

    Pure: no rung, no store, no ``await``. Everything a cascade does apart
    from calling a rung is a function of this value, which is what lets one
    core sit under both flavours.
    """

    query: str
    k: int
    order: tuple[str, ...] = ()
    """Entity ids, in cascade order. Position is fixed by the first rung that
    produced an id and by nothing later."""

    evidence: Mapping[str, tuple[MatchEvidence, ...]] = field(default_factory=dict)
    """Every rung's reason, per id, in the order the rungs were asked."""

    def saturated(self) -> bool:
        """Whether ``k`` is filled, and no further rung need be asked."""
        return len(self.order) >= self.k


def merge_rung(
    state: CascadeState,
    produced: Sequence[EntityCandidate],
    *,
    signal: str,
    kind: EvidenceKind,
) -> CascadeState:
    """Fold one rung's candidates into the state.

    The only thing that decides position, which is why the delegation test
    patches this and asserts that both flavours change: de-duplication written
    twice is invisible until two rungs return the same id.

    An id already held keeps its place and gains this rung's evidence. A new
    id is appended, until ``k`` ids are held -- a rung asked for ``k`` may
    return more than the state has room for, and the ones that do not fit are
    dropped rather than displacing something an earlier rung placed.

    Args:
        state: What the loop holds so far.
        produced: What the rung returned, in its own order.
        signal: The rung's registered name, stamped onto the evidence here.
            The cascade is the only place that knows which rung it just asked,
            so stamping here is what makes ``evidence.signal`` correlate with
            the ``kind:`` a consumer configured, rather than with whatever a
            rung chose to call itself.
        kind: The kind to record where a candidate carries no evidence of its
            own. A rung that supplies evidence keeps its own kind: whether a
            match was declared or inferred is a property of the match, and a
            rung that can do both is the reason this is not read off the rung.
    """
    order = list(state.order)
    evidence = dict(state.evidence)

    for candidate in produced:
        stamped = _stamp(candidate, signal=signal, kind=kind)
        if candidate.entity_id in evidence:
            evidence[candidate.entity_id] = evidence[candidate.entity_id] + stamped
            continue
        if len(order) >= state.k:
            continue
        order.append(candidate.entity_id)
        evidence[candidate.entity_id] = stamped

    return replace(state, order=tuple(order), evidence=evidence)


#: What kind of number a rung that supplied none would have produced.
#:
#: Reached only for a candidate carrying no evidence, which the rungs in this
#: package never return. It is here so that path constructs something honest
#: rather than claiming a measurement: a declared hit is 1.0 by fiat.
_SCORING_FOR = {
    EvidenceKind.DECLARED: Scoring.DECLARED,
    EvidenceKind.INFERRED: Scoring.NATIVE,
}


def _stamp(
    candidate: EntityCandidate, *, signal: str, kind: EvidenceKind
) -> tuple[MatchEvidence, ...]:
    """One rung's evidence for one candidate, carrying the rung's own name."""
    if not candidate.evidence:
        return (
            MatchEvidence(
                signal=signal,
                kind=kind,
                score=candidate.score,
                scoring=_SCORING_FOR[kind],
                matched_text="",
            ),
        )
    return tuple(replace(item, signal=signal) for item in candidate.evidence)


def finish(state: CascadeState, *, compatibility: CompatibilityVerdict | None) -> ResolutionResult:
    """Turn the state into the result a caller gets back.

    A candidate's score is its **rung of record's** -- the first rung that
    produced it -- because no later rung re-scores what an earlier one placed.

    Args:
        state: What the loop ended holding.
        compatibility: The verdict from the search that produced the numbers,
            or ``None`` where nothing searched a corpus. ``None`` becomes
            ``UNKNOWN`` rather than ``COMPATIBLE``: signals that embed nothing
            establish nothing, and saying the corpus was coherent because
            nobody looked is the failure this field exists to prevent.
    """
    candidates = tuple(
        EntityCandidate(
            entity_id=entity_id,
            score=state.evidence[entity_id][0].score,
            evidence=state.evidence[entity_id],
        )
        for entity_id in state.order
    )
    matched = tuple(
        dict.fromkeys(
            item.matched_text
            for candidate in candidates
            for item in candidate.evidence
            if item.matched_text
        )
    )
    # What reached no candidate. These rungs match the whole query, so the
    # residue is all-or-nothing -- but it must still be *reported*, because a
    # miss that says nothing about what it could not place is the quiet
    # failure this field exists to make loud. A caller maintaining a
    # vocabulary reads exactly this to find the next entry to add.
    unmatched = () if matched or not state.query else (state.query,)
    return ResolutionResult(
        candidates=candidates,
        query=state.query,
        compatibility=compatibility if compatibility is not None else CompatibilityVerdict.UNKNOWN,
        coverage=Coverage(matched=matched, unmatched=unmatched),
    )


def _rung_filter(within: Within) -> dict[str, Any] | None:
    """``within`` in the shape a rung's ``filter`` takes, or ``None``.

    One translation, called by both flavours. The scope's meaning -- union
    within an axis, conjunction across them -- is
    :func:`~dataknobs_common.entity_resolution.values.within_admits`'s, so a
    rung honouring this filter and a store honouring the async path's are
    answering the same question.
    """
    axes = within_axes(within)
    if not axes:
        return None
    return {axis: sorted(admitted) for axis, admitted in axes.items()}


def _offered(
    rung: MatchSignal | AsyncMatchSignal, rung_filter: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The filter to hand a rung -- only one that declared it can honour it.

    ``narrows()`` is a rung's own statement, and handing a scope to a rung
    that answered ``False`` is the caller ignoring it. Now that the cascade
    rules on the scope itself, what passes here is purely the rung's chance to
    return ``k`` candidates that are already inside it, so a rung declining is
    a cost and never a correctness question.
    """
    return rung_filter if rung.narrows() else None


def _memberships(entity: Any) -> Mapping[str, str]:
    """What one entity is, per scope axis.

    One axis on this path, because ``describe().declares`` is what an entity
    source publishes as its scope terms and it names types. A source that
    declares more axes gives this function more to say without changing
    anything that calls it.
    """
    return {TAXONOMY_ID_KEY: entity.type}


def _admits(
    found: Mapping[str, Any],
    produced: Sequence[EntityCandidate],
    axes: Mapping[str, frozenset[str]],
) -> tuple[EntityCandidate, ...]:
    """The candidates a scope admits, decided against **the cascade's** source.

    The drop is here rather than in the rungs because the cascade is the one
    component both flavours run, and because ``MatchSignal`` does not require
    two rungs to share a backing: under a rung-side drop, two rungs could
    disagree about whether an entity is in a type and nothing would notice.
    Here there is one authority, so a rung that answers differently -- or
    ignores the filter entirely -- is overruled rather than believed.

    An id the source does not carry is dropped under a scope. It cannot be
    shown to be inside one, and admitting what cannot be checked is the
    failure this whole path exists to refuse.
    """
    return tuple(
        candidate
        for candidate in produced
        if candidate.entity_id in found
        and within_admits(axes, _memberships(found[candidate.entity_id]))
    )


class CascadingResolver:
    """Ask rungs in order until ``k`` is filled.

    The synchronous flavour is the native one here rather than an
    accommodation: over a vocabulary someone typed, every rung is a dictionary
    lookup and there is nothing to await.
    """

    def __init__(self, rungs: Sequence[MatchSignal], entities: EntitySource) -> None:
        """Args:
        rungs: In the order they are asked. The order is the policy.
        entities: The authority a ``within`` scope is decided against. Held
            by the cascade rather than consulted through the rungs, so one
            answer governs however many backings the rungs have between them.
        """
        self._rungs = tuple(rungs)
        self._entities = entities

    @property
    def rungs(self) -> tuple[MatchSignal, ...]:
        """The rungs, in the order they are asked."""
        return self._rungs

    def resolve(self, name: str, *, k: int = 5, within: Within = None) -> ResolutionResult:
        """Place one string, with every rung's reason for each candidate."""
        state = CascadeState(query=name, k=k)
        axes = within_axes(within)
        rung_filter = _rung_filter(within)
        for rung in self._rungs:
            if state.saturated():
                break
            produced = rung.candidates(name, k, filter=_offered(rung, rung_filter))
            scoped: Sequence[EntityCandidate] = produced
            if axes:
                found = self._entities.get_many([c.entity_id for c in produced])
                scoped = _admits(found, produced, axes)
            state = merge_rung(state, scoped, signal=rung.name, kind=_batch_kind(scoped))
        return finish(state, compatibility=None)

    def resolve_many(
        self, names: Sequence[str], *, k: int = 5, within: Within = None
    ) -> list[ResolutionResult]:
        """The bulk form, for a corpus rather than a turn.

        Each rung is asked once for the whole batch, so a rung with a real
        batch path spends one round trip rather than one per query.
        """
        states = [CascadeState(query=name, k=k) for name in names]
        axes = within_axes(within)
        rung_filter = _rung_filter(within)
        for rung in self._rungs:
            if all(state.saturated() for state in states):
                break
            batches: Sequence[Sequence[EntityCandidate]] = rung.candidates_many(
                names, k, filter=_offered(rung, rung_filter)
            )
            if axes:
                found = self._entities.get_many([c.entity_id for batch in batches for c in batch])
                batches = [_admits(found, batch, axes) for batch in batches]
            states = _merge_batch(states, batches, signal=rung.name)
        return [finish(state, compatibility=None) for state in states]


class AsyncCascadingResolver:
    """:class:`CascadingResolver` for rungs that reach for data.

    The same core, the same order and the same saturation rule -- the only
    difference is the ``await`` around each rung. That is the whole reason
    :func:`merge_rung` and :func:`finish` are module functions.
    """

    def __init__(self, rungs: Sequence[AsyncMatchSignal], entities: AsyncEntitySource) -> None:
        """Args:
        rungs: In the order they are asked. The order is the policy.
        entities: The authority a ``within`` scope is decided against.
        """
        self._rungs = tuple(rungs)
        self._entities = entities

    @property
    def rungs(self) -> tuple[AsyncMatchSignal, ...]:
        """The rungs, in the order they are asked."""
        return self._rungs

    async def resolve(self, name: str, *, k: int = 5, within: Within = None) -> ResolutionResult:
        """Place one string, with every rung's reason for each candidate."""
        state = CascadeState(query=name, k=k)
        axes = within_axes(within)
        rung_filter = _rung_filter(within)
        for rung in self._rungs:
            if state.saturated():
                break
            produced = await rung.candidates(name, k, filter=_offered(rung, rung_filter))
            scoped: Sequence[EntityCandidate] = produced
            if axes:
                found = await self._entities.get_many([c.entity_id for c in produced])
                scoped = _admits(found, produced, axes)
            state = merge_rung(state, scoped, signal=rung.name, kind=_batch_kind(scoped))
        return finish(state, compatibility=None)

    async def resolve_many(
        self, names: Sequence[str], *, k: int = 5, within: Within = None
    ) -> list[ResolutionResult]:
        """The bulk form, for a corpus rather than a turn."""
        states = [CascadeState(query=name, k=k) for name in names]
        axes = within_axes(within)
        rung_filter = _rung_filter(within)
        for rung in self._rungs:
            if all(state.saturated() for state in states):
                break
            batches: Sequence[Sequence[EntityCandidate]] = await rung.candidates_many(
                names, k, filter=_offered(rung, rung_filter)
            )
            if axes:
                found = await self._entities.get_many(
                    [c.entity_id for batch in batches for c in batch]
                )
                batches = [_admits(found, batch, axes) for batch in batches]
            states = _merge_batch(states, batches, signal=rung.name)
        return [finish(state, compatibility=None) for state in states]


def _batch_kind(produced: Sequence[EntityCandidate]) -> EvidenceKind:
    """The kind to record for a batch whose candidates carry no evidence.

    Read off the batch rather than off the rung, because the protocol has no
    ``kind`` member and deliberately so: whether a match was declared or
    inferred is a property of the match. A rung that exact-matches some
    queries and fuzzy-matches others reports both from one ``name``.
    """
    for candidate in produced:
        if candidate.evidence:
            return candidate.evidence[0].kind
    return EvidenceKind.DECLARED


def _merge_batch(
    states: list[CascadeState],
    batches: Sequence[Sequence[EntityCandidate]],
    *,
    signal: str,
) -> list[CascadeState]:
    """Fold one rung's batch answer into one state per query.

    A separate function only so both flavours call it; the merging itself is
    still :func:`merge_rung`, once per query.
    """
    return [
        merge_rung(state, produced, signal=signal, kind=_batch_kind(produced))
        for state, produced in zip(states, batches, strict=True)
    ]


class BridgedEntityResolver:
    """An :class:`~dataknobs_common.entity_resolution.protocols.EntityResolver`
    over an asynchronous cascade.

    For a synchronous caller who has no choice: the rungs they need are
    asynchronous, and the call site cannot await. It holds one
    :class:`~dataknobs_common.sync_bridge.SyncLoopBridge` -- a private event
    loop on a daemon thread -- so it is callable from plain synchronous code
    *and* from inside a running loop without the ``run_until_complete``
    deadlock.

    It is **not** a twin of :class:`CascadingResolver`. It satisfies the
    synchronous protocol by forwarding rather than by implementing, which is
    why a signature-parity test over the twins does not include it and why it
    has no asynchronous counterpart: the thing it would be a twin of is the
    resolver it already holds.

    "Callable from inside a running loop" means it does not *deadlock*. It
    still *blocks*: the calling thread waits for the whole cascade, so every
    other task on the caller's loop is stalled meanwhile. From async code,
    ``await resolver.resolve(...)`` directly.

    The bridge costs one daemon thread for the object's lifetime, so build one
    and keep it rather than one per call. It is a daemon, so it can never
    block process exit; :meth:`close` is for deterministic teardown, and the
    class is a context manager for the same reason.
    """

    def __init__(self, inner: AsyncEntityResolver, *, timeout: float | None = None) -> None:
        """Args:
        inner: The asynchronous resolver to reach.
        timeout: Seconds to allow each call, giving a synchronous caller an
            upper bound on a blocking wait it cannot otherwise cancel.
        """
        from dataknobs_common.sync_bridge import SyncLoopBridge

        self._inner = inner
        self._timeout = timeout
        self._bridge = SyncLoopBridge(thread_name="dk-sync-resolver")

    def resolve(self, name: str, *, k: int = 5, within: Within = None) -> ResolutionResult:
        """The inner resolver's, run on the bridge's loop."""
        return self._bridge.run(
            self._inner.resolve(name, k=k, within=within), timeout=self._timeout
        )

    def resolve_many(
        self, names: Sequence[str], *, k: int = 5, within: Within = None
    ) -> list[ResolutionResult]:
        """The inner resolver's, run on the bridge's loop."""
        return self._bridge.run(
            self._inner.resolve_many(names, k=k, within=within), timeout=self._timeout
        )

    def close(self) -> None:
        """Stop the bridge's loop and join its thread."""
        self._bridge.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

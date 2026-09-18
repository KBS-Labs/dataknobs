# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The resolution rung that reads an authority stack.

``dataknobs_common`` ships three rungs that read an entity source and nothing
else, which is what lets a cascade fall back to them when a consumer
configures nothing. This is the rung a consumer *injects* when their
vocabulary is more than a list of forms: an
:class:`~dataknobs_xization.authorities.Authority` matches text the
vocabulary **describes** as well as text it **enumerates**, so a declared
pattern -- a chip number, a date, an account code -- resolves where no
enumeration could have carried it.

**It is an upgrade in one direction and a downgrade in the other, and both
are measured rather than asserted.** Over ``"my golden retriever has been
limping"``, with ``golden retriever`` and ``retriever`` both declared:

* one authority per *form* returns both, at ``(3, 19)`` and ``(10, 19)``;
* one authority per *axis* -- which is how a vocabulary is normally loaded --
  returns the containing form alone, at ``(3, 19)``.

``dataknobs_common``'s scanning rung returns both in either case, because it
probes every token span independently. So the stack reaches forms the scan
cannot and suppresses overlap the scan keeps, and a consumer choosing between
them is choosing which of those two properties they need. Both are pinned by
tests rather than left to this docstring.

**The ids are the authority's.** A rung answers in the resolver's ontology's
id space, and this one reads whatever the authority wrote as its value id --
``canonical_fn``'s answer for a regex authority, the backing frame's id for a
dictionary one. Nothing here coerces or rewrites it: a consumer who injects
this rung is declaring that their authority's value ids *are* their
ontology's entity ids, and an id quietly turned into a plausible neighbour of
itself is the one failure a cascade cannot report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from dataknobs_common.entity_resolution.registry import (
    async_signal_backends,
    signal_backends,
)
from dataknobs_common.entity_resolution.values import (
    EntityCandidate,
    EvidenceKind,
    FormHit,
    MatchEvidence,
    Scoring,
)
from dataknobs_common.hierarchy import K

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataknobs_common.entity_resolution.protocols import AsyncMatchSignal, MatchSignal
    from dataknobs_xization.authorities import Authority

__all__ = [
    "AsyncAuthoritySignal",
    "AuthoritySignal",
]

#: The registered key, and the string a consumer writes as ``kind:``.
_KEY = "authority"

#: A declared hit's score, as ``dataknobs_common``'s own rungs spell it: 1.0
#: by fiat, which is what ``Scoring.DECLARED`` exists to say. Written here
#: rather than imported because ``common`` keeps its copy private -- the
#: invariant is the enum member, and that is shared.
_DECLARED_SCORE = 1.0


def _located(authorities: Authority, query: str) -> list[FormHit[Any]]:
    """Where the stack found each of its matches in ``query``.

    In the stack's own order, which is already the order a rung owes: an
    :class:`~dataknobs_xization.authorities.AuthorityAnnotationsMetaData`
    sorts by start ascending and end **descending**, so a form containing
    another is proposed before the form it contains. A declared score carries
    no order, so this is the only layer that could have said it.

    The column names come from the authority's own metadata rather than from
    the literals they default to -- they are configurable, and a rung reading
    ``"auth_id"`` directly would answer correctly for every consumer who left
    them alone and silently find nothing for the one who did not.
    """
    annotations = authorities.annotate_input(query)
    if annotations.is_empty():
        return []
    columns = authorities.metadata
    return [
        FormHit(
            entity_id=row[columns.auth_id_col],
            span=(int(row[columns.start_pos_col]), int(row[columns.end_pos_col])),
        )
        for _, row in annotations.df.iterrows()
    ]


def _candidates(authorities: Authority, query: str, k: int) -> list[EntityCandidate[Any]]:
    """At most ``k`` entities the stack proposes for ``query``.

    **Shared by both flavours**, and that is the whole of what they share: the
    authority stack is synchronous, so the twin below differs from this one in
    its ``async def`` and in nothing else. Two copies of the grouping would be
    two places for the span arithmetic to drift, in a direction no signature
    parity check could see.

    ``k`` counts **entities**, not hits. A query naming one entity twice is
    one candidate carrying two pieces of evidence -- the same shape a cascade
    produces when two rungs agree -- rather than two of the caller's ``k``
    spent on one answer.

    ``matched_text`` is sliced from the query rather than read from the
    annotation's text column, so it and the span agree by construction
    instead of by trust.
    """
    grouped: dict[Any, list[FormHit[Any]]] = {}
    for hit in _located(authorities, query):
        grouped.setdefault(hit.entity_id, []).append(hit)
    return [
        EntityCandidate(
            entity_id=entity_id,
            score=_DECLARED_SCORE,
            evidence=tuple(
                MatchEvidence(
                    signal=_KEY,
                    kind=EvidenceKind.DECLARED,
                    score=_DECLARED_SCORE,
                    scoring=Scoring.DECLARED,
                    matched_text=query[hit.span[0] : hit.span[1]],
                    span=hit.span,
                )
                for hit in found
            ),
        )
        for entity_id, found in list(grouped.items())[:k]
    ]


class AuthoritySignal(Generic[K]):
    """Propose entities an :class:`~dataknobs_xization.authorities.Authority` found.

    **Written against the bare protocol rather than against**
    :class:`~dataknobs_common.entity_resolution.DeclaredSignal`, which is the
    case that base's own docstring names: a rung whose backing is not a
    dictionary lookup. ``DeclaredSignal`` holds an ``EntitySource`` and
    supplies the rung-side narrowing from it; an authority stack is not one
    and has no declared types to narrow against, which is also why
    :meth:`narrows` answers ``False``.

    **The evidence is ``DECLARED`` at 1.0**, like every other rung over forms
    a vocabulary carries -- by enumeration in the dictionary arm, by
    description in the regex arm. A near-spelling rung, which proposes an
    entity the query did not spell, is ``INFERRED`` with a real number
    underneath it; that is a different kind and is not this class.
    """

    #: The registered key, and the string the evidence carries as its
    #: ``signal``, so a caller reading ``evidence.signal`` can correlate a hit
    #: back to the ``kind:`` they configured.
    key = _KEY

    def __init__(self, authorities: Authority) -> None:
        """Args:
        authorities: The stack to ask. **An**
            :class:`~dataknobs_xization.authorities.Authority`, not an
            :class:`~dataknobs_xization.authorities.AuthoritiesBundle`
            specifically: a bundle *is* one, and requiring the subclass would
            make a consumer with a single regex authority wrap it in a bundle
            of one to satisfy an annotation. The member this rung calls is
            ``annotate_input``, which the base declares.

            A bundle is still what a vocabulary of several axes wants, and
            the class docstring says why: overlap survives *across* member
            authorities and is suppressed *within* one.
        """
        self._authorities = authorities

    @property
    def name(self) -> str:
        """The key this rung is registered under."""
        return self.key

    def narrows(self) -> bool:
        """False: this rung has no vocabulary to read declared types from.

        An authority stack answers *where a form was found*, and holds no
        index of which types an entity belongs to -- so a filter handed here
        could only be ignored or guessed at. Answering ``False`` means the
        cascade never offers one and rules on the candidates itself, which
        costs a wider batch and gets the scope right.
        """
        return False

    def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate[K]]:
        """At most ``k`` entities whose declared forms the stack found."""
        return _candidates(self._authorities, query, k)

    def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate[K]]]:
        """One answer per query, in the order asked.

        A loop: an authority stack matches one text at a time and there is no
        round trip a batch could save, so a batch path here would be the same
        work behind a second spelling of it.
        """
        return [self.candidates(query, k, filter=filter) for query in queries]


class AsyncAuthoritySignal(Generic[K]):
    """:class:`AuthoritySignal` for an asynchronous cascade.

    **The twin exists for composition, not for concurrency.** An authority
    stack is regular expressions and in-memory frames -- no socket, no file,
    no driver -- so there is nothing here to await and nothing to offload:
    :func:`~asyncio.to_thread` around a CPU-bound call buys a context switch
    and no parallelism. What the twin buys is that an asynchronous cascade,
    whose other rungs really do reach a store, can hold this one without a
    bridge.

    That is why it shares :func:`_candidates` verbatim with its synchronous
    twin rather than reimplementing it: the difference between the two is the
    ``async def`` and nothing else, and writing it twice would be two places
    for the same arithmetic to drift.
    """

    key = _KEY

    def __init__(self, authorities: Authority) -> None:
        """Args:
        authorities: The stack to ask, as :class:`AuthoritySignal` takes it.
        """
        self._authorities = authorities

    @property
    def name(self) -> str:
        """The key this rung is registered under."""
        return self.key

    def narrows(self) -> bool:
        """False, for :meth:`AuthoritySignal.narrows`'s reason."""
        return False

    async def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate[K]]:
        """At most ``k`` entities whose declared forms the stack found."""
        return _candidates(self._authorities, query, k)

    async def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate[K]]]:
        """One answer per query, in the order asked."""
        return [await self.candidates(query, k, filter=filter) for query in queries]


def _make_authority(config: dict[str, Any]) -> MatchSignal[Any]:
    return AuthoritySignal(config["authorities"])


def _make_async_authority(config: dict[str, Any]) -> AsyncMatchSignal[Any]:
    return AsyncAuthoritySignal(config["authorities"])


#: What a door reads before building anything. ``needs_io`` is ``False`` for
#: the reason :class:`AsyncAuthoritySignal` gives: the stack reaches no
#: service. ``requires_install`` is what makes the mark this registration
#: clears actionable -- a reader who asked ``common`` for this kind and was
#: told it ships elsewhere can act on the answer.
_AUTHORITY_METADATA = {
    "flavour": "sync",
    "needs_io": False,
    "requires_install": "pip install dataknobs-xization",
}
_ASYNC_AUTHORITY_METADATA = dict(_AUTHORITY_METADATA, flavour="async")

# Registering here, at this module's import, is what clears the mark
# ``dataknobs_common`` leaves for this kind -- see that registry's
# ``declare_unavailable`` call. Until an application imports this module the
# kind is *known and unavailable* rather than unknown, so a consumer who
# wrote ``kind: authority`` is told where it ships instead of being sent
# looking for a typo in a name they spelled correctly.
signal_backends.register(_KEY, _make_authority, metadata=_AUTHORITY_METADATA)
async_signal_backends.register(_KEY, _make_async_authority, metadata=_ASYNC_AUTHORITY_METADATA)

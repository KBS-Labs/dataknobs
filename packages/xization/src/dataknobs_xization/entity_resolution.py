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
    declared_signal_metadata,
    signal_backends,
)
from dataknobs_common.entity_resolution.signals import declared_candidates
from dataknobs_common.entity_resolution.values import EntityCandidate, FormHit
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.hierarchy import K

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataknobs_common.entity_resolution.protocols import AsyncMatchSignal, MatchSignal
    from dataknobs_xization.authorities import Authority, AuthorityAnnotationsMetaData

__all__ = [
    "AsyncAuthoritySignal",
    "AuthoritySignal",
]

#: The registered key, and the string a consumer writes as ``kind:``.
_KEY = "authority"


def _read(
    row: dict[str, Any], vocabularies: Sequence[AuthorityAnnotationsMetaData]
) -> FormHit[Any]:
    """One row, read through the column vocabulary it was actually written in.

    **Not through the vocabulary of the authority the rung was handed**,
    which is the wrong one by construction for a stack of more than one: a
    bundle's members write in their own names, as
    :meth:`~dataknobs_xization.authorities.AuthoritiesBundle.find_matches`
    says outright, and the bundle's names need not reach them. Reading every
    row through one metadata found ``NaN`` for a member that had renamed a
    position column, and ``int(NaN)`` raises several frames below anything
    naming an authority.

    The first vocabulary whose three columns the row carries wins. Where the
    stack shares one -- the ordinary case -- there is one candidate and the
    choice is not a choice; where it does not, exactly one vocabulary reaches
    a given row, because a row carries the columns its finder named and no
    others.

    Raises:
        ValidationError: If no vocabulary in the stack reaches the row. The
            rung wraps the query itself, so every row in these annotations
            was written by this stack -- a row none of its finders can read
            means an authority writes in a vocabulary it does not report, and
            skipping it would lose a match silently.
    """
    read: set[str] = set()
    for columns in vocabularies:
        triple = (columns.auth_id_col, columns.start_pos_col, columns.end_pos_col)
        if all(row.get(column) is not None for column in triple):
            auth_id, start, end = triple
            return FormHit(entity_id=row[auth_id], span=(int(row[start]), int(row[end])))
        read.update(triple)
    raise ValidationError(
        "no authority in this stack names the columns of a row it produced: "
        f"the row carries {sorted(row)} and the stack reads {sorted(read)}. "
        "An Authority that writes rows in a vocabulary its finders() does "
        "not report cannot have them read back."
    )


def _located(authorities: Authority, query: str) -> list[FormHit[Any]]:
    """Where the stack found each of its matches in ``query``.

    **Start ascending and end descending**, so a form containing another is
    proposed before the form it contains. A declared score carries no order,
    so the rung is the only layer that could have said it -- and it says it
    here rather than borrowing the shared frame's sort, which orders by one
    metadata's columns and therefore cannot order a stack that writes in
    several. A row the sort could not read was placed by where its ``NaN``
    landed.

    The rows are read as the authorities wrote them rather than through
    ``annotations.df``, which is the same point one layer down: a frame is a
    single set of columns and these rows need not share one.
    """
    annotations = authorities.annotate_input(query)
    if annotations.is_empty():
        return []
    vocabularies = [finder.metadata for finder in authorities.finders()]
    hits = [_read(row, vocabularies) for row in annotations.ann_row_dicts]
    hits.sort(key=lambda hit: (hit.span[0], -hit.span[1]))
    return hits


def _candidates(authorities: Authority, query: str, k: int) -> list[EntityCandidate[Any]]:
    """At most ``k`` entities the stack proposes for ``query``.

    **Shared by both flavours**, and that is the whole of what they share: the
    authority stack is synchronous, so the twin below differs from this one in
    its ``async def`` and in nothing else.

    The assembly itself is ``dataknobs_common``'s, which is where it belongs:
    what a declared hit's evidence looks like -- the ``1.0`` by fiat, the
    :class:`~dataknobs_common.entity_resolution.values.Scoring` member, the
    ``matched_text`` sliced from the query so it and the span agree by
    construction -- is a property of the *cascade's* vocabulary rather than of
    how this rung found its hits. It was written out here while
    :func:`~dataknobs_common.entity_resolution.declared_candidates` was
    private, which put a second copy of that shape in a second distribution
    with nothing comparing them: a change to what declared evidence carries
    would have reached one and not the other, and nothing would have failed.

    No ``admitted`` argument, because :meth:`AuthoritySignal.narrows` answers
    ``False`` and a rung that does not narrow is never offered a filter -- the
    cascade rules on these candidates itself.
    """
    return declared_candidates(_located(authorities, query), k, signal=_KEY, query=query)


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

    **Generic in the entity key, and the parameter is a claim rather than a
    check.** An authority answers with whatever it wrote as its value id --
    ``canonical_fn``'s answer in the regex arm, the backing frame's index in
    the dictionary one -- and neither has a static type this class could read.
    So ``AuthoritySignal[MyId](stack)`` says *my stack's value ids are* ``MyId``
    *and are my ontology's entity ids*, which is the declaration the module
    docstring describes a consumer as making by injecting this rung at all.
    Nothing here verifies it, and an id quietly turned into a plausible
    neighbour of itself is the one failure a cascade cannot report.
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
        costs a wider batch and decides the scope against the one source that
        can decide it.

        **Which is not the same as every candidate surviving a scope.** The
        cascade rules against *its* entity source, and an id that source does
        not carry cannot be shown to be inside a type -- so it is set aside
        under :attr:`~dataknobs_common.entity_resolution.values.Coverage.beyond_authority`
        rather than returned. That is the designed channel and not a loss, but
        it lands on this rung's headline case: an id a ``canonical_fn``
        computed from the matched text is by construction not one an ontology
        describing the pattern enumerates.
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

    **The twin exists for composition, not for concurrency.** What it buys is
    that an asynchronous cascade, whose other rungs really do reach a store,
    can hold this one without a bridge.

    It shares :func:`_candidates` with its synchronous twin rather than
    reimplementing it: the difference between the two is the ``async def`` and
    nothing else, and writing it twice would be two places for the same
    arithmetic to drift.

    **The work runs on the caller's event loop, and the stack handed here
    must be one that can.** The two arms this package ships are regular
    expressions and in-memory frames -- no socket, no file, no driver -- so
    they perform no I/O, which is what the repository's blocking-transport
    rule is about. They are not free, though: the first call on a dictionary arm
    materializes the whole vocabulary, and a loop is shared, so a co-tenant
    waits for it. That is a cost sized by the consumer's vocabulary rather
    than by their query, and a consumer holding a large one on a busy loop
    should say so by wrapping this rung rather than by being surprised.

    The constructor accepts any
    :class:`~dataknobs_xization.authorities.Authority`, which is a wider type
    than the two shipped arms: it is an abstract base a consumer subclasses.
    **A subclass whose** ``add_annotations`` **reaches a service does not
    belong here** -- there is no ``await`` for it to be reached through, so it
    would block the loop for the length of its round trip. Such an authority
    wants an adapter that offloads it, not this rung.
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
_AUTHORITY_BASE = {
    "flavour": "sync",
    "needs_io": False,
    "requires_install": "pip install dataknobs-xization",
}

#: The base, plus what each rung declares about itself, read off the class by
#: :func:`~dataknobs_common.entity_resolution.declared_signal_metadata` rather
#: than written out here.
#:
#: **This is the registration that shows why that function is published.** The
#: keys it adds are read at *load* time by doors in other distributions --
#: whether a rung reads folded surface forms, and whether its cost is bounded
#: by a number the source may not be able to supply -- and a registration that
#: omits them is not refused, it is simply never asked about. Restating them
#: here would put a second spelling of each fact one package away from the
#: class that owns it, which is the drift the derivation exists to prevent and
#: is worse across a distribution boundary than within one: the two files
#: version separately, so a rung that gained a fact in a release this package
#: has not picked up disagrees silently.
#:
#: Both rungs here answer ``False`` to both, and the answer is the *class's*
#: rather than this dict's. Neither subclasses ``DeclaredSignal`` -- see
#: :class:`AuthoritySignal` -- so neither declares either attribute and the
#: helper reads its default, which is the reading that refuses nothing. An
#: authority stack holds no folded form table and enumerates no window, so
#: that is the right answer; what matters is that it stops being a coincidence
#: the day either rung gains a fact.
_AUTHORITY_METADATA = declared_signal_metadata(AuthoritySignal, _AUTHORITY_BASE)
_ASYNC_AUTHORITY_METADATA = declared_signal_metadata(
    AsyncAuthoritySignal, dict(_AUTHORITY_BASE, flavour="async")
)


def _register_rungs(*, override: bool = False) -> None:
    """Put both flavours in ``dataknobs_common``'s registries.

    Run at this module's import, which is what clears the mark that package
    leaves for this kind -- see its ``declare_unavailable`` calls. Until an
    application imports this module the kind is *known and unavailable*
    rather than unknown, so a consumer who wrote ``kind: authority`` is told
    where it ships instead of being sent looking for a typo in a name they
    spelled correctly.

    **A key a consumer already holds is left alone**, because that registry's
    own prose calls their registration the extension point: *"A consumer who
    writes either kind and registers their own clears the mark the same
    way."* ``register`` refuses a key it already holds, so without the check
    the two orders failed in two ways and the worse one was ours -- a
    consumer who registered first made any later ``import
    dataknobs_xization`` raise out of the import statement, naming a registry
    they need never have heard of and taking down the markdown chunker and
    the normalizer along with the rung.

    Skipping rather than passing ``override=True`` is the other half: not
    crashing is not the same as not clobbering, and a consumer's own rung
    winning is the behaviour the prose promises. It also makes the call
    idempotent, so a reload runs it safely.

    Args:
        override: Replace whatever holds the key. For restoring this
            package's own registration -- a test that stood a rung of its own
            in the way and is putting ours back -- where the consumer-wins
            rule is the thing being worked around rather than honoured.
    """
    # Written out rather than looped: the two registries are parameterized on
    # different protocols, and a loop over the pair widens both factories to
    # their union, which neither registry's `register` accepts.
    if override or not signal_backends.is_registered(_KEY):
        signal_backends.register(
            _KEY, _make_authority, metadata=_AUTHORITY_METADATA, override=override
        )
    if override or not async_signal_backends.is_registered(_KEY):
        async_signal_backends.register(
            _KEY, _make_async_authority, metadata=_ASYNC_AUTHORITY_METADATA, override=override
        )


_register_rungs()

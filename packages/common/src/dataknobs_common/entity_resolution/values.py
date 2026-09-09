"""What a rung produces, what a resolution returns, and what a commit keeps.

Pure data and pure accessors. Nothing here opens a connection, reads a file or
awaits, which is what lets a rung defined in a package with no database
dependency construct every type below.

Three of these -- :class:`Scoring`, :class:`CompatibilityVerdict` and
:class:`ResolutionRef` -- were declared in ``ontology/model.py`` and moved
here. They are the ontology model's only types the resolution family
*constructs* at runtime, and while they lived there the family could not
import them without closing a cycle through ``ontology/__init__``. Their
identity is unchanged: ``dataknobs_common.ontology.Scoring`` is this object,
re-exported, not a copy.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "CompatibilityVerdict",
    "Coverage",
    "EntityCandidate",
    "EvidenceKind",
    "MatchEvidence",
    "ResolutionRef",
    "ResolutionResult",
    "Scoring",
    "TAXONOMY_ID_KEY",
    "Within",
    "within_admits",
    "within_axes",
]


#: A scope to resolve inside: a set id, a collection of them, or a mapping
#: from a published scope axis to either, ``AND``-ed across keys.
#:
#: Published once and referenced, rather than spelled out at each of the seven
#: sites that take one. The union was widened from ``str | Collection[str] |
#: None`` to include the mapping form, and a widening applied to six copies by
#: hand is a widening applied to five of them: there is no second copy here to
#: forget. The bare forms are unchanged and remain sugar for the default axis.
Within = str | Collection[str] | Mapping[str, str | Collection[str]] | None


class Scoring(Enum):
    """What kind of number a match score is.

    A cascade's candidates are heterogeneous by construction -- an exact hit
    and a vector neighbour are not the same measurement -- so the kind travels
    with each piece of evidence rather than being declared once per result.
    """

    DECLARED = "declared"
    RANK_FUSED = "rank_fused"
    NORMALIZED = "normalized"
    NATIVE = "native"
    DECAYED = "decayed"


#: The scoring kinds a distribution may be built from.
#:
#: One member, and each exclusion is a separate argument rather than an
#: oversight. ``DECLARED`` is 1.0 by fiat and carries no information, so it is
#: the kind that exists to say *do not average this*. ``RANK_FUSED`` discards
#: its input scores and is meaningful only inside one result set.
#: ``NATIVE`` is backend-defined. ``DECAYED`` is any of the others multiplied
#: by a hop decay, which leaves it incomparable even to an undecayed score
#: from the same query.
_NORMALIZING = frozenset({Scoring.NORMALIZED})


class CompatibilityVerdict(Enum):
    """Whether a stored corpus was produced by the model now asking of it."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNVERIFIABLE = "unverifiable"
    UNKNOWN = "unknown"


class EvidenceKind(Enum):
    """The only half of a rung's identity a consumer branches on."""

    DECLARED = "declared"
    """A surface form the vocabulary carries matched -- exact, alias."""

    INFERRED = "inferred"
    """A neighbourhood or a near-spelling -- vector, lexical."""


@dataclass(frozen=True)
class MatchEvidence:
    """One rung's reason for producing one candidate.

    ``signal`` is a **string**, not an enum, and that is deliberate: a closed
    set of rung names written today refuses the consumer rung of tomorrow, and
    ``narrows()`` already proves this protocol expects consumer
    implementations. ``kind`` is the closed half, and the only half.
    """

    signal: str
    """The rung's own ``name`` -- open, and the key it is registered under."""

    kind: EvidenceKind
    """Closed, and the only closed half of a rung's identity."""

    score: float

    scoring: Scoring
    """What kind of number ``score`` is -- per rung, not per result."""

    matched_text: str

    span: tuple[int, int] | None = None
    """Half-open, into :attr:`ResolutionResult.query`.

    ``None`` where the rung cannot locate it -- a cosine hit over an embedded
    utterance has no position in it, and a rung matching a whole query string
    has not been asked to find one.
    """


@dataclass(frozen=True)
class EntityCandidate:
    """An entity a cascade produced, with every rung's reason for it."""

    entity_id: str
    """In the **resolver's ontology's** id space.

    A rung reading an index answers in qualified ids and localizes on the way
    out; it can, because a resolver is per-ontology, so there is exactly one
    space to answer in. Increment A's two rungs read a mapping source and are
    local at both ends, so nothing here exercises the conversion -- the field
    still says which space it is in, because the case that needs it is one
    cascade holding rungs that answer in two.
    """

    score: float
    """The candidate's own.

    A field rather than a property returning ``evidence[0].score``, which is
    the obvious move and is wrong: a subclass one phase on carries a score no
    rung produced -- a fused-or-native number multiplied by a hop decay -- and
    a subclass cannot override a base-class property with a field. That fails
    at *every construction* while passing an import check.
    """

    evidence: tuple[MatchEvidence, ...] = ()
    """Ordered by rung. Empty only where a match was inherited rather than
    made, which nothing in this package produces."""

    @property
    def declared(self) -> bool:
        """Some evidence is ``DECLARED`` -- a form the vocabulary carries.

        The negation is deliberately not called ``is_guess``: a match that
        inherits from a more specific one has no surface evidence at all and
        inherits rather than guesses, and those are different things.
        """
        return any(item.kind is EvidenceKind.DECLARED for item in self.evidence)


@dataclass(frozen=True)
class Coverage:
    """Which parts of a query reached a candidate, and which reached none."""

    matched: tuple[str, ...] = ()
    """Contributed to at least one candidate."""

    unmatched: tuple[str, ...] = ()
    """Contributed to none.

    A report, not a verdict. Nothing about an unmatched phrase makes a
    resolution wrong -- an absence is not a falsehood. What it is good for is
    maintenance: the phrases a corpus's users ask about and the vocabulary
    does not cover are the next entries somebody should add.
    """


@dataclass(frozen=True)
class ResolutionResult:
    """What a resolver returns: the candidates, and what they can be trusted for."""

    candidates: tuple[EntityCandidate, ...]
    """Ordered, best first. A miss is an empty tuple -- there is no separate
    outcome enum, because ``RESOLVED`` and ``UNRESOLVED`` between them said
    exactly ``bool(candidates)``."""

    query: str

    compatibility: CompatibilityVerdict = CompatibilityVerdict.UNKNOWN
    """Of the corpus searched. ``UNKNOWN`` where none was, which is the
    authored path: signals that embed nothing establish nothing, and saying
    ``COMPATIBLE`` because nobody looked is the same failure one level over."""

    coverage: Coverage = Coverage()
    """What the query left unaccounted for."""

    def ranked(self) -> tuple[EntityCandidate, ...]:
        """The candidates in order.

        Order survives every scoring kind -- a cascade positions by rung, so
        the ranking does not depend on the numbers being comparable. It does
        **not** survive an ``INCOMPATIBLE`` corpus, where it is an artifact of
        the embedder mix. A caller putting this list in front of a person
        reads :attr:`compatibility` first and shows them unordered if it is
        ``INCOMPATIBLE``, rather than showing a ranking that means nothing.
        """
        return self.candidates

    def as_distribution(self) -> dict[str, float] | None:
        """The scores as a distribution, or ``None`` where they are not one.

        ``None`` rather than an approximation, ever. Two things can refuse:
        an ``INCOMPATIBLE`` corpus, where the numbers were produced by
        different models and comparing them is arithmetic on incomparable
        quantities; and a candidate whose rung of record produced a kind that
        does not normalize.

        Evaluated per candidate rather than once per result, because a
        cascade's tuple holds an exact ``1.0`` beside a cosine ``0.83`` and
        one field cannot describe both. A cascade of exactly one normalizing
        rung can still answer -- this is a method that refuses, not a door
        that fails to open.
        """
        if self.compatibility is CompatibilityVerdict.INCOMPATIBLE:
            return None
        if not self.candidates:
            return None
        for candidate in self.candidates:
            if not candidate.evidence:
                return None
            if candidate.evidence[0].scoring not in _NORMALIZING:
                return None
        total = sum(candidate.score for candidate in self.candidates)
        if total <= 0:
            return None
        return {candidate.entity_id: candidate.score / total for candidate in self.candidates}

    def explain(self, entity_id: str) -> tuple[MatchEvidence, ...]:
        """One candidate's evidence -- the field, not a projection of it.

        A field read over :attr:`candidates` rather than a parallel structure
        kept in step with them by hand, which is the whole reason the evidence
        lives on the object carrying the score. A candidate produced by two
        rungs carries two pieces of evidence and this returns both.

        The return is the tuple rather than a mapping of signal to score:
        that mapping cannot carry ``kind``, ``scoring``, ``matched_text`` or
        ``span``, and since a rung's name is an open string it would silently
        drop one of two rungs that happened to share one.

        Raises:
            KeyError: If no candidate has this id. An empty tuple is the other
                option and is wrong, because ``()`` is what a candidate that
                *inherited* its match rather than making one legitimately
                carries -- so returning it here would make *not a candidate*
                and *a candidate with nothing to say* the same answer.
        """
        for candidate in self.candidates:
            if candidate.entity_id == entity_id:
                return candidate.evidence
        raise KeyError(entity_id)


@dataclass(frozen=True)
class ResolutionRef:
    """What an entity-valued attribute was resolved on.

    Identifiers and numbers. No handle, no I/O: the discipline
    :class:`~dataknobs_common.ontology.model.SourceRef` applies to a database
    row, applied to a search. It identifies a resolution; it does not
    reproduce one.
    """

    query: str
    entity_id: str
    score: float
    scoring: Scoring
    compatibility: CompatibilityVerdict
    corpus: Mapping[str, Any]
    signals: Mapping[str, float] = field(default_factory=dict)
    runners_up: tuple[tuple[str, float], ...] = ()


#: The metadata key a bare ``within`` value scopes on.
#:
#: ``str`` and ``Collection[str]`` are sugar for this axis, which is what keeps
#: the mapping form additive: no call written against the older spelling means
#: anything different now.
TAXONOMY_ID_KEY = "taxonomy_id"


def within_axes(within: Within) -> Mapping[str, frozenset[str]]:
    """``within`` in its one general shape: axis -> the set ids it admits.

    The three spellings collapse here and nowhere else. A scope is enforced by
    a different layer on each flavour -- a vector store on the asynchronous
    path, the cascade or its rungs on the synchronous one -- and each of those
    layers reading the union for itself is how ``AND``-ed and ``OR``-ed
    readings of one value come to coexist in one codebase.

    Returns:
        An empty mapping for ``None``, meaning *no scope*, which is not the
        same as an empty axis: a named axis admitting nothing matches nothing.
    """
    if within is None:
        return {}
    if isinstance(within, str):
        return {TAXONOMY_ID_KEY: frozenset({within})}
    if isinstance(within, Mapping):
        return {
            axis: frozenset({value} if isinstance(value, str) else value)
            for axis, value in within.items()
        }
    return {TAXONOMY_ID_KEY: frozenset(within)}


def within_admits(axes: Mapping[str, frozenset[str]], memberships: Mapping[str, str]) -> bool:
    """Whether something in these axes is inside a scope.

    **Conjunction across keys, union within one.** A scope naming two axes
    admits only what is in *both*, because a consumer whose scope is a kind
    **and** a state has no other way to say so; a scope naming several ids on
    one axis admits any of them, which is what the bare collection form always
    meant.

    Args:
        axes: The scope, from :func:`within_axes`.
        memberships: What the thing being tested is, per axis. An axis absent
            from here is an axis it declares nothing on, so a scope naming
            that axis excludes it -- rather than the more forgiving reading,
            which would let a candidate that says nothing about an axis pass
            every filter on it.
    """
    return all(memberships.get(axis) in admitted for axis, admitted in axes.items())

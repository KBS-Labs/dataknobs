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
from typing import TYPE_CHECKING, Any

from dataknobs_common.entity_resolution.protocols import MembershipOracle
from dataknobs_common.exceptions import ValidationError

if TYPE_CHECKING:
    from dataknobs_common.ontology.model import Entity
    from dataknobs_common.ontology.sources import AsyncEntitySource, EntitySource

__all__ = [
    "ENTITY_TYPE_KEY",
    "CompatibilityVerdict",
    "Coverage",
    "EntityCandidate",
    "EvidenceKind",
    "MatchEvidence",
    "ResolutionRef",
    "ResolutionResult",
    "ScopeAuthority",
    "Scoring",
    "Within",
    "refuse_unknown_axes",
    "within_admits",
    "within_axes",
    "within_axis_names",
    "within_memberships",
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


#: What a scope is decided against: the source an entity came from, either
#: flavour, and possibly an oracle as well.
#:
#: Published once and referenced, for the reason :data:`Within` above gives --
#: three functions take one, and a union widened by hand at three sites is a
#: union widened at two of them.
#:
#: **The oracle arm is load-bearing rather than decorative.** Without it the
#: type checker can prove ``isinstance(source, MembershipOracle)`` false, and
#: reports the branch that exists for it as unreachable: an ordinary
#: ``EntitySource`` declares no ``memberships``, so a union of only the two
#: source protocols excludes exactly the case these functions were written
#: for. A consumer's oracle satisfies both, and this says so.
#:
#: Spelled with ``type`` so the right-hand side is evaluated lazily. The three
#: names in it are importable only under ``TYPE_CHECKING`` -- a runtime import
#: of any of them closes a cycle through ``ontology/__init__``, reproduced
#: rather than assumed -- and a lazy alias is what lets the union still be a
#: real object a consumer can import and annotate with.
type ScopeAuthority = EntitySource | AsyncEntitySource | MembershipOracle | None


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

    beyond_authority: tuple[str, ...] = ()
    """Entity **ids** a rung produced that the scope could not be applied to.

    Not query text, which is what the two fields above hold -- two id spaces
    in one dataclass need the names to carry the difference, and *authority*
    is this family's word for the source a ``within`` scope is decided
    against.

    A rung need not share the cascade's backing, so it can answer with an id
    that backing does not carry. Under a scope such a candidate is dropped:
    it cannot be shown to be inside one, and admitting what cannot be checked
    is what the scope path exists to refuse. Reporting it is what keeps the
    drop from *reading* as a vocabulary miss -- an empty result whose
    ``unmatched`` names the query is also what a correctly spelled scope over
    a vocabulary lacking the phrase returns, and only one of those is the
    caller's to fix. The usual cause is an index gone stale against the
    vocabulary, which is an operational fact rather than a bug, and is why
    this reports rather than refuses.

    **Empty where nothing was scoped**, always: an unscoped resolution asks
    the authority nothing, so there is no check to have failed. Reporting one
    that was never made is the same class of claim this field exists to
    refuse.

    **Not the ordinary exclusion.** A candidate the authority *does* carry and
    the scope rejects is a correct, silent drop and never appears here.

    Accumulated across the rungs of one resolution, first-seen order, without
    duplicates. Untouched by ``k``: these never enter the order, so nothing
    about saturation reaches them.
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


@dataclass(eq=True, frozen=False)
class ResolutionRef:
    """What an entity-valued attribute was resolved on.

    Identifiers and numbers. No handle, no I/O: the discipline
    :class:`~dataknobs_common.ontology.model.SourceRef` applies to a database
    row, applied to a search. It identifies a resolution; it does not
    reproduce one.

    **Compared field-wise, and therefore unhashable**, for the reason that
    class gives: two references recording one resolution are one reference,
    and ``corpus`` and ``signals`` are mappings.
    """

    query: str
    entity_id: str
    score: float
    scoring: Scoring
    compatibility: CompatibilityVerdict
    corpus: Mapping[str, Any]
    signals: Mapping[str, float] = field(default_factory=dict)
    runners_up: tuple[tuple[str, float], ...] = ()


#: The scope axis a bare ``within`` value scopes on.
#:
#: ``str`` and ``Collection[str]`` are sugar for this axis, which is what keeps
#: the mapping form additive: no call written against the older spelling means
#: anything different now.
#:
#: **Named for the values it holds.** The default projection answers with
#: :attr:`~dataknobs_common.ontology.model.Entity.type` -- ``Breed``,
#: ``Species`` -- so that is what the axis is called. It was ``taxonomy_id``
#: once, and the same package keys
#: :attr:`~dataknobs_common.ontology.values.Ontology.taxonomies` by
#: :attr:`~dataknobs_common.ontology.model.TaxonomyDefinition.id`: a consumer
#: holding a genuine taxonomy id was told by the name to pass it here, and
#: doing so returned nothing with nothing said, because
#: :func:`refuse_unknown_axes` refuses an axis a source will not answer for
#: and this was the one it would. A value from the wrong id space under a
#: correctly spelled axis is the single case that refusal cannot see, so the
#: repair was to stop inviting it.
ENTITY_TYPE_KEY = "entity_type"


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
        return {ENTITY_TYPE_KEY: frozenset({within})}
    if isinstance(within, Mapping):
        return {
            axis: frozenset({value} if isinstance(value, str) else value)
            for axis, value in within.items()
        }
    return {ENTITY_TYPE_KEY: frozenset(within)}


def within_memberships(entity: Entity, source: ScopeAuthority = None) -> Mapping[str, str]:
    """What one entity **is**, per scope axis -- the one projection.

    Every place a scope is applied reads membership through here: the cascade,
    which makes the ruling, and the rung-side narrowing on both flavours. That
    is the whole point of publishing it. Three private copies of this
    projection shipped once, agreeing by construction because each read
    ``Entity.type`` -- and a divergence between them could only appear on a
    *second* axis, which is exactly where the suite's only assertion was
    negative. Nothing would have reported it.

    ``Entity`` and the two source protocols are named **under**
    ``TYPE_CHECKING``. Importing any of them at runtime here really does close
    a cycle through ``ontology/__init__`` -- reproduced, not assumed -- but an
    annotation is not a runtime name, and
    :class:`~dataknobs_common.entity_resolution.protocols.MembershipOracle`
    one file over already declares this same parameter that way. So the type
    the oracle branch forwards to and the type this signature admits are one
    type, which is what a wider annotation here could not have given.

    Args:
        entity: The entity to project. The default reads ``type`` alone; an
            oracle may read anything the entity carries, which is why the
            parameter is the class rather than the narrower shape this
            function's own body needs.
        source: The source the entity came from. When it satisfies
            :class:`~dataknobs_common.entity_resolution.protocols.MembershipOracle`
            it is asked instead, which is how a source that knows more than a
            type about its entities makes a multi-axis scope mean something.
            The dispatch lives here rather than at each call site so that
            supplying an oracle cannot reintroduce the divergence this function
            exists to remove.

    Returns:
        Axis name to the one id the entity has on that axis. The default names
        exactly one axis; :func:`within_axis_names` is the reader for which
        axes a given source answers on.
    """
    if isinstance(source, MembershipOracle):
        return source.memberships(entity)
    return {ENTITY_TYPE_KEY: entity.type}


def within_axis_names(source: ScopeAuthority = None) -> frozenset[str]:
    """The axis names a source can be scoped on -- **the legal set**.

    A source satisfying
    :class:`~dataknobs_common.entity_resolution.protocols.MembershipOracle`
    answers for itself; every other source publishes the one axis
    :func:`within_memberships` projects.

    This is the reader
    :attr:`~dataknobs_common.ontology.sources.SourceDescription.declares` was
    twice described as being. ``declares`` holds the type ids a source
    carries -- one axis's *values*, not the set of axis names -- so it could
    never have answered this question.
    """
    if isinstance(source, MembershipOracle):
        return frozenset(source.axes())
    return frozenset({ENTITY_TYPE_KEY})


def refuse_unknown_axes(
    axes: Mapping[str, frozenset[str]],
    source: ScopeAuthority = None,
) -> None:
    """Refuse a scope naming an axis the source does not publish.

    **Because the quiet answer is indistinguishable from a correct one.**
    :func:`within_admits` reads an axis a candidate declares nothing on as
    *excluded*, which is the right reading for an entity and the wrong one for
    a typo: ``{"taxonomy": "Breed"}`` admits no candidate and returns nothing,
    exactly as a correctly spelled scope over a vocabulary holding no ``Breed``
    does. One of those is the caller's mistake and the other is the
    vocabulary's state, and nothing in the result says which.

    Refusing the more forgiving reading was already this family's rule: a
    mis-keyed filter that silently matched *everything* is the failure the
    conjunctive reading exists to refuse. Matching nothing instead is quieter,
    not better -- so the axis name is checked where it enters rather than left
    to mean something by accident.

    Args:
        axes: The scope, from :func:`within_axes`.
        source: The authority the scope will be decided against.

    Raises:
        ValidationError: Naming the axes asked for that this source does not
            publish, and the ones it does.
    """
    published = within_axis_names(source)
    unknown = sorted(set(axes) - published)
    if not unknown:
        return
    raise ValidationError(
        f"within names {'axes' if len(unknown) > 1 else 'an axis'} this source does "
        f"not publish: {unknown}. Axes it publishes: {sorted(published)}",
        context={"unknown": unknown, "published": sorted(published)},
    )


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

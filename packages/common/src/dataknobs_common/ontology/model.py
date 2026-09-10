"""The ontology value types: entities, relations, assertions and their axes.

Pure data. Nothing here opens a connection, reads a file or awaits, which is
what places the module in ``dataknobs-common`` rather than beside a store --
the package's only declared dependency is a marker-scoped typing backport, and
every type below is a value a caller already holds.

The two id constants are the root of the type system, and they are ordinary
entity ids rather than a separate mechanism: an ``EntityType`` is an ``Entity``
whose ``type`` is ``dk:EntityType``, and the root is its own type. That is what
lets one store hold the vocabulary and the things it describes without a second
kind of record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

# Re-exported, not used here: this module is a published import path for these
# three and stayed one when they moved. The redundant-alias spelling of a
# re-export is what `PLC0414` declines, so the directive names `F401` instead.
from dataknobs_common.entity_resolution.values import (  # noqa: F401
    CompatibilityVerdict,
    ResolutionRef,
    Scoring,
)
from dataknobs_common.fields import Field, FieldType

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from dataknobs_common.entity_resolution.values import ResolutionRef

#: The root of the type lattice, and its own type.
DK_ENTITY_TYPE = "dk:EntityType"
#: An ordinary instance of the root, naming the kind a relation is.
DK_RELATION_TYPE = "dk:RelationType"


class InferenceMode(Enum):
    """Whether derived facts are written down or computed per query."""

    MATERIALIZED = "materialized"
    ON_DEMAND = "on_demand"


# `Scoring`, `CompatibilityVerdict` and `ResolutionRef` were declared here and
# now live in `dataknobs_common/entity_resolution/values.py`, re-exported by
# this package's `__init__` **and by this module** so every existing import
# keeps working and keeps resolving to the same object.
#
# Both spellings, because both shipped: a caller who wrote
# `from dataknobs_common.ontology.model import Scoring` reached a real module
# path, and a re-export on the package door alone leaves that one raising
# `ImportError` while the claim above reads as though it did not.
#
# The re-export at the top of this module closes nothing: this direction -- the
# vocabulary reaching the resolution family -- is the one that is allowed, and
# `values` reaches back only under `TYPE_CHECKING`.
#
# They moved because a second family *constructs* them at runtime -- a rung
# building evidence, a cascade building a result -- and a runtime import from
# that family into this one closes a cycle through `ontology/__init__`, which
# imports the loader, which builds a cascade. That failure is by import
# **order**, so a suite importing this package first would never see it.
#
# Placement follows the dependency rather than the family: these three are the
# only types here a resolver has to construct.


@dataclass(frozen=True)
class SourceRef:
    """Where a projected entity or derived assertion came from.

    Data, never a handle. That is what lets it travel out of an ontology to a
    caller who *can* reach the row, from one that cannot.
    """

    source_id: str
    kind: str
    locator: Mapping[str, Any]
    projection_id: str | None = None


@dataclass(frozen=True)
class Provenance:
    """Who said a thing, on what evidence, and when."""

    source: SourceRef | None = None
    extraction_confidence: float | None = None
    resolution: ResolutionRef | None = None
    asserted_by: str | None = None
    asserted_at: datetime | None = None
    derivation: str | None = None


@dataclass
class AttributeDef:
    """One attribute an entity type declares.

    ``description`` is not decoration: it is what an extraction prompt is
    built from, so an empty one costs accuracy rather than tidiness.
    """

    name: str
    value_type: str
    field_type: FieldType | None = None
    entity_type: str | None = None
    required: bool = False
    enum_values: list[str] | None = None
    description: str = ""


@dataclass
class Entity:
    """A thing the vocabulary names.

    ``type`` is itself an entity id, so the type system lives in the graph
    rather than beside it. An empty ``name`` means *use the id*, resolved once
    at construction so no reader has to remember the rule.
    """

    id: str
    type: str
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: SourceRef | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id


@dataclass
class EntityType(Entity):
    """An entity that describes a kind of entity.

    The ``isa`` lattice between *types* is the config's ``isa:`` field and is a
    different store from the ``isa`` assertions between instances.
    """

    type: str = DK_ENTITY_TYPE
    attributes: list[AttributeDef] = field(default_factory=list)


@dataclass
class RelationType(Entity):
    """An entity that describes a kind of edge.

    ``domain`` and ``range`` empty mean unconstrained rather than empty -- a
    relation that permits nothing would be unusable, so the absent case is the
    permissive one.
    """

    type: str = DK_RELATION_TYPE
    domain: frozenset[str] = frozenset()
    range: frozenset[str] = frozenset()
    inverse_of: str | None = None
    symmetric: bool = False
    transitive: bool = False
    inference: InferenceMode = InferenceMode.ON_DEMAND


#: A relation id to resolve, or the definition itself.
RelationRef = str | RelationType


def relation_id(relation: RelationRef) -> str:
    """The id of a relation given either an id or the definition itself.

    Both forms are legal in an assertion, so every comparison goes through
    here rather than each site deciding what it was handed.

    Beside the alias it resolves rather than beside its first caller. It opens
    nothing and awaits nothing, so this module is where the module docstring
    above places it -- and the three holders below canonicalise with it in
    ``__post_init__``, as the two in ``ontology.hierarchy`` do. ``sources``
    imports this module, so a home there would have made those three a
    circular import.
    """
    if isinstance(relation, RelationType):
        return relation.id
    return relation


@dataclass(frozen=True)
class EntityRef:
    """An assertion object that points at another entity."""

    entity_id: str


@dataclass(frozen=True)
class Literal:
    """An assertion object that is a value rather than an entity."""

    value: Any
    type: FieldType
    unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, value: Any, type: FieldType | None = None) -> Literal:
        """Build a literal, detecting the type from the value when omitted.

        Detection goes through :class:`~dataknobs_common.fields.Field` rather
        than a second table of Python types, so a literal and a record field
        agree about what a value is by construction.
        """
        if type is None:
            type = Field(name="", value=value).type
            if type is None:  # pragma: no cover - Field always assigns one
                type = FieldType.STRING
        return cls(value=value, type=type)

    def as_field(self, name: str) -> Field:
        """Project to a ``Field``, for validation or ``convert_to`` coercion."""
        return Field(
            name=name,
            value=self.value,
            type=self.type,
            metadata=dict(self.metadata),
        )


#: What an assertion's object may be.
Term = EntityRef | Literal


class Polarity(Enum):
    """Whether an assertion states a fact or states its negation.

    An ontology is open-world: an assertion nobody wrote is *unknown*, not
    false. ``NEGATED`` is how a document says the other thing -- *this edge
    does not hold* -- as a fact in its own right, with an id, a provenance and
    a place in the file, rather than as the absence of one.

    Two members and no evaluator. Deciding whether a *conditional* assertion
    holds is a later question; stating a negation is this one, and the two are
    separable because nothing evaluates a polarity -- a reader compares it.
    """

    ASSERTED = "asserted"
    NEGATED = "negated"


@dataclass
class Assertion:
    """One stated fact: a subject, a relation, and what it relates to.

    ``derived_from`` is empty for an authored assertion and carries the
    supporting ids for an inferred one, so a consumer can tell the two apart
    without asking where the assertion came from.

    ``polarity`` is what makes *not* statable. It defaults to ``ASSERTED``, so
    every assertion written before the field existed means what it always
    meant, and a reader that ignores it reads a vocabulary of positives.
    """

    id: str
    subject: str
    relation: RelationRef
    object: Term
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None
    derived_from: tuple[str, ...] = ()
    stale: bool = False
    #: Appended **last**, and that is load-bearing rather than tidy: this class
    #: shipped with eight fields, and any earlier position would move
    #: ``metadata`` under a caller who passes it positionally.
    polarity: Polarity = Polarity.ASSERTED

    def __post_init__(self) -> None:
        """Canonicalise the relation, so one fact has one spelling.

        The generated ``__eq__`` compares the field it was given, so the same
        stated fact written by a caller holding the definition compared
        unequal to one written by a caller holding the id. The loader never
        produced that pair -- it coerces with ``str()`` -- but a caller
        building an assertion from a ``RelationType`` it just looked up does.
        """
        self.relation = relation_id(self.relation)


class QualifiedId(NamedTuple):
    """The three parts of a namespaced id.

    A ``NamedTuple`` because every reader unpacks it --
    ``ontology, source, local = split_qualified(...)`` -- while the field names
    keep ``qid.source_id`` legible in a router that does not.
    """

    ontology_id: str
    source_id: str | None
    local_id: str


def qualify(ontology_id: str, local_id: str, source_id: str | None = None) -> str:
    """Build a namespaced id.

    Every site that constructs one calls this. An ``f"{a}:{b}"`` elsewhere is
    the defect, because a malformed id is unfixable once written into stored
    data.

    Args:
        ontology_id: The declaring ontology
        local_id: The id as authored, colons and all
        source_id: The binding, where the ontology binds more than one

    Returns:
        ``ontology:local``, or ``ontology:source:local`` when a source is given
    """
    if source_id is None:
        return f"{ontology_id}:{local_id}"
    return f"{ontology_id}:{source_id}:{local_id}"


def split_qualified(qualified_id: str, source_ids: Collection[str] = ()) -> QualifiedId:
    """Parse a namespaced id, directed by the sources actually in scope.

    Ontology and source ids reject ``:`` at load, so the head of each split is
    unambiguous; the middle segment is decided by *set membership* in the
    declared sources rather than by guessing, which is why the closed set is an
    argument. A parse that reached for a registry to find it would put this
    module in a cycle with the one that builds sources from config.

    The residual ambiguity is stated rather than papered over: a local id whose
    first segment happens to equal a sibling source's id parses as that source.
    Nothing at load can catch it, because it is a property of foreign row
    values and not of the config. It is tolerable because ids are
    *constructed* -- this parse is a convenience for the string-entry path, and
    anything internal that must be exact carries the three parts separately.

    Args:
        qualified_id: The id to parse
        source_ids: The source ids this ontology declares

    Returns:
        The three parts. ``source_id`` is None where no source segment applies
    """
    ontology_id, separator, remainder = qualified_id.partition(":")
    if not separator:
        return QualifiedId(ontology_id="", source_id=None, local_id=qualified_id)

    if len(source_ids) <= 1:
        return QualifiedId(ontology_id=ontology_id, source_id=None, local_id=remainder)

    head, separator, tail = remainder.partition(":")
    if separator and head in source_ids:
        return QualifiedId(ontology_id=ontology_id, source_id=head, local_id=tail)
    return QualifiedId(ontology_id=ontology_id, source_id=None, local_id=remainder)


class CyclePolicy(Enum):
    """What a projection does when the edges it walks form a cycle."""

    REPORT = "report"
    BREAK_AT_REVISIT = "break_at_revisit"


class SiblingOrder(Enum):
    """How the children of one node are ordered."""

    BY_NAME = "by_name"
    BY_INSERTION = "by_insertion"
    BY_METADATA = "by_metadata"
    DECLARED = "declared"


@dataclass(frozen=True)
class ProjectionContext:
    """Everything a :class:`ParentChoice` may consult, gathered before it runs.

    This is why ``choose`` can be synchronous. Both taxonomy flavours share one
    synchronous projection core, and a core that cannot await must be *handed*
    what its policies will read rather than letting them fetch it.
    """

    taxonomy_id: str
    relation: RelationRef
    roots: frozenset[str]
    depths: Mapping[str, int]
    types: Mapping[str, str]

    def __post_init__(self) -> None:
        """Canonicalise the relation; frozen, so through ``object.__setattr__``.

        Nothing in this package constructs a context -- the projection core
        hands one to a policy -- so the only caller who reaches this is a
        consumer writing a ``ParentChoice`` and a context to exercise it
        against, which is the caller least placed to notice that two contexts
        naming one relation compared unequal.
        """
        object.__setattr__(self, "relation", relation_id(self.relation))


@runtime_checkable
class ParentChoice(Protocol):
    """Picks one parent for a node that declares several.

    Returning ``None`` means *this policy has no opinion*, which is what lets
    policies compose without each having to know the others.
    """

    def choose(
        self, node_id: str, parents: Sequence[str], ctx: ProjectionContext
    ) -> str | None: ...


@dataclass(frozen=True)
class TreeProjection:
    """How a multi-parent axis is narrowed to a tree."""

    choice: ParentChoice
    on_cycle: CyclePolicy = CyclePolicy.REPORT
    order: SiblingOrder = SiblingOrder.BY_NAME
    order_key: str | None = None


@dataclass(frozen=True)
class Materialization:
    """Whether an axis's structure and content are written down or derived.

    Per axis rather than per taxonomy: one ontology may hold a materialized
    hierarchy beside an on-demand one. Text is omitted deliberately -- an index
    has no on-demand mode.

    **Both default to the live read, and the two defaults now have different
    reasons.** ``structure: materialized`` is honoured -- a loader door takes
    the copy once, at load -- so its default is a *choice*: the live read is
    what a hand-edited vocabulary wants, because the source is small and the
    freshness is free. ``content: materialized`` is a copy of every entity the
    axis covers and needs a store to hold it, which a module-level door binds
    none of; asking for it is refused at the accessor, naming the axis -- see
    :func:`~dataknobs_common.ontology.values._refuse_a_materialized_content_axis`.
    """

    structure: InferenceMode = InferenceMode.ON_DEMAND
    content: InferenceMode = InferenceMode.ON_DEMAND

    @property
    def structure_is_copied(self) -> bool:
        """Whether the structure axis is a snapshot rather than a live read.

        A named predicate rather than the comparison written twice, because the
        two readings of it are an *invariant pair*: a loader door builds a copy
        for exactly the definitions that say yes here, and
        :func:`~dataknobs_common.ontology.values._structure_for` demands one for
        exactly those. Two spellings of one question is how a door and an
        accessor come to disagree about which axes were copied.
        """
        return self.structure is InferenceMode.MATERIALIZED


@dataclass
class TaxonomyDefinition:
    """Everything a taxonomy is, independent of how its backings are driven.

    Shared by both flavours, and the whole of what serializes to config. The
    built axis is a different object: this is the *definition*, which is a
    value, and it is why an ontology can carry its taxonomies without having
    built any of them.
    """

    id: str
    relation: RelationRef
    name: str = ""
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    projection: TreeProjection | None = None
    materialization: Materialization = Materialization()

    def __post_init__(self) -> None:
        """Default the name to the id, and canonicalise the relation.

        The method predates the second line, which is why a static audit for
        *does this type canonicalise* answered yes about it: the body existed
        and never touched ``relation``.
        """
        if not self.name:
            self.name = self.id
        self.relation = relation_id(self.relation)

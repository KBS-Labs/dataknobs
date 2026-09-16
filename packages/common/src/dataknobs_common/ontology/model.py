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
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Protocol, runtime_checkable

# Re-exported, not used here: this module is a published import path for these
# three and stayed one when they moved. The redundant-alias spelling of a
# re-export is what `PLC0414` declines, so the directive names `F401` instead.
from dataknobs_common.entity_resolution.values import (  # noqa: F401
    CompatibilityVerdict,
    ResolutionRef,
    Scoring,
)
from dataknobs_common.fields import Field, FieldType

# The *same* key parameter the structure axis takes, imported rather than
# declared again. Two ``TypeVar``s with one bound and one default behave
# identically, and a second one would say that the two axes merely happen to
# agree; this says they are one key. The import runs this way only --
# ``hierarchy`` names no ontology type in either position, which is what keeps
# the general module free of the specific package.
from dataknobs_common.hierarchy import K

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


@dataclass(eq=True, frozen=False)
class SourceRef:
    """Where a projected entity or derived assertion came from.

    Data, never a handle. That is what lets it travel out of an ontology to a
    caller who *can* reach the row, from one that cannot.

    **Compared field-wise, and therefore unhashable.** Two references naming
    one row are one reference, and the suite says so -- so equality is the
    member that has to work. ``locator`` is a mapping, so a hash over the
    fields would raise; declaring the type unhashable is how a caller learns
    that from :class:`collections.abc.Hashable` rather than from the call.
    """

    source_id: str
    kind: str
    locator: Mapping[str, Any]
    projection_id: str | None = None


@dataclass(eq=True, frozen=False)
class Provenance(Generic[K]):
    """Who said a thing, on what evidence, and when.

    Generic in the entity key through one field: a resolution names the entity
    it resolved to, so a provenance carried by an ``Assertion[K]`` records one
    in that assertion's space rather than in ``str``.

    Compared field-wise, and unhashable, for the reason :class:`SourceRef`
    gives -- which is also why it is *this* record rather than a further one:
    it holds a ``SourceRef``, so a hash here would reach that mapping through
    it.
    """

    source: SourceRef | None = None
    extraction_confidence: float | None = None
    resolution: ResolutionRef[K] | None = None
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
class Entity(Generic[K]):
    """A thing the vocabulary names.

    ``type`` is itself an entity id, so the type system lives in the graph
    rather than beside it. An empty ``name`` means *use the id*, resolved once
    at construction so no reader has to remember the rule.

    **Generic in the key, defaulted to ``str``**, so a bare ``Entity`` is
    ``Entity[str]`` and every call site written before the parameter existed
    means what it meant. The key is what an ``EntitySource`` is addressed by,
    and it is this field that made the structure axis's genericity incoherent
    while it stayed ``str``: a hierarchy over a caller's own key beside an
    entity lookup that could not be asked about one.

    **``type`` is *not* the key**, and the asymmetry is the design rather than
    an omission. An entity's type is a row in ``entity_types:`` -- an authored
    declaration whose id the document writes as a string -- while an entity's
    *own* id may be whatever the consumer's records are keyed by. So
    :class:`EntityType` and :class:`RelationType` below bind the parameter to
    ``str`` outright: a schema id is a string wherever the instances came
    from.
    """

    id: K
    type: str
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: SourceRef | None = None

    def __post_init__(self) -> None:
        """Default the name to the id, rendered.

        **``str()`` here, and this is the one place in the package that renders
        a key without a codec.** The rule that forbids a default rendering
        elsewhere is about *identity*: a key is addressed by equality, a
        default ``__repr__`` is a function of identity, and two equal keys
        rendering differently would make one node address two entities. None of
        that reaches a **label**. ``name`` addresses nothing, nothing parses it
        back, and two equal keys yielding two spellings of a display string is
        a cosmetic oddity rather than a lost entity -- which is why the rule
        that needs an inverse needs a codec and this does not.

        For ``K = str`` -- every caller today -- ``str(self.id)`` *is*
        ``self.id``, so nothing observable moves. For a key whose repr is its
        identity the name is unhelpful and the caller supplies one, which the
        field already lets them do.
        """
        if not self.name:
            self.name = str(self.id)


@dataclass
class EntityType(Entity[str]):
    """An entity that describes a kind of entity.

    **Binds the key to ``str``**, because a type declaration is authored: its
    id is written in ``entity_types:`` and is a string whatever the instances
    it describes are keyed by. See :class:`Entity` for the asymmetry.

    The ``isa`` lattice between *types* is the config's ``isa:`` field and is a
    different store from the ``isa`` assertions between instances.
    """

    type: str = DK_ENTITY_TYPE
    attributes: list[AttributeDef] = field(default_factory=list)
    isa: str | None = None
    """The type this one specialises, or ``None`` for a root of the lattice.

    **A declared field rather than a documented key in** :attr:`metadata`,
    which is where it was parked while nothing read it. The parking was
    defensible then and is not now, for a reason measured rather than
    anticipated: ``metadata`` is an open dict a loader copies **wholesale**
    from the document, so while the parent lived in it a row could write one
    the loader's own check never saw -- that check reads the declared ``isa:``,
    and a parent written a level down inside ``metadata:`` reached the lattice
    without passing it. A field is the fix at the source: there is no second
    place a parent can be written, so there is nothing for a check to miss.

    Same shape and same reason as :attr:`RelationType.inverse_of` one class
    down -- a scalar reference to another declaration in the same section,
    validated by the loader when the document is read.

    Scalar rather than a collection: a type has at most one parent, which is
    what lets
    :meth:`~dataknobs_common.ontology.taxonomy.Taxonomy.inherited_attributes`
    walk without choosing an order between branches.
    """


@dataclass
class RelationType(Entity[str]):
    """An entity that describes a kind of edge.

    Binds the key to ``str`` for :class:`EntityType`'s reason.

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
class EntityRef(Generic[K]):
    """An assertion object that points at another entity.

    Generic for :class:`Assertion`'s sake: an assertion whose subject is a
    caller's key and whose object is a ``str`` would be an edge between two
    different spaces, which is not an edge.
    """

    entity_id: K


@dataclass(eq=True, frozen=False)
class Literal:
    """An assertion object that is a value rather than an entity.

    Compared field-wise, and unhashable: two literals carrying one value are
    one literal, which shipped tests assert, and ``metadata`` is a dict.
    """

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
#:
#: Generic in the key through its first member: ``Term[Sku]`` is a reference to
#: a ``Sku``-keyed entity or a literal, and a bare ``Term`` is ``Term[str]``. A
#: :class:`Literal` carries a value rather than a key and takes no parameter.
Term = EntityRef[K] | Literal


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
class Assertion(Generic[K]):
    """One stated fact: a subject, a relation, and what it relates to.

    **Generic in the entity key**, which reaches two of its fields and none of
    the others: ``subject`` names an entity and ``object`` may, so both move
    with the key. ``id`` and ``derived_from`` name *assertions*, which are
    minted where the assertion is written rather than supplied by a consumer,
    and ``relation`` names a declaration -- all three stay ``str`` for
    :class:`EntityType`'s reason.

    ``derived_from`` is empty for an authored assertion and carries the
    supporting ids for an inferred one, so a consumer can tell the two apart
    without asking where the assertion came from.

    ``polarity`` is what makes *not* statable. It defaults to ``ASSERTED``, so
    every assertion written before the field existed means what it always
    meant, and a reader that ignores it reads a vocabulary of positives.
    """

    id: str
    subject: K
    relation: RelationRef
    object: Term[K]
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance[K] | None = None
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


@dataclass(eq=True, frozen=False)
class ProjectionContext(Generic[K]):
    """Everything a :class:`ParentChoice` may consult, gathered before it runs.

    This is why ``choose`` can be synchronous. Both taxonomy flavours share one
    synchronous projection core, and a core that cannot await must be *handed*
    what its policies will read rather than letting them fetch it.

    **Compared field-wise, and therefore unhashable.** Equality is the whole
    reason ``__post_init__`` canonicalises the relation, so it is the member
    that has to work; ``depths`` and ``types`` are mappings, so a hash over
    the fields would raise.
    """

    taxonomy_id: str
    relation: RelationRef
    roots: frozenset[K]
    depths: Mapping[K, int]
    #: Node key to entity **type** id. The one mixed annotation here, and
    #: mixed for :class:`Entity`'s reason: the node is the consumer's key and
    #: the type is the schema's string.
    types: Mapping[K, str]

    def __post_init__(self) -> None:
        """Canonicalise the relation, so two spellings of one relation compare equal.

        Nothing in this package constructs a context -- the projection core
        hands one to a policy -- so the only caller who reaches this is a
        consumer writing a ``ParentChoice`` and a context to exercise it
        against, which is the caller least placed to notice that two contexts
        naming one relation compared unequal.
        """
        self.relation = relation_id(self.relation)


@runtime_checkable
class ParentChoice(Protocol[K]):
    """Picks one parent for a node that declares several.

    Returning ``None`` means *this policy has no opinion*, which is what lets
    policies compose without each having to know the others.

    Generic in the node key, defaulted to ``str``: a policy written before the
    parameter existed is a ``ParentChoice[str]`` and satisfies the protocol
    unchanged.
    """

    def choose(self, node_id: K, parents: Sequence[K], ctx: ProjectionContext[K]) -> K | None: ...


@dataclass(frozen=True)
class TreeProjection(Generic[K]):
    """How a multi-parent axis is narrowed to a tree.

    **Generic in the node key**, like the two types it is written over:
    :class:`ParentChoice` picks a parent from ``Sequence[K]`` and
    :class:`ProjectionContext` carries the roots and depths in the same space,
    so a holder that named the policy *bare* pinned both to ``str`` and left a
    consumer's non-``str`` axis unable to declare a projection over itself. The
    default keeps every existing spelling meaning what it meant.
    """

    choice: ParentChoice[K]
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
    #: ``TreeProjection[Any]`` rather than a bare one: this definition is the
    #: authored half of an axis and is ``str``-keyed whatever the *instances*
    #: are keyed by, so it cannot name the key its policy is written over.
    #: ``Any`` admits a projection over any of them; a bare name would admit
    #: only ``str``.
    projection: TreeProjection[Any] | None = None
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

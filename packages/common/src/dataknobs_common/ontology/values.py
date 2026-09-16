"""What a loaded ontology is: the neutral core, and the two flavours over it.

Three dataclasses. :class:`OntologyParts` is everything the two flavours share
and nothing a source must be opened for -- it is what validation produces, and
it is flavour-neutral because the only flavour-specific part of construction is
binding sources.

:class:`Ontology` and :class:`AsyncOntology` differ in exactly three of ten
fields, and all three differ the same way -- a flavoured backing in a
flavour-shaped slot. That is not a coincidence to be tidied away: an ontology
**holds sources, not entities**, so one shape serves both modes and the flavour
is fixed by which door was called rather than inferred from the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

from dataknobs_common.exceptions import NotFoundError, ValidationError
from dataknobs_common.hierarchy import K
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)
from dataknobs_common.ontology.model import InferenceMode, qualify as _qualify
from dataknobs_common.ontology.taxonomy import (
    AsyncTaxonomy,
    Taxonomy,
    _inherited_attributes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dataknobs_common.ontology.taxonomy import TaxonomyView
    from dataknobs_common.hierarchy import AsyncHierarchy, Hierarchy
    from dataknobs_common.ontology.model import (
        Assertion,
        AttributeDef,
        Entity,
        EntityType,
        RelationType,
        TaxonomyDefinition,
    )
    from dataknobs_common.ontology.sources import (
        AssertionSource,
        AsyncAssertionSource,
        AsyncEntitySource,
        EntitySource,
        SourceDescription,
    )

#: Whichever flavour of structure axis is being chosen between. Unbound,
#: because :func:`_structure_for` only ever *returns one of its arguments* --
#: it never reads a member, so the two flavours need nothing in common.
_S = TypeVar("_S")


@runtime_checkable
class KeyCodec(Protocol[K]):
    """How an entity key is written down when it leaves, and read back when it arrives.

    **The owner, the space and the qualification rule, as an object.** The
    content axis was pinned to ``str`` because an entity id carries all three
    and *"none of that survives being parameterised"*. That is true, and it is
    an argument about where the rule lives rather than about whether the axis
    can be generic: here the rule is a value the ontology holds, so it survives
    parameterisation by not being parameterised.

    **A pair rather than a rendering, because a shipped member needs the
    inverse.** :meth:`Ontology.localize` is documented as *what ``entity()``
    takes*; if ``entity()`` takes ``K``, ``localize`` returns ``K``, and that
    is a parse. ``repr()`` has no inverse.

    **There is no default for a non-``str`` key, and that is the whole safety
    property.** *A type that has a string representation* cannot be written as
    a bound, because in Python every type has one -- ``str``, ``int``,
    ``list``, a bare class and ``object()`` all satisfy a structural ``HasStr``
    -- and a bound narrow enough to discriminate would have to be nominal,
    which ``str`` itself would then fail. Nor can ``__repr__`` stand in: the
    key is addressed by **equality** and ``object.__repr__`` is a function of
    **identity**, so two equal keys would render to two strings and one node
    would address two entities, type-checked and silent.

    So it is an argument the type checker will not let you forget: the field on
    :class:`Ontology` is **required**, which is stronger than a defaulted one
    guarded by overloads and needs no overload to say it. Two constructions of
    an ontology exist in this package and both are doors here, so the cost of
    requiring it is two lines rather than a migration.

    Both members are positional-only: a codec is called by this package and its
    parameter names are not a surface a consumer writes.
    """

    def to_id(self, key: K, /) -> str:
        """What the key becomes when it leaves -- an index row, a stored ref."""
        ...

    def from_id(self, rendered: str, /) -> K:
        """What it is again at every door back in. The inverse of :meth:`to_id`."""
        ...


@dataclass(frozen=True)
class StrCodec:
    """The identity, and what ``K = str`` resolves to.

    Ships so that the ``str`` path costs a value rather than a decision: every
    ontology this package loads is keyed by the ids its document authored, and
    this is what those ids are.

    Frozen and stateless, so one instance would do; it is constructed per
    ontology anyway because a value that costs nothing is not worth a module
    singleton whose identity somebody might come to rely on.
    """

    def to_id(self, key: str, /) -> str:
        """The key, unchanged."""
        return key

    def from_id(self, rendered: str, /) -> str:
        """The rendering, unchanged."""
        return rendered


def _definition(taxonomies: Mapping[str, TaxonomyDefinition], name: str) -> TaxonomyDefinition:
    """The definition this ontology declares under ``name``.

    Refuses an undeclared name listing what *is* declared: the caller asked for
    an axis by a name they typed, and the useful answer to a typo is the set it
    was nearly one of.

    Shared by both flavours rather than written into each -- the refusals
    either twin makes are the same refusals, and a rule a twin re-implements is
    a rule that drifts.
    """
    definition = taxonomies.get(name)
    if definition is None:
        raise NotFoundError(
            f"no taxonomy {name!r} in this ontology. Declared: {sorted(taxonomies)}",
            context={"taxonomy": name, "declared": sorted(taxonomies)},
        )
    _refuse_a_materialized_content_axis(definition)
    return definition


def _refuse_a_materialized_content_axis(definition: TaxonomyDefinition) -> None:
    """Refuse an axis asking for a copy of every entity it covers.

    ``Materialization`` is per axis and reuses ``InferenceMode``: a
    ``MATERIALIZED`` axis is a copy with a build time rather than a live read.
    The two axes are no longer answered the same way, which is why only one is
    refused here:

    * **structure** materialized is ids and edges -- the cheap copy, and one a
      loader door takes for itself, once, at load. See :func:`_structure_for`
      for what an accessor then hands back and what it refuses;
    * **content** materialized is every entity on the axis. It is the expensive
      one and needs somewhere to live; an ontology loaded from a hand-edited
      file binds no store, so it is asking for something it cannot be given.

    Refused **naming the axis**, and refused here -- where the caller asked for
    the axis and can act on the answer -- rather than at the first read, which
    is a call site with no idea why it failed.

    What would lift it is a store to hold the copy, which is a binding this
    door does not make rather than an implementation nobody has written. Until
    there is one, refusing is what stands between a consumer and a silent
    downgrade: the alternative is prose saying the mode is recorded but not
    acted on, and that sentence has already been copied into its own opposite
    once.
    """
    if definition.materialization.content is InferenceMode.MATERIALIZED:
        raise ValidationError(
            f"taxonomy {definition.id!r} declares `materialization.content: "
            f"materialized`, which is a copy of every entity on the axis and "
            f"needs a store to hold it; this ontology binds none. Use "
            f"`content: on_demand`, which reads through the entity source",
            context={
                "taxonomy": definition.id,
                "axis": "content",
                "mode": InferenceMode.MATERIALIZED.value,
            },
        )


def _structure_for(
    name: str, definition: TaxonomyDefinition, structures: Mapping[str, _S], live: _S
) -> _S:
    """The structure axis this definition asks for: the copy, or the live read.

    Flavour-free, and it is what lets one rule serve both twins: the choice is
    between two objects the caller already holds, so there is nothing here to
    await and no reason for each flavour to decide it again.

    **Keyed by the name the axis was reached under**, not by ``definition.id``.
    A loader door keys both mappings the same way, so the two agree there --
    but ``taxonomies`` is a mapping a caller may build directly, and a
    definition filed under an alias would then be looked up in ``structures``
    by an id nothing had used as a key. The lookup name is the one the accessor
    was given, so it is the one both mappings answer to.

    ``live`` is built by the caller whether or not it is used. It is a frozen
    pair of references over fields the ontology is already holding, so the
    unused one costs an allocation -- and taking a callable instead would make
    the *one* thing this function does conditional on how it was called.

    **A definition asking for a copy that is not here is refused, not
    downgraded.** A loader door builds one for every such definition, so the
    gap is reachable only by constructing an ontology directly -- and quietly
    substituting the live axis there would hand back a different object than
    the config asked for, under the name the config used, which is the failure
    the materialization refusals exist to prevent.
    """
    if not definition.materialization.structure_is_copied:
        return live
    copied = structures.get(name)
    if copied is None:
        raise ValidationError(
            f"taxonomy {name!r} declares `materialization.structure: "
            f"materialized`, but this ontology carries no copy of that axis. A "
            f"loader door takes one at load; an ontology built directly must be "
            f"given it as `structures={{{name!r}: ...}}`, or declare "
            f"`structure: on_demand` to read the edges through the assertion "
            f"source",
            context={
                "taxonomy": name,
                "axis": "structure",
                "mode": InferenceMode.MATERIALIZED.value,
            },
        )
    return copied


def _localize(ontology_id: str, qualified_id: str) -> str:
    """``qualified_id`` in ``ontology_id``'s own space, or a refusal.

    Shared by both flavours, for :func:`_definition`'s reason: an id belongs
    to an ontology or it does not, and that is not a question a flavour
    changes.

    **It needs no source set, and saying why is the point.** The answer is
    always the remainder after the ontology segment -- ``partition(":")[2]``,
    the slice below -- and no source set can change it. Whether the middle
    segment names a declared source decides how
    :func:`~dataknobs_common.ontology.model.split_qualified` *parses* the id;
    it does not decide what this returns, because both readings put that
    segment on the local side of the ontology segment. So a bare local id
    comes back for the single-source case, and the source segment is kept
    where one applies, which is the space :attr:`Ontology.entities` speaks.

    Nothing is constructed here: the return is a slice of the argument. A
    caller who wants the three parts still asks the parser for them, with the
    declared set it needs -- ``split_qualified(qid, {d.source_id for d in
    onto.describes})``.
    """
    declared_by, separator, remainder = qualified_id.partition(":")
    if not separator or declared_by != ontology_id:
        raise ValidationError(
            f"{qualified_id!r} is not an id of ontology {ontology_id!r}: its "
            f"ontology segment is {declared_by!r}",
            context={
                "qualified_id": qualified_id,
                "ontology": ontology_id,
                "declared_by": declared_by,
            },
        )
    return remainder


@dataclass(frozen=True, eq=False)
class OntologyParts:
    """A validated config, mapped onto values, with no source bound yet.

    Six of :class:`Ontology`'s eleven fields verbatim, plus the declared rows
    and the still-unbound ``sources:`` specs. The five it omits are all
    downstream of binding: the two sources, the descriptions a door derives
    from them, the structure axes a door copies once they exist, and the codec
    a door chooses -- ``StrCodec`` for a document, whose ids are the strings
    its author typed.

    Returning this rather than an ontology is what lets one core serve every
    door: a core that inspected the config to decide which flavour to build
    would make the *return type* a function of the input, which is exactly the
    ambiguity two separate flavours exist to remove.

    **Frozen, and compared by identity**, for the reason :class:`Ontology`
    gives: this is a built value nobody compares field-wise, and several of
    its fields are mappings that no amount of freezing makes hashable.
    """

    id: str
    version: str
    entity_types: Mapping[str, EntityType]
    relation_types: Mapping[str, RelationType]
    taxonomies: Mapping[str, TaxonomyDefinition]
    imports: tuple[str, ...]
    declared_entities: Mapping[str, Entity]
    declared_assertions: tuple[Assertion, ...]
    source_specs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, eq=False)
class Ontology(Generic[K]):
    """A loaded vocabulary whose backings are synchronous.

    A value with accessors. Every member below is pure over the fields --
    nothing here opens anything, and nothing is lazy -- which is what makes an
    ontology safe to hold, share and pass without owning a lifecycle.

    **Frozen, and compared by identity.** Frozen because nothing mutates a
    loaded vocabulary: a different one is a different ``load_ontology`` call.
    Identity because field-wise equality would generate a ``__hash__`` reaching
    ``entity_types`` and ``taxonomies``, which are mappings -- so the type
    would satisfy :class:`collections.abc.Hashable` and raise at the call,
    which is worse than never claiming it. Two ontologies loaded from one
    document are two vocabularies, not one, and nothing in this package
    compares them.
    """

    id: str
    version: str
    entity_types: Mapping[str, EntityType]
    relation_types: Mapping[str, RelationType]
    entities: EntitySource[K]
    assertions: AssertionSource[K]
    taxonomies: Mapping[str, TaxonomyDefinition]
    describes: tuple[SourceDescription, ...]

    #: How this vocabulary's keys are written down and read back --
    #: :class:`KeyCodec`. **Required, and that is the guard**: a defaulted one
    #: would let an ``Ontology[Sku]`` be built carrying the identity codec,
    #: which type-checks and renders a ``Sku`` as its ``repr``. Pass
    #: ``StrCodec()`` for the ``str`` case, which is what both doors here do.
    #:
    #: **Keyword-only, and still required** -- ``kw_only`` decides how an
    #: argument may be *passed*, not whether it may be omitted, so this keeps
    #: the guard above whole while removing the one thing it could not guard
    #: against. A required field may not follow a defaulted one, so this could
    #: not simply be appended the way :attr:`Taxonomy.entity_types` was; it
    #: therefore lands at position nine, where :attr:`structures` used to be,
    #: and a nine-positional construction would bind a structures mapping to a
    #: codec. Both are objects, so nothing reports it at the construction and
    #: the first symptom is a rendered key somewhere else entirely. Nothing
    #: released depends on the order and all four construction sites in this
    #: repository are keyword, so the fix is to make the misbinding
    #: unspellable rather than to preserve a position.
    codec: KeyCodec[K] = field(kw_only=True)

    #: The copied structure axes, keyed by the name the axis is reached
    #: under -- the key in :attr:`taxonomies`, which an alias may spell
    #: differently from ``definition.id``.
    #:
    #: One entry for every definition declaring ``materialization.structure:
    #: materialized``, and none for the rest -- an on-demand axis is built per
    #: call from the assertion source, which is what makes it on-demand. A
    #: loader door fills this; a caller building an ontology directly may fill
    #: it with any :class:`~dataknobs_common.hierarchy.Hierarchy`, and is
    #: refused at :meth:`taxonomy` if they declare a copy and supply none.
    structures: Mapping[str, Hierarchy[K]] = field(default_factory=dict)

    #: The ontology ids this vocabulary declares as ``imports:``, **carried
    #: and never followed**. Resolving a reference across an import needs a
    #: second ontology in scope, which a door loading one file does not have;
    #: dropping the list left the component that *does* have one unable to see
    #: what to resolve against, so it travels even though nothing here spends
    #: it.
    imports: tuple[str, ...] = ()

    def entity(self, entity_id: K) -> Entity[K] | None:
        """The entity with this id, by way of :attr:`entities`.

        An accessor *invokes* the one way rather than reproducing it: this is
        one line, and it is one line on purpose. A second lookup here that
        happened to agree with the source today is the failure this shape
        rules out.
        """
        return self.entities.get(entity_id)

    def by_surface_form(self, form: str) -> frozenset[K]:
        """The ids matching this form, by way of :attr:`entities`.

        ``form`` is surface text a person typed and stays ``str``; what comes
        back is keys. One signature, one annotation of each kind.
        """
        return self.entities.by_surface_form(form)

    def longest_form_tokens(self) -> int | None:
        """The longest declared form's token count, by way of :attr:`entities`.

        ``None`` where the source cannot bound a window at all -- see
        :meth:`~dataknobs_common.ontology.MappingEntitySource.longest_form_tokens`.
        """
        return self.entities.longest_form_tokens()

    def taxonomy(self, name: str) -> Taxonomy[K]:
        """The axis this ontology declares under ``name``, built.

        Built from three fields this object already holds and **no consumer
        input**, which is what keeps it an accessor rather than a factory: the
        structure is this ontology's assertions read for one relation, the
        content is its entity source, and the assertions travel along so a
        cursor over the axis can report the edge it walked.

        **Four fields now, and the fourth is the type store**: an axis that
        could not be asked what a type inherits was an axis holding three of
        the ontology's four relevant fields. The accessor still takes nothing
        but the name, which is what keeps it an accessor.

        The structure is the live read unless the definition asked for a copy,
        in which case it is the one this ontology is carrying -- see
        :func:`_structure_for`. Refuses, naming the axis, a definition whose
        ``materialization`` asks for something this ontology cannot supply --
        see :func:`_refuse_a_materialized_content_axis`.
        """
        definition = _definition(self.taxonomies, name)
        return Taxonomy(
            definition=definition,
            structure=_structure_for(
                name,
                definition,
                self.structures,
                AssertionHierarchy(self.assertions, definition.relation),
            ),
            entities=self.entities,
            assertions=self.assertions,
            entity_types=self.entity_types,
        )

    def inherited_attributes(self, entity_type: str) -> list[AttributeDef]:
        """The attribute declarations ``entity_type`` may be asked for, nearest first.

        **On the object that owns the store it reads.** The walk is over
        :attr:`entity_types` and nothing else -- not :attr:`assertions`, not a
        structure axis -- which is why the same question put to any taxonomy of
        this vocabulary comes back with the same answer. Reaching it only
        through :meth:`taxonomy` would mean building an axis to ask a question
        that is purely about this object's own field, and choosing which axis
        to build would be choosing something the answer does not depend on.

        :meth:`~dataknobs_common.ontology.taxonomy.Taxonomy.inherited_attributes`
        keeps its place and is the same walk over the same store: that is the
        surface a schema projector holding an axis can reach without coming back
        here, and this is the one a caller holding the vocabulary can reach
        without building an axis. Neither is a second implementation -- both are
        one line over the shared walk.

        Args:
            entity_type: The type to read the lattice up from.

        Returns:
            Nearest declaration first, one entry per attribute **name**: a
            nearer declaration shadows a farther one of the same name, because
            a subtype redeclaring a field is specialising it. ``[]`` for a type
            declared with nothing, which is a different answer from the refusal
            below and is why the two are kept apart.

        Raises:
            NotFoundError: Naming a type this vocabulary does not declare --
                whether it is the one asked about or one reached from it.
        """
        return _inherited_attributes(self.entity_types, self.id, entity_type)

    def qualify(self, local_id: K, source_id: str | None = None) -> str:
        """This ontology's external id for ``local_id``.

        Fills the ontology segment from :attr:`id` and composes the rest
        through the free
        :func:`~dataknobs_common.ontology.model.qualify`, which is where the
        spelling of a namespaced id lives. An ``f"{a}:{b}"`` here would be a
        second spelling of it, and a malformed id is unfixable once it is
        written into stored data.

        **One of the four doors a key leaves by**, so it is one of the four
        places :attr:`codec` is spent: the key is rendered here and nowhere
        between here and the value types, which simply carry it.
        """
        return _qualify(self.id, self.codec.to_id(local_id), source_id)

    def localize(self, qualified_id: str) -> K:
        """``qualified_id`` in this ontology's own space -- what :meth:`entity` takes.

        For a single-source ontology that is the bare local id, which is every
        authored vocabulary; for a multi-source one it keeps the source
        segment, because that is the space :attr:`entities` speaks. The name
        invites the first reading, so the second is stated here and asserted
        in a test rather than left to be discovered.

        Refuses an id belonging to another ontology, **naming both**. That
        refusal is what these two members have that the free functions do not:
        only an ontology knows whose ids it is parsing. See :func:`_localize`.

        **The inverse door to :meth:`qualify`**, and the reason the codec is a
        pair rather than a rendering: this member's documented contract is
        *what ``entity()`` takes*, and ``entity()`` takes ``K``. A rendering
        with no inverse could not satisfy it.

        **A codec's own failures are the codec's**, and reach the caller
        unwrapped. This member raises :class:`ValidationError` for an id
        belonging to another ontology -- the part it can judge -- and then hands
        the local segment to :attr:`codec`, whose ``from_id`` is a consumer's
        function parsing a consumer's key space. An ``int("abc")`` in there
        arrives as ``ValueError``. Wrapping it would put this package's name on
        a diagnosis it did not make and cannot improve: the codec knows what
        shape it expected and this frame does not.

        Raises:
            ValidationError: For an id qualified by a different ontology,
                naming both. Whatever :attr:`codec` raises, unchanged, for a
                local segment it cannot read.
        """
        return self.codec.from_id(_localize(self.id, qualified_id))


@dataclass(frozen=True, eq=False)
class AsyncOntology(Generic[K]):
    """The same eleven fields, with asynchronous backings.

    ``entities`` is an :class:`~dataknobs_common.ontology.sources.AsyncEntitySource`
    and ``assertions`` an
    :class:`~dataknobs_common.ontology.sources.AsyncAssertionSource`; everything
    else is identical, including that this is a value and owns nothing.

    Frozen and identity-compared for the reason :class:`Ontology` states, and
    stated here too because a difference between the twins in *how they are
    declared* is one :func:`~dataknobs_common.testing.assert_twin_types_agree`
    cannot see: it reads members, and this is a decision about the type.
    """

    id: str
    version: str
    entity_types: Mapping[str, EntityType]
    relation_types: Mapping[str, RelationType]
    entities: AsyncEntitySource[K]
    assertions: AsyncAssertionSource[K]
    taxonomies: Mapping[str, TaxonomyDefinition]
    describes: tuple[SourceDescription, ...]

    #: :attr:`Ontology.codec`, and required **and keyword-only** here for the
    #: same two reasons. A twin that took it positionally would be the one
    #: place the misbinding stayed spellable.
    codec: KeyCodec[K] = field(kw_only=True)

    #: :attr:`Ontology.structures`, in an asynchronous slot.
    structures: Mapping[str, AsyncHierarchy[K]] = field(default_factory=dict)

    #: :attr:`Ontology.imports`, unflavoured -- a list of ids awaits nothing.
    imports: tuple[str, ...] = ()

    async def entity(self, entity_id: K) -> Entity[K] | None:
        """The entity with this id, by way of :attr:`entities`."""
        return await self.entities.get(entity_id)

    async def by_surface_form(self, form: str) -> frozenset[K]:
        """The ids matching this form, by way of :attr:`entities`."""
        return await self.entities.by_surface_form(form)

    def longest_form_tokens(self) -> int | None:
        """The longest declared form's token count, by way of :attr:`entities`.

        A plain ``def`` on this twin too -- the source answers it without
        awaiting, so there is nothing here to suspend for. ``None`` carries
        the same meaning as on the synchronous twin.
        """
        return self.entities.longest_form_tokens()

    def taxonomy(self, name: str) -> AsyncTaxonomy[K]:
        """The axis this ontology declares under ``name``, built.

        A plain ``def`` on this twin, and deliberately: it constructs over
        fields the object is already holding and awaits nothing. Making it
        awaitable would cost every caller an ``await`` for a lookup and three
        assignments -- and it is the reason a copied axis is taken at the door
        rather than here, since the asynchronous snapshot *is* a coroutine and
        there is nowhere in this signature to await it.
        """
        definition = _definition(self.taxonomies, name)
        return AsyncTaxonomy(
            definition=definition,
            structure=_structure_for(
                name,
                definition,
                self.structures,
                AsyncAssertionHierarchy(self.assertions, definition.relation),
            ),
            entities=self.entities,
            assertions=self.assertions,
            entity_types=self.entity_types,
        )

    def inherited_attributes(self, entity_type: str) -> list[AttributeDef]:
        """The attribute declarations ``entity_type`` may be asked for, nearest first.

        **On the object that owns the store it reads.** The walk is over
        :attr:`entity_types` and nothing else -- not :attr:`assertions`, not a
        structure axis -- which is why the same question put to any taxonomy of
        this vocabulary comes back with the same answer. Reaching it only
        through :meth:`taxonomy` would mean building an axis to ask a question
        that is purely about this object's own field, and choosing which axis
        to build would be choosing something the answer does not depend on.

        :meth:`~dataknobs_common.ontology.taxonomy.Taxonomy.inherited_attributes`
        keeps its place and is the same walk over the same store: that is the
        surface a schema projector holding an axis can reach without coming back
        here, and this is the one a caller holding the vocabulary can reach
        without building an axis. Neither is a second implementation -- both are
        one line over the shared walk.

        **A plain ``def`` on this twin**, for the reason its
        :meth:`taxonomy` gives: the store is a mapping this object is
        already holding, so there is nothing here to suspend for.

        Args:
            entity_type: The type to read the lattice up from.

        Returns:
            Nearest declaration first, one entry per attribute **name**: a
            nearer declaration shadows a farther one of the same name, because
            a subtype redeclaring a field is specialising it. ``[]`` for a type
            declared with nothing, which is a different answer from the refusal
            below and is why the two are kept apart.

        Raises:
            NotFoundError: Naming a type this vocabulary does not declare --
                whether it is the one asked about or one reached from it.
        """
        return _inherited_attributes(self.entity_types, self.id, entity_type)

    def qualify(self, local_id: K, source_id: str | None = None) -> str:
        """:meth:`Ontology.qualify`, unflavoured.

        A plain ``def`` on this twin as well, and for the same reason
        :meth:`taxonomy` is one: it reads two fields and awaits nothing, so
        making it awaitable would cost every caller an ``await`` for a string
        concatenation. One of the four doors, so the codec is spent here.
        """
        return _qualify(self.id, self.codec.to_id(local_id), source_id)

    def localize(self, qualified_id: str) -> K:
        """:meth:`Ontology.localize`, unflavoured."""
        return self.codec.from_id(_localize(self.id, qualified_id))


if TYPE_CHECKING:  # pragma: no cover - checked by the type checker, not run
    from dataclasses import dataclass as _dataclass
    from typing import assert_type

    def _a_consumer_may_bind_a_key_of_their_own() -> None:
        """The whole point of the widening, written as the type checker sees it.

        **This is the red half of the change, and it is red at the type level
        rather than at runtime.** Every call site in this repository binds
        ``K = str``, so a widening that reached the members and not the values
        they carry compiled verbatim here and did not type-check for the first
        consumer who bound anything else. The existing call sites were never
        who the widening was for; this function is.

        Written here rather than as a test for the reason
        :func:`~dataknobs_common.hierarchy._the_key_defaults_to_str` gives:
        this file is type-checked and the test tree is not.

        The codec is what makes it composable rather than merely declarable.
        A ``Sku`` has no useful ``__repr__`` contract -- it is addressed by
        equality and rendered by agreement -- so the rendering is an argument,
        and the field is required so that it cannot be an omission.
        """

        @_dataclass(frozen=True)
        class Sku:
            """A consumer's own key: hashable, value-equal, and not a ``str``."""

            plant: str
            line: int

        class SkuCodec:
            """Their rendering, and its inverse. Two functions, and both are theirs."""

            def to_id(self, key: Sku, /) -> str:
                return f"{key.plant}-{key.line}"

            def from_id(self, rendered: str, /) -> Sku:
                plant, _, line = rendered.rpartition("-")
                return Sku(plant=plant, line=int(line))

        def _composes(
            entities: EntitySource[Sku],
            assertions: AssertionSource[Sku],
            definitions: Mapping[str, TaxonomyDefinition],
            described: tuple[SourceDescription, ...],
            types: Mapping[str, EntityType],
            relations: Mapping[str, RelationType],
        ) -> None:
            onto = Ontology(
                id="acme",
                version="1.0",
                entity_types=types,
                relation_types=relations,
                entities=entities,
                assertions=assertions,
                taxonomies=definitions,
                describes=described,
                codec=SkuCodec(),
            )
            assert_type(onto, "Ontology[Sku]")

            # The content axis answers in the caller's space, all the way down
            # to what the entity itself says its id is.
            assert_type(onto.entity(Sku("ACME", 3)), "Entity[Sku] | None")
            assert_type(onto.by_surface_form("line three"), "frozenset[Sku]")

            # The two doors: out through the codec, and back through it. The
            # inverse is why the codec is a pair -- `localize` is documented as
            # *what entity() takes*, and entity() takes a Sku.
            assert_type(onto.qualify(Sku("ACME", 3)), "str")
            assert_type(onto.localize("acme:ACME-3"), "Sku")

            # And the layer above: the axis, its cursor, and the walk members
            # this leg put on the cursor, all in the same space.
            axis = onto.taxonomy("lines")
            assert_type(axis, "Taxonomy[Sku]")
            assert_type(axis.subtree_keys(Sku("ACME", 3)), "list[Sku]")
            assert_type(axis.at(Sku("ACME", 3)), "TaxonomyView[Sku]")

        def _the_cursor_walks_in_that_space(cursor: TaxonomyView[Sku]) -> None:
            """And the members this leg put on it answer in it too."""
            assert_type(cursor.ancestors(), "tuple[TaxonomyView[Sku], ...]")
            assert_type(cursor.descendants_to_depth(2), "tuple[TaxonomyView[Sku], ...]")
            assert_type(cursor.paths_to_root(), "tuple[tuple[Sku, ...], ...]")
            assert_type(cursor.entity(), "Entity[Sku] | None")

        def _the_str_case_is_unchanged(onto: Ontology) -> None:
            """A bare ``Ontology`` is ``Ontology[str]``, as everything else here is.

            The half that regresses silently: without the default on the key
            parameter a bare annotation would be ``Ontology[Any]``,
            ``disallow_any_generics`` is not set in this repository, and a
            wrong-typed key would then type-check with nothing to report it.
            """
            assert_type(onto, "Ontology[str]")
            assert_type(onto.localize("acme:beagle"), "str")
            assert_type(onto.taxonomy("species").at("beagle").entity(), "Entity[str] | None")

        del _composes, _the_cursor_walks_in_that_space, _the_str_case_is_unchanged

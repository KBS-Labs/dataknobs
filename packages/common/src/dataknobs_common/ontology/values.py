# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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

    from dataknobs_common.hierarchy import AsyncHierarchy, Hierarchy
    from dataknobs_common.ontology.model import (
        Assertion,
        AttributeDef,
        Entity,
        EntityType,
        ParentChoice,
        RelationType,
        TaxonomyDefinition,
        TreeProjection,
    )
    from dataknobs_common.ontology.taxonomy import TaxonomyView
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

    **The space is the ontology's own, which is the whole post-ontology
    remainder and not the bare local id.** Those differ exactly when a
    vocabulary binds more than one source: :meth:`Ontology.localize` keeps the
    source segment then, because that is the space :attr:`Ontology.entities`
    speaks and what a layered source routes on -- and ``localize`` is the
    member documented as *what* ``entity()`` *takes*. So ``from_id`` receives
    that whole remainder, and ``to_id`` must therefore produce it. A codec for
    a multi-source vocabulary renders and parses the segment; one for a
    single-source vocabulary never sees it, because there is none.

    Saying so is not a formality. It is the half that was missing while
    :meth:`Ontology.qualify` took a ``source_id`` and composed a segment behind
    the codec, which handed ``from_id`` a string ``to_id`` had never produced.
    Over ``str`` nothing showed, because the identity codec parses anything.

    Both members are positional-only: a codec is called by this package and its
    parameter names are not a surface a consumer writes.
    """

    def to_id(self, key: K, /) -> str:
        """What the key becomes when it leaves -- an index row, a stored ref.

        The whole of the ontology's local space, which carries the source
        segment for a vocabulary that binds more than one.
        """
        ...

    def from_id(self, rendered: str, /) -> K:
        """What it is again at every door back in. The inverse of :meth:`to_id`.

        It receives exactly what :meth:`to_id` produced, and this package holds
        no door that hands it anything else.
        """
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


def _declared(taxonomies: Mapping[str, TaxonomyDefinition], name: str) -> TaxonomyDefinition:
    """The definition this ontology declares under ``name``, and nothing more.

    Refuses an undeclared name listing what *is* declared: the caller asked for
    an axis by a name they typed, and the useful answer to a typo is the set it
    was nearly one of.

    Shared by both flavours rather than written into each -- the refusals
    either twin makes are the same refusals, and a rule a twin re-implements is
    a rule that drifts.

    **The lookup alone**, which is what lets a caller wanting one *half* of an
    axis ask for it without meeting the other half's refusal. See
    :func:`_definition` for the whole, and :meth:`Ontology.structure_for` for
    the caller that needed the half.
    """
    definition = taxonomies.get(name)
    if definition is None:
        raise NotFoundError(
            f"no taxonomy {name!r} in this ontology. Declared: {sorted(taxonomies)}",
            context={"taxonomy": name, "declared": sorted(taxonomies)},
        )
    return definition


def _definition(taxonomies: Mapping[str, TaxonomyDefinition], name: str) -> TaxonomyDefinition:
    """:func:`_declared`, plus the refusals *building the whole axis* owes.

    What ``taxonomy()`` asks, and the pair is split because the content
    refusal is about an axis's **content** -- so a caller who wants only its
    structure meets a refusal about a half they did not ask for.

    That is not a hypothetical caller. ``OntologyRegistry`` records each axis's
    node set at load, which is a question about the structure alone, and asking
    it through this door made a ``content: materialized`` document raise out of
    ``load()`` -- at a call site with no idea why it failed, which is the exact
    placement :func:`_refuse_a_materialized_content_axis` gives its reason for
    avoiding.
    """
    definition = _declared(taxonomies, name)
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
    """The structure axis for this name: whatever a door bound, else the live read.

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

    **The mapping is consulted first, and that is what makes a *bound* axis
    reachable at all.** ``structures`` holds the axes a **door bound**, of
    which a snapshot is one kind: a door that read a ``parent_id`` column, or a
    caller who built an axis of their own, puts it here, and the accessor hands
    back what it was given under the name it was given it under. The mode is
    then a question about what a door had to do *first* --
    ``materialization.structure: materialized`` is what makes a door take a
    snapshot before filing the axis, and ``on_demand`` is what leaves it live
    -- rather than a gate on whether this mapping is read at all. A live axis
    means ``on_demand``, so a gate on the mode had no way to admit one.

    **A definition asking for a copy that is not here is still refused, not
    downgraded**, and the condition is unchanged: no entry, and a copy asked
    for. A loader door files one for every such definition, so the gap is
    reachable only by constructing an ontology directly -- and quietly
    substituting the live axis there would hand back a different object than
    the config asked for, under the name the config used, which is the failure
    the materialization refusals exist to prevent. Handing back the entry the
    caller *did* supply is the opposite act, and it is what already happened
    whenever the definition said ``materialized``.
    """
    bound = structures.get(name)
    if bound is not None:
        return bound
    if not definition.materialization.structure_is_copied:
        return live
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
    and the still-unbound ``sources:`` and ``taxonomies:`` specs. The five it
    omits are all downstream of binding: the two sources, the descriptions a
    door derives from them, the structure axes a door binds or copies once the
    backings exist, and the codec a door chooses -- ``StrCodec`` for a
    document, whose ids are the strings its author typed.

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

    #: The ``taxonomies:`` rows as written, carried for the same reason
    #: :attr:`source_specs` is: a row may name a backing this door binds no
    #: implementation of, and the door that does binds it in another
    #: distribution. Without the field that door would have to re-read
    #: ``config.taxonomies`` for itself, which is a second reader of one
    #: section and the first place two readings of it could differ --
    #: :attr:`taxonomies` above is what the rows *mean*, and these are the
    #: rows.
    taxonomy_specs: tuple[Mapping[str, Any], ...]


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
    #: not simply be appended the way :attr:`Taxonomy.entity_types` was.
    #: Without ``kw_only`` it would land at position nine, where
    #: :attr:`structures` still is, and a nine-positional construction would
    #: bind a structures mapping to a codec -- both are objects, so nothing
    #: would report it at the construction and the first symptom would be a
    #: rendered key somewhere else entirely. ``kw_only`` takes the field out of
    #: the positional list altogether, so ``structures`` keeps its position and
    #: the misbinding is unspellable rather than merely unlikely.
    codec: KeyCodec[K] = field(kw_only=True)

    #: The structure axes a **door bound**, keyed by the name the axis is
    #: reached under -- the key in :attr:`taxonomies`, which an alias may spell
    #: differently from ``definition.id``.
    #:
    #: *What a door bound*, rather than *what a door copied*: a snapshot is one
    #: kind of bound axis and a live backing over rows is another, and
    #: :meth:`taxonomy` hands back whatever is here under the name it is filed
    #: under. What the definition's ``materialization.structure`` decides is
    #: what a door must do **before** filing it -- ``materialized`` takes a
    #: snapshot, ``on_demand`` does not -- and an axis with no entry at all is
    #: built per call from the assertion source, which is what makes it
    #: on-demand.
    #:
    #: A loader door fills this; a caller building an ontology directly may
    #: fill it with any :class:`~dataknobs_common.hierarchy.Hierarchy`, and is
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

        The structure is whatever a door bound under this name, and the live
        read over the assertions where a door bound nothing -- see
        :func:`_structure_for`. Refuses, naming the axis, a definition whose
        ``materialization`` asks for something this ontology cannot supply --
        see :func:`_refuse_a_materialized_content_axis`.

        Raises:
            NotFoundError: Naming an axis this ontology does not declare. The
                message lists the ones it does, because a caller that got the
                name wrong is usually one edit away from the right one.
            ValidationError: On a declared axis whose ``materialization`` asks
                for a copy of every entity on it, which needs a store this
                ontology was not given -- see
                :func:`_refuse_a_materialized_content_axis`; and on one
                declaring ``materialization.structure: materialized`` that
                this ontology carries no copy of, which escapes through
                :meth:`structure_for` -- see :func:`_structure_for`. A loader
                door files a copy for every definition that asks for one, so
                the second is reached by building the vocabulary directly.
        """
        return Taxonomy(
            definition=_definition(self.taxonomies, name),
            structure=self.structure_for(name),
            entities=self.entities,
            assertions=self.assertions,
            entity_types=self.entity_types,
        )

    def structure_for(self, name: str) -> Hierarchy[K]:
        """The structure axis this ontology answers ``name`` with, and only that.

        :meth:`taxonomy` reads it, and so does anything that wants an axis's
        *shape* without its content -- enumerating its nodes to compare two
        loads of one document is the case that drove publishing it. Such a
        caller asking through :meth:`taxonomy` meets
        :func:`_refuse_a_materialized_content_axis`, which is a refusal about
        a half of the axis they never touch.

        Whatever a door bound under this name, and the live read over the
        assertions where a door bound nothing -- one rule, in
        :func:`_structure_for`, so that what a subscriber holds and what a
        reporter counts cannot be two different axes.

        Args:
            name: The name the axis is reached under, as :attr:`taxonomies`
                keys it

        Returns:
            The axis, live or bound

        Raises:
            NotFoundError: When no taxonomy is declared under ``name``
            ValidationError: When the definition asks for a copy of its
                structure that this ontology does not carry
        """
        definition = _declared(self.taxonomies, name)
        return _structure_for(
            name,
            definition,
            self.structures,
            AssertionHierarchy(self.assertions, definition.relation),
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

    def qualify(self, local_id: K) -> str:
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

        **It takes the key and nothing else, and the parameter it used to take
        is why.** A ``source_id`` here composed a segment *inside* the space
        :attr:`codec` owns -- :meth:`localize` hands the whole post-ontology
        remainder to ``from_id``, so a segment this member prepended arrived at
        a parser with no contract to expect it. Over ``str`` that was invisible,
        because ``StrCodec`` is the identity and every string parses; over a
        key of a consumer's own it produced a key that addresses nothing, with
        nothing raised. Dropping the parameter makes the pair inverses
        unconditionally: **every id this builds, :meth:`localize` reads back.**

        A caller holding the parts separately still composes them with the free
        :func:`~dataknobs_common.ontology.model.qualify`, which is what it is
        for -- ``qualify(onto.id, local, source)``. A caller holding a key of
        this vocabulary calls this, and the key already carries whatever the
        source segment would have said, because that is the space
        :attr:`entities` speaks.
        """
        return _qualify(self.id, self.codec.to_id(local_id))

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

    #: :attr:`Ontology.structures`, in an asynchronous slot -- and the slot a
    #: live backing over rows arrives in, since binding one needs a handle and
    #: a handle is what the asynchronous flavour exists for.
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
        assignments -- and it is the reason an axis is **bound** at the door
        rather than here, whether binding it means opening a handle or taking
        an asynchronous snapshot. Both are coroutines, and there is nowhere in
        this signature to await one.

        Restating rather than pointing, so the clauses are restated too: the
        unit is the contract, not the flavour.

        Raises:
            NotFoundError: Naming an axis this ontology does not declare.
            ValidationError: On a declared axis whose ``materialization`` asks
                for a copy of every entity on it, which needs a store this
                ontology was not given; and on one declaring
                ``materialization.structure: materialized`` that this ontology
                carries no copy of, which escapes through
                :meth:`structure_for`. A loader door files a copy for every
                definition that asks for one, so the second is reached by
                building the vocabulary directly.
        """
        return AsyncTaxonomy(
            definition=_definition(self.taxonomies, name),
            structure=self.structure_for(name),
            entities=self.entities,
            assertions=self.assertions,
            entity_types=self.entity_types,
        )

    def structure_for(self, name: str) -> AsyncHierarchy[K]:
        """:meth:`Ontology.structure_for`, in an asynchronous slot.

        A plain ``def`` for :meth:`taxonomy`'s reason: choosing between two
        objects this ontology is already holding awaits nothing, and the
        binding that *did* await something happened at the door.
        """
        definition = _declared(self.taxonomies, name)
        return _structure_for(
            name,
            definition,
            self.structures,
            AsyncAssertionHierarchy(self.assertions, definition.relation),
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

    def qualify(self, local_id: K) -> str:
        """:meth:`Ontology.qualify`, unflavoured.

        A plain ``def`` on this twin as well, and for the same reason
        :meth:`taxonomy` is one: it reads two fields and awaits nothing, so
        making it awaitable would cost every caller an ``await`` for a string
        concatenation. One of the four doors, so the codec is spent here.
        """
        return _qualify(self.id, self.codec.to_id(local_id))

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
            assert_type(cursor.children_at_depth(2), "tuple[TaxonomyView[Sku], ...]")
            assert_type(cursor.paths_to_root(), "tuple[tuple[Sku, ...], ...]")
            assert_type(cursor.entity(), "Entity[Sku] | None")

        def _a_projection_may_be_written_over_that_key_too(
            choice: ParentChoice[Sku],
        ) -> None:
            """What widening :class:`TreeProjection` buys, written as a call.

            :attr:`TaxonomyDefinition.projection` is ``TreeProjection[Any]``
            because the definition is the *authored* half of an axis and is
            ``str``-keyed whatever the instances are -- so it cannot name the
            key its policy is written over. ``Any`` is what admits a policy over
            any of them; a **bare** ``TreeProjection`` would admit only ``str``,
            because the key parameter defaults. That is the whole of the
            widening's reach, and it is reachable rather than notional: this
            assignment is an error without it.
            """
            declared: TaxonomyDefinition = TaxonomyDefinition(
                id="lines",
                relation="isa",
                projection=TreeProjection(choice=choice),
            )
            assert_type(declared.projection, "TreeProjection[Any] | None")

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
        del _a_projection_may_be_written_over_that_key_too

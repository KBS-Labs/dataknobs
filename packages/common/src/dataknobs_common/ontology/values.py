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
from typing import TYPE_CHECKING, Any, TypeVar

from dataknobs_common.exceptions import NotFoundError, ValidationError
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)
from dataknobs_common.ontology.model import InferenceMode, qualify as _qualify
from dataknobs_common.ontology.taxonomy import AsyncTaxonomy, Taxonomy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dataknobs_common.hierarchy import AsyncHierarchy, Hierarchy
    from dataknobs_common.ontology.model import (
        Assertion,
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

    Six of :class:`Ontology`'s ten fields verbatim, plus the declared rows
    and the still-unbound ``sources:`` specs. The four it omits are all
    downstream of binding: the two sources, the descriptions a door derives
    from them, and the structure axes a door copies once they exist.

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
class Ontology:
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
    entities: EntitySource
    assertions: AssertionSource
    taxonomies: Mapping[str, TaxonomyDefinition]
    describes: tuple[SourceDescription, ...]

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
    structures: Mapping[str, Hierarchy[str]] = field(default_factory=dict)

    #: The ontology ids this vocabulary declares as ``imports:``, **carried
    #: and never followed**. Resolving a reference across an import needs a
    #: second ontology in scope, which a door loading one file does not have;
    #: dropping the list left the component that *does* have one unable to see
    #: what to resolve against, so it travels even though nothing here spends
    #: it.
    imports: tuple[str, ...] = ()

    def entity(self, entity_id: str) -> Entity | None:
        """The entity with this id, by way of :attr:`entities`.

        An accessor *invokes* the one way rather than reproducing it: this is
        one line, and it is one line on purpose. A second lookup here that
        happened to agree with the source today is the failure this shape
        rules out.
        """
        return self.entities.get(entity_id)

    def by_surface_form(self, form: str) -> frozenset[str]:
        """The ids matching this form, by way of :attr:`entities`."""
        return self.entities.by_surface_form(form)

    def taxonomy(self, name: str) -> Taxonomy:
        """The axis this ontology declares under ``name``, built.

        Built from three fields this object already holds and **no consumer
        input**, which is what keeps it an accessor rather than a factory: the
        structure is this ontology's assertions read for one relation, the
        content is its entity source, and the assertions travel along so a
        cursor over the axis can report the edge it walked.

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
        )

    def qualify(self, local_id: str, source_id: str | None = None) -> str:
        """This ontology's external id for ``local_id``.

        Fills the ontology segment from :attr:`id` and composes the rest
        through the free
        :func:`~dataknobs_common.ontology.model.qualify`, which is where the
        spelling of a namespaced id lives. An ``f"{a}:{b}"`` here would be a
        second spelling of it, and a malformed id is unfixable once it is
        written into stored data.
        """
        return _qualify(self.id, local_id, source_id)

    def localize(self, qualified_id: str) -> str:
        """``qualified_id`` in this ontology's own space -- what :meth:`entity` takes.

        For a single-source ontology that is the bare local id, which is every
        authored vocabulary; for a multi-source one it keeps the source
        segment, because that is the space :attr:`entities` speaks. The name
        invites the first reading, so the second is stated here and asserted
        in a test rather than left to be discovered.

        Refuses an id belonging to another ontology, **naming both**. That
        refusal is what these two members have that the free functions do not:
        only an ontology knows whose ids it is parsing. See :func:`_localize`.
        """
        return _localize(self.id, qualified_id)


@dataclass(frozen=True, eq=False)
class AsyncOntology:
    """The same ten fields, with asynchronous backings.

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
    entities: AsyncEntitySource
    assertions: AsyncAssertionSource
    taxonomies: Mapping[str, TaxonomyDefinition]
    describes: tuple[SourceDescription, ...]

    #: :attr:`Ontology.structures`, in an asynchronous slot.
    structures: Mapping[str, AsyncHierarchy[str]] = field(default_factory=dict)

    #: :attr:`Ontology.imports`, unflavoured -- a list of ids awaits nothing.
    imports: tuple[str, ...] = ()

    async def entity(self, entity_id: str) -> Entity | None:
        """The entity with this id, by way of :attr:`entities`."""
        return await self.entities.get(entity_id)

    async def by_surface_form(self, form: str) -> frozenset[str]:
        """The ids matching this form, by way of :attr:`entities`."""
        return await self.entities.by_surface_form(form)

    def taxonomy(self, name: str) -> AsyncTaxonomy:
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
        )

    def qualify(self, local_id: str, source_id: str | None = None) -> str:
        """:meth:`Ontology.qualify`, unflavoured.

        A plain ``def`` on this twin as well, and for the same reason
        :meth:`taxonomy` is one: it reads two fields and awaits nothing, so
        making it awaitable would cost every caller an ``await`` for a string
        concatenation.
        """
        return _qualify(self.id, local_id, source_id)

    def localize(self, qualified_id: str) -> str:
        """:meth:`Ontology.localize`, unflavoured."""
        return _localize(self.id, qualified_id)

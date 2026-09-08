"""What a loaded ontology is: the neutral core, and the two flavours over it.

Three dataclasses. :class:`OntologyParts` is everything the two flavours share
and nothing a source must be opened for -- it is what validation produces, and
it is flavour-neutral because the only flavour-specific part of construction is
binding sources.

:class:`Ontology` and :class:`AsyncOntology` differ in exactly two of eight
fields. That is not a coincidence to be tidied away: an ontology **holds
sources, not entities**, so one shape serves both modes and the flavour is
fixed by which door was called rather than inferred from the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dataknobs_common.exceptions import NotFoundError, ValidationError
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)
from dataknobs_common.ontology.model import InferenceMode
from dataknobs_common.taxonomy import AsyncTaxonomy, Taxonomy

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    _refuse_unbuildable_axis(definition)
    return definition


def _refuse_unbuildable_axis(definition: TaxonomyDefinition) -> None:
    """Refuse an axis whose materialization names a branch nothing here builds.

    ``Materialization`` is per axis and reuses ``InferenceMode``: a
    ``MATERIALIZED`` axis is a snapshot with a build time rather than a live
    read. Both axes have a live implementation and both modes are refused, but
    no longer for the same reason -- which is why the messages differ, and why
    only one of the two is still waiting on something to be written:

    * **content** materialized is a copy of every entity the axis covers. It is
      the expensive one and needs somewhere to live; an ontology loaded from a
      hand-edited file binds no store, so it is asking for something it cannot
      be given;
    * **structure** materialized is ids and edges -- the cheap copy, and cheap
      enough that a store is not what it lacks. It no longer lacks an
      implementation either: :meth:`.MappingHierarchy.snapshot` builds exactly
      this copy from any axis. What it lacks is the *wiring*, and the axis built
      here is why -- an ``AssertionHierarchy`` opens nothing and caches nothing,
      so it *is* the live read, and handing it back for a definition that asked
      for a snapshot would be answering under the wrong name.

    Both are refused **naming the axis**, and refused here -- where the caller
    asked for the axis and can act on the answer -- rather than at the first
    walk, which is a call site with no idea why it failed.

    The structure refusal is temporary by construction. What lifts it is the
    *door* taking the copy, not the copy existing -- and where the door can take
    it is forced rather than chosen: ``taxonomy()`` is a plain ``def`` on both
    twins while the asynchronous snapshot is ``async``, so the only place both
    flavours can take one without changing a published signature is at load,
    once per materialized definition. Taking it inside ``taxonomy()`` fails on
    a second count as well: the method builds afresh on every call, so a
    materialized axis fetched twice would be two snapshots with two build times
    and nothing saying so.

    Until that lands, refusing is the only thing standing between a consumer and
    a silent downgrade, because the alternative to refusing is prose saying the
    mode is recorded but not acted on -- a sentence that has already been copied
    into its own opposite once.
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
    if definition.materialization.structure is InferenceMode.MATERIALIZED:
        raise ValidationError(
            f"taxonomy {definition.id!r} declares `materialization.structure: "
            f"materialized`, which is a snapshot of the axis's ids and edges "
            f"taken at build time; this ontology does not take one for you. "
            f"Use `structure: on_demand`, which reads the edges through the "
            f"assertion source -- and if you want the copy, take it yourself "
            f"with `MappingHierarchy.snapshot(onto.taxonomy(...).structure)`",
            context={
                "taxonomy": definition.id,
                "axis": "structure",
                "mode": InferenceMode.MATERIALIZED.value,
            },
        )


@dataclass(frozen=True)
class OntologyParts:
    """A validated config, mapped onto values, with no source bound yet.

    Six of :class:`Ontology`'s eight fields verbatim, plus the declared rows
    and the still-unbound ``sources:`` specs. The two it omits are the sources,
    which each door binds into its own flavour.

    Returning this rather than an ontology is what lets one core serve every
    door: a core that inspected the config to decide which flavour to build
    would make the *return type* a function of the input, which is exactly the
    ambiguity two separate flavours exist to remove.
    """

    id: str
    version: str
    entity_types: Mapping[str, EntityType]
    relation_types: Mapping[str, RelationType]
    taxonomies: Mapping[str, TaxonomyDefinition]
    declared_entities: Mapping[str, Entity]
    declared_assertions: tuple[Assertion, ...]
    source_specs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Ontology:
    """A loaded vocabulary whose backings are synchronous.

    A value with accessors. Every member below is pure over the fields --
    nothing here opens anything, and nothing is lazy -- which is what makes an
    ontology safe to hold, share and pass without owning a lifecycle.
    """

    id: str
    version: str
    entity_types: Mapping[str, EntityType]
    relation_types: Mapping[str, RelationType]
    entities: EntitySource
    assertions: AssertionSource
    taxonomies: Mapping[str, TaxonomyDefinition]
    describes: tuple[SourceDescription, ...]

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

        Refuses, naming the axis, a definition whose ``materialization`` asks
        for something this ontology cannot supply -- see
        :func:`_refuse_unbuildable_axis`.
        """
        definition = _definition(self.taxonomies, name)
        return Taxonomy(
            definition=definition,
            structure=AssertionHierarchy(self.assertions, definition.relation),
            entities=self.entities,
            assertions=self.assertions,
        )


@dataclass(frozen=True)
class AsyncOntology:
    """The same eight fields, with asynchronous backings.

    ``entities`` is an :class:`~dataknobs_common.ontology.sources.AsyncEntitySource`
    and ``assertions`` an
    :class:`~dataknobs_common.ontology.sources.AsyncAssertionSource`; everything
    else is identical, including that this is a value and owns nothing.
    """

    id: str
    version: str
    entity_types: Mapping[str, EntityType]
    relation_types: Mapping[str, RelationType]
    entities: AsyncEntitySource
    assertions: AsyncAssertionSource
    taxonomies: Mapping[str, TaxonomyDefinition]
    describes: tuple[SourceDescription, ...]

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
        assignments.
        """
        definition = _definition(self.taxonomies, name)
        return AsyncTaxonomy(
            definition=definition,
            structure=AsyncAssertionHierarchy(self.assertions, definition.relation),
            entities=self.entities,
            assertions=self.assertions,
        )

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
    """Refuse an axis whose materialization needs something nothing here holds.

    ``Materialization`` is per axis and reuses ``InferenceMode``: a
    ``MATERIALIZED`` axis is a snapshot with a build time rather than a live
    read. The two axes are not alike in what a snapshot costs, and the shipped
    defaults are exactly the pair that costs nothing to hold:

    * **structure** materialized is ids and edges -- small enough that an
      authored vocabulary serves the axis with no store at all, which is why it
      is the default. It is **recorded rather than acted on**: the axis built
      here is an ``AssertionHierarchy``, which opens nothing and caches nothing,
      so both modes read through the source and neither is refused. Reading the
      declaration back off the definition is the only thing it does today;
    * **content** materialized is a copy of every entity the axis covers, which
      is the expensive one, needs somewhere to live, and is refused below when
      nothing can hold it.

    An ontology loaded from a hand-edited file binds no store, so a definition
    asking for a materialized content axis is asking for something it cannot be
    given. It is refused **naming the axis**, and refused here -- where the
    caller asked for the axis and can act on the answer -- rather than at the
    first walk, which is a call site with no idea why it failed.
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

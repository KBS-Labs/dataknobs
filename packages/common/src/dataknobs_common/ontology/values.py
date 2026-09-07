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

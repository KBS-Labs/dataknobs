"""The typed form of an ``ontology:`` document.

Leaf sections stay raw mappings on purpose. ``sources:`` and ``resolver:`` are
discriminated by a ``kind:`` their entries carry, and the set of kinds is a
registry read rather than a list this module could close over -- so typing them
here would mean naming, in ``dataknobs-common``, kinds that other packages
register. The loader validates what it needs and hands the rest on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataknobs_common.structured_config import StructuredConfig

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class OntologyConfig(StructuredConfig):
    """What a door must be handed to load an ontology.

    Attributes:
        id: The ontology's namespace. Reserved value ``dk`` is refused, and so
            is any id containing ``:``
        version: The vocabulary's own version, not this schema's
        imports: Other ontology ids whose namespaces come into scope
        entity_types: Type declarations, each optionally naming an ``isa``
        relation_types: Relation declarations, loaded into the same store as
            ``entities`` under a typed discriminator
        entities: Inline entity rows, each carrying its own ``id``
        assertions: Inline assertion rows over those entities
        sources: Unbound source specs, discriminated by ``kind:``
        overlay: A default rather than a binding
        taxonomies: Axis definitions -- the definition, never the built axis
        index: The semantic index's configuration, raw
        resolver: The placement cascade's configuration, raw, because its
            ``rungs`` are themselves discriminated by ``kind:``
    """

    id: str
    version: str = "1.0"
    imports: list[str] = field(default_factory=list)
    entity_types: list[Mapping[str, Any]] = field(default_factory=list)
    relation_types: list[Mapping[str, Any]] = field(default_factory=list)
    entities: list[Mapping[str, Any]] = field(default_factory=list)
    assertions: list[Mapping[str, Any]] = field(default_factory=list)
    sources: list[Mapping[str, Any]] = field(default_factory=list)
    overlay: Mapping[str, Any] | None = None
    taxonomies: list[Mapping[str, Any]] = field(default_factory=list)
    index: Mapping[str, Any] | None = None
    resolver: Mapping[str, Any] | None = None

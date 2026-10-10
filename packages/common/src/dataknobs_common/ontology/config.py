# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The typed form of an ``ontology:`` document.

Leaf sections stay raw mappings on purpose. ``sources:`` and ``resolver:`` are
discriminated by a ``kind:`` their entries carry, and the set of kinds is a
registry read rather than a list this module could close over -- so typing them
here would mean naming, in ``dataknobs-common``, kinds that other packages
register. The loader validates what it needs and hands the rest on.

``index:`` and ``event_bus:`` are raw for a different reason, and it is worth
saying which. ``index:``'s blocks are ``$resource`` references into binding
categories -- ``vector_stores``, ``embedders`` -- whose concrete types belong
to ``dataknobs-data`` and ``dataknobs-llm``, and ``event_bus:`` names a backend
whose drivers are optional installs. There is no discriminator to leave open in
either; there is a package boundary. The reader that resolves both is
``dataknobs_data.ontology.OntologyRegistry``, which is also the only door that
binds a live source, for the same reason: it owns a lifecycle and a
module-level loader does not -- so a module-level loader reads neither section
and ignores both.
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

    **Compared field-wise, and not hashable**, as every
    :class:`~dataknobs_common.structured_config.StructuredConfig` is: the
    base declares each subclass unhashable, and its docstring says why.

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
        index: The semantic index's configuration, raw. Two blocks: a
            ``store:`` the reader resolves and opens itself, and an
            ``embedder:`` it refuses rather than builds --- the construct that
            accepts one lives in a package that depends on the reader's, so
            the edge runs the wrong way and the embedder is injected
        resolver: The placement cascade's configuration, raw, because its
            ``rungs`` are themselves discriminated by ``kind:``
        event_bus: The bus a registry announces this vocabulary's arrival and
            departure on, raw, for ``index:``'s reason. A field rather than a
            key read off the mapping beside it: every published construction
            door coerces its argument to this class before a consumer sees it,
            so a section that is not declared here does not survive the trip
            and a door reading one off the raw document disagrees with a door
            that cannot
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
    event_bus: Mapping[str, Any] | None = None

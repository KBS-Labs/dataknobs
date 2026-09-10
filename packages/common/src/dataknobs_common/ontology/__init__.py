"""Vocabularies: entities, the relations between them, and the axes they form.

An ontology here is a **value**. It holds sources rather than entities, owns no
lifecycle, and every accessor on it is pure over its own fields -- so it is
safe to hold, share and pass without anyone having to remember to close it.

Two doors load one::

    from dataknobs_common.ontology import load_ontology

    onto = load_ontology(Path("mammals.yaml"))
    onto.entities.by_surface_form("beagles")     # frozenset({"beagle"})
    onto.entity("beagle").name                   # "Beagle"

``load_ontology`` needs no database, no embedder and no event loop; a file a
person edited is already a list once read. ``async_load_ontology`` returns the
same vocabulary with asynchronous sources, for a caller whose surrounding code
is async. Neither forwards to the other -- they meet at ``build_ontology``,
which validates and maps and binds nothing.

Both refuse a *live* source kind, and the refusal is about ownership rather
than about what is installed: binding a database source creates something that
must be closed, and a module-level function has no ``close()``.
"""

from dataknobs_common.entity_resolution.protocols import (
    AliasFormSource,
    AsyncAliasFormSource,
    MembershipOracle,
)
from dataknobs_common.entity_resolution.values import (
    CompatibilityVerdict,
    ResolutionRef,
    Scoring,
)
from dataknobs_common.ontology.config import OntologyConfig
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
    EdgeCriteria,
    edge_criteria,
)
from dataknobs_common.ontology.loader import (
    AUTHORED_SOURCE_KINDS,
    DEFAULT_NESTED_RELATION,
    ENTITY_TYPE_ISA_KEY,
    RESERVED_ONTOLOGY_ID,
    async_build_resolver,
    async_load_ontology,
    build_ontology,
    build_resolver,
    load_ontology,
)
from dataknobs_common.ontology.model import (
    DK_ENTITY_TYPE,
    DK_RELATION_TYPE,
    Assertion,
    AttributeDef,
    CyclePolicy,
    Entity,
    EntityRef,
    EntityType,
    InferenceMode,
    Literal,
    Materialization,
    ParentChoice,
    Polarity,
    ProjectionContext,
    Provenance,
    QualifiedId,
    RelationRef,
    RelationType,
    SiblingOrder,
    SourceRef,
    TaxonomyDefinition,
    Term,
    TreeProjection,
    qualify,
    relation_id,
    split_qualified,
)
from dataknobs_common.ontology.sources import (
    AUTHORED_SOURCE_ID,
    AssertionSource,
    AsyncAssertionSource,
    AsyncEntitySource,
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    EntitySource,
    MappingAssertionSource,
    MappingEntitySource,
    SourceDescription,
)
from dataknobs_common.text import default_normalizer
from dataknobs_common.ontology.values import AsyncOntology, Ontology, OntologyParts

__all__ = [
    "AUTHORED_SOURCE_ID",
    "AUTHORED_SOURCE_KINDS",
    "DEFAULT_NESTED_RELATION",
    "DK_ENTITY_TYPE",
    "DK_RELATION_TYPE",
    "ENTITY_TYPE_ISA_KEY",
    "RESERVED_ONTOLOGY_ID",
    "AliasFormSource",
    "Assertion",
    "AssertionHierarchy",
    "AssertionSource",
    "AsyncAliasFormSource",
    "AsyncAssertionHierarchy",
    "AsyncAssertionSource",
    "AsyncEntitySource",
    "AsyncMappingAssertionSource",
    "AsyncMappingEntitySource",
    "AsyncOntology",
    "AttributeDef",
    "CompatibilityVerdict",
    "CyclePolicy",
    "EdgeCriteria",
    "Entity",
    "EntityRef",
    "EntitySource",
    "EntityType",
    "InferenceMode",
    "Literal",
    "MappingAssertionSource",
    "MappingEntitySource",
    "Materialization",
    "MembershipOracle",
    "Ontology",
    "OntologyConfig",
    "OntologyParts",
    "ParentChoice",
    "Polarity",
    "ProjectionContext",
    "Provenance",
    "QualifiedId",
    "RelationRef",
    "RelationType",
    "ResolutionRef",
    "Scoring",
    "SiblingOrder",
    "SourceDescription",
    "SourceRef",
    "TaxonomyDefinition",
    "Term",
    "TreeProjection",
    "async_build_resolver",
    "async_load_ontology",
    "build_ontology",
    "build_resolver",
    "default_normalizer",
    "edge_criteria",
    "load_ontology",
    "qualify",
    "relation_id",
    "split_qualified",
]

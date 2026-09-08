"""The doors: one pure core, and a flavour on each side of it.

``build_ontology`` validates a config and maps it onto values. It binds no
source and opens nothing, so it is the same function for every door and every
refusal below fires from it -- which is what makes "the sync and async paths
refuse identically" a property of the code rather than a pair of lists someone
keeps in step.

The two doors bind. Neither forwards to the other, because a synchronous
function cannot call an ``async def``; they meet at the core instead, which is
the only place a shared implementation can actually live.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Container, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataknobs_common.config_loading import load_yaml_or_json
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.fields import FieldType
from dataknobs_common.ontology.config import OntologyConfig
from dataknobs_common.ontology.model import (
    Assertion,
    AttributeDef,
    Entity,
    EntityRef,
    EntityType,
    InferenceMode,
    Literal,
    Materialization,
    RelationType,
    SourceRef,
    TaxonomyDefinition,
    Term,
)
from dataknobs_common.ontology.sources import (
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    MappingAssertionSource,
    MappingEntitySource,
)
from dataknobs_common.ontology.values import AsyncOntology, Ontology, OntologyParts

if TYPE_CHECKING:
    from collections.abc import Callable

#: An ontology may not claim this id: it names the built-in pseudo-ontology
#: that ``dk:EntityType`` and ``dk:RelationType`` belong to.
RESERVED_ONTOLOGY_ID = "dk"

#: The source kinds a module-level door can bind for itself.
#:
#: Everything else is a *live* source, and the refusal is about ownership
#: rather than about what happens to be installed: binding a live source
#: creates something that must be closed, and a function that returns a value
#: has no ``close()`` and nobody to call it. Because this is a literal set,
#: the answer cannot change when another package is imported -- one config
#: means one thing in every environment.
AUTHORED_SOURCE_KINDS = frozenset({"inline", "nested"})

#: What a nested source's parent-child edges are asserted as, absent a
#: ``relation:`` of its own.
DEFAULT_NESTED_RELATION = "isa"

_SLUG_SEPARATORS = re.compile(r"[^a-z0-9/]+")

#: Where an entity type's declared ``isa:`` parent is kept for now.
#:
#: The type lattice is a *different store* from the ``isa`` assertions between
#: instances, and which store it is has not been settled. ``EntityType``
#: declares no field for it, so the alternative to parking it here is dropping
#: a line the document author wrote -- and a validated value that vanishes is
#: worse than one kept under a documented key until its home is decided.
ENTITY_TYPE_ISA_KEY = "isa"


def build_ontology(config: OntologyConfig) -> OntologyParts:
    """Validate a config and map its sections onto values.

    Pure: it reads no file, opens no connection and binds no source. What it
    returns is flavour-neutral, and each door turns it into its own flavour by
    binding ``source_specs`` -- the only part of construction that differs.

    Args:
        config: The typed form of an ``ontology:`` document

    Returns:
        The validated parts, with sources still unbound

    Raises:
        ValidationError: On a reserved or malformed id; on an id duplicated
            within a section or across two of them; on two tree nodes minting
            one id; on an unknown inference mode; or on an ``isa`` naming a
            type the document does not declare. Every message names the
            offending value
    """
    _refuse_reserved_id(config.id)
    _refuse_colon("ontology id", config.id)

    source_specs = tuple(config.sources)
    _validate_source_ids(source_specs)

    entity_types = _build_entity_types(config.entity_types)
    relation_types = _build_relation_types(config.relation_types)
    declared = _build_entities(config.entities)

    _refuse_duplicates(entity_types, relation_types, declared)
    _refuse_undeclared_isa(config.entity_types, entity_types)

    assertions = list(_build_assertions(config.assertions))

    # Declared wins wherever it exists. Minting into a file that already names
    # its entities would give one node two ids, and every authored assertion
    # points at one of them.
    if not declared:
        minted_entities, minted_assertions = _mint_nested(source_specs)
        declared = {**declared, **minted_entities}
        assertions.extend(minted_assertions)
    else:
        _refuse_undeclared_tree_nodes(source_specs, declared)

    _refuse_duplicate_assertion_ids(assertions)

    return OntologyParts(
        id=config.id,
        version=config.version,
        entity_types=entity_types,
        relation_types=relation_types,
        taxonomies=_build_taxonomies(config.taxonomies),
        declared_entities=declared,
        declared_assertions=tuple(assertions),
        source_specs=source_specs,
    )


def load_ontology(
    source: Path | Mapping[str, Any],
    *,
    normalizer: Callable[[str], str] | None = None,
) -> Ontology:
    """Load a vocabulary with synchronous backings.

    No database, no embedder, no event loop. A hand-edited file is already a
    list once read, so nothing here has anything to await and the caller is
    not made to pretend otherwise.

    Args:
        source: A path to a YAML or JSON document, or the document itself
        normalizer: How surface forms are folded for lookup. Defaults to
            strip-and-casefold

    Returns:
        The loaded ontology

    Raises:
        ValidationError: For any refusal in :func:`build_ontology`, and for a
            live source kind this door cannot own
        ConfigLoadError: For any refusal in reading ``source`` as a
            document -- an unrecognised extension, a payload that will not
            parse, a root that is not a mapping, or YAML with PyYAML not
            installed (it ships in this package's ``yaml`` extra).
            Reachable only when ``source`` is a ``Path``: a caller who
            passes the document itself never reaches the read
        OSError: From that same read, for a path that cannot be opened --
            ``FileNotFoundError`` when it is not there
    """
    parts = _validated_parts(_read_config(source))
    entities = MappingEntitySource(parts.declared_entities, normalizer=normalizer)
    return Ontology(
        id=parts.id,
        version=parts.version,
        entity_types=parts.entity_types,
        relation_types=parts.relation_types,
        entities=entities,
        assertions=MappingAssertionSource(parts.declared_assertions),
        taxonomies=parts.taxonomies,
        describes=(entities.describe(),),
    )


async def async_load_ontology(
    source: Path | Mapping[str, Any],
    *,
    normalizer: Callable[[str], str] | None = None,
) -> AsyncOntology:
    """Load a vocabulary with asynchronous backings.

    The same file and the same refusals as :func:`load_ontology`; only the
    flavour of the bound sources differs.

    Args:
        source: A path to a YAML or JSON document, or the document itself
        normalizer: How surface forms are folded for lookup

    Returns:
        The loaded ontology, with asynchronous sources

    Raises:
        ValidationError: Exactly as :func:`load_ontology`
        ConfigLoadError: Exactly as :func:`load_ontology`, propagated out of
            the offloaded read
        OSError: Exactly as :func:`load_ontology`, likewise
    """
    # The read is the only blocking thing either door does, so it is the only
    # thing offloaded -- and only when there is a read: a caller who already
    # holds the document should not pay for a thread to hand it back.
    if isinstance(source, Path):
        config = await asyncio.to_thread(_read_config, source)
    else:
        config = _read_config(source)

    parts = _validated_parts(config)
    entities = AsyncMappingEntitySource(parts.declared_entities, normalizer=normalizer)
    return AsyncOntology(
        id=parts.id,
        version=parts.version,
        entity_types=parts.entity_types,
        relation_types=parts.relation_types,
        entities=entities,
        assertions=AsyncMappingAssertionSource(parts.declared_assertions),
        taxonomies=parts.taxonomies,
        describes=(entities.describe(),),
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _validated_parts(config: OntologyConfig) -> OntologyParts:
    """The half both doors share: validate the grammar, then refuse what a
    module-level loader cannot own.

    Extracted rather than written twice. The two doors already differ in how
    they read and what flavour they bind; letting them also each decide which
    configs to refuse is how "the sync and async paths refuse identically"
    stops being true without anyone editing the sentence that says it does.

    ``build_ontology`` is called by name through this module, so patching it
    is observed by both doors -- which is what makes the delegation test a
    test of delegation rather than of agreement.
    """
    parts = build_ontology(config)
    _refuse_live_sources(parts.source_specs)
    return parts


def _read_config(source: Path | Mapping[str, Any]) -> OntologyConfig:
    """Turn a path or a mapping into a typed config.

    A document may wrap its content under a top-level ``ontology:`` key, which
    is how the section reads in a file that also carries other blocks. Both
    forms are accepted, because refusing the unwrapped one would make the
    mapping and the file spellings differ for no reason.
    """
    raw = load_yaml_or_json(source) if isinstance(source, Path) else dict(source)
    document = raw.get("ontology", raw)
    if not isinstance(document, Mapping):
        raise ValidationError(
            f"`ontology:` must be a mapping, got {type(document).__name__}",
            context={"ontology": document},
        )
    return OntologyConfig.from_dict(document)


# --------------------------------------------------------------------------
# Refusals -- each names the offending value
# --------------------------------------------------------------------------


def _refuse_reserved_id(ontology_id: str) -> None:
    if ontology_id == RESERVED_ONTOLOGY_ID:
        raise ValidationError(
            f"{RESERVED_ONTOLOGY_ID!r} is a reserved ontology id: it names the "
            f"built-in pseudo-ontology that {', '.join(('dk:EntityType', 'dk:RelationType'))} "
            f"belong to. Choose another id",
            context={"ontology_id": ontology_id},
        )


def _refuse_colon(what: str, value: str) -> None:
    if ":" in value:
        raise ValidationError(
            f"{what} {value!r} contains ':', which separates the parts of a "
            f"qualified id. Remove it",
            context={what.replace(" ", "_"): value},
        )


def _validate_source_ids(specs: tuple[Mapping[str, Any], ...]) -> None:
    seen: set[str] = set()
    for spec in specs:
        source_id = str(spec.get("id", ""))
        _refuse_colon("source id", source_id)
        if source_id in seen:
            raise ValidationError(
                f"duplicate source id {source_id!r}",
                context={"source_id": source_id},
            )
        seen.add(source_id)


def _refuse_duplicate_id(claimed: Container[str], candidate: str, section: str) -> None:
    """Refuse an id a section has already claimed.

    Inline at the write, and it has to be there: a dict keyed by id has
    already lost the first row by the time a pass over it could look. That is
    why :func:`_refuse_duplicates`, which reads the built dicts, catches a
    collision *between* two sections and none *within* one.
    """
    if candidate in claimed:
        raise ValidationError(
            f"duplicate id {candidate!r} in `{section}:`",
            context={"id": candidate, "section": section},
        )


def _refuse_duplicate_assertion_ids(assertions: list[Assertion]) -> None:
    """Refuse two assertions sharing one id, authored or minted.

    A pass rather than an inline check, because a list keeps both rows and so
    unlike the id-keyed sections nothing is lost before this can look. It runs
    once over the combined set, so an authored id colliding with a minted one
    is caught as readily as two authored ones.

    What a duplicate costs is not one row. :class:`~.sources._AssertionIndex`
    keys ``by_id`` by id and ``by_subject`` by subject, so a collision makes
    two lookups over the same store disagree about how many assertions there
    are.
    """
    seen: set[str] = set()
    for assertion in assertions:
        _refuse_duplicate_id(seen, assertion.id, "assertions")
        seen.add(assertion.id)


def _inference_mode(declared: Any, field: str) -> InferenceMode:
    """Coerce a declared inference mode, naming an unknown one.

    :class:`ValidationError` does not descend from ``ValueError``, so letting
    the enum raise its own would escape a caller holding the contract both
    doors document -- ``except ValidationError`` -- rather than being caught
    by it.
    """
    try:
        return InferenceMode(str(declared))
    except ValueError as exc:
        raise ValidationError(
            f"{field} is {str(declared)!r}, which is not an inference mode. "
            f"Modes: {sorted(mode.value for mode in InferenceMode)}",
            context={"field": field, "value": declared},
        ) from exc


def _refuse_duplicates(
    entity_types: Mapping[str, EntityType],
    relation_types: Mapping[str, RelationType],
    entities: Mapping[str, Entity],
) -> None:
    """Refuse an id declared twice across the three sections.

    They share one store, so a collision between an ``entity_types`` id and an
    ``entities`` id is as real as one within either.
    """
    seen: dict[str, str] = {}
    for section, declared in (
        ("entity_types", entity_types),
        ("relation_types", relation_types),
        ("entities", entities),
    ):
        for declared_id in declared:
            if declared_id in seen:
                raise ValidationError(
                    f"duplicate id {declared_id!r}: declared in "
                    f"{seen[declared_id]!r} and again in {section!r}",
                    context={"id": declared_id, "sections": [seen[declared_id], section]},
                )
            seen[declared_id] = section


def _refuse_undeclared_isa(
    rows: list[Mapping[str, Any]], entity_types: Mapping[str, EntityType]
) -> None:
    for row in rows:
        parent = row.get("isa")
        if parent is not None and str(parent) not in entity_types:
            raise ValidationError(
                f"entity type {str(row.get('id'))!r} declares `isa: {parent!r}`, "
                f"which no entity type in this document declares. Declared: "
                f"{sorted(entity_types)}",
                context={"entity_type": row.get("id"), "isa": parent},
            )


def _refuse_live_sources(specs: tuple[Mapping[str, Any], ...]) -> None:
    """Refuse a source this door cannot own.

    Computed from the declared kind alone, so it holds identically whether or
    not a package that could bind such a source has been imported.
    """
    for spec in specs:
        kind = str(spec.get("kind", ""))
        if kind not in AUTHORED_SOURCE_KINDS:
            source_id = str(spec.get("id", ""))
            raise ValidationError(
                f"source {source_id!r} declares kind {kind!r}, which is a live "
                f"source: it must be opened, and closed again, and a module-level "
                f"loader owns no lifecycle to do that with. Load this ontology "
                f"through OntologyRegistry, which does. Kinds this loader binds: "
                f"{sorted(AUTHORED_SOURCE_KINDS)}",
                context={"source_id": source_id, "kind": kind},
            )


def _refuse_undeclared_tree_nodes(
    specs: tuple[Mapping[str, Any], ...], declared: Mapping[str, Entity]
) -> None:
    """In the declared regime, a tree node must name an entity that exists."""
    for spec in specs:
        if str(spec.get("kind", "")) != "nested":
            continue
        child_key = str(spec.get("child_key", "children"))
        for node, _path in _walk_tree(
            spec.get("tree"), child_key, str(spec.get("name_key", "name"))
        ):
            node_id = node.get("id")
            if node_id is None or str(node_id) not in declared:
                raise ValidationError(
                    f"tree node {str(node_id)!r} in source "
                    f"{str(spec.get('id'))!r} names no declared entity. This "
                    f"document declares `entities:`, so its tree references "
                    f"those ids and mints none",
                    context={"source_id": spec.get("id"), "node_id": node_id},
                )


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------


def _build_entity_types(rows: list[Mapping[str, Any]]) -> dict[str, EntityType]:
    built: dict[str, EntityType] = {}
    for row in rows:
        type_id = str(row["id"])
        _refuse_colon("entity type id", type_id)
        _refuse_duplicate_id(built, type_id, "entity_types")
        built[type_id] = EntityType(
            id=type_id,
            name=str(row.get("name", "")),
            description=row.get("description"),
            aliases=list(row.get("aliases", [])),
            metadata=_type_metadata(row),
            attributes=[_build_attribute(a) for a in row.get("attributes", [])],
        )
    return built


def _type_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """An entity type's metadata, with its declared ``isa:`` folded in.

    See :data:`ENTITY_TYPE_ISA_KEY` for why the lattice lives here for now.
    """
    metadata = dict(row.get("metadata", {}))
    parent = row.get("isa")
    if parent is not None:
        metadata[ENTITY_TYPE_ISA_KEY] = str(parent)
    return metadata


def _build_attribute(row: Mapping[str, Any]) -> AttributeDef:
    declared = row.get("type")
    return AttributeDef(
        name=str(row["name"]),
        value_type=str(declared) if declared is not None else "string",
        field_type=_field_type(declared),
        entity_type=row.get("entity_type"),
        required=bool(row.get("required", False)),
        enum_values=list(row["enum_values"]) if row.get("enum_values") else None,
        description=str(row.get("description", "")),
    )


def _field_type(declared: Any) -> FieldType | None:
    """The record field type an attribute's declared type maps onto, if any.

    ``entity`` and ``enum`` are vocabulary concepts with no record-field
    counterpart, so they map to None rather than being forced onto one.
    """
    if declared is None:
        return None
    try:
        return FieldType(str(declared))
    except ValueError:
        return None


def _build_relation_types(rows: list[Mapping[str, Any]]) -> dict[str, RelationType]:
    built: dict[str, RelationType] = {}
    for row in rows:
        relation_id = str(row["id"])
        _refuse_colon("relation type id", relation_id)
        _refuse_duplicate_id(built, relation_id, "relation_types")
        built[relation_id] = RelationType(
            id=relation_id,
            name=str(row.get("name", "")),
            description=row.get("description"),
            aliases=list(row.get("aliases", [])),
            metadata=dict(row.get("metadata", {})),
            domain=frozenset(row.get("domain", ())),
            range=frozenset(row.get("range", ())),
            inverse_of=row.get("inverse_of"),
            symmetric=bool(row.get("symmetric", False)),
            transitive=bool(row.get("transitive", False)),
            inference=_inference_mode(
                row.get("inference", InferenceMode.ON_DEMAND.value), "inference"
            ),
        )
    return built


def _build_entities(rows: list[Mapping[str, Any]]) -> dict[str, Entity]:
    built: dict[str, Entity] = {}
    for row in rows:
        entity_id = str(row["id"])
        _refuse_colon("entity id", entity_id)
        if entity_id in built:
            raise ValidationError(
                f"duplicate id {entity_id!r} in `entities:`",
                context={"id": entity_id},
            )
        built[entity_id] = Entity(
            id=entity_id,
            type=str(row["type"]),
            name=str(row.get("name", "")),
            aliases=list(row.get("aliases", [])),
            description=row.get("description"),
            metadata=dict(row.get("metadata", {})),
            source=_build_source_ref(row.get("source")),
        )
    return built


def _build_source_ref(row: Mapping[str, Any] | None) -> SourceRef | None:
    """The reference an entity carries into a consumer's own data.

    Everything the row says beyond the three declared keys becomes the
    ``locator``: it is opaque to us by design, and dropping unknown keys would
    silently discard exactly the part the consumer needs to find the row.
    """
    if row is None:
        return None
    known = {"source_id", "kind", "projection_id"}
    return SourceRef(
        source_id=str(row.get("source_id", "")),
        kind=str(row.get("kind", "record")),
        locator={k: v for k, v in row.items() if k not in known},
        projection_id=row.get("projection_id"),
    )


def _build_assertions(rows: list[Mapping[str, Any]]) -> list[Assertion]:
    built: list[Assertion] = []
    for row in rows:
        subject = str(row["subject"])
        relation = str(row["relation"])
        obj = _build_term(row["object"])
        built.append(
            Assertion(
                id=str(row.get("id") or _mint_assertion_id(subject, relation, obj)),
                subject=subject,
                relation=relation,
                object=obj,
                metadata=dict(row.get("metadata", {})),
            )
        )
    return built


def _build_term(value: Any) -> Term:
    """What an assertion's ``object:`` denotes.

    A bare string names an entity, which is the form a hand-edited file uses
    for every edge. A mapping is explicit either way, and any other scalar is
    a literal whose type is detected the way a record field's is.
    """
    if isinstance(value, str):
        return EntityRef(entity_id=value)
    if isinstance(value, Mapping):
        if "entity" in value:
            return EntityRef(entity_id=str(value["entity"]))
        declared = _field_type(value.get("type"))
        return Literal(
            value=value.get("value"),
            type=declared if declared is not None else FieldType.STRING,
            unit=value.get("unit"),
            metadata=dict(value.get("metadata", {})),
        )
    return Literal.of(value)


def _mint_assertion_id(subject: str, relation: str, obj: Term) -> str:
    """A readable, deterministic id for an assertion the file did not name.

    Derived from the edge rather than from the row's position, so adding a row
    above it does not renumber every assertion below.
    """
    if isinstance(obj, EntityRef):
        tail = obj.entity_id
    else:
        tail = str(obj.value)
    return f"{subject}-{relation}-{tail}"


def _build_taxonomies(rows: list[Mapping[str, Any]]) -> dict[str, TaxonomyDefinition]:
    built: dict[str, TaxonomyDefinition] = {}
    for row in rows:
        taxonomy_id = str(row["id"])
        _refuse_duplicate_id(built, taxonomy_id, "taxonomies")
        materialization = row.get("materialization", {})
        built[taxonomy_id] = TaxonomyDefinition(
            id=taxonomy_id,
            relation=str(row["relation"]),
            name=str(row.get("name", "")),
            description=row.get("description"),
            metadata=dict(row.get("metadata", {})),
            materialization=Materialization(
                structure=_inference_mode(
                    materialization.get("structure", InferenceMode.MATERIALIZED.value),
                    "materialization.structure",
                ),
                content=_inference_mode(
                    materialization.get("content", InferenceMode.ON_DEMAND.value),
                    "materialization.content",
                ),
            ),
        )
    return built


# --------------------------------------------------------------------------
# Nested sources -- the keyless regime
# --------------------------------------------------------------------------


def _mint_nested(
    specs: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Entity], list[Assertion]]:
    """Mint an id per node of every nested tree, with the path as the name.

    A hand-maintained tree has no id field anywhere: a node *is* its path. But
    a path is a name, and a name renames itself when someone edits the file --
    which would re-key every descendant of a renamed interior node at once. So
    the id is a slug of the path taken at first load and the path becomes the
    ``name``, which means renaming a node changes what it is called and not
    what it is.
    """
    entities: dict[str, Entity] = {}
    assertions: list[Assertion] = []
    for spec in specs:
        if str(spec.get("kind", "")) != "nested":
            continue
        source_id = str(spec.get("id", ""))
        child_key = str(spec.get("child_key", "children"))
        name_key = str(spec.get("name_key", "name"))
        node_type = str(spec.get("type", source_id))
        relation = str(spec.get("relation", DEFAULT_NESTED_RELATION))

        minted_from: dict[str, str] = {}
        for node, path in _walk_tree(spec.get("tree"), child_key, name_key):
            name = "/".join(path)
            node_id = _slug(name)
            if node_id in entities:
                raise ValidationError(
                    f"tree node {name!r} in source {source_id!r} mints id "
                    f"{node_id!r}, which {minted_from.get(node_id, node_id)!r} "
                    f"already minted. The slug collapses punctuation and case, "
                    f"so two nodes a person reads as different can name one "
                    f"entity",
                    context={"source_id": source_id, "id": node_id, "path": name},
                )
            minted_from[node_id] = name
            entities[node_id] = Entity(
                id=node_id,
                type=node_type,
                name=name,
                aliases=list(node.get("aliases", [])),
                description=node.get("description"),
                metadata=dict(node.get("metadata", {})),
            )
            if len(path) > 1:
                parent_id = _slug("/".join(path[:-1]))
                assertions.append(
                    Assertion(
                        id=f"{node_id}-{relation}-{parent_id}",
                        subject=node_id,
                        relation=relation,
                        object=EntityRef(entity_id=parent_id),
                    )
                )
    return entities, assertions


def _walk_tree(
    tree: Any, child_key: str, name_key: str
) -> list[tuple[Mapping[str, Any], list[str]]]:
    """Every node of a nested tree, paired with its path from the root.

    Depth-first and iterative. A tree a person maintains is not deep, but a
    recursive walk would turn a malformed self-referential document into a
    stack overflow rather than a refusal.
    """
    if tree is None:
        return []
    roots = tree if isinstance(tree, list) else [tree]
    found: list[tuple[Mapping[str, Any], list[str]]] = []
    stack: list[tuple[Any, list[str]]] = [(node, []) for node in reversed(roots)]
    while stack:
        node, prefix = stack.pop()
        if not isinstance(node, Mapping):
            continue
        path = [*prefix, str(node.get(name_key, node.get("id", "")))]
        found.append((node, path))
        children = node.get(child_key) or []
        stack.extend((child, path) for child in reversed(children))
    return found


def _slug(text: str) -> str:
    """A stable id from a path: lower-cased, with separators collapsed.

    ``/`` survives because it is the path separator and carries the structure;
    everything else non-alphanumeric becomes a single ``-``.
    """
    collapsed = _SLUG_SEPARATORS.sub("-", text.lower()).strip("-")
    return "/".join(part.strip("-") for part in collapsed.split("/"))

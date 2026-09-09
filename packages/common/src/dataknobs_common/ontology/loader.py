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
from collections.abc import Container, Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from dataknobs_common._nested_core import _mint_tree, _walk_tree
from dataknobs_common.config_loading import load_yaml_or_json
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.fields import FieldType
from dataknobs_common.hierarchy import AsyncMappingHierarchy, MappingHierarchy
from dataknobs_common.ontology.config import OntologyConfig
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)
from dataknobs_common.ontology.model import (
    Assertion,
    AttributeDef,
    Entity,
    EntityRef,
    EntityType,
    InferenceMode,
    Literal,
    Materialization,
    Polarity,
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

#: Whichever enum a document is declaring a member of. Bound to ``Enum``
#: because :func:`_declared_enum` calls the type and iterates its members, and
#: to nothing narrower because it does neither of those things per enum.
_E = TypeVar("_E", bound=Enum)

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
        ValidationError: On a row omitting a field its section requires; on a
            reserved or malformed id; on an id duplicated within a section or
            across two of them; on two tree nodes minting one id; on an unknown
            inference mode or polarity; on a key only a later version reads --
            ``condition:``, ``cardinality:``, ``constraints:`` -- which is
            refused rather than dropped; or on an ``isa`` naming a type the
            document does not declare. Every message names the offending value
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
        imports=tuple(config.imports),
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
    assertions = MappingAssertionSource(parts.declared_assertions)
    return Ontology(
        id=parts.id,
        version=parts.version,
        entity_types=parts.entity_types,
        relation_types=parts.relation_types,
        entities=entities,
        assertions=assertions,
        taxonomies=parts.taxonomies,
        describes=(entities.describe(),),
        structures={
            name: MappingHierarchy.snapshot(AssertionHierarchy(assertions, definition.relation))
            for name, definition in _axes_to_copy(parts.taxonomies)
        },
        imports=parts.imports,
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
    assertions = AsyncMappingAssertionSource(parts.declared_assertions)
    return AsyncOntology(
        id=parts.id,
        version=parts.version,
        entity_types=parts.entity_types,
        relation_types=parts.relation_types,
        entities=entities,
        assertions=assertions,
        taxonomies=parts.taxonomies,
        describes=(entities.describe(),),
        structures={
            name: await AsyncMappingHierarchy.snapshot(
                AsyncAssertionHierarchy(assertions, definition.relation)
            )
            for name, definition in _axes_to_copy(parts.taxonomies)
        },
        imports=parts.imports,
    )


def _axes_to_copy(
    taxonomies: Mapping[str, TaxonomyDefinition],
) -> tuple[tuple[str, TaxonomyDefinition], ...]:
    """The definitions whose structure axis this door must copy at load.

    Shared by the doors rather than written into each. What each door does with
    the answer *is* flavoured -- one snapshot is a coroutine and the other is
    not -- but which axes to take is the same question, and it is the question
    :func:`~dataknobs_common.ontology.values._structure_for` asks again on the
    way out. Two spellings of it is how a door and an accessor come to disagree
    about which axes were copied.

    At load rather than at the accessor because that is the only place both
    flavours can take a copy: ``taxonomy()`` is a plain ``def`` on both twins,
    with nowhere to await the asynchronous snapshot, and it builds afresh on
    every call -- so a copy taken there would be a new copy with a new build
    time each time it was fetched, which is the one property a copy exists to
    not have.
    """
    return tuple(
        (name, definition)
        for name, definition in taxonomies.items()
        if definition.materialization.structure_is_copied
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


def _declared_enum(enum_type: type[_E], declared: Any, field: str, *, noun: str, plural: str) -> _E:
    """Coerce a member of ``enum_type`` out of authored text, naming a miss.

    :class:`ValidationError` does not descend from ``ValueError``, so letting
    the enum raise its own would escape a caller holding the contract both
    doors document -- ``except ValidationError`` -- rather than being caught
    by it.

    That is a property of **every** enum a document can spell, not of the
    first one that needed it. Written once for the same reason
    :func:`_required` is: the second reader of an authored enum is where two
    ways of refusing start to differ, and it arrived the day a document could
    say ``polarity: negated``.
    """
    try:
        return enum_type(str(declared))
    except ValueError as exc:
        raise ValidationError(
            f"{field} is {str(declared)!r}, which is not {noun}. "
            f"{plural}: {sorted(str(member.value) for member in enum_type)}",
            context={"field": field, "value": declared},
        ) from exc


def _inference_mode(declared: Any, field: str) -> InferenceMode:
    """The declared inference mode, or a refusal naming what was written."""
    return _declared_enum(InferenceMode, declared, field, noun="an inference mode", plural="Modes")


def _polarity(declared: Any, field: str) -> Polarity:
    """The declared polarity, or a refusal naming what was written."""
    return _declared_enum(Polarity, declared, field, noun="a polarity", plural="Polarities")


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


def _required(row: Mapping[str, Any], key: str, section: str) -> Any:
    """The value at ``key``, or a refusal naming the section and the row.

    One reader for every required field the loader has, because a bare
    ``row[key]`` raises ``KeyError`` -- which does **not** descend from
    ``ValidationError``, so a caller holding either door's documented
    ``Raises:`` caught nothing at all for the commonest authoring mistake there
    is. That is the same defect :func:`_inference_mode` exists to prevent for a
    malformed enum, arriving through a missing key rather than a bad value.

    It was in every section for one reason: each builder indexed its own keys,
    so the refusal had ten places to be written and was written in none. A
    shared reader is what makes the eleventh required field inherit the
    behaviour instead of repeating the omission.

    **The message identifies the row, not only the field.** ``KeyError:
    'type'`` against a fifty-entity file sends the author to bisect it. The id
    is the handle they have, so it is the one used -- and where the missing
    field *is* the id, the keys the row does carry are what is left to find it
    by. An assertion row reaches that branch as a matter of course, since
    assertions mint their ids rather than declaring them.
    """
    if key in row:
        return row[key]
    raise ValidationError(
        f"`{section}:` {_row_handle(row, key)} declares no {key!r}",
        context={"section": section, "field": key, "id": row.get("id")},
    )


def _row_handle(row: Mapping[str, Any], key: str | None = None) -> str:
    """How a refusal points at the row it is about.

    The id, where the row has one and it is not the field being complained
    about; otherwise the keys the row *does* carry, which is all that is left
    to find it by. Shared by every refusal that is about a row rather than a
    value, so a document author meets one way of being pointed at a line
    rather than one per section.
    """
    row_id = row.get("id")
    if key != "id" and row_id is not None:
        return f"row {row_id!r}"
    return f"a row carrying {sorted(row)}"


#: Per section, the keys this version does not read and what each belongs to.
#:
#: Conditions, cardinality and constraints are one later piece of work: each
#: needs an evaluator, and an evaluator needs a truth value for *unknown*,
#: which widens every read member rather than adding a field. So they are
#: deferred together, and named together here.
#:
#: **A section is here only because its builder calls the refusal.** The shape
#: reads as though it covered every section, and it does not: entity types,
#: entities and taxonomies reach no such check, having no deferred key yet.
#: So adding a section here is not sufficient -- its builder has to call
#: :func:`_refuse_a_phase_2_key` as well, or the entry is inert and the key
#: goes on loading and being discarded, which is the one thing this guard
#: exists to end. ``test_the_refusal_cases_are_the_loader_table`` in
#: ``test_ontology_refusals.py`` is what makes that omission a red test
#: rather than a silent no-op.
_PHASE_2_KEYS: Mapping[str, Mapping[str, str]] = {
    "assertions": {"condition": "conditional assertions arrive"},
    "relation_types": {
        "cardinality": "cardinality arrives",
        "condition": "conditional assertions arrive",
        "constraints": "constraints arrive",
    },
}


def _refuse_a_phase_2_key(row: Mapping[str, Any], section: str) -> None:
    """Refuse a key a later version reads, naming it and that version.

    **Refusing is what makes a deferral legible.** These keys used to load and
    be discarded, which from the author's chair is indistinguishable from
    being honoured: a relation type declaring ``cardinality: one_to_many``
    reported no error and constrained nothing. A dropped key and an
    unsupported key have to look different, or the file says one thing and the
    vocabulary means another.

    Named per key rather than through one "unsupported key" message, because
    the author asked a specific question and the useful answer says which
    version answers it.
    """
    for key, arrival in _PHASE_2_KEYS[section].items():
        if key in row:
            raise ValidationError(
                f"`{section}:` {_row_handle(row)} declares {key!r}, which this "
                f"version does not read: {arrival} in phase 2. Remove the key -- "
                f"one that loads and is discarded reads as one that is honoured",
                context={
                    "section": section,
                    "field": key,
                    "id": row.get("id"),
                    "version": "phase 2",
                },
            )


def _build_entity_types(rows: list[Mapping[str, Any]]) -> dict[str, EntityType]:
    built: dict[str, EntityType] = {}
    for row in rows:
        type_id = str(_required(row, "id", "entity_types"))
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
        name=str(_required(row, "name", "entity_types.attributes")),
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
        relation_id = str(_required(row, "id", "relation_types"))
        _refuse_colon("relation type id", relation_id)
        _refuse_duplicate_id(built, relation_id, "relation_types")
        _refuse_a_phase_2_key(row, "relation_types")
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
        entity_id = str(_required(row, "id", "entities"))
        _refuse_colon("entity id", entity_id)
        if entity_id in built:
            raise ValidationError(
                f"duplicate id {entity_id!r} in `entities:`",
                context={"id": entity_id},
            )
        built[entity_id] = Entity(
            id=entity_id,
            type=str(_required(row, "type", "entities")),
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
        subject = str(_required(row, "subject", "assertions"))
        relation = str(_required(row, "relation", "assertions"))
        obj = _build_term(_required(row, "object", "assertions"))
        # After the required reads, as in `relation_types`: a row has to be
        # well formed before it is worth telling its author which of its
        # optional keys a later version reads.
        _refuse_a_phase_2_key(row, "assertions")
        built.append(
            Assertion(
                id=str(row.get("id") or _mint_assertion_id(subject, relation, obj)),
                subject=subject,
                relation=relation,
                object=obj,
                metadata=dict(row.get("metadata", {})),
                polarity=_polarity(row.get("polarity", Polarity.ASSERTED.value), "polarity"),
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
        taxonomy_id = str(_required(row, "id", "taxonomies"))
        _refuse_duplicate_id(built, taxonomy_id, "taxonomies")
        materialization = row.get("materialization", {})
        built[taxonomy_id] = TaxonomyDefinition(
            id=taxonomy_id,
            relation=str(_required(row, "relation", "taxonomies")),
            name=str(row.get("name", "")),
            description=row.get("description"),
            metadata=dict(row.get("metadata", {})),
            materialization=Materialization(
                structure=_inference_mode(
                    materialization.get("structure", InferenceMode.ON_DEMAND.value),
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

    The traversal, the slug and the collision refusal are
    :func:`~dataknobs_common._nested_core._mint_tree`'s, shared with
    :meth:`~dataknobs_common.hierarchy.MappingHierarchy.from_nested` so that one
    document read through either door mints one set of keys. What is this
    function's own is what the minted nodes become here: entities of a declared
    type, and one assertion per parent edge.

    ``claimed`` is threaded across every spec, so two trees in one document that
    slug to a shared id are refused as a collision rather than the second
    silently replacing the first.
    """
    entities: dict[str, Entity] = {}
    assertions: list[Assertion] = []
    claimed: dict[str, str] = {}
    for spec in specs:
        if str(spec.get("kind", "")) != "nested":
            continue
        source_id = str(spec.get("id", ""))
        node_type = str(spec.get("type", source_id))
        relation = str(spec.get("relation", DEFAULT_NESTED_RELATION))

        for minted in _mint_tree(
            spec.get("tree"),
            child_key=str(spec.get("child_key", "children")),
            name_key=str(spec.get("name_key", "name")),
            source=source_id,
            claimed=claimed,
        ):
            entities[minted.id] = Entity(
                id=minted.id,
                type=node_type,
                name=minted.name,
                aliases=list(minted.node.get("aliases", [])),
                description=minted.node.get("description"),
                metadata=dict(minted.node.get("metadata", {})),
            )
            if minted.parent_id is not None:
                assertions.append(
                    Assertion(
                        id=f"{minted.id}-{relation}-{minted.parent_id}",
                        subject=minted.id,
                        relation=relation,
                        object=EntityRef(entity_id=minted.parent_id),
                    )
                )
    return entities, assertions

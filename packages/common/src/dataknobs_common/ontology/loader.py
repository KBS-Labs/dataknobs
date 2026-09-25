# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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
from collections.abc import Collection, Container, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from dataknobs_common._nested_core import _mint_tree, _walk_tree
from dataknobs_common.config_loading import load_yaml_or_json
from dataknobs_common.entity_resolution.cascade import (
    AsyncCascadingResolver,
    CascadingResolver,
)
from dataknobs_common.entity_resolution.registry import (
    async_signal_backends,
    signal_backends,
)
from dataknobs_common.entity_resolution.signals import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncExactNormalizedSignal,
    AsyncScanningSignal,
    ExactNormalizedSignal,
    ScanningSignal,
)
from dataknobs_common.exceptions import NotFoundError, OperationError, ValidationError
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
    _refuse_colon,
)
from dataknobs_common.ontology.sources import (
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
from dataknobs_common.ontology.values import AsyncOntology, Ontology, OntologyParts, StrCodec

if TYPE_CHECKING:
    from collections.abc import Callable

    from dataknobs_common.hierarchy import AsyncHierarchy, Hierarchy
    from dataknobs_common.entity_resolution.protocols import (
        AsyncEntityResolver,
        AsyncMatchSignal,
        EntityResolver,
        MatchSignal,
    )
    from dataknobs_common.registry import PluginRegistry

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
            refused rather than dropped; on a scalar ``domain:`` or ``range:``,
            which is one type name rather than a list of one; on an attribute
            row that is not a mapping, declares a key an attribute is not read
            for, repeats a name its entity type already declares, or carries a
            value of the wrong shape -- a ``required:`` that is not a boolean,
            a ``name:`` that is not a non-empty string, an ``enum_values:``
            that is not a non-empty list of strings, a ``field_type:`` naming
            no record type; or on a
            reference into a section this document declares -- ``isa``, an
            attribute's ``entity_type``, a relation type's ``domain``,
            ``range`` or ``inverse_of``, an entity's ``type``, an assertion's
            or a taxonomy's ``relation`` -- that names nothing the section
            holds. Every message names the offending value.

            A ``relation:`` resolves against ``relation_types:`` union the
            declared attribute names, and is checked only where
            ``relation_types:`` is non-empty; a ``kind:``-bearing taxonomy row
            declares a label rather than a reference and is exempt. See
            :func:`_declared_relations` and
            :func:`_refuse_undeclared_taxonomy_relations`
    """
    _refuse_reserved_id(config.id)
    _refuse_colon("ontology id", config.id)

    source_specs = tuple(config.sources)
    _validate_source_ids(source_specs)

    entity_types = _build_entity_types(config.entity_types)
    relation_types = _build_relation_types(config.relation_types)
    declared = _build_entities(config.entities)

    _refuse_duplicates(entity_types, relation_types, declared)

    # Six of the eight references into sections this document owns in full,
    # over the raw rows and against the built maps. The two `relation:` ones
    # follow their sections' builders, below. Before the mint, deliberately:
    # what `_mint_nested` produces takes BOTH its `type` and its assertions'
    # `relation` from `sources:` -- the node type it derives and
    # `DEFAULT_NESTED_RELATION` -- rather than from a row an author wrote.
    # That is a different question and not yet a ruled one. Both are named
    # here so the next reader does not have to re-derive that the second was
    # considered; `test_a_minted_type_is_not_a_reference_the_loader_checks`
    # pins the ordering this comment describes.
    #
    # `imports:` switches all eight off, because it is the document saying
    # the premise they rest on does not hold -- see `_owns_its_sections`.
    if _owns_its_sections(config):
        _refuse_undeclared_isa(config.entity_types, entity_types)
        _refuse_undeclared_attribute_types(config.entity_types, entity_types)
        _refuse_undeclared_relation_endpoints(config.relation_types, entity_types, relation_types)
        _refuse_undeclared_entity_types(config.entities, entity_types)

    # After the required reads of both sections, as in `relation_types`: a row
    # has to be well formed before it is worth telling its author which of its
    # keys resolves. Run first, these two reported `assertion on subject
    # 'None'` for a row that had simply omitted `subject:`, and carried that
    # `'None'` into the context as the row's id.
    assertions = list(_build_assertions(config.assertions))
    taxonomies = _build_taxonomies(config.taxonomies)

    if _owns_its_sections(config):
        relations = _declared_relations(entity_types, relation_types)
        _refuse_undeclared_assertion_relations(config.assertions, relations)
        _refuse_undeclared_taxonomy_relations(config.taxonomies, relations)

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
        taxonomies=taxonomies,
        imports=tuple(config.imports),
        declared_entities=declared,
        declared_assertions=tuple(assertions),
        source_specs=source_specs,
        taxonomy_specs=tuple(config.taxonomies),
    )


def load_ontology(
    source: Path | Mapping[str, Any],
    *,
    normalizer: Callable[[str], str] | None = None,
) -> Ontology[str]:
    """Load a vocabulary with synchronous backings.

    No database, no embedder, no event loop. A hand-edited file is already a
    list once read, so nothing here has anything to await and the caller is
    not made to pretend otherwise.

    **``Ontology[str]``, and the parameter is bound rather than passed
    through.** This door reads an *authored* document, whose ids are the
    strings its author typed -- it refuses every live source kind, so there is
    no path here by which a caller's own key space could arrive. The codec is
    therefore :class:`~dataknobs_common.ontology.values.StrCodec`, supplied
    rather than defaulted: an ontology's codec field is required, so a door
    that builds one says which it is.

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
    config = _read_config(source)
    parts = _validated_parts(config)
    entities = MappingEntitySource(parts.declared_entities, normalizer=normalizer)
    assertions = MappingAssertionSource(parts.declared_assertions)
    return assemble_ontology(
        parts,
        entities=entities,
        assertions=assertions,
        describes=(entities.describe(),),
    )


async def async_load_ontology(
    source: Path | Mapping[str, Any],
    *,
    normalizer: Callable[[str], str] | None = None,
) -> AsyncOntology[str]:
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
    return await assemble_async_ontology(
        parts,
        entities=entities,
        assertions=assertions,
        describes=(entities.describe(),),
    )


def assemble_ontology(
    parts: OntologyParts,
    *,
    entities: EntitySource[str],
    assertions: AssertionSource[str],
    describes: tuple[SourceDescription, ...],
    structures: Mapping[str, Hierarchy[str]] | None = None,
) -> Ontology[str]:
    """The value a door returns, from the parts every door has in common.

    Everything a vocabulary carries that is not a *bound source* comes off
    ``parts`` and is the same for every door: the id, the version, the two
    type tables, the taxonomy definitions, the imports, the codec, and which
    structure axes are copied at load. What differs between doors is the three
    keywords below, which is why they are the three parameters.

    **Extracted because there are three doors and one of them is in another
    distribution.** ``OntologyRegistry`` (``dataknobs_data.ontology``) is the
    door that owns a lifecycle and binds live sources, and it assembles a
    vocabulary of its own from parts this same function shapes. Written out at
    each door, a field added to :class:`~dataknobs_common.ontology.Ontology`
    has to be added in three places across a package boundary with nothing
    checking -- and the drift that produces is quiet, because each door's own
    tests pass against its own copy.

    ``describes`` is a parameter rather than ``(entities.describe(),)``
    computed here, although a one-tuple is what all three doors pass today.
    The field is plural in the value's own declaration because a binding over
    several sources describes each of them, and a door composing those is what
    the two "not built yet" refusals in the registry stand in for.

    ``Ontology[str]``, bound rather than generic, for :func:`load_ontology`'s
    reason: ``parts`` came from a document, and a document's ids are the
    strings its author typed.

    Args:
        parts: The validated document, as :func:`build_ontology` maps it
        entities: The bound entity source this vocabulary reads through
        assertions: The bound assertion source, which an axis with no backing
            of its own is read over
        describes: One description per bound source, in binding order
        structures: The axes this door bound, keyed by the name each is
            reached under. ``None`` from a door that binds none, which is
            every synchronous door there is -- see :func:`_bound_structures`

    Returns:
        The vocabulary, with synchronous backings
    """
    bound = dict(structures or {})
    copied: dict[str, Hierarchy[str]] = {}
    for name, definition in _axes_to_copy(parts.taxonomies):
        axis = bound.pop(name, None)
        copied[name] = MappingHierarchy.snapshot(
            AssertionHierarchy(assertions, definition.relation) if axis is None else axis
        )
    return Ontology(
        id=parts.id,
        version=parts.version,
        entity_types=parts.entity_types,
        relation_types=parts.relation_types,
        entities=entities,
        assertions=assertions,
        taxonomies=parts.taxonomies,
        describes=describes,
        codec=StrCodec(),
        structures={**bound, **copied},
        imports=parts.imports,
    )


async def assemble_async_ontology(
    parts: OntologyParts,
    *,
    entities: AsyncEntitySource[str],
    assertions: AsyncAssertionSource[str],
    describes: tuple[SourceDescription, ...],
    structures: Mapping[str, AsyncHierarchy[str]] | None = None,
) -> AsyncOntology[str]:
    """:func:`assemble_ontology`'s twin, and the one the third door calls.

    ``async def`` for the one keyword that differs: a copied structure axis is
    snapshotted from an asynchronous hierarchy, and there is nowhere in a
    ``def`` to await that. Everything else is the synchronous twin's, which is
    why the two sit together rather than each beside the door that calls it.

    Args:
        parts: The validated document, as :func:`build_ontology` maps it
        entities: The bound entity source this vocabulary reads through
        assertions: The bound assertion source
        describes: One description per bound source, in binding order
        structures: The axes this door bound over live backings, keyed by the
            name each is reached under -- a ``kind: column`` axis over a table
            is the case, and the door that binds one is in another
            distribution. ``None`` from a door that binds none

    Returns:
        The vocabulary, with asynchronous backings
    """
    bound = dict(structures or {})
    copied: dict[str, AsyncHierarchy[str]] = {}
    for name, definition in _axes_to_copy(parts.taxonomies):
        axis = bound.pop(name, None)
        copied[name] = await AsyncMappingHierarchy.snapshot(
            AsyncAssertionHierarchy(assertions, definition.relation) if axis is None else axis
        )
    return AsyncOntology(
        id=parts.id,
        version=parts.version,
        entity_types=parts.entity_types,
        relation_types=parts.relation_types,
        entities=entities,
        assertions=assertions,
        taxonomies=parts.taxonomies,
        describes=describes,
        codec=StrCodec(),
        structures={**bound, **copied},
        imports=parts.imports,
    )


def _axes_to_copy(
    taxonomies: Mapping[str, TaxonomyDefinition],
) -> tuple[tuple[str, TaxonomyDefinition], ...]:
    """The definitions whose structure axis a door must copy at load.

    Shared by the two assemblers rather than written into each. What each does
    with the answer *is* flavoured -- one snapshot is a coroutine and the other
    is not -- but which axes to take is the same question, and it is the
    question :func:`~dataknobs_common.ontology.values._structure_for` asks
    again on the way out. Two spellings of it is how a door and an accessor
    come to disagree about which axes were copied.

    **What each copies is now a second thing they agree on without sharing a
    line.** A door may arrive holding an axis it *bound* -- a ``kind: column``
    read over a table is the case -- and where it did, the snapshot is taken of
    that axis rather than of an assertion read over edges the document never
    declared. Both assemblers spell that as ``bound.pop(name, None)`` ahead of
    their own flavour of snapshot, and the reason neither can share the line is
    the same reason they cannot share this one.

    **Private again, and that is the point of the assemblers.** It was briefly
    published, because the third door is in another distribution and had to
    ask this question for itself. It no longer asks: it calls
    :func:`assemble_async_ontology`, which asks on its behalf -- so what
    crosses the package boundary is the whole assembly rather than one
    predicate out of it, and the surface a consumer can reach for is smaller
    by exactly the name that was only ever an ingredient.

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
    module-level loader cannot bind.

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
    _refuse_unbindable_axes(parts.taxonomy_specs)
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


def _owns_its_sections(config: OntologyConfig) -> bool:
    """Whether this document declares the sections a reference points into.

    Every one of the eight reference checks reads *a reference into a section
    this document declares in full*, and there are two ways a document says
    that premise does not hold. The first is a section left empty, which
    :func:`_refuse_an_unresolved_reference` reads per reference. The second is
    ``imports:``, which is global to the document and so is read here.

    **An import is carried and never followed.** Resolving across one needs a
    second vocabulary in scope, which a door loading one file does not have
    -- so a name this document does not declare may be one the import
    declares, and this loader cannot tell that from a typo. Refusing would
    refuse the composition ``imports:`` exists for.

    Without this, the checks are shape-dependent in the way the union was:
    an importing document that declares **no** local ``entity_types:`` is
    exempt by the per-section guard and one that declares a single local type
    is refused for every entity the import types -- which is the local half
    deciding a question about the imported half. The cost is that a typo in
    an importing document is not caught here; the component that holds both
    vocabularies is where that check can be right.
    """
    return not config.imports


def _refuse_an_unresolved_reference(
    value: Any,
    declared: Collection[str],
    *,
    referrer: str,
    field: str,
    noun: str,
    context: Mapping[str, Any],
) -> None:
    """Refuse a reference into a section this document owns in full.

    Layer 1 asks whether a name resolves; layer 2 asks whether a statement is
    true of the graph. This is layer 1, and it is the whole of it. A
    ``subject:`` or an ``object:`` is layer 2 -- a claim a live source may
    complete -- and is not checked here or anywhere.

    ``declared`` empty is NO schema rather than an empty one, so the check does
    not fire -- the regime :func:`build_ontology` already names in a comment of
    its own, and the one :func:`_refuse_undeclared_tree_nodes` already fires
    inside. It matters for five of the eight references and cannot fire for the
    other three, which are written in the very section they point at.

    **Seven callers carry the uniform context ``{section, id, field, value}``
    and one does not.** ``isa`` predates this function and was converted into
    a caller of it on the claim that the conversion could not change a
    verdict; its context is the two keys that refusal already carried,
    ``{entity_type, isa}``, and changing them to match the other seven would
    be exactly the verdict change the conversion promised not to make. The
    price is that a programmatic consumer cannot read all eight the same way.
    Unifying them is a deliberate break of a published context, not a tidy-up
    -- and ``test_an_isa_naming_an_undeclared_type_is_refused`` pins the
    current pair so that it fails rather than drifts.

    Args:
        value: The name the referring row declared
        declared: The ids the target section declares
        referrer: The row, named the way its section names one
        field: The key the reference was declared under
        noun: What the target section holds, singular
        context: What the refusal carries, in its section's own vocabulary
    """
    if not declared:
        return
    if str(value) in declared:
        return
    raise ValidationError(
        f"{referrer} declares `{field}: {value!r}`, which no {noun} in this "
        f"document declares. Declared: {sorted(declared)}",
        context=dict(context),
    )


def _refuse_undeclared_isa(
    rows: list[Mapping[str, Any]], entity_types: Mapping[str, EntityType]
) -> None:
    """A lattice edge has to name a type the same document declares.

    The first of the eight, and the one that predates the rule: it is a caller
    of the shared refusal rather than a second spelling of a check it already
    performs. The guard cannot suppress it -- a row declaring ``isa:`` is
    itself in ``entity_types:``, so the target section is non-empty by the
    existence of the referrer.
    """
    for row in rows:
        parent = row.get("isa")
        if parent is None:
            continue
        _refuse_an_unresolved_reference(
            parent,
            entity_types,
            referrer=f"entity type {str(row.get('id'))!r}",
            field="isa",
            noun="entity type",
            context={"entity_type": row.get("id"), "isa": parent},
        )


def _refuse_undeclared_attribute_types(
    rows: list[Mapping[str, Any]], entity_types: Mapping[str, EntityType]
) -> None:
    """An ``entity:``-typed attribute names the type its values point at.

    Written inside ``entity_types:`` and pointing at it, so the guard is inert
    here for the same reason it is inert for ``isa``.
    """
    for row in rows:
        type_id = str(row.get("id"))
        for attribute in row.get("attributes") or ():
            target = attribute.get("entity_type")
            if target is None:
                continue
            name = str(attribute.get("name"))
            _refuse_an_unresolved_reference(
                target,
                entity_types,
                referrer=f"attribute {name!r} of entity type {type_id!r}",
                field="entity_type",
                noun="entity type",
                context={
                    "section": "entity_types.attributes",
                    "id": name,
                    "field": "entity_type",
                    "value": str(target),
                },
            )


def _refuse_undeclared_relation_endpoints(
    rows: list[Mapping[str, Any]],
    entity_types: Mapping[str, EntityType],
    relation_types: Mapping[str, RelationType],
) -> None:
    """The three references a ``relation_types:`` row carries.

    ``domain`` and ``range`` point at ``entity_types:`` -- a different section,
    so the guard is load-bearing for both: a document that declares relation
    types and leaves its type vocabulary to a live source constrains nothing it
    can check. ``inverse_of`` points at the section the row is written in, so
    for that one it is inert.
    """
    for row in rows:
        relation_id = str(row.get("id"))
        referrer = f"relation type {relation_id!r}"
        for field in ("domain", "range"):
            for target in row.get(field, ()):
                _refuse_an_unresolved_reference(
                    target,
                    entity_types,
                    referrer=referrer,
                    field=field,
                    noun="entity type",
                    context={
                        "section": "relation_types",
                        "id": relation_id,
                        "field": field,
                        "value": str(target),
                    },
                )
        inverse = row.get("inverse_of")
        if inverse is None:
            continue
        _refuse_an_unresolved_reference(
            inverse,
            relation_types,
            referrer=referrer,
            field="inverse_of",
            noun="relation type",
            context={
                "section": "relation_types",
                "id": relation_id,
                "field": "inverse_of",
                "value": str(inverse),
            },
        )


def _refuse_undeclared_entity_types(
    rows: list[Mapping[str, Any]], entity_types: Mapping[str, EntityType]
) -> None:
    """An ``entities:`` row names the type it is an instance of.

    The guard is load-bearing: a document may declare instances and leave the
    type vocabulary to an ontology it imports, and such a document is not
    making a claim this loader can check.
    """
    for row in rows:
        entity_id = str(row.get("id"))
        declared_type = row.get("type")
        if declared_type is None:
            continue
        _refuse_an_unresolved_reference(
            declared_type,
            entity_types,
            referrer=f"entity {entity_id!r}",
            field="type",
            noun="entity type",
            context={
                "section": "entities",
                "id": entity_id,
                "field": "type",
                "value": str(declared_type),
            },
        )


#: What both ``relation:`` refusals call the set they resolve against.
#:
#: One spelling, because the two refusals are about one set: a drift here
#: would have the assertion door and the axis door describe the same union
#: differently in their messages.
_RELATION_NOUN = "relation type or declared attribute"


def _declared_relations(
    entity_types: Mapping[str, EntityType], relation_types: Mapping[str, RelationType]
) -> set[str]:
    """What a ``relation:`` may name: a relation type, or a declared attribute.

    An attribute-valued assertion names an attribute, and an attribute is
    declared *inside* an ``entity_types:`` row -- the section label
    :func:`_build_attribute` reads under, ``entity_types.attributes``. There is
    no top-level ``attributes:`` section, so a reading that looked for one
    would compute an empty half and refuse every attribute-valued assertion in
    the vocabulary.

    **``relation_types:`` alone decides whether the reference is checked, and
    the union only decides what resolves.** This is the one reference of the
    eight whose target is a union, so it is the one where *the section is
    empty* and *this document declares no vocabulary here* are not the same
    sentence: a union is non-empty the moment any entity type declares any
    attribute. Measured over the union, an attribute with nothing to do with
    the reference decides the guard -- ``relation_types:`` becomes
    conditionally mandatory, the same typo is caught in one document and
    missed in the next, and a ``kind: column`` axis in a vocabulary that
    declares one attribute is refused for a ``relation:`` its own guide
    publishes as a label. Returning the empty set here restores the rule the
    other seven follow: one declared section decides, and
    :func:`_refuse_an_unresolved_reference`'s guard reads it.
    """
    if not relation_types:
        return set()
    return set(relation_types) | {
        attribute.name for declared in entity_types.values() for attribute in declared.attributes
    }


def _refuse_undeclared_assertion_relations(
    rows: list[Mapping[str, Any]], relations: Collection[str]
) -> None:
    """An assertion's ``relation:`` names an edge this document declares.

    Runs after :func:`_build_assertions`, so every row reaching it carries
    both keys and ``subject`` is the string that section's own refusal would
    have named.
    """
    for row in rows:
        relation = row.get("relation")
        if relation is None:
            continue
        subject = str(row.get("subject"))
        _refuse_an_unresolved_reference(
            relation,
            relations,
            referrer=f"assertion on subject {subject!r}",
            field="relation",
            noun=_RELATION_NOUN,
            context={
                "section": "assertions",
                "id": subject,
                "field": "relation",
                "value": str(relation),
            },
        )


def _refuse_undeclared_taxonomy_relations(
    rows: list[Mapping[str, Any]], relations: Collection[str]
) -> None:
    """An axis walks one relation, and it has to be one the document declares.

    The failure this refuses is silence rather than an error: an axis over a
    relation nothing declares answers empty for every walk, which reads as a
    vocabulary with nothing in it rather than as a misspelt key.

    **A ``kind:``-bearing row is exempt, because its ``relation:`` is a label
    rather than a reference.** That failure is the assertion axis's: the walk
    reads the assertions carrying this relation, so naming one nothing
    declares walks an empty graph. A column axis reads two columns and
    constructs no ``Assertion`` at all, so its ``relation:`` names what its
    edges *mean* -- which is what the registry guide publishes it as, and
    checking it against ``relation_types:`` would refuse a document that guide
    calls valid. Both module doors refuse a ``kind:`` outright
    (:func:`_refuse_unbindable_axes`), so the exemption is reachable only from
    the registry, which is the door such a row is written for.

    Runs after :func:`_build_taxonomies`, so every row reaching it carries
    both ``id`` and ``relation``.
    """
    for row in rows:
        if "kind" in row:
            continue
        relation = row.get("relation")
        if relation is None:
            continue
        taxonomy_id = str(row.get("id"))
        _refuse_an_unresolved_reference(
            relation,
            relations,
            referrer=f"taxonomy {taxonomy_id!r}",
            field="relation",
            noun=_RELATION_NOUN,
            context={
                "section": "taxonomies",
                "id": taxonomy_id,
                "field": "relation",
                "value": str(relation),
            },
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


def _refuse_unbindable_axes(specs: tuple[Mapping[str, Any], ...]) -> None:
    """Refuse a ``taxonomies:`` row naming a backing this door builds none of.

    :func:`_refuse_live_sources`' sibling, and the second half of one rule:
    *this door binds what it can own, and says so by name for everything else.*
    The one axis backing here is the assertion read, which a row asks for by
    naming **no** ``kind:`` at all -- so a ``kind:`` on such a row is a request
    for a backing another distribution binds, and the message points at the
    door that does.

    Computed from the declared kind alone, exactly as the source refusal is, so
    it holds identically whether or not a package that could bind such an axis
    has been imported. It is therefore correct forever rather than until
    something is registered: *no backing for this kind is built here* does not
    stop being true when one is built somewhere else. What that other door then
    answers -- *no binder in this registry is registered for this kind* -- is a
    different question with a different lifetime.

    **Refused rather than dropped, which is the whole of why it exists.**
    ``TaxonomyDefinition`` reads ``id``, ``relation``, ``name``,
    ``description``, ``metadata`` and ``materialization`` and no others, so a
    row carrying ``kind: column``, ``source:`` and ``parent_key:`` used to load
    with all three discarded -- as an assertion axis over assertions the
    document never declared, which answers empty for every walk. A dropped key
    and an unsupported key have to look different, or the file says one thing
    and the vocabulary means another.

    Not :func:`_refuse_a_phase_2_key`'s mechanism, and the difference is the
    message. That one says *this key arrives in a later version*. A ``kind:``
    on an axis row does not arrive later; it arrives in a **different
    distribution**, and a caller reaching this door needs to be told which one.
    """
    for spec in specs:
        if "kind" not in spec:
            continue
        kind = str(spec.get("kind", ""))
        taxonomy_id = str(spec.get("id", ""))
        raise ValidationError(
            f"taxonomy {taxonomy_id!r} declares kind {kind!r}, which this loader "
            f"builds no structure axis for: binding one reads a table, which "
            f"must be opened and closed again, and a module-level loader owns no "
            f"lifecycle to do that with. Load this ontology through "
            f"OntologyRegistry, which does. The axis this loader builds is the "
            f"assertion read, which a `taxonomies:` row asks for by declaring no "
            f"`kind:` at all",
            context={"taxonomy": taxonomy_id, "kind": kind},
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
#:
#: **``taxonomies:`` does reach a refusal, and it is not this one.** A
#: ``kind:`` on an axis row is refused by :func:`_refuse_unbindable_axes`,
#: which is a different message for a different reason: that key does not
#: arrive in a later version, it arrives in another distribution. The two
#: mechanisms sit beside each other and neither covers the other's case.
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
        parent = row.get("isa")
        built[type_id] = EntityType(
            id=type_id,
            name=str(row.get("name", "")),
            description=row.get("description"),
            aliases=list(row.get("aliases", [])),
            metadata=dict(row.get("metadata", {})),
            attributes=_build_attributes(row.get("attributes"), type_id),
            isa=str(parent) if parent is not None else None,
        )
    return built


#: The keys an attribute row is read for, and the only ones.
#:
#: :data:`TAXONOMY_ROW_KEYS`'s rule on the row next door: a key nothing reads
#: loaded and was discarded, so ``enum:`` written for ``enum_values:`` said
#: nothing and enumerated nothing. ``field_type`` is here because the
#: published documents write it -- ``{type: number, field_type: float}`` -- and
#: the loader is what should honour it, not what should refuse it.
ATTRIBUTE_ROW_KEYS: frozenset[str] = frozenset(
    {"name", "type", "field_type", "entity_type", "required", "enum_values", "description"}
)

_ATTRIBUTES_SECTION = "entity_types.attributes"


def _build_attributes(declared: Any, entity_type: str) -> list[AttributeDef]:
    """An entity type's ``attributes:`` list, each row read, no name twice.

    ``null`` is silence, as everywhere in this loader. A mapping here -- the
    natural mistake of keying attributes by name -- and a row that is not a
    mapping both raised a bare ``AttributeError`` before, which no door's
    ``Raises:`` names.

    **A duplicate name is refused within one row only.** It is the file's
    duplicate-id rule, on a list that was never given it: the second
    declaration shadowed the first for every reader. A subtype redeclaring an
    attribute its parent declares is inheritance, which is a different
    question and not this one.
    """
    if declared is None:
        return []
    if not isinstance(declared, list | tuple):
        raise ValidationError(
            f"`{_ATTRIBUTES_SECTION}:` of entity type {entity_type!r} must be a list "
            f"of attribute rows, got {type(declared).__name__}",
            context={"section": _ATTRIBUTES_SECTION, "entity_type": entity_type},
        )
    built: list[AttributeDef] = []
    seen: set[str] = set()
    for row in declared:
        if not isinstance(row, Mapping):
            raise ValidationError(
                f"`{_ATTRIBUTES_SECTION}:` of entity type {entity_type!r} holds "
                f"{row!r}, which is not an attribute row: each is a mapping "
                f"with at least a `name:`",
                context={"section": _ATTRIBUTES_SECTION, "entity_type": entity_type},
            )
        attribute = _build_attribute(row, entity_type)
        if attribute.name in seen:
            raise ValidationError(
                f"entity type {entity_type!r} declares attribute {attribute.name!r} "
                f"twice; the second declaration would shadow the first for every reader",
                context={
                    "section": _ATTRIBUTES_SECTION,
                    "entity_type": entity_type,
                    "attribute": attribute.name,
                },
            )
        seen.add(attribute.name)
        built.append(attribute)
    return built


def _build_attribute(row: Mapping[str, Any], entity_type: str) -> AttributeDef:
    """One attribute row, every key read for its shape and no other key read.

    Each value used to be coerced rather than checked: ``required: "no"``
    went through ``bool(...)`` and loaded as a *required* attribute,
    ``name: null`` and ``description: null`` through ``str(...)`` and loaded
    as the word ``'None'`` -- in the field an extraction prompt is built from
    -- and ``enum_values: []`` was tested for truthiness, so it read as *not
    enumerated*, the opposite of what was written. Every refusal goes through
    :func:`_refuse_attribute_value`, so the eighth key inherits the message
    rather than writing a ninth.

    ``type:`` stays open: ``entity``, ``enum`` and ``number`` are vocabulary
    types with no record counterpart, and deciding which vocabulary types
    exist is not the loader's business. Only a non-string one is refused.
    """
    name = _required(row, "name", _ATTRIBUTES_SECTION)
    if not isinstance(name, str) or not name:
        _refuse_attribute_value(entity_type, None, "name", name, "a non-empty string")
    unread = sorted(set(row) - ATTRIBUTE_ROW_KEYS)
    if unread:
        raise ValidationError(
            f"attribute {name!r} of entity type {entity_type!r} declares {unread}, "
            f"which this loader does not read. An attribute row is read for "
            f"{sorted(ATTRIBUTE_ROW_KEYS)}",
            context={
                "section": _ATTRIBUTES_SECTION,
                "entity_type": entity_type,
                "attribute": name,
                "keys": unread,
            },
        )
    declared = _attribute_str(row, "type", entity_type, name)
    description = _attribute_str(row, "description", entity_type, name)
    required = row.get("required")
    if required is not None and not isinstance(required, bool):
        _refuse_attribute_value(
            entity_type,
            name,
            "required",
            required,
            "a boolean: it is on or off, so it is `true` or `false`",
        )
    return AttributeDef(
        name=name,
        value_type=declared if declared is not None else "string",
        field_type=_attribute_field_type(row, declared, entity_type, name),
        entity_type=row.get("entity_type"),
        required=bool(required),
        enum_values=_attribute_enum_values(row, entity_type, name),
        description=description if description is not None else "",
    )


def _attribute_str(
    row: Mapping[str, Any], key: str, entity_type: str, attribute: str
) -> str | None:
    """A string-valued attribute key, or ``None`` where it is absent or ``null``."""
    value = row.get(key)
    if value is not None and not isinstance(value, str):
        _refuse_attribute_value(entity_type, attribute, key, value, "a string")
    return value


def _attribute_enum_values(
    row: Mapping[str, Any], entity_type: str, attribute: str
) -> list[str] | None:
    """The values an enumerated attribute allows, or ``None`` for *not enumerated*.

    **An empty list is refused rather than read as absent.** A set of allowed
    values with no members allows no value at all, so no instance could ever
    satisfy the attribute. That is a contradiction rather than a policy,
    which is what separates it from ``rungs: []``: an empty rung composition
    can be carried out, and matches nothing.
    """
    values = row.get("enum_values")
    if values is None:
        return None
    if (
        not isinstance(values, list | tuple)
        or not values
        or not all(isinstance(value, str) for value in values)
    ):
        _refuse_attribute_value(
            entity_type, attribute, "enum_values", values, "a non-empty list of strings"
        )
    return list(values)


def _attribute_field_type(
    row: Mapping[str, Any], declared: str | None, entity_type: str, attribute: str
) -> FieldType | None:
    """The record type an attribute is stored as.

    An explicit ``field_type:`` **overrides** the derivation from ``type:``.
    It is the key for the case the derivation cannot cover -- a vocabulary
    type such as ``number`` stored as a :class:`FieldType` -- and the
    published documents write it for exactly that; it was discarded, so every
    one of them loaded with ``field_type=None``.
    """
    explicit = row.get("field_type")
    if explicit is None:
        return _field_type(declared)
    member = _field_type(explicit) if isinstance(explicit, str) else None
    if member is None:
        _refuse_attribute_value(
            entity_type,
            attribute,
            "field_type",
            explicit,
            f"a record field type, one of {[m.value for m in FieldType]}",
        )
    return member


def _refuse_attribute_value(
    entity_type: str, attribute: str | None, key: str, value: Any, wants: str
) -> NoReturn:
    """The one refusal for an attribute key of the wrong shape."""
    subject = f"attribute {attribute!r}" if attribute is not None else "an attribute"
    raise ValidationError(
        f"{subject} of entity type {entity_type!r} declares {key!r} as {value!r}, "
        f"which is not {wants}",
        context={
            "section": _ATTRIBUTES_SECTION,
            "entity_type": entity_type,
            "attribute": attribute,
            "field": key,
            "value": value,
        },
    )


def _field_type(declared: Any) -> FieldType | None:
    """The record field type a declared type maps onto, if any.

    ``entity`` and ``enum`` are vocabulary concepts with no record-field
    counterpart, so they map to None rather than being forced onto one.
    Case is folded, so ``String`` is a spelling of ``string`` -- the reading
    ``dataknobs-data``'s schema reader gives the same word. Shared with a
    literal's ``type:``, so the two readings of one word cannot diverge.
    """
    if declared is None:
        return None
    try:
        return FieldType(str(declared).lower())
    except ValueError:
        return None


def _endpoints(row: Mapping[str, Any], field: str, relation_id: str) -> frozenset[str]:
    """The entity types a relation's ``domain:`` or ``range:`` permits.

    **A bare string is one type name, not its characters.** ``domain: Person``
    is what a hand-edited file carries, and ``frozenset("Person")`` is six
    one-character type names: the reference check downstream then reported
    ``domain: 'P'`` against a declared list holding the exact word the author
    wrote.

    **Refused rather than coerced**, which is where the two precedents for
    this shape part and why the louder one is followed.
    :data:`~dataknobs_common.ontology.tags.NODE_ID_KEY` has a *reader* take a
    bare string as one node, because a row is data arriving from a source and
    refusing it at read time helps nobody. This is an authored document key,
    and the sibling door refuses: ``index.fields`` takes a list, and a scalar
    there is refused as *one name spelled as its characters rather than a list
    of one*. An author can fix what a refusal names, and coercing would make
    the one form that cannot be right into the one silently accepted.

    Coerced to ``str`` per element for the reason ``entity_types:`` is keyed
    by ``str(id)``: an endpoint left uncoerced passed the reference check by
    ``str(value)`` and was then stored as the raw value, so ``domain`` held
    ``1`` against a map keyed ``'1'`` -- resolved at load, matching nothing
    after it.
    """
    value = row.get(field, ())
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValidationError(
            f"relation type {relation_id!r} declares `{field}: {value!r}`; it takes a "
            f"list of entity type names, and a bare string is one name spelled as its "
            f"characters rather than a list of one",
            context={
                "section": "relation_types",
                "id": relation_id,
                "field": field,
                "value": value,
            },
        )
    return frozenset(str(target) for target in value)


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
            domain=_endpoints(row, "domain", relation_id),
            range=_endpoints(row, "range", relation_id),
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


#: The keys a ``taxonomies:`` row is read for, where it names no backing.
#:
#: Published because the check over them cannot be finished here. A row
#: declaring ``kind:`` is asking for a backing another distribution binds, and
#: the keys *that* door reads -- ``source:``, ``parent_key:`` -- are keys this
#: one has never heard of. So the closed set is enforced here for a row with no
#: ``kind:``, and handed over for a row that has one, which is the same split
#: :func:`_refuse_unbindable_axes` makes about the row as a whole.
TAXONOMY_ROW_KEYS: frozenset[str] = frozenset(
    {"id", "relation", "name", "description", "metadata", "materialization"}
)


def _refuse_an_unread_taxonomy_key(row: Mapping[str, Any], taxonomy_id: str) -> None:
    """Refuse a key on an axis row that nothing here reads.

    :func:`_refuse_unbindable_axes` states the rule this enforces -- *a
    dropped key and an unsupported key have to look different, or the file
    says one thing and the vocabulary means another* -- and enforced it for
    ``kind:`` alone. Every other key on the same row went on loading and being
    discarded: a misspelt ``materialisation:`` configured nothing and said
    nothing, which is the failure that rule is about.

    **A row declaring a ``kind:`` is passed over**, because the closed set
    here is not the closed set for such a row: it names a backing whose binder
    reads keys of its own, in a distribution this module does not import. That
    door owns the check over its own rows, and refuses the same way -- see
    ``ColumnAxisBinding.from_mapping`` in ``dataknobs-data``. The module-level
    doors never see such a row at all, :func:`_refuse_unbindable_axes` having
    refused it first.
    """
    if "kind" in row:
        return
    unread = sorted(set(row) - TAXONOMY_ROW_KEYS)
    if not unread:
        return
    raise ValidationError(
        f"`taxonomies:` row {taxonomy_id!r} declares {unread}, which this "
        f"loader does not read. An axis over this document's own assertions is "
        f"read for {sorted(TAXONOMY_ROW_KEYS)}; a row asking for any other "
        f"backing declares `kind:`, and the door that binds that kind reads "
        f"the keys it needs",
        context={"taxonomy": taxonomy_id, "keys": unread},
    )


def _build_taxonomies(rows: list[Mapping[str, Any]]) -> dict[str, TaxonomyDefinition]:
    built: dict[str, TaxonomyDefinition] = {}
    for row in rows:
        taxonomy_id = str(_required(row, "id", "taxonomies"))
        _refuse_duplicate_id(built, taxonomy_id, "taxonomies")
        _refuse_an_unread_taxonomy_key(row, taxonomy_id)
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


# --------------------------------------------------------------------------
# The runtime -- a second function, because a loaded ontology is a value
# --------------------------------------------------------------------------


def build_resolver(
    config: Path | Mapping[str, Any],
    ontology: Ontology[str],
    *,
    handles: Mapping[str, Any] | None = None,
) -> EntityResolver[str]:
    """Build the placement cascade a document configures.

    **The key parameter is bound here rather than carried**, and the binding is
    written down rather than left to the default. The rungs this assembles and
    the cascade under them are ``str``-keyed: every annotation in
    ``entity_resolution.cascade`` and ``entity_resolution.signals`` names the
    key as ``str``, so a resolver built here answers with ``str`` ids whatever
    the ontology handed in is keyed by. Spelling that as ``Ontology[str]``
    makes a vocabulary keyed by something else a **type error at this call**
    instead of an ``Any`` that type-checks and comes back with keys of the
    wrong space. See :class:`~dataknobs_common.entity_resolution.CascadeState`
    for the boundary and what it would take to move it.

    A second function rather than something :func:`load_ontology` returns,
    because an ``Ontology`` is a **value**: it owns no lifecycle and has
    nowhere to put a runtime. A caller wanting both makes two calls and holds
    two objects, which is the price of the value staying a value.

    The first argument is what :func:`load_ontology` takes, so the two read as
    siblings and a caller can pass the path it already has. That means the
    document is read twice, which is likewise what a pure value costs: nothing
    retains the document it came from, so something has to supply it again.

    Args:
        config: A path to a YAML or JSON document, or the document itself --
            the same argument :func:`load_ontology` takes.
        ontology: The loaded vocabulary the rungs match against.
        handles: Live objects a rung is constructed over and a document
            cannot write. :func:`async_build_resolver` documents the channel
            and the merge order, which are the same here.

            **Symmetric even though this distribution ships no synchronous
            rung that needs one**, which is the whole argument for it: the
            registry these doors read is a published extension point, so the
            rung that needs a handle and has a synchronous form is a
            consumer's to write and this is the door they would reach for.
            Withholding the channel would leave them assembling a
            ``CascadingResolver`` beside this function -- a second
            implementation of it, which is the thing
            :func:`_configured_rungs` exists to prevent. The asymmetry was
            argued from the shipped rungs, and the shipped rungs are not who
            an extension point is for.

    Returns:
        A synchronous resolver over the configured rungs.

    Raises:
        ValidationError: For a ``resolver:`` section that is not a mapping
            or declares a key other than ``rungs``, one naming a rung this
            flavour cannot build, a malformed rung entry, an unknown
            ``kind:``, or a rung whose own factory refused the configuration
        ConfigLoadError: For any refusal in reading ``config`` as a document
        OSError: From that same read
    """
    section = _read_config(config).resolver
    _refuse_async_only_rungs(section)
    refuse_unbuildable_rungs(section, registry=signal_backends)
    return CascadingResolver(_sync_rungs(section, ontology, handles=handles), ontology.entities)


async def async_build_resolver(
    config: Path | Mapping[str, Any],
    ontology: AsyncOntology[str],
    *,
    handles: Mapping[str, Any] | None = None,
) -> AsyncEntityResolver[str]:
    """:func:`build_resolver` for a cascade whose rungs reach for data.

    Keyed by ``str`` for the reason the synchronous door states, and declared
    the same way.

    The remedy the synchronous door's refusal names, **for the rungs whose
    remedy it is**. A refusal whose remedy builds nothing is not a remedy,
    which is why this ships in the same increment as the refusal that points
    at it --- and why *handles* exists: a rung reaching for data is
    constructed over a live object, this function's caller may be holding one,
    and until there was a channel for it the refusal pointed here at a door
    that could only fail differently. See
    :func:`_refuse_async_only_rungs`, which now sends such a kind to a door
    that holds what it reaches for.

    ``async`` because the read is offloaded, exactly as
    :func:`async_load_ontology` offloads it. The prefix names the flavour of
    what is built as much as the callability of the builder -- both are true
    here.

    **The synchronous twin takes it too.** It did not, and the asymmetry was
    argued from the shipped rungs -- *the synchronous door builds no rung that
    needs a handle, because the one rung that does has no synchronous form*.
    That is true and is not the question: the registry both doors read is a
    published extension point, so the rung with a synchronous form and a
    handle to be constructed over is a consumer's, and a channel they cannot
    reach is one they reimplement. Both doors carry it, on one body.

    Args:
        config: A path to a YAML or JSON document, or the document itself.
        ontology: The loaded vocabulary the rungs match against.
        handles: Live objects a rung is constructed over and a document
            cannot write --- an index, a store, a client. Forwarded into
            **every** rung's spec, because a factory reads the keys it names
            and ignores the rest, so one mapping serves a composition whose
            rungs need different things.

            **The merge order is a rule, not an implementation detail.**
            Handles beat the document, because a document cannot write a live
            object and a key spelled the same in both is the process's;
            ``entities`` beats handles, because it is this door's one
            guarantee --- that every rung matches against the ontology the
            caller handed in --- and a channel able to displace it is a
            channel able to bypass the door.

            Named ``handles`` rather than typed per rung because this package
            may not name a type from a package that depends on it. What
            arrives is whatever the caller holds, and the rung's own factory
            is what refuses a missing one by name.

    Returns:
        An asynchronous resolver over the configured rungs.

    Raises:
        ValidationError: For a ``resolver:`` section this door cannot build --
            one that is not a mapping or declares a key other than ``rungs``,
            a malformed rung entry, an entry naming no ``kind:``, a ``kind:``
            nothing registers, or a rung whose own factory refused the
            configuration, handles included. **One type for all of them**,
            which is what this line has always said and what
            :func:`_configured_rungs` now makes true
        ConfigLoadError: For any refusal in reading ``config`` as a document
        OSError: From that same read
    """
    if isinstance(config, Path):
        read = await asyncio.to_thread(_read_config, config)
    else:
        read = _read_config(config)
    refuse_unbuildable_rungs(read.resolver, registry=async_signal_backends)
    return AsyncCascadingResolver(
        _async_rungs(read.resolver, ontology, handles=handles), ontology.entities
    )


#: The keys a ``resolver:`` section is read for, and the only ones.
#:
#: Published because two doors read the section: the build doors here, and
#: ``dataknobs-data``'s ``OntologyRegistry``, which refuses a stray key before
#: it opens a store. Both answer from this set, so the key set is written once
#: and in the package that decides what the section means.
RESOLVER_SECTION_KEYS: frozenset[str] = frozenset({"rungs"})


def _rung_specs(section: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...] | None:
    """The rungs a ``resolver:`` section declares, or ``None`` for silence.

    ``None`` and ``()`` are different answers and the difference is
    load-bearing. A document declaring no ``resolver:`` at all has said
    nothing, and gets the default composition. A document declaring
    ``rungs: []`` has written a composition -- an empty one -- and gets a
    cascade that misses everything, because the composition *is* the policy
    and a consumer who wants no rungs must be able to say so. ``{}`` reads the
    same way, as a section that composes nothing.

    **A key other than ``rungs`` is refused**, and that is the third case.
    Read with ``.get`` alone, ``resolver: {rung: [...]}`` -- one letter short
    -- was a section with no ``rungs:``, so it built a cascade that matches
    nothing and said nothing. A section that is not a mapping is refused too;
    it raised a bare ``AttributeError`` before, which no door's ``Raises:``
    names.
    """
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValidationError(
            f"`resolver:` must be a mapping, got {type(section).__name__}",
            context={"resolver": section},
        )
    unread = sorted(set(section) - RESOLVER_SECTION_KEYS)
    if unread:
        raise ValidationError(
            f"`resolver:` declares {unread}, which this loader does not read. The "
            f"section is read for {sorted(RESOLVER_SECTION_KEYS)}, and one with no "
            f"`rungs:` is read as a composition of nothing -- so a key spelt any "
            f"other way builds a cascade that matches nothing",
            context={"keys": unread},
        )
    rungs = section.get("rungs")
    if rungs is None:
        return ()
    if not isinstance(rungs, list):
        raise ValidationError(
            f"`resolver.rungs:` must be a list, got {type(rungs).__name__}",
            context={"rungs": rungs},
        )
    return tuple(rungs)


def _refuse_async_only_rungs(section: Mapping[str, Any] | None) -> None:
    """Refuse a rung kind this door cannot build.

    Computed from the declared **kind** alone, so it holds with nothing
    constructed and identically whether or not a package that implements such
    a rung has been imported.

    **Called by :func:`build_resolver` alone, and by neither loader.** The
    subject of this refusal is *this door cannot build that rung*, which is
    false of a door that builds no rung: an :class:`Ontology` is a value, and
    a caller loading one for its entity types would otherwise be refused over
    a ``resolver:`` section it never reads. The cost is that such a caller is
    no longer told early; the refusal still arrives in full at the first call
    that proposes to build the thing.

    This is the first refusal in this module that is *not* shared by a door
    and its twin, and the asymmetry is the point rather than an oversight.
    The two pairs are different pairs: :func:`_validated_parts` refuses what
    **neither loader** can own, and this refuses what **one build door**
    cannot build.

    **The remedy is per rung, because one remedy was false for two thirds of
    them.** This used to send every refused kind to
    :func:`async_build_resolver`, on the reasoning that the asynchronous door
    must accept what the synchronous one refuses. That holds for a rung whose
    asynchrony is its own -- an ``Ontology`` is all it needs, and that door
    has one. It does not hold for a rung that **reaches for data**: such a
    rung is constructed over a handle, this function's caller has none, and no
    body of :func:`async_build_resolver`'s signature could invent one. A
    caller following the old sentence got a different error rather than a
    rung, which is what
    :func:`async_build_resolver`'s own docstring calls *a refusal whose remedy
    builds nothing*.

    **Nor does it hold for a rung that is simply not here yet**, which is the
    third case and the one the two-way split got wrong. A mark says one of two
    things: *this flavour of the rung does not exist* --- and then the other
    door is where it lives --- or *the rung exists in this flavour and ships
    somewhere else*, and then **this** door builds it as soon as the module
    that registers it is imported. Sending the second case to
    :func:`async_build_resolver` names a door carrying the identical mark, so
    the caller follows the sentence and gets the same refusal back. Measured:
    ``authority`` was the only kind that could reach the old ``else`` at all,
    and the sentence it got was wrong for it.

    The fact that tells the three apart is already on the mark, and it is
    ``flavour`` rather than ``needs_io``: the flavour a mark declares is the
    flavour the rung *has*, so a mark in this registry declaring ``"sync"``
    says the synchronous form exists and is unregistered. ``needs_io`` then
    separates the remaining two. Both are read off the mark rather than
    matched against a list of kinds, so a rung anyone adds is covered without
    an edit here.
    """
    for at, spec in enumerate(_rung_specs(section) or ()):
        # Through the shared reader rather than `spec.get`, which is what this
        # loop used and which raised ``AttributeError`` on a `rungs:` entry
        # that is not a mapping -- the same latent defect as the one
        # :func:`_rung_kind` was written for, one function earlier in the call
        # order, found by reading the neighbours of the first.
        kind = _rung_kind(spec, at)
        reason = signal_backends.unavailable_reason(kind)
        if reason is None:
            continue
        metadata = signal_backends.get_metadata(kind)
        if metadata.get("flavour") == "sync":
            remedy = (
                "A synchronous form of this rung exists and is not registered in "
                "this process, which is what the reason above says how to fix -- "
                "import the module that registers it and this door builds the kind"
            )
        elif metadata.get("needs_io"):
            remedy = (
                "Such a rung is built over a live handle -- an index, a store -- "
                "which this door has no way to supply. Build it with a door that "
                "holds one: dataknobs_data.ontology.OntologyRegistry assembles the "
                "cascade at load from the same `resolver:` section, or pass the "
                "handles yourself to async_build_resolver(..., handles={...})"
            )
        else:
            remedy = (
                "Build it with async_build_resolver, over an ontology from "
                "async_load_ontology, which builds one"
            )
        raise ValidationError(
            f"rung kind {kind!r} cannot be built by this door: {reason}. "
            f"{remedy}. Kinds this door builds: {sorted(signal_backends.list_keys())}",
            context={"kind": kind, "reason": reason},
        )


def _rung_kind(spec: Any, at: int) -> str:
    """The ``kind:`` one rung entry names, or a refusal naming the entry.

    **Two shapes reached the registry un-refused and surfaced as stdlib
    types**, which is the failure the doors' ``Raises:`` sections exist to
    rule out: a ``rungs:`` list holding a bare string produced ``TypeError:
    'str' object is not a mapping`` from the spec merge, and an entry with no
    ``kind:`` produced ``ValueError: config must contain 'kind'`` from inside
    :meth:`~dataknobs_common.registry.PluginRegistry.create`'s key
    resolution --- raised *before* that method's own wrapper, so nothing
    downstream could convert it either.

    Both are documents to fix, and the position is what makes the message
    actionable: a composition of six rungs gives a reader one index rather
    than six candidates.
    """
    if not isinstance(spec, Mapping):
        raise ValidationError(
            f"`resolver.rungs[{at}]` is {type(spec).__name__}, not a mapping. Each "
            f"entry is a rung's own configuration and names its `kind:` -- "
            f"`- kind: exact` rather than `- exact`",
            context={"at": at, "rung": spec},
        )
    kind = spec.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValidationError(
            f"`resolver.rungs[{at}]` names no `kind:`, so there is nothing to build "
            f"it from. Every rung entry names the kind it is; the rest of the entry "
            f"is that kind's own configuration",
            context={"at": at, "rung": dict(spec)},
        )
    return kind


def _rung_fault(exc: Exception) -> str:
    """What to say about a rung a registry could not build.

    :meth:`~dataknobs_common.registry.PluginRegistry.create` bounds its own
    message **on purpose** --- a plugin factory builds a backend from
    deployment configuration, so the exception it wraps can be a driver's text
    carrying a connection URL, and the registry's answer names the key and
    keeps the rest on ``__cause__``. That reasoning is right and this does not
    defeat it.

    What it does is read one exception type back out: a ``ValidationError``
    from a rung factory describes a **document**, and it is the one that says
    which handle was missing or which value was written where a handle
    belongs. Losing it would leave a document author holding "failed to create
    plugin" for a fault they can fix in one line.

    **The claim is about intent, not about provenance.** Every rung factory in
    this distribution raises ``ValidationError`` only for a document fault, so
    for those the unwrapping is exact. A consumer's own factory is theirs, and
    one raising ``ValidationError`` over a connection it could not open would
    have that text read back out here. The trade is deliberate: the alternative
    withholds the diagnosis from every document fault to bound a message no
    in-tree factory writes, and a consumer who wants the bound keeps their own
    failures under a type that is not this one.
    """
    cause = exc.__cause__
    if isinstance(exc, OperationError) and isinstance(cause, ValidationError):
        return str(cause)
    return str(exc)


def _configured_rungs(
    specs: tuple[Any, ...],
    registry: PluginRegistry[Any],
    *,
    entities: Any,
    handles: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Every rung a composition declares, built through one registry.

    **One body for both flavours**, because the two differ in which registry
    they ask and in nothing else --- and because everything *around* the ask
    is what the two kept getting differently. A second copy is where the
    refusal shapes drift, which is exactly what happened: the asynchronous
    side grew a *handles* channel and the refusals were normalized one
    registry at a time.

    **The refusal type is the doors' documented one, for every way a rung can
    fail to build.** Both doors' ``Raises:`` promise ``ValidationError`` for a
    ``resolver:`` section they cannot build; what actually escaped was a
    ``NotFoundError`` for a misspelled ``kind:``, an ``OperationError`` around
    a factory's refusal, and two stdlib types for a malformed entry. A caller
    catching what the docstring named caught none of them, and the one
    consumer that noticed wrote the conversion on its own side --- where it
    served that consumer and no other caller of these doors.

    Args:
        specs: The rung entries, in the order the document writes them.
        registry: The signal registry of this flavour.
        entities: The vocabulary's entity source, which every rung matches
            against and which no document may displace.
        handles: Live objects a rung is constructed over --- see
            :func:`async_build_resolver`, which documents the merge order this
            implements.

    Returns:
        One rung per spec, in order.

    Raises:
        ValidationError: For any entry this registry cannot turn into a rung,
            naming the entry's position and its kind.
    """
    supplied = dict(handles or {})
    if "kind" in supplied:
        # The one key a handle may not carry. Every refusal that runs before
        # construction reads `kind:` off the *document*, and the registry
        # resolves it off the *merged* config -- so a handle spelled this way
        # would have the composition checked as one kind and built as another,
        # silently. `entities` needs no such guard: it is overwritten below
        # rather than read, which is the door's published guarantee.
        raise ValidationError(
            "`handles` may not carry 'kind': it is what names the rung, so a handle "
            "spelled this way would redirect every rung in the composition to one "
            "factory while the document still reads as naming several. Handles are "
            "for the objects a rung is constructed over, not for what it is",
            context={"handles": sorted(supplied)},
        )
    built: list[Any] = []
    for at, spec in enumerate(specs):
        kind = _rung_kind(spec, at)
        try:
            built.append(registry.create(config={**spec, **supplied, "entities": entities}))
        except (NotFoundError, OperationError, ValueError, TypeError) as exc:
            raise ValidationError(
                f"`resolver.rungs[{at}]` names kind {kind!r}, which this door could "
                f"not build: {_rung_fault(exc)}",
                context={"at": at, "kind": kind},
            ) from exc
    return built


def refuse_unbuildable_rungs(
    section: Mapping[str, Any] | None, *, registry: PluginRegistry[Any]
) -> None:
    """Refuse a ``resolver:`` composition a registry cannot build, constructing nothing.

    :func:`_refuse_async_only_rungs`'s property, over the other two ways a
    composition can be unbuildable: an entry that is not a rung's
    configuration at all, and a ``kind:`` nothing registers. Computed from the
    document and the registry alone, so it holds before anything is
    constructed and identically whichever caller asks.

    **Published because the moment matters to a caller that is not a door.**
    :class:`~dataknobs_data.ontology.OntologyRegistry` opens a vector store on
    its way to building a cascade, and its ``index:`` reader is explicit that
    a document which cannot build must not *"leave a handle behind proving it
    tried"*. Its ``resolver:`` reader ran after that store was open, so a
    misspelled ``kind:`` stranded one --- with no registry object yet returned
    for the caller to ``close()``. The refusal it needed is this one, and a
    second copy on that side is a second thing to keep in step with the door.

    Args:
        section: The ``resolver:`` section, or ``None`` for a document that
            declared none --- which is silence and refuses nothing.
        registry: The signal registry of the flavour about to build. The
            answer is per flavour: a kind registered in one and marked in the
            other is buildable by exactly one of these doors.

    Raises:
        ValidationError: For a section that is not a mapping or declares a
            key other than ``rungs``; for an entry that is not a mapping, one
            naming no ``kind:``, or one naming a kind this registry cannot build ---
            carrying the mark's own reason where there is one, because *the
            kind is unknown* and *the kind ships elsewhere* are different
            faults with different remedies.
    """
    for at, spec in enumerate(_rung_specs(section) or ()):
        kind = _rung_kind(spec, at)
        if registry.is_registered(kind):
            continue
        reason = registry.unavailable_reason(kind)
        raise ValidationError(
            f"`resolver.rungs[{at}]` names kind {kind!r}, which this flavour does not "
            f"build: {reason or 'no such rung kind is registered'}. Kinds it builds: "
            f"{sorted(registry.list_keys())}",
            context={"at": at, "kind": kind, "reason": reason},
        )


def _sync_rungs(
    section: Mapping[str, Any] | None,
    ontology: Ontology,
    *,
    handles: Mapping[str, Any] | None = None,
) -> list[MatchSignal]:
    """The synchronous rungs a section configures, or the default composition."""
    specs = _rung_specs(section)
    if specs is None:
        return _default_sync_rungs(ontology)
    rungs: list[MatchSignal] = _configured_rungs(
        specs, signal_backends, entities=ontology.entities, handles=handles
    )
    return rungs


def _async_rungs(
    section: Mapping[str, Any] | None,
    ontology: AsyncOntology,
    *,
    handles: Mapping[str, Any] | None = None,
) -> list[AsyncMatchSignal]:
    """The asynchronous rungs a section configures, or the default composition.

    **The one place a ``resolver:`` section becomes rungs in this flavour**,
    which is what *handles* is for: a caller holding live objects the
    document cannot name reaches rung construction through here rather than
    assembling a list of its own beside it. A second builder would be a
    second implementation of this function, and the one it would drift from
    is this one.

    *handles* reaches the **configured** path only. The default composition
    is this package's three rungs over an entity source, none of which takes
    a handle, so a document that configures nothing has nothing to forward
    them to.
    """
    specs = _rung_specs(section)
    if specs is None:
        return _default_async_rungs(ontology)
    rungs: list[AsyncMatchSignal] = _configured_rungs(
        specs, async_signal_backends, entities=ontology.entities, handles=handles
    )
    return rungs


def _default_sync_rungs(ontology: Ontology) -> list[MatchSignal]:
    """What a document that configures nothing gets.

    The cascade's declared order minus the rung that needs an index -- exact,
    then alias, with no vector leg because a synchronous cascade has none to
    fall back to -- and then the scan, which needs no index either.

    Silence has to build *something*, because the worked call site loads a
    file declaring no ``resolver:`` and then asserts on candidates that report
    the rung which produced them. *Nothing configured* cannot mean *nothing
    built* without that assertion having nothing to be true of.

    **The scan is here because silence has to build something a caller can
    use.** Without it the default compares the whole query and nothing else,
    so a consumer who configures nothing hands over a sentence and gets no
    candidates, an empty ``coverage.matched`` and the whole string unmatched
    -- the question spans were added to answer, unanswerable by the
    composition a document gets for free. It does not replace
    :class:`ExactNormalizedSignal`: a probe is a slice between token
    boundaries, so a declared form whose own first or last character is not
    alphanumeric is reachable by the whole-string rungs and by no scan.

    **Last rather than first**, which decides nothing about what the cascade
    answers and one thing about what it reports. Measured over every query the
    suite and the guides ask, the three rungs return the same candidates at the
    same spans with the same coverage whichever end the scan sits at, because
    the rungs never disagree -- they read one index two ways. What the position
    decides is the rung of *record* for a query the whole-string rungs already
    answer: last leaves that ``exact``, so the slot-filling caller whose string
    *is* the phrase reads the same evidence it always did and the scan is
    appended to it. A caller who wants the locating rung to lead writes the
    composition out; the composition is the policy.

    The rungs inherit the loader's normalizer without being handed one --
    because they do not fold at all. The source folded its forms with whatever
    ``load_ontology`` was given and folds a lookup the same way, so a query
    reaching it matches exactly the way that ontology was loaded. A rung
    folding first would be a *second*, different fold over an answer the
    vocabulary had already decided, and this is the door that cannot pass the
    right one down: an ``Ontology`` is a value, and the normalizer it was
    built with is not on it.

    **That they fold not at all is also what keeps the scan's bound.** A rung
    handed its own ``normalizer`` stops asking the source how wide a window
    may be, since no source-side measurement can see a fold applied after it;
    a document that configures nothing hands none over, so the composition
    here keeps the bound the vocabulary implies.
    """
    return [
        ExactNormalizedSignal(ontology.entities),
        AliasSignal(ontology.entities),
        ScanningSignal(ontology.entities),
    ]


def _default_async_rungs(ontology: AsyncOntology) -> list[AsyncMatchSignal]:
    """:func:`_default_sync_rungs`' twin."""
    return [
        AsyncExactNormalizedSignal(ontology.entities),
        AsyncAliasSignal(ontology.entities),
        AsyncScanningSignal(ontology.entities),
    ]

"""Every refusal fires from both doors, and names what it refused.

Parametrized over the two doors rather than written twice, because the claim
being tested is precisely that the two agree. A pair of hand-written suites
would let one drift and still report green.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.fields import FieldType
from dataknobs_common.ontology import async_load_ontology, build_ontology, load_ontology
from dataknobs_common.ontology import loader as loader_module
from dataknobs_common.ontology.config import OntologyConfig

if TYPE_CHECKING:
    from pathlib import Path

    from dataknobs_common.ontology import AsyncOntology, Ontology
    from dataknobs_common.ontology.model import AttributeDef


def _sync_door(document: Mapping[str, Any]) -> None:
    load_ontology(document)


def _async_door(document: Mapping[str, Any]) -> None:
    asyncio.run(async_load_ontology(document))


DOORS = pytest.mark.parametrize(
    "door", [_sync_door, _async_door], ids=["load_ontology", "async_load_ontology"]
)

Door = Callable[[Mapping[str, Any]], None]


@DOORS
def test_the_reserved_ontology_id_is_refused(door: Door) -> None:
    """``dk`` names the built-in pseudo-ontology, so nobody else may claim it."""
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "dk"})

    assert "'dk'" in str(excinfo.value)
    assert excinfo.value.context["ontology_id"] == "dk"


@DOORS
def test_a_colon_in_the_ontology_id_is_refused(door: Door) -> None:
    """``:`` separates the parts of a qualified id, so an id may not carry one."""
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "retail:catalog"})

    assert "'retail:catalog'" in str(excinfo.value)


@DOORS
def test_a_colon_in_an_entity_id_is_refused(door: Door) -> None:
    """The same rule one level down: local ids are namespaced by construction."""
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "entities": [{"id": "a:b", "type": "T"}]})

    assert "'a:b'" in str(excinfo.value)


@DOORS
def test_a_duplicate_entity_id_is_refused(door: Door) -> None:
    """Two rows, one id: the second would silently win."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entities": [
                    {"id": "dog", "type": "Species"},
                    {"id": "dog", "type": "Breed"},
                ],
            }
        )

    assert "'dog'" in str(excinfo.value)


@DOORS
def test_an_id_declared_in_two_sections_is_refused(door: Door) -> None:
    """Types and entities share one store, so a cross-section collision is real."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entity_types": [{"id": "Species"}],
                "entities": [{"id": "Species", "type": "Species"}],
            }
        )

    message = str(excinfo.value)
    assert "'Species'" in message
    assert "entity_types" in message and "entities" in message


@DOORS
def test_an_isa_naming_an_undeclared_type_is_refused(door: Door) -> None:
    """A lattice edge to a type that does not exist is a typo.

    Not a forward reference: the document declares its own types, so every
    parent it can legally name is somewhere in the same section.

    **Message and context pinned exactly, because this one is a regression
    guard as well as a refusal.** ``isa`` is the check that predates the
    shared refusal and was converted into a caller of it, on the claim that
    the conversion could not change a verdict. A substring assertion would
    pass against a rewritten message or a changed context dict, which is to
    say it would not hold the claim it exists for. The context keeps this
    section's own two keys rather than the uniform four the other seven
    carry -- see :func:`_refuse_an_unresolved_reference` -- and that is
    itself part of what is pinned.
    """
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "entity_types": [{"id": "Breed", "isa": "Speceis"}]})

    assert str(excinfo.value) == (
        "entity type 'Breed' declares `isa: 'Speceis'`, which no entity type "
        "in this document declares. Declared: ['Breed']"
    )
    assert excinfo.value.context == {"entity_type": "Breed", "isa": "Speceis"}


@DOORS
def test_an_isa_declared_later_in_the_file_is_accepted(door: Door) -> None:
    """Order within the section is not meaningful, so it must not decide."""
    door({"id": "x", "entity_types": [{"id": "Breed", "isa": "Species"}, {"id": "Species"}]})


@DOORS
def test_a_live_source_kind_is_refused_naming_id_kind_and_the_loader(
    door: Door,
) -> None:
    """A door cannot own something that has to be closed.

    The message must carry all three: which source, what kind it declared, and
    what to use instead -- an operator reading a boot log has only the message.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "sources": [{"id": "clinic_db", "kind": "record", "table": "species"}],
            }
        )

    message = str(excinfo.value)
    assert "'clinic_db'" in message
    assert "'record'" in message
    assert "OntologyRegistry" in message
    assert excinfo.value.context == {"source_id": "clinic_db", "kind": "record"}


@DOORS
def test_an_axis_kind_this_door_binds_no_backing_for_is_refused(door: Door) -> None:
    """The source refusal's sibling, for the other half of a document.

    `TaxonomyDefinition` reads six keys and `kind:` is not one of them, so a
    row asking for a backing this door does not build used to load with all
    three of its keys discarded -- as an assertion axis over assertions the
    document never declared, which answers empty for every walk. A dropped key
    and an unsupported key have to look different.

    The message carries the same three things the source refusal does: which
    axis, what kind it declared, and what to use instead.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "taxonomies": [
                    {
                        "id": "categories",
                        "kind": "column",
                        "source": "products",
                        "parent_key": "parent_sku",
                        "relation": "parent",
                    }
                ],
            }
        )

    message = str(excinfo.value)
    assert "'categories'" in message
    assert "'column'" in message
    assert "OntologyRegistry" in message
    assert excinfo.value.context == {"taxonomy": "categories", "kind": "column"}


@DOORS
def test_an_axis_declaring_no_kind_is_the_one_this_door_builds(door: Door) -> None:
    """The negative half: silence is how a row asks for the assertion read.

    Without it the refusal above is satisfied by a door that refuses every
    `taxonomies:` row there is, which would be a worse failure than the one it
    replaced.
    """
    door(
        {
            "id": "x",
            "entity_types": [{"id": "Breed"}],
            "entities": [
                {"id": "beagle", "type": "Breed", "name": "Beagle"},
                {"id": "dog", "type": "Breed", "name": "Dog"},
            ],
            "assertions": [{"subject": "beagle", "relation": "isa", "object": "dog"}],
            "taxonomies": [{"id": "kinds", "relation": "isa"}],
        }
    )


@DOORS
def test_a_taxonomy_row_refuses_a_key_this_loader_does_not_read(door: Door) -> None:
    """The rule `kind:` is refused under, applied to the rest of the row.

    `_build_taxonomies` reads six keys. Every other key on a `taxonomies:` row
    loaded and was discarded -- which from the author's chair is
    indistinguishable from being honoured, and is the exact failure the
    `kind:` refusal was added to end. A misspelt `materialisation:` configured
    nothing and said nothing.

    **A row declaring a `kind:` is not this door's to check**, and is refused
    before reaching here anyway: the keys such a row carries belong to
    whichever door binds that backing, and a closed set here would refuse the
    `source:` and `parent_key:` a registry reads. The two halves of the check
    meet at the `kind:` discriminator.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entity_types": [{"id": "Breed"}],
                "entities": [{"id": "dog", "type": "Breed", "name": "Dog"}],
                "taxonomies": [
                    {"id": "kinds", "relation": "isa", "materialisation": {"structure": "copied"}}
                ],
            }
        )

    message = str(excinfo.value)
    assert "'kinds'" in message
    assert "materialisation" in message
    assert "materialization" in message, "the message lists the keys this door does read"
    assert excinfo.value.context["taxonomy"] == "kinds"


@DOORS
def test_a_duplicate_source_id_is_refused(door: Door) -> None:
    """Source ids are the closed set a qualified id is parsed against."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "sources": [
                    {"id": "inline_a", "kind": "inline"},
                    {"id": "inline_a", "kind": "inline"},
                ],
            }
        )

    assert "'inline_a'" in str(excinfo.value)


def test_the_live_source_refusal_does_not_depend_on_what_is_imported() -> None:
    """One config means one thing in every environment.

    The refusal is about *ownership*: a module-level loader has no ``close()``
    whatever is installed. So importing the package that could bind such a
    source must not change the answer -- if it did, the same file would load
    in production and fail in a test, or the reverse.
    """
    document = {"id": "x", "sources": [{"id": "clinic_db", "kind": "record"}]}

    with pytest.raises(ValidationError) as before:
        load_ontology(document)

    import dataknobs_data  # noqa: F401  -- imported for its side effects on the registry

    with pytest.raises(ValidationError) as after:
        load_ontology(document)

    assert str(before.value) == str(after.value)


def test_the_core_refuses_grammar_but_not_ownership() -> None:
    """``build_ontology`` validates; owning a source is the door's question.

    The split matters: a registry *can* own a live source, and it reaches the
    same core. A core that refused live kinds would make that impossible.
    """
    config = OntologyConfig.from_dict(
        {"id": "x", "sources": [{"id": "clinic_db", "kind": "record"}]}
    )

    parts = build_ontology(config)

    assert parts.source_specs[0]["kind"] == "record"


# --------------------------------------------------------------------------
# One id, one thing -- in every section that keys by id
# --------------------------------------------------------------------------
#
# `entities:` was the only section that refused a duplicate. The check was
# written inline there and never extracted, so the five sibling sections that
# key by id inherited nothing and dropped a colliding row in silence. These
# tests pin the whole class, not the one member that happened to be written.


@DOORS
def test_a_duplicate_entity_type_id_is_refused(door: Door) -> None:
    """The same collision as `entities:`, one section over."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entity_types": [
                    {"id": "Species", "name": "Species"},
                    {"id": "Species", "name": "Taxon"},
                ],
            }
        )

    assert "'Species'" in str(excinfo.value)
    assert excinfo.value.context["section"] == "entity_types"


@DOORS
def test_a_duplicate_relation_type_id_is_refused(door: Door) -> None:
    """A relation declared twice would silently keep the second one's inverse."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "relation_types": [
                    {"id": "isa", "symmetric": False},
                    {"id": "isa", "symmetric": True},
                ],
            }
        )

    assert "'isa'" in str(excinfo.value)
    assert excinfo.value.context["section"] == "relation_types"


@DOORS
def test_a_duplicate_taxonomy_id_is_refused(door: Door) -> None:
    """Two taxonomies, one id: the second's relation would win unannounced."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "taxonomies": [
                    {"id": "tree", "relation": "isa"},
                    {"id": "tree", "relation": "part_of"},
                ],
            }
        )

    assert "'tree'" in str(excinfo.value)
    assert excinfo.value.context["section"] == "taxonomies"


@DOORS
def test_a_duplicate_assertion_id_is_refused(door: Door) -> None:
    """Two identical edges mint one id, and the index would disagree with itself.

    ``by_id`` keeps the last and ``by_subject`` keeps both, so a duplicate does
    not merely lose a row -- it makes two lookups over the same store answer
    differently about how many there are.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "assertions": [
                    {"subject": "dog", "relation": "isa", "object": "mammal"},
                    {"subject": "dog", "relation": "isa", "object": "mammal"},
                ],
            }
        )

    assert "'dog-isa-mammal'" in str(excinfo.value)
    assert excinfo.value.context["section"] == "assertions"


@DOORS
def test_two_tree_nodes_that_mint_the_same_id_are_refused(door: Door) -> None:
    """A minted id is a slug of the path, and two paths can slug the same.

    Siblings named ``Late Fees`` and ``late-fees`` are different nodes to the
    person editing the file and one node to the slug, so the second silently
    replaced the first.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "sources": [
                    {
                        "id": "areas",
                        "kind": "nested",
                        "tree": {
                            "name": "Billing",
                            "children": [{"name": "Late Fees"}, {"name": "late-fees"}],
                        },
                    }
                ],
            }
        )

    assert "'billing/late-fees'" in str(excinfo.value)


# --------------------------------------------------------------------------
# A malformed enum is a refusal, not a stray ValueError
# --------------------------------------------------------------------------
#
# `ValidationError` does not descend from `ValueError`, so a caller holding
# the documented contract -- `except ValidationError` around either door --
# did not catch a bad `inference:` at all.


@DOORS
def test_an_unknown_inference_mode_is_refused(door: Door) -> None:
    """Named, so the message says which value and which of the two words to use."""
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "relation_types": [{"id": "isa", "inference": "sometimes"}]})

    message = str(excinfo.value)
    assert "'sometimes'" in message
    assert "on_demand" in message
    assert excinfo.value.context["field"] == "inference"


@DOORS
def test_an_unknown_materialization_mode_is_refused(door: Door) -> None:
    """The same coercion, reached through a taxonomy rather than a relation."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "taxonomies": [
                    {
                        "id": "tree",
                        "relation": "isa",
                        "materialization": {"structure": "sometimes"},
                    }
                ],
            }
        )

    assert "'sometimes'" in str(excinfo.value)
    assert excinfo.value.context["field"] == "materialization.structure"


# --------------------------------------------------------------------------
# A reference is normalized the way the declaration was
# --------------------------------------------------------------------------


@DOORS
def test_a_numeric_entity_type_id_can_still_be_referenced(door: Door) -> None:
    """Unquoted digits are an int in YAML, and the declaration is str-keyed.

    The refusal compared the raw reference against str-keyed declarations, so
    ``isa: 100`` missed ``id: 100`` and a valid document was refused as
    undeclared. Not refusing it is the assertion.
    """
    door(
        {
            "id": "x",
            "entity_types": [{"id": 100, "name": "Root"}, {"id": 200, "isa": 100}],
        }
    )


@DOORS
def test_a_numeric_entity_id_can_still_be_named_by_a_tree(door: Door) -> None:
    """The same mismatch on the declared-tree path."""
    door(
        {
            "id": "x",
            "entities": [{"id": 1, "type": "Node"}, {"id": 2, "type": "Node"}],
            "sources": [
                {
                    "id": "areas",
                    "kind": "nested",
                    "tree": {"id": 1, "name": "Root", "children": [{"id": 2, "name": "Leaf"}]},
                }
            ],
        }
    )


# --------------------------------------------------------------------------
#
# The slug collision, which had no test at either door. It fires from shared
# minting code that `MappingHierarchy.from_nested` calls too -- so the pair
# below is what keeps the two doors' vocabularies one vocabulary, and the
# cross-source case is what kept a message honest that used to name nothing.


@DOORS
def test_two_paths_in_one_tree_that_slug_to_one_id_are_refused(door: Door) -> None:
    """The slug collapses punctuation and case, so a reader sees two nodes."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "sources": [
                    {
                        "id": "areas",
                        "kind": "nested",
                        "tree": {
                            "name": "Billing",
                            "children": [{"name": "Late Fees"}, {"name": "late-fees"}],
                        },
                    }
                ],
            }
        )

    assert "'billing/late-fees'" in str(excinfo.value)
    assert "'Billing/Late Fees'" in str(excinfo.value)
    assert excinfo.value.context["source_id"] == "areas"


@DOORS
def test_a_collision_between_two_trees_names_the_tree_that_got_there_first(
    door: Door,
) -> None:
    """The message named the id back at you, which answered nothing.

    Two nested sources mint into one entity store, and the collision check has
    always spanned them -- but the map from id to the path that minted it was
    rebuilt per source, so a *cross-source* collision found nothing in it and
    fell back to printing the id a second time: ``mints id 'billing', which
    'billing' already minted``. The one thing the reader needed, which of the
    other trees to go and look at, was the one thing absent.

    Now that both doors mint through one core, the map is threaded across
    sources and the message names the colliding node. Pinned here because it is
    the half of the refusal no test reached, which is how it stayed wrong.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "sources": [
                    {"id": "left", "kind": "nested", "tree": {"name": "Billing"}},
                    {"id": "right", "kind": "nested", "tree": {"name": "billing!"}},
                ],
            }
        )

    message = str(excinfo.value)
    assert "which 'Billing' already minted" in message
    assert "which 'billing' already minted" not in message
    assert excinfo.value.context["source_id"] == "right"


# --------------------------------------------------------------------------
# A missing required field is a refusal, not a stray KeyError
# --------------------------------------------------------------------------
#
# The same defect as the stray `ValueError` above, and it fails the contract
# the same way: `KeyError` does not descend from `ValidationError`, so a
# caller holding the documented `Raises:` -- which lists ValidationError,
# ConfigLoadError and OSError -- caught nothing at all for the commonest
# authoring mistake there is. Every section had it, because every builder
# indexed its required keys directly instead of reading them through one
# place that could refuse.

#: One entry per required field the loader reads, and the document that omits
#: it. Listed exhaustively rather than sampled: the reason there were ten of
#: these is that each was written on its own, so a table is the only form that
#: fails when the eleventh is added the old way.
_MISSING_FIELDS = [
    ("entity_types", "id", {"entity_types": [{"name": "Species"}]}),
    (
        "entity_types.attributes",
        "name",
        {"entity_types": [{"id": "Species", "attributes": [{"type": "string"}]}]},
    ),
    ("relation_types", "id", {"relation_types": [{"transitive": True}]}),
    ("entities", "id", {"entities": [{"type": "Species", "name": "Dog"}]}),
    ("entities", "type", {"entities": [{"id": "dog", "name": "Dog"}]}),
    ("assertions", "subject", {"assertions": [{"relation": "isa", "object": "mammal"}]}),
    ("assertions", "relation", {"assertions": [{"subject": "dog", "object": "mammal"}]}),
    ("assertions", "object", {"assertions": [{"subject": "dog", "relation": "isa"}]}),
    ("taxonomies", "id", {"taxonomies": [{"relation": "isa"}]}),
    ("taxonomies", "relation", {"taxonomies": [{"id": "species"}]}),
]


@DOORS
@pytest.mark.parametrize(
    ("section", "missing", "body"),
    _MISSING_FIELDS,
    ids=[f"{section}.{missing}" for section, missing, _ in _MISSING_FIELDS],
)
def test_a_missing_required_field_is_refused_naming_the_section(
    door: Door, section: str, missing: str, body: Mapping[str, Any]
) -> None:
    """Named field, named section, and catchable by the documented contract."""
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", **body})

    message = str(excinfo.value)
    assert repr(missing) in message
    assert f"`{section}:`" in message
    assert excinfo.value.context["section"] == section
    assert excinfo.value.context["field"] == missing


@DOORS
def test_the_refusal_names_the_row_by_its_id_where_it_has_one(door: Door) -> None:
    """A fifty-entity file needs to say *which* row, not only which field.

    The id is the handle a document author has, so it is the one the message
    uses -- ``KeyError: 'type'`` sent them to bisect the file instead.
    """
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "entities": [{"id": "beagle", "name": "Beagle"}]})

    assert "'beagle'" in str(excinfo.value)
    assert excinfo.value.context["id"] == "beagle"


@DOORS
def test_the_refusal_falls_back_to_the_keys_a_row_does_have(door: Door) -> None:
    """The case with no handle: the missing field *is* the id.

    Listing what the row does carry is what makes it findable anyway, and it
    is the only identifying thing left. An assertion row reaches this branch
    too, since assertions mint their ids rather than declaring them.
    """
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "entities": [{"type": "Species", "name": "Beagle"}]})

    message = str(excinfo.value)
    assert "type" in message
    assert "name" in message


# --------------------------------------------------------------------------
# An attribute row is read for its keys, and each key for its shape
# --------------------------------------------------------------------------
#
# `required: "no"` loaded as a required attribute, because the key was read
# with `bool(...)` and a non-empty string is truthy. It was not alone in the
# row: `name` and `description` went through `str(...)`, so `null` became the
# string "None"; `enum_values` was tested for truthiness, so an empty list and
# an absent key read the same; and a key nothing reads -- `enum:` for
# `enum_values:` -- loaded and was discarded. The published `field_type:` key
# was discarded too, which is the one case below that the loader now reads
# rather than refuses.


def _one_attribute(**attribute: Any) -> dict[str, Any]:
    """A document whose one entity type declares the one attribute given."""
    return {"id": "x", "entity_types": [{"id": "Species", "attributes": [attribute]}]}


def _sync_loaded(document: Mapping[str, Any]) -> Ontology:
    return load_ontology(document)


def _async_loaded(document: Mapping[str, Any]) -> AsyncOntology:
    return asyncio.run(async_load_ontology(document))


LOADERS = pytest.mark.parametrize(
    "loaded", [_sync_loaded, _async_loaded], ids=["load_ontology", "async_load_ontology"]
)

Loaded = Callable[[Mapping[str, Any]], "Ontology | AsyncOntology"]


def _the_attribute(loaded: Loaded, **attribute: Any) -> AttributeDef:
    (built,) = loaded(_one_attribute(name="a", **attribute)).entity_types["Species"].attributes
    return built


#: Every attribute a row can be refused for, and the key the refusal is about.
#: Listed as a table so :func:`test_every_attribute_refusal_is_one_exception_type`
#: can hold the claim that each is one type, and so a new refusal has one
#: place to be added.
_REFUSED_ATTRIBUTES: list[tuple[str, str, dict[str, Any]]] = [
    ("required-no", "required", {"name": "a", "required": "no"}),
    ("required-false-string", "required", {"name": "a", "required": "false"}),
    ("required-one", "required", {"name": "a", "required": 1}),
    ("required-zero", "required", {"name": "a", "required": 0}),
    ("name-int", "name", {"name": 5}),
    ("name-null", "name", {"name": None}),
    ("name-empty", "name", {"name": ""}),
    ("description-int", "description", {"name": "a", "description": 5}),
    ("enum-values-empty", "enum_values", {"name": "a", "enum_values": []}),
    ("enum-values-string", "enum_values", {"name": "a", "enum_values": "CS"}),
    ("enum-values-ints", "enum_values", {"name": "a", "enum_values": [1, 2]}),
    ("enum-values-repeated", "enum_values", {"name": "a", "enum_values": ["a", "a"]}),
    ("type-int", "type", {"name": "a", "type": 5}),
    ("field-type-unknown", "field_type", {"name": "a", "field_type": "money"}),
    ("field-type-int", "field_type", {"name": "a", "field_type": 5}),
    (
        "field-type-contradicts-type",
        "field_type",
        {"name": "a", "type": "integer", "field_type": "string"},
    ),
    ("entity-type-int", "entity_type", {"name": "a", "type": "entity", "entity_type": 5}),
    ("entity-type-list", "entity_type", {"name": "a", "type": "entity", "entity_type": ["S"]}),
    ("entity-type-empty", "entity_type", {"name": "a", "type": "entity", "entity_type": ""}),
]

#: One accepted value for every key the row is read for, so the parity guard
#: below can hold that each key has a reading as well as a refusal.
_ACCEPTED_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "name": {"name": "a"},
    "type": {"name": "a", "type": "number"},
    "field_type": {"name": "a", "type": "number", "field_type": "float"},
    "entity_type": {"name": "a", "type": "entity", "entity_type": "Species"},
    "required": {"name": "a", "required": True},
    "enum_values": {"name": "a", "enum_values": ["x", "y"]},
    "description": {"name": "a", "description": "Latin name"},
}


@DOORS
@pytest.mark.parametrize(
    ("key", "attribute"),
    [(key, attribute) for _, key, attribute in _REFUSED_ATTRIBUTES],
    ids=[case for case, _, _ in _REFUSED_ATTRIBUTES],
)
def test_an_attribute_value_of_the_wrong_shape_is_refused(
    door: Door, key: str, attribute: dict[str, Any]
) -> None:
    """Named key, named entity type, and the documented exception type.

    ``required: "no"`` is the case the register row is about: it loaded as a
    *required* attribute. The rest are the same function's other readers,
    each of which coerced rather than checked -- ``name: null`` became the
    attribute ``'None'``, and ``enum_values: "CS"`` became ``['C', 'S']``.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(_one_attribute(**attribute))

    message = str(excinfo.value)
    assert repr(key) in message
    assert "'Species'" in message
    assert excinfo.value.context["section"] == "entity_types.attributes"
    assert excinfo.value.context["entity_type"] == "Species"
    assert excinfo.value.context["field"] == key


@DOORS
def test_a_non_boolean_required_says_it_is_on_or_off(door: Door) -> None:
    """The message a ``"no"`` author needs is the one ``index.aliases`` gives."""
    with pytest.raises(ValidationError) as excinfo:
        door(_one_attribute(name="latin_name", required="no"))

    message = str(excinfo.value)
    assert "'latin_name'" in message
    assert "`true` or `false`" in message


@LOADERS
@pytest.mark.parametrize("attribute", [{}, {"required": None}], ids=["absent", "null"])
def test_an_absent_or_null_required_reads_false(loaded: Loaded, attribute: dict[str, Any]) -> None:
    """``null`` is silence, as it is for ``index.aliases`` and in ``data``'s reader."""
    assert _the_attribute(loaded, **attribute).required is False


@LOADERS
def test_a_boolean_required_is_read_as_written(loaded: Loaded) -> None:
    assert _the_attribute(loaded, required=True).required is True
    assert _the_attribute(loaded, required=False).required is False


@LOADERS
@pytest.mark.parametrize(
    ("attribute", "expected"),
    [({}, ""), ({"description": None}, ""), ({"description": "Latin name"}, "Latin name")],
    ids=["absent", "null", "string"],
)
def test_a_description_is_a_string_and_null_is_silence(
    loaded: Loaded, attribute: dict[str, Any], expected: str
) -> None:
    """``null`` used to reach an extraction prompt as the word ``None``."""
    assert _the_attribute(loaded, **attribute).description == expected


@LOADERS
@pytest.mark.parametrize("values", [["a", "b"], ("a", "b")], ids=["list", "tuple"])
def test_enum_values_read_as_a_list_of_strings(loaded: Loaded, values: Any) -> None:
    assert _the_attribute(loaded, enum_values=values).enum_values == ["a", "b"]


@LOADERS
def test_an_absent_enum_values_is_not_enumerated(loaded: Loaded) -> None:
    assert _the_attribute(loaded).enum_values is None


@DOORS
def test_an_attribute_key_this_loader_does_not_read_is_refused(door: Door) -> None:
    """``enum:`` for ``enum_values:`` loaded, and its values were discarded.

    A dropped key and an unsupported key have to look different -- the rule
    :data:`~dataknobs_common.ontology.TAXONOMY_ROW_KEYS` enforces for an axis
    row, applied to the row next door. The message lists what *is* read, so
    the misspelling is findable from it.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(_one_attribute(name="severity", enum=["a", "b"]))

    message = str(excinfo.value)
    assert "['enum']" in message
    assert "'severity'" in message
    assert str(sorted(loader_module.ATTRIBUTE_ROW_KEYS)) in message
    assert excinfo.value.context["keys"] == ["enum"]


@LOADERS
def test_a_type_is_kept_verbatim_and_its_vocabulary_is_open(loaded: Loaded) -> None:
    """``type:`` names a vocabulary type, and the loader does not decide which exist.

    ``entity``, ``enum`` and ``number`` are vocabulary types with no record
    counterpart, and the published documents write all three. So an unknown
    type loads with no ``field_type``; only a non-string one is refused.
    """
    built = _the_attribute(loaded, type="nonsense")
    assert built.value_type == "nonsense"
    assert built.field_type is None

    assert _the_attribute(loaded).value_type == "string"


@LOADERS
def test_a_type_derives_its_field_type_whatever_its_case(loaded: Loaded) -> None:
    """``type: String`` is a spelling of ``string``; ``value_type`` keeps the author's."""
    built = _the_attribute(loaded, type="String")
    assert built.field_type is FieldType.STRING
    assert built.value_type == "String"


@LOADERS
@pytest.mark.parametrize("spelling", ["float", "Float"])
def test_a_field_type_is_read_and_overrides_the_derivation(loaded: Loaded, spelling: str) -> None:
    """The published key the loader discarded.

    ``{type: number, field_type: float}`` is written in four guides, the
    shared test vocabulary and the proving ground, and loaded as
    ``field_type=None`` -- ``number`` has no record type, and the key that
    said which one was never read.
    """
    built = _the_attribute(loaded, type="number", field_type=spelling)
    assert built.field_type is FieldType.FLOAT
    assert built.value_type == "number"


def test_a_refused_field_type_lists_the_values_it_could_have_been() -> None:
    with pytest.raises(ValidationError) as excinfo:
        load_ontology(_one_attribute(name="price", field_type="money"))

    message = str(excinfo.value)
    for member in FieldType:
        assert repr(member.value) in message


def test_the_guides_vocabulary_reads_its_field_type(mammals_v11_path: Path) -> None:
    """The documents on disk, read through the reader a consumer holds.

    ``lifespan_years`` is written ``{type: number, field_type: float}``, the
    one spelling every published example of the key uses.
    """
    ontology = load_ontology(mammals_v11_path)
    lifespan = {a.name: a for a in ontology.inherited_attributes("Breed")}["lifespan_years"]

    assert lifespan.field_type is FieldType.FLOAT


@DOORS
@pytest.mark.parametrize(
    "entity_type",
    [
        {"id": "Species", "attributes": ["latin_name"]},
        {"id": "Species", "attributes": {"latin_name": {"type": "string"}}},
    ],
    ids=["attribute-not-a-mapping", "attributes-a-mapping"],
)
def test_an_attributes_list_of_the_wrong_shape_is_refused(
    door: Door, entity_type: dict[str, Any]
) -> None:
    """Both raised a bare ``AttributeError``, which no documented ``Raises:`` names."""
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "entity_types": [entity_type]})

    assert "'Species'" in str(excinfo.value)
    assert excinfo.value.context["section"] == "entity_types.attributes"


@DOORS
def test_two_attributes_with_one_name_on_one_type_are_refused(door: Door) -> None:
    """One name, one declaration: the file's duplicate-id rule, on the list it skipped."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entity_types": [
                    {
                        "id": "Species",
                        "attributes": [
                            {"name": "a", "type": "string"},
                            {"name": "a", "type": "number"},
                        ],
                    }
                ],
            }
        )

    assert "'a'" in str(excinfo.value)
    assert "'Species'" in str(excinfo.value)


@DOORS
def test_a_subtype_may_redeclare_its_parent_s_attribute(door: Door) -> None:
    """The duplicate rule is per row. Across ``isa`` it is inheritance, a different question."""
    door(
        {
            "id": "x",
            "entity_types": [
                {"id": "Species", "attributes": [{"name": "a"}]},
                {"id": "Breed", "isa": "Species", "attributes": [{"name": "a"}]},
            ],
        }
    )


def test_every_attribute_key_has_a_reading_and_a_refusal() -> None:
    """Both tables cover every key the row is read for, and each refusal is one type.

    A key added to :data:`ATTRIBUTE_ROW_KEYS` with no refused case is a key
    whose shape nothing checks -- which is how the seven readers above came
    to coerce: each was written on its own. One with no accepted case is a
    key no document could write. ``entity_type`` has no exemption: the
    reference check compares ``str(value)`` and stands down under
    ``imports:``, so it cannot stand in for a shape check.
    """
    refused = {key for _, key, _ in _REFUSED_ATTRIBUTES}
    assert refused == loader_module.ATTRIBUTE_ROW_KEYS
    assert set(_ACCEPTED_ATTRIBUTES) == loader_module.ATTRIBUTE_ROW_KEYS

    for _, _, attribute in _REFUSED_ATTRIBUTES:
        with pytest.raises(ValidationError):
            load_ontology(_one_attribute(**attribute))
    for attribute in _ACCEPTED_ATTRIBUTES.values():
        load_ontology(_one_attribute(**attribute))


@LOADERS
def test_an_absent_type_reads_as_a_string_attribute_all_the_way_down(loaded: Loaded) -> None:
    """``value_type`` said ``string`` and ``field_type`` said nothing.

    ``type: string`` derives ``STRING``, so an attribute that leaves ``type:``
    out -- which reads as ``string`` -- has to derive it too, or the default
    is a different attribute from the one it stands for.
    """
    built = _the_attribute(loaded)
    assert built.value_type == "string"
    assert built.field_type is FieldType.STRING


@LOADERS
@pytest.mark.parametrize(
    ("declared", "explicit"),
    [("float", "float"), ("Float", "FLOAT"), ("number", "float"), ("entity", "string")],
    ids=["agree", "agree-any-case", "number", "entity"],
)
def test_a_field_type_the_type_does_not_contradict_is_read(
    loaded: Loaded, declared: str, explicit: str
) -> None:
    """The override is for a type with no record counterpart, or one that agrees."""
    built = _the_attribute(loaded, type=declared, field_type=explicit)
    assert built.field_type is FieldType(explicit.lower())


@DOORS
def test_a_field_type_contradicting_its_type_names_both(door: Door) -> None:
    """``{type: integer, field_type: string}`` loaded as an integer stored as a string.

    Two keys that each name a record type and disagree is the contradiction
    ``enum_values: []`` is refused for: no reading of the row honours both.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(_one_attribute(name="count", type="integer", field_type="string"))

    message = str(excinfo.value)
    assert "'integer'" in message
    assert "'string'" in message


@LOADERS
@pytest.mark.parametrize("member", list(FieldType), ids=[m.value for m in FieldType])
def test_a_field_type_given_as_a_member_is_read(loaded: Loaded, member: FieldType) -> None:
    """A document built in Python may hand the member itself.

    ``load_ontology`` takes a plain mapping, and ``dataknobs-data``'s schema
    reader already takes a member. This one refused it as *not a record
    field type* while listing the very value it was.
    """
    assert _the_attribute(loaded, type="number", field_type=member).field_type is member


@LOADERS
def test_a_type_given_as_a_member_is_refused_as_not_a_string(loaded: Loaded) -> None:
    """``type:`` is the vocabulary word and stays a string; ``field_type:`` takes the member."""
    with pytest.raises(ValidationError, match="'type'"):
        _the_attribute(loaded, type=FieldType.FLOAT)


@LOADERS
def test_null_attributes_read_as_none(loaded: Loaded) -> None:
    """``attributes: null`` raised a bare ``TypeError`` from the reference check.

    ``null`` is silence everywhere else in this loader, and here it is the
    shape a YAML key with no value takes.
    """
    document = {"id": "x", "entity_types": [{"id": "Species", "attributes": None}]}
    assert loaded(document).entity_types["Species"].attributes == []


@DOORS
def test_an_entity_type_numbered_like_a_type_id_is_still_a_string(door: Door) -> None:
    """The reference check compares ``str(value)``, and type ids are ``str()``'d.

    So ``id: 2024`` beside ``entity_type: 2024`` passed the check and loaded
    an ``int`` into a field typed ``str | None``, where every
    ``attribute.entity_type == type.id`` comparison is false.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entity_types": [
                    {"id": 2024},
                    {
                        "id": "Species",
                        "attributes": [{"name": "a", "type": "entity", "entity_type": 2024}],
                    },
                ],
            }
        )

    assert excinfo.value.context["field"] == "entity_type"


@DOORS
def test_an_entity_type_of_the_wrong_shape_is_refused_under_imports(door: Door) -> None:
    """``imports:`` stands the reference check down, so it cannot be the shape check.

    ``entity_type: [L]`` loaded as a list in a document that imports.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "imports": ["mammals"],
                "entity_types": [
                    {
                        "id": "Species",
                        "attributes": [{"name": "a", "type": "entity", "entity_type": ["L"]}],
                    }
                ],
            }
        )

    assert excinfo.value.context["field"] == "entity_type"


@DOORS
def test_unread_attribute_keys_of_mixed_types_are_refused_by_name(door: Door) -> None:
    """Sorting ``{1, 'foo'}`` raised a bare ``TypeError`` before the refusal was built."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entity_types": [
                    {"id": "Species", "attributes": [{"name": "a", 1: "x", "foo": "y"}]}
                ],
            }
        )

    assert excinfo.value.context["keys"] == ["1", "foo"]


@DOORS
def test_unread_taxonomy_keys_of_mixed_types_are_refused_by_name(door: Door) -> None:
    """The same sort, on the axis row's key set next door."""
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "taxonomies": [{"id": "t", "relation": "r", 1: "x", "foo": "y"}]})

    assert excinfo.value.context["keys"] == ["1", "foo"]


# --------------------------------------------------------------------------
# A key only a later version reads is refused, not dropped
# --------------------------------------------------------------------------
#
# These four used to load and be discarded, which from the author's chair is
# indistinguishable from being honoured: a relation type declaring
# `cardinality: many_to_many` reported no error and constrained nothing.
# Refusing is what makes the deferral legible -- and it is the discipline the
# refusals above already apply, which is to name the offending thing and say
# where what you asked for lives.

#: One entry per key phase 2 brings, and a document that carries it.
#:
#: Listed exhaustively rather than sampled, for the reason the missing-field
#: table is: the fifth key must fail here if it is added the old way.
_PHASE_2_KEYS = [
    (
        "assertions",
        "condition",
        {
            "assertions": [
                {
                    "subject": "dog",
                    "relation": "isa",
                    "object": "mammal",
                    "condition": {"kind": "temporal", "valid_from": "2024-01-01"},
                }
            ]
        },
    ),
    (
        "relation_types",
        "cardinality",
        {"relation_types": [{"id": "sold_by", "cardinality": "many_to_many"}]},
    ),
    (
        "relation_types",
        "condition",
        {"relation_types": [{"id": "sold_by", "condition": {"kind": "temporal"}}]},
    ),
    (
        "relation_types",
        "constraints",
        {"relation_types": [{"id": "isa", "constraints": [{"kind": "acyclic"}]}]},
    ),
]


@DOORS
@pytest.mark.parametrize(
    ("section", "key", "body"),
    _PHASE_2_KEYS,
    ids=[f"{section}.{key}" for section, key, _ in _PHASE_2_KEYS],
)
def test_a_phase_2_key_is_refused_naming_the_key_and_the_version(
    door: Door, section: str, key: str, body: Mapping[str, Any]
) -> None:
    """Named key, named version, from both doors.

    The version half is what separates this from an "unsupported key" error:
    the author asked a specific question, and the useful answer says which
    version answers it rather than that this one does not.
    """
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", **body})

    message = str(excinfo.value)
    assert repr(key) in message
    assert "phase 2" in message
    assert excinfo.value.context["section"] == section
    assert excinfo.value.context["field"] == key
    assert excinfo.value.context["version"] == "phase 2"


def test_the_refusal_cases_are_the_loader_table() -> None:
    """The cases above are the loader's table, not a copy free to drift from it.

    The comment on the list claims the fifth key must fail here if it is added
    the old way, and nothing made that true: a key added to the loader's table
    and not to this one was exercised by nothing at all, so the claim held
    only as long as one author remembered both halves.

    It catches the other direction too, which is the more useful half. Only
    ``relation_types`` and ``assertions`` call the refusal; a *section* added
    to the loader's table -- ``entity_types``, say -- gets a case here to
    satisfy this, and that case then fails, because no builder there refuses
    anything. An entry that names a key nothing checks surfaces as a red test
    rather than as a key that quietly loads and is discarded.
    """
    declared = {
        (section, key) for section, keys in loader_module._PHASE_2_KEYS.items() for key in keys
    }
    exercised = {(section, key) for section, key, _ in _PHASE_2_KEYS}

    assert exercised == declared


@DOORS
def test_the_phase_2_refusal_names_the_row_it_is_about(door: Door) -> None:
    """A fifty-relation file needs to say which row, the way a missing field does.

    The same handle either refusal uses, because a document author meets one
    way of being pointed at a line rather than one per section.
    """
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "relation_types": [{"id": "sold_by", "cardinality": "one_to_many"}]})

    assert "'sold_by'" in str(excinfo.value)
    assert excinfo.value.context["id"] == "sold_by"


@DOORS
def test_an_unknown_polarity_is_refused(door: Door) -> None:
    """The coercion `_inference_mode` already had, on the enum a document gained.

    A stray ``ValueError`` from the enum would escape a caller holding either
    door's documented ``Raises:``, which is the defect one shared reader for
    every authored enum exists to stop repeating.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "assertions": [
                    {
                        "subject": "whale",
                        "relation": "isa",
                        "object": "fish",
                        "polarity": "maybe",
                    }
                ],
            }
        )

    message = str(excinfo.value)
    assert "'maybe'" in message
    assert "negated" in message
    assert excinfo.value.context["field"] == "polarity"


# --------------------------------------------------------------------------
# References into a section this document owns in full
# --------------------------------------------------------------------------
#
# Eight references, one refusal. The `isa` case above is the eighth and was
# already here: it is the one the loader refused before the rule was general,
# and it stays where it is as the regression guard for the conversion.


#: The seven references that resolved against nothing until this rule.
#:
#: Each row is a document constructed dangling, the value that dangles, and
#: the field it dangles from. Table-driven because the claim is uniform -- a
#: reference into a section this document declares must resolve -- and a
#: per-case suite would let one of the seven drift into a different shape.
_DANGLING = [
    (
        "entity_type",
        "Persen",
        {
            "id": "x",
            "entity_types": [
                {
                    "id": "Dog",
                    "attributes": [{"name": "owner", "type": "entity", "entity_type": "Persen"}],
                }
            ],
        },
    ),
    (
        "domain",
        "Persen",
        {
            "id": "x",
            "entity_types": [{"id": "Dog"}],
            "relation_types": [{"id": "owns", "domain": ["Persen"]}],
        },
    ),
    (
        "range",
        "Bicicle",
        {
            "id": "x",
            "entity_types": [{"id": "Dog"}],
            "relation_types": [{"id": "owns", "range": ["Bicicle"]}],
        },
    ),
    (
        "inverse_of",
        "owned_bye",
        {
            "id": "x",
            "relation_types": [{"id": "owns", "inverse_of": "owned_bye"}],
        },
    ),
    (
        "type",
        "Dogg",
        {
            "id": "x",
            "entity_types": [{"id": "Dog"}],
            "entities": [{"id": "beagle", "type": "Dogg"}],
        },
    ),
    (
        "relation",
        "isaa",
        {
            "id": "x",
            "relation_types": [{"id": "isa"}],
            "assertions": [{"subject": "dog", "relation": "isaa", "object": "mammal"}],
        },
    ),
    (
        "relation",
        "isaa",
        {
            "id": "x",
            "relation_types": [{"id": "isa"}],
            "taxonomies": [{"id": "kinds", "relation": "isaa"}],
        },
    ),
]

DANGLING = pytest.mark.parametrize(
    ("field", "dangles", "document"),
    _DANGLING,
    ids=[
        "attribute_entity_type",
        "relation_domain",
        "relation_range",
        "relation_inverse_of",
        "entity_type",
        "assertion_relation",
        "taxonomy_relation",
    ],
)


@DOORS
@DANGLING
def test_a_reference_into_a_section_this_document_declares_must_resolve(
    door: Door, field: str, dangles: str, document: Mapping[str, Any]
) -> None:
    """Seven references that used to resolve against nothing at all.

    Every one of these documents loaded before this rule, and every one of them
    is a typo whose consequence is silence: a relation type whose domain names
    no type constrains nothing, an entity whose type names none is untyped, an
    axis whose relation names none walks an empty graph. The document said
    something and the vocabulary meant nothing.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(document)

    message = str(excinfo.value)
    assert repr(dangles) in message
    assert field in message
    assert excinfo.value.context["value"] == dangles
    assert excinfo.value.context["field"] == field


@DOORS
def test_a_subject_and_an_object_naming_nothing_declared_still_load(door: Door) -> None:
    """Layer 1 asks whether a name resolves; layer 2 whether a statement is true.

    An assertion's ``subject:`` and ``object:`` are layer 2 -- they are claims
    about a graph that a live source may supply -- so a document may assert an
    edge between two ids it does not itself declare. Only the ``relation:`` is
    a reference into a section this document owns in full.
    """
    door(
        {
            "id": "x",
            "relation_types": [{"id": "isa"}],
            "assertions": [{"subject": "ghost", "relation": "isa", "object": "phantom"}],
        }
    )


@DOORS
def test_an_assertion_relation_may_name_a_declared_attribute(door: Door) -> None:
    """The positive control, and the half of this rule that can rot.

    An attribute-valued assertion names an attribute, not a relation type, so
    the set a ``relation:`` resolves against is ``relation_types:`` *union*
    the declared attribute names. A check that looked at ``relation_types:``
    alone would refuse this document -- and a suite that only asserted the
    seven refusals above would pass against a loader that had started refusing
    everything.

    ``relation_types:`` is declared and holds something else, which is what
    puts the check in play at all: the section decides *whether* a
    ``relation:`` is checked, and the union decides what resolves. Without it
    this document is unchecked, and the control would pass against a loader
    that had stopped reading attributes entirely.
    """
    door(
        {
            "id": "x",
            "relation_types": [{"id": "isa"}],
            "entity_types": [
                {"id": "Species", "attributes": [{"name": "lifespan_years", "type": "integer"}]}
            ],
            "assertions": [
                {
                    "subject": "dog",
                    "relation": "lifespan_years",
                    "object": {"value": 12, "type": "integer"},
                }
            ],
        }
    )


@DOORS
def test_an_assertion_relation_naming_neither_half_of_the_union_is_refused(door: Door) -> None:
    """The control's difference, constructed rather than assumed.

    The same document with a *different* attribute declared: both halves of
    the union are present either way, so the guard is not what decides it, and
    the only thing that changed is whether the name is in the union.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "relation_types": [{"id": "isa"}],
                "entity_types": [
                    {"id": "Species", "attributes": [{"name": "weight_kg", "type": "float"}]}
                ],
                "assertions": [
                    {
                        "subject": "dog",
                        "relation": "lifespan_years",
                        "object": {"value": 12, "type": "integer"},
                    }
                ],
            }
        )

    assert "'lifespan_years'" in str(excinfo.value)


@DOORS
def test_a_reference_is_not_checked_against_a_section_the_document_omits(door: Door) -> None:
    """An empty section is NO schema rather than an empty one.

    The regime ``build_ontology`` already runs for tree nodes, generalised:
    a document declaring no ``entity_types:`` at all is not making claims about
    a type vocabulary, so an ``entities:`` row naming a type cannot be wrong
    about one. Refusing here would make the rule unusable for exactly the
    documents a live source completes.
    """
    door({"id": "x", "entities": [{"id": "beagle", "type": "Dog"}]})


# --------------------------------------------------------------------------
# What decides the guard, where the target is a union
# --------------------------------------------------------------------------
#
# Seven of the eight references point at one declared section, so "that
# section is empty" and "this document declares no vocabulary for this
# reference" are the same sentence. `relation:` resolves against
# `relation_types:` *union* the declared attribute names, and there the two
# sentences come apart: the union is non-empty the moment any entity type
# declares any attribute, in a document that declares no relation vocabulary
# at all. The section decides the guard; the union decides what resolves.


@DOORS
def test_an_attribute_elsewhere_does_not_make_relation_types_mandatory(door: Door) -> None:
    """A document with no ``relation_types:`` makes no claim about relations.

    The loader has never required an assertion's ``relation:`` to have a
    ``relation_types:`` row -- ``_build_assertions`` stores the string and the
    axis filters on it -- so a document that names its edges and declares no
    relation vocabulary is the ordinary authored shape, not an incomplete one.

    Measured against the union, this document's guard turns on an attribute
    that has nothing to do with the reference: ``latin_name`` makes the union
    non-empty, and ``isa`` is then refused for not being in it. The same
    document with the attribute deleted loads. A guard a reader cannot predict
    from the reference is worse than no guard.
    """
    door(
        {
            "id": "x",
            "entity_types": [
                {"id": "Species", "attributes": [{"name": "latin_name", "type": "string"}]}
            ],
            "entities": [{"id": "dog", "type": "Species"}, {"id": "mammal", "type": "Species"}],
            "assertions": [{"subject": "dog", "relation": "isa", "object": "mammal"}],
        }
    )


@DOORS
def test_the_same_document_without_the_attribute_loads_too(door: Door) -> None:
    """The other half of the pair, which is what makes the first one a claim.

    Green before this change and after it. A suite holding only the first test
    would pass against a loader that had stopped checking ``relation:``
    altogether; holding both pins that the attribute is not what decides.
    """
    door(
        {
            "id": "x",
            "entity_types": [{"id": "Species"}],
            "entities": [{"id": "dog", "type": "Species"}, {"id": "mammal", "type": "Species"}],
            "assertions": [{"subject": "dog", "relation": "isa", "object": "mammal"}],
        }
    )


@DOORS
def test_a_taxonomy_relation_is_not_checked_where_no_relation_types_are_declared(
    door: Door,
) -> None:
    """The axis half of the same rule, over the section that decides it."""
    door(
        {
            "id": "x",
            "entity_types": [
                {"id": "Species", "attributes": [{"name": "latin_name", "type": "string"}]}
            ],
            "taxonomies": [{"id": "lattice", "relation": "isa"}],
        }
    )


def test_a_kind_bearing_axis_declares_a_label_rather_than_a_reference() -> None:
    """A column axis's ``relation:`` names what its edges mean, not a section.

    Reached through the core rather than a door, as
    ``test_the_core_refuses_grammar_but_not_ownership`` is: a ``kind:`` on an
    axis row is refused by both module doors, so a registry is the only caller
    that gets this far, and the registry is where such a document is loaded.

    Nothing constructs an ``Assertion`` for a column row -- the edges are two
    columns -- so this ``relation:`` resolves against no set of assertions and
    is not a reference into ``relation_types:``. Checking it would refuse a
    document the registry guide publishes as valid.
    """
    config = OntologyConfig.from_dict(
        {
            "id": "catalog",
            "relation_types": [{"id": "part_of"}],
            "entity_types": [{"id": "Product", "attributes": [{"name": "sku", "type": "string"}]}],
            "taxonomies": [
                {
                    "id": "categories",
                    "kind": "column",
                    "source": "products",
                    "parent_key": "parent_sku",
                    "relation": "parent",
                }
            ],
        }
    )

    parts = build_ontology(config)

    assert parts.taxonomy_specs[0]["relation"] == "parent"


# --------------------------------------------------------------------------
# A bare string is one name, not its characters
# --------------------------------------------------------------------------


@DOORS
def test_a_scalar_domain_is_refused_naming_the_scalar(door: Door) -> None:
    """``domain: Person`` is the authoring mistake, and it has to be named.

    ``frozenset("Person")`` is six one-character type names, so the reference
    check downstream reported ``domain: 'P'`` against a declared list holding
    the exact word the author wrote. The sibling door states the rule for the
    same shape -- ``index.fields`` takes a list, and a bare string is one name
    spelled as its characters rather than a list of one.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entity_types": [{"id": "Person"}],
                "relation_types": [{"id": "owns", "domain": "Person"}],
            }
        )

    message = str(excinfo.value)
    assert "'Person'" in message
    assert "'P'" not in message
    assert excinfo.value.context == {
        "section": "relation_types",
        "id": "owns",
        "field": "domain",
        "value": "Person",
    }


@DOORS
def test_a_scalar_range_is_refused_by_the_same_rule(door: Door) -> None:
    """Its sibling key, because one of the two being right is the failure mode.

    ``'Person'`` alone does not distinguish the two refusals -- the character
    refusal names it too, in its ``Declared:`` list -- so this asserts the
    absence of the character the old message led with.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "entity_types": [{"id": "Person"}],
                "relation_types": [{"id": "owns", "range": "Person"}],
            }
        )

    message = str(excinfo.value)
    assert "'Person'" in message
    assert "'P'" not in message
    assert excinfo.value.context["field"] == "range"


@DOORS
def test_a_list_of_one_is_what_the_scalar_was_trying_to_say(door: Door) -> None:
    """The form the refusal points at, loading."""
    door(
        {
            "id": "x",
            "entity_types": [{"id": "Person"}],
            "relation_types": [{"id": "owns", "domain": ["Person"]}],
        }
    )


@DOORS
def test_an_endpoint_is_stored_as_the_string_its_section_is_keyed_by(door: Door) -> None:
    """``entity_types:`` keys by ``str(id)``, so an endpoint has to agree.

    A non-string endpoint passed the reference check by coercion and was then
    stored uncoerced, so ``domain`` held ``1`` against a map keyed ``'1'`` --
    the reference resolved at load and the constraint matched nothing after it.
    """
    del door
    config = OntologyConfig.from_dict(
        {
            "id": "x",
            "entity_types": [{"id": 1}],
            "relation_types": [{"id": "owns", "domain": [1]}],
        }
    )

    parts = build_ontology(config)

    assert parts.relation_types["owns"].domain == frozenset({"1"})


# --------------------------------------------------------------------------
# A malformed row is named by what it carries, not by `str(None)`
# --------------------------------------------------------------------------


@DOORS
def test_an_assertion_missing_its_subject_still_meets_the_missing_field_refusal(
    door: Door,
) -> None:
    """Well-formedness wins over a reference check, as it does in ``relation_types:``.

    ``_build_relation_types`` already sequences this way and says why: a row
    has to be well formed before it is worth telling its author which of its
    keys resolves. A reference check running first reported ``assertion on
    subject 'None'`` and carried ``id: 'None'`` in its context -- a refusal
    about the key the author *did* write, addressed to a row it could not name.
    """
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "relation_types": [{"id": "isa"}],
                "assertions": [{"relation": "isaa", "object": "mammal"}],
            }
        )

    message = str(excinfo.value)
    assert "'subject'" in message
    assert "None" not in message
    assert excinfo.value.context["field"] == "subject"


@DOORS
def test_a_taxonomy_missing_its_id_is_named_by_the_keys_it_carries(door: Door) -> None:
    """The axis half: ``_row_handle``'s other branch, which this reached as ``'None'``."""
    with pytest.raises(ValidationError) as excinfo:
        door(
            {
                "id": "x",
                "relation_types": [{"id": "isa"}],
                "taxonomies": [{"relation": "isaa"}],
            }
        )

    assert "None" not in str(excinfo.value)


# --------------------------------------------------------------------------
# `imports:` is the document saying it does not declare its sections in full
# --------------------------------------------------------------------------


@DOORS
def test_a_document_that_imports_is_not_checked_against_its_local_half(door: Door) -> None:
    """The premise of every one of the eight, stated by the document itself.

    Each check reads *a reference into a section this document declares in
    full*, and the guard on an empty section is how that premise is enforced.
    ``imports:`` is the other way a document says the premise does not hold:
    ``Ontology.imports`` is carried and never followed, because resolving
    across one needs a second vocabulary in scope that a door loading one
    file does not have -- so a name this document does not declare may be one
    the import declares, and refusing it is refusing a document this loader
    cannot adjudicate.

    Measured against the local half alone the check is shape-dependent in
    exactly the way the union was: ``IMPORTING_DOCUMENT`` in
    ``test_ontology_loading.py`` loads only because it declares **no**
    ``entity_types:``, and declaring one local type would refuse an entity
    typed from the import. One local declaration is what this adds.
    """
    door(
        {
            "id": "breeds",
            "imports": ["mammals"],
            "entity_types": [{"id": "Kennel"}],
            "entities": [{"id": "beagle", "type": "Breed"}],
        }
    )


@DOORS
def test_an_importing_document_may_extend_an_imported_type(door: Door) -> None:
    """The lattice half, which is the reference that predates the rule.

    ``isa`` behaved this way before the other seven existed, so this is the
    one member of the family where the narrowing is not new -- and the reason
    it is asserted rather than left implicit.
    """
    door(
        {
            "id": "breeds",
            "imports": ["zoology"],
            "entity_types": [{"id": "Breed", "isa": "Species"}],
        }
    )


@DOORS
def test_a_document_that_imports_nothing_is_checked_as_before(door: Door) -> None:
    """The difference, so the exemption is the import and not the section.

    The same document with the one key removed, which is what makes the pair
    a claim about ``imports:`` rather than about the type it names.
    """
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "breeds", "entity_types": [{"id": "Breed", "isa": "Species"}]})

    assert "'Species'" in str(excinfo.value)

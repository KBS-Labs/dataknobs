"""Every refusal fires from both doors, and names what it refused.

Parametrized over the two doors rather than written twice, because the claim
being tested is precisely that the two agree. A pair of hand-written suites
would let one drift and still report green.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import async_load_ontology, build_ontology, load_ontology
from dataknobs_common.ontology.config import OntologyConfig


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
    """
    with pytest.raises(ValidationError) as excinfo:
        door({"id": "x", "entity_types": [{"id": "Breed", "isa": "Speceis"}]})

    assert "'Speceis'" in str(excinfo.value)


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

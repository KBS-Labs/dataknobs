"""Building and parsing a namespaced id, and the literal vocabulary.

Ids here are *constructed*, never assembled with an f-string at the call site.
The reason is that a malformed id is unfixable once written into stored data:
the row is already there, and nothing in it says which of the two readings was
meant.
"""

from __future__ import annotations

from dataknobs_common.fields import FieldType
from dataknobs_common.ontology import (
    DK_ENTITY_TYPE,
    DK_RELATION_TYPE,
    Entity,
    EntityType,
    Literal,
    QualifiedId,
    RelationType,
    TaxonomyDefinition,
    qualify,
    split_qualified,
)


def test_qualify_builds_both_forms() -> None:
    assert qualify("retail", "cat-1183") == "retail:cat-1183"
    assert qualify("retail", "cat-1183", "catalog") == "retail:catalog:cat-1183"


def test_one_source_means_the_remainder_is_the_local_id() -> None:
    """With no source segment to find, colons in the remainder are the id's.

    This is why the parse takes the declared set rather than guessing: the
    same string means different things in an ontology binding one source and
    one binding several, and only the caller knows which.
    """
    assert split_qualified("retail:catalog:cat-1183") == QualifiedId(
        ontology_id="retail", source_id=None, local_id="catalog:cat-1183"
    )
    assert split_qualified("retail:catalog:cat-1183", ["catalog"]) == QualifiedId(
        ontology_id="retail", source_id=None, local_id="catalog:cat-1183"
    )


def test_a_declared_source_segment_is_recognised() -> None:
    parsed = split_qualified("retail:catalog:cat-1183", ["catalog", "warehouse"])

    assert parsed == QualifiedId(ontology_id="retail", source_id="catalog", local_id="cat-1183")
    assert parsed.source_id == "catalog"


def test_an_undeclared_middle_segment_stays_part_of_the_local_id() -> None:
    """Set membership, not a guess. The set is closed, so this is decidable."""
    assert split_qualified("retail:widgets:w-1", ["catalog", "warehouse"]) == QualifiedId(
        ontology_id="retail", source_id=None, local_id="widgets:w-1"
    )


def test_an_unqualified_id_reports_no_ontology() -> None:
    """A bare local id is not malformed -- it is what a document carries."""
    assert split_qualified("beagle") == QualifiedId(
        ontology_id="", source_id=None, local_id="beagle"
    )


def test_the_round_trip_holds_for_both_forms() -> None:
    assert split_qualified(qualify("retail", "cat-1183")).local_id == "cat-1183"

    both = split_qualified(qualify("retail", "cat-1183", "catalog"), ["catalog", "warehouse"])
    assert both == QualifiedId("retail", "catalog", "cat-1183")


def test_the_root_is_its_own_type() -> None:
    """An entity type is an entity, and the type system is in the graph."""
    assert EntityType(id=DK_ENTITY_TYPE).type == DK_ENTITY_TYPE
    assert RelationType(id="isa").type == DK_RELATION_TYPE


def test_an_entity_with_no_name_uses_its_id() -> None:
    assert Entity(id="dog", type="Species").name == "dog"
    assert Entity(id="dog", type="Species", name="Dog").name == "Dog"


def test_a_taxonomy_definition_with_no_name_uses_its_id() -> None:
    """The same rule, so a reader learns it once."""
    assert TaxonomyDefinition(id="species", relation="isa").name == "species"


def test_a_literal_detects_its_type_the_way_a_field_does() -> None:
    """One detection, so a literal and a record field cannot disagree."""
    assert Literal.of(10).type is FieldType.INTEGER
    assert Literal.of(1.5).type is FieldType.FLOAT
    assert Literal.of("kg").type is FieldType.STRING
    assert Literal.of(True).type is FieldType.BOOLEAN


def test_a_literal_projects_to_a_field_for_validation() -> None:
    """The route to ``validate`` and ``convert_to`` without a second copy."""
    projected = Literal(value=10, type=FieldType.INTEGER).as_field("weight")

    assert projected.name == "weight"
    assert projected.validate() is True
    assert projected.convert_to(FieldType.STRING).value == "10"


def test_a_literals_metadata_is_copied_into_the_field() -> None:
    """The projection must not hand out a reference into the literal.

    A frozen dataclass whose dict a caller can mutate is frozen in name only.
    """
    literal = Literal(value=10, type=FieldType.INTEGER, metadata={"unit": "kg"})
    projected = literal.as_field("weight")

    projected.metadata["unit"] = "lb"

    assert literal.metadata == {"unit": "kg"}

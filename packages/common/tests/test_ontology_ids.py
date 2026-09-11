"""Building and parsing a namespaced id, and the literal vocabulary.

Ids here are *constructed*, never assembled with an f-string at the call site.
The reason is that a malformed id is unfixable once written into stored data:
the row is already there, and nothing in it says which of the two readings was
meant.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.fields import FieldType
from dataknobs_common.ontology import (
    DK_ENTITY_TYPE,
    DK_RELATION_TYPE,
    Entity,
    EntityType,
    Literal,
    Ontology,
    QualifiedId,
    RelationType,
    TaxonomyDefinition,
    async_load_ontology,
    load_ontology,
    qualify,
    split_qualified,
)
from dataknobs_common.ontology.sources import SourceDescription


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


# --------------------------------------------------------------------------
# The two members an ontology has that the free pair cannot
# --------------------------------------------------------------------------


def _retail() -> Ontology:
    """A single-source ontology, which is what every authored vocabulary is."""
    return load_ontology({"id": "retail", "entities": [{"id": "cat-1183", "type": "Item"}]})


def test_the_members_round_trip_a_local_id() -> None:
    """Out through ``qualify``, back through ``localize``, unchanged."""
    onto = _retail()

    assert onto.qualify("cat-1183") == "retail:cat-1183"
    assert onto.localize(onto.qualify("cat-1183")) == "cat-1183"
    assert onto.entity(onto.localize("retail:cat-1183")) is not None


def test_qualify_composes_the_free_function_rather_than_a_second_spelling() -> None:
    """One builder for a namespaced id, because a malformed one is unfixable
    once it has been written into stored data.
    """
    onto = _retail()

    assert onto.qualify("cat-1183") == qualify("retail", "cat-1183")
    assert onto.qualify("cat-1183", "catalog") == qualify("retail", "cat-1183", "catalog")


def test_localize_refuses_another_ontologys_id_naming_both() -> None:
    """The half that earns these members their place over the free functions.

    ``split_qualified`` parses for nobody in particular, so it cannot know
    that ``wholesale:cat-1183`` is not this ontology's to look up. An ontology
    can.
    """
    onto = _retail()

    with pytest.raises(ValidationError) as excinfo:
        onto.localize("wholesale:cat-1183")

    message = str(excinfo.value)
    assert "'wholesale:cat-1183'" in message
    assert "'retail'" in message
    assert excinfo.value.context["ontology"] == "retail"


def test_localize_refuses_an_id_that_is_not_qualified_at_all() -> None:
    """A bare local id is what ``entity()`` takes, so it is not what this takes."""
    with pytest.raises(ValidationError):
        _retail().localize("cat-1183")


def test_localize_keeps_the_source_segment_for_a_multi_source_ontology() -> None:
    """The assertion the name invites the wrong guess about.

    ``localize`` returns the id in *this ontology's* space, which is the bare
    local id exactly when the ontology binds one source. Bind two and the
    space ``entities`` speaks carries the source segment, because that is what
    a layered source routes on.
    """
    onto = replace(
        _retail(),
        describes=(
            SourceDescription("catalog", "memory", None, {}, frozenset()),
            SourceDescription("warehouse", "memory", None, {}, frozenset()),
        ),
    )

    assert onto.localize("retail:catalog:cat-1183") == "catalog:cat-1183"


def test_the_async_twin_has_the_same_two_members() -> None:
    """Neither awaits: both read fields the value is already holding."""
    onto = asyncio.run(
        async_load_ontology({"id": "retail", "entities": [{"id": "cat-1183", "type": "Item"}]})
    )

    assert onto.qualify("cat-1183") == "retail:cat-1183"
    assert onto.localize("retail:cat-1183") == "cat-1183"
    assert not inspect.iscoroutinefunction(type(onto).localize)

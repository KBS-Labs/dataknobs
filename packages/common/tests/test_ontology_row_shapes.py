"""Every authored row is read for its shape, and only for the keys it reads.

The attribute row was given this in its own suite (``test_ontology_refusals.py``),
and the rows around it went on coercing: ``tuple(row.get("aliases", []))``
turned ``aliases: Beagle`` into six one-letter surface forms,
``bool(row.get("symmetric", False))`` read ``symmetric: "no"`` as symmetric,
``str(row.get("name", ""))`` turned ``name: null`` into the word ``'None'``, and
``FrozenMapping(row.get("metadata", {}))`` raised a bare ``TypeError`` for
``metadata: null`` -- an exception no door's ``Raises:`` names. A key nothing
reads -- ``alias:`` for ``aliases:`` -- loaded and was discarded.

Parametrized over both doors, for the reason the refusals suite gives: the
claim is that the two agree.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.fields import FieldType
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology import loader as loader_module
from dataknobs_common.ontology.model import Literal

if TYPE_CHECKING:
    from dataknobs_common.ontology import AsyncOntology, Ontology


def _sync_loaded(document: Mapping[str, Any]) -> Ontology:
    return load_ontology(document)


def _async_loaded(document: Mapping[str, Any]) -> AsyncOntology:
    return asyncio.run(async_load_ontology(document))


LOADERS = pytest.mark.parametrize(
    "loaded", [_sync_loaded, _async_loaded], ids=["load_ontology", "async_load_ontology"]
)

Loaded = Callable[[Mapping[str, Any]], "Ontology | AsyncOntology"]


def _answer(value: Any) -> Any:
    """A source's answer, awaited where the async door's source returned a coroutine."""
    return asyncio.run(value) if asyncio.iscoroutine(value) else value


def _document(**sections: Any) -> dict[str, Any]:
    """A small well-formed document, with any section replaced."""
    document: dict[str, Any] = {
        "id": "v",
        "entity_types": [{"id": "Breed"}],
        "relation_types": [{"id": "likes"}],
        "entities": [{"id": "rex", "type": "Breed"}],
    }
    document.update(sections)
    return document


def _entity_type(**row: Any) -> dict[str, Any]:
    return _document(entity_types=[{"id": "Breed", **row}])


def _relation_type(**row: Any) -> dict[str, Any]:
    return _document(relation_types=[{"id": "likes", **row}])


def _entity(**row: Any) -> dict[str, Any]:
    return _document(entities=[{"id": "rex", "type": "Breed", **row}])


def _assertion(**row: Any) -> dict[str, Any]:
    return _document(assertions=[{"subject": "rex", "relation": "likes", "object": "rex", **row}])


def _literal(**obj: Any) -> dict[str, Any]:
    return _assertion(object={"value": 3, **obj})


def _taxonomy(**row: Any) -> dict[str, Any]:
    return _document(taxonomies=[{"id": "kinds", "relation": "likes", **row}])


def _nested(**node: Any) -> dict[str, Any]:
    return _document(
        entities=[],
        sources=[{"id": "areas", "kind": "nested", "tree": {"name": "Root", **node}}],
    )


# --------------------------------------------------------------------------
# A value of the wrong shape is refused, naming the row and the field
# --------------------------------------------------------------------------

#: (case id, the document, the field the refusal is about)
_REFUSED: list[tuple[str, dict[str, Any], str]] = [
    # aliases -- a bare string was six one-letter aliases
    ("entity-aliases-string", _entity(aliases="Beagle"), "aliases"),
    ("entity-aliases-mapping", _entity(aliases={"a": 1}), "aliases"),
    ("entity-aliases-null-item", _entity(aliases=["x", None]), "aliases"),
    ("entity-aliases-int-items", _entity(aliases=[1, 2]), "aliases"),
    ("entity-aliases-empty-form", _entity(aliases=[""]), "aliases"),
    ("entity-type-aliases-string", _entity_type(aliases="Hound"), "aliases"),
    ("relation-type-aliases-string", _relation_type(aliases="fond of"), "aliases"),
    ("nested-node-aliases-string", _nested(aliases="Top"), "aliases"),
    # metadata -- a mapping, and nothing that dict() happens to accept
    ("entity-metadata-list", _entity(metadata=[1]), "metadata"),
    ("entity-metadata-string", _entity(metadata="x"), "metadata"),
    ("entity-metadata-pairs", _entity(metadata=[["a", 1]]), "metadata"),
    ("entity-type-metadata-list", _entity_type(metadata=[1]), "metadata"),
    ("relation-type-metadata-string", _relation_type(metadata="x"), "metadata"),
    ("assertion-metadata-list", _assertion(metadata=[1]), "metadata"),
    ("literal-metadata-string", _literal(metadata="x"), "metadata"),
    ("taxonomy-metadata-string", _taxonomy(metadata="x"), "metadata"),
    ("nested-node-metadata-list", _nested(metadata=[1]), "metadata"),
    # strings -- a label is a string; null is silence, a number is refused
    ("entity-name-int", _entity(name=3), "name"),
    ("entity-description-int", _entity(description=3), "description"),
    ("entity-type-name-int", _entity_type(name=2024), "name"),
    ("entity-type-description-list", _entity_type(description=["a"]), "description"),
    ("relation-type-name-int", _relation_type(name=5), "name"),
    ("relation-type-description-int", _relation_type(description=5), "description"),
    ("taxonomy-name-int", _taxonomy(name=5), "name"),
    ("taxonomy-description-int", _taxonomy(description=5), "description"),
    ("nested-node-description-int", _nested(description=5), "description"),
    ("literal-unit-int", _literal(unit=5), "unit"),
    ("assertion-id-int", _assertion(id=5), "id"),
    # flags -- on or off; "no" was truthy
    ("relation-type-symmetric-no", _relation_type(symmetric="no"), "symmetric"),
    ("relation-type-transitive-false-string", _relation_type(transitive="false"), "transitive"),
    ("relation-type-symmetric-one", _relation_type(symmetric=1), "symmetric"),
    # references -- an id, not a container
    ("entity-type-isa-list", _entity_type(isa=["Dog"]), "isa"),
    ("relation-type-inverse-of-list", _relation_type(inverse_of=["likes"]), "inverse_of"),
    # nested blocks
    ("entity-source-string", _entity(source="db"), "source"),
    ("entity-source-id-int", _entity(source={"source_id": 5}), "source_id"),
    ("taxonomy-materialization-string", _taxonomy(materialization="eager"), "materialization"),
    # a literal's type names a record type, as an attribute's field_type does
    ("literal-type-unknown", _literal(type="number"), "type"),
    ("literal-type-int", _literal(type=5), "type"),
]


@LOADERS
@pytest.mark.parametrize(
    ("document", "field"),
    [pytest.param(document, field, id=case) for case, document, field in _REFUSED],
)
def test_a_value_of_the_wrong_shape_is_refused_naming_its_field(
    loaded: Loaded, document: dict[str, Any], field: str
) -> None:
    """One exception type for every row, and the field named in it.

    A bare ``TypeError``, ``ValueError`` or ``AttributeError`` escaped every
    door's documented ``Raises:``; a silent misreading escaped everything.
    """
    with pytest.raises(ValidationError) as excinfo:
        loaded(document)

    assert excinfo.value.context["field"] == field
    assert repr(field) in str(excinfo.value)


@LOADERS
def test_a_bare_string_alias_says_how_to_write_one(loaded: Loaded) -> None:
    """The refusal names the row and gives the list the author meant."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(_entity(aliases="Beagle"))

    message = str(excinfo.value)
    assert "'rex'" in message
    assert "['Beagle']" in message
    assert excinfo.value.context["section"] == "entities"
    assert excinfo.value.context["id"] == "rex"


@LOADERS
def test_a_nested_node_is_named_by_the_id_it_was_minted(loaded: Loaded) -> None:
    """A tree node has no ``id:`` of its own to be pointed at by."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(_nested(aliases="Top"))

    assert "areas" in str(excinfo.value)


# --------------------------------------------------------------------------
# null is silence -- every optional field reads as its default
# --------------------------------------------------------------------------


@LOADERS
def test_null_optional_fields_read_as_their_defaults(loaded: Loaded) -> None:
    """``metadata: null`` raised ``TypeError``; ``name: null`` read as ``'None'``."""
    ontology = loaded(
        _document(
            entity_types=[
                {"id": "Breed", "name": None, "aliases": None, "metadata": None, "isa": None}
            ],
            relation_types=[
                {
                    "id": "likes",
                    "aliases": None,
                    "metadata": None,
                    "symmetric": None,
                    "transitive": None,
                    "inverse_of": None,
                }
            ],
            entities=[
                {
                    "id": "rex",
                    "type": "Breed",
                    "name": None,
                    "aliases": None,
                    "description": None,
                    "metadata": None,
                    "source": None,
                }
            ],
            assertions=[
                {"subject": "rex", "relation": "likes", "object": "rex", "metadata": None},
                {
                    "subject": "rex",
                    "relation": "likes",
                    "object": {"value": 3, "metadata": None, "unit": None},
                },
            ],
            taxonomies=[
                {"id": "kinds", "relation": "likes", "metadata": None, "materialization": None}
            ],
        )
    )

    breed = ontology.entity_types["Breed"]
    assert (breed.name, breed.aliases, dict(breed.metadata), breed.isa) == ("Breed", (), {}, None)
    likes = ontology.relation_types["likes"]
    assert (likes.symmetric, likes.transitive, likes.aliases, likes.inverse_of) == (
        False,
        False,
        (),
        None,
    )
    assert ontology.taxonomies["kinds"].metadata == {}


@LOADERS
def test_a_null_entity_field_reads_as_its_default(loaded: Loaded) -> None:
    """The entity half, through the source a loaded vocabulary answers from."""
    ontology = loaded(
        _entity(name=None, aliases=None, description=None, metadata=None, source=None)
    )
    entity = _answer(ontology.entities.get("rex"))

    assert entity is not None
    assert (entity.name, entity.aliases, entity.description, dict(entity.metadata)) == (
        "rex",
        (),
        None,
        {},
    )
    assert entity.source is None


# --------------------------------------------------------------------------
# Well-formed values read as written
# --------------------------------------------------------------------------


@LOADERS
def test_well_formed_values_read_as_written(loaded: Loaded) -> None:
    ontology = loaded(
        _document(
            relation_types=[
                {"id": "likes", "symmetric": True, "transitive": False, "aliases": ["fond of"]}
            ],
            entity_types=[{"id": "Breed", "aliases": ("Kind",), "metadata": {"a": 1}}],
        )
    )

    likes = ontology.relation_types["likes"]
    assert (likes.symmetric, likes.transitive, likes.aliases) == (True, False, ("fond of",))
    assert ontology.entity_types["Breed"].aliases == ("Kind",)
    assert dict(ontology.entity_types["Breed"].metadata) == {"a": 1}


@LOADERS
def test_a_literal_type_naming_a_record_type_is_read(loaded: Loaded) -> None:
    ontology = loaded(_literal(value=3.5, type="Float", unit="kg"))
    (assertion,) = _answer(ontology.assertions.find(subject="rex"))

    assert isinstance(assertion.object, Literal)
    assert (assertion.object.type, assertion.object.unit) == (FieldType.FLOAT, "kg")


# --------------------------------------------------------------------------
# A key nothing reads is refused, naming the key and the keys that are read
# --------------------------------------------------------------------------

#: (case id, the document, the stray key, the section)
_UNREAD: list[tuple[str, dict[str, Any], str, str]] = [
    ("entity-type", _entity_type(alias=["Hound"]), "alias", "entity_types"),
    ("relation-type", _relation_type(symetric=True), "symetric", "relation_types"),
    ("entity", _entity(alias=["Rex"]), "alias", "entities"),
    ("assertion", _assertion(polarty="asserted"), "polarty", "assertions"),
    ("literal", _literal(unti="kg"), "unti", "assertions.object"),
    ("entity-ref", _assertion(object={"entity": "rex", "unit": "kg"}), "unit", "assertions.object"),
    (
        "materialization",
        _taxonomy(materialization={"struture": "copied"}),
        "struture",
        "taxonomies.materialization",
    ),
]


@LOADERS
@pytest.mark.parametrize(
    ("document", "key", "section"),
    [pytest.param(document, key, section, id=case) for case, document, key, section in _UNREAD],
)
def test_a_key_this_loader_does_not_read_is_refused(
    loaded: Loaded, document: dict[str, Any], key: str, section: str
) -> None:
    """A misspelt key configured nothing and said nothing."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(document)

    assert key in str(excinfo.value)
    assert excinfo.value.context["keys"] == [key]
    assert excinfo.value.context["section"] == section


@LOADERS
def test_unread_keys_of_mixed_types_are_refused_by_name(loaded: Loaded) -> None:
    """YAML's ``1: x`` beside ``foo: y`` must not escape the sort as a ``TypeError``."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(_document(entities=[{"id": "rex", "type": "Breed", 1: "x", "foo": "y"}]))

    assert excinfo.value.context["keys"] == ["1", "foo"]


@LOADERS
def test_a_phase_2_key_keeps_its_own_message(loaded: Loaded) -> None:
    """``cardinality:`` is not read *yet*, which is a different answer from never."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(_relation_type(cardinality="one_to_many", symetric=True))

    assert "phase 2" in str(excinfo.value)
    assert excinfo.value.context["field"] == "cardinality"


def test_each_closed_key_set_is_what_its_builder_reads() -> None:
    """The published sets, against the keys a well-formed row may carry.

    A key added to a builder and not to its set is refused for every document
    that writes it; a key added to the set and not read is the discard this
    whole change ends. So the sets are pinned here, once, by name.
    """
    assert {
        "id",
        "name",
        "description",
        "aliases",
        "metadata",
        "attributes",
        "isa",
    } == loader_module.ENTITY_TYPE_ROW_KEYS
    assert {
        "id",
        "name",
        "description",
        "aliases",
        "metadata",
        "domain",
        "range",
        "inverse_of",
        "symmetric",
        "transitive",
        "inference",
    } == loader_module.RELATION_TYPE_ROW_KEYS
    assert {
        "id",
        "type",
        "name",
        "aliases",
        "description",
        "metadata",
        "source",
    } == loader_module.ENTITY_ROW_KEYS
    assert {
        "id",
        "subject",
        "relation",
        "object",
        "metadata",
        "polarity",
    } == loader_module.ASSERTION_ROW_KEYS
    assert {"value", "type", "unit", "metadata"} == loader_module.LITERAL_ROW_KEYS
    assert {"structure", "content"} == loader_module.MATERIALIZATION_KEYS


@LOADERS
def test_a_row_carrying_every_key_it_reads_loads(loaded: Loaded) -> None:
    """The positive half of the pin: every key in every set is accepted."""
    loaded(
        {
            "id": "v",
            "entity_types": [
                {
                    "id": "Breed",
                    "name": "Breed",
                    "description": "a kind",
                    "aliases": ["Kind"],
                    "metadata": {},
                    "attributes": [],
                    "isa": None,
                }
            ],
            "relation_types": [
                {
                    "id": "likes",
                    "name": "likes",
                    "description": "fond",
                    "aliases": [],
                    "metadata": {},
                    "domain": ["Breed"],
                    "range": ["Breed"],
                    "inverse_of": None,
                    "symmetric": True,
                    "transitive": False,
                    "inference": "on_demand",
                }
            ],
            "entities": [
                {
                    "id": "rex",
                    "type": "Breed",
                    "name": "Rex",
                    "aliases": ["R"],
                    "description": "a dog",
                    "metadata": {},
                    "source": {"source_id": "s", "kind": "record", "projection_id": "p", "row": 1},
                }
            ],
            "assertions": [
                {
                    "id": "a1",
                    "subject": "rex",
                    "relation": "likes",
                    "object": {"value": 3, "type": "integer", "unit": "kg", "metadata": {}},
                    "metadata": {},
                    "polarity": "asserted",
                }
            ],
            "taxonomies": [
                {
                    "id": "kinds",
                    "relation": "likes",
                    "materialization": {"structure": "on_demand", "content": "on_demand"},
                }
            ],
        }
    )

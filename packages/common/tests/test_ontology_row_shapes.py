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
from dataknobs_common.ontology.model import InferenceMode, Literal, Polarity

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

    assert "tree node 'root'" in str(excinfo.value)
    assert excinfo.value.context["node_id"] == "root"
    assert excinfo.value.context["source_id"] == "areas"


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
    """The published sets, pinned by name.

    This pins the constants, not what each builder reads: a key added to a
    builder and not to its set is refused for every document that writes it,
    which ``test_a_row_carrying_every_key_it_reads_loads`` turns red. A key
    added to a set and read by nothing would pass both -- review is what
    catches that, and this list is what a reviewer reads it against.
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


# --------------------------------------------------------------------------
# null is silence for every key a row carries, not only the optional labels
# --------------------------------------------------------------------------


@LOADERS
def test_null_enums_and_endpoints_read_as_their_defaults(loaded: Loaded) -> None:
    """``inference: null`` was refused as the mode ``'None'``; ``domain: null`` as a bare string."""
    ontology = loaded(
        _document(
            relation_types=[{"id": "likes", "inference": None, "domain": None, "range": None}],
            assertions=[{"subject": "rex", "relation": "likes", "object": "rex", "polarity": None}],
            taxonomies=[
                {
                    "id": "kinds",
                    "relation": "likes",
                    "materialization": {"structure": None, "content": None},
                }
            ],
        )
    )

    likes = ontology.relation_types["likes"]
    assert likes.inference is InferenceMode.ON_DEMAND
    assert (likes.domain, likes.range) == (frozenset(), frozenset())
    (assertion,) = _answer(ontology.assertions.find(subject="rex"))
    assert assertion.polarity is Polarity.ASSERTED
    materialization = ontology.taxonomies["kinds"].materialization
    assert (materialization.structure, materialization.content) == (
        InferenceMode.ON_DEMAND,
        InferenceMode.ON_DEMAND,
    )


@LOADERS
def test_a_null_kind_on_an_axis_row_is_no_kind(loaded: Loaded) -> None:
    """``kind: null`` asked for no backing, so this door builds the assertion axis."""
    ontology = loaded(_taxonomy(kind=None))

    assert "kinds" in ontology.taxonomies


@LOADERS
def test_a_null_section_reads_as_an_empty_one(loaded: Loaded) -> None:
    ontology = loaded(
        {
            "id": "v",
            "imports": None,
            "entity_types": None,
            "relation_types": None,
            "entities": None,
            "assertions": None,
            "sources": None,
            "taxonomies": None,
        }
    )

    assert dict(ontology.entity_types) == {}


#: (case id, the document, the required field written as null)
_NULL_REQUIRED: list[tuple[str, dict[str, Any], str]] = [
    ("entity-type", _entity(type=None), "type"),
    ("entity-id", _document(entities=[{"id": None, "type": "Breed"}]), "id"),
    ("entity-type-id", _document(entity_types=[{"id": None}]), "id"),
    ("relation-type-id", _document(relation_types=[{"id": None}]), "id"),
    ("assertion-subject", _assertion(subject=None), "subject"),
    ("assertion-relation", _assertion(relation=None), "relation"),
    ("assertion-object", _assertion(object=None), "object"),
    ("taxonomy-relation", _taxonomy(relation=None), "relation"),
    ("entity-ref", _assertion(object={"entity": None}), "entity"),
]


@LOADERS
@pytest.mark.parametrize(
    ("document", "field"),
    [pytest.param(document, field, id=case) for case, document, field in _NULL_REQUIRED],
)
def test_a_required_field_written_as_null_is_missing(
    loaded: Loaded, document: dict[str, Any], field: str
) -> None:
    """``type: null`` loaded an entity typed ``'None'``, and the reference check skipped it."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(document)

    assert excinfo.value.context["field"] == field
    assert "declares no" in str(excinfo.value)


# --------------------------------------------------------------------------
# A required reference is one id, not whatever str() makes of a container
# --------------------------------------------------------------------------

_IMPORTS = {"imports": ["base"]}

#: (case id, the document, the field) -- most under ``imports:``, where the
#: reference checks stand down and ``str()`` was all that read the value.
_BAD_REFS: list[tuple[str, dict[str, Any], str]] = [
    ("entity-type-list", _entity(type=["Breed"]) | _IMPORTS, "type"),
    ("entity-id-list", _document(entities=[{"id": ["rex"], "type": "Breed"}]), "id"),
    ("entity-type-id-mapping", _document(entity_types=[{"id": {"a": 1}}]), "id"),
    ("assertion-relation-list", _assertion(relation=["likes"]) | _IMPORTS, "relation"),
    ("assertion-subject-mapping", _assertion(subject={"a": 1}) | _IMPORTS, "subject"),
    ("taxonomy-relation-list", _taxonomy(relation=["likes"]) | _IMPORTS, "relation"),
    ("entity-ref-list", _assertion(object={"entity": ["rex"]}) | _IMPORTS, "entity"),
    ("entity-type-bool", _entity(type=True) | _IMPORTS, "type"),
    ("domain-element-list", _relation_type(domain=[["Breed"]]), "domain"),
]


@LOADERS
@pytest.mark.parametrize(
    ("document", "field"),
    [pytest.param(document, field, id=case) for case, document, field in _BAD_REFS],
)
def test_a_required_reference_of_the_wrong_shape_is_refused(
    loaded: Loaded, document: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        loaded(document)

    assert excinfo.value.context["field"] == field
    assert "which no" not in str(excinfo.value), "refused for its shape, not as unresolved"


@LOADERS
def test_a_numeric_reference_still_names_the_id_it_renders_as(loaded: Loaded) -> None:
    ontology = loaded(_document(entity_types=[{"id": 7}], entities=[{"id": 1, "type": 7}]))

    entity = _answer(ontology.entities.get("1"))
    assert entity is not None
    assert entity.type == "7"


# --------------------------------------------------------------------------
# A section is a list of rows, and each row a mapping
# --------------------------------------------------------------------------

_BAD_SECTIONS: list[tuple[str, dict[str, Any], str]] = [
    ("entities-string-row", {"id": "v", "entities": ["rex"]}, "entities"),
    ("entity-types-string-row", {"id": "v", "entity_types": ["Breed"]}, "entity_types"),
    ("relation-types-int-row", {"id": "v", "relation_types": [3]}, "relation_types"),
    ("assertions-string-row", {"id": "v", "assertions": ["x"]}, "assertions"),
    ("taxonomies-int-row", {"id": "v", "taxonomies": [5]}, "taxonomies"),
    ("sources-string-row", {"id": "v", "sources": ["x"]}, "sources"),
    ("entities-keyed-by-id", {"id": "v", "entities": {"rex": {"type": "Breed"}}}, "entities"),
    ("sources-string", {"id": "v", "sources": "x"}, "sources"),
    ("imports-string", {"id": "v", "imports": "base"}, "imports"),
    ("imports-int-item", {"id": "v", "imports": [5]}, "imports"),
]


@LOADERS
@pytest.mark.parametrize(
    ("document", "section"),
    [pytest.param(document, section, id=case) for case, document, section in _BAD_SECTIONS],
)
def test_a_section_of_the_wrong_shape_is_refused(
    loaded: Loaded, document: dict[str, Any], section: str
) -> None:
    """``entities: ["rex"]`` raised ``AttributeError``; ``imports: base`` was four imports."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(document)

    assert excinfo.value.context["section"] == section
    assert f"`{section}:`" in str(excinfo.value)


@LOADERS
def test_a_section_keyed_by_id_says_how_to_write_it(loaded: Loaded) -> None:
    with pytest.raises(ValidationError) as excinfo:
        loaded({"id": "v", "entities": {"rex": {"type": "Breed"}}})

    assert "`id:`" in str(excinfo.value)


# --------------------------------------------------------------------------
# What a refusal points at
# --------------------------------------------------------------------------


@LOADERS
def test_an_id_less_assertion_with_mixed_type_keys_is_refused_by_name(loaded: Loaded) -> None:
    """The handle sorted raw keys, so ``1: x`` escaped as a ``TypeError`` from the sort."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(
            _assertion(**{"1": "x"})
            | {"assertions": [{"subject": "rex", "relation": "likes", "object": "rex", 1: "x"}]}
        )

    assert excinfo.value.context["keys"] == ["1"]


@LOADERS
def test_an_assertion_is_pointed_at_by_its_subject_and_relation(loaded: Loaded) -> None:
    """``a row carrying ['object', 'relation', 'subject']`` names none of fifty assertions."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(_literal(metadata="x"))

    message = str(excinfo.value)
    assert "'rex'" in message
    assert "'likes'" in message
    assert "a row carrying" not in message
    assert excinfo.value.context["subject"] == "rex"
    assert excinfo.value.context["relation"] == "likes"


@LOADERS
def test_a_source_refusal_carries_the_entity_not_the_locator(loaded: Loaded) -> None:
    """A locator's own ``id:`` was reported as the row's."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(_entity(source={"source_id": 5, "id": 42}))

    message = str(excinfo.value)
    assert message.startswith("the `source:` of entity 'rex'")
    assert excinfo.value.context["id"] == "rex"


@LOADERS
def test_a_materialization_refusal_carries_its_taxonomy(loaded: Loaded) -> None:
    with pytest.raises(ValidationError) as excinfo:
        loaded(_taxonomy(materialization={"struture": "copied"}))

    assert excinfo.value.context["id"] == "kinds"
    assert str(excinfo.value).startswith("the `materialization:` of taxonomy 'kinds'")


@LOADERS
def test_an_attribute_refusal_carries_no_row_id(loaded: Loaded) -> None:
    """An attribute is named by its entity type and its name; an ``id`` of ``None`` is noise."""
    with pytest.raises(ValidationError) as excinfo:
        loaded(_document(entity_types=[{"id": "Breed", "attributes": [{"name": 5}]}]))

    assert "id" not in excinfo.value.context
    assert excinfo.value.context["entity_type"] == "Breed"


@LOADERS
def test_an_enum_refusal_names_its_row(loaded: Loaded) -> None:
    with pytest.raises(ValidationError) as excinfo:
        loaded(_relation_type(inference="sometimes"))

    assert "'likes'" in str(excinfo.value)
    assert excinfo.value.context["id"] == "likes"
    assert excinfo.value.context["field"] == "inference"


# --------------------------------------------------------------------------
# An assertion id is an id
# --------------------------------------------------------------------------


@LOADERS
def test_an_assertion_id_is_rendered_as_every_id_is(loaded: Loaded) -> None:
    ontology = loaded(_assertion(id=5))

    assert _answer(ontology.assertions.get("5")) is not None


@LOADERS
def test_an_empty_assertion_id_is_minted(loaded: Loaded) -> None:
    ontology = loaded(_assertion(id=""))

    (assertion,) = _answer(ontology.assertions.find(subject="rex"))
    assert assertion.id == "rex-likes-rex"


# --------------------------------------------------------------------------
# A nested source's own keys
# --------------------------------------------------------------------------


@LOADERS
def test_a_nested_spec_with_null_keys_reads_their_defaults(loaded: Loaded) -> None:
    """``type: null`` minted entities typed ``'None'``; ``relation: null`` edges over ``'None'``."""
    ontology = loaded(
        _document(
            entities=[],
            sources=[
                {
                    "id": "areas",
                    "kind": "nested",
                    "type": None,
                    "relation": None,
                    "child_key": None,
                    "name_key": None,
                    "tree": {"name": "Root", "children": [{"name": "Leaf"}]},
                }
            ],
        )
    )

    leaf = _answer(ontology.entities.get("root/leaf"))
    assert leaf is not None
    assert leaf.type == "areas"
    (edge,) = _answer(ontology.assertions.find(subject="root/leaf"))
    assert edge.relation == "isa"


@LOADERS
@pytest.mark.parametrize("key", ["type", "relation", "child_key", "name_key"])
def test_a_nested_spec_key_of_the_wrong_shape_is_refused(loaded: Loaded, key: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        loaded(
            _document(
                entities=[],
                sources=[{"id": "areas", "kind": "nested", key: ["x"], "tree": {"name": "Root"}}],
            )
        )

    assert excinfo.value.context["field"] == key

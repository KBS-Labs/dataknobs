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

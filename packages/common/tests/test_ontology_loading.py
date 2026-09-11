"""A vocabulary loads with nothing standing up behind it.

This is the criterion worth stating twice: it is the one acceptance test in
the ontology work that needs no service. Every other one stands up a store, an
index or a provider first.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from dataknobs_common.ontology import Ontology, async_load_ontology, load_ontology
from dataknobs_common.ontology.sources import MappingAssertionSource, MappingEntitySource


def test_a_file_loads_with_no_database_no_embedder_and_no_loop(
    mammals_path: Path,
) -> None:
    """The whole point of the synchronous door.

    The loop assertion is inside the test rather than around it: a test that
    merely *is not* async proves nothing about the function, since a sync
    function can still be called from inside a running loop.
    """
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    onto = load_ontology(mammals_path)

    assert isinstance(onto, Ontology)
    assert onto.id == "mammals"
    assert onto.version == "1.0"


def test_the_door_is_not_a_coroutine_function() -> None:
    """Not async, and not merely *usable* without awaiting.

    A caller in ordinary synchronous code must be able to write
    ``onto = load_ontology(path)`` with no ``asyncio.run`` around it.
    """
    assert not inspect.iscoroutinefunction(load_ontology)


def test_the_bound_sources_are_the_synchronous_flavour(mammals_path: Path) -> None:
    """The flavour is fixed by which door was called, not inferred."""
    onto = load_ontology(mammals_path)

    assert type(onto.entities) is MappingEntitySource
    assert type(onto.assertions) is MappingAssertionSource


def test_a_mapping_loads_the_same_as_a_file() -> None:
    """A caller who already holds the document need not write it to disk."""
    onto = load_ontology(
        {
            "ontology": {
                "id": "mammals",
                "version": "2.0",
                "entities": [{"id": "dog", "type": "Species", "name": "Dog"}],
            }
        }
    )

    assert onto.version == "2.0"
    assert onto.entity("dog") is not None


def test_an_unwrapped_document_loads_too() -> None:
    """The ``ontology:`` wrapper is how the section reads inside a larger file.

    A document that *is* the ontology needs no wrapper, and demanding one
    would make the file and mapping spellings differ for no reason.
    """
    onto = load_ontology({"id": "mammals", "entities": []})

    assert onto.id == "mammals"


def test_the_sections_map_onto_values(mammals_path: Path) -> None:
    """Types, relations and their declared detail survive the load."""
    onto = load_ontology(mammals_path)

    assert sorted(onto.entity_types) == ["Breed", "Species"]
    assert onto.relation_types["isa"].transitive is True

    latin_name = onto.entity_types["Species"].attributes[0]
    assert latin_name.name == "latin_name"
    assert latin_name.required is True


def test_an_entity_with_no_name_answers_to_its_id() -> None:
    """An empty name means *use the id*, resolved once rather than per reader."""
    onto = load_ontology({"id": "x", "entities": [{"id": "dog", "type": "Species"}]})

    assert onto.entity("dog").name == "dog"


# --------------------------------------------------------------------------
# `imports:` survives the load, carried and never followed
# --------------------------------------------------------------------------

#: A vocabulary that declares another one's namespace comes into scope.
#:
#: The imported ontology is deliberately not in scope here: a file authored
#: for a registry must still load at the file door, or it cannot be tested
#: until the registry exists.
IMPORTING_DOCUMENT = {
    "id": "breeds",
    "imports": ["mammals"],
    "entities": [{"id": "beagle", "type": "Breed"}],
}


def test_imports_survive_the_load() -> None:
    """The config field shipped and the value field did not, so the list was
    read out of the document and dropped on the way out -- leaving the
    component that resolves across an import unable to see what to resolve
    against.
    """
    onto = load_ontology(IMPORTING_DOCUMENT)

    assert onto.imports == ("mammals",)


def test_an_import_is_carried_and_not_followed() -> None:
    """Resolving across one needs a second ontology in scope, which a door
    loading a single file does not have. Carrying the list is the whole of
    what this door owes.
    """
    onto = load_ontology(IMPORTING_DOCUMENT)

    assert onto.entity("dog") is None
    assert onto.entity("beagle") is not None


def test_a_document_with_no_imports_carries_an_empty_tuple() -> None:
    """Absent is empty rather than None, so a reader never branches on it."""
    assert load_ontology({"id": "x"}).imports == ()


def test_the_async_door_carries_imports_too() -> None:
    """A field, not a flavour: a list of ids has nothing to await."""
    onto = asyncio.run(async_load_ontology(IMPORTING_DOCUMENT))

    assert onto.imports == ("mammals",)

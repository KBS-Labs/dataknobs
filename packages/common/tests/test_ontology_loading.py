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

from dataknobs_common.ontology import Ontology, load_ontology
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

"""A vocabulary's fold is injected, and a real second fold reaches it.

The phase-one acceptance clause reads: *normalization is injected, not
defaulted into place -- swapping the default for one that folds differently
changes an exact-match outcome. An injection point is only real if a second
implementation is reachable.*

The injection point is the vocabulary's ``normalizer``: the one
``MappingEntitySource`` keys its index on, which ``load_ontology`` and
``async_load_ontology`` hand to the source they build. An
``ExactNormalizedSignal`` given no ``normalizer`` of its own passes the raw
query to the source, so the source's fold is the only fold.

The second implementation is :func:`basic_normalization_fn`, which ships in
this distribution. ``dataknobs-common`` cannot import it, which is why the test
lives here: reaching the injection point from a different distribution is
what *reachable* means.

**Both directions are asserted.** A row only the swapped fold matches shows
the second implementation reaches the injection point; a row only the default
matches shows the swap replaced the default rather than adding to it. A loader
that ignored ``normalizer=`` fails the first kind, and a resolver that matched
everything fails the second. The positive control -- every authored spelling,
queried as authored, under either fold -- is what keeps the *miss* cells from
passing against a resolver that has stopped matching at all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from dataknobs_common.entity_resolution import ExactNormalizedSignal
from dataknobs_common.ontology import (
    Entity,
    MappingEntitySource,
    async_build_resolver,
    async_load_ontology,
    build_resolver,
    load_ontology,
)
from dataknobs_common.text import default_normalizer

from dataknobs_xization.normalize import basic_normalization_fn

Fold = Callable[[str], str]

# (entity id, authored name)
#
# The ids are opaque on purpose. An entity's id is indexed as one of its
# surface forms, so an id spelled like a query -- ``strasse`` beside a
# ``STRASSE`` row -- matches under any fold that lowercases, and the row would
# measure the id rather than the name.
AUTHORED: tuple[tuple[str, str], ...] = (
    ("term-1", "PowerPoint"),
    ("term-2", "Rock\u2019n\u2019roll"),
    ("term-3", "Stra\u00dfe"),
    ("term-4", "Beagle"),
)

# (query, entity it names, matches under default_normalizer,
#  matches under basic_normalization_fn)
#
# McDonald's is deliberately absent: basic_normalization_fn camelCase-expands
# it to "mc donald's", so it would read as a quote case while measuring two
# steps at once.
DIFFERENCES: tuple[tuple[str, str, bool, bool], ...] = (
    # camelCase expansion, which only basic_normalization_fn does
    ("power point", "term-1", False, True),
    # smart-quote simplification, which only basic_normalization_fn does
    ("rock'n'roll", "term-2", False, True),
    # casefold folds sharp s to "ss"; lower leaves it alone
    ("STRASSE", "term-3", True, False),
    # default_normalizer strips; basic_normalization_fn does not by default
    (" beagle ", "term-4", True, False),
)

FOLDS: dict[str, Fold] = {
    "default_normalizer": default_normalizer,
    "basic_normalization_fn": basic_normalization_fn,
}


def _expected(fold_name: str) -> dict[str, list[str]]:
    """Every query this file asks, and the ids it should place under a fold."""
    swapped = fold_name == "basic_normalization_fn"
    expected = {name: [entity_id] for entity_id, name in AUTHORED}
    for query, entity_id, under_default, under_swap in DIFFERENCES:
        expected[query] = [entity_id] if (under_swap if swapped else under_default) else []
    return expected


def _document() -> dict[str, Any]:
    return {
        "ontology": {
            "id": "injected-fold",
            "entity_types": [{"id": "Term"}],
            "entities": [
                {"id": entity_id, "type": "Term", "name": name} for entity_id, name in AUTHORED
            ],
            "resolver": {"rungs": [{"kind": "exact"}]},
        }
    }


def test_the_table_differs_in_both_directions() -> None:
    """The rows separate the two folds both ways, so the test cannot be one-sided."""
    assert any(not d and s for _, _, d, s in DIFFERENCES)
    assert any(d and not s for _, _, d, s in DIFFERENCES)


@pytest.mark.parametrize("fold_name", FOLDS)
def test_the_synchronous_door_folds_with_the_injected_normalizer(fold_name: str) -> None:
    document = _document()
    onto = load_ontology(document, normalizer=FOLDS[fold_name])
    resolver = build_resolver(document, onto)

    placed = {
        query: [c.entity_id for c in resolver.resolve(query).candidates]
        for query in _expected(fold_name)
    }

    assert placed == _expected(fold_name)


@pytest.mark.parametrize("fold_name", FOLDS)
async def test_the_asynchronous_door_folds_with_the_injected_normalizer(fold_name: str) -> None:
    """The async loader has its own ``normalizer`` channel, so it is read on its own."""
    document = _document()
    onto = await async_load_ontology(document, normalizer=FOLDS[fold_name])
    resolver = await async_build_resolver(document, onto)

    placed = {
        query: [c.entity_id for c in (await resolver.resolve(query)).candidates]
        for query in _expected(fold_name)
    }

    assert placed == _expected(fold_name)


@pytest.mark.parametrize("fold_name", FOLDS)
def test_a_hand_built_source_folds_with_the_injected_normalizer(fold_name: str) -> None:
    """The route a consumer building a source by hand takes, with no loader in between."""
    entities = {
        entity_id: Entity(id=entity_id, type="Term", name=name) for entity_id, name in AUTHORED
    }
    signal = ExactNormalizedSignal(MappingEntitySource(entities, normalizer=FOLDS[fold_name]))

    placed = {
        query: [c.entity_id for c in signal.candidates(query, k=5)]
        for query in _expected(fold_name)
    }

    assert placed == _expected(fold_name)

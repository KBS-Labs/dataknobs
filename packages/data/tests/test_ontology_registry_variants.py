"""``OntologyRegistry.load`` reads a document's ``variants:`` section as the loaders do."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.events import Event, EventType, InMemoryEventBus
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import load_ontology
from dataknobs_common.ontology.config import OntologyConfig

from dataknobs_data.ontology import OntologyRegistry

DEFAULT_SPECS = ["drop_parentheticals", "ampersand", "slash_or", "inflect"]


def grocery(variants: Any = None, **overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "id": "grocery",
        "entity_types": [{"id": "item"}, {"id": "aisle"}],
        "entities": [
            {"id": "apple", "type": "item", "name": "Apple"},
            {"id": "glass", "type": "item", "name": "Glass"},
            {"id": "glasses", "type": "item", "name": "Glasses"},
            {"id": "fruit-veg", "type": "aisle", "name": "Fruit & Veg (fresh)"},
        ],
    }
    if variants is not None:
        document["variants"] = variants
    document.update(overrides)
    return document


async def test_the_registry_admits_the_same_variants_as_the_loader() -> None:
    async with OntologyRegistry() as registry:
        onto = await registry.load(grocery(DEFAULT_SPECS))
        assert await onto.by_surface_form("apples") == frozenset({"apple"})
        assert await onto.by_surface_form("fruit and veg") == frozenset({"fruit-veg"})
        assert onto.variants == load_ontology(grocery(DEFAULT_SPECS)).variants
        assert {each.variant.form for each in onto.variants.refused} == {"Glasses", "Glass"}
        assert onto.longest_form_tokens() == 4


async def test_with_no_section_the_registry_binds_the_vocabulary_as_authored() -> None:
    async with OntologyRegistry() as registry:
        onto = await registry.load(grocery())
        assert onto.variants is None
        assert await onto.by_surface_form("apples") == frozenset()
        assert onto.longest_form_tokens() == 3


async def test_the_registrys_fold_is_the_one_admission_counts_under() -> None:
    def squash(form: str) -> str:
        return "".join(form.split()).casefold()

    document = {
        "id": "fold",
        "entity_types": [{"id": "item"}],
        "entities": [
            {"id": "item-1", "type": "item", "name": "Cup Cake"},
            {"id": "item-2", "type": "item", "name": "Cupcakes"},
        ],
        "variants": ["inflect"],
    }
    async with OntologyRegistry(normalizer=squash) as registry:
        onto = await registry.load(document)
        assert {each.variant.form for each in onto.variants.refused} == {"Cup Cakes", "Cupcake"}


async def test_a_reload_publishes_no_change_the_rules_made() -> None:
    """The diff is over the entities as authored, so variants are not a change in the vocabulary."""
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus = InMemoryEventBus()
    await bus.connect()
    await bus.subscribe("ontology:grocery", record)
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**grocery(DEFAULT_SPECS)), event_bus=bus
    )
    try:
        await registry.load()
        seen.clear()
        await registry.load(grocery(["inflect"]), replace=True)
        updated = [event for event in seen if event.type == EventType.UPDATED]
        assert len(updated) == 1
        payload = updated[0].payload
        assert payload["gone"] == [] and payload["arrived"] == [] and payload["renamed"] == []
    finally:
        await registry.close()
        await bus.close()


async def test_variants_over_a_live_source_are_refused() -> None:
    document = {
        "id": "live",
        "entity_types": [{"id": "Product"}],
        "sources": [
            {
                "id": "products",
                "kind": "record",
                "entity_projection": {
                    "table": "products",
                    "id": "sku",
                    "name": "title",
                    "type": {"const": "Product"},
                },
                "schema": [
                    {"name": "sku", "type": "string"},
                    {"name": "title", "type": "string"},
                ],
            }
        ],
        "variants": ["inflect"],
    }
    async with OntologyRegistry() as registry:
        with pytest.raises(ValidationError, match="variants"):
            await registry.load(document)

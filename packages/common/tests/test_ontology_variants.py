"""An ontology document's ``variants:`` section, through the doors that bind it."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    Entity,
    MappingEntitySource,
    async_load_ontology,
    build_ontology,
    load_ontology,
    with_variants,
)
from dataknobs_common.ontology.config import OntologyConfig
from dataknobs_common.text import inflect, strip_prefix

DEFAULT_SPECS = ["drop_parentheticals", "ampersand", "slash_or", "inflect"]


def grocery(variants: Any = None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "id": "grocery",
        "entity_types": [{"id": "item"}, {"id": "aisle"}],
        "entities": [
            {"id": "apple", "type": "item", "name": "Apple"},
            {"id": "glass", "type": "item", "name": "Glass"},
            {"id": "glasses", "type": "item", "name": "Glasses"},
            {"id": "fruit-veg", "type": "aisle", "name": "Fruit & Veg (fresh)"},
            {
                "id": "acme-pb",
                "type": "item",
                "name": "Acme Peanut Butter",
                "metadata": {"brand": "Acme"},
            },
        ],
    }
    if variants is not None:
        document["variants"] = variants
    return document


def test_with_no_section_nothing_changes() -> None:
    onto = load_ontology(grocery())
    assert onto.variants is None
    assert onto.by_surface_form("apples") == frozenset()
    assert onto.entity("apple").aliases == ()


def test_the_section_admits_variants_as_aliases() -> None:
    onto = load_ontology(grocery(DEFAULT_SPECS))
    assert onto.by_surface_form("apples") == frozenset({"apple"})
    assert onto.by_surface_form("fruit and veg") == frozenset({"fruit-veg"})
    assert "Apples" in onto.entity("apple").aliases


def test_a_refusal_is_carried_on_the_vocabulary() -> None:
    onto = load_ontology(grocery(DEFAULT_SPECS))
    assert onto.variants is not None
    refused = {(each.variant.key, each.variant.form) for each in onto.variants.refused}
    assert refused == {("glass", "Glasses"), ("glasses", "Glass")}
    # The authored forms still answer as authored.
    assert onto.by_surface_form("glass") == frozenset({"glass"})
    assert onto.by_surface_form("glasses") == frozenset({"glasses"})


def test_an_ampersand_variant_moves_the_scan_bound() -> None:
    without = load_ontology(grocery())
    varied = load_ontology(grocery(DEFAULT_SPECS))
    assert without.longest_form_tokens() == 3
    assert varied.longest_form_tokens() == 4  # "fruit and veg (fresh)"


def test_a_rule_reads_the_entitys_metadata() -> None:
    onto = load_ontology(grocery([{"rule": "strip_prefix", "field": "brand"}]))
    assert onto.by_surface_form("peanut butter") == frozenset({"acme-pb"})


async def test_the_async_door_admits_the_same_variants() -> None:
    sync = load_ontology(grocery(DEFAULT_SPECS))
    onto = await async_load_ontology(grocery(DEFAULT_SPECS))
    assert await onto.by_surface_form("apples") == frozenset({"apple"})
    assert onto.variants == sync.variants


def test_the_door_uses_its_own_fold() -> None:
    """A collision is counted under the fold the source is built with."""

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
    onto = load_ontology(document, normalizer=squash)
    assert onto.variants is not None
    assert {each.variant.form for each in onto.variants.refused} == {"Cup Cakes", "Cupcake"}
    # Under the default fold the two never meet.
    assert load_ontology(document).variants.refused == ()


def test_the_parts_keep_the_authored_entities() -> None:
    """Variants are applied at bind time, so a diff of two loads' parts sees authored forms."""
    parts = build_ontology(OntologyConfig.from_dict(grocery(DEFAULT_SPECS)))
    assert parts.declared_entities["apple"].aliases == ()
    assert parts.variant_rules is not None


def test_an_unknown_rule_is_refused_by_name_when_the_document_is_built() -> None:
    with pytest.raises(ValidationError, match="no_such_rule"):
        load_ontology(grocery(["inflect", "no_such_rule"]))


def test_the_load_logs_the_counts_and_each_refusal(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="dataknobs_common.ontology.variants"):
        load_ontology(grocery(DEFAULT_SPECS))
    info = [record for record in caplog.records if record.levelno == logging.INFO]
    debug = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert len(info) == 1
    assert info[0].variants_refused == 2
    assert len(debug) == 2
    assert "'Glasses'" in debug[0].getMessage() or "'Glasses'" in debug[1].getMessage()


def test_with_variants_leaves_its_input_alone() -> None:
    entities = {
        "apple": Entity(id="apple", type="item", name="Apple"),
        "acme-jam": Entity(id="acme-jam", type="item", name="Acme Jam", metadata={"brand": "Acme"}),
    }
    varied, report = with_variants(entities, (strip_prefix("brand"), inflect))
    assert entities["apple"].aliases == ()
    assert varied["apple"].aliases == ("Apples",)
    assert varied["acme-jam"].aliases == ("Jam", "Acme Jams", "Jams")
    source = MappingEntitySource(varied)
    assert source.by_surface_form("jams") == frozenset({"acme-jam"})
    assert report.refused == ()


def test_an_entity_id_is_held_but_not_varied() -> None:
    entities = {
        "cup": Entity(id="cup", type="item", name="Cup"),
        "cups": Entity(id="cups", type="item", name="Mug"),
    }
    varied, report = with_variants(entities, (inflect,))
    assert [each.variant.form for each in report.refused] == ["Cups"]
    assert varied["cups"].aliases == ("Mugs",)

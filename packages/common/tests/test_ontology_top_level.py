"""The document's own top level is read for its shape, as every row is.

Every row of an ontology document is a checked read. The document itself is
read through :meth:`OntologyConfig.from_dict`, and that read used to fail in
the constructor's own words (a ``TypeError`` past both doors' documented
``ValidationError``) or not at all: a misspelt section loaded with nothing
in it, and an empty id loaded although the registry reserves it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigFieldError, ValidationError
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology.config import OntologyConfig


def _sync_door(document: Mapping[str, Any]) -> Any:
    return load_ontology(document)


def _async_door(document: Mapping[str, Any]) -> Any:
    return asyncio.run(async_load_ontology(document))


Door = Callable[[Mapping[str, Any]], Any]

DOORS = pytest.mark.parametrize(
    "door", [_sync_door, _async_door], ids=["load_ontology", "async_load_ontology"]
)


@DOORS
def test_an_empty_document_is_refused_for_its_missing_id(door: Door) -> None:
    with pytest.raises(ValidationError) as excinfo:
        door({})
    assert isinstance(excinfo.value, ConfigFieldError)
    assert excinfo.value.context["missing"] == ["id"]


@DOORS
def test_an_id_written_as_null_is_refused_as_missing(door: Door) -> None:
    with pytest.raises(ValidationError, match="`id:`") as excinfo:
        door({"id": None})
    assert "missing" in str(excinfo.value)


@DOORS
def test_a_numeric_id_is_refused_with_a_hint_to_quote_it(door: Door) -> None:
    with pytest.raises(ValidationError, match="quote it") as excinfo:
        door({"id": 3})
    assert excinfo.value.context == {"section": "id", "value": 3}


@DOORS
def test_an_empty_id_is_refused_since_the_registry_reserves_it(door: Door) -> None:
    with pytest.raises(ValidationError, match="`id:` is the empty string") as excinfo:
        door({"id": ""})
    assert excinfo.value.context == {"section": "id", "value": ""}


@DOORS
def test_a_numeric_version_is_refused_with_a_hint_to_quote_it(door: Door) -> None:
    """YAML reads ``version: 1.10`` as ``1.1``: the author's text is already gone."""
    with pytest.raises(ValidationError, match="quote it") as excinfo:
        door({"id": "o", "version": 1.1})
    assert excinfo.value.context == {"section": "version", "value": 1.1}


def test_a_version_written_as_null_reads_as_absent() -> None:
    """``null`` is silence loader-wide, so the field's default stands."""
    assert load_ontology({"id": "o", "version": None}).version == (
        load_ontology({"id": "o"}).version
    )
    assert load_ontology({"id": "o"}).version == "1.0"


@DOORS
def test_a_misspelt_section_is_refused_naming_the_nearest_one(door: Door) -> None:
    with pytest.raises(ValidationError, match="did you mean 'entities'"):
        door({"id": "o", "entites": [{"id": "rex"}]})


@DOORS
def test_an_overlay_section_is_refused_as_a_key_nothing_reads(door: Door) -> None:
    """The overlay is a component, not a config key, and v1 builds none."""
    with pytest.raises(ValidationError, match="'overlay'"):
        door({"id": "o", "overlay": {}})


@DOORS
def test_a_well_formed_top_level_still_loads(door: Door) -> None:
    ontology = door({"id": "o", "version": "1.10"})
    assert ontology.id == "o"
    assert ontology.version == "1.10"


def test_the_config_declares_a_closed_key_set() -> None:
    assert OntologyConfig._UNKNOWN_KEYS == "raise"
    assert not OntologyConfig.accepts("overlay")


def test_the_wrapped_form_is_still_unwrapped_before_the_key_check() -> None:
    """``ontology:`` wraps the document in a file that carries other blocks."""
    load_ontology({"ontology": {"id": "o"}})


@DOORS
@pytest.mark.parametrize("value", [["o"], {"o": 1}, True], ids=["list", "mapping", "bool"])
def test_a_non_scalar_id_is_refused_without_a_hint_to_quote_it(door: Door, value: Any) -> None:
    """Quoting is the remedy for a number only; a list quoted is still not an id."""
    with pytest.raises(ValidationError, match="not a string") as excinfo:
        door({"id": value})
    assert "quote it" not in str(excinfo.value)


def test_a_float_version_is_not_rendered_back_into_the_hint() -> None:
    """``version: 1.10`` arrives as ``1.1``; suggesting ``'1.1'`` would mislead."""
    with pytest.raises(ValidationError, match="quote it as written") as excinfo:
        load_ontology({"id": "o", "version": 1.1})
    assert "`version: '1.1'`" not in str(excinfo.value)

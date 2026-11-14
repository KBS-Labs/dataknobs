"""The registry's doors refuse a malformed top level as the loader's do.

Three doors reach :class:`OntologyConfig` here: the constructor, through the
mixin's coercion; :meth:`OntologyRegistry.from_config_async`, which loads what
it was constructed from; and :meth:`OntologyRegistry.load`. Each documents
``ValidationError`` for a document it refuses, and each used to let a missing
``id`` out as the constructor's ``TypeError``.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigFieldError, ValidationError
from dataknobs_data.ontology import OntologyRegistry


def test_the_constructor_refuses_a_document_with_no_id() -> None:
    with pytest.raises(ValidationError) as excinfo:
        OntologyRegistry({})
    assert isinstance(excinfo.value, ConfigFieldError)
    assert excinfo.value.context["missing"] == ["id"]


async def test_load_refuses_a_document_with_no_id() -> None:
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load({"entity_types": [{"id": "Pet"}]})
        assert excinfo.value.context["missing"] == ["id"]
    finally:
        await registry.close()


async def test_load_refuses_a_numeric_id_with_a_hint_to_quote_it() -> None:
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError, match="quote it"):
            await registry.load({"id": 3})
    finally:
        await registry.close()


async def test_load_refuses_a_document_claiming_the_empty_id() -> None:
    """The empty id is the registry's own spelling of *nothing configured*.

    That spelling is sound only while no loaded document can claim it, so
    :meth:`~OntologyRegistry.load` refuses one that tries.
    """
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError, match="`id:` is the empty string"):
            await registry.load({"id": "", "entity_types": [{"id": "Pet"}]})
    finally:
        await registry.close()


@pytest.mark.parametrize("block", [5, "memory", ["memory"]])
async def test_an_event_bus_block_that_is_not_a_mapping_is_refused_by_name(block: Any) -> None:
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError, match="`event_bus:`") as excinfo:
            await registry.load({"id": "o", "event_bus": block})
        assert excinfo.value.context["section"] == "event_bus"
        assert registry._event_bus is None, "nothing is built for a refused block"
    finally:
        await registry.close()


async def test_load_refuses_a_misspelt_top_level_section() -> None:
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError, match="did you mean 'event_bus'"):
            await registry.load({"id": "o", "event_buss": {"backend": "memory"}})
    finally:
        await registry.close()


_MEMORY_BUS = {"backend": "memory"}


async def _refused_then_holds_no_bus(registry: OntologyRegistry, document: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        await registry.load(document)
    assert registry._event_bus is None, "a refused document leaves no bus behind"
    assert registry._owns_event_bus is False


@pytest.mark.parametrize(
    "document",
    [
        {"id": "", "event_bus": _MEMORY_BUS},
        {"id": None, "event_bus": _MEMORY_BUS},
        {"id": "o", "index": 5, "event_bus": _MEMORY_BUS},
        {"id": "o", "resolver": 5, "event_bus": _MEMORY_BUS},
    ],
    ids=["empty-id", "null-id", "index-not-a-mapping", "resolver-not-a-mapping"],
)
async def test_a_document_refused_after_its_bus_block_leaves_no_bus(
    document: dict[str, Any],
) -> None:
    registry = OntologyRegistry()
    try:
        await _refused_then_holds_no_bus(registry, document)
        # The next document's own block is the one that takes effect.
        await registry.load({"id": "next", "event_bus": _MEMORY_BUS})
        assert registry._event_bus is not None
        assert registry._owns_event_bus is True
    finally:
        await registry.close()


async def test_a_duplicate_id_refused_leaves_no_bus() -> None:
    registry = OntologyRegistry()
    try:
        await registry.load({"id": "o"})
        await _refused_then_holds_no_bus(registry, {"id": "o", "event_bus": _MEMORY_BUS})
    finally:
        await registry.close()


async def test_an_event_bus_naming_no_registered_backend_is_refused_by_name() -> None:
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError, match="`event_bus:`") as excinfo:
            await registry.load({"id": "o", "event_bus": {"backend": "nope"}})
        assert excinfo.value.context["section"] == "event_bus"
        assert registry._event_bus is None
    finally:
        await registry.close()


@pytest.mark.parametrize("falsy_id", [None, "", 0, False], ids=["null", "empty", "zero", "false"])
async def test_the_constructor_door_refuses_a_document_with_a_falsy_id(falsy_id: Any) -> None:
    """The empty registry is spelled by passing no document, not a falsy id."""
    with pytest.raises(ValidationError):
        registry = await OntologyRegistry.from_config_async(
            {"id": falsy_id, "entity_types": [{"id": "Pet"}]}
        )
        await registry.close()


async def test_an_empty_config_is_still_the_empty_registry() -> None:
    """``OntologyConfig(id="")`` with nothing else in it configures nothing."""
    from dataknobs_common.ontology.config import OntologyConfig

    registry = await OntologyRegistry.from_config_async(OntologyConfig(id=""))
    try:
        with pytest.raises(ValidationError, match="constructed with none"):
            await registry.load()
    finally:
        await registry.close()

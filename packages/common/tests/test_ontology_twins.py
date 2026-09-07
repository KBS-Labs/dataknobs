"""The two doors load the same vocabulary, and both go through one core.

Two different claims. The first is about *results* -- the same file yields the
same four fields either way. The second is about *mechanism*, and it needs a
different kind of test: equality of results cannot distinguish "calls the core"
from "happens to agree with the core today", and only the second survives
someone editing one door.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.ontology import (
    AsyncOntology,
    OntologyParts,
    async_load_ontology,
    load_ontology,
)
from dataknobs_common.ontology import loader as loader_module
from dataknobs_common.ontology.sources import (
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
)


def test_both_doors_yield_the_same_four_fields(mammals_path: Path) -> None:
    """Asserted on the fields, not read off the two implementations."""
    sync = load_ontology(mammals_path)
    async_ = asyncio.run(async_load_ontology(mammals_path))

    assert isinstance(async_, AsyncOntology)
    assert async_.id == sync.id
    assert async_.version == sync.version
    assert async_.entity_types == sync.entity_types
    assert async_.relation_types == sync.relation_types


def test_the_async_door_binds_the_async_flavour(mammals_path: Path) -> None:
    """Same fields, different sources -- which is the whole difference."""
    onto = asyncio.run(async_load_ontology(mammals_path))

    assert type(onto.entities) is AsyncMappingEntitySource
    assert type(onto.assertions) is AsyncMappingAssertionSource


@pytest.mark.asyncio
async def test_the_async_ontology_reads_through_its_sources(
    mammals_path: Path,
) -> None:
    """The async flavour answers the same questions, awaited."""
    onto = await async_load_ontology(mammals_path)

    assert await onto.entities.by_surface_form("beagles") == frozenset({"beagle"})
    beagle = await onto.entities.get("beagle")
    assert beagle is not None
    assert beagle.name == "Beagle"
    assert await onto.entities.by_type("Breed") == frozenset({"beagle"})


def test_both_doors_go_through_build_ontology(
    mammals_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch the core; both doors must observe it.

    A delegation test rather than an equality one. If a door ever grew its own
    copy of the mapping logic, every result assertion above would still pass
    on the day the copy was written -- and would keep passing until the two
    copies drifted, which is exactly when nobody is looking.
    """
    calls: list[str] = []
    real_build = loader_module.build_ontology

    def counting_build(config: Any) -> OntologyParts:
        calls.append(config.id)
        return real_build(config)

    monkeypatch.setattr(loader_module, "build_ontology", counting_build)

    load_ontology(mammals_path)
    assert calls == ["mammals"]

    asyncio.run(async_load_ontology(mammals_path))
    assert calls == ["mammals", "mammals"]


def test_a_patched_core_changes_what_both_doors_return(
    mammals_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stronger half: the patch must reach the *result*, not just be called.

    Counting calls proves the core was reached. It does not prove the door
    used what came back -- a door that called the core and then rebuilt the
    parts itself would pass the counting test.
    """
    real_build = loader_module.build_ontology

    def renaming_build(config: Any) -> OntologyParts:
        parts = real_build(config)
        return OntologyParts(
            id="patched",
            version=parts.version,
            entity_types=parts.entity_types,
            relation_types=parts.relation_types,
            taxonomies=parts.taxonomies,
            declared_entities=parts.declared_entities,
            declared_assertions=parts.declared_assertions,
            source_specs=parts.source_specs,
        )

    monkeypatch.setattr(loader_module, "build_ontology", renaming_build)

    assert load_ontology(mammals_path).id == "patched"
    assert asyncio.run(async_load_ontology(mammals_path)).id == "patched"

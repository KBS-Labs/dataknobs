"""An accessor invokes the one way rather than agreeing with it.

The distinction is invisible to a results test and is the whole point of the
rule: a convenience method that reimplements the lookup passes every equality
assertion on the day it is written, and diverges silently the first time the
source's behaviour changes underneath it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_common.ontology import Entity, async_load_ontology, load_ontology


class _RecordingEntitySource:
    """A source that reports what was asked of it.

    Written rather than mocked so it satisfies the protocol for real: the
    accessors under test call it exactly as they would call the shipped
    concrete, and a signature that stopped matching would fail here.
    """

    def __init__(self, entities: dict[str, Entity]) -> None:
        self.entities = entities
        self.calls: list[tuple[str, str]] = []

    def get(self, entity_id: str) -> Entity | None:
        self.calls.append(("get", entity_id))
        return self.entities.get(entity_id)

    def get_many(self, entity_ids: list[str]) -> dict[str, Entity]:
        self.calls.append(("get_many", ",".join(entity_ids)))
        return {i: self.entities[i] for i in entity_ids if i in self.entities}

    def fetch_origin(self, ref: object) -> None:
        return None

    def fetch_origins(self, refs: object) -> dict[object, object]:
        return {}

    def describe(self) -> object:
        raise NotImplementedError

    def by_surface_form(self, form: str) -> frozenset[str]:
        self.calls.append(("by_surface_form", form))
        return frozenset({"sentinel"})

    def by_type(self, type_id: str) -> frozenset[str]:
        self.calls.append(("by_type", type_id))
        return frozenset()


def test_entity_invokes_the_sources_get(mammals_path: Path) -> None:
    """Replacing the source changes what the accessor answers."""
    onto = load_ontology(mammals_path)
    recording = _RecordingEntitySource({"dog": Entity(id="dog", type="Species")})
    swapped = type(onto)(
        id=onto.id,
        version=onto.version,
        entity_types=onto.entity_types,
        relation_types=onto.relation_types,
        entities=recording,
        assertions=onto.assertions,
        taxonomies=onto.taxonomies,
        describes=onto.describes,
    )

    result = swapped.entity("dog")

    assert recording.calls == [("get", "dog")]
    assert result is recording.entities["dog"]


def test_by_surface_form_invokes_the_sources_member(mammals_path: Path) -> None:
    """The sentinel is what proves invocation.

    ``frozenset({"sentinel"})`` is an answer the real source could never give
    for this file, so the accessor returning it can only mean it asked.
    """
    onto = load_ontology(mammals_path)
    recording = _RecordingEntitySource({})
    swapped = type(onto)(
        id=onto.id,
        version=onto.version,
        entity_types=onto.entity_types,
        relation_types=onto.relation_types,
        entities=recording,
        assertions=onto.assertions,
        taxonomies=onto.taxonomies,
        describes=onto.describes,
    )

    assert swapped.by_surface_form("beagles") == frozenset({"sentinel"})
    assert recording.calls == [("by_surface_form", "beagles")]


def test_the_accessor_and_the_source_agree_on_the_real_file(
    mammals_path: Path,
) -> None:
    """The ordinary case, asserted after the mechanism rather than instead."""
    onto = load_ontology(mammals_path)

    assert onto.entity("beagle") is onto.entities.get("beagle")
    assert onto.by_surface_form("beagles") == onto.entities.by_surface_form("beagles")


@pytest.mark.asyncio
async def test_the_async_accessors_delegate_too(mammals_path: Path) -> None:
    """The twin has the same two accessors over the same two members."""
    onto = await async_load_ontology(mammals_path)

    assert await onto.entity("beagle") == await onto.entities.get("beagle")
    assert await onto.by_surface_form("beagles") == frozenset({"beagle"})

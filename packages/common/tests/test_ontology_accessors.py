"""An accessor invokes the one way rather than agreeing with it.

The distinction is invisible to a results test and is the whole point of the
rule: a convenience method that reimplements the lookup passes every equality
assertion on the day it is written, and diverges silently the first time the
source's behaviour changes underneath it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dataknobs_common.ontology import (
    Entity,
    MappingEntitySource,
    async_load_ontology,
    load_ontology,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataknobs_common.ontology import SourceDescription, SourceRef
    from dataknobs_common.records import Record


class _RecordingEntitySource:
    """A source that records what was asked of it and delegates the answer.

    A spy over the shipped :class:`MappingEntitySource` rather than beside it.
    The rule this file is about applies to the double as well: a second lookup
    written here would agree with the concrete on the day it was written and
    drift the first time the concrete's behaviour changed -- and it had already
    drifted, taking ``list[str]`` where the protocol says ``Sequence[str]`` and
    skipping the normalizer every real answer goes through.

    :meth:`by_surface_form` is the one member that does **not** delegate, and
    the exception is the point: its sentinel is an answer no real source could
    give for this file, which is what separates *the accessor asked* from *the
    accessor agreed*. A delegated empty answer would prove only the weaker one.
    """

    def __init__(self, entities: dict[str, Entity]) -> None:
        self.entities = entities
        self.calls: list[tuple[str, str]] = []
        self._inner = MappingEntitySource(entities)

    def get(self, entity_id: str) -> Entity | None:
        self.calls.append(("get", entity_id))
        return self._inner.get(entity_id)

    def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        self.calls.append(("get_many", ",".join(entity_ids)))
        return self._inner.get_many(entity_ids)

    def fetch_origin(self, ref: SourceRef) -> Record | None:
        return self._inner.fetch_origin(ref)

    def fetch_origins(self, refs: Sequence[SourceRef]) -> dict[SourceRef, Record]:
        return self._inner.fetch_origins(refs)

    def describe(self) -> SourceDescription:
        return self._inner.describe()

    def by_surface_form(self, form: str) -> frozenset[str]:
        self.calls.append(("by_surface_form", form))
        return frozenset({"sentinel"})

    def by_type(self, type_id: str) -> frozenset[str]:
        self.calls.append(("by_type", type_id))
        return self._inner.by_type(type_id)


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

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
    AsyncEntitySource,
    AsyncMappingEntitySource,
    AsyncOntology,
    Entity,
    EntitySource,
    MappingEntitySource,
    Ontology,
    async_load_ontology,
    load_ontology,
)
from dataknobs_common.testing import assert_twin_types_agree

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

    def fetch_origins(self, refs: Sequence[SourceRef]) -> list[Record | None]:
        return self._inner.fetch_origins(refs)

    def describe(self) -> SourceDescription:
        return self._inner.describe()

    def by_surface_form(self, form: str) -> frozenset[str]:
        self.calls.append(("by_surface_form", form))
        return frozenset({"sentinel"})

    def by_alias_form(self, form: str) -> frozenset[str]:
        self.calls.append(("by_alias_form", form))
        return self._inner.by_alias_form(form)

    def by_type(self, type_id: str) -> frozenset[str]:
        self.calls.append(("by_type", type_id))
        return self._inner.by_type(type_id)

    def longest_form_tokens(self) -> int | None:
        self.calls.append(("longest_form_tokens", ""))
        # A sentinel for the same reason ``by_surface_form`` answers one: no
        # real vocabulary in this file declares a 99-token form, so an
        # accessor returning it can only mean it asked.
        return 99


def test_the_double_conforms_to_the_protocol_it_stands_in_for() -> None:
    """The guard whose absence is why widening the protocol broke this quietly.

    A member added to :class:`EntitySource` turns every structural conformer
    from conforming into non-conforming at once, and a double that implements
    the members it happens to need reports nothing: nothing in this file calls
    ``isinstance``, so the suite stayed green while the double had stopped
    satisfying the protocol it exists to stand in for. That is the same break
    a consumer's own source would take, discovered here or discovered by them.

    :class:`~dataknobs_common.ontology.MappingEntitySource` carries this
    assertion already; the double had none, which is exactly the gap.
    """
    from dataknobs_common.ontology import AliasFormSource, EntitySource

    double = _RecordingEntitySource({"dog": Entity(id="dog", type="Species")})

    assert isinstance(double, EntitySource)
    assert isinstance(double, AliasFormSource)


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
        codec=onto.codec,
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
        codec=onto.codec,
    )

    assert swapped.by_surface_form("beagles") == frozenset({"sentinel"})
    assert recording.calls == [("by_surface_form", "beagles")]


def test_longest_form_tokens_invokes_the_sources_member(mammals_path: Path) -> None:
    """The third accessor, held to the second one's standard.

    It shipped with neither a caller nor an assertion: the scan reaches the
    *source* directly, so an ``Ontology`` that answered this from nowhere
    would have passed everything in the tree. That is the shape the accessor
    exists to prevent -- a caller holding the loaded value rather than its
    entities can ask what a window may span, and has to get the vocabulary's
    answer rather than a plausible one.
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
        codec=onto.codec,
    )

    assert swapped.longest_form_tokens() == 99
    assert recording.calls == [("longest_form_tokens", "")]


def test_the_accessor_and_the_source_agree_on_the_real_file(
    mammals_path: Path,
) -> None:
    """The ordinary case, asserted after the mechanism rather than instead."""
    onto = load_ontology(mammals_path)

    assert onto.entity("beagle") is onto.entities.get("beagle")
    assert onto.by_surface_form("beagles") == onto.entities.by_surface_form("beagles")
    assert onto.longest_form_tokens() == onto.entities.longest_form_tokens()


@pytest.mark.asyncio
async def test_the_async_accessors_delegate_too(mammals_path: Path) -> None:
    """The twin has the same three accessors over the same three members.

    ``longest_form_tokens`` is a plain ``def`` on both flavours, so it is the
    one asserted here without an ``await`` -- which is itself the claim.
    """
    onto = await async_load_ontology(mammals_path)

    assert await onto.entity("beagle") == await onto.entities.get("beagle")
    assert await onto.by_surface_form("beagles") == frozenset({"beagle"})
    assert onto.longest_form_tokens() == onto.entities.longest_form_tokens()


#: The members every one of the three pairs owes, and which of them stay
#: synchronous on both halves. ``describe`` and ``longest_form_tokens`` answer
#: from what the object already holds, which is the reason neither is
#: awaitable and the reason both are declared here rather than omitted.
_SOURCE_MEMBERS = [
    "get",
    "get_many",
    "fetch_origin",
    "fetch_origins",
    "describe",
    "by_surface_form",
    "by_type",
    "longest_form_tokens",
]
_SOURCE_UNFLAVOURED = ("describe", "longest_form_tokens")


@pytest.mark.parametrize(
    ("sync_type", "async_type", "members", "unflavoured"),
    [
        (EntitySource, AsyncEntitySource, _SOURCE_MEMBERS, _SOURCE_UNFLAVOURED),
        (
            MappingEntitySource,
            AsyncMappingEntitySource,
            _SOURCE_MEMBERS,
            _SOURCE_UNFLAVOURED,
        ),
        (
            Ontology,
            AsyncOntology,
            ["entity", "by_surface_form", "longest_form_tokens"],
            ("longest_form_tokens",),
        ),
    ],
    ids=["protocols", "mapping-sources", "ontology-values"],
)
def test_the_vocabulary_twins_expose_one_surface(
    sync_type: type, async_type: type, members: list[str], unflavoured: tuple[str, ...]
) -> None:
    """Parity over the three pairs a vocabulary is reached through.

    ``dataknobs_common.entity_resolution`` has held its rungs to this since
    they were twinned; the source protocols, the mapping sources and the
    loaded values had nothing equivalent, and the three were kept in step by
    hand. That worked until a member was added to all three pairs at once --
    which is precisely when hand-maintenance is least reliable and when a miss
    is hardest to see, because each half is individually correct and complete.

    ``longest_form_tokens`` is the member that prompted this and is not the
    reason it should exist: the next one will arrive the same way, and the
    difference between a pair that agrees and a pair that happens to agree is
    a test.

    Returns are compared for the protocol pair and the concrete sources, where
    every member returns the same type on both halves. ``Ontology`` is
    narrowed to the three accessors for the reason it is narrowed: ``taxonomy``
    answers a ``Taxonomy`` against an ``AsyncTaxonomy``, which is a flavoured
    return rather than drift, and it belongs with whichever guard covers that
    pair.
    """
    assert_twin_types_agree(
        sync_type,
        async_type,
        members,
        unflavoured_members=unflavoured,
        compare_return=True,
    )

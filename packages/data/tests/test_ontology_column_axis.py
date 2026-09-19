"""A taxonomy whose edges are a parent column, and what binding one owes.

Four of these are the acceptance surface for the configuration contract --
the id round trip both ways, the column axis walking, that walk agreeing with
the same tree read as assertions, and ``at()`` being the door. The rest are
what those four leave uncovered: a walk that is *correct* says nothing about
how many queries it cost, a snapshot taken by descending from the roots is
correct on every tree without a cycle in it, and the refusal a module-level
door now makes is reached by no criterion at all, because every criterion in
that suite goes through the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.events import Event, InMemoryEventBus
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.hierarchy import AsyncHierarchyView, AsyncMappingHierarchy
from dataknobs_common.ontology import (
    AsyncAssertionHierarchy,
    OntologyConfig,
    async_load_ontology,
)
from dataknobs_common.records import Record
from dataknobs_common.testing import assert_no_blocking

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.ontology.hierarchy import ColumnAxisBinding, ColumnHierarchy
from dataknobs_data.query import Query

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

#: The tree the two backings are compared over, and the fourth row that makes
#: the comparison say something. `root -> mid -> leaf` is a chain both backings
#: agree on by construction; `orphan` is the row they can disagree about,
#: because it is a row of the table and an edge of nothing.
TREE = (
    ("root", "Root", None),
    ("mid", "Mid", "root"),
    ("leaf", "Leaf", "mid"),
    ("orphan", "Orphan", None),
)


def _rows(tree: Sequence[tuple[str, str, str | None]] = TREE) -> list[Record]:
    """The tree as catalogue rows, the parent column absent where there is none."""
    return [
        Record({"sku": sku, "title": title} | ({} if parent is None else {"parent_sku": parent}))
        for sku, title, parent in tree
    ]


async def _store(tree: Sequence[tuple[str, str, str | None]] = TREE) -> AsyncMemoryDatabase:
    db = AsyncMemoryDatabase()
    for record in _rows(tree):
        await db.create(record)
    return db


def _document(**overrides: Any) -> dict[str, Any]:
    """A `kind: record` vocabulary with a `kind: column` axis over its table."""
    document: dict[str, Any] = {
        "id": "catalog",
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
                    {"name": "parent_sku", "type": "string"},
                ],
            }
        ],
        "taxonomies": [
            {
                "id": "categories",
                "kind": "column",
                "source": "products",
                "parent_key": "parent_sku",
                "relation": "parent",
            }
        ],
    }
    document.update(overrides)
    return document


def _as_assertions() -> dict[str, Any]:
    """The same tree authored, which is the other backing the walk is compared to.

    An `entities:` row per node and a `parent` assertion per edge -- and no
    source at all, because a document declaring both an authored vocabulary
    and a live one is refused before either is bound.
    """
    return {
        "id": "authored",
        "entity_types": [{"id": "Product"}],
        "entities": [{"id": sku, "type": "Product", "name": title} for sku, title, _ in TREE],
        "assertions": [
            {"subject": sku, "relation": "parent", "object": parent}
            for sku, _, parent in TREE
            if parent is not None
        ],
        "sources": [],
        "taxonomies": [{"id": "categories", "relation": "parent"}],
    }


# --------------------------------------------------------------------------
# The id round trip, both directions
# --------------------------------------------------------------------------


async def test_the_id_round_trip_both_ways_and_the_wrong_space_answers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`qualify` out, `localize` back, and the qualified id answering None.

    Both clauses, because the second is what makes the first necessary. Over
    `StrCodec` every key is a `str`, so the qualified id type-checks where a
    local one is wanted and comes back `None` rather than raising -- which a
    test asserting only the round trip would pass against a vocabulary that
    had stopped holding the entity at all.
    """
    del caplog
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _store()
    )
    try:
        onto = await registry.load()
        qualified = onto.qualify("leaf")
        assert qualified == "catalog:leaf"
        assert onto.localize(qualified) == "leaf"

        found = await onto.entity(onto.localize(qualified))
        assert found is not None and found.name == "Leaf"
        assert await onto.entity(qualified) is None
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The column axis walks, and it walks the way assertions over it walk
# --------------------------------------------------------------------------


async def test_the_column_axis_walks_the_way_the_same_tree_as_assertions_does() -> None:
    """One tree, two backings, every structural member compared.

    The two agree on a chain by construction, so a vehicle of three chained
    rows would satisfy the criterion's letter while proving nothing. What they
    can disagree about is `orphan` -- a row of the table that no edge mentions
    -- and the protocol decides that case rather than the fixture:
    *"A node absent from every edge of this axis is not in this hierarchy at
    all."* So the last clause is the criterion: both backings answer `False`.

    And the delegation is asserted rather than assumed. The two taxonomies are
    built from one definition through one accessor and differ in `structure`
    alone, so a walk that disagreed would mean the hierarchy family had two
    implementations of one walk rather than two backings under it.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _store()
    )
    try:
        column = (await registry.load()).taxonomy("categories")
        authored = (await async_load_ontology(_as_assertions())).taxonomy("categories")

        assert type(column.structure) is ColumnHierarchy
        assert type(authored.structure) is AsyncAssertionHierarchy
        assert column.definition.relation == authored.definition.relation

        assert sorted(await column.structure.roots()) == sorted(await authored.structure.roots())
        for node, _, _ in TREE:
            assert sorted(await column.structure.parents(node)) == sorted(
                await authored.structure.parents(node)
            )
            assert sorted(await column.structure.children(node)) == sorted(
                await authored.structure.children(node)
            )
            assert await column.structure.contains(node) is await authored.structure.contains(node)
            assert [view.node for view in await column.at(node).ancestors()] == [
                view.node for view in await authored.at(node).ancestors()
            ]

        # The clause that fails if the column axis takes its nodes from the
        # table rather than from its edges.
        assert await column.structure.contains("orphan") is False
        assert await authored.structure.contains("orphan") is False
        assert "orphan" not in await column.structure.roots()
    finally:
        await registry.close()


async def test_a_materialized_column_axis_is_a_snapshot_of_the_column_axis() -> None:
    """The one arrangement the two mechanisms have to compose in.

    `structures` means *the axes a door bound* and `axes_to_copy` still decides
    *which of them a door must snapshot first*, so a `kind: column` axis
    declaring `materialization.structure: materialized` is a copy of the rows
    -- not of an assertion read over edges the document never declared, which
    is what a snapshot taken without consulting what was bound would be.

    Asserted by what the copy holds rather than by its type alone: a snapshot
    of the wrong axis is also an `AsyncMappingHierarchy`, and it is empty.
    """
    materialized = _document(
        taxonomies=[
            {
                "id": "categories",
                "kind": "column",
                "source": "products",
                "parent_key": "parent_sku",
                "relation": "parent",
                "materialization": {"structure": "materialized"},
            }
        ]
    )
    store = await _store()
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**materialized), database=store
    )
    try:
        tax = (await registry.load()).taxonomy("categories")
        assert type(tax.structure) is AsyncMappingHierarchy
        assert sorted(await tax.structure.parent_edges()) == ["leaf", "mid", "root"]
        assert [view.node for view in await tax.at("leaf").ancestors()] == ["mid", "root"]

        # ...and it is a copy: the rows move underneath it and it does not.
        await store.create(Record({"sku": "extra", "title": "Extra", "parent_sku": "root"}))
        assert sorted(await tax.structure.parent_edges()) == ["leaf", "mid", "root"]
    finally:
        await registry.close()


async def test_at_is_the_door_and_an_anchor_the_axis_does_not_hold_answers_empty() -> None:
    """`ancestors` is the cursor's, and the cursor is silent about containment.

    The negative half is the point. `await tax.ancestors(...)` is what a
    caller reaches for first and is not a member of a taxonomy at all; and a
    test asserting only the chain would pass against an axis that had silently
    stopped holding the node, because `at()` does not check containment --
    *"that is the whole precision of it"*.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _store()
    )
    try:
        tax = (await registry.load()).taxonomy("categories")
        assert not hasattr(tax, "ancestors")

        placed = tax.at("leaf")
        assert await placed.exists() is True
        assert [view.node for view in await placed.ancestors()] == ["mid", "root"]

        for absent in ("orphan", "no-such-sku"):
            cursor = tax.at(absent)
            assert await cursor.exists() is False
            assert await cursor.ancestors() == ()
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The refusal a module-level door makes, and the door that answers it
# --------------------------------------------------------------------------


async def test_the_door_that_binds_no_axis_refuses_it_and_names_the_one_that_does() -> None:
    """One test, both halves: the refusal, and the remedy it points at.

    The shape the shipped live-source refusal is already tested in, applied to
    the second thing a module-level door cannot own -- one test holding the
    refusal and the remedy it names, because that is the pairing a message
    pointing at a class can go wrong in.

    Either half alone is green against the wrong thing -- a test
    asserting only the refusal would pass while the row was being dropped in
    silence if the message were changed, and a test asserting only the load
    would not notice that the other door had gone quiet.

    A row whose `kind:` is dropped is the defect: `TaxonomyDefinition` reads
    six keys and none of them is `kind:`, `source:` or `parent_key:`, so such
    a row used to load as an assertion axis over assertions the document never
    declared and answer empty for every walk.
    """
    document = _document(sources=[])

    with pytest.raises(ValidationError) as excinfo:
        await async_load_ontology(document)
    message = str(excinfo.value)
    assert "OntologyRegistry" in message
    assert "'categories'" in message and "'column'" in message

    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _store()
    )
    try:
        walked = (await registry.load()).taxonomy("categories")
        assert [view.node for view in await walked.at("leaf").ancestors()] == ["mid", "root"]
    finally:
        await registry.close()


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"id": "categories", "kind": "adjacency", "source": "products"}, "adjacency"),
        (
            {"id": "categories", "kind": "column", "source": "absent", "parent_key": "parent_sku"},
            "'absent'",
        ),
        ({"id": "categories", "kind": "column", "source": "products"}, "parent_key"),
        (
            {"id": "categories", "kind": "column", "source": "products", "parent_key": "nope"},
            "'nope'",
        ),
    ],
    ids=["unbound-kind", "undeclared-source", "no-parent-key", "undeclared-column"],
)
async def test_the_registry_refuses_an_axis_it_cannot_bind(
    row: dict[str, Any], expected: str
) -> None:
    """Four ways an axis row is wrong, and each refusal naming its own subject.

    The kind is refused by the registry rather than by the door above it,
    because *no binder is registered for this kind* is the one of these that
    stops being true when a binder is registered. The other three are about a
    row that named a backing this registry does have.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(taxonomies=[{**row, "relation": "parent"}])),
        database=await _store(),
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        assert expected in str(excinfo.value)
    finally:
        await registry.close()


async def test_a_column_axis_over_an_authored_source_is_refused_rather_than_downgraded() -> None:
    """An authored source has no table, so there is no column to read.

    Refused rather than quietly read as an assertion axis, which would be the
    silent substitution this whole binding exists to end arriving one step
    later and from the other direction.
    """
    document = _as_assertions()
    document["sources"] = [{"id": "tree", "kind": "inline"}]
    document["taxonomies"] = [
        {
            "id": "categories",
            "kind": "column",
            "source": "tree",
            "parent_key": "parent_sku",
            "relation": "parent",
        }
    ]
    registry = OntologyRegistry.from_components(config=OntologyConfig(**document))
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        message = str(excinfo.value)
        assert "'tree'" in message and "authored" in message
    finally:
        await registry.close()


def test_the_axis_binding_names_the_axis_on_a_refusal() -> None:
    """The parsed row, and what it says when a key is missing."""
    assert ColumnAxisBinding.from_mapping(
        {"source": "products", "parent_key": "parent_sku"}, binding="categories"
    ) == ColumnAxisBinding(source="products", parent_key="parent_sku")

    with pytest.raises(ValidationError, match="categories"):
        ColumnAxisBinding.from_mapping({"source": "products"}, binding="categories")


# --------------------------------------------------------------------------
# What a correct walk does not say: how many queries it cost
# --------------------------------------------------------------------------


class _CountingStore(AsyncMemoryDatabase):
    """An `AsyncMemoryDatabase` that says how many reads went through it.

    A real backend with a counter on its streaming door, rather than a fake:
    what is being measured is how many times the axis asks, and the answers it
    gets have to be the real ones or the walk under measurement is not the
    walk that ships.
    """

    reads: int = 0

    def stream_read(self, query: Query | None = None, config: Any = None) -> AsyncIterator[Record]:
        type(self).reads += 1
        return super().stream_read(query, config)


async def test_a_frontier_costs_one_query_per_level_rather_than_one_per_node() -> None:
    """The bulk members, measured -- correctness is not what they buy.

    A `ColumnHierarchy` carrying only the four singular members passes every
    criterion in this file, because a walk driver falls back to asking per
    node. What it does not do is ask once. The two protocols are opt-in by
    member presence and nothing registers, so an axis that lost them would go
    on answering correctly and quietly cost `N` round trips per level.
    """
    wide = [("root", "Root", None), *((f"n{i}", f"N{i}", "root") for i in range(8))]
    deep = [*wide, *((f"d{i}", f"D{i}", f"n{i}") for i in range(8))]
    store = _CountingStore()
    for record in _rows(deep):
        await store.create(record)
    axis = ColumnHierarchy(store, "products", "sku", "parent_sku")

    frontier = tuple(f"n{i}" for i in range(8))
    _CountingStore.reads = 0
    assert [list(answer) for answer in await axis.parents_many(frontier)] == [["root"]] * 8
    assert _CountingStore.reads == 1

    _CountingStore.reads = 0
    children = await axis.children_many(frontier)
    assert [list(answer) for answer in children] == [[f"d{i}"] for i in range(8)]
    assert _CountingStore.reads == 1

    # ...and the whole axis in one read, which is the member a snapshot takes.
    _CountingStore.reads = 0
    assert len(await axis.parent_edges()) == 17
    assert _CountingStore.reads == 1


class _SingularOnly:
    """The same axis with only the four members every hierarchy has.

    The control the count above needs, and a real object rather than a stub of
    one: it forwards to the axis under test, so the answers are that axis's and
    the only difference is what a driver can *find* on it. Without it the
    measurement has no scale -- `1` means nothing unless the number it is being
    compared against was taken the same way.
    """

    def __init__(self, axis: ColumnHierarchy) -> None:
        self._axis = axis

    async def roots(self) -> Sequence[str]:
        return await self._axis.roots()

    async def parents(self, node_id: str) -> Sequence[str]:
        return await self._axis.parents(node_id)

    async def children(self, node_id: str) -> Sequence[str]:
        return await self._axis.children(node_id)

    async def contains(self, node_id: str) -> bool:
        return await self._axis.contains(node_id)


async def test_a_descent_picks_the_bulk_members_up_and_costs_a_query_a_level() -> None:
    """The driver finds them by member presence, and the saving is the count.

    The measurement above is of the members; this is of the *walk*, which is
    where an axis that quietly lost them would go on answering correctly. The
    two optional protocols register nothing -- `_async_fetch` dispatches on
    `getattr(hierarchy, f"{member}_many", None)` -- so the difference between
    an axis that has them and one that does not is invisible to every
    assertion about what a walk returns.
    """
    wide = [("root", "Root", None), *((f"n{i}", f"N{i}", "root") for i in range(8))]
    deep = [*wide, *((f"d{i}", f"D{i}", f"n{i}") for i in range(8))]
    store = _CountingStore()
    for record in _rows(deep):
        await store.create(record)
    axis = ColumnHierarchy(store, "products", "sku", "parent_sku")

    _CountingStore.reads = 0
    through_bulk = await AsyncHierarchyView(axis, "root").descendants()
    bulk_reads = _CountingStore.reads

    _CountingStore.reads = 0
    through_singular = await AsyncHierarchyView(_SingularOnly(axis), "root").descendants()
    singular_reads = _CountingStore.reads

    # Same walk, same answer -- which is exactly why the count is the only
    # thing that can report the difference.
    assert sorted(view.node for view in through_bulk) == sorted(
        view.node for view in through_singular
    )
    assert len(through_bulk) == 16
    assert bulk_reads == 3, "one read per frontier, plus the one that finds it empty"
    assert singular_reads == 17, "one read per node, which is the cost the members remove"


async def test_a_snapshot_keeps_a_cycle_that_a_descent_from_the_roots_would_lose() -> None:
    """`parent_edges()` is why a copy of this axis is the whole of it.

    A cyclic component with nothing above it has no root to be found from, so
    a snapshot built by descending would be missing it -- and would then
    *refuse an anchor the live axis accepts*, because a walk checks
    containment. The member is what makes the copy one query and complete at
    the same time.
    """
    ring = (("a", "A", "b"), ("b", "B", "c"), ("c", "C", "a"))
    axis = ColumnHierarchy(await _store(ring), "products", "sku", "parent_sku")

    assert await axis.roots() == ()
    assert await axis.contains("a") is True

    copy = await AsyncMappingHierarchy.snapshot(axis)
    assert await copy.contains("a") is True
    assert sorted(await copy.parent_edges()) == ["a", "b", "c"]
    assert list(await copy.parents("a")) == ["b"]


async def test_an_edge_needs_both_ends_so_a_form_row_is_not_a_node() -> None:
    """Over a shared store, the narrowing the entity source needs is free here.

    `memory`, `file`, `s3` and `elasticsearch` hold both kinds of row in one
    store, and a surface-form row carries the projection's id column -- which
    is why the entity reads have to exclude it explicitly. This axis does not:
    a form row has no parent column, and an edge is a row with both ends.
    """
    store = await _store()
    await store.create(Record({"sku": "leaf", "folded_form": "beagle"}))
    axis = ColumnHierarchy(store, "products", "sku", "parent_sku")

    assert sorted(await axis.parent_edges()) == ["leaf", "mid", "root"]
    assert list(await axis.parents("leaf")) == ["mid"]


# --------------------------------------------------------------------------
# What binding one costs the registry: nothing it has to close
# --------------------------------------------------------------------------


async def test_binding_an_axis_opens_no_handle_of_its_own(caplog: pytest.LogCaptureFixture) -> None:
    """The axis reads the source's table, so it is handed the source's handle.

    Asserted as a count rather than argued. A second entry would mean the axis
    resolved a handle for itself, which is the failure ownership-at-acquisition
    exists to prevent: the registry would then be holding two handles over one
    store and closing both.
    """
    del caplog
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _store()
    )
    try:
        onto = await registry.load()
        assert type(onto.taxonomy("categories").structure) is ColumnHierarchy
        assert len(registry._handles) == 1
        handle, owned = registry._handles[0]
        assert owned is False, "an injected handle stays its owner's"
        assert onto.describes[0].table == "products"
    finally:
        await registry.close()
    assert len(registry._handles) == 1, "an injected handle is still recorded after a close"
    del handle


async def test_binding_and_walking_an_axis_does_no_blocking_io_on_the_loop() -> None:
    """Every read this axis makes is a query, and every query is awaited.

    The detector catches an import-time stall on a process's *first* load,
    which is the order-dependence review does not find -- so the load and the
    walk are both inside the block rather than only the part that obviously
    reaches for data.
    """
    store = await _store()
    with assert_no_blocking():
        registry = OntologyRegistry.from_components(
            config=OntologyConfig(**_document()), database=store
        )
        try:
            tax = (await registry.load()).taxonomy("categories")
            assert [view.node for view in await tax.at("leaf").ancestors()] == ["mid", "root"]
        finally:
            await registry.close()


# --------------------------------------------------------------------------
# What a rebuild announces about an axis whose nodes are rows
# --------------------------------------------------------------------------


async def test_a_rebuild_reports_what_a_live_axis_actually_gained_and_lost() -> None:
    """The delta over an axis's own nodes, recorded at each load.

    Computed from the two documents instead, the three sets are the *declared*
    assertions either side -- which is empty for every vocabulary a registry
    exists to bind, so the payload claimed a no-op whatever had changed and
    `an empty payload therefore means a genuine no-op` was false in the common
    case.

    Re-reading the outgoing axis at the rebuild does not fix it either, and
    that is why the population is recorded rather than derived: a live axis
    reads the table, so both sides of such a comparison would answer with the
    rows as they are now. The row added below is the proof -- it arrives in
    the payload only because the earlier population was kept.
    """
    seen: list[Event] = []
    bus = InMemoryEventBus()
    await bus.connect()
    await bus.subscribe("taxonomy:categories", seen.append)

    store = await _store((("root", "Root", None), ("mid", "Mid", "root")))
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=store, event_bus=bus
    )
    try:
        await registry.load()
        assert seen == []

        # One row added and one removed, chosen so that both sets move and
        # both moves are about *edges* rather than about rows. Before: the one
        # edge is `mid -> root`, so the axis holds both. After: `mid`'s row is
        # gone but `leaf` still names it as a parent, so `mid` stays; `root`
        # was named by nothing else, so it leaves.
        await store.create(Record({"sku": "leaf", "title": "Leaf", "parent_sku": "mid"}))
        await store.delete(
            next(
                record.id
                for record in await store.search(Query())
                if record.get_value("sku") == "mid"
            )
        )
        await registry.load(_document(), replace=True)

        assert len(seen) == 1
        payload = seen[0].payload
        assert payload["taxonomy_id"] == "categories"
        assert payload["arrived"] == ["leaf"]
        assert payload["gone"] == ["root"]
        # A rename keeps its id, and only a document can carry a name -- so an
        # axis whose nodes are rows reports the two sets it can compute.
        assert payload["renamed"] == []
    finally:
        await registry.close()
        await bus.close()


async def test_an_axis_a_rebuild_removed_reports_its_whole_population_gone() -> None:
    """The strongest thing a subscriber can be told: the axis is not there."""
    seen: list[Event] = []
    bus = InMemoryEventBus()
    await bus.connect()
    await bus.subscribe("taxonomy:categories", seen.append)

    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _store(), event_bus=bus
    )
    try:
        await registry.load()
        await registry.load(_document(taxonomies=[]), replace=True)

        assert len(seen) == 1
        assert seen[0].payload["gone"] == ["leaf", "mid", "root"]
        assert seen[0].payload["arrived"] == []
    finally:
        await registry.close()
        await bus.close()

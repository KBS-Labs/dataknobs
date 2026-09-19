"""The projection that is data, and the two things a loader cannot introspect.

A binding declares its schema or it is rejected, and a binding read by an
exact rung declares where its folded forms live or it is rejected -- same
rule, two things no ``Database`` in this package can be asked about. Neither
refusal carries an acceptance criterion of its own, which is why they are
here: a test that is not a criterion is still a test.

Both refusals are asserted in **both** directions. A suite carrying only the
refusal half passes against a source that refuses everything, which is the
failure the second of them is itself about.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.capabilities import (
    Capability,
    CapabilityContract,
    CapabilityNotSupportedError,
    require_capability,
    supports_capability,
)
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import OntologyConfig, SourceRef
from dataknobs_common.records import Record

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.ontology import EntityProjection, OntologyRegistry, RecordEntitySource
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.streaming import StreamConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

PROJECTION: dict[str, Any] = {
    "table": "products",
    "id": "sku",
    "name": "title",
    "type": {"const": "Product"},
    "aliases": {"column": "alt_names", "split": ","},
    "description": "blurb",
    "metadata": {"columns": ["owner_team"]},
}

SCHEMA: list[dict[str, str]] = [
    {"name": "sku", "type": "string"},
    {"name": "title", "type": "string"},
    {"name": "alt_names", "type": "string"},
    {"name": "blurb", "type": "string"},
    {"name": "owner_team", "type": "string"},
    {"name": "folded_form", "type": "string"},
]

FOLDED_LOOKUP: dict[str, str] = {
    "table": "product_forms",
    "form": "folded_form",
    "entity": "sku",
}


def _document(
    *, projection: dict[str, Any] | None = None, schema: Any = SCHEMA, **overrides: Any
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "id": "catalog",
        "entity_types": [{"id": "Product"}],
        "sources": [
            {
                "id": "products",
                "kind": "record",
                "entity_projection": dict(PROJECTION if projection is None else projection),
                "schema": schema,
            }
        ],
    }
    if schema is None:
        del document["sources"][0]["schema"]
    document.update(overrides)
    return document


async def _store(*rows: dict[str, Any]) -> AsyncMemoryDatabase:
    db = AsyncMemoryDatabase()
    for row in rows:
        await db.create(Record(dict(row)))
    return db


async def _bound(
    database: AsyncMemoryDatabase, **document_kwargs: Any
) -> tuple[OntologyRegistry, Any]:
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(**document_kwargs)), database=database
    )
    return registry, await registry.load()


# --------------------------------------------------------------------------
# The declared schema, and the column check it buys
# --------------------------------------------------------------------------


async def test_a_binding_with_no_schema_is_rejected_at_load_naming_the_binding() -> None:
    """No database here exposes introspection, so silence would validate nothing."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(schema=None)), database=await _store()
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        assert "'products'" in str(excinfo.value)
        assert "schema" in str(excinfo.value)
    finally:
        await registry.close()


async def test_a_projection_naming_an_undeclared_column_is_rejected_naming_the_column() -> None:
    """What the declaration buys: a column from a config file, checked before use."""
    projection = dict(PROJECTION, name="display_name")
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(projection=projection)), database=await _store()
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        assert "'display_name'" in str(excinfo.value)
    finally:
        await registry.close()


async def test_one_schema_covers_both_of_the_bindings_tables() -> None:
    """The surface-form table's columns are checked against the same list.

    ``FieldSchema`` carries no table, so a name declared once is checked once
    and both uses are checked against it. Asserted from the failing side,
    because that is the direction a hole would hide in: a form column nobody
    declared is refused exactly as an entity column nobody declared is.
    """
    projection = dict(PROJECTION, surface_forms=dict(FOLDED_LOOKUP, form="undeclared_fold"))
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(projection=projection)), database=await _store()
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        assert "'undeclared_fold'" in str(excinfo.value)
    finally:
        await registry.close()


async def test_a_dotted_path_validates_its_root_and_not_what_is_inside_the_value() -> None:
    """Exactly the promise, in both directions, because more would be the quiet failure.

    A declared ``payload`` proves the column exists and says nothing about
    ``legal_name`` inside it -- so the first loads and reads, the second loads
    and comes back empty at read time, and a path whose *root* is undeclared
    does not load at all.
    """
    projection = dict(PROJECTION, name="payload.legal_name")
    schema = [*SCHEMA, {"name": "payload", "type": "json"}]
    database = await _store(
        {"sku": "sku-1", "payload": {"legal_name": "Acme Holdings"}},
        {"sku": "sku-2", "payload": {"other": "x"}},
    )
    registry, ontology = await _bound(database, projection=projection, schema=schema)
    try:
        found = await ontology.entity("sku-1")
        assert found is not None and found.name == "Acme Holdings"
        # The path within the value is not validated, so a row that does not
        # carry it is a read-time miss rather than a load-time refusal.
        missing = await ontology.entity("sku-2")
        assert missing is not None and missing.name == "sku-2"
    finally:
        await registry.close()

    undeclared = OntologyRegistry.from_components(
        config=OntologyConfig(
            **_document(projection=dict(PROJECTION, name="absent_blob.legal_name"))
        ),
        database=await _store(),
    )
    try:
        with pytest.raises(ValidationError, match="absent_blob"):
            await undeclared.load()
    finally:
        await undeclared.close()


# --------------------------------------------------------------------------
# The folded lookup, and what a source with none does
# --------------------------------------------------------------------------


async def test_a_binding_with_a_folded_lookup_answers_a_folded_query() -> None:
    """The source folds both sides, so the case a person typed does not matter."""
    database = await _store(
        {"sku": "sku-4471", "title": "Beagle", "alt_names": "hound,beagle dog"},
        {"folded_form": "beagle", "sku": "sku-4471"},
        {"folded_form": "hound", "sku": "sku-4471"},
    )
    registry, ontology = await _bound(
        database,
        projection=dict(PROJECTION, surface_forms=FOLDED_LOOKUP),
        resolver={"rungs": [{"kind": "exact"}]},
    )
    try:
        assert await ontology.by_surface_form("Beagle") == frozenset({"sku-4471"})
        assert await ontology.by_surface_form("  BEAGLE ") == frozenset({"sku-4471"})
        assert await ontology.by_surface_form("Hound") == frozenset({"sku-4471"})
        # frozenset() means ran and matched nothing -- which is what it means
        # here, and is the reading a source with no lookup must not borrow.
        assert await ontology.by_surface_form("terrier") == frozenset()
        assert Capability.SURFACE_FORM_LOOKUP in ontology.describes[0].capabilities
    finally:
        await registry.close()


async def test_an_exact_rung_over_a_binding_with_no_lookup_is_rejected_at_load() -> None:
    """The other direction, and the one a suite is likely to stop at.

    The first rung of every cascade matches surface forms, the source folds,
    and a live table holds the form as it was written -- so the binding says
    where its folded forms live or the document does not load. The refusal
    names the key, because that is the one thing the consumer can act on.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(resolver={"rungs": [{"kind": "exact"}]})),
        database=await _store(),
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        assert "surface_forms" in str(excinfo.value)
        assert "'products'" in str(excinfo.value)
    finally:
        await registry.close()


async def test_a_binding_with_no_exact_rung_over_it_needs_no_lookup() -> None:
    """The refusal is about the cascade, not about live sources in general."""
    registry, ontology = await _bound(
        await _store({"sku": "sku-1", "title": "Widget"}),
        resolver={"rungs": [{"kind": "semantic"}]},
    )
    try:
        assert (await ontology.entity("sku-1")) is not None
        assert Capability.SURFACE_FORM_LOOKUP not in ontology.describes[0].capabilities
    finally:
        await registry.close()


async def test_a_source_with_no_lookup_refuses_rather_than_answering_the_unfolded_column() -> None:
    """Refusing is the whole point: an unfolded answer is a genuine miss's twin.

    ``frozenset()`` already means *ran and matched nothing*, and a cascade
    falls through to a guessing rung on exactly that reading. A source
    answering over the unfolded column would return it for every query whose
    case differs by one letter, and the cascade would report a vocabulary gap
    that is not there.
    """
    registry, ontology = await _bound(await _store({"sku": "sku-1", "title": "Widget"}))
    try:
        source = ontology.entities
        assert not source.supports(Capability.SURFACE_FORM_LOOKUP)
        with pytest.raises(CapabilityNotSupportedError) as excinfo:
            await source.by_surface_form("widget")
        assert excinfo.value.context["capability"] == "surface_form_lookup"
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The projection itself
# --------------------------------------------------------------------------


async def test_get_is_a_query_because_the_projections_id_is_not_the_storage_id() -> None:
    """Measured rather than assumed, and it is why there is no read path here."""
    database = await _store({"sku": "sku-4471", "title": "Beagle"})
    assert await database.read("sku-4471") is None
    assert await database.search(Query(filters=[Filter("sku", Operator.EQ, "sku-4471")]))

    registry, ontology = await _bound(database)
    try:
        assert (await ontology.entity("sku-4471")) is not None
    finally:
        await registry.close()


async def test_get_many_answers_the_ids_that_name_a_row_and_drops_the_rest() -> None:
    """One filter rather than N reads -- the bulk member is the point, not a tuning."""
    registry, ontology = await _bound(
        await _store(
            {"sku": "sku-1", "title": "One"},
            {"sku": "sku-2", "title": "Two"},
        )
    )
    try:
        found = await ontology.entities.get_many(["sku-1", "sku-2", "sku-404"])
        assert sorted(found) == ["sku-1", "sku-2"]
        assert found["sku-2"].name == "Two"
        assert await ontology.entities.get_many([]) == {}
    finally:
        await registry.close()


async def test_every_declarative_form_projects() -> None:
    """The bare scalars name columns; the braced forms are the ones that differ."""
    registry, ontology = await _bound(
        await _store(
            {
                "sku": "sku-1",
                "title": "Beagle",
                "alt_names": "hound, beagle dog ,",
                "blurb": "A small hound.",
                "owner_team": "catalog",
                "unprojected": "not mine",
            }
        )
    )
    try:
        entity = await ontology.entity("sku-1")
        assert entity is not None
        assert entity.id == "sku-1"
        assert entity.type == "Product"
        assert entity.name == "Beagle"
        assert entity.aliases == ["hound", "beagle dog"]
        assert entity.description == "A small hound."
        assert entity.metadata == {"owner_team": "catalog"}
        assert entity.source is not None
        assert entity.source.source_id == "products"
        assert entity.source.locator == {"table": "products", "sku": "sku-1"}
    finally:
        await registry.close()


async def test_describe_reports_the_projection_and_the_one_type_it_declares() -> None:
    """``declares`` is a set of one: a constant type is known from configuration."""
    registry, ontology = await _bound(await _store({"sku": "sku-1", "title": "Beagle"}))
    try:
        description = ontology.describes[0]
        assert description.source_id == "products"
        assert description.table == "products"
        assert description.declares == frozenset({"Product"})
        assert description.projection["id"] == "sku"
        assert Capability.ORIGIN_FETCH in description.capabilities
        assert await ontology.entities.by_type("Product") == frozenset({"sku-1"})
        assert await ontology.entities.by_type("Widget") == frozenset()
    finally:
        await registry.close()


async def test_expose_origin_false_withholds_the_capability_and_answers_none() -> None:
    """A caller learns it from ``describe()`` rather than from a None it cannot read."""
    database = await _store({"sku": "sku-1", "title": "Beagle", "unprojected": "secret"})
    registry, ontology = await _bound(database, projection=dict(PROJECTION, expose_origin=False))
    try:
        entity = await ontology.entity("sku-1")
        assert entity is not None and entity.source is not None
        assert Capability.ORIGIN_FETCH not in ontology.describes[0].capabilities
        assert await ontology.entities.fetch_origin(entity.source) is None
    finally:
        await registry.close()


async def test_fetch_origin_returns_the_whole_row_where_origins_are_exposed() -> None:
    """The projection is a mapping, not an access boundary, and says so by default."""
    registry, ontology = await _bound(
        await _store({"sku": "sku-1", "title": "Beagle", "unprojected": "visible"})
    )
    try:
        entity = await ontology.entity("sku-1")
        assert entity is not None and entity.source is not None
        origin = await ontology.entities.fetch_origin(entity.source)
        assert origin is not None
        assert origin.get_value("unprojected") == "visible"
    finally:
        await registry.close()


async def test_the_bulk_origin_member_answers_one_slot_per_ref() -> None:
    """One slot per ref, in the order they were asked -- the singular member, lifted.

    ``fetch_origins`` is ``fetch_origin`` over a sequence, so its element type
    is that member's return type and the answer is positional. A caller holds
    the refs it passed, so a positional answer carries strictly more than a
    mapping would: the misses are in it, and two refs naming one row stay two
    slots rather than collapsing into one key.
    """
    registry, ontology = await _bound(
        await _store(
            {"sku": "sku-1", "title": "Beagle", "unprojected": "first"},
            {"sku": "sku-2", "title": "Corgi", "unprojected": "second"},
        )
    )
    try:
        first = await ontology.entity("sku-1")
        second = await ontology.entity("sku-2")
        assert first is not None and first.source is not None
        assert second is not None and second.source is not None

        origins = await ontology.entities.fetch_origins([first.source, second.source])

        assert len(origins) == 2
        assert [o.get_value("unprojected") for o in origins if o is not None] == [
            "first",
            "second",
        ]
    finally:
        await registry.close()


async def test_a_ref_that_reaches_no_row_is_a_none_in_its_own_slot() -> None:
    """A miss is reported where it happened, which a mapping could not have said.

    Three ways to miss, asserted together because the positional answer is what
    makes them distinguishable at all: a ref naming a row that is not there, a
    ref belonging to another source, and a ref carrying no id. A mapping keyed
    by ref would have answered all three by omission, and a caller could not
    tell which of its refs the omission was about.
    """
    registry, ontology = await _bound(await _store({"sku": "sku-1", "title": "Beagle"}))
    try:
        entity = await ontology.entity("sku-1")
        assert entity is not None and entity.source is not None
        mine = entity.source
        absent = SourceRef(source_id=mine.source_id, kind=mine.kind, locator={"sku": "sku-404"})
        foreign = SourceRef(source_id="somebody-else", kind=mine.kind, locator={"sku": "sku-1"})

        origins = await ontology.entities.fetch_origins([absent, mine, foreign])

        assert len(origins) == 3
        assert origins[0] is None
        assert origins[1] is not None and origins[1].get_value("title") == "Beagle"
        assert origins[2] is None
    finally:
        await registry.close()


async def test_the_bulk_member_is_one_read_rather_than_one_per_ref() -> None:
    """What the member is *for*: N refs, one round trip.

    Asserted on the door rather than on the answer, because the answer is the
    same either way -- a loop calling ``fetch_origin`` N times returns exactly
    these rows and is exactly the thing this member exists to avoid. Only the
    read count tells the two apart.
    """
    database = _ReadDoorProbe()
    for row in (
        {"sku": "sku-1", "title": "A"},
        {"sku": "sku-2", "title": "B"},
        {"sku": "sku-3", "title": "C"},
    ):
        await database.create(Record(dict(row)))
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=database
    )
    ontology = await registry.load()
    try:
        refs = []
        for sku in ("sku-1", "sku-2", "sku-3"):
            entity = await ontology.entity(sku)
            assert entity is not None and entity.source is not None
            refs.append(entity.source)

        database.doors.clear()
        origins = await ontology.entities.fetch_origins(refs)

        assert [o.get_value("title") for o in origins if o is not None] == ["A", "B", "C"]
        assert database.doors.count("search") == 1
    finally:
        await registry.close()


async def test_the_capability_it_declares_is_the_one_a_caller_can_spend() -> None:
    """``require_capability`` then call -- the recommended sequence, end to end.

    The guard answering yes and the call raising is the shape this package
    refuses elsewhere, and it is the whole reason the declared return type had
    to change rather than the declaration.
    """
    registry, ontology = await _bound(await _store({"sku": "sku-1", "title": "Beagle"}))
    try:
        source = ontology.entities
        entity = await ontology.entity("sku-1")
        assert entity is not None and entity.source is not None

        require_capability(source, Capability.ORIGIN_FETCH)
        origins = await source.fetch_origins([entity.source])

        assert len(origins) == 1
        assert origins[0] is not None
    finally:
        await registry.close()


async def test_no_refs_is_no_slots_and_no_read() -> None:
    """The empty case answers ``[]`` without reaching the store."""
    database = _ReadDoorProbe()
    await database.create(Record({"sku": "sku-1", "title": "Beagle"}))
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=database
    )
    await registry.load()
    try:
        database.doors.clear()
        assert await registry.get("catalog").entities.fetch_origins([]) == []
        assert database.doors == []
    finally:
        await registry.close()


async def test_longest_form_tokens_is_none_over_a_live_table() -> None:
    """Honest rather than lazy: the longest form is not knowable without a scan."""
    registry, ontology = await _bound(await _store({"sku": "sku-1", "title": "Beagle"}))
    try:
        assert ontology.longest_form_tokens() is None
    finally:
        await registry.close()


async def test_a_type_column_is_refused_while_declares_cannot_say_it_cannot_enumerate() -> None:
    """Refusing costs a `const:`; answering would cost entities that stop being indexed."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(projection=dict(PROJECTION, type="kind"))),
        database=await _store(),
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        assert "'kind'" in str(excinfo.value)
        assert "const" in str(excinfo.value)
    finally:
        await registry.close()


async def test_a_binding_with_neither_a_database_nor_a_handle_is_refused() -> None:
    """A live source is read through a resolved reference or an injected handle."""
    registry = OntologyRegistry(config=OntologyConfig(**_document()))
    try:
        with pytest.raises(ValidationError, match="database"):
            await registry.load()
    finally:
        await registry.close()


def test_the_projection_refuses_a_block_it_cannot_parse() -> None:
    """Every refusal names the binding, because that is what a consumer edits."""
    with pytest.raises(ValidationError, match="'products'"):
        EntityProjection.from_mapping({"id": "sku", "type": {"const": "P"}}, binding="products")
    with pytest.raises(ValidationError, match="'products'"):
        EntityProjection.from_mapping({"table": "t", "type": {"const": "P"}}, binding="products")
    with pytest.raises(ValidationError, match="metadata"):
        EntityProjection.from_mapping(
            {"table": "t", "id": "i", "type": {"const": "P"}, "metadata": ["a"]},
            binding="products",
        )
    with pytest.raises(ValidationError, match="surface_forms"):
        EntityProjection.from_mapping(
            {"table": "t", "id": "i", "type": {"const": "P"}, "surface_forms": "forms"},
            binding="products",
        )


def test_the_source_folds_with_the_normalizer_it_was_built_with() -> None:
    """One callable for both halves, rather than a second knob beside the grammar."""
    projection = EntityProjection.from_mapping(
        dict(PROJECTION, surface_forms=FOLDED_LOOKUP), binding="products"
    )
    source = RecordEntitySource(
        AsyncMemoryDatabase(),
        projection,
        source_id="products",
        normalizer=str.upper,
    )
    assert source._normalizer("beagle") == "BEAGLE"


# --------------------------------------------------------------------------
# One store, two kinds of row -- and the entity side telling them apart
# --------------------------------------------------------------------------


async def _mixed_store() -> AsyncMemoryDatabase:
    """A shared store holding form rows on **both** sides of the entity row.

    Both sides deliberately. ``get`` takes the first row a search answers and
    ``get_many`` keeps the last, so a store with form rows on one side only
    lets whichever member reads from the other end pass by luck. Insertion
    order is not something the source may depend on, and a suite that seeds
    one order cannot say so.
    """
    return await _store(
        {"folded_form": "beagle", "sku": "sku-4471"},
        {"sku": "sku-4471", "title": "Beagle", "alt_names": "hound,beagle dog"},
        {"folded_form": "hound", "sku": "sku-4471"},
        # A form row whose entity row is gone -- a stale index, which is the
        # ordinary state of one written by a separate job.
        {"folded_form": "ghost", "sku": "sku-9999"},
    )


async def test_get_over_a_shared_store_answers_the_entity_row_not_a_form_row() -> None:
    """A form row carries the projection's id column, so an id filter matches it.

    On ``memory``, ``file``, ``s3`` and ``elasticsearch`` the handle *is* the
    store: both kinds of row live in it and one id filter reaches both. The
    form side already tells them apart -- it filters on a column only a form
    row carries -- and this is that same rule applied in the other direction.
    """
    database = await _mixed_store()
    registry, ontology = await _bound(
        database, projection=dict(PROJECTION, surface_forms=FOLDED_LOOKUP)
    )
    try:
        found = await ontology.entity("sku-4471")
        assert found is not None
        assert found.name == "Beagle", "a form row was projected as the entity"
        assert found.aliases == ["hound", "beagle dog"]
    finally:
        await registry.close()


async def test_get_many_over_a_shared_store_answers_the_entity_row_not_a_form_row() -> None:
    """The bulk member keeps the last row per id, so it fails from the other end."""
    database = await _mixed_store()
    registry, ontology = await _bound(
        database, projection=dict(PROJECTION, surface_forms=FOLDED_LOOKUP)
    )
    try:
        found = await ontology.entities.get_many(["sku-4471"])
        assert set(found) == {"sku-4471"}
        assert found["sku-4471"].name == "Beagle", "a form row was projected as the entity"
    finally:
        await registry.close()


async def test_fetch_origin_over_a_shared_store_answers_the_entity_row() -> None:
    """The origin is the row the entity was projected from, not a row beside it."""
    database = await _mixed_store()
    registry, ontology = await _bound(
        database, projection=dict(PROJECTION, surface_forms=FOLDED_LOOKUP)
    )
    try:
        found = await ontology.entity("sku-4471")
        assert found is not None
        origin = await ontology.entities.fetch_origin(found.source)
        assert origin is not None
        assert origin.get_value("title") == "Beagle", "a form row was returned as the origin"
    finally:
        await registry.close()


async def test_by_type_over_a_shared_store_does_not_invent_entities_from_form_rows() -> None:
    """A stale form row names an id no entity row carries; the scan must not report it."""
    database = await _mixed_store()
    registry, ontology = await _bound(
        database, projection=dict(PROJECTION, surface_forms=FOLDED_LOOKUP)
    )
    try:
        assert await ontology.entities.by_type("Product") == frozenset({"sku-4471"})
    finally:
        await registry.close()


async def test_a_binding_with_no_lookup_reads_every_row_it_always_did() -> None:
    """The discrimination costs nothing where there is nothing to discriminate.

    A projection declaring no ``surface_forms:`` has no form column to
    exclude, so the entity side asks exactly what it asked before -- asserted
    because a filter added unconditionally would quietly drop every row of a
    table that happens to carry no such column.
    """
    database = await _store({"sku": "sku-1", "title": "Widget"})
    registry, ontology = await _bound(database)
    try:
        found = await ontology.entity("sku-1")
        assert found is not None and found.name == "Widget"
        assert await ontology.entities.by_type("Product") == frozenset({"sku-1"})
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# Every rung that reads a folded lookup, not only the one that named it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["exact", "scan", "lexical"])
async def test_a_rung_reading_surface_forms_over_a_binding_with_no_lookup_is_refused(
    kind: str,
) -> None:
    """``exact`` is one of three, and the other two fail identically at query time.

    ``ScanningSignal`` and ``LexicalSignal`` both reach ``by_surface_form``;
    a refusal naming only ``exact`` lets a document declaring either of them
    load clean and raise on the first resolve. The set is read from what each
    rung kind declares about itself, so a consumer's own rung is covered by
    declaring the same thing.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(resolver={"rungs": [{"kind": kind}]})),
        database=await _store(),
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        assert "surface_forms" in str(excinfo.value)
        assert "'products'" in str(excinfo.value)
    finally:
        await registry.close()


@pytest.mark.parametrize(
    "resolver",
    [
        {"rungs": [{"kind": "alias"}]},
        {"rungs": []},
        {},
    ],
    ids=["a rung that reads alias forms", "an empty composition", "a section with no rungs"],
)
async def test_a_composition_that_reads_no_surface_forms_loads_unchanged(
    resolver: dict[str, Any],
) -> None:
    """The other direction, so the refusal is not "refuse every record binding".

    ``alias`` reads ``by_alias_form``, and an empty composition reads nothing.
    A document writing either has declared a policy this binding satisfies.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(resolver=resolver)),
        database=await _store({"sku": "sku-1", "title": "Widget"}),
    )
    try:
        ontology = await registry.load()
        assert Capability.SURFACE_FORM_LOOKUP not in ontology.describes[0].capabilities
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The capability contract, answered rather than re-implemented
# --------------------------------------------------------------------------


async def test_the_source_answers_the_capability_contract_it_advertises() -> None:
    """Every member of the contract, not only the one a duck-typed guard reads.

    ``supports()`` worked, because :func:`supports_capability` duck-types on
    it. The other three members did not exist, so a caller enumerating
    capabilities *through the contract* -- ``isinstance(source,
    CapabilityContract)``, then ``supported_capabilities()`` for what the kind
    can do and ``instance_capabilities()`` for what this binding does -- saw a
    source that was not a contract host at all.
    """
    registry, ontology = await _bound(
        await _store({"sku": "sku-1", "title": "Beagle"}),
        projection=dict(PROJECTION, surface_forms=FOLDED_LOOKUP),
    )
    try:
        source = ontology.entities
        assert isinstance(source, CapabilityContract)
        # The classmethod half is the ceiling: what a `kind: record` binding
        # can declare, answerable before one exists.
        assert RecordEntitySource.supported_capabilities() == frozenset(
            {Capability.ORIGIN_FETCH, Capability.SURFACE_FORM_LOOKUP}
        )
        assert source.instance_capabilities() == source.describe().capabilities
        assert source.supports(Capability.SURFACE_FORM_LOOKUP)
        assert source.supports("surface_form_lookup"), "a raw string is the consumer's spelling"
        require_capability(source, Capability.ORIGIN_FETCH)
    finally:
        await registry.close()


async def test_a_binding_without_a_lookup_declares_less_than_its_kind_can() -> None:
    """The instance set is the projection's answer; the class set is the kind's."""
    registry, ontology = await _bound(await _store({"sku": "sku-1", "title": "Beagle"}))
    try:
        source = ontology.entities
        assert Capability.SURFACE_FORM_LOOKUP in RecordEntitySource.supported_capabilities()
        assert Capability.SURFACE_FORM_LOOKUP not in source.instance_capabilities()
        assert not supports_capability(source, Capability.SURFACE_FORM_LOOKUP)
        with pytest.raises(CapabilityNotSupportedError):
            require_capability(source, Capability.SURFACE_FORM_LOOKUP)
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# What the scan costs, on the door it takes
# --------------------------------------------------------------------------


class _ReadDoorProbe(AsyncMemoryDatabase):
    """A real store that records which read door each call arrived through.

    A subclass rather than a mock, so every call still runs the memory
    backend's own code and the rows come back for real -- the recording is
    the only thing added.
    """

    def __init__(self) -> None:
        super().__init__()
        self.doors: list[str] = []

    async def all(self) -> list[Record]:
        self.doors.append("all")
        return await super().all()

    async def search(self, query: Any) -> list[Record]:
        self.doors.append("search")
        return await super().search(query)

    def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        self.doors.append("stream_read")
        return super().stream_read(query, config)


@pytest.mark.parametrize(
    ("projection", "rows", "expected", "door", "refused"),
    [
        pytest.param(
            PROJECTION,
            [{"sku": "sku-1", "title": "Widget"}, {"sku": "sku-2", "title": "Gadget"}],
            frozenset({"sku-1", "sku-2"}),
            "stream_read",
            "all",
            id="whole-table-streams",
        ),
        pytest.param(
            dict(PROJECTION, surface_forms=FOLDED_LOOKUP),
            [
                {"sku": "sku-1", "title": "Widget"},
                {"folded_form": "widget", "sku": "sku-1"},
            ],
            frozenset({"sku-1"}),
            "search",
            "stream_read",
            id="narrowed-does-not",
        ),
    ],
)
async def test_the_type_scan_streams_the_read_that_has_no_filter_to_lose(
    projection: dict[str, Any],
    rows: list[dict[str, Any]],
    expected: frozenset[str],
    door: str,
    refused: str,
) -> None:
    """The whole-table read streams; the narrowed one keeps ``search``, deliberately.

    The asymmetry is the subject. ``stream_read`` bounds what a scan holds,
    which is worth having on the branch that reads the table -- but the two
    members are separate implementations per backend and do not agree
    everywhere: Postgres's ``stream_read`` silently drops non-EQ filters, and
    ``NOT_EXISTS`` on the form column is exactly what the narrowed branch
    sends. Streaming that branch would answer with form rows projected as
    entities, on one backend, with no error.

    ``test_a_sql_backend_can_share_one_store_which_is_why_the_filter_stays_on_search``
    is the other half: it pins that the two cases *can* meet, so this
    asymmetry cannot be argued away later.
    """
    database = _ReadDoorProbe()
    for row in rows:
        await database.create(Record(dict(row)))
    registry, ontology = await _bound(database, projection=projection)
    try:
        database.doors.clear()
        assert await ontology.entities.by_type("Product") == expected
    finally:
        await registry.close()

    assert database.doors[0] == door, f"the scan took {database.doors[0]!r}"
    assert refused not in database.doors


async def test_a_sql_backend_can_share_one_store_which_is_why_the_filter_stays_on_search(
    tmp_path: Path,
) -> None:
    """The argument that would license streaming the narrowed branch is false.

    It goes: a filter is emitted only for a shared store, and a shared store
    is only the backends declaring no ``table``, so Postgres -- the one whose
    ``stream_read`` drops non-EQ filters -- is never reached with one. The
    second step does not hold. A projection whose ``surface_forms:`` names the
    *same* table as its entity rows keys to one handle on **any** backend,
    including one that declares a ``table``, so the store is shared and the
    filter is emitted.

    SQLite stands in for Postgres here because it is the SQL backend a test
    can run; what is being pinned is the *configuration* reaching the filtered
    branch on a table-declaring backend, which is backend-independent.
    """
    document = _document(
        projection=dict(
            PROJECTION,
            surface_forms={"table": "products", "form": "folded_form", "entity": "sku"},
        ),
        schema=[*SCHEMA, {"name": "folded_form", "type": "string"}],
    )
    document["sources"][0]["database"] = {
        "backend": "sqlite",
        "path": str(tmp_path / "catalog.db"),
    }
    registry = OntologyRegistry(config=OntologyConfig(**document), strict_resources=False)
    try:
        ontology = await registry.load()
        source = ontology.entities
        assert len(registry._handles) == 1, "one table named twice is one handle"
        assert source._shared_store is True
        assert [f.operator for f in source._entity_filters()] == [Operator.NOT_EXISTS]
    finally:
        await registry.close()

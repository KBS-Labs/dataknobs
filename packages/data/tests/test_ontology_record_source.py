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
from dataknobs_common.entity_resolution.signals import AsyncScanningSignal
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import OntologyConfig, SourceRef
from dataknobs_common.records import Record
from dataknobs_common.testing import requires_package

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.ontology import EntityProjection, OntologyRegistry, RecordEntitySource
from dataknobs_data.ontology.sources import READ_BATCH_SIZE
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.streaming import StreamConfig
from dataknobs_data.testing import DeterministicEmbedder

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
    """The refusal is about the cascade, not about live sources in general.

    ``semantic`` rather than the ``alias`` the parametrized case below uses,
    because this is the one kind declaring it reads no surface forms that also
    reaches for something the binding does not hold --- so the document
    carries an ``index:`` and the cascade is really built. A section the
    registry cannot construct is refused before this assertion is reached,
    which is what makes the passing half a load rather than a section nothing
    read.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(
            **_document(
                resolver={"rungs": [{"kind": "semantic"}]},
                index={"store": {"backend": "memory", "dimensions": 8}},
            )
        ),
        database=await _store({"sku": "sku-1", "title": "Widget"}),
        embedder=DeterministicEmbedder(dimensions=8),
    )
    try:
        ontology = await registry.load()
        assert (await ontology.entity("sku-1")) is not None
        assert Capability.SURFACE_FORM_LOOKUP not in ontology.describes[0].capabilities
        assert registry.resolver("catalog") is not None
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


async def test_a_projection_typing_every_row_with_an_undeclared_type_is_refused() -> None:
    """The ninth reference of the class the loader refuses eight of.

    ``entity_projection.type: {const: Widget}`` names an ``entity_types:`` id
    exactly as an ``entities:`` row's ``type:`` does, and the consequence of
    its naming nothing is the same one: the entities are untyped, and an index
    enumerating the vocabulary by type finds none of them. It is the worse
    half of the pair, because an authored row mistypes one entity and a
    projection mistypes **every row of a live table**.

    Refused here rather than in ``build_ontology`` with the other eight:
    ``entity_projection`` is this package's schema and ``dataknobs_common``
    has no notion of it, so checking it there would make the core read a
    section only a live binding understands.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(projection=dict(PROJECTION, type={"const": "Widget"}))),
        database=await _store(),
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()

        message = str(excinfo.value)
        assert "'Widget'" in message
        assert "'products'" in message, "the refusal names the binding a consumer edits"
        assert "Product" in message, "and what the document does declare"
    finally:
        await registry.close()


async def test_a_projection_type_is_unchecked_where_no_entity_types_are_declared() -> None:
    """The guard the other eight carry, at the ninth: an empty section is no schema.

    A document that binds a live table and leaves its type vocabulary to an
    ontology it imports is not making a claim this registry can check.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(entity_types=[])),
        database=await _store({"sku": "sku-1", "title": "Beagle"}),
    )
    try:
        ontology = await registry.load()

        assert await ontology.entity("sku-1") is not None
    finally:
        await registry.close()


async def test_a_projection_type_is_unchecked_where_the_document_imports() -> None:
    """The ninth reference inherits the family's other exemption too.

    ``imports:`` is the document saying it does not declare its sections in
    full, and a projection's ``const:`` may name a type the imported
    vocabulary declares. The core switches all eight of its checks off for
    such a document; this one is the same rule over this package's section,
    so it switches off with them or the family disagrees with itself.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(
            **_document(
                projection=dict(PROJECTION, type={"const": "Widget"}),
                imports=["catalogue-core"],
            )
        ),
        database=await _store({"sku": "sku-1", "title": "Beagle"}),
    )
    try:
        ontology = await registry.load()

        assert await ontology.entity("sku-1") is not None
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


async def test_fetch_origins_over_a_shared_store_answers_entity_rows_in_order() -> None:
    """The plural member, which the singular one's pass does not cover.

    ``fetch_origin`` reads one ref through an ``EQ``; ``fetch_origins`` reads
    every ref through one ``IN`` split into batches, which is a second query
    shape over the same rows -- so the narrowing has to be emitted on both
    paths and only one of them was asserted. The stale form row is what makes
    the difference visible: its ``sku`` names no entity row, so a plural read
    that reached form rows would answer that slot with one instead of None,
    and the caller would read a folded form as the origin of an entity that
    is not there.
    """
    database = await _mixed_store()
    registry, ontology = await _bound(
        database, projection=dict(PROJECTION, surface_forms=FOLDED_LOOKUP)
    )
    try:
        present = await ontology.entity("sku-4471")
        assert present is not None
        absent = SourceRef(
            source_id=present.source.source_id,
            kind=present.source.kind,
            locator={"sku": "sku-9999"},
        )

        origins = await ontology.entities.fetch_origins([present.source, absent])

        assert len(origins) == 2, "one slot per ref, in the order asked"
        assert origins[0] is not None
        assert origins[0].get_value("title") == "Beagle", "a form row was returned as an origin"
        assert origins[1] is None, "a stale form row is not an entity's origin"
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
    ("projection", "rows", "expected"),
    [
        pytest.param(
            PROJECTION,
            [{"sku": "sku-1", "title": "Widget"}, {"sku": "sku-2", "title": "Gadget"}],
            frozenset({"sku-1", "sku-2"}),
            id="whole-table",
        ),
        pytest.param(
            dict(PROJECTION, surface_forms=FOLDED_LOOKUP),
            [
                {"sku": "sku-1", "title": "Widget"},
                {"folded_form": "widget", "sku": "sku-1"},
            ],
            frozenset({"sku-1"}),
            id="narrowed",
        ),
    ],
)
async def test_the_type_scan_streams_whichever_branch_it_takes(
    projection: dict[str, Any],
    rows: list[dict[str, Any]],
    expected: frozenset[str],
) -> None:
    """Both branches stream, and the narrowed one could not always.

    The asymmetry this test used to pin was a workaround: ``stream_read`` and
    ``search`` are separate implementations per backend, and Postgres's
    streaming door open-coded its WHERE clause and silently dropped the
    ``NOT_EXISTS`` the narrowed branch sends. That is fixed -- the async
    Postgres ``stream_read`` builds its clause through the same
    ``SQLQueryBuilder`` its ``search`` uses -- so the workaround went with the
    defect it was for.

    What it is replaced with is stronger than symmetry for its own sake. An
    unbounded ``search`` is the read a backend is free to cap, and
    ``AsyncElasticsearchDatabase`` caps one at ``size=10000``; the narrowed
    branch was the branch that took it.

    **The door asserted is the first one, and the qualifier is not a hedge.**
    A ``search`` can still be recorded after it, because ``stream_read`` is
    each backend's own implementation and the memory backend's delegates to
    its own ``search`` for the filtered case. That is the backend's business,
    not a read this source issued, and it is precisely why the caps question
    cannot be settled on this probe: the backend that actually caps answers
    its streaming door through the scroll API and never re-enters ``search``.
    ``test_the_narrowed_type_scan_is_not_truncated_by_a_capping_backend`` is
    that half, over a store that caps for real.
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

    assert database.doors[0] == "stream_read", f"the scan took {database.doors[0]!r}"
    assert "all" not in database.doors, "the scan read the table rather than a query"


@requires_package("aiosqlite")
async def test_a_sql_backend_can_share_one_store_which_is_why_the_filter_must_survive_streaming(
    tmp_path: Path,
) -> None:
    """A shared store is reachable on a table-declaring backend, so the filter is too.

    This used to pin the opposite conclusion -- that the narrowed branch had
    to stay on ``search``, because Postgres's ``stream_read`` dropped non-EQ
    filters and a shared store could reach Postgres. The premise it argued
    against was: a filter is emitted only for a shared store, and a shared
    store is only the backends declaring no ``table``, so the one backend that
    dropped filters is never reached with one.

    **That premise is still false, and that is still the point.** A projection
    whose ``surface_forms:`` names the *same* table as its entity rows keys to
    one handle on **any** backend, including one that declares a ``table``, so
    the store is shared and the filter is emitted. What changed is the remedy:
    rather than routing around a streaming door that lost the filter, the door
    was fixed to carry it. This test is what makes that a requirement rather
    than a nicety -- it is the configuration that puts a ``NOT_EXISTS`` filter
    through a SQL backend's ``stream_read``.

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
        assert [f.operator for f in source.entity_filters()] == [Operator.NOT_EXISTS]
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# A backend that caps an unbounded read, and the two doors that must not lose
# rows to it
# --------------------------------------------------------------------------


class _CappedSearchProbe(AsyncMemoryDatabase):
    """A real store whose ``search`` caps a query carrying no ``limit``.

    This is ``AsyncElasticsearchDatabase``'s contract, at a size a test can
    hold. That backend reads ``size = query.limit_value if query.limit_value
    is not None else 10000``, so an unbounded ``search`` comes back truncated
    -- silently, because a short list is what a matching read of a small table
    looks like. ``stream_read`` there goes through the scroll API, which the
    cap does not reach, and this probe splits the same way.

    A subclass rather than a mock: both doors run the memory backend's own
    matching code and the rows come back for real. The cap is the only thing
    added, and it is applied to the *parent's* result so the override cannot
    be re-entered by a door that delegates.
    """

    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap

    async def search(self, query: Any) -> list[Record]:
        found = await AsyncMemoryDatabase.search(self, query)
        return found if query.limit_value is not None else found[: self.cap]

    def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        return self._scroll(query)

    async def _scroll(self, query: Query | None) -> AsyncIterator[Record]:
        """The door the cap does not reach, as the scroll API is for ES."""
        for record in await AsyncMemoryDatabase.search(self, query or Query()):
            yield record


async def test_the_narrowed_type_scan_is_not_truncated_by_a_capping_backend() -> None:
    """``by_type`` answers every id, including where ``search`` would cap.

    The narrowed branch used to take ``search`` with no ``limit``, so on
    Elasticsearch -- which declares ``index`` rather than ``table``, and is
    therefore always a shared store -- a binding of more rows than the cap
    answered with the cap's worth and reported nothing. The docstring two
    paragraphs above the read claims "on a million-row binding this returns a
    million-element set"; this is that claim, asserted.
    """
    database = _CappedSearchProbe(cap=5)
    for index in range(12):
        await database.create(Record({"sku": f"sku-{index:02d}", "title": f"Widget {index}"}))
        await database.create(Record({"folded_form": f"widget {index}", "sku": f"sku-{index:02d}"}))
    registry, ontology = await _bound(
        database, projection=dict(PROJECTION, surface_forms=FOLDED_LOOKUP)
    )
    try:
        found = await ontology.entities.by_type("Product")
    finally:
        await registry.close()

    assert len(found) == 12, "the narrowed scan answered the backend's cap, not the table"
    assert found == frozenset(f"sku-{index:02d}" for index in range(12))


async def test_get_many_over_more_ids_than_a_backend_will_answer_at_once() -> None:
    """A bulk read of more ids than the cap answers all of them.

    One ``IN`` filter is the member's whole reason to exist, but one ``IN``
    filter is also one read, and a read is what a backend caps. The batch is
    split so that no single read asks for more rows than a backend will
    answer -- which is the same bound that keeps a bind-parameter ceiling out
    of reach on the SQL backends.
    """
    database = _CappedSearchProbe(cap=READ_BATCH_SIZE)
    ids = [f"sku-{index:05d}" for index in range(READ_BATCH_SIZE + 500)]
    for entity_id in ids:
        await database.create(Record({"sku": entity_id, "title": f"Widget {entity_id}"}))
    registry, ontology = await _bound(database)
    try:
        found = await ontology.entities.get_many(ids)
    finally:
        await registry.close()

    assert len(found) == len(ids), "the bulk read answered one read's worth"
    assert set(found) == set(ids)


async def test_fetch_origins_over_more_refs_than_a_backend_will_answer_at_once() -> None:
    """Same bound, same reason -- and one slot per ref however many reads it took."""
    database = _CappedSearchProbe(cap=READ_BATCH_SIZE)
    ids = [f"sku-{index:05d}" for index in range(READ_BATCH_SIZE + 500)]
    for entity_id in ids:
        await database.create(Record({"sku": entity_id, "title": f"Widget {entity_id}"}))
    registry, ontology = await _bound(database)
    refs = [
        SourceRef(source_id="products", kind="record", locator={"sku": entity_id})
        for entity_id in ids
    ]
    try:
        origins = await ontology.entities.fetch_origins(refs)
    finally:
        await registry.close()

    assert len(origins) == len(refs), "a slot per ref survives the split"
    assert all(origin is not None for origin in origins), "a chunked read lost rows"


# --------------------------------------------------------------------------
# What a scan over a live table costs, and the number that bounds it
# --------------------------------------------------------------------------


async def test_a_declared_bound_is_what_the_source_reports() -> None:
    """``longest_form_tokens`` is configuration here, because nothing can measure it.

    An authored index counts its own keys. A live table cannot be counted
    without reading it, and a count taken at load is stale as soon as a row is
    written -- so the number a scanning rung needs is declared beside the
    lookup it describes, by the consumer who knows their own vocabulary.
    """
    projection = EntityProjection.from_mapping(
        dict(PROJECTION, surface_forms=dict(FOLDED_LOOKUP, longest_form_tokens=4)),
        binding="products",
    )
    assert projection.surface_forms is not None
    assert projection.surface_forms.longest_form_tokens == 4
    source = RecordEntitySource(AsyncMemoryDatabase(), projection, source_id="products")
    assert source.longest_form_tokens() == 4


async def test_no_declared_bound_is_still_none_rather_than_a_guess() -> None:
    """The honest answer where none is declared, unchanged.

    ``None`` is this member's published spelling of *I cannot bound this*, and
    inventing a default would make a form longer than it silently unfindable
    -- a cost problem traded for a correctness one. What changes is that the
    combination is refused at load rather than reaching a resolve.
    """
    projection = EntityProjection.from_mapping(
        dict(PROJECTION, surface_forms=FOLDED_LOOKUP), binding="products"
    )
    source = RecordEntitySource(AsyncMemoryDatabase(), projection, source_id="products")
    assert source.longest_form_tokens() is None


async def test_a_declared_bound_makes_the_scan_linear_in_the_query() -> None:
    """The bound's whole purpose, counted rather than argued.

    Unbounded, a scan spends *n(n+1)/2* probes for *n* tokens, and over a live
    binding each probe is a database round trip -- 1,275 of them for a
    fifty-token utterance. Bounded at *L*, it spends at most *n x L*.
    """
    database = _CountingSearchProbe()
    await database.create(Record({"sku": "sku-1", "title": "Beagle"}))
    await database.create(Record({"folded_form": "beagle", "sku": "sku-1"}))
    projection = EntityProjection.from_mapping(
        dict(PROJECTION, surface_forms=dict(FOLDED_LOOKUP, longest_form_tokens=3)),
        binding="products",
    )
    source = RecordEntitySource(database, projection, source_id="products")
    rung = AsyncScanningSignal(source)

    query = " ".join(f"word{index}" for index in range(50))
    database.searches = 0
    await rung.candidates(query, 10)

    assert database.searches <= 50 * 3, f"{database.searches} probes is not linear"
    assert database.searches == sum(50 - length + 1 for length in range(1, 4))


class _CountingSearchProbe(AsyncMemoryDatabase):
    """A real store that counts the reads a rung drives through it."""

    def __init__(self) -> None:
        super().__init__()
        self.searches = 0

    async def search(self, query: Any) -> list[Record]:
        self.searches += 1
        return await super().search(query)

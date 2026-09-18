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

from typing import Any

import pytest

from dataknobs_common.capabilities import Capability, CapabilityNotSupportedError
from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.ontology import OntologyConfig
from dataknobs_common.records import Record

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.ontology import EntityProjection, OntologyRegistry, RecordEntitySource
from dataknobs_data.query import Filter, Operator, Query

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


async def test_the_bulk_origin_member_refuses_because_its_answer_cannot_be_built() -> None:
    """A source that cannot run says so -- including when the reason is the protocol.

    ``fetch_origins`` is declared ``dict[SourceRef, Record]`` and ``SourceRef``
    is deliberately unhashable, so no non-empty answer exists to return. Every
    implementation before this one answered ``{}`` because it could not reach
    an origin at all; this is the first that can, and answering ``{}`` here
    would be indistinguishable from *these refs reached no rows*.
    """
    registry, ontology = await _bound(await _store({"sku": "sku-1", "title": "Beagle"}))
    try:
        entity = await ontology.entity("sku-1")
        assert entity is not None and entity.source is not None
        assert await ontology.entities.fetch_origins([]) == {}
        with pytest.raises(OperationError) as excinfo:
            await ontology.entities.fetch_origins([entity.source])
        assert "fetch_origin" in str(excinfo.value)
        assert excinfo.value.context["source_id"] == "products"
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

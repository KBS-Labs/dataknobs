"""An ontology reads a vocabulary in place, from a native Postgres table it does not own.

A ``database:`` block declares the table's columns and a scope, so one tenant's
categories out of a table holding every tenant's are the vocabulary, with no
view in the owner's database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from _helpdesk import (
    ACME,
    BILLING,
    CATEGORY_FIELDS,
    HARDWARE,
    LAPTOPS,
    PRINTERS,
    SOFTWARE,
    ZENITH,
    helpdesk_schema,
    native_config,
)

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import OntologyConfig
from dataknobs_common.testing import requires_postgres
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.factory import async_database_factory
from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.query import Query
from dataknobs_data.records import Record

pytestmark = requires_postgres

CATEGORY_COLUMNS = ("id", "tenant_id", "name", "parent_id", "aliases")


@pytest.fixture(scope="module")
def pg(
    ensure_postgres_ready: None, postgres_connection_params: dict[str, Any]
) -> Iterator[tuple[dict[str, Any], str]]:
    with helpdesk_schema(postgres_connection_params) as schema:
        yield postgres_connection_params, schema


def block(pg: tuple[dict[str, Any], str], tenant: str, **overrides: object) -> dict[str, Any]:
    """A ``database:`` block for one tenant's slice of ``categories``. The table is the projection's."""
    params, schema = pg
    config = native_config(
        params,
        schema,
        "categories",
        CATEGORY_FIELDS,
        scope=[{"field": "tenant_id", "operator": "=", "value": tenant}],
        **overrides,
    )
    del config["table"]
    return config


def document(
    ontology_id: str,
    database: dict[str, Any] | None = None,
    *,
    surface_forms: dict[str, Any] | None = None,
) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "table": "categories",
        "id": "id",
        "name": "name",
        "type": {"const": "Category"},
        "aliases": {"column": "aliases"},
    }
    columns = list(CATEGORY_COLUMNS)
    if surface_forms is not None:
        projection["surface_forms"] = surface_forms
        columns += [surface_forms["form"], surface_forms["entity"]]
    source: dict[str, Any] = {
        "id": "categories",
        "kind": "record",
        "entity_projection": projection,
        "schema": [{"name": name, "type": "string"} for name in columns],
    }
    if database is not None:
        source["database"] = database
    return {
        "id": ontology_id,
        "version": "1.0",
        "entity_types": [{"id": "Category", "name": "Category"}],
        "relation_types": [{"id": "within", "transitive": True}],
        "sources": [source],
        "taxonomies": [
            {
                "id": "category_tree",
                "kind": "column",
                "source": "categories",
                "parent_key": "parent_id",
                "relation": "within",
            }
        ],
    }


FORMS = {"table": "category_forms", "form": "folded_form", "entity": "entity"}


@pytest.fixture
async def entities(pg: tuple[dict[str, Any], str]) -> AsyncIterator[Any]:
    """The acme slice of ``categories``, opened directly, for a registry to be handed."""
    params, schema = pg
    db = async_database_factory.create(
        **native_config(params, schema, "categories", CATEGORY_FIELDS)
    )
    await db.connect()
    try:
        yield db
    finally:
        await db.close()


async def _forms_for(*rows: tuple[str, object]) -> AsyncMemoryDatabase:
    forms = AsyncMemoryDatabase()
    for form, entity in rows:
        await forms.create(Record({"folded_form": form, "entity": str(entity)}))
    return forms


async def test_two_scopes_of_one_native_table_are_two_vocabularies(
    pg: tuple[dict[str, Any], str],
) -> None:
    """Each tenant's categories, from one table, with no view and no refusal.

    Two bindings over one store are refused when nothing narrows a read to
    either; a scope narrows every read, and each scope is a store of its own.
    """
    registry = OntologyRegistry()
    try:
        acme = await registry.load(document("acme_categories", block(pg, ACME)))
        zenith = await registry.load(document("zenith_categories", block(pg, ZENITH)))

        assert await acme.entity(str(LAPTOPS)) is not None
        assert await acme.entity(str(BILLING)) is None
        assert await zenith.entity(str(BILLING)) is not None
        assert await zenith.entity(str(LAPTOPS)) is None
        acme_tree = acme.taxonomy("category_tree").structure
        zenith_tree = zenith.taxonomy("category_tree").structure
        assert list(await acme_tree.parents(str(LAPTOPS))) == [str(HARDWARE)]
        assert list(await acme_tree.roots()) == [str(HARDWARE)]
        assert not await zenith_tree.contains(str(LAPTOPS))
        assert list(await zenith_tree.roots()) == []
    finally:
        await registry.close()


async def test_a_native_block_cannot_also_describe_a_forms_table(
    pg: tuple[dict[str, Any], str],
) -> None:
    """One block's columns and scope describe one table; the forms table would inherit them."""
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError) as caught:
            await registry.load(document("acme_categories", block(pg, ACME), surface_forms=FORMS))
        message = str(caught.value)
        assert "'categories'" in message and "category_forms" in message
        assert "forms_database" in message
    finally:
        await registry.close()


async def test_a_mistake_in_a_native_block_names_the_ontology_and_binding(
    pg: tuple[dict[str, Any], str],
) -> None:
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError) as caught:
            await registry.load(
                document("acme_categories", block(pg, ACME, id_column="owner_email"))
            )
        message = str(caught.value)
        assert "acme_categories" in message and "'categories'" in message
        assert "id_column" in message
        assert caught.value.context.get("ontology") == "acme_categories"
    finally:
        await registry.close()


async def test_a_surface_form_outside_the_scope_names_no_entity(entities: Any) -> None:
    """The forms store holds another tenant's form; the entity store cannot read its entity."""
    forms = await _forms_for(("notebooks", LAPTOPS), ("invoices", BILLING))
    registry = OntologyRegistry.from_components(
        config=OntologyConfig.from_dict(document("acme_categories", surface_forms=FORMS)),
        database=entities,
        forms_database=forms,
        normalizer=str.casefold,
    )
    async with registry:
        ontology = await registry.load()
        assert await ontology.by_surface_form("NOTEBOOKS") == frozenset({str(LAPTOPS)})
        assert await ontology.by_surface_form("invoices") == frozenset()


async def test_an_ontology_reads_one_tenants_categories_in_place(entities: Any) -> None:
    """A ``record`` source and a ``column`` taxonomy, read in place from native columns.

    The folded surface forms live in a store the consumer owns, a different
    handle, because the fold is the consumer's and the table is not.
    """
    forms = AsyncMemoryDatabase()
    for record in await entities.search(Query()):
        for form in {record.get_value("name"), *record.get_value("aliases")}:
            await forms.create(
                Record({"folded_form": str(form).casefold(), "entity": record.storage_id})
            )
    registry = OntologyRegistry.from_components(
        config=OntologyConfig.from_dict(document("helpdesk", surface_forms=FORMS)),
        database=entities,
        forms_database=forms,
        normalizer=str.casefold,
    )
    async with registry:
        ontology = await registry.load()
        laptops = await ontology.entity(str(LAPTOPS))
        billing = await ontology.entity(str(BILLING))
        by_name = await ontology.entity("Laptops")
        found = await ontology.by_surface_form("NOTEBOOKS")
        tree = ontology.taxonomy("category_tree")
        parents = await tree.structure.parents(str(PRINTERS))
        roots = await tree.structure.roots()
        software_in_tree = await tree.structure.contains(str(SOFTWARE))

    assert laptops is not None and laptops.name == "Laptops"
    assert billing is None, "another tenant's category is outside the scope, so not here"
    assert by_name is None, "a name is not an id, and asking by one is not an error"
    assert found == frozenset({str(LAPTOPS)})
    assert list(parents) == [str(HARDWARE)], "the parent edge is read off the native column"
    # A column taxonomy's nodes are the ends of its edges, so Software, with
    # no parent and no children, is a category but not in the tree.
    assert list(roots) == [str(HARDWARE)]
    assert not software_in_tree

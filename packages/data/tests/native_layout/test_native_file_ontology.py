"""An ontology reads a vocabulary in place, from a native table in somebody else's file.

The ``database:`` block names a SQLite or DuckDB file, declares the table's
columns, and scopes it to one tenant, as a Postgres block does. Nothing runs
but the engine in this process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _helpdesk import (
    ACME,
    BILLING,
    CATEGORIES,
    CATEGORY_FIELDS,
    HARDWARE,
    LAPTOPS,
    PRINTERS,
    ZENITH,
    Helpdesk,
    write_duckdb_file,
    write_sqlite_file,
)

from dataknobs_common.exceptions import ValidationError
from dataknobs_data.ontology import OntologyRegistry


@pytest.fixture(scope="module", params=["sqlite", "duckdb"])
def desk(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Helpdesk:
    directory: Path = tmp_path_factory.mktemp(f"vocabulary-{request.param}")
    if request.param == "sqlite":
        path = write_sqlite_file(directory / "helpdesk.db", (CATEGORIES,))
    else:
        path = write_duckdb_file(directory / "helpdesk.duckdb", (CATEGORIES,))
    return Helpdesk(str(request.param), {"path": str(path)})


def block(desk: Helpdesk, tenant: str, **overrides: object) -> dict[str, Any]:
    """A ``database:`` block for one tenant's slice of ``categories``. The table is the projection's."""
    config = desk.config(
        "categories",
        {name: CATEGORY_FIELDS[name] for name in ("id", "tenant_id", "name", "parent_id")},
        scope=[{"field": "tenant_id", "operator": "=", "value": tenant}],
        **overrides,
    )
    del config["table"]
    return config


def document(ontology_id: str, database: dict[str, Any], **projected: str) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "table": "categories",
        "id": "id",
        "name": "name",
        "type": {"const": "Category"},
        **projected,
    }
    columns = ["id", "tenant_id", "name", "parent_id", "owner_email"]
    return {
        "id": ontology_id,
        "version": "1.0",
        "entity_types": [{"id": "Category", "name": "Category"}],
        "relation_types": [{"id": "within", "transitive": True}],
        "sources": [
            {
                "id": "categories",
                "kind": "record",
                "entity_projection": projection,
                "schema": [{"name": name, "type": "string"} for name in columns],
                "database": database,
            }
        ],
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


async def test_two_scopes_of_one_native_table_are_two_vocabularies(desk: Helpdesk) -> None:
    registry = OntologyRegistry()
    try:
        acme = await registry.load(document("acme_categories", block(desk, ACME)))
        zenith = await registry.load(document("zenith_categories", block(desk, ZENITH)))

        laptops = await acme.entity(str(LAPTOPS))
        assert laptops is not None and laptops.name == "Laptops"
        assert await acme.entity(str(BILLING)) is None
        assert await zenith.entity(str(BILLING)) is not None
        assert await zenith.entity(str(LAPTOPS)) is None
        assert await acme.entity(str(PRINTERS)) is not None
        tree = acme.taxonomy("category_tree").structure
        assert list(await tree.parents(str(LAPTOPS))) == [str(HARDWARE)]
        assert list(await tree.roots()) == [str(HARDWARE)]
    finally:
        await registry.close()


async def test_a_projection_column_the_block_does_not_declare_is_refused(desk: Helpdesk) -> None:
    """Bug: the projection was checked against the binding's ``schema:`` rows
    only. A native read selects only the columns its block declares, so a
    projected column the block left out read as missing on every entity --
    ``owner_email`` here, listed in the rows but not in the block -- rather
    than being refused when the ontology loaded.
    """
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError) as caught:
            await registry.load(
                document("acme_categories", block(desk, ACME), description="owner_email")
            )
        message = str(caught.value)
        assert "'categories'" in message and "owner_email" in message
        assert caught.value.context.get("binding") == "categories"
    finally:
        await registry.close()


async def test_a_taxonomy_parent_key_the_block_does_not_declare_is_refused(
    desk: Helpdesk,
) -> None:
    """The axis reads its parent key off the same handle, so the same check holds."""
    database = desk.config(
        "categories",
        {name: CATEGORY_FIELDS[name] for name in ("id", "tenant_id", "name")},
        scope=[{"field": "tenant_id", "operator": "=", "value": ACME}],
    )
    del database["table"]
    registry = OntologyRegistry()
    try:
        with pytest.raises(ValidationError) as caught:
            await registry.load(document("acme_categories", database))
        assert "parent_id" in str(caught.value)
    finally:
        await registry.close()

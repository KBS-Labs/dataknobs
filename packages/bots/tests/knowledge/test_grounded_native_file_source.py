"""A ``database`` grounded source reads a table in somebody else's SQLite or DuckDB file.

The source's options carry what a native config does -- ``layout: native``,
the key column, a scope -- and its ``schema:`` declares the columns the backend
reads. The file is the owner's: reading it leaves it as it was.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.knowledge.sources.factory import _create_database_source
from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig
from dataknobs_data.sources.base import RetrievalIntent

WIDGET, GADGET, LEDGER = uuid.UUID(int=1), uuid.UUID(int=2), uuid.UUID(int=3)

ROWS = [
    (str(WIDGET), "acme", "Widget recall", "A widget was recalled.", "hardware", "n"),
    (str(GADGET), "acme", "Gadget refund", "A gadget was refunded.", "billing", "n"),
    (str(LEDGER), "zenith", "Widget ledger", "Zenith's widget ledger.", "billing", "n"),
]

#: The columns the source is told about: not ``internal_note``.
FIELDS = [
    {"name": "id", "type": "string", "sql_type": "uuid"},
    {"name": "tenant_id", "type": "string"},
    {"name": "title", "type": "string"},
    {"name": "summary", "type": "text"},
    {"name": "dept", "type": "string", "enum": ["hardware", "billing"]},
]

DDL = (
    "CREATE TABLE cases (id {key} PRIMARY KEY, tenant_id {text} NOT NULL, title {text} NOT NULL, "
    "summary {text} NOT NULL, dept {text} NOT NULL, internal_note {text})"
)


def _write(engine: str, directory: Path) -> Path:
    if engine == "sqlite":
        path = directory / "cases.db"
        conn = sqlite3.connect(path)
        conn.execute(DDL.format(key="TEXT", text="TEXT"))
        conn.executemany("INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?)", ROWS)
        conn.commit()
        conn.close()
        return path
    duckdb = pytest.importorskip("duckdb")
    path = directory / "cases.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(DDL.format(key="UUID", text="VARCHAR"))
    conn.executemany("INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?)", ROWS)
    conn.close()
    return path


@pytest.fixture(params=["sqlite", "duckdb"])
def cases(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[str, Path]:
    return str(request.param), _write(str(request.param), tmp_path)


def _config(engine: str, path: Path, **overrides: Any) -> GroundedSourceConfig:
    options: dict[str, Any] = {
        "backend": engine,
        "path": str(path),
        "table": "cases",
        "layout": "native",
        "id_column": "id",
        "scope": [{"field": "tenant_id", "operator": "=", "value": "acme"}],
        "content_field": "summary",
        "text_search_fields": ["title", "summary"],
        "schema": {"fields": FIELDS},
        **overrides,
    }
    return GroundedSourceConfig(name="case_studies", source_type="database", options=options)


async def test_a_source_reads_one_tenants_rows_from_a_native_table(
    cases: tuple[str, Path],
) -> None:
    engine, path = cases
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    source = await _create_database_source(_config(engine, path))
    try:
        found = await source.query(RetrievalIntent(text_queries=["Widget"]))
        billing = await source.query(RetrievalIntent(filters={"case_studies": {"dept": "billing"}}))
    finally:
        await source.close()

    assert [r.content for r in found] == ["A widget was recalled."], "zenith's row is out of scope"
    assert [r.content for r in billing] == ["A gadget was refunded."]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


async def test_the_filter_schema_is_the_declared_columns(cases: tuple[str, Path]) -> None:
    engine, path = cases
    source = await _create_database_source(_config(engine, path))
    await source.close()

    fields = source.get_schema().fields
    assert "internal_note" not in fields
    assert fields["dept"]["enum"] == ["hardware", "billing"]


async def test_a_native_source_over_a_missing_file_creates_none(tmp_path: Path) -> None:
    for engine in ("sqlite", "duckdb"):
        absent = tmp_path / "absent" / f"cases.{engine}"
        with pytest.raises(RuntimeError, match="layout: native"):
            await _create_database_source(_config(engine, absent))
        assert not absent.parent.exists()

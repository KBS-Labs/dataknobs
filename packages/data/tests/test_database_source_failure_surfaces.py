"""A store the source cannot reach must not read as a store with nothing in it.

``DatabaseSource`` wrapped each of its ``db.search`` calls and returned an
empty list after logging, so every way of failing arrived at the caller as
"no matching records". The two are not the same answer: one says the query
found nothing, the other says the query never ran.

The caller already owns the degradation policy -- the grounded retrieval
loop guards each source and skips one that raises -- so the swallow added
no resilience. What it added was concealment: the guard never fired, and
the source was recorded as having answered.

These pin both search paths against a real backend that cannot serve, which
is the state the defect was found in: a factory-built database that was
never connected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_data import Record, async_database_factory
from dataknobs_data.fields import FieldType
from dataknobs_data.schema import DatabaseSchema
from dataknobs_data.sources.base import RetrievalIntent
from dataknobs_data.sources.database import DatabaseSource

SCHEMA = DatabaseSchema.create(title=FieldType.STRING, summary=FieldType.TEXT)


def _unreachable_source(path: Path) -> DatabaseSource:
    """A source over a real database that was never connected.

    Every backend needing a connection raises on each query until it has
    one. Nothing here is simulated: the database is the one the factory
    builds, and the fault is the one it really raises.
    """
    db = async_database_factory.create(backend="sqlite", path=str(path), table="cases")
    return DatabaseSource(
        db=db,
        schema=SCHEMA,
        name="cases",
        content_field="summary",
        text_search_fields=["title", "summary"],
    )


async def _reachable_source(path: Path) -> DatabaseSource:
    """The same source over the same backend, connected and populated."""
    db = async_database_factory.create(backend="sqlite", path=str(path), table="cases")
    await db.connect()
    db.set_schema(SCHEMA)
    await db.create(Record({"title": "Widget recall", "summary": "A widget was recalled."}))
    return DatabaseSource(
        db=db,
        schema=SCHEMA,
        name="cases",
        content_field="summary",
        text_search_fields=["title", "summary"],
    )


async def test_an_unreachable_store_does_not_read_as_an_empty_one(tmp_path: Path) -> None:
    """The text-search path reports rather than returning no records."""
    source = _unreachable_source(tmp_path / "cases.db")

    with pytest.raises(RuntimeError, match="not connected"):
        await source.query(RetrievalIntent(text_queries=["Widget"]))


async def test_the_structural_only_path_reports_too(tmp_path: Path) -> None:
    """An intent with no text queries takes the other search call.

    Both were guarded, so fixing only the one the defect was found through
    would leave an intent carrying filters alone still reading as empty.
    """
    source = _unreachable_source(tmp_path / "cases.db")

    with pytest.raises(RuntimeError, match="not connected"):
        await source.query(RetrievalIntent())


async def test_a_reachable_store_with_no_match_still_answers_empty(tmp_path: Path) -> None:
    """The distinction the fix exists to draw, from the other side.

    A connected store that simply holds nothing matching is not a failure,
    and must keep returning an empty result rather than raising.
    """
    source = await _reachable_source(tmp_path / "cases.db")

    assert await source.query(RetrievalIntent(text_queries=["nothing matches this"])) == []
    assert [r.content for r in await source.query(RetrievalIntent(text_queries=["Widget"]))] == [
        "A widget was recalled."
    ]

    await source.close()

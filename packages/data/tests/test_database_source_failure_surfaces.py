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
from typing import Any

import pytest

from dataknobs_data import Record, async_database_factory
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
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


class _FailsOnNthSearch(AsyncSQLiteDatabase):
    """A real SQLite database whose ``search`` fails on one nominated call.

    Everything here is the real backend: the record round-trips through
    real SQL, and ``_text_or_search`` runs its real loop. Only one call is
    made to raise, because the condition being pinned -- one
    ``(query, field)`` combination failing after another has already
    matched -- has no natural cause that SQLite can produce. A filter
    naming a column the table does not have does not raise; it matches
    nothing and returns an empty list (see the note in the test below).
    """

    def __init__(self, *args: Any, fail_on_call: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fail_on_call = fail_on_call
        self.search_calls = 0

    async def search(self, query: Any) -> list[Record]:
        self.search_calls += 1
        if self.search_calls == self._fail_on_call:
            raise RuntimeError("injected: this combination's search failed")
        return await super().search(query)


async def test_a_partly_failing_text_search_does_not_return_the_part_that_worked(
    tmp_path: Path,
) -> None:
    """A short answer that looks complete is the same defect in miniature.

    The text-search path ORs across every ``(query, field)`` combination
    and dedups. A failing combination used to be skipped, so the call
    returned what the *others* had matched -- a subset indistinguishable
    from the whole, with nothing to say which records were missing. For a
    grounded source that means an LLM answering confidently from evidence
    that was quietly narrowed, which is the failure grounding exists to
    prevent. So the whole call raises instead.

    How reachable this is depends on the backend, and less than it looks.
    The obvious cause -- a ``text_search_fields`` naming a column that is
    not there -- does not produce it: such a filter matches nothing and
    returns an empty list rather than raising, so it under-retrieves
    silently and this change does not help it. What remains are
    backend-specific per-field faults (a typed column rejecting an
    operator, a per-field permission error, a timeout tripping on one
    expensive field), which is why the fault here is injected rather than
    provoked.
    """
    db = _FailsOnNthSearch(path=str(tmp_path / "cases.db"), table="cases", fail_on_call=2)
    await db.connect()
    db.set_schema(SCHEMA)
    await db.create(Record({"title": "Widget recall", "summary": "A widget was recalled."}))

    source = DatabaseSource(
        db=db,
        schema=SCHEMA,
        name="cases",
        content_field="summary",
        text_search_fields=["title", "summary"],
    )

    with pytest.raises(RuntimeError, match="injected"):
        await source.query(RetrievalIntent(text_queries=["Widget"]))

    # The first combination ran and matched before the second failed, so
    # there really were partial results to discard. Without this the test
    # is satisfied by a source that failed on its first call, which is
    # the sibling tests' case and pins nothing new.
    assert db.search_calls == 2

    await source.close()

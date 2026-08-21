"""One source that cannot serve must not pass for one that found nothing.

The grounded retrieval loop guards each source: one that raises is logged
with its traceback and skipped, and the others still contribute. That
policy is what makes it right for a source to report a failure rather than
absorb it -- and it was unreachable through a ``DatabaseSource``, which
returned an empty list for every way of failing. The guard never fired, and
the strategy recorded the source as having answered with nothing.

These use two real database sources over real backends, one connected and
one not, because the distinction under test is exactly the one a stub would
have to assert into existence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dataknobs_bots.reasoning.grounded import GroundedReasoning
from dataknobs_bots.reasoning.grounded_config import GroundedReasoningConfig
from dataknobs_data import Record, async_database_factory
from dataknobs_data.fields import FieldType
from dataknobs_data.schema import DatabaseSchema
from dataknobs_data.sources.base import RetrievalIntent
from dataknobs_data.sources.database import DatabaseSource

SCHEMA = DatabaseSchema.create(title=FieldType.STRING, summary=FieldType.TEXT)

#: The module holding the per-source guard, so a warning from anywhere else
#: cannot satisfy the assertions below.
STRATEGY_LOGGER = "dataknobs_bots.reasoning.grounded"


def _source(path: Path, name: str) -> DatabaseSource:
    db = async_database_factory.create(backend="sqlite", path=str(path), table="cases")
    return DatabaseSource(
        db=db,
        schema=SCHEMA,
        name=name,
        content_field="summary",
        text_search_fields=["title", "summary"],
    )


async def _connected_source(path: Path, name: str) -> DatabaseSource:
    source = _source(path, name)
    await source._db.connect()
    source._db.set_schema(SCHEMA)
    await source._db.create(Record({"title": "Widget recall", "summary": "A widget was recalled."}))
    return source


async def test_a_failing_source_is_skipped_and_the_others_still_answer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The working source contributes; the broken one is absent, not empty.

    Absent and empty are different downstream: an empty list records a
    source that was consulted and had nothing, which is what a misconfigured
    source looked like on every turn.
    """
    strategy = GroundedReasoning(config=GroundedReasoningConfig())
    strategy.add_source(_source(tmp_path / "broken.db", "cases"))
    strategy.add_source(await _connected_source(tmp_path / "archive.db", "archive"))

    with caplog.at_level(logging.WARNING, logger=STRATEGY_LOGGER):
        results = await strategy._retrieve_from_sources(RetrievalIntent(text_queries=["Widget"]))
    await strategy.close()

    assert "cases" not in results
    assert [r.content for r in results["archive"]] == ["A widget was recalled."]

    named = [r for r in caplog.records if "cases" in r.getMessage()]
    assert named, [r.getMessage() for r in caplog.records]
    assert named[0].exc_info is not None, "the guard reports without the cause"

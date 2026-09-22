# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Reproduce-first tests for ``ETLConfig.source_query``.

The field shipped typed ``str | None`` and defaulting to the raw SQL string
``"SELECT * FROM source_table"``, while its only consumer hands it to
``AsyncDatabase.stream_read``, whose parameter is a :class:`Query`. No SQL
string works on any of the seven backends, so the shipped default could not
run --- and every ETL test in the tree passed ``source_query=None`` to get
around it.

Real constructs only: file-backed ``AsyncDatabase`` source and target.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataknobs_data import AsyncDatabase, Operator, Query, Record
from dataknobs_fsm.core.exceptions import InvalidConfigurationError
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode

ROWS = [
    {"id": "1", "name": "A", "status": "active"},
    {"id": "2", "name": "B", "status": "retired"},
    {"id": "3", "name": "C", "status": "active"},
]


async def _seed(path: str, rows: list[dict]) -> None:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        for row in rows:
            await db.upsert(row["id"], Record(dict(row)))
    finally:
        await db.close()


async def _loaded_ids(path: str) -> set[str]:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        return {record.to_dict()["id"] async for record in db.stream_read(Query())}
    finally:
        await db.close()


def _config(src: str, tgt: str, **overrides) -> ETLConfig:
    return ETLConfig(
        source_db={"type": "file", "path": src},
        target_db={"type": "file", "path": tgt},
        target_table="records",
        key_columns=["id"],
        **overrides,
    )


async def test_the_shipped_default_extracts_every_row(tmp_path: Path) -> None:
    """An ETL that names no query reads the whole source.

    The reproduce: with the default left alone, ``run()`` handed a ``str``
    to ``stream_read`` and died inside the backend rather than extracting
    anything.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(src, ROWS)

    metrics = await DatabaseETL(_config(src, tgt)).run()

    assert metrics["errors"] == 0, metrics
    assert metrics["loaded"] == 3, metrics
    assert await _loaded_ids(tgt) == {"1", "2", "3"}


async def test_a_query_narrows_the_extraction(tmp_path: Path) -> None:
    """A ``Query`` is what the field is for: it filters at the source."""
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(src, ROWS)

    config = _config(src, tgt, source_query=Query().filter("status", Operator.EQ, "active"))
    metrics = await DatabaseETL(config).run()

    assert metrics["loaded"] == 2, metrics
    assert await _loaded_ids(tgt) == {"1", "3"}


async def test_a_dict_query_narrows_the_extraction(tmp_path: Path) -> None:
    """The config-authored form: the same query as a plain dict.

    This is the shape a YAML/JSON pipeline definition can carry, so it is
    the one that has to work through ``from_config``.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(src, ROWS)

    etl = DatabaseETL.from_config(
        {
            "source_db": {"type": "file", "path": src},
            "target_db": {"type": "file", "path": tgt},
            "target_table": "records",
            "key_columns": ["id"],
            "source_query": {"filters": [{"field": "status", "operator": "=", "value": "active"}]},
        }
    )
    metrics = await etl.run()

    assert metrics["loaded"] == 2, metrics
    assert await _loaded_ids(tgt) == {"1", "3"}


def test_a_query_survives_the_json_round_trip() -> None:
    """``ETLConfig``'s round-trip is a tested contract; the query is in it.

    ``dataclasses.asdict`` explodes a ``Query`` (itself a dataclass) into its
    raw attribute names with live ``Operator`` members --- a shape
    ``Query.from_dict`` raises on, and one ``json.dumps`` refuses. So the
    config
    holds the serializable form.
    """
    cfg = ETLConfig(
        source_db={"backend": "memory"},
        target_db={"backend": "memory"},
        source_query=Query().filter("status", Operator.EQ, "active").limit(5),
    )

    restored = ETLConfig.from_dict(json.loads(json.dumps(cfg.to_json_dict())))

    assert restored == cfg


async def test_incremental_mode_keeps_the_configured_query(tmp_path: Path) -> None:
    """An incremental run narrows *further*; it does not discard the query.

    The incremental filter was built from scratch, so a pipeline that named
    a source query and ran incrementally silently extracted rows the query
    excluded.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(
        src,
        [
            {"id": "1", "name": "A", "status": "active", "updated_at": "2026-01-02"},
            {"id": "2", "name": "B", "status": "retired", "updated_at": "2026-01-02"},
            {"id": "3", "name": "C", "status": "active", "updated_at": "2025-01-01"},
        ],
    )

    config = _config(
        src,
        tgt,
        mode=ETLMode.INCREMENTAL,
        source_query=Query().filter("status", Operator.EQ, "active"),
    )
    etl = DatabaseETL(config)
    # The position a previous run would have left behind.
    etl._watermark = "2026-01-01"

    metrics = await etl.run()

    # id 2 is excluded by the configured query, id 3 by the incremental
    # filter. Only id 1 satisfies both.
    assert await _loaded_ids(tgt) == {"1"}, metrics


def test_a_sql_string_is_refused_at_construction() -> None:
    """The field's own former default, named as the thing it cannot be.

    Nothing ``source_query`` reaches takes SQL, so a string is rejected
    where it is written rather than inside whichever backend is configured.
    """
    with pytest.raises(InvalidConfigurationError, match="not SQL"):
        ETLConfig(
            source_db={"backend": "memory"},
            target_db={"backend": "memory"},
            source_query="SELECT * FROM source_table",  # type: ignore[arg-type]
        )

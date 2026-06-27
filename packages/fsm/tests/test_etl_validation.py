"""Reproduce-first tests: the ETL ``validate`` stage must actually gate records.

``DatabaseETL`` accepts a ``validation_schema`` but historically never read it:
the ``validate`` state was an unconditional passthrough, so a row violating the
configured schema loaded into the target exactly like a valid one. A consumer
who set ``validation_schema`` got a silent no-op — the run *looked* validated.

These tests assert that an invalid row is diverted to a non-loading terminal and
counted separately from a hard error. They FAIL against the unwired passthrough
(all rows load; there is no ``rejected`` metric) and PASS once ``validate`` is a
real arc-condition gate.

Real constructs only: file-backed ``AsyncDatabase`` source/target, real
``DatabaseETL`` build — no mocks. The three accepted validation-spec forms
(friendly dict schema / library ``IValidationFunction`` / callable predicate)
are each exercised end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.testing import assert_no_blocking
from dataknobs_data import AsyncDatabase, Query, Record
from dataknobs_fsm.core.exceptions import ETLError
from dataknobs_fsm.functions.library.validators import RangeValidator
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode


async def _seed_source(path: str, rows: list[dict]) -> None:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        for row in rows:
            await db.upsert(row["id"], Record(dict(row)))
    finally:
        await db.close()


async def _read_target(path: str) -> list[dict]:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        return [record.to_dict() async for record in db.stream_read(Query())]
    finally:
        await db.close()


def _build_etl(
    src: str,
    tgt: str,
    *,
    validation_schema: Any = None,
    reject_counts_as_error: bool = False,
) -> DatabaseETL:
    # ``reject_counts_as_error`` is forwarded only when non-default so the
    # core gate tests construct against the pre-fix config too (reproduce-first:
    # they must fail on "validation_schema ignored", not on an unknown kwarg).
    extra = (
        {"reject_counts_as_error": True} if reject_counts_as_error else {}
    )
    return DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            validation_schema=validation_schema,
            **extra,
        )
    )


# Rows: Bob (age 15) violates "age >= 18"; Alice and Carol pass.
_MIXED_ROWS = [
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 15},
    {"id": "3", "name": "Carol", "age": 40},
]


@pytest.mark.asyncio
async def test_dict_schema_gate_rejects_invalid_rows(tmp_path: Path) -> None:
    """A friendly dict ``validation_schema`` must divert violating rows."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, _MIXED_ROWS)

    etl = _build_etl(src, tgt, validation_schema={"age": {"type": "int", "min": 18}})
    metrics = await etl.run()

    rows = await _read_target(tgt)
    persisted_ids = {r["id"] for r in rows}
    assert persisted_ids == {"1", "3"}, (
        f"validation_schema was ignored — invalid row loaded: {persisted_ids}"
    )
    assert metrics["loaded"] == 2, metrics
    assert metrics["rejected"] == 1, metrics
    assert metrics["errors"] == 0, metrics


@pytest.mark.asyncio
async def test_passthrough_parity_when_no_schema(tmp_path: Path) -> None:
    """No ``validation_schema`` → every row loads; ``rejected`` stays zero."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, _MIXED_ROWS)

    etl = _build_etl(src, tgt, validation_schema=None)
    metrics = await etl.run()

    rows = await _read_target(tgt)
    assert {r["id"] for r in rows} == {"1", "2", "3"}, rows
    assert metrics["loaded"] == 3, metrics
    assert metrics["rejected"] == 0, metrics
    assert metrics["errors"] == 0, metrics


@pytest.mark.asyncio
async def test_library_validation_function_form(tmp_path: Path) -> None:
    """A library ``IValidationFunction`` gates rows identically to the dict form."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, _MIXED_ROWS)

    etl = _build_etl(
        src, tgt, validation_schema=RangeValidator({"age": {"min": 18}})
    )
    metrics = await etl.run()

    assert {r["id"] for r in await _read_target(tgt)} == {"1", "3"}
    assert metrics["loaded"] == 2, metrics
    assert metrics["rejected"] == 1, metrics


@pytest.mark.asyncio
async def test_callable_predicate_form(tmp_path: Path) -> None:
    """A plain ``record -> bool`` callable gates rows."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, _MIXED_ROWS)

    etl = _build_etl(
        src, tgt, validation_schema=lambda r: r.get("age", 0) >= 18
    )
    metrics = await etl.run()

    assert {r["id"] for r in await _read_target(tgt)} == {"1", "3"}
    assert metrics["loaded"] == 2, metrics
    assert metrics["rejected"] == 1, metrics


def _ten_rows_three_invalid() -> list[dict]:
    rows = [{"id": str(i), "name": f"n{i}", "age": 25} for i in range(1, 8)]
    rows += [{"id": str(i), "name": f"n{i}", "age": 10} for i in range(8, 11)]
    return rows


@pytest.mark.asyncio
async def test_reject_does_not_trip_error_threshold_by_default(
    tmp_path: Path,
) -> None:
    """Rejections above ``error_threshold`` do not abort the run by default."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, _ten_rows_three_invalid())  # 30% rejected (> 5%)

    etl = _build_etl(src, tgt, validation_schema={"age": {"type": "int", "min": 18}})
    metrics = await etl.run()  # must NOT raise ETLError

    assert metrics["loaded"] == 7, metrics
    assert metrics["rejected"] == 3, metrics
    assert metrics["errors"] == 0, metrics


@pytest.mark.asyncio
async def test_reject_counts_as_error_flips_threshold(tmp_path: Path) -> None:
    """``reject_counts_as_error=True`` makes excess rejections abort the run."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, _ten_rows_three_invalid())  # 30% rejected (> 5%)

    etl = _build_etl(
        src,
        tgt,
        validation_schema={"age": {"type": "int", "min": 18}},
        reject_counts_as_error=True,
    )
    with pytest.raises(ETLError):
        await etl.run()


@pytest.mark.asyncio
async def test_no_double_counting_and_non_blocking(tmp_path: Path) -> None:
    """``extracted == loaded + rejected + errors`` and the run never blocks."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, _MIXED_ROWS)

    etl = _build_etl(src, tgt, validation_schema={"age": {"type": "int", "min": 18}})
    with assert_no_blocking():
        metrics = await etl.run()

    assert metrics["extracted"] == (
        metrics["loaded"] + metrics["rejected"] + metrics["errors"]
    ), metrics

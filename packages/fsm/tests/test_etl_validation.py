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
from dataknobs_fsm.core.exceptions import ETLError, InvalidConfigurationError
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


@pytest.mark.asyncio
async def test_gate_infrastructure_error_counts_as_error_not_reject(
    tmp_path: Path,
) -> None:
    """A gate raising an UNEXPECTED error makes the record an *error*, not a reject.

    Reproduce-first: before the engine distinguished reject from error, a gate
    predicate that *raised* (a validator bug, a reference table down) was
    swallowed to ``False``, so the record was silently routed to the
    ``rejected`` terminal — counted as a clean data-quality drop with
    ``errors == 0``, hiding the failure as a green run. Now an unexpected
    exception surfaces as a record error.

    ``error_threshold`` is set to 1.0 so the run completes and the metrics can
    be inspected (otherwise the errors would correctly abort the run). Against
    the old behaviour this asserts ``errors == 3`` / ``rejected == 0`` and
    FAILS (it would have been ``rejected == 3`` / ``errors == 0``).
    """
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, _MIXED_ROWS)

    def exploding_gate(_record: dict) -> bool:
        # Simulates an infrastructure failure inside the gate (e.g. a reference
        # lookup against a DB that is down) — NOT a "record is invalid" verdict.
        raise RuntimeError("reference table unavailable")

    etl = DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            validation_schema=exploding_gate,
            error_threshold=1.0,  # don't abort; inspect the metrics
        )
    )
    metrics = await etl.run()

    assert metrics["errors"] == 3, metrics
    assert metrics["rejected"] == 0, metrics
    assert metrics["loaded"] == 0, metrics
    assert await _read_target(tgt) == [], "no row should load when the gate errors"


@pytest.mark.asyncio
async def test_resource_backed_gate_via_validation_resources(tmp_path: Path) -> None:
    """A resource-reading async ``validation_schema`` reaches its declared resource.

    Reproduce-first for the resource-backed gate via ``DatabaseETL``: the
    auto-built ``valid`` arc must carry the ``validation_resources`` binding, or
    the gate predicate's ``context.resources`` is empty and ``require_resource``
    raises — so (per the error-vs-reject contract) every record would *error*.
    With the binding wired, the async predicate validates each row against a
    reference-table resource: ids present in the reference pass; the rest are
    rejected. Also exercises the async-predicate ETL path end-to-end.
    """
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    ref = str(tmp_path / "reference.json")
    await _seed_source(src, _MIXED_ROWS)

    # Reference table: only ids "1" and "3" are allowed.
    ref_db = await AsyncDatabase.from_backend("file", {"type": "file", "path": ref})
    try:
        for rid in ("1", "3"):
            await ref_db.upsert(rid, Record({"id": rid}))
    finally:
        await ref_db.close()

    async def id_in_reference(record: dict, context: Any) -> bool:
        reference = context.require_resource("ref_db")
        rows = await reference.execute_query(Query())
        allowed = {row.get("id") for row in rows}
        return record.get("id") in allowed

    etl = DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            validation_schema=id_in_reference,
            validation_resources={
                "ref_db": {
                    "type": "async_database",
                    "config": {"type": "file", "path": ref},
                }
            },
        )
    )
    metrics = await etl.run()

    assert {r["id"] for r in await _read_target(tgt)} == {"1", "3"}, metrics
    assert metrics["loaded"] == 2, metrics
    assert metrics["rejected"] == 1, metrics
    assert metrics["errors"] == 0, metrics


def test_validation_resources_without_schema_is_rejected() -> None:
    """`validation_resources` with no `validation_schema` is a misconfiguration.

    The resources would be registered but never bound to a gate arc (which is
    wired only when `validation_schema` is set) — a silent no-op. Construction
    rejects it instead.
    """
    with pytest.raises(InvalidConfigurationError):
        ETLConfig(
            source_db={"type": "file", "path": "s.json"},
            target_db={"type": "file", "path": "t.json"},
            target_table="records",
            key_columns=["id"],
            validation_resources={
                "ref_db": {"type": "async_database",
                           "config": {"type": "file", "path": "r.json"}},
            },
        )

"""Reproduce-first tests: the ETL ``enrich`` stage must actually enrich records.

``DatabaseETL`` accepts ``enrichment_sources`` but historically never read them:
the ``enrich`` state was an unconditional passthrough, so a configured
enrichment (a computed field, a reference-table lookup) silently did nothing —
the run *looked* enriched (records route through the stage) but emitted
unenriched rows. Same "homework-assignment, not a tool" failure mode the
``validate`` gate fixed for validation.

These tests assert that each accepted enrichment spec form actually adds fields
to the loaded records, that a reference lookup resolves through an injected
async resource, that a failing enricher is counted as an error (not silently
dropped), and that the no-enrichment path stays byte-identical. They FAIL
against the unwired passthrough (enriched fields never appear in the target) and
PASS once ``enrich`` is a real per-record state transform.

Real constructs only: file-backed ``AsyncDatabase`` source/target **and** a
file-backed reference DB for lookups — no mocks. The four accepted spec forms
(field→value map / reference-table ``match`` lookup / ``ITransformFunction`` /
callable) are each exercised end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.testing import assert_no_blocking
from dataknobs_data import AsyncDatabase, Query, Record
from dataknobs_fsm.functions.library.transformers import DataEnricher
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode


async def _seed(path: str, rows: list[dict]) -> None:
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
    enrichment_sources: Any = None,
    enrichment_on_missing: str | None = None,
    error_threshold: float | None = None,
) -> DatabaseETL:
    # ``enrichment_on_missing`` is forwarded only when non-default so the core
    # enrich tests construct against the pre-fix config too (reproduce-first:
    # they must fail on "enrichment_sources ignored", not on an unknown kwarg).
    extra: dict[str, Any] = {}
    if enrichment_on_missing is not None:
        extra["enrichment_on_missing"] = enrichment_on_missing
    if error_threshold is not None:
        extra["error_threshold"] = error_threshold
    return DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            enrichment_sources=enrichment_sources,
            **extra,
        )
    )


_ROWS = [
    {"id": "1", "name": "Alice", "country_code": "US"},
    {"id": "2", "name": "Bo", "country_code": "CA"},
    {"id": "3", "name": "Carol", "country_code": "GB"},
]


@pytest.mark.asyncio
async def test_field_map_enrichment_static_and_callable(tmp_path: Path) -> None:
    """A field→value map adds static and computed fields to every record."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed(src, _ROWS)

    etl = _build_etl(
        src,
        tgt,
        enrichment_sources=[{"tier": "gold", "name_len": lambda r: len(r["name"])}],
    )
    metrics = await etl.run()

    rows = {r["id"]: r for r in await _read_target(tgt)}
    assert rows["1"]["tier"] == "gold", f"enrichment_sources ignored: {rows['1']}"
    assert rows["1"]["name_len"] == 5, rows["1"]
    assert rows["2"]["name_len"] == 2, rows["2"]
    assert metrics["loaded"] == 3, metrics
    assert metrics["errors"] == 0, metrics


@pytest.mark.asyncio
async def test_reference_table_lookup_by_match(tmp_path: Path) -> None:
    """A reference-table ``match`` lookup merges looked-up fields per record.

    Guards G2 (the enrich state sees the enrichment resource) and G3 (the
    resource is an async backend — exercised via ``await execute_query``).
    """
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    ref = str(tmp_path / "ref.json")
    await _seed(src, _ROWS)
    await _seed(
        ref,
        [
            {"id": "US", "code": "US", "name": "United States", "region": "NA"},
            {"id": "CA", "code": "CA", "name": "Canada", "region": "NA"},
            {"id": "GB", "code": "GB", "name": "United Kingdom", "region": "EU"},
        ],
    )

    etl = _build_etl(
        src,
        tgt,
        enrichment_sources=[
            {
                "database": {"type": "file", "path": ref},
                "match": {"country_code": "code"},
                "fields": ["region"],
            }
        ],
    )
    metrics = await etl.run()

    rows = {r["id"]: r for r in await _read_target(tgt)}
    assert rows["1"]["region"] == "NA", f"lookup not merged: {rows['1']}"
    assert rows["3"]["region"] == "EU", rows["3"]
    assert metrics["loaded"] == 3, metrics


@pytest.mark.asyncio
async def test_lookup_overwrite_policy(tmp_path: Path) -> None:
    """A looked-up field colliding with an existing key honors ``overwrite``."""
    src = str(tmp_path / "source.json")
    ref = str(tmp_path / "ref.json")
    await _seed(src, _ROWS)
    await _seed(
        ref,
        [
            {"id": "US", "code": "US", "name": "United States"},
            {"id": "CA", "code": "CA", "name": "Canada"},
            {"id": "GB", "code": "GB", "name": "United Kingdom"},
        ],
    )

    # overwrite=False (default): the record's own ``name`` survives the collision.
    tgt_keep = str(tmp_path / "keep.json")
    etl = _build_etl(
        src,
        tgt_keep,
        enrichment_sources=[
            {
                "database": {"type": "file", "path": ref},
                "match": {"country_code": "code"},
                "fields": ["name"],
            }
        ],
    )
    await etl.run()
    keep = {r["id"]: r for r in await _read_target(tgt_keep)}
    assert keep["1"]["name"] == "Alice", keep["1"]

    # overwrite=True: the looked-up ``name`` replaces the record's own.
    tgt_over = str(tmp_path / "over.json")
    etl = _build_etl(
        src,
        tgt_over,
        enrichment_sources=[
            {
                "database": {"type": "file", "path": ref},
                "match": {"country_code": "code"},
                "fields": ["name"],
                "overwrite": True,
            }
        ],
    )
    await etl.run()
    over = {r["id"]: r for r in await _read_target(tgt_over)}
    assert over["1"]["name"] == "United States", over["1"]


@pytest.mark.asyncio
async def test_on_missing_ignore_null_error(tmp_path: Path) -> None:
    """``enrichment_on_missing`` controls how an unmatched lookup is handled."""
    rows = [{"id": "1", "name": "Alice", "country_code": "ZZ"}]  # no ZZ in ref
    ref = str(tmp_path / "ref.json")
    await _seed(
        ref, [{"id": "US", "code": "US", "name": "United States", "region": "NA"}]
    )

    def source_with(sub: str) -> str:
        return str(tmp_path / f"src_{sub}.json")

    spec = {
        "database": {"type": "file", "path": ref},
        "match": {"country_code": "code"},
        "fields": ["region"],
    }

    # ignore (default): the unmatched row passes through unchanged and loads.
    s_ign = source_with("ign")
    t_ign = str(tmp_path / "t_ign.json")
    await _seed(s_ign, rows)
    etl = _build_etl(s_ign, t_ign, enrichment_sources=[spec])
    m = await etl.run()
    out = await _read_target(t_ign)
    assert m["loaded"] == 1 and "region" not in out[0], (m, out)

    # null: the requested fields are set to None.
    s_null = source_with("null")
    t_null = str(tmp_path / "t_null.json")
    await _seed(s_null, rows)
    etl = _build_etl(
        s_null, t_null, enrichment_sources=[spec], enrichment_on_missing="null"
    )
    m = await etl.run()
    out = await _read_target(t_null)
    assert m["loaded"] == 1 and out[0]["region"] is None, (m, out)

    # error: the unmatched row becomes a counted error (permissive threshold so
    # the all-error single-row run reports metrics instead of aborting).
    s_err = source_with("err")
    t_err = str(tmp_path / "t_err.json")
    await _seed(s_err, rows)
    etl = _build_etl(
        s_err,
        t_err,
        enrichment_sources=[spec],
        enrichment_on_missing="error",
        error_threshold=1.0,
    )
    m = await etl.run()
    assert m["errors"] == 1 and m["loaded"] == 0, m


@pytest.mark.asyncio
async def test_enrichment_failure_counts_as_error(tmp_path: Path) -> None:
    """A failing enricher counts as ``errors``; no new terminal/metric key."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed(src, _ROWS)

    def boom(_record: dict) -> dict:
        raise RuntimeError("enricher blew up")

    # Permissive threshold so the all-error run reports metrics rather than
    # aborting — the assertion is about *classification*, not the threshold.
    etl = _build_etl(src, tgt, enrichment_sources=[boom], error_threshold=1.0)
    metrics = await etl.run()

    assert metrics["errors"] == 3, metrics
    assert metrics["loaded"] == 0, metrics
    assert metrics["extracted"] == (
        metrics["loaded"] + metrics["rejected"] + metrics["errors"]
    ), metrics
    # Metric keys are unchanged vs the no-enrichment shape (no new terminal).
    assert set(metrics) == {
        "extracted",
        "transformed",
        "loaded",
        "rejected",
        "errors",
        "skipped",
    }, metrics


@pytest.mark.asyncio
async def test_passthrough_parity_when_no_enrichment(tmp_path: Path) -> None:
    """No ``enrichment_sources`` → every row loads unchanged (regression guard)."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed(src, _ROWS)

    etl = _build_etl(src, tgt, enrichment_sources=None)
    metrics = await etl.run()

    rows = {r["id"]: r for r in await _read_target(tgt)}
    assert set(rows) == {"1", "2", "3"}, rows
    assert "tier" not in rows["1"] and "region" not in rows["1"], rows["1"]
    assert metrics["loaded"] == 3, metrics


@pytest.mark.asyncio
async def test_instance_and_callable_forms(tmp_path: Path) -> None:
    """A pre-built ``DataEnricher`` and a bare callable both enrich identically."""
    src = str(tmp_path / "source.json")
    await _seed(src, _ROWS)

    # ITransformFunction instance (a richer library construct, overwrite=True).
    t_inst = str(tmp_path / "inst.json")
    etl = _build_etl(
        src,
        t_inst,
        enrichment_sources=[DataEnricher({"name": "REPLACED"}, overwrite=True)],
    )
    await etl.run()
    inst = {r["id"]: r for r in await _read_target(t_inst)}
    assert inst["1"]["name"] == "REPLACED", inst["1"]

    # Bare callable record -> dict.
    t_call = str(tmp_path / "call.json")
    etl = _build_etl(
        src,
        t_call,
        enrichment_sources=[lambda r: {**r, "upper": r["name"].upper()}],
    )
    await etl.run()
    call = {r["id"]: r for r in await _read_target(t_call)}
    assert call["1"]["upper"] == "ALICE", call["1"]


@pytest.mark.asyncio
async def test_lookup_run_is_non_blocking(tmp_path: Path) -> None:
    """An enriching run with a reference lookup never blocks the event loop.

    Pins the async-resource fix (G3): a sync ``database`` registration would
    block on the reference query.
    """
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    ref = str(tmp_path / "ref.json")
    await _seed(src, _ROWS)
    await _seed(
        ref,
        [
            {"id": "US", "code": "US", "region": "NA"},
            {"id": "CA", "code": "CA", "region": "NA"},
            {"id": "GB", "code": "GB", "region": "EU"},
        ],
    )

    etl = _build_etl(
        src,
        tgt,
        enrichment_sources=[
            {
                "database": {"type": "file", "path": ref},
                "match": {"country_code": "code"},
                "fields": ["region"],
            }
        ],
    )
    with assert_no_blocking():
        metrics = await etl.run()
    assert metrics["loaded"] == 3, metrics
    rows = {r["id"]: r for r in await _read_target(tgt)}
    assert rows["1"]["region"] == "NA", f"lookup not merged: {rows['1']}"


@pytest.mark.asyncio
async def test_api_enrichment_source_is_rejected(tmp_path: Path) -> None:
    """An ``api`` lookup source raises a clear error rather than silently no-op."""
    from dataknobs_common.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        ETLConfig(
            source_db={"type": "file", "path": "s"},
            target_db={"type": "file", "path": "t"},
            enrichment_sources=[{"api": {"url": "http://x"}, "match": {"a": "b"}}],
        )

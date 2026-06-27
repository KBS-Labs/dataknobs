"""Unit coverage for the shared enricher builder + lookup-merge enricher.

These exercise :func:`build_record_enricher`'s four dispatch branches, the
``merge_enrichment_field`` overwrite primitive, and
:class:`LookupMergeEnricher`'s match/miss behavior directly — the per-branch
verification the ETL integration tests (``test_etl_enrichment.py``) rely on but
do not isolate. Real constructs only: a real async memory database resource for
the lookup branch (no mocks).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from dataknobs_data import Record
from dataknobs_fsm.functions.base import TransformError
from dataknobs_fsm.functions.library.enrichers import (
    LookupMergeEnricher,
    build_record_enricher,
)
from dataknobs_fsm.functions.library.transformers import (
    DataEnricher,
    merge_enrichment_field,
)
from dataknobs_fsm.patterns.etl import _ETLEnrich
from dataknobs_fsm.resources.database import AsyncDatabaseResourceAdapter


async def _call(enricher: Any, record: dict, context: Any = None) -> dict:
    out = enricher(record, context)
    if inspect.isawaitable(out):
        out = await out
    return out


# --- merge_enrichment_field primitive -------------------------------------

def test_merge_primitive_overwrite_policy() -> None:
    rec = {"a": 1}
    assert merge_enrichment_field(rec, "b", 2, overwrite=False) is True
    assert rec["b"] == 2
    # present + not overwrite -> skipped
    assert merge_enrichment_field(rec, "a", 9, overwrite=False) is False
    assert rec["a"] == 1
    # present + overwrite -> replaced
    assert merge_enrichment_field(rec, "a", 9, overwrite=True) is True
    assert rec["a"] == 9


# --- field->value map branch ----------------------------------------------

@pytest.mark.asyncio
async def test_field_map_static_and_callable() -> None:
    enr = build_record_enricher({"tier": "gold", "n": lambda r: len(r["name"])})
    out = await _call(enr, {"name": "Alice"})
    assert out["tier"] == "gold"
    assert out["n"] == 5


@pytest.mark.asyncio
async def test_field_map_does_not_overwrite_existing() -> None:
    enr = build_record_enricher({"name": "OVERRIDE"})
    out = await _call(enr, {"name": "keep"})
    assert out["name"] == "keep"  # bare field-map form is overwrite=False


# --- ITransformFunction branch --------------------------------------------

@pytest.mark.asyncio
async def test_instance_branch_uses_construct_directly() -> None:
    enr = build_record_enricher(DataEnricher({"name": "REPLACED"}, overwrite=True))
    out = await _call(enr, {"name": "old"})
    assert out["name"] == "REPLACED"


# --- callable branch (arity + await normalization) ------------------------

@pytest.mark.asyncio
async def test_callable_one_arg() -> None:
    enr = build_record_enricher(lambda r: {**r, "x": 1})
    assert (await _call(enr, {"id": "1"}))["x"] == 1


@pytest.mark.asyncio
async def test_callable_two_arg_receives_context() -> None:
    def fn(record: dict, context: Any) -> dict:
        return {**record, "ctx_seen": context is not None}

    enr = build_record_enricher(fn)
    assert (await _call(enr, {"id": "1"}, context=object()))["ctx_seen"] is True


@pytest.mark.asyncio
async def test_async_callable_is_awaited() -> None:
    async def fn(record: dict) -> dict:
        return {**record, "async": True}

    enr = build_record_enricher(fn)
    assert inspect.iscoroutinefunction(enr)
    assert (await _call(enr, {"id": "1"}))["async"] is True


# --- dispatch errors -------------------------------------------------------

def test_database_key_rejected_in_library_builder() -> None:
    with pytest.raises(TypeError, match="ETL-level convenience"):
        build_record_enricher({"database": {"type": "memory"}, "match": {"a": "b"}})


def test_resource_lookup_requires_match() -> None:
    with pytest.raises(ValueError, match="non-empty 'match'"):
        build_record_enricher({"resource": "ref"})


def test_unsupported_spec_raises_type_error() -> None:
    with pytest.raises(TypeError):
        build_record_enricher(42)  # type: ignore[arg-type]


# --- _ETLEnrich non-dict return -------------------------------------------

@pytest.mark.asyncio
async def test_etl_enrich_rejects_non_dict_return() -> None:
    def bad(_record: dict) -> list:
        return ["not", "a", "dict"]

    step = _ETLEnrich([build_record_enricher(bad)])
    with pytest.raises(TransformError, match="must return a dict"):
        await step.transform({"id": "1"})


# --- LookupMergeEnricher direct (match / miss policies) -------------------

async def _ref_resource(rows: list[dict]) -> AsyncDatabaseResourceAdapter:
    adapter = AsyncDatabaseResourceAdapter("ref", type="memory")
    db = await adapter._ensure_db()
    for row in rows:
        await db.upsert(row["id"], Record(dict(row)))
    return adapter


@pytest.mark.asyncio
async def test_lookup_match_merges_fields() -> None:
    adapter = await _ref_resource(
        [{"id": "US", "code": "US", "region": "NA"}]
    )
    enr = LookupMergeEnricher("ref", {"country": "code"}, fields=["region"])
    ctx = SimpleNamespace(resources={"ref": adapter})
    out = await enr.transform({"country": "US"}, ctx)
    assert out["region"] == "NA"


@pytest.mark.asyncio
async def test_lookup_on_missing_policies() -> None:
    adapter = await _ref_resource([{"id": "US", "code": "US", "region": "NA"}])
    ctx = SimpleNamespace(resources={"ref": adapter})
    spec = ("ref", {"country": "code"})

    ignore = LookupMergeEnricher(*spec, fields=["region"], on_missing="ignore")
    assert "region" not in await ignore.transform({"country": "ZZ"}, ctx)

    null = LookupMergeEnricher(*spec, fields=["region"], on_missing="null")
    assert (await null.transform({"country": "ZZ"}, ctx))["region"] is None

    err = LookupMergeEnricher(*spec, fields=["region"], on_missing="error")
    with pytest.raises(TransformError):
        await err.transform({"country": "ZZ"}, ctx)


def test_lookup_validates_construction() -> None:
    with pytest.raises(ValueError, match="non-empty 'match'"):
        LookupMergeEnricher("ref", {})
    with pytest.raises(ValueError, match="unknown on_missing"):
        LookupMergeEnricher("ref", {"a": "b"}, on_missing="bogus")

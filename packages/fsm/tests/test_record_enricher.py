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


def test_match_without_resource_is_rejected() -> None:
    """A 'match' join spec with no 'resource' is a malformed lookup, not a map.

    Reproduces the silent mis-enrich: without the guard this spec falls through
    to the field→value map branch and adds literal ``match`` / ``fields`` columns
    to every record instead of doing a reference lookup.
    """
    with pytest.raises(TypeError, match="'match' join spec but no 'resource'"):
        build_record_enricher(
            {"match": {"country_code": "code"}, "fields": ["region"]}
        )


def test_overwrite_without_fields_is_rejected() -> None:
    """``overwrite=True`` with no explicit ``fields`` would clobber record keys."""
    with pytest.raises(
        ValueError, match="overwrite=True requires an explicit 'fields'"
    ):
        LookupMergeEnricher("ref", {"country": "code"}, overwrite=True)


def test_null_on_missing_without_fields_is_rejected() -> None:
    """``on_missing='null'`` with no ``fields`` would null nothing — a no-op."""
    with pytest.raises(
        ValueError, match="on_missing='null' requires an explicit 'fields'"
    ):
        LookupMergeEnricher("ref", {"country": "code"}, on_missing="null")


# --- shared step loop: non-dict guard for both stages ---------------------

@pytest.mark.asyncio
async def test_etl_enrich_rejects_non_dict_return() -> None:
    def bad(_record: dict) -> list:
        return ["not", "a", "dict"]

    step = _ETLEnrich([build_record_enricher(bad)])
    with pytest.raises(TransformError, match="enrichment #0 must return a dict"):
        await step.transform({"id": "1"})


@pytest.mark.asyncio
async def test_etl_transform_rejects_non_dict_return() -> None:
    """The shared step loop guards the transform stage with the same shape.

    Locks the :func:`_apply_record_steps` extraction: ``_ETLTransform`` and
    ``_ETLEnrich`` share the non-dict guard, differing only in the ``label``.
    """
    from dataknobs_fsm.patterns.etl import _ETLTransform

    step = _ETLTransform(
        field_mappings=None, transformations=[lambda _record: ["nope"]]
    )
    with pytest.raises(
        TransformError, match="transformation #0 must return a dict"
    ):
        await step.transform({"id": "1"})


@pytest.mark.asyncio
async def test_etl_transform_awaits_async_callable() -> None:
    """An async transformation callable is awaited by the shared step loop.

    Locks the async path of ``_ETLTransform``'s lambda wrapper +
    :func:`_apply_record_steps`: the coroutine returned by ``fn(record)`` is
    detected via ``inspect.isawaitable`` and awaited. The sync non-dict test
    above does not exercise this branch.
    """
    from dataknobs_fsm.patterns.etl import _ETLTransform

    async def add_flag(record: dict) -> dict:
        return {**record, "async": True}

    step = _ETLTransform(field_mappings=None, transformations=[add_flag])
    out = await step.transform({"id": "1"})
    assert out == {"id": "1", "async": True}


# --- DataEnricher: shared collision predicate short-circuits the callable --

def test_data_enricher_skips_callable_when_present_and_not_overwrite() -> None:
    """A present, not-overwritten field never evaluates its callable value.

    Pins the shared ``_enrichment_collides`` predicate: the collision is decided
    BEFORE a (potentially side-effecting) callable runs.
    """
    calls: list[int] = []

    def compute(_record: dict) -> str:
        calls.append(1)
        return "computed"

    enr = DataEnricher({"name": compute}, overwrite=False)
    out = enr.transform({"name": "keep"})
    assert out["name"] == "keep"
    assert calls == [], "callable was evaluated despite the collision skip"


# --- LookupMergeEnricher direct (match / miss policies) -------------------

async def _ref_resource(rows: list[dict]) -> AsyncDatabaseResourceAdapter:
    # Seed through the adapter's public ``upsert`` (the same surface the ETL
    # load step writes through), not the private ``_ensure_db()`` internal.
    adapter = AsyncDatabaseResourceAdapter("ref", type="memory")
    if rows:
        await adapter.upsert("ref", [dict(r) for r in rows], key_columns=["id"])
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

    # The null policy is subject to the overwrite gate (it merges via the same
    # collision predicate as a hit): a field the record already carries is left
    # untouched on a miss when overwrite=False, and nulled only when
    # overwrite=True. Pins the documented interaction (overwrite=False uniformly
    # means "do not touch existing fields", in both the hit and miss paths).
    null_keep = LookupMergeEnricher(*spec, fields=["region"], on_missing="null")
    kept = await null_keep.transform({"country": "ZZ", "region": "EU"}, ctx)
    assert kept["region"] == "EU", "overwrite=False must preserve existing on miss"

    null_over = LookupMergeEnricher(
        *spec, fields=["region"], on_missing="null", overwrite=True
    )
    over = await null_over.transform({"country": "ZZ", "region": "EU"}, ctx)
    assert over["region"] is None, "overwrite=True must null existing on miss"

    err = LookupMergeEnricher(*spec, fields=["region"], on_missing="error")
    with pytest.raises(TransformError):
        await err.transform({"country": "ZZ"}, ctx)


def test_lookup_validates_construction() -> None:
    with pytest.raises(ValueError, match="non-empty 'match'"):
        LookupMergeEnricher("ref", {})
    with pytest.raises(ValueError, match="unknown on_missing"):
        LookupMergeEnricher("ref", {"a": "b"}, on_missing="bogus")


@pytest.mark.asyncio
async def test_lookup_composite_key_match() -> None:
    """A multi-field ``match`` AND-combines: only the row matching BOTH wins.

    Locks the composite-key contract (the ``match`` items compile to AND-combined
    equality filters), which the single-field tests did not exercise.
    """
    adapter = await _ref_resource(
        [
            {"id": "1", "code": "US", "tier": "gold", "perk": "lounge"},
            {"id": "2", "code": "US", "tier": "silver", "perk": "wifi"},
        ]
    )
    enr = LookupMergeEnricher(
        "ref", {"country": "code", "level": "tier"}, fields=["perk"]
    )
    ctx = SimpleNamespace(resources={"ref": adapter})
    out = await enr.transform({"country": "US", "level": "silver"}, ctx)
    assert out["perk"] == "wifi", out

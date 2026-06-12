"""Behavior tests for the vector-partition resolver reference implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from dataknobs_common.resolver import (
    CallablePartitionResolver,
    JoiningPartitionResolver,
    MetadataKeyPartitionResolver,
    NullPartitionResolver,
    TemporalPartitionResolver,
)


@dataclass
class _FakeRecord:
    """Minimal record stub for partition-resolver tests."""

    metadata: dict[str, Any] = field(default_factory=dict)


# ---- NullPartitionResolver ----


def test_null_partition_resolver_returns_default() -> None:
    r = NullPartitionResolver()
    assert r.resolve(_FakeRecord()) == "default"


def test_null_partition_resolver_custom_default() -> None:
    r = NullPartitionResolver(default="prod")
    assert r.resolve(_FakeRecord()) == "prod"


# ---- MetadataKeyPartitionResolver ----


def test_metadata_key_resolver_extracts_value() -> None:
    r = MetadataKeyPartitionResolver(metadata_key="tenant_id")
    record = _FakeRecord(metadata={"tenant_id": "acme"})
    assert r.resolve(record) == "acme"


def test_metadata_key_resolver_uses_default_on_missing_key() -> None:
    r = MetadataKeyPartitionResolver(metadata_key="tenant_id", default="default")
    record = _FakeRecord(metadata={})
    assert r.resolve(record) == "default"


def test_metadata_key_resolver_uses_default_on_missing_metadata() -> None:
    r = MetadataKeyPartitionResolver(metadata_key="tenant_id")
    record = object()  # no metadata attribute
    assert r.resolve(record) == "default"


def test_metadata_key_resolver_str_coerces_scalar_value() -> None:
    r = MetadataKeyPartitionResolver(metadata_key="version")
    record = _FakeRecord(metadata={"version": 42})
    assert r.resolve(record) == "42"


def test_metadata_key_resolver_non_scalar_falls_back_to_default() -> None:
    """Non-scalar values (list/dict/set/custom object) resolve to default.

    Silent ``str()`` coercion would corrupt partition names like
    ``"[1, 2]"`` or ``"{'k': 'v'}"`` which are typically used as table
    suffixes, collection names, or filesystem paths.
    """
    r = MetadataKeyPartitionResolver(metadata_key="tenant_id", default="default")
    for bad_value in [[1, 2], {"nested": "value"}, {1, 2, 3}, object()]:
        record = _FakeRecord(metadata={"tenant_id": bad_value})
        assert r.resolve(record) == "default", f"failed for {bad_value!r}"


def test_metadata_key_resolver_accepts_all_scalar_types() -> None:
    r = MetadataKeyPartitionResolver(metadata_key="key")
    cases: list[tuple[Any, str]] = [
        ("string_value", "string_value"),
        (42, "42"),
        (3.14, "3.14"),
        (True, "True"),
    ]
    for value, expected in cases:
        record = _FakeRecord(metadata={"key": value})
        assert r.resolve(record) == expected


# ---- TemporalPartitionResolver ----


def test_temporal_resolver_quarter_bucket() -> None:
    r = TemporalPartitionResolver(timestamp_key="ts", bucket="quarter")
    record = _FakeRecord(metadata={"ts": datetime(2026, 3, 15)})
    assert r.resolve(record) == "2026_q1"
    record_q2 = _FakeRecord(metadata={"ts": datetime(2026, 5, 1)})
    assert r.resolve(record_q2) == "2026_q2"
    record_q4 = _FakeRecord(metadata={"ts": datetime(2026, 11, 1)})
    assert r.resolve(record_q4) == "2026_q4"


def test_temporal_resolver_month_bucket() -> None:
    r = TemporalPartitionResolver(timestamp_key="ts", bucket="month")
    record = _FakeRecord(metadata={"ts": datetime(2026, 3, 15)})
    assert r.resolve(record) == "2026_m03"


def test_temporal_resolver_year_bucket() -> None:
    r = TemporalPartitionResolver(timestamp_key="ts", bucket="year")
    record = _FakeRecord(metadata={"ts": datetime(2026, 3, 15)})
    assert r.resolve(record) == "2026"


def test_temporal_resolver_iso_string_timestamp() -> None:
    r = TemporalPartitionResolver(timestamp_key="ts", bucket="quarter")
    record = _FakeRecord(metadata={"ts": "2026-03-15T10:00:00"})
    assert r.resolve(record) == "2026_q1"


def test_temporal_resolver_unparseable_timestamp_returns_default() -> None:
    r = TemporalPartitionResolver(timestamp_key="ts", bucket="quarter")
    record = _FakeRecord(metadata={"ts": "not-a-date"})
    assert r.resolve(record) == "default"


def test_temporal_resolver_missing_timestamp_returns_default() -> None:
    r = TemporalPartitionResolver(timestamp_key="ts", bucket="quarter")
    record = _FakeRecord(metadata={})
    assert r.resolve(record) == "default"


def test_temporal_resolver_unsupported_bucket_raises_at_construction() -> None:
    """Bad bucket fails fast at construction, not lazily on first valid value.

    Previously the validation lived in ``_format_bucket`` and only fired
    when a record with a parseable timestamp was seen — so a typo
    (``bucket="quarterly"``) silently routed every record to ``default``
    until the first valid datetime crashed. ``__post_init__`` raises
    immediately so the typo can't ship.
    """
    with pytest.raises(ValueError, match="Unsupported bucket"):
        TemporalPartitionResolver(timestamp_key="ts", bucket="decade")
    with pytest.raises(ValueError, match="Unsupported bucket"):
        TemporalPartitionResolver(timestamp_key="ts", bucket="quarterly")


# ---- CallablePartitionResolver ----


def test_callable_partition_resolver_dispatches() -> None:
    r = CallablePartitionResolver(
        fn=lambda record: f"custom_{record.metadata.get('x', 'none')}"
    )
    record = _FakeRecord(metadata={"x": "value"})
    assert r.resolve(record) == "custom_value"


# ---- JoiningPartitionResolver ----


def test_joining_partition_resolver_joins_with_unambiguous_separator() -> None:
    """Recommended usage: pick a separator the sub-resolvers don't produce.

    ``TemporalPartitionResolver`` returns strings containing ``_``
    (``"2026_q2"`` / ``"2026_m05"``), so ``"::"`` (or ``"/"`` / ``"|"``)
    keeps the joined key reversibly parseable.
    """
    r = JoiningPartitionResolver(
        resolvers=[
            MetadataKeyPartitionResolver("tenant_id"),
            TemporalPartitionResolver("ts", bucket="quarter"),
        ],
        sep="::",
    )
    record = _FakeRecord(metadata={
        "tenant_id": "acme",
        "ts": datetime(2026, 5, 1),
    })
    assert r.resolve(record) == "acme::2026_q2"


def test_joining_partition_resolver_custom_separator() -> None:
    r = JoiningPartitionResolver(
        resolvers=[
            MetadataKeyPartitionResolver("tenant_id"),
            MetadataKeyPartitionResolver("content_type"),
        ],
        sep="/",
    )
    record = _FakeRecord(metadata={
        "tenant_id": "acme",
        "content_type": "legal",
    })
    assert r.resolve(record) == "acme/legal"


def test_joining_partition_resolver_none_subresolver_short_circuits() -> None:
    """If any sub-resolver returns None, the joiner returns None.

    The in-tree partition resolvers default to a string rather than None,
    so this short-circuit requires either a custom resolver or a record
    genuinely unable to produce a partition for one of the component
    dimensions.
    """
    r = JoiningPartitionResolver(
        resolvers=[
            MetadataKeyPartitionResolver("tenant_id"),
            CallablePartitionResolver(lambda record: None),
        ],
    )
    record = _FakeRecord(metadata={"tenant_id": "acme"})
    assert r.resolve(record) is None


def test_joining_partition_resolver_empty_chain_returns_empty_string() -> None:
    """Empty chain joins to empty string."""
    r = JoiningPartitionResolver(resolvers=[], sep="_")
    record = _FakeRecord(metadata={})
    assert r.resolve(record) == ""

"""Tests for VectorStoreBase timestamp config + helpers (Item 36, Phase 2).

Covers the shared timestamp exposure contract that all vector store
backends inherit: ``timestamps`` sub-config parsing, ``_format_timestamp``
across the three formats, and ``_inject_timestamps`` with collision
handling. Uses a minimal concrete subclass (no mocks) to exercise the
helpers without requiring any backend runtime.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pytest

from dataknobs_data.vector.stores.common import VectorStoreBase


class _ConcreteStore(VectorStoreBase):
    """Minimal concrete subclass for helper exercise — no backend I/O."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)


class TestTimestampFormat:
    """``_format_timestamp`` across the three supported formats."""

    def test_format_iso(self):
        store = _ConcreteStore({"dimensions": 4, "timestamps": {"format": "iso"}})
        dt = datetime(2026, 4, 22, 14, 23, 45, 123456, tzinfo=timezone.utc)
        assert store._format_timestamp(dt) == "2026-04-22T14:23:45.123456+00:00"

    def test_format_epoch(self):
        store = _ConcreteStore(
            {"dimensions": 4, "timestamps": {"format": "epoch"}}
        )
        dt = datetime(2026, 4, 22, 14, 23, 45, tzinfo=timezone.utc)
        result = store._format_timestamp(dt)
        assert isinstance(result, float)
        assert result == dt.timestamp()

    def test_format_datetime(self):
        store = _ConcreteStore(
            {"dimensions": 4, "timestamps": {"format": "datetime"}}
        )
        dt = datetime(2026, 4, 22, 14, 23, 45, tzinfo=timezone.utc)
        assert store._format_timestamp(dt) is dt

    def test_format_none_returns_none(self):
        store = _ConcreteStore({"dimensions": 4})
        assert store._format_timestamp(None) is None

    def test_default_format_is_iso(self):
        store = _ConcreteStore({"dimensions": 4})
        assert store.timestamps_format == "iso"

    def test_invalid_format_raises_at_config_parse(self):
        with pytest.raises(ValueError, match="timestamps.format"):
            _ConcreteStore({"dimensions": 4, "timestamps": {"format": "bogus"}})


class TestInjectTimestamps:
    """``_inject_timestamps`` contract: new dict, collision handling."""

    def test_inject_creates_new_dict(self):
        store = _ConcreteStore({"dimensions": 4})
        original: dict[str, Any] = {"k": "v"}
        dt = datetime(2026, 4, 22, tzinfo=timezone.utc)
        result = store._inject_timestamps(original, created=dt, updated=dt)
        assert result is not original
        assert original == {"k": "v"}, "input must not be mutated"
        assert result["k"] == "v"
        assert result["_created_at"] == dt.isoformat()
        assert result["_updated_at"] == dt.isoformat()

    def test_inject_none_meta_yields_dict(self):
        store = _ConcreteStore({"dimensions": 4})
        dt = datetime(2026, 4, 22, tzinfo=timezone.utc)
        result = store._inject_timestamps(None, created=dt, updated=dt)
        assert result == {
            "_created_at": dt.isoformat(),
            "_updated_at": dt.isoformat(),
        }

    def test_inject_both_timestamps_none_yields_none_values(self):
        store = _ConcreteStore({"dimensions": 4})
        result = store._inject_timestamps({"k": "v"}, created=None, updated=None)
        assert result["_created_at"] is None
        assert result["_updated_at"] is None
        assert result["k"] == "v"

    def test_inject_collision_keeps_consumer_value(self, caplog):
        store = _ConcreteStore({"dimensions": 4})
        dt = datetime(2026, 4, 22, tzinfo=timezone.utc)
        consumer_meta: dict[str, Any] = {"_created_at": "consumer-value"}

        with caplog.at_level(logging.WARNING):
            result = store._inject_timestamps(
                consumer_meta, created=dt, updated=dt
            )

        assert result["_created_at"] == "consumer-value"
        # Updated key was not in consumer's dict, so it is injected normally.
        assert result["_updated_at"] == dt.isoformat()
        assert any(
            "_created_at" in record.message and "skipped" in record.message
            for record in caplog.records
        )

    def test_inject_collision_warns_once_per_key(self, caplog):
        store = _ConcreteStore({"dimensions": 4})
        dt = datetime(2026, 4, 22, tzinfo=timezone.utc)

        with caplog.at_level(logging.WARNING):
            store._inject_timestamps(
                {"_created_at": "x"}, created=dt, updated=dt
            )
            store._inject_timestamps(
                {"_created_at": "y"}, created=dt, updated=dt
            )
            store._inject_timestamps(
                {"_created_at": "z"}, created=dt, updated=dt
            )

        collision_records = [
            r for r in caplog.records
            if "_created_at" in r.message and "skipped" in r.message
        ]
        assert len(collision_records) == 1, (
            "Collision warning must fire exactly once per key per store "
            "instance per process, not once per injection call."
        )

    def test_inject_different_stores_warn_independently(self, caplog):
        """Two different store instances each get their own warning.

        Warning state lives on the instance (per-instance
        ``_timestamp_collision_warned`` set), so each instance tracks
        its own keys and both should warn independently.
        """
        store_a = _ConcreteStore({"dimensions": 4})
        store_b = _ConcreteStore({"dimensions": 4})
        dt = datetime(2026, 4, 22, tzinfo=timezone.utc)

        with caplog.at_level(logging.WARNING):
            store_a._inject_timestamps(
                {"_created_at": "x"}, created=dt, updated=dt
            )
            store_b._inject_timestamps(
                {"_created_at": "y"}, created=dt, updated=dt
            )

        collision_records = [
            r for r in caplog.records
            if "_created_at" in r.message and "skipped" in r.message
        ]
        assert len(collision_records) == 2, (
            "Separate store instances must warn independently."
        )

    def test_inject_custom_keys(self):
        store = _ConcreteStore({
            "dimensions": 4,
            "timestamps": {
                "created_key": "freshness.created",
                "updated_key": "freshness.updated",
            },
        })
        dt = datetime(2026, 4, 22, tzinfo=timezone.utc)
        result = store._inject_timestamps({}, created=dt, updated=dt)
        assert "freshness.created" in result
        assert "freshness.updated" in result
        assert "_created_at" not in result
        assert "_updated_at" not in result

    def test_inject_applies_configured_format(self):
        store = _ConcreteStore(
            {"dimensions": 4, "timestamps": {"format": "epoch"}}
        )
        dt = datetime(2026, 4, 22, 14, 23, 45, tzinfo=timezone.utc)
        result = store._inject_timestamps({}, created=dt, updated=dt)
        assert isinstance(result["_created_at"], float)
        assert isinstance(result["_updated_at"], float)
        assert result["_created_at"] == dt.timestamp()

"""Unit tests for SQLRecordSerializer static helpers.

Pins the serializer's contract so the inbound (`row_to_record`) and
outbound (`record_to_row`) directions cannot silently drift again — the
same drift that produced the async-postgres id round-trip asymmetry.
"""

from __future__ import annotations

import json
import re
import uuid

from dataknobs_data import Record
from dataknobs_data.backends.sql_base import SQLRecordSerializer

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class TestRowToRecord:
    """Inbound direction: dict-shaped row -> Record."""

    def test_populates_id_from_row(self):
        record = SQLRecordSerializer.row_to_record(
            {"id": "abc-123", "data": '{"x": 1}', "metadata": None}
        )
        assert record.id == "abc-123"
        assert record.storage_id == "abc-123"
        assert record.fields["x"].value == 1

    def test_missing_id_does_not_raise(self):
        """Some test/in-memory paths may pass rows without ``id``."""
        record = SQLRecordSerializer.row_to_record(
            {"data": '{"x": 1}', "metadata": None}
        )
        # Without an ``id`` in the row, ``ensure_record_id`` is skipped.
        # The record's id is whatever ``json_to_record`` assigned (None).
        assert record.fields["x"].value == 1

    def test_metadata_string_decoded(self):
        record = SQLRecordSerializer.row_to_record(
            {
                "id": "k1",
                "data": '{"x": 1}',
                "metadata": '{"source": "unit-test"}',
            }
        )
        assert record.metadata == {"source": "unit-test"}

    def test_metadata_dict_serialized_then_decoded(self):
        """Some drivers (asyncpg/JSONB) hand back dicts; serializer copes."""
        record = SQLRecordSerializer.row_to_record(
            {
                "id": "k1",
                "data": {"x": 1},
                "metadata": {"source": "unit-test"},
            }
        )
        assert record.fields["x"].value == 1
        assert record.metadata == {"source": "unit-test"}

    def test_metadata_none_yields_empty_dict(self):
        record = SQLRecordSerializer.row_to_record(
            {"id": "k1", "data": '{"x": 1}', "metadata": None}
        )
        assert record.metadata == {}

    def test_metadata_literal_null_string_treated_as_empty(self):
        """JSONB null sometimes round-trips as the string 'null'."""
        record = SQLRecordSerializer.row_to_record(
            {"id": "k1", "data": '{"x": 1}', "metadata": "null"}
        )
        assert record.metadata == {}


class TestRecordToRow:
    """Outbound direction: Record -> dict-shaped row."""

    def test_uses_provided_id(self):
        rec = Record({"x": 1, "y": "two"}, id="explicit-id")
        row = SQLRecordSerializer.record_to_row(rec, id="explicit-id")
        assert row["id"] == "explicit-id"

    def test_generates_uuid_when_id_missing(self):
        """``id=None`` produces a fresh hyphenated 36-char UUID."""
        row = SQLRecordSerializer.record_to_row(Record({"x": 1}))
        assert UUID_RE.match(row["id"]), row["id"]
        # Round-trip through uuid.UUID to confirm validity:
        uuid.UUID(row["id"])

    def test_generates_unique_ids(self):
        """Each call without an id gets its own UUID (no reuse)."""
        seen = {
            SQLRecordSerializer.record_to_row(Record({"x": i}))["id"]
            for i in range(5)
        }
        assert len(seen) == 5

    def test_data_is_serialized_json_string(self):
        row = SQLRecordSerializer.record_to_row(
            Record({"x": 1, "y": "two"}), id="k"
        )
        assert isinstance(row["data"], str)
        decoded = json.loads(row["data"])
        assert decoded == {"x": 1, "y": "two"}

    def test_metadata_dict_is_serialized(self):
        rec = Record({"x": 1}, metadata={"source": "unit-test"})
        row = SQLRecordSerializer.record_to_row(rec, id="k")
        assert row["metadata"] is not None
        assert json.loads(row["metadata"]) == {"source": "unit-test"}

    def test_empty_metadata_becomes_none(self):
        """Empty metadata dict serializes to ``None`` (not ``"{}"``)."""
        rec = Record({"x": 1})
        assert rec.metadata in (None, {})
        row = SQLRecordSerializer.record_to_row(rec, id="k")
        assert row["metadata"] is None

    def test_keys_are_exactly_id_data_metadata(self):
        rec = Record({"x": 1}, metadata={"source": "u"})
        row = SQLRecordSerializer.record_to_row(rec, id="k")
        assert set(row.keys()) == {"id", "data", "metadata"}


class TestRoundTrip:
    """row_to_record(record_to_row(r)) preserves data + metadata."""

    def test_round_trip_basic(self):
        original = Record({"x": 1, "y": "two", "z": [1, 2, 3]}, id="r1")
        row = SQLRecordSerializer.record_to_row(original, id="r1")
        restored = SQLRecordSerializer.row_to_record(row)

        assert restored.id == "r1"
        assert restored.fields["x"].value == 1
        assert restored.fields["y"].value == "two"
        assert restored.fields["z"].value == [1, 2, 3]

    def test_round_trip_preserves_metadata(self):
        original = Record(
            {"k": "v"}, id="r2", metadata={"source": "doc-1", "version": 3}
        )
        row = SQLRecordSerializer.record_to_row(original, id="r2")
        restored = SQLRecordSerializer.row_to_record(row)

        assert restored.metadata == {"source": "doc-1", "version": 3}

    def test_round_trip_assigns_generated_id_when_omitted(self):
        original = Record({"k": "v"})
        row = SQLRecordSerializer.record_to_row(original)  # no id arg
        restored = SQLRecordSerializer.row_to_record(row)

        # Restored record's id matches the row's generated id.
        assert restored.id == row["id"]
        assert UUID_RE.match(restored.id)

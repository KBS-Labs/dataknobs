"""Tests for :func:`iter_latest_records`.

The helper is shared by ``ArtifactRegistry`` (``query``, ``count``) and
``RubricRegistry`` (``get_for_target``, ``list_all``).  Both compose
:class:`AsyncKeyedRecordStore` with a dual-write storage shape:

- *Latest pointer* row: stored at ``entity.id``; carries
  ``data["_version_key"] = f"{id}:{version}"``.
- *Versioned snapshot* row: stored at ``f"{id}:{version}"``; carries
  the same ``_version_key``.

These tests verify the helper drops snapshots, dedupes pointers, and
preserves input ordering (which matters for callers that push a sort
spec to the database).
"""

from __future__ import annotations

from dataknobs_data import Record

from dataknobs_bots.utils.versioned_records import iter_latest_records


def _pointer(entity_id: str, version: str) -> Record:
    """Build a latest-pointer record (storage_id != _version_key)."""
    return Record(
        id=entity_id,
        data={"id": entity_id, "_version_key": f"{entity_id}:{version}"},
    )


def _snapshot(entity_id: str, version: str) -> Record:
    """Build a versioned-snapshot record (storage_id == _version_key)."""
    key = f"{entity_id}:{version}"
    return Record(
        id=key,
        data={"id": entity_id, "_version_key": key},
    )


class TestIterLatestRecords:
    def test_empty_input_yields_nothing(self) -> None:
        assert list(iter_latest_records([])) == []

    def test_single_pointer_passes_through(self) -> None:
        records = [_pointer("a1", "1.0")]
        result = list(iter_latest_records(records))
        assert len(result) == 1
        assert result[0].data["id"] == "a1"

    def test_single_snapshot_is_dropped(self) -> None:
        records = [_snapshot("a1", "1.0")]
        assert list(iter_latest_records(records)) == []

    def test_pointer_and_snapshot_pair_yields_pointer_only(self) -> None:
        # The canonical case: one write produces one pointer + one snapshot.
        records = [_pointer("a1", "1.0"), _snapshot("a1", "1.0")]
        result = list(iter_latest_records(records))
        assert [r.storage_id or r.id for r in result] == ["a1"]

    def test_multiple_versions_only_pointer_survives(self) -> None:
        # After revising twice, storage has the latest pointer plus all
        # snapshot rows (v1, v2, v3).  Only the pointer should survive —
        # NOT the most recent snapshot (it has the same content as the
        # pointer for the current version but is still dropped by rule).
        records = [
            _pointer("a1", "3.0"),
            _snapshot("a1", "1.0"),
            _snapshot("a1", "2.0"),
            _snapshot("a1", "3.0"),
        ]
        result = list(iter_latest_records(records))
        assert len(result) == 1
        assert (result[0].storage_id or result[0].id) == "a1"

    def test_dedupe_repeated_pointer(self) -> None:
        # Defensive dedup: if the backend returns the same pointer twice,
        # only the first occurrence is yielded.
        records = [_pointer("a1", "1.0"), _pointer("a1", "1.0")]
        result = list(iter_latest_records(records))
        assert len(result) == 1

    def test_multiple_entities_each_yielded_once(self) -> None:
        records = [
            _pointer("a1", "1.0"),
            _snapshot("a1", "1.0"),
            _pointer("a2", "1.0"),
            _snapshot("a2", "1.0"),
        ]
        result = list(iter_latest_records(records))
        ids = [r.data["id"] for r in result]
        assert ids == ["a1", "a2"]

    def test_input_order_preserved(self) -> None:
        # First-occurrence-per-id rule preserves input order; this is the
        # contract callers depend on when pushing a sort spec to the DB.
        records = [
            _pointer("c", "1.0"),
            _pointer("a", "1.0"),
            _pointer("b", "1.0"),
        ]
        result = list(iter_latest_records(records))
        assert [r.data["id"] for r in result] == ["c", "a", "b"]

    def test_snapshots_interleaved_with_other_entities(self) -> None:
        # Old snapshots from a prior version of one entity should not
        # be confused for a different entity's pointer.
        records = [
            _snapshot("a1", "1.0"),   # old snapshot of a1
            _pointer("a2", "1.0"),    # pointer of a2
            _pointer("a1", "2.0"),    # current pointer of a1
            _snapshot("a1", "2.0"),   # current snapshot of a1
        ]
        result = list(iter_latest_records(records))
        ids = [r.data["id"] for r in result]
        assert ids == ["a2", "a1"]
        # And the surviving a1 record IS the pointer (not a snapshot).
        a1_record = next(r for r in result if r.data["id"] == "a1")
        assert (a1_record.storage_id or a1_record.id) == "a1"

    def test_record_without_version_key_is_yielded(self) -> None:
        # Records that don't follow the dual-write convention (no
        # ``_version_key``) pass through.  Useful for backwards-compat
        # reads of pre-migration data.
        record = Record(id="legacy", data={"id": "legacy"})
        result = list(iter_latest_records([record]))
        assert len(result) == 1

    def test_record_without_id_is_yielded_but_not_deduped(self) -> None:
        # Records without an ``id`` field bypass the dedup table — both
        # would be yielded.  Data-integrity issue in practice, but the
        # helper must not silently drop them or it could mask the bug.
        rec1 = Record(id="x", data={"_version_key": "x:1"})
        rec2 = Record(id="y", data={"_version_key": "y:1"})
        # rec1.storage_id is "x", _version_key is "x:1" → not a snapshot.
        # Same for rec2.  Neither has data["id"].
        result = list(iter_latest_records([rec1, rec2]))
        assert len(result) == 2

    def test_empty_data_does_not_crash(self) -> None:
        record = Record(id="r1", data=None)
        result = list(iter_latest_records([record]))
        assert len(result) == 1

"""Native per-version snapshot diff for :class:`S3KnowledgeBackend`.

Two ``change_detection_mode`` strategies are exercised with the real
aioboto3 client against **LocalStack** (the project's S3 integration
harness — a real S3 service, no mocks; ``moto``'s ``mock_aws`` is
incompatible with aiobotocore):

- ``"snapshot"``: a ``{path: checksum}`` object written per version.
- ``"s3_versioning"``: the metadata object's own S3 version history is
  the snapshot store (no extra objects). Requires bucket versioning;
  with it off, a stale version safely falls back to a full re-ingest.

Both must produce the same minimal, disjoint :class:`ChangeSet` the
in-memory and file backends already produce — change *detection* was
always correct; this proves the *diff* is now minimal for S3 too.

Start LocalStack with ``bin/dk up``; the integration tests skip when it
is unavailable.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge.storage import InvalidVersionError
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_common.testing import requires_localstack

# --- Pure config tests (no S3 service needed) ---


def test_unknown_change_detection_mode_raises() -> None:
    """Fail closed: an unrecognized mode is rejected, never defaulted."""
    with pytest.raises(ValueError, match="Unknown change_detection_mode"):
        S3KnowledgeBackend(bucket="b", change_detection_mode="bogus")


def test_from_config_threads_change_detection_mode() -> None:
    """``change_detection_mode`` flows through ``from_config``."""
    be = S3KnowledgeBackend.from_config(
        {"bucket": "b", "change_detection_mode": "s3_versioning"}
    )
    assert be._change_detection_mode == "s3_versioning"
    # Default when omitted.
    assert (
        S3KnowledgeBackend.from_config({"bucket": "b"})
        ._change_detection_mode
        == "snapshot"
    )


# --- LocalStack integration tests ---
#
# The marks are applied per-class (NOT module-level ``pytestmark``) so the
# two pure config tests above always run; only the service-backed classes
# require LocalStack.

_INTEGRATION_MARKS = [
    pytest.mark.integration,
    pytest.mark.s3,
    requires_localstack,
]


async def _backend(
    cfg: dict[str, Any], mode: str
) -> S3KnowledgeBackend:
    """Build and initialize a backend against the LocalStack bucket."""
    be = S3KnowledgeBackend.from_config(
        {**cfg, "change_detection_mode": mode}
    )
    await be.initialize()
    return be


class TestSnapshotMode:
    pytestmark = _INTEGRATION_MARKS

    async def test_minimal_disjoint_diff(self, s3_kb_config) -> None:
        be = await _backend(s3_kb_config, "snapshot")
        try:
            await be.create_kb("d")
            await be.put_file("d", "a.md", b"A1")
            await be.put_file("d", "b.md", b"B1")
            version = await be.get_checksum("d")

            await be.put_file("d", "a.md", b"A2")  # modified
            await be.put_file("d", "c.md", b"C1")  # added
            await be.delete_file("d", "b.md")  # deleted

            cs = await be.list_changes_since("d", version)
            assert sorted(f.path for f in cs.added) == ["c.md"]
            assert sorted(f.path for f in cs.modified) == ["a.md"]
            assert sorted(cs.deleted) == ["b.md"]
            assert cs.version == await be.get_checksum("d")
        finally:
            await be.close()

    async def test_empty_baseline_round_trips(self, s3_kb_config) -> None:
        be = await _backend(s3_kb_config, "snapshot")
        try:
            await be.create_kb("d")
            baseline = await be.get_checksum("d")
            assert baseline == ""
            await be.put_file("d", "a.md", b"A")

            cs = await be.list_changes_since("d", baseline)
            assert sorted(f.path for f in cs.added) == ["a.md"]
            assert not cs.modified and not cs.deleted
        finally:
            await be.close()

    async def test_unretained_version_raises_and_is_swallowed(
        self, s3_kb_config
    ) -> None:
        be = await _backend(s3_kb_config, "snapshot")
        try:
            await be.create_kb("d")
            await be.put_file("d", "a.md", b"A")

            with pytest.raises(InvalidVersionError):
                await be.list_changes_since("d", "not-a-real-snapshot")
            assert await be.has_changes_since("d", "stale") is True
        finally:
            await be.close()


class TestS3VersioningMode:
    pytestmark = _INTEGRATION_MARKS

    async def test_minimal_diff_via_version_history(
        self, s3_kb_versioned_config
    ) -> None:
        """No snapshot objects written — the diff is reconstructed from
        the metadata object's S3 version history."""
        be = await _backend(s3_kb_versioned_config, "s3_versioning")
        try:
            await be.create_kb("d")
            await be.put_file("d", "a.md", b"A1")
            await be.put_file("d", "b.md", b"B1")
            version = await be.get_checksum("d")

            await be.put_file("d", "a.md", b"A2")  # modified
            await be.put_file("d", "c.md", b"C1")  # added
            await be.delete_file("d", "b.md")  # deleted

            cs = await be.list_changes_since("d", version)
            assert sorted(f.path for f in cs.added) == ["c.md"]
            assert sorted(f.path for f in cs.modified) == ["a.md"]
            assert sorted(cs.deleted) == ["b.md"]

            # Fast path writes NO _snapshots/ objects.
            snapshot_prefix = (
                f"{be._prefix}d/{be.SNAPSHOTS_DIR}/"
            )
            async with be._session.client(
                "s3", **be._client_kwargs
            ) as s3:
                listed = await s3.list_objects_v2(
                    Bucket=be._bucket, Prefix=snapshot_prefix
                )
            assert listed.get("KeyCount", 0) == 0
        finally:
            await be.close()

    async def test_versioning_disabled_falls_back_safely(
        self, s3_kb_config
    ) -> None:
        """With bucket versioning off only the current metadata version
        is listed, so a stale version is unresolvable → InvalidVersionError
        → ``has_changes_since`` reports "changed" (a correct, non-minimal
        full re-ingest — never a wrong diff).

        Uses the *unversioned* bucket so the s3_versioning walk finds only
        the current metadata version.
        """
        be = await _backend(s3_kb_config, "s3_versioning")
        try:
            await be.create_kb("d")
            await be.put_file("d", "a.md", b"A1")
            stale = await be.get_checksum("d")
            await be.put_file("d", "b.md", b"B1")  # version now differs

            with pytest.raises(InvalidVersionError):
                await be.list_changes_since("d", stale)
            assert await be.has_changes_since("d", stale) is True
        finally:
            await be.close()

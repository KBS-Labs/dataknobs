"""Native per-version snapshot diff for :class:`S3KnowledgeBackend`.

Two ``change_detection_mode`` strategies are exercised with real boto3
against ``moto`` (the project's S3 testing construct — a real client, no
mocks):

- ``"snapshot"``: a ``{path: checksum}`` object written per version.
- ``"s3_versioning"``: the metadata object's own S3 version history is
  the snapshot store (no extra objects). Requires bucket versioning;
  with it off, a stale version safely falls back to a full re-ingest.

Both must produce the same minimal, disjoint :class:`ChangeSet` the
in-memory and file backends already produce — change *detection* was
always correct; this proves the *diff* is now minimal for S3 too.
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from dataknobs_bots.knowledge.storage import InvalidVersionError
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend

BUCKET = "kb-snapshot-bucket"


@pytest.fixture(autouse=True)
def _isolate_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient AWS env so moto (not a running LocalStack) serves.

    Mirrors ``test_s3_region_fallback``: ``bin/test.sh`` exports
    ``AWS_ENDPOINT_URL*`` for integration runs and botocore 1.34+ honors
    them inside ``mock_aws()``, which would route to LocalStack and
    persist bucket state across tests.
    """
    for key in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "LOCALSTACK_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


async def _backend(mode: str, *, versioning: bool) -> S3KnowledgeBackend:
    """Create the bucket (optionally versioned) and an initialized backend."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    if versioning:
        client.put_bucket_versioning(
            Bucket=BUCKET,
            VersioningConfiguration={"Status": "Enabled"},
        )
    be = S3KnowledgeBackend(
        bucket=BUCKET, prefix="kb/", change_detection_mode=mode
    )
    await be.initialize()
    return be


def test_unknown_change_detection_mode_raises() -> None:
    """Fail closed: an unrecognized mode is rejected, never defaulted."""
    with pytest.raises(ValueError, match="Unknown change_detection_mode"):
        S3KnowledgeBackend(bucket=BUCKET, change_detection_mode="bogus")


def test_from_config_threads_change_detection_mode() -> None:
    """``change_detection_mode`` flows through ``from_config``."""
    be = S3KnowledgeBackend.from_config(
        {"bucket": BUCKET, "change_detection_mode": "s3_versioning"}
    )
    assert be._change_detection_mode == "s3_versioning"
    # Default when omitted.
    assert (
        S3KnowledgeBackend.from_config({"bucket": BUCKET})
        ._change_detection_mode
        == "snapshot"
    )


class TestSnapshotMode:
    async def test_minimal_disjoint_diff(self) -> None:
        with mock_aws():
            be = await _backend("snapshot", versioning=False)
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
            await be.close()

    async def test_empty_baseline_round_trips(self) -> None:
        with mock_aws():
            be = await _backend("snapshot", versioning=False)
            await be.create_kb("d")
            baseline = await be.get_checksum("d")
            assert baseline == ""
            await be.put_file("d", "a.md", b"A")

            cs = await be.list_changes_since("d", baseline)
            assert sorted(f.path for f in cs.added) == ["a.md"]
            assert not cs.modified and not cs.deleted
            await be.close()

    async def test_unretained_version_raises_and_is_swallowed(self) -> None:
        with mock_aws():
            be = await _backend("snapshot", versioning=False)
            await be.create_kb("d")
            await be.put_file("d", "a.md", b"A")

            with pytest.raises(InvalidVersionError):
                await be.list_changes_since("d", "not-a-real-snapshot")
            assert await be.has_changes_since("d", "stale") is True
            await be.close()


class TestS3VersioningMode:
    async def test_minimal_diff_via_version_history(self) -> None:
        """No snapshot objects written — the diff is reconstructed from
        the metadata object's S3 version history."""
        with mock_aws():
            be = await _backend("s3_versioning", versioning=True)
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
            client = boto3.client("s3", region_name="us-east-1")
            listed = client.list_objects_v2(
                Bucket=BUCKET, Prefix="kb/d/_snapshots/"
            )
            assert listed.get("KeyCount", 0) == 0
            await be.close()

    async def test_versioning_disabled_falls_back_safely(self) -> None:
        """With bucket versioning off only the current metadata version
        is listed, so a stale version is unresolvable → InvalidVersionError
        → ``has_changes_since`` reports "changed" (a correct, non-minimal
        full re-ingest — never a wrong diff)."""
        with mock_aws():
            be = await _backend("s3_versioning", versioning=False)
            await be.create_kb("d")
            await be.put_file("d", "a.md", b"A1")
            stale = await be.get_checksum("d")
            await be.put_file("d", "b.md", b"B1")  # version now differs

            with pytest.raises(InvalidVersionError):
                await be.list_changes_since("d", stale)
            assert await be.has_changes_since("d", stale) is True
            await be.close()

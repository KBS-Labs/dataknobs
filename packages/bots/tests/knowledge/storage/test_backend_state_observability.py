"""Cross-backend conformance: state writes fire observability events.

Every in-tree backend fires ``ingest:metadata:write`` on a metadata
state write and ``ingest:snapshot:write`` on a snapshot state write,
through the shared ``_fire_state_write`` helper on
``KnowledgeResourceBackendMixin``. The helper is zero-overhead when no
callbacks were ever registered (the registry is not constructed).
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from dataknobs_bots.knowledge import (
    INGEST_METADATA_WRITE,
    INGEST_SNAPSHOT_WRITE,
)
from dataknobs_bots.knowledge.storage import (
    FileKnowledgeBackend,
    InMemoryKnowledgeBackend,
    KnowledgeKeyKind,
    S3KnowledgeBackend,
)

S3_BUCKET = "kb-state-observability-bucket"


async def _build(kind: str, tmp_path) -> object:
    if kind == "memory":
        backend: object = InMemoryKnowledgeBackend()
    else:
        backend = FileKnowledgeBackend(str(tmp_path / "kb"))
    await backend.initialize()
    await backend.create_kb("d1")
    return backend


@pytest.fixture
def _isolate_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient AWS env so moto (not a running LocalStack) serves.

    Mirrors ``test_s3_snapshot_diff``: ``bin/test.sh`` exports
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


async def _s3_backend() -> S3KnowledgeBackend:
    """Create the bucket and an initialized S3 backend (inside mock_aws)."""
    boto3.client("s3", region_name="us-east-1").create_bucket(
        Bucket=S3_BUCKET
    )
    backend = S3KnowledgeBackend(bucket=S3_BUCKET, prefix="kb/")
    await backend.initialize()
    await backend.create_kb("d1")
    return backend


@pytest.mark.parametrize("kind", ["memory", "file"])
@pytest.mark.asyncio
async def test_metadata_write_fires(kind: str, tmp_path) -> None:
    backend = await _build(kind, tmp_path)
    events: list[dict] = []
    backend.state_write_callbacks.register(
        INGEST_METADATA_WRITE, events.append
    )

    # set_ingestion_status is a metadata state write on every backend.
    await backend.set_ingestion_status("d1", "ready")

    assert len(events) >= 1
    ev = events[-1]
    assert ev["domain_id"] == "d1"
    assert ev["kind"] is KnowledgeKeyKind.METADATA
    assert isinstance(ev["byte_size"], int)
    assert isinstance(ev["key"], str) and ev["key"]


@pytest.mark.parametrize("kind", ["memory", "file"])
@pytest.mark.asyncio
async def test_snapshot_write_fires(kind: str, tmp_path) -> None:
    backend = await _build(kind, tmp_path)
    events: list[dict] = []
    backend.state_write_callbacks.register(
        INGEST_SNAPSHOT_WRITE, events.append
    )

    # put_file records a content snapshot on every backend.
    await backend.put_file("d1", "intro.md", b"# Intro\n")

    assert len(events) >= 1
    ev = events[-1]
    assert ev["domain_id"] == "d1"
    assert ev["kind"] is KnowledgeKeyKind.SNAPSHOT
    assert isinstance(ev["byte_size"], int)


@pytest.mark.parametrize("kind", ["memory", "file"])
@pytest.mark.asyncio
async def test_zero_overhead_when_no_callbacks(kind: str, tmp_path) -> None:
    """No callback ever registered ⇒ the registry is never constructed."""
    backend = await _build(kind, tmp_path)

    await backend.put_file("d1", "intro.md", b"# Intro\n")
    await backend.set_ingestion_status("d1", "ready")

    assert getattr(backend, "_state_write_callbacks", None) is None


@pytest.mark.parametrize("kind", ["memory", "file"])
@pytest.mark.asyncio
async def test_state_write_callbacks_stable_identity(
    kind: str, tmp_path
) -> None:
    backend = await _build(kind, tmp_path)
    assert backend.state_write_callbacks is backend.state_write_callbacks


# ---------------------------------------------------------------------------
# S3 backend — the third in-tree backend, advertised the same
# BACKEND_STATE_OBSERVABILITY capability, exercised here against moto so
# its metadata/snapshot fires are not advertised-but-unproven.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s3_metadata_write_fires(_isolate_aws_env) -> None:
    with mock_aws():
        backend = await _s3_backend()
        events: list[dict] = []
        backend.state_write_callbacks.register(
            INGEST_METADATA_WRITE, events.append
        )

        await backend.set_ingestion_status("d1", "ready")

        assert len(events) >= 1
        ev = events[-1]
        assert ev["domain_id"] == "d1"
        assert ev["kind"] is KnowledgeKeyKind.METADATA
        assert isinstance(ev["byte_size"], int)
        assert isinstance(ev["key"], str) and ev["key"]


@pytest.mark.asyncio
async def test_s3_snapshot_write_fires(_isolate_aws_env) -> None:
    with mock_aws():
        backend = await _s3_backend()
        events: list[dict] = []
        backend.state_write_callbacks.register(
            INGEST_SNAPSHOT_WRITE, events.append
        )

        await backend.put_file("d1", "intro.md", b"# Intro\n")

        assert len(events) >= 1
        ev = events[-1]
        assert ev["domain_id"] == "d1"
        assert ev["kind"] is KnowledgeKeyKind.SNAPSHOT
        assert isinstance(ev["byte_size"], int)


@pytest.mark.asyncio
async def test_s3_zero_overhead_when_no_callbacks(_isolate_aws_env) -> None:
    """No callback ever registered ⇒ the registry is never constructed."""
    with mock_aws():
        backend = await _s3_backend()

        await backend.put_file("d1", "intro.md", b"# Intro\n")
        await backend.set_ingestion_status("d1", "ready")

        assert getattr(backend, "_state_write_callbacks", None) is None

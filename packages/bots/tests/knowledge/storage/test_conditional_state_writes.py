"""Optimistic-concurrency (conditional) state writes on knowledge backends.

The public state-write entry, ``set_ingestion_status``, does a
read-modify-write on the whole KB metadata document: load it, mutate
``ingestion_status`` / ``ingestion_error`` / ``generation``, save it. Two
writers interleaving here last-writer-wins clobber — writer B's load
predates writer A's save, so B's save silently drops A's status
transition. ``get_state_version`` + the ``expected_version`` guard close
that race: capture the current opaque token, pass it back, and a stale
token raises :class:`ConcurrencyError` instead of clobbering.

Reproduce-first: each backend is first shown to clobber on the
unconditional path (``test_unconditional_writes_last_writer_wins``), then
shown to convert that clobber into a ``ConcurrencyError`` once the stale
token is supplied (``test_cas_conflict_raises_concurrency_error``).

Scope: the metadata document only. Snapshots are content-addressed /
write-once by identity (every writer of a given key agrees on its bytes),
so they need no CAS and are out of scope here.

Backends:
    - memory + file run in plain CI (this module's parametrized tests).
    - S3 runs against LocalStack (``@requires_localstack``), exercising
      the server-enforced ``If-Match`` precondition.

The file backend's guarantee is multi-process on one POSIX host (an
ephemeral ``fcntl.flock`` over the read-check-write critical section). A
genuine two-process race is heavier than these single-process,
deterministic stale-token tests; a cross-process ``multiprocessing`` race
asserting exactly one write lands is a captured follow-up. The
single-process tests pin that a stale token raises and a fresh token
succeeds, and that the advisory lock sidecar is created only on the CAS
path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dataknobs_common.capabilities import Capability
from dataknobs_common.exceptions import ConcurrencyError

from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend
from dataknobs_bots.knowledge.storage.memory import InMemoryKnowledgeBackend
from dataknobs_bots.knowledge.storage.models import IngestionStatus
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend

try:
    from dataknobs_common.testing import requires_localstack
except ImportError:  # pragma: no cover - defensive
    requires_localstack = pytest.mark.skip(reason="requires_localstack unavailable")


# ---------------------------------------------------------------------------
# memory + file (plain CI)
# ---------------------------------------------------------------------------


async def _make_local_backend(
    kind: str, tmp_path: Path
) -> InMemoryKnowledgeBackend | FileKnowledgeBackend:
    """Build + initialize an in-CI backend (memory or file)."""
    backend: InMemoryKnowledgeBackend | FileKnowledgeBackend
    if kind == "memory":
        backend = InMemoryKnowledgeBackend()
    else:
        backend = FileKnowledgeBackend(base_path=str(tmp_path / "kb"))
    await backend.initialize()
    return backend


_LOCAL_KINDS = ["memory", "file"]


@pytest.mark.parametrize("kind", _LOCAL_KINDS)
async def test_advertises_transactional_metadata(
    kind: str, tmp_path: Path
) -> None:
    """Each in-tree backend enforces CAS, so it advertises the capability."""
    backend = await _make_local_backend(kind, tmp_path)
    try:
        assert backend.supports(Capability.TRANSACTIONAL_METADATA)
    finally:
        await backend.close()


@pytest.mark.parametrize("kind", _LOCAL_KINDS)
async def test_state_version_none_for_missing_kb(
    kind: str, tmp_path: Path
) -> None:
    """No state document ⇒ ``get_state_version`` returns ``None``."""
    backend = await _make_local_backend(kind, tmp_path)
    try:
        assert await backend.get_state_version("nope") is None
    finally:
        await backend.close()


@pytest.mark.parametrize("kind", _LOCAL_KINDS)
async def test_state_version_present_after_create_kb(
    kind: str, tmp_path: Path
) -> None:
    """The single-tenant state document exists from ``create_kb`` onward."""
    backend = await _make_local_backend(kind, tmp_path)
    try:
        await backend.create_kb("d")
        assert await backend.get_state_version("d") is not None
    finally:
        await backend.close()


@pytest.mark.parametrize("kind", _LOCAL_KINDS)
async def test_unconditional_writes_last_writer_wins(
    kind: str, tmp_path: Path
) -> None:
    """Characterization: without ``expected_version`` the later write
    silently clobbers — the race the CAS guard exists to close.
    """
    backend = await _make_local_backend(kind, tmp_path)
    try:
        await backend.create_kb("d")
        # A token captured before either write — both writers "decided"
        # against this baseline but write unconditionally.
        baseline = await backend.get_state_version("d")

        await backend.set_ingestion_status("d", IngestionStatus.READY)
        await backend.set_ingestion_status("d", IngestionStatus.ERROR)

        info = await backend.get_info("d")
        assert info is not None
        # No guard ⇒ last writer wins; A's READY transition was dropped.
        assert info.ingestion_status == IngestionStatus.ERROR
        # The baseline token is now stale (every write advanced it).
        assert await backend.get_state_version("d") != baseline
    finally:
        await backend.close()


@pytest.mark.parametrize("kind", _LOCAL_KINDS)
async def test_cas_conflict_raises_concurrency_error(
    kind: str, tmp_path: Path
) -> None:
    """A stale ``expected_version`` raises ``ConcurrencyError`` and the
    winning writer's transition survives intact.
    """
    backend = await _make_local_backend(kind, tmp_path)
    try:
        await backend.create_kb("d")
        token = await backend.get_state_version("d")
        assert token is not None

        # Writer A wins with the shared token.
        await backend.set_ingestion_status(
            "d", IngestionStatus.READY, expected_version=token
        )

        # Writer B still holds the now-stale token.
        with pytest.raises(ConcurrencyError) as exc_info:
            await backend.set_ingestion_status(
                "d", IngestionStatus.ERROR, expected_version=token
            )
        assert exc_info.value.context["domain_id"] == "d"
        assert exc_info.value.context["expected_version"] == token

        # The conflict left A's transition untouched (no partial write).
        info = await backend.get_info("d")
        assert info is not None
        assert info.ingestion_status == IngestionStatus.READY
    finally:
        await backend.close()


@pytest.mark.parametrize("kind", _LOCAL_KINDS)
async def test_state_version_token_round_trips(
    kind: str, tmp_path: Path
) -> None:
    """Read token → conditional write → read a *different* token; an
    idempotent re-write with the fresh token succeeds.
    """
    backend = await _make_local_backend(kind, tmp_path)
    try:
        await backend.create_kb("d")
        t0 = await backend.get_state_version("d")
        assert t0 is not None

        await backend.set_ingestion_status(
            "d", IngestionStatus.INGESTING, expected_version=t0
        )
        t1 = await backend.get_state_version("d")
        assert t1 is not None
        assert t1 != t0

        # The fresh token writes cleanly.
        await backend.set_ingestion_status(
            "d", IngestionStatus.READY, expected_version=t1
        )
        t2 = await backend.get_state_version("d")
        assert t2 is not None
        assert t2 != t1
    finally:
        await backend.close()


@pytest.mark.parametrize("kind", _LOCAL_KINDS)
async def test_default_path_writes_unconditionally(
    kind: str, tmp_path: Path
) -> None:
    """``expected_version=None`` (the default) preserves the
    unconditional write — a stale token never blocks it — and the file
    backend never creates its lock sidecar on this path.
    """
    backend = await _make_local_backend(kind, tmp_path)
    try:
        await backend.create_kb("d")
        stale = await backend.get_state_version("d")
        assert stale is not None

        # First default write advances the token, making `stale` stale.
        await backend.set_ingestion_status("d", IngestionStatus.READY)
        # A second default write still succeeds despite the stale token
        # (the default path does no version check).
        await backend.set_ingestion_status("d", IngestionStatus.ERROR)

        info = await backend.get_info("d")
        assert info is not None
        assert info.ingestion_status == IngestionStatus.ERROR

        if kind == "file":
            lock = (
                tmp_path
                / "kb"
                / "d"
                / FileKnowledgeBackend.METADATA_LOCK_FILE
            )
            assert not lock.exists()
    finally:
        await backend.close()


async def test_file_cas_path_creates_lock_sidecar(tmp_path: Path) -> None:
    """The file backend's CAS path opens the advisory lock sidecar.

    Complements ``test_default_path_writes_unconditionally`` (which pins
    the sidecar is absent on the unconditional path).
    """
    backend = FileKnowledgeBackend(base_path=str(tmp_path / "kb"))
    await backend.initialize()
    try:
        await backend.create_kb("d")
        token = await backend.get_state_version("d")
        await backend.set_ingestion_status(
            "d", IngestionStatus.READY, expected_version=token
        )
        lock = (
            tmp_path / "kb" / "d" / FileKnowledgeBackend.METADATA_LOCK_FILE
        )
        assert lock.exists()
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# S3 (LocalStack — server-enforced If-Match)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.s3
@requires_localstack
async def test_s3_state_version_token_round_trips(s3_kb_config) -> None:
    """S3 ``get_state_version`` returns the ETag and round-trips."""
    backend = S3KnowledgeBackend.from_config(s3_kb_config)
    await backend.initialize()
    try:
        assert await backend.get_state_version("missing") is None

        await backend.create_kb("d")
        t0 = await backend.get_state_version("d")
        assert t0 is not None

        await backend.set_ingestion_status(
            "d", IngestionStatus.READY, expected_version=t0
        )
        t1 = await backend.get_state_version("d")
        assert t1 is not None
        assert t1 != t0
    finally:
        await backend.close()


@pytest.mark.integration
@pytest.mark.s3
@requires_localstack
async def test_s3_if_match_conflict_raises_concurrency_error(
    s3_kb_config,
) -> None:
    """A stale ``If-Match`` precondition (HTTP 412) maps to
    ``ConcurrencyError``; the winner's transition survives.

    If the running LocalStack build does not enforce conditional PUT, the
    second write would succeed — skip explicitly (no silent pass) rather
    than report a misleading failure.
    """
    backend = S3KnowledgeBackend.from_config(s3_kb_config)
    await backend.initialize()
    try:
        await backend.create_kb("d")
        token = await backend.get_state_version("d")
        assert token is not None

        # Writer A wins; this advances the object ETag.
        await backend.set_ingestion_status(
            "d", IngestionStatus.READY, expected_version=token
        )

        # Writer B holds A's pre-write (now stale) ETag.
        try:
            await backend.set_ingestion_status(
                "d", IngestionStatus.ERROR, expected_version=token
            )
        except ConcurrencyError as exc:
            assert exc.context["domain_id"] == "d"
            assert exc.context["expected_version"] == token
        else:
            pytest.skip(
                "LocalStack build does not enforce S3 If-Match "
                "conditional PUT"
            )

        info = await backend.get_info("d")
        assert info is not None
        assert info.ingestion_status == IngestionStatus.READY
    finally:
        await backend.close()


@pytest.mark.integration
@pytest.mark.s3
@requires_localstack
async def test_s3_default_path_writes_unconditionally(s3_kb_config) -> None:
    """``expected_version=None`` against S3 stays an unconditional PUT."""
    backend = S3KnowledgeBackend.from_config(s3_kb_config)
    await backend.initialize()
    try:
        await backend.create_kb("d")
        stale = await backend.get_state_version("d")
        assert stale is not None

        await backend.set_ingestion_status("d", IngestionStatus.READY)
        # Stale token does not block a default (no expected_version) write.
        await backend.set_ingestion_status("d", IngestionStatus.ERROR)

        info = await backend.get_info("d")
        assert info is not None
        assert info.ingestion_status == IngestionStatus.ERROR
    finally:
        await backend.close()

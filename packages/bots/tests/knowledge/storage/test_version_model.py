"""Version-model behavior: KnowledgeBaseInfo / IngestionStatus / ChangeSet.

Real constructs only — `InMemoryKnowledgeBackend` / `FileKnowledgeBackend`
are the documented testing backends; no mocks.

RC1 regression: `get_checksum()` and `has_changes_since()` are
documented as a change-detection pair, but historically lived in
different value spaces (`get_checksum` → content-snapshot MD5;
`has_changes_since` → monotonic `info.version` counter). A consumer
doing the intuitive `v = await be.get_checksum(d); ...
await be.has_changes_since(d, v)` therefore got ``True`` forever
(checksum != counter) → permanent spurious full re-ingest. The
``test_rc1_*`` cases below fail on pre-fix code and pass after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.knowledge import (
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)
from dataknobs_bots.knowledge.storage import (
    ChangeSet,
    FileKnowledgeBackend,
    InMemoryKnowledgeBackend,
    InvalidVersionError,
)


@pytest.fixture(params=["memory", "file"])
async def backend(request: pytest.FixtureRequest, tmp_path: Path):
    """Each non-S3 in-tree backend (S3 needs boto3; covered elsewhere)."""
    if request.param == "memory":
        be = InMemoryKnowledgeBackend()
    else:
        be = FileKnowledgeBackend(base_path=tmp_path / "kb")
    await be.initialize()
    yield be
    await be.close()


class TestRC1ChecksumChangeDetectionRoundTrip:
    """`get_checksum()` ⇄ `has_changes_since()` must agree."""

    async def test_rc1_unchanged_kb_reports_no_changes(self, backend) -> None:
        """Capture the checksum, change nothing, expect no changes.

        Pre-fix this asserts ``False`` but gets ``True`` because
        ``has_changes_since`` compares the monotonic counter against
        the content-snapshot MD5 the consumer captured.
        """
        await backend.create_kb("d")
        await backend.put_file("d", "a.md", b"alpha")

        version = await backend.get_checksum("d")
        # Nothing mutated between capture and check.
        assert await backend.has_changes_since("d", version) is False

    async def test_rc1_changed_kb_reports_changes(self, backend) -> None:
        """A real edit after capture is still detected."""
        await backend.create_kb("d")
        await backend.put_file("d", "a.md", b"alpha")

        version = await backend.get_checksum("d")
        await backend.put_file("d", "b.md", b"beta")

        assert await backend.has_changes_since("d", version) is True

    async def test_rc1_empty_kb_baseline_round_trips(self, backend) -> None:
        """The empty-KB checksum is a valid baseline to diff against."""
        await backend.create_kb("d")

        baseline = await backend.get_checksum("d")  # "" for an empty KB
        assert await backend.has_changes_since("d", baseline) is False

        await backend.put_file("d", "a.md", b"alpha")
        assert await backend.has_changes_since("d", baseline) is True


class TestChangeSetInvariants:
    """`ChangeSet` shape + the disjointness/`is_empty` contract."""

    def test_is_empty_true_when_nothing_changed(self) -> None:
        cs = ChangeSet(added=[], modified=[], deleted=[], version="v")
        assert cs.is_empty is True

    def test_is_empty_false_with_any_change(self) -> None:
        cs = ChangeSet(added=[], modified=[], deleted=["x"], version="v")
        assert cs.is_empty is False

    def test_frozen(self) -> None:
        cs = ChangeSet(added=[], modified=[], deleted=[], version="v")
        with pytest.raises((AttributeError, TypeError)):
            cs.version = "other"  # type: ignore[misc]

    async def test_memory_diff_disjoint_and_correct(self) -> None:
        """Memory's per-version snapshot yields a correct minimal diff.

        added / modified / deleted must be pairwise disjoint and the
        ChangeSet version must equal the current canonical checksum.
        """
        be = InMemoryKnowledgeBackend()
        await be.initialize()
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

        added = {f.path for f in cs.added}
        modified = {f.path for f in cs.modified}
        deleted = set(cs.deleted)
        assert added & modified == set()
        assert added & deleted == set()
        assert modified & deleted == set()

        assert cs.version == await be.get_checksum("d")
        assert cs.is_empty is False
        await be.close()

    async def test_unchanged_short_circuit_is_empty(self, backend) -> None:
        """Equal version ⇒ empty ChangeSet for every backend (no snapshot
        store needed — this is the RC1 fix that works universally)."""
        await backend.create_kb("d")
        await backend.put_file("d", "a.md", b"A")
        version = await backend.get_checksum("d")

        cs = await backend.list_changes_since("d", version)
        assert cs.is_empty
        assert cs.version == version


class TestNaiveBackendChangeSet:
    """File/S3 backends have no per-version store: a differing version
    reports every current file as ``added`` — correct, non-minimal."""

    async def test_file_backend_differing_version_all_added(
        self, tmp_path: Path
    ) -> None:
        be = FileKnowledgeBackend(base_path=tmp_path / "kb")
        await be.initialize()
        await be.create_kb("d")
        await be.put_file("d", "a.md", b"A")
        old = await be.get_checksum("d")
        await be.put_file("d", "b.md", b"B")  # version now differs

        cs = await be.list_changes_since("d", old)
        # Naive: no retained snapshot ⇒ all current files are "added".
        assert sorted(f.path for f in cs.added) == ["a.md", "b.md"]
        assert not cs.modified
        assert not cs.deleted
        assert cs.is_empty is False  # detection still correct
        await be.close()


class TestInvalidVersionError:
    """Memory has a real store ⇒ an unretained version is reported, not
    silently treated as the empty snapshot."""

    async def test_list_changes_since_raises_for_unretained(self) -> None:
        be = InMemoryKnowledgeBackend()
        await be.initialize()
        await be.create_kb("d")
        await be.put_file("d", "a.md", b"A")
        with pytest.raises(InvalidVersionError):
            await be.list_changes_since("d", "not-a-real-snapshot-id")
        await be.close()

    async def test_has_changes_since_swallows_invalid_version(self) -> None:
        """`has_changes_since` maps an unresolvable version to "changed"
        so callers safely re-ingest (no exception leaks)."""
        be = InMemoryKnowledgeBackend()
        await be.initialize()
        await be.create_kb("d")
        await be.put_file("d", "a.md", b"A")
        assert await be.has_changes_since("d", "stale-unknown") is True
        await be.close()


class TestIngestIfChangedRoundTrip:
    """RC1 at the manager layer: capturing get_current_version and
    passing it back must NOT spuriously re-ingest an unchanged KB."""

    async def test_no_spurious_reingest_then_detects_real_change(
        self,
    ) -> None:
        backend = InMemoryKnowledgeBackend()
        await backend.initialize()
        await backend.create_kb("d1")
        await backend.put_file("d1", "docs/topic.md", b"# Topic\n\nHi.\n")

        rag = await RAGKnowledgeBase.from_config(
            {
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_provider": "echo",
                "embedding_model": "test",
            }
        )
        manager = KnowledgeIngestionManager(source=backend, destination=rag)

        first = await manager.ingest("d1")
        assert first.files_processed >= 1

        # Canonical version (== get_checksum), captured by a consumer.
        version = await manager.get_current_version("d1")
        assert version

        # Nothing changed → must skip (pre-fix this re-ingested forever).
        skipped = await manager.ingest_if_changed("d1", last_version=version)
        assert skipped is None

        # A real edit → re-ingest happens.
        await backend.put_file("d1", "docs/topic.md", b"# Topic\n\nEdited.\n")
        redone = await manager.ingest_if_changed("d1", last_version=version)
        assert redone is not None
        assert redone.files_processed >= 1

        await rag.close()
        await backend.close()


def test_normalize_ingestion_status_rejects_unknown_with_validation_error() -> None:
    """An invalid status string raises ``ValidationError``.

    ``normalize_ingestion_status`` previously surfaced the bare
    ``ValueError`` from ``IngestionStatus(<bad>)`` — an opaque
    ``'xyz' is not a valid IngestionStatus`` with no list of accepted
    values, and the wrong exception type for the project's contract
    (``ValidationError`` is a ``DataknobsError``, **not** a
    ``ValueError`` subclass). The message must enumerate the valid
    statuses so a caller can self-correct.
    """
    from dataknobs_common.exceptions import ValidationError

    from dataknobs_bots.knowledge.storage.models import (
        IngestionStatus,
        normalize_ingestion_status,
    )

    # The typed and valid-string forms still pass through unchanged.
    assert (
        normalize_ingestion_status(IngestionStatus.READY)
        is IngestionStatus.READY
    )
    assert (
        normalize_ingestion_status("swapping")
        is IngestionStatus.SWAPPING
    )

    with pytest.raises(ValidationError) as excinfo:
        normalize_ingestion_status("not-a-real-status")

    msg = str(excinfo.value)
    for member in IngestionStatus:
        assert member.value in msg
    # Contract: ValidationError is NOT a ValueError (a bare
    # `except ValueError` must not silently swallow it).
    assert not isinstance(excinfo.value, ValueError)

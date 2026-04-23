"""Tests for :meth:`RAGKnowledgeBase.ingest_from_backend` (Phase 2).

Cross-backend parity guarantees the unified ingest pipeline behaves
identically whether driven from a local directory, an in-memory
backend, or a file backend. Pattern-based behavior that
``KnowledgeIngestionManager`` silently dropped before the unification
is now exercised end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_bots.knowledge.storage import (
    FileKnowledgeBackend,
    InMemoryKnowledgeBackend,
)
from dataknobs_xization.ingestion import (
    BackendDocumentSource,
    FilePatternConfig,
    KnowledgeBaseConfig,
)


async def _make_kb() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


async def _collected_paths(
    source: BackendDocumentSource, patterns: list[str]
) -> list[str]:
    out: list[str] = []
    async for ref in source.iter_files(patterns):
        out.append(ref.path)
    return out


@pytest.fixture
async def populated_memory_backend() -> InMemoryKnowledgeBackend:
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    await backend.create_kb("d1")
    await backend.put_file("d1", "intro.md", b"# Intro\n\nHello.\n")
    await backend.put_file("d1", "guide.md", b"# Guide\n\nBody.\n")
    await backend.put_file("d1", "drafts/skip.md", b"# Skip me\n")
    await backend.put_file(
        "d1", "data.jsonl", b'{"title": "A"}\n{"title": "B"}\n'
    )
    return backend


@pytest.mark.asyncio
async def test_local_ingest_equivalent_to_load_from_directory(
    tmp_path: Path,
) -> None:
    """Same corpus via local path and FileKnowledgeBackend → same stats."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "intro.md").write_text("# Intro\n\nHello.\n")
    (corpus / "guide.md").write_text("# Guide\n\nBody.\n")

    config = KnowledgeBaseConfig(name="t")

    kb_local = await _make_kb()
    local_stats = await kb_local.load_from_directory(corpus, config=config)

    # Use FileKnowledgeBackend rooted at tmp_path; store docs under its
    # managed "content/" layout via put_file.
    backend = FileKnowledgeBackend.from_config({"path": str(tmp_path / "store")})
    await backend.initialize()
    await backend.create_kb("d1")
    for src in corpus.iterdir():
        await backend.put_file("d1", src.name, src.read_bytes())

    kb_backend = await _make_kb()
    backend_stats = await kb_backend.ingest_from_backend(
        backend, "d1", config=config
    )
    await backend.close()

    assert local_stats["total_files"] == backend_stats["total_files"]
    assert local_stats["total_chunks"] == backend_stats["total_chunks"]
    assert (
        local_stats["files_by_type"] == backend_stats["files_by_type"]
    )


@pytest.mark.asyncio
async def test_ingest_from_memory_backend(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    kb = await _make_kb()
    stats = await kb.ingest_from_backend(
        populated_memory_backend, "d1", config=KnowledgeBaseConfig(name="t")
    )
    # 3 markdown + 1 jsonl (drafts/skip.md is matched by default **/*.md)
    assert stats["total_files"] == 4
    assert stats["total_chunks"] > 0
    assert stats["files_by_type"]["markdown"] == 3
    assert stats["files_by_type"]["jsonl"] == 1


@pytest.mark.asyncio
async def test_ingest_respects_exclude_patterns(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    config = KnowledgeBaseConfig(
        name="t", exclude_patterns=["drafts/**"]
    )
    kb = await _make_kb()
    stats = await kb.ingest_from_backend(
        populated_memory_backend, "d1", config=config
    )
    # drafts/skip.md excluded → 2 markdown + 1 jsonl
    assert stats["total_files"] == 3
    assert stats["files_by_type"]["markdown"] == 2


@pytest.mark.asyncio
async def test_ingest_respects_patterns(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    """Only markdown files at the root match; JSONL excluded."""
    config = KnowledgeBaseConfig(
        name="t",
        patterns=[FilePatternConfig(pattern="*.md")],
    )
    kb = await _make_kb()
    stats = await kb.ingest_from_backend(
        populated_memory_backend, "d1", config=config
    )
    # Only intro.md and guide.md at the root (drafts/skip.md is nested)
    assert stats["total_files"] == 2
    assert stats["files_by_type"]["markdown"] == 2
    assert stats["files_by_type"]["jsonl"] == 0


@pytest.mark.asyncio
async def test_ingest_progress_callback_invoked(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    recorded: list[tuple[str, int]] = []

    def cb(path: str, n: int) -> None:
        recorded.append((path, n))

    kb = await _make_kb()
    stats = await kb.ingest_from_backend(
        populated_memory_backend, "d1", progress_callback=cb
    )
    assert len(recorded) == stats["total_files"]
    assert all(isinstance(p, str) and isinstance(n, int) for p, n in recorded)


@pytest.mark.asyncio
async def test_ingest_reads_config_from_backend_metadata(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    """When no config is passed, backend _metadata/ provides one."""
    import json as json_lib

    payload = json_lib.dumps(
        {
            "name": "backend-config",
            "exclude_patterns": ["drafts/**"],
        }
    ).encode("utf-8")
    await populated_memory_backend.put_file(
        "d1", "_metadata/knowledge_base.json", payload
    )

    kb = await _make_kb()
    stats = await kb.ingest_from_backend(populated_memory_backend, "d1")
    # Exclude honored → drafts/skip.md skipped
    assert stats["files_by_type"]["markdown"] == 2


@pytest.mark.asyncio
async def test_ingest_uses_defaults_when_no_config(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    kb = await _make_kb()
    stats = await kb.ingest_from_backend(populated_memory_backend, "d1")
    # No config → defaults match everything including drafts/
    assert stats["total_files"] == 4


@pytest.mark.asyncio
async def test_backend_document_source_iter_files_matches_fnmatch(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    source = BackendDocumentSource(populated_memory_backend, "d1")
    md_paths = sorted(await _collected_paths(source, ["*.md"]))
    assert md_paths == ["guide.md", "intro.md"]

    nested_md = sorted(await _collected_paths(source, ["**/*.md"]))
    assert "drafts/skip.md" in nested_md


@pytest.mark.asyncio
async def test_backend_document_source_empty_patterns_yields_all(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    source = BackendDocumentSource(populated_memory_backend, "d1")
    all_paths = sorted(await _collected_paths(source, []))
    assert "intro.md" in all_paths
    assert "data.jsonl" in all_paths


@pytest.mark.asyncio
async def test_pattern_intersection_across_local_and_backend(
    tmp_path: Path,
) -> None:
    """LocalDocumentSource and BackendDocumentSource(FileKnowledgeBackend)
    yield the same set of files for the same patterns."""
    from typing import Any

    from dataknobs_xization.ingestion.source import LocalDocumentSource

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "intro.md").write_text("# Intro\n")
    (corpus / "guide.md").write_text("# Guide\n")
    (corpus / "sub").mkdir()
    (corpus / "sub" / "note.md").write_text("# Note\n")

    local = LocalDocumentSource(corpus)
    backend = FileKnowledgeBackend.from_config({"path": str(tmp_path / "store")})
    await backend.initialize()
    await backend.create_kb("corpus")
    for p in corpus.rglob("*.md"):
        rel = p.relative_to(corpus).as_posix()
        await backend.put_file("corpus", rel, p.read_bytes())

    remote = BackendDocumentSource(backend, "corpus")

    async def _collect_from(source: Any) -> set[str]:
        seen: set[str] = set()
        async for ref in source.iter_files(["**/*.md"]):
            seen.add(ref.path)
        return seen

    local_set = await _collect_from(local)
    remote_set = await _collect_from(remote)
    await backend.close()
    assert local_set == remote_set


@pytest.mark.asyncio
async def test_ingest_captures_errors_without_failing_batch(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    await populated_memory_backend.put_file(
        "d1", "broken.json", b"{ not valid json"
    )
    kb = await _make_kb()
    stats = await kb.ingest_from_backend(populated_memory_backend, "d1")
    # At least the broken file is listed as errored; everything else processed
    assert len(stats["errors"]) >= 1
    assert any("broken.json" in e["file"] for e in stats["errors"])
    # Valid files still processed
    assert stats["total_files"] >= 4


@pytest.mark.asyncio
async def test_ingest_raises_ingestion_config_error_on_malformed_json(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    """Malformed JSON in backend config raises ``IngestionConfigError``.

    Symmetric with :meth:`KnowledgeBaseConfig.load` on a local
    directory: both loaders must fail loudly rather than silently
    falling back to defaults, so a broken config doesn't quietly
    corrupt every downstream ingest.
    """
    from dataknobs_xization.ingestion.config import IngestionConfigError

    await populated_memory_backend.put_file(
        "d1", "_metadata/knowledge_base.json", b"{ not valid json"
    )
    kb = await _make_kb()
    with pytest.raises(IngestionConfigError, match="Failed to parse"):
        await kb.ingest_from_backend(populated_memory_backend, "d1")


@pytest.mark.asyncio
async def test_ingest_raises_ingestion_config_error_on_non_dict(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    """Config that parses to a non-dict (e.g. a list) raises
    ``IngestionConfigError`` — symmetric with
    :meth:`KnowledgeBaseConfig._load_file`."""
    from dataknobs_xization.ingestion.config import IngestionConfigError

    await populated_memory_backend.put_file(
        "d1", "knowledge_base.json", b"[1, 2, 3]"
    )
    kb = await _make_kb()
    with pytest.raises(IngestionConfigError, match="did not decode to a dict"):
        await kb.ingest_from_backend(populated_memory_backend, "d1")


@pytest.mark.asyncio
async def test_ingest_root_config_takes_precedence_over_metadata(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    """Root-level ``knowledge_base.*`` wins over ``_metadata/``
    when both are present — matches the candidate order in
    :meth:`RAGKnowledgeBase._load_kb_config_from_backend`.
    """
    import json as json_lib

    root_cfg = {"name": "root", "exclude_patterns": ["drafts/**"]}
    meta_cfg = {"name": "meta", "exclude_patterns": []}
    await populated_memory_backend.put_file(
        "d1", "knowledge_base.json", json_lib.dumps(root_cfg).encode("utf-8")
    )
    await populated_memory_backend.put_file(
        "d1",
        "_metadata/knowledge_base.json",
        json_lib.dumps(meta_cfg).encode("utf-8"),
    )

    kb = await _make_kb()
    stats = await kb.ingest_from_backend(populated_memory_backend, "d1")
    # Root excluded drafts/; meta did not. If root wins, drafts/skip.md
    # is skipped → only 2 markdown docs (intro + guide).
    assert stats["files_by_type"]["markdown"] == 2


@pytest.mark.asyncio
async def test_ingest_isolates_per_file_embed_store_failures(
    populated_memory_backend: InMemoryKnowledgeBackend,
) -> None:
    """A failure during embed/store for one file does not abort the
    batch — other files are still ingested, and the failing file is
    captured in ``stats["errors"]``.

    Regression for H#4: :meth:`_ingest_from_processor_async` must wrap
    each doc's :meth:`_embed_and_store_chunks` call in its own
    try/except. Without per-file isolation a single flaky embed would
    drop the rest of the batch — the pre-unification
    ``KnowledgeIngestionManager._ingest_file`` semantics guaranteed
    isolation and the unified pipeline must preserve it.
    """
    kb = await _make_kb()

    real_embed_and_store = kb._embed_and_store_chunks  # type: ignore[attr-defined]

    async def _flaky(
        *args: Any, source_file: str = "", **kwargs: Any
    ) -> int:
        if "guide.md" in source_file:
            raise RuntimeError("simulated embed/store failure on guide.md")
        return await real_embed_and_store(
            *args, source_file=source_file, **kwargs
        )

    kb._embed_and_store_chunks = _flaky  # type: ignore[assignment,method-assign]

    stats = await kb.ingest_from_backend(populated_memory_backend, "d1")

    # guide.md shows up as an error but other files are still ingested.
    assert any(
        "guide.md" in e["file"] and "simulated" in e["error"]
        for e in stats["errors"]
    ), f"Expected guide.md to be flagged as errored; got {stats['errors']}"
    # Exactly one file failed; the rest succeeded.
    failed = {Path(e["file"]).name for e in stats["errors"]}
    assert failed == {"guide.md"}
    # At least intro.md, drafts/skip.md and data.jsonl made it through.
    assert stats["total_files"] >= 3
    assert stats["total_chunks"] >= 3

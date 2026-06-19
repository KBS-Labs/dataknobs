"""Async-correctness tests for ``FileKnowledgeBackend`` (Shape-B offload).

The file backend does synchronous disk I/O (``open``, ``json.load``,
``tempfile``/``os.replace``, ``shutil.rmtree``) in every async storage
method. Run directly on the event loop those calls stall it. The fix
offloads them via ``asyncio.to_thread``.

These reproduce-first tests wrap each awaited operation in
``assert_no_blocking()``: each FAILS against the pre-offload code (the
detector catches the blocking syscall) and PASSES once the backend
offloads. They need no external service — a temp directory is the whole
fixture — so they run in the always-on unit pass. Functional round-trip
assertions guard against the refactor breaking behavior. No mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend
from dataknobs_common.testing import assert_no_blocking, is_blockbuster_available

requires_blockbuster = pytest.mark.skipif(
    not is_blockbuster_available(),
    reason="blockbuster not installed",
)


async def _make_backend(tmp_path: Path) -> FileKnowledgeBackend:
    backend = FileKnowledgeBackend(base_path=tmp_path / "kb")
    await backend.initialize()
    return backend


@requires_blockbuster
async def test_create_kb_does_not_block(tmp_path: Path) -> None:
    backend = await _make_backend(tmp_path)
    with assert_no_blocking():
        await backend.create_kb("domain")


@requires_blockbuster
async def test_put_and_get_file_do_not_block(tmp_path: Path) -> None:
    backend = await _make_backend(tmp_path)
    await backend.create_kb("domain")
    with assert_no_blocking():
        await backend.put_file("domain", "doc.md", b"# Hello")
        content = await backend.get_file("domain", "doc.md")
    assert content == b"# Hello"


@requires_blockbuster
async def test_stream_file_does_not_block(tmp_path: Path) -> None:
    backend = await _make_backend(tmp_path)
    await backend.create_kb("domain")
    await backend.put_file("domain", "doc.md", b"abcdefgh" * 4)
    with assert_no_blocking():
        stream = await backend.stream_file("domain", "doc.md", chunk_size=8)
        assert stream is not None
        chunks = [chunk async for chunk in stream]
    assert b"".join(chunks) == b"abcdefgh" * 4


@requires_blockbuster
async def test_delete_file_does_not_block(tmp_path: Path) -> None:
    backend = await _make_backend(tmp_path)
    await backend.create_kb("domain")
    await backend.put_file("domain", "doc.md", b"data")
    with assert_no_blocking():
        deleted = await backend.delete_file("domain", "doc.md")
    assert deleted is True


@requires_blockbuster
async def test_set_ingestion_status_does_not_block(tmp_path: Path) -> None:
    backend = await _make_backend(tmp_path)
    await backend.create_kb("domain")
    with assert_no_blocking():
        await backend.set_ingestion_status("domain", "ready")


@requires_blockbuster
async def test_list_and_info_do_not_block(tmp_path: Path) -> None:
    backend = await _make_backend(tmp_path)
    await backend.create_kb("domain")
    await backend.put_file("domain", "doc.md", b"data")
    with assert_no_blocking():
        files = await backend.list_files("domain")
        info = await backend.get_info("domain")
        kbs = await backend.list_kbs()
    assert [f.path for f in files] == ["doc.md"]
    assert info is not None and info.domain_id == "domain"
    assert [kb.domain_id for kb in kbs] == ["domain"]


@requires_blockbuster
async def test_delete_kb_does_not_block(tmp_path: Path) -> None:
    backend = await _make_backend(tmp_path)
    await backend.create_kb("domain")
    with assert_no_blocking():
        deleted = await backend.delete_kb("domain")
    assert deleted is True


async def test_functional_roundtrip_preserved(tmp_path: Path) -> None:
    """The offload refactor must not change observable behavior."""
    backend = await _make_backend(tmp_path)
    await backend.create_kb("domain")
    await backend.put_file("domain", "a/doc.md", b"alpha")
    await backend.put_file("domain", "b/doc.json", b"{}")

    assert await backend.get_file("domain", "a/doc.md") == b"alpha"
    assert await backend.file_exists("domain", "b/doc.json") is True
    assert await backend.get_file("domain", "missing.md") is None

    files = await backend.list_files("domain")
    assert sorted(f.path for f in files) == ["a/doc.md", "b/doc.json"]

    assert await backend.delete_file("domain", "a/doc.md") is True
    assert await backend.get_file("domain", "a/doc.md") is None
    assert await backend.delete_kb("domain") is True
    assert await backend.get_info("domain") is None

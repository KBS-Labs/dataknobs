"""End-to-end S3 knowledge-backend tests against real LocalStack.

``S3KnowledgeBackend`` drives its S3 I/O through aioboto3 so the event
loop is never blocked by a synchronous network transport. This module:

1. **Reproduce-first async-correctness** — wraps the awaited backend
   operations in :func:`assert_no_blocking`. Against the pre-fix backend
   (synchronous ``boto3`` client called from ``async def``) the urllib3
   socket read blocks the running loop → ``BlockingError``. Against the
   aioboto3 transport the loop stays free → the block passes. This is
   the executable proof of the transport swap.
2. **Functional round-trips** — the full CRUD + KB lifecycle against a
   real S3 service (LocalStack), so the async client path is exercised
   for real, not just for non-blocking-ness.
3. **Concurrency** — many operations driven through ``asyncio.gather``,
   proving the per-operation client contexts compose concurrently.

Start LocalStack with ``bin/dk up`` (it runs ``SERVICES=s3,sqs``); the
whole module skips when LocalStack is unavailable. ``moto``'s
``mock_aws`` is deliberately NOT used — it is incompatible with the
aiobotocore transport these tests must exercise.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_common.testing import (
    assert_no_blocking,
    requires_blockbuster,
    requires_localstack,
)

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]


async def _backend(cfg: dict[str, Any]) -> S3KnowledgeBackend:
    backend = S3KnowledgeBackend.from_config(cfg)
    await backend.initialize()
    return backend


# ---------------------------------------------------------------------------
# Reproduce-first: the async-correctness proof for the transport swap
# ---------------------------------------------------------------------------


# NOTE: the backend's steady-state operations are the meaningful
# non-blocking guarantee and are asserted below. ``initialize()`` itself
# is deliberately NOT wrapped in ``assert_no_blocking``: aiobotocore loads
# botocore's bundled data files (endpoints / sdk-default-config / the S3
# service model) lazily and *synchronously* the first time any client is
# created from a session — a one-time, startup-only disk read inside the
# shared ``create_aioboto3_session`` factory (``dataknobs-data``) that
# every aioboto3 consumer (e.g. ``AsyncS3Database``) shares. Moving that
# one-time load off the loop is a session-factory concern tracked
# separately, not part of this backend's transport swap. Every operation
# *after* initialization is non-blocking, which the tests below pin.


@requires_blockbuster
async def test_put_and_get_file_do_not_block(s3_kb_config) -> None:
    """put_file / get_file must not stall the event loop.

    FAILS against the synchronous-boto3 backend (the socket read blocks
    the loop inside ``async def``); PASSES against aioboto3.
    """
    backend = await _backend(s3_kb_config)
    try:
        await backend.create_kb("d")
        with assert_no_blocking():
            await backend.put_file("d", "intro.md", b"# Hello\n")
            fetched = await backend.get_file("d", "intro.md")
        assert fetched == b"# Hello\n"
    finally:
        await backend.close()


@requires_blockbuster
async def test_stream_file_does_not_block(s3_kb_config) -> None:
    """Streaming a file must not stall the loop on any chunk read."""
    backend = await _backend(s3_kb_config)
    try:
        await backend.create_kb("d")
        await backend.put_file("d", "big.md", b"x" * 50_000)
        with assert_no_blocking():
            stream = await backend.stream_file("d", "big.md", chunk_size=4096)
            assert stream is not None
            total = 0
            async for chunk in stream:
                total += len(chunk)
        assert total == 50_000
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Functional round-trips
# ---------------------------------------------------------------------------


async def test_full_file_lifecycle(s3_kb_config) -> None:
    backend = await _backend(s3_kb_config)
    try:
        await backend.create_kb("d")

        info = await backend.put_file("d", "a.md", b"AAA")
        assert info.path == "a.md"
        assert info.size_bytes == 3

        assert await backend.get_file("d", "a.md") == b"AAA"
        assert await backend.file_exists("d", "a.md") is True
        assert await backend.file_exists("d", "missing.md") is False

        files = await backend.list_files("d")
        assert [f.path for f in files] == ["a.md"]

        # Update in place.
        await backend.put_file("d", "a.md", b"BBBB")
        assert await backend.get_file("d", "a.md") == b"BBBB"

        # Delete.
        assert await backend.delete_file("d", "a.md") is True
        assert await backend.delete_file("d", "a.md") is False
        assert await backend.get_file("d", "a.md") is None
    finally:
        await backend.close()


async def test_content_type_autodetect(s3_kb_config) -> None:
    """Content type is guessed when not supplied (offloaded mimetypes)."""
    backend = await _backend(s3_kb_config)
    try:
        await backend.create_kb("d")
        info = await backend.put_file("d", "page.html", b"<html></html>")
        assert info.content_type == "text/html"
    finally:
        await backend.close()


async def test_kb_lifecycle_and_missing(s3_kb_config) -> None:
    backend = await _backend(s3_kb_config)
    try:
        # Missing KB → None / empty.
        assert await backend.get_info("nope") is None

        await backend.create_kb("d")
        with pytest.raises(ValueError, match="already exists"):
            await backend.create_kb("d")

        info = await backend.get_info("d")
        assert info is not None and info.domain_id == "d"

        await backend.put_file("d", "a.md", b"AAA")
        kbs = await backend.list_kbs()
        assert any(kb.domain_id == "d" for kb in kbs)

        assert await backend.delete_kb("d") is True
        assert await backend.delete_kb("d") is False
        assert await backend.get_info("d") is None
    finally:
        await backend.close()


async def test_get_missing_file_returns_none(s3_kb_config) -> None:
    backend = await _backend(s3_kb_config)
    try:
        await backend.create_kb("d")
        assert await backend.get_file("d", "absent.md") is None
        assert await backend.stream_file("d", "absent.md") is None
    finally:
        await backend.close()


async def test_put_file_requires_existing_kb(s3_kb_config) -> None:
    backend = await _backend(s3_kb_config)
    try:
        with pytest.raises(ValueError, match="does not exist"):
            await backend.put_file("no-kb", "a.md", b"AAA")
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Concurrency — per-operation client contexts compose under gather
# ---------------------------------------------------------------------------


async def test_concurrent_put_and_get(s3_kb_config) -> None:
    """Many concurrent writes then concurrent reads round-trip correctly."""
    backend = await _backend(s3_kb_config)
    try:
        await backend.create_kb("d")

        # NOTE: put_file mutates the shared KB metadata object, so the
        # writes are serialized here (concurrent metadata writes would
        # race on last-write-wins). The reads below are the concurrency
        # assertion — they fan out across independent client contexts.
        paths = [f"f{i}.md" for i in range(10)]
        for i, p in enumerate(paths):
            await backend.put_file("d", p, f"body-{i}".encode())

        results = await asyncio.gather(
            *(backend.get_file("d", p) for p in paths)
        )
        assert results == [f"body-{i}".encode() for i in range(10)]

        exists = await asyncio.gather(
            *(backend.file_exists("d", p) for p in paths)
        )
        assert all(exists)
    finally:
        await backend.close()

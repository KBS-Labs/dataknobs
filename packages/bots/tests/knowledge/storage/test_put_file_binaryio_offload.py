"""Async-correctness test for ``put_file`` with a file-like ``content``.

Every backend's ``put_file`` accepts ``bytes | BinaryIO``. A real OS file
handle's ``read()`` is a blocking syscall; called directly on the event loop
it stalls it. The shared ``KnowledgeResourceBackendMixin._read_content_bytes``
helper offloads the read via ``asyncio.to_thread`` for all three backends.

This reproduce-first test passes a real open file handle to ``put_file`` and
wraps the call in ``assert_no_blocking()`` (FAILS against the inline
``content.read()``, PASSES once routed through the offloading helper). The
in-memory backend is used so the file-handle read is the ONLY potential
blocking source — no backend disk I/O confuses the detector. A functional
assertion guards the round-trip. No mocks.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_bots.knowledge.storage.memory import InMemoryKnowledgeBackend
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster


@requires_blockbuster
async def test_put_file_with_file_handle_does_not_block(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"binary payload" * 64)

    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    await backend.create_kb("domain")

    # Open a real OS file handle: its blocking read must be offloaded.
    with open(source, "rb") as handle:
        with assert_no_blocking():
            file_info = await backend.put_file("domain", "doc.bin", handle)

    assert file_info.size_bytes == len(b"binary payload" * 64)
    assert await backend.get_file("domain", "doc.bin") == b"binary payload" * 64

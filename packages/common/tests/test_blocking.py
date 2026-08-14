"""Behavioral tests for the ``assert_no_blocking`` detector construct.

These prove the reproduce-first tool actually works: a blocking syscall on
the event loop is detected (the red), and the same work done via an async
transport or ``asyncio.to_thread`` offload is allowed (the green). Every
async-correctness fix downstream relies on this construct distinguishing
the two, so it is itself tested against real blocking and non-blocking
calls — no mocks.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from dataknobs_common.testing import (
    assert_no_blocking,
    blocking_error_type,
    is_blockbuster_available,
    requires_blockbuster,
)
from dataknobs_common.testing import blocking as blocking_module


def test_blockbuster_is_available_in_dev_env() -> None:
    """blockbuster is a declared dev dependency, so the detector is usable."""
    assert is_blockbuster_available() is True


def test_blocking_error_type_is_an_exception() -> None:
    assert issubclass(blocking_error_type(), Exception)


@requires_blockbuster
async def test_detects_blocking_sleep_on_loop() -> None:
    """A synchronous ``time.sleep`` on the loop is caught (the red)."""
    with pytest.raises(blocking_error_type()):
        with assert_no_blocking():
            # Waived per line, because this file takes no per-file waiver: it is
            # the detector's own suite, so switching ASYNC off over it would
            # waive the check across the code most likely to need it. The sleep
            # must be synchronous and must run on a live loop — that is the
            # thing being detected — so neither `await` nor a sync test is the
            # fix here.
            time.sleep(0.01)  # noqa: ASYNC251


@requires_blockbuster
async def test_allows_async_sleep() -> None:
    """An ``await asyncio.sleep`` yields to the loop — allowed (the green)."""
    with assert_no_blocking():
        await asyncio.sleep(0.01)


@requires_blockbuster
async def test_detects_blocking_file_read_on_loop(tmp_path: Path) -> None:
    """Synchronous ``open()``/read on the loop is caught — models the file
    backend defect (sync disk I/O inside an ``async def``)."""
    target = tmp_path / "doc.txt"
    target.write_text("payload")

    with pytest.raises(blocking_error_type()):
        with assert_no_blocking():
            # Deliberate, and for the same reason as the sleep above: the
            # blocking open IS the input to the detector under test.
            with open(target) as handle:  # noqa: ASYNC230
                handle.read()


@requires_blockbuster
async def test_allows_offloaded_file_read(tmp_path: Path) -> None:
    """The same blocking read offloaded via ``asyncio.to_thread`` is allowed —
    models the Shape-B fix the file backends adopt."""
    target = tmp_path / "doc.txt"
    target.write_text("payload")

    with assert_no_blocking():
        content = await asyncio.to_thread(target.read_text)

    assert content == "payload"


async def test_raises_runtime_error_when_detector_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without blockbuster the context manager fails loudly rather than
    silently passing without detection."""
    monkeypatch.setattr(blocking_module, "is_blockbuster_available", lambda: False)
    with pytest.raises(RuntimeError, match="blockbuster"):
        with assert_no_blocking():
            pass


@requires_blockbuster
async def test_no_blocking_fixture_allows_clean_async_work(
    no_blocking: None,
) -> None:
    """The ``no_blocking`` fixture wraps the whole test; non-blocking work
    passes under it."""
    await asyncio.sleep(0)
    await asyncio.to_thread(lambda: time.sleep(0.001))

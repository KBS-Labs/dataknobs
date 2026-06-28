"""Timeout behaviour of the Simple API (``api/simple.py``) — real constructs only.

Every synchronous Simple-API surface now runs on the single
``AsyncExecutionEngine`` via the FSM's async->sync bridge, and ``timeout=`` is
bounded by that bridge: on expiry the in-flight coroutine is cancelled
(best-effort, at its next ``await`` point) and the caller's wait returns
promptly, instead of the old ``ThreadPoolExecutor`` that blocked on
``shutdown(wait=True)`` until the work finished anyway.

These tests pin that behaviour at each Simple-API entry point with **real FSM
builds and real transforms** — no ``Mock``/``patch``. A slow ``async`` transform
(``asyncio.sleep`` far longer than the timeout) stands in for genuinely slow
work; a bounded timeout proves the wait is cut short rather than waiting it out.

Covered surfaces:

* ``SimpleFSM.process`` — success, no-timeout, bounded-timeout error result +
  message shape, and resource release on ``close()`` after a timeout.
* ``process_file`` module helper — success over a real file, bounded timeout.
* ``batch_process`` module helper — success over real records, bounded timeout.

The bridge-routing / no-leaked-thread guarantees of these same entry points are
covered in ``test_sync_entrypoint_bridge.py``; this file focuses on the
timeout-result *shape*, *message*, and *resource cleanup* contracts.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from dataknobs_fsm.api.simple import SimpleFSM, batch_process, process_file


def _linear_config(name: str, transform_name: str) -> dict[str, Any]:
    """A minimal start->end FSM whose only arc runs ``transform_name``."""
    return {
        "name": name,
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "end",
                        "name": "go",
                        "transform": {"type": "registered", "name": transform_name},
                    }
                ],
            }
        ],
    }


async def _slow_transform(data: Any, context: Any) -> Any:
    """A transform that sleeps far longer than any test timeout."""
    import asyncio

    await asyncio.sleep(5)
    return data


# --------------------------------------------------------------------------- #
# SimpleFSM.process
# --------------------------------------------------------------------------- #


def test_process_within_timeout_succeeds() -> None:
    """``process(timeout=)`` returns a success result when work finishes in time."""
    ran: list[str] = []

    def stamp(data: Any, context: Any) -> Any:
        ran.append("ran")
        return data

    fsm = SimpleFSM(_linear_config("quick_fsm", "stamp"), custom_functions={"stamp": stamp})
    try:
        result = fsm.process({"input": "test"}, timeout=5.0)
        assert result["success"] is True
        assert result["final_state"] == "end"
        assert result.get("error") in (None, "")
        assert ran == ["ran"], "the real transform did not run on the timeout path"
    finally:
        fsm.close()


def test_process_without_timeout_succeeds() -> None:
    """``process()`` with no timeout runs on the bridge and succeeds."""
    fsm = SimpleFSM(
        _linear_config("notimeout_fsm", "stamp"),
        custom_functions={"stamp": lambda d, c: d},
    )
    try:
        result = fsm.process({"input": "test"})
        assert result["success"] is True
        assert result["final_state"] == "end"
    finally:
        fsm.close()


def test_process_timeout_returns_bounded_error_result() -> None:
    """``process(timeout=)`` returns an error result promptly, not after the work.

    The slow transform sleeps 5s; a 0.2s timeout must cut the wait short (well
    under 2s) and surface a timeout error result, rather than blocking until the
    transform finishes anyway.
    """
    fsm = SimpleFSM(
        _linear_config("slow_fsm", "slow"), custom_functions={"slow": _slow_transform}
    )
    try:
        start = time.monotonic()
        result = fsm.process({"input": "test"}, timeout=0.2)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, (
            f"process(timeout=0.2) was not bounded — took {elapsed:.2f}s "
            "(the slow transform sleeps 5s)"
        )
        assert result["success"] is False
        error = (result.get("error") or "").lower()
        assert "timeout" in error
        assert "0.2" in error
        assert "seconds" in error
    finally:
        fsm.close()


def test_process_timeout_releases_resources_on_close() -> None:
    """A timed-out run still releases its acquired state resources on ``close()``.

    ``start`` declares a real in-memory database resource, so entering it
    acquires ``scratch_db`` before the slow transform runs. The timeout cancels
    the in-flight coroutine; ``close()`` must then leave no held resource and no
    bridge thread behind.
    """
    reached: list[str] = []

    async def slow_after_entry(data: Any, context: Any) -> Any:
        # Reaching the transform proves ``start`` was entered, so ``scratch_db``
        # was acquired — making the post-close release assertion non-vacuous.
        import asyncio

        reached.append("transform")
        await asyncio.sleep(5)
        return data

    config = {
        "name": "resource_fsm",
        "main_network": "main",
        "resources": [
            {"name": "scratch_db", "type": "async_database", "config": {"type": "memory"}},
        ],
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True, "resources": ["scratch_db"]},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "end",
                        "name": "go",
                        "transform": {"type": "registered", "name": "slow"},
                    }
                ],
            }
        ],
    }
    fsm = SimpleFSM(config, custom_functions={"slow": slow_after_entry})

    result = fsm.process({"id": 1}, timeout=0.2)
    assert result["success"] is False
    assert reached == ["transform"], (
        "run never reached the transform — resource acquisition was not exercised"
    )

    # Capture THIS FSM's shared bridge thread before close, so the leak check is
    # specific to this FSM. ``process()`` above already created the bridge, so
    # the accessor returns the existing instance rather than spinning up a new
    # one. (A global name scan would falsely fail on other tests' live bridges.)
    bridge_thread = fsm._fsm.get_sync_bridge()._thread

    fsm.close()

    manager = fsm._resource_manager
    assert manager._resources == {}, (
        f"resources leaked after timeout + close: {manager._resources!r}"
    )
    assert all(not owners for owners in manager._resource_owners.values()), (
        f"resource owners leaked after timeout + close: {manager._resource_owners!r}"
    )
    assert not bridge_thread.is_alive(), (
        "this FSM's bridge thread was not joined by close()"
    )


# --------------------------------------------------------------------------- #
# process_file module helper
# --------------------------------------------------------------------------- #


def test_process_file_within_timeout_succeeds() -> None:
    """``process_file(timeout=)`` processes a real file and returns real stats."""
    ran: list[str] = []

    def stamp(data: Any, context: Any) -> Any:
        ran.append("ran")
        return data

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as handle:
        handle.write(json.dumps({"data": "a"}) + "\n")
        handle.write(json.dumps({"data": "b"}) + "\n")
        input_path = Path(handle.name)

    try:
        result = process_file(
            fsm_config=_linear_config("file_fsm", "stamp"),
            input_file=str(input_path),
            timeout=5.0,
            custom_functions={"stamp": stamp},
        )
        assert result["total_processed"] == 2
        assert result["failed"] == 0
        assert len(ran) == 2, "the real transform did not run once per record"
    finally:
        input_path.unlink(missing_ok=True)


def test_process_file_timeout_raises_bounded() -> None:
    """``process_file(timeout=)`` raises ``TimeoutError`` promptly, bounded."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as handle:
        handle.write(json.dumps({"data": "a"}) + "\n")
        handle.write(json.dumps({"data": "b"}) + "\n")
        input_path = Path(handle.name)

    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="File processing exceeded timeout"):
            process_file(
                fsm_config=_linear_config("slow_file_fsm", "slow"),
                input_file=str(input_path),
                timeout=0.3,
                custom_functions={"slow": _slow_transform},
            )
        elapsed = time.monotonic() - start
        assert elapsed < 3.0, (
            f"process_file(timeout=0.3) was not bounded — took {elapsed:.2f}s"
        )
    finally:
        input_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# batch_process module helper
# --------------------------------------------------------------------------- #


def test_batch_process_within_timeout_succeeds() -> None:
    """``batch_process(timeout=)`` processes real records and returns per-record results."""
    ran: list[str] = []

    def stamp(data: Any, context: Any) -> Any:
        ran.append("ran")
        return data

    data = [{"id": i} for i in range(5)]
    result = batch_process(
        fsm_config=_linear_config("batch_fsm", "stamp"),
        data=data,
        batch_size=2,
        max_workers=2,
        timeout=5.0,
        custom_functions={"stamp": stamp},
    )
    assert len(result) == 5
    assert len(ran) == 5, "the real transform did not run once per record"


def test_batch_process_timeout_raises_bounded() -> None:
    """``batch_process(timeout=)`` raises ``TimeoutError`` promptly, bounded."""
    data = [{"id": i} for i in range(5)]
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="Batch processing exceeded timeout"):
        batch_process(
            fsm_config=_linear_config("slow_batch_fsm", "slow"),
            data=data,
            batch_size=2,
            max_workers=2,
            timeout=0.3,
            custom_functions={"slow": _slow_transform},
        )
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, (
        f"batch_process(timeout=0.3) was not bounded — took {elapsed:.2f}s"
    )

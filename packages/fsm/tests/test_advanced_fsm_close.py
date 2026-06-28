"""Reproduce-first: ``AdvancedFSM.close()`` releases the per-FSM bridge thread.

``AdvancedFSM.execute_step_sync`` drives the single async engine through the
FSM's shared async→sync bridge — a daemon event-loop thread created lazily on
the first synchronous step. Before ``AdvancedFSM`` grew a ``close()`` /
``aclose()`` lifecycle, an ``AdvancedFSM`` that only ever stepped synchronously
had no ergonomic way to reach ``FSM.close()`` (the CHANGELOG claimed the bridge
was released by ``FSM.close()`` / ``SimpleFSM.close()`` — neither reachable from
an ``AdvancedFSM`` holder), so that bridge thread lived until process exit.

These tests step synchronously (spinning up the shared bridge), then assert
``close()`` / ``aclose()`` / context-manager exit join its thread. Each captures
the specific bridge thread before teardown, so the leak check is scoped to this
FSM and cannot be confused by another test's live bridge.

Real constructs only — a real FSM build and the real bridge thread; no mocks.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_fsm import AdvancedFSM, create_advanced_fsm


def _stepping_fsm() -> AdvancedFSM:
    """A minimal two-step linear FSM driven via ``execute_step_sync``."""
    config: dict[str, Any] = {
        "name": "stepping_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "middle"},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {"from": "start", "to": "middle", "name": "go"},
                    {"from": "middle", "to": "end", "name": "finish"},
                ],
            }
        ],
    }
    return create_advanced_fsm(config)


def _step_to_start_bridge(fsm: AdvancedFSM) -> Any:
    """Run one sync step (lazily creating the shared bridge) and return its thread."""
    context = fsm.create_context({"value": 1})
    fsm.execute_step_sync(context)
    thread = fsm.fsm.get_sync_bridge()._thread
    assert thread.is_alive(), (
        "execute_step_sync did not start the FSM's shared bridge thread"
    )
    return thread


def test_close_joins_bridge_thread() -> None:
    """``close()`` stops and joins the bridge thread that sync stepping created."""
    fsm = _stepping_fsm()
    thread = _step_to_start_bridge(fsm)

    fsm.close()

    assert not thread.is_alive(), (
        "AdvancedFSM.close() did not join the FSM's shared bridge thread — a "
        "sync-only AdvancedFSM would leak it until process exit"
    )
    # Idempotent — a second close after the bridge is gone must not raise.
    fsm.close()


@pytest.mark.asyncio
async def test_aclose_joins_bridge_thread() -> None:
    """``aclose()`` is the async-context counterpart and joins the same thread."""
    fsm = _stepping_fsm()
    thread = _step_to_start_bridge(fsm)

    await fsm.aclose()

    assert not thread.is_alive(), (
        "AdvancedFSM.aclose() did not join the FSM's shared bridge thread"
    )


def test_context_manager_closes_bridge_thread() -> None:
    """Using ``AdvancedFSM`` as a sync context manager closes the bridge on exit."""
    fsm = _stepping_fsm()
    with fsm:
        thread = _step_to_start_bridge(fsm)
    assert not thread.is_alive(), (
        "exiting the AdvancedFSM context manager did not close the bridge thread"
    )

"""Tests for the strategy-agnostic ``LifecycleHooks`` surface.

Exercises the class directly (no wizard dependency) so that
non-wizard ``ReasoningStrategy`` adopters can rely on the same
contract the wizard's ``WizardHooks`` composes. Callbacks and
triggers exchange a single opaque ``event: dict[str, Any]``
payload — adopters add subsystem-specific keys without extending
the trigger signature. See the
:mod:`dataknobs_bots.reasoning.lifecycle` module docstring for
canonical event keys.
"""
from __future__ import annotations

from typing import Any

import pytest


def _import_lifecycle():
    """Lazy import — until the module ships these tests collect-fail (RED)."""
    from dataknobs_bots.reasoning.lifecycle import (
        LifecycleHooks,
        TurnHookCallback,
    )
    return LifecycleHooks, TurnHookCallback


def _evt(stage: str = "s", **extra: Any) -> dict[str, Any]:
    """Canonical event payload for tests — minimal shape plus overrides."""
    return {"stage": stage, "phase": "start", "reason": "normal", **extra}


# ---------------------------------------------------------------------------
# Registration + triggering — async callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_start_fires_registered_async_callback() -> None:
    LifecycleHooks, _ = _import_lifecycle()
    fired: list[dict[str, Any]] = []

    async def cb(event: dict[str, Any]) -> None:
        fired.append(event)

    hooks = LifecycleHooks()
    hooks.on_turn_start(cb)

    event = _evt(stage="stage_a", manager=object(), state=object())
    await hooks.trigger_turn_start(event)

    assert len(fired) == 1
    assert fired[0] is event
    assert fired[0]["stage"] == "stage_a"


@pytest.mark.asyncio
async def test_on_turn_end_fires_registered_async_callback() -> None:
    LifecycleHooks, _ = _import_lifecycle()
    fired: list[str] = []

    async def cb(event: dict[str, Any]) -> None:
        fired.append(event["stage"])

    hooks = LifecycleHooks()
    hooks.on_turn_end(cb)

    await hooks.trigger_turn_end(_evt(stage="stage_b", phase="end"))
    assert fired == ["stage_b"]


# ---------------------------------------------------------------------------
# Sync callbacks accepted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_callback_accepted_for_turn_start() -> None:
    LifecycleHooks, _ = _import_lifecycle()
    fired = 0

    def sync_cb(event: dict[str, Any]) -> None:
        nonlocal fired
        fired += 1

    hooks = LifecycleHooks()
    hooks.on_turn_start(sync_cb)
    await hooks.trigger_turn_start(_evt())
    assert fired == 1


# ---------------------------------------------------------------------------
# Stage scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_scoped_hook_fires_only_for_its_stage() -> None:
    LifecycleHooks, _ = _import_lifecycle()
    seen: list[str] = []

    async def cb(event: dict[str, Any]) -> None:
        seen.append(event["stage"])

    hooks = LifecycleHooks()
    hooks.on_turn_start(cb, stage="alpha")

    await hooks.trigger_turn_start(_evt(stage="alpha"))
    await hooks.trigger_turn_start(_evt(stage="beta"))
    assert seen == ["alpha"]


@pytest.mark.asyncio
async def test_global_hook_fires_for_every_stage() -> None:
    LifecycleHooks, _ = _import_lifecycle()
    seen: list[str] = []

    async def cb(event: dict[str, Any]) -> None:
        seen.append(event["stage"])

    hooks = LifecycleHooks()
    hooks.on_turn_start(cb)  # no stage = global

    await hooks.trigger_turn_start(_evt(stage="alpha"))
    await hooks.trigger_turn_start(_evt(stage="beta"))
    assert seen == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Multiple callbacks fire in registration order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_callbacks_fire_in_registration_order() -> None:
    LifecycleHooks, _ = _import_lifecycle()
    order: list[str] = []

    async def first(event: dict[str, Any]) -> None:
        order.append("first")

    async def second(event: dict[str, Any]) -> None:
        order.append("second")

    hooks = LifecycleHooks()
    hooks.on_turn_start(first)
    hooks.on_turn_start(second)

    await hooks.trigger_turn_start(_evt())
    assert order == ["first", "second"]


# ---------------------------------------------------------------------------
# Empty registry: trigger is a clean no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_on_empty_registry_is_noop() -> None:
    LifecycleHooks, _ = _import_lifecycle()
    hooks = LifecycleHooks()
    # Must not raise — no hooks registered.
    await hooks.trigger_turn_start(_evt())
    await hooks.trigger_turn_end(_evt(phase="end"))


# ---------------------------------------------------------------------------
# from_config: dotted-path callback resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_config_loads_dotted_path_callbacks(monkeypatch) -> None:
    """Pins the config-driven shape used by wizard YAML adopters."""
    LifecycleHooks, _ = _import_lifecycle()

    # Register a callable in this module's namespace so the dotted-path
    # resolver can find it.
    fired: list[str] = []

    async def _registered_hook(event: dict[str, Any]) -> None:
        fired.append(event["stage"])

    import sys
    sys.modules[__name__]._registered_hook = _registered_hook  # type: ignore[attr-defined]

    config = {
        "on_turn_start": [
            {"function": f"{__name__}:_registered_hook"},
        ],
        "on_turn_end": [
            f"{__name__}:_registered_hook",
        ],
    }
    hooks = LifecycleHooks.from_config(config)
    await hooks.trigger_turn_start(_evt(stage="x"))
    await hooks.trigger_turn_end(_evt(stage="y", phase="end"))
    assert fired == ["x", "y"]


def test_from_config_with_unparseable_path_skips_silently(caplog) -> None:
    """Malformed dotted paths log a WARNING and skip — no crash."""
    LifecycleHooks, _ = _import_lifecycle()
    caplog.set_level("WARNING", logger="dataknobs_bots.reasoning.lifecycle")
    hooks = LifecycleHooks.from_config(
        {"on_turn_start": [{"function": "not_a_valid_path"}]},
    )
    assert len(hooks._turn_start_hooks) == 0


def test_from_config_with_stage_scoping_round_trips() -> None:
    """Per-stage scoping carries through from_config."""
    LifecycleHooks, _ = _import_lifecycle()
    config = {
        "on_turn_start": [
            {"function": f"{__name__}:_registered_hook_for_stage", "stage": "alpha"},
        ],
    }

    async def _registered_hook_for_stage(event: dict[str, Any]) -> None:
        pass

    import sys
    sys.modules[__name__]._registered_hook_for_stage = _registered_hook_for_stage  # type: ignore[attr-defined]

    hooks = LifecycleHooks.from_config(config)
    assert len(hooks._turn_start_hooks) == 1
    assert hooks._turn_start_hooks[0].stage == "alpha"


# ---------------------------------------------------------------------------
# Public introspection + reset
# ---------------------------------------------------------------------------


def test_turn_count_properties_track_registrations() -> None:
    """``turn_start_count`` / ``turn_end_count`` are the public surfaces
    ``WizardHooks.hook_count`` and consumer diagnostics use; they replace
    the prior private-list-length accesses.
    """
    LifecycleHooks, _ = _import_lifecycle()
    hooks = LifecycleHooks()
    assert hooks.turn_start_count == 0
    assert hooks.turn_end_count == 0

    async def noop(event: dict[str, Any]) -> None:
        pass

    hooks.on_turn_start(noop)
    hooks.on_turn_start(noop, stage="alpha")
    hooks.on_turn_end(noop)
    assert hooks.turn_start_count == 2
    assert hooks.turn_end_count == 1


@pytest.mark.asyncio
async def test_clear_drains_in_place_and_preserves_identity() -> None:
    """``clear()`` drops every registered callback AND preserves the
    instance identity (so a consumer holding a reference via
    :attr:`WizardHooks.lifecycle` remains attached after a clear).
    """
    LifecycleHooks, _ = _import_lifecycle()
    fired: list[str] = []

    async def cb(event: dict[str, Any]) -> None:
        fired.append(event["stage"])

    hooks = LifecycleHooks()
    hooks.on_turn_start(cb)
    hooks.on_turn_end(cb)
    assert hooks.turn_start_count == 1
    assert hooks.turn_end_count == 1

    hooks.clear()

    assert hooks.turn_start_count == 0
    assert hooks.turn_end_count == 0
    # Post-clear, the registry still triggers cleanly (no AttributeError
    # from internal-state replacement) and the callback does NOT fire.
    await hooks.trigger_turn_start(_evt(stage="x"))
    await hooks.trigger_turn_end(_evt(stage="y", phase="end"))
    assert fired == []

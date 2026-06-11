"""Tests for the wizard turn-lifecycle hooks ``on_turn_start`` /
``on_turn_end``.

These hooks are the consumer extension surface for pre-turn /
post-turn logic. The manager-metadata inbox bridge is the
dataknobs-shipped reference adopter; this file exercises the hook
surface itself (so consumers can rely on the contract).
"""
from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_hooks import WizardHooks
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm import EchoProvider


def _minimal_wizard_dict() -> dict[str, Any]:
    return {
        "name": "hook-test",
        "version": "1.0",
        "stages": [
            {
                "name": "only",
                "is_start": True,
                "is_end": True,
                "prompt": "noop",
            },
        ],
    }


def _build_wizard(
    *,
    hooks: WizardHooks | None = None,
) -> WizardReasoning:
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(_minimal_wizard_dict())
    return WizardReasoning(
        wizard_fsm=fsm,
        hooks=hooks,
        strict_validation=False,
    )


class _FakeManager:
    """Minimal manager: ``metadata`` dict + ``get_messages()`` empty list."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self._messages: list[dict[str, Any]] = []

    def get_messages(self) -> list[dict[str, Any]]:
        return self._messages


def _dummy_llm() -> Any:
    return EchoProvider({"provider": "echo", "model": "test"})


# ---------------------------------------------------------------------------
# Pin: on_turn_start registration → trigger from begin_turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_start_hook_fires_with_manager_and_state() -> None:
    """Pins the basic registration → trigger contract."""
    fired_with: list[tuple[Any, Any, str]] = []

    async def my_hook(
        manager: Any, wizard_state: Any, stage_name: str,
    ) -> None:
        fired_with.append((manager, wizard_state, stage_name))

    hooks = WizardHooks()
    hooks.on_turn_start(my_hook)
    wizard = _build_wizard(hooks=hooks)
    manager = _FakeManager()

    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    assert len(fired_with) == 1
    fired_manager, fired_state, fired_stage = fired_with[0]
    assert fired_manager is manager
    assert fired_state is handle.wizard_state
    assert fired_stage == "only"


# ---------------------------------------------------------------------------
# Pin: greet inherits the hook surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_start_hook_fires_from_greet_too() -> None:
    """Bot-initiated greet inherits the hook surface symmetrically.

    The hook fires before ``generate_stage_response`` runs, so the
    minimal ``_FakeManager`` (which lacks the full ``system_prompt`` /
    branching surface used by stage rendering) raises further along.
    The hook-fire pin lives upstream of that raise — we catch the
    downstream error and assert on the hook side effect.
    """
    fired = 0

    async def my_hook(
        manager: Any, wizard_state: Any, stage_name: str,
    ) -> None:
        nonlocal fired
        fired += 1

    hooks = WizardHooks()
    hooks.on_turn_start(my_hook)
    wizard = _build_wizard(hooks=hooks)
    manager = _FakeManager()

    try:
        await wizard.greet(manager, llm=_dummy_llm())
    except AttributeError:
        # Stub manager doesn't carry the stage-rendering surface;
        # the hook already fired upstream of the crash.
        pass

    assert fired == 1


# ---------------------------------------------------------------------------
# Pin: ordering — runs AFTER the per-turn ephemeral key clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_start_hook_runs_after_per_turn_clear() -> None:
    """A hook can re-populate ephemeral keys that the per-turn clear
    just wiped — proving the trigger fires AFTER the clear, not before.

    Set a stage-level ``_per_turn_keys`` so the wizard clears keys in
    that set at the start of every turn. Hook re-populates the same key;
    if the trigger ran BEFORE the clear, this assertion would fail.
    """
    wizard_dict = _minimal_wizard_dict()
    wizard_dict["settings"] = {"per_turn_keys": ["_intent"]}

    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(wizard_dict)

    async def my_hook(
        manager: Any, wizard_state: Any, stage_name: str,
    ) -> None:
        wizard_state.data["_intent"] = "hook_set"

    hooks = WizardHooks()
    hooks.on_turn_start(my_hook)
    wizard = WizardReasoning(
        wizard_fsm=fsm, hooks=hooks, strict_validation=False,
    )
    manager = _FakeManager()
    manager.metadata["wizard"] = {
        "fsm_state": {
            "current_stage": "only",
            "data": {"_intent": "stale_prior_turn"},
            "history": [],
            "completed": False,
            "clarification_attempts": 0,
            "transitions": [],
            "stage_entry_time": 0,
            "tasks": {},
            "subflow_stack": [],
        },
    }

    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    # Stale value cleared, then hook re-populated → "hook_set" survives.
    assert handle.wizard_state.data["_intent"] == "hook_set"


# ---------------------------------------------------------------------------
# Pin: stage-scoped hook only fires for its stage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_start_stage_specific_scoping() -> None:
    fired_stages: list[str] = []

    async def my_hook(
        manager: Any, wizard_state: Any, stage_name: str,
    ) -> None:
        fired_stages.append(stage_name)

    hooks = WizardHooks()
    hooks.on_turn_start(my_hook, stage="only")
    wizard = _build_wizard(hooks=hooks)
    manager = _FakeManager()

    await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)
    assert fired_stages == ["only"]


@pytest.mark.asyncio
async def test_on_turn_start_stage_scoped_hook_skips_other_stages() -> None:
    fired_stages: list[str] = []

    async def my_hook(
        manager: Any, wizard_state: Any, stage_name: str,
    ) -> None:
        fired_stages.append(stage_name)

    hooks = WizardHooks()
    hooks.on_turn_start(my_hook, stage="other_stage_not_in_wizard")
    wizard = _build_wizard(hooks=hooks)
    manager = _FakeManager()

    await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)
    assert fired_stages == []


# ---------------------------------------------------------------------------
# Pin: on_turn_end fires from the canonical finalize_turn exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_end_hook_fires_after_finalize() -> None:
    """``on_turn_end`` pins the symmetric post-turn observation point.

    Uses :class:`BotTestHarness` because driving a full ``generate``
    requires real stage-rendering infrastructure (system_prompt,
    branching, conversation tree) — beyond what a minimal manager
    stub provides. ``on_turn_end`` fires AFTER ``_save_wizard_state``
    in finalize_turn, so any failure upstream in stage rendering would
    short-circuit the trigger.
    """
    from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

    config = (
        WizardConfigBuilder("on-end-test")
        .stage("only", is_start=True, is_end=True, prompt="Hi.")
        .build()
    )

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Hello!"],
    ) as harness:
        fired = 0

        async def my_hook(
            manager: Any, wizard_state: Any, stage_name: str,
        ) -> None:
            nonlocal fired
            fired += 1

        # The wizard strategy has its own WizardHooks (None when no
        # hooks config is supplied) — attach one if needed and register.
        strategy = harness.bot.reasoning_strategy
        if strategy._hooks is None:
            strategy._hooks = WizardHooks()
        strategy._hooks.on_turn_end(my_hook)

        await harness.chat("hello")

        assert fired == 1


# ---------------------------------------------------------------------------
# Pin: WizardHooks.from_config loads turn-lifecycle callbacks
# ---------------------------------------------------------------------------


async def _track_calls(
    manager: Any, wizard_state: Any, stage_name: str,
) -> None:
    """Module-level callable used by the from_config test."""


def test_wizard_hooks_from_config_loads_turn_start_callbacks() -> None:
    """Config-driven hook registration follows existing dotted-path shape."""
    config = {
        "on_turn_start": [
            {"function": f"{__name__}:_track_calls"},
        ],
        "on_turn_end": [
            f"{__name__}:_track_calls",
        ],
    }
    hooks = WizardHooks.from_config(config)
    # WizardHooks composes LifecycleHooks; turn-lifecycle storage lives
    # on the embedded instance.
    assert len(hooks._lifecycle._turn_start_hooks) == 1
    assert len(hooks._lifecycle._turn_end_hooks) == 1


# ---------------------------------------------------------------------------
# Back-compat: zero turn-hooks → no behaviour change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wizard_without_turn_hooks_runs_unchanged() -> None:
    """A wizard with WizardHooks but no on_turn_start/end works fine.

    Pre-fix, this asserts only that begin_turn still produces a handle
    (no AttributeError from missing lifecycle composition).
    """
    hooks = WizardHooks()
    hooks.on_enter(lambda stage, data: None)  # existing surface only
    wizard = _build_wizard(hooks=hooks)
    manager = _FakeManager()

    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)
    assert handle.wizard_state is not None

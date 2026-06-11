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

        # Public runtime-attach surface — lazy-creates WizardHooks
        # internally when no hooks config was supplied. Pairs with the
        # canonical `_fire_turn_end_hook` trigger that runs from every
        # finalize_turn exit (normal + subflow-push, sync + streaming).
        strategy = harness.bot.reasoning_strategy
        strategy.add_turn_end_hook(my_hook)

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
    # Public introspection surface — `hook_count` includes the composed
    # turn-lifecycle keys alongside the legacy wizard hooks.
    counts = hooks.hook_count
    assert counts["turn_start"] == 1
    assert counts["turn_end"] == 1


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


# ---------------------------------------------------------------------------
# WizardHooks.clear() drops EVERY hook type, including the composed turn surface
# ---------------------------------------------------------------------------


def test_wizard_hooks_clear_drops_turn_hooks_too() -> None:
    """``clear()`` must reset the embedded LifecycleHooks alongside the
    legacy lists. Otherwise the auto-registered inbox bridge (or any
    runtime-registered turn hook) survives a clear silently.
    """
    hooks = WizardHooks()
    hooks.on_enter(lambda stage, data: None)
    hooks.on_turn_start(lambda manager, state, stage_name: None)
    hooks.on_turn_end(lambda manager, state, stage_name: None)

    counts = hooks.hook_count
    assert counts["enter"] == 1
    assert counts["turn_start"] == 1
    assert counts["turn_end"] == 1

    hooks.clear()

    counts = hooks.hook_count
    assert counts["enter"] == 0
    assert counts["turn_start"] == 0
    assert counts["turn_end"] == 0


def test_wizard_hooks_clear_preserves_lifecycle_identity() -> None:
    """An external reference to ``hooks.lifecycle`` survives a clear.

    The clear drains the lifecycle in place rather than rebinding it,
    matching the legacy list-clear semantics for the other hook types
    so consumers can safely cache the lifecycle reference.
    """
    hooks = WizardHooks()
    cached_lifecycle = hooks.lifecycle
    hooks.on_turn_start(lambda manager, state, stage_name: None)

    hooks.clear()

    assert hooks.lifecycle is cached_lifecycle
    assert cached_lifecycle.turn_start_count == 0


def test_wizard_hooks_hook_count_keys() -> None:
    """``hook_count`` reports every hook type a consumer can register
    through this class — including the composed turn-lifecycle surface.
    Diagnostic consumers (e.g., debug logging, admin endpoints) rely
    on this to surface a complete picture.
    """
    hooks = WizardHooks()
    counts = hooks.hook_count

    expected_keys = {
        "enter",
        "exit",
        "complete",
        "restart",
        "error",
        "turn_start",
        "turn_end",
    }
    assert set(counts.keys()) == expected_keys
    assert all(v == 0 for v in counts.values())


# ---------------------------------------------------------------------------
# Public runtime-attach surface on WizardReasoning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_turn_start_hook_lazy_creates_hooks() -> None:
    """``add_turn_start_hook`` works even when no hooks were supplied
    at construction — lazy-creates a :class:`WizardHooks` internally.
    """
    fired_stages: list[str] = []

    async def my_hook(manager: Any, wizard_state: Any, stage_name: str) -> None:
        fired_stages.append(stage_name)

    wizard = _build_wizard(hooks=None)  # NO hooks at construction
    wizard.add_turn_start_hook(my_hook)

    await wizard.begin_turn(_FakeManager(), llm=_dummy_llm(), tools=None)
    assert fired_stages == ["only"]


@pytest.mark.asyncio
async def test_add_turn_end_hook_lazy_creates_hooks() -> None:
    """Symmetric to ``add_turn_start_hook`` — the runtime-attach flow
    must lazy-create the :class:`WizardHooks` AND deliver the callback
    to the canonical fire-point.

    Behavioural — drives a full turn through :class:`BotTestHarness`
    and asserts ``my_hook`` fires from ``finalize_turn``'s
    ``_fire_turn_end_hook`` exit. ``finalize_turn`` requires real
    stage-rendering infrastructure, so the harness path mirrors how
    consumers attach hooks at runtime (no construction-time hooks
    block in the bot config).
    """
    from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

    config = (
        WizardConfigBuilder("lazy-end-test")
        .stage("only", is_start=True, is_end=True, prompt="Hi.")
        .build()
    )

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Reply"],
    ) as harness:
        fired = 0

        async def my_hook(*_: Any) -> None:
            nonlocal fired
            fired += 1

        # No hooks block in the bot config above → wizard's _hooks is None.
        # The public surface must lazy-create + delegate so this works.
        harness.bot.reasoning_strategy.add_turn_end_hook(my_hook)

        await harness.chat("hello")

        assert fired == 1


def test_add_turn_start_hook_appends_to_existing_hooks() -> None:
    """When a :class:`WizardHooks` was supplied at construction,
    ``add_turn_start_hook`` extends it rather than replacing.
    """
    hooks = WizardHooks()
    hooks.on_enter(lambda stage, data: None)  # pre-existing legacy hook
    wizard = _build_wizard(hooks=hooks)

    async def my_hook(*_: Any) -> None:
        pass

    wizard.add_turn_start_hook(my_hook)

    counts = hooks.hook_count
    assert counts["enter"] == 1  # preserved
    assert counts["turn_start"] == 1  # added


# ---------------------------------------------------------------------------
# Pin Tier A: stream_finalize_turn normal path fires on_turn_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_end_fires_from_stream_finalize_turn() -> None:
    """``stream_finalize_turn`` fires ``on_turn_end`` after the
    fully-consumed save — symmetric with ``finalize_turn``.

    Subflow-push exits (sync + streaming) share the same
    ``_fire_turn_end_hook`` helper, so they're transitively covered.
    Paths still skipped (tracked as 164-FU1): early-returns from
    begin_turn / process_input, stream abandonment via ``aclose()``,
    and the non-conversational ``advance`` API.
    """
    from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

    config = (
        WizardConfigBuilder("stream-end-test")
        .stage("only", is_start=True, is_end=True, prompt="Hi.")
        .build()
    )

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Streamed reply"],
    ) as harness:
        fired = 0

        async def my_hook(*_: Any) -> None:
            nonlocal fired
            fired += 1

        harness.bot.reasoning_strategy.add_turn_end_hook(my_hook)

        # Fully consume the stream — the canonical exit is reached only
        # when the async iterator completes.
        async for _chunk in harness.bot.stream_chat("hello", harness.context):
            pass

        assert fired == 1, (
            "on_turn_end must fire after stream_finalize_turn's "
            "save_wizard_state — closes 164-FU1 Tier A for streaming."
        )


# ---------------------------------------------------------------------------
# Pin Tier A: subflow-push exit fires on_turn_end (sync path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_end_fires_from_finalize_turn_subflow_push() -> None:
    """The subflow-push branch in ``finalize_turn`` MUST fire
    ``on_turn_end`` after its ``_save_wizard_state``.

    The sync and streaming subflow-push branches both call the shared
    ``_fire_turn_end_hook`` helper, so this single sync integration
    test transitively pins the streaming variant's save→fire ordering
    too. Without this test, the subflow-push branch could be modified
    to call fire-before-save (breaking writer hooks that publish to
    ``manager.metadata`` expecting persisted state) and no test would
    catch it.

    Configuration: the start stage transitions unconditionally into
    a subflow on first turn, so a single ``chat()`` drives the wizard
    through the subflow-push exit.
    """
    from dataknobs_bots.testing import BotTestHarness

    config = {
        "name": "subflow-end-test",
        "version": "1.0",
        "stages": [
            {
                "name": "parent",
                "is_start": True,
                "prompt": "Parent stage",
                "transitions": [
                    {
                        "target": "_subflow",
                        "subflow": {
                            "network": "sub",
                            "return_stage": "parent_end",
                        },
                    },
                ],
            },
            {
                "name": "parent_end",
                "is_end": True,
                "prompt": "Done!",
            },
        ],
        "subflows": {
            "sub": {
                "stages": [
                    {
                        "name": "sub_only",
                        "is_start": True,
                        "is_end": True,
                        "prompt": "In subflow",
                    },
                ],
            },
        },
    }

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Parent reply", "Subflow reply"],
    ) as harness:
        fire_log: list[str] = []

        async def my_hook(
            manager: Any, wizard_state: Any, stage_name: str,
        ) -> None:
            # Capture the stage name to confirm the active subflow FSM is
            # reported (not the parent's stage) — a save→fire reorder
            # would also drop this signal.
            fire_log.append(stage_name)

        harness.bot.reasoning_strategy.add_turn_end_hook(my_hook)

        await harness.chat("hello")

        assert fire_log, (
            "on_turn_end did not fire from the subflow-push exit — "
            "the shared `_fire_turn_end_hook` helper is not wired into "
            "the subflow branch."
        )
        # Active FSM after the push is the subflow's; current_stage is
        # the subflow's start. The helper resolves via
        # `SubflowManager.get_active_fsm()` and must report the subflow
        # stage, not the parent's.
        assert fire_log[-1] == "sub_only", (
            f"Expected subflow start stage 'sub_only'; got {fire_log!r}. "
            "If this reports the parent stage, the helper's active-FSM "
            "lookup is wrong on the subflow-push exit."
        )

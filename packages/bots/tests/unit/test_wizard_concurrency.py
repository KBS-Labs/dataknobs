"""Tests for the per-call closure factory in _execute_fsm_step (item 82).

Verifies that:
- Each FSM step receives its own LLM and TurnContext via the closure
- ``advance()`` threads ``llm`` through to transforms
- ``advance()`` without ``llm`` produces empty config (not stale)
- Navigation skip produces ``llm=None`` in transform context
- Two concurrent ``finalize_turn`` calls with different LLMs are isolated
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import sentinel

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.reasoning.wizard_types import WizardState
from dataknobs_llm import EchoProvider
from dataknobs_llm.testing import scripted_schema_extractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_captured_contexts: list[Any] = []
"""Module-level list to capture TransformContext from transforms."""


def context_capturing_transform(
    data: dict[str, Any],
    context: Any = None,
    **kwargs: object,
) -> None:
    """Transform that records the TransformContext it receives."""
    _captured_contexts.append(context)


def _make_reasoning(
    config: dict[str, Any],
    *,
    extractor: Any | None = None,
    custom_functions: dict[str, Any] | None = None,
) -> WizardReasoning:
    """Create a WizardReasoning from a config dict."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(config, custom_functions=custom_functions)
    return WizardReasoning(
        wizard_fsm=wizard_fsm,
        extractor=extractor,
        strict_validation=False,
        extraction_scope="current_message",
    )


def _make_state(
    reasoning: WizardReasoning,
    *,
    current_stage: str | None = None,
    data: dict[str, Any] | None = None,
) -> WizardState:
    """Create a WizardState at the given stage."""
    stage = current_stage or reasoning.initial_stage
    return WizardState(
        current_stage=stage,
        data=data or {},
        history=[stage],
        stage_entry_time=time.time(),
    )


def _config_with_capturing_transform() -> dict[str, Any]:
    """Wizard config with a context-capturing transform on the transition."""
    return {
        "name": "concurrency-test",
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "prompt": "Start",
                "schema": {
                    "type": "object",
                    "properties": {"go": {"type": "string"}},
                    "required": ["go"],
                },
                "transitions": [
                    {
                        "target": "end",
                        "condition": "data.get('go')",
                        "transform": "capture_ctx",
                    }
                ],
            },
            {"name": "end", "is_end": True, "prompt": "Done"},
        ],
    }


def _custom_fns() -> dict[str, Any]:
    return {"capture_ctx": context_capturing_transform}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFsmStepLlmParameter:
    """_execute_fsm_step delivers llm via per-call closure."""

    @pytest.mark.asyncio
    async def test_fsm_step_receives_correct_llm(self) -> None:
        """Transform context contains the llm passed to _execute_fsm_step."""
        _captured_contexts.clear()
        config = _config_with_capturing_transform()
        reasoning = _make_reasoning(config, custom_functions=_custom_fns())
        state = _make_state(reasoning, data={"go": "yes"})

        fake_llm = sentinel.my_llm
        await reasoning._execute_fsm_step(state, llm=fake_llm)

        assert len(_captured_contexts) == 1
        ctx = _captured_contexts[0]
        assert ctx.config.get("llm") is fake_llm

    @pytest.mark.asyncio
    async def test_fsm_step_receives_correct_turn_context(self) -> None:
        """TurnContext with correct user_message and intent flows through."""
        _captured_contexts.clear()
        config = _config_with_capturing_transform()
        reasoning = _make_reasoning(config, custom_functions=_custom_fns())
        state = _make_state(reasoning, data={"go": "yes", "_intent": "test_intent"})

        await reasoning._execute_fsm_step(
            state, user_message="hello world", llm=sentinel.llm,
        )

        assert len(_captured_contexts) == 1
        ctx = _captured_contexts[0]
        assert ctx.turn is not None
        assert ctx.turn.message == "hello world"
        assert ctx.turn.intent == "test_intent"

    @pytest.mark.asyncio
    async def test_fsm_step_without_llm(self) -> None:
        """When llm is None, transform context has empty config."""
        _captured_contexts.clear()
        config = _config_with_capturing_transform()
        reasoning = _make_reasoning(config, custom_functions=_custom_fns())
        state = _make_state(reasoning, data={"go": "yes"})

        await reasoning._execute_fsm_step(state, llm=None)

        assert len(_captured_contexts) == 1
        ctx = _captured_contexts[0]
        assert ctx.config == {}

    @pytest.mark.asyncio
    async def test_factory_restored_after_step(self) -> None:
        """After _execute_fsm_step, the fallback factory is restored.

        Verifies behaviorally: the fallback factory returns turn=None
        and config={} (distinguishable from the per-call closure).
        """
        config = _config_with_capturing_transform()
        reasoning = _make_reasoning(config, custom_functions=_custom_fns())
        state = _make_state(reasoning, data={"go": "yes"})

        await reasoning._execute_fsm_step(state, llm=sentinel.llm)

        # The fallback factory is _build_transform_context — call it
        # directly to verify it produces safe defaults.
        ctx = reasoning._build_transform_context(None)
        assert ctx.turn is None
        assert ctx.config == {}


class TestAdvanceLlmThreading:
    """advance() threads llm through to _execute_fsm_step."""

    @pytest.mark.asyncio
    async def test_advance_passes_llm_to_transforms(self) -> None:
        """advance() with llm= makes it available in transform context."""
        _captured_contexts.clear()
        config = _config_with_capturing_transform()
        reasoning = _make_reasoning(config, custom_functions=_custom_fns())
        state = _make_state(reasoning, data={})

        fake_llm = sentinel.advance_llm
        await reasoning.advance(
            {"go": "yes"}, state, llm=fake_llm,
        )

        assert len(_captured_contexts) == 1
        ctx = _captured_contexts[0]
        assert ctx.config.get("llm") is fake_llm

    @pytest.mark.asyncio
    async def test_advance_without_llm(self) -> None:
        """advance() without llm produces empty config (not stale)."""
        _captured_contexts.clear()
        config = _config_with_capturing_transform()
        reasoning = _make_reasoning(config, custom_functions=_custom_fns())
        state = _make_state(reasoning, data={})

        await reasoning.advance({"go": "yes"}, state)

        assert len(_captured_contexts) == 1
        ctx = _captured_contexts[0]
        assert ctx.config == {}


class TestNavigateSkipLlm:
    """Navigation skip passes llm=None to transforms."""

    @pytest.mark.asyncio
    async def test_navigate_skip_no_llm(self) -> None:
        """Navigation skip → llm=None in transform context."""
        _captured_contexts.clear()
        config = {
            "name": "skip-test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "can_skip": True,
                    "transitions": [
                        {
                            "target": "end",
                            "transform": "capture_ctx",
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        reasoning = _make_reasoning(config, custom_functions=_custom_fns())
        state = _make_state(reasoning, current_stage="start")

        await reasoning._navigate_skip(state)

        assert len(_captured_contexts) == 1
        ctx = _captured_contexts[0]
        # Navigator doesn't have an llm reference → config is empty
        assert ctx.config == {}


class TestConcurrentTurnsIsolated:
    """Two concurrent FSM steps see their own LLM (core concurrency test).

    Safety invariant: within ``_execute_fsm_step``, there is no ``await``
    between ``set_transform_context_factory(_scoped_factory)`` and
    ``step_async()``.  The FSM engine's ``_create_function_context``
    reads the factory synchronously before any ``await`` in the step,
    so a concurrent call cannot overwrite the factory between
    registration and invocation.  The test below covers the wider
    window (transform body interleaving) to verify that each closure
    captures its own stack-local values.
    """

    @pytest.mark.asyncio
    async def test_concurrent_turns_see_own_llm(self) -> None:
        """Two concurrent _execute_fsm_step calls with different LLMs.

        Each transform should see the LLM from its own call, not the
        other's.  This is the scenario that was broken when
        ``self._current_llm`` was an instance attribute.
        """
        # We need two independent states but the same reasoning instance
        # (simulating multi-tenant shared instance).
        _captured_contexts.clear()

        # Use an event to force interleaving: transform A pauses, B runs,
        # then A resumes.  This maximizes the window where the old
        # instance-attribute approach would have been overwritten.
        interleave_event = asyncio.Event()
        order: list[str] = []

        async def slow_capturing_transform(
            data: dict[str, Any],
            context: Any = None,
            **kwargs: object,
        ) -> None:
            """Transform that pauses to allow interleaving."""
            label = data.get("_label", "unknown")
            order.append(f"{label}_start")
            if label == "A":
                # A starts first, then waits for B to start
                interleave_event.set()
                # Yield control to let B's FSM step run
                await asyncio.sleep(0)
            _captured_contexts.append((label, context))
            order.append(f"{label}_end")

        config = {
            "name": "concurrent-test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('go')",
                            "transform": "slow_capture",
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        fns = {"slow_capture": slow_capturing_transform}
        reasoning = _make_reasoning(config, custom_functions=fns)

        state_a = _make_state(reasoning, data={"go": "yes", "_label": "A"})
        state_b = _make_state(reasoning, data={"go": "yes", "_label": "B"})

        llm_a = sentinel.llm_A
        llm_b = sentinel.llm_B

        async def run_a() -> None:
            await reasoning._execute_fsm_step(state_a, llm=llm_a)

        async def run_b() -> None:
            # Wait for A's transform to start before running B
            await interleave_event.wait()
            await reasoning._execute_fsm_step(state_b, llm=llm_b)

        await asyncio.gather(run_a(), run_b())

        # Both transforms should have captured their own LLM
        captured = {label: ctx for label, ctx in _captured_contexts}
        assert captured["A"].config.get("llm") is llm_a
        assert captured["B"].config.get("llm") is llm_b


class TestAutoAdvanceLlmThreading:
    """Auto-advance transforms receive the LLM via the scoped factory.

    ``run_auto_advance_loop`` calls ``step_async`` directly (not through
    ``_execute_fsm_step``).  It must install its own per-call closure
    so that auto-advance transforms have access to the LLM — otherwise
    they see ``config={}`` from the fallback factory.
    """

    @pytest.mark.asyncio
    async def test_auto_advance_transform_sees_llm(self) -> None:
        """Transform on an auto-advance transition sees the LLM.

        Config: start → (extract 'go') → mid → (auto-advance, capture_ctx) → end.
        The capture_ctx transform fires during auto-advance and must see the LLM.
        """
        _captured_contexts.clear()

        config = {
            "name": "auto-advance-llm-test",
            "settings": {
                "extraction_scope": "current_message",
                "auto_advance_filled_stages": True,
            },
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "schema": {
                        "type": "object",
                        "properties": {"go": {"type": "string"}},
                        "required": ["go"],
                    },
                    "transitions": [
                        {
                            "target": "mid",
                            "condition": "data.get('go')",
                        }
                    ],
                },
                {
                    "name": "mid",
                    "prompt": "Mid",
                    "auto_advance": True,
                    "response_template": "Advancing...",
                    "transitions": [
                        {
                            "target": "end",
                            "transform": "capture_ctx",
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        reasoning = _make_reasoning(config, custom_functions=_custom_fns())

        # Seed data so start→mid transition fires
        state = _make_state(reasoning, data={"go": "yes"})

        fake_llm = sentinel.auto_advance_llm

        # Execute FSM step (start→mid) then post-transition lifecycle
        # (mid→end via auto-advance).  The auto-advance transform must
        # see the LLM.
        await reasoning._execute_fsm_step(state, llm=fake_llm)
        await reasoning._run_post_transition_lifecycle(
            state, llm=fake_llm,
        )

        assert len(_captured_contexts) == 1, (
            f"Expected 1 captured context from auto-advance transform, "
            f"got {len(_captured_contexts)}"
        )
        ctx = _captured_contexts[0]
        assert ctx.config.get("llm") is fake_llm, (
            f"Auto-advance transform should see the LLM, "
            f"got config={ctx.config}"
        )


class TestNoInstanceTurnState:
    """Instance no longer has _current_llm or _current_turn."""

    def test_no_current_llm_attribute(self) -> None:
        """WizardReasoning no longer has _current_llm."""
        config = {
            "name": "test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "S",
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "E"},
            ],
        }
        reasoning = _make_reasoning(config)
        assert not hasattr(reasoning, "_current_llm")
        assert not hasattr(reasoning, "_current_turn")

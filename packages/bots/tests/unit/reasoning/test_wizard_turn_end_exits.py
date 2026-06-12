"""Tests for ``on_turn_end`` firing on every wizard exit path.

Closes the observability gap where ``on_turn_end`` only fired on
the canonical ``finalize_turn`` / ``stream_finalize_turn`` exits.
Every early-return, the stream-abandonment path, and the
non-conversational ``advance()`` API now fire ``on_turn_end`` with
a per-site ``reason`` discriminator on the opaque event payload.

These tests pin each fire-point through behavioural setup so a
regression that drops one of the new awaits is caught immediately.
"""
from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_hooks import WizardHooks
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


# ---------------------------------------------------------------------------
# Tier B: begin_turn early-returns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_end_fires_on_navigation_early_return() -> None:
    """The navigation early-return in ``begin_turn`` fires ``on_turn_end``
    with ``reason="navigation"``.

    Drives a two-turn wizard: the first turn advances past the start
    stage; the second turn sends ``"back"`` to trigger the navigation
    handler's early-return. The fired event must carry the navigation
    discriminator and the manager / state references.
    """
    config = (
        WizardConfigBuilder("nav-test")
        .stage("first", is_start=True, prompt="Tell me a name.")
            .field("name", field_type="string", required=True)
            .transition("second", "data.get('name')")
        .stage("second", is_end=True, prompt="Second stage.")
        .build()
    )

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["First reply", "Second reply", "Back reply"],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        fired: list[dict[str, Any]] = []

        async def my_hook(event: dict[str, Any]) -> None:
            fired.append(event)

        harness.bot.reasoning_strategy.add_turn_end_hook(my_hook)

        # Advance into the second stage.
        await harness.chat("My name is Alice")
        canonical_count = len(fired)

        # Now send "back" — the navigator's keyword match triggers the
        # navigation early-return in begin_turn.
        await harness.chat("back")

        # Exactly one additional fire — from the navigation early-return,
        # NOT from finalize_turn.
        assert len(fired) == canonical_count + 1
        nav_event = fired[-1]
        assert nav_event["reason"] == "navigation"
        assert nav_event["phase"] == "end"
        assert nav_event["state"] is not None


# ---------------------------------------------------------------------------
# Tier B: process_input early-returns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_end_fires_on_clarification_early_return() -> None:
    """Low-confidence extraction routes through the clarification
    early-return, which fires ``on_turn_end`` with ``reason="clarification"``.
    """
    config = (
        WizardConfigBuilder("clarification-test")
        .stage("ask", is_start=True, prompt="Tell me a name and age.")
            .field("name", field_type="string", required=True)
            .field("age", field_type="string", required=True)
            .transition("done", "data.get('name') and data.get('age')")
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )

    # Extraction returns an empty dict at low confidence → confidence
    # gate fires.
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Could you clarify?"],
        extraction_results=[[{}]],
    ) as harness:
        # Override the auto-wired extractor with a low-confidence one so
        # is_confident=False.
        from dataknobs_llm.testing import (
            ConfigurableExtractor,
            SimpleExtractionResult,
        )
        low_conf = ConfigurableExtractor(
            results=[SimpleExtractionResult(data={}, confidence=0.0)],
        )
        from dataknobs_bots.testing import inject_providers
        inject_providers(harness.bot, extractor=low_conf)

        fired: list[dict[str, Any]] = []

        async def my_hook(event: dict[str, Any]) -> None:
            fired.append(event)

        harness.bot.reasoning_strategy.add_turn_end_hook(my_hook)

        await harness.chat("...")

        assert len(fired) == 1
        assert fired[0]["reason"] == "clarification"
        assert fired[0]["phase"] == "end"


@pytest.mark.asyncio
async def test_on_turn_end_fires_on_collection_help_early_return() -> None:
    """A help-intent message in a collection-mode stage routes through
    the collection_help early-return, which fires ``on_turn_end`` with
    ``reason="collection_help"``.
    """
    config = {
        "name": "collection-help-test",
        "version": "1.0",
        "stages": [
            {
                "name": "collect",
                "is_start": True,
                "is_end": True,
                "prompt": "Collect items.",
                "collection_mode": "collection",
                "collection_config": {
                    "bank": "items",
                    "done_keywords": ["done"],
                },
                "schema": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                    },
                    "required": ["item"],
                },
            },
        ],
    }

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Here is some help."],
    ) as harness:
        fired: list[dict[str, Any]] = []

        async def my_hook(event: dict[str, Any]) -> None:
            fired.append(event)

        harness.bot.reasoning_strategy.add_turn_end_hook(my_hook)

        # "help" matches the built-in help heuristic
        # (classify_collection_intent).
        await harness.chat("help")

        assert any(e["reason"] == "collection_help" for e in fired), (
            f"Expected reason='collection_help'; got {[e['reason'] for e in fired]}"
        )


# ---------------------------------------------------------------------------
# Tier C1: stream abandonment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_end_fires_on_stream_abandonment() -> None:
    """When the stream consumer abandons the iterator via ``aclose()``,
    the GeneratorExit branch in ``stream_finalize_turn`` fires
    ``on_turn_end`` with ``reason="abandoned"`` and
    ``state_saved=False``.

    Drives ``stream_finalize_turn`` directly (not through
    ``DynaBot.stream_chat``) so the ``aclose()`` deterministically
    propagates ``GeneratorExit`` into the wizard's generator — going
    through the bot's outer generator would close the nested wizard
    generator only at garbage-collection time, which races with the
    assertion.
    """
    config = (
        WizardConfigBuilder("stream-abandon-test")
        .stage("only", is_start=True, is_end=True, prompt="Hi.")
        .build()
    )

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Streamed reply"],
    ) as harness:
        fired: list[dict[str, Any]] = []

        async def my_hook(event: dict[str, Any]) -> None:
            fired.append(event)

        strategy = harness.bot.reasoning_strategy
        strategy.add_turn_end_hook(my_hook)

        # Build a real ConversationManager via the bot's internal
        # helper, then drive begin_turn → process_input →
        # stream_finalize_turn manually. Reaching for the helper
        # rather than the public chat()/stream_chat() surfaces is
        # deliberate: aclose() on DynaBot.stream_chat propagates
        # GeneratorExit only into the outer generator, leaving the
        # nested wizard generator to close at GC time, which races
        # with the assertion. Driving stream_finalize_turn directly
        # exercises the same code under test (the try/except
        # branch we added) deterministically.
        manager = await harness.bot._get_or_create_conversation(
            harness.context,
        )
        await manager.add_message(role="user", content="hello")

        handle = await strategy.begin_turn(
            manager, harness.bot.llm, tools=None,
        )
        result = await strategy.process_input(handle)
        assert result.early_response is None, (
            "Test setup expected the wizard to reach finalize_turn; "
            "got an early_response from process_input."
        )

        stream = strategy.stream_finalize_turn(handle)
        # Drive one chunk so the generator is suspended at a yield
        # point — aclose() then injects GeneratorExit there, hitting
        # the wizard's try/except branch.
        await stream.__anext__()
        await stream.aclose()

        assert len(fired) == 1, (
            f"Expected exactly one on_turn_end fire (abandonment); "
            f"got {len(fired)} with reasons "
            f"{[e['reason'] for e in fired]}"
        )
        evt = fired[0]
        assert evt["reason"] == "abandoned"
        assert evt["state_saved"] is False
        assert evt["phase"] == "end"


# ---------------------------------------------------------------------------
# Tier C2: advance() non-conversational API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_end_fires_on_advance() -> None:
    """``advance()`` fires ``on_turn_end`` with ``reason="advance"`` and
    ``manager=None`` — the non-conversational API has no manager.

    Consumers depending on the manager filter on ``reason`` or check
    for the key's absence.
    """
    config = {
        "name": "advance-test",
        "version": "1.0",
        "stages": [
            {
                "name": "first",
                "is_start": True,
                "prompt": "Provide name.",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "transitions": [
                    {
                        "target": "done",
                        "condition": "data.get('name')",
                    },
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "All set.",
            },
        ],
    }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)

    hooks = WizardHooks()
    fired: list[dict[str, Any]] = []

    async def my_hook(event: dict[str, Any]) -> None:
        fired.append(event)

    hooks.on_turn_end(my_hook)

    wizard = WizardReasoning(
        wizard_fsm=fsm, hooks=hooks, strict_validation=False,
    )
    state = WizardState(current_stage="first")
    state.history = ["first"]

    # Dict-mode advance — merges directly into state.data without
    # touching the extractor.
    result = await wizard.advance(
        user_input={"name": "Alice"}, state=state,
    )

    assert result.transitioned is True
    assert len(fired) == 1
    evt = fired[0]
    assert evt["reason"] == "advance"
    assert evt["manager"] is None
    assert evt["state"] is state
    assert evt["phase"] == "end"


# ---------------------------------------------------------------------------
# Canonical normal-path still fires with reason="normal"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_turn_end_canonical_path_publishes_reason_normal() -> None:
    """Sanity pin — the canonical ``finalize_turn`` exit publishes
    ``reason="normal"``, distinguishing it from every early-return path.
    """
    config = (
        WizardConfigBuilder("normal-reason-test")
        .stage("only", is_start=True, is_end=True, prompt="Hi.")
        .build()
    )

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Hello there"],
    ) as harness:
        fired: list[dict[str, Any]] = []

        async def my_hook(event: dict[str, Any]) -> None:
            fired.append(event)

        harness.bot.reasoning_strategy.add_turn_end_hook(my_hook)

        await harness.chat("hi")

        assert len(fired) == 1
        assert fired[0]["reason"] == "normal"
        assert fired[0]["phase"] == "end"

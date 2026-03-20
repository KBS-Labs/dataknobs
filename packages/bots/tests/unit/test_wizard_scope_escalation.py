"""Tests for wizard extraction scope escalation.

When extraction_scope is "current_message" and required fields are still
missing after the initial extraction + merge, scope escalation retries
with a broader scope (e.g. "wizard_session") to recover fields from
earlier conversation turns.

Also tests the ``recent_messages`` scope and the ``SCOPE_BREADTH``
constant.

These tests exercise the full DynaBot.from_config() → bot.chat() path
to verify scope escalation through the public API.
"""

from typing import Any

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.reasoning.wizard import SCOPE_BREADTH
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.testing import ConfigurableExtractor, SimpleExtractionResult


# ---------------------------------------------------------------------------
# Shared wizard config: gather stage with 3 required fields
# ---------------------------------------------------------------------------

WIZARD_CONFIG: dict[str, Any] = {
    "name": "escalation-test",
    "version": "1.0",
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me your name, domain_name, and domain_id.",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "domain_name": {"type": "string"},
                    "domain_id": {"type": "string"},
                },
                "required": ["name", "domain_name", "domain_id"],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": (
                        "data.get('name') and data.get('domain_name') "
                        "and data.get('domain_id')"
                    ),
                },
            ],
        },
        {
            "name": "done",
            "is_end": True,
            "prompt": "All done!",
        },
    ],
}


def _bot_config(
    wizard_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a DynaBot config with inline wizard reasoning."""
    wizard_cfg = {**WIZARD_CONFIG}
    if wizard_settings:
        wizard_cfg["settings"] = wizard_settings
    return {
        "llm": {"provider": "echo", "model": "echo-test"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": {
            "strategy": "wizard",
            "wizard_config": wizard_cfg,
            "extraction_config": {
                "provider": "echo",
                "model": "echo-extraction",
            },
        },
    }


def _make_context(conv_id: str = "test-conv") -> BotContext:
    return BotContext(conversation_id=conv_id, client_id="test")


def _get_wizard_state(bot: DynaBot, conv_id: str = "test-conv") -> dict[str, Any]:
    """Read wizard state from the bot's conversation manager metadata.

    Uses the manager metadata path that the wizard writes to via
    ``_save_wizard_state``.
    """
    manager = bot._conversation_managers.get(conv_id)
    if manager is None:
        return {}
    wizard_meta = manager.metadata.get("wizard", {})
    return wizard_meta.get("fsm_state", {})


def _get_wizard_data(bot: DynaBot, conv_id: str = "test-conv") -> dict[str, Any]:
    """Read wizard_state.data from the bot."""
    return _get_wizard_state(bot, conv_id).get("data", {})


def _get_wizard_stage(bot: DynaBot, conv_id: str = "test-conv") -> str:
    """Read current wizard stage from the bot."""
    return _get_wizard_state(bot, conv_id).get("current_stage", "")


def _inject_extractor(bot: DynaBot, extractor: ConfigurableExtractor) -> None:
    """Replace the wizard's schema extractor with a ConfigurableExtractor.

    This is the one internal touch point — equivalent to what
    inject_providers() does for LLM providers, but for the extraction
    subsystem which uses a ConfigurableExtractor (not an AsyncLLMProvider).
    """
    strategy = bot.reasoning_strategy
    if strategy is not None and hasattr(strategy, "_extractor"):
        strategy._extractor = extractor


# ---------------------------------------------------------------------------
# Tests: Scope escalation
# ---------------------------------------------------------------------------


class TestScopeEscalation:
    """Verify scope escalation triggers and behaviour via bot.chat()."""

    @pytest.mark.asyncio
    async def test_escalation_fills_missing_fields(self) -> None:
        """Escalation re-extracts with session scope when fields are missing.

        Turn 1: user provides name.
        Turn 2: user provides domain_id and domain_name.  Extractor
        (current_message) returns only domain_id.  Escalated extraction
        (wizard_session) returns all three fields.
        """
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            # Turn 1: name only
            SimpleExtractionResult(
                data={"name": "Alice"}, confidence=0.9,
            ),
            # Turn 2: current_message — domain_id only
            SimpleExtractionResult(
                data={"domain_id": "chess-champ"}, confidence=0.9,
            ),
            # Turn 2: escalated (session scope) — all fields
            SimpleExtractionResult(
                data={
                    "name": "Alice",
                    "domain_name": "Chess Champ",
                    "domain_id": "chess-champ",
                },
                confidence=0.9,
            ),
        ])

        main_provider.set_responses(["Got it!", "All set!"])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        # Turn 1
        await bot.chat("My name is Alice", ctx)
        assert _get_wizard_data(bot)["name"] == "Alice"

        # Turn 2: current_message misses domain_name, escalation fills it
        await bot.chat("Call it Chess Champ, ID chess-champ", ctx)
        data = _get_wizard_data(bot)
        assert data["name"] == "Alice"
        assert data["domain_name"] == "Chess Champ"
        assert data["domain_id"] == "chess-champ"

        # 3 extract calls: turn 1 + turn 2 initial + turn 2 escalated
        assert len(extractor.extract_calls) == 3

        await bot.close()

    @pytest.mark.asyncio
    async def test_no_escalation_when_fields_satisfied(self) -> None:
        """Fast path: no second extraction when all required fields filled."""
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            SimpleExtractionResult(
                data={
                    "name": "Alice",
                    "domain_name": "Chess Champ",
                    "domain_id": "chess-champ",
                },
                confidence=0.9,
            ),
        ])

        main_provider.set_responses(["All set!"])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        await bot.chat("Alice, Chess Champ, chess-champ", ctx)

        # Only 1 extract call — no escalation needed
        assert len(extractor.extract_calls) == 1

        await bot.close()

    @pytest.mark.asyncio
    async def test_no_escalation_when_already_session_scope(self) -> None:
        """No escalation when extraction_scope is already wizard_session."""
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "wizard_session",
            "scope_escalation": {"enabled": True},
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            SimpleExtractionResult(
                data={"name": "Alice", "domain_id": "chess-champ"},
                confidence=0.5,
            ),
        ])

        main_provider.set_responses(["What's the domain name?"])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        await bot.chat("Alice, chess-champ", ctx)

        # Only 1 call — scope is already broadest
        assert len(extractor.extract_calls) == 1

        await bot.close()

    @pytest.mark.asyncio
    async def test_escalation_disabled_by_default(self) -> None:
        """Escalation is disabled when not configured."""
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            SimpleExtractionResult(
                data={"name": "Alice"}, confidence=0.5,
            ),
        ])

        main_provider.set_responses(["Tell me more."])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        await bot.chat("I'm Alice", ctx)

        # Only 1 extract call — escalation not enabled
        assert len(extractor.extract_calls) == 1

        await bot.close()

    @pytest.mark.asyncio
    async def test_escalation_with_grounding_protects_existing(self) -> None:
        """Grounding filter protects existing data during escalation.

        Escalated extraction may return values for already-filled fields.
        The grounding filter should block ungrounded overwrites.
        """
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
            "extraction_grounding": True,
            "scope_escalation": {"enabled": True},
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            # Turn 1: domain_id only
            SimpleExtractionResult(
                data={"domain_id": "chess-champ"}, confidence=0.9,
            ),
            # Turn 2: current_message — name only
            SimpleExtractionResult(
                data={"name": "Alice"}, confidence=0.9,
            ),
            # Turn 2 escalated: all 3 but domain_id differs (ungrounded)
            SimpleExtractionResult(
                data={
                    "name": "Alice",
                    "domain_name": "Chess Champ",
                    "domain_id": "wrong-id",
                },
                confidence=0.9,
            ),
        ])

        main_provider.set_responses(["Got it!", "All set!"])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        # Turn 1
        await bot.chat("ID is chess-champ", ctx)
        assert _get_wizard_data(bot)["domain_id"] == "chess-champ"

        # Turn 2: escalation fires, but grounding blocks "wrong-id"
        await bot.chat("Alice, Chess Champ", ctx)
        data = _get_wizard_data(bot)
        assert data["domain_name"] == "Chess Champ"
        assert data["domain_id"] == "chess-champ"  # Protected

        await bot.close()

    @pytest.mark.asyncio
    async def test_escalation_to_recent_messages(self) -> None:
        """Escalation can target recent_messages as the broadened scope."""
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
            "scope_escalation": {
                "enabled": True,
                "escalation_scope": "recent_messages",
                "recent_messages_count": 3,
            },
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            # Turn 1: name
            SimpleExtractionResult(
                data={"name": "Alice"}, confidence=0.9,
            ),
            # Turn 2: current_message — domain_id only
            SimpleExtractionResult(
                data={"domain_id": "chess-champ"}, confidence=0.9,
            ),
            # Turn 2 escalated (recent_messages) — all fields
            SimpleExtractionResult(
                data={
                    "name": "Alice",
                    "domain_name": "Chess Champ",
                    "domain_id": "chess-champ",
                },
                confidence=0.9,
            ),
        ])

        main_provider.set_responses(["Tell me the domain.", "All set!"])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        await bot.chat("I'm Alice", ctx)
        await bot.chat("Call it Chess Champ, ID chess-champ", ctx)

        data = _get_wizard_data(bot)
        assert data["domain_name"] == "Chess Champ"
        # 3 extract calls: turn 1 + turn 2 initial + turn 2 escalated
        assert len(extractor.extract_calls) == 3

        await bot.close()

    @pytest.mark.asyncio
    async def test_escalation_higher_confidence_replaces(self) -> None:
        """When escalated extraction has higher confidence, wizard proceeds
        without clarification."""
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            # Turn 1: establish name in history
            SimpleExtractionResult(
                data={"name": "Alice"}, confidence=0.9,
            ),
            # Turn 2 initial: low confidence, partial
            SimpleExtractionResult(
                data={"domain_id": "chess-champ"}, confidence=0.3,
            ),
            # Turn 2 escalated: high confidence, full
            SimpleExtractionResult(
                data={
                    "name": "Alice",
                    "domain_name": "Chess Champ",
                    "domain_id": "chess-champ",
                },
                confidence=0.95,
            ),
        ])

        main_provider.set_responses(["Got it!", "All set!"])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        # Turn 1: establish history
        await bot.chat("I'm Alice", ctx)
        # Turn 2: escalation fires, high confidence replaces low
        await bot.chat("Chess Champ, chess-champ", ctx)

        data = _get_wizard_data(bot)
        assert data["name"] == "Alice"
        assert data["domain_name"] == "Chess Champ"
        assert data["domain_id"] == "chess-champ"

        await bot.close()

    @pytest.mark.asyncio
    async def test_no_escalation_on_first_turn(self) -> None:
        """Escalation does not fire on the first turn (no prior history).

        On turn 1 there are no prior user messages, so escalating to a
        broader scope would just re-extract the same content at higher
        cost for zero benefit.
        """
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            # Only returns partial data — missing 2/3 required fields
            SimpleExtractionResult(
                data={"name": "Alice"}, confidence=0.5,
            ),
        ])

        main_provider.set_responses(["Tell me the domain info."])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        await bot.chat("I'm Alice", ctx)

        # Only 1 extract call — no escalation on first turn
        assert len(extractor.extract_calls) == 1

        await bot.close()

    @pytest.mark.asyncio
    async def test_escalation_with_empty_result(self) -> None:
        """When escalated extraction returns empty data, wizard falls
        through to the confidence gate with the original extraction."""
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            # Turn 1: name only
            SimpleExtractionResult(
                data={"name": "Alice"}, confidence=0.9,
            ),
            # Turn 2: partial data
            SimpleExtractionResult(
                data={"domain_id": "chess-champ"}, confidence=0.5,
            ),
            # Turn 2 escalated: empty result
            SimpleExtractionResult(
                data={}, confidence=0.3,
            ),
        ])

        main_provider.set_responses(["Got it!", "What's the domain name?"])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        await bot.chat("I'm Alice", ctx)
        await bot.chat("ID chess-champ", ctx)

        data = _get_wizard_data(bot)
        # Partial data from initial extraction persists
        assert data.get("name") == "Alice"
        assert data.get("domain_id") == "chess-champ"
        # domain_name still missing — escalation returned nothing
        assert data.get("domain_name") is None

        # Escalation did fire (3 calls) but produced no data
        assert len(extractor.extract_calls) == 3

        await bot.close()


# ---------------------------------------------------------------------------
# Tests: recent_messages scope
# ---------------------------------------------------------------------------


class TestRecentMessagesScope:
    """Verify the recent_messages extraction scope via bot.chat()."""

    @pytest.mark.asyncio
    async def test_recent_messages_limits_context(self) -> None:
        """recent_messages scope passes only last N user messages."""
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "recent_messages",
            "scope_escalation": {"recent_messages_count": 2},
        }))
        ctx = _make_context()
        main_provider = EchoProvider({"provider": "echo", "model": "test"})

        extractor = ConfigurableExtractor(results=[
            SimpleExtractionResult(data={}, confidence=0.5),
            SimpleExtractionResult(data={}, confidence=0.5),
            SimpleExtractionResult(data={}, confidence=0.5),
            SimpleExtractionResult(data={}, confidence=0.5),
        ])

        main_provider.set_responses(["OK", "OK", "OK", "OK"])
        bot.llm = main_provider
        _inject_extractor(bot, extractor)

        for msg in ["Msg one", "Msg two", "Msg three", "Msg four"]:
            await bot.chat(msg, ctx)

        # Turn 4 extraction: with recent_messages_count=2, context should
        # include msgs 2+3 (not msg 1), because msg 4 is the current message
        last_call = extractor.extract_calls[-1]
        extraction_text = last_call["text"]

        assert "Msg one" not in extraction_text
        assert "Msg two" in extraction_text
        assert "Msg three" in extraction_text

        await bot.close()


# ---------------------------------------------------------------------------
# Tests: Config loading via from_config()
# ---------------------------------------------------------------------------


class TestScopeEscalationConfig:
    """Verify scope_escalation settings loaded through DynaBot.from_config()."""

    @pytest.mark.asyncio
    async def test_config_loading_enabled(self) -> None:
        """scope_escalation settings are wired through from_config()."""
        bot = await DynaBot.from_config(_bot_config({
            "extraction_scope": "current_message",
            "scope_escalation": {
                "enabled": True,
                "escalation_scope": "recent_messages",
                "recent_messages_count": 5,
            },
        }))

        strategy = bot.reasoning_strategy
        assert strategy._scope_escalation_enabled is True
        assert strategy._scope_escalation_scope == "recent_messages"
        assert strategy._recent_messages_count == 5

        await bot.close()

    @pytest.mark.asyncio
    async def test_config_defaults(self) -> None:
        """Without scope_escalation config, defaults apply."""
        bot = await DynaBot.from_config(_bot_config())

        strategy = bot.reasoning_strategy
        assert strategy._scope_escalation_enabled is False
        assert strategy._scope_escalation_scope == "wizard_session"
        assert strategy._recent_messages_count == 3

        await bot.close()


# ---------------------------------------------------------------------------
# Tests: SCOPE_BREADTH constant
# ---------------------------------------------------------------------------


class TestScopeBreadth:
    """Verify the scope breadth ordering constant."""

    def test_breadth_ordering(self) -> None:
        """current_message < recent_messages < wizard_session."""
        assert SCOPE_BREADTH["current_message"] < SCOPE_BREADTH["recent_messages"]
        assert SCOPE_BREADTH["recent_messages"] < SCOPE_BREADTH["wizard_session"]

    def test_all_scopes_present(self) -> None:
        """All three scopes are in the breadth map."""
        assert set(SCOPE_BREADTH.keys()) == {
            "current_message",
            "recent_messages",
            "wizard_session",
        }

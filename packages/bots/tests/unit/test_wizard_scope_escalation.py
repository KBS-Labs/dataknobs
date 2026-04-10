"""Tests for wizard extraction scope escalation.

When extraction_scope is "current_message" and required fields are still
missing after the initial extraction + merge, scope escalation retries
with a broader scope (e.g. "wizard_session") to recover fields from
earlier conversation turns.

Also tests the ``recent_messages`` scope and the ``SCOPE_BREADTH``
constant.

These tests exercise the full DynaBot.from_config() → bot.chat() path
via ``BotTestHarness`` to verify scope escalation through the public API.
"""

import pytest

from dataknobs_bots.reasoning.wizard import SCOPE_BREADTH
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


# ---------------------------------------------------------------------------
# Shared wizard config builder
# ---------------------------------------------------------------------------


def _wizard_config() -> dict:
    """Gather stage with 3 required fields."""
    return (
        WizardConfigBuilder("escalation-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your name, domain_name, and domain_id.",
        )
        .field("name", field_type="string", required=True)
        .field("domain_name", field_type="string", required=True)
        .field("domain_id", field_type="string", required=True)
        .transition(
            "done",
            "data.get('name') and data.get('domain_name') "
            "and data.get('domain_id')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


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
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                # Turn 1: name only
                [{"name": "Alice"}],
                # Turn 2: current_message — domain_id; escalated — all fields
                [
                    {"domain_id": "chess-champ"},
                    {
                        "name": "Alice",
                        "domain_name": "Chess Champ",
                        "domain_id": "chess-champ",
                    },
                ],
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            assert harness.wizard_data["name"] == "Alice"

            await harness.chat("Call it Chess Champ, ID chess-champ")
            assert harness.wizard_data["name"] == "Alice"
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_data["domain_id"] == "chess-champ"

            # 3 extract calls: turn 1 + turn 2 initial + turn 2 escalated
            assert harness.extractor is not None
            assert len(harness.extractor.extract_calls) == 3

    @pytest.mark.asyncio
    async def test_no_escalation_when_fields_satisfied(self) -> None:
        """Fast path: no second extraction when all required fields filled."""
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["All set!"],
            extraction_results=[
                [
                    {
                        "name": "Alice",
                        "domain_name": "Chess Champ",
                        "domain_id": "chess-champ",
                    },
                ],
            ],
        ) as harness:
            await harness.chat("Alice, Chess Champ, chess-champ")
            assert harness.extractor is not None
            assert len(harness.extractor.extract_calls) == 1

    @pytest.mark.asyncio
    async def test_no_escalation_when_already_session_scope(self) -> None:
        """No escalation when extraction_scope is already wizard_session."""
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "wizard_session",
            "scope_escalation": {"enabled": True},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["What's the domain name?"],
            extraction_results=[
                [{"name": "Alice", "domain_id": "chess-champ"}],
            ],
            extraction_scope="wizard_session",
        ) as harness:
            await harness.chat("Alice, chess-champ")
            assert harness.extractor is not None
            assert len(harness.extractor.extract_calls) == 1

    @pytest.mark.asyncio
    async def test_escalation_disabled_by_default(self) -> None:
        """Escalation is disabled when not configured."""
        async with await BotTestHarness.create(
            wizard_config=_wizard_config(),
            main_responses=["Tell me more."],
            extraction_results=[
                [{"name": "Alice"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            assert harness.extractor is not None
            assert len(harness.extractor.extract_calls) == 1

    @pytest.mark.asyncio
    async def test_escalation_with_grounding_protects_existing(self) -> None:
        """Grounding filter protects existing data during escalation.

        Escalated extraction may return values for already-filled fields.
        The grounding filter should block ungrounded overwrites.
        """
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "extraction_grounding": True,
            "scope_escalation": {"enabled": True},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                # Turn 1: domain_id only
                [{"domain_id": "chess-champ"}],
                # Turn 2: current = name; escalated = all 3 but wrong domain_id
                [
                    {"name": "Alice"},
                    {
                        "name": "Alice",
                        "domain_name": "Chess Champ",
                        "domain_id": "wrong-id",
                    },
                ],
            ],
        ) as harness:
            await harness.chat("ID is chess-champ")
            assert harness.wizard_data["domain_id"] == "chess-champ"

            await harness.chat("Alice, Chess Champ")
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_data["domain_id"] == "chess-champ"  # Protected

    @pytest.mark.asyncio
    async def test_escalation_to_recent_messages(self) -> None:
        """Escalation can target recent_messages as the broadened scope."""
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {
                "enabled": True,
                "escalation_scope": "recent_messages",
                "recent_messages_count": 3,
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Tell me the domain.", "All set!"],
            extraction_results=[
                [{"name": "Alice"}],
                [
                    {"domain_id": "chess-champ"},
                    {
                        "name": "Alice",
                        "domain_name": "Chess Champ",
                        "domain_id": "chess-champ",
                    },
                ],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("Call it Chess Champ, ID chess-champ")
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.extractor is not None
            assert len(harness.extractor.extract_calls) == 3

    @pytest.mark.asyncio
    async def test_escalation_higher_confidence_replaces(self) -> None:
        """When escalated extraction has higher confidence, wizard proceeds."""
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                [{"name": "Alice"}],
                [
                    {"domain_id": "chess-champ"},
                    {
                        "name": "Alice",
                        "domain_name": "Chess Champ",
                        "domain_id": "chess-champ",
                    },
                ],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("Chess Champ, chess-champ")
            assert harness.wizard_data["name"] == "Alice"
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_data["domain_id"] == "chess-champ"

    @pytest.mark.asyncio
    async def test_no_escalation_on_first_turn(self) -> None:
        """Escalation does not fire on the first turn (no prior history)."""
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Tell me the domain info."],
            extraction_results=[
                [{"name": "Alice"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            assert harness.extractor is not None
            assert len(harness.extractor.extract_calls) == 1

    @pytest.mark.asyncio
    async def test_escalation_with_empty_result(self) -> None:
        """When escalated extraction returns empty data, wizard falls
        through to the confidence gate with the original extraction."""
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "What's the domain name?"],
            extraction_results=[
                [{"name": "Alice"}],
                [{"domain_id": "chess-champ"}, {}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("ID chess-champ")
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_data.get("domain_id") == "chess-champ"
            assert harness.wizard_data.get("domain_name") is None
            assert harness.extractor is not None
            assert len(harness.extractor.extract_calls) == 3


# ---------------------------------------------------------------------------
# Tests: recent_messages scope
# ---------------------------------------------------------------------------


class TestRecentMessagesScope:
    """Verify the recent_messages extraction scope via bot.chat()."""

    @pytest.mark.asyncio
    async def test_recent_messages_limits_context(self) -> None:
        """recent_messages scope passes only last N user messages."""
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "recent_messages",
            "scope_escalation": {"recent_messages_count": 2},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["OK", "OK", "OK", "OK"],
            extraction_results=[
                [{}],
                [{}],
                [{}],
                [{}],
            ],
            extraction_scope="recent_messages",
        ) as harness:
            for msg in ["Msg one", "Msg two", "Msg three", "Msg four"]:
                await harness.chat(msg)

            # Turn 4 extraction: with recent_messages_count=2, context
            # includes msgs 2+3 (not msg 1)
            assert harness.extractor is not None
            last_call = harness.extractor.extract_calls[-1]
            extraction_text = last_call["text"]

            assert "Msg one" not in extraction_text
            assert "Msg two" in extraction_text
            assert "Msg three" in extraction_text


# ---------------------------------------------------------------------------
# Tests: Config loading via from_config()
# ---------------------------------------------------------------------------


class TestScopeEscalationConfig:
    """Verify scope_escalation settings loaded through DynaBot.from_config()."""

    @pytest.mark.asyncio
    async def test_config_loading_enabled(self) -> None:
        """scope_escalation settings are wired through from_config()."""
        config = _wizard_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {
                "enabled": True,
                "escalation_scope": "recent_messages",
                "recent_messages_count": 5,
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert strategy._extraction._scope_escalation_enabled is True
            assert strategy._extraction._scope_escalation_scope == "recent_messages"
            assert strategy._extraction._recent_messages_count == 5

    @pytest.mark.asyncio
    async def test_config_defaults(self) -> None:
        """Without scope_escalation config, defaults apply."""
        async with await BotTestHarness.create(
            wizard_config=_wizard_config(),
        ) as harness:
            strategy = harness.bot.reasoning_strategy
            assert strategy._extraction._scope_escalation_enabled is False
            assert strategy._extraction._scope_escalation_scope == "wizard_session"
            assert strategy._extraction._recent_messages_count == 3


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

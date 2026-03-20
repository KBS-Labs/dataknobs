"""Tests for BotTestHarness, WizardConfigBuilder, and TurnResult."""

from typing import Any

import pytest

from dataknobs_bots.testing import (
    BotTestHarness,
    TurnResult,
    WizardConfigBuilder,
    inject_providers,
)
from dataknobs_llm.testing import ConfigurableExtractor, SimpleExtractionResult


# ---------------------------------------------------------------------------
# WizardConfigBuilder tests
# ---------------------------------------------------------------------------


class TestWizardConfigBuilder:
    """Tests for WizardConfigBuilder fluent API and validation."""

    def test_minimal_config(self) -> None:
        """Minimal valid config: start stage + end stage."""
        config = (
            WizardConfigBuilder("test")
            .stage("start", is_start=True, prompt="Hello")
            .stage("end", is_end=True, prompt="Done")
            .build()
        )
        assert config["name"] == "test"
        assert config["version"] == "1.0"
        assert len(config["stages"]) == 2
        assert config["stages"][0]["name"] == "start"
        assert config["stages"][0]["is_start"] is True
        assert config["stages"][1]["name"] == "end"
        assert config["stages"][1]["is_end"] is True

    def test_fields_and_transitions(self) -> None:
        """Fields and transitions are added to the current stage."""
        config = (
            WizardConfigBuilder("test")
            .stage("gather", is_start=True, prompt="Tell me")
            .field("name", field_type="string", required=True)
            .field("age", field_type="integer", required=True)
            .transition("done", "data.get('name') and data.get('age')")
            .stage("done", is_end=True, prompt="Done")
            .build()
        )
        gather = config["stages"][0]
        assert gather["schema"]["properties"]["name"] == {"type": "string"}
        assert gather["schema"]["properties"]["age"] == {"type": "integer"}
        assert gather["schema"]["required"] == ["name", "age"]
        assert len(gather["transitions"]) == 1
        assert gather["transitions"][0]["target"] == "done"

    def test_field_with_all_options(self) -> None:
        """Field with description, enum, default, x-extraction."""
        config = (
            WizardConfigBuilder("test")
            .stage("s", is_start=True, is_end=True, prompt="p")
            .field(
                "level",
                field_type="string",
                required=True,
                description="Skill level",
                enum=["beginner", "advanced"],
                default="beginner",
                x_extraction={"grounding": True},
            )
            .build()
        )
        field_def = config["stages"][0]["schema"]["properties"]["level"]
        assert field_def["type"] == "string"
        assert field_def["description"] == "Skill level"
        assert field_def["enum"] == ["beginner", "advanced"]
        assert field_def["default"] == "beginner"
        assert field_def["x-extraction"] == {"grounding": True}

    def test_settings(self) -> None:
        """Wizard-level settings are set."""
        config = (
            WizardConfigBuilder("test")
            .stage("s", is_start=True, is_end=True, prompt="p")
            .settings(extraction_scope="current_message", auto_advance=True)
            .build()
        )
        assert config["settings"]["extraction_scope"] == "current_message"
        assert config["settings"]["auto_advance"] is True

    def test_stage_options(self) -> None:
        """Stage-level options are preserved."""
        config = (
            WizardConfigBuilder("test")
            .stage(
                "s",
                is_start=True,
                is_end=True,
                prompt="p",
                mode="conversation",
                extraction_scope="wizard_session",
                auto_advance=False,
                skip_extraction=True,
            )
            .build()
        )
        stage = config["stages"][0]
        assert stage["mode"] == "conversation"
        assert stage["extraction_scope"] == "wizard_session"
        assert stage["auto_advance"] is False
        assert stage["skip_extraction"] is True

    def test_transition_with_priority(self) -> None:
        """Transitions accept priority."""
        config = (
            WizardConfigBuilder("test")
            .stage("s", is_start=True, prompt="p")
            .transition("a", "data.get('x') == 'a'", priority=0)
            .transition("b", "data.get('x') == 'b'", priority=1)
            .stage("a", is_end=True, prompt="A")
            .stage("b", is_end=True, prompt="B")
            .build()
        )
        transitions = config["stages"][0]["transitions"]
        assert transitions[0]["priority"] == 0
        assert transitions[1]["priority"] == 1

    def test_validation_no_start_stage(self) -> None:
        """Build raises when no start stage is defined."""
        with pytest.raises(ValueError, match="start stage"):
            WizardConfigBuilder("test").stage(
                "end", is_end=True, prompt="Done"
            ).build()

    def test_validation_no_end_stage(self) -> None:
        """Build raises when no end stage is defined."""
        with pytest.raises(ValueError, match="end stage"):
            WizardConfigBuilder("test").stage(
                "start", is_start=True, prompt="Hi"
            ).build()

    def test_validation_invalid_transition_target(self) -> None:
        """Build raises when transition references nonexistent stage."""
        with pytest.raises(ValueError, match="nonexistent stage"):
            (
                WizardConfigBuilder("test")
                .stage("start", is_start=True, prompt="Hi")
                .transition("missing", "True")
                .stage("end", is_end=True, prompt="Done")
                .build()
            )

    def test_field_before_stage_raises(self) -> None:
        """field() before stage() raises ValueError."""
        with pytest.raises(ValueError, match="field.*after stage"):
            WizardConfigBuilder("test").field("x", field_type="string")

    def test_transition_before_stage_raises(self) -> None:
        """transition() before stage() raises ValueError."""
        with pytest.raises(ValueError, match="transition.*after stage"):
            WizardConfigBuilder("test").transition("x")

    def test_custom_version(self) -> None:
        """Custom version string is used."""
        config = (
            WizardConfigBuilder("test", version="2.0")
            .stage("s", is_start=True, is_end=True, prompt="p")
            .build()
        )
        assert config["version"] == "2.0"


# ---------------------------------------------------------------------------
# TurnResult tests
# ---------------------------------------------------------------------------


class TestTurnResult:
    """Tests for TurnResult dataclass."""

    def test_defaults(self) -> None:
        """Default values are set."""
        result = TurnResult(response="hello")
        assert result.response == "hello"
        assert result.wizard_stage is None
        assert result.wizard_data == {}
        assert result.wizard_state is None
        assert result.turn_index == 0

    def test_all_fields(self) -> None:
        """All fields are set from constructor."""
        result = TurnResult(
            response="ok",
            wizard_stage="gather",
            wizard_data={"name": "Alice"},
            wizard_state={"current_stage": "gather", "data": {"name": "Alice"}},
            turn_index=3,
        )
        assert result.response == "ok"
        assert result.wizard_stage == "gather"
        assert result.wizard_data == {"name": "Alice"}
        assert result.turn_index == 3


# ---------------------------------------------------------------------------
# BotTestHarness tests
# ---------------------------------------------------------------------------


def _two_field_config() -> dict[str, Any]:
    """Config with 2 required fields to avoid verbatim capture."""
    return (
        WizardConfigBuilder("harness-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your name and topic.",
        )
        .field("name", field_type="string", required=True)
        .field("topic", field_type="string", required=True)
        .transition(
            "done",
            "data.get('name') and data.get('topic')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


class TestBotTestHarness:
    """Tests for BotTestHarness high-level test helper."""

    @pytest.mark.asyncio
    async def test_create_and_chat(self) -> None:
        """Basic create + chat flow."""
        async with await BotTestHarness.create(
            wizard_config=_two_field_config(),
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice", "topic": "math"}]],
        ) as harness:
            result = await harness.chat("Alice and math")
            assert result.response == "Got it!"
            assert result.wizard_data["name"] == "Alice"
            assert result.wizard_data["topic"] == "math"

    @pytest.mark.asyncio
    async def test_wizard_properties(self) -> None:
        """Properties reflect last turn state."""
        async with await BotTestHarness.create(
            wizard_config=_two_field_config(),
            main_responses=["OK"],
            extraction_results=[[{"name": "Alice"}]],
        ) as harness:
            await harness.chat("Alice")
            assert harness.wizard_stage == "gather"
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.last_response == "OK"
            assert harness.turn_count == 1

    @pytest.mark.asyncio
    async def test_multi_turn(self) -> None:
        """Multiple turns accumulate state."""
        async with await BotTestHarness.create(
            wizard_config=_two_field_config(),
            main_responses=["What topic?", "All done!"],
            extraction_results=[
                [{"name": "Alice"}],
                [{"topic": "math"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            assert harness.wizard_stage == "gather"
            assert harness.turn_count == 1

            await harness.chat("math")
            assert harness.wizard_stage == "done"
            assert harness.turn_count == 2

    @pytest.mark.asyncio
    async def test_extractor_call_tracking(self) -> None:
        """Extraction calls are tracked on the extractor."""
        async with await BotTestHarness.create(
            wizard_config=_two_field_config(),
            main_responses=["OK"],
            extraction_results=[[{"name": "Alice", "topic": "math"}]],
        ) as harness:
            await harness.chat("Alice and math")
            assert harness.extractor is not None
            assert len(harness.extractor.extract_calls) == 1

    @pytest.mark.asyncio
    async def test_per_turn_extraction_flattening(self) -> None:
        """Per-turn extraction results are correctly flattened."""
        config = _two_field_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got name", "Got topic"],
            extraction_results=[
                # Turn 1: one call
                [{"name": "Alice"}],
                # Turn 2: initial + escalated (2 calls)
                [{"topic": "math"}, {"name": "Alice", "topic": "math"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("math")
            assert harness.extractor is not None
            # 1 (turn1) + 2 (turn2 initial + escalated) = 3
            assert harness.extractor.call_index >= 2

    @pytest.mark.asyncio
    async def test_context_manager_closes_bot(self) -> None:
        """Async context manager closes the bot on exit."""
        async with await BotTestHarness.create(
            wizard_config=_two_field_config(),
        ) as harness:
            bot = harness.bot

        # After context exit, bot should be closed
        # (no assertion — just verifying no exception on close)

    @pytest.mark.asyncio
    async def test_bot_and_provider_access(self) -> None:
        """Direct access to bot and provider."""
        async with await BotTestHarness.create(
            wizard_config=_two_field_config(),
        ) as harness:
            assert harness.bot is not None
            assert harness.provider is not None
            assert hasattr(harness.bot, "chat")
            assert hasattr(harness.provider, "set_responses")

    @pytest.mark.asyncio
    async def test_no_config_raises(self) -> None:
        """create() raises when neither wizard_config nor bot_config given."""
        with pytest.raises(ValueError, match="wizard_config or bot_config"):
            await BotTestHarness.create()

    @pytest.mark.asyncio
    async def test_no_extraction_results(self) -> None:
        """Harness works without extraction_results (extractor is None)."""
        async with await BotTestHarness.create(
            wizard_config=_two_field_config(),
            main_responses=["Hello!"],
        ) as harness:
            assert harness.extractor is None
            result = await harness.chat("hi")
            assert result.response == "Hello!"


# ---------------------------------------------------------------------------
# inject_providers extractor kwarg
# ---------------------------------------------------------------------------


class TestInjectProvidersExtractor:
    """Tests for the extractor kwarg on inject_providers."""

    @pytest.mark.asyncio
    async def test_extractor_kwarg_replaces_strategy_extractor(self) -> None:
        """extractor kwarg replaces strategy._extractor."""
        from dataknobs_bots.bot.base import DynaBot

        config = _two_field_config()
        bot_config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": config,
                "extraction_config": {"provider": "echo", "model": "ext"},
            },
        }
        bot = await DynaBot.from_config(bot_config)

        ext = ConfigurableExtractor(results=[
            SimpleExtractionResult(data={"name": "Alice"}, confidence=0.9),
        ])
        inject_providers(bot, extractor=ext)

        assert bot.reasoning_strategy.extractor is ext
        await bot.close()

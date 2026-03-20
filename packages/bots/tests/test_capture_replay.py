"""Tests for capture-replay infrastructure in dataknobs_bots.testing.

Tests CaptureReplay loading/provider creation and inject_providers wiring.
No Ollama dependency — uses EchoProvider throughout.
"""

import json
from typing import Any

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.testing import CaptureReplay, WizardConfigBuilder, inject_providers
from dataknobs_llm import EchoProvider
from dataknobs_llm.testing import ConfigurableExtractor, SimpleExtractionResult

# =============================================================================
# Test fixtures
# =============================================================================


def _make_capture_data(
    turns: list | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build a minimal capture data dict for testing."""
    return {
        "format_version": "1.0",
        "metadata": metadata or {"description": "test capture"},
        "turns": turns or [],
    }


def _make_turn(
    turn_index: int,
    turn_type: str = "chat",
    user_message: str | None = "Hello",
    bot_response: str = "Hi there",
    llm_calls: list | None = None,
) -> dict:
    """Build a minimal turn dict."""
    return {
        "turn_index": turn_index,
        "type": turn_type,
        "user_message": user_message,
        "bot_response": bot_response,
        "wizard_state_before": None,
        "wizard_state_after": None,
        "llm_calls": llm_calls or [],
    }


def _make_llm_call(
    call_index: int,
    role: str = "main",
    content: str = "test response",
    model: str = "test-model",
) -> dict:
    """Build a minimal LLM call dict."""
    return {
        "call_index": call_index,
        "role": role,
        "request": {
            "messages": [{"role": "user", "content": "test"}],
        },
        "response": {
            "content": content,
            "model": model,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    }


def _wizard_bot_config() -> dict[str, Any]:
    """Bot config with wizard reasoning and extraction."""
    config = (
        WizardConfigBuilder("test")
        .stage("gather", is_start=True, prompt="Tell me your name and topic.")
        .field("name", field_type="string", required=True)
        .field("topic", field_type="string", required=True)
        .transition("done", "data.get('name') and data.get('topic')")
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )
    return {
        "llm": {"provider": "echo", "model": "original-main"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": {
            "strategy": "wizard",
            "wizard_config": config,
            "extraction_config": {
                "provider": "echo",
                "model": "original-extraction",
            },
        },
    }


def _simple_bot_config() -> dict[str, Any]:
    """Bot config with no reasoning strategy."""
    return {
        "llm": {"provider": "echo", "model": "original-main"},
        "conversation_storage": {"backend": "memory"},
    }


# =============================================================================
# CaptureReplay tests
# =============================================================================


class TestCaptureReplay:
    """CaptureReplay loading and provider creation."""

    def test_from_dict_basic(self):
        data = _make_capture_data(metadata={"description": "basic test"})
        replay = CaptureReplay.from_dict(data)
        assert replay.format_version == "1.0"
        assert replay.metadata["description"] == "basic test"
        assert replay.turns == []

    def test_from_dict_with_turns(self):
        turns = [
            _make_turn(0, llm_calls=[_make_llm_call(0)]),
            _make_turn(1, llm_calls=[_make_llm_call(1)]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        assert len(replay.turns) == 2

    def test_main_provider_has_responses(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="main", content="hello main"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        provider = replay.main_provider()
        assert provider.config.model == "capture-replay"

    def test_extraction_provider_has_responses(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="extraction", content="extracted"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        provider = replay.extraction_provider()
        assert provider.config.model == "capture-replay"

    def test_separates_main_and_extraction_calls(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="main", content="main response"),
                _make_llm_call(1, role="extraction", content="ext response"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        assert len(replay._main_responses) == 1
        assert len(replay._extraction_responses) == 1

    def test_from_file(self, tmp_path):
        data = _make_capture_data(
            turns=[_make_turn(0, llm_calls=[_make_llm_call(0)])],
            metadata={"description": "file test"},
        )
        path = tmp_path / "test_capture.json"
        path.write_text(json.dumps(data))

        replay = CaptureReplay.from_file(path)
        assert replay.metadata["description"] == "file test"
        assert len(replay.turns) == 1


# =============================================================================
# inject_providers tests — using real DynaBot instances
# =============================================================================


class TestInjectProviders:
    """inject_providers wires providers into real DynaBot instances."""

    @pytest.mark.asyncio
    async def test_injects_main_provider(self) -> None:
        bot = await DynaBot.from_config(_wizard_bot_config())
        new_main = EchoProvider({"provider": "echo", "model": "new-main"})
        inject_providers(bot, main_provider=new_main)
        assert bot.llm is new_main
        await bot.close()

    @pytest.mark.asyncio
    async def test_injects_extraction_provider(self) -> None:
        bot = await DynaBot.from_config(_wizard_bot_config())
        new_ext = EchoProvider({"provider": "echo", "model": "new-ext"})
        inject_providers(bot, extraction_provider=new_ext)
        assert bot.reasoning_strategy.extractor.provider is new_ext
        await bot.close()

    @pytest.mark.asyncio
    async def test_injects_both(self) -> None:
        bot = await DynaBot.from_config(_wizard_bot_config())
        new_main = EchoProvider({"provider": "echo", "model": "new-main"})
        new_ext = EchoProvider({"provider": "echo", "model": "new-ext"})
        inject_providers(bot, new_main, new_ext)
        assert bot.llm is new_main
        assert bot.reasoning_strategy.extractor.provider is new_ext
        await bot.close()

    @pytest.mark.asyncio
    async def test_none_keeps_existing(self) -> None:
        bot = await DynaBot.from_config(_wizard_bot_config())
        original_main = bot.llm
        original_ext_provider = bot.reasoning_strategy.extractor.provider
        inject_providers(bot)
        assert bot.llm is original_main
        assert bot.reasoning_strategy.extractor.provider is original_ext_provider
        await bot.close()

    @pytest.mark.asyncio
    async def test_extractor_kwarg_replaces_extractor(self) -> None:
        bot = await DynaBot.from_config(_wizard_bot_config())
        ext = ConfigurableExtractor(results=[
            SimpleExtractionResult(data={"name": "Alice"}, confidence=0.9),
        ])
        inject_providers(bot, extractor=ext)
        assert bot.reasoning_strategy.extractor is ext
        await bot.close()

    @pytest.mark.asyncio
    async def test_extractor_and_extraction_provider_mutually_exclusive(
        self,
    ) -> None:
        bot = await DynaBot.from_config(_wizard_bot_config())
        ext = ConfigurableExtractor()
        provider = EchoProvider({"provider": "echo", "model": "new"})
        with pytest.raises(ValueError, match="mutually exclusive"):
            inject_providers(bot, extraction_provider=provider, extractor=ext)
        await bot.close()

    @pytest.mark.asyncio
    async def test_no_strategy_skips_extraction_injection(self) -> None:
        """Bot without reasoning_strategy skips extraction injection."""
        bot = await DynaBot.from_config(_simple_bot_config())
        new_ext = EchoProvider({"provider": "echo", "model": "new"})
        # Should not raise
        inject_providers(bot, extraction_provider=new_ext)
        await bot.close()

    @pytest.mark.asyncio
    async def test_role_provider_unclaimed_does_not_raise(self) -> None:
        """Injecting a role no subsystem claims succeeds silently."""
        bot = await DynaBot.from_config(_wizard_bot_config())
        extra = EchoProvider({"provider": "echo", "model": "orphan"})
        inject_providers(bot, custom_role=extra)
        await bot.close()


class TestCaptureReplayInjectIntoBot:
    """CaptureReplay.inject_into_bot integrates with inject_providers."""

    @pytest.mark.asyncio
    async def test_injects_captured_providers(self) -> None:
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="main", content="captured main"),
                _make_llm_call(1, role="extraction", content="captured ext"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        bot = await DynaBot.from_config(_wizard_bot_config())
        replay.inject_into_bot(bot)

        assert bot.llm.config.model == "capture-replay"
        assert bot.reasoning_strategy.extractor.provider.config.model == "capture-replay"
        await bot.close()

    @pytest.mark.asyncio
    async def test_skips_extraction_when_no_extraction_calls(self) -> None:
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="main", content="only main"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        bot = await DynaBot.from_config(_wizard_bot_config())
        original_ext_provider = bot.reasoning_strategy.extractor.provider
        replay.inject_into_bot(bot)

        assert bot.llm.config.model == "capture-replay"
        assert bot.reasoning_strategy.extractor.provider is original_ext_provider
        await bot.close()

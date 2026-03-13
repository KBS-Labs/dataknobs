"""Tests for capture-replay infrastructure in dataknobs_bots.testing.

Tests CaptureReplay loading/provider creation and inject_providers wiring.
No Ollama dependency — uses EchoProvider throughout.
"""

import pytest

from dataknobs_bots.testing import CaptureReplay, inject_providers
from dataknobs_llm import EchoProvider

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
    content: str = "response text",
    model: str = "test-model",
) -> dict:
    """Build a minimal LLM call dict."""
    return {
        "call_index": call_index,
        "role": role,
        "messages": [{"role": "user", "content": "test"}],
        "response": {"content": content, "model": model},
        "config_overrides": None,
        "duration_seconds": 0.5,
    }


# =============================================================================
# CaptureReplay tests
# =============================================================================


class TestCaptureReplayFromDict:
    """CaptureReplay.from_dict creates correct instances."""

    def test_empty_capture(self):
        data = _make_capture_data()
        replay = CaptureReplay.from_dict(data)
        assert replay.format_version == "1.0"
        assert replay.metadata["description"] == "test capture"
        assert replay.turns == []

    def test_metadata_preserved(self):
        data = _make_capture_data(metadata={
            "description": "ConfigBot basic",
            "domain_id": "configbot",
            "captured_at": "2026-03-04T12:00:00",
        })
        replay = CaptureReplay.from_dict(data)
        assert replay.metadata["domain_id"] == "configbot"

    def test_turns_preserved(self):
        turns = [
            _make_turn(0, "greet", user_message=None, bot_response="Welcome!"),
            _make_turn(1, "chat", user_message="Hi", bot_response="Hello!"),
        ]
        data = _make_capture_data(turns=turns)
        replay = CaptureReplay.from_dict(data)
        assert len(replay.turns) == 2
        assert replay.turns[0]["type"] == "greet"
        assert replay.turns[1]["user_message"] == "Hi"


class TestCaptureReplayProviders:
    """CaptureReplay creates correctly queued EchoProviders."""

    def test_main_provider_with_responses(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="main", content="First"),
                _make_llm_call(1, role="main", content="Second"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        provider = replay.main_provider()
        assert isinstance(provider, EchoProvider)

    def test_extraction_provider_with_responses(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="extraction", content='{"name": "Test"}'),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        provider = replay.extraction_provider()
        assert isinstance(provider, EchoProvider)

    def test_responses_separated_by_role(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="extraction", content="extracted"),
                _make_llm_call(1, role="main", content="main response"),
            ]),
            _make_turn(1, llm_calls=[
                _make_llm_call(2, role="main", content="second main"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))

        # Main should have 2 responses
        assert len(replay._main_responses) == 2
        assert replay._main_responses[0].content == "main response"
        assert replay._main_responses[1].content == "second main"

        # Extraction should have 1 response
        assert len(replay._extraction_responses) == 1
        assert replay._extraction_responses[0].content == "extracted"

    def test_empty_capture_creates_empty_providers(self):
        replay = CaptureReplay.from_dict(_make_capture_data())
        main = replay.main_provider()
        ext = replay.extraction_provider()
        assert isinstance(main, EchoProvider)
        assert isinstance(ext, EchoProvider)

    @pytest.mark.asyncio()
    async def test_main_provider_responses_in_order(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="main", content="first"),
            ]),
            _make_turn(1, llm_calls=[
                _make_llm_call(1, role="main", content="second"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        provider = replay.main_provider()

        r1 = await provider.complete("test 1")
        assert r1.content == "first"

        r2 = await provider.complete("test 2")
        assert r2.content == "second"

    @pytest.mark.asyncio()
    async def test_extraction_provider_responses_in_order(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="extraction", content='{"a": 1}'),
                _make_llm_call(1, role="extraction", content='{"b": 2}'),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        provider = replay.extraction_provider()

        r1 = await provider.complete("extract 1")
        assert r1.content == '{"a": 1}'

        r2 = await provider.complete("extract 2")
        assert r2.content == '{"b": 2}'


class TestCaptureReplayFromFile:
    """CaptureReplay.from_file loads from disk."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            CaptureReplay.from_file("/nonexistent/path.json")

    def test_loads_from_file(self, tmp_path):
        import json

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
# inject_providers tests
# =============================================================================


class _FakeExtractor:
    """Minimal extractor stub with _provider attribute."""

    def __init__(self):
        self._provider = EchoProvider({"provider": "echo", "model": "original-ext"})


class _FakeStrategy:
    """Minimal reasoning strategy stub with _extractor."""

    def __init__(self):
        self._extractor = _FakeExtractor()


class _FakeMemory:
    """Minimal memory stub with set_provider support."""

    def __init__(self):
        self.embedding_provider = EchoProvider({"provider": "echo", "model": "original-mem"})

    def set_provider(self, role: str, provider: object) -> bool:
        from dataknobs_bots.bot.base import PROVIDER_ROLE_MEMORY_EMBEDDING

        if role == PROVIDER_ROLE_MEMORY_EMBEDDING:
            self.embedding_provider = provider
            return True
        return False


class _FakeBot:
    """Minimal DynaBot stub for testing inject_providers."""

    def __init__(self, *, with_memory: bool = False):
        self.llm = EchoProvider({"provider": "echo", "model": "original-main"})
        self.reasoning_strategy = _FakeStrategy()
        self.memory = _FakeMemory() if with_memory else None
        self.knowledge_base = None


class TestInjectProviders:
    """inject_providers wires providers into bot attributes."""

    def test_injects_main_provider(self):
        bot = _FakeBot()
        new_main = EchoProvider({"provider": "echo", "model": "new-main"})
        inject_providers(bot, main_provider=new_main)
        assert bot.llm is new_main

    def test_injects_extraction_provider(self):
        bot = _FakeBot()
        new_ext = EchoProvider({"provider": "echo", "model": "new-ext"})
        inject_providers(bot, extraction_provider=new_ext)
        assert bot.reasoning_strategy._extractor._provider is new_ext

    def test_injects_both(self):
        bot = _FakeBot()
        new_main = EchoProvider({"provider": "echo", "model": "new-main"})
        new_ext = EchoProvider({"provider": "echo", "model": "new-ext"})
        inject_providers(bot, new_main, new_ext)
        assert bot.llm is new_main
        assert bot.reasoning_strategy._extractor._provider is new_ext

    def test_none_keeps_existing(self):
        bot = _FakeBot()
        original_main = bot.llm
        original_ext = bot.reasoning_strategy._extractor._provider
        inject_providers(bot)
        assert bot.llm is original_main
        assert bot.reasoning_strategy._extractor._provider is original_ext

    def test_no_strategy_logs_warning(self):
        """Bot without reasoning_strategy skips extraction injection."""

        class BotNoStrategy:
            llm = EchoProvider({"provider": "echo", "model": "test"})

        bot = BotNoStrategy()
        new_ext = EchoProvider({"provider": "echo", "model": "new"})
        # Should not raise
        inject_providers(bot, extraction_provider=new_ext)

    def test_no_extractor_logs_warning(self):
        """Strategy without _extractor skips extraction injection."""

        class FakeStrategyNoExtractor:
            pass

        class BotNoExtractor:
            llm = EchoProvider({"provider": "echo", "model": "test"})
            reasoning_strategy = FakeStrategyNoExtractor()

        bot = BotNoExtractor()
        new_ext = EchoProvider({"provider": "echo", "model": "new"})
        # Should not raise
        inject_providers(bot, extraction_provider=new_ext)

    def test_role_provider_wired_into_memory_subsystem(self):
        """Role provider injection updates the actual memory subsystem."""
        from dataknobs_bots.bot.base import PROVIDER_ROLE_MEMORY_EMBEDDING

        bot = _FakeBot(with_memory=True)
        original = bot.memory.embedding_provider
        new_embed = EchoProvider({"provider": "echo", "model": "injected-embed"})

        inject_providers(bot, **{PROVIDER_ROLE_MEMORY_EMBEDDING: new_embed})

        # Subsystem should use the injected provider
        assert bot.memory.embedding_provider is new_embed
        assert bot.memory.embedding_provider is not original

    def test_role_provider_unclaimed_does_not_raise(self):
        """Injecting a role no subsystem claims succeeds silently."""
        bot = _FakeBot()
        extra = EchoProvider({"provider": "echo", "model": "orphan"})
        # Should not raise — just logs a debug message
        inject_providers(bot, custom_role=extra)

    def test_extraction_uses_set_provider_on_strategy(self):
        """Extraction provider injection uses set_provider when available."""
        from dataknobs_bots.bot.base import PROVIDER_ROLE_EXTRACTION
        from dataknobs_bots.reasoning.base import ReasoningStrategy

        class StubStrategy(ReasoningStrategy):
            def __init__(self):
                super().__init__()
                self.provider_for_role: dict[str, object] = {}

            def set_provider(self, role: str, provider: object) -> bool:
                if role == PROVIDER_ROLE_EXTRACTION:
                    self.provider_for_role[role] = provider
                    return True
                return False

            async def generate(self, manager, llm, tools=None, **kwargs):
                pass

        class BotWithStrategy:
            llm = EchoProvider({"provider": "echo", "model": "test"})
            reasoning_strategy = StubStrategy()

        bot = BotWithStrategy()
        new_ext = EchoProvider({"provider": "echo", "model": "new-ext"})
        inject_providers(bot, extraction_provider=new_ext)

        assert bot.reasoning_strategy.provider_for_role[PROVIDER_ROLE_EXTRACTION] is new_ext


class TestCaptureReplayInjectIntoBot:
    """CaptureReplay.inject_into_bot integrates with inject_providers."""

    def test_injects_captured_providers(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="main", content="captured main"),
                _make_llm_call(1, role="extraction", content="captured ext"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        bot = _FakeBot()
        replay.inject_into_bot(bot)

        # Main provider should be replaced with replay provider
        assert bot.llm.config.model == "capture-replay"
        # Extraction provider should be replaced
        assert bot.reasoning_strategy._extractor._provider.config.model == "capture-replay"

    def test_skips_extraction_when_no_extraction_calls(self):
        turns = [
            _make_turn(0, llm_calls=[
                _make_llm_call(0, role="main", content="only main"),
            ]),
        ]
        replay = CaptureReplay.from_dict(_make_capture_data(turns=turns))
        bot = _FakeBot()
        original_ext = bot.reasoning_strategy._extractor._provider
        replay.inject_into_bot(bot)

        # Main should be replaced
        assert bot.llm.config.model == "capture-replay"
        # Extraction should be unchanged (no extraction calls in capture)
        assert bot.reasoning_strategy._extractor._provider is original_ext

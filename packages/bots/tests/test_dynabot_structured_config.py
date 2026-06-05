"""Tests for the typed ``DynaBotConfig`` and DynaBot's structured-config adoption.

DynaBot is a structured-config consumer: it carries a typed
``DynaBotConfig`` snapshot, builds through the
``from_config`` → ``from_config_async`` → ``__init__`` → ``_setup`` →
``_ainit`` lifecycle, and keeps a dual-input constructor so the historical
pre-built shape (``DynaBot(llm=provider, prompt_builder=..., ...)``) keeps
working verbatim.

All tests use real constructs (``EchoProvider``, ``AsyncMemoryDatabase``,
real prompt builders) — no mocks.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.config import DynaBotConfig
from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import (
    assert_structured_config_consumer,
    assert_structured_config_roundtrip,
)
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import EchoProvider
from dataknobs_llm.conversations import DataknobsConversationStorage
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.prompts.implementations import CompositePromptLibrary

# Ctor parameters that are pre-built collaborators or derived scalars rather
# than ``DynaBotConfig`` fields. They live on the dual-input constructor for
# the pre-built shape and are not part of the serializable config snapshot.
_PREBUILT_CTOR_PARAMS = {
    "prompt_builder",
    "tool_registry",
    "kb_auto_context",
    "reasoning_strategy",
    "system_prompt_name",
    "system_prompt_content",
    "system_prompt_rag_configs",
    "default_temperature",
    "default_max_tokens",
    "prompt_resolver",
}


def _make_prompt_builder() -> AsyncPromptBuilder:
    return AsyncPromptBuilder(CompositePromptLibrary())


async def _make_storage() -> DataknobsConversationStorage:
    return DataknobsConversationStorage(AsyncMemoryDatabase())


# ---------------------------------------------------------------------------
# DynaBotConfig — typed config
# ---------------------------------------------------------------------------

class TestDynaBotConfig:
    """The typed top-level config snapshot."""

    def test_is_structured_config(self) -> None:
        assert issubclass(DynaBotConfig, StructuredConfig)

    def test_from_dict_field_parity(self) -> None:
        cfg = DynaBotConfig.from_dict(
            {
                "llm": {"provider": "echo", "model": "test", "temperature": 0.3},
                "conversation_storage": {"backend": "memory"},
                "max_tool_iterations": 7,
                "tool_timeout": 12.5,
                "tool_loop_timeout": 60.0,
            }
        )
        assert cfg.llm["temperature"] == 0.3
        assert cfg.conversation_storage == {"backend": "memory"}
        assert cfg.max_tool_iterations == 7
        assert cfg.tool_timeout == 12.5
        assert cfg.tool_loop_timeout == 60.0

    def test_polymorphic_sections_pass_through_verbatim(self) -> None:
        """memory/knowledge_base/reasoning/llm stay raw — never rebuilt."""
        memory = {"type": "buffer", "max_messages": 5}
        kb = {"enabled": True, "type": "rag"}
        reasoning = {"strategy": "grounded", "sources": [{"type": "kb"}]}
        cfg = DynaBotConfig.from_dict(
            {
                "llm": {"provider": "echo", "model": "test"},
                "memory": memory,
                "knowledge_base": kb,
                "reasoning": reasoning,
            }
        )
        # Values pass through verbatim as plain dicts (not typed sub-configs).
        assert cfg.memory == memory
        assert cfg.knowledge_base == kb
        assert cfg.reasoning == reasoning
        assert isinstance(cfg.memory, dict)
        assert isinstance(cfg.reasoning, dict)

    def test_unknown_keys_ignored(self) -> None:
        cfg = DynaBotConfig.from_dict(
            {"llm": {"provider": "echo"}, "not_a_field": "dropped"}
        )
        assert not hasattr(cfg, "not_a_field")

    def test_roundtrip(self) -> None:
        cfg = DynaBotConfig.from_dict(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "memory": {"type": "buffer"},
                "tools": [{"class": "x.Y"}],
                "tool_definitions": {"y": {"class": "x.Y"}},
                "system_prompt": {"name": "assistant"},
                "context_transform": "pkg.mod:fn",
                "max_tool_iterations": 3,
            }
        )
        assert_structured_config_roundtrip(cfg)

    def test_post_init_rejects_negative_tool_timeout(self) -> None:
        with pytest.raises(ValueError, match="tool_timeout must be non-negative"):
            DynaBotConfig(tool_timeout=-1.0)

    def test_post_init_rejects_negative_tool_loop_timeout(self) -> None:
        with pytest.raises(
            ValueError, match="tool_loop_timeout must be non-negative"
        ):
            DynaBotConfig(tool_loop_timeout=-1.0)


# ---------------------------------------------------------------------------
# Parity guard
# ---------------------------------------------------------------------------

class TestParityGuard:
    """DynaBot satisfies the structured-config consumer contract."""

    def test_consumer_parity(self) -> None:
        assert_structured_config_consumer(
            DynaBot, ignore_params=_PREBUILT_CTOR_PARAMS
        )


# ---------------------------------------------------------------------------
# Dual-input constructor — the back-compat guarantee (write-first discipline)
# ---------------------------------------------------------------------------

class TestDualInputConstructor:
    """The pre-built collaborator shape must keep working verbatim."""

    @pytest.mark.asyncio
    async def test_keyword_prebuilt_construction(self) -> None:
        provider = EchoProvider({"provider": "echo", "model": "test"})
        storage = await _make_storage()
        bot = DynaBot(
            llm=provider,
            prompt_builder=_make_prompt_builder(),
            conversation_storage=storage,
        )
        assert bot.llm is provider
        assert bot.conversation_storage is storage
        assert bot._owns_llm is True
        assert bot._prebuilt is True
        assert isinstance(bot.config, DynaBotConfig)

    @pytest.mark.asyncio
    async def test_positional_prebuilt_construction(self) -> None:
        provider = EchoProvider({"provider": "echo", "model": "test"})
        builder = _make_prompt_builder()
        storage = await _make_storage()
        bot = DynaBot(provider, builder, storage)
        assert bot.llm is provider
        assert bot.prompt_builder is builder
        assert bot.conversation_storage is storage

    @pytest.mark.asyncio
    async def test_prebuilt_ainit_short_circuits(self) -> None:
        """_ainit is a no-op for the pre-built shape (collaborators wired)."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        bot = DynaBot(
            llm=provider,
            prompt_builder=_make_prompt_builder(),
            conversation_storage=await _make_storage(),
        )
        await bot._ainit()  # must not rebuild or raise
        assert bot.llm is provider

    @pytest.mark.asyncio
    async def test_prebuilt_scalar_knobs_preserved(self) -> None:
        provider = EchoProvider({"provider": "echo", "model": "test"})
        bot = DynaBot(
            llm=provider,
            prompt_builder=_make_prompt_builder(),
            conversation_storage=await _make_storage(),
            default_temperature=0.25,
            default_max_tokens=512,
            max_tool_iterations=9,
        )
        assert bot.default_temperature == 0.25
        assert bot.default_max_tokens == 512
        assert bot._max_tool_iterations == 9

    @pytest.mark.asyncio
    async def test_prebuilt_prompt_envelope_default_is_markdown(self) -> None:
        """Omitting prompt_envelope on the pre-built shape uses the markdown default."""
        from dataknobs_bots.prompts import PromptEnvelopeStyle

        provider = EchoProvider({"provider": "echo", "model": "test"})
        bot = DynaBot(
            llm=provider,
            prompt_builder=_make_prompt_builder(),
            conversation_storage=await _make_storage(),
        )
        assert bot.config.prompt_envelope == "markdown"
        assert bot._prompt_envelope.style is PromptEnvelopeStyle.MARKDOWN

    @pytest.mark.asyncio
    async def test_prebuilt_prompt_envelope_xml_pins_legacy_shape(self) -> None:
        """Pre-built shape honors prompt_envelope='xml' just like the config path."""
        from dataknobs_bots.prompts import PromptEnvelopeStyle

        provider = EchoProvider({"provider": "echo", "model": "test"})
        bot = DynaBot(
            llm=provider,
            prompt_builder=_make_prompt_builder(),
            conversation_storage=await _make_storage(),
            prompt_envelope="xml",
        )
        assert bot.config.prompt_envelope == "xml"
        assert bot._prompt_envelope.style is PromptEnvelopeStyle.XML

    @pytest.mark.asyncio
    async def test_prebuilt_prompt_envelope_prose_round_trips(self) -> None:
        """Pre-built shape accepts the 'prose' style end-to-end."""
        from dataknobs_bots.prompts import PromptEnvelopeStyle

        provider = EchoProvider({"provider": "echo", "model": "test"})
        bot = DynaBot(
            llm=provider,
            prompt_builder=_make_prompt_builder(),
            conversation_storage=await _make_storage(),
            prompt_envelope="prose",
        )
        assert bot.config.prompt_envelope == "prose"
        assert bot._prompt_envelope.style is PromptEnvelopeStyle.PROSE

    def test_no_args_raises(self) -> None:
        with pytest.raises(TypeError, match="`llm` is required"):
            DynaBot()

    def test_prebuilt_without_storage_raises(self) -> None:
        """A built bot needs conversation storage — omitting it is rejected.

        ``ConversationManager`` (driven by every ``chat()``) requires a
        non-None storage, so a pre-built bot lacking it would fail on first
        use. The constructor rejects it up front instead of building a
        broken bot.
        """
        provider = EchoProvider({"provider": "echo", "model": "test"})
        with pytest.raises(TypeError, match="conversation_storage"):
            DynaBot(llm=provider, prompt_builder=_make_prompt_builder())

    @pytest.mark.asyncio
    async def test_prebuilt_without_prompt_builder_raises(self) -> None:
        """A built bot needs a prompt builder — omitting it is rejected."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        with pytest.raises(TypeError, match="prompt_builder"):
            DynaBot(llm=provider, conversation_storage=await _make_storage())

    def test_typed_config_with_collaborator_kwarg_raises(self) -> None:
        cfg = DynaBotConfig.from_dict({"llm": {"provider": "echo"}})
        with pytest.raises(TypeError, match="cannot mix a config"):
            DynaBot(cfg, prompt_builder=_make_prompt_builder())

    def test_typed_config_with_loose_kwarg_raises(self) -> None:
        # A typed config mixed with any loose kwarg is rejected by the mixin
        # (a dict config merges loose kwargs instead — that is the mixin's
        # documented dict-construction behavior, not an error).
        cfg = DynaBotConfig.from_dict({"llm": {"provider": "echo"}})
        with pytest.raises(TypeError):
            DynaBot(cfg, some_loose_kwarg=1)


# ---------------------------------------------------------------------------
# from_components — additive canonical alias of the pre-built shape
# ---------------------------------------------------------------------------

class TestFromComponents:
    @pytest.mark.asyncio
    async def test_adopts_prebuilt_collaborators(self) -> None:
        provider = EchoProvider({"provider": "echo", "model": "test"})
        builder = _make_prompt_builder()
        storage = await _make_storage()
        bot = DynaBot.from_components(
            llm=provider,
            prompt_builder=builder,
            conversation_storage=storage,
        )
        assert bot.llm is provider
        assert bot.prompt_builder is builder
        assert bot.conversation_storage is storage
        assert bot._prebuilt is True

    def test_requires_llm(self) -> None:
        with pytest.raises(TypeError, match="requires a built `llm`"):
            DynaBot.from_components(prompt_builder=_make_prompt_builder())

    def test_requires_prompt_builder_and_storage(self) -> None:
        """``from_components`` shares the pre-built collaborator contract:
        a built bot needs both a prompt builder and conversation storage."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        with pytest.raises(
            TypeError, match="prompt_builder and conversation_storage"
        ):
            DynaBot.from_components(llm=provider)


# ---------------------------------------------------------------------------
# Config-driven lifecycle
# ---------------------------------------------------------------------------

class TestConfigDrivenLifecycle:
    @pytest.mark.asyncio
    async def test_from_config_dict(self) -> None:
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        async with bot:
            assert isinstance(bot.config, DynaBotConfig)
            assert bot.config.llm == {"provider": "echo", "model": "test"}
            assert bot.llm is not None
            assert bot._owns_llm is True
            assert bot._prebuilt is False

    @pytest.mark.asyncio
    async def test_from_config_typed_equals_dict(self) -> None:
        raw = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "max_tool_iterations": 4,
        }
        bot_dict = await DynaBot.from_config(raw)
        bot_typed = await DynaBot.from_config(DynaBotConfig.from_dict(raw))
        async with bot_dict, bot_typed:
            assert bot_dict.config == bot_typed.config
            assert bot_dict._max_tool_iterations == bot_typed._max_tool_iterations == 4

    @pytest.mark.asyncio
    async def test_from_config_without_llm_raises_clear_error(self) -> None:
        """No ``llm`` section and no injected provider → a clear error.

        Without the guard this fails deep inside ``LLMConfig.from_dict({})``
        with an opaque ``TypeError`` about missing positional arguments. The
        guard surfaces a ``ConfigurationError`` naming the real problem.
        """
        from dataknobs_common.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="'llm' section"):
            await DynaBot.from_config(
                {"conversation_storage": {"backend": "memory"}}
            )

    @pytest.mark.asyncio
    async def test_from_config_injected_llm_not_owned(self) -> None:
        shared = EchoProvider({"provider": "echo", "model": "test"})
        await shared.initialize()
        bot = await DynaBot.from_config(
            {"conversation_storage": {"backend": "memory"}}, llm=shared
        )
        assert bot.llm is shared
        assert bot._owns_llm is False
        # bot.close() must not close a caller-owned provider.
        await bot.close()
        assert shared.close_count == 0
        await shared.close()

    @pytest.mark.asyncio
    async def test_from_config_middleware_override(self) -> None:
        from dataknobs_bots.middleware.logging import LoggingMiddleware

        mw = LoggingMiddleware()
        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[mw],
        )
        async with bot:
            assert bot.middleware == [mw]

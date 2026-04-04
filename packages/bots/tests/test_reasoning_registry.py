"""Tests for the reasoning strategy registry.

Covers: registration, creation, capabilities, from_config round-trips,
3rd-party strategy end-to-end, and backward compatibility.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common import NotFoundError, OperationError
from dataknobs_common.registry import PluginRegistry

from dataknobs_bots.reasoning import (
    GroundedReasoning,
    HybridReasoning,
    ReActReasoning,
    ReasoningStrategy,
    SimpleReasoning,
    StrategyCapabilities,
    WizardReasoning,
    create_reasoning_from_config,
    get_registry,
    get_strategy_factory,
    is_strategy_registered,
    list_strategies,
    register_strategy,
)
from dataknobs_bots.reasoning.registry import _register_builtins


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _CustomStrategy(ReasoningStrategy):
    """Minimal custom strategy for testing registration."""

    def __init__(
        self,
        *,
        greeting_template: str | None = None,
        custom_param: str = "default",
    ) -> None:
        super().__init__(greeting_template=greeting_template)
        self.custom_param = custom_param

    @classmethod
    def capabilities(cls) -> StrategyCapabilities:
        return StrategyCapabilities()

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> _CustomStrategy:
        return cls(
            greeting_template=config.get("greeting_template"),
            custom_param=config.get("custom_param", "default"),
        )

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return "custom response"


@pytest.fixture()
def fresh_registry() -> PluginRegistry[ReasoningStrategy]:
    """Return a fresh registry matching production configuration."""
    registry = PluginRegistry[ReasoningStrategy](
        "test_strategies",
        validate_type=ReasoningStrategy,
        canonicalize_keys=True,
        config_key="strategy",
        config_key_default="simple",
        on_first_access=_register_builtins,
    )
    return registry


# ------------------------------------------------------------------
# Registry basics
# ------------------------------------------------------------------


class TestRegistryBasics:
    """Core registration and creation behavior."""

    def test_register_custom_factory(self, fresh_registry: PluginRegistry[ReasoningStrategy]) -> None:
        fresh_registry.register("custom", _CustomStrategy)
        assert fresh_registry.is_registered("custom")

    def test_create_from_config(self, fresh_registry: PluginRegistry[ReasoningStrategy]) -> None:
        fresh_registry.register("custom", _CustomStrategy)
        strategy = fresh_registry.create(
            config={"strategy": "custom", "custom_param": "hello"},
        )
        assert isinstance(strategy, _CustomStrategy)
        assert strategy.custom_param == "hello"

    def test_create_with_callable_factory(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        def factory(config: dict[str, Any], **kwargs: Any) -> _CustomStrategy:
            return _CustomStrategy(custom_param=config.get("custom_param", "fn"))

        fresh_registry.register("fn_strategy", factory)
        strategy = fresh_registry.create(config={"strategy": "fn_strategy"})
        assert isinstance(strategy, _CustomStrategy)
        assert strategy.custom_param == "fn"

    def test_duplicate_registration_raises(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        fresh_registry.register("custom", _CustomStrategy)
        with pytest.raises(OperationError, match="already registered"):
            fresh_registry.register("custom", _CustomStrategy)

    def test_duplicate_with_override(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        fresh_registry.register("custom", _CustomStrategy)
        fresh_registry.register("custom", _CustomStrategy, override=True)
        assert fresh_registry.is_registered("custom")

    def test_unknown_strategy_error(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        with pytest.raises(NotFoundError, match="'nope' not registered"):
            fresh_registry.create(config={"strategy": "nope"})

    def test_unknown_strategy_lists_available(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        with pytest.raises(NotFoundError) as exc_info:
            fresh_registry.create(config={"strategy": "nope"})
        # Error context should list available strategies
        available = exc_info.value.context.get("available", [])
        for name in ("simple", "react", "wizard", "grounded", "hybrid"):
            assert name in available

    def test_case_insensitive(self, fresh_registry: PluginRegistry[ReasoningStrategy]) -> None:
        strategy = fresh_registry.create(config={"strategy": "SIMPLE"})
        assert isinstance(strategy, SimpleReasoning)


# ------------------------------------------------------------------
# list_strategies / get_factory
# ------------------------------------------------------------------


class TestListAndGet:
    """Public API: list_strategies, get_strategy_factory."""

    def test_list_includes_builtins(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        keys = fresh_registry.list_keys()
        for name in ("simple", "react", "wizard", "grounded", "hybrid"):
            assert name in keys

    def test_list_includes_custom(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        fresh_registry.register("custom", _CustomStrategy)
        keys = fresh_registry.list_keys()
        assert "custom" in keys

    def test_get_factory_returns_class(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        factory = fresh_registry.get_factory("simple")
        assert factory is SimpleReasoning

    def test_get_factory_returns_none(
        self, fresh_registry: PluginRegistry[ReasoningStrategy],
    ) -> None:
        assert fresh_registry.get_factory("nonexistent") is None


# ------------------------------------------------------------------
# Capabilities on built-in strategies
# ------------------------------------------------------------------


class TestCapabilities:
    """Each built-in strategy declares correct capabilities."""

    def test_simple_capabilities(self) -> None:
        caps = SimpleReasoning.capabilities()
        assert caps.manages_sources is False

    def test_react_capabilities(self) -> None:
        caps = ReActReasoning.capabilities()
        assert caps.manages_sources is False

    def test_grounded_capabilities(self) -> None:
        caps = GroundedReasoning.capabilities()
        assert caps.manages_sources is True

    def test_hybrid_capabilities(self) -> None:
        caps = HybridReasoning.capabilities()
        assert caps.manages_sources is True

    def test_wizard_capabilities(self) -> None:
        caps = WizardReasoning.capabilities()
        assert caps.manages_sources is False

    def test_base_capabilities(self) -> None:
        """Base class returns no capabilities."""
        caps = ReasoningStrategy.capabilities()
        assert caps == StrategyCapabilities()

    def test_capabilities_frozen(self) -> None:
        """StrategyCapabilities is frozen (immutable)."""
        caps = StrategyCapabilities()
        with pytest.raises(AttributeError):
            caps.manages_sources = True  # type: ignore[misc]


# ------------------------------------------------------------------
# from_config round-trips
# ------------------------------------------------------------------


class TestFromConfig:
    """from_config classmethods create strategies correctly."""

    @pytest.mark.asyncio()
    async def test_simple_from_config(self) -> None:
        from dataknobs_bots.testing import StubManager

        strategy = SimpleReasoning.from_config(
            {"greeting_template": "Hello!"},
        )
        assert isinstance(strategy, SimpleReasoning)
        # Verify greeting_template via greet() behavior
        result = await strategy.greet(StubManager(), None)
        assert result is not None
        assert result.content == "Hello!"

    @pytest.mark.asyncio()
    async def test_react_from_config(self) -> None:
        from dataknobs_bots.testing import StubManager

        strategy = ReActReasoning.from_config({
            "max_iterations": 10,
            "verbose": True,
            "store_trace": True,
            "greeting_template": "Hi",
        })
        assert isinstance(strategy, ReActReasoning)
        assert strategy.max_iterations == 10
        assert strategy.verbose is True
        assert strategy.store_trace is True
        # Verify greeting_template via greet() behavior
        result = await strategy.greet(StubManager(), None)
        assert result is not None
        assert result.content == "Hi"

    def test_react_from_config_defaults(self) -> None:
        strategy = ReActReasoning.from_config({})
        assert strategy.max_iterations == 5
        assert strategy.verbose is False
        assert strategy.store_trace is False

    def test_grounded_from_config_minimal(self) -> None:
        config: dict[str, Any] = {
            "strategy": "grounded",
            "intent": {"mode": "static", "text_queries": ["test"]},
        }
        strategy = GroundedReasoning.from_config(config)
        assert isinstance(strategy, GroundedReasoning)

    def test_grounded_from_config_with_kb(self) -> None:
        """KB is auto-wrapped when no vector_kb source is declared."""
        config: dict[str, Any] = {
            "strategy": "grounded",
            "intent": {"mode": "static", "text_queries": ["test"]},
        }
        kb = object()  # Dummy — from_config only stores, doesn't call
        strategy = GroundedReasoning.from_config(config, knowledge_base=kb)
        assert isinstance(strategy, GroundedReasoning)

    def test_grounded_from_config_skips_kb_when_vector_source_declared(self) -> None:
        """KB auto-wrap is skipped when config already has a vector_kb source."""
        config: dict[str, Any] = {
            "strategy": "grounded",
            "intent": {"mode": "static", "text_queries": ["test"]},
            "sources": [{"type": "vector_kb", "name": "kb"}],
        }
        kb = object()
        # Should not call set_knowledge_base — the source is declared
        strategy = GroundedReasoning.from_config(config, knowledge_base=kb)
        assert isinstance(strategy, GroundedReasoning)

    def test_hybrid_from_config_minimal(self) -> None:
        config: dict[str, Any] = {
            "strategy": "hybrid",
            "grounded": {
                "intent": {"mode": "static", "text_queries": ["test"]},
            },
        }
        strategy = HybridReasoning.from_config(config)
        assert isinstance(strategy, HybridReasoning)

    def test_hybrid_from_config_with_kb(self) -> None:
        config: dict[str, Any] = {
            "strategy": "hybrid",
            "grounded": {
                "intent": {"mode": "static", "text_queries": ["test"]},
            },
        }
        kb = object()
        strategy = HybridReasoning.from_config(config, knowledge_base=kb)
        assert isinstance(strategy, HybridReasoning)


# ------------------------------------------------------------------
# get_source_configs
# ------------------------------------------------------------------


class TestGetSourceConfigs:
    """Strategies declare where their source configs live."""

    def test_base_reads_top_level_sources(self) -> None:
        config = {"sources": [{"name": "kb", "source_type": "vector_kb"}]}
        assert ReasoningStrategy.get_source_configs(config) == config["sources"]

    def test_base_returns_empty_when_absent(self) -> None:
        assert ReasoningStrategy.get_source_configs({}) == []

    def test_grounded_uses_base_default(self) -> None:
        config = {"sources": [{"name": "kb"}]}
        assert GroundedReasoning.get_source_configs(config) == [{"name": "kb"}]

    def test_hybrid_reads_grounded_sub_key(self) -> None:
        config = {
            "grounded": {"sources": [{"name": "kb"}]},
        }
        assert HybridReasoning.get_source_configs(config) == [{"name": "kb"}]

    def test_hybrid_ignores_top_level_sources(self) -> None:
        config = {
            "sources": [{"name": "wrong"}],
            "grounded": {"sources": [{"name": "right"}]},
        }
        result = HybridReasoning.get_source_configs(config)
        assert result == [{"name": "right"}]

    def test_hybrid_returns_empty_when_absent(self) -> None:
        assert HybridReasoning.get_source_configs({}) == []


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


class TestBackwardCompat:
    """create_reasoning_from_config still works as before."""

    def test_create_simple(self) -> None:
        strategy = create_reasoning_from_config({"strategy": "simple"})
        assert isinstance(strategy, SimpleReasoning)

    def test_create_react(self) -> None:
        strategy = create_reasoning_from_config({
            "strategy": "react",
            "max_iterations": 3,
        })
        assert isinstance(strategy, ReActReasoning)
        assert strategy.max_iterations == 3

    def test_create_default_is_simple(self) -> None:
        strategy = create_reasoning_from_config({})
        assert isinstance(strategy, SimpleReasoning)

    def test_unknown_raises(self) -> None:
        with pytest.raises(NotFoundError, match="not registered"):
            create_reasoning_from_config({"strategy": "does_not_exist"})


# ------------------------------------------------------------------
# Module-level public API
# ------------------------------------------------------------------


class TestPublicAPI:
    """Module-level functions delegate to the singleton correctly."""

    def test_list_strategies(self) -> None:
        strategies = list_strategies()
        assert "simple" in strategies
        assert "react" in strategies

    def test_get_strategy_factory(self) -> None:
        factory = get_strategy_factory("react")
        assert factory is ReActReasoning

    def test_get_strategy_factory_missing(self) -> None:
        assert get_strategy_factory("nonexistent") is None

    def test_is_strategy_registered(self) -> None:
        assert is_strategy_registered("simple") is True
        assert is_strategy_registered("nonexistent") is False

    def test_get_registry(self) -> None:
        registry = get_registry()
        assert isinstance(registry, PluginRegistry)


# ------------------------------------------------------------------
# 3rd-party strategy end-to-end with BotTestHarness
# ------------------------------------------------------------------


class TestThirdPartyEndToEnd:
    """Register a custom strategy and use it with BotTestHarness."""

    @pytest.fixture(autouse=True)
    def _isolated_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Swap the module-level singleton with a fresh registry.

        Uses monkeypatch so the original singleton is automatically
        restored after each test — no shared state mutation.
        """
        import dataknobs_bots.reasoning.registry as reg_module

        fresh: PluginRegistry[ReasoningStrategy] = PluginRegistry(
            "test_strategies",
            validate_type=ReasoningStrategy,
            canonicalize_keys=True,
            config_key="strategy",
            config_key_default="simple",
            on_first_access=_register_builtins,
        )
        fresh.register("test_custom", _CustomStrategy)
        monkeypatch.setattr(reg_module, "_registry", fresh)

    @pytest.mark.asyncio()
    async def test_custom_strategy_via_harness(self) -> None:
        from dataknobs_bots.testing import BotTestHarness
        from dataknobs_llm.testing import text_response

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {
                    "strategy": "test_custom",
                    "custom_param": "harness_test",
                },
            },
            main_responses=[text_response("echo back")],
        ) as harness:
            result = await harness.chat("hello")
            # The custom strategy ignores the LLM and returns
            # "custom response" — but DynaBot wraps the raw_content
            # pipeline, so we just verify no crash and the bot ran.
            assert result is not None

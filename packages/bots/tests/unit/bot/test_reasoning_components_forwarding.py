"""Tests for ``DynaBot.from_config(reasoning_components=...)`` (161-D).

Bot-layer config-time forwarding of consumer-supplied collaborators
into the reasoning strategy's ``StructuredConfigConsumer.components``
channel.

Strategies that read their components from ``self.components`` (e.g.
``ReActReasoning``) pick up the forwarded keys; unknown keys are
silently absorbed by the mixin; bot-managed component names
(``knowledge_base``, ``prompt_resolver``, ``prompt_envelope``) collide
loudly with a ``ConfigurationError``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from dataknobs_bots import BotContext, DynaBot
from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.reasoning.registry import get_registry, register_strategy
from dataknobs_common.exceptions import ConfigurationError, NotFoundError
from dataknobs_common.structured_config import StructuredConfig, StructuredConfigConsumer

# =====================================================================
# Test fixtures: a minimal strategy that captures its components.
# =====================================================================


@dataclass(frozen=True)
class _CaptureStrategyConfig(StructuredConfig):
    """Empty config — the capture strategy has no scalar knobs."""


class _ComponentCaptureStrategy(
    StructuredConfigConsumer[_CaptureStrategyConfig], ReasoningStrategy
):
    """Captures the components mapping that ``_setup`` receives.

    Registered in the strategy registry under a unique test key so the
    bot's config-driven path can construct it.
    """

    CONFIG_CLS: ClassVar[type[_CaptureStrategyConfig]] = _CaptureStrategyConfig

    captured_components: dict[str, Any]

    def _setup(self) -> None:
        # Snapshot what the mixin handed us, so the test can inspect it.
        self.captured_components = dict(self.components)

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError


_TEST_STRATEGY_KEY = "_test_component_capture"


@pytest.fixture(scope="module", autouse=True)
def _register_capture_strategy() -> Iterator[None]:
    """Register the capture strategy for this module's tests and remove it
    on teardown so the process-global strategy registry doesn't accumulate
    test-only keys across runs."""
    register_strategy(
        _TEST_STRATEGY_KEY,
        _ComponentCaptureStrategy,
        override=True,
    )
    try:
        yield
    finally:
        # Already unregistered (e.g. a parallel test cleaned up) is fine.
        with contextlib.suppress(NotFoundError):
            get_registry().unregister(_TEST_STRATEGY_KEY)


# =====================================================================
# Forwarding lands on the strategy.
# =====================================================================


_BASE_CAPTURE_CONFIG: dict[str, Any] = {
    "llm": {"provider": "echo", "model": "test"},
    "conversation_storage": {"backend": "memory"},
    "reasoning": {"strategy": _TEST_STRATEGY_KEY},
}


class TestReasoningComponentsForwarding:
    """161-D coverage at the bot's config-time forwarding layer."""

    @pytest.mark.asyncio
    async def test_forwarding_lands_in_components(self) -> None:
        bot = await DynaBot.from_config(
            _BASE_CAPTURE_CONFIG,
            reasoning_components={"my_key": "my_value"},
        )
        try:
            strategy = bot.reasoning_strategy
            assert isinstance(strategy, _ComponentCaptureStrategy)
            assert strategy.captured_components["my_key"] == "my_value"
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_none_and_empty_dict_behave_identically(self) -> None:
        # ``None`` — default, no forwarding.
        bot_none = await DynaBot.from_config(_BASE_CAPTURE_CONFIG)
        try:
            strategy = bot_none.reasoning_strategy
            assert isinstance(strategy, _ComponentCaptureStrategy)
            assert "my_key" not in strategy.captured_components
        finally:
            await bot_none.close()

        # Empty dict — explicit no-forwarding.
        bot_empty = await DynaBot.from_config(
            _BASE_CAPTURE_CONFIG, reasoning_components={}
        )
        try:
            strategy = bot_empty.reasoning_strategy
            assert isinstance(strategy, _ComponentCaptureStrategy)
            assert "my_key" not in strategy.captured_components
        finally:
            await bot_empty.close()

    @pytest.mark.asyncio
    async def test_unknown_component_name_is_silently_absorbed(self) -> None:
        """SimpleReasoning ignores unknown components without raising —
        the mixin's signature-aware absorb contract."""
        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }
        bot = await DynaBot.from_config(
            config, reasoning_components={"nonsense_key": "x"}
        )
        try:
            # Construction succeeded; the unknown key was absorbed.
            assert bot.reasoning_strategy is not None
        finally:
            await bot.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "managed_name",
        ["knowledge_base", "prompt_resolver", "prompt_envelope"],
    )
    async def test_collision_with_bot_managed_name_raises(
        self, managed_name: str
    ) -> None:
        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }
        with pytest.raises(ConfigurationError) as exc_info:
            await DynaBot.from_config(
                config,
                reasoning_components={managed_name: object()},
            )
        assert managed_name in str(exc_info.value)


# =====================================================================
# End-to-end: ReAct ``extra_context`` flows through to a tool call.
# Migrated from ``tests/test_reasoning.py::test_extra_context_propagates_to_tools``
# (which previously reached for ``bot.reasoning_strategy._extra_context``).
# =====================================================================


class TestReActExtraContextEndToEnd:
    @pytest.mark.asyncio
    async def test_extra_context_propagates_to_tools(self) -> None:
        from dataknobs_llm.testing import text_response, tool_call_response
        from dataknobs_llm.tools import ContextAwareTool, ToolExecutionContext

        captured_contexts: list[ToolExecutionContext] = []

        class ContextCaptureTool(ContextAwareTool):
            def __init__(self) -> None:
                super().__init__(
                    name="capture",
                    description="Captures execution context",
                )

            @property
            def schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def execute_with_context(
                self, context: ToolExecutionContext, **kwargs: Any
            ) -> str:
                captured_contexts.append(context)
                return "captured"

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 3,
            },
        }

        bot = await DynaBot.from_config(
            config,
            reasoning_components={
                "extra_context": {"my_key": "my_value"},
            },
        )
        try:
            bot.tool_registry.register_tool(ContextCaptureTool())
            context = BotContext(
                conversation_id="conv-extra-ctx", client_id="test-client"
            )

            bot.llm.set_responses([
                tool_call_response("capture", {}),
                text_response("Done"),
            ])

            await bot.chat("Capture context", context)

            assert len(captured_contexts) == 1
            tool_ctx = captured_contexts[0]
            assert tool_ctx.extra.get("my_key") == "my_value"
        finally:
            await bot.close()

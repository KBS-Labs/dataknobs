"""Tests for template-based greetings on ReasoningStrategy.

Validates:
- SimpleReasoning renders greeting_template with initial_context
- ReActReasoning renders greeting_template with initial_context
- No template configured returns None
- DynaBot.greet() delegates template greetings correctly
- WizardReasoning resolves stage-level and strategy-level greetings in order
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning import (
    SimpleReasoning,
    ReActReasoning,
    WizardReasoning,
    create_reasoning_from_config,
)
from dataknobs_bots.reasoning.wizard_config import WizardReasoningConfig
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm import LLMResponse
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider


class TestSimpleReasoningGreeting:
    """Template-based greetings for SimpleReasoning."""

    @pytest.mark.asyncio
    async def test_greet_with_template(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """greeting_template renders and returns LLMResponse."""
        manager, _ = conversation_manager_pair
        strategy = SimpleReasoning(greeting_template="Welcome to the bot!")

        response = await strategy.greet(manager, llm=None)

        assert isinstance(response, LLMResponse)
        assert response.content == "Welcome to the bot!"
        assert response.model == "template"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_greet_with_initial_context(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """initial_context variables are available in the template."""
        manager, _ = conversation_manager_pair
        strategy = SimpleReasoning(greeting_template="Hello {{ user_name }}! Welcome to {{ app }}.")

        response = await strategy.greet(
            manager,
            llm=None,
            initial_context={"user_name": "Alice", "app": "DataKnobs"},
        )

        assert response is not None
        assert response.content == "Hello Alice! Welcome to DataKnobs."

    @pytest.mark.asyncio
    async def test_greet_without_template_returns_none(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """No greeting_template means greet() returns None."""
        manager, _ = conversation_manager_pair
        strategy = SimpleReasoning()

        response = await strategy.greet(manager, llm=None)

        assert response is None

    @pytest.mark.asyncio
    async def test_greet_with_empty_initial_context(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Template renders with empty context (undefined vars left empty)."""
        manager, _ = conversation_manager_pair
        strategy = SimpleReasoning(greeting_template="Hello {{ name }}!")

        response = await strategy.greet(manager, llm=None, initial_context={})

        assert response is not None
        # jinja2.Undefined renders as empty string
        assert response.content == "Hello !"


class TestReActReasoningGreeting:
    """Template-based greetings for ReActReasoning."""

    @pytest.mark.asyncio
    async def test_greet_with_template(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """ReActReasoning also supports greeting_template."""
        manager, _ = conversation_manager_pair
        strategy = ReActReasoning(greeting_template="Hello {{ user }}! I can help with tools.")

        response = await strategy.greet(
            manager,
            llm=None,
            initial_context={"user": "Bob"},
        )

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello Bob! I can help with tools."


class TestCreateReasoningFromConfig:
    """greeting_template flows through config factory."""

    def test_simple_with_greeting_template(self) -> None:
        """Config-created SimpleReasoning has greeting_template."""
        config: dict[str, Any] = {
            "strategy": "simple",
            "greeting_template": "Hello {{ name }}!",
        }
        strategy = create_reasoning_from_config(config)

        assert isinstance(strategy, SimpleReasoning)
        assert strategy.greeting_template == "Hello {{ name }}!"

    def test_react_with_greeting_template(self) -> None:
        """Config-created ReActReasoning has greeting_template."""
        config: dict[str, Any] = {
            "strategy": "react",
            "greeting_template": "Hi there!",
        }
        strategy = create_reasoning_from_config(config)

        assert isinstance(strategy, ReActReasoning)
        assert strategy.greeting_template == "Hi there!"

    def test_simple_without_greeting_template(self) -> None:
        """Config without greeting_template creates strategy with None."""
        config: dict[str, Any] = {"strategy": "simple"}
        strategy = create_reasoning_from_config(config)

        assert isinstance(strategy, SimpleReasoning)
        assert strategy.greeting_template is None


def _wizard_config(
    *,
    stage_greeting: str | None = None,
    response_template: str | None = None,
) -> dict[str, Any]:
    """A two-stage wizard whose start stage carries the given templates."""
    start: dict[str, Any] = {
        "name": "start",
        "is_start": True,
        "prompt": "Ask user something",
        "transitions": [{"target": "done"}],
    }
    if stage_greeting:
        start["greeting_template"] = stage_greeting
    if response_template:
        start["response_template"] = response_template
    return {
        "name": "greeting-test",
        "version": "1.0",
        "stages": [start, {"name": "done", "is_end": True, "prompt": "Done"}],
    }


#: ``(stage greeting, strategy greeting, response_template, expected)``.
#:
#: Row 3 is the contract this file used to pin as *"WizardReasoning ignores
#: base template"* — a start stage with a ``response_template`` and nothing
#: else still greets with it, which is why that row is unchanged. Rows 2 and 4
#: are what supersedes the old one: the strategy-level field is now the start
#: stage's default rather than a discarded value.
_PRECEDENCE_TABLE: list[tuple[str | None, str | None, str | None, str]] = [
    ("STAGE", "STRATEGY", "RESPONSE", "STAGE"),
    (None, "STRATEGY", "RESPONSE", "STRATEGY"),
    (None, None, "RESPONSE", "RESPONSE"),
    (None, "STRATEGY", None, "STRATEGY"),
]


class TestWizardGreetingPrecedence:
    """A wizard resolves its greeting from the stage, then the strategy.

    ``ReasoningStrategy`` documents ``greeting_template`` as universal, and
    ``WizardReasoning`` overrides ``greet()`` — so until the stage-level field
    existed there was nowhere for the strategy-level one to land, and it was
    discarded. Now the two compose: the stage field is the mechanism, and the
    strategy field is the start stage's default.
    """

    @pytest.mark.parametrize("case", _PRECEDENCE_TABLE)
    @pytest.mark.asyncio
    async def test_greeting_precedence(
        self,
        case: tuple[str | None, str | None, str | None, str],
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """The first template that applies wins, and only that one renders."""
        stage_greeting, strategy_greeting, response_template, expected = case
        manager, provider = conversation_manager_pair
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(
            _wizard_config(
                stage_greeting=stage_greeting,
                response_template=response_template,
            )
        )
        kwargs: dict[str, Any] = {}
        if strategy_greeting:
            kwargs["greeting_template"] = strategy_greeting
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False, **kwargs)

        response = await reasoning.greet(manager, llm=provider)

        assert response is not None
        assert response.content == expected

    @pytest.mark.asyncio
    async def test_the_strategy_greeting_renders_with_initial_context(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """It is a Jinja2 template at stage scope, as it is at strategy scope."""
        manager, provider = conversation_manager_pair
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(_wizard_config())
        reasoning = WizardReasoning(
            wizard_fsm=fsm,
            strict_validation=False,
            greeting_template="Hello {{ user_name }}!",
        )

        response = await reasoning.greet(
            manager, llm=provider, initial_context={"user_name": "Alice"}
        )

        assert response is not None
        assert response.content == "Hello Alice!"

    @pytest.mark.asyncio
    async def test_the_strategy_greeting_is_said_once(self) -> None:
        """It goes through the stage's greeting count, so it is stepped over.

        A greeting bolted onto ``greet()`` alone would be re-rendered the next
        time the start stage produced output. Routing the strategy-level
        default through the same per-stage counter is what makes "once" mean
        once, rather than "once per call to greet()".
        """
        bot_config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": _wizard_config(),
                "greeting_template": "STRATEGY GREETING",
                "strict_validation": False,
            },
        }
        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=["LLM-1", "LLM-2"],
        ) as harness:
            opening = await harness.greet()
            assert opening.response == "STRATEGY GREETING"

            answer = await harness.chat("something")
            assert "STRATEGY GREETING" not in answer.response

    @pytest.mark.asyncio
    async def test_the_greeting_travels_the_ordinary_stage_path(self) -> None:
        """It is the start stage's greeting, not a shortcut around the stage.

        A greeting rendered and returned directly from ``greet()`` produces
        the same opening line, and every test above still passes against
        that version — the difference only shows when the start stage does
        something *after* speaking. Auto-advance is that case: the wizard
        must greet, step through the message stage, and land on the next
        stage in the same turn.
        """
        wizard: dict[str, Any] = {
            "name": "greeting-auto-advance",
            "version": "1.0",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "auto_advance": True,
                    "prompt": "Say hello",
                    "transitions": [{"target": "collect"}],
                },
                {
                    "name": "collect",
                    "prompt": "Ask for a name",
                    "response_template": "COLLECT",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True, "prompt": "Done"},
            ],
        }
        bot_config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": wizard,
                "greeting_template": "STRATEGY",
                "strict_validation": False,
            },
        }
        async with await BotTestHarness.create(
            bot_config=bot_config,
            main_responses=["LLM-1"],
        ) as harness:
            opening = await harness.greet()

            assert "STRATEGY" in opening.response
            assert "COLLECT" in opening.response
            assert opening.response.index("STRATEGY") < opening.response.index("COLLECT")
            assert harness.wizard_stage == "collect"

    @pytest.mark.asyncio
    async def test_wizard_reasoning_config_carries_the_field(self) -> None:
        """``from_dict`` used to project the key away without complaint."""
        config = WizardReasoningConfig.from_dict(
            {"wizard_config": "some/path.yaml", "greeting_template": "Hi!"}
        )

        assert config.greeting_template == "Hi!"


class TestDynaBotGreetingIntegration:
    """DynaBot.greet() works with template-based strategies."""

    @pytest.mark.asyncio
    async def test_dynabot_greet_with_simple_template(self) -> None:
        """DynaBot.greet() delegates to SimpleReasoning template greeting."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "simple",
                "greeting_template": "Hello {{ name }}! How can I help?",
            },
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="test-greeting-1",
                client_id="test",
            )

            result = await bot.greet(
                context,
                initial_context={"name": "Alice"},
            )

            assert result == "Hello Alice! How can I help?"
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_dynabot_greet_without_template_returns_none(self) -> None:
        """DynaBot.greet() returns None when no template configured."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="test-greeting-2",
                client_id="test",
            )

            result = await bot.greet(context)

            assert result is None
        finally:
            await bot.close()

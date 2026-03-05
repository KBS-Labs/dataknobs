"""Tests for template-based greetings on ReasoningStrategy.

Validates:
- SimpleReasoning renders greeting_template with initial_context
- ReActReasoning renders greeting_template with initial_context
- No template configured returns None
- DynaBot.greet() delegates template greetings correctly
- WizardReasoning ignores base template (uses FSM-driven greetings)
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
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
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
        manager, provider = conversation_manager_pair
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
        manager, provider = conversation_manager_pair
        strategy = SimpleReasoning(
            greeting_template="Hello {{ user_name }}! Welcome to {{ app }}."
        )

        response = await strategy.greet(
            manager, llm=None,
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
        manager, provider = conversation_manager_pair
        strategy = SimpleReasoning()

        response = await strategy.greet(manager, llm=None)

        assert response is None

    @pytest.mark.asyncio
    async def test_greet_with_empty_initial_context(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Template renders with empty context (undefined vars left empty)."""
        manager, provider = conversation_manager_pair
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
        manager, provider = conversation_manager_pair
        strategy = ReActReasoning(
            greeting_template="Hello {{ user }}! I can help with tools."
        )

        response = await strategy.greet(
            manager, llm=None,
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
        assert strategy._greeting_template == "Hello {{ name }}!"

    def test_react_with_greeting_template(self) -> None:
        """Config-created ReActReasoning has greeting_template."""
        config: dict[str, Any] = {
            "strategy": "react",
            "greeting_template": "Hi there!",
        }
        strategy = create_reasoning_from_config(config)

        assert isinstance(strategy, ReActReasoning)
        assert strategy._greeting_template == "Hi there!"

    def test_simple_without_greeting_template(self) -> None:
        """Config without greeting_template creates strategy with None."""
        config: dict[str, Any] = {"strategy": "simple"}
        strategy = create_reasoning_from_config(config)

        assert isinstance(strategy, SimpleReasoning)
        assert strategy._greeting_template is None


class TestWizardGreetingIgnoresBase:
    """WizardReasoning uses FSM greetings, not base template."""

    @pytest.mark.asyncio
    async def test_wizard_uses_fsm_not_base_template(
        self,
        conversation_manager: ConversationManager,
    ) -> None:
        """WizardReasoning.greet() generates FSM start stage response."""
        config: dict[str, Any] = {
            "name": "greeting-test",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Ask user something",
                    "response_template": "Welcome to the wizard!",
                    "transitions": [{"target": "done"}],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        response = await reasoning.greet(conversation_manager, llm=None)

        # Wizard uses its own FSM-driven greeting, not base template
        assert response is not None
        assert response.content == "Welcome to the wizard!"


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

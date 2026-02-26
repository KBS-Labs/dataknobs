"""Tests for bot-first greeting in wizard scenarios.

Validates:
- ReasoningStrategy.greet() default returns None
- WizardReasoning.greet() initializes state and generates a response
- DynaBot.greet() delegates correctly to the reasoning strategy
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.simple import SimpleReasoning
from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def greeting_wizard_config() -> dict[str, Any]:
    """Wizard config with a response_template on the start stage."""
    return {
        "name": "greeting-wizard",
        "version": "1.0",
        "stages": [
            {
                "name": "greeting",
                "is_start": True,
                "prompt": "Greet the user and ask for their name",
                "response_template": "Hello! Welcome to the wizard. What is your name?",
                "schema": {
                    "type": "object",
                    "properties": {"user_name": {"type": "string"}},
                    "required": ["user_name"],
                },
                "transitions": [
                    {"target": "done", "condition": "data.get('user_name')"},
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "All done!",
            },
        ],
    }


@pytest.fixture
def greeting_wizard_reasoning(
    greeting_wizard_config: dict[str, Any],
) -> WizardReasoning:
    """WizardReasoning with a template-driven greeting stage."""
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(greeting_wizard_config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


# ---------------------------------------------------------------------------
# TestWizardReasoningGreet
# ---------------------------------------------------------------------------

class TestWizardReasoningGreet:
    """Tests for WizardReasoning.greet() directly."""

    @pytest.mark.asyncio
    async def test_greet_initializes_wizard_state(
        self,
        greeting_wizard_reasoning: WizardReasoning,
        conversation_manager: ConversationManager,
    ) -> None:
        """greet() creates wizard state at the start stage."""
        await greeting_wizard_reasoning.greet(conversation_manager, llm=None)

        wizard_meta = conversation_manager.metadata.get("wizard", {})
        assert wizard_meta.get("current_stage") == "greeting"
        fsm_state = wizard_meta.get("fsm_state", {})
        assert fsm_state.get("current_stage") == "greeting"
        assert "greeting" in fsm_state.get("history", [])

    @pytest.mark.asyncio
    async def test_greet_returns_response(
        self,
        greeting_wizard_reasoning: WizardReasoning,
        conversation_manager: ConversationManager,
    ) -> None:
        """greet() returns a response object with content."""
        response = await greeting_wizard_reasoning.greet(conversation_manager, llm=None)

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_greet_with_response_template(
        self,
        greeting_wizard_reasoning: WizardReasoning,
        conversation_manager: ConversationManager,
    ) -> None:
        """greet() renders the start stage's response_template."""
        response = await greeting_wizard_reasoning.greet(conversation_manager, llm=None)

        assert response.content == "Hello! Welcome to the wizard. What is your name?"

    @pytest.mark.asyncio
    async def test_greet_saves_wizard_state(
        self,
        greeting_wizard_reasoning: WizardReasoning,
        conversation_manager: ConversationManager,
    ) -> None:
        """greet() persists wizard state to manager.metadata."""
        await greeting_wizard_reasoning.greet(conversation_manager, llm=None)

        assert "wizard" in conversation_manager.metadata
        assert "fsm_state" in conversation_manager.metadata["wizard"]

    @pytest.mark.asyncio
    async def test_greet_adds_assistant_message(
        self,
        greeting_wizard_reasoning: WizardReasoning,
        conversation_manager: ConversationManager,
    ) -> None:
        """greet() adds an assistant message (no user message) to history."""
        await greeting_wizard_reasoning.greet(conversation_manager, llm=None)

        messages = conversation_manager.get_messages()
        # Real CM includes system message as first message
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "Welcome to the wizard" in assistant_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_greet_without_template_uses_llm(
        self,
        wizard_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """greet() on a stage without response_template uses LLM."""
        manager, provider = conversation_manager_pair
        provider.set_responses(["Welcome! What would you like?"])

        response = await wizard_reasoning.greet(manager, llm=None)

        assert response is not None
        assert response.content == "Welcome! What would you like?"


# ---------------------------------------------------------------------------
# TestReasoningStrategyGreetDefault
# ---------------------------------------------------------------------------

class TestReasoningStrategyGreetDefault:
    """Tests that non-wizard strategies return None from greet()."""

    @pytest.mark.asyncio
    async def test_simple_reasoning_greet_returns_none(
        self, conversation_manager: ConversationManager
    ) -> None:
        """SimpleReasoning inherits the base default (returns None)."""
        strategy = SimpleReasoning()

        result = await strategy.greet(conversation_manager, llm=None)

        assert result is None


# ---------------------------------------------------------------------------
# TestDynaBotGreet (integration)
# ---------------------------------------------------------------------------

class TestDynaBotGreet:
    """Integration tests using DynaBot.from_config()."""

    @pytest.mark.asyncio
    async def test_greet_wizard_bot(self) -> None:
        """Full integration: wizard bot greet() returns template response."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": {
                    "name": "greet-test",
                    "stages": [
                        {
                            "name": "welcome",
                            "is_start": True,
                            "prompt": "Welcome the user",
                            "response_template": "Hello! Welcome to the wizard.",
                            "transitions": [{"target": "done"}],
                        },
                        {
                            "name": "done",
                            "is_end": True,
                            "prompt": "Done",
                        },
                    ],
                },
            },
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="test-greet-1",
                client_id="test",
            )

            greeting = await bot.greet(context)

            assert greeting is not None
            assert greeting == "Hello! Welcome to the wizard."
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_greet_non_wizard_bot_returns_none(self) -> None:
        """Non-wizard bot (simple strategy) returns None from greet()."""
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
                conversation_id="test-greet-2",
                client_id="test",
            )

            greeting = await bot.greet(context)

            assert greeting is None
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_greet_bot_without_strategy_returns_none(self) -> None:
        """Bot without reasoning strategy returns None from greet()."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="test-greet-3",
                client_id="test",
            )

            greeting = await bot.greet(context)

            assert greeting is None
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_greet_sets_up_conversation(self) -> None:
        """greet() creates a conversation with the greeting in history."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": {
                    "name": "greet-conv-test",
                    "stages": [
                        {
                            "name": "welcome",
                            "is_start": True,
                            "prompt": "Welcome the user",
                            "response_template": "Hi there!",
                            "transitions": [{"target": "done"}],
                        },
                        {
                            "name": "done",
                            "is_end": True,
                            "prompt": "Done",
                        },
                    ],
                },
            },
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="test-greet-conv",
                client_id="test",
            )

            await bot.greet(context)

            # Verify conversation exists and has wizard state
            wizard_state = await bot.get_wizard_state("test-greet-conv")
            assert wizard_state is not None
            assert wizard_state["current_stage"] == "welcome"
        finally:
            await bot.close()

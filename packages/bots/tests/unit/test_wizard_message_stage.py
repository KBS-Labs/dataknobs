"""Tests for wizard message stages (auto_advance without schema).

A "message stage" is a stage with:
- response_template (the message to display)
- auto_advance: true
- No schema (no data collection)
- At least one transition

This pattern enables informational stages that display a message and
auto-advance to the next stage without waiting for user input.

The implementation extends the existing auto_advance mechanism:
1. _can_auto_advance allows schema-less stages with explicit auto_advance: true
2. The auto-advance loop renders templates from intermediate message stages
3. The rendered messages are prepended to the final stage's response
4. greet() also applies auto-advance logic for message start stages
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def message_stage_config() -> dict[str, Any]:
    """Wizard with a message stage between two data stages."""
    return {
        "name": "message-stage-wizard",
        "stages": [
            {
                "name": "collect_name",
                "is_start": True,
                "prompt": "What is your name?",
                "schema": {
                    "type": "object",
                    "properties": {"user_name": {"type": "string"}},
                    "required": ["user_name"],
                },
                "transitions": [
                    {"target": "confirmation", "condition": "data.get('user_name')"},
                ],
            },
            {
                "name": "confirmation",
                "response_template": "Thanks {{ user_name }}! Let's continue.",
                "auto_advance": True,
                "transitions": [
                    {"target": "collect_email", "condition": "true"},
                ],
            },
            {
                "name": "collect_email",
                "prompt": "What is your email?",
                "schema": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                    "required": ["email"],
                },
                "transitions": [
                    {"target": "done", "condition": "data.get('email')"},
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "response_template": "All done!",
            },
        ],
    }


@pytest.fixture
def greeting_message_config() -> dict[str, Any]:
    """Wizard where the start stage is a message stage that auto-advances."""
    return {
        "name": "greeting-message-wizard",
        "stages": [
            {
                "name": "welcome",
                "is_start": True,
                "response_template": "Welcome to the onboarding wizard!",
                "auto_advance": True,
                "transitions": [
                    {"target": "collect_name", "condition": "true"},
                ],
            },
            {
                "name": "collect_name",
                "prompt": "What is your name?",
                "response_template": "Please tell me your name.",
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
                "response_template": "All done!",
            },
        ],
    }


@pytest.fixture
def chained_message_config() -> dict[str, Any]:
    """Wizard with two consecutive message stages."""
    return {
        "name": "chained-message-wizard",
        "stages": [
            {
                "name": "collect_name",
                "is_start": True,
                "prompt": "What is your name?",
                "schema": {
                    "type": "object",
                    "properties": {"user_name": {"type": "string"}},
                    "required": ["user_name"],
                },
                "transitions": [
                    {"target": "ack", "condition": "data.get('user_name')"},
                ],
            },
            {
                "name": "ack",
                "response_template": "Got it, {{ user_name }}.",
                "auto_advance": True,
                "transitions": [
                    {"target": "info", "condition": "true"},
                ],
            },
            {
                "name": "info",
                "response_template": "Now let's set up your profile.",
                "auto_advance": True,
                "transitions": [
                    {"target": "collect_email", "condition": "true"},
                ],
            },
            {
                "name": "collect_email",
                "prompt": "What is your email?",
                "schema": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                    "required": ["email"],
                },
                "transitions": [
                    {"target": "done", "condition": "data.get('email')"},
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "response_template": "All done!",
            },
        ],
    }


@pytest.fixture
def conditional_message_config() -> dict[str, Any]:
    """Wizard with a message stage whose transition depends on collected data."""
    return {
        "name": "conditional-message-wizard",
        "stages": [
            {
                "name": "collect_role",
                "is_start": True,
                "prompt": "What is your role?",
                "schema": {
                    "type": "object",
                    "properties": {"role": {"type": "string"}},
                    "required": ["role"],
                },
                "transitions": [
                    {"target": "routing", "condition": "data.get('role')"},
                ],
            },
            {
                "name": "routing",
                "response_template": "Setting up your {{ role }} workspace...",
                "auto_advance": True,
                "transitions": [
                    {
                        "target": "admin_setup",
                        "condition": "data.get('role') == 'admin'",
                        "priority": 0,
                    },
                    {
                        "target": "user_setup",
                        "condition": "true",
                        "priority": 1,
                    },
                ],
            },
            {
                "name": "admin_setup",
                "is_end": True,
                "response_template": "Admin workspace ready!",
            },
            {
                "name": "user_setup",
                "is_end": True,
                "response_template": "User workspace ready!",
            },
        ],
    }


# ---------------------------------------------------------------------------
# TestCanAutoAdvanceMessageStage — unit tests for _can_auto_advance
# ---------------------------------------------------------------------------

class TestCanAutoAdvanceMessageStage:
    """Tests that _can_auto_advance correctly handles schema-less stages."""

    def test_message_stage_with_explicit_auto_advance(self) -> None:
        """A schema-less stage with auto_advance: true CAN auto-advance."""
        config = {
            "name": "test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                    "transitions": [{"target": "msg", "condition": "data.get('x')"}],
                },
                {
                    "name": "msg",
                    "response_template": "Hello!",
                    "auto_advance": True,
                    "transitions": [{"target": "end", "condition": "true"}],
                },
                {"name": "end", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm)

        # Advance FSM to the message stage
        fsm.step({"x": "value"})
        assert fsm.current_stage == "msg"

        state = WizardState(current_stage="msg", data={"x": "value"})
        stage = fsm.current_metadata

        assert reasoning._can_auto_advance(state, stage) is True

    def test_schema_less_stage_without_auto_advance_cannot_advance(self) -> None:
        """A schema-less stage WITHOUT auto_advance: true cannot auto-advance."""
        config = {
            "name": "test",
            "stages": [
                {
                    "name": "msg",
                    "is_start": True,
                    "response_template": "Hello!",
                    # No auto_advance
                    "transitions": [{"target": "end", "condition": "true"}],
                },
                {"name": "end", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm)
        state = WizardState(current_stage="msg", data={})
        stage = fsm.current_metadata

        assert reasoning._can_auto_advance(state, stage) is False

    def test_global_auto_advance_does_not_affect_schema_less_stages(self) -> None:
        """Global auto_advance_filled_stages alone does NOT enable schema-less advance.

        The global setting means "skip stages whose required fields are filled."
        A stage with no fields has no fields to check — it's a different concept
        from "always advance this stage." Only explicit per-stage auto_advance
        should enable schema-less advancement.
        """
        config = {
            "name": "test",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "msg",
                    "is_start": True,
                    "response_template": "Hello!",
                    # No auto_advance, no schema
                    "transitions": [{"target": "end", "condition": "true"}],
                },
                {"name": "end", "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(
            wizard_fsm=fsm, auto_advance_filled_stages=True
        )
        state = WizardState(current_stage="msg", data={})
        stage = fsm.current_metadata

        assert reasoning._can_auto_advance(state, stage) is False

    def test_message_stage_end_stage_cannot_auto_advance(self) -> None:
        """End stages cannot auto-advance even with auto_advance: true."""
        config = {
            "name": "test",
            "stages": [
                {
                    "name": "end",
                    "is_start": True,
                    "is_end": True,
                    "response_template": "Done!",
                    "auto_advance": True,
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm)
        state = WizardState(current_stage="end", data={})
        stage = fsm.current_metadata

        assert reasoning._can_auto_advance(state, stage) is False


# ---------------------------------------------------------------------------
# TestMessageStageGenerate — integration tests via generate()
# ---------------------------------------------------------------------------

class TestMessageStageGenerate:
    """Tests that generate() auto-advances through message stages.

    These tests use auto_advance_filled_stages + pre-seeded data to bypass
    the data-collection stage, focusing exclusively on verifying message
    stage auto-advance and template rendering behavior.
    """

    @pytest.mark.asyncio
    async def test_message_stage_auto_advances_with_template_in_response(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After transition to a message stage, the response includes its
        template AND the wizard lands on the next data-collection stage."""
        manager, provider = conversation_manager_pair

        config: dict[str, Any] = {
            "name": "msg-advance-test",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "collect_name",
                    "is_start": True,
                    "prompt": "What is your name?",
                    "schema": {
                        "type": "object",
                        "properties": {"user_name": {"type": "string"}},
                        "required": ["user_name"],
                    },
                    "transitions": [
                        {"target": "confirmation", "condition": "data.get('user_name')"},
                    ],
                },
                {
                    "name": "confirmation",
                    "response_template": "Thanks {{ user_name }}! Let's continue.",
                    "auto_advance": True,
                    "transitions": [
                        {"target": "collect_email", "condition": "true"},
                    ],
                },
                {
                    "name": "collect_email",
                    "prompt": "What is your email?",
                    "response_template": "Please enter your email.",
                    "schema": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                        "required": ["email"],
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('email')"},
                    ],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "response_template": "All done!",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(
            wizard_fsm=fsm,
            strict_validation=False,
            auto_advance_filled_stages=True,
        )

        # Pre-seed wizard state with user_name so collect_name auto-advances
        # via global auto_advance_filled_stages, then confirmation should
        # auto-advance via its per-stage auto_advance: true.
        from tests.unit.conftest import set_wizard_state

        set_wizard_state(manager, {
            "current_stage": "collect_name",
            "data": {"user_name": "Alice"},
            "history": ["collect_name"],
        })

        # User says anything — extraction doesn't matter because the stage
        # will auto-advance with pre-filled data.
        provider.set_responses(["ok"])
        await manager.add_message(role="user", content="Alice")

        response = await reasoning.generate(manager, llm=None)

        # The response should include the confirmation message template
        assert "Thanks Alice! Let's continue." in response.content

        # Wizard should have landed on collect_email, not confirmation
        wizard_meta = manager.metadata.get("wizard", {})
        current_stage = wizard_meta.get("current_stage")
        assert current_stage == "collect_email"

    @pytest.mark.asyncio
    async def test_chained_message_stages(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Two consecutive message stages both render and auto-advance."""
        manager, provider = conversation_manager_pair

        config: dict[str, Any] = {
            "name": "chained-msg-test",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "collect_name",
                    "is_start": True,
                    "prompt": "What is your name?",
                    "schema": {
                        "type": "object",
                        "properties": {"user_name": {"type": "string"}},
                        "required": ["user_name"],
                    },
                    "transitions": [
                        {"target": "ack", "condition": "data.get('user_name')"},
                    ],
                },
                {
                    "name": "ack",
                    "response_template": "Got it, {{ user_name }}.",
                    "auto_advance": True,
                    "transitions": [
                        {"target": "info", "condition": "true"},
                    ],
                },
                {
                    "name": "info",
                    "response_template": "Now let's set up your profile.",
                    "auto_advance": True,
                    "transitions": [
                        {"target": "collect_email", "condition": "true"},
                    ],
                },
                {
                    "name": "collect_email",
                    "prompt": "What is your email?",
                    "response_template": "Please enter your email.",
                    "schema": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                        "required": ["email"],
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('email')"},
                    ],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "response_template": "All done!",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(
            wizard_fsm=fsm,
            strict_validation=False,
            auto_advance_filled_stages=True,
        )

        from tests.unit.conftest import set_wizard_state

        set_wizard_state(manager, {
            "current_stage": "collect_name",
            "data": {"user_name": "Bob"},
            "history": ["collect_name"],
        })

        provider.set_responses(["ok"])
        await manager.add_message(role="user", content="Bob")

        response = await reasoning.generate(manager, llm=None)

        # Both message stage templates should appear
        assert "Got it, Bob." in response.content
        assert "Now let's set up your profile." in response.content

        # Should land on collect_email
        wizard_meta = manager.metadata.get("wizard", {})
        assert wizard_meta.get("current_stage") == "collect_email"

    @pytest.mark.asyncio
    async def test_conditional_message_stage_routes_correctly(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Message stage evaluates conditions based on collected data."""
        manager, provider = conversation_manager_pair

        config: dict[str, Any] = {
            "name": "conditional-msg-test",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "collect_role",
                    "is_start": True,
                    "prompt": "What is your role?",
                    "schema": {
                        "type": "object",
                        "properties": {"role": {"type": "string"}},
                        "required": ["role"],
                    },
                    "transitions": [
                        {"target": "routing", "condition": "data.get('role')"},
                    ],
                },
                {
                    "name": "routing",
                    "response_template": "Setting up your {{ role }} workspace...",
                    "auto_advance": True,
                    "transitions": [
                        {
                            "target": "admin_setup",
                            "condition": "data.get('role') == 'admin'",
                            "priority": 0,
                        },
                        {
                            "target": "user_setup",
                            "condition": "true",
                            "priority": 1,
                        },
                    ],
                },
                {
                    "name": "admin_setup",
                    "is_end": True,
                    "response_template": "Admin workspace ready!",
                },
                {
                    "name": "user_setup",
                    "is_end": True,
                    "response_template": "User workspace ready!",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(
            wizard_fsm=fsm,
            strict_validation=False,
            auto_advance_filled_stages=True,
        )

        from tests.unit.conftest import set_wizard_state

        set_wizard_state(manager, {
            "current_stage": "collect_role",
            "data": {"role": "admin"},
            "history": ["collect_role"],
        })

        provider.set_responses(["ok"])
        await manager.add_message(role="user", content="admin")

        response = await reasoning.generate(manager, llm=None)

        # Should see the routing message
        assert "Setting up your admin workspace" in response.content

        # Should land on admin_setup (not user_setup)
        wizard_meta = manager.metadata.get("wizard", {})
        assert wizard_meta.get("current_stage") == "admin_setup"

    @pytest.mark.asyncio
    async def test_message_stage_records_transition(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Auto-advance through a message stage records transition with
        trigger='auto_advance'."""
        manager, provider = conversation_manager_pair

        config: dict[str, Any] = {
            "name": "transition-record-test",
            "settings": {"auto_advance_filled_stages": True},
            "stages": [
                {
                    "name": "collect_name",
                    "is_start": True,
                    "prompt": "What is your name?",
                    "schema": {
                        "type": "object",
                        "properties": {"user_name": {"type": "string"}},
                        "required": ["user_name"],
                    },
                    "transitions": [
                        {"target": "confirmation", "condition": "data.get('user_name')"},
                    ],
                },
                {
                    "name": "confirmation",
                    "response_template": "Thanks {{ user_name }}!",
                    "auto_advance": True,
                    "transitions": [
                        {"target": "collect_email", "condition": "true"},
                    ],
                },
                {
                    "name": "collect_email",
                    "prompt": "What is your email?",
                    "response_template": "Please enter your email.",
                    "schema": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                        "required": ["email"],
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('email')"},
                    ],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "response_template": "All done!",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(
            wizard_fsm=fsm,
            strict_validation=False,
            auto_advance_filled_stages=True,
        )

        from tests.unit.conftest import set_wizard_state

        set_wizard_state(manager, {
            "current_stage": "collect_name",
            "data": {"user_name": "Alice"},
            "history": ["collect_name"],
        })

        provider.set_responses(["ok"])
        await manager.add_message(role="user", content="Alice")

        await reasoning.generate(manager, llm=None)

        # Check transitions in wizard state
        wizard_meta = manager.metadata.get("wizard", {})
        fsm_state = wizard_meta.get("fsm_state", {})
        transitions = fsm_state.get("transitions", [])

        # Should have: collect_name -> confirmation (auto_advance from global)
        #              confirmation -> collect_email (auto_advance from per-stage)
        auto_advance_transitions = [
            t for t in transitions if t.get("trigger") == "auto_advance"
        ]
        assert len(auto_advance_transitions) >= 1
        # At least one auto-advance transition should be from the message stage
        msg_transitions = [
            t for t in auto_advance_transitions
            if t["from_stage"] == "confirmation"
        ]
        assert len(msg_transitions) == 1
        assert msg_transitions[0]["to_stage"] == "collect_email"


# ---------------------------------------------------------------------------
# TestMessageStageGreet — greet() with message start stages
# ---------------------------------------------------------------------------

class TestMessageStageGreet:
    """Tests for greet() when the start stage is a message stage."""

    @pytest.mark.asyncio
    async def test_greet_auto_advances_through_message_start_stage(
        self,
        greeting_message_config: dict[str, Any],
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """greet() on a message start stage renders its template and
        auto-advances to the next stage."""
        manager, _provider = conversation_manager_pair
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(greeting_message_config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        response = await reasoning.greet(manager, llm=None)

        # Should include the welcome message
        assert response is not None
        assert "Welcome to the onboarding wizard!" in response.content

        # Should have auto-advanced to collect_name
        wizard_meta = manager.metadata.get("wizard", {})
        fsm_state = wizard_meta.get("fsm_state", {})
        assert fsm_state.get("current_stage") == "collect_name"

    @pytest.mark.asyncio
    async def test_greet_message_stage_includes_next_stage_prompt(
        self,
        greeting_message_config: dict[str, Any],
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """greet() renders both the message stage template and the
        landing stage's template (if any)."""
        manager, _provider = conversation_manager_pair
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(greeting_message_config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        response = await reasoning.greet(manager, llm=None)

        # Should include both the welcome message and the next stage's prompt
        assert "Welcome to the onboarding wizard!" in response.content
        assert "Please tell me your name." in response.content


# ---------------------------------------------------------------------------
# TestMessageStageDynaBot — full integration via DynaBot
# ---------------------------------------------------------------------------

class TestMessageStageDynaBot:
    """Integration tests using DynaBot for message stages."""

    @pytest.mark.asyncio
    async def test_chat_greet_auto_advances_through_message_stage(self) -> None:
        """DynaBot.greet() with initial_context auto-advances through
        a data stage AND a message stage in one call.

        Uses auto_advance_filled_stages + initial_context to pre-seed data
        so the collect stage auto-advances, then the message stage also
        auto-advances, exercising the full chain via DynaBot.
        """
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": {
                    "name": "msg-test",
                    "settings": {"auto_advance_filled_stages": True},
                    "stages": [
                        {
                            "name": "collect",
                            "is_start": True,
                            "prompt": "What is your name?",
                            "response_template": "Please provide your name.",
                            "schema": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            },
                            "transitions": [
                                {
                                    "target": "thanks",
                                    "condition": "data.get('name')",
                                },
                            ],
                        },
                        {
                            "name": "thanks",
                            "response_template": "Thanks {{ name }}!",
                            "auto_advance": True,
                            "transitions": [
                                {"target": "done", "condition": "true"},
                            ],
                        },
                        {
                            "name": "done",
                            "is_end": True,
                            "response_template": "All done!",
                        },
                    ],
                },
            },
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="msg-test-1",
                client_id="test",
            )

            # Pre-seed name via initial_context — collect auto-advances
            # (filled fields), then thanks auto-advances (message stage).
            greeting = await bot.greet(
                context, initial_context={"name": "Alice"}
            )
            assert greeting is not None
            assert "Thanks Alice!" in greeting

            # Should land on "done"
            wizard_state = await bot.get_wizard_state("msg-test-1")
            assert wizard_state is not None
            assert wizard_state.get("current_stage") == "done"
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_greet_with_message_start_stage(self) -> None:
        """DynaBot.greet() auto-advances through a message start stage."""
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": {
                    "name": "greet-msg-test",
                    "stages": [
                        {
                            "name": "welcome",
                            "is_start": True,
                            "response_template": "Welcome aboard!",
                            "auto_advance": True,
                            "transitions": [
                                {"target": "collect", "condition": "true"},
                            ],
                        },
                        {
                            "name": "collect",
                            "prompt": "What is your name?",
                            "response_template": "Please enter your name.",
                            "schema": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            },
                            "transitions": [
                                {"target": "done", "condition": "data.get('name')"},
                            ],
                        },
                        {
                            "name": "done",
                            "is_end": True,
                            "response_template": "Done!",
                        },
                    ],
                },
            },
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="greet-msg-test-1",
                client_id="test",
            )

            greeting = await bot.greet(context)

            assert greeting is not None
            assert "Welcome aboard!" in greeting

            # Should have auto-advanced to "collect"
            wizard_state = await bot.get_wizard_state("greet-msg-test-1")
            assert wizard_state is not None
            assert wizard_state.get("current_stage") == "collect"
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_stream_chat_after_message_stage_greet(self) -> None:
        """DynaBot.stream_chat() works after a greet that auto-advanced
        through a message stage.

        Verifies that the wizard state is correct after greet auto-advances
        through a message stage, and that a subsequent stream_chat can
        interact with the landing stage.
        """
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_bots.bot.context import BotContext

        config: dict[str, Any] = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": {
                    "name": "stream-msg-test",
                    "stages": [
                        {
                            "name": "welcome",
                            "is_start": True,
                            "response_template": "Welcome!",
                            "auto_advance": True,
                            "transitions": [
                                {"target": "collect", "condition": "true"},
                            ],
                        },
                        {
                            "name": "collect",
                            "prompt": "What is your name?",
                            "response_template": "Please tell me your name.",
                            "schema": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            },
                            "transitions": [
                                {
                                    "target": "done",
                                    "condition": "data.get('name')",
                                },
                            ],
                        },
                        {
                            "name": "done",
                            "is_end": True,
                            "response_template": "All done!",
                        },
                    ],
                },
            },
        }
        bot = await DynaBot.from_config(config)
        try:
            context = BotContext(
                conversation_id="stream-msg-test-1",
                client_id="test",
            )

            # Greet auto-advances through "welcome" message stage
            greeting = await bot.greet(context)
            assert greeting is not None
            assert "Welcome!" in greeting

            # Wizard should be on "collect" now
            wizard_state = await bot.get_wizard_state("stream-msg-test-1")
            assert wizard_state is not None
            assert wizard_state.get("current_stage") == "collect"

            # Stream a response — the user provides their name.
            # Verbatim capture works here because "collect" has a single
            # required string field.  However, there's a bot response in
            # history from greet, so extraction may fall through to the
            # heuristic.  Stream the response and verify we get something.
            chunks: list[str] = []
            async for chunk in bot.stream_chat("Alice", context):
                if chunk.delta:
                    chunks.append(chunk.delta)

            full_response = "".join(chunks)
            # We got a streamed response (content depends on extraction
            # behavior with echo provider, but streaming itself works)
            assert len(full_response) > 0
        finally:
            await bot.close()

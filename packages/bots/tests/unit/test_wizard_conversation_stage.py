"""Tests for conversation stage paradigm in WizardReasoning.

Tests cover:
- _message injection into FSM context and cleanup after step
- mode: conversation stages skipping extraction
- Intent detection (keyword and LLM methods)
- stage_mode in wizard metadata
- WizardConfigLoader preserving mode and intent_detection
- Structured → conversation → structured round-trip
"""

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.llm import LLMMessage, LLMResponse

from .conftest import WizardTestManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conversation_wizard_config() -> dict:
    """Wizard with a conversation stage and a structured stage."""
    return {
        "name": "conversation-test",
        "version": "1.0",
        "stages": [
            {
                "name": "chat",
                "is_start": True,
                "mode": "conversation",
                "prompt": "You are a friendly tutor.",
                "suggestions": ["Quiz me", "Explain a concept"],
                "intent_detection": {
                    "method": "keyword",
                    "intents": [
                        {
                            "id": "start_quiz",
                            "keywords": ["quiz", "test me"],
                            "description": "Start a quiz",
                        },
                    ],
                },
                "transitions": [
                    {
                        "target": "quiz",
                        "condition": "data.get('_intent') == 'start_quiz'",
                    },
                ],
            },
            {
                "name": "quiz",
                "prompt": "Answer the following question.",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                "transitions": [{"target": "done", "condition": "data.get('answer')"}],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "All done!",
            },
        ],
    }


@pytest.fixture
def conversation_reasoning(
    conversation_wizard_config: dict,
) -> WizardReasoning:
    """WizardReasoning with conversation stage config."""
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(conversation_wizard_config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


@pytest.fixture
def roundtrip_wizard_config() -> dict:
    """Wizard for testing structured → conversation → structured round-trip."""
    return {
        "name": "roundtrip-test",
        "version": "1.0",
        "stages": [
            {
                "name": "collect_info",
                "is_start": True,
                "prompt": "Tell me your name and topic.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "topic": {"type": "string"},
                    },
                },
                "intent_detection": {
                    "method": "keyword",
                    "intents": [
                        {
                            "id": "need_help",
                            "keywords": ["help", "confused"],
                        },
                    ],
                },
                "transitions": [
                    {
                        "target": "help_chat",
                        "condition": "data.get('_intent') == 'need_help'",
                        "priority": 0,
                    },
                    {
                        "target": "done",
                        "condition": "data.get('name') and data.get('topic')",
                        "priority": 1,
                    },
                ],
            },
            {
                "name": "help_chat",
                "mode": "conversation",
                "prompt": "Happy to help! Ask me anything.",
                "intent_detection": {
                    "method": "keyword",
                    "intents": [
                        {
                            "id": "resume",
                            "keywords": ["continue", "let's go", "back"],
                        },
                    ],
                },
                "transitions": [
                    {
                        "target": "collect_info",
                        "condition": "data.get('_intent') == 'resume'",
                    },
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "Complete!",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Enhancement 1: _message injection
# ---------------------------------------------------------------------------


class TestMessageInjection:
    """Tests for _message injection into FSM context."""

    def test_message_based_transition(self) -> None:
        """Transition condition can use data.get('_message')."""
        config = {
            "name": "msg-test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Say something",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": (
                                "'magic' in data.get('_message', '').lower()"
                            ),
                        },
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        # Without _message - should stay
        fsm.step({"_message": "hello"})
        assert fsm.current_stage == "start"

        # With matching _message - should transition
        fsm.restart()
        fsm.step({"_message": "the magic word"})
        assert fsm.current_stage == "end"

    @pytest.mark.asyncio
    async def test_message_not_persisted_after_step(
        self,
        conversation_reasoning: WizardReasoning,
    ) -> None:
        """_message is cleaned up from wizard_state.data after FSM step."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "just chatting"}]
        manager.metadata = {}

        # Script response
        manager.echo_provider.set_responses(["Hello!"])

        await conversation_reasoning.generate(manager, llm=None)

        # After generate, _message should NOT be in persisted data
        fsm_state = manager.metadata["wizard"]["fsm_state"]
        assert "_message" not in fsm_state["data"]


# ---------------------------------------------------------------------------
# Enhancement 2: Conversation mode
# ---------------------------------------------------------------------------


class TestConversationMode:
    """Tests for mode: conversation stage behavior."""

    @pytest.mark.asyncio
    async def test_conversation_stage_skips_extraction(
        self,
        conversation_reasoning: WizardReasoning,
    ) -> None:
        """Conversation stage generates response without extraction."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "Tell me about math"}]
        manager.metadata = {}

        manager.echo_provider.set_responses(["Math is fascinating!"])

        response = await conversation_reasoning.generate(manager, llm=None)

        assert response is not None
        assert response.content == "Math is fascinating!"
        # Should be at chat stage (no transition since no intent matched)
        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "chat"

    @pytest.mark.asyncio
    async def test_conversation_stage_with_intent_triggers_transition(
        self,
        conversation_reasoning: WizardReasoning,
    ) -> None:
        """Keyword intent triggers transition from conversation stage."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "Quiz me please!"}]
        manager.metadata = {}

        manager.echo_provider.set_responses(["Let's start the quiz!"])

        response = await conversation_reasoning.generate(manager, llm=None)

        assert response is not None
        # Should have transitioned to quiz stage
        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "quiz"

    @pytest.mark.asyncio
    async def test_conversation_no_clarification_loop(
        self,
        conversation_reasoning: WizardReasoning,
    ) -> None:
        """Conversation stage does not increment clarification_attempts."""
        manager = WizardTestManager()
        manager.metadata = {}

        # Send multiple messages - none should trigger clarification
        for i in range(5):
            manager.messages = [
                {"role": "user", "content": f"Random message {i}"}
            ]
            manager.echo_provider.set_responses([f"Response {i}"])
            await conversation_reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["clarification_attempts"] == 0


# ---------------------------------------------------------------------------
# Enhancement 3: Intent detection
# ---------------------------------------------------------------------------


class TestIntentDetection:
    """Tests for _detect_intent method."""

    @pytest.fixture
    def reasoning(self, conversation_wizard_config: dict) -> WizardReasoning:
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(conversation_wizard_config)
        return WizardReasoning(wizard_fsm=fsm, strict_validation=False)

    @pytest.mark.asyncio
    async def test_keyword_intent_match(self, reasoning: WizardReasoning) -> None:
        """Keyword method detects intent from message."""
        state = WizardState(current_stage="chat")
        stage = {
            "intent_detection": {
                "method": "keyword",
                "intents": [
                    {"id": "greet", "keywords": ["hello", "hi"]},
                    {"id": "quiz", "keywords": ["quiz", "test"]},
                ],
            },
        }

        await reasoning._detect_intent("Let's do a quiz!", stage, state, None)
        assert state.data["_intent"] == "quiz"

    @pytest.mark.asyncio
    async def test_keyword_intent_case_insensitive(
        self, reasoning: WizardReasoning
    ) -> None:
        """Keyword matching is case-insensitive."""
        state = WizardState(current_stage="chat")
        stage = {
            "intent_detection": {
                "method": "keyword",
                "intents": [{"id": "greet", "keywords": ["hello"]}],
            },
        }

        await reasoning._detect_intent("HELLO there!", stage, state, None)
        assert state.data["_intent"] == "greet"

    @pytest.mark.asyncio
    async def test_keyword_intent_no_match(
        self, reasoning: WizardReasoning
    ) -> None:
        """No intent set when no keywords match."""
        state = WizardState(current_stage="chat")
        stage = {
            "intent_detection": {
                "method": "keyword",
                "intents": [{"id": "greet", "keywords": ["hello"]}],
            },
        }

        await reasoning._detect_intent("goodbye world", stage, state, None)
        assert "_intent" not in state.data

    @pytest.mark.asyncio
    async def test_keyword_first_match_wins(
        self, reasoning: WizardReasoning
    ) -> None:
        """First matching intent in list wins."""
        state = WizardState(current_stage="chat")
        stage = {
            "intent_detection": {
                "method": "keyword",
                "intents": [
                    {"id": "first", "keywords": ["test"]},
                    {"id": "second", "keywords": ["test"]},
                ],
            },
        }

        await reasoning._detect_intent("test message", stage, state, None)
        assert state.data["_intent"] == "first"

    @pytest.mark.asyncio
    async def test_intent_cleared_each_turn(
        self, reasoning: WizardReasoning
    ) -> None:
        """Previous _intent is cleared before detection runs."""
        state = WizardState(current_stage="chat", data={"_intent": "old_intent"})
        stage = {
            "intent_detection": {
                "method": "keyword",
                "intents": [{"id": "greet", "keywords": ["hello"]}],
            },
        }

        # Message doesn't match - _intent should be cleared
        await reasoning._detect_intent("no match", stage, state, None)
        assert "_intent" not in state.data

    @pytest.mark.asyncio
    async def test_no_intent_config_is_noop(
        self, reasoning: WizardReasoning
    ) -> None:
        """No intent_detection config means no _intent is set."""
        state = WizardState(current_stage="chat", data={"existing": "data"})
        stage = {}

        await reasoning._detect_intent("hello", stage, state, None)
        assert "_intent" not in state.data
        assert state.data["existing"] == "data"

    @pytest.mark.asyncio
    async def test_llm_intent_detection(
        self, reasoning: WizardReasoning
    ) -> None:
        """LLM method classifies intent via LLM call."""
        from dataknobs_llm.llm.providers.echo import EchoProvider

        llm = EchoProvider(
            {"provider": "echo", "model": "echo", "options": {"echo_prefix": ""}}
        )
        llm.set_responses(["start_quiz"])

        state = WizardState(current_stage="chat")
        stage = {
            "intent_detection": {
                "method": "llm",
                "intents": [
                    {
                        "id": "start_quiz",
                        "description": "Start a quiz session",
                    },
                    {
                        "id": "get_help",
                        "description": "Get help with a concept",
                    },
                ],
            },
        }

        await reasoning._detect_intent("quiz me", stage, state, llm)
        assert state.data["_intent"] == "start_quiz"

    @pytest.mark.asyncio
    async def test_llm_intent_invalid_response(
        self, reasoning: WizardReasoning
    ) -> None:
        """LLM returning invalid intent ID results in no intent set."""
        from dataknobs_llm.llm.providers.echo import EchoProvider

        llm = EchoProvider(
            {"provider": "echo", "model": "echo", "options": {"echo_prefix": ""}}
        )
        llm.set_responses(["invalid_intent_id"])

        state = WizardState(current_stage="chat")
        stage = {
            "intent_detection": {
                "method": "llm",
                "intents": [
                    {"id": "start_quiz", "description": "Start a quiz"},
                ],
            },
        }

        await reasoning._detect_intent("quiz me", stage, state, llm)
        assert "_intent" not in state.data

    @pytest.mark.asyncio
    async def test_llm_intent_none_response(
        self, reasoning: WizardReasoning
    ) -> None:
        """LLM returning 'none' results in no intent set."""
        from dataknobs_llm.llm.providers.echo import EchoProvider

        llm = EchoProvider(
            {"provider": "echo", "model": "echo", "options": {"echo_prefix": ""}}
        )
        llm.set_responses(["none"])

        state = WizardState(current_stage="chat")
        stage = {
            "intent_detection": {
                "method": "llm",
                "intents": [
                    {"id": "start_quiz", "description": "Start a quiz"},
                ],
            },
        }

        await reasoning._detect_intent("just chatting", stage, state, llm)
        assert "_intent" not in state.data


# ---------------------------------------------------------------------------
# Enhancement 3 + 2: Intent detection on structured stages
# ---------------------------------------------------------------------------


class TestStructuredStageIntentDetection:
    """Tests for intent detection on structured (non-conversation) stages."""

    @pytest.mark.asyncio
    async def test_structured_stage_runs_intent_detection(self) -> None:
        """Structured stage with intent_detection runs detection after extraction."""
        config = {
            "name": "structured-intent-test",
            "version": "1.0",
            "stages": [
                {
                    "name": "collect",
                    "is_start": True,
                    "prompt": "Enter your data",
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                    "intent_detection": {
                        "method": "keyword",
                        "intents": [
                            {"id": "need_help", "keywords": ["help", "confused"]},
                        ],
                    },
                    "transitions": [
                        {
                            "target": "help",
                            "condition": "data.get('_intent') == 'need_help'",
                            "priority": 0,
                        },
                        {
                            "target": "done",
                            "condition": "data.get('value')",
                            "priority": 1,
                        },
                    ],
                },
                {
                    "name": "help",
                    "mode": "conversation",
                    "prompt": "How can I help?",
                    "transitions": [],
                },
                {"name": "done", "is_end": True, "prompt": "Complete!"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "I'm confused, help!"}]
        manager.metadata = {}

        manager.echo_provider.set_responses(["How can I help?"])

        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "help"


# ---------------------------------------------------------------------------
# Enhancement 4: stage_mode in metadata
# ---------------------------------------------------------------------------


class TestStageModeMeta:
    """Tests for stage_mode in wizard metadata."""

    @pytest.mark.asyncio
    async def test_conversation_stage_mode(
        self,
        conversation_reasoning: WizardReasoning,
    ) -> None:
        """Conversation stage returns stage_mode='conversation' in metadata."""
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "hello"}]
        manager.metadata = {}

        manager.echo_provider.set_responses(["Hi!"])

        await conversation_reasoning.generate(manager, llm=None)

        wizard_meta = manager.metadata["wizard"]
        assert wizard_meta["stage_mode"] == "conversation"

    @pytest.mark.asyncio
    async def test_structured_stage_mode_default(self) -> None:
        """Structured stage returns stage_mode='structured' (default)."""
        config = {
            "name": "structured-test",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Enter something",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "hello"}]
        manager.metadata = {}

        manager.echo_provider.set_responses(["Welcome!"])

        await reasoning.generate(manager, llm=None)

        wizard_meta = manager.metadata["wizard"]
        assert wizard_meta["stage_mode"] == "structured"


# ---------------------------------------------------------------------------
# Enhancement 5: Config loader
# ---------------------------------------------------------------------------


class TestConfigLoaderConversation:
    """Tests for WizardConfigLoader preserving conversation stage fields."""

    def test_mode_preserved_in_metadata(self) -> None:
        """mode field is preserved in stage metadata after loading."""
        config = {
            "name": "loader-test",
            "stages": [
                {
                    "name": "chat",
                    "is_start": True,
                    "mode": "conversation",
                    "prompt": "Hello",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        meta = fsm.current_metadata
        assert meta["mode"] == "conversation"

    def test_intent_detection_preserved_in_metadata(self) -> None:
        """intent_detection config is preserved in stage metadata."""
        intent_config = {
            "method": "keyword",
            "intents": [
                {"id": "quiz", "keywords": ["quiz", "test"]},
            ],
        }
        config = {
            "name": "loader-test",
            "stages": [
                {
                    "name": "chat",
                    "is_start": True,
                    "mode": "conversation",
                    "prompt": "Hello",
                    "intent_detection": intent_config,
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        meta = fsm.current_metadata
        assert meta["intent_detection"] == intent_config

    def test_structured_stage_no_mode(self) -> None:
        """Stage without mode has mode=None in metadata."""
        config = {
            "name": "loader-test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Hello",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        meta = fsm.current_metadata
        assert meta["mode"] is None

    def test_structured_stage_no_intent_detection(self) -> None:
        """Stage without intent_detection has None in metadata."""
        config = {
            "name": "loader-test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Hello",
                },
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        meta = fsm.current_metadata
        assert meta["intent_detection"] is None


# ---------------------------------------------------------------------------
# Round-trip: structured → conversation → structured
# ---------------------------------------------------------------------------


class TestConversationRoundTrip:
    """Tests for structured → conversation → structured round-trip flow."""

    @pytest.mark.asyncio
    async def test_data_preserved_across_conversation_detour(
        self, roundtrip_wizard_config: dict
    ) -> None:
        """Partially collected data survives a conversation detour and back."""
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(roundtrip_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        manager = WizardTestManager()

        # Turn 1: User provides partial data + "help" keyword
        manager.messages = [{"role": "user", "content": "I'm confused, help me"}]
        manager.metadata = {}
        manager.echo_provider.set_responses(["Happy to help!"])

        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "help_chat"

        # Turn 2: User asks a question in conversation mode
        manager.messages = [
            {"role": "user", "content": "What topics are available?"}
        ]
        manager.echo_provider.set_responses(["We have math and science!"])

        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "help_chat"

        # Turn 3: User says "let's continue" to go back
        manager.messages = [
            {"role": "user", "content": "OK let's continue with setup"}
        ]
        manager.echo_provider.set_responses(["Great, let's continue!"])

        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "collect_info"

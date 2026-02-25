"""Integration tests for wizard metadata persistence on conversation nodes.

Validates that all code paths in ``_generate_stage_response`` persist
``metadata.wizard`` on the assistant conversation node via the real
``ConversationManager`` pipeline backed by ``EchoProvider``.

The four paths tested:
1. **Template pure** — ``response_template`` with no ``llm_assist``
2. **Template + llm_assist** — ``response_template`` with ``llm_assist: true``
3. **LLM single-call** — no template, no tools (or tools with ``single`` reasoning)
4. **LLM ReAct** — tools with ``reasoning: react``
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import EchoProvider
from dataknobs_llm.conversations.manager import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.prompts.builders import AsyncPromptBuilder
from dataknobs_llm.prompts.implementations.config_library import ConfigPromptLibrary
from dataknobs_llm.testing import text_response, tool_call_response
from dataknobs_llm.tools.base import Tool


# ---------------------------------------------------------------------------
# Test tool for ReAct path
# ---------------------------------------------------------------------------


class LookupTool(Tool):
    """Minimal tool for exercising the ReAct path."""

    def __init__(self) -> None:
        super().__init__(name="lookup", description="Look something up")
        self.call_count = 0

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        return {"result": "found it"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def conversation_storage() -> DataknobsConversationStorage:
    """Real conversation storage backed by in-memory database."""
    return DataknobsConversationStorage(AsyncMemoryDatabase())


@pytest_asyncio.fixture
async def echo_provider() -> EchoProvider:
    """EchoProvider with empty prefix for predictable content."""
    return EchoProvider(
        {"provider": "echo", "model": "echo-test", "options": {"echo_prefix": ""}}
    )


@pytest_asyncio.fixture
async def conversation_manager(
    echo_provider: EchoProvider,
    conversation_storage: DataknobsConversationStorage,
) -> ConversationManager:
    """ConversationManager backed by EchoProvider + memory storage."""
    library = ConfigPromptLibrary(
        {"system": {"wizard_test": {"template": "You are a test wizard."}}}
    )
    builder = AsyncPromptBuilder(library=library)
    manager = await ConversationManager.create(
        llm=echo_provider,
        prompt_builder=builder,
        storage=conversation_storage,
        system_prompt_name="wizard_test",
    )
    return manager


def _make_config(
    *,
    response_template: str | None = None,
    llm_assist: bool = False,
    reasoning: str | None = None,
    tools: list[str] | None = None,
) -> dict[str, Any]:
    """Build a 2-stage wizard config with customisable start stage."""
    start_stage: dict[str, Any] = {
        "name": "collect",
        "is_start": True,
        "prompt": "Tell me something",
        "schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
        },
        "transitions": [{"target": "done", "condition": "data.get('topic')"}],
    }
    if response_template is not None:
        start_stage["response_template"] = response_template
    if llm_assist:
        start_stage["llm_assist"] = True
        start_stage["llm_assist_prompt"] = "Help the user with their question."
    if reasoning is not None:
        start_stage["reasoning"] = reasoning
    if tools is not None:
        start_stage["tools"] = tools

    return {
        "name": "metadata-test",
        "version": "1.0",
        "stages": [
            start_stage,
            {"name": "done", "is_end": True, "prompt": "Complete"},
        ],
    }


def _last_assistant_node(manager: ConversationManager) -> Any:
    """Return the last assistant conversation node from the manager state."""
    nodes = manager.state.get_current_nodes()
    for node in reversed(nodes):
        if node.message.role == "assistant":
            return node
    raise AssertionError("No assistant node found in conversation")


def _assert_wizard_metadata(node: Any, expected_stage: str) -> None:
    """Assert that a conversation node has a valid wizard metadata snapshot."""
    assert node.metadata is not None, "Node metadata is None"
    assert "wizard" in node.metadata, (
        f"Node metadata missing 'wizard' key. Keys: {list(node.metadata.keys())}"
    )
    wizard = node.metadata["wizard"]
    assert wizard["current_stage"] == expected_stage
    assert "stage_index" in wizard
    assert "total_stages" in wizard
    assert "progress_percent" in wizard
    assert "stages" in wizard


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWizardMetadataPersistence:
    """Verify wizard metadata is persisted on conversation nodes for all paths."""

    @pytest.mark.asyncio
    async def test_template_pure_persists_wizard_metadata(
        self,
        echo_provider: EchoProvider,
        conversation_manager: ConversationManager,
    ) -> None:
        """Path 1: template-only stage persists wizard metadata via add_message."""
        config = _make_config(response_template="Hello {{ topic | default('world') }}!")
        fsm = WizardConfigLoader().load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Seed wizard state so _generate_stage_response finds it
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "astronomy"
        reasoning._save_wizard_state(conversation_manager, state)

        # Add a user message (non-question, so llm_assist is not triggered)
        await conversation_manager.add_message(role="user", content="astronomy")

        # Call _generate_stage_response in template-pure mode
        stage = fsm.current_metadata
        response = await reasoning._generate_stage_response(
            conversation_manager, echo_provider, stage, state, tools=[]
        )

        assert "astronomy" in response.content

        node = _last_assistant_node(conversation_manager)
        _assert_wizard_metadata(node, "collect")

    @pytest.mark.asyncio
    async def test_template_llm_assist_persists_wizard_metadata(
        self,
        echo_provider: EchoProvider,
        conversation_manager: ConversationManager,
    ) -> None:
        """Path 2: template + llm_assist stage persists wizard metadata via complete()."""
        config = _make_config(
            response_template="Hello {{ topic | default('world') }}!",
            llm_assist=True,
        )
        fsm = WizardConfigLoader().load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "biology"
        reasoning._save_wizard_state(conversation_manager, state)

        # User asks a question — triggers llm_assist path
        await conversation_manager.add_message(
            role="user", content="What topics can I choose?"
        )

        # Script EchoProvider for the assist response
        echo_provider.set_responses([
            text_response("You can choose any science topic!"),
        ])

        stage = fsm.current_metadata
        response = await reasoning._generate_stage_response(
            conversation_manager, echo_provider, stage, state, tools=[]
        )

        assert response.content == "You can choose any science topic!"

        node = _last_assistant_node(conversation_manager)
        _assert_wizard_metadata(node, "collect")

    @pytest.mark.asyncio
    async def test_llm_single_call_persists_wizard_metadata(
        self,
        echo_provider: EchoProvider,
        conversation_manager: ConversationManager,
    ) -> None:
        """Path 3: LLM single-call stage persists wizard metadata via complete()."""
        config = _make_config()  # No template — uses LLM mode
        fsm = WizardConfigLoader().load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        state = reasoning._get_wizard_state(conversation_manager)
        reasoning._save_wizard_state(conversation_manager, state)

        await conversation_manager.add_message(
            role="user", content="I want to learn about physics"
        )

        echo_provider.set_responses([
            text_response("Great choice! Tell me more about what interests you."),
        ])

        stage = fsm.current_metadata
        response = await reasoning._generate_stage_response(
            conversation_manager, echo_provider, stage, state, tools=[]
        )

        assert "Great choice" in response.content

        node = _last_assistant_node(conversation_manager)
        _assert_wizard_metadata(node, "collect")

    @pytest.mark.asyncio
    async def test_llm_react_persists_wizard_metadata(
        self,
        echo_provider: EchoProvider,
        conversation_manager: ConversationManager,
    ) -> None:
        """Path 4: LLM ReAct stage persists wizard metadata via complete()."""
        config = _make_config(reasoning="react", tools=["lookup"])
        fsm = WizardConfigLoader().load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        state = reasoning._get_wizard_state(conversation_manager)
        reasoning._save_wizard_state(conversation_manager, state)

        await conversation_manager.add_message(
            role="user", content="Look up astronomy for me"
        )

        lookup_tool = LookupTool()

        # Script: one tool call, then a final text response
        echo_provider.set_responses([
            tool_call_response("lookup", {}),
            text_response("I looked it up — astronomy is fascinating!"),
        ])

        stage = fsm.current_metadata
        response = await reasoning._generate_stage_response(
            conversation_manager, echo_provider, stage, state, tools=[lookup_tool]
        )

        assert lookup_tool.call_count == 1
        assert "astronomy" in response.content

        # The final assistant node (text response) should have wizard metadata
        node = _last_assistant_node(conversation_manager)
        _assert_wizard_metadata(node, "collect")

    @pytest.mark.asyncio
    async def test_react_max_iterations_persists_wizard_metadata(
        self,
        echo_provider: EchoProvider,
        conversation_manager: ConversationManager,
    ) -> None:
        """Path 4b: ReAct max-iterations fallback persists wizard metadata."""
        config = _make_config(reasoning="react", tools=["lookup"])
        # Override max_iterations to 1 so we hit the fallback quickly
        fsm = WizardConfigLoader().load_from_dict(config)
        fsm.current_metadata["max_iterations"] = 1
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        state = reasoning._get_wizard_state(conversation_manager)
        reasoning._save_wizard_state(conversation_manager, state)

        await conversation_manager.add_message(
            role="user", content="Look up everything"
        )

        lookup_tool = LookupTool()

        # Script: one tool call (hits max_iterations=1), then forced text response
        echo_provider.set_responses([
            tool_call_response("lookup", {}),
            text_response("Done after max iterations."),
        ])

        stage = fsm.current_metadata
        response = await reasoning._generate_stage_response(
            conversation_manager, echo_provider, stage, state, tools=[lookup_tool]
        )

        assert response.content == "Done after max iterations."

        node = _last_assistant_node(conversation_manager)
        _assert_wizard_metadata(node, "collect")

    @pytest.mark.asyncio
    async def test_metadata_survives_storage_roundtrip(
        self,
        echo_provider: EchoProvider,
        conversation_manager: ConversationManager,
        conversation_storage: DataknobsConversationStorage,
    ) -> None:
        """Wizard metadata on nodes survives save → load from storage."""
        config = _make_config()
        fsm = WizardConfigLoader().load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "chemistry"
        reasoning._save_wizard_state(conversation_manager, state)

        await conversation_manager.add_message(
            role="user", content="chemistry"
        )

        echo_provider.set_responses([
            text_response("Chemistry is a great topic!"),
        ])

        stage = fsm.current_metadata
        await reasoning._generate_stage_response(
            conversation_manager, echo_provider, stage, state, tools=[]
        )

        # Save to storage and reload
        conv_id = conversation_manager.state.conversation_id
        await conversation_storage.save_conversation(conversation_manager.state)
        loaded = await conversation_storage.load_conversation(conv_id)
        assert loaded is not None

        # Find the assistant node in the loaded conversation
        assistant_nodes = [
            n for n in loaded.get_current_nodes() if n.message.role == "assistant"
        ]
        assert len(assistant_nodes) >= 1

        last_node = assistant_nodes[-1]
        assert last_node.metadata is not None
        assert "wizard" in last_node.metadata
        assert last_node.metadata["wizard"]["current_stage"] == "collect"

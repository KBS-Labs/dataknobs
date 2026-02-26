"""Tests for ReasoningManagerProtocol conformance.

Verifies that ConversationManager satisfies the ReasoningManagerProtocol,
preventing interface drift.
"""

import pytest

from dataknobs_bots.reasoning.base import ReasoningManagerProtocol
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider


class TestProtocolConformance:
    """Verify that ConversationManager satisfies the protocol."""

    @pytest.mark.asyncio
    async def test_conversation_manager_satisfies_protocol(
        self,
        conversation_manager: ConversationManager,
    ) -> None:
        """ConversationManager satisfies ReasoningManagerProtocol."""
        assert isinstance(conversation_manager, ReasoningManagerProtocol)


class TestRealManagerProtocol:
    """Verify protocol methods work on real ConversationManager."""

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt_override(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Real ConversationManager.complete() uses system_prompt_override."""
        manager, provider = conversation_manager_pair

        await manager.add_message(role="user", content="Hello")
        provider.set_responses(["Response with override"])

        await manager.complete(system_prompt_override="Override prompt")

        last_call = provider.get_last_call()
        messages = last_call["messages"]
        system_msg = next(m for m in messages if m.role == "system")
        assert system_msg.content == "Override prompt"

    @pytest.mark.asyncio
    async def test_complete_with_tools(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Real ConversationManager.complete() forwards tools to provider."""
        manager, provider = conversation_manager_pair

        await manager.add_message(role="user", content="Use tools")
        provider.set_responses(["Tool response"])

        class MockTool:
            name = "test_tool"
            description = "A test tool"
            schema = {"type": "object"}

        await manager.complete(tools=[MockTool()])

        last_call = provider.get_last_call()
        assert last_call["tools"] is not None
        assert len(last_call["tools"]) == 1

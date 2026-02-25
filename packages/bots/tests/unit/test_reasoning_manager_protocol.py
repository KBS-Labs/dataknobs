"""Tests for ReasoningManagerProtocol conformance.

Verifies that ConversationManager and WizardTestManager both satisfy
the ReasoningManagerProtocol, preventing interface drift.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_bots.reasoning.base import ReasoningManagerProtocol
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.llm import LLMConfig
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary


def _create_prompts(prompt_dir: Path) -> None:
    """Create minimal prompt files."""
    system_dir = prompt_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "assistant.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
    )


class TestProtocolConformance:
    """Verify that real and test managers satisfy the protocol."""

    @pytest.mark.asyncio
    async def test_conversation_manager_satisfies_protocol(self) -> None:
        """ConversationManager satisfies ReasoningManagerProtocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir) / "prompts"
            _create_prompts(prompt_dir)

            config = LLMConfig(provider="echo", model="echo-test", options={"echo_prefix": ""})
            provider = EchoProvider(config)
            library = FileSystemPromptLibrary(prompt_dir)
            builder = AsyncPromptBuilder(library=library)
            storage = DataknobsConversationStorage(AsyncMemoryDatabase())

            manager = await ConversationManager.create(
                llm=provider,
                prompt_builder=builder,
                storage=storage,
                system_prompt_name="assistant",
            )

            assert isinstance(manager, ReasoningManagerProtocol)
            await provider.close()

    def test_wizard_test_manager_satisfies_protocol(self, test_manager: Any) -> None:
        """WizardTestManager satisfies ReasoningManagerProtocol."""
        assert isinstance(test_manager, ReasoningManagerProtocol)


class TestProtocolMethods:
    """Verify that protocol methods work correctly on WizardTestManager."""

    @pytest.mark.asyncio
    async def test_add_message_keyword_args(self, test_manager: Any) -> None:
        """add_message() works with keyword arguments (matching protocol)."""
        await test_manager.add_message(role="user", content="Hello")
        await test_manager.add_message(role="assistant", content="Hi there")
        await test_manager.add_message(
            role="system", content="Context update", metadata={"source": "test"}
        )

        messages = test_manager.get_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "system"
        assert messages[2]["metadata"] == {"source": "test"}

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt_override(self, test_manager: Any) -> None:
        """complete() accepts system_prompt_override."""
        test_manager.echo_provider.set_responses(["Test response"])
        test_manager.messages.append({"role": "user", "content": "Hello"})

        response = await test_manager.complete(
            system_prompt_override="Custom prompt"
        )

        assert response.content == "Test response"
        assert test_manager.complete_calls[-1]["system_prompt_override"] == "Custom prompt"

    @pytest.mark.asyncio
    async def test_complete_with_tools(self, test_manager: Any) -> None:
        """complete() accepts tools parameter."""
        test_manager.echo_provider.set_responses(["Using tools"])
        test_manager.messages.append({"role": "user", "content": "Search"})

        class MockTool:
            name = "search"
            description = "Search"
            schema = {}

        response = await test_manager.complete(tools=[MockTool()])

        assert response.content == "Using tools"
        assert test_manager.complete_calls[-1]["tools"] is not None
        assert len(test_manager.complete_calls[-1]["tools"]) == 1

    def test_system_prompt_property(self, test_manager: Any) -> None:
        """system_prompt property is accessible."""
        assert test_manager.system_prompt == "You are a helpful assistant."

    def test_metadata_property(self, test_manager: Any) -> None:
        """metadata property is accessible and writable."""
        test_manager.metadata["test_key"] = "test_value"
        assert test_manager.metadata["test_key"] == "test_value"

    def test_get_messages(self, test_manager: Any) -> None:
        """get_messages() returns list of dicts."""
        test_manager.add_user_message("Hello")
        messages = test_manager.get_messages()
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestRealManagerProtocol:
    """Verify protocol methods work on real ConversationManager."""

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt_override(
        self, real_conversation_manager: dict[str, Any]
    ) -> None:
        """Real ConversationManager.complete() uses system_prompt_override."""
        manager = real_conversation_manager["manager"]
        provider: EchoProvider = real_conversation_manager["provider"]

        await manager.add_message(role="user", content="Hello")
        provider.set_responses(["Response with override"])

        await manager.complete(system_prompt_override="Override prompt")

        last_call = provider.get_last_call()
        messages = last_call["messages"]
        system_msg = next(m for m in messages if m.role == "system")
        assert system_msg.content == "Override prompt"

    @pytest.mark.asyncio
    async def test_complete_with_tools(
        self, real_conversation_manager: dict[str, Any]
    ) -> None:
        """Real ConversationManager.complete() forwards tools to provider."""
        manager = real_conversation_manager["manager"]
        provider: EchoProvider = real_conversation_manager["provider"]

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

"""Tests for ConversationManager.complete() system_prompt_override and tools parameters.

These tests verify the fixes for the WizardReasoning / ConversationManager
interface disconnect where system_prompt_override was silently dropped and
tools handling was inconsistent across providers.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.llm import LLMConfig, LLMMessage, LLMResponse
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary


def _create_prompts(prompt_dir: Path) -> None:
    """Create test prompt files."""
    system_dir = prompt_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "helpful.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
    )
    user_dir = prompt_dir / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "greeting.yaml").write_text(
        yaml.dump({"template": "Hello!"})
    )


@pytest.fixture
async def manager_with_provider():
    """Create a ConversationManager + EchoProvider for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        _create_prompts(prompt_dir)

        config = LLMConfig(
            provider="echo",
            model="echo-test",
            options={"echo_prefix": ""},
        )
        provider = EchoProvider(config)
        library = FileSystemPromptLibrary(prompt_dir)
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        manager = await ConversationManager.create(
            llm=provider,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="helpful",
        )

        yield {"manager": manager, "provider": provider}
        await provider.close()


class TestSystemPromptOverride:
    """Tests for system_prompt_override parameter on ConversationManager.complete()."""

    @pytest.mark.asyncio
    async def test_override_replaces_system_message(self, manager_with_provider: dict[str, Any]) -> None:
        """system_prompt_override replaces the system message sent to the LLM."""
        manager = manager_with_provider["manager"]
        provider: EchoProvider = manager_with_provider["provider"]

        # Add a user message
        await manager.add_message(role="user", content="Hello")

        # Complete with override
        provider.set_responses(["Override response"])
        await manager.complete(system_prompt_override="You are a wizard at step 2")

        # Verify the provider saw the overridden system prompt
        last_call = provider.get_last_call()
        assert last_call is not None
        messages = last_call["messages"]
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "You are a wizard at step 2"

    @pytest.mark.asyncio
    async def test_override_does_not_mutate_tree(self, manager_with_provider: dict[str, Any]) -> None:
        """system_prompt_override must NOT mutate the conversation tree."""
        manager = manager_with_provider["manager"]
        provider: EchoProvider = manager_with_provider["provider"]

        # Add user message
        await manager.add_message(role="user", content="Hello")

        # Complete with override
        provider.set_responses(["Response 1"])
        await manager.complete(system_prompt_override="Temporary override")

        # Verify the tree's system prompt is unchanged
        assert manager.system_prompt == "You are a helpful assistant"

        # Verify messages in tree still have original system prompt
        tree_messages = manager.get_messages()
        system_msgs = [m for m in tree_messages if m.get("role") == "system"]
        assert any("helpful assistant" in m.get("content", "") for m in system_msgs)

    @pytest.mark.asyncio
    async def test_no_override_uses_original(self, manager_with_provider: dict[str, Any]) -> None:
        """Without system_prompt_override, the original system prompt is used."""
        manager = manager_with_provider["manager"]
        provider: EchoProvider = manager_with_provider["provider"]

        await manager.add_message(role="user", content="Hello")
        provider.set_responses(["Normal response"])
        await manager.complete()

        last_call = provider.get_last_call()
        assert last_call is not None
        messages = last_call["messages"]
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_override_with_no_existing_system_message(self) -> None:
        """system_prompt_override prepends system message if none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir) / "prompts"
            _create_prompts(prompt_dir)

            config = LLMConfig(provider="echo", model="echo-test", options={"echo_prefix": ""})
            provider = EchoProvider(config)
            library = FileSystemPromptLibrary(prompt_dir)
            builder = AsyncPromptBuilder(library=library)
            storage = DataknobsConversationStorage(AsyncMemoryDatabase())

            # Create WITHOUT system prompt
            manager = await ConversationManager.create(
                llm=provider,
                prompt_builder=builder,
                storage=storage,
            )

            await manager.add_message(role="user", content="Hello")
            provider.set_responses(["Response"])
            await manager.complete(system_prompt_override="Injected system prompt")

            last_call = provider.get_last_call()
            messages = last_call["messages"]
            assert messages[0].role == "system"
            assert messages[0].content == "Injected system prompt"

            await provider.close()


class TestToolsParameter:
    """Tests for tools parameter on ConversationManager.complete()."""

    @pytest.mark.asyncio
    async def test_tools_forwarded_to_provider(self, manager_with_provider: dict[str, Any]) -> None:
        """tools parameter is forwarded to the LLM provider."""
        manager = manager_with_provider["manager"]
        provider: EchoProvider = manager_with_provider["provider"]

        await manager.add_message(role="user", content="Search for something")
        provider.set_responses(["I'll search for that"])

        # Create a minimal tool-like object
        class MockTool:
            name = "search"
            description = "Search for information"
            schema = {"type": "object", "properties": {"query": {"type": "string"}}}

        await manager.complete(tools=[MockTool()])

        last_call = provider.get_last_call()
        assert last_call is not None
        # EchoProvider records tools in call history
        assert last_call.get("tools") is not None
        assert len(last_call["tools"]) == 1

    @pytest.mark.asyncio
    async def test_no_tools_passes_none(self, manager_with_provider: dict[str, Any]) -> None:
        """Without tools, None is passed to the provider."""
        manager = manager_with_provider["manager"]
        provider: EchoProvider = manager_with_provider["provider"]

        await manager.add_message(role="user", content="Hello")
        provider.set_responses(["Hi"])
        await manager.complete()

        last_call = provider.get_last_call()
        assert last_call.get("tools") is None

    @pytest.mark.asyncio
    async def test_both_override_and_tools(self, manager_with_provider: dict[str, Any]) -> None:
        """system_prompt_override and tools can be used together."""
        manager = manager_with_provider["manager"]
        provider: EchoProvider = manager_with_provider["provider"]

        await manager.add_message(role="user", content="Help me")
        provider.set_responses(["Using tools with override"])

        class MockTool:
            name = "helper"
            description = "Help with tasks"
            schema = {"type": "object", "properties": {}}

        await manager.complete(
            system_prompt_override="You are a wizard with tools",
            tools=[MockTool()],
        )

        last_call = provider.get_last_call()
        messages = last_call["messages"]
        system_msgs = [m for m in messages if m.role == "system"]
        assert system_msgs[0].content == "You are a wizard with tools"
        assert last_call.get("tools") is not None

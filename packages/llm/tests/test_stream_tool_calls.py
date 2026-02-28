"""Tests for streaming tool call support.

Validates that:
- LLMStreamResponse carries tool_calls on final chunks
- EchoProvider.stream_complete() forwards tools and carries tool_calls
- EchoProvider.stream_complete() handles empty content (tool-call-only responses)
- ConversationManager.stream_complete() passes tools to the provider
- ConversationManager.stream_complete() preserves tool_calls in the conversation tree
- ConversationManager.complete() still works (regression)
- Both paths produce identical conversation tree state for the same tool call response
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import LLMConfig, LLMStreamResponse
from dataknobs_llm.llm.base import ToolCall
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_llm.testing import text_response, tool_call_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_config() -> dict:
    return {"provider": "echo", "model": "echo-test", "options": {"echo_prefix": ""}}


@pytest.fixture
def provider(echo_config: dict) -> EchoProvider:
    return EchoProvider(echo_config)


def _create_test_prompts(prompt_dir: Path) -> None:
    system_dir = prompt_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "test.yaml").write_text(
        yaml.dump({"template": "You are a test assistant"})
    )


@pytest.fixture
async def manager_components():
    """Yield (llm, storage, builder) for ConversationManager tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        _create_test_prompts(prompt_dir)

        config = LLMConfig(
            provider="echo",
            model="echo-test",
            options={"echo_prefix": ""},
        )
        llm = EchoProvider(config)
        library = FileSystemPromptLibrary(prompt_dir)
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        yield {"llm": llm, "builder": builder, "storage": storage}
        await llm.close()


async def _make_manager(components: dict) -> ConversationManager:
    """Helper to create a ConversationManager and add one user message."""
    mgr = await ConversationManager.create(
        llm=components["llm"],
        prompt_builder=components["builder"],
        storage=components["storage"],
        system_prompt_name="test",
    )
    await mgr.add_message(role="user", content="Hello")
    return mgr


# ---------------------------------------------------------------------------
# 1. LLMStreamResponse accepts tool_calls field
# ---------------------------------------------------------------------------


class TestLLMStreamResponseToolCalls:
    def test_defaults_to_none(self) -> None:
        chunk = LLMStreamResponse(delta="hi")
        assert chunk.tool_calls is None

    def test_accepts_tool_calls(self) -> None:
        tc = ToolCall(name="search", parameters={"q": "test"}, id="tc-1")
        chunk = LLMStreamResponse(
            delta="",
            is_final=True,
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1
        assert chunk.tool_calls[0].name == "search"
        assert chunk.tool_calls[0].parameters == {"q": "test"}


# ---------------------------------------------------------------------------
# 2. EchoProvider.stream_complete() carries tool_calls on final chunk
# ---------------------------------------------------------------------------


class TestEchoStreamToolCalls:
    @pytest.mark.asyncio
    async def test_tool_calls_on_final_chunk(self, provider: EchoProvider) -> None:
        """Streaming a tool_call_response should carry tool_calls on the final chunk."""
        provider.set_responses([
            tool_call_response("get_weather", {"city": "NYC"}, content="Checking weather"),
        ])

        chunks: list[LLMStreamResponse] = []
        async for chunk in provider.stream_complete("What's the weather?"):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Only the last chunk should be final
        assert chunks[-1].is_final is True
        assert chunks[-1].tool_calls is not None
        assert chunks[-1].tool_calls[0].name == "get_weather"
        assert chunks[-1].tool_calls[0].parameters == {"city": "NYC"}

        # Non-final chunks must NOT carry tool_calls
        for c in chunks[:-1]:
            assert c.tool_calls is None

    @pytest.mark.asyncio
    async def test_finish_reason_preserved(self, provider: EchoProvider) -> None:
        """finish_reason from the underlying response should propagate."""
        provider.set_responses([
            tool_call_response("search", {"q": "test"}),
        ])

        chunks = [c async for c in provider.stream_complete("search")]
        final = chunks[-1]
        # tool_call_response sets finish_reason="tool_calls"
        assert final.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_no_tool_calls_when_text_only(self, provider: EchoProvider) -> None:
        """Regular text responses should NOT have tool_calls."""
        provider.set_responses([text_response("Just a text answer")])

        chunks = [c async for c in provider.stream_complete("Hello")]
        assert chunks[-1].is_final is True
        assert chunks[-1].tool_calls is None


# ---------------------------------------------------------------------------
# 3. EchoProvider.stream_complete() handles empty content (tool call only)
# ---------------------------------------------------------------------------


class TestEchoStreamEmptyContent:
    @pytest.mark.asyncio
    async def test_empty_content_tool_call(self, provider: EchoProvider) -> None:
        """A tool_call_response with content='' should yield exactly one final chunk."""
        provider.set_responses([
            tool_call_response("do_thing", {"x": 1}, content=""),
        ])

        chunks = [c async for c in provider.stream_complete("trigger")]
        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].delta == ""
        assert chunks[0].tool_calls is not None
        assert chunks[0].tool_calls[0].name == "do_thing"


# ---------------------------------------------------------------------------
# 4. EchoProvider.stream_complete() forwards tools parameter
# ---------------------------------------------------------------------------


class TestEchoStreamToolsForwarding:
    @pytest.mark.asyncio
    async def test_tools_recorded_in_call_history(self, provider: EchoProvider) -> None:
        """tools= parameter should be forwarded to complete() and visible in call history."""
        provider.set_responses([text_response("ok")])

        class FakeTool:
            name = "my_tool"
            description = "A test tool"
            schema = {"type": "object"}

        _ = [c async for c in provider.stream_complete("test", tools=[FakeTool()])]

        last_call = provider.get_last_call()
        assert last_call is not None
        assert last_call["tools"] is not None
        assert len(last_call["tools"]) == 1


# ---------------------------------------------------------------------------
# 5. ConversationManager.stream_complete() passes tools to provider
# ---------------------------------------------------------------------------


class TestManagerStreamToolPassing:
    @pytest.mark.asyncio
    async def test_tools_reach_provider(self, manager_components: dict) -> None:
        llm: EchoProvider = manager_components["llm"]
        llm.set_responses([text_response("ok")])
        mgr = await _make_manager(manager_components)

        class FakeTool:
            name = "my_tool"
            description = "A test tool"
            schema = {"type": "object"}

        _ = [c async for c in mgr.stream_complete(tools=[FakeTool()])]

        last_call = llm.get_last_call()
        assert last_call is not None
        assert last_call["tools"] is not None
        assert last_call["tools"][0].name == "my_tool"


# ---------------------------------------------------------------------------
# 6. ConversationManager.stream_complete() preserves tool_calls in tree
# ---------------------------------------------------------------------------


class TestManagerStreamTreePreservation:
    @pytest.mark.asyncio
    async def test_tool_calls_in_tree_metadata(self, manager_components: dict) -> None:
        llm: EchoProvider = manager_components["llm"]
        llm.set_responses([
            tool_call_response("search", {"query": "python"}, content="searching"),
        ])
        mgr = await _make_manager(manager_components)

        _ = [c async for c in mgr.stream_complete()]

        # The assistant node should have tool_calls in metadata
        node = mgr.state.get_current_node()
        assert node is not None
        meta = node.data.metadata
        assert "tool_calls" in meta
        assert meta["tool_calls"][0]["name"] == "search"
        assert meta["tool_calls"][0]["parameters"] == {"query": "python"}

    @pytest.mark.asyncio
    async def test_tool_calls_on_assistant_message(
        self, manager_components: dict
    ) -> None:
        llm: EchoProvider = manager_components["llm"]
        llm.set_responses([
            tool_call_response("calc", {"expr": "2+2"}, content="calculating"),
        ])
        mgr = await _make_manager(manager_components)

        _ = [c async for c in mgr.stream_complete()]

        # The LLMMessage on the assistant node should carry tool_calls
        node = mgr.state.get_current_node()
        assert node is not None
        msg = node.data.message
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].name == "calc"


# ---------------------------------------------------------------------------
# 7. ConversationManager.complete() still works (regression)
# ---------------------------------------------------------------------------


class TestManagerCompleteRegression:
    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self, manager_components: dict) -> None:
        llm: EchoProvider = manager_components["llm"]
        llm.set_responses([
            tool_call_response("search", {"q": "test"}, content="result"),
        ])
        mgr = await _make_manager(manager_components)

        response = await mgr.complete()

        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "search"

        # Verify tree metadata
        node = mgr.state.get_current_node()
        assert node is not None
        assert "tool_calls" in node.data.metadata

    @pytest.mark.asyncio
    async def test_complete_text_only(self, manager_components: dict) -> None:
        llm: EchoProvider = manager_components["llm"]
        llm.set_responses([text_response("just text")])
        mgr = await _make_manager(manager_components)

        response = await mgr.complete()

        assert response.content == "just text"
        assert response.tool_calls is None


# ---------------------------------------------------------------------------
# 8. Both paths produce identical tree state for the same tool call response
# ---------------------------------------------------------------------------


class TestCompletionPathParity:
    @pytest.mark.asyncio
    async def test_tree_state_matches(self, manager_components: dict) -> None:
        """complete() and stream_complete() should produce identical tree metadata."""
        llm: EchoProvider = manager_components["llm"]

        # --- complete() path ---
        llm.set_responses([
            tool_call_response(
                "search", {"q": "test"}, content="found", tool_id="tc-fixed"
            ),
        ])
        mgr1 = await _make_manager(manager_components)
        await mgr1.complete()
        node1 = mgr1.state.get_current_node()
        assert node1 is not None
        meta1 = node1.data.metadata

        # --- stream_complete() path --- (fresh manager)
        llm.set_responses([
            tool_call_response(
                "search", {"q": "test"}, content="found", tool_id="tc-fixed"
            ),
        ])
        # Need a new storage to avoid collision
        manager_components["storage"] = DataknobsConversationStorage(
            AsyncMemoryDatabase()
        )
        mgr2 = await _make_manager(manager_components)
        _ = [c async for c in mgr2.stream_complete()]
        node2 = mgr2.state.get_current_node()
        assert node2 is not None
        meta2 = node2.data.metadata

        # Compare the key metadata fields
        assert meta1["tool_calls"] == meta2["tool_calls"]
        assert meta1["finish_reason"] == meta2["finish_reason"]
        assert node1.data.message.tool_calls[0].name == node2.data.message.tool_calls[0].name
        assert node1.data.message.tool_calls[0].parameters == node2.data.message.tool_calls[0].parameters


# ---------------------------------------------------------------------------
# 9. LLMStreamResponse model field parity
# ---------------------------------------------------------------------------


class TestStreamResponseModelField:
    def test_model_defaults_to_none(self) -> None:
        chunk = LLMStreamResponse(delta="hi")
        assert chunk.model is None

    def test_model_set_on_final_chunk(self) -> None:
        chunk = LLMStreamResponse(
            delta="", is_final=True, finish_reason="stop", model="gpt-4"
        )
        assert chunk.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_echo_sets_model_on_final_chunk(self, provider: EchoProvider) -> None:
        """EchoProvider should set model on the final streaming chunk."""
        provider.set_responses([text_response("hello", model="custom-model")])

        chunks = [c async for c in provider.stream_complete("test")]
        # Non-final chunks should not have model
        for c in chunks[:-1]:
            assert c.model is None
        # Final chunk must carry the model
        assert chunks[-1].is_final is True
        assert chunks[-1].model == "custom-model"

    @pytest.mark.asyncio
    async def test_echo_empty_content_sets_model(self, provider: EchoProvider) -> None:
        """Empty-content responses should also carry model on the final chunk."""
        provider.set_responses([
            tool_call_response("do_thing", {"x": 1}, content=""),
        ])

        chunks = [c async for c in provider.stream_complete("trigger")]
        assert len(chunks) == 1
        assert chunks[0].model is not None

    @pytest.mark.asyncio
    async def test_manager_uses_model_from_stream(
        self, manager_components: dict
    ) -> None:
        """ConversationManager should use the model from the final chunk."""
        llm: EchoProvider = manager_components["llm"]
        llm.set_responses([
            text_response("hi", model="streamed-model"),
        ])
        mgr = await _make_manager(manager_components)

        _ = [c async for c in mgr.stream_complete()]

        node = mgr.state.get_current_node()
        assert node is not None
        assert node.data.metadata["model"] == "streamed-model"

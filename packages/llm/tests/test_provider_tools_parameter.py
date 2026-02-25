"""Tests for tools parameter on provider complete() methods.

Verifies that the tools parameter is consistently handled across all
LLM providers as part of the interface disconnect fix.
"""

from typing import Any

import pytest

from dataknobs_llm.llm import LLMConfig, LLMMessage
from dataknobs_llm.llm.providers.echo import EchoProvider


class MockTool:
    """Minimal tool-like object for testing."""

    def __init__(self, name: str = "test_tool") -> None:
        self.name = name
        self.description = f"A tool called {name}"
        self.schema = {"type": "object", "properties": {"arg": {"type": "string"}}}


class TestEchoProviderTools:
    """Test that EchoProvider records tools in call history."""

    @pytest.mark.asyncio
    async def test_tools_recorded_in_history(self) -> None:
        """EchoProvider records tools passed to complete()."""
        config = LLMConfig(provider="echo", model="echo-test", options={"echo_prefix": ""})
        provider = EchoProvider(config)
        await provider.initialize()

        provider.set_responses(["Response with tools"])
        tools = [MockTool("search"), MockTool("calculator")]

        await provider.complete(
            [LLMMessage(role="user", content="Hello")],
            tools=tools,
        )

        last_call = provider.get_last_call()
        assert last_call is not None
        assert last_call["tools"] is not None
        assert len(last_call["tools"]) == 2
        assert last_call["tools"][0].name == "search"
        assert last_call["tools"][1].name == "calculator"

        await provider.close()

    @pytest.mark.asyncio
    async def test_no_tools_recorded_as_none(self) -> None:
        """EchoProvider records None when no tools are passed."""
        config = LLMConfig(provider="echo", model="echo-test", options={"echo_prefix": ""})
        provider = EchoProvider(config)
        await provider.initialize()

        provider.set_responses(["Response without tools"])
        await provider.complete([LLMMessage(role="user", content="Hello")])

        last_call = provider.get_last_call()
        assert last_call["tools"] is None

        await provider.close()

    @pytest.mark.asyncio
    async def test_tools_with_config_overrides(self) -> None:
        """EchoProvider handles both config_overrides and tools."""
        config = LLMConfig(provider="echo", model="echo-test", options={"echo_prefix": ""})
        provider = EchoProvider(config)
        await provider.initialize()

        provider.set_responses(["Combined response"])
        await provider.complete(
            [LLMMessage(role="user", content="Hello")],
            config_overrides={"temperature": 0.5},
            tools=[MockTool()],
        )

        last_call = provider.get_last_call()
        assert last_call["tools"] is not None
        assert last_call["config_overrides"] == {"temperature": 0.5}

        await provider.close()


class TestHuggingFaceProviderTools:
    """Test that HuggingFaceProvider raises on tools."""

    @pytest.mark.asyncio
    async def test_tools_raises_not_supported(self) -> None:
        """HuggingFaceProvider raises ToolsNotSupportedError when tools are passed."""
        from dataknobs_llm.exceptions import ToolsNotSupportedError
        from dataknobs_llm.llm.providers.huggingface import HuggingFaceProvider

        config = LLMConfig(
            provider="huggingface",
            model="test-model",
            api_base="http://localhost:8080",
        )
        provider = HuggingFaceProvider(config)

        with pytest.raises(ToolsNotSupportedError):
            await provider.complete(
                [LLMMessage(role="user", content="Hello")],
                tools=[MockTool()],
            )

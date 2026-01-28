"""Tests for testing utilities module."""

import json

import pytest

from dataknobs_llm.testing import (
    text_response,
    tool_call_response,
    multi_tool_response,
    extraction_response,
    ResponseSequenceBuilder,
)
from dataknobs_llm.llm.providers import EchoProvider


class TestTextResponse:
    """Tests for text_response builder."""

    def test_basic_text_response(self) -> None:
        """Create simple text response."""
        response = text_response("Hello, world!")

        assert response.content == "Hello, world!"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.tool_calls is None

    def test_text_response_with_options(self) -> None:
        """Create text response with custom options."""
        response = text_response(
            "Hello!",
            model="custom-model",
            finish_reason="length",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        assert response.model == "custom-model"
        assert response.finish_reason == "length"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] == 10

    def test_text_response_with_metadata(self) -> None:
        """Create text response with metadata."""
        response = text_response(
            "Hello!",
            metadata={"request_id": "abc123"},
        )

        assert response.metadata == {"request_id": "abc123"}

    def test_text_response_empty_content(self) -> None:
        """Create text response with empty content."""
        response = text_response("")

        assert response.content == ""
        assert response.finish_reason == "stop"


class TestToolCallResponse:
    """Tests for tool_call_response builder."""

    def test_basic_tool_call(self) -> None:
        """Create single tool call response."""
        response = tool_call_response("preview_config", {"format": "yaml"})

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "preview_config"
        assert response.tool_calls[0].parameters == {"format": "yaml"}
        assert response.finish_reason == "tool_calls"

    def test_tool_call_with_id(self) -> None:
        """Create tool call with specific ID."""
        response = tool_call_response("test", {}, tool_id="my-id")

        assert response.tool_calls is not None
        assert response.tool_calls[0].id == "my-id"

    def test_tool_call_auto_generates_id(self) -> None:
        """Tool call IDs are auto-generated."""
        response = tool_call_response("test", {})

        assert response.tool_calls is not None
        assert response.tool_calls[0].id is not None
        assert response.tool_calls[0].id.startswith("tc-")

    def test_tool_call_empty_arguments(self) -> None:
        """Create tool call with no arguments."""
        response = tool_call_response("no_args_tool")

        assert response.tool_calls is not None
        assert response.tool_calls[0].parameters == {}

    def test_tool_call_with_content(self) -> None:
        """Create tool call with accompanying text content."""
        response = tool_call_response(
            "preview_config",
            {},
            content="Let me preview that for you.",
        )

        assert response.content == "Let me preview that for you."
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

    def test_multiple_tool_calls(self) -> None:
        """Create response with multiple tool calls."""
        response = tool_call_response(
            "first_tool",
            {"a": 1},
            additional_tools=[
                ("second_tool", {"b": 2}),
                ("third_tool", {}),
            ],
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 3
        assert response.tool_calls[0].name == "first_tool"
        assert response.tool_calls[1].name == "second_tool"
        assert response.tool_calls[2].name == "third_tool"
        assert response.tool_calls[1].parameters == {"b": 2}


class TestMultiToolResponse:
    """Tests for multi_tool_response builder."""

    def test_multi_tool_response(self) -> None:
        """Create response with multiple tools."""
        response = multi_tool_response([
            ("preview", {}),
            ("validate", {"strict": True}),
        ])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "preview"
        assert response.tool_calls[1].parameters == {"strict": True}
        assert response.finish_reason == "tool_calls"

    def test_multi_tool_response_empty_list(self) -> None:
        """Create response with empty tool list."""
        response = multi_tool_response([])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 0

    def test_multi_tool_response_with_content(self) -> None:
        """Create multi-tool response with content."""
        response = multi_tool_response(
            [("tool1", {}), ("tool2", {})],
            content="Executing both tools",
        )

        assert response.content == "Executing both tools"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 2


class TestExtractionResponse:
    """Tests for extraction_response builder."""

    def test_extraction_response(self) -> None:
        """Create extraction response with JSON content."""
        response = extraction_response({"name": "Test", "count": 42})

        data = json.loads(response.content)
        assert data == {"name": "Test", "count": 42}
        assert response.finish_reason == "stop"

    def test_extraction_response_nested_data(self) -> None:
        """Create extraction response with nested data."""
        response = extraction_response({
            "bot": {
                "name": "Math Tutor",
                "settings": {"difficulty": 5},
            },
            "tags": ["math", "education"],
        })

        data = json.loads(response.content)
        assert data["bot"]["name"] == "Math Tutor"
        assert data["tags"] == ["math", "education"]

    def test_extraction_response_empty_dict(self) -> None:
        """Create extraction response with empty data."""
        response = extraction_response({})

        data = json.loads(response.content)
        assert data == {}


class TestResponseSequenceBuilder:
    """Tests for ResponseSequenceBuilder."""

    def test_build_sequence(self) -> None:
        """Build a sequence of responses."""
        responses = (
            ResponseSequenceBuilder()
            .add_text("Hello")
            .add_tool_call("test_tool", {"arg": 1})
            .add_text("Done")
            .build()
        )

        assert len(responses) == 3
        assert responses[0].content == "Hello"
        assert responses[1].tool_calls is not None
        assert responses[1].tool_calls[0].name == "test_tool"
        assert responses[2].content == "Done"

    def test_build_empty_sequence(self) -> None:
        """Build empty sequence."""
        responses = ResponseSequenceBuilder().build()

        assert responses == []

    def test_custom_model(self) -> None:
        """Builder uses custom model for all responses."""
        responses = (
            ResponseSequenceBuilder(model="custom-model")
            .add_text("Hello")
            .add_tool_call("test", {})
            .build()
        )

        assert responses[0].model == "custom-model"
        assert responses[1].model == "custom-model"

    def test_add_multi_tool(self) -> None:
        """Add multi-tool response via builder."""
        responses = (
            ResponseSequenceBuilder()
            .add_multi_tool([
                ("tool1", {"x": 1}),
                ("tool2", {"y": 2}),
            ])
            .build()
        )

        assert len(responses) == 1
        assert responses[0].tool_calls is not None
        assert len(responses[0].tool_calls) == 2

    def test_add_extraction(self) -> None:
        """Add extraction response via builder."""
        responses = (
            ResponseSequenceBuilder()
            .add_extraction({"name": "Test Bot"})
            .build()
        )

        assert len(responses) == 1
        data = json.loads(responses[0].content)
        assert data["name"] == "Test Bot"

    def test_add_custom_response(self) -> None:
        """Add custom LLMResponse via builder."""
        custom = text_response("Custom!", model="special-model")

        responses = (
            ResponseSequenceBuilder()
            .add(custom)
            .add_text("Regular")
            .build()
        )

        assert len(responses) == 2
        assert responses[0].model == "special-model"
        assert responses[1].model == "test-model"

    def test_configure_provider(self) -> None:
        """Builder can configure an EchoProvider."""
        provider = EchoProvider({"provider": "echo", "model": "test"})

        (
            ResponseSequenceBuilder()
            .add_text("Response 1")
            .add_text("Response 2")
            .configure(provider)
        )

        # Responses should be set (accessing internal for test)
        assert len(provider._response_queue) == 2


class TestEchoProviderIntegration:
    """Integration tests with EchoProvider."""

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self) -> None:
        """EchoProvider returns tool call then text."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            tool_call_response("my_tool", {"x": 1}),
            text_response("Tool executed!"),
        ])

        # First call returns tool call
        response1 = await provider.complete("Do something")
        assert response1.tool_calls is not None
        assert response1.tool_calls[0].name == "my_tool"

        # Second call returns text
        response2 = await provider.complete("Continue")
        assert response2.content == "Tool executed!"
        assert response2.tool_calls is None

    @pytest.mark.asyncio
    async def test_extraction_sequence(self) -> None:
        """Test extraction response in sequence."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            extraction_response({"name": "Test Bot"}),
        ])

        response = await provider.complete("Extract data")
        data = json.loads(response.content)

        assert data["name"] == "Test Bot"

    @pytest.mark.asyncio
    async def test_complex_wizard_sequence(self) -> None:
        """Test a realistic wizard-like sequence."""
        provider = EchoProvider({"provider": "echo", "model": "test"})

        # Simulate: extraction -> tool call -> text
        (
            ResponseSequenceBuilder()
            .add_extraction({"stage": "identity", "domain_id": "math-tutor"})
            .add_tool_call("preview_config", {"format": "yaml"})
            .add_text("Configuration complete!")
            .configure(provider)
        )

        # Step 1: Extraction
        resp1 = await provider.complete("Configure identity")
        data = json.loads(resp1.content)
        assert data["stage"] == "identity"

        # Step 2: Tool call
        resp2 = await provider.complete("Show preview")
        assert resp2.tool_calls is not None
        assert resp2.tool_calls[0].name == "preview_config"

        # Step 3: Completion message
        resp3 = await provider.complete("Done")
        assert resp3.content == "Configuration complete!"

    @pytest.mark.asyncio
    async def test_multi_tool_response_integration(self) -> None:
        """Test multi-tool response with EchoProvider."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            multi_tool_response([
                ("preview_config", {}),
                ("validate_config", {"strict": True}),
            ]),
            text_response("All done!"),
        ])

        # First call returns multiple tools
        response1 = await provider.complete("Preview and validate")
        assert response1.tool_calls is not None
        assert len(response1.tool_calls) == 2
        assert response1.tool_calls[0].name == "preview_config"
        assert response1.tool_calls[1].name == "validate_config"

        # Second call returns text
        response2 = await provider.complete("Continue")
        assert response2.content == "All done!"

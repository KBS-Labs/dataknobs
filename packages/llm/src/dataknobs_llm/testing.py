"""Testing utilities for dataknobs-llm.

This module provides convenience builders for creating test LLM responses,
designed for use with EchoProvider in unit and integration tests.

Example:
    ```python
    from dataknobs_llm.llm.providers import EchoProvider
    from dataknobs_llm.testing import tool_call_response, text_response

    provider = EchoProvider({"provider": "echo", "model": "test"})
    provider.set_responses([
        tool_call_response("preview_config", {"format": "yaml"}),
        text_response("Here's your config preview!")
    ])
    ```
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from .llm.base import LLMResponse, ToolCall

if TYPE_CHECKING:
    from .llm.providers.echo import EchoProvider


def text_response(
    content: str,
    *,
    model: str = "test-model",
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Create a simple text LLMResponse.

    Args:
        content: Response text content
        model: Model identifier (default: "test-model")
        finish_reason: Why generation stopped (default: "stop")
        usage: Optional token usage dict
        metadata: Optional metadata dict

    Returns:
        LLMResponse with text content

    Example:
        >>> response = text_response("Hello, world!")
        >>> response.content
        'Hello, world!'
    """
    return LLMResponse(
        content=content,
        model=model,
        finish_reason=finish_reason,
        usage=usage,
        metadata=metadata or {},
    )


def tool_call_response(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    tool_id: str | None = None,
    content: str = "",
    model: str = "test-model",
    additional_tools: list[tuple[str, dict[str, Any]]] | None = None,
) -> LLMResponse:
    """Create an LLMResponse with tool call(s).

    Args:
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool (default: {})
        tool_id: Unique ID for the tool call (auto-generated if not provided)
        content: Optional text content alongside tool call
        model: Model identifier (default: "test-model")
        additional_tools: Additional tool calls as (name, args) tuples

    Returns:
        LLMResponse with tool_calls populated

    Example:
        >>> response = tool_call_response("get_weather", {"city": "NYC"})
        >>> response.tool_calls[0].name
        'get_weather'
        >>> response.tool_calls[0].parameters
        {'city': 'NYC'}

        # Multiple tool calls
        >>> response = tool_call_response(
        ...     "preview_config", {},
        ...     additional_tools=[("validate_config", {})]
        ... )
        >>> len(response.tool_calls)
        2
    """
    tools = [
        ToolCall(
            name=tool_name,
            parameters=arguments or {},
            id=tool_id or f"tc-{uuid.uuid4().hex[:8]}",
        )
    ]

    if additional_tools:
        for name, args in additional_tools:
            tools.append(
                ToolCall(
                    name=name,
                    parameters=args,
                    id=f"tc-{uuid.uuid4().hex[:8]}",
                )
            )

    return LLMResponse(
        content=content,
        model=model,
        finish_reason="tool_calls",
        tool_calls=tools,
    )


def multi_tool_response(
    tools: list[tuple[str, dict[str, Any]]],
    *,
    content: str = "",
    model: str = "test-model",
) -> LLMResponse:
    """Create an LLMResponse with multiple tool calls.

    Args:
        tools: List of (tool_name, arguments) tuples
        content: Optional text content alongside tool calls
        model: Model identifier (default: "test-model")

    Returns:
        LLMResponse with multiple tool_calls

    Example:
        >>> response = multi_tool_response([
        ...     ("preview_config", {}),
        ...     ("validate_config", {"strict": True}),
        ... ])
        >>> len(response.tool_calls)
        2
    """
    tool_calls = [
        ToolCall(
            name=name,
            parameters=args,
            id=f"tc-{uuid.uuid4().hex[:8]}",
        )
        for name, args in tools
    ]

    return LLMResponse(
        content=content,
        model=model,
        finish_reason="tool_calls",
        tool_calls=tool_calls,
    )


def extraction_response(
    data: dict[str, Any],
    *,
    model: str = "test-model",
) -> LLMResponse:
    """Create an LLMResponse for schema extraction.

    The data is JSON-encoded as the response content, mimicking
    how extraction LLMs return structured data.

    Args:
        data: Extracted data dict
        model: Model identifier (default: "test-model")

    Returns:
        LLMResponse with JSON content

    Example:
        >>> response = extraction_response({"name": "Math Tutor", "level": 5})
        >>> import json
        >>> json.loads(response.content)
        {'name': 'Math Tutor', 'level': 5}
    """
    return LLMResponse(
        content=json.dumps(data),
        model=model,
        finish_reason="stop",
    )


class ResponseSequenceBuilder:
    """Builder for creating sequences of LLM responses.

    Provides a fluent API for building test response sequences,
    useful for testing multi-turn conversations and ReAct loops.

    Example:
        ```python
        responses = (
            ResponseSequenceBuilder()
            .add_tool_call("list_templates", {})
            .add_tool_call("get_template", {"name": "quiz"})
            .add_text("I found the quiz template!")
            .build()
        )

        provider.set_responses(responses)
        ```
    """

    def __init__(self, model: str = "test-model"):
        """Initialize builder.

        Args:
            model: Model identifier for all responses
        """
        self._model = model
        self._responses: list[LLMResponse] = []

    def add_text(self, content: str) -> ResponseSequenceBuilder:
        """Add a text response to the sequence.

        Args:
            content: Response text

        Returns:
            Self for chaining
        """
        self._responses.append(text_response(content, model=self._model))
        return self

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ResponseSequenceBuilder:
        """Add a tool call response to the sequence.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Self for chaining
        """
        self._responses.append(
            tool_call_response(tool_name, arguments, model=self._model)
        )
        return self

    def add_multi_tool(
        self,
        tools: list[tuple[str, dict[str, Any]]],
    ) -> ResponseSequenceBuilder:
        """Add a multi-tool call response to the sequence.

        Args:
            tools: List of (name, args) tuples

        Returns:
            Self for chaining
        """
        self._responses.append(multi_tool_response(tools, model=self._model))
        return self

    def add_extraction(
        self,
        data: dict[str, Any],
    ) -> ResponseSequenceBuilder:
        """Add an extraction response to the sequence.

        Args:
            data: Extracted data dict

        Returns:
            Self for chaining
        """
        self._responses.append(extraction_response(data, model=self._model))
        return self

    def add(self, response: LLMResponse) -> ResponseSequenceBuilder:
        """Add a custom LLMResponse to the sequence.

        Args:
            response: Custom response object

        Returns:
            Self for chaining
        """
        self._responses.append(response)
        return self

    def build(self) -> list[LLMResponse]:
        """Build and return the response sequence.

        Returns:
            List of LLMResponse objects
        """
        return list(self._responses)

    def configure(self, provider: EchoProvider) -> EchoProvider:
        """Configure an EchoProvider with this sequence.

        Args:
            provider: EchoProvider to configure

        Returns:
            The configured provider
        """
        provider.set_responses(self._responses)
        return provider

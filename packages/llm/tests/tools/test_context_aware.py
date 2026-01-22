"""Tests for ContextAwareTool and ContextEnhancedTool."""

from typing import Any

import pytest

from dataknobs_llm.tools import Tool
from dataknobs_llm.tools.context import ToolExecutionContext, WizardStateSnapshot
from dataknobs_llm.tools.context_aware import (
    ContextAwareTool,
    ContextEnhancedTool,
    default_wizard_data_injector,
)


class SampleContextAwareTool(ContextAwareTool):
    """Sample tool for testing that uses context."""

    def __init__(self) -> None:
        super().__init__(
            name="sample_context_tool",
            description="A sample tool that uses context",
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input value"},
            },
            "required": ["input"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        input: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute and return context info for testing."""
        return {
            "input": input,
            "conversation_id": context.conversation_id,
            "user_id": context.user_id,
            "has_wizard_state": context.wizard_state is not None,
            "wizard_data": (
                context.wizard_state.collected_data
                if context.wizard_state
                else {}
            ),
        }


class SampleRegularTool(Tool):
    """Sample regular tool for testing wrapping."""

    def __init__(self) -> None:
        super().__init__(
            name="sample_regular_tool",
            description="A sample regular tool",
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "wizard_data": {"type": "object"},
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        wizard_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute and return what was received."""
        return {
            "query": query,
            "wizard_data": wizard_data or {},
            "extra_kwargs": kwargs,
        }


class TestContextAwareTool:
    """Tests for ContextAwareTool base class."""

    @pytest.mark.asyncio
    async def test_execute_with_context_provided(self) -> None:
        """Test that execute passes context to execute_with_context."""
        tool = SampleContextAwareTool()

        wizard_state = WizardStateSnapshot(
            current_stage="configure",
            collected_data={"domain_id": "test-bot"},
        )
        context = ToolExecutionContext(
            conversation_id="conv-123",
            user_id="user-456",
            wizard_state=wizard_state,
        )

        result = await tool.execute(input="test-input", _context=context)

        assert result["input"] == "test-input"
        assert result["conversation_id"] == "conv-123"
        assert result["user_id"] == "user-456"
        assert result["has_wizard_state"] is True
        assert result["wizard_data"]["domain_id"] == "test-bot"

    @pytest.mark.asyncio
    async def test_execute_without_context(self) -> None:
        """Test that execute works without context (uses empty context)."""
        tool = SampleContextAwareTool()

        result = await tool.execute(input="test-input")

        assert result["input"] == "test-input"
        assert result["conversation_id"] is None
        assert result["user_id"] is None
        assert result["has_wizard_state"] is False
        assert result["wizard_data"] == {}

    @pytest.mark.asyncio
    async def test_context_extracted_from_kwargs(self) -> None:
        """Test that _context is properly extracted from kwargs."""
        tool = SampleContextAwareTool()

        context = ToolExecutionContext(conversation_id="conv-abc")

        # _context should be extracted, not passed to execute_with_context
        result = await tool.execute(input="test", _context=context)

        assert result["conversation_id"] == "conv-abc"

    def test_tool_properties(self) -> None:
        """Test that tool properties are correctly set."""
        tool = SampleContextAwareTool()

        assert tool.name == "sample_context_tool"
        assert tool.description == "A sample tool that uses context"
        assert tool.schema["type"] == "object"
        assert "input" in tool.schema["properties"]


class TestContextEnhancedTool:
    """Tests for ContextEnhancedTool wrapper."""

    @pytest.mark.asyncio
    async def test_wraps_tool_without_injector(self) -> None:
        """Test that wrapper works without injector."""
        inner_tool = SampleRegularTool()
        wrapper = ContextEnhancedTool(inner_tool)

        context = ToolExecutionContext(conversation_id="conv-123")
        result = await wrapper.execute(query="test-query", _context=context)

        assert result["query"] == "test-query"
        assert result["wizard_data"] == {}

    @pytest.mark.asyncio
    async def test_wraps_tool_with_injector(self) -> None:
        """Test that wrapper injects values from context."""
        inner_tool = SampleRegularTool()

        def custom_injector(context: ToolExecutionContext) -> dict[str, Any]:
            return {"wizard_data": {"injected": True, "conv_id": context.conversation_id}}

        wrapper = ContextEnhancedTool(inner_tool, context_injector=custom_injector)

        context = ToolExecutionContext(conversation_id="conv-123")
        result = await wrapper.execute(query="test-query", _context=context)

        assert result["query"] == "test-query"
        assert result["wizard_data"]["injected"] is True
        assert result["wizard_data"]["conv_id"] == "conv-123"

    @pytest.mark.asyncio
    async def test_explicit_kwargs_override_injected(self) -> None:
        """Test that explicit kwargs are not overridden by injector."""
        inner_tool = SampleRegularTool()

        def custom_injector(context: ToolExecutionContext) -> dict[str, Any]:
            return {"wizard_data": {"from_injector": True}}

        wrapper = ContextEnhancedTool(inner_tool, context_injector=custom_injector)

        context = ToolExecutionContext(conversation_id="conv-123")
        result = await wrapper.execute(
            query="test-query",
            wizard_data={"explicit": True},
            _context=context,
        )

        # Explicit value should be preserved
        assert result["wizard_data"]["explicit"] is True
        assert "from_injector" not in result["wizard_data"]

    def test_wrapper_preserves_tool_properties(self) -> None:
        """Test that wrapper preserves inner tool properties."""
        inner_tool = SampleRegularTool()
        wrapper = ContextEnhancedTool(inner_tool)

        assert wrapper.name == inner_tool.name
        assert wrapper.description == inner_tool.description
        assert wrapper.schema == inner_tool.schema
        assert wrapper.metadata == inner_tool.metadata


class TestDefaultWizardDataInjector:
    """Tests for default_wizard_data_injector function."""

    def test_injects_wizard_data_when_present(self) -> None:
        """Test that wizard data is injected when available."""
        wizard_state = WizardStateSnapshot(
            current_stage="configure",
            collected_data={"domain_id": "test-bot", "name": "Test Bot"},
        )
        context = ToolExecutionContext(wizard_state=wizard_state)

        result = default_wizard_data_injector(context)

        assert "wizard_data" in result
        assert result["wizard_data"]["domain_id"] == "test-bot"
        assert result["wizard_data"]["name"] == "Test Bot"

    def test_returns_empty_when_no_wizard_state(self) -> None:
        """Test that empty dict is returned when no wizard state."""
        context = ToolExecutionContext()

        result = default_wizard_data_injector(context)

        assert result == {}

    @pytest.mark.asyncio
    async def test_integration_with_context_enhanced_tool(self) -> None:
        """Test default injector works with ContextEnhancedTool."""
        inner_tool = SampleRegularTool()
        wrapper = ContextEnhancedTool(
            inner_tool,
            context_injector=default_wizard_data_injector,
        )

        wizard_state = WizardStateSnapshot(
            current_stage="review",
            collected_data={"template": "tutor", "name": "My Bot"},
        )
        context = ToolExecutionContext(wizard_state=wizard_state)

        result = await wrapper.execute(query="preview", _context=context)

        assert result["wizard_data"]["template"] == "tutor"
        assert result["wizard_data"]["name"] == "My Bot"


class TestToolConversionMethods:
    """Tests for tool format conversion methods."""

    def test_to_function_definition(self) -> None:
        """Test OpenAI function definition format."""
        tool = SampleContextAwareTool()
        definition = tool.to_function_definition()

        assert definition["name"] == "sample_context_tool"
        assert definition["description"] == "A sample tool that uses context"
        assert definition["parameters"]["type"] == "object"
        assert "input" in definition["parameters"]["properties"]

    def test_to_anthropic_tool_definition(self) -> None:
        """Test Anthropic tool definition format."""
        tool = SampleContextAwareTool()
        definition = tool.to_anthropic_tool_definition()

        assert definition["name"] == "sample_context_tool"
        assert definition["description"] == "A sample tool that uses context"
        assert definition["input_schema"]["type"] == "object"
        assert "input" in definition["input_schema"]["properties"]

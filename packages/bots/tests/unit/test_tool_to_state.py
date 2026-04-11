"""Tests for tool-to-state integration (item 79).

Verifies:
- ToolCallSpec construction and ProcessResult with pending calls
- End-to-end: extraction → tool execution → result mapping → transition
- Tool error with on_error=skip (state unpolluted, stage stays)
- Tool error with on_error=fail (error flag in state)
- Tool not found (graceful skip)
- Non-dict tool result (maps to first target key)
- Missing params (state key not yet populated → param omitted)
- Config loading (KNOWN_STAGE_FIELDS, _extract_metadata, wizard_builder)
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_llm.tools import Tool

from dataknobs_bots.reasoning.base import ProcessResult, ToolCallSpec
from dataknobs_bots.reasoning.wizard_types import (
    ToolResultMappingEntry,
    WizardTurnHandle,
)
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


# =========================================================================
# Test tools
# =========================================================================


class ProductLookupTool(Tool):
    """Tool that returns a product dict given a query."""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        super().__init__(
            name="product_lookup",
            description="Look up a product by name",
        )
        self._result = result or {
            "product_id": "PROD-123",
            "category": "electronics",
        }

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

    async def execute(self, query: str = "", **_kwargs: Any) -> dict[str, Any]:
        return self._result


class FailingTool(Tool):
    """Tool that always raises an exception."""

    def __init__(self) -> None:
        super().__init__(
            name="failing_tool",
            description="A tool that fails",
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kwargs: Any) -> Any:
        raise RuntimeError("Tool execution failed")


class ScalarTool(Tool):
    """Tool that returns a scalar (non-dict) result."""

    def __init__(self, result: Any = 42) -> None:
        super().__init__(
            name="scalar_tool",
            description="Returns a scalar value",
        )
        self._result = result

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
            },
        }

    async def execute(self, **_kwargs: Any) -> Any:
        return self._result


# =========================================================================
# Type construction tests
# =========================================================================


class TestToolCallSpecConstruction:
    """Verify ToolCallSpec and ProcessResult with pending calls."""

    def test_tool_call_spec_frozen(self) -> None:
        spec = ToolCallSpec(name="my_tool", parameters={"key": "val"})
        assert spec.name == "my_tool"
        assert spec.parameters == {"key": "val"}
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore[misc]

    def test_process_result_with_pending_calls(self) -> None:
        specs = [
            ToolCallSpec(name="t1", parameters={"a": 1}),
            ToolCallSpec(name="t2", parameters={}),
        ]
        result = ProcessResult(
            needs_tool_execution=True,
            pending_tool_calls=specs,
            action="extracted",
        )
        assert result.needs_tool_execution is True
        assert len(result.pending_tool_calls) == 2
        assert result.pending_tool_calls[0].name == "t1"

    def test_process_result_default_empty(self) -> None:
        result = ProcessResult()
        assert result.pending_tool_calls == []
        assert result.needs_tool_execution is False


class TestToolResultMappingEntry:
    """Verify ToolResultMappingEntry type."""

    def test_construction(self) -> None:
        entry = ToolResultMappingEntry(
            tool_name="product_lookup",
            params={"query": "product_name"},
            mapping={"product_id": "product_id"},
        )
        assert entry.tool_name == "product_lookup"
        assert entry.on_error == "skip"

    def test_frozen(self) -> None:
        entry = ToolResultMappingEntry(
            tool_name="t", params={}, mapping={},
        )
        with pytest.raises(AttributeError):
            entry.tool_name = "other"  # type: ignore[misc]


class TestWizardTurnHandleToolMapping:
    """Verify WizardTurnHandle has tool_result_mapping field."""

    def test_default_empty(self) -> None:
        handle = WizardTurnHandle(manager=None, llm=None)
        assert handle.tool_result_mapping == []

    def test_set_mapping(self) -> None:
        handle = WizardTurnHandle(manager=None, llm=None)
        entry = ToolResultMappingEntry(tool_name="t", params={}, mapping={})
        handle.tool_result_mapping = [entry]
        assert len(handle.tool_result_mapping) == 1
        assert handle.tool_result_mapping[0].tool_name == "t"


# =========================================================================
# Config loading tests
# =========================================================================


class TestConfigLoading:
    """Verify tool_result_mapping flows through config loading."""

    def test_known_stage_fields_includes_tool_result_mapping(self) -> None:
        from dataknobs_bots.reasoning.wizard_loader import KNOWN_STAGE_FIELDS
        assert "tool_result_mapping" in KNOWN_STAGE_FIELDS

    def test_extract_metadata_includes_tool_result_mapping(self) -> None:
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        loader = WizardConfigLoader()
        config = {
            "stages": [
                {
                    "name": "lookup",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Enter product",
                    "schema": {
                        "type": "object",
                        "properties": {"product_name": {"type": "string"}},
                    },
                    "tool_result_mapping": [
                        {
                            "tool": "product_lookup",
                            "params": {"query": "product_name"},
                            "mapping": {"product_id": "product_id"},
                        }
                    ],
                }
            ]
        }
        metadata = loader._extract_metadata(config)
        assert "tool_result_mapping" in metadata["lookup"]
        assert len(metadata["lookup"]["tool_result_mapping"]) == 1

    def test_wizard_builder_stage_config_roundtrip(self) -> None:
        from dataknobs_bots.config.wizard_builder import StageConfig

        stage = StageConfig(
            name="lookup",
            prompt="Enter product",
            tool_result_mapping=(
                {
                    "tool": "product_lookup",
                    "params": {"query": "product_name"},
                    "mapping": {"product_id": "product_id"},
                },
            ),
        )
        d = stage.to_dict()
        assert "tool_result_mapping" in d
        assert d["tool_result_mapping"][0]["tool"] == "product_lookup"

    def test_wizard_builder_from_dict_roundtrip(self) -> None:
        from dataknobs_bots.config.wizard_builder import _stage_from_dict

        d = {
            "name": "lookup",
            "prompt": "Enter product",
            "tool_result_mapping": [
                {"tool": "t", "params": {}, "mapping": {"a": "b"}},
            ],
        }
        stage = _stage_from_dict(d)
        assert len(stage.tool_result_mapping) == 1
        assert stage.tool_result_mapping[0]["tool"] == "t"


# =========================================================================
# End-to-end behavioral tests (via BotTestHarness)
# =========================================================================


def _build_lookup_wizard_config(
    tool_result_mapping: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a wizard config with a tool_result_mapping stage."""
    return (
        WizardConfigBuilder("tool-test")
        .stage(
            "lookup",
            is_start=True,
            prompt="What product are you looking for?",
            tool_result_mapping=tool_result_mapping,
        )
            .field("product_name", field_type="string", required=True)
            .transition("review", "has('product_id')")
        .stage("review", is_end=True, prompt="Here are the details.")
        .build()
    )


class TestToolToStateEndToEnd:
    """End-to-end: extraction → tool execution → result mapping → transition."""

    @pytest.mark.asyncio
    async def test_extraction_triggers_tool_and_maps_results(self) -> None:
        """Full flow: extract product_name → tool lookup → product_id in state → transition."""
        config = _build_lookup_wizard_config(tool_result_mapping=[
            {
                "tool": "product_lookup",
                "params": {"query": "product_name"},
                "mapping": {
                    "product_id": "product_id",
                    "category": "product_category",
                },
            },
        ])

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Looking up...", "Here are the details."],
            extraction_results=[[{"product_name": "Widget"}]],
            tools=[ProductLookupTool()],
        ) as harness:
            await harness.chat("I want Widget")

            # Tool result should be mapped into state
            assert harness.wizard_data.get("product_id") == "PROD-123"
            assert harness.wizard_data.get("product_category") == "electronics"
            assert harness.wizard_data.get("product_name") == "Widget"
            # Transition should have fired (product_id is now set)
            assert harness.wizard_stage == "review"

    @pytest.mark.asyncio
    async def test_tool_error_skip(self) -> None:
        """Tool error with on_error=skip: state not polluted, stage stays."""
        config = _build_lookup_wizard_config(tool_result_mapping=[
            {
                "tool": "failing_tool",
                "params": {},
                "mapping": {"result": "result_value"},
                "on_error": "skip",
            },
        ])

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Hmm...", "Try again."],
            extraction_results=[[{"product_name": "Widget"}]],
            tools=[FailingTool()],
        ) as harness:
            await harness.chat("I want Widget")

            # Error should be silently skipped
            assert "result_value" not in harness.wizard_data
            assert "_tool_error_failing_tool" not in harness.wizard_data
            # No product_id → no transition
            assert harness.wizard_stage == "lookup"

    @pytest.mark.asyncio
    async def test_tool_error_fail(self) -> None:
        """Tool error with on_error=fail: error flag written to state."""
        config = _build_lookup_wizard_config(tool_result_mapping=[
            {
                "tool": "failing_tool",
                "params": {},
                "mapping": {"result": "result_value"},
                "on_error": "fail",
            },
        ])

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Error occurred.", "Try again."],
            extraction_results=[[{"product_name": "Widget"}]],
            tools=[FailingTool()],
        ) as harness:
            await harness.chat("I want Widget")

            # Error flag should be in state
            assert "_tool_error_failing_tool" in harness.wizard_data
            assert "result_value" not in harness.wizard_data

    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        """Tool not in registry: graceful skip via ToolExecution error record."""
        config = _build_lookup_wizard_config(tool_result_mapping=[
            {
                "tool": "nonexistent_tool",
                "params": {},
                "mapping": {"result": "result_value"},
            },
        ])

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Ok.", "What else?"],
            extraction_results=[[{"product_name": "Widget"}]],
            # Register a real tool so registry is truthy; nonexistent_tool
            # is still absent, exercising the NotFoundError path.
            tools=[ProductLookupTool()],
        ) as harness:
            await harness.chat("I want Widget")

            # Should not crash; mapping should be skipped (tool not found
            # produces a ToolExecution with error, which on_error=skip ignores)
            assert "result_value" not in harness.wizard_data

    @pytest.mark.asyncio
    async def test_non_dict_tool_result(self) -> None:
        """Non-dict tool result maps to first target key."""
        config = _build_lookup_wizard_config(tool_result_mapping=[
            {
                "tool": "scalar_tool",
                "params": {"input": "product_name"},
                "mapping": {"value": "computed_score"},
            },
        ])

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got score.", "Done."],
            extraction_results=[[{"product_name": "Widget"}]],
            tools=[ScalarTool(result=99)],
        ) as harness:
            await harness.chat("I want Widget")

            # Scalar result (99) should map to first target key
            assert harness.wizard_data.get("computed_score") == 99

    @pytest.mark.asyncio
    async def test_missing_params_omitted(self) -> None:
        """When a state key for a param is not yet populated, param is omitted."""
        config = _build_lookup_wizard_config(tool_result_mapping=[
            {
                "tool": "product_lookup",
                "params": {
                    "query": "product_name",
                    "filter": "nonexistent_key",  # Not in state
                },
                "mapping": {"product_id": "product_id"},
            },
        ])

        tool = ProductLookupTool()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Found it.", "Details."],
            extraction_results=[[{"product_name": "Widget"}]],
            tools=[tool],
        ) as harness:
            await harness.chat("I want Widget")

            # Tool should still execute (with just query param)
            assert harness.wizard_data.get("product_id") == "PROD-123"

    @pytest.mark.asyncio
    async def test_no_tool_result_mapping_no_tool_execution(self) -> None:
        """Stages without tool_result_mapping do not trigger tool execution."""
        config = (
            WizardConfigBuilder("no-tools")
            .stage("gather", is_start=True, prompt="Enter your name.")
                .field("name", field_type="string", required=True)
                .transition("done", "has('name')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Done!"],
            extraction_results=[[{"name": "Alice"}]],
            tools=[ProductLookupTool()],
        ) as harness:
            await harness.chat("My name is Alice")

            # Should transition normally without any tool involvement
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_extraction_fails_tools_do_not_fire(self) -> None:
        """When extraction fails confidence gate (clarification), tools do NOT fire."""
        config = _build_lookup_wizard_config(tool_result_mapping=[
            {
                "tool": "product_lookup",
                "params": {"query": "product_name"},
                "mapping": {"product_id": "product_id"},
            },
        ])

        # Extraction returns empty data → confidence gate fails → clarification.
        # Tools should NOT run on this turn.
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Could you clarify?", "Still waiting..."],
            extraction_results=[[{}]],
            tools=[ProductLookupTool()],
        ) as harness:
            await harness.chat("hmm not sure")

            # Should stay on lookup stage (clarification, no transition)
            assert harness.wizard_stage == "lookup"
            # Tool should NOT have been called — no product_id in state
            assert "product_id" not in harness.wizard_data

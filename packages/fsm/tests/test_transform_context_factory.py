"""Tests for the transform_context_factory mechanism on ExecutionContext.

Verifies that:
- exec_context.variables propagates to FunctionContext.variables
- Network stack top propagates to FunctionContext.network_name
- Default (no factory): transforms receive plain FunctionContext
- With factory: transforms receive factory output
- Factory receives complete FunctionContext with all fields populated
- In-place transforms (returning None) preserve input data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import FunctionContext


def _make_arc(
    transform_name: str = "my_transform",
    source: str = "state_a",
    target: str = "state_b",
) -> ArcExecution:
    """Build a minimal ArcExecution for testing."""
    arc_def = ArcDefinition(
        target_state=target,
        transform=transform_name,
    )
    return ArcExecution(
        arc_def=arc_def,
        source_state=source,
        function_registry={},
    )


class TestVariablesPropagation:
    """exec_context.variables must flow to FunctionContext.variables."""

    def test_variables_propagated(self) -> None:
        arc = _make_arc()
        ctx = ExecutionContext()
        ctx.variables = {"user_id": "abc", "session": 42}

        func_ctx = arc._create_function_context(ctx, resources={"r": 1})

        assert isinstance(func_ctx, FunctionContext)
        assert func_ctx.variables == {"user_id": "abc", "session": 42}

    def test_empty_variables_default(self) -> None:
        arc = _make_arc()
        ctx = ExecutionContext()

        func_ctx = arc._create_function_context(ctx)

        assert func_ctx.variables == {}


class TestNetworkNamePropagation:
    """Network stack top must flow to FunctionContext.network_name."""

    def test_network_name_from_stack(self) -> None:
        arc = _make_arc()
        ctx = ExecutionContext()
        ctx.network_stack = [("main", None), ("sub_flow", "return_state")]

        func_ctx = arc._create_function_context(ctx)

        assert func_ctx.network_name == "sub_flow"

    def test_network_name_none_when_empty_stack(self) -> None:
        arc = _make_arc()
        ctx = ExecutionContext()

        func_ctx = arc._create_function_context(ctx)

        assert func_ctx.network_name is None


class TestTransformContextFactory:
    """transform_context_factory on ExecutionContext."""

    def test_default_no_factory(self) -> None:
        """Without factory, _create_function_context returns FunctionContext."""
        arc = _make_arc()
        ctx = ExecutionContext()

        result = arc._create_function_context(ctx)

        assert isinstance(result, FunctionContext)
        assert result.state_name == "state_a"

    def test_factory_called(self) -> None:
        """With factory, _create_function_context returns factory output."""

        @dataclass
        class CustomContext:
            original: FunctionContext
            extra: str = "custom"

        def factory(fc: FunctionContext) -> CustomContext:
            return CustomContext(original=fc)

        arc = _make_arc()
        ctx = ExecutionContext()
        ctx.transform_context_factory = factory

        result = arc._create_function_context(ctx)

        assert isinstance(result, CustomContext)
        assert result.extra == "custom"
        assert result.original.state_name == "state_a"

    def test_factory_receives_complete_context(self) -> None:
        """Factory receives FunctionContext with all fields populated."""
        captured: list[FunctionContext] = []

        def factory(fc: FunctionContext) -> FunctionContext:
            captured.append(fc)
            return fc

        arc = _make_arc(transform_name="do_work", source="src", target="dst")
        ctx = ExecutionContext()
        ctx.variables = {"key": "val"}
        ctx.network_stack = [("net1", None)]
        ctx.transform_context_factory = factory

        arc._create_function_context(ctx, resources={"gpu": True})

        assert len(captured) == 1
        fc = captured[0]
        assert fc.state_name == "src"
        assert fc.function_name == "do_work"
        assert fc.resources == {"gpu": True}
        assert fc.variables == {"key": "val"}
        assert fc.network_name == "net1"
        assert fc.metadata["target_state"] == "dst"


class TestNoneReturnPreservation:
    """Transforms returning None should preserve input data."""

    def test_sync_none_return_preserves_data(self) -> None:
        """Sync transform returning None → input data preserved."""

        def mutating_transform(data: dict, ctx: Any) -> None:
            data["added"] = True
            # Returns None (in-place mutation)

        arc_def = ArcDefinition(target_state="b", transform="mutate")
        arc = ArcExecution(
            arc_def=arc_def,
            source_state="a",
            function_registry={"mutate": mutating_transform},
        )
        ctx = ExecutionContext()

        func_ctx = arc._create_function_context(ctx)
        result = arc._execute_single_transform("mutate", {"x": 1}, func_ctx)

        assert result == {"x": 1, "added": True}

    @pytest.mark.asyncio
    async def test_async_none_return_preserves_data(self) -> None:
        """Async transform returning None → input data preserved."""

        async def async_mutating_transform(data: dict, ctx: Any) -> None:
            data["added"] = True

        arc_def = ArcDefinition(target_state="b", transform="mutate")
        arc = ArcExecution(
            arc_def=arc_def,
            source_state="a",
            function_registry={"mutate": async_mutating_transform},
        )
        ctx = ExecutionContext()

        func_ctx = arc._create_function_context(ctx)
        result = await arc._execute_single_transform_async(
            "mutate", {"x": 1}, func_ctx
        )

        assert result == {"x": 1, "added": True}

    def test_sync_explicit_return_used(self) -> None:
        """Sync transform with explicit return → return value used."""

        def replacing_transform(data: dict, ctx: Any) -> dict:
            return {"replaced": True}

        arc_def = ArcDefinition(target_state="b", transform="replace")
        arc = ArcExecution(
            arc_def=arc_def,
            source_state="a",
            function_registry={"replace": replacing_transform},
        )
        ctx = ExecutionContext()

        func_ctx = arc._create_function_context(ctx)
        result = arc._execute_single_transform("replace", {"x": 1}, func_ctx)

        assert result == {"replaced": True}

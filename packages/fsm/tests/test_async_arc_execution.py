"""Tests for async arc execution methods.

Validates that ArcExecution.can_execute_async(), execute_async(),
and _execute_single_transform_async() properly handle both sync
and async functions by using inspect.isawaitable().
"""

import pytest

from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution
from dataknobs_fsm.core.exceptions import FunctionError
from dataknobs_fsm.functions.base import ExecutionResult, FunctionContext


class StubContext:
    """Minimal execution context for arc tests."""

    pass


def sync_add_ten(data: dict, context: FunctionContext) -> dict:
    """Sync transform that adds 10."""
    result = dict(data)
    result["value"] = result.get("value", 0) + 10
    return result


async def async_double(data: dict, context: FunctionContext) -> dict:
    """Async transform that doubles."""
    result = dict(data)
    result["value"] = result.get("value", 0) * 2
    return result


async def async_fail(data: dict, context: FunctionContext) -> dict:
    """Async transform that raises."""
    raise ValueError("async failure")


def sync_pretest_true(data: dict, context: object) -> bool:
    """Sync pre-test that returns True."""
    return True


def sync_pretest_false(data: dict, context: object) -> bool:
    """Sync pre-test that returns False."""
    return False


async def async_pretest_true(data: dict, context: object) -> bool:
    """Async pre-test that returns True."""
    return True


async def async_pretest_false(data: dict, context: object) -> bool:
    """Async pre-test that returns False."""
    return False


class TestExecuteAsyncWithSyncTransform:
    """Sync transforms work through the async path."""

    @pytest.mark.asyncio
    async def test_sync_transform_returns_result(self) -> None:
        registry = {"add_ten": sync_add_ten}
        arc_def = ArcDefinition(target_state="next", transform="add_ten")
        arc_exec = ArcExecution(arc_def, "current", registry)

        result = await arc_exec.execute_async(StubContext(), {"value": 5})
        assert result["value"] == 15

    @pytest.mark.asyncio
    async def test_statistics_updated(self) -> None:
        registry = {"add_ten": sync_add_ten}
        arc_def = ArcDefinition(target_state="next", transform="add_ten")
        arc_exec = ArcExecution(arc_def, "current", registry)

        await arc_exec.execute_async(StubContext(), {"value": 0})
        assert arc_exec.execution_count == 1
        assert arc_exec.success_count == 1
        assert arc_exec.failure_count == 0


class TestExecuteAsyncWithAsyncTransform:
    """Async transforms are properly awaited."""

    @pytest.mark.asyncio
    async def test_async_transform_returns_result(self) -> None:
        registry = {"double": async_double}
        arc_def = ArcDefinition(target_state="next", transform="double")
        arc_exec = ArcExecution(arc_def, "current", registry)

        result = await arc_exec.execute_async(StubContext(), {"value": 7})
        assert result["value"] == 14

    @pytest.mark.asyncio
    async def test_async_transform_error_propagates(self) -> None:
        registry = {"fail": async_fail}
        arc_def = ArcDefinition(target_state="next", transform="fail")
        arc_exec = ArcExecution(arc_def, "current", registry)

        with pytest.raises(FunctionError, match="Arc execution failed"):
            await arc_exec.execute_async(StubContext(), {"value": 0})

        assert arc_exec.failure_count == 1


class TestExecuteAsyncWithMixedChain:
    """A chain of [sync, async, sync] transforms all execute in order."""

    @pytest.mark.asyncio
    async def test_mixed_chain_produces_correct_result(self) -> None:
        registry = {
            "add_ten": sync_add_ten,
            "double": async_double,
        }
        arc_def = ArcDefinition(
            target_state="next",
            transform=["add_ten", "double", "add_ten"],
        )
        arc_exec = ArcExecution(arc_def, "current", registry)

        # value=5 -> +10=15 -> *2=30 -> +10=40
        result = await arc_exec.execute_async(StubContext(), {"value": 5})
        assert result["value"] == 40


class TestCanExecuteAsyncWithSyncPretest:
    """Sync pre-test works through can_execute_async."""

    @pytest.mark.asyncio
    async def test_sync_pretest_true(self) -> None:
        registry = {"check": sync_pretest_true}
        arc_def = ArcDefinition(target_state="next", pre_test="check")
        arc_exec = ArcExecution(arc_def, "current", registry)

        assert await arc_exec.can_execute_async(StubContext(), {}) is True

    @pytest.mark.asyncio
    async def test_sync_pretest_false(self) -> None:
        registry = {"check": sync_pretest_false}
        arc_def = ArcDefinition(target_state="next", pre_test="check")
        arc_exec = ArcExecution(arc_def, "current", registry)

        assert await arc_exec.can_execute_async(StubContext(), {}) is False


class TestCanExecuteAsyncWithAsyncPretest:
    """Async pre-test is awaited in can_execute_async."""

    @pytest.mark.asyncio
    async def test_async_pretest_true(self) -> None:
        registry = {"check": async_pretest_true}
        arc_def = ArcDefinition(target_state="next", pre_test="check")
        arc_exec = ArcExecution(arc_def, "current", registry)

        assert await arc_exec.can_execute_async(StubContext(), {}) is True

    @pytest.mark.asyncio
    async def test_async_pretest_false(self) -> None:
        registry = {"check": async_pretest_false}
        arc_def = ArcDefinition(target_state="next", pre_test="check")
        arc_exec = ArcExecution(arc_def, "current", registry)

        assert await arc_exec.can_execute_async(StubContext(), {}) is False

    @pytest.mark.asyncio
    async def test_no_pretest_returns_true(self) -> None:
        arc_def = ArcDefinition(target_state="next")
        arc_exec = ArcExecution(arc_def, "current", {})

        assert await arc_exec.can_execute_async(StubContext(), {}) is True


class TestExecuteAsyncNoTransform:
    """execute_async with no transform passes data through."""

    @pytest.mark.asyncio
    async def test_passthrough(self) -> None:
        arc_def = ArcDefinition(target_state="next")
        arc_exec = ArcExecution(arc_def, "current", {})

        result = await arc_exec.execute_async(StubContext(), {"key": "value"})
        assert result == {"key": "value"}


class TestExecuteAsyncExecutionResult:
    """Async path handles ExecutionResult wrappers."""

    @pytest.mark.asyncio
    async def test_success_result_unwrapped(self) -> None:
        async def wrapped(data: dict, ctx: FunctionContext) -> ExecutionResult:
            return ExecutionResult.success_result({"answer": 42})

        registry = {"wrapped": wrapped}
        arc_def = ArcDefinition(target_state="next", transform="wrapped")
        arc_exec = ArcExecution(arc_def, "current", registry)

        result = await arc_exec.execute_async(StubContext(), {})
        assert result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_failure_result_raises(self) -> None:
        async def wrapped(data: dict, ctx: FunctionContext) -> ExecutionResult:
            return ExecutionResult.failure_result("bad input")

        registry = {"wrapped": wrapped}
        arc_def = ArcDefinition(target_state="next", transform="wrapped")
        arc_exec = ArcExecution(arc_def, "current", registry)

        with pytest.raises(FunctionError, match="bad input"):
            await arc_exec.execute_async(StubContext(), {})

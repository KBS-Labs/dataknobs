"""Tests for multi-transform arc execution.

Tests the chained transform pipeline in ArcExecution where a list of
transform functions are executed sequentially, passing data through each.

Also tests that list transforms work through AdvancedFSM.execute_step_sync(),
which was previously a divergent code path that stringified list transforms.
"""

import pytest

from dataknobs_fsm.api.advanced import AdvancedFSM, create_advanced_fsm
from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution
from dataknobs_fsm.core.exceptions import FunctionError
from dataknobs_fsm.functions.base import ExecutionResult, FunctionContext


class MockContext:
    """Minimal execution context for arc tests."""

    pass


class TestArcMultiTransform:
    """Tests for multi-transform arc execution."""

    def setup_method(self) -> None:
        """Set up test fixtures."""

        def add_ten(data: dict, context: FunctionContext) -> dict:
            result = dict(data)
            result["value"] = result.get("value", 0) + 10
            return result

        def double(data: dict, context: FunctionContext) -> dict:
            result = dict(data)
            result["value"] = result.get("value", 0) * 2
            return result

        def fail_transform(data: dict, context: FunctionContext) -> dict:
            raise ValueError("Transform failed deliberately")

        def return_success(data: dict, context: FunctionContext) -> ExecutionResult:
            return ExecutionResult.success_result({"value": 999})

        def return_failure(data: dict, context: FunctionContext) -> ExecutionResult:
            return ExecutionResult.failure_result("deliberate failure")

        self.function_registry: dict = {
            "add_ten": add_ten,
            "double": double,
            "fail_transform": fail_transform,
            "return_success": return_success,
            "return_failure": return_failure,
        }

    def test_single_transform_executes(self) -> None:
        """A single string transform works normally."""
        arc_def = ArcDefinition(target_state="next", transform="add_ten")
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        result = arc_exec.execute(MockContext(), {"value": 5})
        assert result["value"] == 15

    def test_multi_transform_chained(self) -> None:
        """List of transforms pipe data sequentially."""
        arc_def = ArcDefinition(
            target_state="next",
            transform=["add_ten", "double"],
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        # value=5 → add_ten → 15 → double → 30
        result = arc_exec.execute(MockContext(), {"value": 5})
        assert result["value"] == 30

    def test_multi_transform_reverse_order(self) -> None:
        """Transform order matters: double then add_ten gives different result."""
        arc_def = ArcDefinition(
            target_state="next",
            transform=["double", "add_ten"],
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        # value=5 → double → 10 → add_ten → 20
        result = arc_exec.execute(MockContext(), {"value": 5})
        assert result["value"] == 20

    def test_multi_transform_error_propagates(self) -> None:
        """Error in second transform raises FunctionError."""
        arc_def = ArcDefinition(
            target_state="next",
            transform=["add_ten", "fail_transform"],
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        with pytest.raises(FunctionError):
            arc_exec.execute(MockContext(), {"value": 5})

    def test_transform_missing_raises(self) -> None:
        """Non-existent transform name raises FunctionError."""
        arc_def = ArcDefinition(
            target_state="next",
            transform="nonexistent_func",
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        with pytest.raises(FunctionError, match="not found"):
            arc_exec.execute(MockContext(), {"value": 1})

    def test_transform_returns_execution_result_success(self) -> None:
        """ExecutionResult with success unwraps to .data."""
        arc_def = ArcDefinition(target_state="next", transform="return_success")
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        result = arc_exec.execute(MockContext(), {"value": 1})
        assert result == {"value": 999}

    def test_transform_returns_execution_result_failure(self) -> None:
        """ExecutionResult with failure raises FunctionError."""
        arc_def = ArcDefinition(target_state="next", transform="return_failure")
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        with pytest.raises(FunctionError, match="deliberate failure"):
            arc_exec.execute(MockContext(), {"value": 1})

    def test_no_transform_passthrough(self) -> None:
        """Null transform passes data through unchanged."""
        arc_def = ArcDefinition(target_state="next", transform=None)
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        data = {"value": 42}
        result = arc_exec.execute(MockContext(), data)
        assert result is data

    def test_multi_transform_updates_statistics(self) -> None:
        """Multi-transform arc tracks execution count and success count."""
        arc_def = ArcDefinition(
            target_state="next",
            transform=["add_ten", "double"],
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        arc_exec.execute(MockContext(), {"value": 1})
        assert arc_exec.execution_count == 1
        assert arc_exec.success_count == 1
        assert arc_exec.failure_count == 0

    def test_multi_transform_failure_updates_statistics(self) -> None:
        """Failed multi-transform arc tracks failure count."""
        arc_def = ArcDefinition(
            target_state="next",
            transform=["add_ten", "fail_transform"],
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        with pytest.raises(FunctionError):
            arc_exec.execute(MockContext(), {"value": 1})
        assert arc_exec.execution_count == 1
        assert arc_exec.failure_count == 1
        assert arc_exec.success_count == 0

    def test_missing_in_multi_transform_chain_raises(self) -> None:
        """Missing function in a multi-transform chain raises FunctionError."""
        arc_def = ArcDefinition(
            target_state="next",
            transform=["add_ten", "nonexistent_func", "double"],
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)

        with pytest.raises(FunctionError, match="not found"):
            arc_exec.execute(MockContext(), {"value": 5})


class TestStepSyncMultiTransform:
    """Tests that list transforms work through AdvancedFSM.execute_step_sync().

    These exercise the _execute_arc_transform delegation to ArcExecution,
    which previously stringified list transforms and silently skipped them.
    """

    @staticmethod
    def _make_config(
        arc_transform: dict | list[dict] | None = None,
    ) -> dict:
        """Build a minimal two-state FSM config with an arc transform.

        Args:
            arc_transform: Transform spec(s) for the start->end arc.

        Returns:
            FSM config dict ready for ``create_advanced_fsm``.
        """
        arc: dict = {"from": "start", "to": "end"}
        if arc_transform is not None:
            arc["transform"] = arc_transform
        return {
            "name": "test_multi",
            "version": "1.0",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True},
                        {"name": "end", "is_end": True},
                    ],
                    "arcs": [arc],
                }
            ],
        }

    def test_step_sync_multi_transform(self) -> None:
        """List transforms execute sequentially through execute_step_sync."""

        def add_ten(data: dict, ctx: FunctionContext) -> dict:
            out = dict(data)
            out["value"] = out.get("value", 0) + 10
            return out

        def double(data: dict, ctx: FunctionContext) -> dict:
            out = dict(data)
            out["value"] = out.get("value", 0) * 2
            return out

        config = self._make_config(
            arc_transform=[
                {"type": "registered", "name": "add_ten"},
                {"type": "registered", "name": "double"},
            ]
        )
        fsm = create_advanced_fsm(
            config,
            custom_functions={"add_ten": add_ten, "double": double},
        )
        context = fsm.create_context({"value": 5})

        result = fsm.execute_step_sync(context)

        assert result.success, f"Step failed: {result.error}"
        assert result.to_state == "end"
        # value=5 -> add_ten -> 15 -> double -> 30
        assert context.data["value"] == 30

    def test_step_sync_single_transform_still_works(self) -> None:
        """Regression guard: a single string transform still works."""

        def add_ten(data: dict, ctx: FunctionContext) -> dict:
            out = dict(data)
            out["value"] = out.get("value", 0) + 10
            return out

        config = self._make_config(
            arc_transform={"type": "registered", "name": "add_ten"},
        )
        fsm = create_advanced_fsm(
            config,
            custom_functions={"add_ten": add_ten},
        )
        context = fsm.create_context({"value": 5})

        result = fsm.execute_step_sync(context)

        assert result.success, f"Step failed: {result.error}"
        assert result.to_state == "end"
        assert context.data["value"] == 15

    def test_step_sync_stringified_list_compat(self) -> None:
        """Backward compat: stringified list transforms are parsed correctly."""
        from dataknobs_fsm.api.advanced import AdvancedFSM

        parsed = AdvancedFSM._normalize_transform_names("['add_ten', 'double']")
        assert parsed == ["add_ten", "double"]

        # Single string stays as single-element list
        parsed = AdvancedFSM._normalize_transform_names("add_ten")
        assert parsed == ["add_ten"]

        # Actual list passes through
        parsed = AdvancedFSM._normalize_transform_names(["a", "b"])
        assert parsed == ["a", "b"]

        # None / empty
        parsed = AdvancedFSM._normalize_transform_names(None)
        assert parsed == []

    def test_step_sync_no_transform(self) -> None:
        """Arc with no transform passes data through unchanged."""
        config = self._make_config(arc_transform=None)
        fsm = create_advanced_fsm(config)
        context = fsm.create_context({"value": 42})

        result = fsm.execute_step_sync(context)

        assert result.success
        assert result.to_state == "end"
        assert context.data["value"] == 42

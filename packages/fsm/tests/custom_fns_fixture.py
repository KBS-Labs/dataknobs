"""Importable helper module for config-authored ``custom`` function tests.

A real, importable module (under the test package) referenced by
``{"type": "custom", "module": "tests.custom_fns_fixture", "name": ...}`` so the
custom-function-reference tests exercise the real ``importlib.import_module`` +
``getattr`` resolution path with no mocks.
"""

from __future__ import annotations

from typing import Any, Dict

from dataknobs_fsm.functions.base import ExecutionResult, ITransformFunction


class AddMarker(ITransformFunction):
    """Custom transform that stamps a configured marker onto the record.

    Constructed from the config reference's ``params`` (``key`` / ``value``),
    exercising the class-materialization path for ``type: custom``.
    """

    def __init__(self, key: str, value: Any = True) -> None:
        self.key = key
        self.value = value

    def transform(
        self,
        data: Dict[str, Any],
        context: Any = None,
    ) -> Dict[str, Any]:
        data[self.key] = self.value
        return data

    def get_transform_description(self) -> str:
        return f"AddMarker(key={self.key!r})"


def make_execution_result(data: Dict[str, Any]) -> ExecutionResult:
    """Trivial helper kept importable for ad-hoc custom-reference probes."""
    return ExecutionResult.success_result(data)

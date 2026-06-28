"""Importable helper module for config-authored ``custom`` function tests.

A real, importable module (under the test package) referenced by
``{"type": "custom", "module": "tests.custom_fns_fixture", "name": ...}`` so the
custom-function-reference tests exercise the real ``importlib.import_module`` +
``getattr`` resolution path with no mocks.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from dataknobs_fsm.functions.base import (
    ExecutionResult,
    IStateTestFunction,
    ITransformFunction,
    IValidationFunction,
)


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


class AsyncAddMarker(ITransformFunction):
    """Custom transform whose ``transform`` is ``async def``.

    Exercises the async-adapter path: the engine must *await* the coroutine the
    interface method returns rather than storing it as the record's data. A
    no-op ``await`` (``asyncio.sleep(0)``) forces a real suspension point so a
    "ran it synchronously and dropped the coroutine" regression cannot pass.
    """

    def __init__(self, key: str, value: Any = True) -> None:
        self.key = key
        self.value = value

    async def transform(
        self,
        data: Dict[str, Any],
        context: Any = None,
    ) -> Dict[str, Any]:
        import asyncio

        await asyncio.sleep(0)
        data[self.key] = self.value
        return data

    def get_transform_description(self) -> str:
        return f"AsyncAddMarker(key={self.key!r})"


class AsyncRequireField(IValidationFunction):
    """Custom validator whose ``validate`` is ``async def`` (gates on a field).

    Returns ``False`` (failing entry) when the required field is absent. Used as
    a ``pre_validators:`` ref to prove an async custom validator is awaited and
    its boolean result honored, not stored as a truthy coroutine.
    """

    def __init__(self, field: str) -> None:
        self.field = field

    async def validate(self, data: Any, context: Any = None) -> bool:
        import asyncio

        await asyncio.sleep(0)
        return self.field in data

    def get_validation_rules(self) -> Dict[str, Any]:
        return {"required": self.field}


class EnrichViaValidate(IValidationFunction):
    """Custom validator that returns a dict (merged by the post-entry path).

    Referenced under ``validators:`` (the post-entry validator phase, distinct
    from the gating ``pre_validators:`` phase). That phase calls
    ``validate(state_obj)`` single-arg and merges a dict result into the record,
    so this proves the adapter is dispatched correctly on that path too.
    """

    def __init__(self, key: str, value: Any = True) -> None:
        self.key = key
        self.value = value

    def validate(self, data: Any) -> Dict[str, Any]:
        return {self.key: self.value}

    def get_validation_rules(self) -> Dict[str, Any]:
        return {"enriches": self.key}


class HasField(IStateTestFunction):
    """Custom arc-condition test: passes when a configured field is present.

    Exercises the ``IStateTestFunction`` adapter on the arc-condition dispatch
    path (``test`` returning ``(passed, reason)``), which no built-in library
    function covers.
    """

    def __init__(self, field: str) -> None:
        self.field = field

    def test(self, data: Any, context: Any = None) -> Tuple[bool, str | None]:
        if self.field in data:
            return True, None
        return False, f"missing {self.field}"

    def get_test_description(self) -> str:
        return f"HasField({self.field!r})"


class AsyncHasField(IStateTestFunction):
    """Custom arc-condition test whose ``test`` is ``async def``.

    Exercises the *async* adapter on the ``IStateTestFunction`` arc-condition
    path: the engine must *await* the coroutine and gate the transition on its
    ``(passed, reason)`` result, not on the truthiness of an un-awaited
    coroutine object (which is always truthy and would wrongly pass the arc).
    A no-op ``await`` forces a real suspension point.
    """

    def __init__(self, field: str) -> None:
        self.field = field

    async def test(self, data: Any, context: Any = None) -> Tuple[bool, str | None]:
        import asyncio

        await asyncio.sleep(0)
        if self.field in data:
            return True, None
        return False, f"missing {self.field}"

    def get_test_description(self) -> str:
        return f"AsyncHasField({self.field!r})"


def stamp_processed(data: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Plain custom ``(data, context)`` transform *function* (not a class).

    Referenced as ``{type: custom, ...}`` with no ``params``: it must resolve
    through the standard wrapper path (``classes_only`` keeps it from being
    mis-invoked as a zero-arg factory) and run, stamping ``processed``.
    """
    data["processed"] = True
    return data


def make_execution_result(data: Dict[str, Any]) -> ExecutionResult:
    """Trivial helper kept importable for ad-hoc custom-reference probes."""
    return ExecutionResult.success_result(data)

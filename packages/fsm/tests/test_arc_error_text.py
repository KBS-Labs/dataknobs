"""What an arc's error is allowed to say.

A transform or pre-test is consumer code, and what it raises is usually its
dependency's text. A driver that cannot connect says so by quoting the
connection URL. The arc's error names the exception's type and keeps the
original as its cause.

The errors dataknobs words itself -- a missing function, a refused resource, a
transform that reports its own failure -- keep their words.
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_fsm.api.advanced import create_advanced_fsm
from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution
from dataknobs_fsm.core.exceptions import FunctionError
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import ExecutionResult
from dataknobs_fsm.resources.manager import ResourceManager

_DSN = "postgresql://svc:hunter2@db.internal/orders"


def fail_to_connect(data, context):
    raise ConnectionError(f"could not connect to {_DSN}")


def _arc(function, *, pre_test: bool = False, **kwargs) -> ArcExecution:
    field = "pre_test" if pre_test else "transform"
    registry = {"f": function} if function else {}
    return ArcExecution(
        ArcDefinition(target_state="next", **{field: "f"}, **kwargs), "now", registry
    )


async def _failure(arc: ArcExecution, context: ExecutionContext | None = None) -> FunctionError:
    context = context or ExecutionContext()
    run = arc.can_execute_async if arc.arc_def.pre_test else arc.execute_async
    with pytest.raises(FunctionError) as excinfo:
        await run(context, {})
    return excinfo.value


@pytest.mark.parametrize(
    ("pre_test", "message"),
    [
        (False, "Arc execution failed (ConnectionError)"),
        (True, "Pre-test execution failed (ConnectionError)"),
    ],
)
async def test_what_the_function_raised_is_the_cause_and_not_the_message(pre_test, message):
    error = await _failure(_arc(fail_to_connect, pre_test=pre_test))

    assert str(error) == message
    assert _DSN in str(error.__cause__)
    assert (error.from_state, error.to_state) == ("now", "next")


async def test_a_failure_is_counted_once():
    arc = _arc(fail_to_connect)
    await _failure(arc)
    assert (arc.execution_count, arc.success_count, arc.failure_count) == (1, 0, 1)


async def test_a_missing_transform_is_named():
    error = await _failure(_arc(None))
    assert str(error) == "Arc execution failed: Transform function 'f' not found"


async def test_a_transform_that_is_not_callable_is_named():
    arc = _arc(object())
    error = await _failure(arc)
    assert str(error) == "Arc execution failed: Transform f is not callable"
    assert arc.failure_count == 1


async def test_a_missing_pre_test_is_named():
    error = await _failure(_arc(None, pre_test=True))
    assert str(error) == "Pre-test function 'f' not found"


async def test_a_failure_the_transform_reports_keeps_its_words():
    error = await _failure(_arc(lambda data, ctx: ExecutionResult.failure_result("no such order")))
    assert str(error) == "Arc execution failed: no such order"


@pytest.mark.parametrize("pre_test", [False, True])
async def test_an_error_raised_as_a_function_error_keeps_its_words(pre_test):
    def refuse(data, context):
        raise FunctionError("order is closed")

    error = await _failure(_arc(refuse, pre_test=pre_test))
    assert str(error).endswith("execution failed: order is closed")


@pytest.mark.parametrize("pre_test", [False, True])
async def test_a_resource_that_cannot_be_acquired_is_named(pre_test):
    context = ExecutionContext()
    context.resource_manager = ResourceManager()
    arc = _arc(lambda data, ctx: data, pre_test=pre_test, required_resources={"db": "orders_db"})

    error = await _failure(arc, context)

    assert "Resource 'orders_db' error: Failed to acquire arc resource" in str(error)


def _stepping_fsm():
    return create_advanced_fsm(
        {
            "name": "Stepping",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True},
                        {"name": "done", "is_end": True},
                    ],
                    "arcs": [
                        {
                            "from": "start",
                            "to": "done",
                            "transform": {"type": "registered", "name": "fail_to_connect"},
                        }
                    ],
                }
            ],
        },
        custom_functions={"fail_to_connect": fail_to_connect},
    )


async def test_a_step_reports_the_type_and_logs_the_original(caplog):
    fsm = _stepping_fsm()
    with caplog.at_level(logging.ERROR, logger="dataknobs_fsm.api.advanced"):
        result = await fsm.execute_step_async(fsm.create_context({}))

    assert result.success is False
    assert result.error == "Arc execution failed (ConnectionError)"
    (record,) = [r for r in caplog.records if r.name == "dataknobs_fsm.api.advanced"]
    assert record.exc_info is not None
    assert _DSN in str(record.exc_info[1].__cause__)

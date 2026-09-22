# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A consumer function that fails is reported, not re-run with other arguments.

Four sites decided a callable's arity by *calling it* and catching the
failure::

    try:
        return func(state_obj)
    except (TypeError, AttributeError):
        return func(record, context)

An ``except`` cannot tell where the exception came from. Argument binding
raises ``TypeError`` **before the callable's frame exists**, which is the case
these clauses were written for --- but a ``TypeError`` or ``AttributeError``
raised *inside the body*, after the work has been done, looks exactly the
same from outside. So a transform with an ordinary bug in it (``.upper()`` on
a number, ``"x" + 1``) was run a second time, with different arguments,
because it failed the first time.

Measured before the fix, through the public pre-built-:class:`FSM` door ---
:meth:`StateDefinition.add_transform_function` takes a plain callable, and
``AdvancedFSM(fsm)`` accepts a pre-built FSM:

============================================  ==============  ==============
site                                          body raises      body succeeds
============================================  ==============  ==============
``AsyncExecutionEngine`` state transform      2 invocations   1 invocation
``AsyncExecutionEngine`` validator            2 invocations   1 invocation
``AdvancedFSM`` stepped transform             2 invocations   1 invocation
``evaluate_arc_condition_common``             2 invocations   1 invocation
============================================  ==============  ==============

The first call received a ``StateDataWrapper`` and the second a ``dict``, so
"it failed, try it differently" is not even a no-op for a function that only
reads its record: anything it wrote, sent, appended or counted happened twice.
The validator site is the quietest of the four --- its loop swallows every
exception, so the run still reports success while the consumer's function has
run twice and failed twice.

``AttributeError`` never belonged in that clause at all. Argument binding
cannot raise it, so catching it could only ever re-run a body that had
already failed.

The fix reads the signature instead of the failure, through the reading this
package already has: :func:`~dataknobs_fsm.functions.base.accepts_context`
answers "does it want the context", and its sibling
:func:`~dataknobs_fsm.functions.base.accepts_one_argument` answers "will it
take the state object alone". Which argument each callable receives is
unchanged --- the shadow census over all 1970 tests in this package found two
distinct callables reaching these sites and the new reading agrees with the
old outcome for both --- so what goes away is only the second call.

Real constructs only: real configs, a real :class:`FSM`, real engines.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.config.builder import build_fsm
from dataknobs_fsm.core.context_factory import ContextFactory
from dataknobs_fsm.core.data_wrapper import StateDataWrapper
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine

CONFIG: Dict[str, Any] = {
    "name": "retry_probe",
    "data_mode": "copy",
    "resources": [],
    "states": [
        {"name": "start", "is_start": True, "resources": []},
        {"name": "work", "resources": []},
        {"name": "done", "is_end": True},
    ],
    "arcs": [
        {"from": "start", "to": "work", "name": "go"},
        {"from": "work", "to": "done", "name": "fin"},
    ],
}


class Shout:
    """A consumer transform with an ordinary bug: ``.upper()`` on a number.

    Records what it was handed each time, so a second invocation is visible
    as a second entry rather than merely as a different answer.
    """

    def __init__(self) -> None:
        self.received: List[str] = []

    def __call__(self, record: Any, context: Any = None) -> Dict[str, Any]:
        self.received.append(type(record).__name__)
        return {**record, "shout": record["name"].upper()}


class ShoutValidator:
    """The same bug, on the interface the validator loop calls."""

    def __init__(self) -> None:
        self.received: List[str] = []

    def validate(self, record: Any, context: Any = None) -> Dict[str, Any]:
        self.received.append(type(record).__name__)
        return {"shout": record["name"].upper()}


def _fsm(attribute: str, function: Any) -> Any:
    """A built FSM with ``function`` attached to ``work`` as a plain callable.

    ``add_transform_function`` and ``add_validation_function`` both declare
    ``RegisteredFunction``, which includes a bare ``Callable`` --- so this is
    the shape a consumer building an FSM programmatically produces, and the
    one that reaches these sites without a wrapper in between.
    """
    fsm = build_fsm(dict(CONFIG))
    getattr(fsm.states["work"], attribute).append(function)
    return fsm


async def _run_engine(attribute: str, function: Any, data: Dict[str, Any]) -> bool:
    fsm = _fsm(attribute, function)
    engine = AsyncExecutionEngine(fsm)
    context = ContextFactory.create_context(fsm, data, data_mode=ProcessingMode.SINGLE)
    success, _ = await engine.execute(context, data)
    return bool(success)


# --------------------------------------------------------------------------- #
# The engine's state-transform dispatch
# --------------------------------------------------------------------------- #


async def test_a_state_transform_whose_body_fails_is_not_run_again() -> None:
    """``record['name'].upper()`` on an integer is the consumer's bug, run once.

    It was run twice: the ``AttributeError`` from the body was read as "wrong
    signature", so the engine called the same function again with a ``dict``
    and a context.
    """
    shout = Shout()

    success = await _run_engine("transform_functions", shout, {"name": 5})

    assert shout.received == ["StateDataWrapper"], shout.received
    assert not success


async def test_a_working_state_transform_still_runs_exactly_once() -> None:
    """The control: no fallback was ever involved when nothing raised."""
    shout = Shout()

    success = await _run_engine("transform_functions", shout, {"name": "ada"})

    assert shout.received == ["StateDataWrapper"], shout.received
    assert success


async def test_a_state_transform_that_cannot_take_the_state_object_still_gets_both() -> None:
    """The case the clause was written for: it is decided by reading, not failing.

    Two *required* positional parameters cannot be called with the state
    object alone, so this callable receives the record and the context --- and
    it always did. What changed is that nothing has to fail for it to.
    """
    seen: List[Any] = []

    def needs_both(record: Dict[str, Any], context: Any) -> Dict[str, Any]:
        seen.append((type(record).__name__, context is not None))
        return {**record, "seen": True}

    success = await _run_engine("transform_functions", needs_both, {"name": "ada"})

    assert seen == [("dict", True)], seen
    assert success


# --------------------------------------------------------------------------- #
# The engine's validator loop
# --------------------------------------------------------------------------- #


async def test_a_validator_whose_body_fails_is_not_run_again() -> None:
    """The quietest of the four: the loop swallows the failure either way.

    So the run reports success while the consumer's validator has run twice
    and raised twice, and nothing anywhere says so.
    """
    validator = ShoutValidator()

    success = await _run_engine("validation_functions", validator, {"name": 5})

    assert validator.received == ["StateDataWrapper"], validator.received
    assert success, "a failing validator does not fail the record — only the count changed"


async def test_a_working_validator_still_runs_exactly_once() -> None:
    validator = ShoutValidator()

    await _run_engine("validation_functions", validator, {"name": "ada"})

    assert validator.received == ["StateDataWrapper"], validator.received


# --------------------------------------------------------------------------- #
# The stepping API's own copy of the dispatch
# --------------------------------------------------------------------------- #


async def test_a_stepped_transform_whose_body_fails_is_not_run_again() -> None:
    """``AdvancedFSM`` keeps its own copy of this dispatch, and had the same bug."""
    shout = Shout()
    advanced = AdvancedFSM(_fsm("transform_functions", shout))
    context = advanced.create_context({"name": 5})

    result = await advanced.execute_step_async(context)

    assert shout.received == ["StateDataWrapper"], shout.received
    assert not result.success


async def test_a_working_stepped_transform_still_runs_exactly_once() -> None:
    shout = Shout()
    advanced = AdvancedFSM(_fsm("transform_functions", shout))
    context = advanced.create_context({"name": "ada"})

    result = await advanced.execute_step_async(context)

    assert shout.received == ["StateDataWrapper"], shout.received
    assert result.success


# --------------------------------------------------------------------------- #
# The arc-condition helper
# --------------------------------------------------------------------------- #


def _arc_condition_engine() -> tuple[Any, Any, Any]:
    fsm = build_fsm(dict(CONFIG))
    engine = AsyncExecutionEngine(fsm)
    context = ContextFactory.create_context(fsm, {"name": 5}, data_mode=ProcessingMode.SINGLE)
    return fsm, engine, context


def test_an_arc_condition_whose_body_fails_is_not_run_again() -> None:
    """``evaluate_arc_condition_common`` has no caller in this tree, and is fixed anyway.

    It is a published method on the shared engine base, so a consumer driving
    traversal themselves reaches it; "nothing in here calls it" is a fact
    about this repository, not about the method. Its preference is the
    mirror image of the transform sites --- it tries ``(record, context)``
    first --- and it had the same defect, with a worse ending: the second
    failure is swallowed by the enclosing ``except`` and the arc simply
    reports False, so a condition with a bug in it reads as a condition that
    said no.
    """
    calls: List[str] = []
    fsm, engine, context = _arc_condition_engine()

    def broken(record: Dict[str, Any], ctx: Any = None) -> bool:
        calls.append(type(record).__name__)
        return bool(record["name"] + 1)  # TypeError once ``name`` is not a number

    arc = fsm.states["start"].outgoing_arcs[0]
    arc.condition = broken
    context.data = {"name": "ada"}

    assert engine.evaluate_arc_condition_common(arc, context) is False
    assert calls == ["dict"], calls


def test_an_arc_condition_still_receives_the_context_when_it_declares_one() -> None:
    """The preference at this site is unchanged: two positionals get both."""
    seen: List[Any] = []
    fsm, engine, context = _arc_condition_engine()

    def wants_context(record: Dict[str, Any], ctx: Any = None) -> bool:
        seen.append((type(record).__name__, ctx is not None))
        return True

    arc = fsm.states["start"].outgoing_arcs[0]
    arc.condition = wants_context

    assert engine.evaluate_arc_condition_common(arc, context) is True
    assert seen == [("dict", True)], seen


def test_a_one_argument_arc_condition_is_called_with_the_record_alone() -> None:
    """And the shape that used to need a failed call to be discovered."""
    seen: List[Any] = []
    fsm, engine, context = _arc_condition_engine()

    def record_only(record: Dict[str, Any]) -> bool:
        seen.append(type(record).__name__)
        return True

    arc = fsm.states["start"].outgoing_arcs[0]
    arc.condition = record_only

    assert engine.evaluate_arc_condition_common(arc, context) is True
    assert seen == ["dict"], seen


# --------------------------------------------------------------------------- #
# The reading itself
# --------------------------------------------------------------------------- #


def test_the_arity_reading_agrees_with_what_a_call_would_do() -> None:
    """``accepts_one_argument`` answers the question the failed call used to.

    Every shape that reaches these sites, read rather than tried. The
    wrapper row is the one that matters in practice: it is what the
    registered-function door produces, and it is why this dispatch keeps
    preferring the state object rather than switching to
    :func:`accepts_context`, which would answer ``True`` for it and change
    what every registered transform receives.
    """
    from dataknobs_fsm.functions.base import accepts_one_argument

    assert accepts_one_argument(lambda state: state)
    assert accepts_one_argument(lambda record, context=None: record)
    assert accepts_one_argument(lambda *args: args)
    assert accepts_one_argument(lambda record, *args, **kwargs: record)
    assert not accepts_one_argument(lambda record, context: record)
    assert not accepts_one_argument(lambda: None)
    assert not accepts_one_argument(lambda record, *, context: record)

    wrapper = StateDataWrapper({"a": 1})
    assert accepts_one_argument(wrapper.get)

    # Unreadable signatures answer the default rather than guessing, and the
    # default here is the historical first attempt: the state object alone.
    assert accepts_one_argument(object(), default=True)
    assert not accepts_one_argument(object(), default=False)


def test_a_required_keyword_only_is_read_past_var_positional() -> None:
    """``*args`` does not excuse a required keyword-only parameter.

    ``accepts_one_argument`` states the rule outright: a required
    keyword-only parameter cannot be filled by a positional call, so such a
    callable answers ``False`` and is given ``(record, context)`` instead,
    where a keyword ``context`` at least has a chance of binding by name.
    The reading applied that rule only when it reached the parameter ---
    and ``*args`` returned before it could, because ``*args`` alone does
    satisfy the positional half of the question.

    So the one shape where both halves disagree, ``(*args, mandatory)``, was
    read as one-argument-callable and called with the state object alone.
    The call raises before the body exists, which is the failure the whole
    reading was written to stop guessing at --- and it is asserted here
    rather than described, so the reading is checked against what a call
    does rather than against this docstring.
    """
    from dataknobs_fsm.functions.base import accepts_one_argument

    def var_positional_then_required_keyword(*args: Any, mandatory: Any) -> Any:
        return args, mandatory

    with pytest.raises(TypeError):
        var_positional_then_required_keyword({"a": 1})

    assert not accepts_one_argument(var_positional_then_required_keyword)

    # The keyword-only parameters that a positional call CAN leave unbound
    # are unaffected: one with a default, and ``**kwargs``, both still
    # answer ``True`` beyond a ``*args``.
    assert accepts_one_argument(lambda *args, context=None: args)
    assert accepts_one_argument(lambda *args, **kwargs: args)

"""Config -> runtime threading for push-arc data isolation.

These tests pin that a config-authored ``data_isolation`` on a push arc reaches
the runtime ``PushArc.isolation_mode`` field, that the config field speaks the
same isolation enum the runtime honors (``copy``/``reference``/``serialize``),
and that the legacy ``direct`` value (which never had push-isolation meaning)
now fails loud at parse time rather than being silently dropped.

The config field is exercised on **both** authoring paths: direct typed
``PushArcConfig`` construction, and the dict/YAML path that real configs use
(raw arc dicts coerced through ``StateConfig.validate_arcs``). The dict path is
the one that matters for the "``direct`` fails loud at load" migration
guarantee -- a raw dict whose ``PushArcConfig`` fails validation must raise,
not silently degrade to a plain ``ArcConfig`` that drops ``target_network``.

Execution-status note: the async ``AsyncExecutionEngine`` now executes push
arcs -- it pushes the sub-network, isolates the data via the shared
``DataIsolationMode.apply`` helper, enters the sub-network, and pops back on
completion (proven below on the ``AsyncSimpleFSM`` path). ``NetworkExecutor`` --
a public, exported executor -- also traverses sub-networks and honors all three
modes at the data-input boundary (proven below). The sync ``ExecutionEngine``
likewise traverses sub-networks via its own push/pop path.
"""

import inspect

import pytest
from pydantic import ValidationError

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    FunctionReference,
    NetworkConfig,
    PushArcConfig,
    StateConfig,
)
from dataknobs_fsm.core.arc import DataIsolationMode, PushArc
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.network import NetworkExecutor
from dataknobs_fsm.functions.base import StateTransitionError


def _build_fsm_with_push(isolation_value: str):
    """Build a two-network FSM whose start state has a push arc carrying the
    given ``data_isolation`` config value.
    """
    config = FSMConfig(
        name="iso_fsm",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[
                            PushArcConfig(
                                target="after",
                                target_network="sub",
                                return_state="after",
                                data_isolation=isolation_value,
                            )
                        ],
                    ),
                    StateConfig(name="after", arcs=[ArcConfig(target="end")]),
                    StateConfig(name="end", is_end=True),
                ],
            ),
            NetworkConfig(
                name="sub",
                states=[
                    StateConfig(name="s1", is_start=True, arcs=[ArcConfig(target="s2")]),
                    StateConfig(name="s2", is_end=True),
                ],
            ),
        ],
    )
    return FSMBuilder().build(config)


@pytest.mark.parametrize(
    "value, expected_mode",
    [
        ("copy", DataIsolationMode.COPY),
        ("reference", DataIsolationMode.REFERENCE),
        ("serialize", DataIsolationMode.SERIALIZE),
    ],
)
def test_builder_threads_isolation_mode_to_runtime_arc(value, expected_mode):
    """The configured ``data_isolation`` reaches ``PushArc.isolation_mode``.

    Reproduces the dropped-value bug: before threading, the builder never
    passed ``isolation_mode=`` so the runtime arc always carried the default
    ``COPY`` regardless of config, and ``serialize`` could not even be authored.
    """
    fsm = _build_fsm_with_push(value)
    arc = fsm.networks["main"].states["start"].arcs[0]
    assert isinstance(arc, PushArc)
    assert arc.isolation_mode == expected_mode


def test_pusharc_config_parses_serialize():
    """``serialize`` is now an authorable push-arc isolation value.

    Before the enum retype the config field used the state-level data-mode enum
    (copy/reference/direct), which raised on ``serialize`` at parse time.
    """
    cfg = PushArcConfig(target="after", target_network="sub", data_isolation="serialize")
    assert cfg.data_isolation == DataIsolationMode.SERIALIZE


def test_pusharc_config_rejects_direct_migration():
    """``direct`` is no longer a valid push-arc isolation value.

    ``direct`` is a state-level data-handling mode; it never had push-isolation
    semantics and was silently dropped at build time. After the field speaks the
    isolation enum, authoring it fails loud at parse rather than no-op'ing.
    """
    with pytest.raises(ValidationError):
        PushArcConfig(target="after", target_network="sub", data_isolation="direct")


# --- Dict / YAML load path (the path real configs use) -----------------------
#
# Raw arc dicts are coerced to PushArcConfig/ArcConfig by
# ``StateConfig.validate_arcs``. These proofs matter most for the migration
# guarantee: if that validator does not fire (e.g. a reversed
# ``@field_validator``/``@classmethod`` decorator order leaves it unregistered),
# a push-arc dict falls back to smart-union coercion, and a ``direct`` value --
# invalid for PushArcConfig -- silently degrades to a plain ArcConfig that drops
# ``target_network`` instead of raising. That is the exact opposite of "fails
# loud at load", so it must be pinned on the dict path, not just typed
# construction.


def _state_from_arc_dict(arc: dict) -> StateConfig:
    """Coerce a raw arc dict through ``StateConfig.validate_arcs``."""
    return StateConfig(name="start", is_start=True, arcs=[arc])


def _build_fsm_with_push_dict(isolation_value: str):
    """Build the two-network FSM via the raw-dict load path.

    The push arc is a plain dict, so it is coerced through
    ``StateConfig.validate_arcs`` exactly as a YAML/JSON config would be.
    """
    config = FSMConfig.model_validate(
        {
            "name": "iso_fsm",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {
                            "name": "start",
                            "is_start": True,
                            "arcs": [
                                {
                                    "target": "after",
                                    "target_network": "sub",
                                    "return_state": "after",
                                    "data_isolation": isolation_value,
                                }
                            ],
                        },
                        {"name": "after", "arcs": [{"target": "end"}]},
                        {"name": "end", "is_end": True},
                    ],
                },
                {
                    "name": "sub",
                    "states": [
                        {"name": "s1", "is_start": True, "arcs": [{"target": "s2"}]},
                        {"name": "s2", "is_end": True},
                    ],
                },
            ],
        }
    )
    return FSMBuilder().build(config)


@pytest.mark.parametrize(
    "value, expected_mode",
    [
        ("copy", DataIsolationMode.COPY),
        ("reference", DataIsolationMode.REFERENCE),
        ("serialize", DataIsolationMode.SERIALIZE),
    ],
)
def test_dict_load_path_preserves_push_arc_and_isolation(value, expected_mode):
    """A raw push-arc dict coerces to PushArcConfig and keeps its fields.

    Pins that ``validate_arcs`` actually fires: a dict with ``target_network``
    must become a ``PushArcConfig`` (not silently a plain ``ArcConfig``), keep
    ``target_network``, and carry the isolation value.
    """
    state = _state_from_arc_dict(
        {"target": "after", "target_network": "sub", "data_isolation": value}
    )
    arc = state.arcs[0]
    assert isinstance(arc, PushArcConfig)
    assert arc.target_network == "sub"
    assert arc.data_isolation == expected_mode


@pytest.mark.parametrize(
    "value, expected_mode",
    [
        ("copy", DataIsolationMode.COPY),
        ("reference", DataIsolationMode.REFERENCE),
        ("serialize", DataIsolationMode.SERIALIZE),
    ],
)
def test_dict_load_path_threads_isolation_mode_to_runtime_arc(value, expected_mode):
    """The dict-authored ``data_isolation`` reaches ``PushArc.isolation_mode``.

    End-to-end on the path real YAML/JSON configs travel: raw dict ->
    validate_arcs -> PushArcConfig -> builder -> runtime ``PushArc``.
    """
    fsm = _build_fsm_with_push_dict(value)
    arc = fsm.networks["main"].states["start"].arcs[0]
    assert isinstance(arc, PushArc)
    assert arc.isolation_mode == expected_mode


def test_dict_load_path_rejects_direct_migration():
    """``direct`` raises at load on the dict path -- it does not degrade.

    The migration guarantee: a ``direct`` push-arc dict must fail loud, not fall
    back to a plain ``ArcConfig`` that silently drops ``target_network``.
    """
    with pytest.raises(ValidationError):
        _state_from_arc_dict(
            {"target": "after", "target_network": "sub", "data_isolation": "direct"}
        )


def test_no_dead_isolation_branch_in_network_executor():
    """The push-isolation handler has no unreachable mode branch.

    Guards against the regression where the handler referenced a non-existent
    arc attribute (``data_isolation_mode``) and compared against a string that
    is not an isolation-enum member -- a branch that could never execute and a
    comment describing behavior no code path produced.
    """
    source = inspect.getsource(NetworkExecutor._handle_push_arc)
    assert "data_isolation_mode" not in source
    assert "'partial'" not in source
    assert '"partial"' not in source


def test_network_executor_copy_push_still_enters_subflow():
    """The live push branches still traverse the sub-network after the
    dead-branch removal (regression guard for W2).

    Sub-network entry is proven by a marker the sub start-state transform sets,
    since ``state_history`` records only main-network states.
    """
    mark_sub = FunctionReference(
        type="inline",
        code=(
            "def transform(data, context):\n"
            "    context.variables.setdefault('visited', []).append('s1')\n"
            "    return data\n"
        ),
    )
    config = FSMConfig(
        name="iso_fsm",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[
                            PushArcConfig(
                                target="after",
                                target_network="sub",
                                return_state="after",
                                data_isolation="copy",
                            )
                        ],
                    ),
                    StateConfig(name="after", arcs=[ArcConfig(target="end")]),
                    StateConfig(name="end", is_end=True),
                ],
            ),
            NetworkConfig(
                name="sub",
                states=[
                    StateConfig(
                        name="s1",
                        is_start=True,
                        arcs=[ArcConfig(target="s2")],
                        transforms=[mark_sub],
                    ),
                    StateConfig(name="s2", is_end=True),
                ],
            ),
        ],
    )
    fsm = FSMBuilder().build(config)
    executor = NetworkExecutor(fsm)
    context = ExecutionContext()
    context.data = {"id": 1}
    success, _result = executor.execute_network("main", context, context.data)
    # The sub-network's start state transform must have run (subflow traversed).
    assert "s1" in context.variables.get("visited", [])
    assert success is True


def _run_network_push_isolation(isolation_value: str):
    """Run a push through ``NetworkExecutor`` whose sub start-state transform
    mutates the data, and report whether the parent's *original* data object was
    touched.

    Returns ``(original_obj, parent_context, success)``. Under data-isolating
    modes the sub-network mutates an isolated snapshot, so the parent's original
    object is left untouched; under REFERENCE the sub shares the parent's object
    and mutates it in place.
    """
    touch_sub = FunctionReference(
        type="inline",
        code=(
            "def transform(data, context):\n"
            "    data['sub_touched'] = True\n"
            "    return data\n"
        ),
    )
    config = FSMConfig(
        name="iso_fsm",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[
                            PushArcConfig(
                                target="after",
                                target_network="sub",
                                return_state="after",
                                data_isolation=isolation_value,
                            )
                        ],
                    ),
                    StateConfig(name="after", arcs=[ArcConfig(target="end")]),
                    StateConfig(name="end", is_end=True),
                ],
            ),
            NetworkConfig(
                name="sub",
                states=[
                    StateConfig(
                        name="s1",
                        is_start=True,
                        arcs=[ArcConfig(target="s2")],
                        transforms=[touch_sub],
                    ),
                    StateConfig(name="s2", is_end=True),
                ],
            ),
        ],
    )
    fsm = FSMBuilder().build(config)
    executor = NetworkExecutor(fsm)
    context = ExecutionContext()
    original = {"id": 1}
    context.data = original
    success, _result = executor.execute_network("main", context, context.data)
    return original, context, success


@pytest.mark.parametrize("value", ["copy", "serialize"])
def test_network_executor_isolating_modes_leave_parent_data_untouched(value):
    """COPY/SERIALIZE give the sub-network an isolated snapshot of the data.

    A sub-network mutation does not leak into the parent's original data object;
    the result is merged back onto ``context.data`` (a distinct object) on
    success. This is the behavioral payoff of threading ``data_isolation``
    through to a public, sub-network-traversing executor.
    """
    original, context, success = _run_network_push_isolation(value)
    assert success is True
    assert "sub_touched" not in original  # parent's original object untouched
    assert context.data is not original  # merged-back result is a fresh object
    assert context.data.get("sub_touched") is True  # result still propagates


def test_network_executor_reference_mode_shares_parent_data():
    """REFERENCE shares the parent's data object with the sub-network.

    The contrast case to the isolating modes: the sub mutates the parent's
    object in place, so the marker appears on the original object.
    """
    original, context, success = _run_network_push_isolation("reference")
    assert success is True
    assert original.get("sub_touched") is True  # mutated in place
    assert context.data is original  # same object, no snapshot


def test_serialize_apply_uses_project_encoder_not_stdlib_json():
    """SERIALIZE round-trips through the *project* JSON encoder, not stdlib json.

    ``DataIsolationMode.apply`` must use ``dataknobs_fsm.utils.json_encoder`` so the
    FSM-specific types it special-cases (here, an object exposing ``to_dict()``)
    survive the round-trip. Plain ``json.dumps`` raises ``TypeError`` on such an
    object, so this pins that ``arc.py`` did not regress to stdlib ``json`` (the
    original divergence the shared helper was introduced to fix).
    """
    import json as _stdlib_json

    class _HasToDict:
        def to_dict(self) -> dict:
            return {"kind": "widget", "n": 3}

    payload = {"id": 1, "obj": _HasToDict()}

    # stdlib json cannot serialize the custom object...
    with pytest.raises(TypeError):
        _stdlib_json.dumps(payload)

    # ...but the project encoder converts it via to_dict() and round-trips.
    out = DataIsolationMode.SERIALIZE.apply(payload)
    assert out == {"id": 1, "obj": {"kind": "widget", "n": 3}}
    assert out is not payload  # a fresh, isolated snapshot


def test_network_executor_enforces_max_depth_across_nested_push_arcs():
    """``max_depth`` is enforced across nested push arcs, not just within one
    context's stack.

    Each sub-network now runs in a *fresh* ``ExecutionContext`` whose
    ``network_stack`` starts empty, so a stack-length depth check would see 0 at
    every nesting level and never fire -- an unconditionally self-recursive push
    network would recurse until Python's own recursion limit (``RecursionError``)
    instead of failing loud at the configured ceiling. Depth is carried forward
    on an explicit counter so the bounded ``StateTransitionError`` is raised at
    ``max_depth``.
    """
    config = FSMConfig(
        name="recur_fsm",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="s0",
                        is_start=True,
                        arcs=[
                            PushArcConfig(
                                target="end",
                                target_network="loop",
                                return_state="end",
                            )
                        ],
                    ),
                    StateConfig(name="end", is_end=True),
                ],
            ),
            NetworkConfig(
                name="loop",
                states=[
                    StateConfig(
                        name="p",
                        is_start=True,
                        arcs=[
                            # Unconditionally pushes back into itself -> would
                            # recurse without bound absent depth enforcement.
                            PushArcConfig(
                                target="done",
                                target_network="loop",
                                return_state="done",
                            )
                        ],
                    ),
                    StateConfig(name="done", is_end=True),
                ],
            ),
        ],
    )
    fsm = FSMBuilder().build(config)
    executor = NetworkExecutor(fsm, max_depth=3)
    context = ExecutionContext()
    context.data = {"id": 1}
    with pytest.raises(StateTransitionError, match="depth"):
        executor.execute_network("main", context, context.data)


async def test_async_push_arc_enters_subflow():
    """The async engine executes a push arc and enters the sub-network.

    Reproduces (and now confirms the fix for) the gap where the async engine
    treated a push arc as a flat transition with no sub-network push/pop, so
    the subflow was never entered. After wiring async push-arc execution the
    sub-network's start-state transform runs and sets ``entered_subflow``.
    ``SimpleFSM.process()`` shares this async engine, so the same behavior now
    holds there too.
    """
    from dataknobs_fsm.api.async_simple import AsyncSimpleFSM

    config = {
        "name": "iso_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {
                        "name": "start",
                        "is_start": True,
                        "arcs": [
                            {
                                "target": "after",
                                "target_network": "sub",
                                "return_state": "after",
                            }
                        ],
                    },
                    {"name": "after", "arcs": [{"target": "end"}]},
                    {"name": "end", "is_end": True},
                ],
            },
            {
                "name": "sub",
                "states": [
                    {
                        "name": "s1",
                        "is_start": True,
                        "arcs": [{"target": "s2"}],
                        "transforms": [
                            {
                                "type": "inline",
                                "code": (
                                    "def transform(data, context):\n"
                                    "    data['entered_subflow'] = True\n"
                                    "    return data\n"
                                ),
                            }
                        ],
                    },
                    {"name": "s2", "is_end": True},
                ],
            },
        ],
    }
    fsm = AsyncSimpleFSM(config)
    try:
        result = await fsm.process({"id": 1})
    finally:
        await fsm.close()
    assert result.get("data", {}).get("entered_subflow") is True


def _async_push_isolation_config(isolation_value: str | None) -> dict:
    """Two-network config whose sub start-state transform mutates the data.

    The push arc carries ``data_isolation`` (omitted when ``None`` so the
    runtime default COPY applies). The sub-network's ``s1`` transform sets
    ``sub_touched`` on the data it receives, and the main flow runs
    start -(push)-> sub(s1->s2) -(pop)-> after -> end.
    """
    arc: dict = {
        "target": "after",
        "target_network": "sub",
        "return_state": "after",
    }
    if isolation_value is not None:
        arc["data_isolation"] = isolation_value
    return {
        "name": "iso_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True, "arcs": [arc]},
                    {"name": "after", "arcs": [{"target": "end"}]},
                    {"name": "end", "is_end": True},
                ],
            },
            {
                "name": "sub",
                "states": [
                    {
                        "name": "s1",
                        "is_start": True,
                        "arcs": [{"target": "s2"}],
                        "transforms": [
                            {
                                "type": "inline",
                                "code": (
                                    "def transform(data, context):\n"
                                    "    data['sub_touched'] = True\n"
                                    "    return data\n"
                                ),
                            }
                        ],
                    },
                    {"name": "s2", "is_end": True},
                ],
            },
        ],
    }


@pytest.mark.parametrize("isolation_value", ["copy", "serialize", None])
async def test_async_push_arc_isolating_modes_propagate_result(isolation_value):
    """COPY/SERIALIZE (and the default) enter the subflow and merge the result.

    The async engine isolates the sub-network's data view via
    ``DataIsolationMode.apply``; the sub mutates its isolated snapshot and the
    result is merged back onto ``context.data`` so it still propagates to the
    final result. (``None`` exercises the runtime default, which is COPY.)
    """
    from dataknobs_fsm.api.async_simple import AsyncSimpleFSM

    fsm = AsyncSimpleFSM(_async_push_isolation_config(isolation_value))
    try:
        result = await fsm.process({"id": 1})
    finally:
        await fsm.close()
    assert result.get("data", {}).get("sub_touched") is True


async def test_async_push_arc_reference_mode_enters_subflow():
    """REFERENCE shares the parent's data object with the sub-network.

    The contrast case to the isolating modes: with REFERENCE the sub-network
    operates on the parent's data by reference, so its mutation is visible in
    the final result. The async engine still enters and pops the subflow.
    """
    from dataknobs_fsm.api.async_simple import AsyncSimpleFSM

    fsm = AsyncSimpleFSM(_async_push_isolation_config("reference"))
    try:
        result = await fsm.process({"id": 1})
    finally:
        await fsm.close()
    assert result.get("data", {}).get("sub_touched") is True

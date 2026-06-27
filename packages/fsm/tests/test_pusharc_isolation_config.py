"""Config -> runtime threading for push-arc data isolation.

These tests pin that a config-authored ``data_isolation`` on a push arc reaches
the runtime ``PushArc.isolation_mode`` field, that the config field speaks the
same isolation enum the runtime honors (``copy``/``reference``/``serialize``),
and that the legacy ``direct`` value (which never had push-isolation meaning)
now fails loud at parse time rather than being silently dropped.

Scope note: these are *config/build-boundary* proofs. No reachable execution
path today runs a push arc through a full sub-network while honoring
``isolation_mode`` (the sync ``ExecutionEngine`` does not traverse sub-networks;
the async engine does not execute push arcs at all; the test-only
``NetworkExecutor`` shares the data reference in every mode). So threading the
value changes no run-time behavior yet -- the behavioral payoff arrives when
push-arc execution with isolation is wired on an engine. These tests therefore
assert what is observable today: that the configured value is carried onto the
runtime arc and that the enum/migration shape is settled.
"""

import inspect

import pytest
from pydantic import ValidationError

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    NetworkConfig,
    PushArcConfig,
    StateConfig,
)
from dataknobs_fsm.core.arc import DataIsolationMode, PushArc
from dataknobs_fsm.execution.network import NetworkExecutor


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
    from dataknobs_fsm.config.schema import FunctionReference
    from dataknobs_fsm.execution.context import ExecutionContext

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


@pytest.mark.xfail(
    reason=(
        "The async execution engine does not yet execute push arcs: a push arc "
        "is treated as a flat transition with no sub-network push/pop, so the "
        "subflow is never entered. Flip to a real assertion when async push-arc "
        "execution is wired."
    ),
    strict=False,
)
async def test_async_push_arc_enters_subflow():
    """Pins the bound that push arcs do not execute on the async path today.

    ``SimpleFSM.process()`` shares this async engine, so the same flat-transition
    behavior holds there too.
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

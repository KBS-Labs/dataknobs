"""Async-path coverage for subflow (push-arc) behaviors.

These exercise push-arc behaviors end-to-end on the single async engine via the
public ``AsyncSimpleFSM`` API. They replace the equivalent scenarios that were
previously only proven through the removed synchronous ``NetworkExecutor`` /
``ExecutionEngine`` subflow paths:

- **max-depth enforcement** — a self-recursive push must terminate at the
  engine's nesting ceiling instead of recursing without bound.
- **missing target network** — a push to an undefined network fails cleanly
  (the record errors) rather than entering nothing silently.
- **custom initial state** — the ``network:state`` target syntax enters the
  sub-network at the named state, skipping its declared start state.
- **multi-state resource inheritance** — a resource inherited from the pushing
  state is visible to *every* state in the sub-network, and is gone again once
  the run returns to the parent.

Real constructs only (real FSM builds, inline transforms, a real
``AsyncDatabase`` memory backend for the resource test — no mocks).
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode


# --------------------------------------------------------------------------- #
# max-depth enforcement on a self-recursive push
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_push_arc_enforces_max_depth() -> None:
    """A self-recursive push terminates at the nesting ceiling, not unbounded.

    ``loop.p`` unconditionally pushes back into ``loop``; absent depth
    enforcement this would recurse until Python's own recursion limit. The async
    engine's ``subflow_depth_exceeded`` guard fails the push once the nesting
    ceiling is reached, so the run terminates with a clean failure rather than a
    ``RecursionError``. The test completing at all is the proof that nesting is
    bounded.
    """
    config = {
        "name": "recur_fsm",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {
                        "name": "s0",
                        "is_start": True,
                        "arcs": [
                            {
                                "target": "end",
                                "target_network": "loop",
                                "return_state": "end",
                            }
                        ],
                    },
                    {"name": "end", "is_end": True},
                ],
            },
            {
                "name": "loop",
                "states": [
                    {
                        "name": "p",
                        "is_start": True,
                        # Unconditionally pushes back into itself.
                        "arcs": [
                            {
                                "target": "done",
                                "target_network": "loop",
                                "return_state": "done",
                            }
                        ],
                    },
                    {"name": "done", "is_end": True},
                ],
            },
        ],
    }
    fsm = AsyncSimpleFSM(config)
    try:
        result = await fsm.process({"id": 1})
    finally:
        await fsm.close()

    assert result["success"] is False, (
        "an unbounded self-recursive push was expected to fail at the nesting "
        f"ceiling, not complete successfully (got {result})"
    )


# --------------------------------------------------------------------------- #
# push to a non-existent target network fails cleanly
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_push_to_nonexistent_network_fails() -> None:
    """A push whose target network does not exist fails the record cleanly.

    ``_execute_push_arc`` returns False when the target network is missing, so
    the only transition out of ``start`` fails and the run reports failure
    rather than silently entering nothing.
    """
    config = {
        "name": "missing_target",
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
                                "target_network": "does_not_exist",
                                "return_state": "after",
                            }
                        ],
                    },
                    {"name": "after", "arcs": [{"target": "end"}]},
                    {"name": "end", "is_end": True},
                ],
            },
        ],
    }
    fsm = AsyncSimpleFSM(config)
    try:
        result = await fsm.process({"id": 1})
    finally:
        await fsm.close()

    assert result["success"] is False, (
        "a push to an undefined network should fail the record, not succeed "
        f"(got {result})"
    )


# --------------------------------------------------------------------------- #
# custom initial state via the network:state target syntax
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_push_arc_custom_initial_state() -> None:
    """``target_network: "sub:second"`` enters the sub-network at ``second``.

    The ``network:state`` syntax overrides the sub-network's declared start
    state, so the push enters ``second`` directly: its transform runs and the
    skipped start state ``first`` does not.
    """
    config = {
        "name": "custom_initial",
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
                                "target_network": "sub:second",
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
                        "name": "first",
                        "is_start": True,
                        "arcs": [{"target": "second"}],
                        "transforms": [
                            {
                                "type": "inline",
                                "code": (
                                    "def transform(data, context):\n"
                                    "    data['first_ran'] = True\n"
                                    "    return data\n"
                                ),
                            }
                        ],
                    },
                    {
                        "name": "second",
                        "arcs": [{"target": "done"}],
                        "transforms": [
                            {
                                "type": "inline",
                                "code": (
                                    "def transform(data, context):\n"
                                    "    data['second_ran'] = True\n"
                                    "    return data\n"
                                ),
                            }
                        ],
                    },
                    {"name": "done", "is_end": True},
                ],
            },
        ],
    }
    fsm = AsyncSimpleFSM(config)
    try:
        result = await fsm.process({"id": 1})
    finally:
        await fsm.close()

    assert result["success"], f"FSM did not complete cleanly: {result}"
    data = result.get("data", {})
    assert data.get("second_ran") is True, (
        "the sub-network was not entered at the custom initial state 'second'"
    )
    assert data.get("first_ran") is not True, (
        "the declared start state 'first' ran — the custom initial state did not "
        "override it"
    )


# --------------------------------------------------------------------------- #
# a resource inherited from the pushing state is visible across all sub states
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_subflow_inherited_resource_visible_across_substates() -> None:
    """An inherited pushing-state resource reaches every state in the sub-network.

    ``main.start`` declares ``target_db`` and pushes into ``sub``; neither sub
    state declares the resource, so each can only see it by inheritance. Both
    ``s1`` and ``s2`` record whether ``target_db`` is in their function context,
    and the post-return ``after`` state records that it is gone again once the
    subflow has popped (the pushing state's owned resource is released on pop).
    """
    config = {
        "name": "multistate_inherit",
        "main_network": "main",
        "resources": [
            {"name": "target_db", "type": "async_database", "config": {"type": "memory"}},
        ],
        "networks": [
            {
                "name": "main",
                "states": [
                    {
                        "name": "start",
                        "is_start": True,
                        "resources": ["target_db"],
                        "arcs": [
                            {
                                "target": "after",
                                "target_network": "sub",
                                "return_state": "after",
                            }
                        ],
                    },
                    {
                        "name": "after",
                        "arcs": [{"target": "end"}],
                        "transforms": [
                            {
                                "type": "inline",
                                "code": (
                                    "def transform(data, context):\n"
                                    "    res = getattr(context, 'resources', None) or {}\n"
                                    "    data['after_has_db'] = 'target_db' in res\n"
                                    "    return data\n"
                                ),
                            }
                        ],
                    },
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
                                    "    res = getattr(context, 'resources', None) or {}\n"
                                    "    data['s1_has_db'] = 'target_db' in res\n"
                                    "    return data\n"
                                ),
                            }
                        ],
                    },
                    {
                        "name": "s2",
                        "arcs": [{"target": "s3"}],
                        "transforms": [
                            {
                                "type": "inline",
                                "code": (
                                    "def transform(data, context):\n"
                                    "    res = getattr(context, 'resources', None) or {}\n"
                                    "    data['s2_has_db'] = 'target_db' in res\n"
                                    "    return data\n"
                                ),
                            }
                        ],
                    },
                    {"name": "s3", "is_end": True},
                ],
            },
        ],
    }
    fsm = AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY)
    try:
        result = await fsm.process({"id": "1"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()

    data = result.get("data", {})
    assert data.get("s1_has_db") is True, (
        "the first sub-network state did not inherit the pushing state's "
        "'target_db' resource"
    )
    assert data.get("s2_has_db") is True, (
        "a later sub-network state did not see the inherited 'target_db' "
        "resource — inheritance must span every state in the sub-network"
    )
    assert data.get("after_has_db") is False, (
        "the pushing state's 'target_db' resource was still present after the "
        "subflow returned — it should be released on pop"
    )


# --------------------------------------------------------------------------- #
# resource inheritance accumulates across three nested push levels
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_three_level_subflow_resource_inheritance() -> None:
    """Resources accumulate down three nested push levels.

    ``level1`` declares ``r1`` and pushes into ``level2`` (declares ``r2``),
    which pushes into ``level3`` (declares ``r3``). The deepest state must see
    all three — ``r1`` and ``r2`` inherited from the two pushing states above it,
    plus its own ``r3`` — proving inheritance is carried forward across every
    push level, not just one. The check is recorded on the data (which merges
    back up the pop chain), not on ``context.variables`` (which a transform's
    function context does not share).
    """
    check = (
        "def transform(data, context):\n"
        "    res = getattr(context, 'resources', None) or {}\n"
        "    data['l3_sees'] = sorted(n for n in ('r1', 'r2', 'r3') if n in res)\n"
        "    return data\n"
    )
    config = {
        "name": "three_level",
        "main_network": "level1",
        "resources": [
            {"name": "r1", "type": "async_database", "config": {"type": "memory"}},
            {"name": "r2", "type": "async_database", "config": {"type": "memory"}},
            {"name": "r3", "type": "async_database", "config": {"type": "memory"}},
        ],
        "networks": [
            {
                "name": "level1",
                "states": [
                    {
                        "name": "l1_start",
                        "is_start": True,
                        "resources": ["r1"],
                        "arcs": [
                            {"target": "l1_end", "target_network": "level2", "return_state": "l1_end"}
                        ],
                    },
                    {"name": "l1_end", "is_end": True},
                ],
            },
            {
                "name": "level2",
                "states": [
                    {
                        "name": "l2_start",
                        "is_start": True,
                        "resources": ["r2"],
                        "arcs": [
                            {"target": "l2_end", "target_network": "level3", "return_state": "l2_end"}
                        ],
                    },
                    {"name": "l2_end", "is_end": True},
                ],
            },
            {
                "name": "level3",
                "states": [
                    {
                        "name": "l3_start",
                        "is_start": True,
                        "resources": ["r3"],
                        "arcs": [{"target": "l3_end"}],
                        "transforms": [{"type": "inline", "code": check}],
                    },
                    {"name": "l3_end", "is_end": True},
                ],
            },
        ],
    }
    fsm = AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY)
    try:
        result = await fsm.process({"id": "1"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()

    assert result.get("data", {}).get("l3_sees") == ["r1", "r2", "r3"], (
        "the deepest sub-network state did not inherit resources from all parent "
        f"push levels (got {result.get('data', {}).get('l3_sees')!r})"
    )

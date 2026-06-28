"""Reproduce-first tests for async subflow state-resource lifecycle.

Two coupled correctness properties of the async engine's state entry, both of
which only manifest with a real ``ResourceManager`` plus state
``resource_requirements`` (so the rest of the suite does not exercise them):

1. **Inheritance from the pushing state, regardless of how it was entered.** A
   push arc must let the sub-network inherit the pushing state's acquired
   resources. Inheritance seeds from ``current_state_resources``; before the
   initial-state and regular-transition entries were routed through the shared
   ``enter_state`` they never populated it, so a push inherited nothing and a
   sub-network transform relying on an inherited resource silently no-op'd. This
   is covered for both entry paths: a pushing state entered via the initial path
   (``is_start``) and one entered via a regular A→B transition.

2. **Release on pop.** A state holds the resources it acquired itself while it
   is active (so a nested push can inherit them); when it is left — a regular
   transition, or a subflow pop — those owned resources must be released. The
   pushing state's resources are held through the whole subflow and released for
   the parent level on pop. Without the release-on-exit wiring they leaked
   (``ExecutionEngine.exit_state`` existed but was never called).

Both use real constructs (a real ``AsyncDatabase`` memory backend via the
resource manager, a real FSM build, the real ``DatabaseUpsert`` transform — no
mocks).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode


def _inherit_fsm() -> AsyncSimpleFSM:
    """Two-network FSM where the *parent* declares the resource and the
    *sub-network* transform checks whether it inherited it.

    ``main.start`` (the regularly-entered pushing state) declares ``target_db``
    and pushes into ``sub``. ``sub.s1`` does **not** declare ``target_db``; its
    transform records whether the resource is present in its function context,
    which can only be true if the sub-network inherited it from the pushing
    state.
    """
    config = {
        "name": "inherit_fsm",
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
                        # NOTE: s1 does NOT declare target_db — it can only see
                        # the resource by inheriting it from the pushing state.
                        "arcs": [{"target": "s2"}],
                        "transforms": [
                            {
                                "type": "inline",
                                "code": (
                                    "def transform(data, context):\n"
                                    "    res = getattr(context, 'resources', None) or {}\n"
                                    "    data['inherited_target_db'] = 'target_db' in res\n"
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
    return AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY)


@pytest.mark.asyncio
async def test_async_push_inherits_initial_entered_parent_resources() -> None:
    """A push from an initial-state-entered parent inherits that state's resources.

    Here the pushing state ``start`` is ``is_start`` — entered via the
    initial-state path (``_enter_initial_state``), which now routes through the
    shared ``enter_state``. The sub-network transform sees ``target_db`` in its
    function context only via inheritance from the pushing state. Before that
    routing the parent's ``current_state_resources`` was never populated, so the
    sub inherited nothing and the resource was absent from the transform's
    context. (The sibling test below covers a pushing state entered via a
    regular A→B transition.)
    """
    fsm = _inherit_fsm()
    try:
        result = await fsm.process({"id": "1", "name": "Alice"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()

    assert result.get("data", {}).get("inherited_target_db") is True, (
        "The sub-network did not inherit the pushing parent's 'target_db' "
        "resource — it was absent from the sub-network transform's context"
    )


def _inherit_via_regular_transition_fsm() -> AsyncSimpleFSM:
    """Like ``_inherit_fsm`` but the pushing state is entered by a regular arc.

    ``main.start`` declares nothing and transitions (a regular A→B arc) to
    ``mid``, which declares ``target_db`` and pushes into ``sub``. So the pushing
    state ``mid`` is entered via the regular-transition path (``_execute_transition``
    → ``enter_state``), not the initial-state path — exercising that the regular
    path also populates ``current_state_resources`` for the push to inherit.
    """
    config = {
        "name": "inherit_regular_fsm",
        "main_network": "main",
        "resources": [
            {"name": "target_db", "type": "async_database", "config": {"type": "memory"}},
        ],
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True, "arcs": [{"target": "mid"}]},
                    {
                        "name": "mid",
                        "resources": ["target_db"],
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
                                    "    res = getattr(context, 'resources', None) or {}\n"
                                    "    data['inherited_target_db'] = 'target_db' in res\n"
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
    return AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY)


@pytest.mark.asyncio
async def test_async_push_inherits_regular_transition_entered_parent_resources() -> None:
    """A push from a regular-transition-entered parent inherits its resources.

    The pushing state ``mid`` is reached by a regular A→B arc from ``start``; its
    ``target_db`` must reach the sub-network transform via inheritance, proving
    the regular-transition entry (``_execute_transition`` → ``enter_state``)
    populates ``current_state_resources`` the same way the initial-state entry
    does.
    """
    fsm = _inherit_via_regular_transition_fsm()
    try:
        result = await fsm.process({"id": "1", "name": "Alice"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()

    assert result.get("data", {}).get("inherited_target_db") is True, (
        "The sub-network did not inherit the regular-transition-entered parent's "
        "'target_db' resource — it was absent from the sub-network transform's "
        "context"
    )


def _pushing_resource_fsm() -> AsyncSimpleFSM:
    """FSM whose pushing state declares a resource and whose final states do not.

    ``main.start`` declares ``held_db`` and pushes into ``sub``; ``after`` and
    ``end`` declare nothing, and the sub-network states declare nothing. After a
    full run the pushing state's resource must have been released on pop — the
    resource manager should show no live owner for it.
    """
    config = {
        "name": "release_fsm",
        "main_network": "main",
        "resources": [
            {"name": "held_db", "type": "async_database", "config": {"type": "memory"}},
        ],
        "networks": [
            {
                "name": "main",
                "states": [
                    {
                        "name": "start",
                        "is_start": True,
                        "resources": ["held_db"],
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
                    {"name": "s1", "is_start": True, "arcs": [{"target": "s2"}]},
                    {"name": "s2", "is_end": True},
                ],
            },
        ],
    }
    return AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY)


@pytest.mark.asyncio
async def test_async_subflow_releases_pushing_state_resources_on_pop() -> None:
    """The pushing state's resources are released for the parent level on pop.

    ``start`` acquires ``held_db`` and holds it through the subflow (so the
    sub-network could inherit it); on pop, when the parent resumes at the return
    state, that resource must be released. With the release-on-exit wiring
    removed, the pushing state's acquisition leaks — the resource manager keeps a
    live owner for ``held_db`` after the run. This is a regression guard for the
    acquire-and-release logic introduced together (on ``origin/main`` the pushing
    state never acquired the resource at all, so the guard cannot fail-before
    that baseline); the inheritance test above is the strict reproduce-first one.
    """
    fsm = _pushing_resource_fsm()
    try:
        result = await fsm.process({"id": "1"})
        assert result["success"], f"FSM did not complete cleanly: {result}"

        owners = fsm._resource_manager._resource_owners.get("held_db", set())
        assert not owners, (
            "The pushing state's 'held_db' resource was not released on pop — "
            f"the resource manager still lists live owner(s) {owners!r}"
        )
        # No live acquisitions should remain at all once the run has unwound.
        assert not fsm._resource_manager._resources, (
            "State resources leaked after the run: "
            f"{list(fsm._resource_manager._resources)}"
        )
    finally:
        await fsm.close()


def _terminal_resource_fsm() -> AsyncSimpleFSM:
    """FSM whose FINAL state declares a resource.

    ``start`` declares nothing and transitions to ``end``; ``end`` is a final
    state that declares ``end_db``. Entering ``end`` (a regular transition now
    routed through ``enter_state``) acquires ``end_db`` — but the run finalizes
    there and the state is never "left", so without a release at run completion
    its acquisition would be stranded until the resource manager is torn down.
    """
    config = {
        "name": "terminal_release_fsm",
        "main_network": "main",
        "resources": [
            {"name": "end_db", "type": "async_database", "config": {"type": "memory"}},
        ],
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True, "arcs": [{"target": "end"}]},
                    {"name": "end", "is_end": True, "resources": ["end_db"]},
                ],
            },
        ],
    }
    return AsyncSimpleFSM(config, data_mode=DataHandlingMode.COPY)


@pytest.mark.asyncio
async def test_async_terminal_state_resources_released_at_completion() -> None:
    """The final state's owned resources are released when the run completes.

    Each state's owned resources are released as it is *left* (regular
    transition or subflow pop), but the terminal state the run ends on is never
    left. Before the run-completion release was wired, ``end_db`` (acquired on
    entering the final ``end`` state) stayed held — the resource manager listed
    a live owner for it after the run. The release-on-completion closes that gap
    so no acquisition survives the run.
    """
    fsm = _terminal_resource_fsm()
    try:
        result = await fsm.process({"id": "1"})
        assert result["success"], f"FSM did not complete cleanly: {result}"

        owners = fsm._resource_manager._resource_owners.get("end_db", set())
        assert not owners, (
            "The terminal state's 'end_db' resource was not released at run "
            f"completion — the resource manager still lists owner(s) {owners!r}"
        )
        assert not fsm._resource_manager._resources, (
            "State resources leaked after the run: "
            f"{list(fsm._resource_manager._resources)}"
        )
    finally:
        await fsm.close()

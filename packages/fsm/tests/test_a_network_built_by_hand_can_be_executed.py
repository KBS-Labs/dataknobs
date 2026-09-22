# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A network built through ``StateNetwork``'s own API can be executed.

``StateNetwork`` publishes a complete-looking construction API --- ``add_state``,
``add_arc``, ``get_state``, ``initial_states``, ``validate``, ``find_cycles``,
``to_dict`` / ``from_dict`` --- and it is the only door into the FSM that is not
a YAML document. It is the door a consumer reaches for to build a network from
something that is not a config file: a database table, a user's drawing, a
generated graph.

It had never worked, and it failed twice over, either half sufficient on its
own:

1. ``add_state`` accepted a ``State`` --- a second, three-attribute state class
   --- that the engines cannot read. They ask a state for ``outgoing_arcs`` and
   ``type``; ``State`` has neither, so execution returned
   ``(False, "'State' object has no attribute 'outgoing_arcs'")``.
2. ``add_arc`` recorded the arc in ``_arcs`` and ``_arc_index`` and **not** on
   the source state's ``outgoing_arcs``, which is the only list the engines
   consult. So a network whose states were all ``StateDefinition`` still had no
   transitions: ``(False, "No valid transitions from state: start")``.

Neither is visible from inside the class. ``validate()`` passes, ``find_cycles``
answers, ``get_arcs_from_state`` returns the arcs --- every accessor on the
network agrees the network is fine, because they all read the index the engines
do not.

The config path escaped both because :class:`FSMBuilder` did the work itself:
it constructed ``StateDefinition`` directly and hand-appended an
``ArcDefinition`` to ``state_def.outgoing_arcs`` one line before calling
``add_arc``. That is why this survived --- the working implementation of "add an
arc to a network" lived in the builder, not in the class whose method it is, so
every test of the supported path exercised the copy that was complete.

These tests drive the public API only. Nothing here reaches past it to arrange
what the engine reads, because arranging that by hand is precisely the bug.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_fsm.core.arc import ArcDefinition
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateDefinition, StateType
from dataknobs_fsm.execution.context import ExecutionContext


def linear_fsm() -> FSM:
    """``start -> process -> end``, built entirely through the public API."""
    fsm = FSM(name="hand_built")
    network = StateNetwork(name="main")

    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="process"))
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    network.add_arc("start", "process")
    network.add_arc("process", "end")

    fsm.add_network(network)
    return fsm


def run(fsm: FSM, record: dict[str, Any]) -> tuple[bool, Any, str | None]:
    """Execute one record, returning ``(success, result, final_state)``."""
    context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
    try:
        success, result = fsm.get_sync_bridge().run(fsm.get_async_engine().execute(context, record))
        return success, result, context.current_state
    finally:
        fsm.close()


def test_a_hand_built_network_runs_to_its_end_state() -> None:
    """The whole point: the public API produces an executable network."""
    success, result, final_state = run(linear_fsm(), {"id": 1, "value": 25.0})

    assert success, f"a hand-built network did not execute: {result!r}"
    assert final_state == "end"


def test_add_arc_records_the_arc_where_the_engine_reads_it() -> None:
    """``add_arc`` is the single writer of every index the network keeps.

    The engines read ``state.outgoing_arcs`` and nothing else; the network's own
    accessors read ``_arcs`` / ``_arc_index``. A writer that maintains one and
    not the other is how the two came apart, so the assertion is that one call
    reaches all of them --- and with the *same* object, because two arcs that
    are equal today drift apart the first time one of them is edited.
    """
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    arc = network.add_arc("start", "end", pre_test="is_ready")

    assert network.get_arcs_from_state("start") == [arc]
    assert network.states["start"].outgoing_arcs == [arc]
    assert network.states["start"].outgoing_arcs[0] is arc
    assert arc.source_state == "start"
    assert arc.target_state == "end"
    assert arc.pre_test == "is_ready"


def test_a_prepared_arc_definition_is_the_one_that_is_stored() -> None:
    """A caller that has already built an ``ArcDefinition`` hands it over.

    The builder resolves functions, priorities and a definition order that the
    keyword form cannot express. Before this, it appended its own arc to the
    state and then called ``add_arc`` to make a *second*, lossy one for the
    network. Passing it through is what collapses those two into one.
    """
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    prepared = ArcDefinition(target_state="end", priority=7, definition_order=3)
    stored = network.add_arc("start", "end", definition=prepared)

    assert stored is prepared
    assert network.states["start"].outgoing_arcs[0] is prepared
    assert network.get_arcs_from_state("start")[0] is prepared
    assert prepared.source_state == "start", "add_arc stamps the source it was given"
    assert prepared.priority == 7, "the prepared arc's own fields survive"


def test_removing_an_arc_removes_it_from_every_index() -> None:
    """The inverse of the writer: one removal, every index."""
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)

    arc = network.add_arc("start", "end")
    network.remove_arc(arc)

    assert network.get_arcs_from_state("start") == []
    assert network.states["start"].outgoing_arcs == []


def test_an_arc_to_an_unknown_state_is_refused_before_anything_is_written() -> None:
    """A refused arc leaves no partial write behind in any index."""
    network = StateNetwork(name="main")
    network.add_state(StateDefinition(name="start", type=StateType.START), initial=True)

    with pytest.raises(ValueError, match="nowhere"):
        network.add_arc("start", "nowhere")

    assert network.get_arcs_from_state("start") == []
    assert network.states["start"].outgoing_arcs == []


def test_a_plain_callable_validator_on_a_hand_built_state_is_run() -> None:
    """A validator that is an ordinary function is called, not skipped.

    ``StateDefinition.validation_functions`` is a list of
    ``RegisteredFunction`` --- deliberately, because what the builder puts
    there is whatever ``_resolve_function`` returned, which includes a plain
    callable. The async engine reached each entry as ``validator.validate``,
    which only the interface form and the builder's ``InterfaceWrapper``
    answer. A plain function raises ``AttributeError`` there, and the loop
    around it swallows every exception, so the validator did not run and the
    record was reported as validated.

    Invisible until now for a reason worth recording: the only way to put a
    plain callable in that list is to build the state yourself, and a
    hand-built network could not be executed at all. Making the door work is
    what put a record in front of this.
    """
    calls: list[dict[str, Any]] = []

    def record_seen(data: dict[str, Any], context: Any = None) -> bool:
        calls.append(dict(data))
        return True

    fsm = FSM(name="plain_validator")
    network = StateNetwork(name="main")
    start = StateDefinition(name="start", type=StateType.START)
    start.validation_functions = [record_seen]
    network.add_state(start, initial=True)
    network.add_state(StateDefinition(name="end", type=StateType.END), final=True)
    network.add_arc("start", "end")
    fsm.add_network(network)

    success, result, _ = run(fsm, {"id": 7})

    assert success, result
    assert calls == [{"id": 7}], "the validator was never called"

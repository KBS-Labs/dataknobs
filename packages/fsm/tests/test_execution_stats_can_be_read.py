# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``ExecutionContext.get_execution_stats`` answers instead of raising.

It read ``self.transition_count`` and ``self.execution_id``, and
``ExecutionContext`` has never had either attribute --- so every call raised
``AttributeError: 'ExecutionContext' object has no attribute
'transition_count'``, on the first line of the dict it was building. The
method has no caller in this tree, and could not have had one anywhere:
there is no input for which it returned.

Both values are already tracked, under other names:

- ``state_history`` gains an entry each time :meth:`set_state` leaves a
  state, so its length *is* the number of transitions taken. The sibling
  :meth:`get_performance_stats` already reports ``len(state_history)``, under
  the name ``states_visited``.
- ``execution_id`` is carried in ``metadata`` --- it is what
  :class:`ExecutionHistory` keys a run by, and what ``ResultFormatter`` copies
  through to a caller. ``None`` when the run was not given one.

The type checker had named both, by code and by line, for as long as the
cell has been measured.
"""

from __future__ import annotations

from typing import Any, Dict

from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.context import ExecutionContext


def _traversed() -> ExecutionContext:
    """A context that has walked ``start -> work -> done``."""
    context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
    for state in ("start", "work", "done"):
        context.set_state(state)
    return context


def test_a_context_can_report_its_execution_stats() -> None:
    """Every call raised before the dict was finished being built."""
    stats: Dict[str, Any] = _traversed().get_execution_stats()

    assert stats["states_visited"] == 2
    assert stats["current_state"] == "done"
    assert stats["previous_state"] == "work"
    assert stats["data_mode"] == ProcessingMode.SINGLE.value


def test_the_transition_count_is_the_number_of_states_left() -> None:
    """Two transitions to walk three states, and none before the first."""
    fresh = ExecutionContext()
    assert fresh.get_execution_stats()["transition_count"] == 0

    fresh.set_state("start")
    assert fresh.get_execution_stats()["transition_count"] == 0, "entering is not transitioning"

    fresh.set_state("work")
    assert fresh.get_execution_stats()["transition_count"] == 1

    fresh.set_state("done")
    assert fresh.get_execution_stats()["transition_count"] == 2


def test_the_execution_id_comes_from_the_metadata_that_carries_it() -> None:
    """The name a run is keyed by everywhere else it is written down."""
    context = _traversed()
    assert context.get_execution_stats()["execution_id"] is None

    context.metadata["execution_id"] = "exec-1"
    assert context.get_execution_stats()["execution_id"] == "exec-1"


def test_the_sibling_that_always_worked_still_agrees() -> None:
    """A control: the two stats methods overlap, and must not disagree."""
    context = _traversed()

    stats = context.get_execution_stats()
    performance = context.get_performance_stats()

    assert stats["states_visited"] == performance["states_visited"]
    assert stats["current_state"] == performance["current_state"]

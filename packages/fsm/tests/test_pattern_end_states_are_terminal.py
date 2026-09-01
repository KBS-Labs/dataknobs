"""A pattern's terminal state has to be terminal to the engine, not just named one.

``ErrorRecoveryWorkflow`` and ``APIOrchestrator`` marked their last state with
``{"name": "end", "type": "terminal"}``. ``StateConfig`` has never declared a
``type``, and the schema discarded keys it did not declare, so the word carried
no weight: the builder derives a state's kind from ``is_start`` / ``is_end``
(``config/builder.py``), found neither, and made the end state ``NORMAL``.

Their two sibling patterns --- ``DatabaseETL`` and ``FileProcessor`` --- spell
it ``is_end: True`` and carry no ``type`` at all. Four patterns, two
conventions, and only one of them was a convention the schema knew.

The check is over every pattern rather than over the two that were wrong: the
next pattern to be written is the one at risk, and naming the two known cases
would pass while it went the same way.
"""

from __future__ import annotations

import pytest

from dataknobs_fsm.patterns.api_orchestration import (
    APIEndpoint,
    APIOrchestrationConfig,
    APIOrchestrator,
)
from dataknobs_fsm.patterns.error_recovery import (
    ErrorRecoveryConfig,
    ErrorRecoveryWorkflow,
    RecoveryStrategy,
)


def _error_recovery() -> ErrorRecoveryWorkflow:
    return ErrorRecoveryWorkflow(ErrorRecoveryConfig(primary_strategy=RecoveryStrategy.RETRY))


def _api_orchestrator() -> APIOrchestrator:
    return APIOrchestrator(
        APIOrchestrationConfig(endpoints=[APIEndpoint(name="one", url="https://example.invalid")])
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_error_recovery, id="error_recovery"),
        pytest.param(_api_orchestrator, id="api_orchestration"),
    ],
)
def test_a_pattern_workflow_has_a_state_the_engine_treats_as_final(build) -> None:
    """Not "is there a state called end" --- whether the engine agrees it ends."""
    fsm = build()._build_fsm()

    finals = [name for name in fsm.get_states() if fsm.get_state(name).is_end]

    assert finals, (
        "no state in this pattern is marked final, so the engine has nothing to "
        "stop at --- a terminal state named in a key the schema discards is not "
        "a terminal state"
    )

"""A pattern's terminal state has to be terminal to the engine, not just named one.

``ErrorRecoveryWorkflow`` and ``APIOrchestrator`` marked their last state with
``{"name": "end", "type": "terminal"}``. ``StateConfig`` has never declared a
``type``, and the schema discarded keys it did not declare, so the word carried
no weight: the builder derives a state's kind from ``is_start`` / ``is_end``
(``config/builder.py``), found neither, and made the end state ``NORMAL``.

Their two sibling patterns --- ``DatabaseETL`` and ``FileProcessor`` --- spell
it ``is_end: True`` and carry no ``type`` at all. Four patterns, two
conventions, and only one of them was a convention the schema knew.

Every pattern is checked, and every branch each builder takes on its config.
An earlier draft of this file parametrised over two builders at their default
config and passed while ``circuit_check`` --- reachable only from
``RecoveryStrategy.CIRCUIT_BREAKER``, in the very file the fix was editing ---
still carried a ``"type": "decision"`` key. A builder that branches hides a
whole shape behind a config value, so the branch is the unit to cover, not the
class.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from dataknobs_fsm.patterns.api_orchestration import (
    APIEndpoint,
    APIOrchestrationConfig,
    APIOrchestrator,
    OrchestrationMode,
)
from dataknobs_fsm.patterns.error_recovery import (
    ErrorRecoveryConfig,
    ErrorRecoveryWorkflow,
    RecoveryStrategy,
)
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode
from dataknobs_fsm.patterns.file_processing import (
    FileProcessingConfig,
    FileProcessor,
    ProcessingMode,
)

_ENDPOINTS = [
    APIEndpoint(name="one", url="https://example.invalid"),
    APIEndpoint(name="two", url="https://example.invalid"),
]
_DB = {"backend": "memory"}


def _cases() -> list[Any]:
    """Every pattern builder, once per branch it takes on its own config."""
    cases: list[Any] = []
    for strategy in RecoveryStrategy:
        cases.append(
            pytest.param(
                lambda s=strategy: ErrorRecoveryWorkflow(ErrorRecoveryConfig(primary_strategy=s)),
                id=f"error_recovery-{strategy.value}",
            )
        )
    for mode in OrchestrationMode:
        cases.append(
            pytest.param(
                lambda m=mode: APIOrchestrator(
                    APIOrchestrationConfig(endpoints=_ENDPOINTS, mode=m)
                ),
                id=f"api_orchestration-{mode.value}",
            )
        )
    for etl_mode in ETLMode:
        cases.append(
            pytest.param(
                lambda m=etl_mode: DatabaseETL(
                    ETLConfig(source_db=dict(_DB), target_db=dict(_DB), mode=m)
                ),
                id=f"etl-{etl_mode.value}",
            )
        )
    for proc_mode in ProcessingMode:
        cases.append(
            pytest.param(
                lambda m=proc_mode: FileProcessor(
                    FileProcessingConfig(input_path="in.csv", mode=m)
                ),
                id=f"file_processing-{proc_mode.value}",
            )
        )
    return cases


@pytest.mark.parametrize("build", _cases())
def test_a_pattern_workflow_has_a_state_the_engine_treats_as_final(
    build: Callable[[], Any],
) -> None:
    """Not "is there a state called end" --- whether the engine agrees it ends."""
    fsm = build()._build_fsm()

    finals = [name for name in fsm.get_states() if fsm.get_state(name).is_end]

    assert finals, (
        "no state in this pattern is marked final, so the engine has nothing to "
        "stop at --- a terminal state named in a key the schema discards is not "
        "a terminal state"
    )


@pytest.mark.parametrize("build", _cases())
def test_a_pattern_workflow_builds_at_all(build: Callable[[], Any]) -> None:
    """The weaker claim, stated separately because it fails differently.

    Once unknown keys are refused, a pattern carrying one does not build a
    wrong FSM --- it raises during validation. Both assertions live in the same
    call, but the messages a reader gets are worth keeping apart: this one says
    the configuration is invalid, the other says it is valid and wrong.
    """
    fsm = build()._build_fsm()

    assert fsm.get_states(), "the pattern built an FSM with no states"

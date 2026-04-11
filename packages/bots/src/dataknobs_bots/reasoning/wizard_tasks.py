"""Wizard task tracking.

Functions for building and updating the wizard task list based on
extracted fields, tool invocations, and stage transitions.  Extracted
from ``wizard.py`` (item 77a).
"""

from __future__ import annotations

import logging
from typing import Any

from .observability import WizardTask, WizardTaskList
from .wizard_types import WizardState

logger = logging.getLogger(__name__)


def build_initial_tasks(
    stage_metadata: dict[str, dict[str, Any]],
) -> WizardTaskList:
    """Build initial task list from wizard configuration.

    Extracts task definitions from stage metadata and creates
    a :class:`WizardTaskList` with all tasks in pending status.

    Args:
        stage_metadata: Mapping of stage name → stage metadata dict.

    Returns:
        WizardTaskList with initial tasks.
    """
    tasks: list[WizardTask] = []
    global_tasks_added = False

    # Extract tasks from each stage's metadata
    for stage_name, stage_meta in stage_metadata.items():
        # Per-stage tasks
        stage_tasks = stage_meta.get("tasks", [])
        for task_def in stage_tasks:
            if task_def.get("id"):  # Only add if id is defined
                tasks.append(WizardTask(
                    id=task_def.get("id"),
                    description=task_def.get("description", task_def.get("id", "")),
                    status="pending",
                    stage=stage_name,
                    required=task_def.get("required", True),
                    depends_on=task_def.get("depends_on", []),
                    completed_by=task_def.get("completed_by"),
                    field_name=task_def.get("field_name"),
                    tool_name=task_def.get("tool_name"),
                ))

        # Global tasks (only need to add once)
        if not global_tasks_added:
            global_tasks = stage_meta.get("_global_tasks", [])
            for task_def in global_tasks:
                if task_def.get("id"):  # Only add if id is defined
                    tasks.append(WizardTask(
                        id=task_def.get("id"),
                        description=task_def.get(
                            "description", task_def.get("id", "")
                        ),
                        status="pending",
                        stage=None,  # Global task
                        required=task_def.get("required", True),
                        depends_on=task_def.get("depends_on", []),
                        completed_by=task_def.get("completed_by"),
                        field_name=task_def.get("field_name"),
                        tool_name=task_def.get("tool_name"),
                    ))
            if global_tasks:
                global_tasks_added = True

    return WizardTaskList(tasks=tasks)


def update_field_tasks(
    state: WizardState, extracted_data: dict[str, Any],
) -> None:
    """Mark field-extraction tasks as complete when fields are collected.

    Args:
        state: Current wizard state.
        extracted_data: Data that was just extracted.
    """
    for field_name, value in extracted_data.items():
        if value is not None and not field_name.startswith("_"):
            for task in state.tasks.tasks:
                if (
                    task.completed_by == "field_extraction"
                    and task.field_name == field_name
                    and task.status == "pending"
                ):
                    state.tasks.complete_task(task.id)
                    logger.debug("Task %s completed via field extraction", task.id)


def update_tool_tasks(
    state: WizardState, tool_name: str, success: bool,
) -> None:
    """Mark tool-result tasks as complete when tools succeed.

    Called from :meth:`WizardReasoning.finalize_turn` when DynaBot
    passes tool execution results through the phased protocol.

    Args:
        state: Current wizard state.
        tool_name: Name of the tool that was executed.
        success: Whether the tool execution succeeded.
    """
    if success:
        for task in state.tasks.tasks:
            if (
                task.completed_by == "tool_result"
                and task.tool_name == tool_name
                and task.status == "pending"
            ):
                state.tasks.complete_task(task.id)
                logger.debug("Task %s completed via tool result", task.id)


def update_stage_exit_tasks(state: WizardState, stage: str) -> None:
    """Mark stage-exit tasks as complete when leaving a stage.

    Args:
        state: Current wizard state.
        stage: The stage being exited.
    """
    for task in state.tasks.tasks:
        if (
            task.completed_by == "stage_exit"
            and task.stage == stage
            and task.status == "pending"
        ):
            state.tasks.complete_task(task.id)
            logger.debug("Task %s completed via stage exit", task.id)

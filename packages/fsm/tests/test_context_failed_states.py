"""Reproduce-first: ExecutionContext.failed_states lifecycle.

``failed_states`` records the states whose transform raised for a record. It is
load-bearing for data integrity — it gates whether downstream transforms run
(``should_skip_state_transforms``) and whether the record is reported as a
failure (``finalize_single_result``). Its propagation rules must therefore be
exact:

* a fresh context starts with an empty set (initialized in ``__init__``);
* ``clone()`` does NOT carry it (per-record / batch-item isolation — one
  record's failure must not taint the next record's persistence decision);
* ``create_child_context()`` does NOT carry it (a parallel sub-path starts
  clean);
* ``merge_child_context()`` unions the child's failures back into the parent, so
  a failure on a parallel / subnetwork sub-path is not lost from the parent's
  result.

The merge assertion is reproduce-first: before the union was added,
``merge_child_context`` dropped the child's ``failed_states`` and the parent
finalized as a success despite a sub-path failure.
"""

from __future__ import annotations

from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.context import ExecutionContext


def test_fresh_context_has_empty_failed_states() -> None:
    context = ExecutionContext()
    assert context.failed_states == set()


def test_clone_does_not_carry_failed_states() -> None:
    context = ExecutionContext()
    context.failed_states.add("transform_state")

    clone = context.clone()

    assert clone.failed_states == set(), (
        "clone must start clean so one record's failure does not taint the next"
    )
    # And the original is unaffected by mutating the clone.
    clone.failed_states.add("other")
    assert context.failed_states == {"transform_state"}


def test_child_context_starts_clean() -> None:
    parent = ExecutionContext()
    parent.failed_states.add("upstream")

    child = parent.create_child_context("path-1")

    assert child.failed_states == set(), "a parallel sub-path starts clean"


def test_merge_child_context_propagates_failed_states() -> None:
    parent = ExecutionContext(data_mode=ProcessingMode.SINGLE)
    child = parent.create_child_context("path-1")

    # The child sub-path fails a transform.
    child.failed_states.add("child_transform")

    assert parent.merge_child_context("path-1") is True
    assert "child_transform" in parent.failed_states, (
        "a sub-path transform failure must surface in the parent's failed_states"
    )


def test_merge_child_context_unions_with_existing_parent_failures() -> None:
    parent = ExecutionContext()
    parent.failed_states.add("parent_transform")
    child = parent.create_child_context("path-1")
    child.failed_states.add("child_transform")

    parent.merge_child_context("path-1")

    assert parent.failed_states == {"parent_transform", "child_transform"}

"""Behavior tests for :class:`CallbackOrdering` reference implementations."""

from __future__ import annotations

import pytest

from dataknobs_common.callbacks import (
    CallbackEntry,
    CompositeOrdering,
    FIFOOrdering,
    PriorityOrdering,
    StageOrdering,
)


def _entry(
    *,
    priority: int = 0,
    stage: str = "main",
    seq: int = 0,
) -> CallbackEntry:
    return CallbackEntry(
        topic="t",
        callback=lambda _: None,
        priority=priority,
        stage=stage,
        registration_seq=seq,
    )


def test_fifo_ordering_compares_by_seq() -> None:
    ord_ = FIFOOrdering()
    assert ord_.compare(_entry(seq=1), _entry(seq=2)) == -1
    assert ord_.compare(_entry(seq=2), _entry(seq=1)) == 1
    assert ord_.compare(_entry(seq=1), _entry(seq=1)) == 0


def test_priority_ordering_lower_wins_and_ties_to_zero() -> None:
    ord_ = PriorityOrdering()
    assert (
        ord_.compare(
            _entry(priority=-1, seq=2),
            _entry(priority=0, seq=1),
        )
        == -1
    )
    # Equal priority compares 0 — the registry's stable sort preserves
    # registration order, and CompositeOrdering can pass the tie to the
    # next inner ordering.
    assert (
        ord_.compare(
            _entry(priority=0, seq=1),
            _entry(priority=0, seq=2),
        )
        == 0
    )
    assert (
        ord_.compare(
            _entry(priority=0, seq=2),
            _entry(priority=0, seq=1),
        )
        == 0
    )


def test_stage_ordering_requires_nonempty_stages() -> None:
    with pytest.raises(ValueError, match="at least one"):
        StageOrdering(stages=())


def test_stage_ordering_unknown_stage_sorts_to_end() -> None:
    ord_ = StageOrdering(stages=("pre", "main"))
    assert (
        ord_.compare(
            _entry(stage="pre", seq=2),
            _entry(stage="main", seq=1),
        )
        == -1
    )
    assert (
        ord_.compare(
            _entry(stage="main", seq=1),
            _entry(stage="unknown", seq=0),
        )
        == -1
    )


def test_stage_ordering_within_stage_ties_to_zero() -> None:
    # Same-stage entries compare 0 — the registry's stable sort preserves
    # registration order without baking FIFO into the comparator (which
    # would prevent CompositeOrdering from passing the tie to a later
    # ordering tier).
    ord_ = StageOrdering(stages=("a",))
    assert (
        ord_.compare(
            _entry(stage="a", seq=1),
            _entry(stage="a", seq=2),
        )
        == 0
    )


def test_composite_ordering_requires_inner() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeOrdering()


def test_composite_ordering_first_nonzero_wins() -> None:
    ord_ = CompositeOrdering(
        StageOrdering(stages=("pre", "main", "post")),
        PriorityOrdering(),
        FIFOOrdering(),
    )
    # Stage decides when stages differ
    assert (
        ord_.compare(
            _entry(stage="pre", priority=10, seq=2),
            _entry(stage="main", priority=-10, seq=1),
        )
        == -1
    )
    # Within same stage, priority decides
    assert (
        ord_.compare(
            _entry(stage="main", priority=-1, seq=2),
            _entry(stage="main", priority=0, seq=1),
        )
        == -1
    )
    # Same stage + priority → FIFO tiebreak
    assert (
        ord_.compare(
            _entry(stage="main", priority=0, seq=1),
            _entry(stage="main", priority=0, seq=2),
        )
        == -1
    )

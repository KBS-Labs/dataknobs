"""Protocol-conformance tests for :class:`CallbackOrdering`."""

from __future__ import annotations

from dataknobs_common.callbacks import (
    CallbackEntry,
    CallbackOrdering,
    CompositeOrdering,
    FIFOOrdering,
    PriorityOrdering,
    StageOrdering,
)


def test_built_in_orderings_conform_to_protocol() -> None:
    assert isinstance(FIFOOrdering(), CallbackOrdering)
    assert isinstance(PriorityOrdering(), CallbackOrdering)
    assert isinstance(StageOrdering(stages=("a", "b")), CallbackOrdering)
    assert isinstance(
        CompositeOrdering(FIFOOrdering()),
        CallbackOrdering,
    )


def test_ordering_protocol_is_runtime_checkable() -> None:
    class CustomOrdering:
        def compare(
            self,
            a: CallbackEntry,
            b: CallbackEntry,
        ) -> int:
            return 0

    assert isinstance(CustomOrdering(), CallbackOrdering)


def test_ordering_protocol_rejects_non_conformer() -> None:
    class NotAnOrdering:
        def order(self, a: CallbackEntry, b: CallbackEntry) -> int:
            return 0

    assert not isinstance(NotAnOrdering(), CallbackOrdering)

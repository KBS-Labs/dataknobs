"""``VectorStoreBase._overfetch_sizes`` — the shared over-fetch policy.

Three backends compensate for a post-filter by asking their index for
more rows than the caller wanted, and this generator is the single place
that decides how many. It was extracted from three independently written
copies, so it is the one piece of that consolidation with no prior
behaviour to inherit — and the escalation loop in particular was reached
only by its first yield, leaving the doubling, the ceiling cap and the
anti-stall step untested.

Driven through a real store rather than a constructed base instance: the
method is inherited unchanged by every backend and needs no state beyond
what an ordinary store already has.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.vector.stores.common import POST_FILTER_OVERFETCH
from dataknobs_data.vector.stores.memory import MemoryVectorStore

# Enough to prove a sequence terminates without hanging the suite if it
# ever stops doing so.
_RUNAWAY = 200


@pytest.fixture
def store() -> MemoryVectorStore:
    return MemoryVectorStore({"dimensions": 4})


def _sizes(store: Any, k: int, *, has_post_filter: bool, ceiling: int | None = None) -> list[int]:
    """Drain the generator, refusing to spin forever on a broken one."""
    out: list[int] = []
    for size in store._overfetch_sizes(k, has_post_filter=has_post_filter, ceiling=ceiling):
        out.append(size)
        assert len(out) <= _RUNAWAY, f"sequence did not terminate: {out[:20]}..."
    return out


def test_without_a_post_filter_the_index_truncation_is_already_exact(store: Any) -> None:
    """No post-filter, no over-fetch: asking for more than ``k`` is waste."""
    assert _sizes(store, 10, has_post_filter=False) == [10]


def test_without_a_post_filter_a_ceiling_still_caps_the_ask(store: Any) -> None:
    """A caller that knows the corpus is smaller than ``k`` asks for the corpus."""
    assert _sizes(store, 10, has_post_filter=False, ceiling=3) == [3]
    assert _sizes(store, 10, has_post_filter=False, ceiling=50) == [10]


def test_an_unbounded_post_filter_takes_one_over_fetch_and_stops(store: Any) -> None:
    """With no ceiling there is nothing to escalate towards.

    This is the heuristic on its own, and it is why the constant is
    documented as a heuristic: a filter matching fewer than one candidate
    in ``POST_FILTER_OVERFETCH`` still under-returns here.
    """
    assert _sizes(store, 10, has_post_filter=True) == [10 * POST_FILTER_OVERFETCH]


def test_a_ceiling_escalates_by_doubling_and_ends_exactly_on_it(store: Any) -> None:
    """The sequence doubles, is capped at the ceiling, and ends there.

    Ending *on* the ceiling is the part that matters: at that size the
    index has returned every row, so the post-filter's answer is exact
    rather than merely over-fetched.
    """
    sizes = _sizes(store, 3, has_post_filter=True, ceiling=100)

    assert sizes[0] == 3 * POST_FILTER_OVERFETCH
    assert sizes == [12, 24, 48, 96, 100]
    assert sizes[-1] == 100
    assert sizes == sorted(sizes), "sizes must widen monotonically"


def test_a_ceiling_below_the_first_over_fetch_is_the_only_size(store: Any) -> None:
    """Never ask for more rows than exist."""
    assert _sizes(store, 10, has_post_filter=True, ceiling=7) == [7]


def test_a_ceiling_reached_exactly_does_not_yield_twice(store: Any) -> None:
    """The first size landing on the ceiling terminates the sequence."""
    assert _sizes(store, 3, has_post_filter=True, ceiling=3 * POST_FILTER_OVERFETCH) == [12]


def test_an_empty_corpus_yields_one_zero_and_stops(store: Any) -> None:
    """A ceiling of zero terminates rather than escalating from nothing."""
    assert _sizes(store, 10, has_post_filter=True, ceiling=0) == [0]


@pytest.mark.parametrize("k", [0, -1, -5])
def test_a_non_positive_k_still_terminates(store: Any, k: int) -> None:
    """The anti-stall step is what keeps this from spinning at zero.

    Doubling alone leaves a non-positive size where it started forever,
    so the escalation takes ``max(capped * 2, capped + 1)``. Callers
    normalize ``k`` before reaching here; this pins that the generator
    does not depend on their doing so.
    """
    sizes = _sizes(store, k, has_post_filter=True, ceiling=5)

    assert sizes[-1] == 5
    assert sizes == sorted(sizes)
    assert len(sizes) == len(set(sizes)), f"a size repeated, so it was stalling: {sizes}"

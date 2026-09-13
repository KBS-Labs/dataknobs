"""Tests for :func:`assert_twins_agree` and :func:`assert_twin_types_agree`.

Every assertion the guard makes is exercised in both directions: a pair that
should pass, and a pair that must make it red. A parity guard that cannot fail
is worth less than no guard, because it also reports green -- so the red half
is the half that carries the value here, and each ``pytest.raises`` below names
the drift it is standing in for.

The subjects are ordinary functions defined in this module rather than stand-in
objects. The guard reads signatures, so a real ``def`` is both the simplest
fixture available and the thing production code will actually hand it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence

import pytest

from dataknobs_common.testing import assert_twin_types_agree, assert_twins_agree

# --------------------------------------------------------------------------
# The agreeing pair, and the shapes that must not
# --------------------------------------------------------------------------


def fetch(node: str, *, limit: int = 10) -> Sequence[str]:
    """The synchronous half of a well-behaved pair."""
    return [node] * limit


async def async_fetch(node: str, *, limit: int = 10) -> Sequence[str]:
    """Its twin: same parameters, same defaults, same annotations."""
    return [node] * limit


async def async_fetch_bounded(
    node: str, *, limit: int = 10, max_concurrency: int = 8
) -> Sequence[str]:
    """A twin carrying one genuine asynchronous-only knob."""
    return [node] * (limit + max_concurrency)


def test_an_agreeing_pair_passes() -> None:
    """The baseline: nothing declared, nothing wrong."""
    assert_twins_agree(fetch, async_fetch, compare_return=True)


def test_a_declared_async_only_parameter_passes() -> None:
    """A real asymmetry, named, is what the parameter is for."""
    assert_twins_agree(fetch, async_fetch_bounded, async_only={"max_concurrency"})


def test_an_undeclared_async_only_parameter_is_refused() -> None:
    """The drift the guard exists for: a knob added to one half only."""
    with pytest.raises(AssertionError, match="max_concurrency"):
        assert_twins_agree(fetch, async_fetch_bounded)


def test_a_stale_async_only_entry_is_refused() -> None:
    """A declaration that stopped being true must fail, not go quiet.

    The sibling guards in this package need an explicit staleness check because
    their suppression lists are subsets. This one gets it from comparing by
    equality, and that is only worth claiming if it is checked.
    """
    with pytest.raises(AssertionError, match="max_concurrency"):
        assert_twins_agree(fetch, async_fetch, async_only={"max_concurrency"})


# --------------------------------------------------------------------------
# Each remaining assertion, in both directions
# --------------------------------------------------------------------------


async def async_fetch_missing_a_parameter(node: str) -> Sequence[str]:
    """A twin that dropped a parameter the synchronous half still takes."""
    return [node]


def test_a_parameter_absent_from_the_async_half_is_refused() -> None:
    """Flavour-agnostic code would pass a keyword the asynchronous half lacks."""
    with pytest.raises(AssertionError, match="limit"):
        assert_twins_agree(fetch, async_fetch_missing_a_parameter)


async def async_fetch_other_default(node: str, *, limit: int = 25) -> Sequence[str]:
    """Same surface, different default -- the same call gives different answers."""
    return [node] * limit


def test_a_differing_default_is_refused() -> None:
    with pytest.raises(AssertionError, match="defaults to"):
        assert_twins_agree(fetch, async_fetch_other_default)


async def async_fetch_positional(node: str, limit: int = 10) -> Sequence[str]:
    """Same names and defaults; ``limit`` is no longer keyword-only."""
    return [node] * limit


def test_a_differing_parameter_kind_is_refused() -> None:
    """Names and defaults can agree while the call that works does not."""
    with pytest.raises(AssertionError, match="keyword-only"):
        assert_twins_agree(fetch, async_fetch_positional)


async def async_fetch_other_annotation(node: bytes, *, limit: int = 10) -> Sequence[str]:
    """A parameter that changed type on one side only."""
    return [node.decode()] * limit


def test_an_undeclared_annotation_difference_is_refused() -> None:
    with pytest.raises(AssertionError, match="annotated differently"):
        assert_twins_agree(fetch, async_fetch_other_annotation)


def test_a_declared_flavour_typed_parameter_passes() -> None:
    """The legitimate case: a parameter that is itself flavoured."""
    assert_twins_agree(fetch, async_fetch_other_annotation, flavour_typed={"node"})


def test_a_stale_flavour_typed_entry_is_refused() -> None:
    """Declared flavoured, actually identical -- the exception outlived its reason."""
    with pytest.raises(AssertionError, match="annotated differently"):
        assert_twins_agree(fetch, async_fetch, flavour_typed={"node"})


# --------------------------------------------------------------------------
# Flavour, including the async generator that the obvious check misses
# --------------------------------------------------------------------------


def walk(node: str) -> Iterator[str]:
    """A synchronous generator function."""
    yield node


async def async_walk(node: str) -> AsyncIterator[str]:
    """An async *generator* function: not a coroutine function.

    The case that makes ``inspect.iscoroutinefunction`` alone the wrong check.
    A streaming twin is the most asymmetric pair a codebase has, so a guard
    that reads it as synchronous fails exactly where it is needed.
    """
    yield node


def test_an_async_generator_counts_as_the_asynchronous_half() -> None:
    assert_twins_agree(walk, async_walk)


def test_a_streaming_pair_may_differ_by_return_annotation() -> None:
    """``Iterator`` against ``AsyncIterator`` is the flavour, not the contract."""
    assert_twins_agree(walk, async_walk, compare_return=False)


def test_a_streaming_pair_is_refused_when_the_return_is_compared() -> None:
    with pytest.raises(AssertionError, match="AsyncIterator"):
        assert_twins_agree(walk, async_walk, compare_return=True)


def test_the_halves_may_not_be_swapped() -> None:
    """Passing the pair the wrong way round is drift the guard must catch."""
    with pytest.raises(AssertionError, match="synchronous twin is an async callable"):
        assert_twins_agree(async_fetch, fetch)


def test_a_synchronous_second_half_is_refused() -> None:
    with pytest.raises(AssertionError, match="neither a coroutine function"):
        assert_twins_agree(fetch, fetch)


# --------------------------------------------------------------------------
# Members that are synchronous on both halves
# --------------------------------------------------------------------------


def rank(hits: frozenset[str], *, descending: bool = False) -> Sequence[str]:
    """A member a twinned type keeps synchronous on both halves.

    Ordering a set that has already arrived reaches for nothing, so the
    asynchronous twin gains an ``await`` and no I/O by being made a coroutine.
    Real instances of this shape: a ``name`` property, a ``narrows()``
    predicate, an ordering hook over hits the source already returned.
    """
    return sorted(hits, reverse=descending)


def rank_twin(hits: frozenset[str], *, descending: bool = False) -> Sequence[str]:
    """Its opposite number: same surface, synchronous for the same reason."""
    return sorted(hits, reverse=descending)


def rank_twin_drifted(
    hits: frozenset[str], *, descending: bool = False, limit: int = 0
) -> Sequence[str]:
    """The drift, arriving on a pair that is synchronous on both halves."""
    return sorted(hits, reverse=descending)[:limit]


def test_a_pair_synchronous_on_both_halves_passes_when_declared() -> None:
    assert_twins_agree(rank, rank_twin, unflavoured=True, compare_return=True)


def test_a_pair_synchronous_on_both_halves_is_refused_undeclared() -> None:
    """The gap the parameter closes, pinned as a gap.

    Before it, such a pair failed the flavour assertion outright, so the only
    way to keep the check green was to leave the member out of the list --
    unguarded, which is precisely where drift lives unnoticed.
    """
    with pytest.raises(AssertionError, match="neither a coroutine function"):
        assert_twins_agree(rank, rank_twin)


def test_drift_is_still_caught_in_a_pair_declared_unflavoured() -> None:
    """``unflavoured`` relaxes the flavour assertion and nothing else.

    The half of this that carries the value: a parameter that switched the
    check off for a member would be worse than not listing the member, since
    it would also report green.
    """
    with pytest.raises(AssertionError, match="limit"):
        assert_twins_agree(rank, rank_twin_drifted, unflavoured=True)


def test_a_flavoured_pair_declared_unflavoured_is_refused() -> None:
    """A declaration that stops being true fails rather than going quiet."""
    with pytest.raises(AssertionError, match="declared unflavoured"):
        assert_twins_agree(fetch, async_fetch, unflavoured=True)


# --------------------------------------------------------------------------
# The types accessor
# --------------------------------------------------------------------------


class Reader:
    """A synchronous surface with two members."""

    def read(self, key: str) -> str:
        return key

    def keys(self) -> Sequence[str]:
        return []


class AsyncReader:
    """Its twin, with ``keys`` drifted: the default moved."""

    async def read(self, key: str) -> str:
        return key

    async def keys(self, *, prefix: str = "") -> Sequence[str]:
        return [prefix]


def test_the_types_accessor_checks_every_named_member() -> None:
    assert_twin_types_agree(Reader, AsyncReader, ("read",), compare_return=True)


def test_the_types_accessor_names_the_member_that_drifted() -> None:
    """A loop over members is only useful if the failure says which one."""
    with pytest.raises(AssertionError, match=r"Reader/AsyncReader\.keys"):
        assert_twin_types_agree(Reader, AsyncReader, ("read", "keys"))


class Vocabulary:
    """A surface mixing the two kinds of member in one type."""

    def lookup(self, key: str) -> str:
        return key

    def rank(self, hits: frozenset[str]) -> Sequence[str]:
        return sorted(hits)


class AsyncVocabulary:
    """Its twin: ``lookup`` crosses the loop, ``rank`` deliberately does not."""

    async def lookup(self, key: str) -> str:
        return key

    def rank(self, hits: frozenset[str]) -> Sequence[str]:
        return sorted(hits)


def test_the_types_accessor_mixes_flavoured_and_unflavoured_members() -> None:
    """One call over a real surface, which is the shape callers have.

    Splitting into a call per flavour would work and would also let a member
    drop out of both lists without anything noticing.
    """
    assert_twin_types_agree(
        Vocabulary,
        AsyncVocabulary,
        ("lookup", "rank"),
        unflavoured_members={"rank"},
        compare_return=True,
    )


def test_an_unflavoured_member_outside_the_checked_list_is_refused() -> None:
    """The declaration names members, so equality cannot self-check it.

    An entry outside ``members`` claims an exception for a comparison nobody
    is making -- a stale suppression that reads as a clean scan, which is the
    failure this module's other declarations avoid by being compared against
    what is observed.
    """
    with pytest.raises(AssertionError, match="not among the members"):
        assert_twin_types_agree(
            Vocabulary, AsyncVocabulary, ("lookup",), unflavoured_members={"rank"}
        )


def test_the_listed_members_are_what_is_checked() -> None:
    """Omitting the drifted member passes -- which is why the list is explicit.

    Pinned rather than assumed: it is the reason the member names are written
    out at each call site instead of being discovered from the type.
    """
    assert_twin_types_agree(Reader, AsyncReader, ("read",))

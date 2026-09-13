"""Drift-guard for a sync/async twin pair: one surface, one stated difference.

A twinned API is two callables a caller is invited to treat as one. That
invitation is honest only while the pair actually agrees -- a keyword added to
the asynchronous half and forgotten on the synchronous one fails nothing at the
time, and surfaces later as flavour-agnostic code that is wrong against
whichever half its author did not reach for.

Nothing else in this package catches it. The factory-parity guards next door
compare a config surface to a constructor; these compare one flavour to its
twin, which is a different axis and a different bug.

**Why the allowed difference is a named set rather than a count.** Real
asymmetries exist: ``max_concurrency`` bounds a fan-out the synchronous lane
cannot have, and a knob that does nothing is worse than an asymmetry that is
stated. Naming it makes a *second* divergence fail rather than quietly join the
first, which a tolerance of "at most one" would not.

**Staleness cannot arise here, and the reason is worth stating.** The
suppression lists on this package's other guards need their own check that
every entry still matches something, because a suppression whose site moved is
a hole that reads as a clean scan. ``async_only`` needs no such check: it is
compared by *equality* against the observed difference, so an entry naming a
parameter since adopted, renamed or removed fails this assertion directly
rather than going quiet.

``unflavoured_members`` is the one declaration here naming *members* rather
than parameters, so equality against an observed difference is not available
to it -- an entry outside the list being checked would claim an exception for
a comparison nobody is making. It carries an explicit subset check instead,
which is the same guarantee reached the only way its shape allows.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = ["assert_twin_types_agree", "assert_twins_agree"]


def _is_async(impl: Callable[..., Any]) -> bool:
    """Whether a callable delivers its result by way of the event loop.

    Both forms count, and the second is the one that matters.
    ``inspect.iscoroutinefunction`` alone is the obvious check and the wrong
    one: an ``async def`` containing a ``yield`` is an *async generator*, for
    which it answers False. A streaming twin would then read as synchronous and
    fail the flavour assertion -- on precisely the pair whose two halves differ
    most, which is the pair most worth checking.
    """
    return inspect.iscoroutinefunction(impl) or inspect.isasyncgenfunction(impl)


def assert_twins_agree(
    sync_impl: Callable[..., Any],
    async_impl: Callable[..., Any],
    *,
    async_only: Iterable[str] = (),
    flavour_typed: Iterable[str] = (),
    unflavoured: bool = False,
    compare_return: bool = False,
    label: str = "",
) -> None:
    """Assert a synchronous callable and its asynchronous twin expose one surface.

    Args:
        sync_impl: The synchronous half. Must not be a coroutine or async
            generator function.
        async_impl: The asynchronous half. Must be one or the other.
        async_only: Parameter names the asynchronous half may have and the
            synchronous half may not, each for a reason worth writing down.
            Compared by equality, so an entry that stops being true fails here.
        flavour_typed: Parameter names whose *annotation* is expected to differ
            between the halves, because the parameter is itself flavoured -- a
            driver takes ``Hierarchy[K]`` on one side and ``AsyncHierarchy[K]``
            on the other. Compared by equality like ``async_only``: a name
            listed here whose annotations have since converged fails, so the
            exception cannot outlive its reason.
        unflavoured: Whether this member is deliberately synchronous on *both*
            halves. A twinned type has them -- a name property, a predicate, an
            ordering hook over data that has already arrived -- and they reach
            for nothing, so making one awaitable would cost every caller an
            ``await`` for no I/O. Without this the pair fails the flavour
            assertion, which left exactly those members unguarded: the only way
            to keep them green was to leave them out of the list, and a keyword
            added to one half and not the other does not stop being the drift
            this function exists to catch because both halves are synchronous.
        compare_return: Whether the return annotations must match too. False by
            default because a streaming twin's differs by flavour --
            ``Iterator[str]`` against ``AsyncIterator[str]`` -- and comparing
            them would assert the flavour rather than the contract. True where
            both halves return the same type.
        label: Prepended to failure messages, for a caller checking many pairs
            in a loop.

    Raises:
        AssertionError: On a flavour mismatch, a parameter present on one half
            only and not declared in ``async_only``, a shared parameter whose
            default or kind differs, or an annotation difference not declared
            in ``flavour_typed``.
    """
    where = f"{label}: " if label else ""

    assert not _is_async(sync_impl), f"{where}the synchronous twin is an async callable"
    if unflavoured:
        assert not _is_async(async_impl), (
            f"{where}this member is declared unflavoured, and the asynchronous twin's "
            f"half is an async callable. Drop `unflavoured` where the pair is genuinely "
            f"flavoured — like the other declarations here it is compared against what "
            f"is observed, so an entry that stops being true fails rather than going quiet"
        )
    else:
        assert _is_async(async_impl), (
            f"{where}the asynchronous twin is neither a coroutine function nor an "
            f"async generator function. A member deliberately synchronous on both "
            f"halves is declared with `unflavoured`, not omitted from the check"
        )

    sync_signature = inspect.signature(sync_impl)
    async_signature = inspect.signature(async_impl)
    sync_params = sync_signature.parameters
    async_params = async_signature.parameters

    expected = set(async_only)
    surplus = async_params.keys() - sync_params.keys()
    assert surplus == expected, (
        f"{where}the asynchronous twin's extra parameters are {sorted(surplus)}, "
        f"declared {sorted(expected)}. A difference the pair genuinely needs goes in "
        f"`async_only` with its reason; anything else is drift"
    )

    absent = sync_params.keys() - async_params.keys()
    assert absent == set(), (
        f"{where}{sorted(absent)} are on the synchronous twin and missing from the "
        f"asynchronous one — a caller writing flavour-agnostic code cannot pass them"
    )

    for name, parameter in sync_params.items():
        twin = async_params[name]
        assert parameter.default == twin.default, (
            f"{where}parameter {name!r} defaults to {parameter.default!r} on the "
            f"synchronous twin and {twin.default!r} on the asynchronous one"
        )
        assert parameter.kind == twin.kind, (
            f"{where}parameter {name!r} is {parameter.kind.description} on the "
            f"synchronous twin and {twin.kind.description} on the asynchronous one"
        )

    differing = {
        name
        for name, parameter in sync_params.items()
        if parameter.annotation != async_params[name].annotation
    }
    assert differing == set(flavour_typed), (
        f"{where}the parameters annotated differently are {sorted(differing)}, "
        f"declared {sorted(set(flavour_typed))}. A parameter that is itself flavoured "
        f"goes in `flavour_typed` with its reason; anything else is drift"
    )

    if compare_return:
        assert sync_signature.return_annotation == async_signature.return_annotation, (
            f"{where}the twins return {sync_signature.return_annotation!r} and "
            f"{async_signature.return_annotation!r}. Pass `compare_return=False` where "
            f"the difference is the flavour rather than the contract"
        )


def assert_twin_types_agree(
    sync_type: type,
    async_type: type,
    members: Sequence[str],
    *,
    async_only: Iterable[str] = (),
    flavour_typed: Iterable[str] = (),
    unflavoured_members: Iterable[str] = (),
    compare_return: bool = False,
) -> None:
    """:func:`assert_twins_agree` over the named members of two twinned types.

    The shape a protocol pair or a backend pair takes, where the surface to
    check is every member rather than one function. Each member is checked
    under the same terms, and the failure names it.

    Args:
        sync_type: The synchronous class or protocol.
        async_type: Its asynchronous twin.
        members: Member names to compare. Named rather than discovered, so that
            a member added to one half and not the other is caught by the test
            that lists it rather than silently dropping out of the comparison.
        async_only: As :func:`assert_twins_agree`, applied to every member.
        flavour_typed: As :func:`assert_twins_agree`, applied to every member.
        unflavoured_members: Member *names* — unlike the two sets above, which
            name parameters — that are synchronous on both halves. Passed as
            :func:`assert_twins_agree`'s ``unflavoured`` for those members and
            not for the rest, so one call covers a surface that mixes the two
            rather than splitting into a call per flavour. Must name members
            that are actually being checked; an entry outside ``members``
            declares an exception for a comparison nobody is making.
        compare_return: As :func:`assert_twins_agree`, applied to every member.

    Raises:
        AssertionError: From the first member that disagrees, naming it; or for
            an ``unflavoured_members`` entry absent from ``members``.
    """
    unflavoured = set(unflavoured_members)
    stale = unflavoured - set(members)
    assert stale == set(), (
        f"{sync_type.__name__}/{async_type.__name__}: {sorted(stale)} are declared "
        f"unflavoured but are not among the members being checked, so the declaration "
        f"guards nothing. Add them to `members` or drop them"
    )

    for name in members:
        assert_twins_agree(
            getattr(sync_type, name),
            getattr(async_type, name),
            async_only=async_only,
            flavour_typed=flavour_typed,
            unflavoured=name in unflavoured,
            compare_return=compare_return,
            label=f"{sync_type.__name__}/{async_type.__name__}.{name}",
        )

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
    assert _is_async(async_impl), (
        f"{where}the asynchronous twin is neither a coroutine function nor an "
        f"async generator function"
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
        compare_return: As :func:`assert_twins_agree`, applied to every member.

    Raises:
        AssertionError: From the first member that disagrees, naming it.
    """
    for name in members:
        assert_twins_agree(
            getattr(sync_type, name),
            getattr(async_type, name),
            async_only=async_only,
            flavour_typed=flavour_typed,
            compare_return=compare_return,
            label=f"{sync_type.__name__}/{async_type.__name__}.{name}",
        )

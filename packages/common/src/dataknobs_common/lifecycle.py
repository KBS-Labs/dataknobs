"""Collaborator-lifetime helpers for owned-vs-injected teardown.

A class that holds a collaborator (a database connection, an LLM
provider, a vector store, a connection pool) faces a recurring question
at ``close()`` time: *did I build this, or was it handed to me?* A
collaborator the holder built is owned — the holder must close it. A
collaborator injected by a caller (via a constructor kwarg, a
``from_components`` channel, or a shared-resource pattern) is
*caller-owned* — closing it would tear down a resource other holders
still depend on.

The settled idiom across dataknobs records that distinction in an
``_owns_*`` flag and gates the cascade::

    if self._owns_db and self._db is not None and hasattr(self._db, "close"):
        await self._db.close()

These helpers encapsulate that guard in one place so the dozen-plus
sites carrying it stay consistent. Three variants are provided, and
which one to reach for is decided by the *collaborator's* interface, not
by the caller's:

============================  ==========================
The collaborator exposes      Use
============================  ==========================
a synchronous ``close()``     :func:`close_if_owned_sync`
an ``async def close()``      :func:`close_if_owned`
an ``aclose()``, closed       :func:`aclose_if_owned`
from async
============================  ==========================

The third row is the one worth stating explicitly, because it is the one
where picking by the *caller's* context rather than the collaborator's
interface goes wrong. Take a collaborator that mirrors the sync/async
lifecycle pair — a synchronous ``close()`` alongside an ``aclose()`` that
awaits coroutine cleanup the sync form skips. From async code, neither
sibling is merely suboptimal for it: :func:`close_if_owned` would
``await`` a ``None`` return (``TypeError``), and
:func:`close_if_owned_sync` would silently take the lossy half.

The row is stated as "an ``aclose()``" rather than "both ``close()`` and
``aclose()``" because the probe is what decides. A collaborator whose
``close()`` is itself ``async`` and whose ``aclose()`` is an alias for it
(``AsyncMemoryBank``) satisfies two rows, and either helper is correct
there — the rows are not disjoint, and do not need to be.

The second row's blessing of an ``async def close()`` holds for a
collaborator whose shape the holder knows **statically** — which is the
case these helpers are for, since choosing between them is itself a
static decision. It does not extend to a collaborator held in a
heterogeneous *registry*, where the holder has many objects of unrelated
types and the method's name is the only thing it can route on. There an
``async def close()`` is indistinguishable from a synchronous one until
it has already been called and its coroutine discarded, so the name has
to carry the distinction: ``close()`` synchronous, ``aclose()`` awaited,
enforced when the collaborator is registered.
``dataknobs_fsm.resources.ResourceManager`` is such a registry.

What *is* worth knowing: each helper probes for exactly one method name
and skips when it is absent, so reaching for :func:`aclose_if_owned` on a
plain ``close()``-only collaborator closes **nothing**. That skip is
logged (see :func:`_report_unclosable`) rather than silent.

Error isolation is offered as an opt-in on all three: at a teardown
*cascade* (a bot closing knowledge base, then memory, then storage) one
failing subsystem must not abort the others. Pass ``on_error`` to catch
the exception and hand it to a callback (typically a logger) instead of
letting it propagate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "aclose_if_owned",
    "close_if_owned",
    "close_if_owned_sync",
]


def _report_unclosable(resource: Any, method: str) -> None:
    """Record that an owned collaborator exposes no way to close it.

    ``owns=True`` is a claim of responsibility for teardown, so finding no
    ``method`` to call means that responsibility cannot be discharged and
    whatever the collaborator holds is retained. Skipping is still the right
    behavior — raising would make a collaborator that legitimately needs no
    teardown impossible to hold — but doing it in total silence is not.

    That matters most for :func:`aclose_if_owned`, whose probe is the
    *unusual* name: called on a plain ``close()``-only collaborator it closes
    nothing at all, which is a worse outcome than either sibling's (one
    raises loudly, the other at least closes something).

    Logged at DEBUG rather than WARNING because, unlike a misspelled
    registry key, "no teardown method" has a legitimate shape — a frozen
    config, a plain mapping, a value object. A warning that fires on those
    is one people learn to ignore, which is the same silence by another
    route.
    """
    logger.debug(
        "Owned %s exposes no %s(); nothing was closed. If it holds a "
        "resource, either it needs that method or this holder should not "
        "claim ownership of it.",
        type(resource).__name__,
        method,
    )


async def close_if_owned(
    resource: Any,
    owns: bool,
    *,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Close ``resource`` iff this holder owns it (async).

    Closes ``resource`` only when ``owns`` is True, ``resource`` is not
    None, and it exposes a ``close()`` method. An injected (not-owned)
    collaborator is left untouched for its owner to close.

    Args:
        resource: The collaborator to (maybe) close. May be None.
        owns: Whether this holder owns ``resource``'s lifecycle. When
            False, ``resource`` is left untouched.
        on_error: Optional callback invoked with the exception when
            ``close()`` raises. When provided, the close is
            *error-isolated* — the ``Exception`` is caught and passed to
            ``on_error`` rather than propagating, so one failing subsystem
            in a teardown cascade does not abort the rest. When None (the
            default), exceptions propagate. Only ``Exception`` subclasses
            are isolated; ``asyncio.CancelledError`` and the other
            ``BaseException`` subclasses (``KeyboardInterrupt``,
            ``SystemExit``) always propagate regardless, so cancellation
            and interpreter shutdown are never swallowed.
    """
    if not (owns and resource is not None):
        return
    if not hasattr(resource, "close"):
        _report_unclosable(resource, "close")
        return
    if on_error is None:
        await resource.close()
    else:
        try:
            await resource.close()
        except Exception as exc:  # error isolation is the contract
            on_error(exc)


def close_if_owned_sync(
    resource: Any,
    owns: bool,
    *,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Close ``resource`` iff this holder owns it (synchronous).

    The synchronous counterpart of :func:`close_if_owned`, for holders
    whose collaborator exposes a synchronous ``close()`` (e.g. a sync
    database connection). Same ownership guard and same opt-in error
    isolation.

    Args:
        resource: The collaborator to (maybe) close. May be None.
        owns: Whether this holder owns ``resource``'s lifecycle. When
            False, ``resource`` is left untouched.
        on_error: Optional callback invoked with the exception when
            ``close()`` raises. When provided, the close is
            error-isolated; when None (the default), exceptions propagate.
            Only ``Exception`` subclasses are isolated; ``BaseException``
            subclasses (``KeyboardInterrupt``, ``SystemExit``) always
            propagate.
    """
    if not (owns and resource is not None):
        return
    if not hasattr(resource, "close"):
        _report_unclosable(resource, "close")
        return
    if on_error is None:
        resource.close()
    else:
        try:
            resource.close()
        except Exception as exc:  # error isolation is the contract
            on_error(exc)


async def aclose_if_owned(
    resource: Any,
    owns: bool,
    *,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Close ``resource`` via ``aclose()`` iff this holder owns it (async).

    The counterpart of :func:`close_if_owned` for collaborators that
    mirror the sync/async lifecycle pair — a synchronous ``close()``
    alongside an ``aclose()``. For those, neither sibling is correct:
    :func:`close_if_owned` would ``await`` what ``close()`` returns
    (``None``, raising ``TypeError``), and :func:`close_if_owned_sync`
    would call ``close()``, skipping the coroutine cleanup ``aclose()``
    exists to perform.

    A collaborator whose ``close()`` is itself ``async`` and whose
    ``aclose()`` merely aliases it is served correctly by either this or
    :func:`close_if_owned`; the two are not mutually exclusive.

    The ownership guard, the ``on_error`` contract, and the set of
    exceptions isolated are identical to :func:`close_if_owned`; only the
    probed method differs. That makes the ``hasattr`` check meaningful
    rather than decorative here — it is what discriminates a dual-method
    collaborator from a plain one.

    Args:
        resource: The collaborator to (maybe) close. May be None. Left
            untouched when it exposes no ``aclose()``.
        owns: Whether this holder owns ``resource``'s lifecycle. When
            False, ``resource`` is left untouched.
        on_error: Optional callback invoked with the exception when
            ``aclose()`` raises. When provided, the close is
            *error-isolated* — the ``Exception`` is caught and passed to
            ``on_error`` rather than propagating, so one failing subsystem
            in a teardown cascade does not abort the rest. When None (the
            default), exceptions propagate. Only ``Exception`` subclasses
            are isolated; ``asyncio.CancelledError`` and the other
            ``BaseException`` subclasses (``KeyboardInterrupt``,
            ``SystemExit``) always propagate regardless, so cancellation
            and interpreter shutdown are never swallowed.
    """
    if not (owns and resource is not None):
        return
    if not hasattr(resource, "aclose"):
        _report_unclosable(resource, "aclose")
        return
    if on_error is None:
        await resource.aclose()
    else:
        try:
            await resource.aclose()
        except Exception as exc:  # error isolation is the contract
            on_error(exc)

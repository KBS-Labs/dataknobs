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
sites carrying it stay consistent. Both an async and a sync variant are
provided because some collaborators (a sync database connection) expose
a synchronous ``close()``.

Error isolation is offered as an opt-in: at a teardown *cascade* (a bot
closing knowledge base, then memory, then storage) one failing subsystem
must not abort the others. Pass ``on_error`` to catch the exception and
hand it to a callback (typically a logger) instead of letting it
propagate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "close_if_owned",
    "close_if_owned_sync",
]


async def close_if_owned(
    resource: Any,
    owns: bool,
    *,
    on_error: Callable[[BaseException], None] | None = None,
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
            *error-isolated* — the exception is caught and passed to
            ``on_error`` rather than propagating, so one failing subsystem
            in a teardown cascade does not abort the rest. When None (the
            default), exceptions propagate. ``asyncio.CancelledError`` and
            other ``BaseException`` subclasses always propagate regardless,
            so cancellation is never swallowed.
    """
    if owns and resource is not None and hasattr(resource, "close"):
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
    on_error: Callable[[BaseException], None] | None = None,
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
    """
    if owns and resource is not None and hasattr(resource, "close"):
        if on_error is None:
            resource.close()
        else:
            try:
                resource.close()
            except Exception as exc:  # error isolation is the contract
                on_error(exc)

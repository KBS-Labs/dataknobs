"""``AsyncDatabase.from_backend`` resolves and builds off the event loop.

The door is ``async def`` and its docstring promises a *connected* instance,
so everything between the backend name and that instance is work this method
owns. Two parts of it block: resolving the name imports the backend
implementation through ``PluginRegistry``'s ``on_first_access`` hook, which
reads the module off disk, and constructing the backend's config normalizes
a filesystem path. Neither is I/O the method can avoid doing -- both are I/O
it can avoid doing *here*.

The harm is the ordinary one: a multi-tenant server building a handle for one
request stalls every other task on that loop for the duration. That it is
cheap and usually once per process is a reason it went unnoticed, not a
reason it is allowed -- the same argument would excuse any single blocking
call, and this one is on a published factory that a request path may reach.

``OntologyRegistry._database_handle`` in this package already runs the same
resolution inside ``asyncio.to_thread`` for exactly this reason, so the
offload is the established shape here rather than a new one.

Against the pre-fix code these FAIL with ``blockbuster.BlockingError``; after
the offload they PASS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_data.database import AsyncDatabase

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.asyncio, requires_blockbuster]


async def test_from_backend_does_not_resolve_a_file_backend_on_the_loop(
    tmp_path: Path,
) -> None:
    """The sqlite door, whose config normalizes a path as well as importing."""
    target = tmp_path / "nested" / "db.sqlite"

    with assert_no_blocking():
        db = await AsyncDatabase.from_backend("sqlite", {"path": str(target)})

    try:
        assert isinstance(db, AsyncDatabase)
    finally:
        await db.disconnect()


async def test_from_backend_does_not_resolve_an_in_process_backend_on_the_loop() -> None:
    """The memory door, which has no file to touch and still imports a module.

    Kept beside the sqlite case because the two block for different reasons
    and an offload that covered only the path handling would still leave the
    import on the loop -- for whichever backend a given process reaches first.
    """
    with assert_no_blocking():
        db = await AsyncDatabase.from_backend("memory")

    try:
        assert isinstance(db, AsyncDatabase)
    finally:
        await db.disconnect()


async def test_an_unknown_backend_still_refuses_by_name(tmp_path: Path) -> None:
    """The offload must not swallow or reshape the refusal it wraps.

    ``select_backend`` raises for an unrecognised name, and moving it into a
    worker thread re-raises the exception in the awaiting coroutine. Asserted
    because a ``to_thread`` that dropped it would look exactly like a pass.
    """
    with pytest.raises(ValueError, match="not-a-backend"):
        await AsyncDatabase.from_backend("not-a-backend", {"path": str(tmp_path)})

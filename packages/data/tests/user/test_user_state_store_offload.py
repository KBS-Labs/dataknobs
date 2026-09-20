"""``AsyncUserStateStore.from_config_async`` builds its database off the loop.

The store's async door promises a usable store, so everything between the
config and that store is work the door owns -- and where no handle was
injected, that includes resolving a backend name. Resolving one imports the
backend implementation through ``PluginRegistry``'s ``on_first_access`` hook,
which reads a module off disk.

**The sixth offload on this branch, and the one nothing pinned.** The other
five are covered -- ``AsyncDatabase.from_backend`` by
``test_async_from_backend_offload``, and the registry's environment read,
backend resolution, handle build and bus build by
``test_ontology_registry``'s two brackets. A remedy applied five times and
asserted four is the shape that comes undone quietly, because the uncovered
site is indistinguishable from the covered ones until someone edits it.

The harm is the ordinary one: a multi-tenant server building a store for one
request stalls every other task on that loop for the duration. Cheap and
once per process is why it goes unnoticed, not why it is allowed.

Against the un-offloaded code this FAILS with ``blockbuster.BlockingError``;
with the offload it PASSES.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster, requires_package

from dataknobs_data.user.store import AsyncUserStateStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.asyncio, requires_blockbuster]

SECTIONS = [{"name": "notes", "kind": "collection"}]


async def test_an_in_process_backend_is_not_resolved_on_the_loop() -> None:
    """The default door: no backend named, and a module still imported."""
    with assert_no_blocking():
        store = await AsyncUserStateStore.from_config_async({"sections": list(SECTIONS)})

    try:
        assert store is not None
    finally:
        await store.close()


@requires_package("aiosqlite")
async def test_a_file_backend_is_neither_resolved_nor_opened_on_the_loop(
    tmp_path: Path,
) -> None:
    """The backend whose config also normalizes a path, and which really connects.

    Kept beside the in-process case because the two block for different
    reasons: an offload covering only the import would leave the path work on
    the loop, and one covering only the build would leave ``connect()`` there
    -- and this door does both before it returns.
    """
    config: dict[str, Any] = {
        "backend": "sqlite",
        "path": str(tmp_path / "state.db"),
        "sections": list(SECTIONS),
    }

    with assert_no_blocking():
        store = await AsyncUserStateStore.from_config_async(config)

    try:
        await store.add_record("user-1", "notes", {"text": "hello"})
        assert [r.get_value("text") for r in await store.query("user-1", "notes")] == ["hello"]
    finally:
        await store.close()

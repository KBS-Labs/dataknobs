"""The async close half must match the sync half — and not stall the loop.

``AdvancedFSM`` and ``SimpleFSM`` each expose a sync ``close()`` and an
``aclose()``. Callers are told to prefer ``aclose()`` from async code because
it awaits provider cleanup the sync form skips. That advice is only safe if
the async half is otherwise a *superset* of the sync one, and if it is
actually non-blocking. Three ways it was not:

* ``AdvancedFSM.aclose()`` joined the bridge thread inline, stalling the
  caller's event loop for the whole shutdown — the defect
  ``.claude/rules/async-transport.md`` exists to prevent, on the path the
  documentation recommends.
* ``SimpleFSM.aclose()`` never released the bridge at all, so the thread its
  sync sibling reclaims leaked for the life of the process.
* ``ResourceManager.cleanup()`` (what ``aclose`` drives) awaited async
  providers but skipped the acquired-resource release, the pool close, and
  the closed-flag its sync sibling performs — so a pooled connection
  outlived the manager that owned it.

Real constructs only: real FSM builds, the real bridge, and
``PropertiesResource`` — an in-tree provider whose live instances are
directly observable, so "was it actually released?" is an assertion rather
than an inference.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from dataknobs_common.testing import (
    assert_no_blocking,
    assert_no_leaked_bridge_threads,
)

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    NetworkConfig,
    StateConfig,
)
from dataknobs_fsm.resources.base import ResourceStatus
from dataknobs_fsm.resources.database import AsyncDatabaseResourceAdapter
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.resources.pool import PoolConfig
from dataknobs_fsm.resources.properties import PropertiesResource


def _trivial_dict() -> dict[str, object]:
    """The same FSM in the dict form ``SimpleFSM`` accepts."""
    return {
        "name": "trivial",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [{"from": "start", "to": "end", "name": "go"}],
            }
        ],
    }


def _trivial_config() -> FSMConfig:
    """A minimal start→end FSM (no transforms, no resources)."""
    return FSMConfig(
        name="trivial",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(name="start", is_start=True, arcs=[ArcConfig(target="end")]),
                    StateConfig(name="end", is_end=True),
                ],
            )
        ],
    )


# --------------------------------------------------------------------------- #
# aclose() must not block the event loop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_advanced_fsm_aclose_does_not_block_the_event_loop() -> None:
    """``AdvancedFSM.aclose()`` offloads the bridge join.

    The bridge's ``close()`` stops its loop and *joins* the daemon thread.
    Joining inline from an ``async def`` freezes every other task on the
    caller's loop until the thread's in-flight step and async-generator
    shutdown complete — unbounded, and invisible in a single-request test.

    The bridge is allocated outside the guarded block: creating it requires a
    synchronous step, which is itself blocking and would otherwise be the
    thing detected.
    """
    fsm = AdvancedFSM(FSMBuilder().build(_trivial_config()))
    fsm.fsm.get_sync_bridge()  # allocate the daemon thread to be joined
    try:
        with assert_no_blocking():
            await fsm.aclose()
    finally:
        fsm.close()  # idempotent; reclaims the thread if the assert fired


@pytest.mark.asyncio
async def test_simple_fsm_aclose_does_not_block_the_event_loop() -> None:
    """Same contract on the sibling API, which has the same dual shape."""
    fsm = SimpleFSM(_trivial_dict())
    fsm._fsm.get_sync_bridge()
    try:
        with assert_no_blocking():
            await fsm.aclose()
    finally:
        fsm._fsm.close()


# --------------------------------------------------------------------------- #
# aclose() must release everything close() releases
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_simple_fsm_aclose_releases_the_bridge_thread() -> None:
    """``SimpleFSM.aclose()`` reclaims the thread its sync sibling reclaims.

    ``close()`` ends with ``self._fsm.close()``; ``aclose()`` awaited the
    async FSM and stopped there, so choosing the async form — the one the
    docs recommend — leaked the daemon thread outright.
    """
    with assert_no_leaked_bridge_threads():
        fsm = SimpleFSM(_trivial_dict())
        fsm._fsm.get_sync_bridge()
        await fsm.aclose()


@pytest.mark.asyncio
async def test_advanced_fsm_aclose_releases_the_bridge_thread() -> None:
    """The same guarantee on ``AdvancedFSM``, pinned so the offload keeps it.

    Moving the join off the loop must not turn it into a fire-and-forget:
    the thread has to be gone when ``aclose()`` returns, not merely
    scheduled for collection.
    """
    with assert_no_leaked_bridge_threads():
        fsm = AdvancedFSM(FSMBuilder().build(_trivial_config()))
        fsm.fsm.get_sync_bridge()
        await fsm.aclose()


# --------------------------------------------------------------------------- #
# Every API class offers the context-manager form of its lifecycle
# --------------------------------------------------------------------------- #


def test_simple_fsm_is_a_sync_context_manager() -> None:
    """``with SimpleFSM(...)`` closes on exit.

    ``AdvancedFSM`` had this and its siblings did not, which left the
    reliable way to avoid leaking the bridge available on one of three
    classes with the same lifecycle. Asking callers to remember ``close()``
    is how the leak happened in the first place.
    """
    with assert_no_leaked_bridge_threads():
        with SimpleFSM(_trivial_dict()) as fsm:
            fsm._fsm.get_sync_bridge()


@pytest.mark.asyncio
async def test_simple_fsm_is_an_async_context_manager() -> None:
    """``async with SimpleFSM(...)`` routes through the non-blocking half."""
    with assert_no_leaked_bridge_threads():
        async with SimpleFSM(_trivial_dict()) as fsm:
            fsm._fsm.get_sync_bridge()


@pytest.mark.asyncio
async def test_async_simple_fsm_is_an_async_context_manager() -> None:
    """``AsyncSimpleFSM`` closes its resource manager on context exit.

    It allocates no bridge — being async throughout, it never needs one — so
    what its context manager has to release is the resource manager. The
    gap it closes is the same one: a lifecycle whose only spelling was a
    method call the caller had to remember.
    """
    async with AsyncSimpleFSM(_trivial_dict()) as fsm:
        provider = PropertiesResource("props", initial_properties={"k": "v"})
        fsm._resource_manager.register_provider("props", provider)
        fsm._resource_manager.acquire("props", owner_id="state_a")
        assert provider.get_all_instances()

    assert provider.get_all_instances() == {}, (
        "exiting the AsyncSimpleFSM context manager did not clean up the resource manager"
    )


# --------------------------------------------------------------------------- #
# ResourceManager.cleanup() must be a superset of close()
# --------------------------------------------------------------------------- #


def _manager_with_pooled_provider() -> tuple[ResourceManager, PropertiesResource]:
    """A manager holding one pooled provider with one acquired resource."""
    manager = ResourceManager()
    provider = PropertiesResource("props", initial_properties={"k": "v"})
    manager.register_provider("props", provider, pool_config=PoolConfig(min_size=1, max_size=2))
    manager.acquire("props", owner_id="state_a")
    return manager, provider


def test_close_releases_acquired_resources() -> None:
    """Baseline: the sync half's behavior, so the parity claim is anchored.

    Without this, a regression that broke *both* halves would leave the
    parity test passing on two equally-wrong sides.
    """
    manager, provider = _manager_with_pooled_provider()
    assert provider.get_all_instances()

    manager.close()

    assert provider.get_all_instances() == {}


@pytest.mark.asyncio
async def test_cleanup_releases_acquired_resources() -> None:
    """``cleanup()`` returns acquired resources to their provider.

    Held to the same standard as the sync half. This provider's own
    ``close()`` happens to drop its instances, so the property survived the
    missing ``release_all`` by luck; a provider that releases only on
    ``release()`` would not have. Pinned so the parity is a contract rather
    than a coincidence of which provider is in the test.
    """
    manager, provider = _manager_with_pooled_provider()
    assert provider.get_all_instances()

    await manager.cleanup()

    assert provider.get_all_instances() == {}


@pytest.mark.asyncio
async def test_cleanup_closes_pools() -> None:
    """``cleanup()`` closes each pool rather than dropping the reference.

    ``_pools.clear()`` alone leaves the pool's own resources unreleased and
    its closed flag unset — the connections stay open until GC, if ever.
    """
    manager, _provider = _manager_with_pooled_provider()
    pool = manager._pools["props"]
    assert not pool._closed

    await manager.cleanup()

    assert pool._closed


@pytest.mark.asyncio
async def test_cleanup_marks_the_manager_closed() -> None:
    """Both close forms must be terminal in the *same* way.

    ``close()`` sets ``_closed``, so a later ``acquire`` reports "Resource
    manager is closed". ``cleanup()`` left the flag unset while clearing the
    providers, so the same call reported "Unknown resource" — the same
    outcome, diagnosed as a different bug.
    """
    manager, _provider = _manager_with_pooled_provider()

    await manager.cleanup()

    with pytest.raises(Exception, match="closed"):
        manager.acquire("props", owner_id="state_b")


# --------------------------------------------------------------------------- #
# What "reusable after close" does and does not cover
# --------------------------------------------------------------------------- #


def test_close_is_terminal_for_resources() -> None:
    """Closing releases providers; it does not re-register them.

    Documented as "the FSM stays usable after close", which is true of the
    bridge — a later synchronous step rebuilds it — and false of the
    resource manager. Pinned here so the two halves of that claim cannot be
    confused again: the manager rejects a later acquire outright.
    """
    manager, _provider = _manager_with_pooled_provider()

    manager.close()

    with pytest.raises(Exception, match="closed"):
        manager.acquire("props", owner_id="state_b")


@pytest.mark.asyncio
async def test_both_close_forms_are_terminal_the_same_way() -> None:
    """The sync and async halves must not disagree about what closed means.

    They did: ``close()`` set the closed flag, so a later acquire reported
    "Resource manager is closed", while ``cleanup()`` left the flag unset
    and merely dropped the providers, so the same call reported "Unknown
    resource" — one state, two diagnoses, and only one of them pointing at
    what actually happened.
    """
    closed_sync, _ = _manager_with_pooled_provider()
    closed_sync.close()
    with pytest.raises(Exception) as sync_error:
        closed_sync.acquire("props", owner_id="x")

    closed_async, _ = _manager_with_pooled_provider()
    await closed_async.cleanup()
    with pytest.raises(Exception) as async_error:
        closed_async.acquire("props", owner_id="x")

    assert str(sync_error.value) == str(async_error.value)


# --------------------------------------------------------------------------- #
# …and it must actually close the providers
# --------------------------------------------------------------------------- #
#
# Everything above compares what the two halves do *around* provider teardown —
# the acquired-resource release, the pools, the closed flag, the terminality —
# and nothing compared the teardown itself, which is the thing ``cleanup()``
# exists to do. That gap is why the case below survived: an
# ``AsyncDatabaseResourceAdapter`` does not override ``close()``, so the sync
# half runs the inherited ``BaseResourceProvider.close``, releases the handle
# list, and never touches the database. No coroutine is created, so nothing
# warns; the manager then clears its registry and the object holding the open
# connection becomes unreachable.


def _manager_with_async_database(tmp_path: Path) -> tuple[ResourceManager, object]:
    """A manager holding one async-database provider with an open backend."""
    manager = ResourceManager()
    adapter = AsyncDatabaseResourceAdapter(
        "target_db", type="file", path=str(tmp_path / "target.json")
    )
    manager.register_provider("target_db", adapter)
    return manager, adapter


@pytest.mark.asyncio
async def test_cleanup_closes_an_async_database_provider(tmp_path: Path) -> None:
    """The working half, pinned before the reporting half is added.

    ``cleanup()`` gets this right by accident of ordering — it probes
    ``aclose`` before falling through to the sync bucket — so this passes
    today and exists to keep it passing.
    """
    manager, adapter = _manager_with_async_database(tmp_path)
    await adapter._ensure_db()
    assert adapter._database is not None

    await manager.cleanup()

    assert adapter._database is None, "cleanup() left the database open"
    assert adapter.status is ResourceStatus.CLOSED


def test_close_reports_the_async_database_it_cannot_close(tmp_path: Path) -> None:
    """The sync half cannot await, so it must say so rather than claim success.

    The connection really does stay open — a synchronous ``close()`` has no
    loop to await ``aclose()`` on, and running one is not available to it:
    ``close()`` is reachable from ``__exit__``, where a loop may already be
    running in this thread. What is available is telling the truth about it,
    which is what makes the leak diagnosable instead of invisible.
    """
    manager, adapter = _manager_with_async_database(tmp_path)
    asyncio.run(adapter._ensure_db())
    assert adapter._database is not None

    manager.close()

    assert "target_db" in manager.unclosed_providers, (
        "a provider whose teardown must be awaited was closed synchronously and reported as closed"
    )

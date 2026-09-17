# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The versioning managers read from the store they write to.

Every test in this file fails against the code it replaces, and most of them
fail by reporting success. The three managers took a ``storage: Any`` and
duck-typed it for ``set``, ``append`` and ``delete``; against a real DataKnobs
backend the first two matched nothing, so every write was dropped in silence,
while ``delete`` matched by name and fired --- against ids that had therefore
never been written. Reads came from instance dictionaries the writes shadowed
and nothing replayed. Handed ``AsyncMemoryDatabase``, the layer wrote nothing,
read nothing, deleted rows that did not exist, and said ``True``.

The last two tests use SQLite rather than the memory backend, because two of
the claims here cannot be made against memory at all: that a version outlives
the connection that wrote it, and that these coroutines *suspend*. A
pure-Python ``async def`` awaiting only other pure-Python ``async def``s never
yields, so the memory backend is green on the suspension probe both before and
after this change. Only a real transport separates them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from dataknobs_common.testing import requires_package
from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_llm.prompts import VersionedPromptLibrary
from dataknobs_llm.prompts.versioning import (
    ABTestManager,
    DatabaseVersionStore,
    InMemoryVersionStore,
    MetricsCollector,
    PromptVariant,
    PromptVersion,
    VersionManager,
    VersionStatus,
)


class DuckTypedBackend:
    """What ``storage`` used to be handed, spelled out.

    Three coroutines, none of which any DataKnobs backend has under these
    names. It is here to be refused.
    """

    async def set(self, key: str, value: Any) -> None: ...

    async def append(self, key: str, value: Any) -> None: ...

    async def delete(self, key: str) -> None: ...


def completes_without_yielding(coro: Any) -> bool:
    """Drive a coroutine one step: did it finish without ever suspending?"""
    try:
        coro.send(None)
    except StopIteration:
        return True
    coro.close()
    return False


# ===== The acceptance case =====


@pytest.mark.asyncio
async def test_a_version_written_by_one_manager_is_read_by_another() -> None:
    """The claim the whole item rests on, and the one that could not be made.

    Two managers, one database, nothing shared in memory. Before this change
    the second saw ``None``, because the first had written nothing anywhere.
    """
    db = AsyncMemoryDatabase()

    writer = VersionManager(DatabaseVersionStore(db))
    written = await writer.create_version(
        name="greeting", prompt_type="system", template="Hello {{name}}!", version="1.0.0"
    )

    reader = VersionManager(DatabaseVersionStore(db))
    read = await reader.get_version("greeting", "system")

    assert read is not None
    assert read.version_id == written.version_id
    assert read.template == "Hello {{name}}!"
    assert await reader.list_names("system") == {"greeting"}
    assert [v.version for v in await reader.list_versions("greeting", "system")] == ["1.0.0"]


@pytest.mark.asyncio
async def test_the_write_reaches_the_database() -> None:
    """The row is there. It is worth asserting separately from reading it back.

    A read-back test can be satisfied by a cache; this one cannot. Before the
    change the count was ``0`` --- ``create_version`` returned a
    ``PromptVersion`` and persisted nothing.
    """
    db = AsyncMemoryDatabase()
    manager = VersionManager(DatabaseVersionStore(db))

    await manager.create_version(
        name="greeting", prompt_type="system", template="Hello!", version="1.0.0"
    )

    assert await db.count() == 1


@pytest.mark.asyncio
async def test_a_delete_reports_what_the_store_did() -> None:
    """The lie with teeth: ``True`` for a row the database had never held.

    ``delete_version`` consulted an instance dictionary the backend knew
    nothing about, so over an empty database it reported a deletion and issued
    one anyway. It now reports what the store reports.
    """
    db = AsyncMemoryDatabase()
    manager = VersionManager(DatabaseVersionStore(db))
    version = await manager.create_version(
        name="greeting", prompt_type="system", template="Hello!", version="1.0.0"
    )

    assert await manager.delete_version(version.version_id) is True
    assert await db.count() == 0
    assert await manager.delete_version(version.version_id) is False
    assert await manager.delete_version("never-existed") is False


@pytest.mark.asyncio
async def test_a_manager_answers_from_a_store_it_did_not_write_to() -> None:
    """No shadow dictionary left to read from.

    The store is populated behind the manager's back, so a manager still
    reading its own instance state would find nothing. This is the in-memory
    twin of the two-manager test above, and it is the one that would catch a
    cache quietly reintroduced in front of the store.
    """
    store = InMemoryVersionStore()
    await store.save_version(
        PromptVersion(
            version_id="v1",
            name="greeting",
            prompt_type="system",
            version="1.0.0",
            template="Hello!",
            status=VersionStatus.ACTIVE,
        )
    )

    manager = VersionManager(store)

    found = await manager.get_version("greeting", "system")
    assert found is not None
    assert found.version_id == "v1"
    assert await manager.list_names("system") == {"greeting"}


@pytest.mark.asyncio
async def test_experiments_and_their_assignments_persist() -> None:
    """Sticky assignment is only sticky if it outlives the object that made it."""
    db = AsyncMemoryDatabase()

    writer = ABTestManager(DatabaseVersionStore(db))
    experiment = await writer.create_experiment(
        name="greeting",
        prompt_type="system",
        variants=[PromptVariant("1.0.0", 0.5, "Control"), PromptVariant("1.0.1", 0.5, "Treatment")],
    )
    assigned = await writer.get_variant_for_user(experiment.experiment_id, "user123")

    reader = ABTestManager(DatabaseVersionStore(db))

    assert (await reader.get_experiment(experiment.experiment_id)) is not None
    assert await reader.get_user_assignment(experiment.experiment_id, "user123") == assigned
    assert await reader.get_variant_for_user(experiment.experiment_id, "user123") == assigned
    assert await reader.get_experiment_assignments(experiment.experiment_id) == {
        "user123": assigned
    }
    assert [e.experiment_id for e in await reader.list_experiments(status="running")] == [
        experiment.experiment_id
    ]


@pytest.mark.asyncio
async def test_metric_events_and_their_aggregate_persist() -> None:
    """Both entities, because they were persisted by different broken verbs.

    The aggregate went through ``set`` and the events through ``append``;
    neither exists on any backend, so a metrics collector over a database
    recorded nothing at all.
    """
    db = AsyncMemoryDatabase()

    writer = MetricsCollector(DatabaseVersionStore(db))
    await writer.record_event(version_id="v1", success=True, response_time=0.5, tokens=100)
    await writer.record_event(version_id="v1", success=False, response_time=1.5, tokens=200)

    reader = MetricsCollector(DatabaseVersionStore(db))
    metrics = await reader.get_metrics("v1")

    assert metrics.total_uses == 2
    assert metrics.success_count == 1
    assert metrics.error_count == 1
    assert metrics.avg_tokens == 150.0
    assert len(await reader.get_events("v1")) == 2

    assert await reader.reset_metrics("v1") is True
    assert (await MetricsCollector(DatabaseVersionStore(db)).get_metrics("v1")).total_uses == 0


# ===== The library =====


@pytest.mark.asyncio
async def test_the_library_hands_one_store_to_all_three_managers() -> None:
    """One object, three protocols --- exactly as one ``storage`` used to be."""
    store = InMemoryVersionStore()
    library = VersionedPromptLibrary(store=store)

    assert library.store is store
    assert library.version_manager.store is store
    assert library.ab_test_manager.store is store
    assert library.metrics_collector.store is store


@pytest.mark.asyncio
async def test_the_library_defaults_to_the_in_memory_store() -> None:
    """Constructing one with no arguments keeps working, and says what it got."""
    library = VersionedPromptLibrary()

    assert isinstance(library.store, InMemoryVersionStore)
    assert library.get_metadata() == {
        "type": "VersionedPromptLibrary",
        "store": "InMemoryVersionStore",
        "has_base_library": False,
    }


@pytest.mark.asyncio
async def test_the_library_reads_back_what_it_wrote_through_a_database() -> None:
    """End to end: the accessor a consumer actually calls, over a real backend."""
    db = AsyncMemoryDatabase()

    writer = VersionedPromptLibrary(store=DatabaseVersionStore(db))
    await writer.create_version(
        name="greeting", prompt_type="system", template="Hello {{name}}!", version="1.0.0"
    )

    reader = VersionedPromptLibrary(store=DatabaseVersionStore(db))
    template = await reader.get_system_prompt("greeting")

    assert template is not None
    assert template["template"] == "Hello {{name}}!"
    assert await reader.list_system_prompts() == ["greeting"]
    assert reader.get_metadata()["store"] == "DatabaseVersionStore"


# ===== Refusing the parameter this replaced =====


@pytest.mark.parametrize(
    ("holder", "protocol_name"),
    [
        (VersionManager, "VersionStore"),
        (ABTestManager, "ExperimentStore"),
        (MetricsCollector, "MetricsStore"),
        (VersionedPromptLibrary, "VersioningStore"),
    ],
)
def test_the_backend_this_replaces_is_refused_at_construction(
    holder: type, protocol_name: str
) -> None:
    """Where the author is still looking, rather than at the first dropped write.

    Each manager names the protocol it needs, so the message says which part
    of the surface is wanted --- and every one of them says how to get one.
    """
    with pytest.raises(TypeError) as caught:
        holder(DuckTypedBackend())

    message = str(caught.value)
    assert f"needs a {protocol_name}" in message
    assert "DuckTypedBackend is missing" in message
    assert "InMemoryVersionStore()" in message


# ===== Against a real async transport =====


@requires_package("aiosqlite")
@pytest.mark.asyncio
async def test_a_version_outlives_the_connection_that_wrote_it(tmp_path: Path) -> None:
    """Persistence, in the sense a consumer means it: on disk, after a close.

    The memory backend cannot make this claim --- it *is* the process. This is
    the shape the item's acceptance criterion was written for, and the layer
    could not satisfy it under any backend.
    """
    from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

    path = str(tmp_path / "prompts.db")

    db = AsyncSQLiteDatabase({"path": path})
    await db.connect()
    library = VersionedPromptLibrary(store=DatabaseVersionStore(db))
    await library.create_version(
        name="greeting", prompt_type="system", template="Hello {{name}}!", version="1.0.0"
    )
    await db.close()

    reopened = AsyncSQLiteDatabase({"path": path})
    await reopened.connect()
    try:
        later = VersionedPromptLibrary(store=DatabaseVersionStore(reopened))
        template = await later.get_system_prompt("greeting")
        assert template is not None
        assert template["template"] == "Hello {{name}}!"
        assert await later.list_system_prompts() == ["greeting"]
    finally:
        await reopened.close()


@requires_package("aiosqlite")
@pytest.mark.asyncio
async def test_the_coroutines_suspend_against_a_real_transport(tmp_path: Path) -> None:
    """The ``async`` on these methods stops being decorative.

    Every one of them used to complete on a single ``send(None)``: a method
    that never reads has nothing to await, and the writes went nowhere. Driven
    against a database with a real transport underneath, ``create_version``
    now yields to the loop before it finishes --- which is what makes it
    honest for a caller to await.

    The memory backend would not show this and does not disprove it: its
    coroutines never suspend either, so the probe is green on both sides of
    this change. That is why the assertion is made here.
    """
    from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

    db = AsyncSQLiteDatabase({"path": str(tmp_path / "prompts.db")})
    await db.connect()
    try:
        manager = VersionManager(DatabaseVersionStore(db))

        assert not completes_without_yielding(
            manager.create_version(
                name="greeting", prompt_type="system", template="Hello!", version="1.0.0"
            )
        )

        # The control: the same probe over a dictionary, where nothing
        # suspends because nothing has to.
        assert completes_without_yielding(
            VersionManager(InMemoryVersionStore()).create_version(
                name="greeting", prompt_type="system", template="Hello!", version="1.0.0"
            )
        )
    finally:
        await db.close()
        await asyncio.sleep(0)

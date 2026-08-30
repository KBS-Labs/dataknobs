"""A backend's vector methods belong to the lane the backend is in.

``VectorOperationsMixin`` declared every one of its methods ``async`` and was
mixed into sync backends as readily as async ones --- five of the seven sync
backends carry a vector surface, and all five got the async mixin. Three of
the methods it *implements* (rather than declares abstract) then called
``await self.read(...)`` / ``await self.delete(...)`` / ``await
self.search(...)`` on a sync database, so on a sync backend they raised
``TypeError: object NoneType can't be used in 'await' expression`` --- and the
sibling defect on the declared half was that those five overrode an async
declaration with a sync definition, which a type checker reported and nothing
else reported at all.

This is the same defect the branch already fixed one module away, in the
opposite direction: every async backend had inherited the *sync*
``BulkEmbedMixin``, whose ``self.exists`` / ``self.update`` / ``self.create``
calls produced coroutines nobody awaited --- and a coroutine object is truthy,
so the ``exists`` branch was taken unconditionally and nothing was ever
written, with no exception anywhere.

``TestEveryBackendsVectorMethodsMatchItsLane`` is the guard that would have
caught both. It is a property of the whole family rather than of any one
backend, which is why it is a sweep over all fourteen rather than a cell per
class.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.database import AsyncDatabase, SyncDatabase

BACKEND_MODULES = (
    "memory",
    "file",
    "sqlite",
    "sqlite_async",
    "s3",
    "s3_async",
    "duckdb",
    "postgres",
    "elasticsearch",
    "elasticsearch_async",
)

# Every method of the vector surface whose laneness a caller depends on ---
# the two abstract ones, the ones the mixin implements, and the two private
# helpers ``hybrid_search`` awaits, which have to agree with it or it breaks.
LANE_METHODS = (
    "vector_search",
    "bulk_embed_and_store",
    "update_vector",
    "delete_from_index",
    "hybrid_search",
    "create_vector_index",
    "drop_vector_index",
    "get_vector_index_stats",
    "_text_search_for_hybrid",
    "_supports_native_hybrid",
)


def _backend_classes() -> list[type]:
    found: list[type] = []
    for name in BACKEND_MODULES:
        module = importlib.import_module(f"dataknobs_data.backends.{name}")
        found.extend(
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and obj.__module__ == module.__name__
            and issubclass(obj, (AsyncDatabase, SyncDatabase))
        )
    return found


BACKENDS = _backend_classes()


def _cases() -> list[tuple[type, str]]:
    return [(cls, method) for cls in BACKENDS for method in LANE_METHODS]


class TestEveryBackendsVectorMethodsMatchItsLane:
    """The recurrence guard, over the whole family at once."""

    def test_the_sweep_found_the_backends(self) -> None:
        """A parity sweep that enumerates nothing passes vacuously."""
        assert len(BACKENDS) == 14

    @pytest.mark.parametrize(
        ("cls", "method"), _cases(), ids=lambda x: x if isinstance(x, str) else x.__name__
    )
    def test_laneness(self, cls: type, method: str) -> None:
        function = getattr(cls, method, None)
        if function is None:
            pytest.skip(f"{cls.__name__} does not offer {method}")

        lane_is_async = issubclass(cls, AsyncDatabase)

        assert inspect.iscoroutinefunction(function) is lane_is_async, (
            f"{cls.__name__}.{method} is "
            f"{'async' if inspect.iscoroutinefunction(function) else 'sync'} "
            f"but {cls.__name__} is a "
            f"{'AsyncDatabase' if lane_is_async else 'SyncDatabase'}"
        )


class TestTheInheritedMethodsRunOnASyncBackend:
    """The reproduce cells: three ``TypeError``s, measured before the split."""

    @pytest.fixture
    def db(self) -> SyncMemoryDatabase:
        database = SyncMemoryDatabase(config={"vector_enabled": True})
        database.create(
            Record(data={"id": "r1", "content": "hello world", "embedding": [1.0, 0.0]})
        )
        return database

    def test_update_vector(self, db: SyncMemoryDatabase) -> None:
        assert db.update_vector("r1", "embedding", [0.0, 1.0]) is True

    def test_update_vector_missing_record(self, db: SyncMemoryDatabase) -> None:
        assert db.update_vector("nope", "embedding", [0.0, 1.0]) is False

    def test_delete_from_index(self, db: SyncMemoryDatabase) -> None:
        assert db.delete_from_index("r1") is True

    def test_hybrid_search(self, db: SyncMemoryDatabase) -> None:
        results = db.hybrid_search("hello", np.array([1.0, 0.0]), text_fields=["content"])

        assert [r.record.get_value("content") for r in results] == ["hello world"]

    def test_the_index_helpers(self, db: SyncMemoryDatabase) -> None:
        assert db.create_vector_index("embedding") is True
        assert db.drop_vector_index("embedding") is True
        assert db.get_vector_index_stats("embedding")["field"] == "embedding"


class TestTheAsyncLaneStillWorks:
    """The other half of the split, unchanged in behaviour."""

    @pytest.fixture
    async def db(self) -> AsyncMemoryDatabase:
        database = AsyncMemoryDatabase(config={"vector_enabled": True})
        await database.create(
            Record(data={"id": "r1", "content": "hello world", "embedding": [1.0, 0.0]})
        )
        return database

    @pytest.mark.asyncio
    async def test_update_vector(self, db: AsyncMemoryDatabase) -> None:
        assert await db.update_vector("r1", "embedding", [0.0, 1.0]) is True

    @pytest.mark.asyncio
    async def test_delete_from_index(self, db: AsyncMemoryDatabase) -> None:
        assert await db.delete_from_index("r1") is True

    @pytest.mark.asyncio
    async def test_hybrid_search(self, db: AsyncMemoryDatabase) -> None:
        results = await db.hybrid_search("hello", np.array([1.0, 0.0]), text_fields=["content"])

        assert [r.record.get_value("content") for r in results] == ["hello world"]

    @pytest.mark.asyncio
    async def test_the_index_helpers(self, db: AsyncMemoryDatabase) -> None:
        assert await db.create_vector_index("embedding") is True
        assert await db.drop_vector_index("embedding") is True
        assert (await db.get_vector_index_stats("embedding"))["field"] == "embedding"


class TestBothLanesAreImportable:
    """Named separately, so a backend author picks rather than inherits one."""

    def test_the_sync_lane_declares_sync_methods(self) -> None:
        from dataknobs_data.vector import SyncVectorOperationsMixin

        for method in LANE_METHODS:
            function = getattr(SyncVectorOperationsMixin, method)
            assert not inspect.iscoroutinefunction(function), method

    def test_the_async_lane_declares_async_methods(self) -> None:
        from dataknobs_data.vector import AsyncVectorOperationsMixin

        for method in LANE_METHODS:
            function = getattr(AsyncVectorOperationsMixin, method)
            assert inspect.iscoroutinefunction(function), method

    def test_the_old_name_still_resolves_to_the_async_lane(self) -> None:
        """Back-compat: the bare name has always meant the async lane."""
        from dataknobs_data.vector import (
            AsyncVectorOperationsMixin,
            VectorOperationsMixin,
        )

        assert VectorOperationsMixin is AsyncVectorOperationsMixin

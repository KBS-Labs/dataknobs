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
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.file import SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
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


# Exactly the keywords ``hybrid_search`` passes to ``vector_search`` in both
# lanes. An implementation that does not name every one of them is one the
# mixin cannot call, whichever lane it is in.
HYBRID_CALL_KEYWORDS = ("query_vector", "vector_field", "k", "metric", "filter")


def _named_parameters(function: object) -> dict[str, inspect.Parameter]:
    """The parameters a caller can reach *by name*.

    ``**kwargs`` is deliberately not consulted. A keyword it swallows binds
    at the signature and then goes wherever the body forwards it, which for
    ``SyncSQLiteDatabase`` was into a second value for a parameter already
    passed positionally --- a `TypeError` one frame further down than any
    signature check looks.
    """
    parameters = inspect.signature(function).parameters  # type: ignore[arg-type]
    return {
        name: parameter
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
    }


class TestEveryBackendsVectorMethodsMatchItsLane:
    """The recurrence guard, over the whole family at once."""

    def test_the_sweep_found_the_backends(self) -> None:
        """A parity sweep that enumerates nothing passes vacuously."""
        assert len(BACKENDS) == 14

    def test_the_sweep_found_live_cells(self) -> None:
        """...and neither does one whose every cell skips.

        ``test_laneness`` skips a `(class, method)` pair the class does not
        offer, which is legitimate --- both DuckDB classes carry no vector
        surface and `SyncElasticsearchDatabase` carries two methods of it.
        But a mixin method *renamed* would empty every cell while the count
        above still passed, so the floor is asserted rather than assumed.
        """
        live = sum(1 for cls, method in _cases() if getattr(cls, method, None) is not None)

        assert live >= 100, f"only {live} of {len(_cases())} cells are live"

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


class TestEveryBackendCanBeCalledTheWayTheMixinCallsIt:
    """Laneness was the wrong property to stop at.

    The split made ``hybrid_search`` reachable on five sync backends, and it
    calls ``vector_search`` by keyword. Four of the twelve implementations
    named the field parameter ``field_name`` instead --- so two of the five
    raised rather than searched, and the same spelling every shipped example
    uses (``vector_field=``) had never worked on them through a direct call
    either:

        SyncSQLiteDatabase   TypeError: python_vector_search_sync() got
                             multiple values for keyword argument
                             'vector_field'
        SyncPostgresDatabase TypeError: vector_search() got an unexpected
                             keyword argument 'vector_field'

    `test_laneness` cannot see this --- both methods are in the right lane.
    """

    VECTOR_SEARCHERS: ClassVar[list[type]] = [
        cls for cls in BACKENDS if getattr(cls, "vector_search", None)
    ]

    @pytest.mark.parametrize("cls", VECTOR_SEARCHERS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("keyword", HYBRID_CALL_KEYWORDS)
    def test_the_keyword_is_a_named_parameter(self, cls: type, keyword: str) -> None:
        named = _named_parameters(cls.vector_search)

        assert keyword in named, (
            f"{cls.__name__}.vector_search has no parameter named {keyword!r} "
            f"(it has {sorted(named)}) --- the mixin's hybrid_search passes it"
        )

    @pytest.mark.parametrize("cls", VECTOR_SEARCHERS, ids=lambda c: c.__name__)
    def test_only_the_query_vector_is_required(self, cls: type) -> None:
        """A field the caller must supply is one ``hybrid_search`` cannot omit.

        Both Postgres classes made ``vector_field`` a required positional
        where the other ten default it, so the twelve did not agree on what a
        minimal call looks like either.
        """
        required = [
            name
            for name, parameter in _named_parameters(cls.vector_search).items()
            if parameter.default is inspect.Parameter.empty and name != "query_vector"
        ]

        assert required == [], f"{cls.__name__}.vector_search also requires {required}"


@pytest.fixture(params=["memory", "file", "sqlite"])
def sync_db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SyncDatabase]:
    """Every sync vector backend that runs in-process.

    Parametrised rather than hard-coded to memory, because memory is one of
    the three that happened to work --- the two that raised were sqlite and
    postgres, and a single-backend cell is what let that through.
    ``SyncS3Database`` and ``SyncPostgresDatabase`` are the other two sync
    backends with a vector surface; both need a service, so the signature
    sweep above is what covers them here.
    """
    backend = request.param
    if backend == "memory":
        database: SyncDatabase = SyncMemoryDatabase(config={"vector_enabled": True})
    elif backend == "file":
        database = SyncFileDatabase(
            config={"path": str(tmp_path / "records"), "vector_enabled": True}
        )
    else:
        database = SyncSQLiteDatabase(
            config={"path": str(tmp_path / "records.db"), "vector_enabled": True}
        )
    database.connect()
    database.create(Record(data={"id": "r1", "content": "hello world", "embedding": [1.0, 0.0]}))
    try:
        yield database
    finally:
        database.close()


class TestTheInheritedMethodsRunOnASyncBackend:
    """The reproduce cells: three ``TypeError``s, measured before the split."""

    def test_update_vector(self, sync_db: SyncDatabase) -> None:
        assert sync_db.update_vector("r1", "embedding", [0.0, 1.0]) is True

    def test_update_vector_actually_stores_the_vector(self, sync_db: SyncDatabase) -> None:
        """``update`` returns ``bool``, and ``bool is not None`` is always true.

        So the return value said nothing about whether the write landed ---
        a version conflict answered ``True`` exactly as a success did.
        """
        sync_db.update_vector("r1", "embedding", [0.0, 1.0])

        stored = sync_db.read("r1")
        assert stored is not None
        assert list(stored.get_value("embedding")) == [0.0, 1.0]

    def test_update_vector_missing_record(self, sync_db: SyncDatabase) -> None:
        assert sync_db.update_vector("nope", "embedding", [0.0, 1.0]) is False

    def test_update_vector_reports_a_write_that_did_not_happen(self, sync_db: SyncDatabase) -> None:
        """``update`` returns ``bool``; ``False is not None`` is ``True``.

        The record is readable when ``update_vector`` reads it and gone by
        the time it writes --- the shape of a delete landing between the two,
        which is the one path that reaches a ``False`` from ``update`` on a
        backend where the id exists at read time. The subclass exists to make
        that interleaving happen at a fixed point rather than by luck; it is
        a real ``SyncMemoryDatabase`` in every other respect, and it exercises
        the mixin's real code path.
        """

        class DeletedBetweenReadAndWrite(SyncMemoryDatabase):
            def update(self, id: str, record: Record, **kwargs: object) -> bool:
                self.delete(id)
                return super().update(id, record, **kwargs)  # type: ignore[arg-type]

        database = DeletedBetweenReadAndWrite(config={"vector_enabled": True})
        database.create(Record(data={"id": "r1", "content": "x", "embedding": [1.0, 0.0]}))

        assert database.update_vector("r1", "embedding", [0.0, 1.0]) is False

    def test_delete_from_index(self, sync_db: SyncDatabase) -> None:
        assert sync_db.delete_from_index("r1") is True

    def test_hybrid_search(self, sync_db: SyncDatabase) -> None:
        results = sync_db.hybrid_search("hello", np.array([1.0, 0.0]), text_fields=["content"])

        assert [r.record.get_value("content") for r in results] == ["hello world"]

    def test_vector_search_takes_the_documented_keyword(self, sync_db: SyncDatabase) -> None:
        """``vector_field=`` is the spelling every shipped example uses."""
        results = sync_db.vector_search(query_vector=np.array([1.0, 0.0]), vector_field="embedding")

        assert [r.record.get_value("content") for r in results] == ["hello world"]

    def test_the_index_helpers(self, sync_db: SyncDatabase) -> None:
        assert sync_db.create_vector_index("embedding") is True
        assert sync_db.drop_vector_index("embedding") is True
        assert sync_db.get_vector_index_stats("embedding")["field"] == "embedding"


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

"""Id-keyed record store abstractions over ``AsyncDatabase`` / ``SyncDatabase``.

``AsyncKeyedRecordStore[T]`` and ``SyncKeyedRecordStore[T]`` are generic
wrappers that encapsulate ``Record`` construction for id-keyed
registry / pointer-table use cases.  The store preserves the ``Record``
two-column (``data`` and ``metadata``) shape *by construction*: the
serializer signature ``(T) -> (dict, dict)`` makes the metadata channel
part of the function's type, so a consumer cannot accidentally route
metadata into the data column or omit metadata entirely.

Use this instead of building ``Record(...)`` inline whenever a backend
is being used as an id-keyed key/value store with structured payloads.

Example:
    ```python
    from dataclasses import dataclass
    from dataknobs_data import AsyncKeyedRecordStore
    from dataknobs_data.backends.memory import AsyncMemoryDatabase

    @dataclass
    class Registration:
        bot_id: str
        config: dict
        status: str
        tenant_id: str | None = None

    def to_columns(r: Registration) -> tuple[dict, dict]:
        data = {"bot_id": r.bot_id, "config": r.config, "status": r.status}
        metadata = {"tenant_id": r.tenant_id} if r.tenant_id else {}
        return data, metadata

    def from_record(record) -> Registration:
        return Registration(
            bot_id=record.get_value("bot_id"),
            config=record.get_value("config"),
            status=record.get_value("status"),
            tenant_id=record.metadata.get("tenant_id"),
        )

    db = AsyncMemoryDatabase()
    await db.connect()
    store = AsyncKeyedRecordStore[Registration](
        db, serializer=to_columns, deserializer=from_record,
    )
    await store.put("bot-a", Registration("bot-a", {}, "active", "t1"))
    t1_regs = await store.list(filter_metadata={"tenant_id": "t1"})
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .query import Filter, Operator, Query
from .records import Record

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence

    from .database import AsyncDatabase, SyncDatabase
    from .query import SortSpec, VectorQuery
    from .streaming import StreamConfig


T = TypeVar("T")


def _build_query(
    *,
    filter_data: Mapping[str, Any] | None,
    filter_metadata: Mapping[str, Any] | None,
    sort: list[SortSpec] | None = None,
    limit: int | None = None,
    offset: int | None = None,
    vector_query: VectorQuery | None = None,
) -> Query:
    """Compose a ``Query`` from typed-channel filter maps.

    ``filter_metadata`` entries are routed to the ``metadata`` column via
    the ``metadata.X`` field-path convention already honored by all
    backends (SQL backends route through their JSONB column; the memory
    backend resolves the path through ``Record.get_nested_value``).
    """
    filters: list[Filter] = []
    for k, v in (filter_data or {}).items():
        filters.append(Filter(k, Operator.EQ, v))
    for k, v in (filter_metadata or {}).items():
        filters.append(Filter(f"metadata.{k}", Operator.EQ, v))

    q = Query(filters=filters)
    if sort is not None:
        q.sort_specs = list(sort)
    if limit is not None:
        q.limit_value = limit
    if offset is not None:
        q.offset_value = offset
    if vector_query is not None:
        q.vector_query = vector_query
    return q


def _build_record(
    key: str,
    value: T,
    serializer: Callable[[T], tuple[dict[str, Any], dict[str, Any]]],
) -> Record:
    """Build a ``Record`` from a typed value via its serializer.

    Centralizing record construction here is the structural prevention
    contract: every store operation routes through this single site.
    """
    data, metadata = serializer(value)
    return Record(data=dict(data), metadata=dict(metadata), storage_id=key)


class AsyncKeyedRecordStore(Generic[T]):
    """Generic id-keyed record store over ``AsyncDatabase``.

    The serializer signature ``(T) -> (dict, dict)`` (returning a tuple
    of ``data`` and ``metadata`` dicts, not a ``Record``) is load-bearing:
    it makes the metadata channel part of the function's type so that no
    consumer can construct a ``Record`` that forgets the metadata column.
    Filter channels mirror the same separation — ``filter_data`` and
    ``filter_metadata`` route to the correct column without callers
    needing to know the field-path convention.
    """

    def __init__(
        self,
        db: AsyncDatabase,
        *,
        serializer: Callable[[T], tuple[dict[str, Any], dict[str, Any]]],
        deserializer: Callable[[Record], T],
    ) -> None:
        """Initialize the store.

        Args:
            db: An ``AsyncDatabase`` instance.  Lifecycle (``connect`` /
                ``close``) remains the caller's responsibility.
            serializer: Function returning the ``(data, metadata)`` tuple
                for a typed value.  Each dict is copied before being stored.
            deserializer: Function reconstructing the typed value from a
                retrieved ``Record``.  Receives the full record so that
                both ``record.data`` / ``record.get_value(...)`` and
                ``record.metadata`` are available.
        """
        self._db = db
        self._serializer = serializer
        self._deserializer = deserializer

    @property
    def db(self) -> AsyncDatabase:
        """The underlying ``AsyncDatabase``.

        Exposed read-only for lifecycle management (``initialize``,
        ``connect``, ``close``) and for callers that legitimately need to
        drop to the raw surface (e.g. schema inspection).  Consumers
        SHOULD NOT use this to bypass the store for CRUD on records the
        store also manages.
        """
        return self._db

    async def put(self, key: str, value: T) -> None:
        """Insert or update the record for ``key``."""
        record = _build_record(key, value, self._serializer)
        await self._db.upsert(key, record)

    async def get(self, key: str) -> T | None:
        """Return the value for ``key``, or ``None`` if not found."""
        record = await self._db.read(key)
        if record is None:
            return None
        return self._deserializer(record)

    async def exists(self, key: str) -> bool:
        """Check whether the key is present in the underlying database."""
        return await self._db.exists(key)

    async def delete(self, key: str) -> bool:
        """Delete the record for ``key``.  Returns ``True`` if deleted."""
        return await self._db.delete(key)

    async def put_batch(self, items: Mapping[str, T]) -> None:
        """Insert or update multiple records.

        Default implementation is sequential ``put`` calls; backends that
        offer batched upsert can be wired in later without changing the
        consumer surface.
        """
        for k, v in items.items():
            await self.put(k, v)

    async def get_batch(self, keys: Sequence[str]) -> list[T | None]:
        """Return values for ``keys`` in order (``None`` for misses)."""
        records = await self._db.read_batch(list(keys))
        return [self._deserializer(r) if r is not None else None for r in records]

    async def delete_batch(self, keys: Sequence[str]) -> int:
        """Delete records for ``keys``; returns the number actually deleted."""
        results = await self._db.delete_batch(list(keys))
        return sum(1 for r in results if r)

    async def clear(self) -> int:
        """Delete every record in the underlying database. Returns count deleted.

        Default implementation enumerates via ``AsyncDatabase.all()`` and
        deletes by ``storage_id``.  Backends that ship a native truncate
        (``TRUNCATE TABLE``, ``delete_by_query``, multi-object delete) can
        add a fast path on the database surface later and every consumer
        of this store inherits it transparently.

        Warning:
            **This deletes every record in the underlying database, not
            just records previously written through this store.**  When
            multiple logical stores share a single database (the FSM
            history+steps shared backend is the canonical example), a
            call to ``clear()`` on one store will wipe the other's
            records too.  Consumers sharing a database must:

            - Scope ``clear`` at the adapter / owning-object level
              (e.g., the registry that composes the store), so the
              caller intent is "clear all records this entity owns"
              rather than "clear the entire backing database"; or
            - Use ``delete_batch(...)`` with a list of keys filtered
              by a discriminator in the serializer's data shape
              (e.g., a ``"_kind"`` field); or
            - Keep each logical store in its own database.

            ``clear()`` is provided as a convenience for the
            single-store case (most tests, simple deployments).
        """
        records = await self._db.all()
        keys = [r.storage_id for r in records if r.storage_id]
        if not keys:
            return 0
        return await self.delete_batch(keys)

    async def list(
        self,
        *,
        filter_data: Mapping[str, Any] | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        vector_query: VectorQuery | None = None,
    ) -> list[T]:
        """List values matching the supplied filters.

        ``filter_data`` and ``filter_metadata`` are AND-combined.  Empty
        dicts are equivalent to ``None`` — both mean "no filter on that
        channel".
        """
        q = _build_query(
            filter_data=filter_data,
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
            vector_query=vector_query,
        )
        records = await self._db.search(q)
        return [self._deserializer(r) for r in records]

    async def count(
        self,
        *,
        filter_data: Mapping[str, Any] | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count matching records.

        Routes through ``AsyncDatabase.count(query)`` so that backends
        which ship pushdown counts (``SELECT COUNT(*) WHERE ...``)
        benefit every consumer without API changes.
        """
        if not filter_data and not filter_metadata:
            return await self._db.count()
        q = _build_query(filter_data=filter_data, filter_metadata=filter_metadata)
        return await self._db.count(q)

    async def stream(
        self,
        *,
        filter_data: Mapping[str, Any] | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        config: StreamConfig | None = None,
    ) -> AsyncIterator[T]:
        """Stream values matching the supplied filters one at a time."""
        q: Query | None = None
        if filter_data or filter_metadata:
            q = _build_query(
                filter_data=filter_data, filter_metadata=filter_metadata
            )
        async for record in self._db.stream_read(q, config):
            yield self._deserializer(record)

    async def search(self, query: Query) -> Sequence[Record]:
        """Escape hatch for ``Query`` / ``ComplexQuery`` / vector scores.

        Returns raw ``Record`` instances rather than ``T`` so that scored
        results from vector search remain accessible.  Consumers that
        need typed values should call ``list(...)`` instead.

        Typed as ``Sequence[Record]`` (rather than ``list[Record]``)
        because the ``list`` method on this class would otherwise shadow
        the builtin in this annotation context.
        """
        return await self._db.search(query)


class SyncKeyedRecordStore(Generic[T]):
    """Generic id-keyed record store over ``SyncDatabase``.

    Mirrors ``AsyncKeyedRecordStore`` for synchronous use.  Same
    serializer / deserializer contract.
    """

    def __init__(
        self,
        db: SyncDatabase,
        *,
        serializer: Callable[[T], tuple[dict[str, Any], dict[str, Any]]],
        deserializer: Callable[[Record], T],
    ) -> None:
        """See ``AsyncKeyedRecordStore.__init__``."""
        self._db = db
        self._serializer = serializer
        self._deserializer = deserializer

    @property
    def db(self) -> SyncDatabase:
        """See ``AsyncKeyedRecordStore.db``."""
        return self._db

    def put(self, key: str, value: T) -> None:
        """See ``AsyncKeyedRecordStore.put``."""
        record = _build_record(key, value, self._serializer)
        self._db.upsert(key, record)

    def get(self, key: str) -> T | None:
        """See ``AsyncKeyedRecordStore.get``."""
        record = self._db.read(key)
        if record is None:
            return None
        return self._deserializer(record)

    def exists(self, key: str) -> bool:
        """See ``AsyncKeyedRecordStore.exists``."""
        return self._db.exists(key)

    def delete(self, key: str) -> bool:
        """See ``AsyncKeyedRecordStore.delete``."""
        return self._db.delete(key)

    def put_batch(self, items: Mapping[str, T]) -> None:
        """See ``AsyncKeyedRecordStore.put_batch``."""
        for k, v in items.items():
            self.put(k, v)

    def get_batch(self, keys: Sequence[str]) -> list[T | None]:
        """See ``AsyncKeyedRecordStore.get_batch``."""
        records = self._db.read_batch(list(keys))
        return [self._deserializer(r) if r is not None else None for r in records]

    def delete_batch(self, keys: Sequence[str]) -> int:
        """See ``AsyncKeyedRecordStore.delete_batch``."""
        results = self._db.delete_batch(list(keys))
        return sum(1 for r in results if r)

    def clear(self) -> int:
        """See ``AsyncKeyedRecordStore.clear``."""
        records = self._db.all()
        keys = [r.storage_id for r in records if r.storage_id]
        if not keys:
            return 0
        return self.delete_batch(keys)

    def list(
        self,
        *,
        filter_data: Mapping[str, Any] | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        vector_query: VectorQuery | None = None,
    ) -> list[T]:
        """See ``AsyncKeyedRecordStore.list``."""
        q = _build_query(
            filter_data=filter_data,
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
            vector_query=vector_query,
        )
        records = self._db.search(q)
        return [self._deserializer(r) for r in records]

    def count(
        self,
        *,
        filter_data: Mapping[str, Any] | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """See ``AsyncKeyedRecordStore.count``."""
        if not filter_data and not filter_metadata:
            return self._db.count()
        q = _build_query(filter_data=filter_data, filter_metadata=filter_metadata)
        return self._db.count(q)

    def stream(
        self,
        *,
        filter_data: Mapping[str, Any] | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        config: StreamConfig | None = None,
    ) -> Iterator[T]:
        """See ``AsyncKeyedRecordStore.stream``."""
        q: Query | None = None
        if filter_data or filter_metadata:
            q = _build_query(
                filter_data=filter_data, filter_metadata=filter_metadata
            )
        for record in self._db.stream_read(q, config):
            yield self._deserializer(record)

    def search(self, query: Query) -> Sequence[Record]:
        """See ``AsyncKeyedRecordStore.search``."""
        return self._db.search(query)


__all__ = ["AsyncKeyedRecordStore", "SyncKeyedRecordStore"]

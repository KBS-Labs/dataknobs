# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Async SQLite backend implementation using aiosqlite."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import aiosqlite
from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import AsyncDatabase, enforce_content_version
from ..exceptions import DuplicateRecordError
from ..query import Query
from ..query_logic import ComplexQuery
from ..vector import AsyncVectorOperationsMixin
from ..vector.bulk_embed_mixin import AsyncBulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .config import AsyncSQLiteDatabaseConfig
from .sql_base import (
    SQLTableManager,
    constraint_violation_error,
    is_duplicate_key_error,
)
from .sqlite_mixins import (
    REGEXP_FUNCTION,
    SQLiteLayoutMixin,
    SQLiteVectorSupport,
    sqlite_max_parameters,
    sqlite_regexp,
)
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import ClassVar

    import numpy as np

    from ..records import Record
    from ..streaming import StreamConfig, StreamResult
    from ..vector.types import DistanceMetric, VectorSearchResult


logger = logging.getLogger(__name__)

#: The stores whose operation the running task is inside: a statement an
#: operation makes through another public method is part of it, not a new one.
_OPERATING: ContextVar[tuple[AsyncSQLiteDatabase, ...]] = ContextVar(
    "sqlite_async_operating", default=()
)


class AsyncSQLiteDatabase(
    StructuredConfigConsumer[AsyncSQLiteDatabaseConfig],
    SQLiteLayoutMixin,
    AsyncDatabase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,  # Provides python_vector_search_async
    AsyncBulkEmbedMixin,  # Must come before AsyncVectorOperationsMixin to override bulk_embed_and_store
    AsyncVectorOperationsMixin,
):
    """Asynchronous SQLite database backend using aiosqlite.

    Constructed through :class:`AsyncSQLiteDatabaseConfig` — every
    documented config key is a typed field on that dataclass, so
    ``self.config`` is the typed config (not a dict) and the
    ``from_config`` / factory paths share one construction route.

    With ``layout: native`` it reads a table in somebody else's file through
    the table's own columns, opening the file read-only (see
    :class:`~dataknobs_data.backends.config.SQLiteDatabaseConfigBase`).
    """

    CONFIG_CLS: ClassVar[type[AsyncSQLiteDatabaseConfig]] = AsyncSQLiteDatabaseConfig

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op for SQLite — connection setup is
        deferred to :meth:`connect`). ``journal_mode`` defaults to
        ``"WAL"`` for file-based databases; that default depends on the
        resolved ``path`` so it is computed here rather than as a static
        config field default. A native table's file is somebody else's, and
        its journal mode is theirs: none is set.
        """
        cfg = self.config
        self.db_path = cfg.path
        self.table_name = cfg.table
        self.timeout = cfg.timeout
        self.synchronous = cfg.synchronous
        self.pool_size = cfg.pool_size
        self.auto_create_table = cfg.creates_table

        self.table_manager = SQLTableManager(self.table_name, dialect="sqlite")
        # The one query builder, made with the table's layout. It needs no
        # connection, so a native configuration the layout refuses fails here.
        self._setup_layout()
        self._connect_to = self._connect_target()
        self.journal_mode = (
            cfg.journal_mode
            if cfg.journal_mode is not None
            else ("WAL" if self.db_path != ":memory:" and not self.native else None)
        )

        self.db: aiosqlite.Connection | None = None
        self._connected = False
        # The operations running on the connection, which ``close`` waits for.
        self._running = 0
        self._idle = asyncio.Event()
        self._idle.set()

        # Serializes conditional (compare-and-set) writes so a concurrent pair
        # on one instance yields exactly one winner. aiosqlite queues each
        # statement independently, so without this the read and the write of a
        # conditional update could interleave across coroutines.
        self._cas_lock = asyncio.Lock()

        # Initialize vector support
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)
        self._init_vector_state()

    async def connect(self) -> None:
        """Connect to the SQLite database."""
        if self._connected:
            return

        # Off the loop. A native table is in somebody else's file, opened
        # read-only, never made.
        directory = self._directory_to_make()
        if directory is not None:
            await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)

        target, uri = self._connect_to
        with self._refusing_unopened(sqlite3.OperationalError):
            self.db = await aiosqlite.connect(target, timeout=self.timeout, uri=uri)
        await self.db.create_function(*REGEXP_FUNCTION, sqlite_regexp, deterministic=True)

        # Enable row factory for dict-like access
        self.db.row_factory = aiosqlite.Row

        try:
            # The first statements to read the file: on a native table, what
            # fails here is reading it -- a lock its owner holds, a WAL file's
            # side files that cannot be made.
            with self._refusing_unopened(sqlite3.OperationalError):
                await self._configure_sqlite()
                # Create table if it doesn't exist
                await self._ensure_table()
        except BaseException:
            # Refused after opening -- a native table that is not there, a file
            # that cannot be written -- so close what was opened: its worker
            # thread would otherwise outlive the refusal and the process.
            await self.db.close()
            self.db = None
            raise

        self._connected = True
        logger.info(f"Connected to async SQLite database: {self.db_path}")

    async def close(self) -> None:
        """Close the database connection.

        Refuses every operation from the moment it starts, and closes the
        connection once the operations already running on it are done. A
        transaction is not one operation: closing in its body discards it, and
        its commit is refused.
        """
        was_connected, self._connected = self._connected, False
        while self._running:
            await self._idle.wait()
        db, self.db = self.db, None
        if db is not None:
            await db.close()
        if was_connected:
            logger.info(f"Disconnected from async SQLite database: {self.db_path}")

    async def _configure_sqlite(self) -> None:
        """Configure SQLite settings for performance."""
        if not self.db:
            return

        # Set journal mode if specified
        if self.journal_mode:
            await self.db.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            logger.debug(f"Set journal_mode to {self.journal_mode}")

        # Set synchronous mode
        await self.db.execute(f"PRAGMA synchronous = {self.synchronous}")
        logger.debug(f"Set synchronous to {self.synchronous}")

        # Enable foreign keys
        await self.db.execute("PRAGMA foreign_keys = ON")

        # Optimize for performance
        await self.db.execute("PRAGMA temp_store = MEMORY")
        await self.db.execute("PRAGMA mmap_size = 30000000000")

        await self.db.commit()

    async def _ensure_table(self) -> None:
        """Ensure the table exists.

        When ``auto_create_table=True`` (default), runs ``CREATE TABLE IF NOT
        EXISTS …``. When ``auto_create_table=False``, verifies the table is
        present and raises ``RuntimeError`` if it isn't.
        """
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

        if not self.auto_create_table:
            exists_sql, params = self._relation_exists_query()
            async with self.db.execute(exists_sql, params) as cursor:
                row = await cursor.fetchone()
                exists = bool(row[0]) if row else False
            if not exists:
                raise self._missing_relation_error()
            return

        await self.db.executescript(self.table_manager.get_create_table_sql())
        await self.db.commit()

    def _check_connection(self) -> None:
        """Check if database is connected."""
        self._require_conn()

    def _require_conn(self) -> aiosqlite.Connection:
        """The connection, or the refusal :meth:`_check_connection` gives.

        The same test, returning what it tested: a caller that holds the
        result has the connection narrowed for the type checker, which a
        check in another method cannot give it.
        """
        if not self._connected or self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.db

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[aiosqlite.Connection]:
        """The connection one operation runs on, held open until it is done.

        Refused before :meth:`connect` and from the moment :meth:`close`
        starts. Once admitted, the operation is counted until it returns, and
        ``close`` closes the connection only when none is: an operation is
        several statements, each its own await, and ``close`` may start between
        two of them. An operation another one makes -- a conditional update
        reading the record it compares -- is part of that one, and is admitted
        whatever ``close`` has started.
        """
        operating = _OPERATING.get()
        if any(store is self for store in operating):
            if self.db is None:  # the outer operation holds it open
                raise RuntimeError("Database not connected. Call connect() first.")
            yield self.db
            return
        db = self._require_conn()
        token = _OPERATING.set((*operating, self))
        try:
            async with self._counted():
                yield db
        finally:
            _OPERATING.reset(token)

    @asynccontextmanager
    async def _counted(self) -> AsyncIterator[None]:
        """Count what runs inside among the operations :meth:`close` waits for."""
        self._running += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._running -= 1
            if not self._running:
                self._idle.set()

    async def create(self, record: Record) -> str:
        """Create a new record."""
        async with self._operation() as db:
            record_id = record.id or self._generate_id()
            query, params = self.query_builder.build_create_query(record, record_id=record_id)

            try:
                await db.execute(query, params)
                await db.commit()

                # SQLite doesn't support RETURNING, so we use the ID we generated
                return record_id
            except aiosqlite.IntegrityError as e:
                await db.rollback()
                if is_duplicate_key_error(e):
                    raise DuplicateRecordError(params[0]) from e
                # NOT NULL / CHECK / other column constraint — surface truthfully
                # instead of mislabeling it as a duplicate id.
                raise constraint_violation_error(params[0]) from e

    async def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        async with self._operation() as db:
            query, params = self.query_builder.build_read_query(id)

            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()

                if row:
                    return self.query_builder.record_from_row(dict(row))
                return None

    async def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional optimistic-concurrency token from
                ``get_version(id)`` (a content hash for SQLite). When provided,
                a stale token raises ``ConcurrencyError`` instead of
                overwriting. When ``None`` the update is unconditional,
                byte-identical to prior behavior.

        Returns:
            True if the record was updated, False if no record with the given ID exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        async with self._operation() as db:
            # Conditional write: hold the CAS lock across the read-compare-write so
            # a concurrent conditional pair on this instance yields exactly one
            # winner. Reusing read() guarantees the token compared here matches the
            # one get_version() returned. Cross-connection atomicity is out of
            # scope (see the module docs on the in-process content-hash backends).
            if expected_version is not None:
                async with self._cas_lock:
                    current = await self.read(id)
                    if current is None:
                        return False
                    enforce_content_version(id, expected_version, current)
                    query, params = self.query_builder.build_update_query(id, record)
                    cursor = await db.execute(query, params)
                    await db.commit()
                    return cursor.rowcount > 0

            query, params = self.query_builder.build_update_query(id, record)

            cursor = await db.execute(query, params)
            await db.commit()
            rows_affected = cursor.rowcount

            if rows_affected == 0:
                logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")

            return rows_affected > 0

    async def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        When ``expected_version`` is provided the read-compare-delete runs
        under the CAS lock so a concurrent conditional pair on this instance
        yields exactly one winner; a stale token raises ``ConcurrencyError``
        and a missing record returns ``False``. Cross-connection atomicity is
        out of scope (see the module docs on the in-process content-hash
        backends). When ``None`` the delete is unconditional, byte-identical to
        prior behavior.
        """
        async with self._operation() as db:
            if expected_version is not None:
                async with self._cas_lock:
                    current = await self.read(id)
                    if current is None:
                        return False
                    enforce_content_version(id, expected_version, current)
                    query, params = self.query_builder.build_delete_query(id)
                    cursor = await db.execute(query, params)
                    await db.commit()
                    return cursor.rowcount > 0

            query, params = self.query_builder.build_delete_query(id)

            cursor = await db.execute(query, params)
            await db.commit()
            return cursor.rowcount > 0

    async def exists(self, id: str) -> bool:
        """Check if a record exists."""
        async with self._operation() as db:
            query, params = self.query_builder.build_exists_query(id)

            async with db.execute(query, params) as cursor:
                result = await cursor.fetchone()
                return result is not None

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query."""
        async with self._operation() as db:
            # Handle ComplexQuery with native SQL support
            if isinstance(query, ComplexQuery):
                sql_query, params = self.query_builder.build_complex_search_query(query)
            else:
                sql_query, params = self.query_builder.build_search_query(query)

            async with db.execute(sql_query, params) as cursor:
                rows = await cursor.fetchall()

                records = [self.query_builder.record_from_row(dict(row)) for row in rows]

                # Apply field projection if specified
                if query.fields:
                    records = [r.project(query.fields) for r in records]

                return records

    async def count(self, query: Query | None = None) -> int:
        """Count records matching a query."""
        async with self._operation() as db:
            sql_query, params = self.query_builder.build_count_query(query)

            async with db.execute(sql_query, params) as cursor:
                result = await cursor.fetchone()
                return result[0] if result else 0

    def supports_transactions(self) -> bool:
        """SQLite batch ops run inside an explicit ``BEGIN``/``COMMIT``."""
        return True

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Open one native transaction on the shared aiosqlite connection.

        Issues a single ``BEGIN TRANSACTION`` and yields the connection as
        the handle; the batch methods run their DML on it and skip their own
        ``BEGIN``/``commit`` when a handle is threaded (``_tx is not None``), so
        a multi-kind buffered-transaction flush commits (or rolls back) as one
        unit. Concurrency: the connection is single, so — as the module docs
        note — two buffered-transaction commits must not run against this
        instance concurrently; the ``BEGIN``/``COMMIT`` boundaries would
        interleave.
        """
        async with self._operation() as db:
            await db.execute("BEGIN TRANSACTION")
        try:
            yield db
            async with self._operation() as current:
                await current.commit()
        except BaseException:
            await self._rollback_if_open()
            raise

    async def _rollback_if_open(self) -> None:
        """Roll back the open transaction, unless :meth:`close` took the connection.

        Closing the connection discarded the transaction, and a refusal here
        would only bury the error that brought the rollback about. Not refused
        while ``close`` waits, which then waits for it too.
        """
        db = self.db
        if db is None:
            return
        async with self._counted():
            await db.rollback()

    async def _existing_ids(self, db: aiosqlite.Connection, ids: list[str]) -> set[str]:
        """Which of ``ids`` are stored, asked on ``db`` in one statement whatever their number."""
        if not ids:
            return set()
        query, params = self.query_builder.build_existing_ids_query(ids)
        async with db.execute(query, params) as cursor:
            return {row[0] for row in await cursor.fetchall()}

    async def create_batch(self, records: list[Record], *, _tx: Any = None) -> list[str]:
        """Create multiple records efficiently, in one transaction.

        Uses multi-value INSERTs, as many as SQLite's parameter limit
        requires. Like ``create()``, this fails closed: a colliding id (or a
        duplicate id within the batch) raises ``DuplicateRecordError`` and the
        transaction is rolled back so nothing is written. A caller-supplied
        ``record.id`` is honored (the shared query builder mints a uuid only
        when a record has none).

        When ``_tx`` is supplied (a multi-kind buffered-transaction flush), the
        DML joins that outer transaction and this method skips its own
        ``BEGIN``/``commit``/``rollback`` — the outer :meth:`_transaction` owns
        the boundary.
        """
        if not records:
            return []

        async with self._operation() as db:
            # Use the shared batch create query builder (honors record.id, mints via
            # _generate_id; raises DuplicateRecordError up front on a within-batch
            # duplicate id).
            statements, ids = self.query_builder.build_batch_create_queries(
                records, id_factory=self._generate_id, max_parameters=sqlite_max_parameters()
            )

            own_tx = _tx is None
            if own_tx:
                await db.execute("BEGIN TRANSACTION")
            else:
                # Inside a wider transaction a failed statement is undone alone, so
                # the rows this batch's earlier statements wrote stay visible and a
                # lookup after the failure could name one of them. Ask first.
                stored = await self._existing_ids(db, [r.id for r in records if r.id])
                if stored:
                    raise DuplicateRecordError(next(r.id for r in records if r.id in stored))

            try:
                for query, params in statements:
                    await db.execute(query, params)
                if own_tx:
                    await db.commit()
                return ids
            except aiosqlite.IntegrityError as e:
                if own_tx:
                    await db.rollback()
                if is_duplicate_key_error(e):
                    colliding = ids[0]
                    # Name the colliding id precisely on the error path only, once
                    # the rollback has undone everything this batch wrote.
                    if own_tx:
                        stored = await self._existing_ids(db, [r.id for r in records if r.id])
                        colliding = next((r.id for r in records if r.id in stored), colliding)
                    raise DuplicateRecordError(colliding) from e
                raise constraint_violation_error() from e
            except Exception:
                if own_tx:
                    await db.rollback()
                raise

    async def upsert_batch(self, records: list[Record], *, _tx: Any = None) -> list[str]:
        """Insert-or-overwrite multiple records efficiently, in one transaction.

        Uses ``INSERT ... ON CONFLICT (id) DO UPDATE``, as many statements as
        SQLite's parameter limit requires. Honors a caller-supplied
        ``record.id`` (minting a uuid only when absent); a colliding id is
        overwritten (never raised). Returns ids in input order.

        When ``_tx`` is supplied the DML joins that outer transaction and this
        method skips its own boundary (see :meth:`create_batch`).
        """
        if not records:
            return []

        async with self._operation() as db:
            statements, ids = self.query_builder.build_batch_upsert_queries(
                records, id_factory=self._generate_id, max_parameters=sqlite_max_parameters()
            )

            own_tx = _tx is None
            if own_tx:
                await db.execute("BEGIN TRANSACTION")
            try:
                for query, params in statements:
                    await db.execute(query, params)
                if own_tx:
                    await db.commit()
                return ids
            except Exception:
                if own_tx:
                    await db.rollback()
                raise

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently, in one transaction.

        Each record takes its own update; a repeated id takes its last. An id
        not stored is reported ``False`` and written nowhere.
        """
        if not updates:
            return []

        async with self._operation() as db:
            # One statement per update rather than a join: UPDATE … FROM needs
            # SQLite 3.33, and executemany binds three values per run.
            query, rows = self.query_builder.build_batch_update_rows(updates)

            await db.execute("BEGIN TRANSACTION")
            try:
                await db.executemany(query, rows)
                await db.commit()

                # SQLite's UPDATE returns nothing here, so ask which ids exist.
                existing_ids = await self._existing_ids(db, [record_id for record_id, _ in updates])
                return [record_id in existing_ids for record_id, _ in updates]
            except Exception:
                await db.rollback()
                raise

    async def delete_batch(self, ids: list[str], *, _tx: Any = None) -> list[bool]:
        """Delete multiple records efficiently using a single query.

        Uses a single DELETE, whatever the number of ids.

        When ``_tx`` is supplied the DML joins that outer transaction and this
        method skips its own boundary (see :meth:`create_batch`).
        """
        if not ids:
            return []

        async with self._operation() as db:
            # Check which IDs exist before deletion
            existing_ids = await self._existing_ids(db, ids)

            query, params = self.query_builder.build_batch_delete_query(ids)

            own_tx = _tx is None
            if own_tx:
                await db.execute("BEGIN TRANSACTION")

            try:
                await db.execute(query, params)
                if own_tx:
                    await db.commit()
                return [id in existing_ids for id in ids]
            except Exception:
                if own_tx:
                    await db.rollback()
                raise

    def _initialize(self) -> None:
        """Initialize method - connection setup handled in connect()."""
        pass

    async def _count_all(self) -> int:
        """Count all records in the database: every row of a native table's scope."""
        return await self.count()

    async def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream the records a query matches, a page of ``search`` at a time.

        Each page is its own statement, sorted by the query's sort and then by
        the key (see :func:`~dataknobs_data.streaming.stream_page`), so no
        statement stays open between records: an open SQLite read would lock
        the file's owner out of writing it. The query's limit, offset and
        projection hold; with no sort the stream promises no order.
        """
        from ..streaming import aiter_search_pages

        async for record in aiter_search_pages(self.search, query, config):
            yield record

    async def stream_write(
        self, records: AsyncIterator[Record], config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database.

        Honors ``config.on_conflict`` via the shared conflict resolver: INSERT
        uses the ``create_batch`` bulk fast-path with a per-record ``create``
        fallback (so a colliding id fails closed and is attributed as a failure,
        not silently overwritten); UPSERT uses ``upsert_batch``; SKIP writes
        per-record via ``create`` and counts duplicates as skips.
        """
        from ..streaming import (
            StreamConfig,
            async_run_stream_write,
            resolve_conflict_write,
        )

        config = config or StreamConfig()

        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            insert_batch_func=self.create_batch,
            single_create_func=self.create,
            upsert_func=self.upsert,
            upsert_batch_func=self.upsert_batch,
        )
        return await async_run_stream_write(
            records,
            batch_write_func=batch_write_func,
            single_write_func=single_write_func,
            skip_on_duplicate=skip_on_duplicate,
            config=config,
        )

    async def _vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str,
        k: int,
        metric: DistanceMetric,
        filter: Query | None,
    ) -> list[VectorSearchResult]:
        """Raw k-NN over every record, in Python.

        SQLite has no vector operators, so the similarity is computed here
        rather than in the query.
        """
        async with self._operation():
            return await self.python_vector_search_async(
                query_vector=query_vector,
                vector_field=vector_field,
                k=k,
                filter=filter,
                metric=metric,
            )

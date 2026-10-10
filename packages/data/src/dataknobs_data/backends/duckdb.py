# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""DuckDB backend implementation for analytical workloads.

DuckDB is an embedded columnar database optimized for analytics,
providing 10-100x performance improvement over SQLite for
aggregations, joins, and analytical queries.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

import duckdb
from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import AsyncDatabase, SyncDatabase, enforce_content_version
from ..exceptions import DuplicateRecordError
from ..query import Query
from ..query_logic import ComplexQuery
from .config import AsyncDuckDBDatabaseConfig, SyncDuckDBDatabaseConfig
from .layout_backend import FileLayoutMixin
from .sql_base import (
    SQLQueryBuilder,
    SQLRecordSerializer,
    SQLTableManager,
    constraint_violation_error,
    is_duplicate_key_error,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from typing import ClassVar

    from ..records import Record
    from ..streaming import StreamConfig, StreamResult


logger = logging.getLogger(__name__)


def _existing_ids(
    conn: duckdb.DuckDBPyConnection, builder: SQLQueryBuilder, ids: list[str]
) -> set[str]:
    """Which of ``ids`` are stored, asked in one statement on ``conn``."""
    if not ids:
        return set()
    query, params = builder.build_existing_ids_query(ids)
    return {row[0] for row in conn.execute(query, params).fetchall()}


class DuckDBLayoutMixin(FileLayoutMixin):
    """What reading a table through a column layout means on DuckDB, for both twins.

    A native table is in somebody else's file, which is opened ``read_only``:
    DuckDB then writes nothing to it and creates no file that is not there.
    DuckDB's table lookup lists views, so one is read in place.

    **A native store holds no connection between reads.** DuckDB locks a file
    for as long as any connection holds it, and a read-only connection's lock
    refuses every process that opens the file for writing -- the owner's
    included. So :meth:`connect` opens the file to check the table and closes
    it, and each read opens the file for its own statement.

    DuckDB lets one process write a file, or any number read it, never both,
    and it waits for neither: a lock it cannot take fails at once. So the owner
    is kept out while a statement runs, and is refused if it opens the file
    then; a read is refused as long as the owner holds the file open for
    writing. An owner that keeps one connection open for its whole life keeps
    every native read out for that long. Either side retries.

    Each read pays for opening the file. Reads on separate connections need no
    lock between them, so they run side by side on the async twin's pool.
    """

    _DIALECT: ClassVar[str] = "duckdb"
    _PARAM_STYLE: ClassVar[str] = "qmark"
    _READ_ONLY_NEEDS: ClassVar[str] = (
        "DuckDB also refuses a read-only connection to a file another connection holds "
        "open for writing."
    )
    _HELD_WHILE: ClassVar[str] = (
        "DuckDB opens no file another process holds open for writing, and does not wait for it"
    )

    read_only: bool
    auto_create_table: bool
    conn: duckdb.DuckDBPyConnection | None
    _connected: bool
    _lock: threading.Lock

    def _open(self) -> duckdb.DuckDBPyConnection:
        """Open the file as configured; refuse by name a native table's that will not open."""
        with self._refusing_unopened(duckdb.IOException, duckdb.ConnectionException):
            return duckdb.connect(self.db_path, read_only=self.read_only)

    def _file_refusal(self, error: Exception) -> Exception | None:
        """A file that opened at connect and will not open for a read is held by its owner.

        DuckDB raises one ``IOException`` for a file that is not there and for
        one whose lock it cannot take, so when it fails decides which it is.
        At connect it may be either, and the refusal says both.
        """
        if self._connected:
            return self._held_file_error(error)
        return self._unopened_file_error(error)

    def _connect_file(self) -> duckdb.DuckDBPyConnection | None:
        """Open the file and check the table: the connection to hold, or ``None`` under native.

        Blocking -- a directory made, a file opened -- so the async twin runs
        it on its pool. A file whose table is refused is closed before the
        refusal reaches the caller.
        """
        directory = self._directory_to_make()
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        conn = self._open()
        try:
            self._check_relation(conn)
        except BaseException:
            conn.close()
            raise
        if self.native:
            conn.close()
            return None
        return conn

    def _check_relation(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Ensure the table is there, or that it may be created.

        When ``auto_create_table`` is on (the JSON layout's default), creates
        the table; off, verifies it exists and raises ``RuntimeError`` if it is
        missing. Under ``read_only`` with the JSON layout the check is skipped:
        no DDL is meaningful in read-only mode, and the existence check is
        skipped too, so do not rely on ``auto_create_table=False`` to detect a
        missing table there. A native table is always read-only, and is always
        checked, so a table that is not there is named on connect.
        """
        if self.read_only and not self.native:
            if not self.auto_create_table:
                logger.warning(
                    "auto_create_table=False has no effect when read_only=True — "
                    "the table existence check is skipped in read-only mode."
                )
            return
        if not self.auto_create_table:
            exists_sql, params = self._relation_exists_query()
            row = conn.execute(exists_sql, list(params)).fetchone()
            if not (row and row[0]):
                raise self._missing_relation_error()
            return
        conn.execute(self.table_manager.get_create_table_sql())

    def _check_connection(self) -> None:
        """Refuse an operation before :meth:`connect`."""
        if not self._connected:
            raise RuntimeError("Database not connected. Call connect() first.")

    def _require_conn(self) -> duckdb.DuckDBPyConnection:
        """The connection the store holds, or the refusal :meth:`_check_connection` gives.

        A caller that holds the result has the connection narrowed for the
        type checker, which a check in another method cannot give it. A native
        store holds none; its reads go through :meth:`_reading`.
        """
        if not self._connected or self.conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn

    @contextmanager
    def _reading(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """The connection a read runs on.

        The store's own, under the lock every statement on it holds; under
        native, the file opened for this read alone, which needs none.
        """
        if not self.native:
            with self._lock:
                yield self._require_conn()
            return
        self._check_connection()
        conn = self._open()
        try:
            yield conn
        finally:
            conn.close()

    def _close_held(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Close the store's connection once no statement runs on it."""
        with self._lock:
            conn.close()

    def _rows(self, sql: str, params: Any) -> list[dict[str, Any]]:
        """Run one read, every row as a column-to-value mapping."""
        with self._reading() as conn:
            result = conn.execute(sql, params)
            names = [column[0] for column in result.description]
            return [dict(zip(names, row, strict=True)) for row in result.fetchall()]

    def _read_rows(self, id: str) -> Record | None:
        """The record stored under ``id``, or ``None``: the body both twins' ``read`` run."""
        rows = self._rows(*self.query_builder.build_read_query(id))
        return self.query_builder.record_from_row(rows[0]) if rows else None

    def _exists_rows(self, id: str) -> bool:
        """Whether a record is stored under ``id``: the body both twins' ``exists`` run."""
        return bool(self._rows(*self.query_builder.build_exists_query(id)))

    def _search_rows(self, query: Query | ComplexQuery) -> list[Record]:
        """The records ``query`` matches: the body both twins' ``search`` run."""
        if isinstance(query, ComplexQuery):
            sql, params = self.query_builder.build_complex_search_query(query)
        else:
            sql, params = self.query_builder.build_search_query(query)
        return self.query_builder.records_from_rows(self._rows(sql, params), query)

    def _page_rows(
        self, page: Query, after: Sequence[Any] | None
    ) -> tuple[list[Record], list[Any] | None]:
        """One page of a stream: the body both twins' page reads run."""
        sql, params, keys = self.query_builder.build_page_query(page, after)
        return self.query_builder.page_records(self._rows(sql, params), page, keys)

    def _count_rows(self, query: Query | None = None) -> int:
        """How many records ``query`` matches: the body both twins' ``count`` run."""
        rows = self._rows(*self.query_builder.build_count_query(query))
        return int(next(iter(rows[0].values()))) if rows else 0


class AsyncDuckDBDatabase(
    StructuredConfigConsumer[AsyncDuckDBDatabaseConfig],
    DuckDBLayoutMixin,
    AsyncDatabase,
):
    """Asynchronous DuckDB database backend for analytical workloads.

    DuckDB is an embedded columnar database optimized for analytics.
    Provides 10-100x performance improvement over SQLite for
    aggregations, joins, and analytical queries.

    Features:
    - Columnar storage for fast analytical queries
    - Parallel execution for multi-threaded query processing
    - Native Parquet integration for efficient data import/export
    - Advanced analytics support (window functions, CTEs, complex aggregations)

    Usage:
        ```python
        from dataknobs_data import async_database_factory

        # File-based database
        db = async_database_factory("duckdb:///path/to/data.duckdb")

        # In-memory database
        db = async_database_factory("duckdb:///:memory:")

        async with db:
            # Perform CRUD operations
            await db.create(record)
            results = await db.search(query)
        ```
    """

    CONFIG_CLS: ClassVar[type[AsyncDuckDBDatabaseConfig]] = AsyncDuckDBDatabaseConfig

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op — connection setup is deferred to
        :meth:`connect`).
        """
        cfg = self.config
        self.db_path = cfg.path
        self.table_name = cfg.table
        self.timeout = cfg.timeout
        self.max_workers = cfg.max_workers
        self.read_only = cfg.opens_read_only
        self.auto_create_table = cfg.creates_table

        # Thread pool for async operations (DuckDB has no native async support)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.serializer = SQLRecordSerializer()
        self.table_manager = SQLTableManager(self.table_name, dialect="duckdb")
        # The one query builder, made with the table's layout. It needs no
        # connection, so a native configuration the layout refuses fails here.
        self._setup_layout()

        self.conn: duckdb.DuckDBPyConnection | None = None
        self._connected = False
        self._lock = threading.Lock()  # Thread safety lock for DuckDB connection

    async def connect(self) -> None:
        """Connect to the DuckDB database: the file is opened on the pool, off the loop."""
        if self._connected:
            return
        loop = asyncio.get_running_loop()
        try:
            self.conn = await loop.run_in_executor(self.executor, self._connect_file)
        except BaseException:
            # Refused -- a file that will not open, a table that is not there:
            # the pool's thread would outlive the refusal, so replace the pool.
            await self._replace_executor()
            raise
        self._connected = True
        logger.info(f"Connected to async DuckDB database: {self.db_path}")

    async def _replace_executor(self) -> None:
        """Shut the pool down off the loop, and leave a fresh one, which starts no thread until used."""
        executor = self.executor
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        await asyncio.to_thread(executor.shutdown, wait=True)

    async def close(self) -> None:
        """Close the database connection, and the pool's threads.

        Refuses every operation from the moment it starts, and closes the
        connection once the statement running on it, if any, is done.
        """
        conn, self.conn = self.conn, None
        was_connected, self._connected = self._connected, False
        if conn is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self._close_held, conn)
        if was_connected:
            logger.info(f"Disconnected from async DuckDB database: {self.db_path}")
        await self._replace_executor()

    async def create(self, record: Record) -> str:
        """Create a new record.

        Args:
            record: The record to create

        Returns:
            The record ID
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._create_sync, record)

    def _create_sync(self, record: Record) -> str:
        """Synchronous create implementation."""
        record_id = record.id or self._generate_id()
        query, params = self.query_builder.build_create_query(record, record_id=record_id)

        try:
            with self._lock:
                self._require_conn().execute(query, params)
            # DuckDB doesn't support RETURNING, so we use the ID we generated
            return record_id
        except duckdb.ConstraintException as e:
            if is_duplicate_key_error(e):
                raise DuplicateRecordError(params[0]) from e
            # NOT NULL / CHECK / other column constraint — surface truthfully
            # instead of mislabeling it as a duplicate id.
            raise constraint_violation_error(params[0]) from e

    async def read(self, id: str) -> Record | None:
        """Read a record by ID.

        Args:
            id: The record ID

        Returns:
            The record if found, None otherwise
        """
        self._check_connection()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._read_rows, id)

    async def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional optimistic-concurrency token from
                ``get_version(id)`` (a content hash for DuckDB). When provided,
                the read-compare-write runs inside the connection lock so the
                compare-and-set is atomic within the connection; a stale token
                raises ``ConcurrencyError`` instead of overwriting. When
                ``None`` the update is unconditional, byte-identical to prior
                behavior.

        Returns:
            True if the record was updated, False if no record exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._update_sync,
            id,
            record,
            expected_version,
        )

    def _update_sync(self, id: str, record: Record, expected_version: str | None = None) -> bool:
        """Synchronous update implementation."""
        query, params = self.query_builder.build_update_query(id, record)

        with self._lock:
            conn = self._require_conn()
            if expected_version is not None:
                # Conditional write: read the current row and apply the update
                # under the connection lock so the compare-and-set is atomic.
                read_query, read_params = self.query_builder.build_read_query(id)
                result = conn.execute(read_query, read_params).fetchone()
                if result is None:
                    # Conditional update of an absent record is a documented
                    # False return, not a warning-worthy event (matches the
                    # SQLite backend's quiet miss).
                    return False
                columns = conn.description
                row_dict = {columns[i][0]: result[i] for i in range(len(columns))}
                current = self.query_builder.record_from_row(row_dict)
                enforce_content_version(id, expected_version, current)
                conn.execute(query, params)
                return True

            # Check if record exists
            exists_query, exists_params = self.query_builder.build_exists_query(id)
            exists = conn.execute(exists_query, exists_params).fetchone() is not None

            if exists:
                conn.execute(query, params)
                return True

            logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")
            return False

    async def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID
            expected_version: Optional content-hash token from
                ``get_version(id)``. When provided, the read-compare-delete
                runs under the connection lock so the compare-and-set is atomic
                within the connection; a stale token raises ``ConcurrencyError``
                and a missing record returns ``False``. When ``None`` the
                delete is unconditional, byte-identical to prior behavior.

        Returns:
            True if deleted, False if not found

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._delete_sync,
            id,
            expected_version,
        )

    def _delete_sync(self, id: str, expected_version: str | None = None) -> bool:
        """Synchronous delete implementation."""
        query, params = self.query_builder.build_delete_query(id)

        with self._lock:
            conn = self._require_conn()
            if expected_version is not None:
                # Conditional delete: read the current row and apply the delete
                # under the connection lock so the compare-and-set is atomic.
                read_query, read_params = self.query_builder.build_read_query(id)
                result = conn.execute(read_query, read_params).fetchone()
                if result is None:
                    return False
                columns = conn.description
                row_dict = {columns[i][0]: result[i] for i in range(len(columns))}
                current = self.query_builder.record_from_row(row_dict)
                enforce_content_version(id, expected_version, current)
                conn.execute(query, params)
                return True

            # First check if the record exists
            exists_query, exists_params = self.query_builder.build_exists_query(id)
            exists = conn.execute(exists_query, exists_params).fetchone() is not None

            if exists:
                conn.execute(query, params)
                return True
        return False

    async def exists(self, id: str) -> bool:
        """Check if a record exists.

        Args:
            id: The record ID

        Returns:
            True if exists, False otherwise
        """
        self._check_connection()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._exists_rows, id)

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query.

        Args:
            query: The query specification

        Returns:
            List of matching records
        """
        self._check_connection()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._search_rows, query)

    async def count(self, query: Query | None = None) -> int:
        """Count records matching a query.

        Args:
            query: Optional query specification

        Returns:
            Count of matching records
        """
        self._check_connection()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._count_rows, query)

    def supports_transactions(self) -> bool:
        """DuckDB batch ops run inside an explicit ``begin``/``commit``."""
        return True

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[duckdb.DuckDBPyConnection]:
        """Open one native transaction on the shared DuckDB connection.

        Runs the outer ``begin`` / ``commit`` / ``rollback`` on the executor
        under ``self._lock`` (matching how every op reaches the connection) and
        yields the connection the transaction began on as the handle. The batch sync cores run their DML
        under the same lock and skip their own ``begin``/``commit`` when a handle
        is threaded, so a multi-kind buffered-transaction flush commits (or rolls
        back) as one unit. As the module docs note, two buffered-transaction
        commits must not run against this instance concurrently.
        """
        self._check_connection()
        loop = asyncio.get_event_loop()
        conn = await loop.run_in_executor(self.executor, self._begin_sync)
        try:
            yield conn
            # Commit inside the ``try`` (not an ``else``) so a failure of the
            # commit itself rolls the connection back — otherwise the shared
            # connection is left with an open/aborted transaction and the next
            # ``begin()`` raises "cannot start a transaction within a
            # transaction". Mirrors the sqlite ``_transaction`` sibling.
            await loop.run_in_executor(self.executor, self._commit_sync)
        except BaseException:
            await loop.run_in_executor(self.executor, self._rollback_sync)
            raise

    def _begin_sync(self) -> duckdb.DuckDBPyConnection:
        """Begin a transaction on the connection, and return it (executor thread, under lock)."""
        with self._lock:
            conn = self._require_conn()
            conn.begin()
            return conn

    def _commit_sync(self) -> None:
        """Commit the connection's transaction (executor thread, under lock)."""
        with self._lock:
            self._require_conn().commit()

    def _rollback_sync(self) -> None:
        """Roll back the connection's transaction (executor thread, under lock).

        Nothing to do once the store is closed: closing the connection
        discarded the transaction, and a refusal here would only bury the
        error that brought the rollback about.
        """
        with self._lock:
            if self.conn is not None:
                self.conn.rollback()

    async def create_batch(self, records: list[Record], *, _tx: Any = None) -> list[str]:
        """Create multiple records efficiently.

        Args:
            records: List of records to create
            _tx: Internal. When supplied (a multi-kind buffered-transaction
                flush), the DML joins the outer :meth:`_transaction` and the sync
                core skips its own ``begin``/``commit``.

        Returns:
            List of record IDs
        """
        if not records:
            return []

        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._create_batch_sync,
            records,
            _tx is None,
        )

    def _create_batch_sync(self, records: list[Record], own_tx: bool = True) -> list[str]:
        """Synchronous batch create implementation.

        Fails closed like ``create()``: a colliding id (or a within-batch
        duplicate, raised up front by the shared query builder) raises
        ``DuplicateRecordError`` and the transaction is rolled back so nothing is
        written; a caller-supplied ``record.id`` is honored. When ``own_tx`` is
        ``False`` the DML runs inside an outer :meth:`_transaction` and this core
        skips its own ``begin``/``commit``/``rollback``.
        """
        # Use the shared batch create query builder
        statements, ids = self.query_builder.build_batch_create_queries(
            records, id_factory=self._generate_id
        )

        # Execute the batch insert in a transaction
        with self._lock:
            conn = self._require_conn()
            try:
                if own_tx:
                    conn.begin()
                else:
                    # The probe after a failure cannot run inside a wider
                    # transaction (see below), so ask before writing instead.
                    stored = _existing_ids(
                        conn, self.query_builder, [r.id for r in records if r.id]
                    )
                    if stored:
                        raise DuplicateRecordError(next(r.id for r in records if r.id in stored))
                for query, params in statements:
                    conn.execute(query, params)
                if own_tx:
                    conn.commit()
                return ids
            except duckdb.ConstraintException as e:
                if own_tx:
                    conn.rollback()
                if is_duplicate_key_error(e):
                    colliding = ids[0]
                    # Precise colliding-id naming needs a read probe, but DuckDB
                    # (like Postgres) aborts the whole transaction on a
                    # constraint violation. Probe only on the owned path, where
                    # we just rolled back and the connection is queryable again;
                    # on the multi-kind flush path (own_tx=False) the transaction
                    # is still aborted — a probe would raise
                    # ``duckdb.TransactionException`` and mask the
                    # ``DuplicateRecordError`` — so that path asked before
                    # writing, and a collision it did not see is reported as the
                    # first batch id while the outer ``_transaction`` rolls the
                    # whole flush back.
                    # We are on the executor thread holding the lock, so probe
                    # the raw connection directly rather than the async exists()
                    # coroutine.
                    if own_tx:
                        stored = _existing_ids(
                            conn,
                            self.query_builder,
                            [r.id for r in records if r.id],
                        )
                        colliding = next((r.id for r in records if r.id in stored), colliding)
                    raise DuplicateRecordError(colliding) from e
                raise constraint_violation_error() from e
            except Exception:
                if own_tx:
                    conn.rollback()
                raise

    async def upsert_batch(self, records: list[Record], *, _tx: Any = None) -> list[str]:
        """Insert-or-overwrite multiple records efficiently in one statement.

        Uses ``INSERT ... ON CONFLICT (id) DO UPDATE``. Honors a caller-supplied
        ``record.id`` (minting a uuid only when absent); a colliding id is
        overwritten (never raised). Returns ids in input order. When ``_tx`` is
        supplied the DML joins the outer :meth:`_transaction` (see
        :meth:`create_batch`).
        """
        if not records:
            return []

        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._upsert_batch_sync,
            records,
            _tx is None,
        )

    def _upsert_batch_sync(self, records: list[Record], own_tx: bool = True) -> list[str]:
        """Synchronous batch upsert implementation."""
        statements, ids = self.query_builder.build_batch_upsert_queries(
            records, id_factory=self._generate_id
        )

        with self._lock:
            conn = self._require_conn()
            try:
                if own_tx:
                    conn.begin()
                for query, params in statements:
                    conn.execute(query, params)
                if own_tx:
                    conn.commit()
                return ids
            except Exception:
                if own_tx:
                    conn.rollback()
                raise

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently.

        Args:
            updates: List of (record_id, record) tuples

        Returns:
            List of success indicators
        """
        if not updates:
            return []

        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._update_batch_sync, updates)

    def _update_batch_sync(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Synchronous batch update implementation."""
        # Use the shared batch update query builder
        statements = self.query_builder.build_batch_update_queries(updates)

        # Execute the batch update in a transaction
        with self._lock:
            conn = self._require_conn()
            try:
                conn.begin()
                for query, params in statements:
                    conn.execute(query, params)
                conn.commit()

                # DuckDB's UPDATE returns nothing here, so ask which ids exist.
                existing_ids = _existing_ids(
                    conn,
                    self.query_builder,
                    [record_id for record_id, _ in updates],
                )

                # Return results for each update
                results = []
                for record_id, _ in updates:
                    results.append(record_id in existing_ids)

                return results
            except Exception:
                conn.rollback()
                raise

    async def delete_batch(self, ids: list[str], *, _tx: Any = None) -> list[bool]:
        """Delete multiple records efficiently.

        Args:
            ids: List of record IDs to delete
            _tx: Internal. When supplied the DML joins the outer
                :meth:`_transaction` (see :meth:`create_batch`).

        Returns:
            List of success indicators
        """
        if not ids:
            return []

        self._check_connection()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._delete_batch_sync,
            ids,
            _tx is None,
        )

    def _delete_batch_sync(self, ids: list[str], own_tx: bool = True) -> list[bool]:
        """Synchronous batch delete implementation."""
        with self._lock:
            conn = self._require_conn()
            # Check which IDs exist before deletion
            existing_ids = _existing_ids(conn, self.query_builder, ids)

            # Use the shared batch delete query builder
            query, params = self.query_builder.build_batch_delete_query(ids)

            # Execute the batch delete in a transaction
            try:
                if own_tx:
                    conn.begin()
                conn.execute(query, params)
                if own_tx:
                    conn.commit()

                # Return results based on which IDs existed
                results = []
                for id in ids:
                    results.append(id in existing_ids)

                return results
            except Exception:
                if own_tx:
                    conn.rollback()
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
        """Stream the records a query matches, a page at a time.

        Each page is its own statement, sorted by the query's sort and then by
        the key, and each after the first begins after the last row read (see
        :func:`~dataknobs_data.streaming.stream_page`). Each runs under the
        connection lock as every statement here does: a DuckDB result left
        open between records is cut short, with no error, by the next
        statement on its connection. A row written or removed ahead of the
        stream's position between two pages moves no other row. The query's
        limit, offset and projection hold; with no sort the stream promises no
        order.
        """
        from ..streaming import aiter_search_pages

        async for record in aiter_search_pages(self._search_page, query, config):
            yield record

    async def _search_page(
        self, page: Query, after: Sequence[Any] | None
    ) -> tuple[list[Record], list[Any] | None]:
        """One page of a stream, after the row ``after`` holds the sort keys of."""
        self._check_connection()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._page_rows, page, after)

    async def stream_write(
        self, records: AsyncIterator[Record], config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database.

        Args:
            records: Async iterator of records
            config: Stream configuration

        Returns:
            Stream result with statistics

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


class SyncDuckDBDatabase(
    StructuredConfigConsumer[SyncDuckDBDatabaseConfig],
    DuckDBLayoutMixin,
    SyncDatabase,
):
    """Synchronous DuckDB database backend for analytical workloads.

    DuckDB is an embedded columnar database optimized for analytics.
    Provides 10-100x performance improvement over SQLite for
    aggregations, joins, and analytical queries.

    Features:
    - Columnar storage for fast analytical queries
    - Native Parquet integration for efficient data import/export
    - Advanced analytics support (window functions, CTEs, complex aggregations)

    Usage:
        ```python
        from dataknobs_data.backends.duckdb import SyncDuckDBDatabase

        # File-based database
        db = SyncDuckDBDatabase({"path": "/path/to/data.duckdb"})

        # In-memory database
        db = SyncDuckDBDatabase({"path": ":memory:"})

        with db:
            # Perform CRUD operations
            db.create(record)
            results = db.search(query)
        ```
    """

    CONFIG_CLS: ClassVar[type[SyncDuckDBDatabaseConfig]] = SyncDuckDBDatabaseConfig

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op — connection setup is deferred to
        :meth:`connect`).
        """
        cfg = self.config
        self.db_path = cfg.path
        self.table_name = cfg.table
        self.timeout = cfg.timeout
        self.read_only = cfg.opens_read_only
        self.auto_create_table = cfg.creates_table

        self.serializer = SQLRecordSerializer()
        self.table_manager = SQLTableManager(self.table_name, dialect="duckdb")
        # The one query builder, made with the table's layout. It needs no
        # connection, so a native configuration the layout refuses fails here.
        self._setup_layout()

        self.conn: duckdb.DuckDBPyConnection | None = None
        self._connected = False
        # What the read cores and ``close`` share with the async twin, whose
        # pool needs one statement at a time on the connection. Writes here
        # do not take it, so a store shared across threads is not serialized.
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Connect to the DuckDB database."""
        if self._connected:
            return
        self.conn = self._connect_file()
        self._connected = True
        logger.info(f"Connected to sync DuckDB database: {self.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        conn, self.conn = self.conn, None
        if conn is not None:
            self._close_held(conn)
        if self._connected:
            self._connected = False
            logger.info(f"Disconnected from sync DuckDB database: {self.db_path}")

    def create(self, record: Record) -> str:
        """Create a new record.

        Args:
            record: The record to create

        Returns:
            The record ID
        """
        conn = self._require_conn()
        record_id = record.id or self._generate_id()
        query, params = self.query_builder.build_create_query(record, record_id=record_id)

        try:
            conn.execute(query, params)
            return record_id
        except duckdb.ConstraintException as e:
            if is_duplicate_key_error(e):
                raise DuplicateRecordError(params[0]) from e
            # NOT NULL / CHECK / other column constraint — surface truthfully
            # instead of mislabeling it as a duplicate id.
            raise constraint_violation_error(params[0]) from e

    def read(self, id: str) -> Record | None:
        """Read a record by ID.

        Args:
            id: The record ID

        Returns:
            The record if found, None otherwise
        """
        self._check_connection()
        return self._read_rows(id)

    def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional optimistic-concurrency token from
                ``get_version(id)`` (a content hash for DuckDB). When provided,
                a stale token raises ``ConcurrencyError`` instead of
                overwriting. When ``None`` the update is unconditional,
                byte-identical to prior behavior.

        Returns:
            True if the record was updated, False if no record exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        conn = self._require_conn()
        query, params = self.query_builder.build_update_query(id, record)

        # Conditional write: compare the current content-hash token before
        # issuing the UPDATE. Reusing read() guarantees the token compared
        # here matches the one get_version() returned. On a single connection
        # the read and write are serialized; cross-connection atomicity is out
        # of scope (see the module docs on the in-process content-hash
        # backends).
        if expected_version is not None:
            current = self.read(id)
            if current is None:
                # Conditional update of an absent record is a documented False
                # return, not a warning-worthy event (matches SQLite).
                return False
            enforce_content_version(id, expected_version, current)

        # Check if record exists
        exists_query, exists_params = self.query_builder.build_exists_query(id)
        exists = conn.execute(exists_query, exists_params).fetchone() is not None

        if exists:
            conn.execute(query, params)
            return True

        logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")
        return False

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID
            expected_version: Optional content-hash token from
                ``get_version(id)``. When provided, a stale token raises
                ``ConcurrencyError`` and a missing record returns ``False``.
                When ``None`` the delete is unconditional, byte-identical to
                prior behavior.

        Returns:
            True if deleted, False if not found

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        conn = self._require_conn()
        query, params = self.query_builder.build_delete_query(id)

        if expected_version is not None:
            current = self.read(id)
            if current is None:
                return False
            enforce_content_version(id, expected_version, current)
            conn.execute(query, params)
            return True

        # First check if the record exists
        exists_query, exists_params = self.query_builder.build_exists_query(id)
        exists = conn.execute(exists_query, exists_params).fetchone() is not None

        if exists:
            conn.execute(query, params)
            return True
        return False

    def exists(self, id: str) -> bool:
        """Check if a record exists.

        Args:
            id: The record ID

        Returns:
            True if exists, False otherwise
        """
        self._check_connection()
        return self._exists_rows(id)

    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query.

        Args:
            query: The query specification

        Returns:
            List of matching records
        """
        self._check_connection()
        return self._search_rows(query)

    def count(self, query: Query | None = None) -> int:
        """Count records matching a query.

        Args:
            query: Optional query specification

        Returns:
            Count of matching records
        """
        self._check_connection()
        return self._count_rows(query)

    def _insert_batch_atomic(self) -> bool:
        # create_batch runs a multi-value INSERT inside an explicit transaction
        # (begin/commit, rollback on error): a colliding id rolls the batch back
        # so nothing is written on raise and the migrator's INSERT bulk
        # fast-path is safe.
        return True

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently.

        Uses a multi-value INSERT. Like ``create()``, this fails closed: a
        colliding id (or a duplicate id within the batch) raises
        ``DuplicateRecordError`` and the transaction is rolled back so nothing is
        written. A caller-supplied ``record.id`` is honored (the shared query
        builder mints a uuid only when a record has none).

        Args:
            records: List of records to create

        Returns:
            List of record IDs
        """
        if not records:
            return []

        conn = self._require_conn()
        statements, ids = self.query_builder.build_batch_create_queries(
            records, id_factory=self._generate_id
        )

        try:
            conn.begin()
            for query, params in statements:
                conn.execute(query, params)
            conn.commit()
            return ids
        except duckdb.ConstraintException as e:
            conn.rollback()
            if is_duplicate_key_error(e):
                stored = _existing_ids(conn, self.query_builder, [r.id for r in records if r.id])
                colliding = next((r.id for r in records if r.id in stored), ids[0])
                raise DuplicateRecordError(colliding) from e
            raise constraint_violation_error() from e
        except Exception:
            conn.rollback()
            raise

    def upsert_batch(self, records: list[Record]) -> list[str]:
        """Insert-or-overwrite multiple records efficiently in one statement.

        Uses ``INSERT ... ON CONFLICT (id) DO UPDATE``. Honors a caller-supplied
        ``record.id`` (minting a uuid only when absent); a colliding id is
        overwritten (never raised). Returns ids in input order.
        """
        if not records:
            return []

        conn = self._require_conn()
        statements, ids = self.query_builder.build_batch_upsert_queries(
            records, id_factory=self._generate_id
        )

        try:
            conn.begin()
            for query, params in statements:
                conn.execute(query, params)
            conn.commit()
            return ids
        except Exception:
            conn.rollback()
            raise

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently.

        Args:
            updates: List of (record_id, record) tuples

        Returns:
            List of success indicators
        """
        if not updates:
            return []

        conn = self._require_conn()
        statements = self.query_builder.build_batch_update_queries(updates)

        try:
            conn.begin()
            for query, params in statements:
                conn.execute(query, params)
            conn.commit()

            # DuckDB's UPDATE returns nothing here, so ask which ids exist.
            existing_ids = _existing_ids(
                conn, self.query_builder, [record_id for record_id, _ in updates]
            )

            results = []
            for record_id, _ in updates:
                results.append(record_id in existing_ids)

            return results
        except Exception:
            conn.rollback()
            raise

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently.

        Args:
            ids: List of record IDs to delete

        Returns:
            List of success indicators
        """
        if not ids:
            return []

        conn = self._require_conn()

        # Check which IDs exist before deletion

        existing_ids = _existing_ids(conn, self.query_builder, ids)

        query, params = self.query_builder.build_batch_delete_query(ids)

        try:
            conn.begin()
            conn.execute(query, params)
            conn.commit()

            results = []
            for id in ids:
                results.append(id in existing_ids)

            return results
        except Exception:
            conn.rollback()
            raise

    def _initialize(self) -> None:
        """Initialize method - connection setup handled in connect()."""
        pass

    def _count_all(self) -> int:
        """Count all records in the database: every row of a native table's scope."""
        return self.count()

    def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream the records a query matches, a page at a time.

        Each page is its own statement, sorted by the query's sort and then by
        the key, and each after the first begins after the last row read (see
        :func:`~dataknobs_data.streaming.stream_page`): a DuckDB result left
        open between records is cut short, with no error, by the next
        statement on its connection. A row written or removed ahead of the
        stream's position between two pages moves no other row. The query's
        limit, offset and projection hold; with no sort the stream promises no
        order.
        """
        from ..streaming import iter_search_pages

        yield from iter_search_pages(self._search_page, query, config)

    def _search_page(
        self, page: Query, after: Sequence[Any] | None
    ) -> tuple[list[Record], list[Any] | None]:
        """One page of a stream, after the row ``after`` holds the sort keys of."""
        self._check_connection()
        return self._page_rows(page, after)

    def stream_write(
        self, records: Iterator[Record], config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database.

        Args:
            records: Iterator of records
            config: Stream configuration

        Returns:
            Stream result with statistics

        Honors ``config.on_conflict`` via the shared conflict resolver: INSERT
        uses the ``create_batch`` bulk fast-path with a per-record ``create``
        fallback (so a colliding id fails closed and is attributed as a failure,
        not silently overwritten); UPSERT uses ``upsert_batch``; SKIP writes
        per-record via ``create`` and counts duplicates as skips.
        """
        from ..streaming import (
            StreamConfig,
            resolve_conflict_write,
            run_stream_write,
        )

        config = config or StreamConfig()

        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            insert_batch_func=self.create_batch,
            single_create_func=self.create,
            upsert_func=self.upsert,
            upsert_batch_func=self.upsert_batch,
        )
        return run_stream_write(
            records,
            batch_write_func=batch_write_func,
            single_write_func=single_write_func,
            skip_on_duplicate=skip_on_duplicate,
            config=config,
        )

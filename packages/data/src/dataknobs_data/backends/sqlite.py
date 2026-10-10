# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""SQLite backend implementation with sync and async support."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import SyncDatabase, enforce_content_version
from ..exceptions import DuplicateRecordError
from ..query import Query
from ..query_logic import ComplexQuery
from ..records import Record
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.mixins import SyncVectorOperationsMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .config import SyncSQLiteDatabaseConfig
from .sql_base import (
    SQLRecordSerializer,
    SQLTableManager,
    constraint_violation_error,
    is_duplicate_key_error,
)
from .sqlite_mixins import SQLiteLayoutMixin, SQLiteVectorSupport, register_regexp
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import ClassVar
    from ..streaming import StreamConfig, StreamResult
    from ..vector.types import DistanceMetric, VectorSearchResult


logger = logging.getLogger(__name__)


class SyncSQLiteDatabase(
    StructuredConfigConsumer[SyncSQLiteDatabaseConfig],
    SQLiteLayoutMixin,
    SyncDatabase,
    VectorConfigMixin,
    PythonVectorSearchMixin,  # Provides python_vector_search_sync
    BulkEmbedMixin,  # Must come before SyncVectorOperationsMixin to override bulk_embed_and_store
    SyncVectorOperationsMixin,
    SQLiteVectorSupport,
    SQLRecordSerializer,  # Use the standard SQL serializer
):
    """Synchronous SQLite database backend.

    Constructed through :class:`SyncSQLiteDatabaseConfig` — every
    documented config key is a typed field on that dataclass, so
    ``self.config`` is the typed config (not a dict) and the
    ``from_config`` / factory paths share one construction route.

    With ``layout: native`` it reads a table in somebody else's file through
    the table's own columns, opening the file read-only (see
    :class:`~dataknobs_data.backends.config.SQLiteDatabaseConfigBase`).
    """

    CONFIG_CLS: ClassVar[type[SyncSQLiteDatabaseConfig]] = SyncSQLiteDatabaseConfig

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op for SQLite — connection setup is
        deferred to :meth:`connect`).
        """
        cfg = self.config
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)
        self._init_vector_state()

        self.db_path = cfg.path
        self.table_name = cfg.table
        self.timeout = cfg.timeout
        self.check_same_thread = cfg.check_same_thread
        self.journal_mode = cfg.journal_mode
        self.synchronous = cfg.synchronous
        self.auto_create_table = cfg.auto_create_table

        self.table_manager = SQLTableManager(self.table_name, dialect="sqlite")
        # The one query builder, made with the table's layout. It needs no
        # connection, so a native configuration the layout refuses fails here.
        self._setup_layout()
        self._connect_to = self._connect_target()

        self.conn: sqlite3.Connection | None = None
        self._connected = False

    def connect(self) -> None:
        """Connect to the SQLite database."""
        if self._connected:
            return

        # Create directory if needed for file-based database. A native table
        # is in somebody else's file, which is opened read-only and never made.
        if self.db_path != ":memory:" and not self.native:
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

        target, uri = self._connect_to
        try:
            self.conn = sqlite3.connect(
                target, timeout=self.timeout, check_same_thread=self.check_same_thread, uri=uri
            )
        except sqlite3.OperationalError as e:
            if self.native:
                raise self._unopened_file_error(e) from e
            raise
        register_regexp(self.conn)

        # Enable row factory for dict-like access
        self.conn.row_factory = sqlite3.Row

        try:
            self._configure_sqlite()
            # Create table if it doesn't exist
            self._ensure_table()
        except BaseException as e:
            # Refused after opening, so close what was opened.
            self.conn.close()
            self.conn = None
            if self.native and isinstance(e, sqlite3.OperationalError):
                # The first statements to read the file: what fails here on a
                # native table is reading it, as a WAL file's side files that
                # cannot be made.
                raise self._unopened_file_error(e) from e
            raise

        self._connected = True
        logger.info(f"Connected to SQLite database: {self.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self._connected = False
            logger.info(f"Disconnected from SQLite database: {self.db_path}")

    def _configure_sqlite(self) -> None:
        """Configure SQLite settings for performance."""
        if not self.conn:
            return

        cursor = self.conn.cursor()

        # Set journal mode if specified
        if self.journal_mode:
            cursor.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            logger.debug(f"Set journal_mode to {self.journal_mode}")

        # Set synchronous mode if specified
        if self.synchronous:
            cursor.execute(f"PRAGMA synchronous = {self.synchronous}")
            logger.debug(f"Set synchronous to {self.synchronous}")

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        # Optimize for performance
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA mmap_size = 30000000000")

        cursor.close()

    def _ensure_table(self) -> None:
        """Ensure the table exists.

        When ``auto_create_table=True`` (default), runs ``CREATE TABLE IF NOT
        EXISTS …``. When ``auto_create_table=False``, verifies the table is
        present and raises ``RuntimeError`` if it isn't.
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        if not self.auto_create_table:
            exists_sql, params = self._relation_exists_query()
            cursor = self.conn.cursor()
            try:
                cursor.execute(exists_sql, params)
                row = cursor.fetchone()
                exists = bool(row[0]) if row else False
            finally:
                cursor.close()
            if not exists:
                raise self._missing_relation_error()
            return

        cursor = self.conn.cursor()
        try:
            cursor.executescript(self.table_manager.get_create_table_sql())
            self.conn.commit()
        finally:
            cursor.close()

    def _check_connection(self) -> None:
        """Check if database is connected."""
        self._require_conn()

    def _require_conn(self) -> sqlite3.Connection:
        """The connection, or the refusal :meth:`_check_connection` gives.

        The same test, returning what it tested: a caller that holds the
        result has the connection narrowed for the type checker, which a
        check in another method cannot give it.
        """
        if not self._connected or self.conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn

    def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()

        # Update vector dimensions tracking if needed
        if self._has_vector_fields(record):
            self._update_vector_dimensions(record)

        # Use centralized method to prepare record
        record, storage_id = self._prepare_record_for_storage(record)

        # Use the standard SQL serializer
        data_json = self.record_to_json(record)
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        # Build insert query for SQLite's standard table structure
        query = f"INSERT INTO {self.table_manager.qualified_table} (id, data, metadata) VALUES (?, ?, ?)"
        params = [storage_id, data_json, metadata_json]

        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            self.conn.commit()
            return storage_id
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            if is_duplicate_key_error(e):
                raise DuplicateRecordError(storage_id) from e
            # NOT NULL / CHECK / other column constraint — surface truthfully
            # instead of mislabeling it as a duplicate id.
            raise constraint_violation_error(storage_id) from e
        finally:
            cursor.close()

    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()

        query, params = self.query_builder.build_read_query(id)
        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            row = cursor.fetchone()

            if row:
                record = self.query_builder.record_from_row(dict(row))
                # Use centralized method to prepare record
                return self._prepare_record_from_storage(record, id)
            return None
        finally:
            cursor.close()

    def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional optimistic-concurrency token from
                ``get_version(id)`` (a content hash for SQLite). When provided,
                the read-compare-write runs inside one transaction so the
                compare-and-set is atomic within the connection; a stale token
                raises ``ConcurrencyError`` instead of overwriting. When
                ``None`` the update is unconditional, byte-identical to prior
                behavior.

        Returns:
            True if the record was updated, False if no record with the given ID exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current version token.
        """
        self._check_connection()

        # Update vector dimensions tracking if needed
        if self._has_vector_fields(record):
            self._update_vector_dimensions(record)

        # Use the standard SQL serializer
        data_json = self.record_to_json(record)
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        # Build update query
        query = (
            f"UPDATE {self.table_manager.qualified_table} SET data = ?, metadata = ? WHERE id = ?"
        )
        params = [data_json, metadata_json, id]

        # Conditional write: compare the current content-hash token before
        # issuing the UPDATE. Reusing read() guarantees the token compared
        # here is byte-identical to the one get_version() returned. On a
        # single connection the read and write are effectively serialized;
        # cross-connection atomicity is out of scope (see the module docs on
        # the in-process content-hash backends).
        if expected_version is not None:
            current = self.read(id)
            if current is None:
                return False
            enforce_content_version(id, expected_version, current)

        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            self.conn.commit()
            rows_affected = cursor.rowcount

            if rows_affected == 0:
                logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")

            return rows_affected > 0
        finally:
            cursor.close()

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        When ``expected_version`` is provided the current content-hash token is
        compared before the ``DELETE``; a stale token raises
        ``ConcurrencyError`` and a missing record returns ``False``. Reusing
        ``read()`` keeps the compared token byte-identical to ``get_version()``.
        On a single connection the read and delete are serialized;
        cross-connection atomicity is out of scope (see the module docs on the
        in-process content-hash backends). When ``None`` the delete is
        unconditional, byte-identical to prior behavior.
        """
        self._check_connection()

        if expected_version is not None:
            current = self.read(id)
            if current is None:
                return False
            enforce_content_version(id, expected_version, current)

        query, params = self.query_builder.build_delete_query(id)
        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            self.conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()

        query, params = self.query_builder.build_exists_query(id)
        cursor = self.conn.cursor()

        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result is not None
        finally:
            cursor.close()

    def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()

        cursor = self.conn.cursor()
        try:
            # Get count before clearing
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_manager.qualified_table}")
            count = cursor.fetchone()[0]

            # Clear the table
            cursor.execute(f"DELETE FROM {self.table_manager.qualified_table}")
            self.conn.commit()

            return count
        finally:
            cursor.close()

    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query."""
        self._check_connection()

        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql_query, params = self.query_builder.build_complex_search_query(query)
        else:
            sql_query, params = self.query_builder.build_search_query(query)

        cursor = self.conn.cursor()

        try:
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()

            # The layout sets each record's storage id from the row's key.
            records = [self.query_builder.record_from_row(dict(row)) for row in rows]

            # Apply field projection if specified
            if query.fields:
                records = [r.project(query.fields) for r in records]

            return records
        finally:
            cursor.close()

    def count(self, query: Query | None = None) -> int:
        """Count records matching a query."""
        self._check_connection()

        sql_query, params = self.query_builder.build_count_query(query)
        cursor = self.conn.cursor()

        try:
            cursor.execute(sql_query, params)
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()

    def _insert_batch_atomic(self) -> bool:
        # create_batch runs its INSERT statements inside one transaction: a
        # colliding id rolls the whole batch back, so nothing is written on
        # raise and the migrator's INSERT bulk fast-path is safe.
        return True

    def _max_parameters(self) -> int:
        """The most parameters one statement may bind on this connection.

        Read from the connection each time, since a connection may lower it
        (``setlimit``) and a build may ship a default below 32766.
        """
        return int(self._require_conn().getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))

    def _existing_ids(self, cursor: sqlite3.Cursor, ids: list[str]) -> set[str]:
        """Which of ``ids`` are stored, in one statement whatever their number."""
        if not ids:
            return set()
        query, params = self.query_builder.build_existing_ids_query(ids)
        cursor.execute(query, params)
        return {row[0] for row in cursor.fetchall()}

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently, in one transaction.

        Uses multi-value INSERTs, as many as the connection's parameter limit
        requires. Like ``create()``, this fails closed: a colliding id (or a
        duplicate id within the batch) raises ``DuplicateRecordError`` and,
        because the INSERTs run in one transaction, the whole batch is rolled
        back so nothing is written. A caller-supplied ``record.id`` is honored
        (the shared query builder mints a uuid only when a record has none).
        """
        if not records:
            return []

        self._check_connection()

        # Use the shared batch create query builder (honors record.id, mints via
        # _generate_id; raises DuplicateRecordError up front on a within-batch
        # duplicate id).
        statements, ids = self.query_builder.build_batch_create_queries(
            records, id_factory=self._generate_id, max_parameters=self._max_parameters()
        )

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            for query, params in statements:
                cursor.execute(query, params)
            self.conn.commit()
            return ids
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            if is_duplicate_key_error(e):
                # Name the colliding id precisely on the error path (cheap — only
                # runs on a failed batch, never on the happy path).
                stored = self._existing_ids(cursor, [r.id for r in records if r.id])
                colliding = next((r.id for r in records if r.id in stored), ids[0])
                raise DuplicateRecordError(colliding) from e
            # NOT NULL / CHECK / other column constraint — surface truthfully
            # instead of mislabeling it as a duplicate id.
            raise constraint_violation_error() from e
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def upsert_batch(self, records: list[Record]) -> list[str]:
        """Insert-or-overwrite multiple records efficiently, in one transaction.

        Uses ``INSERT ... ON CONFLICT (id) DO UPDATE``, as many statements as
        the connection's parameter limit requires. Honors a caller-supplied
        ``record.id`` (minting a uuid only when absent); a colliding id is
        overwritten (never raised). Returns ids in input order.
        """
        if not records:
            return []

        self._check_connection()

        statements, ids = self.query_builder.build_batch_upsert_queries(
            records, id_factory=self._generate_id, max_parameters=self._max_parameters()
        )

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            for query, params in statements:
                cursor.execute(query, params)
            self.conn.commit()
            return ids
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently, in one transaction.

        Each record takes its own update; a repeated id takes its last. An id
        not stored is reported ``False`` and written nowhere.
        """
        if not updates:
            return []

        self._check_connection()

        # One statement per update rather than a join: UPDATE … FROM needs
        # SQLite 3.33, and executemany binds three values per run.
        query, rows = self.query_builder.build_batch_update_rows(updates)

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            cursor.executemany(query, rows)
            self.conn.commit()

            # SQLite's UPDATE returns nothing here, so ask which ids exist.
            existing_ids = self._existing_ids(cursor, [record_id for record_id, _ in updates])
            return [record_id in existing_ids for record_id, _ in updates]
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently using a single query.

        Uses a single DELETE, whatever the number of ids.
        """
        if not ids:
            return []

        self._check_connection()

        cursor = self.conn.cursor()
        try:
            # Check which IDs exist before deletion
            existing_ids = self._existing_ids(cursor, ids)

            query, params = self.query_builder.build_batch_delete_query(ids)
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute(query, params)
            self.conn.commit()

            return [id in existing_ids for id in ids]
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def _initialize(self) -> None:
        """Initialize method - connection setup handled in connect()."""
        pass

    def _count_all(self) -> int:
        """Count all records in the database: every row of a native table's scope."""
        return self.count()

    def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream the records a query matches, a page of ``search`` at a time.

        Each page is its own statement, sorted by the query's sort and then by
        the key (see :func:`~dataknobs_data.streaming.stream_page`), so no
        statement stays open between records: an open SQLite read would lock
        the file's owner out of writing it. The query's limit, offset and
        projection hold; with no sort the stream promises no order.
        """
        from ..streaming import iter_search_pages

        yield from iter_search_pages(self.search, query, config)

    def stream_write(
        self, records: Iterator[Record], config: StreamConfig | None = None
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

    # Vector support methods
    def has_vector_support(self) -> bool:
        """Check if this backend has vector support.

        Returns:
            False - SQLite has no native vector support, uses Python-based similarity
        """
        return False  # No native vector support

    def enable_vector_support(self) -> bool:
        """Enable vector support for this backend.

        Returns:
            True - Vector support is always available (Python-based)
        """
        # SQLite doesn't need any special setup for vector support
        # We handle vectors as JSON strings
        self.vector_enabled = True
        return True

    def _vector_search(
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
        self._check_connection()

        return self.python_vector_search_sync(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
        )

    def add_vectors(
        self,
        vectors: list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        field_name: str = "embedding",
    ) -> list[str]:
        """Add vectors to the database.

        Args:
            vectors: List of vectors to add
            ids: Optional list of IDs
            metadata: Optional list of metadata dicts
            field_name: Name of the vector field

        Returns:
            List of created record IDs
        """
        from collections import OrderedDict

        from ..fields import VectorField

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        # Create records with vector fields
        records = []
        for i, vector in enumerate(vectors):
            # Create vector field
            vector_field = VectorField(
                name=field_name,
                value=vector,
                dimensions=len(vector) if isinstance(vector, (list, np.ndarray)) else None,
            )

            # Create record
            record_metadata = metadata[i] if metadata and i < len(metadata) else {}
            record = Record(
                data=OrderedDict({field_name: vector_field}),
                metadata=record_metadata,
                storage_id=ids[i],
            )
            records.append(record)

        # Use batch create for efficiency
        return self.create_batch(records)

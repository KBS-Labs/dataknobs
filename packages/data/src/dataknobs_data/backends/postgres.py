# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL backend implementation with proper connection management and vector support."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypedDict, cast

import asyncpg
import psycopg2
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.lifecycle import close_if_owned_sync
from dataknobs_common.structured_config import StructuredConfigConsumer

from dataknobs_utils.sql_utils import PostgresDB, quote_ident

from ..database import AsyncDatabase, SyncDatabase, version_conflict_error
from ..exceptions import DuplicateRecordError
from ..pooling import ConnectionPoolManager
from ..pooling.postgres import PostgresPoolConfig, create_asyncpg_pool, validate_asyncpg_pool
from ..query import Query
from ..query_logic import ComplexQuery
from ..streaming import (
    StreamConfig,
    StreamResult,
    async_run_stream_write,
    resolve_conflict_write,
    run_stream_write,
)
from ..vector.bulk_embed_mixin import AsyncBulkEmbedMixin, BulkEmbedMixin
from ..vector.mixins import (
    AsyncVectorOperationsMixin,
    SyncVectorOperationsMixin,
    resolve_metric,
)
from .config import PostgresDatabaseConfig
from .postgres_vector import format_vector_for_postgres
from .postgres_mixins import (
    PostgresBaseConfig,
    PostgresConnectionValidator,
    PostgresErrorHandler,
    PostgresTableManager,
    PostgresVectorSupport,
)
from .sql_base import (
    SQLQueryBuilder,
    SQLRecordSerializer,
    SQLTableManager,
    constraint_violation_error,
    validate_field_name,
    validate_field_path,
)
from ..vector.types import DistanceMetric


# Raised by both twins' ``create_vector_index``. The mixin declares
# ``dimensions`` optional because most backends ignore it; pgvector cannot
# index an expression whose width is not fixed, and saying so as a refusal
# rather than as a required argument is what lets a caller written against the
# mixin reach this method and be told why.
_DIMENSIONS_REQUIRED = (
    "dimensions is required to build a pgvector index: the index expression "
    "casts to ``vector(n)`` and pgvector will not index a column whose width "
    "is not fixed."
)


def _query_vector_values(query_vector: np.ndarray | list[float] | VectorField) -> Any:
    """Unwrap whatever the caller handed us down to the numbers.

    The width of what comes back is the width of the stored vectors too ---
    pgvector refuses to compare vectors of different widths --- which is how
    the search learns the ``dimensions`` an index was built for without being
    told.

    Args:
        query_vector: A ``VectorField``, a numpy array, or a sequence.

    Returns:
        The underlying sequence of numbers.
    """
    from ..fields import VectorField as _VectorField

    return query_vector.value if isinstance(query_vector, _VectorField) else query_vector


class _ConnConfig(TypedDict):
    """psycopg2 connection parameters, one declared type per key.

    A plain dict literal would infer ``dict[str, object]`` from the five
    different value types, and ``object`` is not something ``PostgresDB`` or
    ``validate_database_name`` accept — so every read became an error at the
    call it fed, for a dict whose contents were fully known.
    """

    host: str
    port: int
    database: str
    user: str
    password: str


if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from collections.abc import AsyncIterator, Iterator, Callable, Awaitable
    from typing import ClassVar
    from ..fields import VectorField
    from ..records import Record
    from ..vector.types import VectorSearchResult

logger = logging.getLogger(__name__)


def _ssl_to_sslmode(ssl: Any) -> str | None:
    """Translate the asyncpg-native ``ssl`` config to a psycopg2 ``sslmode``.

    The unified :class:`PostgresDatabaseConfig` ``ssl`` field keeps asyncpg
    semantics; the sync (psycopg2) backend speaks ``sslmode`` instead.
    ``None`` → no ``sslmode`` (libpq's own default), ``str`` → that mode,
    ``True`` → ``"require"``, ``False`` → ``"disable"``. An unsupported
    value (e.g. an ``ssl.SSLContext``, which psycopg2's ``connect`` cannot
    accept) raises rather than silently degrading.
    """
    if ssl is None:
        return None
    if isinstance(ssl, str):
        return ssl
    if isinstance(ssl, bool):
        return "require" if ssl else "disable"
    raise ConfigurationError(
        f"Sync Postgres backend cannot translate ssl={ssl!r} "
        f"({type(ssl).__name__}) to a psycopg2 sslmode. Pass an sslmode "
        "string (e.g. 'require'), a bool, or use the async backend for "
        "SSLContext-based configuration."
    )


class SyncPostgresDatabase(
    StructuredConfigConsumer[PostgresDatabaseConfig],
    SyncDatabase,
    BulkEmbedMixin,  # Must come before SyncVectorOperationsMixin to override bulk_embed_and_store
    SyncVectorOperationsMixin,
    SQLRecordSerializer,
    PostgresBaseConfig,
    PostgresTableManager,
    PostgresVectorSupport,
    PostgresConnectionValidator,
    PostgresErrorHandler,
):
    """Synchronous PostgreSQL database backend with proper connection management.

    Constructed through the unified :class:`PostgresDatabaseConfig` (shared
    with :class:`AsyncPostgresDatabase`), so both Postgres backends accept
    an identical connection surface. The async-only knobs
    (``min_pool_size``/``max_pool_size``/``command_timeout``) are inert here;
    ``ssl`` is honored via translation to a psycopg2 ``sslmode``.
    """

    CONFIG_CLS: ClassVar[type[PostgresDatabaseConfig]] = PostgresDatabaseConfig

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Connection setup itself is deferred to :meth:`connect` (``_initialize``
        is a no-op); this only computes the attributes ``connect`` consumes.
        """
        cfg = self.config
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)

        self.table_name = cfg.table
        self.schema_name = cfg.schema_name
        self._q_table = quote_ident(self.table_name)
        self._q_schema = quote_ident(self.schema_name)
        self._q_qualified = f"{self._q_schema}.{self._q_table}"
        self._connected = False
        self._ensure_database_enabled = cfg.ensure_database
        self.auto_create_table = cfg.auto_create_table
        self._init_vector_state()

        # Table manager for parameterized existence checks (psycopg2 pyformat style)
        self.table_manager = SQLTableManager(
            self.table_name,
            schema_name=self.schema_name,
            dialect="postgres",
            param_style="pyformat",
        )

        # Connection params consumed by connect()/_create_database.
        # Typed per key rather than left to inference: the values have five
        # different types, so an inferred dict joins them to ``object`` and
        # every read out of it becomes an error at the call it feeds. The
        # config class already declares each one — this carries that through
        # instead of discarding it at the dict boundary.
        self._conn_config: _ConnConfig = {
            "host": cfg.host,
            "port": cfg.port,
            "database": cfg.database,
            "user": cfg.user,
            "password": cfg.password,
        }
        # asyncpg-native ssl translated to psycopg2 sslmode (fail-fast on
        # an unsupported value such as an SSLContext).
        self._sslmode = _ssl_to_sslmode(cfg.ssl)

        # Annotated rather than inferred. A bare ``= None`` makes mypy read the
        # attribute's type as ``None``, so every ``self.db.query(...)`` below
        # was an error against ``None`` and the bodies around them were
        # written off as unreachable — which is what forced the
        # ``type: ignore[unreachable]`` that used to sit on ``close``, and what
        # kept mypy from ever type-checking the code it had declared dead.
        #
        # Declared as the type it holds in every method that uses it: those
        # methods have always assumed ``connect()`` ran first, and none of them
        # guards. The ``None`` is the pre-connect window, narrowed at the one
        # place that can be reached before ``connect()``.
        self.db: PostgresDB = None  # type: ignore[assignment]  # set in connect()
        self._owns_db = False  # set in _open_connection(), which builds it
        # The same treatment, for the same reason: left inferred, this reads as
        # ``None`` and every ``self.query_builder.build_*`` below becomes an
        # attribute error on ``None``. It sits one line from the attribute that
        # made the point, and had the identical defect.
        self.query_builder: SQLQueryBuilder = None  # type: ignore[assignment]  # set in connect()

    def connect(self) -> None:
        """Connect to the PostgreSQL database."""
        if self._connected:
            return  # Already connected

        # Initialize query builder with pyformat style for psycopg2
        self.query_builder = SQLQueryBuilder(
            self.table_name, self.schema_name, dialect="postgres", param_style="pyformat"
        )

        # Open connection and ensure table; if the database doesn't exist
        # and ensure_database is enabled, create it and retry.
        # Note: PostgresDB connects lazily — the actual psycopg2 connection
        # happens on first query (in _ensure_table), not in _open_connection.
        try:
            self._open_connection()
            self._ensure_table()
        except Exception as e:
            if self._ensure_database_enabled and self._is_invalid_catalog_error(e):
                self._create_database()
                # The first PostgresDB is about to be replaced, so it is closed
                # rather than dropped. It holds no live connection in the case
                # that gets here — connecting is what failed — but "the object
                # that is going away is closed by whoever owns it" is the rule
                # the rest of this class follows, and an ownership discipline
                # with one path exempted is one nobody can rely on.
                close_if_owned_sync(self.db, self._owns_db)
                self._open_connection()
                self._ensure_table()
            else:
                raise

        # Detect and enable vector support if requested
        if self.vector_enabled:
            self._detect_vector_support()

        self._connected = True
        self.log_operation("connect", f"Connected to table: {self.schema_name}.{self.table_name}")

    def _open_connection(self) -> None:
        """Open the PostgresDB connection using stored config.

        The config dict is normalized up-front by
        ``normalize_postgres_connection_config`` (via
        ``_parse_postgres_config``), so ``host``/``database``/``user``/
        ``port`` are already populated from ``connection_string``, explicit
        keys, or ``POSTGRES_*`` env-var fallbacks. No secondary dotenv
        lookup is needed here — the normalizer is the single env-var
        contract.
        """
        # Indexed, not ``.get(key, default)``: ``_ConnConfig`` is total and is
        # built with all five keys, so every default was unreachable while
        # reading as though it still applied. The normalizer above is where a
        # missing value acquires one.
        self.db = PostgresDB(
            host=self._conn_config["host"],
            db=self._conn_config["database"],
            user=self._conn_config["user"],
            pwd=self._conn_config["password"],
            port=self._conn_config["port"],
            sslmode=self._sslmode,
        )
        # Constructed here, so closed by close(). Recorded at the point of the
        # decision rather than re-derived from the attribute later.
        self._owns_db = True

    @staticmethod
    def _is_invalid_catalog_error(exc: Exception) -> bool:
        """Check if an exception indicates the database does not exist.

        psycopg2 raises ``OperationalError`` for connection-level failures
        but does not set ``pgcode`` on these (it's ``None``).  We fall back
        to checking the error message for the PostgreSQL FATAL text.
        """
        import psycopg2

        if not isinstance(exc, psycopg2.OperationalError):
            return False
        pgcode = getattr(exc, "pgcode", None)
        if pgcode == "3D000":
            return True
        # Connection-level errors have pgcode=None; match on message
        if pgcode is None and "does not exist" in str(exc):
            return True
        return False

    def _create_database(self) -> None:
        """Create the target database via the ``postgres`` maintenance database.

        Called only when the initial connection fails because the target
        database does not exist.
        """
        import psycopg2
        import psycopg2.sql
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        from .postgres_mixins import validate_database_name

        target_db = self._conn_config["database"]
        validate_database_name(target_db)

        conn_kwargs: dict[str, Any] = {
            "host": self._conn_config["host"],
            "port": self._conn_config["port"],
            "user": self._conn_config["user"],
            "password": self._conn_config["password"],
            "database": "postgres",
            "connect_timeout": 10,
        }
        if self._sslmode is not None:
            conn_kwargs["sslmode"] = self._sslmode
        conn = psycopg2.connect(**conn_kwargs)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()
        try:
            cursor.execute(
                psycopg2.sql.SQL("CREATE DATABASE {}").format(psycopg2.sql.Identifier(target_db))
            )
            logger.info("Created PostgreSQL database %r", target_db)
        except psycopg2.errors.DuplicateDatabase:
            pass  # Race condition: another process created it
        finally:
            cursor.close()
            conn.close()

    def close(self) -> None:
        """Close the database connection.

        This used to close nothing, on the stated grounds that "PostgresDB
        manages its own connections via context managers". It does not:
        psycopg2's ``with conn`` is a transaction scope, not a close. What kept
        that from showing up as exhausted connections was CPython refcounting
        reclaiming each connection when the frame exited — an interpreter
        detail, and one that left this method's own contract unmet.

        Ownership is recorded where it is decided — :meth:`_open_connection`,
        the one place that constructs a ``PostgresDB`` — rather than inferred
        here from the attribute being set. The two agree today, because
        construction is the only way ``self.db`` becomes non-``None``; passing
        ``self.db is not None`` would say *existence* while meaning
        *ownership*, and the day an injection path is added the two stop being
        the same thing silently. ``close_if_owned_sync`` null-guards its own
        argument, so the existence half needs no help from the caller.

        ``self.db`` is left in place rather than cleared: :meth:`connect`
        rebuilds it through :meth:`_open_connection`, so clearing it would buy
        nothing and would contradict the attribute's declared type.
        """
        close_if_owned_sync(self.db, self._owns_db)
        self._owns_db = False
        self._connected = False

    def _initialize(self) -> None:
        """Initialize method - connection setup moved to connect()."""
        # Configuration parsing stays here, actual connection in connect()
        pass

    def _detect_vector_support(self) -> None:
        """Detect and enable vector support if pgvector is available.

        The ``except Exception`` below is what lets a database genuinely
        without pgvector answer ``False`` rather than raise. An unconnected one
        used to take that same branch --- ``self.db`` is ``None``, the probe
        raised ``AttributeError``, and the handler reported it as "no vector
        support" for a database whose extensions it never got to read. A
        missing extension is still ``False``; a missing connection is not, so
        it is refused ahead of the probe, with the message every other door on
        this class gives.
        """
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

        from .postgres_vector import check_pgvector_extension_sync, install_pgvector_extension_sync

        try:
            # Check if pgvector is installed
            if check_pgvector_extension_sync(self.db):
                self._vector_enabled = True
                logger.info("pgvector extension detected and enabled")
            else:
                # Try to install it
                if install_pgvector_extension_sync(self.db):
                    self._vector_enabled = True
                    logger.info("pgvector extension installed and enabled")
                else:
                    logger.debug("pgvector extension not available")
        except Exception as e:
            logger.debug(f"Could not enable vector support: {e}")
            self._vector_enabled = False

    def _ensure_table(self) -> None:
        """Ensure the records table exists.

        When ``auto_create_table=True`` (default), runs ``CREATE TABLE IF NOT
        EXISTS …``. When ``auto_create_table=False``, verifies the table is
        present and raises ``RuntimeError`` if it isn't — for consumers
        managing DDL via Alembic / Flyway / Sqitch.
        """
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

        if not self.auto_create_table:
            exists_sql, params = self.table_manager.get_table_exists_sql()
            # get_table_exists_sql returns a positional tuple or a named dict
            # depending on param_style; this manager is built with
            # param_style="pyformat" (see __init__), which pins the dict branch
            # -- the one PostgresDB.query binds.
            df = self.db.query(exists_sql, cast("dict[str, Any]", params))
            exists = bool(df.iloc[0, 0]) if not df.empty else False
            if not exists:
                raise RuntimeError(
                    f"Table {self.schema_name}.{self.table_name} does not exist "
                    "and auto_create_table is disabled. Run your migrations "
                    "before starting the application."
                )
            return

        create_table_sql = self.get_create_table_sql(self.schema_name, self.table_name)
        self.db.execute(create_table_sql)

    def _record_to_row(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to a database row (delegates to shared serializer)."""
        return SQLRecordSerializer.record_to_row(record, id)

    @staticmethod
    def _frame_row_to_dict(row: pd.Series[Any]) -> dict[str, Any]:
        """A DataFrame row as the ``dict[str, Any]`` the serializer declares.

        ``Series.to_dict()`` is typed ``dict[Hashable, Any]``, because a
        DataFrame's labels need not be strings. A row out of these queries
        always has string column names, so the conversion is safe — but it has
        to be *stated*, and stating it once is the point. Written out at each
        call site it was stated three times and forgotten at a fourth, where
        ``vector_search`` passed the ``Series`` itself: harmless today only
        because ``Series`` happens to answer ``.get``/``in``/``[]`` the way a
        mapping does, and a latent bug the moment the serializer uses anything
        a ``Series`` implements differently.
        """
        return {str(key): value for key, value in row.to_dict().items()}

    def _row_to_record(self, row: dict[str, Any]) -> Record:
        """Convert a database row to a Record (delegates to shared serializer)."""
        return self.row_to_record(row)

    def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()
        # Use record's ID if it has one, otherwise generate a new one
        id = record.id if record.id else self._generate_id()
        row = self._record_to_row(record, id)

        sql = f"""
        INSERT INTO {self._q_qualified} (id, data, metadata)
        VALUES (%(id)s, %(data)s, %(metadata)s)
        """
        try:
            self.db.execute(sql, row)
        except psycopg2.errors.UniqueViolation as e:
            raise DuplicateRecordError(id) from e
        except psycopg2.IntegrityError as e:
            # Every other constraint the deployment put on this table. Caught
            # by the DB-API base rather than by naming NOT NULL / CHECK /
            # FOREIGN KEY, so a constraint kind not listed here still maps.
            # `UniqueViolation` is a subclass, so it must precede this clause.
            raise constraint_violation_error(id) from e
        return id

    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()
        sql = f"""
        SELECT id, data, metadata
        FROM {self._q_qualified}
        WHERE id = %(id)s
        """
        df = self.db.query(sql, {"id": id})

        if df.empty:
            return None

        return self._row_to_record(self._frame_row_to_dict(df.iloc[0]))

    def get_version(self, id: str) -> str | None:
        """Return the row's ``xmin`` transaction id as the version token.

        ``xmin`` is PostgreSQL's system column holding the id of the
        transaction that last inserted/updated the row; it advances on every
        UPDATE, so it is a native monotonic-per-row version — ABA-safe, unlike
        the base content-hash default this overrides.
        """
        self._check_connection()
        sql = f"""
        SELECT xmin::text AS version
        FROM {self._q_qualified}
        WHERE id = %(id)s
        """
        df = self.db.query(sql, {"id": id})
        if df.empty:
            return None
        return str(df.iloc[0]["version"])

    def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional ``xmin`` token from ``get_version(id)``.
                When provided, the ``UPDATE`` carries an ``AND xmin = …``
                predicate so the compare-and-set is enforced atomically by the
                server; a stale token raises ``ConcurrencyError``. When ``None``
                the update is unconditional, byte-identical to prior behavior.

        Returns:
            True if the record was updated, False if no record with the given ID exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current ``xmin`` token.
        """
        self._check_connection()
        row = self._record_to_row(record, id)

        if expected_version is not None:
            sql = f"""
            UPDATE {self._q_qualified}
            SET data = %(data)s, metadata = %(metadata)s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %(id)s AND xmin::text = %(expected_version)s
            """
            params = dict(row)
            params["expected_version"] = expected_version
            result = self.db.execute(sql, params)
            rows_affected = result if isinstance(result, int) else 0
            if rows_affected == 0:
                # The atomic UPDATE matched nothing: either the row is gone
                # (update never inserts -> False) or the token is stale
                # (concurrent modification -> raise). A follow-up read only
                # picks the right disposition/message; the CAS itself already
                # happened server-side.
                current = self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current)
            return True

        sql = f"""
        UPDATE {self._q_qualified}
        SET data = %(data)s, metadata = %(metadata)s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %(id)s
        """
        result = self.db.execute(sql, row)

        # PostgresDB.execute returns number of affected rows
        rows_affected = result if isinstance(result, int) else 0

        if rows_affected == 0:
            logger.warning(f"Update affected 0 rows for id={id}. Record may not exist.")

        return rows_affected > 0

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        When ``expected_version`` is provided the ``DELETE`` carries an
        ``AND xmin = …`` predicate so the compare-and-set is enforced
        atomically by the server; a stale token raises ``ConcurrencyError``
        and a missing row returns ``False``. When ``None`` the delete is
        unconditional, byte-identical to prior behavior.
        """
        self._check_connection()

        if expected_version is not None:
            sql = f"""
            DELETE FROM {self._q_qualified}
            WHERE id = %(id)s AND xmin::text = %(expected_version)s
            """
            result = self.db.execute(sql, {"id": id, "expected_version": expected_version})
            rows_affected = result if isinstance(result, int) else 0
            if rows_affected == 0:
                # The atomic DELETE matched nothing: either the row is gone
                # (delete of an absent id -> False) or the token is stale
                # (concurrent modification -> raise). The follow-up read only
                # picks the disposition; the CAS already happened server-side.
                current = self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current)
            return True

        sql = f"""
        DELETE FROM {self._q_qualified}
        WHERE id = %(id)s
        """
        result = self.db.execute(sql, {"id": id})
        return result > 0 if isinstance(result, int) else False

    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()
        sql = f"""
        SELECT 1 FROM {self._q_qualified}
        WHERE id = %(id)s
        LIMIT 1
        """
        df = self.db.query(sql, {"id": id})
        return not df.empty

    def upsert(
        self,
        id_or_record: str | Record,
        record: Record | None = None,
        *,
        expected_version: str | None = None,
    ) -> str:
        """Update or insert a record.

        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic

        When ``expected_version`` is provided the upsert is conditional: the
        record must already exist with a matching ``xmin`` token, otherwise it
        raises ``ConcurrencyError``. A conditional upsert never inserts.
        """
        self._check_connection()

        # Resolve the storage id via the shared helper (honors an explicit id;
        # mints via the overridable _generate_id() hook when the record has none).
        id, record = self._resolve_upsert_id(id_or_record, record)

        # Conditional upsert: delegate to update()'s atomic xmin
        # compare-and-set. A True return is the update; a stale token raises
        # straight out; a False return means the row is absent, which for a
        # conditional upsert is itself a conflict (it never inserts). update()'s
        # conditional path reports an absent row without logging, so this stays
        # quiet on the miss.
        if expected_version is not None:
            if self.update(id, record, expected_version=expected_version):
                return id
            raise version_conflict_error(id, expected_version, None)

        # Unconditional upsert: probe existence, but ACT on update()'s return so
        # a concurrent delete between exists() and update() falls through to the
        # insert instead of claiming a false success (the old code ignored
        # update()'s return). Gating update() behind exists() also skips a
        # doomed UPDATE (and its log) on the common insert path.
        if self.exists(id) and self.update(id, record):
            return id
        # Insert with specific ID (unconditional upsert of an absent row).
        row = self._record_to_row(record, id)
        sql = f"""
        INSERT INTO {self._q_qualified} (id, data, metadata)
        VALUES (%(id)s, %(data)s, %(metadata)s)
        """
        self.db.execute(sql, row)
        return id

    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()

        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql_query, params_list = self.query_builder.build_complex_search_query(query)
        else:
            sql_query, params_list = self.query_builder.build_search_query(query)

        # Build params dict for psycopg2
        # The query builder now generates %(p0)s style placeholders directly
        params_dict = {}
        if params_list:
            for i, param in enumerate(params_list):
                params_dict[f"p{i}"] = param

        # Execute query
        df = self.db.query(sql_query, params_dict)

        # Convert to records
        records = []
        for _, row in df.iterrows():
            record = self._row_to_record(self._frame_row_to_dict(row))

            # Apply field projection if specified
            if query.fields:
                record = record.project(query.fields)

            records.append(record)

        return records

    def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()
        sql = f"SELECT COUNT(*) as count FROM {self._q_qualified}"
        df = self.db.query(sql)
        return int(df.iloc[0]["count"]) if not df.empty else 0

    def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()
        # Get count first
        count = self._count_all()

        # Delete all records
        sql = f"TRUNCATE TABLE {self._q_qualified}"
        self.db.execute(sql)

        return count

    def _insert_batch_atomic(self) -> bool:
        # create_batch is a single multi-value INSERT: the whole statement
        # commits or aborts as a unit, so a colliding id writes nothing and the
        # migrator's INSERT bulk fast-path is safe.
        return True

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently using a single query.

        Uses multi-value INSERT for better performance.

        Args:
            records: List of records to create

        Returns:
            List of created record IDs
        """
        if not records:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL with pyformat style
        from .sql_base import SQLQueryBuilder

        query_builder = SQLQueryBuilder(
            self.table_name, self.schema_name, dialect="postgres", param_style="pyformat"
        )

        # Use the shared batch create query builder (honors record.id, mints via
        # _generate_id; raises DuplicateRecordError up front on a within-batch
        # duplicate id).
        query, params_list, ids = query_builder.build_batch_create_query(
            records, id_factory=self._generate_id
        )

        # Build params dict for psycopg2
        params_dict = {}
        for i, param in enumerate(params_list):
            params_dict[f"p{i}"] = param

        # Execute the batch insert and get returned IDs. Like create(), a
        # colliding id fails closed: the single INSERT is transactional, so the
        # unique-violation aborts the whole batch (nothing written).
        try:
            result_df = self.db.query(query, params_dict)
        except psycopg2.errors.UniqueViolation as e:
            colliding = next((r.id for r in records if r.id and self.exists(r.id)), ids[0])
            raise DuplicateRecordError(colliding) from e
        except psycopg2.IntegrityError as e:
            # No id: a batch INSERT is one statement, and the driver names the
            # constraint rather than the row that tripped it.
            raise constraint_violation_error() from e

        # PostgreSQL RETURNING clause gives us the actual inserted IDs
        if not result_df.empty:
            return result_df["id"].tolist()
        return ids

    def upsert_batch(self, records: list[Record]) -> list[str]:
        """Insert-or-overwrite multiple records in a single statement.

        Uses ``INSERT ... ON CONFLICT (id) DO UPDATE``. Honors a caller-supplied
        ``record.id`` (minting a uuid only when absent); a colliding id is
        overwritten (never raised). Returns ids in input order.
        """
        if not records:
            return []

        self._check_connection()

        from .sql_base import SQLQueryBuilder

        query_builder = SQLQueryBuilder(
            self.table_name, self.schema_name, dialect="postgres", param_style="pyformat"
        )
        query, params_list, ids = query_builder.build_batch_upsert_query(
            records, id_factory=self._generate_id
        )

        params_dict = {}
        for i, param in enumerate(params_list):
            params_dict[f"p{i}"] = param

        # RETURNING order is not guaranteed under ON CONFLICT, so return the
        # builder's input-order ids rather than the result-set order.
        self.db.query(query, params_dict)
        return ids

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently using a single query.

        Uses single DELETE with IN clause for better performance.

        Args:
            ids: List of record IDs to delete

        Returns:
            List of success flags for each deletion
        """
        if not ids:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL with pyformat style
        from .sql_base import SQLQueryBuilder

        query_builder = SQLQueryBuilder(
            self.table_name, self.schema_name, dialect="postgres", param_style="pyformat"
        )

        # Use the shared batch delete query builder (includes RETURNING clause)
        query, params_list = query_builder.build_batch_delete_query(ids)

        # Build params dict for psycopg2
        params_dict = {}
        for i, param in enumerate(params_list):
            params_dict[f"p{i}"] = param

        # Execute the batch delete and get returned IDs
        result_df = self.db.query(query, params_dict)

        # Get list of deleted IDs from RETURNING clause
        deleted_ids = set(result_df["id"].tolist()) if not result_df.empty else set()

        # Return results based on which IDs were actually deleted
        results = []
        for id in ids:
            results.append(id in deleted_ids)

        return results

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using a single query.

        Uses PostgreSQL's CASE expressions for batch updates via shared SQL builder.

        Args:
            updates: List of (id, record) tuples to update

        Returns:
            List of success flags for each update
        """
        if not updates:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL with pyformat style
        from .sql_base import SQLQueryBuilder

        query_builder = SQLQueryBuilder(
            self.table_name, self.schema_name, dialect="postgres", param_style="pyformat"
        )

        # Use the shared batch update query builder
        query, params_list = query_builder.build_batch_update_query(updates)

        # Build params dict for psycopg2
        params_dict = {}
        for i, param in enumerate(params_list):
            params_dict[f"p{i}"] = param

        # Execute the batch update and get returned IDs (query now includes RETURNING clause)
        result_df = self.db.query(query, params_dict)

        # Get list of updated IDs from RETURNING clause
        updated_ids = set(result_df["id"].tolist()) if not result_df.empty else set()

        results = []
        for record_id, _ in updates:
            results.append(record_id in updated_ids)

        return results

    def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from PostgreSQL."""
        # Pre-flight the field grammar before a connection is acquired, through
        # the same check ``build_where_clause`` applies at the point of
        # interpolation below. Both sites called a grammar of their own once,
        # and the two disagreed about a dotted path.
        if query and query.filters:
            for f in query.filters:
                validate_field_path(f.field)
        self._check_connection()
        config = config or StreamConfig()

        # Build SQL query
        sql = f"SELECT id, data, metadata FROM {self._q_qualified}"
        params = {}

        # Through the same builder ``search`` uses, so the two doors apply one
        # Query the same way. Open-coded here, this loop emitted a clause only
        # for ``Operator.EQ`` and dropped every other operator in silence, so a
        # caller who swapped ``search`` for ``stream_read`` to bound memory got
        # back rows it had filtered out.
        where_clause, filter_params = self.query_builder.build_where_clause(query)
        if where_clause:
            # ``build_where_clause`` is written to extend a predicate that is
            # already there and so opens with " AND ". ``WHERE TRUE`` gives it
            # the one it expects, which is cheaper to read than slicing the
            # prefix back off and cannot go wrong when the clause changes shape.
            sql += " WHERE TRUE" + where_clause
            params.update({f"p{i}": value for i, value in enumerate(filter_params)})

        # Use cursor for streaming
        # Note: PostgresDB may need modification to support cursors
        # For now, we'll fetch in batches
        sql += f" LIMIT {config.batch_size} OFFSET %(offset)s"

        offset = 0
        while True:
            params["offset"] = offset
            df = self.db.query(sql, params)

            if df.empty:
                break

            for _, row in df.iterrows():
                record = self._row_to_record(self._frame_row_to_dict(row))
                if query and query.fields:
                    record = record.project(query.fields)
                yield record

            offset += config.batch_size

            # If we got less than batch_size, we're done
            if len(df) < config.batch_size:
                break

    def stream_write(
        self, records: Iterator[Record], config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into PostgreSQL.

        Honors ``config.on_conflict``: INSERT uses the batch fast-path
        (``_write_batch``) with a ``create`` per-record fallback; UPSERT/SKIP
        write per-record via ``upsert``/``create``.
        """
        self._check_connection()
        config = config or StreamConfig()
        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            insert_batch_func=self._write_batch,
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

    def _write_batch(self, records: list[Record]) -> list[str]:
        """Write a batch of records with a single multi-row INSERT.

        The INSERT fast-path for the streaming INSERT policy. Like ``create()``,
        it honors a caller-supplied ``record.id`` (minting only when absent) and
        fails closed on a colliding id — the single INSERT is atomic, so the
        unique-violation aborts the whole batch and the streaming loop falls back
        to per-record ``create()`` to attribute the specific collision.

        Returns:
            List of created record IDs
        """
        # Build batch insert SQL
        values = []
        params = {}
        ids = []

        for i, record in enumerate(records):
            id = record.id if record.id else self._generate_id()
            ids.append(id)
            row = self._record_to_row(record, id)
            values.append(f"(%(id_{i})s, %(data_{i})s, %(metadata_{i})s)")
            params[f"id_{i}"] = row["id"]
            params[f"data_{i}"] = row["data"]
            params[f"metadata_{i}"] = row["metadata"]

        sql = f"""
        INSERT INTO {self._q_qualified} (id, data, metadata)
        VALUES {", ".join(values)}
        """
        try:
            self.db.execute(sql, params)
        except psycopg2.errors.UniqueViolation as e:
            colliding = next((r.id for r in records if r.id and self.exists(r.id)), ids[0])
            raise DuplicateRecordError(colliding) from e
        except psycopg2.IntegrityError as e:
            raise constraint_violation_error() from e
        return ids

    def _vector_search(
        self,
        query_vector: np.ndarray | list[float] | VectorField,
        *,
        vector_field: str,
        k: int,
        metric: DistanceMetric,
        filter: Query | None,
    ) -> list[VectorSearchResult]:
        """Raw k-NN through pgvector's distance operators.

        The threshold and the source assembly are the mixin's; this is the
        part that knows how to ask pgvector.

        Reads the vector out of the JSON ``data`` column, which is where
        every write path on both twins puts it. :meth:`AsyncPostgresDatabase.
        _vector_search` used to read a dedicated ``vector_<field>`` column
        instead --- a location three of the fourteen write paths across the two
        classes filled --- so the two twins searched different storage on the
        same table and a corpus written by one was invisible to the other.
        Both ask the same question of the same column now, through
        :func:`build_vector_search_sql`.

        Args:
            query_vector: Query vector (numpy array, list, or VectorField)
            vector_field: Name of the vector field to search, read out of the
                JSON ``data`` column
            k: Maximum number of results to return
            metric: Distance metric, already resolved by the mixin
            filter: Optional query filter to apply before vector search

        Returns:
            List of VectorSearchResult objects ordered by similarity
        """
        if not self._vector_enabled:
            raise RuntimeError("Vector search not available - pgvector not installed")

        self._check_connection()

        from ..vector.types import VectorSearchResult
        from .postgres_vector import build_vector_search_sql, distance_to_score

        values = _query_vector_values(query_vector)
        vector_str = format_vector_for_postgres(values)

        params: list[Any] = [vector_str, vector_field]

        # Add filters if provided using the query builder. The builder emits
        # pyformat placeholders because this class configures it that way.
        filter_clause = ""
        if filter:
            filter_clause, filter_params = self.query_builder.build_where_clause(
                filter, len(params) + 1
            )
            params.extend(filter_params)

        sql = build_vector_search_sql(
            q_qualified=self._q_qualified,
            vector_field=vector_field,
            dimensions=len(values),
            metric=metric,
            vector_placeholder="%(p0)s",
            field_placeholder="%(p1)s",
            limit_clause=f"LIMIT %(p{len(params)})s",
            filter_clause=filter_clause,
        )
        params.append(k)

        param_dict = {f"p{i}": param for i, param in enumerate(params)}
        df = self.db.query(sql, param_dict)

        # See the async twin: a ``VectorField`` object stored without its
        # ``value`` key passes the presence predicate and yields no distance.
        return [
            VectorSearchResult(
                record=self._row_to_record(self._frame_row_to_dict(row)),
                score=distance_to_score(metric, float(row["distance"])),
                vector_field=vector_field,
                metadata={"distance": float(row["distance"]), "metric": metric.value},
            )
            for _, row in df.iterrows()
            if row["distance"] is not None
        ]

    def create_vector_index(
        self,
        vector_field: str = "embedding",
        dimensions: int | None = None,
        metric: DistanceMetric | str | None = None,
        index_type: str = "ivfflat",
        lists: int | None = None,
    ) -> bool:
        """Create a vector index for efficient similarity search.

        The twin's implementation, so that the backend that can build an
        index is not the only one that cannot use it. This class inherited
        the mixin's ``return True`` no-op, which reported success and created
        nothing.

        ``dimensions`` is declared optional because the mixin declares it so,
        and refused when absent because pgvector cannot index an expression
        of unfixed width.

        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions in the vectors. Required here,
                unlike on the backends that ignore it.
            metric: Distance metric for the index
            index_type: Type of index (ivfflat, hnsw)
            lists: Number of lists for IVFFlat index

        Returns:
            True if index was created successfully
        """
        from .postgres_vector import (
            build_vector_index_sql,
            build_vector_value_expression,
            get_optimal_index_type,
            get_vector_count_sql,
        )

        self._check_connection()

        if not self._vector_enabled:
            return False

        if dimensions is None:
            raise ValueError(_DIMENSIONS_REQUIRED)

        if not lists and index_type == "ivfflat":
            count_df = self.db.query(
                get_vector_count_sql(self._q_schema, self._q_table, vector_field)
            )
            count = int(count_df.iloc[0]["count"]) if not count_df.empty else 0
            _, params = get_optimal_index_type(count)
            lists = params.get("lists", 100)

        index_sql = build_vector_index_sql(
            q_table_name=self._q_table,
            q_schema_name=self._q_schema,
            column_name=build_vector_value_expression(vector_field, dimensions),
            dimensions=dimensions,
            metric=resolve_metric(self, metric),
            index_type=index_type,
            index_params={"lists": lists} if lists else None,
            field_name=vector_field,
        )

        try:
            logger.debug(f"Creating vector index with SQL: {index_sql}")
            self.db.execute(index_sql)
            return True
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")
            logger.debug(f"Index SQL was: {index_sql}")
            return False

    def drop_vector_index(
        self, vector_field: str = "embedding", metric: DistanceMetric | str | None = None
    ) -> bool:
        """Drop a vector index.

        Args:
            vector_field: Name of the vector field
            metric: Distance metric used in the index

        Returns:
            True if index was dropped successfully
        """
        from .postgres_vector import get_vector_index_name

        self._check_connection()

        index_name = get_vector_index_name(
            self.table_name, vector_field, resolve_metric(self, metric).value
        )

        try:
            self.db.execute(f"DROP INDEX IF EXISTS {self._q_schema}.{quote_ident(index_name)}")
            return True
        except Exception as e:
            logger.warning(f"Failed to drop vector index: {e}")
            return False

    def get_vector_index_stats(self, vector_field: str = "embedding") -> dict[str, Any]:
        """Get statistics about a vector field and its index.

        Args:
            vector_field: Name of the vector field

        Returns:
            Dictionary with index statistics
        """
        from .postgres_vector import get_index_check_sql, get_vector_count_sql

        self._check_connection()

        stats: dict[str, Any] = {"field": vector_field, "indexed": False, "vector_count": 0}

        try:
            count_df = self.db.query(
                get_vector_count_sql(self._q_schema, self._q_table, vector_field)
            )
            stats["vector_count"] = int(count_df.iloc[0]["count"]) if not count_df.empty else 0

            # Raw (unquoted) names are correct here: the catalog query binds
            # them as text against pg_indexes columns, not as identifiers.
            index_sql, params = get_index_check_sql(self.schema_name, self.table_name, vector_field)
            # asyncpg's $N numbering, rewritten for psycopg2's pyformat.
            for i in range(len(params), 0, -1):
                index_sql = index_sql.replace(f"${i}", f"%(p{i - 1})s")
            index_df = self.db.query(index_sql, {f"p{i}": value for i, value in enumerate(params)})
            stats["indexed"] = bool(index_df.iloc[0]["has_index"]) if not index_df.empty else False
        except Exception as e:
            # Reported rather than only logged. The defaults above are a
            # truthful answer to "no index, no vectors" and an untruthful one
            # to "the query failed", and a caller reading the returned dict
            # could not tell which it had --- the two differ by a log line it
            # is not looking at.
            logger.warning(f"Failed to get vector index stats: {e}")
            stats["error"] = str(e)

        return stats

    def has_vector_support(self) -> bool:
        """Check if this database has vector support enabled.

        Returns:
            True if vector operations are supported
        """
        return self._vector_enabled

    def enable_vector_support(self) -> bool:
        """Enable vector support for this database if possible.

        Returns:
            True if vector support is now enabled
        """
        if self._vector_enabled:
            return True

        self._detect_vector_support()
        return self._vector_enabled


# Global pool manager instance for async PostgreSQL connections
_pool_manager = ConnectionPoolManager[asyncpg.Pool]()


class AsyncPostgresDatabase(
    StructuredConfigConsumer[PostgresDatabaseConfig],
    AsyncDatabase,
    AsyncBulkEmbedMixin,  # Must come before AsyncVectorOperationsMixin to override bulk_embed_and_store
    AsyncVectorOperationsMixin,
    PostgresBaseConfig,
    PostgresTableManager,
    PostgresVectorSupport,
    PostgresConnectionValidator,
    PostgresErrorHandler,
):
    """Native async PostgreSQL database backend with vector support and event loop-aware connection pooling.

    Constructed through the unified :class:`PostgresDatabaseConfig` (shared
    with :class:`SyncPostgresDatabase`). This backend honors the full
    parameter set, including the pool bounds (``min_pool_size`` /
    ``max_pool_size``), ``command_timeout``, and asyncpg-native ``ssl``.
    """

    CONFIG_CLS: ClassVar[type[PostgresDatabaseConfig]] = PostgresDatabaseConfig

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Pool creation is deferred to :meth:`connect` (``_initialize`` is a
        no-op); this only assembles the :class:`PostgresPoolConfig` and the
        attributes ``connect`` consumes.
        """
        cfg = self.config
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)

        self.table_name = cfg.table
        self.schema_name = cfg.schema_name
        self._q_table = quote_ident(self.table_name)
        self._q_schema = quote_ident(self.schema_name)
        self._q_qualified = f"{self._q_schema}.{self._q_table}"
        self._connected = False
        self._ensure_database_enabled = cfg.ensure_database
        self.auto_create_table = cfg.auto_create_table
        self._init_vector_state()

        # Declared here for the sync twin's stated reason, and it was the twin
        # difference that made this class's own readers disagree: ``search``
        # re-created the builder under ``hasattr`` while ``stream_read`` reached
        # for it directly, which reads as one of the two missing a guard.
        # Neither is. ``connect()`` binds this before it sets ``_connected``,
        # and every reader passes ``_check_connection()`` first, so a builder
        # that is not there yet is unreachable from any of them.
        self.query_builder: SQLQueryBuilder = None  # type: ignore[assignment]  # set in connect()

        # Table manager for parameterized existence checks (asyncpg numeric style)
        self.table_manager = SQLTableManager(
            self.table_name,
            schema_name=self.schema_name,
            dialect="postgres",
            param_style="numeric",
        )

        # Assemble the pool config directly from the typed fields (the
        # connection layer is already resolved in PostgresDatabaseConfig).
        self._pool_config = PostgresPoolConfig(
            host=cfg.host,
            port=cfg.port,
            database=cfg.database,
            user=cfg.user,
            password=cfg.password,
            min_size=cfg.min_pool_size,
            max_size=cfg.max_pool_size,
            command_timeout=cfg.command_timeout,
            ssl=cfg.ssl,
        )
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Connect to the database."""
        if self._connected:
            return

        # Attempt pool creation; if the database doesn't exist and
        # ensure_database is enabled, create it and retry.
        from ..pooling import BasePoolConfig

        try:
            self._pool = await _pool_manager.get_pool(
                self._pool_config,
                cast("Callable[[BasePoolConfig], Awaitable[Any]]", create_asyncpg_pool),
                validate_asyncpg_pool,
            )
        except asyncpg.InvalidCatalogNameError:
            if self._ensure_database_enabled:
                await self._create_database()
                self._pool = await _pool_manager.get_pool(
                    self._pool_config,
                    cast("Callable[[BasePoolConfig], Awaitable[Any]]", create_asyncpg_pool),
                    validate_asyncpg_pool,
                )
            else:
                raise

        # get_pool incremented this holder's claim on the shared pool. If
        # table/vector setup fails we never set _connected, so balance the
        # increment here rather than relying on the caller invoking close()
        # after a failed connect() (which would otherwise leak the holder).
        try:
            # Initialize query builder
            self.query_builder = SQLQueryBuilder(
                self.table_name, self.schema_name, dialect="postgres"
            )

            # Ensure table exists
            await self._ensure_table()

            # Check and enable vector support if requested
            if self.vector_enabled:
                await self._detect_vector_support()
        except Exception:
            await _pool_manager.release_pool(self._pool_config)
            self._pool = None
            raise

        self._connected = True
        self.log_operation("connect", f"Connected to table: {self.schema_name}.{self.table_name}")

    async def _create_database(self) -> None:
        """Create the target database via the ``postgres`` maintenance database.

        Called only when pool creation fails with ``InvalidCatalogNameError``,
        indicating the target database does not exist.
        """
        from .postgres_mixins import validate_database_name

        target_db = self._pool_config.database
        # validate_database_name restricts to [a-zA-Z_][a-zA-Z0-9_]* which
        # is safe for double-quoted identifiers in CREATE DATABASE.  asyncpg
        # has no public sql.Identifier equivalent, so validation is the
        # primary defence against injection here.
        validate_database_name(target_db)

        conn = await asyncpg.connect(
            host=self._pool_config.host,
            port=self._pool_config.port,
            user=self._pool_config.user,
            password=self._pool_config.password,
            database="postgres",
            ssl=self._pool_config.ssl,
            timeout=10,
        )
        try:
            await conn.execute(f'CREATE DATABASE "{target_db}"')
            logger.info("Created PostgreSQL database %r", target_db)
        except asyncpg.exceptions.DuplicateDatabaseError:
            pass  # Race condition: another process created it
        finally:
            await conn.close()

    async def close(self) -> None:
        """Release this holder's claim on the shared connection pool.

        Pools are shared by DSN across every ``AsyncPostgresDatabase`` on
        the same event loop (``ConnectionPoolManager`` keys on
        host/port/database/user, not table). ``close()`` is a *release*,
        not a teardown: it decrements the manager's holder count and the
        pool is closed only when the last holder releases. This prevents
        one instance's ``close()`` from closing the pool out from under
        siblings that still hold it.
        """
        if self._pool is not None:
            try:
                await _pool_manager.release_pool(self._pool_config)
            except Exception as e:
                logger.warning("Error releasing connection pool: %s", e)
            # Guarding on ``self._pool is not None`` makes a double-close a
            # no-op (no double-decrement) — release_pool is idempotent for
            # an already-evicted config, but this avoids the call entirely.
            self._pool = None
        self._connected = False

    def _initialize(self) -> None:
        """Initialize is handled in connect."""
        pass

    async def _ensure_table(self) -> None:
        """Ensure the records table exists.

        When ``auto_create_table=True`` (default), runs ``CREATE TABLE IF NOT
        EXISTS …``. When ``auto_create_table=False``, verifies the table is
        present and raises ``RuntimeError`` if it isn't — for consumers
        managing DDL via Alembic / Flyway / Sqitch.
        """
        if not self.auto_create_table:
            exists_sql, params = self.table_manager.get_table_exists_sql()
            async with self._require_pool().acquire() as conn:
                exists = await conn.fetchval(exists_sql, *params)
            if not exists:
                raise RuntimeError(
                    f"Table {self.schema_name}.{self.table_name} does not exist "
                    "and auto_create_table is disabled. Run your migrations "
                    "before starting the application."
                )
            return

        create_table_sql = self.get_create_table_sql(self.schema_name, self.table_name)
        async with self._require_pool().acquire() as conn:
            await conn.execute(create_table_sql)

    async def _detect_vector_support(self) -> None:
        """Detect and enable vector support if pgvector is available.

        The sync twin wraps its probe in ``except Exception``; this one does
        not, and does not need to --- ``install_pgvector_extension`` already
        answers ``False`` for a database that will not take the extension. An
        unconnected database is refused by :meth:`_require_pool` below, with
        the message every other door on this class gives.
        """
        from .postgres_vector import check_pgvector_extension, install_pgvector_extension

        async with self._require_pool().acquire() as conn:
            # Check if pgvector is available
            if await check_pgvector_extension(conn):
                self._vector_enabled = True
                logger.info("pgvector extension detected and enabled")
            else:
                # Try to install it
                if await install_pgvector_extension(conn):
                    self._vector_enabled = True
                    logger.info("pgvector extension installed and enabled")
                else:
                    logger.debug("pgvector extension not available")

    def _check_connection(self) -> None:
        """Check if async database is connected."""
        self._check_async_connection()

    def _require_pool(self) -> asyncpg.Pool:
        """The pool, or the same refusal every other door on this class gives.

        ``_check_async_connection`` does test the pool --- but through
        ``getattr(self, "_pool", None)``, on a mixin that never declares the
        attribute, so nothing narrows it here. That is why all twenty-seven
        ``self._pool.acquire()`` sites read to the type checker as a
        dereference of ``None`` whether or not their method checked first, and
        why the one site that genuinely did not check was indistinguishable
        from the twenty-six that did.

        For a caller that has already passed ``_check_connection()`` the raise
        is unreachable and the return is the narrowing. For
        ``_detect_vector_support`` --- reached from the public
        ``enable_vector_support()``, which checks nothing --- it is the whole
        fix: that path used to answer ``AttributeError: 'NoneType' object has
        no attribute 'acquire'``.

        This tests the pool and not ``_connected``, deliberately: ``connect()``
        runs ``_ensure_table`` and ``_detect_vector_support`` after assigning
        the pool and before setting the flag, so a flag test would refuse the
        connect path itself.
        """
        if self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    def _record_to_row(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to a database row (delegates to shared serializer).

        Mirrors :meth:`SyncPostgresDatabase._record_to_row`: both sync
        and async route through ``SQLRecordSerializer.record_to_row``
        so the outbound ``id`` / ``data`` / ``metadata`` shape lives
        in one place and cannot drift between siblings (the same trap
        that produced the inbound ``_row_to_record`` divergence).
        """
        return SQLRecordSerializer.record_to_row(record, id)

    def _row_to_record(self, row: asyncpg.Record) -> Record:
        """Convert a database row to a Record (delegates to shared serializer).

        Mirrors :meth:`SyncPostgresDatabase._row_to_record`: delegates
        to ``SQLRecordSerializer.row_to_record`` so the
        ``ensure_record_id`` step is applied uniformly. Without the
        delegation, async ``read()`` returned records where
        ``record.id`` / ``record.storage_id`` were whatever was in the
        JSON payload (typically ``None``) — silently differing from
        ``db.read(id)`` on the sync backend for the same on-disk row.
        ``asyncpg.Record`` supports ``dict()`` conversion via the
        mapping protocol; the explicit cast keeps the static helper's
        ``dict[str, Any]`` contract clean.
        """
        return SQLRecordSerializer.row_to_record(dict(row))

    async def create(self, record: Record) -> str:
        """Create a new record with vector support."""
        self._check_connection()

        id = record.id if record.id else self._generate_id()
        row = self._record_to_row(record, id)
        values: list[Any] = [row["id"], row["data"], row["metadata"]]

        sql = f"""
        INSERT INTO {self._q_qualified} (id, data, metadata)
        VALUES ($1, $2, $3)
        """

        try:
            async with self._require_pool().acquire() as conn:
                await conn.execute(sql, *values)
        except asyncpg.exceptions.UniqueViolationError as e:
            raise DuplicateRecordError(id) from e
        except asyncpg.exceptions.IntegrityConstraintViolationError as e:
            raise constraint_violation_error(id) from e

        return id

    async def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()
        sql = f"""
        SELECT id, data, metadata
        FROM {self._q_qualified}
        WHERE id = $1
        """

        async with self._require_pool().acquire() as conn:
            row = await conn.fetchrow(sql, id)

        if not row:
            return None

        return self._row_to_record(row)

    async def get_version(self, id: str) -> str | None:
        """Return the row's ``xmin`` transaction id as the version token.

        ``xmin`` is PostgreSQL's system column holding the id of the
        transaction that last inserted/updated the row; it advances on every
        UPDATE, so it is a native monotonic-per-row version — ABA-safe, unlike
        the base content-hash default this overrides.
        """
        self._check_connection()
        sql = f"""
        SELECT xmin::text AS version
        FROM {self._q_qualified}
        WHERE id = $1
        """
        async with self._require_pool().acquire() as conn:
            version = await conn.fetchval(sql, id)
        return None if version is None else str(version)

    async def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record.

        Args:
            id: The record ID to update
            record: The record data to update with
            expected_version: Optional ``xmin`` token from ``get_version(id)``.
                When provided, the ``UPDATE`` carries an ``AND xmin = …``
                predicate so the compare-and-set is enforced atomically by the
                server; a stale token raises ``ConcurrencyError``. When ``None``
                the update is unconditional, byte-identical to prior behavior.

        Returns:
            True if the record was updated, False if no record with the given ID exists

        Raises:
            ConcurrencyError: If ``expected_version`` does not match the
                record's current ``xmin`` token.
        """
        self._check_connection()

        row = self._record_to_row(record, id)

        set_clauses = ["data = $2", "metadata = $3", "updated_at = CURRENT_TIMESTAMP"]
        values: list[Any] = [id, row["data"], row["metadata"]]

        where = "WHERE id = $1"
        if expected_version is not None:
            where += f" AND xmin::text = ${len(values) + 1}"
            values.append(expected_version)

        sql = f"""
        UPDATE {self._q_qualified}
        SET {", ".join(set_clauses)}
        {where}
        """

        async with self._require_pool().acquire() as conn:
            result = await conn.execute(sql, *values)

        # Returns UPDATE n where n is rows affected
        rows_affected = int(result.split()[-1])

        if rows_affected == 0:
            if expected_version is not None:
                # The atomic UPDATE matched nothing: either the row is gone
                # (update never inserts -> False) or the token is stale
                # (concurrent modification -> raise). The follow-up read only
                # picks the disposition/message; the CAS already happened
                # server-side.
                current = await self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current)
            logger.warning("Update affected 0 rows for id=%s. Record may not exist.", id)

        return rows_affected > 0

    async def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        When ``expected_version`` is provided the ``DELETE`` carries an
        ``AND xmin = …`` predicate so the compare-and-set is enforced
        atomically by the server; a stale token raises ``ConcurrencyError``
        and a missing row returns ``False``. When ``None`` the delete is
        unconditional, byte-identical to prior behavior.
        """
        self._check_connection()

        if expected_version is not None:
            sql = f"""
            DELETE FROM {self._q_qualified}
            WHERE id = $1 AND xmin::text = $2
            """
            async with self._require_pool().acquire() as conn:
                # asyncpg returns the command tag -- "DELETE n" -- as a str.
                # Saying so once, on this method's first binding of ``result``,
                # is what types both branches: asyncpg ships no stubs, so
                # ``conn`` is untyped and the tag arrives as Any otherwise.
                result: str = await conn.execute(sql, id, expected_version)
            rows_affected = int(result.rsplit(maxsplit=1)[-1])
            if rows_affected == 0:
                # The atomic DELETE matched nothing: either the row is gone
                # (delete of an absent id -> False) or the token is stale
                # (concurrent modification -> raise). The follow-up read only
                # picks the disposition; the CAS already happened server-side.
                current = await self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current)
            return True

        sql = f"""
        DELETE FROM {self._q_qualified}
        WHERE id = $1
        """

        async with self._require_pool().acquire() as conn:
            result = await conn.execute(sql, id)

        # Returns DELETE n where n is rows affected
        return result.split()[-1] != "0"

    async def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()
        sql = f"""
        SELECT 1 FROM {self._q_qualified}
        WHERE id = $1
        LIMIT 1
        """

        async with self._require_pool().acquire() as conn:
            row = await conn.fetchrow(sql, id)

        return row is not None

    async def upsert(
        self,
        id_or_record: str | Record,
        record: Record | None = None,
        *,
        expected_version: str | None = None,
    ) -> str:
        """Update or insert a record.

        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic

        When ``expected_version`` is provided the upsert is conditional: the
        record must already exist with a matching ``xmin`` token, otherwise it
        raises ``ConcurrencyError``. A conditional upsert never inserts, so it
        takes the explicit compare-and-set path rather than ``ON CONFLICT``.
        """
        self._check_connection()

        # Resolve the storage id via the shared helper (honors an explicit id;
        # mints via the overridable _generate_id() hook when the record has none).
        id, record = self._resolve_upsert_id(id_or_record, record)

        if expected_version is not None:
            # A conditional upsert never inserts. Delegate to update()'s atomic
            # xmin compare-and-set: a True return is the update; a stale token
            # raises straight out; a False return means the row is absent
            # (decided server-side), which for a conditional upsert is itself a
            # conflict. Acting on the return (not a separate exists() probe)
            # closes the exists()->update() TOCTOU.
            if await self.update(id, record, expected_version=expected_version):
                return id
            raise version_conflict_error(id, expected_version, None)

        row = self._record_to_row(record, id)
        values: list[Any] = [row["id"], row["data"], row["metadata"]]
        update_clauses = [
            "data = EXCLUDED.data",
            "metadata = EXCLUDED.metadata",
            "updated_at = CURRENT_TIMESTAMP",
        ]

        sql = f"""
        INSERT INTO {self._q_qualified} (id, data, metadata)
        VALUES ($1, $2, $3)
        ON CONFLICT (id) DO UPDATE
        SET {", ".join(update_clauses)}
        """

        async with self._require_pool().acquire() as conn:
            await conn.execute(sql, *values)

        return id

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()

        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql, params = self.query_builder.build_complex_search_query(query)
        else:
            sql, params = self.query_builder.build_search_query(query)

        # Execute query with asyncpg (already uses positional parameters)
        async with self._require_pool().acquire() as conn:
            rows = await conn.fetch(sql, *params)

        # Convert to records
        records = []
        for row in rows:
            record = self._row_to_record(row)

            # Apply field projection if specified
            if query.fields:
                record = record.project(query.fields)

            records.append(record)

        return records

    async def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()
        sql = f"SELECT COUNT(*) as count FROM {self._q_qualified}"

        async with self._require_pool().acquire() as conn:
            row = await conn.fetchrow(sql)

        return row["count"] if row else 0

    async def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()
        # Get count first
        count = await self._count_all()

        # Delete all records
        sql = f"TRUNCATE TABLE {self._q_qualified}"

        async with self._require_pool().acquire() as conn:
            await conn.execute(sql)

        return count

    def supports_transactions(self) -> bool:
        """Postgres batch ops are atomic (single multi-row DML statement)."""
        return True

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[Any]:
        """Pin one pooled connection in a transaction spanning a commit flush.

        Acquires a single connection from the pool and opens a
        ``conn.transaction()`` on it, yielding the connection as the handle. The
        batch methods run their DML on this exact connection (via
        :meth:`_acquire`) and skip opening a nested transaction, so a multi-kind
        buffered-transaction flush commits (or rolls back) as one unit. Pinning
        the connection is why the batch methods **must** route through the
        threaded handle: a fall-through to a second ``_require_pool().acquire()`` while
        this one is held would deadlock a size-1 pool.
        """
        self._check_connection()
        async with self._require_pool().acquire() as conn:
            async with conn.transaction():
                yield conn

    @asynccontextmanager
    async def _acquire(self, _tx: Any) -> AsyncIterator[Any]:
        """Yield the threaded flush connection, or acquire a fresh pooled one.

        When ``_tx`` is supplied (inside a :meth:`_transaction` flush) it is the
        pinned connection — yield it directly and never acquire a second one
        (the size-1-pool deadlock trap). When ``None`` (the direct path) acquire
        a fresh pooled connection for the batch's own implicit transaction, the
        pre-existing behavior.
        """
        if _tx is not None:
            yield _tx
        else:
            async with self._require_pool().acquire() as conn:
                yield conn

    async def create_batch(self, records: list[Record], *, _tx: Any = None) -> list[str]:
        """Create multiple records efficiently using a single query.

        Uses multi-value INSERT with RETURNING for better performance.

        Args:
            records: List of records to create
            _tx: Internal. When supplied (a multi-kind buffered-transaction
                flush), the DML runs on that pinned connection inside the outer
                :meth:`_transaction`; when ``None`` a fresh pooled connection
                runs it as an implicit transaction (the pre-existing path).

        Returns:
            List of created record IDs
        """
        if not records:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder

        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Use the shared batch create query builder (honors record.id, mints via
        # _generate_id; raises DuplicateRecordError up front on a within-batch
        # duplicate id).
        query, params, ids = query_builder.build_batch_create_query(
            records, id_factory=self._generate_id
        )

        # Execute the batch insert with RETURNING. Like create(), a colliding id
        # fails closed: the single INSERT is atomic, so the unique-violation
        # aborts the whole batch (nothing written).
        try:
            async with self._acquire(_tx) as conn:
                rows = await conn.fetch(query, *params)
        except asyncpg.exceptions.UniqueViolationError as e:
            colliding = ids[0]
            # Precise colliding-id naming needs a probe on a *fresh* connection.
            # That is safe only off the pinned-tx path: inside a flush the pool
            # connection is held (a probe could exhaust a size-1 pool) and the
            # aborted transaction cannot be queried anyway. On the ``_tx`` path
            # report the first batch id and let the outer transaction roll back.
            if _tx is None:
                for record in records:
                    if record.id and await self.exists(record.id):
                        colliding = record.id
                        break
            raise DuplicateRecordError(colliding) from e
        except asyncpg.exceptions.IntegrityConstraintViolationError as e:
            raise constraint_violation_error() from e

        # Return the actual inserted IDs from RETURNING clause
        if rows:
            return [row["id"] for row in rows]
        return ids  # Fallback to generated IDs

    async def upsert_batch(self, records: list[Record], *, _tx: Any = None) -> list[str]:
        """Insert-or-overwrite multiple records in a single statement.

        Uses ``INSERT ... ON CONFLICT (id) DO UPDATE``. Honors a caller-supplied
        ``record.id`` (minting a uuid only when absent); a colliding id is
        overwritten (never raised). Returns ids in input order. When ``_tx`` is
        supplied the DML runs on the pinned flush connection (see
        :meth:`create_batch`).
        """
        if not records:
            return []

        self._check_connection()

        from .sql_base import SQLQueryBuilder

        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")
        query, params, ids = query_builder.build_batch_upsert_query(
            records, id_factory=self._generate_id
        )

        async with self._acquire(_tx) as conn:
            await conn.execute(query, *params)

        # RETURNING order is not guaranteed under ON CONFLICT, so return the
        # builder's input-order ids.
        return ids

    async def delete_batch(self, ids: list[str], *, _tx: Any = None) -> list[bool]:
        """Delete multiple records efficiently using a single query.

        Uses single DELETE with IN clause and RETURNING for verification.

        Args:
            ids: List of record IDs to delete
            _tx: Internal. When supplied the DML runs on the pinned flush
                connection (see :meth:`create_batch`).

        Returns:
            List of success flags for each deletion
        """
        if not ids:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder

        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Use the shared batch delete query builder
        query, params = query_builder.build_batch_delete_query(ids)

        # Execute the batch delete with RETURNING
        async with self._acquire(_tx) as conn:
            rows = await conn.fetch(query, *params)

        # Convert returned rows to set of deleted IDs
        deleted_ids = {row["id"] for row in rows}

        # Return results for each deletion
        results = []
        for id in ids:
            results.append(id in deleted_ids)

        return results

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using a single query.

        Uses PostgreSQL's CASE expressions for batch updates with native asyncpg.

        Args:
            updates: List of (id, record) tuples to update

        Returns:
            List of success flags for each update
        """
        if not updates:
            return []

        self._check_connection()

        # Create a query builder for PostgreSQL
        from .sql_base import SQLQueryBuilder

        query_builder = SQLQueryBuilder(self.table_name, self.schema_name, dialect="postgres")

        # Use the shared batch update query builder. It already
        # produces positional parameters ($1, $2) AND appends
        # ``RETURNING id`` when ``dialect="postgres"`` -- see
        # ``SQLQueryBuilder.build_batch_update_query`` -- so do NOT append
        # a second ``RETURNING id`` here, that produces invalid SQL.
        query, params = query_builder.build_batch_update_query(updates)

        # Execute the batch update
        async with self._require_pool().acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Convert returned rows to set of updated IDs
        updated_ids = {row["id"] for row in rows}

        # Return results for each update
        results = []
        for record_id, _ in updates:
            results.append(record_id in updated_ids)

        return results

    async def _vector_search(
        self,
        query_vector: np.ndarray | list[float] | VectorField,
        *,
        vector_field: str,
        k: int,
        metric: DistanceMetric,
        filter: Query | None,
    ) -> list[VectorSearchResult]:
        """Raw k-NN through pgvector's distance operators.

        The threshold and the source assembly are the mixin's; this is the
        part that knows how to ask pgvector.

        Reads the vector out of the JSON ``data`` column, the same storage
        :meth:`SyncPostgresDatabase._vector_search` reads and the same
        storage every write path on both twins fills. This method used to
        read a dedicated ``vector_<field>`` column that only ``create``,
        ``update`` and ``upsert`` on this class ever wrote --- so its own
        ``create_batch`` produced records it could not find, and a
        sync-written corpus was invisible to it entirely.

        Args:
            query_vector: Query vector (numpy array, list, or VectorField)
            vector_field: Name of the vector field to search, read out of the
                JSON ``data`` column
            k: Maximum number of results to return
            metric: Distance metric, already resolved by the mixin
            filter: Optional query filter to apply before vector search

        Returns:
            List of VectorSearchResult objects ordered by similarity
        """
        if not self._vector_enabled:
            raise RuntimeError("Vector search not available - pgvector not installed")

        self._check_connection()

        from ..vector.types import VectorSearchResult
        from .postgres_vector import build_vector_search_sql, distance_to_score

        values = _query_vector_values(query_vector)
        params: list[Any] = [format_vector_for_postgres(values), vector_field]

        # This class builds its query builder without a ``param_style``, which
        # defaults to ``"numeric"`` --- so the clause already carries ``$3``,
        # ``$4``, ... and needs no rewriting. It was being rewritten anyway,
        # by a loop replacing ``%s`` in a string that has never contained one,
        # under a comment asserting the opposite. The numbering is the
        # builder's, from the start index passed here: $1 and $2 are the
        # vector and the field name.
        filter_clause = ""
        if filter:
            filter_clause, filter_params = self.query_builder.build_where_clause(
                filter, len(params) + 1
            )
            params.extend(filter_params)

        sql = build_vector_search_sql(
            q_qualified=self._q_qualified,
            vector_field=vector_field,
            dimensions=len(values),
            metric=metric,
            vector_placeholder="$1",
            field_placeholder="$2",
            # Interpolated rather than bound: this is the one integer in the
            # statement and binding it would renumber every filter parameter
            # behind it. ``k`` reaches here from the mixin's declared ``int``.
            limit_clause=f"LIMIT {int(k)}",
            filter_clause=filter_clause,
        )

        async with self._require_pool().acquire() as conn:
            rows = await conn.fetch(sql, *params)

        # A ``NULL`` distance survives the presence predicate in one shape:
        # a ``VectorField`` object stored without its ``value`` key, where
        # ``->>'value'`` yields NULL and the cast carries it through. Dropping
        # the row is the graceful answer; ``float(None)`` would fail the whole
        # search over one malformed record.
        return [
            VectorSearchResult(
                record=self._row_to_record(row),
                score=distance_to_score(metric, float(row["distance"])),
                vector_field=vector_field,
                metadata={"distance": float(row["distance"]), "metric": metric.value},
            )
            for row in rows
            if row["distance"] is not None
        ]

    async def enable_vector_support(self) -> bool:
        """Enable vector support for this database.

        Returns:
            True if vector support is enabled
        """
        if self._vector_enabled:
            return True

        await self._detect_vector_support()
        return self._vector_enabled

    async def has_vector_support(self) -> bool:
        """Check if this database has vector support enabled.

        Returns:
            True if vector support is available
        """
        return self._vector_enabled

    async def create_vector_index(
        self,
        vector_field: str = "embedding",
        dimensions: int | None = None,
        metric: DistanceMetric | str | None = None,
        index_type: str = "ivfflat",
        lists: int | None = None,
    ) -> bool:
        """Create a vector index for efficient similarity search.

        ``dimensions`` is declared optional because the mixin declares it so,
        and is then refused when absent because pgvector cannot index a
        column of unfixed width --- the cast this builds is what gives the
        expression a width at all. Stating that as a refusal rather than as a
        required argument is what lets a caller written against the mixin
        reach this method and be told why, instead of being turned away by
        the signature on every backend at once.

        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions in the vectors. Required here,
                unlike on the backends that ignore it.
            metric: Distance metric for the index
            index_type: Type of index (ivfflat, hnsw)
            lists: Number of lists for IVFFlat index

        Returns:
            True if index was created successfully
        """
        from .postgres_vector import (
            build_vector_index_sql,
            build_vector_value_expression,
            get_optimal_index_type,
            get_vector_count_sql,
        )

        self._check_connection()

        if not self._vector_enabled:
            return False

        if dimensions is None:
            raise ValueError(_DIMENSIONS_REQUIRED)

        # Determine optimal parameters if not provided
        if not lists and index_type == "ivfflat":
            # Count vectors to determine optimal lists
            count_sql = get_vector_count_sql(self._q_schema, self._q_table, vector_field)
            async with self._require_pool().acquire() as conn:
                count = await conn.fetchval(count_sql) or 0
                _, params = get_optimal_index_type(count)
                lists = params.get("lists", 100)

        # The expression the searches use, so the planner sees one expression
        # rather than two. This built ``(data->'f'->>'value')::vector(n)``
        # while both searches asked a ``CASE`` that also tolerates a bare
        # array, which no query could match.
        index_sql = build_vector_index_sql(
            q_table_name=self._q_table,
            q_schema_name=self._q_schema,
            column_name=build_vector_value_expression(vector_field, dimensions),
            dimensions=dimensions,
            metric=resolve_metric(self, metric),
            index_type=index_type,
            index_params={"lists": lists} if lists else None,
            field_name=vector_field,
        )

        # Create the index
        try:
            logger.debug(f"Creating vector index with SQL: {index_sql}")
            async with self._require_pool().acquire() as conn:
                await conn.execute(index_sql)
            return True
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")
            logger.debug(f"Index SQL was: {index_sql}")
            return False

    async def drop_vector_index(
        self, vector_field: str = "embedding", metric: DistanceMetric | str | None = None
    ) -> bool:
        """Drop a vector index.

        Args:
            vector_field: Name of the vector field
            metric: Distance metric used in the index

        Returns:
            True if index was dropped successfully
        """
        from .postgres_vector import get_vector_index_name

        self._check_connection()

        index_name = get_vector_index_name(
            self.table_name, vector_field, resolve_metric(self, metric).value
        )

        try:
            async with self._require_pool().acquire() as conn:
                await conn.execute(
                    f"DROP INDEX IF EXISTS {self._q_schema}.{quote_ident(index_name)}"
                )
            return True
        except Exception as e:
            logger.warning(f"Failed to drop vector index: {e}")
            return False

    async def get_vector_index_stats(self, vector_field: str = "embedding") -> dict[str, Any]:
        """Get statistics about a vector field and its index.

        Args:
            vector_field: Name of the vector field

        Returns:
            Dictionary with index statistics
        """
        from .postgres_vector import get_index_check_sql, get_vector_count_sql

        self._check_connection()

        stats: dict[str, Any] = {
            "field": vector_field,
            "indexed": False,
            "vector_count": 0,
        }

        try:
            async with self._require_pool().acquire() as conn:
                # Count vectors
                count_sql = get_vector_count_sql(self._q_schema, self._q_table, vector_field)
                stats["vector_count"] = await conn.fetchval(count_sql) or 0

                # Check for index — raw (unquoted) names are correct here:
                # get_index_check_sql queries pg_indexes using $1/$2 parameterized
                # bindings against catalog text columns (schemaname, tablename), not
                # via identifier interpolation, so quoting would produce wrong matches.
                index_sql, params = get_index_check_sql(
                    self.schema_name, self.table_name, vector_field
                )
                stats["indexed"] = await conn.fetchval(index_sql, *params) or False
        except Exception as e:
            # See the sync twin: the defaults are indistinguishable from a
            # successful "nothing here" without this.
            logger.warning(f"Failed to get vector index stats: {e}")
            stats["error"] = str(e)

        return stats

    async def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from PostgreSQL using cursor."""
        # Pre-flight the field grammar -- see the sync twin.
        if query and query.filters:
            for f in query.filters:
                validate_field_path(f.field)
        self._check_connection()
        config = config or StreamConfig()

        # Build SQL query
        sql = f"SELECT id, data, metadata FROM {self._q_qualified}"
        params = []

        # Through the same builder ``search`` uses -- see the sync twin for the
        # half they shared. This one also had a louder half of its own: the
        # placeholder counter advanced once per *filter* while ``params`` grew
        # only for EQ, so one non-EQ filter ahead of an EQ one shifted every
        # later placeholder past its argument and the SQL named a ``$N``
        # nothing had bound.
        where_clause, filter_params = self.query_builder.build_where_clause(query)
        if where_clause:
            sql += " WHERE TRUE" + where_clause
            params.extend(filter_params)

        # Use cursor for efficient streaming. asyncpg's
        # ``conn.cursor(sql, *args)`` returns a ``CursorFactory`` that
        # supports ``async for``; ``await``-ing it returns a ``Cursor``
        # object intended for the explicit-fetch API
        # (``await cur.fetch(n)``) and is NOT an async iterator.
        async with self._require_pool().acquire() as conn:
            async with conn.transaction():
                batch = []
                async for row in conn.cursor(sql, *params):
                    record = self._row_to_record(row)
                    if query and query.fields:
                        record = record.project(query.fields)

                    batch.append(record)

                    if len(batch) >= config.batch_size:
                        for rec in batch:
                            yield rec
                        batch = []

                # Yield remaining records
                for rec in batch:
                    yield rec

    async def stream_write(
        self, records: AsyncIterator[Record], config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into PostgreSQL using batch inserts.

        Honors ``config.on_conflict``: INSERT uses the COPY batch fast-path with
        a ``create`` per-record fallback; UPSERT/SKIP write per-record via
        ``upsert``/``create``.
        """
        self._check_connection()
        config = config or StreamConfig()

        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            # Passed directly, as the sync twin passes it: _write_batch already
            # returns the authoritative ids, minting a uuid where record.id is
            # absent. The wrapper this replaces existed to return
            # [r.id for r in b] -- None for an id-less record -- and once that
            # was corrected it forwarded and did nothing else.
            insert_batch_func=self._write_batch,
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

    async def _write_batch(self, records: list[Record]) -> list[str]:
        """Write a batch of records using COPY for performance.

        The INSERT fast-path for the streaming INSERT policy. Like ``create()``,
        it honors a caller-supplied ``record.id`` (minting only when absent) and
        fails closed on a colliding id — ``COPY`` runs in one transaction, so the
        unique-violation aborts the whole batch and the streaming loop falls back
        to per-record ``create()`` to attribute the specific collision.

        Returns:
            List of created record IDs
        """
        if not records:
            return []

        # Prepare data for COPY; honor record.id, mint via _generate_id only
        # when absent (uniform with create() and the sync _write_batch).
        rows = []
        ids = []
        for record in records:
            record_id = record.id or self._generate_id()
            row_data = self._record_to_row(record, record_id)
            ids.append(row_data["id"])
            rows.append((row_data["id"], row_data["data"], row_data["metadata"]))

        # Use COPY for efficient bulk insert
        try:
            async with self._require_pool().acquire() as conn:
                await conn.copy_records_to_table(
                    self.table_name,
                    schema_name=self.schema_name or None,
                    records=rows,
                    columns=["id", "data", "metadata"],
                )
        except asyncpg.exceptions.UniqueViolationError as e:
            colliding = ids[0]
            for record in records:
                if record.id and await self.exists(record.id):
                    colliding = record.id
                    break
            raise DuplicateRecordError(colliding) from e
        except asyncpg.exceptions.IntegrityConstraintViolationError as e:
            raise constraint_violation_error() from e

        return ids

    async def _supports_native_hybrid(self) -> bool:
        """Check if this PostgreSQL backend supports native hybrid search.

        PostgreSQL with pgvector and full-text search (tsvector) supports
        native hybrid search.

        Returns:
            True if vector support is enabled (pgvector available)
        """
        return self._vector_enabled

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray | list[float],
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        k: int = 10,
        config: Any = None,  # HybridSearchConfig
        filter: Query | None = None,
        metric: DistanceMetric | str | None = None,
    ) -> list[Any]:  # list[HybridSearchResult]
        """Perform hybrid search using PostgreSQL full-text search and pgvector.

        Combines PostgreSQL's tsvector full-text search with pgvector similarity
        search using configurable fusion strategies.

        Args:
            query_text: Text query for full-text matching
            query_vector: Vector for pgvector similarity search
            text_fields: Fields to search for text matching
            vector_field: Name of the vector field to search
            k: Number of results to return
            config: Hybrid search configuration (weights, fusion strategy)
            filter: Optional additional filters to apply
            metric: Distance metric for vector search

        Returns:
            List of HybridSearchResult ordered by combined score (descending)
        """
        from ..vector.hybrid import (
            FusionStrategy,
            HybridSearchConfig,
            HybridSearchResult,
            reciprocal_rank_fusion,
        )
        from .postgres_vector import build_hybrid_search_sql, distance_to_score

        self._check_connection()

        config = config or HybridSearchConfig()

        # For NATIVE strategy with pgvector, we can do a combined query
        # For other strategies, use the parent implementation
        if config.fusion_strategy not in (FusionStrategy.NATIVE, FusionStrategy.RRF):
            from ..vector.mixins import AsyncVectorOperationsMixin

            return await AsyncVectorOperationsMixin.hybrid_search(
                self,
                query_text=query_text,
                query_vector=query_vector,
                text_fields=text_fields,
                vector_field=vector_field,
                k=k,
                config=config,
                filter=filter,
                metric=metric,
            )

        # Use config.text_fields if provided, otherwise use parameter
        search_text_fields = config.text_fields or text_fields or ["content", "title", "text"]

        # Get more results for fusion
        fetch_k = min(k * 3, 100)

        # Prepare vector search
        if isinstance(query_vector, (list, tuple)):
            import numpy as np

            query_vector = np.array(query_vector, dtype=np.float32)

        vector_str = format_vector_for_postgres(query_vector)

        resolved = resolve_metric(self, metric)

        params: list[Any] = [query_text, vector_str, vector_field]

        # Applied to both arms, or to neither. ``filter`` was declared, bound
        # into nothing, and never consulted: the parameter list was built as
        # the three above and stopped, so a filtered hybrid search read the
        # whole table and said nothing. Fusing a filtered ranking with an
        # unfiltered one would be a second way to get the same wrong answer,
        # which is why the clause goes in both.
        filter_clause = ""
        if filter:
            filter_clause, filter_params = self.query_builder.build_where_clause(
                filter, len(params) + 1
            )
            params.extend(filter_params)

        # The JSON ``data`` column, as both ``_vector_search`` twins now read
        # --- this CTE read the ``vector_<field>`` column, so a hybrid search
        # over a batch-written or sync-written corpus contributed no vector
        # half at all and silently degraded to a text search. Built by the
        # same function rather than restated here, which is what stops it
        # drifting away from them again.
        sql = build_hybrid_search_sql(
            q_qualified=self._q_qualified,
            text_concat=self._build_text_field_concat(search_text_fields),
            vector_field=vector_field,
            dimensions=len(query_vector),
            metric=resolved,
            fetch_k=fetch_k,
            text_placeholder="$1",
            vector_placeholder="$2",
            field_placeholder="$3",
            filter_clause=filter_clause,
        )

        try:
            async with self._require_pool().acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except Exception as e:
            # If full-text search fails, fall back to client-side fusion
            logger.warning(
                f"Native PostgreSQL hybrid search failed ({e}), falling back to client-side"
            )
            from ..vector.mixins import AsyncVectorOperationsMixin

            return await AsyncVectorOperationsMixin.hybrid_search(
                self,
                query_text=query_text,
                query_vector=query_vector,
                text_fields=text_fields,
                vector_field=vector_field,
                k=k,
                config=HybridSearchConfig(
                    text_weight=config.text_weight,
                    vector_weight=config.vector_weight,
                    fusion_strategy=FusionStrategy.RRF,
                    rrf_k=config.rrf_k,
                    text_fields=config.text_fields,
                ),
                filter=filter,
                metric=metric,
            )

        # Build result lists for fusion
        records_by_id: dict[str, Record] = {}
        text_scores: list[tuple[str, float]] = []
        vector_scores: list[tuple[str, float]] = []

        for row in rows:
            record = self._row_to_record(row)
            record_id = row["id"]
            records_by_id[record_id] = record

            if row["text_score"] is not None:
                text_scores.append((record_id, float(row["text_score"])))
            if row["vector_distance"] is not None:
                # Converted here rather than in SQL, where it was a hardcoded
                # ``1.0 - distance`` that is the cosine formula applied to
                # whichever metric the caller asked for.
                vector_scores.append(
                    (record_id, distance_to_score(resolved, float(row["vector_distance"])))
                )

        # Sort by score for rank-based fusion
        text_scores.sort(key=lambda x: x[1], reverse=True)
        vector_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply RRF fusion
        fused = reciprocal_rank_fusion(
            text_results=text_scores,
            vector_results=vector_scores,
            k=config.rrf_k,
            text_weight=config.text_weight,
            vector_weight=config.vector_weight,
        )

        # Build HybridSearchResult objects
        text_score_map = dict(text_scores)
        vector_score_map = dict(vector_scores)
        text_rank_map = {rid: i + 1 for i, (rid, _) in enumerate(text_scores)}
        vector_rank_map = {rid: i + 1 for i, (rid, _) in enumerate(vector_scores)}

        results: list[HybridSearchResult] = []
        for record_id, combined_score in fused[:k]:
            if record_id not in records_by_id:
                continue

            results.append(
                HybridSearchResult(
                    record=records_by_id[record_id],
                    combined_score=combined_score,
                    text_score=text_score_map.get(record_id),
                    vector_score=vector_score_map.get(record_id),
                    text_rank=text_rank_map.get(record_id),
                    vector_rank=vector_rank_map.get(record_id),
                    metadata={
                        "fusion_strategy": config.fusion_strategy.value,
                        "text_weight": config.text_weight,
                        "vector_weight": config.vector_weight,
                        "backend": "postgresql",
                    },
                )
            )

        return results

    def _build_text_field_concat(self, text_fields: list[str]) -> str:
        """Build SQL expression to concatenate text fields for full-text search.

        Args:
            text_fields: List of field names to concatenate

        Returns:
            SQL expression for concatenated text fields
        """
        if not text_fields:
            return "COALESCE(data->>'content', '')"

        for field in text_fields:
            validate_field_name(field)

        parts = [f"COALESCE(data->>'{field}', '')" for field in text_fields]
        return " || ' ' || ".join(parts)

    async def _text_search_for_hybrid(
        self,
        query_text: str,
        text_fields: list[str] | None,
        k: int,
        filter: Query | None = None,
    ) -> list[tuple[Record, float]]:
        """Perform PostgreSQL full-text search for hybrid search fusion.

        Uses PostgreSQL's tsvector/tsquery full-text search with ts_rank_cd scoring.

        Args:
            query_text: Text to search for
            text_fields: Fields to search in
            k: Maximum results to return
            filter: Additional filters

        Returns:
            List of (record, score) tuples ordered by text relevance
        """
        self._check_connection()

        search_fields = text_fields or ["content", "title", "text"]
        text_concat = self._build_text_field_concat(search_fields)

        sql = f"""
        SELECT
            id,
            data,
            metadata,
            ts_rank_cd(
                to_tsvector('english', {text_concat}),
                plainto_tsquery('english', $1)
            ) as score
        FROM {self._q_qualified}
        WHERE to_tsvector('english', {text_concat}) @@ plainto_tsquery('english', $1)
        ORDER BY score DESC
        LIMIT {k}
        """

        try:
            async with self._require_pool().acquire() as conn:
                rows = await conn.fetch(sql, query_text)
        except Exception as e:
            # Fall back to LIKE-based search if full-text search fails
            logger.warning(f"PostgreSQL full-text search failed ({e}), falling back to LIKE")
            from ..vector.mixins import AsyncVectorOperationsMixin

            return await AsyncVectorOperationsMixin._text_search_for_hybrid(
                self,
                query_text=query_text,
                text_fields=text_fields,
                k=k,
                filter=filter,
            )

        # Normalize scores
        results: list[tuple[Record, float]] = []
        max_score = max((float(row["score"]) for row in rows), default=1.0) or 1.0

        for row in rows:
            record = self._row_to_record(row)
            score = float(row["score"]) / max_score if max_score > 0 else 0.0
            results.append((record, score))

        return results

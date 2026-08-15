"""SQL database utility functions and connection management.

Provides utilities for working with SQL databases including PostgreSQL,
with support for connection management, query execution, and data loading.
"""

from __future__ import annotations

import contextlib
import operator
import os
import threading
import weakref
from abc import ABC, abstractmethod
from types import TracebackType
from typing import IO, Any, Dict, List, Self

import numpy as np
import pandas as pd
import psycopg2

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without python-dotenv
    # dotenv is optional. The stub takes the real function's parameters and
    # returns its type: a narrower stand-in is a different function under the
    # same name, and which one a caller gets depends on what happens to be
    # installed.
    def load_dotenv(
        dotenv_path: str | os.PathLike[str] | None = None,
        stream: IO[str] | None = None,
        verbose: bool = False,
        override: bool = False,
        interpolate: bool = True,
        encoding: str | None = "utf-8",
    ) -> bool:
        return False


from dataknobs_common.lifecycle import close_if_owned_sync

from dataknobs_utils.sys_utils import load_project_vars


CALLER_LANE = "caller"
"""The connection :meth:`PostgresDB.get_conn` hands to a caller.

A caller owns the transaction on the connection it is given, so nothing else
may commit or roll it back. See :attr:`INTERNAL_LANE`.
"""

INTERNAL_LANE = "internal"
"""The connection ``query`` / ``execute`` / ``upload`` use among themselves.

Kept apart from :attr:`CALLER_LANE` because a transaction belongs to a
*connection*: those methods each enter ``with conn``, which commits on the way
out, so sharing one connection with a caller would make an ordinary ``query``
commit whatever that caller had left open — with nothing nested and nothing
raised. A caller that never calls ``get_conn`` opens one connection per thread,
exactly as before.
"""


def quote_ident(name: str, dialect: str = "postgres") -> str:
    """Return *name* as a double-quoted SQL identifier.

    All three supported dialects (``postgres``, ``sqlite``, ``duckdb``)
    follow the same SQL-standard rule: surround the name with ``"``,
    escaping any internal ``"`` as ``""``.

    Raises ``ValueError`` for empty or non-string input. Does **not**
    split qualified names — ``"schema.table"`` is treated as a single
    identifier that produces ``'"schema.table"'``. Splitting a qualified
    name into parts and quoting each is the caller's responsibility.

    Not for test code — use ``dataknobs_common.testing.safe_sql_ident``
    for test-fixture identifier validation instead.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    _supported = {"postgres", "sqlite", "duckdb"}
    if dialect not in _supported:
        raise ValueError(f"Unsupported dialect: {dialect!r}. Supported: {_supported}")
    # Standard SQL double-quoting rule (correct for postgres, sqlite, duckdb)
    return '"' + name.replace('"', '""') + '"'


class RecordFetcher(ABC):
    """Abstract base class for fetching records from a data source.

    Provides a common interface for retrieving records by ID from various
    data sources (databases, DataFrames, dictionaries, etc.) with support
    for zero-based and one-based ID systems.

    Attributes:
        id_field_name: Name of the ID field in the data source.
        fields_to_retrieve: Subset of fields to retrieve (None for all).
        one_based: True if data source uses 1-based IDs.
    """

    def __init__(
        self,
        id_field_name: str = "id",
        fields_to_retrieve: List[str] | None = None,
        one_based_ids: bool = False,
    ) -> None:
        """Initialize the record fetcher.

        Args:
            id_field_name: Name of the integer ID field. Defaults to "id".
            fields_to_retrieve: Subset of fields to retrieve. If None, retrieves
                all fields. Defaults to None.
            one_based_ids: True if data source uses 1-based IDs, False for 0-based.
                Defaults to False.
        """
        self.id_field_name = id_field_name
        self.fields_to_retrieve = fields_to_retrieve
        self.one_based = one_based_ids

    @abstractmethod
    def get_records(
        self, ids: List[int], one_based: bool = False, fields_to_retrieve: List[str] | None = None
    ) -> pd.DataFrame:
        """Fetch records by ID from the data source.

        Args:
            ids: Collection of record IDs to retrieve.
            one_based: True if the provided IDs are 1-based. Defaults to False.
            fields_to_retrieve: Subset of fields for this call, overriding
                instance default. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved records.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class DotenvPostgresConnector:
    """PostgreSQL connection manager using environment variables and project vars.

    Loads database connection parameters from environment variables (.env),
    project variables file, or constructor arguments, with environment variables
    taking precedence.

    Attributes:
        host: Database host address.
        database: Database name.
        user: Database username.
        password: Database password.
        port: Database port number.
    """

    def __init__(
        self,
        host: str | None = None,
        db: str | None = None,
        user: str | None = None,
        pwd: str | None = None,
        port: int | None = None,
        pvname: str = ".project_vars",
        sslmode: str | None = None,
        validate_on_reuse: bool = True,
    ) -> None:
        """Initialize PostgreSQL connector with environment-based configuration.

        Args:
            host: Database host. If None, uses POSTGRES_HOST environment variable
                or "localhost". Defaults to None.
            db: Database name. If None, uses POSTGRES_DB environment variable
                or "postgres". Defaults to None.
            user: Username. If None, uses POSTGRES_USER environment variable
                or "postgres". Defaults to None.
            pwd: Password. If None, uses POSTGRES_PASSWORD environment variable.
                Defaults to None.
            port: Port number. If None, uses POSTGRES_PORT environment variable
                or 5432. Defaults to None.
            pvname: Project variables filename to load. Defaults to ".project_vars".
            sslmode: psycopg2 ``sslmode`` (e.g. ``"require"``, ``"verify-full"``,
                ``"disable"``). When None (the default), no ``sslmode`` is passed
                and libpq's own default applies — preserving prior behavior.
            validate_on_reuse: Whether to confirm a cached connection is still
                alive before handing it back, with a ``SELECT 1``. Defaults to
                True. A server-side drop leaves ``connection.closed`` reading 0
                and cannot be detected locally, so without this a connection
                idle long enough to be dropped fails on the next statement.
                Costs 0.29 ms against the 4.1 ms handshake reuse saves; set
                False to trade correctness for that last 7% where connections
                are known not to sit idle.
        """
        self.sslmode = sslmode
        self.validate_on_reuse = validate_on_reuse
        config = load_project_vars(pvname=pvname)
        if host is None or db is None or user is None or pwd is None or port is None:
            load_dotenv()

        self.host = (
            os.getenv(
                "POSTGRES_HOST", config.get("POSTGRES_HOST", "localhost") if config else "localhost"
            )
            if host is None
            else host
        )
        self.database = (
            os.getenv(
                "POSTGRES_DB", config.get("POSTGRES_DB", "postgres") if config else "postgres"
            )
            if db is None
            else db
        )
        self.user = (
            os.getenv(
                "POSTGRES_USER", config.get("POSTGRES_USER", "postgres") if config else "postgres"
            )
            if user is None
            else user
        )
        self.password = (
            os.getenv(
                "POSTGRES_PASSWORD", config.get("POSTGRES_PASSWORD", None) if config else None
            )
            if pwd is None
            else pwd
        )
        self.port = (
            int(os.getenv("POSTGRES_PORT", config.get("POSTGRES_PORT", 5432) if config else 5432))
            if port is None
            else port
        )
        # Per thread, not per connector — see get_conn. The registry beside it
        # is what lets close() reach a connection another thread opened.
        #
        # Weak, so that reachability does not become ownership. A strong set
        # would keep every connection alive for the life of the connector, and
        # a worker thread that opens one and exits would strand its backend
        # with nothing able to reach it — the caller's frame is gone and only
        # an explicit close() remains. Holding weakly restores the reclamation
        # that made the pre-reuse code safe: the thread-local slot dies with
        # the thread, the last strong reference goes with it, and psycopg2's
        # dealloc closes the socket. close() still sees every connection whose
        # thread is alive, which is the set it needs to reach.
        self._local = threading.local()
        self._open_conns: weakref.WeakSet[Any] = weakref.WeakSet()
        self._conns_lock = threading.Lock()

    def _is_usable(self, conn: Any) -> bool:
        """Whether ``conn`` can still carry a statement.

        ``connection.closed`` answers a narrower question than it appears to:
        it reports what *this process* did to the connection. psycopg2 sets it
        when it closes the connection itself, or when it detects a broken one
        during an operation — so a backend killed by ``pg_terminate_backend``,
        an idle timeout, or a pooler eviction leaves it reading 0 until
        something tries to use the connection and fails.

        **There is no correct local answer.** A readable socket means EOF, an
        error, *or* data, and the three are indistinguishable without reading
        the wire protocol: a terminated backend sends an error message before
        closing, so peeking at the first byte finds a byte there exactly as a
        pending ``NOTIFY`` would. A check built on readability alone therefore
        discards healthy connections — dropping a ``LISTEN`` subscription with
        them — while a check built on ``conn.closed`` alone misses the case it
        exists for. Only a round trip separates them.

        So the connection is asked. ``SELECT 1`` costs 0.29 ms against the
        4.1 ms handshake it saves, leaving reuse ~93% ahead of reconnecting per
        call, and it is unambiguous. Set ``validate_on_reuse=False`` to trade
        that back for the last 7% where a caller knows its connections are not
        idle long enough to be dropped.

        The probe runs under ``autocommit`` when the connection is idle, so it
        does not open a transaction that nothing closes — a connection handed
        back idle-in-transaction holds a snapshot and blocks VACUUM. When the
        caller already has a transaction open the probe joins it and no
        autocommit change is made, because switching mid-transaction is both an
        error and none of this method's business.
        """
        if conn is None or conn.closed:
            return False
        if not self.validate_on_reuse:
            return True
        status = conn.get_transaction_status()  # local; no round trip
        if status == psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN:
            return False
        if status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            # Live, and inside a transaction the caller has to unwind itself.
            # Replacing it here would discard their work without telling them.
            return True
        was_autocommit = conn.autocommit
        idle = status == psycopg2.extensions.TRANSACTION_STATUS_IDLE
        try:
            if idle:
                conn.autocommit = True
            with conn.cursor() as curs:
                curs.execute("SELECT 1")
            return True
        except psycopg2.Error:
            return False
        finally:
            if idle:
                with contextlib.suppress(psycopg2.Error):
                    conn.autocommit = was_autocommit

    def get_conn(self, lane: str = CALLER_LANE) -> Any:
        """Return this thread's PostgreSQL connection, opening it if needed.

        The connection is **reused within a thread**. It used to be built fresh
        on every call, which put a full TCP+auth handshake in front of every
        ``query`` / ``execute`` / ``upload`` — measured at 79% of wall time for
        a trivial ``SELECT 1``, and one handshake per CRUD operation for the
        backends built on top of this class.

        Nothing accumulated, which is why the cost went unnoticed: psycopg2's
        ``with conn`` is a *transaction* scope and does not close, so every
        connection was left open and then reclaimed by CPython refcounting when
        the caller's frame exited. Reuse replaces that accident with a
        lifecycle — one connection per thread per lane, closed by :meth:`close`
        or, if the thread ends first, reclaimed with the thread-local that held
        it. That second half is not a detail: the registry backing
        :meth:`close` holds its connections weakly precisely so a worker that
        exits cannot strand one.

        **Per thread rather than per connector**, because reuse must not become
        sharing. psycopg2 is threadsafety level 2, so a connection *may* be
        passed between threads; that is not the same as a shareable transaction
        block. Two threads entering ``with conn`` on one connection raise
        ``ProgrammingError: the connection cannot be re-entered recursively``,
        and beneath that error they would be inside one transaction, where
        either thread's commit commits the other's uncommitted work. The error
        is the louder half and the transaction is the worse one.

        **Per lane** for the same reason, one thread down. A transaction
        belongs to a connection, and ``PostgresDB``'s wrapper methods each
        enter ``with conn`` — which commits on the way out. Sharing one
        connection between a caller and those wrappers would mean a ``query``
        call committing whatever the caller had open, silently and without
        anything being nested. ``CALLER_LANE`` is what :meth:`PostgresDB.get_conn`
        hands out; ``INTERNAL_LANE`` is what the wrappers use. A caller that
        never touches ``get_conn`` opens one connection per thread as before.

        A connection closed underneath us — by a caller that closed what it was
        handed, or by the server dropping it — is replaced rather than
        returned, so a stale cache cannot surface as ``InterfaceError:
        connection already closed`` or ``OperationalError: server closed the
        connection unexpectedly`` on a call that did nothing wrong. See
        :meth:`_is_usable` for why that needs a round trip.

        Args:
            lane: Which of this thread's connections to return. Defaults to
                ``CALLER_LANE``; the wrapper methods pass ``INTERNAL_LANE`` so
                that their transactions and the caller's stay separate.

        Returns:
            psycopg2.connection: Active database connection using configured parameters.
        """
        conns = self._lane_conns()
        conn = conns.get(lane)
        if self._is_usable(conn):
            return conn
        kwargs: dict[str, Any] = {
            "host": self.host,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "port": self.port,
        }
        # Only pass sslmode when explicitly configured so libpq's own
        # default applies otherwise (preserving prior behavior).
        if self.sslmode is not None:
            kwargs["sslmode"] = self.sslmode
        new_conn = psycopg2.connect(**kwargs)
        # Publication and registration under one lock, so a connection is never
        # live-but-unregistered. Assigning the thread-local first would leave a
        # window in which close() drains a registry this connection has not
        # entered yet, and the caller would keep using a connection the closed
        # connector no longer knows about.
        with self._conns_lock:
            if conn is not None:
                # WeakSet.discard(None) raises rather than no-opping.
                self._open_conns.discard(conn)
            self._open_conns.add(new_conn)
            conns[lane] = new_conn
        return new_conn

    def _lane_conns(self) -> dict[str, Any]:
        """This thread's connection-per-lane mapping, created on first use."""
        conns: dict[str, Any] | None = getattr(self._local, "conns", None)
        if conns is None:
            conns = {}
            self._local.conns = conns
        return conns

    def close(self) -> None:
        """Close every connection this connector opened, on any thread.

        Reaching other threads' connections is the point: a connector used from
        a thread pool holds one per worker, and a ``close`` that reached only
        the caller's would leave the rest open with no handle to them.

        The calling thread's slot is cleared, and the others are left pointing
        at a closed connection — :meth:`get_conn` tests for that and reopens, so
        a thread that closes while another is idle does not break the other.
        Idempotent, and safe on a connector that never connected.

        This is a **shutdown** operation and does not wait for quiescence. A
        thread closed out from under mid-statement sees ``InterfaceError:
        connection already closed``; callers that close while other threads may
        still be working are responsible for joining them first.
        """
        with self._conns_lock:
            conns, self._open_conns = list(self._open_conns), weakref.WeakSet()
        for conn in conns:
            if not conn.closed:
                conn.close()
        self._lane_conns().clear()


class PostgresDB:
    """PostgreSQL database wrapper with utilities for querying and managing tables.

    Provides high-level interface for executing queries, managing tables, and
    uploading DataFrames to PostgreSQL databases.

    Attributes:
        _connector: Connection manager for database operations.
    """

    def __init__(
        self,
        host: str | DotenvPostgresConnector | None = None,
        db: str | None = None,
        user: str | None = None,
        pwd: str | None = None,
        port: int | None = None,
        sslmode: str | None = None,
        validate_on_reuse: bool = True,
    ) -> None:
        """Initialize PostgreSQL database wrapper.

        Args:
            host: Database host or DotenvPostgresConnector instance. If None,
                uses environment configuration. Defaults to None.
            db: Database name. If None, uses environment configuration.
                Defaults to None.
            user: Username. If None, uses environment configuration. Defaults to None.
            pwd: Password. If None, uses environment configuration. Defaults to None.
            port: Port number. If None, uses environment configuration. Defaults to None.
            sslmode: psycopg2 ``sslmode`` forwarded to the connector (e.g.
                ``"require"``). Ignored when ``host`` is an already-built
                ``DotenvPostgresConnector`` (it carries its own ``sslmode``).
            validate_on_reuse: Forwarded to the connector — whether to confirm a
                cached connection is alive before reusing it. Defaults to True.
                Ignored when ``host`` is an already-built connector (it carries
                its own setting).
        """
        # Allow passing a connector directly (for backward compatibility).
        # A connector handed in belongs to the caller: it may be shared with
        # another PostgresDB, and closing it here would close a connection this
        # instance did not open. Only a connector built here is ours to close.
        if isinstance(host, DotenvPostgresConnector):
            self._connector = host
            self._owns_connector = False
        else:
            self._connector = DotenvPostgresConnector(
                host=host,
                db=db,
                user=user,
                pwd=pwd,
                port=port,
                sslmode=sslmode,
                validate_on_reuse=validate_on_reuse,
            )
            self._owns_connector = True
        self._tables_df: pd.DataFrame | None = None
        self._table_names: List[str] | None = None

    @property
    def table_names(self) -> List[str]:
        """Get list of all table names in the database.

        Returns:
            List[str]: List of table names from the public schema.
        """
        if self._table_names is None:
            self._table_names = self._do_get_table_names()
        return self._table_names

    @property
    def tables_df(self) -> pd.DataFrame:
        """Get DataFrame of database table metadata.

        Note:
            The exact schema is database-specific. For PostgreSQL, queries
            information_schema.tables.

        Returns:
            pd.DataFrame: Table metadata from information_schema.tables.
        """
        if self._tables_df is None:
            self._tables_df = self._do_get_tables_df()
        return self._tables_df

    def get_columns(self, table_name: str) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM information_schema.columns WHERE table_name = %(table_name)s",
            params={"table_name": table_name},
        )

    def table_head(self, table_name: str, n: int = 10) -> pd.DataFrame:
        """Get the first N rows from a table.

        The table name is quoted and the row count is bound. ``n`` is the last
        caller value in this module still reaching SQL by interpolation, which
        it did for no reason other than being a number rather than a name —
        the same "it cannot be text, so it needs no binding" reasoning that
        turned out to have gaps in the fetcher's ``ids``.

        Args:
            table_name: Name of the table to sample.
            n: Number of rows to return. Defaults to 10.

        Returns:
            pd.DataFrame: First N rows from the table.
        """
        return self.query(
            f"SELECT * FROM {quote_ident(table_name)} LIMIT %(row_limit)s",
            params={"row_limit": int(n)},
        )

    def get_conn(self) -> Any:
        """Get a connection to the PostgreSQL database.

        The connection is owned by the connector and reused across calls. Do
        **not** close it — call :meth:`close` on this object instead, or use it
        as a context manager. Closing the returned connection directly is
        tolerated (the connector reopens on the next call) but discards the
        reuse this method exists to provide.

        **The transaction on this connection is yours.** :meth:`query`,
        :meth:`execute` and :meth:`upload` run on a separate connection, so a
        wrapper call cannot commit or roll back work you have left open — and
        equally, nothing you do here is committed by them. Commit it yourself.

        Returns:
            psycopg2.connection: Active database connection.
        """
        return self._connector.get_conn(lane=CALLER_LANE)

    def _internal_conn(self) -> Any:
        """The connection the wrapper methods share among themselves.

        Separate from what :meth:`get_conn` hands out, because ``with conn``
        commits the whole connection: on one shared connection an ordinary
        :meth:`query` would commit a caller's open transaction, silently.
        """
        return self._connector.get_conn(lane=INTERNAL_LANE)

    def close(self) -> None:
        """Close the connection, if this instance owns the connector.

        A connector passed to ``__init__`` belongs to its caller and is left
        alone; one built here is closed. Idempotent.
        """
        close_if_owned_sync(self._connector, self._owns_connector)

    def __enter__(self) -> Self:
        """Enter a context that closes the connection on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the connection, whether or not the block raised."""
        self.close()

    def _do_get_tables_df(self) -> pd.DataFrame:
        """Do the work of getting the tables dataframe."""
        return self.query("SELECT * FROM information_schema.tables WHERE table_schema = 'public'")

    def _do_get_table_names(self) -> List[str]:
        """Do the work of getting table names."""
        return self.tables_df["table_name"].tolist()

    def query(
        self,
        query: str,
        params: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Uses parameterized queries for safe injection of values.

        Args:
            query: SQL query string to execute.
            params: Dictionary of parameters to safely inject. Each parameter
                "param" should appear as "%(param)s" in the query string.
                Defaults to None.

        Returns:
            pd.DataFrame: Query results with column names from the cursor.
        """
        with self._internal_conn() as conn:
            with conn.cursor() as curs:
                if params is None:
                    curs.execute(query)
                else:
                    curs.execute(query, params)
                df = pd.DataFrame(curs.fetchall(), columns=[desc[0] for desc in curs.description])
        return df

    def execute(self, stmt: str, params: Dict[str, Any] | None = None) -> int:
        """Execute a SQL statement and commit changes.

        Args:
            stmt: SQL statement to execute.
            params: Optional dictionary of parameters for safe injection.
                Defaults to None.

        Returns:
            int: Number of rows affected by the statement.
        """
        with self._internal_conn() as conn:
            with conn.cursor() as curs:
                curs.execute(stmt, params)
                rowcount: int = curs.rowcount
                conn.commit()
        return rowcount

    @staticmethod
    def _build_insert_columns(columns: list[str]) -> str:
        """Build a quoted comma-separated column list for INSERT statements."""
        return ", ".join(quote_ident(col) for col in columns)

    @staticmethod
    def _dtype_itemsize(dtype: Any) -> int:
        """Width in bytes of the values behind ``dtype``.

        A pandas ExtensionDtype — ``Int64``, ``Float64`` — carries no
        ``itemsize`` of its own but exposes the numpy dtype it is backed by, so
        a nullable 64-bit column is measured as 64-bit rather than falling
        through to the default.

        The default is 8, the widest this method distinguishes: a dtype we
        cannot measure gets the SQL type that fits the most, because guessing
        narrow costs the value while guessing wide costs storage.
        """
        base = getattr(dtype, "numpy_dtype", dtype)
        itemsize = getattr(base, "itemsize", None)
        return int(itemsize) if itemsize else 8

    @staticmethod
    def _psql_integer_type(dtype: Any) -> str:
        """The narrowest PostgreSQL integer type that holds every ``dtype`` value.

        Width alone does not decide this, because **PostgreSQL has no unsigned
        integer types**. ``integer`` is a *signed* int4 and stops at 2^31-1,
        while a 4-byte unsigned dtype runs to 2^32-1 — so reading the itemsize
        without reading the signedness reproduces, one dtype family over, the
        exact defect this ladder exists to fix: a column that its own data
        cannot enter.

        ``uint64`` overflows ``bigint`` for the same reason and has no wider
        integer type to be promoted into, so it becomes ``numeric``, which is
        unbounded. The precision is stated rather than left open because 20
        digits is what 2^64-1 needs, and saying so keeps the declaration
        readable as a range.
        """
        base = getattr(dtype, "numpy_dtype", dtype)
        itemsize = PostgresDB._dtype_itemsize(dtype)
        if getattr(base, "kind", "") == "u":
            if itemsize >= 8:
                return "numeric(20)"
            return "bigint" if itemsize >= 4 else "integer"
        return "bigint" if itemsize > 4 else "integer"

    @staticmethod
    def _psql_type_for_dtype(dtype: Any) -> str | None:
        """The SQL type for ``dtype``, or ``None`` if it has none of its own.

        ``None`` means "this becomes ``varchar``" — the ladder's catch-all,
        whose width cannot be decided from the dtype alone because it depends
        on the values. Separating the two questions is what lets
        :meth:`_psql_schema_line` and :meth:`_column_is_text` share one ladder
        instead of keeping two in step by hand: the declaration side asks what
        the type is, the write side asks only whether it came back ``None``.

        That drift is not hypothetical. The write side previously restated the
        ladder as its own negated predicate, and a rendering that disagreed
        with the declaration is how ``'nan'`` came to be sent into a typed
        column.
        """
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        # The SQL type comes from the dtype's own range rather than from its
        # family. `integer` is int4 and `real` is float4, while pandas defaults
        # to 64 bits for both: an int64 past 2^31 produced a column its own data
        # could not enter, and a float64 was accepted into float4's ~7
        # significant digits and silently rounded. The two fail differently and
        # the silent one is worse, since the round trip looks successful.
        # Integers additionally have to account for signedness — see
        # _psql_integer_type — because PostgreSQL has no unsigned types.
        if pd.api.types.is_integer_dtype(dtype):
            return PostgresDB._psql_integer_type(dtype)
        if pd.api.types.is_float_dtype(dtype):
            return "double precision" if PostgresDB._dtype_itemsize(dtype) > 4 else "real"
        # Tz-aware first: is_datetime64_any_dtype is True for both, and emitting
        # a bare ``timestamp`` for a tz-aware column silently drops the offset.
        if isinstance(dtype, pd.DatetimeTZDtype):
            return "timestamptz"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "timestamp"
        if pd.api.types.is_timedelta64_dtype(dtype):
            return "interval"
        return None

    @staticmethod
    def _psql_schema_line(df: pd.DataFrame, col: str) -> str:
        """Build a single quoted column definition line for CREATE TABLE."""
        q_col = quote_ident(col)
        dtype = df[col].dtype
        sql_type = PostgresDB._psql_type_for_dtype(dtype)
        if sql_type is not None:
            return f"{q_col} {sql_type}"
        return f"{q_col} varchar({PostgresDB._psql_varchar_width(df[col])})"

    @staticmethod
    def _psql_varchar_width(values: pd.Series) -> int:
        """Width of the widest value in ``values`` once rendered as text.

        ``upload`` renders the cells of a ``varchar`` column with ``str`` — the
        one kind of column it still renders rather than sending typed — so the
        width that has to fit is the rendered one, and it is measured with that
        same ``str``.
        ``astype(str)`` is a different renderer and disagrees on some object
        payloads (``bytes`` renders as ``abc`` there and ``b'abc'`` here),
        which would declare the column narrower than the text written into it.
        Calling ``.str.len()`` on the column instead assumed it already held
        strings and raised ``AttributeError`` on any object column that did
        not, so the assumption is dropped rather than guarded.

        The floor of 1 covers a column with nothing to measure and one whose
        values are all the empty string alike: PostgreSQL rejects ``varchar(0)``
        at declaration, so a width of 0 costs the whole table rather than a
        value.

        Nulls are excluded, and that is now correct rather than merely narrow.
        It used to disagree with ``upload``, which rendered a null as the text
        ``'nan'``/``'<NA>'`` — so ``['a', None]`` declared ``varchar(1)`` and
        then sent three characters into it. The rendering is what changed:
        ``upload`` sends SQL ``NULL``, which occupies no width, so measuring
        the non-null values is measuring what arrives.
        """
        raw = values.dropna().map(str).str.len().max()
        return max(1, int(raw)) if pd.notna(raw) else 1

    @staticmethod
    def _column_is_text(dtype: Any) -> bool:
        """Whether ``_psql_schema_line`` types this dtype as ``varchar``.

        The two methods have to agree: a column declared ``varchar`` is written
        as text and measured as text, and a column with a real SQL type is
        written as a typed value. Drift between them is how the previous
        rendering came to send ``'nan'`` into a typed column.

        So this asks the ladder rather than restating it. Written out as its
        own negated predicate — which is what it used to be — agreement was a
        property two lists of ``pd.api.types`` calls happened to have, and
        adding a branch to one of them was enough to lose it silently. Derived,
        there is nothing to keep in step.
        """
        return PostgresDB._psql_type_for_dtype(dtype) is None

    @staticmethod
    def _column_values_for_insert(values: pd.Series) -> list[Any]:
        """One column's values in the form the INSERT should carry.

        ``upload`` used to render *every* cell with ``str`` over
        ``df.to_records()``, so nothing reached psycopg2 as a typed parameter.
        Four things followed, and none of them announced itself:

        * a null became the text ``'nan'`` / ``'<NA>'``, which no typed column
          accepts at any width;
        * ``to_records()`` upcasts a nullable extension dtype to ``float64``,
          so an ``Int64`` column sent ``'1.0'`` into the integer column the
          schema ladder had just created for it;
        * ``str`` on a timedelta follows the *column's resolution*, so a
          ``timedelta64[ns]`` column produced ``'86400000000000 nanoseconds'``
          and PostgreSQL rejected it outright — ``interval`` has no unit finer
          than a microsecond;
        * psycopg2's own adaptation was bypassed for every dtype, leaving
          PostgreSQL's unknown-literal coercion to do the typing.

        Iterating the Series rather than a records array is what fixes the
        second: each column keeps its own dtype instead of being upcast to a
        common one across the row.

        Text columns keep going through ``str``, deliberately. The ``varchar``
        branch is the ladder's fallback for everything without a SQL type of
        its own, and handing psycopg2 an arbitrary object raises
        ``ProgrammingError: can't adapt type``. Rendering them here with the
        same ``str`` that :meth:`_psql_varchar_width` measures keeps the
        declared width and the written value in agreement.

        The null test is guarded by ``is_scalar`` because ``pd.isna`` is
        *elementwise*: handed a ``list`` or an ``ndarray`` it answers about each
        element and returns an array, which the surrounding ``or`` then asks for
        a single truth value it cannot give. Asking only about scalars keeps the
        question well-posed, and loses nothing — every null sentinel pandas
        recognises (``None``, ``nan``, ``NA``, ``NaT``) is itself a scalar.
        """
        as_text = PostgresDB._column_is_text(values.dtype)
        out: list[Any] = []
        for value in values:
            if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
                out.append(None)
            elif as_text:
                out.append(str(value))
            elif isinstance(value, pd.Timestamp):
                out.append(value.to_pydatetime())
            elif isinstance(value, pd.Timedelta):
                out.append(value.to_pytimedelta())
            elif isinstance(value, np.generic):
                # psycopg2 has no adapter for numpy scalars; .item() hands over
                # the Python built-in it wraps.
                out.append(value.item())
            else:
                out.append(value)
        return out

    @staticmethod
    def _require_usable_column_labels(df: pd.DataFrame) -> None:
        """Reject column labels that cannot become SQL identifiers, and say why.

        Rejecting is the right answer — an unnamed SQL column is not something
        ``upload`` should invent a name for, and coercing with ``str`` would
        create columns called ``0`` and ``1`` that nobody asked for. It is also
        what the rest of this module does with identifiers.

        What was not right was the diagnostic. ``pd.DataFrame([[1, 2]])`` gets
        pandas' default integer labels, and the caller saw
        ``Invalid SQL identifier: 0`` from deep inside the schema builder, with
        nothing to say the subject was a column label, that pandas supplied it,
        or that ``df.columns = [...]`` is the one-line fix.

        Checked here rather than in ``quote_ident``, which quotes identifiers
        for the whole module and cannot know that this one came from a
        DataFrame; and up-front rather than per column, so the message names
        every offending label instead of stopping at the first.

        A *repeated* label is refused in the same place, for the same reason
        one step later: ``df[col]`` returns a DataFrame rather than a Series
        when the label appears twice, so the value conversion died on
        ``values.dtype`` with ``AttributeError: 'DataFrame' object has no
        attribute 'dtype'`` — an internal type, from a helper the caller never
        called. It could not have succeeded either way, since the INSERT names
        each column once and two columns of one name have no distinguishable
        destination.

        Raises:
            ValueError: If any column label is not a non-empty string, or if
                any label is repeated.
        """
        bad = [
            (position, label)
            for position, label in enumerate(df.columns)
            if not isinstance(label, str) or not label
        ]
        if bad:
            described = ", ".join(
                f"position {position}: {label!r} ({type(label).__name__})"
                for position, label in bad
            )
            raise ValueError(
                f"DataFrame column labels must be non-empty strings to be used as SQL "
                f"identifiers; {len(bad)} of {len(df.columns)} are not — {described}. "
                f"A DataFrame built without column names gets pandas' default integer "
                f"labels; set them with df.columns = [...] before uploading."
            )

        positions: dict[Any, list[int]] = {}
        for position, label in enumerate(df.columns):
            positions.setdefault(label, []).append(position)
        repeated = {label: at for label, at in positions.items() if len(at) > 1}
        if repeated:
            described = ", ".join(
                f"{label!r} at positions {at}" for label, at in sorted(repeated.items())
            )
            raise ValueError(
                f"DataFrame column labels must be unique to be used as SQL identifiers; "
                f"{len(repeated)} label(s) are duplicated — {described}. Each column of "
                f"the INSERT is named once, so repeated labels have no distinguishable "
                f"destination; rename them before uploading."
            )

    def upload(self, table_name: str, df: pd.DataFrame) -> None:
        """Upload DataFrame data to a database table.

        Creates the table if it doesn't exist, inferring schema from DataFrame types.

        Args:
            table_name: Name of the table to insert data into.
            df: DataFrame with columns matching table fields and data to upload.
        """
        self._require_usable_column_labels(df)
        fields = self._build_insert_columns(list(df.columns))
        template = ", ".join(["%s"] * len(df.columns))
        if table_name not in self.table_names:
            self._create_table(table_name, df)
        # Built per column and then transposed, so every value keeps its own
        # column's dtype. Going row-first through ``to_records()`` upcast a
        # nullable Int64 to float64 and sent '1.0' into an integer column.
        by_column = [self._column_values_for_insert(df[col]) for col in df.columns]
        with self._internal_conn() as conn:
            with conn.cursor() as curs:
                sql = f"INSERT INTO {quote_ident(table_name)} ({fields}) VALUES " + ",".join(
                    curs.mogrify(f"({template})", list(row)).decode("utf-8")
                    for row in zip(*by_column, strict=True)
                )
                curs.execute(sql)

    def _create_table(self, table_name: str, df: pd.DataFrame) -> None:
        """Create a table with schema inferred from DataFrame.

        Creates the table structure based on DataFrame column types but doesn't
        populate it with data.

        Args:
            table_name: Name of the table to create.
            df: DataFrame whose columns and types define the table schema.
        """
        schema_lines = ",".join(self._psql_schema_line(df, col) for col in df.columns)
        sql = f"CREATE TABLE IF NOT EXISTS {quote_ident(table_name)} ({schema_lines})"
        self._tables_df = None
        self._table_names = None
        self.execute(sql)


class PostgresRecordFetcher(RecordFetcher):
    """Fetch records from a PostgreSQL table by ID.

    Attributes:
        db: PostgreSQL database connection wrapper.
        table_name: Name of the table to query.
    """

    def __init__(
        self,
        db: PostgresDB,
        table_name: str,
        id_field_name: str = "id",
        fields_to_retrieve: List[str] | None = None,
        one_based_ids: bool = False,
    ) -> None:
        """Initialize PostgreSQL record fetcher.

        Args:
            db: PostgresDB instance for database operations.
            table_name: Name of the table to fetch records from.
            id_field_name: Name of the integer ID field. Defaults to "id".
            fields_to_retrieve: Subset of fields to retrieve. If None, retrieves
                all fields. Defaults to None.
            one_based_ids: True if data source uses 1-based IDs. Defaults to False.
        """
        super().__init__(
            id_field_name=id_field_name,
            fields_to_retrieve=fields_to_retrieve,
            one_based_ids=one_based_ids,
        )
        self.db = db
        self.table_name = table_name

    def get_records(
        self,
        ids: List[int],
        one_based: bool = False,
        fields_to_retrieve: List[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch records from PostgreSQL table by IDs.

        Every identifier inlined below is quoted. ``fields_to_retrieve`` is a
        list of **column names**, matching what the sibling fetchers do with the
        same parameter (``df[fields_to_retrieve]`` — selection by bare label);
        a name is not a SQL expression, and passing one that reaches outside the
        configured table used to work.

        ``ids`` are **bound**, not inlined. Inlining them was safe by side
        effect — ``str(value + offset)`` raises ``TypeError`` on anything
        non-numeric, so caller text could not reach the SQL — but a side effect
        only covers what it happens to cover, and two things fell outside it:
        an empty list built ``IN ()``, which is a syntax error, and a ``nan``
        or ``inf`` survived the arithmetic to arrive as a bare literal the
        server then rejected. Binding removes the question instead of answering
        it, and ``int()`` states the numeric requirement the arithmetic used to
        imply.

        An empty ``ids`` returns an empty frame **with the columns a populated
        one would have**, matching both sibling fetchers. Returning a
        zero-column frame instead would make ``got["id"]`` raise ``KeyError``
        on a result that merely has no rows, and would make ``pd.concat`` over
        batched calls produce something that is not the union of the non-empty
        batches' columns.

        Args:
            ids: Collection of record IDs to retrieve.
            one_based: True if provided IDs are 1-based. Defaults to False.
            fields_to_retrieve: Subset of *column names* for this call,
                overriding instance default. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved records.

        Raises:
            ValueError: If a field name, the table name or the ID field name is
                not a usable SQL identifier.
            TypeError: If an entry in ``ids`` is not an integer.
        """
        if fields_to_retrieve is None:
            fields_to_retrieve = self.fields_to_retrieve
        if fields_to_retrieve is not None:
            fields = ", ".join(quote_ident(field) for field in fields_to_retrieve)
        else:
            fields = "*"
        offset = 0
        if one_based != self.one_based:
            offset = 1 if self.one_based else -1
        # operator.index rather than int(): ``ids`` is declared List[int] and
        # this is the "must be an integer" test. int() accepts "5" and silently
        # truncates 1.9 to 1 — returning a different, wrong row rather than the
        # empty result such a value used to produce. index() rejects both while
        # still accepting numpy integers, which is what a DataFrame column
        # yields.
        wanted = tuple(operator.index(value) + offset for value in ids)
        if not wanted:
            # ``IN ()`` is a syntax error, so ask for a row that cannot exist.
            # The round trip is what gives the empty frame its columns; deriving
            # them locally would mean reimplementing ``SELECT *``.
            wanted = (None,)  # type: ignore[assignment]  # renders as IN (NULL)
        return self.db.query(
            f"""
           SELECT {fields}
           FROM {quote_ident(self.table_name)}
           WHERE {quote_ident(self.id_field_name)} IN %(ids)s
        """,
            params={"ids": wanted},
        )


class DictionaryRecordFetcher(RecordFetcher):
    """Fetch records from a dictionary mapping IDs to record values.

    Attributes:
        the_dict: Dictionary mapping IDs to record value lists.
        field_names: Field names corresponding to record value positions.
    """

    def __init__(
        self,
        the_dict: Dict[int, List[Any]],
        all_field_names: List[str],
        id_field_name: str = "id",
        fields_to_retrieve: List[str] | None = None,
        one_based_ids: bool = False,
    ):
        """Initialize dictionary record fetcher.

        Args:
            the_dict: Dictionary mapping IDs to lists of record values.
            all_field_names: Field names in same order as record value lists.
            id_field_name: Name of the integer ID field. Defaults to "id".
            fields_to_retrieve: Subset of fields to retrieve. If None, retrieves
                all fields. Defaults to None.
            one_based_ids: True if dictionary uses 1-based IDs. Defaults to False.
        """
        super().__init__(
            id_field_name=id_field_name,
            fields_to_retrieve=fields_to_retrieve,
            one_based_ids=one_based_ids,
        )
        self.the_dict = the_dict
        self.field_names = all_field_names

    def get_records(
        self,
        ids: List[int],
        one_based: bool = False,
        fields_to_retrieve: List[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch records from dictionary by IDs.

        Args:
            ids: Collection of record IDs to retrieve.
            one_based: True if provided IDs are 1-based. Defaults to False.
            fields_to_retrieve: Subset of fields for this call, overriding
                instance default. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved records, with None
                values for missing IDs.
        """
        offset = 0
        if one_based != self.one_based:
            offset = 1 if self.one_based else -1
        offset_ids = [an_id + offset for an_id in ids]
        records = [
            self.the_dict.get(an_id, [an_id] + [None] * (len(self.field_names) - 1))
            for an_id in offset_ids
        ]
        df = pd.DataFrame(records, columns=self.field_names)
        if fields_to_retrieve is None:
            fields_to_retrieve = self.fields_to_retrieve
        if fields_to_retrieve is not None:
            df = df[fields_to_retrieve]
        return df


class DataFrameRecordFetcher(RecordFetcher):
    """Fetch records from a pandas DataFrame by ID.

    Attributes:
        df: DataFrame containing records to fetch from.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_field_name: str = "id",
        fields_to_retrieve: List[str] | None = None,
        one_based_ids: bool = False,
    ) -> None:
        """Initialize DataFrame record fetcher.

        Args:
            df: DataFrame containing records.
            id_field_name: Name of the integer ID field. Defaults to "id".
            fields_to_retrieve: Subset of fields to retrieve. If None, retrieves
                all fields. Defaults to None.
            one_based_ids: True if DataFrame uses 1-based IDs. Defaults to False.
        """
        super().__init__(
            id_field_name=id_field_name,
            fields_to_retrieve=fields_to_retrieve,
            one_based_ids=one_based_ids,
        )
        self.df = df

    def get_records(
        self,
        ids: List[int],
        one_based: bool = False,
        fields_to_retrieve: List[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch records from DataFrame by IDs.

        Args:
            ids: Collection of record IDs to retrieve.
            one_based: True if provided IDs are 1-based. Defaults to False.
            fields_to_retrieve: Subset of fields for this call, overriding
                instance default. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved records.
        """
        offset = 0
        if one_based != self.one_based:
            offset = 1 if self.one_based else -1
        adjusted_ids = [an_id + offset for an_id in ids]
        df = self.df[self.df[self.id_field_name].isin(adjusted_ids)]
        if fields_to_retrieve is None:
            fields_to_retrieve = self.fields_to_retrieve
        if fields_to_retrieve is not None:
            df = df[fields_to_retrieve]
        return df

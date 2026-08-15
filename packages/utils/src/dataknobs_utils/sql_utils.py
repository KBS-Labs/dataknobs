"""SQL database utility functions and connection management.

Provides utilities for working with SQL databases including PostgreSQL,
with support for connection management, query execution, and data loading.
"""

from __future__ import annotations

import os
import threading
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
        """
        self.sslmode = sslmode
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
        self._local = threading.local()
        self._open_conns: set[Any] = set()
        self._conns_lock = threading.Lock()

    def get_conn(self) -> Any:
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
        lifecycle — one connection per thread, closed by :meth:`close`.

        **Per thread rather than per connector**, because reuse must not become
        sharing. psycopg2 is threadsafety level 2, so a connection *may* be
        passed between threads; that is not the same as a shareable transaction
        block. Two threads entering ``with conn`` on one connection raise
        ``ProgrammingError: the connection cannot be re-entered recursively``,
        and beneath that error they would be inside one transaction, where
        either thread's commit commits the other's uncommitted work. The error
        is the louder half and the transaction is the worse one.

        A connection closed underneath us (a dropped server-side connection, or
        a caller that closed what it was handed) is replaced rather than
        returned, so a stale cache cannot surface as ``InterfaceError:
        connection already closed`` on a call that did nothing wrong.

        Returns:
            psycopg2.connection: Active database connection using configured parameters.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
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
        self._local.conn = new_conn
        with self._conns_lock:
            self._open_conns.discard(conn)
            self._open_conns.add(new_conn)
        return new_conn

    def close(self) -> None:
        """Close every connection this connector opened, on any thread.

        Reaching other threads' connections is the point: a connector used from
        a thread pool holds one per worker, and a ``close`` that reached only
        the caller's would leave the rest open with no handle to them.

        The calling thread's slot is cleared, and the others are left pointing
        at a closed connection — :meth:`get_conn` tests for that and reopens, so
        a thread that closes while another is idle does not break the other.
        Idempotent, and safe on a connector that never connected.
        """
        with self._conns_lock:
            conns, self._open_conns = list(self._open_conns), set()
        for conn in conns:
            if not conn.closed:
                conn.close()
        self._local.conn = None


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
                host=host, db=db, user=user, pwd=pwd, port=port, sslmode=sslmode
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

        Args:
            table_name: Name of the table to sample.
            n: Number of rows to return. Defaults to 10.

        Returns:
            pd.DataFrame: First N rows from the table.
        """
        return self.query(f"""SELECT * FROM {quote_ident(table_name)} LIMIT {n}""")

    def get_conn(self) -> Any:
        """Get a connection to the PostgreSQL database.

        The connection is owned by the connector and reused across calls. Do
        **not** close it — call :meth:`close` on this object instead, or use it
        as a context manager. Closing the returned connection directly is
        tolerated (the connector reopens on the next call) but discards the
        reuse this method exists to provide.

        Returns:
            psycopg2.connection: Active database connection.
        """
        return self._connector.get_conn()

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
        with self.get_conn() as conn:
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
        with self.get_conn() as conn:
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
    def _psql_schema_line(df: pd.DataFrame, col: str) -> str:
        """Build a single quoted column definition line for CREATE TABLE."""
        q_col = quote_ident(col)
        dtype = df[col].dtype
        # ``pd.api.types`` predicates rather than ``np.issubdtype``: the latter
        # raises TypeError on every pandas ExtensionDtype — which is why this
        # ladder used to be written twice, once per branch of an
        # ``isinstance(dtype, np.dtype)`` split — and it reports timedelta64 as
        # an integer, since timedelta64 subclasses np.signedinteger. One ladder
        # answers correctly for numpy dtypes and ExtensionDtypes alike.
        if pd.api.types.is_bool_dtype(dtype):
            return f"{q_col} boolean"
        # Width comes from the dtype rather than from the family. `integer` is
        # int4 and `real` is float4, while pandas defaults to 64 bits for both:
        # an int64 past 2^31 produced a column its own data could not enter,
        # and a float64 was accepted into float4's ~7 significant digits and
        # silently rounded. The two fail differently and the silent one is
        # worse, since the round trip looks successful.
        if pd.api.types.is_integer_dtype(dtype):
            return f"{q_col} {'bigint' if PostgresDB._dtype_itemsize(dtype) > 4 else 'integer'}"
        if pd.api.types.is_float_dtype(dtype):
            precision = "double precision" if PostgresDB._dtype_itemsize(dtype) > 4 else "real"
            return f"{q_col} {precision}"
        # Tz-aware first: is_datetime64_any_dtype is True for both, and emitting
        # a bare ``timestamp`` for a tz-aware column silently drops the offset.
        if isinstance(dtype, pd.DatetimeTZDtype):
            return f"{q_col} timestamptz"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return f"{q_col} timestamp"
        if pd.api.types.is_timedelta64_dtype(dtype):
            return f"{q_col} interval"
        return f"{q_col} varchar({PostgresDB._psql_varchar_width(df[col])})"

    @staticmethod
    def _psql_varchar_width(values: pd.Series) -> int:
        """Width of the widest value in ``values`` once rendered as text.

        ``upload`` sends ``str(value)`` for every cell, so the width that has to
        fit is the rendered one — and it is measured with that same ``str``.
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
        written as a typed value. Reading the same predicate ladder from one
        place is what keeps the write side from drifting from the declaration
        side, which is how the previous rendering came to send ``'nan'`` into a
        typed column.
        """
        return not (
            pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_integer_dtype(dtype)
            or pd.api.types.is_float_dtype(dtype)
            or pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_timedelta64_dtype(dtype)
        )

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
        """
        as_text = PostgresDB._column_is_text(values.dtype)
        out: list[Any] = []
        for value in values:
            if value is None or (value is not pd.NaT and pd.isna(value)) or value is pd.NaT:
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

        Raises:
            ValueError: If any column label is not a non-empty string.
        """
        bad = [
            (position, label)
            for position, label in enumerate(df.columns)
            if not isinstance(label, str) or not label
        ]
        if not bad:
            return
        described = ", ".join(
            f"position {position}: {label!r} ({type(label).__name__})" for position, label in bad
        )
        raise ValueError(
            f"DataFrame column labels must be non-empty strings to be used as SQL "
            f"identifiers; {len(bad)} of {len(df.columns)} are not — {described}. "
            f"A DataFrame built without column names gets pandas' default integer "
            f"labels; set them with df.columns = [...] before uploading."
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
        with self.get_conn() as conn:
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

        ``ids`` needs no quoting and gets none: values go through
        ``str(value + offset)``, which raises ``TypeError`` on anything that is
        not a number, so the clause cannot carry caller text.

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
            TypeError: If an entry in ``ids`` is not a number.
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
        values = ", ".join([str(value + offset) for value in ids])
        return self.db.query(f"""
           SELECT {fields}
           FROM {quote_ident(self.table_name)}
           WHERE {quote_ident(self.id_field_name)} IN ({values})
        """)


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

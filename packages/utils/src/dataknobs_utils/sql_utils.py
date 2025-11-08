"""SQL database utility functions and connection management.

Provides utilities for working with SQL databases including PostgreSQL,
with support for connection management, query execution, and data loading.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psycopg2

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except ImportError:
    # dotenv is optional - provide a stub if not installed
    def load_dotenv() -> None:
        pass


from dataknobs_utils.sys_utils import load_project_vars


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
        """
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
            int(os.getenv(
                "POSTGRES_PORT", config.get("POSTGRES_PORT", 5432) if config else 5432
            ))
            if port is None
            else port
        )

    def get_conn(self) -> Any:
        """Create and return a PostgreSQL database connection.

        Returns:
            psycopg2.connection: Active database connection using configured parameters.
        """
        return psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
            port=self.port,
        )


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
        """
        # Allow passing a connector directly (for backward compatibility)
        if isinstance(host, DotenvPostgresConnector):
            self._connector = host
        else:
            self._connector = DotenvPostgresConnector(host=host, db=db, user=user, pwd=pwd, port=port)
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
        return self.query(f"""
            SELECT *
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """)

    def table_head(self, table_name: str, n: int = 10) -> pd.DataFrame:
        """Get the first N rows from a table.

        Args:
            table_name: Name of the table to sample.
            n: Number of rows to return. Defaults to 10.

        Returns:
            pd.DataFrame: First N rows from the table.
        """
        return self.query(f"""SELECT * FROM {table_name} LIMIT {n}""")

    def get_conn(self) -> Any:
        """Get a connection to the PostgreSQL database.

        Returns:
            psycopg2.connection: Active database connection.
        """
        return self._connector.get_conn()

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
                rowcount = curs.rowcount
                conn.commit()
        return rowcount

    def upload(self, table_name: str, df: pd.DataFrame) -> None:
        """Upload DataFrame data to a database table.

        Creates the table if it doesn't exist, inferring schema from DataFrame types.

        Args:
            table_name: Name of the table to insert data into.
            df: DataFrame with columns matching table fields and data to upload.
        """
        fields = ", ".join(df.columns)
        template = ", ".join(["%s"] * len(df.columns))
        if table_name not in self.table_names:
            self._create_table(table_name, df)
        with self.get_conn() as conn:
            with conn.cursor() as curs:
                sql = f"INSERT INTO {table_name} ({fields}) VALUES " + ",".join(
                    curs.mogrify(
                        f"({template})",
                        [str(row[col]) for col in df.columns],
                    ).decode("utf-8")
                    for row in df.to_records()
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

        def psql_schema_line(df: pd.DataFrame, col: str) -> str:
            line = None
            dtype = df[col].dtype
            # Check if it's a numpy dtype (not an ExtensionDtype)
            if hasattr(dtype, "type"):
                if np.issubdtype(dtype, np.integer):
                    line = f"{col} integer"
                elif np.issubdtype(dtype, np.float64):
                    line = f"{col} real"
                else:
                    maxlen = max(df[col].str.len())
                    line = f"{col} varchar({maxlen})"
            else:
                # Handle ExtensionDtype or other types as varchar
                maxlen = max(df[col].str.len())
                line = f"{col} varchar({maxlen})"
            return line

        def build_create_table_sql(df: pd.DataFrame, table_name: str) -> str:
            schema_lines = ",".join([psql_schema_line(df, col) for col in df.columns])
            return f"CREATE TABLE IF NOT EXISTS {table_name} ({schema_lines})"

        self._tables_df = None
        self._table_names = None
        self.execute(build_create_table_sql(df, table_name))


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

        Args:
            ids: Collection of record IDs to retrieve.
            one_based: True if provided IDs are 1-based. Defaults to False.
            fields_to_retrieve: Subset of fields for this call, overriding
                instance default. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved records.
        """
        if fields_to_retrieve is None:
            fields_to_retrieve = self.fields_to_retrieve
        if fields_to_retrieve is not None:
            fields = ", ".join(fields_to_retrieve)
        else:
            fields = "*"
        offset = 0
        if one_based != self.one_based:
            offset = 1 if self.one_based else -1
        values = ", ".join([str(value + offset) for value in ids])
        return self.db.query(f"""
           SELECT {fields}
           FROM {self.table_name}
           WHERE {self.id_field_name} IN ({values})
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

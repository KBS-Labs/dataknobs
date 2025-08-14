import os
from abc import ABC, abstractmethod
from functools import lru_cache
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
    """Abstract base class for fetching records from a data source."""

    def __init__(
        self,
        id_field_name: str = "id",
        fields_to_retrieve: List[str] | None = None,
        one_based_ids: bool = False,
    ) -> None:
        """:param id_field_name: The name of the integer "id" field for the data source.
        :param fields_to_retrieve: The subset of fields to retrieve (retrieve all if None).
        :param one_based_ids: True if the data source IDs are 1-based.
        """
        self.id_field_name = id_field_name
        self.fields_to_retrieve = fields_to_retrieve
        self.one_based = one_based_ids

    @abstractmethod
    def get_records(
        self, ids: List[int], one_based: bool = False, fields_to_retrieve: List[str] | None = None
    ) -> pd.DataFrame:
        """:param ids: The collection of IDs of the records to retrieve
        :param one_based: True if the ids are one-based.
        :param fields_to_retrieve: Overriding subset of self.fields_to_retrieve for this call.
        :return: A pandas dataframe with the retrieved records.
        """
        raise NotImplementedError


class DotenvPostgresConnector:
    def __init__(
        self,
        host: str | None = None,
        db: str | None = None,
        user: str | None = None,
        pwd: str | None = None,
        port: int | None = None,
        pvname: str = ".project_vars",
    ) -> None:
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
        """Get a connection to the database.

        :return: The database connection
        """
        return psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
            port=self.port,
        )


class PostgresDB:
    def __init__(
        self,
        host: str | None = None,
        db: str | None = None,
        user: str | None = None,
        pwd: str | None = None,
        port: int | None = None,
    ) -> None:
        # Allow passing a connector directly (for backward compatibility)
        if isinstance(host, DotenvPostgresConnector):
            self._connector = host
        else:
            self._connector = DotenvPostgresConnector(host=host, db=db, user=user, pwd=pwd, port=port)
        self._tables_df: pd.DataFrame | None = None
        self._table_names: List[str] | None = None

    @property
    def table_names(self) -> List[str]:
        """Get the database table names as a list of strings."""
        if self._table_names is None:
            self._table_names = self._do_get_table_names()
        return self._table_names

    @property
    def tables_df(self) -> pd.DataFrame:
        """Get the database tables dataframe. Note that this is non-standard
        across db types.

        :return: The database tables dataframe
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
        """Get a sample of N rows from the table.
        :param table_name: The table whose rows to sample
        :param n: The number of rows to return
        """
        return self.query(f"""SELECT * FROM {table_name} LIMIT {n}""")

    def get_conn(self) -> Any:
        """Get a connection to the database.

        :return: The database connection
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
        """Submit a query, returning the results as a dataframe.

        :param query: The sql query to execute
        :param params: Parameters to safely inject into the query string, where
            each parameter "param" has the form "%(param)s" in the query string
        :return: A dataframe with the results
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
        """Execute a statement.

        :param stmt: The sql statement to execute
        :param params: Optional dictionary of parameters for the query
        :return: Number of rows affected
        """
        with self.get_conn() as conn:
            with conn.cursor() as curs:
                curs.execute(stmt, params)
                rowcount = curs.rowcount
                conn.commit()
        return rowcount

    def upload(self, table_name: str, df: pd.DataFrame) -> None:
        """Upload the dataframe data to the table.

        :param table_name: The name of the table to upload the data to
        :param df: A dataframe whose columns match the table fields that holds
                   the data to upload
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
        """Create a table for the dataframe if it doesn't already exist.
        Do not populate the table with the contents of the dataframe here.
        :param table_name: The name of the table to create
        :param df: A dataframe whose columns represent the table schema
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
    """Class for fetching records from a Postgres DB."""

    def __init__(
        self,
        db: PostgresDB,
        table_name: str,
        id_field_name: str = "id",
        fields_to_retrieve: List[str] | None = None,
        one_based_ids: bool = False,
    ) -> None:
        """:param db: The postgres database
        :param table_name: The name of the table of the data source.
        :param id_field_name: The name of the integer "id" field in the table.
        :param fields_to_retrieve: The subset of fields to retrieve (retrieve all if None).
        :param one_based_ids: True if the data source IDs are 1-based.
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
        """Get the records as a dataframe for the given IDs.

        :param ids: The collection of IDs of the records to retrieve
        :param one_based: True if these ids are one-based.
        :param fields_to_retrieve: Overriding subset of self.fields_to_retrieve for this call.
        :return: A pandas dataframe with the retrieved records.
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
    """Class for fetching records from a dictionary of IDs mapped to record values."""

    def __init__(
        self,
        the_dict: Dict[int, List[Any]],
        all_field_names: List[str],
        id_field_name: str = "id",
        fields_to_retrieve: List[str] | None = None,
        one_based_ids: bool = False,
    ):
        """:param db: The postgres database
        :param the_dict: The dictionary mapping IDs to records
        :param all_field_names: All field names in the same order as the records
        :param id_field_name: The name of the integer "id" field in the table.
        :param fields_to_retrieve: The subset of fields to retrieve (retrieve all if None).
        :param one_based_ids: True if the data source IDs are 1-based.
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
        """Get the records as a dataframe for the given IDs.

        :param ids: The collection of IDs of the records to retrieve
        :param one_based: True if these ids are one-based.
        :param fields_to_retrieve: Overriding subset of self.fields_to_retrieve for this call.
        :return: A pandas dataframe with the retrieved records.
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
    """Class for fetching records from a pandas dataframe."""

    def __init__(
        self,
        df: pd.DataFrame,
        id_field_name: str = "id",
        fields_to_retrieve: List[str] | None = None,
        one_based_ids: bool = False,
    ) -> None:
        """:param df: The dataframe of records
        :param id_field_name: The name of the integer "id" field in the table.
        :param fields_to_retrieve: The subset of fields to retrieve (retrieve all if None).
        :param one_based_ids: True if the data source IDs are 1-based.
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
        """Get the records as a dataframe for the given IDs.

        :param ids: The collection of IDs of the records to retrieve
        :param one_based: True if these ids are one-based.
        :param fields_to_retrieve: Overriding subset of self.fields_to_retrieve for this call.
        :return: A pandas dataframe with the retrieved records.
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

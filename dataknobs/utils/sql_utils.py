import numpy as np
import os
import pandas as pd
import psycopg2
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from functools import lru_cache
from typing import Any, Dict, List


class RecordFetcher(ABC):
    '''
    Abstract base class for fetching records from a data source.
    '''

    def __init__(
            self,
            id_field_name='id',
            fields_to_retrieve=None,
            one_based_ids=False
    ):
        '''
        :param id_field_name: The name of the integer "id" field for the data source.
        :param fields_to_retrieve: The subset of fields to retrieve (retrieve all if None).
        :param one_based_ids: True if the data source IDs are 1-based.
        '''
        self.id_field_name = id_field_name
        self.fields_to_retrieve = fields_to_retrieve
        self.one_based = one_based_ids

    @abstractmethod
    def get_records(
            self, ids: List[int],
            one_based: bool = False,
            fields_to_retrieve: List[str] = None
    ) -> pd.DataFrame:
        '''
        :param ids: The collection of IDs of the records to retrieve
        :param one_based: True if the ids are one-based.
        :param fields_to_retrieve: Overriding subset of self.fields_to_retrieve for this call.
        :return: A pandas dataframe with the retrieved records.
        '''
        raise NotImplementedError


class DotenvPostgresConnector:
    def __init__(self, host=None, db=None, user=None, pwd=None):
        if host is None or db is None or user is None or pwd is None:
            load_dotenv()
        
        self.host = os.getenv('POSTGRES_HOST', 'localhost') if host is None else host
        self.database = os.getenv('POSTGRES_DB', 'postgres') if db is None else db
        self.user = os.getenv('POSTGRES_USER', 'postgres') if user is None else user
        self.password = os.getenv('POSTGRES_PASSWORD', None) if pwd is None else pwd

    def get_conn(self):
        '''
        Get a connection to the database.

        :return: The database connection
        '''
        return psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        

class PostgresDB():

    def __init__(self, host=None, db=None, user=None, pwd=None):
        self._connector = DotenvPostgresConnector(
            host=host, db=db, user=user, pwd=pwd
        )
        self._tables_df = None
        self._table_names = None

    @property
    def table_names(self) -> List[str]:
        '''
        Get the database table names as a list of strings.
        '''
        if self._table_names is None:
            self._table_names = self._do_get_table_names()
        return self._table_names

    @property
    def tables_df(self) -> pd.DataFrame:
        '''
        Get the database tables dataframe. Note that this is non-standard
        across db types.

        :return: The database tables dataframe
        '''
        if self._tables_df is None:
            self._tables_df = self._do_get_tables_df()
        return self._tables_df

    def get_columns(self, table_name: str) -> pd.DataFrame:
        return self.query(f'''
            SELECT *
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        ''')

    def table_head(self, table_name: str, n: int = 10) -> pd.DataFrame:
        '''
        Get a sample of N rows from the table.
        :param table_name: The table whose rows to sample
        :param n: The number of rows to return
        '''
        return self.query(f'''SELECT * FROM {table_name} LIMIT {n}''')

    def get_conn(self):
        '''
        Get a connection to the database.

        :return: The database connection
        '''
        return self._connector.get_conn()

    def _do_get_tables_df(self) -> pd.DataFrame:
        '''
        Do the work of getting the tables dataframe.
        '''
        return self.query(
            "SELECT * FROM information_schema.tables WHERE table_schema = 'public'"
        )

    def _do_get_table_names(self) -> List[str]:
        '''
        Do the work of getting table names.
        '''
        return self.tables_df['table_name'].tolist()

    @lru_cache(maxsize=2)
    def query(self, query: str) -> pd.DataFrame:
        '''
        Submit a query, returning the results as a dataframe.

        :param query: The sql query to execute
        :return: A dataframe with the results
        '''
        df = None
        with self.get_conn() as conn:
            with conn.cursor() as curs:
                curs.execute(query)
                df = pd.DataFrame(
                    curs.fetchall(),
                    columns=[
                        desc[0] for desc in curs.description
                    ]
                )
        return df

    def execute(self, stmt: str):
        '''
        Execute a statement.

        :param stmt: The sql statement to execute
        '''
        with self.get_conn() as conn:
            with conn.cursor() as curs:
                curs.execute(stmt)

    def upload(self, table_name, df):
        '''
        Upload the dataframe data to the table.

        :param table_name: The name of the table to upload the data to
        :param df: A dataframe whose columns match the table fields that holds
                   the data to upload
        '''
        fields = ', '.join(df.columns)
        template = ', '.join(['%s'] * len(df.columns))
        if table_name not in self.table_names:
            self._create_table(table_name, df)
        with self.get_conn() as conn:
            with conn.cursor() as curs:
                sql = f'INSERT INTO {table_name} ({fields}) VALUES ' + \
                    ','.join(
                        curs.mogrify(
                            f'({template})',
                            [str(row[col]) for col in df.columns],
                        ).decode('utf-8')
                        for row in df.to_records()
                    )
                curs.execute(sql)

    def _create_table(self, table_name, df):
        '''
        Create a table for the dataframe if it doesn't already exist.
        Do not populate the table with the contents of the dataframe here.
        :param table_name: The name of the table to create
        :param df: A dataframe whose columns represent the table schema
        '''
        def psql_schema_line(df, col):
            line = None
            if np.issubdtype(df[col].dtype, np.integer):
                line = f'{col} integer'
            elif np.issubdtype(df[col].dtype, np.float64):
                line = f'{col} real'
            else:
                maxlen = max(df[col].str.len())
                line = f'{col} varchar({maxlen})'
            return line
        def build_create_table_sql(df, table_name):
            schema_lines = ','.join([
                psql_schema_line(df, col)
                for col in df.columns
            ])
            return f'CREATE TABLE IF NOT EXISTS {table_name} ({schema_lines})'
        self._tables_df = None
        self._table_names = None
        self.execute(build_create_table_sql(df, table_name))


class PostgresRecordFetcher(RecordFetcher):
    '''
    Class for fetching records from a Postgres DB.
    '''

    def __init__(
            self,
            db,
            table_name,
            id_field_name='id',
            fields_to_retrieve=None,
            one_based_ids=False
    ):
        '''
        :param db: The postgres database
        :param table_name: The name of the table of the data source.
        :param id_field_name: The name of the integer "id" field in the table.
        :param fields_to_retrieve: The subset of fields to retrieve (retrieve all if None).
        :param one_based_ids: True if the data source IDs are 1-based.
        '''
        super().__init__(
            id_field_name=id_field_name,
            fields_to_retrieve=fields_to_retrieve,
            one_based_ids=one_based_ids
        )
        self.db = db
        self.table_name = table_name

    def get_records(
            self,
            ids: List[int],
            one_based: bool = False,
            fields_to_retrieve: List[str] = None,
    ) -> pd.DataFrame:
        '''
        Get the records as a dataframe for the given IDs.
        
        :param ids: The collection of IDs of the records to retrieve
        :param one_based: True if these ids are one-based.
        :param fields_to_retrieve: Overriding subset of self.fields_to_retrieve for this call.
        :return: A pandas dataframe with the retrieved records.
        '''
        fields = '*'
        if fields_to_retrieve is None:
            fields_to_retrieve = self.fields_to_retrieve
        if fields_to_retrieve is not None:
            fields = ', '.join(fields_to_retrieve)
        offset = 0
        if one_based != self.one_based:
            offset = 1 if self.one_based else -1
        values = ', '.join([
            str(value + offset)
            for value in ids
        ])
        return self.db.query(f'''
           SELECT {fields}
           FROM {self.table_name}
           WHERE {self.id_field_name} IN ({values})
        ''')


class DictionaryRecordFetcher(RecordFetcher):
    '''
    Class for fetching records from a dictionary of IDs mapped to record values.
    '''

    def __init__(
            self,
            the_dict: Dict[int, List[Any]],
            all_field_names: List[str],
            id_field_name='id',
            fields_to_retrieve=None,
            one_based_ids=False
    ):
        '''
        :param db: The postgres database
        :param the_dict: The dictionary mapping IDs to records
        :param all_field_names: All field names in the same order as the records
        :param id_field_name: The name of the integer "id" field in the table.
        :param fields_to_retrieve: The subset of fields to retrieve (retrieve all if None).
        :param one_based_ids: True if the data source IDs are 1-based.
        '''
        super().__init__(
            id_field_name=id_field_name,
            fields_to_retrieve=fields_to_retrieve,
            one_based_ids=one_based_ids
        )
        self.the_dict = the_dict
        self.field_names = all_field_names

    def get_records(
            self,
            ids: List[int],
            one_based: bool = False,
            fields_to_retrieve: List[str] = None,
    ) -> pd.DataFrame:
        '''
        Get the records as a dataframe for the given IDs.
        
        :param ids: The collection of IDs of the records to retrieve
        :param one_based: True if these ids are one-based.
        :param fields_to_retrieve: Overriding subset of self.fields_to_retrieve for this call.
        :return: A pandas dataframe with the retrieved records.
        '''
        offset = 0
        if one_based != self.one_based:
            offset = 1 if self.one_based else -1
        records = [
            self.the_dict.get(an_id, [an_id+offset] + [None]*(len(self.field_names)-1))
            for an_id in ids
        ]
        df = pd.DataFrame(records, columns=self.field_names)
        if fields_to_retrieve is None:
            fields_to_retrieve = self.fields_to_retrieve
        if fields_to_retrieve is not None:
            df = df[fields_to_retrieve]
        return df


class DataFrameRecordFetcher(RecordFetcher):
    '''
    Class for fetching records from a pandas dataframe.
    '''

    def __init__(
            self,
            df: pd.DataFrame,
            id_field_name='id',
            fields_to_retrieve=None,
            one_based_ids=False
    ):
        '''
        :param df: The dataframe of records
        :param id_field_name: The name of the integer "id" field in the table.
        :param fields_to_retrieve: The subset of fields to retrieve (retrieve all if None).
        :param one_based_ids: True if the data source IDs are 1-based.
        '''
        super().__init__(
            id_field_name=id_field_name,
            fields_to_retrieve=fields_to_retrieve,
            one_based_ids=one_based_ids
        )
        self.df = df

    def get_records(
            self,
            ids: List[int],
            one_based: bool = False,
            fields_to_retrieve: List[str] = None,
    ) -> pd.DataFrame:
        '''
        Get the records as a dataframe for the given IDs.
        
        :param ids: The collection of IDs of the records to retrieve
        :param one_based: True if these ids are one-based.
        :param fields_to_retrieve: Overriding subset of self.fields_to_retrieve for this call.
        :return: A pandas dataframe with the retrieved records.
        '''
        offset = 0
        if one_based != self.one_based:
            offset = 1 if self.one_based else -1
        ids = [
            an_id + offset
            for an_id in ids
        ]
        df = self.df[self.df[self.id_field_name].isin(ids)]
        if fields_to_retrieve is None:
            fields_to_retrieve = self.fields_to_retrieve
        if fields_to_retrieve is not None:
            df = self.df[fields_to_retrieve]
        return df

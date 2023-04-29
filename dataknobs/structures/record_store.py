import json
import os
import pandas as pd
from typing import Any, Dict, List


class RecordStore:
    '''
    Wrapper around a sequence of records represented in memory as a list of
    dictionaries and/or as a dataframe and as a tsv file on disk.
    '''

    def __init__(
            self,
            tsv_fpath: str,
            df: pd.DataFrame = None,
    ):
        '''
        :param tsv_fpath: The path to the tsv file on disk. If None or
            empty, then data will not be persisted.
        '''
        self.tsv_fpath = tsv_fpath
        self._df = None
        self._recs = None  # List[Dict[str, Any]]
        self._init_data(df)

    def _init_data(self, df: pd.DataFrame = None):
        '''
        Initialize store data from the tsv file.
        '''
        if os.path.exists(self.tsv_fpath):
            self._df = pd.read_csv(self.tsv_fpath, sep='\t')
        else:
            self._df = df
        self._recs = self._build_recs_from_df()

    def _build_recs_from_df(self) -> List[Dict[str, Any]]:
        ''' Build records from the dataframe '''
        if self._df is not None:
            recs = [
                json.loads(rec)
                for rec in self._df.to_json(
                        orient='records', lines=True
                ).strip().split('\n')
            ]
        else:
            recs = list()
        return recs

    @property
    def df(self) -> pd.DataFrame:
        ''' Get the records as a dataframe '''
        if self._df is None:
            self._df = pd.DataFrame(self._recs)
        return self._df

    @property
    def records(self) -> List[Dict[str, Any]]:
        ''' Get the records as a list of dictionaries '''
        return self._recs

    def add_rec(self, rec: Dict[str, Any]):
        ''' Add the record '''
        self._recs.append(rec)
        self._df = None

    def save(self):
        ''' Save the records to disk as a tsv '''
        self.df.to_csv(self.tsv_fpath, sep='\t', index=False)

    def restore(self):
        ''' Restore records from the version on disk, discarding any changes '''
        self._init_data()

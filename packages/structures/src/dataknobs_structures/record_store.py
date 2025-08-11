import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


class RecordStore:
    """Wrapper around a sequence of records represented in memory as a list of
    dictionaries and/or as a dataframe and as a tsv file on disk.
    """

    def __init__(
        self,
        tsv_fpath: Optional[str],
        df: Optional[pd.DataFrame] = None,
        sep: str = "\t",
    ):
        """:param tsv_fpath: The path to the tsv file on disk. If None or
            empty, then data will not be persisted.
        :param df: An initial dataframe
        :param sep: The file separator to use (if not a tab)
        """
        self.tsv_fpath = tsv_fpath
        self.init_df = df
        self.sep = sep
        self._df: Optional[pd.DataFrame] = None
        self._recs: List[Dict[str, Any]] = []  # Initialize as empty list, not None
        self._init_data(df)

    def _init_data(self, df: Optional[pd.DataFrame] = None) -> None:
        """Initialize store data from the tsv file."""
        if self.tsv_fpath is not None and os.path.exists(self.tsv_fpath):
            self._df = pd.read_csv(self.tsv_fpath, sep=self.sep)
        else:
            self._df = df.copy() if df is not None else None
        self._recs = self._build_recs_from_df()

    def _build_recs_from_df(self) -> List[Dict[str, Any]]:
        """Build records from the dataframe"""
        if self._df is not None:
            recs = [
                json.loads(rec)
                for rec in self._df.to_json(orient="records", lines=True).strip().split("\n")
            ]
        else:
            recs = []
        return recs

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """Get the records as a dataframe"""
        if self._df is None and self._recs is not None:
            self._df = pd.DataFrame(self._recs)
        return self._df

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Get the records as a list of dictionaries"""
        return self._recs or []

    def clear(self) -> None:
        """Clear the contents, starting from empty, but don't auto-"save"."""
        self._recs.clear()
        self._df = None

    def add_rec(self, rec: Dict[str, Any]) -> None:
        """Add the record"""
        self._recs.append(rec)
        self._df = None

    def save(self) -> None:
        """Save the records to disk as a tsv"""
        if self.tsv_fpath is not None and self.df is not None:
            self.df.to_csv(self.tsv_fpath, sep=self.sep, index=False)

    def restore(self, df: Optional[pd.DataFrame] = None) -> None:
        """Restore records from the version on disk, discarding any changes.
        NOTE: If there is no backing file (e.g., tsv_fpath is None), then
        restore will discard all data and restart with the given df (if not
        None,) the init df or start anew.
        """
        self._init_data(df if df is not None else self.init_df)

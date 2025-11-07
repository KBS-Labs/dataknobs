"""Record storage with in-memory and disk persistence.

This module provides RecordStore, a flexible container for managing tabular data
as both in-memory records (list of dictionaries) and pandas DataFrames, with
optional persistence to disk as TSV/CSV files.

The RecordStore class is useful for:
- Managing datasets that need both dict and DataFrame representations
- Automatic synchronization between memory and disk
- Simple CRUD operations on tabular data
- Data analysis workflows that switch between pandas and native Python

Typical usage example:

    ```python
    from dataknobs_structures import RecordStore

    # Create store with disk persistence
    store = RecordStore("/path/to/data.tsv")

    # Add records
    store.add_rec({"id": 1, "name": "Alice", "score": 95})
    store.add_rec({"id": 2, "name": "Bob", "score": 87})

    # Access as DataFrame or records
    df = store.df
    records = store.records

    # Save to disk
    store.save()
    ```
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd


class RecordStore:
    """Container for tabular records with memory and disk representations.

    Manages a sequence of records (rows) that can be represented as a list of
    dictionaries, a pandas DataFrame, and/or a TSV/CSV file on disk. The store
    automatically synchronizes between these representations and provides simple
    CRUD operations.

    Attributes:
        tsv_fpath: Path to the backing file on disk (None if not persisted).
        df: The records as a pandas DataFrame (lazily created from records).
        records: The records as a list of dictionaries.

    Example:
        ```python
        # Create with disk persistence
        store = RecordStore("/data/results.tsv")

        # Add records
        store.add_rec({"user": "alice", "score": 100})
        store.add_rec({"user": "bob", "score": 95})

        # Access data
        print(len(store.records))  # 2
        print(store.df.shape)      # (2, 2)

        # Save and restore
        store.save()
        store.clear()
        store.restore()  # Reloads from disk
        ```

    Note:
        If tsv_fpath is None, the store operates entirely in memory with no
        disk persistence.
    """

    def __init__(
        self,
        tsv_fpath: str | None,
        df: pd.DataFrame | None = None,
        sep: str = "\t",
    ):
        r"""Initialize record store with optional file backing.

        Args:
            tsv_fpath: Path to TSV/CSV file on disk. If None, operates without
                disk persistence. If the file exists, loads data from it.
            df: Optional initial DataFrame to populate the store. Ignored if
                tsv_fpath points to an existing file.
            sep: File separator character. Defaults to tab ("\\t") for TSV files.
                Use "," for CSV.

        Example:
            ```python
            # In-memory only
            store = RecordStore(None)

            # With disk persistence
            store = RecordStore("/data/records.tsv")

            # CSV with comma separator
            store = RecordStore("/data/records.csv", sep=",")

            # Initialize with DataFrame
            import pandas as pd
            df = pd.DataFrame([{"a": 1}, {"a": 2}])
            store = RecordStore("/data/data.tsv", df=df)
            ```
        """
        self.tsv_fpath = tsv_fpath
        self.init_df = df
        self.sep = sep
        self._df: pd.DataFrame | None = None
        self._recs: List[Dict[str, Any]] = []  # Initialize as empty list, not None
        self._init_data(df)

    def _init_data(self, df: pd.DataFrame | None = None) -> None:
        """Initialize store data from file or DataFrame.

        Args:
            df: Optional DataFrame to use if no backing file exists.

        Note:
            Internal method called during initialization.
        """
        if self.tsv_fpath is not None and os.path.exists(self.tsv_fpath):
            self._df = pd.read_csv(self.tsv_fpath, sep=self.sep)
        else:
            self._df = df.copy() if df is not None else None
        self._recs = self._build_recs_from_df()

    def _build_recs_from_df(self) -> List[Dict[str, Any]]:
        """Build records list from DataFrame.

        Returns:
            List of dictionaries, one per DataFrame row.

        Note:
            Internal method for synchronizing DataFrame to records.
        """
        if self._df is not None:
            recs = [
                json.loads(rec)
                for rec in self._df.to_json(orient="records", lines=True).strip().split("\n")
            ]
        else:
            recs = []
        return recs

    @property
    def df(self) -> pd.DataFrame | None:
        """Get records as a pandas DataFrame.

        Lazily creates the DataFrame from records if it doesn't exist.

        Returns:
            DataFrame representation of records, or None if no records.

        Example:
            ```python
            store = RecordStore(None)
            store.add_rec({"name": "Alice", "age": 30})
            store.add_rec({"name": "Bob", "age": 25})

            df = store.df
            print(df.shape)  # (2, 2)
            print(df.columns.tolist())  # ['name', 'age']
            ```
        """
        if self._df is None and self._recs is not None:
            self._df = pd.DataFrame(self._recs)
        return self._df

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Get records as a list of dictionaries.

        Returns:
            List of record dictionaries.

        Example:
            ```python
            store = RecordStore(None)
            store.add_rec({"id": 1, "value": "A"})

            for rec in store.records:
                print(rec["id"], rec["value"])
            ```
        """
        return self._recs or []

    def clear(self) -> None:
        """Clear all records without saving changes.

        Removes all records from memory but does not affect the backing file.
        Use save() afterwards if you want to persist the empty state to disk.

        Example:
            ```python
            store = RecordStore("/data/records.tsv")
            print(len(store.records))  # 100

            store.clear()
            print(len(store.records))  # 0

            # File still contains 100 records until save() is called
            ```
        """
        self._recs.clear()
        self._df = None

    def add_rec(self, rec: Dict[str, Any]) -> None:
        """Add a record to the store.

        Appends a new record and invalidates the DataFrame cache (will be
        rebuilt on next access to df property).

        Args:
            rec: Dictionary representing a single record (row).

        Example:
            ```python
            store = RecordStore("/data/users.tsv")

            store.add_rec({"user_id": 1, "name": "Alice", "active": True})
            store.add_rec({"user_id": 2, "name": "Bob", "active": False})

            store.save()  # Persist to disk
            ```
        """
        self._recs.append(rec)
        self._df = None

    def save(self) -> None:
        """Save records to the backing file.

        Writes the current records to disk as a TSV/CSV file. Does nothing if
        tsv_fpath is None (in-memory only mode).

        Example:
            ```python
            store = RecordStore("/data/results.tsv")
            store.add_rec({"metric": "accuracy", "value": 0.95})
            store.add_rec({"metric": "precision", "value": 0.92})

            store.save()  # Writes to /data/results.tsv
            ```

        Note:
            The file is written with headers and without row indices.
        """
        if self.tsv_fpath is not None and self.df is not None:
            self.df.to_csv(self.tsv_fpath, sep=self.sep, index=False)

    def restore(self, df: pd.DataFrame | None = None) -> None:
        """Restore records from disk, discarding in-memory changes.

        Reloads data from the backing file (if it exists) or from the provided
        DataFrame. All current in-memory changes are lost.

        Args:
            df: Optional DataFrame to restore from. If None, uses the backing
                file (if available) or the initial DataFrame.

        Example:
            ```python
            store = RecordStore("/data/records.tsv")
            original_count = len(store.records)

            # Make changes
            store.add_rec({"new": "data"})
            store.clear()

            # Undo changes
            store.restore()
            print(len(store.records))  # Back to original_count
            ```

        Note:
            If tsv_fpath is None and no df is provided, restores to the initial
            DataFrame or creates an empty store.
        """
        self._init_data(df if df is not None else self.init_df)

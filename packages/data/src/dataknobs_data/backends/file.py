"""File-based database backend implementation."""

from __future__ import annotations

import asyncio
import csv
import gzip
import json
import os
import platform
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

from dataknobs_config import ConfigurableBase

from ..database import AsyncDatabase, SyncDatabase
from ..query import Query
from ..records import Record
from ..streaming import AsyncStreamingMixin, StreamConfig, StreamingMixin, StreamResult
from ..vector import VectorOperationsMixin
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class FileLock:
    """Cross-platform file locking."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lockfile = filepath + ".lock"
        self.lock_handle = None

    def acquire(self):
        """Acquire the file lock."""
        if platform.system() == "Windows":
            import msvcrt

            while True:
                try:
                    self.lock_handle = open(self.lockfile, "wb")
                    msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if self.lock_handle:
                        self.lock_handle.close()  # type: ignore[unreachable]
                    import time

                    time.sleep(0.01)
        else:
            import fcntl

            self.lock_handle = open(self.lockfile, "wb")
            fcntl.lockf(self.lock_handle, fcntl.LOCK_EX)

    def release(self):
        """Release the file lock."""
        if self.lock_handle:
            if platform.system() == "Windows":  # type: ignore[unreachable]
                import msvcrt

                try:
                    msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            self.lock_handle.close()
            try:
                os.remove(self.lockfile)
            except (OSError, FileNotFoundError):
                pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FileFormat:
    """Base class for file format handlers."""

    @staticmethod
    def load(filepath: str) -> dict[str, dict[str, Any]]:
        """Load data from file."""
        raise NotImplementedError

    @staticmethod
    def save(filepath: str, data: dict[str, dict[str, Any]]):
        """Save data to file."""
        raise NotImplementedError


class JSONFormat(FileFormat):
    """JSON file format handler."""

    @staticmethod
    def load(filepath: str) -> dict[str, dict[str, Any]]:
        """Load data from JSON file."""
        if not os.path.exists(filepath):
            return {}

        # Check if file is empty
        if os.path.getsize(filepath) == 0:
            return {}

        try:
            if filepath.endswith(".gz"):
                try:
                    with gzip.open(filepath, "rt", encoding="utf-8") as f:
                        content = f.read()
                        if not content.strip():
                            return {}
                        data = json.loads(content)
                except (gzip.BadGzipFile, OSError):
                    # File has .gz extension but isn't gzipped, treat as regular file
                    with open(filepath, encoding="utf-8") as f:
                        content = f.read()
                        if not content.strip():
                            return {}
                        data = json.loads(content)
            else:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        return {}
                    data = json.loads(content)

            return data
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def save(filepath: str, data: dict[str, dict[str, Any]]):
        """Save data to JSON file."""
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


class CSVFormat(FileFormat):
    """CSV file format handler."""

    @staticmethod
    def load(filepath: str) -> dict[str, dict[str, Any]]:
        """Load data from CSV file."""
        if not os.path.exists(filepath):
            return {}

        # Check if file is empty
        if os.path.getsize(filepath) == 0:
            return {}

        data = {}
        try:
            if filepath.endswith(".gz"):
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "__id__" in row:
                            record_id = row.pop("__id__")
                            # Try to deserialize JSON strings back to objects
                            fields = {}
                            for key, value in row.items():
                                if value and isinstance(value, str):
                                    # Try to parse as JSON if it looks like JSON
                                    if (value.startswith('{') and value.endswith('}')) or \
                                       (value.startswith('[') and value.endswith(']')):
                                        try:
                                            fields[key] = json.loads(value)
                                        except json.JSONDecodeError:
                                            fields[key] = value
                                    else:
                                        fields[key] = value
                                else:
                                    fields[key] = value
                            data[record_id] = {"fields": fields}
            else:
                with open(filepath, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "__id__" in row:
                            record_id = row.pop("__id__")
                            # Try to deserialize JSON strings back to objects
                            fields = {}
                            for key, value in row.items():
                                if value and isinstance(value, str):
                                    # Try to parse as JSON if it looks like JSON
                                    if (value.startswith('{') and value.endswith('}')) or \
                                       (value.startswith('[') and value.endswith(']')):
                                        try:
                                            fields[key] = json.loads(value)
                                        except json.JSONDecodeError:
                                            fields[key] = value
                                    else:
                                        fields[key] = value
                                else:
                                    fields[key] = value
                            data[record_id] = {"fields": fields}
        except (OSError, csv.Error):
            return {}

        return data

    @staticmethod
    def save(filepath: str, data: dict[str, dict[str, Any]]):
        """Save data to CSV file."""
        if not data:
            if filepath.endswith(".gz"):
                with gzip.open(filepath, "wt", encoding="utf-8") as f:
                    f.write("")
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("")
            return

        # Extract all field names and prepare flattened data
        all_fields = set()
        flattened_data = {}
        for record_id, record_data in data.items():
            if "fields" in record_data:
                # Flatten field values for CSV format
                flat_fields = {}
                for field_name, field_data in record_data["fields"].items():
                    # Handle both full field dicts and simple values
                    if isinstance(field_data, dict) and "value" in field_data:
                        value = field_data["value"]
                    else:
                        value = field_data

                    # Serialize complex types as JSON strings
                    if isinstance(value, (dict, list)):
                        flat_fields[field_name] = json.dumps(value)
                    else:
                        flat_fields[field_name] = value
                    all_fields.add(field_name)
                flattened_data[record_id] = flat_fields

        fieldnames = ["__id__"] + sorted(list(all_fields))

        if filepath.endswith(".gz"):
            with gzip.open(filepath, "wt", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record_id, fields in flattened_data.items():
                    row = {"__id__": record_id}
                    row.update(fields)
                    writer.writerow(row)
        else:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record_id, fields in flattened_data.items():
                    row = {"__id__": record_id}
                    row.update(fields)
                    writer.writerow(row)


class ParquetFormat(FileFormat):
    """Parquet file format handler."""

    @staticmethod
    def load(filepath: str) -> dict[str, dict[str, Any]]:
        """Load data from Parquet file."""
        if not os.path.exists(filepath):
            return {}

        try:
            import pandas as pd

            df = pd.read_parquet(filepath)
            data = {}

            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                if "__id__" in row_dict:
                    record_id = row_dict.pop("__id__")
                else:
                    record_id = str(idx)

                # Remove NaN values
                fields = {k: v for k, v in row_dict.items() if pd.notna(v)}
                data[record_id] = {"fields": fields}

            return data
        except ImportError as e:
            raise ImportError("Parquet support requires pandas and pyarrow packages") from e

    @staticmethod
    def save(filepath: str, data: dict[str, dict[str, Any]]):
        """Save data to Parquet file."""
        try:
            import pandas as pd

            if not data:
                # Create empty DataFrame
                df = pd.DataFrame()
            else:
                rows = []
                for record_id, record_data in data.items():
                    row = {"__id__": record_id}
                    if "fields" in record_data:
                        # Flatten field values for Parquet format
                        for field_name, field_data in record_data["fields"].items():
                            # Handle both full field dicts and simple values
                            if isinstance(field_data, dict) and "value" in field_data:
                                row[field_name] = field_data["value"]
                            else:
                                row[field_name] = field_data
                    rows.append(row)

                df = pd.DataFrame(rows)

            df.to_parquet(filepath, index=False, compression="snappy")
        except ImportError as e:
            raise ImportError("Parquet support requires pandas and pyarrow packages") from e


class AsyncFileDatabase(  # type: ignore[misc]
    AsyncDatabase,
    AsyncStreamingMixin,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    """Async file-based database implementation."""

    FORMAT_HANDLERS = {
        ".json": JSONFormat,
        ".csv": CSVFormat,
        ".tsv": CSVFormat,
        ".parquet": ParquetFormat,
        ".pq": ParquetFormat,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # If no path specified, use a temporary file instead of polluting CWD
        if "path" not in self.config:
            # Create a unique temporary file that won't conflict
            temp_file = tempfile.NamedTemporaryFile(
                prefix="dataknobs_async_db_",
                suffix=".json",
                delete=False
            )
            self.filepath = temp_file.name
            temp_file.close()
            self._is_temp_file = True
        else:
            self.filepath = self.config["path"]
            self._is_temp_file = False

        self.format = self.config.get("format")
        self.compression = self.config.get("compression", None)
        self._lock = asyncio.Lock()
        self._file_lock = FileLock(self.filepath)

        # Detect format from file extension if not specified
        if not self.format:
            path = Path(self.filepath)
            # Check for compression
            if path.suffix == ".gz":
                self.compression = "gzip"
                path = Path(path.stem)

            ext = path.suffix.lower()
            if ext in self.FORMAT_HANDLERS:
                self.format = ext.lstrip(".")
            else:
                self.format = "json"  # Default to JSON

        # Apply compression to filepath if specified
        if self.compression == "gzip" and not self.filepath.endswith(".gz"):
            self.filepath += ".gz"

        # Get the appropriate format handler
        ext = f".{self.format}"
        self.handler = self.FORMAT_HANDLERS.get(ext, JSONFormat)

        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()

    @classmethod
    def from_config(cls, config: dict) -> AsyncFileDatabase:
        """Create from config dictionary."""
        return cls(config)

    def _generate_id(self) -> str:
        """Generate a unique ID for a record."""
        return str(uuid.uuid4())

    async def _load_data(self) -> dict[str, Record]:
        """Load all data from file."""
        with self._file_lock:
            raw_data = self.handler.load(self.filepath)
            data = {}
            for record_id, record_dict in raw_data.items():
                data[record_id] = Record.from_dict(record_dict)
            return data

    async def _save_data(self, data: dict[str, Record]):
        """Save all data to file atomically."""
        # Convert records to dictionaries
        raw_data = {}
        for record_id, record in data.items():
            raw_data[record_id] = record.to_dict(include_metadata=True, flatten=False)

        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.filepath) or ".")
        os.close(temp_fd)

        try:
            with self._file_lock:
                self.handler.save(temp_path, raw_data)
                # Atomic rename
                os.replace(temp_path, self.filepath)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    async def create(self, record: Record) -> str:
        """Create a new record in the file."""
        async with self._lock:
            data = await self._load_data()
            # Use centralized method to prepare record
            record_copy, storage_id = self._prepare_record_for_storage(record)
            data[storage_id] = record_copy
            await self._save_data(data)
            return storage_id

    async def read(self, id: str) -> Record | None:
        """Read a record from the file."""
        async with self._lock:
            data = await self._load_data()
            record = data.get(id)
            # Use centralized method to prepare record
            return self._prepare_record_from_storage(record, id)

    async def update(self, id: str, record: Record) -> bool:
        """Update a record in the file."""
        async with self._lock:
            data = await self._load_data()
            if id in data:
                data[id] = record.copy(deep=True)
                await self._save_data(data)
                return True
            return False

    async def delete(self, id: str) -> bool:
        """Delete a record from the file."""
        async with self._lock:
            data = await self._load_data()
            if id in data:
                del data[id]
                await self._save_data(data)
                return True
            return False

    async def exists(self, id: str) -> bool:
        """Check if a record exists in the file."""
        async with self._lock:
            data = await self._load_data()
            return id in data

    async def upsert(self, id_or_record: str | Record, record: Record | None = None) -> str:
        """Update or insert a record.
        
        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic
        """
        # Determine ID and record based on arguments
        if isinstance(id_or_record, str):
            id = id_or_record
            if record is None:
                raise ValueError("Record required when ID is provided")
        else:
            record = id_or_record
            id = record.id
            if id is None:
                import uuid  # type: ignore[unreachable]
                id = str(uuid.uuid4())
                record.storage_id = id
        
        async with self._lock:
            data = await self._load_data()
            data[id] = record.copy(deep=True)
            await self._save_data(data)
            return id

    async def search(self, query: Query) -> list[Record]:
        """Search for records matching the query."""
        async with self._lock:
            data = await self._load_data()
            results = []

            for record_id, record in data.items():
                # Apply filters
                matches = True
                for filter in query.filters:
                    # Special handling for 'id' field
                    if filter.field == 'id':
                        field_value = record_id
                    else:
                        field_value = record.get_value(filter.field)
                    if not filter.matches(field_value):
                        matches = False
                        break

                if matches:
                    results.append((record_id, record))

            # Use the helper method from base class
            return self._process_search_results(results, query, deep_copy=True)

    async def _count_all(self) -> int:
        """Count all records in the file."""
        async with self._lock:
            data = await self._load_data()
            return len(data)

    async def clear(self) -> int:
        """Clear all records from the file."""
        async with self._lock:
            data = await self._load_data()
            count = len(data)
            await self._save_data({})
            return count

    async def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently."""
        async with self._lock:
            data = await self._load_data()
            ids = []
            for record in records:
                record_id = self._generate_id()
                data[record_id] = record.copy(deep=True)
                ids.append(record_id)
            await self._save_data(data)
            return ids

    async def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records efficiently."""
        async with self._lock:
            data = await self._load_data()
            results = []
            for record_id in ids:
                record = data.get(record_id)
                results.append(record.copy(deep=True) if record else None)
            return results

    async def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently."""
        async with self._lock:
            data = await self._load_data()
            results = []
            modified = False
            for record_id in ids:
                if record_id in data:
                    del data[record_id]
                    results.append(True)
                    modified = True
                else:
                    results.append(False)

            if modified:
                await self._save_data(data)

            return results

    async def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from file."""
        # For file backend, we can use the default implementation
        # since we need to load all data anyway
        config = config or StreamConfig()

        # Use search to get all matching records
        if query:
            records = await self.search(query)
        else:
            records = await self.search(Query())

        # Yield records in batches for consistency
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record

    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into file."""
        # Use the default implementation from mixin
        return await self._default_stream_write(records, config)

    async def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """Perform vector similarity search using Python calculations.
        
        Note: This implementation reads all records from disk to perform
        the search locally. For better performance with large datasets,
        consider using SQLite or a dedicated vector database.
        """
        return await self.python_vector_search_async(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )

    async def close(self) -> None:
        """Close the database and clean up temporary files if needed."""
        # Clean up temporary file if it was created
        if getattr(self, '_is_temp_file', False) and self.filepath:
            try:
                if os.path.exists(self.filepath):
                    Path(self.filepath).unlink()
                # Also remove lock file if it exists
                lock_file = self.filepath + ".lock"
                if os.path.exists(lock_file):
                    Path(lock_file).unlink()
            except OSError:
                pass  # Best effort cleanup


class SyncFileDatabase(  # type: ignore[misc]
    SyncDatabase,
    StreamingMixin,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    """Synchronous file-based database implementation."""

    FORMAT_HANDLERS = {
        ".json": JSONFormat,
        ".csv": CSVFormat,
        ".tsv": CSVFormat,
        ".parquet": ParquetFormat,
        ".pq": ParquetFormat,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # If no path specified, use a temporary file instead of polluting CWD
        if "path" not in self.config:
            # Create a unique temporary file that won't conflict
            temp_file = tempfile.NamedTemporaryFile(
                prefix="dataknobs_sync_db_",
                suffix=".json",
                delete=False
            )
            self.filepath = temp_file.name
            temp_file.close()
            self._is_temp_file = True
        else:
            self.filepath = self.config["path"]
            self._is_temp_file = False

        self.format = self.config.get("format")
        self.compression = self.config.get("compression", None)
        self._lock = threading.RLock()
        self._file_lock = FileLock(self.filepath)

        # Detect format from file extension if not specified
        if not self.format:
            path = Path(self.filepath)
            # Check for compression
            if path.suffix == ".gz":
                self.compression = "gzip"
                path = Path(path.stem)

            ext = path.suffix.lower()
            if ext in self.FORMAT_HANDLERS:
                self.format = ext.lstrip(".")
            else:
                self.format = "json"  # Default to JSON

        # Apply compression to filepath if specified
        if self.compression == "gzip" and not self.filepath.endswith(".gz"):
            self.filepath += ".gz"

        # Get the appropriate format handler
        ext = f".{self.format}"
        self.handler = self.FORMAT_HANDLERS.get(ext, JSONFormat)

        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()

    @classmethod
    def from_config(cls, config: dict) -> SyncFileDatabase:
        """Create from config dictionary."""
        return cls(config)

    def _generate_id(self) -> str:
        """Generate a unique ID for a record."""
        return str(uuid.uuid4())

    def _load_data(self) -> dict[str, Record]:
        """Load all data from file."""
        with self._file_lock:
            raw_data = self.handler.load(self.filepath)
            data = {}
            for record_id, record_dict in raw_data.items():
                data[record_id] = Record.from_dict(record_dict)
            return data

    def _save_data(self, data: dict[str, Record]):
        """Save all data to file atomically."""
        # Convert records to dictionaries
        raw_data = {}
        for record_id, record in data.items():
            raw_data[record_id] = record.to_dict(include_metadata=True, flatten=False)

        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.filepath) or ".")
        os.close(temp_fd)

        try:
            with self._file_lock:
                self.handler.save(temp_path, raw_data)
                # Atomic rename
                os.replace(temp_path, self.filepath)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _do_set_data(self, data: dict[str, Record], record: Record) -> str:
        """Ensure record has a storage ID, set data[id]=record.copy() and return the ID"""
        # Use centralized method to prepare record
        record_copy, storage_id = self._prepare_record_for_storage(record)

        # Store the record
        data[storage_id] = record_copy
        return storage_id

    def create(self, record: Record) -> str:
        """Create a new record in the file."""
        with self._lock:
            data = self._load_data()
            # Use record's ID if it has one, otherwise generate a new one
            record_id = self._do_set_data(data, record)
            self._save_data(data)
            return record_id

    def read(self, id: str) -> Record | None:
        """Read a record from the file."""
        with self._lock:
            data = self._load_data()
            record = data.get(id)
            # Use centralized method to prepare record
            return self._prepare_record_from_storage(record, id)

    def update(self, id: str, record: Record) -> bool:
        """Update a record in the file."""
        with self._lock:
            data = self._load_data()
            if id in data:
                data[id] = record.copy(deep=True)
                self._save_data(data)
                return True
            return False

    def delete(self, id: str) -> bool:
        """Delete a record from the file."""
        with self._lock:
            data = self._load_data()
            if id in data:
                del data[id]
                self._save_data(data)
                return True
            return False

    def exists(self, id: str) -> bool:
        """Check if a record exists in the file."""
        with self._lock:
            data = self._load_data()
            return id in data

    def upsert(self, id_or_record: str | Record, record: Record | None = None) -> str:
        """Update or insert a record.
        
        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic
        """
        # Determine ID and record based on arguments
        if isinstance(id_or_record, str):
            id = id_or_record
            if record is None:
                raise ValueError("Record required when ID is provided")
        else:
            record = id_or_record
            id = record.id
            if id is None:
                import uuid  # type: ignore[unreachable]
                id = str(uuid.uuid4())
                record.storage_id = id
        
        with self._lock:
            data = self._load_data()
            data[id] = record.copy(deep=True)
            self._save_data(data)
            return id

    def search(self, query: Query) -> list[Record]:
        """Search for records matching the query."""
        with self._lock:
            data = self._load_data()
            results = []

            for record_id, record in data.items():
                # Apply filters
                matches = True
                for filter in query.filters:
                    # Special handling for 'id' field
                    if filter.field == 'id':
                        field_value = record_id
                    else:
                        field_value = record.get_value(filter.field)
                    if not filter.matches(field_value):
                        matches = False
                        break

                if matches:
                    results.append((record_id, record))

            # Use the helper method from base class
            return self._process_search_results(results, query, deep_copy=True)

    def _count_all(self) -> int:
        """Count all records in the file."""
        with self._lock:
            data = self._load_data()
            return len(data)

    def clear(self) -> int:
        """Clear all records from the file."""
        with self._lock:
            data = self._load_data()
            count = len(data)
            self._save_data({})
            return count

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently."""
        with self._lock:
            data = self._load_data()
            ids = []
            for record in records:
                record_id = self._do_set_data(data, record)
                ids.append(record_id)
            self._save_data(data)
            return ids

    def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records efficiently."""
        with self._lock:
            data = self._load_data()
            results = []
            for record_id in ids:
                record = data.get(record_id)
                results.append(record.copy(deep=True) if record else None)
            return results

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently."""
        with self._lock:
            data = self._load_data()
            results = []
            modified = False
            for record_id in ids:
                if record_id in data:
                    del data[record_id]
                    results.append(True)
                    modified = True
                else:
                    results.append(False)

            if modified:
                self._save_data(data)

            return results

    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from file."""
        # For file backend, we can use the default implementation
        # since we need to load all data anyway
        config = config or StreamConfig()

        # Use search to get all matching records
        if query:
            records = self.search(query)
        else:
            records = self.search(Query())

        # Yield records in batches for consistency
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record

    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into file."""
        # Use the default implementation
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        quitting = False

        def do_write_batch(batch: list) -> bool:
            """Write batch with individual retries, return False to quit"""
            retval = True
            try:
                ids = self.create_batch(batch)
                result.successful += len(ids)
                result.total_processed += len(batch)
            except Exception:
                # Try creating each item again and catch specific error items
                for rec in batch:
                    result.total_processed += 1
                    try:
                        self.create(rec)
                        result.successful += 1
                    except Exception as e:
                        # This item failed again
                        result.failed += 1
                        result.add_error(None, e)
                        if config.on_error:
                            if not config.on_error(e, rec):
                                retval = False
                                break
                        else:
                            # Without "on_error", quit streaming
                            retval = False
                            break
            return retval

        batch = []
        for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write batch
                quitting = not do_write_batch(batch)
                if quitting:
                    # Got signal to quit
                    break
                batch = []

        # Write remaining batch
        if batch and not quitting:
            do_write_batch(batch)

        result.duration = time.time() - start_time
        return result

    def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """Perform vector similarity search using Python calculations.
        
        Note: This implementation reads all records from disk to perform
        the search locally. For better performance with large datasets,
        consider using SQLite or a dedicated vector database.
        """
        return self.python_vector_search_sync(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )

    def close(self) -> None:
        """Close the database and clean up temporary files if needed."""
        # Clean up temporary file if it was created
        if getattr(self, '_is_temp_file', False) and self.filepath:
            try:
                if os.path.exists(self.filepath):
                    Path(self.filepath).unlink()
                # Also remove lock file if it exists
                lock_file = self.filepath + ".lock"
                if os.path.exists(lock_file):
                    Path(lock_file).unlink()
            except OSError:
                pass  # Best effort cleanup

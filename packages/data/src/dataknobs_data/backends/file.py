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
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import (
    AsyncDatabase,
    SyncDatabase,
    enforce_content_version,
    prepare_atomic_batch,
)
from ..exceptions import DuplicateRecordError
from ..query import Query
from ..records import Record
from ..streaming import (
    AsyncStreamingMixin,
    StreamConfig,
    StreamingMixin,
    StreamResult,
    resolve_conflict_write,
    run_stream_write,
)
from ..vector import VectorOperationsMixin
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .config import FileDatabaseConfig
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from typing import ClassVar


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


def _load_file_data(
    handler: type[FileFormat],
    filepath: str,
    file_lock: FileLock,
) -> dict[str, Record]:
    """Load and deserialize all records from ``filepath`` under ``file_lock``.

    The single synchronous implementation of the file read shared by both
    backends: :class:`SyncFileDatabase` calls it directly, while
    :class:`AsyncFileDatabase` runs it via :func:`asyncio.to_thread` so the
    blocking ``FileLock`` acquire (``fcntl.lockf`` / Windows spin) and the
    handler's disk I/O stay off the event loop. Keeping one implementation
    is why the async and sync paths cannot drift.
    """
    with file_lock:
        raw_data = handler.load(filepath)
        data = {}
        for record_id, record_dict in raw_data.items():
            data[record_id] = Record.from_dict(record_dict)
        return data


def _save_file_data(
    handler: type[FileFormat],
    filepath: str,
    file_lock: FileLock,
    data: dict[str, Record],
) -> None:
    """Atomically serialize ``data`` to ``filepath`` under ``file_lock``.

    Writes to a temp file in the same directory, then ``os.replace``s it
    into place under the lock. The single synchronous implementation of
    the file write shared by both backends (sync calls it directly; async
    via :func:`asyncio.to_thread`) — the ``mkstemp`` + lock + atomic
    rename logic lives exactly once.
    """
    # Convert records to dictionaries
    raw_data = {}
    for record_id, record in data.items():
        raw_data[record_id] = record.to_dict(include_metadata=True, flatten=False)

    # Write to temporary file first
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filepath) or ".")
    os.close(temp_fd)

    try:
        with file_lock:
            handler.save(temp_path, raw_data)
            # Atomic rename
            os.replace(temp_path, filepath)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


class AsyncFileDatabase(  # type: ignore[misc]
    StructuredConfigConsumer[FileDatabaseConfig],
    AsyncDatabase,
    AsyncStreamingMixin,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    """Async file-based database implementation.

    Constructed through :class:`FileDatabaseConfig` (shared with the sync
    backend) — every documented config key is a typed field on that
    dataclass, so ``self.config`` is the typed config (not a dict).
    """

    FORMAT_HANDLERS = {
        ".json": JSONFormat,
        ".csv": CSVFormat,
        ".tsv": CSVFormat,
        ".parquet": ParquetFormat,
        ".pq": ParquetFormat,
    }

    CONFIG_CLS: ClassVar[type[FileDatabaseConfig]] = FileDatabaseConfig
    _TEMP_PREFIX = "dataknobs_async_db_"

    def _setup(self) -> None:
        """Resolve filepath/format/handler and vector state from the config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize``.
        """
        cfg = self.config

        # If no path specified, use a temporary file instead of polluting CWD
        if cfg.path is None:
            # Create a unique temporary file that won't conflict
            temp_file = tempfile.NamedTemporaryFile(
                prefix=self._TEMP_PREFIX,
                suffix=".json",
                delete=False
            )
            self.filepath = temp_file.name
            temp_file.close()
            self._is_temp_file = True
        else:
            self.filepath = cfg.path
            self._is_temp_file = False

        self.format = cfg.format
        self.compression = cfg.compression
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
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)
        self._init_vector_state()

    def _generate_id(self) -> str:
        """Generate a unique ID for a record."""
        return str(uuid.uuid4())

    async def _load_data(self) -> dict[str, Record]:
        """Load all data from file.

        Offloads the blocking ``FileLock`` acquire + disk read onto a
        worker thread via :func:`asyncio.to_thread` so the whole critical
        section runs off the event loop. Shares the synchronous
        implementation (:func:`_load_file_data`) with
        :class:`SyncFileDatabase` — no third copy.
        """
        return await asyncio.to_thread(
            _load_file_data, self.handler, self.filepath, self._file_lock
        )

    async def _save_data(self, data: dict[str, Record]):
        """Save all data to file atomically.

        Offloads the ``mkstemp`` + ``FileLock`` acquire + write + atomic
        ``os.replace`` onto a worker thread via :func:`asyncio.to_thread`
        so the entire blocking critical section runs off the event loop.
        Shares the synchronous implementation (:func:`_save_file_data`)
        with :class:`SyncFileDatabase`.
        """
        await asyncio.to_thread(
            _save_file_data,
            self.handler,
            self.filepath,
            self._file_lock,
            data,
        )

    async def create(self, record: Record) -> str:
        """Create a new record in the file."""
        async with self._lock:
            data = await self._load_data()
            # Use centralized method to prepare record
            record_copy, storage_id = self._prepare_record_for_storage(record)
            # Atomic insert: fail closed on a colliding id rather than overwrite
            if storage_id in data:
                raise DuplicateRecordError(storage_id)
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

    async def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        """Update a record in the file."""
        async with self._lock:
            data = await self._load_data()
            if id not in data:
                return False
            # Conditional write: compare the content-hash token inside the
            # lock so the check and the write are atomic (no TOCTOU race).
            enforce_content_version(
                id, expected_version, self._prepare_record_from_storage(data.get(id), id)
            )
            data[id] = record.copy(deep=True)
            await self._save_data(data)
            return True

    async def delete(
        self, id: str, *, expected_version: str | None = None
    ) -> bool:
        """Delete a record from the file.

        When ``expected_version`` is provided the content-hash token is
        compared inside the lock so the check and the delete are atomic (no
        TOCTOU); a stale token raises ``ConcurrencyError`` and a missing record
        returns ``False``. When ``None`` the delete is unconditional,
        byte-identical to prior behavior.
        """
        async with self._lock:
            data = await self._load_data()
            if id not in data:
                return False
            enforce_content_version(
                id, expected_version, self._prepare_record_from_storage(data.get(id), id)
            )
            del data[id]
            await self._save_data(data)
            return True

    async def exists(self, id: str) -> bool:
        """Check if a record exists in the file."""
        async with self._lock:
            data = await self._load_data()
            return id in data

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
        record must already exist with a matching token, otherwise it raises
        ``ConcurrencyError``. A conditional upsert never inserts (an absent
        record's token is ``None``, which never matches).
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
                id = str(uuid.uuid4())  # type: ignore[unreachable]
                record.storage_id = id

        async with self._lock:
            data = await self._load_data()
            enforce_content_version(
                id, expected_version, self._prepare_record_from_storage(data.get(id), id)
            )
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

    async def create_batch(
        self, records: list[Record], *, _tx: Any = None
    ) -> list[str]:
        """Create multiple records, failing closed on any colliding id.

        Matches ``create``'s atomic-insert contract: a colliding id — against an
        existing record or a duplicate within the same batch — raises
        ``DuplicateRecordError`` before any record is written, and the record's
        own id is honored (this path previously minted a fresh id and ignored
        ``record.id``). The pre-scan runs before ``_save_data``, so the batch is
        all-or-nothing and the streaming INSERT fast-path fails closed.

        ``_tx`` is accepted for interface parity with the transactional backends
        and ignored — the file backend has no native transaction to join.
        """
        async with self._lock:
            data = await self._load_data()
            prepared = prepare_atomic_batch(
                records, data, self._prepare_record_for_storage
            )
            ids = []
            for record_copy, storage_id in prepared:
                data[storage_id] = record_copy
                ids.append(storage_id)
            await self._save_data(data)
            return ids

    async def upsert_batch(
        self, records: list[Record], *, _tx: Any = None
    ) -> list[str]:
        """Insert-or-overwrite multiple records in one load/save cycle.

        The batch sibling of ``create_batch``, with upsert semantics: a
        colliding id is overwritten (never raised), a caller-supplied
        ``record.id`` is honored, and ids are returned in input order. The whole
        batch is applied under one ``_load_data`` / ``_save_data`` (a single file
        rewrite) rather than the per-record load+save the ABC default loop would
        incur. ``_tx`` is accepted for interface parity and ignored (see
        :meth:`create_batch`).
        """
        if not records:
            return []
        async with self._lock:
            data = await self._load_data()
            ids = []
            for record in records:
                record_copy, storage_id = self._prepare_record_for_storage(record)
                data[storage_id] = record_copy  # overwrite (upsert)
                ids.append(storage_id)
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

    async def delete_batch(
        self, ids: list[str], *, _tx: Any = None
    ) -> list[bool]:
        """Delete multiple records efficiently.

        ``_tx`` is accepted for interface parity and ignored (see
        :meth:`create_batch`).
        """
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
        # Clean up temporary file if it was created. The existence stats
        # and unlinks are blocking disk I/O, so run them off the loop.
        if getattr(self, '_is_temp_file', False) and self.filepath:
            await asyncio.to_thread(self._cleanup_temp_files)

    def _cleanup_temp_files(self) -> None:
        """Best-effort temp/lock-file removal — run via ``to_thread``."""
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
    StructuredConfigConsumer[FileDatabaseConfig],
    SyncDatabase,
    StreamingMixin,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    """Synchronous file-based database implementation.

    Constructed through :class:`FileDatabaseConfig` (shared with the async
    backend) — every documented config key is a typed field on that
    dataclass, so ``self.config`` is the typed config (not a dict).
    """

    FORMAT_HANDLERS = {
        ".json": JSONFormat,
        ".csv": CSVFormat,
        ".tsv": CSVFormat,
        ".parquet": ParquetFormat,
        ".pq": ParquetFormat,
    }

    CONFIG_CLS: ClassVar[type[FileDatabaseConfig]] = FileDatabaseConfig
    _TEMP_PREFIX = "dataknobs_sync_db_"

    def _setup(self) -> None:
        """Resolve filepath/format/handler and vector state from the config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize``.
        """
        cfg = self.config

        # If no path specified, use a temporary file instead of polluting CWD
        if cfg.path is None:
            # Create a unique temporary file that won't conflict
            temp_file = tempfile.NamedTemporaryFile(
                prefix=self._TEMP_PREFIX,
                suffix=".json",
                delete=False
            )
            self.filepath = temp_file.name
            temp_file.close()
            self._is_temp_file = True
        else:
            self.filepath = cfg.path
            self._is_temp_file = False

        self.format = cfg.format
        self.compression = cfg.compression
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
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)
        self._init_vector_state()

    def _generate_id(self) -> str:
        """Generate a unique ID for a record."""
        return str(uuid.uuid4())

    def _load_data(self) -> dict[str, Record]:
        """Load all data from file.

        Delegates to the shared synchronous :func:`_load_file_data` — the
        same implementation :class:`AsyncFileDatabase` offloads via
        :func:`asyncio.to_thread`.
        """
        return _load_file_data(self.handler, self.filepath, self._file_lock)

    def _save_data(self, data: dict[str, Record]):
        """Save all data to file atomically.

        Delegates to the shared synchronous :func:`_save_file_data` — the
        same implementation :class:`AsyncFileDatabase` offloads via
        :func:`asyncio.to_thread`.
        """
        _save_file_data(self.handler, self.filepath, self._file_lock, data)

    def create(self, record: Record) -> str:
        """Create a new record in the file."""
        with self._lock:
            data = self._load_data()
            # Use record's ID if it has one, otherwise generate a new one
            record_copy, storage_id = self._prepare_record_for_storage(record)
            # Atomic insert: fail closed on a colliding id rather than overwrite
            if storage_id in data:
                raise DuplicateRecordError(storage_id)
            data[storage_id] = record_copy
            self._save_data(data)
            return storage_id

    def read(self, id: str) -> Record | None:
        """Read a record from the file."""
        with self._lock:
            data = self._load_data()
            record = data.get(id)
            # Use centralized method to prepare record
            return self._prepare_record_from_storage(record, id)

    def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        """Update a record in the file."""
        with self._lock:
            data = self._load_data()
            if id not in data:
                return False
            # Conditional write: compare the content-hash token inside the
            # lock so the check and the write are atomic (no TOCTOU race).
            enforce_content_version(
                id, expected_version, self._prepare_record_from_storage(data.get(id), id)
            )
            data[id] = record.copy(deep=True)
            self._save_data(data)
            return True

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record from the file.

        When ``expected_version`` is provided the content-hash token is
        compared inside the lock so the check and the delete are atomic (no
        TOCTOU); a stale token raises ``ConcurrencyError`` and a missing record
        returns ``False``. When ``None`` the delete is unconditional,
        byte-identical to prior behavior.
        """
        with self._lock:
            data = self._load_data()
            if id not in data:
                return False
            enforce_content_version(
                id, expected_version, self._prepare_record_from_storage(data.get(id), id)
            )
            del data[id]
            self._save_data(data)
            return True

    def exists(self, id: str) -> bool:
        """Check if a record exists in the file."""
        with self._lock:
            data = self._load_data()
            return id in data

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
        record must already exist with a matching token, otherwise it raises
        ``ConcurrencyError``. A conditional upsert never inserts (an absent
        record's token is ``None``, which never matches).
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
                id = str(uuid.uuid4())  # type: ignore[unreachable]
                record.storage_id = id

        with self._lock:
            data = self._load_data()
            enforce_content_version(
                id, expected_version, self._prepare_record_from_storage(data.get(id), id)
            )
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

    def _insert_batch_atomic(self) -> bool:
        # create_batch pre-scans for collisions (prepare_atomic_batch) before the
        # single _save_data, so a colliding id raises before anything is written
        # and the migrator's INSERT bulk fast-path is safe.
        return True

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records, failing closed on any colliding id.

        Matches ``create``'s atomic-insert contract: a colliding id — against an
        existing record or a duplicate within the same batch — raises
        ``DuplicateRecordError`` before any record is written (this path
        previously overwrote on collision). The pre-scan runs before
        ``_save_data``, so the batch is all-or-nothing and the streaming INSERT
        fast-path fails closed.
        """
        with self._lock:
            data = self._load_data()
            prepared = prepare_atomic_batch(
                records, data, self._prepare_record_for_storage
            )
            ids = []
            for record_copy, storage_id in prepared:
                data[storage_id] = record_copy
                ids.append(storage_id)
            self._save_data(data)
            return ids

    def upsert_batch(self, records: list[Record]) -> list[str]:
        """Insert-or-overwrite multiple records in one load/save cycle.

        The batch sibling of ``create_batch``, with upsert semantics: a
        colliding id is overwritten (never raised), a caller-supplied
        ``record.id`` is honored, and ids are returned in input order. The whole
        batch is applied under one ``_load_data`` / ``_save_data`` (a single file
        rewrite) rather than the per-record load+save the ABC default loop would
        incur.
        """
        if not records:
            return []
        with self._lock:
            data = self._load_data()
            ids = []
            for record in records:
                record_copy, storage_id = self._prepare_record_for_storage(record)
                data[storage_id] = record_copy  # overwrite (upsert)
                ids.append(storage_id)
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
        """Stream records into file.

        Honors ``config.on_conflict``: INSERT uses the ``create_batch``
        fast-path with a ``create`` per-record fallback; UPSERT uses the
        ``upsert_batch`` fast-path with an ``upsert`` per-record fallback; SKIP
        writes per-record via ``create``.
        """
        config = config or StreamConfig()
        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            insert_batch_func=self.create_batch,
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

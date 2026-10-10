# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""SQLite-specific mixins for vector support and other functionality."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import closing
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from typing import TYPE_CHECKING
from ..fields import VectorField
from ..vector.types import DistanceMetric
from .layout_backend import FileLayoutMixin

if TYPE_CHECKING:
    from ..records import Record


logger = logging.getLogger(__name__)


@cache
def sqlite_max_parameters() -> int:
    """The most parameters one statement may bind under this SQLite library.

    The library's compiled default for a new connection (32766 since SQLite
    3.32, 999 before), read once. A connection can lower its own limit, so a
    backend that holds a :class:`sqlite3.Connection` reads the connection's;
    one reached through aiosqlite has no public way to change it, so it keeps
    this default.
    """
    with closing(sqlite3.connect(":memory:")) as probe:
        return int(probe.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))


def sqlite_regexp(pattern: Any, value: Any) -> bool:
    """SQLite's ``value REGEXP pattern``, answered as ``Filter.matches`` answers ``REGEX``.

    SQLite parses the operator and calls a function named ``REGEXP`` with the
    pattern first, but ships none: :func:`register_regexp` registers this one.
    An unanchored :func:`re.search` over a string, and no match for a value
    that is not one -- ``NULL``, a number, a blob.
    """
    if not isinstance(value, str):
        return False
    return re.search(pattern, value) is not None


#: The name and arity SQLite calls ``x REGEXP y`` through.
REGEXP_FUNCTION: tuple[str, int] = ("REGEXP", 2)


def register_regexp(conn: sqlite3.Connection) -> None:
    """Register :func:`sqlite_regexp` on ``conn``, so ``Operator.REGEX`` answers there."""
    conn.create_function(*REGEXP_FUNCTION, sqlite_regexp, deterministic=True)


class SQLiteLayoutMixin(FileLayoutMixin):
    """What reading a table through a column layout means on SQLite, for both twins.

    A native table is in somebody else's file, opened read-only through a
    ``file:`` URI with ``mode=ro``: SQLite then writes nothing to the file and
    makes no database file that is not there. It may be a view as well as a
    table.

    **A file in WAL mode is the exception to "makes nothing".** SQLite reads
    one through its ``-wal`` and ``-shm`` files even read-only, and makes them
    beside the file when they are not there. They are what the owner's own
    connections make and use, and they hold none of the store's data; but they
    need a directory this process can write, unless the owner has the file
    open and they are there already.
    """

    _DIALECT: ClassVar[str] = "sqlite"
    _PARAM_STYLE: ClassVar[str] = "qmark"
    _READ_ONLY_NEEDS: ClassVar[str] = (
        "A file in WAL mode is read through its `-wal` and `-shm` files, which SQLite "
        "makes beside it when they are not there, so it needs a directory this process "
        "can write unless the owner has the file open."
    )
    _HELD_WHILE: ClassVar[str] = (
        "SQLite waited `timeout` seconds for the owner's lock on it, then gave up"
    )
    #: Primary result codes SQLite gives a file it cannot read as a database here.
    _UNOPENED_CODES: ClassVar[frozenset[int]] = frozenset(
        {
            sqlite3.SQLITE_CANTOPEN,
            sqlite3.SQLITE_READONLY,
            sqlite3.SQLITE_NOTADB,
            sqlite3.SQLITE_PERM,
        }
    )
    #: Primary result codes SQLite gives a file whose owner holds a lock on it.
    _HELD_CODES: ClassVar[frozenset[int]] = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})

    def _file_refusal(self, error: Exception) -> Exception | None:
        """Which refusal SQLite's result code makes ``error``; any other code is raised as is."""
        code = getattr(error, "sqlite_errorcode", None)
        if code is None:
            return None
        primary = code & 0xFF
        if primary in self._HELD_CODES:
            return self._held_file_error(error)
        if primary in self._UNOPENED_CODES:
            return self._unopened_file_error(error)
        return None

    def _connect_target(self) -> tuple[str, bool]:
        """What the driver opens, and whether it is a URI.

        Under the native layout, the file as a read-only URI. The path is made
        absolute and quoted by :meth:`pathlib.Path.as_uri`, so a name holding
        ``?`` or ``#`` is the file's name rather than the URI's query.
        """
        if not self.native:
            return self.db_path, False
        return f"{Path(self.db_path).absolute().as_uri()}?mode=ro", True

    def _relation_exists_query(self) -> tuple[str, Any]:
        """Under the native layout, a view is read in place as a table is.

        The engine's lookup reads ``sqlite_master`` for tables only, which is
        right for the table this package creates.
        """
        if not self.native:
            return super()._relation_exists_query()
        return (
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name = ? COLLATE NOCASE",
            (self.table_name,),
        )


class SQLiteVectorSupport:
    """Vector support for SQLite using JSON storage and Python-based similarity."""

    def __init__(self) -> None:
        """Initialize vector support tracking."""
        self._vector_dimensions: dict[str, int] = {}
        self._vector_fields: dict[str, Any] = {}

    def _has_vector_fields(self, record: Record) -> bool:
        """Check if record has vector fields.

        Args:
            record: Record to check

        Returns:
            True if record has vector fields
        """
        return any(isinstance(field, VectorField) for field in record.fields.values())

    def _extract_vector_dimensions(self, record: Record) -> dict[str, int]:
        """Extract dimensions from vector fields in a record.

        Args:
            record: Record containing potential vector fields

        Returns:
            Dictionary mapping field names to dimensions
        """
        dimensions = {}
        for name, field in record.fields.items():
            if isinstance(field, VectorField):
                if field.value is not None:
                    if isinstance(field.value, np.ndarray):
                        dimensions[name] = field.value.shape[0]
                    elif isinstance(field.value, list):
                        dimensions[name] = len(field.value)
                elif field.dimensions:
                    dimensions[name] = field.dimensions
        return dimensions

    def _update_vector_dimensions(self, record: Record) -> None:
        """Update tracked vector dimensions from a record.

        Args:
            record: Record containing vector fields
        """
        dimensions = self._extract_vector_dimensions(record)
        self._vector_dimensions.update(dimensions)

        # Track which fields are vectors
        for name, field in record.fields.items():
            if isinstance(field, VectorField):
                self._vector_fields[name] = {
                    "dimensions": dimensions.get(name),
                    "source_field": field.source_field,
                    "model_name": field.model_name,
                    "model_version": field.model_version,
                }

    def _serialize_vector(self, vector: np.ndarray | list) -> str:
        """Serialize a vector to JSON string for storage.

        Args:
            vector: Vector as numpy array or list

        Returns:
            JSON string representation
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        return json.dumps(vector)

    def _deserialize_vector(self, vector_str: str) -> np.ndarray | None:
        """Deserialize a vector from JSON string.

        Args:
            vector_str: JSON string representation

        Returns:
            Numpy array
        """
        if not vector_str:
            return None
        try:
            vector_list = json.loads(vector_str)
            return np.array(vector_list, dtype=np.float32)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _compute_similarity(
        self,
        vec1: np.ndarray | None,
        vec2: np.ndarray | None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> float:
        """Compute similarity between two vectors.

        The scoring table for every backend with no native k-NN ---
        ``PythonVectorSearchMixin._score_and_rank`` calls this for all eight,
        on every search. It branched on the *member*, so two of the six fell
        into its ``else`` and raised ``Unsupported metric``: a database
        configured ``vector_metric="l2"`` --- a legitimate member value, which
        the config parser accepts without a warning, and which the published
        settings table names --- could not run a search at all.

        Keyed on :meth:`DistanceMetric.canonical` now, so the table has four
        entries and covers all six. ``L1`` is computed rather than refused:
        it is the Manhattan distance, the enum has always named it, and only
        the absence of a branch here made it unavailable.

        Args:
            vec1: First vector
            vec2: Second vector
            metric: Distance metric to use, in any accepted spelling

        Returns:
            Similarity score (higher is more similar)

        Raises:
            ValueError: If the vectors differ in shape, or the metric is not
                an accepted spelling.
        """
        if vec1 is None or vec2 is None:
            return 0.0

        # Ensure vectors are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1, dtype=np.float32)  # type: ignore[unreachable]
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2, dtype=np.float32)  # type: ignore[unreachable]

        # Check dimensions match
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector dimensions don't match: {vec1.shape} vs {vec2.shape}")

        canonical = DistanceMetric.resolve(metric).canonical()

        if canonical is DistanceMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        if canonical is DistanceMetric.EUCLIDEAN:
            # Convert Euclidean distance to similarity (inverse)
            distance = float(np.linalg.norm(vec1 - vec2))
            return 1.0 / (1.0 + distance)

        if canonical is DistanceMetric.DOT_PRODUCT:
            # Dot product similarity
            return float(np.dot(vec1, vec2))

        # L1 (Manhattan). Unbounded above like Euclidean, so mapped to (0, 1]
        # the same way --- which is also how ``distance_to_score`` maps both,
        # so a corpus scores alike on a Python-path backend and on pgvector.
        distance = float(np.abs(vec1 - vec2).sum())
        return 1.0 / (1.0 + distance)

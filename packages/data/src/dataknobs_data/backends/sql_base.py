# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Base SQL functionality shared between SQL database backends."""

from __future__ import annotations

import inspect
import json
import re
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from datetime import datetime, time, timedelta
from decimal import Decimal
from numbers import Integral
from types import MappingProxyType
from typing import Any, TYPE_CHECKING, TypeVar

from dataknobs_utils.sql_utils import quote_ident

from dataknobs_common.exceptions import OperationError, ValidationError

from ..exceptions import DuplicateRecordError, RecordValidationError
from ..query import (
    Filter,
    Operator,
    Query,
    SortOrder,
    is_storage_key_field,
    NAIVE_TIMESTAMP_SHAPE,
    ZONED_TIMESTAMP_SHAPE,
    membership_values,
    value_kind,
)
from ..records import Record
from .column_layout import ColumnLayout, JsonbLayout
from .sql_types import (
    DOUBLE as _DOUBLE,
    HUGEINT as _HUGEINT,
    INT64_OR_DOUBLE as _INT64_OR_DOUBLE,
    INTEGER_NUMBER as _INTEGER_NUMBER,
    MATCHES_ALL as _MATCHES_ALL,
    MATCHES_NONE as _MATCHES_NONE,
    REAL_NUMBER as _REAL_NUMBER,
    NumberDomain,
    comparand,
    NAIVE_TIME as _NAIVE_TIME,
    NEVER as _NEVER,
    TIME_READINGS as _TIME_READINGS,
    ZONED_INSTANT as _ZONED_INSTANT,
    ZONED_WALL_CLOCK as _ZONED_WALL_CLOCK,
)

# Field name segments must be valid identifiers to prevent SQL injection.
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# A field prefixed ``metadata.`` addresses the metadata JSONB column; any other
# (non-storage-key) field addresses the data JSONB column.
_METADATA_FIELD_PREFIX = "metadata."

#: One statement ready to run: its SQL, and the values its placeholders bind in
#: order.
SQLStatement = tuple[str, list[Any]]


def _warn_renamed(old: str, new: str) -> None:
    warnings.warn(
        f"SQLQueryBuilder.{old} is deprecated; use {new}, which splits a batch "
        "by a parameter ceiling and returns a list of statements",
        DeprecationWarning,
        stacklevel=3,
    )


def _json_can_carry(value: Any) -> bool:
    """Whether ``value`` has a JSON form, and so could be a stored record's value."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def resolve_json_column_and_path(field: str) -> tuple[str, str]:
    """Resolve the JSON column and nested path for a non-storage-key field.

    A ``metadata.<path>`` field addresses the ``metadata`` JSONB column at
    ``<path>``; every other field addresses the ``data`` column at the field
    itself. The single source of truth for the ``metadata.`` prefix routing that
    every SQL filter/sort translator consults instead of testing the prefix
    inline, so the filter and sort paths cannot disagree on where a field lives.

    The reserved storage-key field addresses the record's ``id`` column, not a
    JSON column, so callers MUST gate on :func:`is_storage_key_field` first; the
    precondition is enforced (a fail-loud ``ValueError`` rather than a silent
    mis-route to ``("data", "id")``) so a future caller cannot reopen the
    storage-key-shadowing footgun.

    Returns:
        A ``(column, nested_path)`` pair naming the target JSONB column and the
        dot-notation path within it.

    Raises:
        ValueError: if *field* is the reserved storage-key field.
    """
    if is_storage_key_field(field):
        raise ValueError(
            f"resolve_json_column_and_path() is for non-storage-key fields, but "
            f"{field!r} is the reserved storage-key field (it addresses the 'id' "
            f"column, not a JSON column). Gate the call on is_storage_key_field()."
        )
    if field.startswith(_METADATA_FIELD_PREFIX):
        return "metadata", field[len(_METADATA_FIELD_PREFIX) :]
    return "data", field


def validate_field_name(field: str) -> None:
    """Raise ValueError if *field* is not a safe SQL identifier segment.

    Valid names match ``[A-Za-z_][A-Za-z0-9_]*``.  This check guards
    string-literal positions in SQL (e.g. JSONB key slots) where
    ``quote_ident()`` does not apply.

    This is the **single-segment** check, for a position that takes one JSON
    key and gives a dot no special meaning — ``get_vector_extraction_sql`` and
    ``_build_text_field_concat``.  A *filter field* is a dot-separated path, and
    applying this check to one rejects every nested field; use
    :func:`validate_field_path` there.
    """
    if not _FIELD_NAME_RE.match(field):
        raise ValueError(
            f"Invalid field name: {field!r}. Field names must match [A-Za-z_][A-Za-z0-9_]*."
        )


def validate_field_path(field: str) -> None:
    """Raise ValueError if any dot-separated segment of *field* is unsafe.

    The grammar for a **query field path**, where a dot is a path separator and
    each segment reaches a JSONB key in SQL string-literal position.  It is one
    function because it has two callers that must not drift:
    :meth:`SQLQueryBuilder._build_json_field_expr`, which validates at the point
    of interpolation, and the Postgres ``stream_read`` twins, which pre-flight
    the query so a malformed field is refused before a connection is acquired.

    Those two were separate implementations once, and they disagreed: the
    pre-flight applied :func:`validate_field_name` to the whole dotted string,
    so ``metadata.work_order_id`` raised there while ``search`` over the same
    ``Query`` answered rows through the builder.  Anything that needs this
    grammar calls this function; nothing re-spells it.
    """
    for part in field.split("."):
        if not _FIELD_NAME_RE.match(part):
            raise ValueError(
                f"Invalid field name segment {part!r} in {field!r}. "
                f"Field segments must match [A-Za-z_][A-Za-z0-9_]*."
            )


def escape_like_prefix(prefix: str) -> str:
    r"""Escape a literal string for use as a ``LIKE ... ESCAPE '\'`` prefix.

    Escapes the LIKE metacharacters ``%`` and ``_`` and the escape character
    ``\`` itself so that the prefix is matched *verbatim* — a ``_`` or ``%``
    in the caller's prefix matches a literal ``_`` or ``%``, not a wildcard.
    The caller appends the trailing ``%`` wildcard after escaping.

    Args:
        prefix: The literal prefix string.

    Returns:
        The prefix with ``\\``, ``%``, and ``_`` backslash-escaped.
    """
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def prefix_upper_bound(prefix: str) -> str | None:
    """Compute the exclusive upper bound for a literal-prefix range scan.

    Returns the smallest string strictly greater than every string that begins
    with *prefix*, formed by incrementing the last code point of *prefix*. A
    half-open range ``value >= prefix AND value < upper_bound`` then matches
    exactly the strings with that literal prefix, using a plain index range
    scan under BINARY collation (case-sensitive).

    Edge cases:
    - An empty prefix has no upper bound (every string starts with it) →
      returns ``None``; callers emit an always-true clause instead.
    - A prefix whose final code point is the maximum representable
      (``U+10FFFF``) cannot be incremented in place; the last incrementable
      code point is advanced and the maximal tail is dropped. If every code
      point is maximal, returns ``None`` (unbounded above).

    Args:
        prefix: The literal prefix string.

    Returns:
        The exclusive upper-bound string, or ``None`` if the prefix has no
        finite upper bound.
    """
    if not prefix:
        return None
    # Walk from the end, incrementing the first code point that is not already
    # the maximum. Everything after it (all maximal code points) is dropped.
    for i in range(len(prefix) - 1, -1, -1):
        if ord(prefix[i]) < 0x10FFFF:
            return prefix[:i] + chr(ord(prefix[i]) + 1)
    return None


#: Each negated operator, and the operator it is the negation of.
_NEGATIONS: Mapping[Operator, Operator] = MappingProxyType(
    {
        Operator.NEQ: Operator.EQ,
        Operator.NOT_IN: Operator.IN,
        Operator.NOT_BETWEEN: Operator.BETWEEN,
    }
)

#: The dialects whose drivers bind a ``Decimal`` as a decimal, exactly.
_DECIMAL_DIALECTS = frozenset({"postgres", "duckdb"})

#: Each dialect's names for a JSON value's type, as its type function
#: (``jsonb_typeof`` / ``json_type``) reports them. A dialect missing here
#: cannot read a JSON type, and compares untyped.
_JSON_TYPE_NAMES: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "postgres": {"number": ("number",), "boolean": ("boolean",), "string": ("string",)},
        "sqlite": {
            "number": ("integer", "real"),
            "boolean": ("true", "false"),
            "string": ("text",),
        },
        "duckdb": {
            "number": ("BIGINT", "UBIGINT", "DOUBLE"),
            "boolean": ("BOOLEAN",),
            "string": ("VARCHAR",),
        },
    }
)

#: The type DuckDB casts a value of each non-string kind to.
_DUCKDB_CASTS: Mapping[str, str] = MappingProxyType(
    {"number": "DOUBLE", "boolean": "BOOLEAN", "timestamp": "TIMESTAMP"}
)

#: The date that opens that shape, as SQLite's ``GLOB`` states it.
_DATE_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"

# The ways a stored string is read as a time (``_NAIVE_TIME``,
# ``_ZONED_INSTANT``, ``_ZONED_WALL_CLOCK``, imported above from
# :mod:`.sql_types`, where a native column's type declares its kinds). A naive
# time is a string with no zone; a zoned one is read by the instant it names
# or, against a ``date``, by its own wall clock.

_DAY = timedelta(days=1)
_MICROSECOND = timedelta(microseconds=1)

#: A zone at the end of a string of :data:`ZONED_TIMESTAMP_SHAPE`.
_ZONE_SUFFIX = "(Z|[+-][0-9][0-9]:[0-9][0-9])$"


def _readings(bound: Any) -> dict[str | None, str | None]:
    """The values a bound relates to, each with the reading it compares them under.

    Keyed by the population a value belongs to: its :func:`value_kind`, with
    a time string split into naive and zoned, which Python never orders
    against each other. So an aware ``datetime`` relates to zoned strings, by
    instant; a naive one to naive strings; and a ``date`` to both, each by its
    own wall-clock day, as :func:`~dataknobs_data.query._align_temporal`
    reads it. Two bounds of one comparison relate to the populations both
    relate to.
    """
    kind = value_kind(bound)
    if kind != "timestamp":
        return {kind: kind}
    if not isinstance(bound, datetime):
        return {"naive": _NAIVE_TIME, "zoned": _ZONED_WALL_CLOCK}
    if bound.utcoffset() is None:
        return {"naive": _NAIVE_TIME}
    return {"zoned": _ZONED_INSTANT}


def _split_numbers(bound: Any) -> dict[str | None, str | None]:
    """:func:`_readings`, with a number related to integers and reals apart."""
    readings = _readings(bound)
    if "number" not in readings:
        return readings
    return {"integer": _INTEGER_NUMBER, "real": _REAL_NUMBER}


def _takes_operator(bind: Callable[..., Any]) -> bool:
    """Whether a ``bind`` hook takes the operator as a third argument.

    Before it did, a hook took the reading and the bound; one written then
    is still called with those two.
    """
    try:
        inspect.signature(bind).bind(None, None, Operator.EQ)
    except TypeError:
        return False
    except ValueError:  # no signature to read: a builtin
        return True
    return True


def _utc_text(bound: datetime) -> str:
    """An aware bound as SQLite writes the instant a zoned string names.

    In UTC, fixed width, to the microsecond, and a day early. SQLite writes
    the years 0000 to 9999; an offset moves a time Python holds by up to a
    day either side of its own 0001 to 9999, so the instant a day early is
    one SQLite can always write, and shifting both sides keeps their order.
    The arithmetic is on whole microseconds, where no step can leave
    Python's range; a result before 0001-01-01 falls on 0000-12-30 or -31,
    which Python cannot hold, so it is written by hand.
    """
    offset = bound.utcoffset()
    if offset is None:
        raise ValueError(f"A naive datetime names no instant: {bound!r}")
    since_min = (bound.replace(tzinfo=None) - datetime.min) // _MICROSECOND
    early = since_min - (offset + _DAY) // _MICROSECOND
    days, rest = divmod(early, _DAY // _MICROSECOND)
    clock = (datetime.min + timedelta(microseconds=rest)).time()
    if days >= 0:
        return datetime.combine(datetime.min + timedelta(days=days), clock).isoformat(
            sep="T", timespec="microseconds"
        )
    return f"0000-12-{32 + days}T{clock.isoformat(timespec='microseconds')}"


def is_duplicate_key_error(exc: BaseException) -> bool:
    """Return True if a SQL integrity/constraint error is a primary-key
    (duplicate-id) collision rather than another column-constraint violation.

    ``create()`` maps a colliding id to ``DuplicateRecordError``, but the raw
    driver exceptions (``sqlite3.IntegrityError``, ``duckdb.ConstraintException``)
    also fire for ``NOT NULL`` and ``CHECK`` violations on the ``data`` /
    ``metadata`` columns. Those must not be mislabeled as a duplicate id — this
    predicate distinguishes them by the driver's error text:

    - SQLite enforces a non-integer ``PRIMARY KEY`` via a unique index, so a
      colliding id surfaces as ``UNIQUE constraint failed``.
    - DuckDB reports ``Duplicate key ... violates primary key constraint``.

    ``NOT NULL`` / ``CHECK`` violations carry their own distinct markers and
    return ``False`` here.
    """
    msg = str(exc).lower()
    return (
        "unique constraint failed" in msg  # sqlite (id PK -> unique index)
        or "primary key" in msg  # duckdb
        or "duplicate key" in msg  # duckdb / generic
    )


def constraint_violation_error(record_id: str | None = None) -> RecordValidationError:
    """Build the error for a non-duplicate constraint violation.

    The counterpart to :func:`is_duplicate_key_error`: once that returns
    ``False``, every SQL backend raises this. One factory rather than eight
    copies, because eight copies is why the message they all built was wrong in
    eight places at once.

    Two qualifications on "every". Postgres does not use the text predicate —
    psycopg2 and asyncpg both expose the distinction as an exception type, so
    it splits on ``UniqueViolation`` vs the ``IntegrityError`` base and reaches
    this factory from the second clause. And Elasticsearch is not a SQL backend
    at all: it has no ``NOT NULL`` or ``CHECK`` to violate, so a version
    conflict is the only write rejection it can produce, and that is already a
    ``DuplicateRecordError``.

    What they built was ``RecordValidationError(str(exc))``, relaying the
    driver's text verbatim. That text names the physical schema —
    ``NOT NULL constraint failed: records.tenant_secret`` — and
    ``RecordValidationError`` is a ``ValidationError``, which the
    ``dataknobs-bots`` API layer returns to the caller as a disclosed 422. So
    a rejected write published the table and column it was rejected by.

    The driver's text is not lost, only moved: every call site raises
    ``from exc``, so it stays on ``__cause__`` — in the traceback a library
    caller sees, and in the line the API handler logs. Only the response body
    loses it, which is the audience it was never written for.

    Args:
        record_id: The id being written, when the caller knows which one it
            was. Batch writes do not: the driver reports the constraint, not
            the row.

    Returns:
        The error to raise ``from`` the driver's exception.
    """
    if record_id:
        return RecordValidationError(f"Record '{record_id}' was rejected by a database constraint")
    return RecordValidationError("A record was rejected by a database constraint")


if TYPE_CHECKING:
    from ..query_logic import ComplexQuery


class SQLRecordSerializer:
    """Mixin for SQL record serialization/deserialization with vector support."""

    @staticmethod
    def record_to_json(record: Record) -> str:
        """Convert a Record to JSON string for storage.

        Handles VectorField serialization to preserve metadata.
        """
        from ..fields import VectorField

        data = {}
        for field_name, field_obj in record.fields.items():
            # Handle VectorField - preserve full metadata
            if isinstance(field_obj, VectorField):
                data[field_name] = field_obj.to_dict()
            # Handle other special fields that have to_list
            elif hasattr(field_obj, "to_list") and callable(field_obj.to_list):
                data[field_name] = field_obj.to_list()
            else:
                data[field_name] = field_obj.value
        return json.dumps(data)

    @staticmethod
    def get_vector_extraction_sql(field_name: str, dialect: str = "postgres") -> str:
        """Get SQL expression to extract vector from JSON field.

        Handles both raw arrays and VectorField dict formats.

        Args:
            field_name: Name of the vector field
            dialect: SQL dialect (postgres, sqlite, etc.)

        Returns:
            SQL expression to extract vector value
        """
        validate_field_name(field_name)

        if dialect == "postgres":
            # PostgreSQL: Handle both formats - raw array or VectorField dict
            return f"""CASE
                WHEN jsonb_typeof(data->'{field_name}') = 'object'
                THEN (data->'{field_name}'->>'value')::vector
                ELSE (data->>'{field_name}')::vector
            END"""
        elif dialect == "sqlite":
            # SQLite doesn't have native vector type, return JSON string
            return f"""CASE
                WHEN json_type(json_extract(data, '$.{field_name}')) = 'object'
                THEN json_extract(data, '$.{field_name}.value')
                ELSE json_extract(data, '$.{field_name}')
            END"""
        else:
            # Generic fallback
            return f"data->'{field_name}'"

    @staticmethod
    def json_to_record(data_json: str, metadata_json: str | None = None) -> Record:
        """Convert JSON strings to a Record.

        Reconstructs VectorField objects from serialized format.
        """
        from ..fields import Field, VectorField

        data = json.loads(data_json) if data_json else {}
        metadata = json.loads(metadata_json) if metadata_json and metadata_json != "null" else {}

        # Reconstruct fields properly, especially VectorFields. Annotated to
        # the base: the first branch assigns a ``VectorField`` and the second a
        # plain ``Field``, and an unannotated dict takes its value type from
        # whichever it sees first.
        fields: dict[str, Field] = {}
        for field_name, field_value in data.items():
            # Check if this is a serialized VectorField
            if isinstance(field_value, dict) and field_value.get("type") == "vector":
                # Ensure the field has a 'name' key for from_dict (in case it's missing)
                if "name" not in field_value:
                    field_value["name"] = field_name
                # Reconstruct VectorField from dict
                fields[field_name] = VectorField.from_dict(field_value)
            else:
                # Regular field
                fields[field_name] = Field(name=field_name, value=field_value)

        # Create Record with properly typed fields
        record = Record(metadata=metadata)
        record.fields.update(fields)
        return record

    @staticmethod
    def row_to_record(row: dict[str, Any]) -> Record:
        """Convert a database row to a Record.

        Args:
            row: Database row as dictionary with 'id', 'data' and optional 'metadata' fields

        Returns:
            Reconstructed Record object with ID set
        """
        data_json = row.get("data", {})
        if not isinstance(data_json, str):
            data_json = json.dumps(data_json)

        metadata_json = row.get("metadata")
        if metadata_json and not isinstance(metadata_json, str):
            metadata_json = json.dumps(metadata_json)

        record = SQLRecordSerializer.json_to_record(data_json, metadata_json)

        # Ensure the record has its ID set from the row
        from ..database_utils import ensure_record_id

        if "id" in row:
            record = ensure_record_id(record, row["id"])

        return record

    @staticmethod
    def record_to_row(record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to a database row.

        Outbound counterpart to :meth:`row_to_record`. Centralizes the
        ``id`` / ``data`` / ``metadata`` shape so sync and async SQL
        backends do not duplicate the body and silently drift (the
        same shape that produced the inbound `_row_to_record`
        divergence).

        Args:
            record: Record to serialize.
            id: Row id; if ``None``, a fresh ``str(uuid.uuid4())``
                (hyphenated 36-character form) is used.

        Returns:
            Dict with keys ``id`` (str), ``data`` (JSON str), and
            ``metadata`` (JSON str or ``None``).
        """
        return {
            "id": id or str(uuid.uuid4()),
            "data": SQLRecordSerializer.record_to_json(record),
            "metadata": (json.dumps(record.metadata) if record.metadata else None),
        }


_Builds = TypeVar("_Builds", bound=Callable[..., Any])


def _writes(method: _Builds) -> _Builds:
    """Mark a builder method as one that writes, refused under a read-only layout.

    The refusal is in the builder, so a backend that forgets to refuse a write
    still cannot emit one against a table it does not own.
    """

    @wraps(method)
    def refused_unless_writable(self: SQLQueryBuilder, *args: Any, **kwargs: Any) -> Any:
        if not self.layout.writable:
            raise OperationError(
                f"table {self.table_name!r} is read through a read-only layout; "
                f"{method.__name__} would write it",
                context={"table": self.table_name, "operation": method.__name__},
            )
        return method(self, *args, **kwargs)

    refused_unless_writable.writes = True  # type: ignore[attr-defined]
    return refused_unless_writable  # type: ignore[return-value]


class SQLQueryBuilder:
    """Builds SQL queries from Query objects.

    The builder reads its table through a :class:`~.column_layout.ColumnLayout`.
    The default, :class:`~.column_layout.JsonbLayout`, is the table this package
    creates; :class:`~.column_layout.NativeColumnLayout` reads a table with
    ordinary typed columns, ANDs its scope into every read and refuses every
    write.
    """

    def __init__(
        self,
        table_name: str,
        schema_name: str | None = None,
        dialect: str = "standard",
        param_style: str = "numeric",
        layout: ColumnLayout | None = None,
    ):
        """Initialize the SQL query builder.

        Args:
            table_name: Name of the database table
            schema_name: Optional schema name
            dialect: SQL dialect ('postgres', 'sqlite', 'standard')
            param_style: Parameter style ('numeric' for $1, 'qmark' for ?, 'pyformat' for %(name)s)
            layout: How the table's rows are laid out; the JSON layout this
                package creates when omitted.

        Raises:
            ValidationError: When ``layout`` cannot render for ``dialect``, or
                is not a :class:`~.column_layout.JsonbLayout` and claims it
                may write.
        """
        self.table_name = table_name
        self.schema_name = schema_name
        self.dialect = dialect
        self.param_style = param_style
        self.qualified_table = self._get_qualified_table_name()
        self._layout = layout if layout is not None else JsonbLayout()
        if self._layout.writable and not isinstance(self._layout, JsonbLayout):
            # Every statement that writes is rendered for the JSON layout's
            # ``id``, ``data`` and ``metadata`` columns.
            raise ValidationError(
                f"{type(self._layout).__name__} claims it may write table {table_name!r}, "
                f"but only the JSON layout writes: the builder's write statements name "
                f"its id, data and metadata columns",
                context={"table": table_name, "layout": type(self._layout).__name__},
            )
        self._layout.check_dialect(dialect)

    @property
    def layout(self) -> ColumnLayout:
        """How the table's rows are laid out."""
        return self._layout

    def _get_qualified_table_name(self) -> str:
        """Get the fully qualified table name."""
        if self.schema_name:
            return f"{quote_ident(self.schema_name)}.{quote_ident(self.table_name)}"
        return quote_ident(self.table_name)

    def param_placeholder(self, param_num: int, param_name: str | None = None) -> str:
        """The placeholder for parameter ``param_num``, in the builder's ``param_style``.

        A layout numbers its parameters from the ``param_start`` it is given,
        and the next clause numbers on from however many it returned.

        Args:
            param_num: Parameter number (1-based)
            param_name: Optional parameter name for pyformat style

        Returns:
            Parameter placeholder string
        """
        if self.param_style == "numeric":
            return f"${param_num}"
        elif self.param_style == "qmark":
            return "?"
        elif self.param_style == "pyformat":
            name = param_name or f"p{param_num - 1}"  # 0-based for pyformat
            return f"%({name})s"
        else:
            # Default to numeric for postgres dialect, qmark for others
            if self.dialect == "postgres":
                return f"${param_num}"
            else:
                return "?"

    @_writes
    def build_create_query(
        self, record: Record, record_id: str | None = None
    ) -> tuple[str, list[Any]]:
        """Build an INSERT query for creating a record.

        Args:
            record: The record to insert
            record_id: The resolved storage id. Backends resolve the mint
                through their ``_generate_id()`` hook and pass it here; the
                inline ``record.id or str(uuid.uuid4())`` below is only a
                defensive fallback for a direct caller that supplies neither.

        Returns:
            Tuple of (SQL query, parameters)
        """
        record_id = record_id or record.id or str(uuid.uuid4())
        data = SQLRecordSerializer.record_to_json(record)
        metadata = json.dumps(record.metadata) if record.metadata else None

        p1 = self.param_placeholder(1)
        p2 = self.param_placeholder(2)
        p3 = self.param_placeholder(3)

        query = f"""
            INSERT INTO {self.qualified_table} (id, data, metadata, created_at, updated_at)
            VALUES ({p1}, {p2}, {p3}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        if self.dialect == "postgres":
            query += " RETURNING id"

        params = [record_id, data, metadata]

        return query, params

    def build_read_query(self, record_id: str) -> tuple[str, list[Any]]:
        """Build a SELECT query for reading a record by ID.

        Args:
            record_id: The record ID

        Returns:
            Tuple of (SQL query, parameters)
        """
        where, params = self._keyed_clause(record_id)
        return (
            f"SELECT {self.layout.select_list(self)} FROM {self.qualified_table} WHERE {where}",
            params,
        )

    @_writes
    def build_update_query(self, record_id: str, record: Record) -> tuple[str, list[Any]]:
        """Build an UPDATE query for updating a record.

        Args:
            record_id: The record ID
            record: The updated record

        Returns:
            Tuple of (SQL query, parameters)
        """
        data = self._record_to_json(record)
        metadata = json.dumps(record.metadata) if record.metadata else None

        if self.param_style == "qmark":
            # SQLite: data, metadata, then id
            query = f"""
                UPDATE {self.qualified_table}
                SET data = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            params = [data, metadata, record_id]
        else:
            # PostgreSQL: id first, then data, metadata
            p1 = self.param_placeholder(1)
            p2 = self.param_placeholder(2)
            p3 = self.param_placeholder(3)
            query = f"""
                UPDATE {self.qualified_table}
                SET data = {p2}, metadata = {p3}, updated_at = CURRENT_TIMESTAMP
                WHERE id = {p1}
            """
            params = [record_id, data, metadata]

        return query, params

    @_writes
    def build_delete_query(self, record_id: str) -> tuple[str, list[Any]]:
        """Build a DELETE query for deleting a record.

        Args:
            record_id: The record ID

        Returns:
            Tuple of (SQL query, parameters)
        """
        p1 = self.param_placeholder(1)
        query = f"DELETE FROM {self.qualified_table} WHERE id = {p1}"

        return query, [record_id]

    def build_exists_query(self, record_id: str) -> tuple[str, list[Any]]:
        """Build a query to check if a record exists.

        Args:
            record_id: The record ID

        Returns:
            Tuple of (SQL query, parameters)
        """
        where, params = self._keyed_clause(record_id)
        return f"SELECT 1 FROM {self.qualified_table} WHERE {where} LIMIT 1", params

    def _keyed_clause(self, record_id: str) -> tuple[str, list[Any]]:
        """The row stored under ``record_id``, within the layout's scope."""
        scope, params = self._filters_clause([])
        key, key_params = self.layout.key_clause(self, record_id, 1 + len(params))
        return (f"{scope} AND {key}" if scope else key), params + key_params

    def build_complex_search_query(self, query: ComplexQuery) -> tuple[str, list[Any]]:
        """Build a SELECT query from a ComplexQuery object with boolean logic.

        Args:
            query: The ComplexQuery object

        Returns:
            Tuple of (SQL query, parameters)

        Raises:
            ValueError: If any filter field contains invalid characters.
        """
        sql_parts = [f"SELECT {self.layout.select_list(self)} FROM {self.qualified_table}"]

        # The layout's scope goes above the condition, so an OR in it cannot
        # reach rows outside the scope.
        scope, params = self._filters_clause([])
        if query.condition:
            where_clause, where_params = self._build_complex_condition(
                query.condition, 1 + len(params)
            )
            if where_clause:
                if scope:
                    where_clause = f"{scope} AND ({where_clause})"
                sql_parts.append(f"WHERE {where_clause}")
                params.extend(where_params)
        elif scope:
            sql_parts.append(f"WHERE {scope}")

        sql_parts.extend(self._paging_clauses(query))

        return " ".join(sql_parts), params

    def _build_complex_condition(self, condition: Any, param_start: int) -> tuple[str, list[Any]]:
        """Build WHERE clause for complex boolean logic conditions.

        The clause selects exactly the records ``condition.matches`` accepts.
        A comparison on a missing field is ``NULL`` in SQL. Every leaf clause
        :meth:`_build_filter_clause` renders is ``NULL`` only where
        ``Filter.matches`` answers ``False``, and ``NOT`` relies on that: a
        leaf that could be ``NULL`` on a match would make its ``NOT`` wrong.
        ``AND`` and ``OR`` keep ``NULL`` where ``matches`` answers ``False``,
        and ``NOT`` is rendered ``(<clause>) IS NOT TRUE``, which reads it as
        ``False`` before negating it. So
        ``NOT(colour == 'blue')`` matches a record without a colour, as
        ``matches`` does, while ``colour != 'blue'`` does not.

        ``NOT`` over several conditions matches when none of them does. An
        empty ``AND`` is ``TRUE`` and an empty ``OR`` is ``FALSE``, so an
        empty group constrains its parent as ``matches`` says it does.

        Args:
            condition: The Condition object (LogicCondition or FilterCondition)
            param_start: Starting parameter number

        Returns:
            Tuple of (SQL clause, parameters)

        Raises:
            TypeError: If the tree holds a condition other than a
                ``FilterCondition`` or ``LogicCondition``, which no clause
                renders.
        """
        from ..query_logic import FilterCondition, LogicCondition, LogicOperator

        # Handle FilterCondition (leaf node)
        if isinstance(condition, FilterCondition):
            clause, filter_params = self._build_filter_clause(condition.filter, param_start)
            return clause, filter_params

        # Handle LogicCondition (branch node)
        elif isinstance(condition, LogicCondition):
            clauses: list[str] = []
            params: list[Any] = []
            for sub_condition in condition.conditions:
                sub_clause, sub_params = self._build_complex_condition(
                    sub_condition, param_start + len(params)
                )
                clauses.append(sub_clause)
                params.extend(sub_params)

            if condition.operator == LogicOperator.AND:
                return (f"({' AND '.join(clauses)})" if clauses else "TRUE"), params

            any_clause = f"({' OR '.join(clauses)})" if clauses else "FALSE"
            if condition.operator == LogicOperator.OR:
                return any_clause, params
            if condition.operator == LogicOperator.NOT:
                return f"({any_clause} IS NOT TRUE)", params

        # Rendered as nothing, it matched every record whatever it answers.
        raise TypeError(f"Cannot render {type(condition).__name__} as a SQL condition")

    def build_where_clause(
        self, query: Query | None, param_start: int = 1
    ) -> tuple[str, list[Any]]:
        """Build just the WHERE clause from a Query object.

        Args:
            query: The Query object (can be None)
            param_start: Starting parameter number for placeholders

        Returns:
            Tuple of (WHERE clause SQL, parameters)
            Returns empty string and empty list if no filters
        """
        where, params = self._filters_clause(query.filters if query else [], param_start)
        if not where:
            return "", []
        return " AND " + where, params

    def _filters_clause(
        self, filters: Sequence[Filter], param_start: int = 1
    ) -> tuple[str, list[Any]]:
        """AND filters together, numbering placeholders from ``param_start``.

        The one place a filter list becomes SQL, shared by the search, the
        count and :meth:`build_where_clause`, so the three cannot disagree on
        which rows a query selects. The layout's scope comes first, so its
        parameters are numbered first.

        Args:
            filters: The filters to combine.
            param_start: The number of the first placeholder.

        Returns:
            Tuple of (the clause without ``WHERE``, parameters), or
            ``("", [])`` when there are no filters.
        """
        where_clauses = []
        params: list[Any] = []
        param_count = param_start - 1

        for filter_spec in [*self.layout.scope, *filters]:
            param_count += 1
            clause, new_params = self._build_filter_clause(filter_spec, param_count)
            where_clauses.append(clause)
            params.extend(new_params)
            param_count += len(new_params) - 1  # Adjust for multiple params

        return " AND ".join(where_clauses), params

    def _paging_clauses(self, query: Query | ComplexQuery) -> list[str]:
        """The ``ORDER BY``, ``LIMIT`` and ``OFFSET`` clauses a query asks for.

        Shared by the plain and the complex search. A count takes none of
        them (see :meth:`build_count_query`).

        Args:
            query: The query whose sort and paging to render.

        Returns:
            The clauses in statement order; empty when the query names none.
        """
        clauses = []
        if query.sort_specs:
            order_parts: list[str] = []
            for sort_spec in query.sort_specs:
                direction = "DESC" if sort_spec.order == SortOrder.DESC else "ASC"
                order_parts.extend(
                    f"{key} {direction}" for key in self._build_sort_keys(sort_spec.field)
                )
            clauses.append("ORDER BY " + ", ".join(order_parts))

        # ``is not None`` so ``limit=0`` becomes ``LIMIT 0`` (zero rows)
        # rather than being silently dropped.
        if query.limit_value is not None:
            clauses.append(f"LIMIT {query.limit_value}")
        elif query.offset_value is not None and self.dialect == "sqlite":
            # SQLite accepts OFFSET only after a LIMIT; -1 is its "no limit".
            clauses.append("LIMIT -1")
        if query.offset_value is not None:
            clauses.append(f"OFFSET {query.offset_value}")
        return clauses

    def build_search_query(self, query: Query) -> tuple[str, list[Any]]:
        """Build a SELECT query from a Query object.

        Args:
            query: The Query object

        Returns:
            Tuple of (SQL query, parameters)

        Raises:
            ValueError: If any filter field contains invalid characters.
        """
        sql_parts = [f"SELECT {self.layout.select_list(self)} FROM {self.qualified_table}"]

        where, params = self._filters_clause(query.filters)
        if where:
            sql_parts.append("WHERE " + where)

        sql_parts.extend(self._paging_clauses(query))

        return " ".join(sql_parts), params

    def _values_statements(
        self,
        rows: Sequence[Sequence[Any]],
        render: Callable[[str], str],
        max_parameters: int | None,
        *,
        row_suffix: str = "",
    ) -> list[SQLStatement]:
        """``rows`` as a ``VALUES`` list, over as few statements as the ceiling allows.

        Each statement numbers its placeholders from 1 and binds at most
        ``max_parameters`` values (``None``: no ceiling, so one statement).
        The caller runs them in one transaction; that is what keeps a batch
        all-or-nothing however many statements it takes.

        Args:
            rows: The rows, each the values its tuple binds, all one width.
            render: The statement around the rendered ``VALUES`` list.
            max_parameters: The most values one statement may bind.
            row_suffix: Text closing each tuple after its placeholders.

        Raises:
            ValueError: a ceiling too low to carry one row.
        """
        width = len(rows[0])
        per_statement = len(rows) if max_parameters is None else max_parameters // width
        if per_statement < 1:
            raise ValueError(
                f"a statement of at most {max_parameters} parameters cannot carry "
                f"one row of {width}"
            )
        statements: list[SQLStatement] = []
        for start in range(0, len(rows), per_statement):
            params: list[Any] = []
            tuples: list[str] = []
            for row in rows[start : start + per_statement]:
                marks = ", ".join(
                    self.param_placeholder(len(params) + k) for k in range(1, width + 1)
                )
                tuples.append(f"({marks}{row_suffix})")
                params.extend(row)
            statements.append((render(", ".join(tuples)), params))
        return statements

    @_writes
    def build_batch_update_queries(
        self, updates: list[tuple[str, Record]], *, max_parameters: int | None = None
    ) -> list[SQLStatement]:
        """Build the statements that update records by id, as a join.

        Each record's data and metadata are replaced by its update's. The
        updates are a ``VALUES`` list joined to the table on ``id``, so the
        work grows with the batch rather than with its square, and each row
        reads its own values by column rather than by position. A repeated id
        takes its **last** update, as a loop of ``update`` calls leaves it.

        On PostgreSQL each statement returns the ids it updated; other dialects
        return nothing, and the caller asks which ids exist
        (:meth:`build_existing_ids_query`).

        ``UPDATE … FROM`` needs SQLite 3.33 or later, so the SQLite backends
        use :meth:`build_batch_update_rows` instead, which runs on any version.

        Args:
            updates: ``(id, record)`` pairs, in order.
            max_parameters: The most values one statement may bind; ``None``
                for no ceiling.

        Returns:
            The statements, to run in one transaction; none for no updates.
        """
        if not updates:
            return []

        # Ordered id -> (data_json, metadata_json), last occurrence wins.
        rows: dict[str, tuple[str, str | None]] = {}
        for record_id, record in updates:
            metadata_json = json.dumps(record.metadata) if record.metadata else None
            rows[record_id] = (self._record_to_json(record), metadata_json)

        if self.dialect == "sqlite":
            # SQLite names the columns of a VALUES list column1, column2, ...
            # and takes no alias list for them.
            alias, id_col, data, metadata = "v", "v.column1", "v.column2", "v.column3"
        else:
            alias, id_col, data, metadata = "v(id, data, metadata)", "v.id", "v.data", "v.metadata"
        if self.dialect == "postgres":
            # A parameter in a VALUES list has no column to take a type from.
            data, metadata = f"{data}::jsonb", f"{metadata}::jsonb"
        returning = " RETURNING target.id" if self.dialect == "postgres" else ""

        def render(values: str) -> str:
            return (
                f"UPDATE {self.qualified_table} AS target"
                f" SET data = {data}, metadata = {metadata}, updated_at = CURRENT_TIMESTAMP"
                f" FROM (VALUES {values}) AS {alias}"
                f" WHERE target.id = {id_col}{returning}"
            )

        return self._values_statements(
            [(record_id, *values) for record_id, values in rows.items()], render, max_parameters
        )

    @_writes
    def build_batch_update_rows(
        self, updates: list[tuple[str, Record]]
    ) -> tuple[str, list[list[Any]]]:
        """Build one update by id and its parameters for each update, for ``executemany``.

        The statement is :meth:`build_update_query`'s, so each row binds three
        values whatever the batch's size, and each record reads its own. Run in
        order, a repeated id ends as its **last** update.

        Args:
            updates: ``(id, record)`` pairs, in order.

        Returns:
            The statement and one parameter list per update; an empty query
            for no updates.
        """
        if not updates:
            return "", []
        query, _ = self.build_update_query(*updates[0])
        return query, [self.build_update_query(rid, record)[1] for rid, record in updates]

    @_writes
    def build_batch_create_queries(
        self,
        records: list[Record],
        id_factory: Callable[[], str] | None = None,
        *,
        max_parameters: int | None = None,
    ) -> tuple[list[SQLStatement], list[str]]:
        """Build the multi-row INSERT statements that create records.

        Like ``create()``, a caller-supplied ``record.id`` is honored (an id is
        minted only when a record has none); the id primary-key constraint makes
        a colliding id fail closed at the store (the executor translates the
        driver violation to ``DuplicateRecordError``). A duplicate id *within*
        the batch is detected here and raises ``DuplicateRecordError`` before
        any SQL runs, since it would otherwise surface as an opaque constraint
        error, or not at all when the two rows land in different statements.

        Args:
            records: List of records to insert
            id_factory: The backend's ``_generate_id`` hook, used to mint an id
                for a record that carries none — so a consumer overriding the
                hook governs batch mints too. Defaults to a random UUID4 when a
                direct caller supplies no factory.
            max_parameters: The most values one statement may bind; ``None``
                for no ceiling.

        Returns:
            The statements, to run in one transaction, and the ids in input
            order.

        Raises:
            DuplicateRecordError: two records in the batch share an id.
        """
        if not records:
            return [], []

        mint = id_factory if id_factory is not None else (lambda: str(uuid.uuid4()))

        ids: list[str] = []
        rows: list[tuple[str, str, str | None]] = []
        seen: set[str] = set()
        for record in records:
            record_id = record.id or mint()
            if record_id in seen:
                raise DuplicateRecordError(record_id)
            seen.add(record_id)
            ids.append(record_id)
            metadata_json = json.dumps(record.metadata) if record.metadata else None
            rows.append((record_id, self._record_to_json(record), metadata_json))

        returning = " RETURNING id" if self.dialect == "postgres" else ""

        def render(values: str) -> str:
            return (
                f"INSERT INTO {self.qualified_table} (id, data, metadata, created_at, updated_at)"
                f" VALUES {values}{returning}"
            )

        statements = self._values_statements(
            rows, render, max_parameters, row_suffix=", CURRENT_TIMESTAMP, CURRENT_TIMESTAMP"
        )
        return statements, ids

    @_writes
    def build_batch_upsert_queries(
        self,
        records: list[Record],
        id_factory: Callable[[], str] | None = None,
        *,
        max_parameters: int | None = None,
    ) -> tuple[list[SQLStatement], list[str]]:
        """Build the INSERT ... ON CONFLICT DO UPDATE statements that upsert records.

        The batch analogue of :meth:`build_batch_create_queries` with upsert
        semantics: a caller-supplied ``record.id`` is honored (an id is minted
        only when absent) and an id already present is overwritten (never
        raised). All three SQL dialects (PostgreSQL, SQLite, DuckDB) support
        ``ON CONFLICT (id) DO UPDATE``.

        One ``ON CONFLICT`` statement cannot affect the same row twice, so
        within-batch duplicate ids are coalesced keeping the **last** occurrence
        (last-wins, matching a per-record ``upsert`` loop) before the rows are
        split into statements; the returned id list still carries one entry per
        input record, in input order.

        Args:
            records: List of records to upsert
            id_factory: The backend's ``_generate_id`` hook, used to mint an id
                for a record that carries none — so a consumer overriding the
                hook governs batch upsert mints too. Defaults to a random UUID4
                when a direct caller supplies no factory.
            max_parameters: The most values one statement may bind; ``None``
                for no ceiling.

        Returns:
            The statements, to run in one transaction, and the ids in input
            order.
        """
        if not records:
            return [], []

        mint = id_factory if id_factory is not None else (lambda: str(uuid.uuid4()))

        ids: list[str] = []
        # Ordered id -> (data_json, metadata_json), last occurrence wins.
        rows: dict[str, tuple[str, str | None]] = {}
        for record in records:
            record_id = record.id or mint()
            ids.append(record_id)
            metadata_json = json.dumps(record.metadata) if record.metadata else None
            rows[record_id] = (self._record_to_json(record), metadata_json)

        returning = " RETURNING id" if self.dialect == "postgres" else ""

        # EXCLUDED.updated_at is the CURRENT_TIMESTAMP from this row's VALUES
        # clause (i.e. "now"); referencing it instead of a bare CURRENT_TIMESTAMP
        # in the SET clause keeps the statement portable — DuckDB's binder
        # rejects a bare CURRENT_TIMESTAMP there, PostgreSQL/SQLite accept both.
        def render(values: str) -> str:
            return (
                f"INSERT INTO {self.qualified_table} (id, data, metadata, created_at, updated_at)"
                f" VALUES {values}"
                " ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data,"
                " metadata = EXCLUDED.metadata, updated_at = EXCLUDED.updated_at"
                f"{returning}"
            )

        statements = self._values_statements(
            [(record_id, *values) for record_id, values in rows.items()],
            render,
            max_parameters,
            row_suffix=", CURRENT_TIMESTAMP, CURRENT_TIMESTAMP",
        )
        return statements, ids

    @_writes
    def build_batch_create_query(
        self, records: list[Record], id_factory: Callable[[], str] | None = None
    ) -> tuple[str, list[Any], list[str]]:
        """Deprecated: use :meth:`build_batch_create_queries`.

        The whole batch as one statement, with no parameter ceiling.
        """
        _warn_renamed("build_batch_create_query", "build_batch_create_queries")
        [(query, params)], ids = self.build_batch_create_queries(records, id_factory)
        return query, params, ids

    @_writes
    def build_batch_upsert_query(
        self, records: list[Record], id_factory: Callable[[], str] | None = None
    ) -> tuple[str, list[Any], list[str]]:
        """Deprecated: use :meth:`build_batch_upsert_queries`.

        The whole batch as one statement, with no parameter ceiling.
        """
        _warn_renamed("build_batch_upsert_query", "build_batch_upsert_queries")
        [(query, params)], ids = self.build_batch_upsert_queries(records, id_factory)
        return query, params, ids

    @_writes
    def build_batch_update_query(self, updates: list[tuple[str, Record]]) -> SQLStatement:
        """Deprecated: use :meth:`build_batch_update_queries`.

        The whole batch as one statement, with no parameter ceiling. It is now
        the join :meth:`build_batch_update_queries` builds, so on SQLite it
        needs 3.33 or later; :meth:`build_batch_update_rows` does not.
        """
        _warn_renamed("build_batch_update_query", "build_batch_update_queries")
        if not updates:
            return "", []
        [statement] = self.build_batch_update_queries(updates)
        return statement

    @_writes
    def build_batch_delete_query(self, ids: list[str]) -> SQLStatement:
        """Build one DELETE statement for records by id, whatever their number.

        The ids are a membership list, bound as :meth:`membership_clause`
        binds one, so a dialect whose driver caps a statement's parameters
        still deletes any number in one statement. On PostgreSQL it returns
        the ids it deleted.

        Args:
            ids: List of record IDs to delete

        Returns:
            Tuple of (SQL query, parameters); an empty query for no ids.
        """
        if not ids:
            return "", []
        clause, params = self.membership_clause("id", Operator.IN, ids, 1)
        returning = " RETURNING id" if self.dialect == "postgres" else ""
        return f"DELETE FROM {self.qualified_table} WHERE {clause}{returning}", params

    @_writes
    def build_existing_ids_query(self, ids: list[str]) -> SQLStatement:
        """Build one SELECT of which of ``ids`` are stored, whatever their number.

        Bound as :meth:`build_batch_delete_query` binds its ids.

        Args:
            ids: Record IDs to look for.

        Returns:
            Tuple of (SQL query, parameters); an empty query for no ids.
        """
        if not ids:
            return "", []
        clause, params = self.membership_clause("id", Operator.IN, ids, 1)
        return f"SELECT id FROM {self.qualified_table} WHERE {clause}", params

    def build_count_query(self, query: Query | None = None) -> tuple[str, list[Any]]:
        """Build a COUNT query.

        The count is of the whole match: the query's sort, limit and offset
        never enter the statement, so one ``Query`` can page a search and
        total it.

        Args:
            query: Optional Query object for filtering

        Returns:
            Tuple of (SQL query, parameters)
        """
        sql = f"SELECT COUNT(*) FROM {self.qualified_table}"
        where, params = self._filters_clause(query.filters if query is not None else [])
        if not where:
            return sql, []
        return f"{sql} WHERE {where}", params

    def _build_sort_keys(self, field: str) -> list[str]:
        """Build the ``ORDER BY`` keys for one sorted field, through the layout.

        Every sort reaches the layout through this method.

        Args:
            field: The sorted field.

        Returns:
            The SQL expressions to order by, most significant first; the
            caller applies the direction to each.
        """
        return self.layout.sort_keys(self, field)

    def _jsonb_sort_keys(self, field: str) -> list[str]:
        """The JSON layout's ``ORDER BY`` keys for one field, supporting dot-notation.

        Routes ``id`` to the ``id`` column, ``metadata.*`` fields to the
        ``metadata`` column, and everything else to ``data``.  Uses the
        JSON-preserving extraction (``->`` in Postgres) so that sorting
        preserves JSON type ordering.

        Text sorts by code point, as the in-memory sort does. SQLite and
        DuckDB already compare strings that way. Postgres compares ``jsonb``
        strings under the database collation, and ``COLLATE`` cannot attach to
        ``jsonb``, so a Postgres JSON field takes two keys: the ``jsonb`` value
        with every string folded to ``""``, which keeps the JSON type order and
        ties the strings, then the text value in code-point order, which breaks
        that tie. See :meth:`code_point_order` for the ``id`` column.

        Args:
            field: Field name, optionally dot-separated.

        Returns:
            The SQL expressions to order by, most significant first; the
            caller applies the direction to each.
        """
        if is_storage_key_field(field):
            return [self.code_point_order("id")]

        column, nested_path = resolve_json_column_and_path(field)
        typed = self._build_json_field_expr(nested_path, column=column, as_text=False)
        if self.dialect != "postgres":
            return [typed]
        text = self._build_json_field_expr(nested_path, column=column)
        return [
            f"CASE WHEN jsonb_typeof({typed}) = 'string' THEN '\"\"'::jsonb ELSE {typed} END",
            self.code_point_order(text),
        ]

    def code_point_order(self, expr: str) -> str:
        """Make a text expression compare by code point, as ``Filter.matches`` does.

        Postgres compares text under the database collation, so under
        ``en_US.utf8`` ``'apple' > 'B'`` where Python says the opposite.
        ``COLLATE "C"`` compares bytes, and UTF-8 byte order is code-point
        order. SQLite (``BINARY``) and DuckDB compare that way already.

        The records table declares ``id`` with ``COLLATE "C"``, so on a table
        this package created the clause matches the primary-key index and the
        index still serves range and sort. On a table created before that,
        the answer is right and the index no longer serves those two; equality
        never takes the clause.
        """
        if self.dialect == "postgres":
            return f'({expr}) COLLATE "C"'
        return expr

    def _build_json_field_expr(
        self,
        field: str,
        column: str = "data",
        *,
        as_text: bool = True,
    ) -> str:
        """Build a SQL expression to extract a value from a JSON/JSONB column.

        Supports dot-notation for nested field access. For example,
        ``"config.timeout"`` extracts the nested path ``data->'config'->>'timeout'``
        in PostgreSQL, ``json_extract(data, '$.config.timeout')`` in SQLite, etc.

        .. important::

           Dot-notation is a **path separator** — a dot in a field name is
           *always* interpreted as nesting.  If a JSON key literally contains a
           dot (e.g. ``{"my.field": 1}``), it **cannot** be addressed through
           this query interface.  This matches the semantics of
           ``Record.get_value()``, which applies the same convention.

        Args:
            field: Field name, optionally dot-separated for nested access.
            column: The SQL column to extract from — must be ``"data"`` or
                ``"metadata"``.  Other values raise ``ValueError``.
            as_text: If ``True`` (default), extract the leaf value as text
                (``->>`` in Postgres).  If ``False``, preserve the JSON type
                (``->`` in Postgres) — useful for ORDER BY.

        Returns:
            A SQL expression that extracts the field value.

        Raises:
            ValueError: If *column* is not an allowed column name, or if any
                field name segment contains characters outside
                ``[A-Za-z_][A-Za-z0-9_]*``.
        """
        _allowed_columns = {"data", "metadata"}
        if column not in _allowed_columns:
            raise ValueError(f"column must be one of {_allowed_columns!r}, got {column!r}")

        # Validate field path segments to prevent SQL injection. The same
        # function the Postgres ``stream_read`` twins pre-flight, so what they
        # accept and what this interpolates cannot drift apart.
        validate_field_path(field)
        parts = field.split(".")

        if self.dialect == "postgres":
            # Build chained extraction operators
            chain = column
            for part in parts[:-1]:
                chain += f"->'{part}'"
            # Final segment: ->> for text, -> for jsonb
            leaf_op = "->>" if as_text else "->"
            return f"{chain}{leaf_op}'{parts[-1]}'"
        elif self.dialect == "sqlite":
            # json_extract returns typed values in SQLite — as_text has no effect
            return f"json_extract({column}, '$.{field}')"
        elif self.dialect == "duckdb":
            func = "json_extract_string" if as_text else "json_extract"
            return f"{func}({column}, '$.{field}')"
        else:
            return field

    #: Operators that order their operands, and so depend on a collation when
    #: the operands are text: a layout wraps a text expression compared by one
    #: in :meth:`code_point_order`.
    ORDERED_OPERATORS = frozenset(
        {
            Operator.GT,
            Operator.GTE,
            Operator.LT,
            Operator.LTE,
            Operator.BETWEEN,
            Operator.NOT_BETWEEN,
        }
    )

    #: Operators that compare a value with a bound, and so match only a value
    #: of the bound's kind: the operators :meth:`typed_clause` renders.
    TYPED_OPERATORS = ORDERED_OPERATORS | {
        Operator.EQ,
        Operator.NEQ,
        Operator.IN,
        Operator.NOT_IN,
    }

    #: Operators whose in-memory ``Filter.matches`` contract requires the field
    #: value to be a string (a non-string value never matches). A layout
    #: renders one over a text expression with :meth:`operator_clause`, and
    #: answers ``FALSE`` for a value that is not text. On a JSON field the
    #: SQL text projection would otherwise coerce non-string values to text
    #: and match, so these get a JSON-string-type guard AND'd in --- see
    #: :meth:`_json_string_guard` and :meth:`_build_filter_clause`.
    STRING_ONLY_OPERATORS = frozenset(
        {Operator.LIKE, Operator.NOT_LIKE, Operator.REGEX, Operator.STARTS_WITH}
    )

    def _json_type_is(self, field: str, column: str, kind: str) -> str | None:
        """A predicate that the JSON value at ``column.field`` is of ``kind``.

        A time is stored as a JSON string, so it asks for a string.
        Returns ``None`` for a dialect that cannot read a JSON type. ``field``
        is validated by the :meth:`_build_json_field_expr` call that precedes
        every use; ``column`` is our own ``"data"``/``"metadata"``.
        """
        names = _JSON_TYPE_NAMES.get(self.dialect)
        if names is None:
            return None
        options = names["string" if kind in _TIME_READINGS else kind]
        if self.dialect == "postgres":
            json_expr = self._build_json_field_expr(field, column=column, as_text=False)
            type_expr = f"jsonb_typeof({json_expr})"
        else:
            type_expr = f"json_type({column}, '$.{field}')"
        if len(options) == 1:
            return f"{type_expr} = '{options[0]}'"
        return f"{type_expr} IN ({', '.join(f"'{name}'" for name in options)})"

    def _json_string_guard(self, field: str, column: str) -> str | None:
        """Return a predicate asserting the JSON value at ``column.field`` is a string.

        The string-only operators (LIKE / NOT_LIKE / REGEX / STARTS_WITH) match
        only string values in the in-memory :meth:`Filter.matches` matcher
        (``isinstance(record_value, str)``). Without this guard the SQL push-down
        diverges: the text projection (``data->>'f'`` / ``json_extract`` /
        ``json_extract_string``) coerces a numeric/boolean JSON value to text and
        matches it. AND'ing this guard in keeps SQL and in-memory in agreement.

        Returns ``None`` for a dialect without JSON type introspection (the
        ``standard`` fallback), leaving behavior unchanged there.
        """
        return self._json_type_is(field, column, "string")

    def _typed_json_value(self, field: str, column: str, kind: str) -> tuple[str, str] | None:
        """A test that the value at ``column.field`` is of ``kind``, and the value.

        The value is in ``kind``'s SQL type. A string needs no cast, so it is
        the text projection itself, and an index on that serves an equality or
        membership comparison. Any other kind is cast inside a ``CASE`` on the
        same test, so a value of another kind is never cast: PostgreSQL does
        not promise to evaluate an ``AND``-ed test before a cast.

        A time is a JSON string that names one, read as ``kind`` says
        (:meth:`time_reading`).

        Returns ``None`` for a dialect that cannot read a JSON type.
        """
        text = self._build_json_field_expr(field, column=column)
        if kind in (_INTEGER_NUMBER, _REAL_NUMBER):
            is_integer, is_real = self._duckdb_number_populations(field, column)
            if kind == _INTEGER_NUMBER:
                return is_integer, f"CASE WHEN {is_integer} THEN TRY_CAST({text} AS HUGEINT) END"
            return is_real, f"CASE WHEN {is_real} THEN TRY_CAST({text} AS DOUBLE) END"
        is_kind = self._json_type_is(field, column, kind)
        if is_kind is None:
            return None
        if kind in _TIME_READINGS:
            names_time, value = self.time_reading(text, kind)
            is_kind = f"{is_kind} AND {names_time}"
        elif kind == "string" or self.dialect == "sqlite":
            # SQLite's json_extract already answers in the value's own type.
            return is_kind, text
        elif self.dialect == "postgres":
            value = f"({text})::{'numeric' if kind == 'number' else 'boolean'}"
        else:
            value = f"TRY_CAST({text} AS {_DUCKDB_CASTS[kind]})"
        return is_kind, f"CASE WHEN {is_kind} THEN {value} END"

    def _duckdb_number_populations(self, field: str, column: str) -> tuple[str, str]:
        """Tests that the JSON number at ``column.field`` is an integer, or a real.

        DuckDB has no type holding every JSON number exactly: a ``DOUBLE``
        rounds an integer past 2**53, and a ``HUGEINT`` rounds a fraction. So
        a number written as an integer literal that a ``HUGEINT`` holds is an
        integer, read as one, and any other number is a real, read as a
        ``DOUBLE``, which holds the float Python wrote exactly. DuckDB types an
        integer literal past 64 bits ``DOUBLE``, so the literal is asked, not
        the type. An integer past 128 bits is read as a ``DOUBLE``: the one
        place DuckDB cannot read a JSON number exactly.

        Each test is false, not ``NULL``, for a present value it excludes.
        """
        text = self._build_json_field_expr(field, column=column)
        is_number = self._json_type_is(field, column, "number")
        integral = (
            f"regexp_full_match({text}, '-?[0-9]+') AND TRY_CAST({text} AS HUGEINT) IS NOT NULL"
        )
        return f"({is_number} AND {integral})", f"({is_number} AND NOT ({integral}))"

    def time_reading(self, text: str, reading: str) -> tuple[str, str]:
        """A test that the string ``text`` names a time of ``reading``'s kind, and the time.

        ``Filter.matches`` reads a string as a time through
        :func:`~dataknobs_data.query.read_timestamp`: ISO extended form, each
        field in its range, naming a real day, and with or without a zone.
        With none it is a naive ``datetime``, which ``Filter.matches`` never
        orders against an aware one, so the two are separate kinds here too:

        - :data:`~.sql_types.NAIVE_TIME`: a string of
          :data:`~dataknobs_data.query.NAIVE_TIMESTAMP_SHAPE`, read as the
          time it names.
        - :data:`~.sql_types.ZONED_INSTANT`: a string of
          :data:`~dataknobs_data.query.ZONED_TIMESTAMP_SHAPE`, read as the
          instant it names. Both sides carry their offset, so no session time
          zone enters the comparison.
        - :data:`~.sql_types.ZONED_WALL_CLOCK`: the same strings, read as the time on their
          own clock, which is how a ``date`` bound orders one.

        A string naming none is a string, as ``Filter.matches`` reads it:
        unmatched, and matched by a negated operator. Refusing it in the test,
        rather than letting its cast answer ``NULL``, is what keeps the clause
        two-valued, so ``NOT`` over it still matches.

        The shape bounds each field. What it cannot refuse is a day past its
        month's end (``2024-02-30``): DuckDB's ``TRY_CAST`` answers ``NULL``
        for it, and SQLite, which reads it as the next month, is asked whether
        the date comes back as written. PostgreSQL 15 has no cast that answers
        ``NULL`` (``pg_input_is_valid`` arrives in 16), so such a value still
        raises there. SQLite has no timestamp type: ``strftime`` reads the
        value and it is compared as fixed-width text, which the bound is
        rendered to match (:meth:`bind_bound`), a zoned value in UTC and a
        day early, since its date functions write only the years 0000 to 9999
        (:func:`_utc_text`).

        ``text`` is a text expression: a JSON field, the ``id`` column, or a
        layout's text column. The value is to be read only where the test
        holds, as ``CASE WHEN <test> THEN <time> END``.
        """
        if reading == _NAIVE_TIME:
            return self._naive_time_reading(text)
        by_instant = reading == _ZONED_INSTANT
        if self.dialect == "postgres":
            # A zone in the input of a ``timestamp`` cast is ignored, which
            # leaves the wall clock.
            cast = "timestamptz" if by_instant else "timestamp"
            return f"{text} ~ '{ZONED_TIMESTAMP_SHAPE}'", f"({text})::{cast}"
        wall_clock_text = (
            f"regexp_replace({text}, '{_ZONE_SUFFIX}', '')"
            if self.dialect == "duckdb"
            else f"substr({text}, 1, length({text}) - CASE WHEN {text} GLOB '*Z' THEN 1 ELSE 6 END)"
        )
        naive_test, wall_clock = self._naive_time_reading(wall_clock_text)
        if self.dialect == "duckdb":
            # DuckDB reads a zone only after the seconds.
            with_seconds = (
                f"CASE WHEN substr({text}, 17, 1) IN ('Z', '+', '-') "
                f"THEN substr({text}, 1, 16) || ':00' || substr({text}, 17) ELSE {text} END"
            )
            instant = f"TRY_CAST({with_seconds} AS TIMESTAMPTZ)"
            names_time = (
                f"regexp_full_match({text}, '{ZONED_TIMESTAMP_SHAPE}') "
                f"AND {self._when_present(text, f'{instant} IS NOT NULL')}"
            )
            return names_time, instant if by_instant else wall_clock
        # A zone after a time, in range; the time before it as a naive one;
        # and an instant SQLite reads, which it writes in UTC, a day early.
        seconds_utc = f"strftime('%Y-%m-%dT%H:%M:%S', {text}, '-1 day')"
        names_time = (
            f"({text} GLOB '*Z' OR ({text} GLOB '*[+-][0-9][0-9]:[0-9][0-9]' "
            f"AND substr({text}, -5, 2) < '24' AND substr({text}, -2) < '60')) "
            f"AND length({wall_clock_text}) > 10 AND {naive_test} "
            f"AND {self._when_present(text, f'{seconds_utc} IS NOT NULL')}"
        )
        if not by_instant:
            return names_time, wall_clock
        return names_time, f"{seconds_utc} || {self._sqlite_fraction(wall_clock_text)}"

    def _naive_time_reading(self, text: str) -> tuple[str, str]:
        """:meth:`time_reading` for a string with no zone."""
        if self.dialect == "postgres":
            return f"{text} ~ '{NAIVE_TIMESTAMP_SHAPE}'", f"({text})::timestamp"
        if self.dialect == "duckdb":
            readable = f"TRY_CAST({text} AS TIMESTAMP) IS NOT NULL"
            return (
                f"regexp_full_match({text}, '{NAIVE_TIMESTAMP_SHAPE}') "
                f"AND {self._when_present(text, readable)}",
                f"TRY_CAST({text} AS TIMESTAMP)",
            )
        # The date, alone or followed by a time made of digits, ':' and '.';
        # a year from 0001; a date that comes back as written (SQLite rolls
        # 02-30 over to 03-01, and only the date is re-read, since a time of
        # .999999 rounds into the next day); an hour below 24, which SQLite
        # keeps; and a minute and second ``strftime`` accepts. ``IS`` rather
        # than ``=``, so an unreadable date is false, not ``NULL``.
        readable = f"strftime('%H:%M:%S', {text}) IS NOT NULL"
        names_time = (
            f"({text} GLOB '{_DATE_GLOB}' OR ({text} GLOB "
            f"'{_DATE_GLOB}[T ][0-9][0-9]:[0-9][0-9]*' AND substr({text}, 17) "
            f"NOT GLOB '*[^0-9:.]*' AND substr({text}, 12, 2) < '24')) "
            f"AND substr({text}, 1, 4) <> '0000' "
            f"AND date(substr({text}, 1, 10), '+0 days') IS substr({text}, 1, 10) "
            f"AND {self._when_present(text, readable)}"
        )
        return (
            names_time,
            f"strftime('%Y-%m-%dT%H:%M:%S', {text}) || {self._sqlite_fraction(text)}",
        )

    @staticmethod
    def _sqlite_fraction(text: str) -> str:
        """The fraction of a second in the zone-less time ``text``, to six digits."""
        return (
            f"CASE WHEN instr({text}, '.') > 0 THEN "
            f"substr(substr({text}, instr({text}, '.')) || '000000', 1, 7) "
            "ELSE '.000000' END"
        )

    @staticmethod
    def _when_present(text: str, test: str) -> str:
        """``test`` for a present value, and unknown (``NULL``) for a missing one.

        A test that reads ``NULL`` as a fact (``x IS NOT NULL``) would make a
        kind test false for a missing value where every other kind test is
        unknown, and a ``NOT`` around the clause would then match the missing
        value for this kind alone. How a ``NOT`` treats a missing value is a
        question of its own, so it is answered the same way for every kind.
        """
        return f"CASE WHEN {text} IS NOT NULL THEN {test} END"

    def bind_bound(self, kind: str | None, bound: Any) -> Any:
        """A bound as it is passed to be compared with a value of its kind.

        - A plain ``date`` is its midnight, as ``Filter.matches`` reads it.
          SQLite compares a time as text --- fixed width, ``T``-separated,
          to the microsecond, so text order is time order, as
          :meth:`time_reading` renders the value --- rather than handed
          to the driver's default datetime adapter, which writes a space for
          the ``T``. An aware bound is written in UTC there
          (:func:`_utc_text`), as a zoned value is read.
        - A number no driver takes as it is --- a numpy number, a ``Decimal``
          on SQLite, which cannot bind one --- is an ``int`` when integral and
          a ``float`` otherwise. PostgreSQL and DuckDB take a ``Decimal`` as
          the decimal type that holds it, exactly.
        - A numpy boolean is a ``bool``.
        """
        if kind in _TIME_READINGS:
            if not isinstance(bound, datetime):
                bound = datetime.combine(bound, time.min)
            if self.dialect != "sqlite":
                return bound
            if kind == _ZONED_INSTANT:
                return _utc_text(bound)
            return bound.isoformat(sep="T", timespec="microseconds")
        if kind == "boolean":
            return bool(bound)
        if kind == "number" and type(bound) not in (int, float):
            if isinstance(bound, Integral):
                return int(bound)
            if not (self.dialect in _DECIMAL_DIALECTS and isinstance(bound, Decimal)):
                return float(bound)
        return bound

    def number_domain(self, reading: str | None) -> NumberDomain | None:
        """The numbers a JSON value read under ``reading`` can be, on this dialect.

        ``None`` where a bound needs no comparand: a reading that is not a
        number, or an engine that compares every number exactly as it is sent
        (PostgreSQL reads a JSON number as ``numeric``).
        """
        if reading == "number" and self.dialect == "sqlite":
            return _INT64_OR_DOUBLE
        if reading == _INTEGER_NUMBER:
            return _HUGEINT
        if reading == _REAL_NUMBER:
            return _DOUBLE
        return None

    def bind_comparand(self, reading: str | None, bound: Any, op: Operator) -> Any:
        """A bound as it is passed to be compared by ``op`` under ``reading``.

        :meth:`bind_bound`, except for a number read in a domain the engine
        cannot hold every number in (:meth:`number_domain`): that bound is
        replaced by :func:`~.sql_types.comparand`'s value for it, which the
        engine compares exactly and the driver can always bind. So a bound
        outside 64 bits neither raises on SQLite nor rounds on DuckDB.

        The default ``bind`` of :meth:`typed_clause`.
        """
        domain = self.number_domain(reading)
        if domain is None:
            return self.bind_bound(reading, bound)
        return comparand(domain, op, bound)

    def typed_clause(
        self,
        op: Operator,
        value: Any,
        param_start: int,
        expr_for: Callable[[str | None], tuple[str | None, str] | None],
        present: str,
        *,
        bind: Callable[..., Any] | None = None,
        cast_for: Callable[[str | None, Sequence[Any]], str | None] | None = None,
        readings: Callable[[Any], Mapping[str | None, str | None]] | None = None,
    ) -> tuple[str, list[Any]]:
        """A comparison that matches only values of its bound's kind.

        The clause a layout renders a filter's typed operator with
        (:attr:`TYPED_OPERATORS`): the layout says what a value is under each
        reading of a bound (``expr_for``), and this renders the rest as
        ``Filter.matches`` answers it.

        ``Filter.matches`` relates a value only to a bound of its own kind
        (see :func:`~dataknobs_data.query.value_kind`), and never matches a missing value. So:

        - A positive operator is the field's kind test ``AND``-ed in front of
          the comparison (``expr_for``). A present value of another kind makes
          the clause false rather than ``NULL``, so a ``NOT`` around it still
          reads that value as unmatched, and the comparison itself is left as
          an index on the expression can serve it.
        - A time bound compares under each reading it has (:func:`_readings`):
          a ``date`` is compared with naive and zoned values alike, one part
          for each, ``OR``-ed.
        - A membership list is split by reading, each part compared in its
          own, and the parts ``OR``-ed: ``IN [5, 'c']`` matches ``5`` and ``'c'``.
        - A ``BETWEEN`` compares the values both bounds relate to, each bound
          under its own reading of them, so bounds of different kinds match
          nothing; and so does a ``None`` or NaN bound, which equals and orders
          against nothing (reading :data:`~.sql_types.NEVER`; ``expr_for``
          answers ``None`` for it).
        - A negated operator (``NEQ``, ``NOT_IN``, ``NOT_BETWEEN``) is every
          ``present`` value its positive form does not match, so a value of
          another kind matches it.

        Args:
            op: The filter's operator, one of :attr:`TYPED_OPERATORS`.
            value: The filter's value.
            param_start: Starting parameter number for placeholders.
            expr_for: For a bound's reading, the predicate a value must meet
                to be read so (``None`` when every value is) and the expression
                compared; or ``None`` when no such value can be stored there.
                The predicate is ``AND``-ed in front of the comparison, so it
                must be false, not ``NULL``, for a value it excludes. A reading
                is :func:`~dataknobs_data.query.value_kind`'s name for the
                bound (``"string"``, ``"number"``, ``"boolean"``, or
                :data:`~.sql_types.NEVER`), one of
                :data:`~.sql_types.TIME_READINGS` for a time, or ``None`` for a
                bound with no kind.
            present: A predicate that the field holds a value.
            bind: How a bound is passed as a parameter, given its reading,
                the bound and the operator it is compared by (``GTE`` and
                ``LTE`` for the two sides of a ``BETWEEN``, ``IN`` for a
                membership member); :meth:`bind_comparand` when omitted. A
                layout that sends a bound in its column's own type wraps
                :meth:`bind_bound`. A ``bind`` taking only the reading and the
                bound is called with those two. ``bind`` may answer
                :data:`~.sql_types.MATCHES_NONE` or
                :data:`~.sql_types.MATCHES_ALL` (see
                :func:`~.sql_types.comparand`) for a bound no stored value
                equals, and the comparison is then that constant.
            cast_for: For a reading and the bounds it binds (as bound), the SQL
                type their placeholders are cast to, or ``None`` to leave them
                untyped. For a column whose own type a driver would otherwise
                give the placeholder, converting the bound.
            readings: The populations a bound relates to, each with the reading
                it compares them under; by kind when omitted, with a time
                split into naive and zoned. A layout whose engine reads two
                populations of one kind in different types splits that kind
                too, as DuckDB's JSON layout splits a number into
                :data:`~.sql_types.INTEGER_NUMBER` and
                :data:`~.sql_types.REAL_NUMBER`.

        Returns:
            Tuple of (SQL clause, parameters).
        """
        positive = _NEGATIONS.get(op, op)
        read = _readings if readings is None else readings
        bind_with_op = bind is None or _takes_operator(bind)

        def cast(reading: str | None, bound_values: Sequence[Any]) -> str | None:
            return None if cast_for is None else cast_for(reading, bound_values)

        def bound_as_bound(kind: str | None, bound: Any, compared_by: Operator) -> Any:
            if bind is None:
                return self.bind_comparand(kind, bound, compared_by)
            if bind_with_op:
                return bind(kind, bound, compared_by)
            return bind(kind, bound)

        def guarded(kind_test: str | None, clause: str) -> str:
            return clause if kind_test is None else f"({kind_test} AND {clause})"

        parts: list[str] = []
        params: list[Any] = []
        if positive == Operator.IN:
            groups: dict[str | None, list[Any]] = {}
            for member in membership_values(value):
                for reading in read(member).values():
                    groups.setdefault(reading, []).append(member)
            for reading, members in groups.items():
                target = expr_for(reading)
                if target is None:
                    continue
                bound_members = [
                    bound
                    for bound in (
                        bound_as_bound(reading, member, Operator.IN) for member in members
                    )
                    if bound is not _MATCHES_NONE
                ]
                if bound_members:
                    part, part_params = self.membership_clause(
                        target[1],
                        Operator.IN,
                        bound_members,
                        param_start + len(params),
                        placeholder_type=cast(reading, bound_members),
                    )
                    parts.append(guarded(target[0], part))
                    params.extend(part_params)
        else:
            bounds = value if positive == Operator.BETWEEN else [value]
            by_bound = [read(bound) for bound in bounds]
            for population in by_bound[0]:
                per_bound = [r.get(population, _NEVER) for r in by_bound]
                targets = [t for t in map(expr_for, per_bound) if t is not None]
                if len(targets) < len(per_bound):
                    continue  # a bound relates to none of these values
                # Strings order by code point, as ``Filter.matches`` compares str.
                ordered = positive in self.ORDERED_OPERATORS
                exprs = [
                    self.code_point_order(expr) if r == "string" and ordered else expr
                    for r, (_, expr) in zip(per_bound, targets, strict=True)
                ]
                # Each bound with the operator it is compared by: a range's
                # two sides are its low and high halves.
                ops = [Operator.GTE, Operator.LTE] if positive == Operator.BETWEEN else [positive]
                bound_values = [
                    bound_as_bound(r, b, o) for r, b, o in zip(per_bound, bounds, ops, strict=True)
                ]
                if any(b is _MATCHES_NONE for b in bound_values):
                    continue  # no value of this population meets a side
                sides = [
                    (expr, o, b, r)
                    for expr, o, b, r in zip(exprs, ops, bound_values, per_bound, strict=True)
                    if b is not _MATCHES_ALL
                ]
                start = param_start + len(params)
                if not sides:
                    part, part_params = "TRUE", []
                elif positive != Operator.BETWEEN:
                    part, part_params = self.operator_clause(
                        exprs[0],
                        positive,
                        bound_values[0],
                        start,
                        placeholder_type=cast(per_bound[0], bound_values),
                    )
                elif len(sides) == 2 and per_bound[0] == per_bound[1]:
                    part, part_params = self.operator_clause(
                        exprs[0],
                        positive,
                        bound_values,
                        start,
                        placeholder_type=cast(per_bound[0], bound_values),
                    )
                else:
                    # A date and an aware bound read a zoned value two ways,
                    # and a side every value meets drops out: each side that
                    # is left is one parameter.
                    halves: list[str] = []
                    part_params = []
                    for expr, side_op, side_value, reading in sides:
                        half, half_params = self.operator_clause(
                            expr,
                            side_op,
                            side_value,
                            start + len(part_params),
                            placeholder_type=cast(reading, [side_value]),
                        )
                        halves.append(half)
                        part_params.extend(half_params)
                    part = halves[0] if len(halves) == 1 else f"({' AND '.join(halves)})"
                parts.append(guarded(targets[0][0], part))
                params.extend(part_params)

        if not parts:
            # Nothing of the bound's kind can match: the positive form is
            # false, and its negation is every value the field holds.
            return ("FALSE" if positive is op else present), []
        matched = parts[0] if len(parts) == 1 else f"({' OR '.join(parts)})"
        if positive is op:
            return matched, params
        return f"({present} AND NOT {matched})", params

    def operator_clause(
        self,
        field_expr: str,
        op: Operator,
        value: Any,
        param_start: int,
        *,
        placeholder_type: str | None = None,
    ) -> tuple[str, list[Any]]:
        """Build the comparison clause for a given operator.

        Args:
            field_expr: The fully-qualified, possibly cast, SQL field expression.
            op: The filter ``Operator``.
            value: The filter value.
            param_start: Starting parameter number for placeholders.
            placeholder_type: The SQL type each placeholder of a comparison,
                membership or range is cast to; untyped when ``None``. The cast
                is on the placeholder, never the field, so an index on the
                field still serves the comparison.

        Returns:
            Tuple of (SQL clause, parameters). A clause may bind no parameters
            (``EXISTS``, an empty membership list), so the caller numbers the
            next placeholder from the length of the list returned.
        """

        def placeholder(param_num: int) -> str:
            rendered = self.param_placeholder(param_num)
            return (
                rendered if placeholder_type is None else f"CAST({rendered} AS {placeholder_type})"
            )

        param_placeholder = placeholder(param_start)

        if op == Operator.EQ:
            return f"{field_expr} = {param_placeholder}", [value]
        elif op == Operator.NEQ:
            return f"{field_expr} != {param_placeholder}", [value]
        elif op == Operator.GT:
            return f"{field_expr} > {param_placeholder}", [value]
        elif op == Operator.GTE:
            return f"{field_expr} >= {param_placeholder}", [value]
        elif op == Operator.LT:
            return f"{field_expr} < {param_placeholder}", [value]
        elif op == Operator.LTE:
            return f"{field_expr} <= {param_placeholder}", [value]
        elif op in (Operator.LIKE, Operator.NOT_LIKE):
            return self._build_like_clause(field_expr, op, value, param_placeholder)
        elif op in (Operator.IN, Operator.NOT_IN):
            return self.membership_clause(
                field_expr, op, value, param_start, placeholder_type=placeholder_type
            )
        elif op == Operator.BETWEEN:
            placeholder1 = placeholder(param_start)
            placeholder2 = placeholder(param_start + 1)
            return f"{field_expr} BETWEEN {placeholder1} AND {placeholder2}", list(value)
        elif op == Operator.NOT_BETWEEN:
            placeholder1 = placeholder(param_start)
            placeholder2 = placeholder(param_start + 1)
            return f"{field_expr} NOT BETWEEN {placeholder1} AND {placeholder2}", list(value)
        elif op == Operator.EXISTS:
            return f"{field_expr} IS NOT NULL", []
        elif op == Operator.NOT_EXISTS:
            return f"{field_expr} IS NULL", []
        elif op == Operator.REGEX:
            if self.dialect == "postgres":
                return f"{field_expr} ~ {param_placeholder}", [value]
            elif self.dialect == "duckdb":
                return f"regexp_matches({field_expr}, {param_placeholder})", [value]
            else:
                return f"{field_expr} REGEXP {param_placeholder}", [value]
        elif op == Operator.STARTS_WITH:
            return self._build_starts_with_clause(field_expr, value, param_start)
        else:
            raise ValueError(f"Unsupported operator: {op}")

    def _build_like_clause(
        self,
        field_expr: str,
        op: Operator,
        value: Any,
        param_placeholder: str,
    ) -> tuple[str, list[Any]]:
        r"""Build a ``LIKE`` / ``NOT LIKE`` clause that answers as ``Filter.matches`` does.

        The contract is case-insensitive, with ``%`` and ``_`` the only
        wildcards and every other character, ``\`` among them, matched
        verbatim. Each engine needs its own shape to say that:

        - **postgres** --- ``ILIKE ... ESCAPE ''``. Its ``LIKE`` is
          case-sensitive, and it reads ``\`` as an escape unless told there is
          none. Case folding follows the database's ``LC_CTYPE``.
        - **duckdb** --- ``ILIKE``. Its ``LIKE`` is case-sensitive and has no
          escape character by default.
        - **sqlite** (and any other dialect) --- plain ``LIKE``, which is
          already case-insensitive with no escape character. It folds ASCII
          case only, a documented variation.

        No index is given up: the JSONB layout's GIN indexes serve neither
        form, and ``STARTS_WITH`` keeps its own case-sensitive shape.
        """
        negate = "NOT " if op == Operator.NOT_LIKE else ""
        if self.dialect == "postgres":
            return f"{field_expr} {negate}ILIKE {param_placeholder} ESCAPE ''", [value]
        if self.dialect == "duckdb":
            return f"{field_expr} {negate}ILIKE {param_placeholder}", [value]
        return f"{field_expr} {negate}LIKE {param_placeholder}", [value]

    def membership_clause(
        self,
        field_expr: str,
        op: Operator,
        value: Any,
        param_start: int,
        *,
        placeholder_type: str | None = None,
    ) -> tuple[str, list[Any]]:
        """Build an ``IN`` / ``NOT IN`` clause that answers as ``Filter.matches`` does.

        Only the members that can match are rendered (see
        :func:`~dataknobs_data.query.membership_values`): a ``None`` member
        would make every ``NOT IN`` row ``NULL`` under SQL's three-valued
        logic. When none are left, the clause is written out rather than
        rendered as ``()``, which Postgres and DuckDB reject:

        - ``IN`` an empty list is ``FALSE`` --- nothing is in it.
        - ``NOT IN`` an empty list is ``<field> IS NOT NULL``, not ``TRUE``:
          ``Filter.matches`` never matches a record whose field has no value,
          so this is every record that has one. The empty ``STARTS_WITH``
          prefix takes the same answer for the same reason.

        Neither binds a parameter, so the placeholders that follow are
        numbered as though the clause were absent.

        A dialect whose driver caps how many parameters a statement binds
        passes the whole list as one, so a list of any length is one clause:

        - **postgres** --- ``= ANY(<array>)`` / ``<> ALL(<array>)``. Both
          drivers pass a list as an array; asyncpg caps a statement at 32767
          parameters.
        - **sqlite** --- ``IN (SELECT value FROM json_each(<json>))``, the list
          written as a JSON array. SQLite caps a statement at its connection's
          variable limit, 32766 by default. A member JSON cannot carry
          (``bytes``, a ``UUID``) is left out, as one no stored record holds.
        - Any other dialect --- one placeholder per member.

        ``placeholder_type`` casts the members as
        :meth:`operator_clause` casts a bound: the Postgres array as an
        array of it, and each placeholder of the other dialects. SQLite's
        JSON array is not cast; SQLite compares a member as it is.
        """
        members = membership_values(value)
        if self.dialect == "sqlite":
            # A record's data is stored as JSON, so a member JSON cannot carry
            # is one no record holds, and it cannot go in the JSON array below.
            members = [m for m in members if _json_can_carry(m)]
        if not members:
            if op == Operator.IN:
                return "FALSE", []
            return f"{field_expr} IS NOT NULL", []
        placeholder = self.param_placeholder(param_start)
        if self.dialect == "postgres":
            if placeholder_type is not None:
                placeholder = f"CAST({placeholder} AS {placeholder_type}[])"
            if op == Operator.IN:
                return f"{field_expr} = ANY({placeholder})", [members]
            return f"{field_expr} <> ALL({placeholder})", [members]
        if self.dialect == "sqlite":
            keyword = "IN" if op == Operator.IN else "NOT IN"
            return (
                f"{field_expr} {keyword} (SELECT value FROM json_each({placeholder}))",
                [json.dumps(members)],
            )
        placeholders = ", ".join(
            (
                self.param_placeholder(i)
                if placeholder_type is None
                else f"CAST({self.param_placeholder(i)} AS {placeholder_type})"
            )
            for i in range(param_start, param_start + len(members))
        )
        keyword = "IN" if op == Operator.IN else "NOT IN"
        return f"{field_expr} {keyword} ({placeholders})", members

    def _build_starts_with_clause(
        self,
        field_expr: str,
        value: Any,
        param_start: int,
    ) -> tuple[str, list[Any]]:
        r"""Build a case-sensitive literal-prefix clause for ``STARTS_WITH``.

        Case-sensitivity parity with the in-memory ``str.startswith`` matcher
        is the constraint, and it drives a per-dialect shape:

        - **postgres / duckdb** — ``LIKE '<escaped-prefix>%' ESCAPE '\'``.
          ``LIKE`` is case-sensitive on both engines, and the literal prefix is
          escaped so ``%``/``_`` in it match verbatim.
        - **sqlite** — a half-open range ``field >= prefix AND field <
          upper_bound``. SQLite ``LIKE`` is case-*insensitive* for ASCII, so it
          cannot be used here; ``>=``/``<`` on TEXT use BINARY collation
          (case-sensitive) and ride the ``id`` primary-key index. Two prefixes
          have no finite upper bound: an **empty** prefix (matches every
          non-null value → an always-true clause, no bound params) and a
          **non-empty all-maximal-code-point** prefix (still lower-bounded →
          ``field >= prefix``, one bound param). These are distinct: the
          all-maximal case must keep the lower bound or it over-matches every
          row, diverging from ``str.startswith``.
        """
        if self.dialect in ("postgres", "duckdb"):
            placeholder = self.param_placeholder(param_start)
            escaped = escape_like_prefix(str(value)) + "%"
            return f"{field_expr} LIKE {placeholder} ESCAPE '\\'", [escaped]

        # sqlite (and any other dialect): case-sensitive half-open range scan.
        prefix = str(value)
        upper = prefix_upper_bound(prefix)
        if upper is None:
            if not prefix:
                # Empty prefix — every non-null value starts with it. Always-true
                # clause, no bound parameters consumed.
                return f"{field_expr} IS NOT NULL", []
            # Non-empty prefix whose every code point is maximal (U+10FFFF):
            # no exclusive upper bound exists, but the lower bound still holds.
            # ``>= prefix`` matches exactly the strings at-or-above it (which are
            # precisely those starting with this all-maximal prefix), unlike a
            # bare ``IS NOT NULL`` that would return every row.
            placeholder = self.param_placeholder(param_start)
            return f"{field_expr} >= {placeholder}", [prefix]
        placeholder1 = self.param_placeholder(param_start)
        placeholder2 = self.param_placeholder(param_start + 1)
        return (
            f"({field_expr} >= {placeholder1} AND {field_expr} < {placeholder2})",
            [prefix, upper],
        )

    def _build_filter_clause(self, filter_spec: Filter, param_start: int) -> tuple[str, list[Any]]:
        """Build a WHERE clause for a single filter, through the layout.

        Every filter, the layout's scope and a ``ComplexQuery``'s leaves
        included, reaches the layout through this method.

        Args:
            filter_spec: The filter.
            param_start: Starting parameter number for placeholders.

        Returns:
            Tuple of (SQL clause, parameters).
        """
        return self.layout.filter_clause(self, filter_spec, param_start)

    def _jsonb_filter_clause(self, filter_spec: Filter, param_start: int) -> tuple[str, list[Any]]:
        """The JSON layout's WHERE clause for a single filter.

        Supports dot-notation for nested JSON field access.  A field name
        like ``"metadata.work_order_id"`` is routed to the ``metadata`` JSONB
        column with proper nested path extraction, while a field like
        ``"config.timeout"`` is routed to the ``data`` column.

        .. important::

           Dots in field names are **always** interpreted as path separators,
           matching the semantics of ``Record.get_value()``.  JSON keys that
           literally contain a dot cannot be queried through this interface.

        Args:
            filter_spec: The filter specification (has ``.field``, ``.operator``,
                ``.value`` attributes).
            param_start: Starting parameter number for placeholders.

        Returns:
            Tuple of (SQL clause, parameters).

        Raises:
            ValueError: If a field name segment contains invalid characters.
        """
        field = filter_spec.field
        op = filter_spec.operator
        value = filter_spec.value

        # The reserved storage-key field is a real column, not inside JSON.
        # It is always a string, so a string bound compares it as it is, a
        # time bound reads it as the time it names (if it names one of the
        # bound's kind), and a bound of any other kind never matches it.
        if is_storage_key_field(field):
            if op in self.TYPED_OPERATORS:

                def storage_key_for(kind: str | None) -> tuple[str | None, str] | None:
                    if kind is None or kind == "string":
                        return None, "id"
                    if kind in _TIME_READINGS and self.dialect in _JSON_TYPE_NAMES:
                        names_time, time_value = self.time_reading("id", kind)
                        return names_time, f"CASE WHEN {names_time} THEN {time_value} END"
                    return None

                return self.typed_clause(
                    op,
                    value,
                    param_start,
                    storage_key_for,
                    "id IS NOT NULL",
                )
            return self.operator_clause("id", op, value, param_start)

        # Other fields target the data JSONB column, or the metadata column
        # when prefixed ``metadata.`` (dot-notation nesting).
        column, nested_path = resolve_json_column_and_path(field)
        text_expr = self._build_json_field_expr(nested_path, column=column)

        if op in self.TYPED_OPERATORS and self.dialect in _JSON_TYPE_NAMES:

            def json_value_for(kind: str | None) -> tuple[str | None, str] | None:
                if kind is None:
                    return None, text_expr
                if kind == _NEVER:
                    return None
                return self._typed_json_value(nested_path, column, kind)

            return self.typed_clause(
                op,
                value,
                param_start,
                json_value_for,
                f"{text_expr} IS NOT NULL",
                readings=_split_numbers if self.dialect == "duckdb" else None,
            )

        clause, params = self.operator_clause(text_expr, op, value, param_start)

        # String-only operators match only string values (in-memory contract).
        # On a JSON field, AND in a JSON-string-type guard so a non-string value
        # the text projection would coerce-and-match is excluded — keeping the
        # SQL push-down in agreement with Filter.matches across every backend.
        if op in self.STRING_ONLY_OPERATORS:
            guard = self._json_string_guard(nested_path, column)
            if guard:
                clause = f"({guard} AND {clause})"

        return clause, params

    def _record_to_json(self, record: Record) -> str:
        """Convert a Record to JSON string for storage."""
        return SQLRecordSerializer.record_to_json(record)

    @staticmethod
    def row_to_record(row: dict[str, Any]) -> Record:
        """Convert a database row of the JSON layout to a Record.

        Args:
            row: Database row as dictionary

        Returns:
            Record object
        """
        return SQLRecordSerializer.row_to_record(row)

    def record_from_row(self, row: Mapping[str, Any]) -> Record:
        """Convert a row this builder's statements selected to a Record, by its layout.

        Args:
            row: Database row, by column name.

        Returns:
            Record object
        """
        return self.layout.record_from_row(row)


class SQLTableManager:
    """Manages SQL table creation and schema."""

    def __init__(
        self,
        table_name: str,
        schema_name: str | None = None,
        dialect: str = "standard",
        param_style: str = "qmark",
    ):
        """Initialize the table manager.

        Args:
            table_name: Name of the database table
            schema_name: Optional schema name
            dialect: SQL dialect ('postgres', 'duckdb', 'sqlite', 'standard')
            param_style: Parameter placeholder style — ``'qmark'`` (``?``,
                default, works for DuckDB/SQLite), ``'numeric'`` (``$1``/``$2``
                for asyncpg), or ``'pyformat'`` (``%(name)s`` for psycopg2).
                ``'qmark'`` is invalid for ``dialect='postgres'``; use
                ``'numeric'`` (asyncpg) or ``'pyformat'`` (psycopg2).
        """
        if dialect == "postgres" and param_style == "qmark":
            raise ValueError(
                "param_style='qmark' is not valid for dialect='postgres'. "
                "Use param_style='numeric' for asyncpg or 'pyformat' for psycopg2."
            )
        self.table_name = table_name
        self.schema_name = schema_name
        self.dialect = dialect
        self.param_style = param_style
        self.qualified_table = self._get_qualified_table_name()

    def _get_qualified_table_name(self) -> str:
        """Get the fully qualified table name."""
        if self.schema_name:
            return f"{quote_ident(self.schema_name)}.{quote_ident(self.table_name)}"
        return quote_ident(self.table_name)

    def get_create_table_sql(self) -> str:
        """Get the CREATE TABLE SQL statement.

        Returns:
            SQL statement for creating the table
        """
        if self.dialect == "postgres":
            # ``COLLATE "C"`` so the primary-key index serves the code-point
            # range and sort ``SQLQueryBuilder`` renders.
            return f"""
            CREATE TABLE IF NOT EXISTS {self.qualified_table} (
                id VARCHAR(255) COLLATE "C" PRIMARY KEY,
                data JSONB NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS {quote_ident(f"idx_{self.table_name}_data")}
            ON {self.qualified_table} USING GIN (data);

            CREATE INDEX IF NOT EXISTS {quote_ident(f"idx_{self.table_name}_metadata")}
            ON {self.qualified_table} USING GIN (metadata);
            """
        elif self.dialect == "sqlite":
            # SQLite doesn't have JSONB, uses TEXT for JSON storage
            return f"""
            CREATE TABLE IF NOT EXISTS {self.qualified_table} (
                id VARCHAR(255) PRIMARY KEY,
                data TEXT NOT NULL CHECK (json_valid(data)),
                metadata TEXT CHECK (metadata IS NULL OR json_valid(metadata)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS {quote_ident(f"idx_{self.table_name}_created")}
            ON {self.qualified_table} (created_at);

            CREATE INDEX IF NOT EXISTS {quote_ident(f"idx_{self.table_name}_updated")}
            ON {self.qualified_table} (updated_at);
            """
        elif self.dialect == "duckdb":
            # DuckDB has native JSON type for efficient JSON storage and querying
            return f"""
            CREATE TABLE IF NOT EXISTS {self.qualified_table} (
                id VARCHAR(255) PRIMARY KEY,
                data JSON NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS {quote_ident(f"idx_{self.table_name}_created")}
            ON {self.qualified_table} (created_at);

            CREATE INDEX IF NOT EXISTS {quote_ident(f"idx_{self.table_name}_updated")}
            ON {self.qualified_table} (updated_at);
            """
        else:
            # Generic SQL
            return f"""
            CREATE TABLE IF NOT EXISTS {self.qualified_table} (
                id VARCHAR(255) PRIMARY KEY,
                data TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """

    def _placeholder(self, n: int, name: str) -> str:
        """Return the correct parameter placeholder for this instance's param_style.

        Args:
            n: 1-based positional index (used by 'numeric' style)
            name: Named parameter key (used by 'pyformat' style)
        """
        if self.param_style == "numeric":
            return f"${n}"
        if self.param_style == "pyformat":
            return f"%({name})s"
        return "?"

    def get_table_exists_sql(self) -> tuple[str, tuple[Any, ...] | dict[str, Any]]:
        """Get a parameterized table-existence-check query.

        Returns a ``(sql, params)`` pair. Execute as
        ``cursor.execute(sql, params)`` and read the first column of the
        first row as a bool.

        The placeholder style and param type depend on ``param_style``:
        - ``'qmark'`` (default): ``?`` placeholders, positional tuple
        - ``'numeric'``: ``$1``/``$2`` placeholders (asyncpg), positional tuple
        - ``'pyformat'``: ``%(name)s`` placeholders (psycopg2), named dict
        """
        if self.dialect == "postgres":
            p1 = self._placeholder(1, "schema")
            p2 = self._placeholder(2, "table")
            sql = (
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                f"WHERE table_schema = {p1} AND table_name = {p2}"
                ")"
            )
            schema = self.schema_name or "public"
            if self.param_style == "pyformat":
                return sql, {"schema": schema, "table": self.table_name}
            return sql, (schema, self.table_name)
        # DuckDB and SQLite resolve a table name, quoted or not, whatever its
        # case, so the lookup does too: otherwise a table their queries read is
        # reported missing.
        elif self.dialect == "duckdb":
            sql = (
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE lower(table_schema) = lower(?) AND lower(table_name) = lower(?)"
                ")"
            )
            schema = self.schema_name or "main"
            return sql, (schema, self.table_name)
        elif self.dialect == "sqlite":
            sql = (
                "SELECT EXISTS (SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = ? COLLATE NOCASE)"
            )
            return sql, (self.table_name,)
        else:
            sql = "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = ?)"
            return sql, (self.table_name,)

    @staticmethod
    def coerce_bool(value: Any, default: bool = True) -> bool:
        """Coerce a config value to bool, handling YAML/env string values.

        ``None`` → ``default``. Strings ``"false"`` / ``"0"`` / ``"no"``
        (case-insensitive) → ``False``; any other string → ``True``.
        Non-strings go through ``bool(value)``.
        """
        if value is None:
            return default
        if isinstance(value, str):
            return value.lower() not in ("false", "0", "no", "")
        return bool(value)

    def get_drop_table_sql(self) -> str:
        """Get the DROP TABLE SQL statement.

        Returns:
            SQL statement for dropping the table
        """
        return f"DROP TABLE IF EXISTS {self.qualified_table}"

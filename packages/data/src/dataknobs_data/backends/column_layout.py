# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""How a table's rows are laid out, for :class:`~dataknobs_data.backends.sql_base.SQLQueryBuilder`.

A builder reads its table through one :class:`ColumnLayout`, which decides
what a filter's or a sort's field is in SQL, what a statement selects, how
the key is compared, how a row becomes a :class:`~dataknobs_data.records.Record`,
and whether the table can be written.

- :class:`JsonbLayout` is the table this package creates: ``id``, ``data``
  and ``metadata``, every field a path into one of the two JSON columns. It
  is every builder's default.
- :class:`NativeColumnLayout` is a table somebody else owns, with ordinary
  typed columns. It reads the columns a schema declares and nothing else,
  ANDs a scope into every read, refuses every write, and answers every filter
  as :meth:`~dataknobs_data.query.Filter.matches` answers it over the record
  it returns.

:func:`read_layout_config` reads the three configuration keys a backend
takes for this: ``layout:``, ``id_column:`` and ``scope:``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.exceptions import ValidationError
from dataknobs_utils.sql_utils import quote_ident

from ..fields import FieldType
from ..query import (
    RESERVED_KEY_FIELD,
    Filter,
    Operator,
    is_storage_key_field,
    membership_values,
    value_kind,
)
from ..records import Record
from ..schema import SQL_TYPE_KEY
from .sql_types import (
    NAIVE_TIME,
    TIME_READINGS,
    ZONED_INSTANT,
    ZONED_WALL_CLOCK,
    SqlType,
    field_type_answer,
    sql_types,
)

if TYPE_CHECKING:
    from ..schema import DatabaseSchema
    from .sql_base import SQLQueryBuilder

#: The dialects a native layout renders for: each needs its own text cast,
#: time reading and placeholder type.
NATIVE_DIALECTS = frozenset({"postgres", "sqlite", "duckdb"})

#: The values ``layout:`` takes.
LAYOUTS = ("jsonb", "native")

_EQUALITY = frozenset({Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN})
_RANGES = frozenset({Operator.BETWEEN, Operator.NOT_BETWEEN})
_MEMBERSHIP = frozenset({Operator.IN, Operator.NOT_IN})
_STRING_ONLY = frozenset({Operator.LIKE, Operator.NOT_LIKE, Operator.REGEX, Operator.STARTS_WITH})
_PRESENCE = frozenset({Operator.EXISTS, Operator.NOT_EXISTS})

_TEXT_TYPE = MappingProxyType({"postgres": "TEXT", "sqlite": "TEXT", "duckdb": "VARCHAR"})


class ColumnLayout(ABC):
    """How one table's rows are laid out: what a field is, and what a row becomes."""

    #: Whether the builder may build a statement that writes the table.
    writable: ClassVar[bool]

    @property
    def scope(self) -> tuple[Filter, ...]:
        """The filters every read is ANDed with: which rows the table holds for this store."""
        return ()

    def check_dialect(self, dialect: str) -> None:  # noqa: B027 - most layouts take any
        """Refuse a dialect this layout cannot render for. Called by the builder."""

    @abstractmethod
    def filter_clause(
        self, builder: SQLQueryBuilder, spec: Filter, param_start: int
    ) -> tuple[str, list[Any]]:
        """One filter as a SQL clause and its parameters, numbered from ``param_start``."""

    @abstractmethod
    def sort_keys(self, builder: SQLQueryBuilder, field: str) -> list[str]:
        """The ``ORDER BY`` keys for one field, most significant first."""

    @abstractmethod
    def select_list(self, builder: SQLQueryBuilder) -> str:
        """What a statement reading whole rows selects."""

    @abstractmethod
    def key_clause(
        self, builder: SQLQueryBuilder, record_id: str, param_start: int
    ) -> tuple[str, list[Any]]:
        """The clause selecting the row stored under ``record_id``."""

    @abstractmethod
    def record_from_row(self, row: Mapping[str, Any]) -> Record:
        """A row the driver returned, as a record."""


class JsonbLayout(ColumnLayout):
    """The table this package creates: ``id``, and the record in ``data`` and ``metadata``.

    Every field is a path into ``data``, or into ``metadata`` when it is
    prefixed ``metadata.``; the reserved key field is the ``id`` column.
    """

    writable = True

    def filter_clause(
        self, builder: SQLQueryBuilder, spec: Filter, param_start: int
    ) -> tuple[str, list[Any]]:
        return builder._jsonb_filter_clause(spec, param_start)

    def sort_keys(self, builder: SQLQueryBuilder, field: str) -> list[str]:
        return builder._jsonb_sort_keys(field)

    def select_list(self, builder: SQLQueryBuilder) -> str:
        return "*"

    def key_clause(
        self, builder: SQLQueryBuilder, record_id: str, param_start: int
    ) -> tuple[str, list[Any]]:
        return f"id = {builder._get_param_placeholder(param_start)}", [record_id]

    def record_from_row(self, row: Mapping[str, Any]) -> Record:
        from .sql_base import SQLRecordSerializer

        return SQLRecordSerializer.row_to_record(dict(row))


@dataclass(frozen=True)
class _Column:
    """A declared column: its quoted name, and what it holds (``None``: structured)."""

    name: str
    quoted: str
    field_type: FieldType
    sql_type: SqlType | None


class NativeColumnLayout(ColumnLayout):
    """A table somebody else owns, read through the columns a schema declares.

    - **Only declared columns.** A filter, a sort or a scope naming any other
      column is refused before SQL is built, and a statement selects the
      declared columns rather than ``*``. The reserved key field
      (:data:`~dataknobs_data.query.RESERVED_KEY_FIELD`) is ``id_column``.
    - **Every filter answers as** :meth:`~dataknobs_data.query.Filter.matches`
      **answers over the record returned.** Each column's type says which
      kinds of value it holds (:class:`~dataknobs_data.backends.sql_types.SqlType`).
      A bound of another kind matches nothing, and its negation every present
      value; it is never sent to a driver that would refuse it, or convert it
      and answer wrongly. A ``json``, ``binary`` or vector column is refused for
      every operator but ``EXISTS`` / ``NOT_EXISTS``: no engine compares one
      the way ``Filter.matches`` does.
    - **The scope** is ANDed into every read the builder builds, and a scope
      that could match no row (a value its column cannot hold) is refused.
    - **Read-only.** The builder refuses every statement that writes.

    A declared type is what the answers rest on, and nothing reads the table
    to check it: declare each column as the table holds it.

    Args:
        schema: The table's columns. A column's ``FieldType`` decides what it
            holds unless its ``metadata["sql_type"]`` names a registered
            :class:`~dataknobs_data.backends.sql_types.SqlType`.
        id_column: The declared column that keys a row.
        scope: Filters fixing which rows of the table this store is.

    Raises:
        ValidationError: When the schema declares no column, ``id_column`` is
            not declared, a ``sql_type`` is not a non-empty string or is not
            registered, or a scope filter names an undeclared column or a value
            its column cannot hold.
    """

    writable = False

    def __init__(
        self,
        schema: DatabaseSchema,
        *,
        id_column: str,
        scope: Sequence[Filter] = (),
    ) -> None:
        if not schema.fields:
            raise ValidationError(
                "a native layout reads the columns its schema declares, and this schema "
                "declares none",
                context={"id_column": id_column},
            )
        self._columns: Mapping[str, _Column] = MappingProxyType(
            {
                name: _resolve_column(name, field.type, field.metadata)
                for name, field in schema.fields.items()
            }
        )
        if id_column not in self._columns:
            raise ValidationError(
                f"id_column {id_column!r} is not a declared column; declared: "
                f"{sorted(self._columns)}",
                context={"id_column": id_column, "declared": sorted(self._columns)},
            )
        if self._columns[id_column].sql_type is None:
            raise ValidationError(
                f"id_column {id_column!r} is declared {self._columns[id_column].field_type.value}, "
                f"which a key cannot be compared as",
                context={"id_column": id_column},
            )
        self.id_column = id_column
        for spec in scope:
            self._check_scope_filter(spec)
        self._scope = tuple(scope)

    @property
    def scope(self) -> tuple[Filter, ...]:
        return self._scope

    @property
    def columns(self) -> tuple[str, ...]:
        """The declared columns, in declaration order."""
        return tuple(self._columns)

    def check_dialect(self, dialect: str) -> None:
        if dialect not in NATIVE_DIALECTS:
            raise ValidationError(
                f"a native layout renders for {sorted(NATIVE_DIALECTS)}, not {dialect!r}: "
                f"how a column is compared as text, as a time and with a typed "
                f"placeholder differs by dialect",
                context={"dialect": dialect},
            )

    # -- what a field is ------------------------------------------------------

    def _column(self, field: str, *, table: str | None = None) -> _Column:
        if is_storage_key_field(field):
            return self._columns[self.id_column]
        column = self._columns.get(field)
        if column is None:
            where = f"table {table!r}" if table else "this native table"
            raise ValidationError(
                f"{where} has no declared column {field!r}; declared: {sorted(self._columns)}",
                context={"table": table, "column": field, "declared": sorted(self._columns)},
            )
        return column

    @staticmethod
    def _refuse_structured(column: _Column, op: Operator, table: str | None) -> None:
        raise ValidationError(
            f"column {column.name!r} is declared {column.field_type.value}, which no engine "
            f"compares as a filter does; only EXISTS and NOT_EXISTS apply to it, not "
            f"{op.value}",
            context={"table": table, "column": column.name, "operator": op.value},
        )

    @staticmethod
    def _relates(sql_type: SqlType, reading: str | None) -> bool:
        """Whether a value of the column can be compared with a bound read so."""
        if reading is None or reading == "never":
            return False
        if reading in TIME_READINGS:
            if sql_type.stores_text:
                return True  # a string naming a time, read as one
            if reading == NAIVE_TIME:
                return NAIVE_TIME in sql_type.kinds
            return ZONED_INSTANT in sql_type.kinds
        return reading in sql_type.kinds

    @staticmethod
    def _bounds(op: Operator, value: Any) -> list[Any]:
        if op in _MEMBERSHIP:
            return membership_values(value)
        if op in _RANGES:
            return list(value)
        return [value]

    def _admits(self, column: _Column, spec: Filter) -> bool:
        """Whether ``spec`` can match some value of ``column`` (structured: refused)."""
        if spec.operator in _PRESENCE:
            return True
        sql_type = column.sql_type
        if sql_type is None:
            self._refuse_structured(column, spec.operator, None)
            raise AssertionError  # unreachable
        if spec.operator in _STRING_ONLY:
            return sql_type.stores_text or sql_type.reads_as_text
        bounds = [sql_type.bind(b) for b in self._bounds(spec.operator, spec.value)]
        related = [self._relates(sql_type, value_kind(b)) for b in bounds]
        if spec.operator in _RANGES:
            return all(related)
        return any(related)

    def _check_scope_filter(self, spec: Filter) -> None:
        if not isinstance(spec, Filter):
            raise ValidationError(
                f"a scope is a list of filters, got {type(spec).__name__}",
                context={"got": type(spec).__name__},
            )
        column = self._column(spec.field)
        if not self._admits(column, spec):
            raise ValidationError(
                f"scope filter {spec.field} {spec.operator.value} {spec.value!r} can match no "
                f"row: column {column.name!r} cannot hold that value, so the store would "
                f"always be empty",
                context={"column": column.name, "operator": spec.operator.value},
            )

    # -- rendering ------------------------------------------------------------

    @staticmethod
    def _text(builder: SQLQueryBuilder, column: _Column) -> str:
        sql_type = column.sql_type
        if sql_type is not None and sql_type.stores_text:
            return column.quoted
        return f"CAST({column.quoted} AS {_TEXT_TYPE[builder.dialect]})"

    def _time_expr(
        self, builder: SQLQueryBuilder, column: _Column, reading: str
    ) -> tuple[str | None, str]:
        sql_type = column.sql_type
        assert sql_type is not None
        if reading == ZONED_WALL_CLOCK and not sql_type.stores_text:
            # A zoned column is read in UTC, so a date orders it by UTC's
            # clock: as the instant of its midnight in UTC (see ``_bind``).
            reading = ZONED_INSTANT
        if sql_type.stores_text or builder.dialect == "sqlite":
            # Text, or SQLite, which has no time type and stores the text: read
            # it as the time it names, as a JSON string is read.
            names_time, time_value = builder._time_reading(column.quoted, reading)
            return names_time, f"CASE WHEN {names_time} THEN {time_value} END"
        return None, column.quoted

    def filter_clause(
        self, builder: SQLQueryBuilder, spec: Filter, param_start: int
    ) -> tuple[str, list[Any]]:
        column = self._column(spec.field, table=builder.table_name)
        op = spec.operator
        if op == Operator.EXISTS:
            return f"{column.quoted} IS NOT NULL", []
        if op == Operator.NOT_EXISTS:
            return f"{column.quoted} IS NULL", []
        sql_type = column.sql_type
        if sql_type is None:
            self._refuse_structured(column, op, builder.table_name)
            raise AssertionError  # unreachable

        if op in _STRING_ONLY:
            if not (sql_type.stores_text or sql_type.reads_as_text):
                # ``Filter.matches`` answers False for a value that is not a
                # string, NOT_LIKE included.
                return "FALSE", []
            return builder._build_operator_clause(
                self._text(builder, column), op, spec.value, param_start
            )

        if op not in builder._TYPED_OPERATORS:
            raise ValueError(f"Unsupported operator: {op}")

        bounds = [sql_type.bind(b) for b in self._bounds(op, spec.value)]
        value: Any = bounds if op in _MEMBERSHIP | _RANGES else bounds[0]
        # A column that reads as text compares itself in its own type only for
        # equality with bounds it holds, which keeps an index on it.
        own_type = not sql_type.reads_as_text or (
            op in _EQUALITY
            and sql_type.holds is not None
            and all(sql_type.holds(b) for b in bounds if b is not None)
        )

        def expr_for(reading: str | None) -> tuple[str | None, str] | None:
            if not self._relates(sql_type, reading):
                return None
            if reading in TIME_READINGS:
                assert reading is not None
                return self._time_expr(builder, column, reading)
            if reading == "string" and not own_type:
                return None, self._text(builder, column)
            return None, column.quoted

        def bind(reading: str | None, bound: Any) -> Any:
            if reading == ZONED_WALL_CLOCK and not sql_type.stores_text:
                # The record holds the instant in UTC, so a date is its
                # midnight there.
                start = bound if isinstance(bound, datetime) else datetime.combine(bound, time.min)
                return builder._bind_bound(ZONED_INSTANT, start.replace(tzinfo=UTC))
            return builder._bind_bound(reading, bound)

        def cast_for(reading: str | None, bound_values: Sequence[Any]) -> str | None:
            if sql_type.placeholder is None or not own_type or reading in TIME_READINGS:
                return None
            return sql_type.placeholder(builder.dialect, bound_values)

        return builder._build_typed_clause(
            op,
            value,
            param_start,
            expr_for,
            f"{column.quoted} IS NOT NULL",
            bind=bind,
            cast_for=cast_for,
        )

    def sort_keys(self, builder: SQLQueryBuilder, field: str) -> list[str]:
        column = self._column(field, table=builder.table_name)
        sql_type = column.sql_type
        if sql_type is None:
            raise ValidationError(
                f"column {column.name!r} is declared {column.field_type.value}, which has no "
                f"order a sort can follow",
                context={"table": builder.table_name, "column": column.name},
            )
        if sql_type.stores_text or sql_type.reads_as_text:
            return [builder._code_point_order(self._text(builder, column))]
        return [column.quoted]

    def select_list(self, builder: SQLQueryBuilder) -> str:
        """The declared columns, each under its own name.

        DuckDB's driver needs ``pytz`` to return a zoned time, so a column
        holding zoned instants is selected there as its time in UTC, which
        the type's ``read`` takes as UTC.
        """
        selected = []
        for column in self._columns.values():
            sql_type = column.sql_type
            if (
                builder.dialect == "duckdb"
                and sql_type is not None
                and ZONED_INSTANT in sql_type.kinds
                and not sql_type.stores_text
            ):
                selected.append(f"timezone('UTC', {column.quoted}) AS {column.quoted}")
            else:
                selected.append(column.quoted)
        return ", ".join(selected)

    def key_clause(
        self, builder: SQLQueryBuilder, record_id: str, param_start: int
    ) -> tuple[str, list[Any]]:
        return self.filter_clause(
            builder, Filter(RESERVED_KEY_FIELD, Operator.EQ, record_id), param_start
        )

    def record_from_row(self, row: Mapping[str, Any]) -> Record:
        data: dict[str, Any] = {}
        for name, column in self._columns.items():
            raw = row[name]
            data[name] = (
                raw if raw is None or column.sql_type is None else column.sql_type.read(raw)
            )
        key = data[self.id_column]
        return Record(data, storage_id=None if key is None else str(key))


def _resolve_column(name: str, field_type: Any, metadata: Mapping[str, Any]) -> _Column:
    """A declared column, with the answer for what it holds."""
    member = FieldType.lookup(field_type)
    if member is None:
        raise ValidationError(
            f"column {name!r} is declared as {field_type!r}, which is not a field type; "
            f"one of {[t.value for t in FieldType]}",
            context={"column": name, "type": field_type},
        )
    declared = metadata.get(SQL_TYPE_KEY)
    if declared is None:
        return _Column(name, quote_ident(name), member, field_type_answer(member))
    if not isinstance(declared, str) or not declared:
        raise ValidationError(
            f"column {name!r} declares `{SQL_TYPE_KEY}: {declared!r}`; it is the name of a "
            f"registered SQL type",
            context={"column": name, SQL_TYPE_KEY: declared},
        )
    sql_type = sql_types.get_optional(declared)
    if sql_type is None:
        raise ValidationError(
            f"column {name!r} declares `{SQL_TYPE_KEY}: {declared}`, which is not registered; "
            f"registered: {sorted(sql_types.list_keys())}. Register it with "
            f"`sql_types.register({declared!r}, SqlType(...))`",
            context={
                "column": name,
                SQL_TYPE_KEY: declared,
                "registered": sorted(sql_types.list_keys()),
            },
        )
    return _Column(name, quote_ident(name), member, sql_type)


def read_layout_config(
    config: Mapping[str, Any],
    schema: DatabaseSchema | None,
    *,
    origin: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> ColumnLayout:
    """Read a backend's ``layout:``, ``id_column:`` and ``scope:`` into a layout.

    The one reader of the three keys, so every SQL backend takes them the same
    way.

    - ``layout:`` is ``jsonb`` (the default: the table this package creates)
      or ``native`` (a table with ordinary typed columns).
    - ``id_column:`` and ``scope:`` are refused unless ``layout: native``, and
      so is a column declaring ``metadata.sql_type``, which nothing under the
      JSON layout reads.
    - ``native`` takes the schema's declared columns, an ``id_column`` among
      them, and an optional ``scope:``: a list of filters, each a
      :class:`~dataknobs_data.query.Filter` or a ``{field, operator, value}``
      mapping.

    Args:
        config: The backend's configuration.
        schema: The backend's declared schema, if any.
        origin: Where the configuration came from, prefixed to every refusal.
        context: Carried into every refusal's ``context``.

    Returns:
        The layout the configuration describes.

    Raises:
        ValidationError: When a key is refused, as above, or the native layout
            refuses its schema or scope.
    """
    prefix = f"{origin}: " if origin else ""
    base: dict[str, Any] = dict(context or {})

    def refuse(message: str, **extra: Any) -> ValidationError:
        return ValidationError(f"{prefix}{message}", context={**base, **extra})

    layout = config.get("layout") or "jsonb"
    if layout not in LAYOUTS:
        raise refuse(f"`layout:` is one of {list(LAYOUTS)}, got {layout!r}", layout=layout)

    if layout == "jsonb":
        for key in ("id_column", "scope"):
            if config.get(key) is not None:
                raise refuse(
                    f"`{key}:` is read only with `layout: native`; the JSON layout's key is "
                    f"`id` and it has no scope",
                    key=key,
                )
        declared = [
            name
            for name, field in (schema.fields.items() if schema else ())
            if SQL_TYPE_KEY in field.metadata
        ]
        if declared:
            raise refuse(
                f"columns {declared} declare `{SQL_TYPE_KEY}`, which only `layout: native` reads",
                columns=declared,
            )
        return JsonbLayout()

    id_column = config.get("id_column")
    if not isinstance(id_column, str) or not id_column:
        raise refuse(
            "`layout: native` needs `id_column:`, the declared column that keys a row",
            id_column=id_column,
        )
    raw_scope = config.get("scope") or []
    if not isinstance(raw_scope, Sequence) or isinstance(raw_scope, (str, bytes)):
        raise refuse(
            f"`scope:` is a list of filters, got {type(raw_scope).__name__}",
            got=type(raw_scope).__name__,
        )
    scope: list[Filter] = []
    for entry in raw_scope:
        if isinstance(entry, Filter):
            scope.append(entry)
        elif isinstance(entry, Mapping):
            try:
                scope.append(Filter.from_dict(dict(entry)))
            except (KeyError, TypeError, ValueError) as e:
                raise refuse(
                    f"`scope:` entry {dict(entry)!r} is not a filter "
                    f"(`{{field, operator, value}}`): {e}",
                    entry=dict(entry),
                ) from e
        else:
            raise refuse(
                f"`scope:` entry {entry!r} is not a filter (`{{field, operator, value}}`)",
                entry=entry,
            )
    if schema is None:
        raise refuse("`layout: native` needs a `schema:` declaring the table's columns")
    try:
        return NativeColumnLayout(schema, id_column=id_column, scope=scope)
    except ValidationError as e:
        raise refuse(str(e), **dict(e.context or {})) from e

# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""What a column of a table the store does not own holds, and how it is compared.

A table with ordinary typed columns is read through
:class:`~dataknobs_data.backends.column_layout.NativeColumnLayout`, which
answers every filter as :meth:`~dataknobs_data.query.Filter.matches` answers
it over the record it returns. To do that it has to know, for each column,
which values the column can hold. A :class:`SqlType` says so.

Each :class:`~dataknobs_common.fields.FieldType` a schema can declare has a
built-in answer (:func:`field_type_answer`). A column whose SQL type
``FieldType`` does not name declares it through
``FieldSchema.metadata["sql_type"]`` (:data:`SQL_TYPE_KEY` in
:mod:`dataknobs_data.schema`), and the name is looked up in :data:`sql_types`.
Two ship: ``uuid`` and ``timestamptz``. A consumer registers its own::

    from dataknobs_data import SqlType, sql_types

    sql_types.register("citext", SqlType(kinds=frozenset({"string"}), text="stored"))

The kinds are :func:`~dataknobs_data.query.value_kind`'s names, ``"string"``,
``"number"`` and ``"boolean"``, plus the two ways a time is held,
:data:`NAIVE_TIME` and :data:`ZONED_INSTANT`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, Literal, get_args

from dataknobs_common.registry import Registry

from ..fields import FieldType
from ..query import read_timestamp

#: A time held with no zone, compared by its own clock. The name is
#: :func:`~dataknobs_data.query.value_kind`'s for a time, which a naive one keeps.
NAIVE_TIME: Final = "timestamp"

#: A time held with a zone, compared by the instant it names.
ZONED_INSTANT: Final = "zoned timestamp"

#: A zoned time compared by its own wall clock, which is how a plain ``date``
#: bound orders one. Never declared: a column holding :data:`ZONED_INSTANT`
#: answers it.
ZONED_WALL_CLOCK = "zoned wall-clock timestamp"

#: Every reading of a time.
TIME_READINGS = frozenset({NAIVE_TIME, ZONED_INSTANT, ZONED_WALL_CLOCK})

#: A kind of value a :class:`SqlType` may declare it holds.
SqlKind = Literal["string", "number", "boolean", "timestamp", "zoned timestamp"]

#: How a column relates to text: its value is text, or it is compared as its text.
SqlText = Literal["stored", "cast"]

#: The kinds a :class:`SqlType` may declare.
DECLARABLE_KINDS: frozenset[str] = frozenset(get_args(SqlKind))

_INT64 = range(-(2**63), 2**63)


def _same(value: Any) -> Any:
    return value


@dataclass(frozen=True)
class SqlType:
    """What a column of one SQL type holds, and how a filter is compared with it.

    Attributes:
        kinds: The kinds of value the column holds (see the module docstring).
            A filter whose bound is of another kind matches nothing, and its
            negation every present value, as ``Filter.matches`` answers.
        bind: A filter's bound as a record of this column would hold it, or
            ``None`` when no value of the column can equal it. Applied to each
            bound and membership member before it is compared, so the oracle
            and the statement compare like with like.
        read: A value the driver returned for the column, as the record holds
            it. Every engine then gives the record one Python type per column.
        text: How the column relates to text. ``"stored"``: its SQL value is
            text, so it is compared as it is with a string bound, read as a time
            by a time bound, and the string-only operators (``LIKE``,
            ``REGEX``, ``STARTS_WITH``) apply. ``"cast"``: it is not text, but
            the record holds its text, so a string bound and the string-only
            operators compare ``CAST(column AS TEXT)``. ``None``: neither.
        holds: For a column that reads as text, whether an equality bound (after
            ``bind``) can be compared with the column in its own type, which
            keeps an index on it. ``None``: always compared as text.
        placeholder: The SQL type a bound is sent as when it is compared in the
            column's own type, given the dialect and the bounds; ``None`` to
            send it untyped. A driver that infers the placeholder's type from
            the column can otherwise convert the bound, and answer wrongly.
    """

    kinds: frozenset[SqlKind]
    bind: Callable[[Any], Any] = _same
    read: Callable[[Any], Any] = _same
    text: SqlText | None = None
    holds: Callable[[Any], bool] | None = None
    placeholder: Callable[[str, Sequence[Any]], str | None] | None = None

    def __post_init__(self) -> None:
        unknown = set(self.kinds) - DECLARABLE_KINDS
        if not self.kinds or unknown:
            raise ValueError(
                f"An SqlType holds one or more of {sorted(DECLARABLE_KINDS)}; got "
                f"{sorted(self.kinds)!r}"
            )
        if self.text is not None and self.text not in get_args(SqlText):
            raise ValueError(
                f"An SqlType's text is one of {list(get_args(SqlText))} or None; got {self.text!r}"
            )

    @property
    def stores_text(self) -> bool:
        """Whether the column's SQL value is text."""
        return self.text == "stored"

    @property
    def reads_as_text(self) -> bool:
        """Whether the column is not text but the record holds its text."""
        return self.text == "cast"


#: The SQL types a schema can name through ``metadata["sql_type"]``.
sql_types: Registry[SqlType] = Registry("sql_types")


def _number_placeholder(dialect: str, bounds: Sequence[Any]) -> str | None:
    """``bigint`` for whole numbers, ``numeric`` for a ``Decimal`` or one past 64 bits.

    Only PostgreSQL needs it: asyncpg types a placeholder by the column it is
    compared with, so ``integer_column >= $1`` sends ``3.5`` as ``3``. The cast
    is on the placeholder, so an index on the column still serves it; a
    fractional bound, compared as ``double precision``, is the one that cannot.
    """
    if dialect != "postgres":
        return None
    if any(isinstance(b, Decimal) for b in bounds):
        return "numeric"
    if all(isinstance(b, Integral) for b in bounds):
        return "bigint" if all(int(b) in _INT64 for b in bounds) else "numeric"
    return "double precision"


def _canonical_uuid(value: Any) -> Any:
    """A UUID, or a string naming one, as its canonical lower-case text."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str):
        try:
            return str(uuid.UUID(value))
        except ValueError:
            return value
    return value


def _is_canonical_uuid(value: Any) -> bool:
    """Whether ``value`` is a uuid's canonical text, which a uuid column can be compared with."""
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _uuid_placeholder(dialect: str, bounds: Sequence[Any]) -> str | None:
    return {"postgres": "uuid", "duckdb": "UUID"}.get(dialect)


def _bind_time(value: Any) -> Any:
    """A string bound read as the time it names, or ``None`` when it names none."""
    if isinstance(value, str):
        return read_timestamp(value)
    return value


def _read_naive_time(value: Any) -> Any:
    """A time column's value as a ``datetime``: SQLite returns the text it stores."""
    if isinstance(value, str):
        read = read_timestamp(value)
        return value if read is None else read
    return value


def _read_zoned_time(value: Any) -> Any:
    """A zoned column's value as an aware ``datetime`` in UTC.

    A naive value is the column's time in UTC: a native layout selects it so
    where the driver cannot return an aware one (DuckDB, without ``pytz``).
    """
    value = _read_naive_time(value)
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return value


def _read_boolean(value: Any) -> Any:
    """SQLite returns a boolean column as ``0`` / ``1``."""
    return bool(value) if isinstance(value, int) and not isinstance(value, bool) else value


sql_types.register(
    "uuid",
    SqlType(
        kinds=frozenset({"string"}),
        bind=_canonical_uuid,
        read=_canonical_uuid,
        text="cast",
        holds=_is_canonical_uuid,
        placeholder=_uuid_placeholder,
    ),
)

sql_types.register(
    "timestamptz",
    SqlType(
        kinds=frozenset({ZONED_INSTANT}),
        bind=_bind_time,
        read=_read_zoned_time,
    ),
)

_TEXT = SqlType(kinds=frozenset({"string"}), text="stored")
_NUMBER = SqlType(kinds=frozenset({"number"}), placeholder=_number_placeholder)

#: The built-in answer for each ``FieldType`` a column can be compared as.
#: Held here, not in :data:`sql_types`, so a registration cannot change every
#: column of a built-in type.
_FIELD_TYPE_ANSWERS: Mapping[FieldType, SqlType] = MappingProxyType(
    {
        FieldType.STRING: _TEXT,
        FieldType.TEXT: _TEXT,
        FieldType.INTEGER: _NUMBER,
        FieldType.FLOAT: _NUMBER,
        FieldType.BOOLEAN: SqlType(kinds=frozenset({"boolean"}), read=_read_boolean),
        FieldType.DATETIME: SqlType(
            kinds=frozenset({NAIVE_TIME}), bind=_bind_time, read=_read_naive_time
        ),
    }
)


def field_type_answer(field_type: FieldType) -> SqlType | None:
    """The built-in :class:`SqlType` for ``field_type``, or ``None``.

    ``None`` for the structured types (``json``, ``binary`` and the vectors),
    which no engine compares one way, so a filter on one is refused by name
    rather than answered.
    """
    return _FIELD_TYPE_ANSWERS.get(field_type)

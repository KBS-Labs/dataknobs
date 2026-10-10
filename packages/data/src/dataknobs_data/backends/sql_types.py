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

import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, Literal, get_args

from dataknobs_common.exceptions import ValidationError
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

#: The reading of a bound no value relates to: ``None`` or NaN, which equal
#: nothing and order against nothing (:func:`~dataknobs_data.query.value_kind`'s
#: name for them). A layout's ``expr_for`` answers ``None`` for it.
NEVER: Final = "never"

#: A kind of value a :class:`SqlType` may declare it holds.
SqlKind = Literal["string", "number", "boolean", "timestamp", "zoned timestamp"]

#: How a column relates to text: its value is text, or it is compared as its text.
SqlText = Literal["stored", "cast"]

#: The kinds a :class:`SqlType` may declare.
DECLARABLE_KINDS: frozenset[str] = frozenset(get_args(SqlKind))

_INT64 = range(-(2**63), 2**63)

#: The integers each dialect's driver binds as one, where it has a bound: past
#: them a Python ``int`` raises, and so past them no integer column holds a value.
_BOUND_INTEGERS: Mapping[str, range] = MappingProxyType(
    {"sqlite": _INT64, "duckdb": range(-(2**127), 2**127)}
)


def _same(value: Any) -> Any:
    return value


def _same_sent(value: Any, dialect: str) -> Any:
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
        holds: Whether some value of the column can equal a bound (after
            ``bind``) of a kind it holds; ``None``: any can. A scope equal to a
            value no row holds is refused by it, and a column that reads as text
            is compared with such a bound as its text, but in its own type, which
            keeps an index on it, with one it holds.
        own: A bound compared with the column in its own type, as the value
            sent for it on a dialect: one every value of the column compares
            with as it compares with the bound. A uuid column takes its text as
            it is; an integer key's text ``"10"`` is sent as ``10``; an integer
            column's ``2.0`` is sent as ``2``, and ``2.25`` as ``2.5``, which no
            engine rounds onto an integer.
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
    own: Callable[[Any, str], Any] = _same_sent
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
    fractional bound, compared as ``numeric`` or ``double precision``, is the
    one that cannot.
    """
    if dialect != "postgres":
        return None
    if any(isinstance(b, Decimal) for b in bounds):
        return "numeric"
    if all(isinstance(b, Integral) for b in bounds):
        return "bigint" if all(int(b) in _INT64 for b in bounds) else "numeric"
    return "double precision"


#: The magnitude past which DuckDB has no decimal holding a number to a tenth.
_DUCKDB_DECIMAL_LIMIT = 10**37


def _integer_placeholder(dialect: str, bounds: Sequence[Any]) -> str | None:
    """As :func:`_number_placeholder`, refusing what DuckDB cannot compare exactly.

    DuckDB types a ``Decimal`` as the ``DECIMAL`` that holds it, and gives the
    bounds of one ``BETWEEN`` or ``IN`` a common type: a fractional bound
    beside an integer past 37 digits makes that a ``DECIMAL(38,1)`` the integer
    does not fit, and a fraction that wide is typed ``DOUBLE``. Either compares
    the column inexactly, so it is refused rather than answered.
    """
    if dialect != "duckdb":
        return _number_placeholder(dialect, bounds)
    numbers = [b for b in bounds if isinstance(b, (int, Decimal)) and not isinstance(b, bool)]
    if any(isinstance(b, Decimal) for b in numbers):
        for b in numbers:
            if not -_DUCKDB_DECIMAL_LIMIT < b < _DUCKDB_DECIMAL_LIMIT:
                raise ValidationError(
                    f"DuckDB holds a number to a tenth only in a DECIMAL of 38 digits, which "
                    f"{b!r} does not fit beside a fractional bound; compare it in a filter of "
                    f"its own",
                    context={"bound": str(b), "dialect": dialect},
                )
    return None


def _is_whole(value: Any) -> bool:
    """Whether an integer column can hold ``value``: a number with no fraction."""
    if isinstance(value, Integral):
        return True
    try:
        return bool(value == math.floor(value))
    except (TypeError, OverflowError, ValueError):
        return False


def _integer_comparand(value: Any, dialect: str) -> Any:
    """A number as an integer column compares with it, sent exactly on ``dialect``.

    - A whole ``float`` or ``Decimal`` is its ``int``, so ``2.0**60`` is not
      sent as a ``float`` an engine compares by rounding the column.
    - A whole number past the integers the driver binds is past every value
      the column holds, so it is the infinity on its side, which every engine
      compares with an integer exactly.
    - A fractional one is the ``Decimal`` halfway between the two integers it
      lies between, which every integer compares with as it compares with the
      bound, and which no engine rounds onto either. SQLite has no decimal, so
      there it is the ``float`` that holds it, and short of 2**52 one does;
      past that no SQLite number lies between two integers, and the bound is
      refused rather than rounded onto one.

    An infinity is left as it is, and so is a number of another type (a
    ``Fraction``), which the builder sends as it would.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        whole = int(value)
    elif isinstance(value, (float, Decimal)):
        try:
            whole = math.floor(value)
        except (OverflowError, ValueError):
            return value
    else:
        return value
    bound = _BOUND_INTEGERS.get(dialect)
    if bound is not None and whole not in bound:
        return math.copysign(math.inf, whole)
    if value == whole:
        return whole
    # Built from text: Decimal arithmetic rounds to 28 digits.
    halfway = Decimal(f"{10 * whole + 5}E-1")
    if dialect != "sqlite":
        return halfway
    if Decimal(float(halfway)) != halfway:
        raise ValidationError(
            f"SQLite has no number between {whole} and {whole + 1}, so an integer column "
            f"cannot be compared with {value!r} exactly",
            context={"bound": str(value), "dialect": dialect},
        )
    return float(halfway)


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
_INTEGER = SqlType(
    kinds=frozenset({"number"}),
    holds=_is_whole,
    own=_integer_comparand,
    placeholder=_integer_placeholder,
)

#: The built-in answer for each ``FieldType`` a column can be compared as.
#: Held here, not in :data:`sql_types`, so a registration cannot change every
#: column of a built-in type.
_FIELD_TYPE_ANSWERS: Mapping[FieldType, SqlType] = MappingProxyType(
    {
        FieldType.STRING: _TEXT,
        FieldType.TEXT: _TEXT,
        FieldType.INTEGER: _INTEGER,
        FieldType.FLOAT: _NUMBER,
        FieldType.BOOLEAN: SqlType(kinds=frozenset({"boolean"}), read=_read_boolean),
        FieldType.DATETIME: SqlType(
            kinds=frozenset({NAIVE_TIME}), bind=_bind_time, read=_read_naive_time
        ),
    }
)


def _is_integer_text(value: Any) -> bool:
    """Whether ``value`` is a 64-bit integer's text, as ``str`` writes it."""
    if not isinstance(value, str):
        return False
    try:
        number = int(value)
    except ValueError:
        return False
    return str(number) == value and number in _INT64


def _integer_of_text(value: str, dialect: str) -> int:
    """An integer's text, which :func:`_is_integer_text` held, as the integer."""
    return int(value)


#: An integer column keying a row: compared as its text, which is the
#: record's storage id, and in its own type for an integer's own text.
_INTEGER_KEY = SqlType(
    kinds=frozenset({"string"}),
    text="cast",
    holds=_is_integer_text,
    own=_integer_of_text,
    placeholder=_number_placeholder,
)


def key_answer(field_type: FieldType, sql_type: SqlType | None) -> SqlType | None:
    """What a column keying a row is compared as, or ``None`` when it cannot key one.

    A record's storage id is its key column's value as text, and the reserved
    key field is compared with that text, as ``Filter.matches`` compares it
    with the storage id. So a key column must have one text on every engine,
    and the one Python writes:

    - A column holding strings that is text or is read as its text (``string``,
      ``text``, ``uuid``) is compared as it already is.
    - An ``integer`` column is compared as its text, and in its own type for an
      equality with an integer's own text, so an index on it still serves a
      read by key.
    - Anything else is ``None``: each engine writes a float, a boolean or a
      time as different text, and a structured value has no text to key by.

    Args:
        field_type: The column's declared field type.
        sql_type: What the column holds, as the layout resolved it.

    Returns:
        The :class:`SqlType` the key is compared as, or ``None``.
    """
    if sql_type is None:
        return None
    if "string" in sql_type.kinds and (sql_type.stores_text or sql_type.reads_as_text):
        return sql_type
    if field_type is FieldType.INTEGER and sql_type is _FIELD_TYPE_ANSWERS[FieldType.INTEGER]:
        return _INTEGER_KEY
    return None


def field_type_answer(field_type: FieldType) -> SqlType | None:
    """The built-in :class:`SqlType` for ``field_type``, or ``None``.

    ``None`` for the structured types (``json``, ``binary`` and the vectors),
    which no engine compares one way, so a filter on one is refused by name
    rather than answered.
    """
    return _FIELD_TYPE_ANSWERS.get(field_type)

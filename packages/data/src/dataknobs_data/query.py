# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Query construction and filtering for database operations.

This module provides classes for building queries with filters, operators,
sorting, pagination, and vector similarity search for database operations.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Collection, Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import Enum
from functools import cached_property, cmp_to_key, lru_cache
from numbers import Number
from typing import TYPE_CHECKING, Any, SupportsFloat

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from .query_logic import ComplexQuery
    from .vector.types import DistanceMetric


class Operator(Enum):
    """Query operators for filtering.

    Operators used to build filter conditions in queries. Supports comparison,
    pattern matching, existence checks, and range queries.

    Example:
        ```python
        from dataknobs_data import Filter, Operator, Query

        # Equality
        filter_eq = Filter("age", Operator.EQ, 30)

        # Comparison
        filter_gt = Filter("score", Operator.GT, 90)

        # Pattern matching (SQL LIKE)
        filter_like = Filter("name", Operator.LIKE, "A%")  # Names starting with 'A'

        # IN operator
        filter_in = Filter("status", Operator.IN, ["active", "pending"])

        # Range query
        filter_between = Filter("age", Operator.BETWEEN, [20, 40])

        # Build query
        query = Query(filters=[filter_gt, filter_like])
        ```
    """

    EQ = "="  # Equal
    NEQ = "!="  # Not equal
    GT = ">"  # Greater than
    GTE = ">="  # Greater than or equal
    LT = "<"  # Less than
    LTE = "<="  # Less than or equal
    IN = "in"  # In list
    NOT_IN = "not_in"  # Not in list
    LIKE = "like"  # String pattern matching (SQL LIKE)
    NOT_LIKE = "not_like"  # String pattern not matching (SQL NOT LIKE)
    REGEX = "regex"  # Regular expression matching
    STARTS_WITH = "starts_with"  # Literal, case-sensitive prefix match (escape-safe)
    EXISTS = "exists"  # Field exists
    NOT_EXISTS = "not_exists"  # Field does not exist
    BETWEEN = "between"  # Value between two bounds (inclusive)
    NOT_BETWEEN = "not_between"  # Value not between two bounds


class SortOrder(Enum):
    """Sort order for query results."""

    ASC = "asc"
    DESC = "desc"


# The one spelling neither the enum values nor the normalization below can
# reach. Every other alias the fluent path ever accepted --- ``IN``, ``LIKE``,
# ``NOT IN``, ``STARTS_WITH``, ``BETWEEN`` --- is an enum value in a different
# case or with a space for the underscore, so it is derived rather than listed.
# That is the whole point: the private table this replaces was a hand-written
# second copy of the vocabulary, and it had already drifted --- ``not_like``
# was missing from it, so ``filter("name", "not_like", "A%")`` meant
# ``name = "A%"``.
_OPERATOR_ALIASES: dict[str, Operator] = {"==": Operator.EQ}


def coerce_operator(operator: str | Operator) -> Operator:
    """The one reading of a filter operator, whatever spelling it arrives in.

    Accepts an :class:`Operator`, any operator's own value (``"!="``,
    ``"not_like"``), and case/space variants of those (``"IN"``,
    ``"NOT BETWEEN"``), plus ``"=="`` for equality.

    Args:
        operator: The operator, as an enum member or a string spelling.

    Returns:
        The :class:`Operator` the spelling names.

    Raises:
        ValueError: If the spelling names no operator. It used to name
            :attr:`Operator.EQ` --- silently, so a typo inverted a query
            rather than failing it, and no return value, log line or
            exception let the caller find out.
    """
    if isinstance(operator, Operator):
        return operator
    if isinstance(operator, str):
        if operator in _OPERATOR_ALIASES:
            return _OPERATOR_ALIASES[operator]
        for candidate in (operator, operator.strip().lower().replace(" ", "_")):
            try:
                return Operator(candidate)
            except ValueError:
                continue
    known = ", ".join(sorted({member.value for member in Operator} | set(_OPERATOR_ALIASES)))
    raise ValueError(f"Unknown query operator {operator!r}; expected one of: {known}")


def coerce_sort_order(order: str | SortOrder) -> SortOrder:
    """The one reading of a sort order, whatever spelling it arrives in.

    Args:
        order: The order, as an enum member or a string spelling
            (case-insensitive).

    Returns:
        The :class:`SortOrder` the spelling names.

    Raises:
        ValueError: If the spelling names no order. It used to name
            :attr:`SortOrder.DESC` --- every string but ``"asc"`` did, so
            ``sort_by("score", "ascending")`` sorted descending.
    """
    if isinstance(order, SortOrder):
        return order
    if isinstance(order, str):
        try:
            return SortOrder(order.strip().lower())
        except ValueError:
            pass
    known = ", ".join(member.value for member in SortOrder)
    raise ValueError(f"Unknown sort order {order!r}; expected one of: {known}")


RESERVED_KEY_FIELD = "id"
"""The single query/sort field name routed to a record's storage key.

Both ``Filter(RESERVED_KEY_FIELD, ...)`` and ``SortSpec(RESERVED_KEY_FIELD, ...)``
target the record's storage key on every backend — never a value stored under the
same name in the record's ``data``. This is the only reserved *bare* field name:
``metadata.<x>`` is a separate SQL-only prefix that routes to the metadata column,
and any ``<entity>_id`` (e.g. ``node_id``, ``sku``) is an ordinary data field. To
query a secondary identifier held in ``data``, name that field something other
than ``id`` so it is not shadowed by the storage-key routing.
"""


def is_storage_key_field(field_name: str) -> bool:
    """Return True if ``field_name`` addresses the record's storage key.

    The single source of truth every backend's filter/sort translation consults
    instead of comparing ``field == "id"`` inline, so all backends agree on the
    reserved name by construction.
    """
    return field_name == RESERVED_KEY_FIELD


def _hashable(value: Any) -> Any:
    """Project a filter value onto a shape that hashes.

    ``Filter.value`` is annotated ``Any`` because it holds whatever the field
    holds, and four operators hold a *list*: ``IN`` and ``NOT_IN`` take the
    candidate set, ``BETWEEN`` and ``NOT_BETWEEN`` take the two bounds. A hash
    over the field tuple alone would therefore refuse exactly the operators
    whose value is most often written as a literal.

    The projection is for the hash and for nothing else --- :attr:`Filter.value`
    keeps the type it was handed (a list or set as a copy of it), so ``to_dict()`` answers in the shape
    ``from_dict()`` takes and every backend reads the list it was given.
    Normalising at construction instead would be cheaper here and wrong
    everywhere else: ``Filter("tags", Operator.EQ, ["a"])`` compares its value
    against what a JSON column hands back, and a tuple does not equal a list.

    Equal values project onto equal shapes, which is the half of the hash
    contract a hand-written ``__hash__`` owes. The converse is not promised and
    need not be: ``["a"]`` and ``("a",)`` are unequal filters that land on one
    hash, which is an ordinary collision.
    """
    if isinstance(value, Mapping):
        return frozenset((key, _hashable(item)) for key, item in value.items())
    if isinstance(value, AbstractSet):
        return frozenset(_hashable(item) for item in value)
    if isinstance(value, bytearray):
        # Unhashable, and equal to the ``bytes`` it holds, so it projects there.
        return bytes(value)
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Collection):
        # Every other collection a membership value may be --- a list, a tuple,
        # ``dict.values()`` --- is compared in iteration order, so it projects
        # onto a tuple in that order.
        return tuple(_hashable(item) for item in value)
    return value


#: The two operators whose value is a collection of candidates rather than one.
_MEMBERSHIP_OPERATORS = frozenset({Operator.IN, Operator.NOT_IN})

#: The two operators whose value is a range: a lower and an upper bound.
_RANGE_OPERATORS = frozenset({Operator.BETWEEN, Operator.NOT_BETWEEN})

#: How much of a refused value's ``repr`` a refusal quotes.
_REFUSED_VALUE_REPR_LIMIT = 40


def membership_values(value: Collection[Any]) -> list[Any]:
    """The members of an ``IN`` / ``NOT IN`` value that can match a record.

    :meth:`Filter.matches` never matches a ``None`` record value, so a ``None``
    member can never select anything. SQL disagrees in the way that matters:
    ``x NOT IN (NULL, ...)`` is ``NULL`` for every ``x``, so one ``None``
    member empties a whole ``NOT IN``. A backend that renders the list renders
    these members instead, and handles the list they leave empty.

    The filter keeps the value it was given; this is how it is *evaluated*,
    not a rewrite of it.
    """
    return [member for member in value if member is not None]


#: The date of an ISO timestamp in extended form, each field in its range and
#: the year from 0001, as :class:`datetime` counts. ``[0-9]`` rather than
#: ``\d``, which some regex engines read as any Unicode digit.
_ISO_DATE = (
    r"([1-9][0-9]{3}|0[1-9][0-9]{2}|00[1-9][0-9]|000[1-9])"
    r"-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
)
#: The optional time that follows it: hours and minutes, then optional seconds
#: and a fraction of any length, after a ``T`` or a space.
_ISO_TIME = r"[T ]([01][0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9](\.[0-9]+)?)?"

#: The zone that may follow the time: ``Z``, or an offset in hours and minutes.
_ISO_ZONE = r"(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])"

#: The shape of a timestamp string with no zone: :func:`read_timestamp`'s
#: shape without its zone. A backend that renders a filter as a query tests a
#: stored string against this before reading it as a time, so it and
#: ``Filter.matches`` read the same strings as times.
NAIVE_TIMESTAMP_SHAPE = f"^{_ISO_DATE}({_ISO_TIME})?$"

#: The shape of a timestamp string with a zone, which follows a time: the
#: strings :func:`read_timestamp` reads as an aware ``datetime``. A backend
#: tests a stored string against this before reading it as an instant.
ZONED_TIMESTAMP_SHAPE = f"^{_ISO_DATE}{_ISO_TIME}{_ISO_ZONE}$"

_TIMESTAMP = re.compile(f"^{_ISO_DATE}({_ISO_TIME}{_ISO_ZONE}?)?$")


def read_timestamp(text: str) -> datetime | None:
    """The time a string names, or ``None`` when it names none.

    A timestamp is ISO 8601 in extended form: a date (``2024-01-02``),
    optionally a time after ``T`` or a space (``10:30``, ``10:30:05``,
    ``10:30:05.25``), and optionally a zone (``Z``, ``+01:00``), each field in
    its range and naming a real day. That is the form ``datetime.isoformat``
    writes and every backend stores. The other forms :meth:`datetime.fromisoformat`
    also reads --- ``20240102``, ``2024-W01-2``, an hour alone --- are strings:
    no backend's query language reads them as times, so a filter does not.

    This is the one reading ``Filter.matches`` uses wherever a string meets a
    date or datetime, and the one the SQL backends render
    (:data:`NAIVE_TIMESTAMP_SHAPE`, :data:`ZONED_TIMESTAMP_SHAPE`). A string
    with a zone is an aware ``datetime`` and one without is naive, and Python
    orders neither against the other, so an aware bound relates only to zoned
    strings and a naive bound only to naive ones; a ``date`` bound relates to
    both, by each value's own wall-clock day (:func:`_align_temporal`). Two
    strings never meet a time: they compare as text.
    """
    if not _TIMESTAMP.match(text):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        # The right shape, and no real day: 2024-02-30.
        return None


def _python_scalar(value: Any) -> Any:
    """A numpy scalar as the Python value it holds; any other value as given.

    numpy's scalars compare with Python's own, but not with every numeric type
    --- ``np.int64(5) == Decimal(5)`` raises --- so a filter compares the
    Python value a numpy scalar stands for.
    """
    if type(value).__module__ == "numpy" and getattr(value, "ndim", None) == 0:
        return value.item()
    return value


def _is_boolean(value: Any) -> bool:
    """A ``bool``, or numpy's boolean scalar, which does not subclass it."""
    kind = type(value)
    return isinstance(value, bool) or (
        kind.__module__ == "numpy" and kind.__name__ in ("bool", "bool_")
    )


def value_kind(value: Any) -> str | None:
    """The kind a filter relates ``value`` by, as JSON types a stored value.

    - ``"boolean"``: a ``bool`` (or numpy boolean). Never a number, although
      Python makes ``True == 1``.
    - ``"number"``: any other real number --- ``int``, ``float``,
      ``Decimal``, a numpy number.
    - ``"timestamp"``: a ``date`` or ``datetime``. A plain ``date`` is its own
      midnight (see :func:`_align_temporal`).
    - ``"string"``: a ``str``, which a timestamp bound reads through
      :func:`read_timestamp`.
    - ``"never"``: ``None`` or NaN, which equal nothing and order against
      nothing, so a comparison with one matches nothing and its negation every
      present value.
    - ``None``: anything else (a list, a mapping), compared as it is.

    ``Filter.matches`` and every backend that renders a comparison ask this,
    so the two cannot disagree about which values a bound can relate to.
    """
    if value is None:
        return "never"
    if _is_boolean(value):
        return "boolean"
    if isinstance(value, Number) and not isinstance(value, complex):
        try:
            if isinstance(value, SupportsFloat) and math.isnan(value):
                return "never"
        except OverflowError:
            pass  # an integer too large for a float, and so no NaN
        except ValueError:
            # A signalling Decimal NaN refuses even to become a float.
            return "never"
        return "number"
    if isinstance(value, date):
        return "timestamp"
    if isinstance(value, str):
        return "string"
    return None


def _bool_against_number(a: Any, b: Any) -> bool:
    """Whether one of two values is a boolean and the other a number.

    Python makes ``bool`` a subclass of ``int``, so ``True == 1`` and
    ``False < 5``. JSON keeps the two apart, as PostgreSQL, DuckDB and
    Elasticsearch store them, so a filter never relates a boolean to a number:
    a comparison between them is false, and its negation true.
    """
    return {value_kind(a), value_kind(b)} == {"boolean", "number"}


def _string_against_temporal(a: Any, b: Any) -> bool:
    """Whether one of two values is a string and the other a date or datetime."""
    return (isinstance(a, str) and isinstance(b, date)) or (
        isinstance(b, str) and isinstance(a, date)
    )


def _read_temporal_string(a: Any, b: Any) -> tuple[Any, Any] | None:
    """Read the string of a string/temporal pair as the time it names.

    A temporal value stored as JSON comes back a string, so the string is read
    as the datetime it names (:func:`read_timestamp`). ``None`` when it names
    none, since then the two are unrelated. Call only for a pair
    :func:`_string_against_temporal` holds.
    """
    if isinstance(a, str):
        read = read_timestamp(a)
        return None if read is None else (read, b)
    read = read_timestamp(b)
    return None if read is None else (a, read)


def _values_equal(a: Any, b: Any) -> bool:
    """``a == b`` as a filter asks it, agreeing with its ordering.

    A boolean never equals a number (see :func:`_bool_against_number`), and
    ``None`` or NaN equals nothing. A string against a date or datetime is read
    as the datetime it names, and a date against a datetime is aligned, as the
    ordering operators do --- so a value that is ``>=`` and ``<=`` a bound also
    equals it.

    ``bool(...)``, because ``==`` between two values of unknown type is not
    required to answer with one --- a numpy array answers with an array. A
    caller that does ``if f.matches(v)`` was going to raise on the array
    anyway; this raises at the comparison instead of one frame later.
    """
    a, b = _python_scalar(a), _python_scalar(b)
    if _bool_against_number(a, b) or "never" in (value_kind(a), value_kind(b)):
        return False
    if _string_against_temporal(a, b):
        pair = _read_temporal_string(a, b)
        if pair is None:
            return False
        a, b = pair
    a, b = _align_temporal(a, b)
    return bool(a == b)


class _Times:
    """Dates and datetimes, indexed so a date or datetime probes them by hash.

    :func:`_align_temporal` equates a plain date with a datetime at that
    date's midnight in the datetime's own zone. So a datetime equals a member
    date when it falls at midnight on it, and a date equals a member datetime
    that does: the midnights are indexed by their date.
    """

    def __init__(self) -> None:
        self.datetimes: set[datetime] = set()
        self.dates: set[date] = set()
        self.midnights: set[date] = set()

    def add(self, value: date) -> None:
        if isinstance(value, datetime):
            self.datetimes.add(value)
            if value.time() == time.min:
                self.midnights.add(value.date())
        else:
            self.dates.add(value)

    def __contains__(self, value: date) -> bool:
        if isinstance(value, datetime):
            return value in self.datetimes or (
                value.time() == time.min and value.date() in self.dates
            )
        return value in self.dates or value in self.midnights


class _Membership:
    """The members of an ``IN`` / ``NOT IN`` value, indexed for equality.

    Membership is :func:`_values_equal` against any member, and ``in`` answers
    that wrongly across kinds: it finds ``True`` in ``[1]``, and never finds
    ``'2024-01-01'`` in ``[date(2024, 1, 1)]``. So the members are split by
    :func:`value_kind` once, when a filter is first matched, and every value
    is looked up by hash among the members of its own kind. A string member
    that names a time is read once, here, and found by a date or datetime
    value; a string value that names one finds a date or datetime member.

    A member that does not hash --- a list, for a field holding lists --- is
    compared with ``==``, as ``in`` over a list compares it.
    """

    def __init__(self, members: Collection[Any]) -> None:
        self.booleans: set[bool] = set()
        self.numbers: set[Any] = set()
        self.times = _Times()
        self.named_times = _Times()
        self.hashed: set[Any] = set()
        self.unhashed: list[Any] = []
        for member in members:
            kind = value_kind(member)
            if kind == "boolean":
                self.booleans.add(bool(member))
            elif kind == "number":
                self.numbers.add(_python_scalar(member))
            elif kind == "timestamp":
                self.times.add(member)
            elif kind != "never":
                if kind == "string" and (named := read_timestamp(member)) is not None:
                    self.named_times.add(named)
                try:
                    self.hashed.add(member)
                except TypeError:
                    self.unhashed.append(member)

    def __contains__(self, value: Any) -> bool:
        kind = value_kind(value)
        if kind == "boolean":
            return bool(value) in self.booleans
        if kind == "number":
            return _python_scalar(value) in self.numbers
        if kind == "timestamp":
            return value in self.times or value in self.named_times
        if kind == "never":
            return False
        try:
            if value in self.hashed:
                return True
        except TypeError:
            # A value that does not hash equals no member that does.
            pass
        if kind == "string":
            named = read_timestamp(value)
            return named is not None and named in self.times
        return any(value == member for member in self.unhashed)


@lru_cache(maxsize=256)
def _like_regex(pattern: str) -> re.Pattern[str]:
    r"""Compile a SQL ``LIKE`` pattern for a whole-value, case-insensitive match.

    Only ``%`` (any run, newlines included) and ``_`` (any one character) are
    wildcards; every other character, ``\`` among them, matches verbatim, as
    in sqlite and DuckDB. Postgres reads ``\`` as an escape by default.
    """
    body = "".join(
        ".*" if char == "%" else "." if char == "_" else re.escape(char) for char in pattern
    )
    return re.compile(body, re.IGNORECASE | re.DOTALL)


def _align_temporal(a: Any, b: Any) -> tuple[Any, Any]:
    """Promote a plain ``date`` ordered against a ``datetime`` to midnight.

    Python refuses to order a ``datetime`` against a ``date`` --- ``TypeError``,
    although the one subclasses the other --- where PostgreSQL and DuckDB
    promote the ``date`` to that day's midnight over a native ``timestamp``
    column. This takes their reading, so the in-memory answer agrees with
    theirs. A ``date`` carries no
    zone, so its midnight is taken in the ``datetime``'s own zone, which also
    keeps an aware ``datetime`` comparable.

    Any other pair is returned as given.
    """
    if isinstance(a, datetime) and _is_plain_date(b):
        return a, datetime.combine(b, time.min, tzinfo=a.tzinfo)
    if isinstance(b, datetime) and _is_plain_date(a):
        return datetime.combine(a, time.min, tzinfo=b.tzinfo), b
    return a, b


def _is_plain_date(value: Any) -> bool:
    """A ``date`` that is not a ``datetime``, which subclasses it."""
    return isinstance(value, date) and not isinstance(value, datetime)


def _order(a: Any, b: Any) -> int:
    """Three-way comparison under the same alignment ``Filter.matches`` uses."""
    a, b = _align_temporal(a, b)
    return int(a > b) - int(a < b)


_aligned_key = cmp_to_key(_order)


def _raw_key(value: Any) -> Any:
    return value


def sort_key_for(values: Iterable[Any]) -> Callable[[Any], Any]:
    """The key an in-memory sort orders these values by.

    A field mixing a plain ``date`` with a ``datetime`` is ordered under the
    alignment ``Filter.matches`` uses, so a sort agrees with the ordering
    operators. That key is a Python callback per comparison, so it is taken
    only when the field needs it; any other field sorts by its raw values,
    as it always has. Values no ordering relates still raise ``TypeError``,
    as ``sorted`` does.

    Args:
        values: Every value the sort will order.

    Returns:
        A key function for ``sorted`` / ``list.sort``.
    """
    has_date = has_datetime = False
    for value in values:
        if isinstance(value, datetime):
            has_datetime = True
        elif isinstance(value, date):
            has_date = True
        if has_date and has_datetime:
            return _aligned_key
    return _raw_key


@dataclass(frozen=True)
class Filter:
    """Represents a filter condition.

    A Filter combines a field name, an operator, and a value to create a query condition.
    Multiple filters can be combined in a Query for complex filtering.

    **A value, and frozen so that it hashes.** A condition is described rather
    than built up: nothing in this repository has ever assigned to one of these
    fields after construction, and a filter that could be edited under a caller
    holding it is the reason a mutable type is conventionally refused a hash.
    Frozen, it can sit in a set, key a cache, or be a field of another frozen
    value --- which is what :class:`~dataknobs_data.ontology.ColumnHierarchy`
    does with the narrowing its binding hands it.

    Hashability is the whole of that promise rather than most of it. A frozen
    dataclass satisfies :class:`~collections.abc.Hashable` whatever its fields
    hold, so one whose field tuple sometimes refuses the call answers the check
    a caller is supposed to ask with and then raises anyway. The list-valued
    operators are not a corner here --- ``IN`` is the common case --- so
    :meth:`__hash__` is written out over a projected value instead.

    **Its own copy of a list or set.** A list or set value is copied when the
    filter is built, so appending to the caller's list afterwards changes
    neither the filter's hash, nor what it matches in memory, nor the query a
    backend renders from it. A tuple or frozenset needs no copy. Any other
    collection --- ``dict.keys()`` among them --- is held as handed, and must
    not change while the filter is in use.

    **Membership.** ``IN`` and ``NOT_IN`` take a collection --- a list, tuple,
    set, ``dict.keys()`` or any other :class:`~collections.abc.Collection` that
    is neither a string nor a mapping --- and anything else is refused here,
    when the filter is built. A string used to mean substring match in memory
    and single-character match in SQL; a mapping is its keys in memory and in
    SQL, and a terms *lookup* in Elasticsearch. The SQL backends and the
    backends that filter in memory then answer as :meth:`matches` does: nothing
    is in an empty list, a ``None`` member matches nothing, and ``NOT_IN``
    selects only records whose field has a value, as ``NEQ`` does.
    Elasticsearch does not yet: its ``NOT_IN`` also selects a document that
    lacks the field, and it chooses a string field's exact-match path from
    the first member only.

    **Range.** ``BETWEEN`` and ``NOT_BETWEEN`` take exactly two bounds, as a
    list or tuple, and anything else is refused here too.

    Attributes:
        field: The field name to filter on
        operator: The comparison operator
        value: The value to compare against (optional for EXISTS/NOT_EXISTS operators)

    Example:
        ```python
        from dataknobs_data import Filter, Operator, Query, database_factory

        # Create filters
        age_filter = Filter("age", Operator.GT, 25)
        name_filter = Filter("name", Operator.LIKE, "A%")
        status_filter = Filter("status", Operator.IN, ["active", "pending"])

        # Use in query
        query = Query(filters=[age_filter, name_filter])

        # Search database
        db = database_factory("memory")
        results = db.search(query)
        ```
    """

    field: str
    operator: Operator
    value: Any = None

    def __post_init__(self) -> None:
        """Refuse a membership value or a range of the wrong shape, and own the value.

        A list or set value is replaced by a copy (see the class docstring).

        Raises:
            ValueError: If the operator is ``IN`` or ``NOT_IN`` and the value
                is not a collection, or is a string or a mapping; or if it is
                ``BETWEEN`` or ``NOT_BETWEEN`` and the value is not a list or
                tuple of exactly two bounds (a set has no order to say which
                bound is which). ``ValueError`` because it is what this module
                raises for a bad operator or sort order, so an
                ``except ValueError`` around query building catches it.
        """
        if self.operator in _MEMBERSHIP_OPERATORS:
            if isinstance(self.value, Mapping):
                remedy = "Pass the mapping's keys, as a list or as .keys()."
            elif isinstance(self.value, (str, bytes, bytearray)) or not isinstance(
                self.value, Collection
            ):
                remedy = "Wrap a single value in a list, or use EQ."
            else:
                self._own_value()
                return
            needs = "needs a list of values"
        elif self.operator in _RANGE_OPERATORS:
            if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
                self._own_value()
                return
            needs = "needs two bounds, as a list or tuple"
            remedy = "Pass [lower, upper]."
        else:
            self._own_value()
            return
        shown = repr(self.value)
        if len(shown) > _REFUSED_VALUE_REPR_LIMIT:
            shown = shown[: _REFUSED_VALUE_REPR_LIMIT - 3] + "..."
        raise ValueError(
            f"Filter({self.field!r}, {self.operator.name}) {needs}; "
            f"got {type(self.value).__name__} {shown}. {remedy}"
        )

    def _own_value(self) -> None:
        """Hold a copy of a list or set value, so the caller's cannot change it."""
        if isinstance(self.value, (list, set)):
            object.__setattr__(self, "value", copy.copy(self.value))

    def __hash__(self) -> int:
        """Hash the condition, projecting a container value onto a hashable shape.

        Written out rather than generated, because the generated one would
        raise for every list-valued operator --- see :func:`_hashable`, which
        is where that projection and its cost are argued.

        Equality stays the generated one, over the value as held. So two
        filters that compare equal hash equal, and the pair that does not ---
        a list value against the equivalent tuple --- is unequal and collides,
        which is the direction the contract allows.
        """
        return hash((self.field, self.operator, _hashable(self.value)))

    @cached_property
    def _membership(self) -> _Membership:
        """The ``IN`` / ``NOT IN`` members, indexed once for every match.

        A frozen filter's value is not reassigned, and a list or set value is
        the filter's own copy, so the index stays true.
        """
        return _Membership(self.value)

    def matches(self, record_value: Any) -> bool:
        """Check if a record value matches this filter.

        Supports type-aware comparisons for ranges and special handling
        for datetime/date objects.
        """
        if self.operator == Operator.EXISTS:
            return record_value is not None
        elif self.operator == Operator.NOT_EXISTS:
            return record_value is None
        elif record_value is None:
            return False

        if self.operator == Operator.EQ:
            return _values_equal(record_value, self.value)
        elif self.operator == Operator.NEQ:
            return not _values_equal(record_value, self.value)
        elif self.operator == Operator.GT:
            return self._compare_values(record_value, self.value, lambda a, b: a > b)
        elif self.operator == Operator.GTE:
            return self._compare_values(record_value, self.value, lambda a, b: a >= b)
        elif self.operator == Operator.LT:
            return self._compare_values(record_value, self.value, lambda a, b: a < b)
        elif self.operator == Operator.LTE:
            return self._compare_values(record_value, self.value, lambda a, b: a <= b)
        elif self.operator == Operator.IN:
            return record_value in self._membership
        elif self.operator == Operator.NOT_IN:
            return record_value not in self._membership
        elif self.operator == Operator.BETWEEN:
            lower, upper = self.value
            return self._compare_values(
                record_value, lower, lambda a, b: a >= b
            ) and self._compare_values(record_value, upper, lambda a, b: a <= b)
        elif self.operator == Operator.NOT_BETWEEN:
            lower, upper = self.value
            return not (
                self._compare_values(record_value, lower, lambda a, b: a >= b)
                and self._compare_values(record_value, upper, lambda a, b: a <= b)
            )
        elif self.operator in (Operator.LIKE, Operator.NOT_LIKE):
            if not isinstance(self.value, str):
                raise ValueError(f"LIKE/NOT_LIKE pattern must be a string, got: {self.value!r}")
            if not isinstance(record_value, str):
                return False
            matched = _like_regex(self.value).fullmatch(record_value) is not None
            return matched if self.operator == Operator.LIKE else not matched
        elif self.operator == Operator.REGEX:
            if not isinstance(record_value, str):
                return False
            return bool(re.search(self.value, record_value))
        elif self.operator == Operator.STARTS_WITH:
            # Literal, case-sensitive prefix match. Unlike LIKE, the prefix is
            # matched verbatim (a ``_`` or ``%`` in it is not a wildcard).
            # String-only, like LIKE/REGEX: a non-string value never matches.
            # The SQL backends enforce the same contract with a JSON-string-type
            # guard (see SQLQueryBuilder._json_string_guard), so every backend
            # agrees.
            return isinstance(record_value, str) and record_value.startswith(self.value)
        else:
            # This should never be reached as all operators are handled above
            raise ValueError(f"Unknown operator: {self.operator}")

    def _compare_values(self, a: Any, b: Any, comparator: Callable[[Any, Any], bool]) -> bool:
        """Compare two values with type awareness.

        Handles special cases:
        - A string against a date or datetime is read as the time it names
          (:func:`read_timestamp`); two strings compare as text, by code
          point, whatever either names --- the bound's kind decides, so a
          caller wanting time order passes a date or datetime
        - Mixed numeric types are converted appropriately
        - String comparisons are case-sensitive
        - A boolean never orders against a number (see
          :func:`_bool_against_number`)
        """
        a, b = _python_scalar(a), _python_scalar(b)
        if _bool_against_number(a, b):
            return False

        # Handle datetime/date comparisons
        if _string_against_temporal(a, b):
            pair = _read_temporal_string(a, b)
            if pair is None:
                return False
            a, b = pair

        a, b = _align_temporal(a, b)

        # Handle numeric comparisons
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return comparator(a, b)

        # Try direct comparison
        try:
            return comparator(a, b)
        except TypeError:
            # Types not comparable
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert filter to dictionary representation.

        A membership value that is not already a list or tuple --- a set,
        ``dict.keys()`` --- is written as a list, so the representation is
        JSON and :meth:`from_dict` reads back the same members.
        """
        value = self.value
        if self.operator in _MEMBERSHIP_OPERATORS and not isinstance(value, (list, tuple)):
            value = list(value)
        return {"field": self.field, "operator": self.operator.value, "value": value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Filter:
        """Create filter from dictionary representation."""
        return cls(
            field=data["field"],
            operator=coerce_operator(data["operator"]),
            value=data.get("value"),
        )


@dataclass(frozen=True)
class SortSpec:
    """Represents a sort specification.

    Frozen for :class:`Filter`'s reason and with none of its difficulty: this
    is the other spec a :class:`Query` holds, it describes a sort rather than
    accumulating one, and nothing assigns to a field of one after construction.
    Both its fields are hashable on their own, so the generated hash is correct
    here and no projection is needed.
    """

    field: str
    order: SortOrder = SortOrder.ASC

    def to_dict(self) -> dict[str, str]:
        """Convert sort spec to dictionary representation."""
        return {"field": self.field, "order": self.order.value}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SortSpec:
        """Create sort spec from dictionary representation."""
        return cls(field=data["field"], order=coerce_sort_order(data.get("order", "asc")))


@dataclass
class VectorQuery:
    """Represents a vector similarity search query.

    This dataclass encapsulates all parameters needed for vector similarity search,
    including the query vector, distance metric, and various search options.
    """

    vector: np.ndarray | list[float]  # Query vector or embeddings
    field_name: str = "embedding"  # Vector field name to search
    k: int = 10  # Number of results (top-k)
    metric: DistanceMetric | str = "cosine"  # Distance metric
    include_source: bool = True  # Include source text in results
    score_threshold: float | None = None  # Minimum similarity score
    rerank: bool = False  # Whether to rerank results
    rerank_k: int | None = None  # Number of results to rerank (default: 2*k)
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def to_dict(self) -> dict[str, Any]:
        """Convert vector query to dictionary representation."""
        import numpy as np

        # Handle vector serialization
        vector_data = self.vector
        if isinstance(vector_data, np.ndarray):
            vector_data = vector_data.tolist()

        # Handle metric serialization
        metric_value = self.metric
        if hasattr(metric_value, "value"):  # DistanceMetric enum
            metric_value = metric_value.value

        result = {
            "vector": vector_data,
            "field": self.field_name,
            "k": self.k,
            "metric": metric_value,
            "include_source": self.include_source,
        }

        if self.score_threshold is not None:
            result["score_threshold"] = self.score_threshold
        if self.rerank:
            result["rerank"] = self.rerank
            if self.rerank_k is not None:
                result["rerank_k"] = self.rerank_k
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorQuery:
        """Create vector query from dictionary representation."""
        import numpy as np

        from .vector.types import DistanceMetric

        # Handle vector deserialization
        vector_data = data["vector"]
        if not isinstance(vector_data, np.ndarray):
            vector_data = np.array(vector_data, dtype=np.float32)

        # Handle metric deserialization
        metric_value = data.get("metric", "cosine")
        if isinstance(metric_value, str):
            try:
                metric_value = DistanceMetric(metric_value)
            except ValueError:
                # Keep as string if not a valid enum value
                pass

        return cls(
            vector=vector_data,
            field_name=data.get("field", "embedding"),
            k=data.get("k", 10),
            metric=metric_value,
            include_source=data.get("include_source", True),
            score_threshold=data.get("score_threshold"),
            rerank=data.get("rerank", False),
            rerank_k=data.get("rerank_k"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Query:
    """Represents a database query with filters, sorting, pagination, and vector search.

    A Query combines multiple filter conditions, sort specifications, and pagination
    options to retrieve records from a database. Supports fluent interface for building queries.

    Attributes:
        filters: List of filter conditions
        sort_specs: List of sort specifications
        limit_value: Maximum number of results
        offset_value: Number of results to skip
        fields: List of field names to include (projection)
        vector_query: Optional vector similarity search parameters

    Example:
        ```python
        from dataknobs_data import Query, Filter, Operator, SortOrder, SortSpec, database_factory

        # Simple query with filters
        query = Query(
            filters=[
                Filter("age", Operator.GT, 25),
                Filter("status", Operator.EQ, "active")
            ]
        )

        # Using fluent interface
        query = (
            Query()
            .filter("age", Operator.GT, 25)
            .filter("status", Operator.EQ, "active")
            .sort_by("age", SortOrder.DESC)
            .limit(10)
            .offset(20)
        )

        # With field projection
        query = (
            Query()
            .filter("age", Operator.GT, 25)
            .select("name", "age", "email")
        )

        # Execute query
        db = database_factory("memory")
        results = db.search(query)
        ```
    """

    filters: list[Filter] = field(default_factory=list)
    sort_specs: list[SortSpec] = field(default_factory=list)
    limit_value: int | None = None
    offset_value: int | None = None
    fields: list[str] | None = None  # Field projection
    vector_query: VectorQuery | None = None  # Vector similarity search

    @property
    def sort_property(self) -> list[SortSpec]:
        """Get sort specifications (backward compatibility)."""
        return self.sort_specs

    @property
    def limit_property(self) -> int | None:
        """Get limit value (backward compatibility)."""
        return self.limit_value

    @property
    def offset_property(self) -> int | None:
        """Get offset value (backward compatibility)."""
        return self.offset_value

    def filter(self, field: str, operator: str | Operator, value: Any = None) -> Query:
        """Add a filter to the query (fluent interface).

        Args:
            field: The field name to filter on
            operator: The operator, as an :class:`Operator` or any spelling
                :func:`coerce_operator` accepts
            value: The value to compare against

        Returns:
            Self for method chaining

        Raises:
            ValueError: If ``operator`` names no operator. It used to mean
                equality instead, so a typo returned the wrong rows rather
                than failing.
        """
        self.filters.append(Filter(field=field, operator=coerce_operator(operator), value=value))
        return self

    def sort_by(self, field: str, order: str | SortOrder = "asc") -> Query:
        """Add a sort specification to the query (fluent interface).

        Args:
            field: The field name to sort by
            order: The sort order ("asc", "desc", case-insensitive, or a
                :class:`SortOrder`)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If ``order`` names no order. Every string but
                ``"asc"`` used to mean descending, so ``"ascending"`` sorted
                the wrong way.
        """
        self.sort_specs.append(SortSpec(field=field, order=coerce_sort_order(order)))
        return self

    def sort(self, field: str, order: str | SortOrder = "asc") -> Query:
        """Add sorting (fluent interface)."""
        return self.sort_by(field, order)

    def set_limit(self, limit: int) -> Query:
        """Set the result limit (fluent interface).

        Args:
            limit: Maximum number of results

        Returns:
            Self for method chaining
        """
        self.limit_value = limit
        return self

    def limit(self, value: int) -> Query:
        """Set limit (fluent interface)."""
        return self.set_limit(value)

    def set_offset(self, offset: int) -> Query:
        """Set the result offset (fluent interface).

        Args:
            offset: Number of results to skip

        Returns:
            Self for method chaining
        """
        self.offset_value = offset
        return self

    def offset(self, value: int) -> Query:
        """Set offset (fluent interface)."""
        return self.set_offset(value)

    def select(self, *fields: str) -> Query:
        """Set field projection (fluent interface).

        Args:
            fields: Field names to include in results

        Returns:
            Self for method chaining
        """
        self.fields = list(fields) if fields else None
        return self

    def clear_filters(self) -> Query:
        """Clear all filters (fluent interface)."""
        self.filters = []
        return self

    def clear_sort(self) -> Query:
        """Clear all sort specifications (fluent interface)."""
        self.sort_specs = []
        return self

    def similar_to(
        self,
        vector: np.ndarray | list[float],
        field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric | str = "cosine",
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> Query:
        """Add vector similarity search to the query.

        This method sets up a vector similarity search that will find the k most
        similar vectors to the provided query vector.

        Args:
            vector: Query vector to search for similar vectors
            field: Vector field name to search (default: "embedding")
            k: Number of results to return (default: 10)
            metric: Distance metric to use (default: "cosine")
            include_source: Whether to include source text in results (default: True)
            score_threshold: Minimum similarity score threshold (optional)

        Returns:
            Self for method chaining
        """
        self.vector_query = VectorQuery(
            vector=vector,
            field_name=field,
            k=k,
            metric=metric,
            include_source=include_source,
            score_threshold=score_threshold,
        )
        # Always update limit to match k
        self.limit_value = k
        return self

    def near_text(
        self,
        text: str,
        embedding_fn: Callable[[str], np.ndarray | list[float]],
        field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric | str = "cosine",
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> Query:
        """Add text-based vector similarity search to the query.

        This is a convenience method that converts text to a vector using the
        provided embedding function, then performs vector similarity search.

        Args:
            text: Text to convert to vector for similarity search
            embedding_fn: Function to convert text to vector. The list
                return is admitted because :meth:`similar_to`, where this
                result goes, already accepts one --- and because it is the
                shape :class:`~dataknobs_data.vector.SyncTextEmbedder`
                returns, which is the supported way to reach an async
                embedder from this synchronous call.
            field: Vector field name to search (default: "embedding")
            k: Number of results to return (default: 10)
            metric: Distance metric to use (default: "cosine")
            include_source: Whether to include source text in results (default: True)
            score_threshold: Minimum similarity score threshold (optional)

        Returns:
            Self for method chaining
        """
        # Convert text to vector using provided embedding function
        vector = embedding_fn(text)
        return self.similar_to(
            vector=vector,
            field=field,
            k=k,
            metric=metric,
            include_source=include_source,
            score_threshold=score_threshold,
        )

    def hybrid(
        self,
        text_query: str | None = None,
        vector: np.ndarray | list[float] | None = None,
        text_field: str = "content",
        vector_field: str = "embedding",
        alpha: float = 0.5,
        k: int = 10,
        metric: DistanceMetric | str = "cosine",
    ) -> Query:
        """Create a hybrid query combining text and vector search.

        This method combines traditional text search with vector similarity search,
        allowing for more nuanced queries that leverage both exact text matching
        and semantic similarity.

        Args:
            text_query: Text to search for (optional)
            vector: Vector for similarity search (optional)
            text_field: Field for text search (default: "content")
            vector_field: Field for vector search (default: "embedding")
            alpha: Weight balance between text (0.0) and vector (1.0) search (default: 0.5)
            k: Number of results to return (default: 10)
            metric: Distance metric for vector search (default: "cosine")

        Returns:
            Self for method chaining

        Note:
            - alpha=0.0 gives full weight to text search
            - alpha=1.0 gives full weight to vector search
            - alpha=0.5 gives equal weight to both
        """
        # Add text filter if provided
        if text_query:
            self.filter(text_field, Operator.LIKE, f"%{text_query}%")

        # Add vector search if provided
        if vector is not None:
            self.vector_query = VectorQuery(
                vector=vector,
                field_name=vector_field,
                k=k,
                metric=metric,
                include_source=True,
            )
            # Store alpha in vector query metadata for backend to use
            self.vector_query.metadata = {"hybrid_alpha": alpha}

        # Set limit if not already set
        if self.limit_value is None:
            self.limit_value = k

        return self

    def with_reranking(self, rerank_k: int | None = None) -> Query:
        """Enable result reranking for vector queries.

        Args:
            rerank_k: Number of results to rerank (default: 2*k from vector query)

        Returns:
            Self for method chaining
        """
        if self.vector_query:
            self.vector_query.rerank = True
            self.vector_query.rerank_k = rerank_k or (self.vector_query.k * 2)
        return self

    def clear_vector(self) -> Query:
        """Clear vector search from the query (fluent interface)."""
        self.vector_query = None
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert query to dictionary representation."""
        result: dict[str, Any] = {
            "filters": [f.to_dict() for f in self.filters],
            "sort": [s.to_dict() for s in self.sort_specs],
        }
        if self.limit_value is not None:
            result["limit"] = self.limit_value
        if self.offset_value is not None:
            result["offset"] = self.offset_value
        if self.fields is not None:
            result["fields"] = self.fields
        if self.vector_query is not None:
            result["vector_query"] = self.vector_query.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Query:
        """Create query from dictionary representation."""
        query = cls()

        for filter_data in data.get("filters", []):
            query.filters.append(Filter.from_dict(filter_data))

        for sort_data in data.get("sort", []):
            query.sort_specs.append(SortSpec.from_dict(sort_data))

        query.limit_value = data.get("limit")
        query.offset_value = data.get("offset")
        query.fields = data.get("fields")

        if "vector_query" in data:
            query.vector_query = VectorQuery.from_dict(data["vector_query"])

        return query

    def copy(self) -> Query:
        """Create a copy of the query."""
        import copy

        return Query(
            filters=copy.deepcopy(self.filters),
            sort_specs=copy.deepcopy(self.sort_specs),
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            fields=self.fields.copy() if self.fields else None,
            vector_query=copy.deepcopy(self.vector_query) if self.vector_query else None,
        )

    def or_(self, *filters: Filter | Query) -> ComplexQuery:
        """Create a ComplexQuery with OR logic.

        The current query's filters become an AND group, combined with OR conditions.
        Example: Query with filters [A, B] calling or_(C, D) creates: (A AND B) AND (C OR D)

        Args:
            filters: Filter objects or Query objects to OR together

        Returns:
            ComplexQuery with OR logic
        """
        from .query_logic import (
            ComplexQuery,
            Condition,
            FilterCondition,
            LogicCondition,
            LogicOperator,
        )

        # Build OR conditions from the arguments
        or_conditions: list[Condition] = []
        for item in filters:
            if isinstance(item, Filter):
                or_conditions.append(FilterCondition(item))
            elif isinstance(item, Query):
                if len(item.filters) == 1:
                    or_conditions.append(FilterCondition(item.filters[0]))
                elif item.filters:
                    and_cond = LogicCondition(operator=LogicOperator.AND)
                    for f in item.filters:
                        and_cond.conditions.append(FilterCondition(f))
                    or_conditions.append(and_cond)

        # Create the OR condition group
        or_group: Condition | None = None
        if or_conditions:
            if len(or_conditions) == 1:
                or_group = or_conditions[0]
            else:
                or_group = LogicCondition(operator=LogicOperator.OR, conditions=or_conditions)

        # Combine with existing filters (if any) using AND
        if self.filters:
            # Create AND condition for existing filters
            existing: Condition
            if len(self.filters) == 1:
                existing = FilterCondition(self.filters[0])
            else:
                # Built through its own name, because appending to
                # `.conditions` is a LogicCondition capability and the
                # variable that leaves this block is a Condition.
                group = LogicCondition(operator=LogicOperator.AND)
                for f in self.filters:
                    group.conditions.append(FilterCondition(f))
                existing = group

            # Combine existing AND new OR group with AND
            root_condition: Condition | None
            if or_group:
                root_condition = LogicCondition(
                    operator=LogicOperator.AND, conditions=[existing, or_group]
                )
            else:
                root_condition = existing
        else:
            # No existing filters, just use OR group
            root_condition = or_group

        return ComplexQuery(
            condition=root_condition,
            sort_specs=self.sort_specs.copy(),
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            fields=self.fields.copy() if self.fields else None,
        )

    def and_(self, *filters: Filter | Query) -> Query:
        """Add more filters with AND logic (convenience method).

        Args:
            filters: Filter objects or Query objects to AND together

        Returns:
            Self for chaining
        """
        for item in filters:
            if isinstance(item, Filter):
                self.filters.append(item)
            elif isinstance(item, Query):
                self.filters.extend(item.filters)
        return self

    def not_(self, filter: Filter) -> ComplexQuery:
        """Create a ComplexQuery with NOT logic.

        The ``NOT`` is the filter's complement, so it matches a record without
        the filter's field, or with a ``null`` one: ``not_(Filter("c",
        Operator.EQ, "x"))`` keeps such a record where ``Filter("c",
        Operator.NEQ, "x")`` does not. See :class:`LogicCondition`.

        Args:
            filter: Filter to negate

        Returns:
            ComplexQuery with NOT logic
        """
        from .query_logic import (
            ComplexQuery,
            Condition,
            FilterCondition,
            LogicCondition,
            LogicOperator,
        )

        # Current filters as AND
        conditions: list[Condition] = []
        if self.filters:
            if len(self.filters) == 1:
                conditions.append(FilterCondition(self.filters[0]))
            else:
                and_cond = LogicCondition(operator=LogicOperator.AND)
                for f in self.filters:
                    and_cond.conditions.append(FilterCondition(f))
                conditions.append(and_cond)

        # Add NOT condition
        not_cond = LogicCondition(operator=LogicOperator.NOT, conditions=[FilterCondition(filter)])
        conditions.append(not_cond)

        # Create root condition
        if len(conditions) == 1:
            root_condition = conditions[0]
        else:
            root_condition = LogicCondition(operator=LogicOperator.AND, conditions=conditions)

        return ComplexQuery(
            condition=root_condition,
            sort_specs=self.sort_specs.copy(),
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            fields=self.fields.copy() if self.fields else None,
        )

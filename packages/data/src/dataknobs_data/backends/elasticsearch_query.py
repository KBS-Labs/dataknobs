# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Shared Elasticsearch filter-to-Query-DSL translation.

The sync and async Elasticsearch backends and the vector-pre-filter mixin all
translate a :class:`~dataknobs_data.query.Filter` into Elasticsearch Query DSL
through the functions here, so the translation lives in exactly one place. The
functions are pure — they take query objects and how the index maps the fields
they name (:class:`FieldTypes`) and return ``dict`` clauses with no I/O — so
every operator's emitted DSL can be pinned in fast offline unit tests that run
wherever CI runs, not only against a live Elasticsearch.

Field-path rules:

* The ``id`` field is the record's storage key. The document carries it as a
  top-level ``id`` keyword mirroring ``_id``; unlike the ``_id`` metafield it
  supports the full operator set (term/terms/range/prefix/wildcard/regexp/
  exists), so ``Filter("id", …)`` is a first-class query target. It is already
  a keyword and never takes a ``.keyword`` suffix. Because this targets the
  stamped top-level ``id`` field (not the ``_id`` metafield), ``id`` filtering
  only sees documents written with that field present — every write path stamps
  it now, but records indexed by an older version that did not stamp a *minted*
  ``id`` must be reindexed to become ``id``-queryable.
* A record data field literally named ``id`` (``record.data["id"]``) is not
  reachable through the query API — ``Filter("id", …)`` always means the storage
  key. Query such a field under a different name.
* Other fields live under ``data.<field>``. A field mapped as analyzed text is
  reached through its keyword sub-field by every operator but existence, and by
  a sort, so a comparison is against the **full, un-analyzed** value. Any other
  field (keyword, number, date, boolean) is reached directly. A pattern on a
  field that holds no strings (a number, boolean, object or vector) matches
  nothing, negated or not.
* What the index cannot answer raises ``ValueError``, negated or not: any
  comparison or pattern on a text field with no keyword sub-field (a custom
  mapping), which has no full value to reach, and a pattern on a field that is
  not matched as text, such as a date or an ``ip``. Elasticsearch itself refuses
  to sort by the former. Under an alias, one index that maps the field this
  way is enough, except where another gives it a keyword sub-field, which is
  then used.

Semantics:

* ``LIKE``/``NOT_LIKE`` translate SQL wildcards (``%``→``*``, ``_``→``?``) and
  match case-insensitively, consistent with the in-memory and SQL backends,
  though Elasticsearch folds ASCII case only (``'é'`` does not match ``'É'``),
  as SQLite does. Any other character — including the Lucene wildcard
  metacharacters ``*`` ``?`` and the backslash escape — is escaped so it
  matches literally, mirroring SQL ``LIKE`` where only ``%`` and ``_`` are
  wildcards. The case-insensitive ``wildcard`` form requires Elasticsearch
  ≥ 7.10.
* ``REGEX`` runs against the **full field value** via the keyword sub-field
  (case-sensitive), so a pattern matches the whole string — matching the
  in-memory (``re.search``) and SQL backends. Against the analyzed base path a
  ``regexp`` would match per-token (and lowercased), which is why the keyword
  sub-field is used. Note Elasticsearch ``regexp`` is anchored (the pattern must
  match the entire value) and uses Lucene RegExp syntax, which differs from
  Python ``re`` (no ``^``/``$`` anchors, no look-around).
* ``STARTS_WITH`` is a case-sensitive ``prefix`` query.
* A bound relates only to values of its own kind, as ``Filter.matches`` has it.
  The mapping says so for a number or boolean field, and the storage key is
  always a string: a bound of another kind matches nothing there. Nor does a
  boolean on a date field, or anything but a string on an ``ip`` field. ``None`` and
  NaN match nothing anywhere, and a ``None`` member of a list is dropped. A
  bound that is not a string is still compared as text with a string field,
  whose index holds ``5`` and ``"5"`` alike, and a number with a date field as
  epoch milliseconds. A list is compared element by
  element, and equality with a list is refused by Elasticsearch.
* A negated operator (``NEQ``/``NOT_IN``/``NOT_LIKE``/``NOT_BETWEEN``) matches a
  document that **has a value** the operator it negates does not match: ``exists``
  with a ``must_not``, because a bare ``must_not`` also keeps a document without
  the field. Each clause is self-contained, so callers only ever wrap the
  returned clauses in ``bool``/``must``.
* An unsupported operator raises ``ValueError`` rather than silently matching
  every document — a dropped filter that falls back to ``match_all`` returns
  everything, the worst failure mode for a query engine. ``ValueError`` matches
  the in-memory matcher's own unknown-operator raise (``Filter.matches``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..query import (
    Operator,
    SortOrder,
    is_storage_key_field,
    value_kind,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping, Sequence

    from ..query import Filter, Query, SortSpec
    from ..query_logic import ComplexQuery, Condition


#: Each negated operator, and the one whose matches it leaves out.
_NEGATED = {
    Operator.NEQ: Operator.EQ,
    Operator.NOT_IN: Operator.IN,
    Operator.NOT_BETWEEN: Operator.BETWEEN,
    Operator.NOT_LIKE: Operator.LIKE,
}
_RANGE_BOUNDS = {
    Operator.GT: "gt",
    Operator.GTE: "gte",
    Operator.LT: "lt",
    Operator.LTE: "lte",
}
#: Operators whose operand is a pattern over the full, un-analyzed string.
_PATTERN_OPS = frozenset({Operator.LIKE, Operator.STARTS_WITH, Operator.REGEX})
_EXISTENCE_OPS = frozenset({Operator.EXISTS, Operator.NOT_EXISTS})

#: Elasticsearch types whose terms are words rather than the stored value.
_ANALYZED_TYPES = frozenset({"text", "match_only_text", "search_as_you_type"})
#: Elasticsearch types that hold a string's full value.
_EXACT_STRING_TYPES = frozenset({"keyword", "constant_keyword", "wildcard", "flattened"})
#: Elasticsearch types a pattern can be tried against.
_PATTERN_TYPES = _EXACT_STRING_TYPES | {"version"}
#: The Elasticsearch types that hold one kind of value alone, and it.
_SCALAR_TYPES = {
    **dict.fromkeys(
        (
            "byte",
            "short",
            "integer",
            "long",
            "unsigned_long",
            "half_float",
            "float",
            "double",
            "scaled_float",
        ),
        "number",
    ),
    "boolean": "boolean",
}
#: Elasticsearch types no string is stored in, so no pattern matches.
_STRINGLESS_TYPES = frozenset({*_SCALAR_TYPES, "object", "nested", "dense_vector"})
#: The kinds of bound (:func:`~dataknobs_data.query.value_kind`) that can relate
#: to a value of an Elasticsearch type. It reads a bound of another kind as a
#: spelling of its own (``"3"`` as the number) or refuses it. A date takes a
#: number, because it can hold one: epoch milliseconds. A type not named here is
#: left to read the bound itself; a string field may hold what was written as a
#: number.
_BOUND_KINDS: dict[str, frozenset[str]] = {
    **{name: frozenset({kind}) for name, kind in _SCALAR_TYPES.items()},
    **dict.fromkeys(("date", "date_nanos"), frozenset({"string", "timestamp", "number"})),
    "ip": frozenset({"string"}),
}


class FieldTypes:
    """How one index maps the fields queries name, learned as they are named.

    The I/O is the backend's. It asks :meth:`unknown` which fields a query names
    that are not known yet, requests the field capabilities of
    :meth:`request_fields`, and hands the response to :meth:`learn`. A field the
    index does not map yet is not remembered, so it is asked about again: a later
    write may map it. A mapped field's type never changes within an index, so it
    is kept for as long as this object is.
    """

    def __init__(self) -> None:
        self._types: dict[str, frozenset[str]] = {}
        self._exact: dict[str, str] = {}

    def unknown(self, names: Iterable[str]) -> list[str]:
        """The data fields among ``names`` whose mapping is not known."""
        return sorted({n for n in names if not is_storage_key_field(n) and n not in self._types})

    @staticmethod
    def request_fields(names: Iterable[str]) -> list[str]:
        """The field patterns to request capabilities for: each field and its sub-fields."""
        return [path for n in names for path in (f"data.{n}", f"data.{n}.*")]

    def learn(self, names: Iterable[str], capabilities: Mapping[str, Mapping[str, Any]]) -> None:
        """Record what a field-capabilities response says of ``names``.

        Args:
            names: The fields that were asked about.
            capabilities: The response's ``fields``: path to its types.
        """
        for name in names:
            path = f"data.{name}"
            # An index alias may span indices that do not all map the field.
            types = frozenset(capabilities.get(path, {})) - {"unmapped"}
            if not types:
                continue
            self._types[name] = types
            if types & _ANALYZED_TYPES:
                exact = sorted(
                    sub
                    for sub, sub_types in capabilities.items()
                    if sub.startswith(f"{path}.") and _EXACT_STRING_TYPES & set(sub_types)
                )
                if exact:
                    preferred = f"{path}.keyword"
                    self._exact[name] = preferred if preferred in exact else exact[0]

    def value_path(self, name: str) -> str:
        """The path holding a field's full value; the field itself when nothing else does."""
        if is_storage_key_field(name):
            # A top-level keyword mirroring ``_id``, which cannot itself be sorted.
            return "id"
        return self._exact.get(name, f"data.{name}")

    def takes_patterns(self, name: str) -> bool:
        """Whether :meth:`value_path` can be matched against a pattern.

        True for a field that is not mapped, which no document has a value for.
        """
        if is_storage_key_field(name) or name in self._exact or name not in self._types:
            return True
        return bool(self._types[name] & _PATTERN_TYPES)

    def relates(self, name: str, bound: Any) -> bool:
        """Whether a bound can relate to any value a field holds.

        False for ``None`` and NaN, which relate to nothing, and for a bound of
        a kind the field's type cannot hold. True for a field that is not
        mapped, which has no value to relate to, and for a list or mapping,
        which Elasticsearch refuses: a field of any type can hold a list.
        """
        kind = value_kind(bound)
        if kind == "never":
            return False
        if is_storage_key_field(name):
            # Always a string. A time is compared as the text it is written as.
            return kind not in ("number", "boolean")
        types = self._types.get(name)
        if kind is None or types is None:
            return True
        return any(t not in _BOUND_KINDS or kind in _BOUND_KINDS[t] for t in types)

    def holds_only_words(self, name: str) -> bool:
        """Whether a field is text whose full value nothing holds, in any index.

        Its own terms are words, and a word is not the value.
        """
        types = self._types.get(name)
        return types is not None and name not in self._exact and bool(types & _ANALYZED_TYPES)

    def holds_no_strings(self, name: str) -> bool:
        """Whether no value of a field is a string."""
        types = self._types.get(name)
        return types is not None and types <= _STRINGLESS_TYPES

    def described(self, name: str) -> str:
        """The types a mapped field has, for an error to name."""
        return " and ".join(sorted(self._types[name]))


def condition_fields(condition: Condition | None) -> set[str]:
    """The fields a condition tree's filters name."""
    from ..query_logic import FilterCondition, LogicCondition

    if isinstance(condition, FilterCondition):
        return {condition.filter.field}
    if isinstance(condition, LogicCondition):
        return {name for sub in condition.conditions for name in condition_fields(sub)}
    return set()


def query_fields(query: Query | ComplexQuery | None) -> set[str]:
    """The fields a query filters or sorts by, whose mapping its translation needs."""
    from ..query_logic import ComplexQuery

    if query is None:
        return set()
    if isinstance(query, ComplexQuery):
        named = condition_fields(query.condition)
    else:
        named = {f.field for f in query.filters}
    return named | {s.field for s in query.sort_specs}


def build_query(query: Query | ComplexQuery, types: FieldTypes) -> dict[str, Any]:
    """Translate either kind of query's filters; ``match_all`` when it has none."""
    from ..query_logic import ComplexQuery

    if not isinstance(query, ComplexQuery):
        return build_bool_query(query.filters, types)
    if query.condition is None:
        return {"match_all": {}}
    return build_complex_es_query(query.condition, types)


def build_sort(sort_specs: Sequence[SortSpec], types: FieldTypes) -> list[dict[str, Any]]:
    """Translate a query's sort, reaching each field's full value.

    A record with no value sorts last in either direction, as on every backend.
    ``unmapped_type`` says the same of a field no record has, which Elasticsearch
    otherwise refuses to sort by.
    """
    return [
        {
            types.value_path(spec.field): {
                "order": "desc" if spec.order == SortOrder.DESC else "asc",
                "missing": "_last",
                "unmapped_type": "keyword",
            }
        }
        for spec in sort_specs
    ]


def _field_path(filter_obj: Filter, types: FieldTypes) -> str:
    """The document path an operator's clause should target for this filter."""
    name = filter_obj.field
    if filter_obj.operator in _EXISTENCE_OPS:
        return _presence_path(name)
    return types.value_path(name)


def _presence_path(name: str) -> str:
    """The path that says whether a field has a value.

    Not a sub-field, which leaves out a value past its ``ignore_above``.
    """
    return name if is_storage_key_field(name) else f"data.{name}"


def _sql_wildcard_to_es(pattern: str) -> str:
    """Translate a SQL ``LIKE`` pattern to Elasticsearch ``wildcard`` syntax.

    Only ``%`` and ``_`` are SQL wildcards; every other character is literal —
    including the Lucene wildcard metacharacters ``*`` ``?`` and the backslash
    escape. Those are escaped first (so they match literally), *then* the SQL
    wildcards are mapped onto the ES forms, so a freshly-introduced ``*``/``?``
    is never re-escaped. Raises ``ValueError`` for a non-string pattern.
    """
    if not isinstance(pattern, str):
        raise ValueError(f"LIKE/NOT_LIKE pattern must be a string, got: {pattern!r}")
    escaped = (
        pattern.replace("\\", "\\\\")  # escape the escape char first
        .replace("*", "\\*")  # literal SQL '*' -> ES literal
        .replace("?", "\\?")  # literal SQL '?' -> ES literal
    )
    return escaped.replace("%", "*").replace("_", "?")


def build_filter_es_query(filter_obj: Filter, types: FieldTypes) -> dict[str, Any]:
    """Translate one :class:`Filter` into an Elasticsearch Query-DSL clause.

    Pure and self-contained, so a caller only ever composes the returned clauses
    under ``bool``/``must``. A negated operator matches a document that has a
    value the operator it negates does not match. Raises ``ValueError`` for an
    operator this translator cannot express, or that the index cannot answer.
    """
    op = filter_obj.operator
    if op == Operator.NOT_EXISTS:
        return {"bool": {"must_not": {"exists": {"field": _field_path(filter_obj, types)}}}}
    if op not in _NEGATED:
        clause = _positive_clause(op, filter_obj, types)
        return {"match_none": {}} if clause is None else clause

    negated = _positive_clause(_NEGATED[op], filter_obj, types)
    if op == Operator.NOT_LIKE and types.holds_no_strings(filter_obj.field):
        # A pattern asks for a string, negated or not.
        return {"match_none": {}}
    present = {"exists": {"field": _presence_path(filter_obj.field)}}
    if negated is None:
        return present
    return {"bool": {"filter": present, "must_not": negated}}


def _positive_clause(op: Operator, filter_obj: Filter, types: FieldTypes) -> dict[str, Any] | None:
    """The clause for an operator that is not negated; ``None`` if nothing can match."""
    name = filter_obj.field
    path = _field_path(filter_obj, types)
    value = filter_obj.value

    if op == Operator.EXISTS:
        return {"exists": {"field": path}}
    if op in _PATTERN_OPS:
        clause = _pattern_clause(op, path, value)
        if types.holds_no_strings(name):
            return None
        if types.holds_only_words(name) or not types.takes_patterns(name):
            raise _unanswerable(filter_obj, types)
        return clause
    if op == Operator.EQ:
        return {"term": {path: value}} if _relates(filter_obj, value, types) else None
    if op == Operator.IN:
        members = [m for m in value if _relates(filter_obj, m, types)]
        return {"terms": {path: members}} if members else None
    if op in _RANGE_BOUNDS:
        if not _relates(filter_obj, value, types):
            return None
        return {"range": {path: {_RANGE_BOUNDS[op]: value}}}
    if op == Operator.BETWEEN:
        # A Filter refuses a range that is not two bounds when it is built.
        lower, upper = value
        if not all(types.relates(name, bound) for bound in value):
            return None
        if not _same_kind(lower, upper):
            return None
        if types.holds_only_words(name):
            raise _unanswerable(filter_obj, types)
        return {"range": {path: {"gte": lower, "lte": upper}}}

    raise ValueError(f"Unsupported operator: {op}")


def _pattern_clause(op: Operator, path: str, value: Any) -> dict[str, Any]:
    if op == Operator.LIKE:
        return {"wildcard": {path: {"value": _sql_wildcard_to_es(value), "case_insensitive": True}}}
    if op == Operator.REGEX:
        # ``path`` holds the full value, not a single analyzed token.
        return {"regexp": {path: value}}
    # Literal, case-sensitive prefix: no case_insensitive flag.
    return {"prefix": {path: value}}


def _relates(filter_obj: Filter, bound: Any, types: FieldTypes) -> bool:
    """:meth:`FieldTypes.relates`, raising where the index cannot say further."""
    if not types.relates(filter_obj.field, bound):
        return False
    if types.holds_only_words(filter_obj.field):
        raise _unanswerable(filter_obj, types)
    return True


def _unanswerable(filter_obj: Filter, types: FieldTypes) -> ValueError:
    """The error for an operator the index holds nothing to answer with."""
    name = filter_obj.field
    asked = f"Elasticsearch cannot answer {filter_obj.operator.name} on {name!r}, "
    mapped = f"which the index has mapped as {types.described(name)}"
    if types.holds_only_words(name):
        return ValueError(
            f"{asked}{mapped} with no keyword sub-field: only its words can be reached. "
            "Map it with a keyword sub-field, or as a keyword."
        )
    return ValueError(f"{asked}{mapped}: a pattern is matched against a keyword.")


def _same_kind(lower: Any, upper: Any) -> bool:
    """Whether one value can lie between two bounds.

    A time may be given as a string, so it goes with any other kind here and
    Elasticsearch reads both bounds as the field's type.
    """
    return len({value_kind(lower), value_kind(upper)} - {"timestamp"}) < 2


def build_bool_query(filters: Sequence[Filter], types: FieldTypes) -> dict[str, Any]:
    """Wrap per-filter clauses in a single ``bool``/``must`` query.

    Returns ``{"match_all": {}}`` for an empty filter sequence. This is the
    outer wrapper both backends use for the plain ``Query`` path.
    """
    must = [build_filter_es_query(f, types) for f in filters]
    return {"bool": {"must": must}} if must else {"match_all": {}}


def build_complex_es_query(condition: Condition, types: FieldTypes) -> dict[str, Any]:
    """Translate a ``ComplexQuery`` condition tree into a nested ``bool`` query.

    ``AND`` → ``must``, ``OR`` → ``should`` (``minimum_should_match: 1``),
    ``NOT`` → ``must_not`` over every condition it holds, which matches when
    none of them does; leaf filters delegate to
    :func:`build_filter_es_query`. A single-clause ``AND``/``OR`` collapses to
    that clause. An empty ``AND`` or ``NOT`` matches every document and an
    empty ``OR`` matches none, as ``LogicCondition.matches`` answers them.

    Raises ``TypeError`` for a condition other than a ``FilterCondition`` or
    ``LogicCondition``, which no clause renders.
    """
    from ..query_logic import FilterCondition, LogicCondition, LogicOperator

    if isinstance(condition, FilterCondition):
        return build_filter_es_query(condition.filter, types)

    if isinstance(condition, LogicCondition):
        clauses = [build_complex_es_query(sub, types) for sub in condition.conditions]

        if condition.operator == LogicOperator.AND:
            if not clauses:
                return {"match_all": {}}
            if len(clauses) == 1:
                return clauses[0]
            return {"bool": {"must": clauses}}

        if condition.operator == LogicOperator.OR:
            if not clauses:
                return {"match_none": {}}
            if len(clauses) == 1:
                return clauses[0]
            return {"bool": {"should": clauses, "minimum_should_match": 1}}

        if condition.operator == LogicOperator.NOT:
            if clauses:
                return {"bool": {"must_not": clauses}}
            return {"match_all": {}}

    # Rendered as match_all, it matched every document whatever it answers.
    raise TypeError(f"Cannot render {type(condition).__name__} as an Elasticsearch query")


# --------------------------------------------------------------------------
# Reading a whole match: from/size inside the result window, search_after past it
# --------------------------------------------------------------------------

#: Elasticsearch's default ``index.max_result_window``. A ``from``/``size``
#: request whose ``from + size`` passes it is refused, not truncated.
DEFAULT_MAX_RESULT_WINDOW = 10_000

#: Hits asked for per request when a read pages with ``search_after``.
SEARCH_AFTER_PAGE_SIZE = 1_000

#: How long a paged read's point in time is kept open between two requests.
PIT_KEEP_ALIVE = "1m"

#: The tiebreaker that makes a paged sort total. Elasticsearch's shard-local
#: document order, which needs no mapping and is valid only in a point in time.
_TIEBREAK: dict[str, Any] = {"_shard_doc": {"order": "asc"}}


def plan_search(
    query: dict[str, Any],
    sort: list[dict[str, Any]],
    offset: int | None,
    limit: int | None,
    *,
    window: int = DEFAULT_MAX_RESULT_WINDOW,
    page_size: int = SEARCH_AFTER_PAGE_SIZE,
) -> Generator[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Plan the requests that answer one search, whatever page it names.

    A ``Query`` with no ``limit`` means every match, and one request cannot
    say that: with no ``size`` Elasticsearch answers ten hits, and a ``size``
    large enough to mean "all" is refused once ``from + size`` passes
    ``index.max_result_window``. So:

    * a bounded read inside the window is one ``from``/``size`` request,
      exactly as before (``limit=0`` is ``size=0``, zero hits);
    * anything else pages with ``search_after`` in a point in time, over the
      caller's sort (relevance when there is none) tie-broken on
      ``_shard_doc``, so the order is total and the pages come from one
      snapshot. An offset is skipped by reading past it, since
      ``search_after`` has no ``from``.

    A paged request carries a ``pit`` key holding only ``keep_alive``; the
    driver opens the point in time and fills in its id. The plan does no I/O,
    so the sync and async backends drive the same one
    (:func:`run_search_plan`, :func:`arun_search_plan`) and cannot disagree on
    which hits a query returns.

    Args:
        query: The Query DSL clause.
        sort: The sort clauses, empty for relevance order.
        offset: Hits to skip, or None.
        limit: Hits to return, or None for every match.
        window: The index's result window.
        page_size: Hits per ``search_after`` request.

    Yields:
        Each request body, in order. The driver sends back that request's hits.

    Returns:
        The hits asked for, in order.
    """
    start = offset or 0
    if limit is not None and start + limit <= window:
        body: dict[str, Any] = {"query": query, "from": start, "size": limit}
        if sort:
            body["sort"] = sort
        return (yield body)

    paged_sort = [*(sort or [{"_score": {"order": "desc"}}]), _TIEBREAK]
    hits: list[dict[str, Any]] = []
    after: list[Any] | None = None
    skip = start
    while limit is None or len(hits) < limit:
        body = {
            "query": query,
            "sort": paged_sort,
            "size": page_size,
            "pit": {"keep_alive": PIT_KEEP_ALIVE},
        }
        if after is not None:
            body["search_after"] = after
        page = yield body
        if not page:
            break
        after = page[-1]["sort"]
        hits.extend(page[skip:])
        skip = max(0, skip - len(page))
        if len(page) < page_size:
            break
    return hits if limit is None else hits[:limit]


def client_search_kwargs(body: dict[str, Any], index: str) -> dict[str, Any]:
    """Spell a planned request as keyword arguments to the official client.

    A request in a point in time names no index, since the point in time does.

    Args:
        body: A request body from :func:`plan_search`, its ``pit`` id filled in.
        index: The index a request outside a point in time searches.

    Returns:
        Keyword arguments for ``Elasticsearch.search`` / ``AsyncElasticsearch.search``.
    """
    kwargs: dict[str, Any] = {
        "query": body["query"],
        "size": body["size"],
        "sort": body.get("sort"),
        "from_": body.get("from"),
        "search_after": body.get("search_after"),
    }
    if "pit" in body:
        kwargs["pit"] = body["pit"]
    else:
        kwargs["index"] = index
    return kwargs


def run_search_plan(
    plan: Generator[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    execute: Callable[[dict[str, Any]], tuple[list[dict[str, Any]], str | None]],
    open_pit: Callable[[], str],
    close_pit: Callable[[str], object],
) -> list[dict[str, Any]]:
    """Drive a search plan with sync I/O.

    Opens a point in time for the first request that asks for one, sends each
    later request with the newest id Elasticsearch returned, and closes it on
    every path out.

    Args:
        plan: From :func:`plan_search`.
        execute: Sends one request; returns its hits and point-in-time id.
        open_pit: Opens a point in time on the index; returns its id.
        close_pit: Closes a point in time by id.

    Returns:
        The hits the plan returns.
    """
    pit_id: str | None = None
    try:
        body = next(plan)
        while True:
            if "pit" in body:
                if pit_id is None:
                    pit_id = open_pit()
                body = {**body, "pit": {**body["pit"], "id": pit_id}}
            hits, returned_id = execute(body)
            pit_id = returned_id or pit_id
            body = plan.send(hits)
    except StopIteration as done:
        return list(done.value)
    finally:
        plan.close()
        if pit_id is not None:
            close_pit(pit_id)


async def arun_search_plan(
    plan: Generator[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    execute: Callable[[dict[str, Any]], Awaitable[tuple[list[dict[str, Any]], str | None]]],
    open_pit: Callable[[], Awaitable[str]],
    close_pit: Callable[[str], Awaitable[object]],
) -> list[dict[str, Any]]:
    """Drive a search plan with async I/O; see :func:`run_search_plan`."""
    pit_id: str | None = None
    try:
        body = next(plan)
        while True:
            if "pit" in body:
                if pit_id is None:
                    pit_id = await open_pit()
                body = {**body, "pit": {**body["pit"], "id": pit_id}}
            hits, returned_id = await execute(body)
            pit_id = returned_id or pit_id
            body = plan.send(hits)
    except StopIteration as done:
        return list(done.value)
    finally:
        plan.close()
        if pit_id is not None:
            await close_pit(pit_id)

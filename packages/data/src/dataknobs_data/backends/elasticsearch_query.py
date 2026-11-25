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
  field (keyword, number, date, boolean) is reached directly. A string equality
  or a pattern on a field that holds no strings matches nothing.
* A text field with no keyword sub-field (a custom mapping) has no full value to
  reach: an equality or pattern on it matches nothing, a range compares its
  analyzed terms, and Elasticsearch refuses to sort by it.

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
* Every negation (``NEQ``/``NOT_IN``/``NOT_EXISTS``/``NOT_LIKE``/``NOT_BETWEEN``)
  returns a self-contained ``{"bool": {"must_not": …}}`` clause, so callers only
  ever wrap the returned clauses in ``bool``/``must``. Per Elasticsearch's
  three-valued ``must_not`` semantics, a document that is *missing* the field is
  included by a negation (e.g. ``NEQ`` matches docs without the field) — this
  differs from SQL ``!=``, which excludes NULLs.
* An unsupported operator raises ``ValueError`` rather than silently matching
  every document — a dropped filter that falls back to ``match_all`` returns
  everything, the worst failure mode for a query engine. ``ValueError`` matches
  the in-memory matcher's own unknown-operator raise (``Filter.matches``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..query import Operator, SortOrder, is_storage_key_field

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping, Sequence

    from ..query import Filter, Query, SortSpec
    from ..query_logic import ComplexQuery, Condition


# Operators that match a string exactly.
_KEYWORD_EQUALITY_OPS = frozenset({Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN})
# Operators whose operand is a pattern over the full, un-analyzed string.
_KEYWORD_PATTERN_OPS = frozenset(
    {Operator.LIKE, Operator.NOT_LIKE, Operator.STARTS_WITH, Operator.REGEX}
)


_EXISTENCE_OPS = frozenset({Operator.EXISTS, Operator.NOT_EXISTS})

#: Elasticsearch types whose terms are words rather than the stored value.
_ANALYZED_TYPES = frozenset({"text", "match_only_text"})
#: Elasticsearch types that hold a string's full value.
_EXACT_STRING_TYPES = frozenset({"keyword", "constant_keyword", "wildcard", "flattened"})
#: Elasticsearch types for which a string is not a value but a spelling of one:
#: it reads ``"3"`` as the number and refuses a string that spells none.
_NOT_STRING_TYPES = frozenset(
    {
        "byte",
        "short",
        "integer",
        "long",
        "unsigned_long",
        "half_float",
        "float",
        "double",
        "scaled_float",
        "boolean",
    }
)


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

    def holds_exact_strings(self, name: str) -> bool:
        """Whether :meth:`value_path` can be matched against a pattern.

        True for a field that is not mapped, which no document has a value for.
        """
        if is_storage_key_field(name) or name in self._exact or name not in self._types:
            return True
        return bool(self._types[name] & _EXACT_STRING_TYPES)

    def equals_no_string(self, name: str) -> bool:
        """Whether no string equals a field's value, by how it is mapped.

        True for a number or a boolean, and for text whose full value nothing
        holds: its own terms are words, and a word is not the value. False for a
        date, an IP address or any other type a string is the way to write:
        Elasticsearch reads the operand as the type, as it does a range's.
        """
        if name in self._exact or name not in self._types:
            return False
        return self._types[name] <= _NOT_STRING_TYPES | _ANALYZED_TYPES


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


def _is_string_value(value: Any) -> bool:
    """Whether a value (or the first element of a list) is a string.

    For a membership list this inspects only the first element — a
    heterogeneous ``IN``/``NOT_IN`` list keys its field path off ``value[0]``.
    """
    if isinstance(value, str):
        return True
    return bool(value) and isinstance(value, list) and isinstance(value[0], str)


def _field_path(filter_obj: Filter, types: FieldTypes) -> str:
    """The document path an operator's clause should target for this filter."""
    name = filter_obj.field
    op = filter_obj.operator
    if op in _EXISTENCE_OPS and not is_storage_key_field(name):
        # A sub-field leaves out a value past its ``ignore_above``.
        return f"data.{name}"
    if op in _KEYWORD_PATTERN_OPS:
        matches_nothing = not types.holds_exact_strings(name)
    elif op in _KEYWORD_EQUALITY_OPS and _is_string_value(filter_obj.value):
        matches_nothing = types.equals_no_string(name)
    else:
        matches_nothing = False
    if matches_nothing:
        # No string is there to match, and a path nothing maps says so.
        return f"data.{name}.keyword"
    return types.value_path(name)


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

    Pure and self-contained: negations wrap themselves in ``bool``/``must_not``,
    so a caller only ever composes the returned clauses under ``bool``/``must``.
    Raises ``ValueError`` for an operator this translator cannot express.
    """
    op = filter_obj.operator
    field_path = _field_path(filter_obj, types)
    value = filter_obj.value

    if op == Operator.EQ:
        return {"term": {field_path: value}}
    if op == Operator.NEQ:
        return {"bool": {"must_not": {"term": {field_path: value}}}}
    if op == Operator.GT:
        return {"range": {field_path: {"gt": value}}}
    if op == Operator.GTE:
        return {"range": {field_path: {"gte": value}}}
    if op == Operator.LT:
        return {"range": {field_path: {"lt": value}}}
    if op == Operator.LTE:
        return {"range": {field_path: {"lte": value}}}
    if op == Operator.LIKE:
        return {
            "wildcard": {
                field_path: {
                    "value": _sql_wildcard_to_es(value),
                    "case_insensitive": True,
                }
            }
        }
    if op == Operator.NOT_LIKE:
        return {
            "bool": {
                "must_not": {
                    "wildcard": {
                        field_path: {
                            "value": _sql_wildcard_to_es(value),
                            "case_insensitive": True,
                        }
                    }
                }
            }
        }
    if op == Operator.IN:
        return {"terms": {field_path: value}}
    if op == Operator.NOT_IN:
        return {"bool": {"must_not": {"terms": {field_path: value}}}}
    if op == Operator.EXISTS:
        return {"exists": {"field": field_path}}
    if op == Operator.NOT_EXISTS:
        return {"bool": {"must_not": {"exists": {"field": field_path}}}}
    if op == Operator.REGEX:
        # ``field_path`` holds the full value, not a single analyzed token.
        return {"regexp": {field_path: value}}
    if op == Operator.STARTS_WITH:
        # Literal, case-sensitive prefix — no case_insensitive flag.
        return {"prefix": {field_path: value}}
    if op == Operator.BETWEEN:
        # A Filter refuses a range that is not two bounds when it is built.
        lower, upper = value
        return {"range": {field_path: {"gte": lower, "lte": upper}}}
    if op == Operator.NOT_BETWEEN:
        lower, upper = value
        return {"bool": {"must_not": {"range": {field_path: {"gte": lower, "lte": upper}}}}}

    raise ValueError(f"Unsupported operator: {op}")


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

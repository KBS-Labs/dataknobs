# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Shared Elasticsearch filter-to-Query-DSL translation.

The sync and async Elasticsearch backends and the vector-pre-filter mixin all
translate a :class:`~dataknobs_data.query.Filter` into Elasticsearch Query DSL
through the functions here, so the translation lives in exactly one place. The
functions are pure — they take query objects and return ``dict`` clauses with
no I/O and no backend state — so every operator's emitted DSL can be pinned in
fast offline unit tests that run wherever CI runs, not only against a live
Elasticsearch.

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
* Other fields live under ``data.<field>``. The ``.keyword`` sub-field is used
  wherever matching is against the **full, un-analyzed** value — equality,
  membership, wildcard, prefix, and regex on string values; the analyzed base
  path is used only for range and existence.

Semantics:

* ``LIKE``/``NOT_LIKE`` translate SQL wildcards (``%``→``*``, ``_``→``?``) and
  match case-insensitively, consistent with the in-memory and SQL backends,
  though Elasticsearch folds ASCII case only (``'é'`` does not match ``'É'``),
  as SQLite does. Any other character — including the Lucene wildcard
  metacharacters ``*`` ``?`` and the backslash escape — is escaped so it
  matches literally, mirroring SQL ``LIKE`` where only ``%`` and ``_`` are
  wildcards. The case-insensitive ``wildcard`` form requires Elasticsearch
  ≥ 7.10.
* ``REGEX`` runs against the **full field value** via the ``.keyword`` sub-field
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

from ..query import Operator, is_storage_key_field

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Sequence

    from ..query import Filter
    from ..query_logic import Condition


# Operators that use the ``.keyword`` sub-field for exact matching on strings.
_KEYWORD_EQUALITY_OPS = frozenset({Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN})
# Pattern operators always match against the full, un-analyzed value, so they
# unconditionally target the ``.keyword`` sub-field. ``REGEX`` is here (not on
# the analyzed base path) so a pattern matches the whole string rather than a
# single analyzed token — see the module Semantics note.
_KEYWORD_PATTERN_OPS = frozenset(
    {Operator.LIKE, Operator.NOT_LIKE, Operator.STARTS_WITH, Operator.REGEX}
)


def _is_string_value(value: Any) -> bool:
    """Whether a value (or the first element of a list) is a string.

    For a membership list this inspects only the first element — a
    heterogeneous ``IN``/``NOT_IN`` list keys its field path off ``value[0]``.
    """
    if isinstance(value, str):
        return True
    return bool(value) and isinstance(value, list) and isinstance(value[0], str)


def _field_path(filter_obj: Filter) -> str:
    """The document path an operator's clause should target for this filter.

    ``id`` resolves to the top-level ``id`` keyword; other fields to
    ``data.<field>`` with the ``.keyword`` suffix appended where exact matching
    on a string applies.
    """
    if is_storage_key_field(filter_obj.field):
        # Already a keyword — never suffixed.
        return "id"

    base = f"data.{filter_obj.field}"
    op = filter_obj.operator
    if op in _KEYWORD_PATTERN_OPS:
        return f"{base}.keyword"
    if op in _KEYWORD_EQUALITY_OPS and _is_string_value(filter_obj.value):
        return f"{base}.keyword"
    return base


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


def build_filter_es_query(filter_obj: Filter) -> dict[str, Any]:
    """Translate one :class:`Filter` into an Elasticsearch Query-DSL clause.

    Pure and self-contained: negations wrap themselves in ``bool``/``must_not``,
    so a caller only ever composes the returned clauses under ``bool``/``must``.
    Raises ``ValueError`` for an operator this translator cannot express.
    """
    op = filter_obj.operator
    field_path = _field_path(filter_obj)
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
        # ``field_path`` is the ``.keyword`` sub-field (or ``id``), so the
        # regexp matches the full value, not a single analyzed token.
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


def build_bool_query(filters: Sequence[Filter]) -> dict[str, Any]:
    """Wrap per-filter clauses in a single ``bool``/``must`` query.

    Returns ``{"match_all": {}}`` for an empty filter sequence. This is the
    outer wrapper both backends use for the plain ``Query`` path.
    """
    must = [build_filter_es_query(f) for f in filters]
    return {"bool": {"must": must}} if must else {"match_all": {}}


def build_complex_es_query(condition: Condition) -> dict[str, Any]:
    """Translate a ``ComplexQuery`` condition tree into a nested ``bool`` query.

    ``AND`` → ``must``, ``OR`` → ``should`` (``minimum_should_match: 1``),
    ``NOT`` → ``must_not``; leaf filters delegate to
    :func:`build_filter_es_query`. A single-clause ``AND``/``OR`` collapses to
    that clause. An empty branch is ``{"match_all": {}}``.
    """
    from ..query_logic import FilterCondition, LogicCondition, LogicOperator

    if isinstance(condition, FilterCondition):
        return build_filter_es_query(condition.filter)

    if isinstance(condition, LogicCondition):
        clauses = [build_complex_es_query(sub) for sub in condition.conditions]
        clauses = [c for c in clauses if c]

        if condition.operator == LogicOperator.AND:
            if not clauses:
                return {"match_all": {}}
            if len(clauses) == 1:
                return clauses[0]
            return {"bool": {"must": clauses}}

        if condition.operator == LogicOperator.OR:
            if not clauses:
                return {"match_all": {}}
            if len(clauses) == 1:
                return clauses[0]
            return {"bool": {"should": clauses, "minimum_should_match": 1}}

        if condition.operator == LogicOperator.NOT:
            if clauses:
                return {"bool": {"must_not": clauses[0]}}
            return {"match_all": {}}

    return {"match_all": {}}


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

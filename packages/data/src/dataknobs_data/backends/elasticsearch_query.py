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
  a keyword and never takes a ``.keyword`` suffix.
* Other fields live under ``data.<field>``. The ``.keyword`` sub-field is used
  where exact matching applies — equality, membership, wildcard and prefix on
  string values; the analyzed base path is used for range, existence and regex.

Semantics:

* ``LIKE``/``NOT_LIKE`` translate SQL wildcards (``%``→``*``, ``_``→``?``) and
  match case-insensitively, consistent with the in-memory and SQL backends.
* ``STARTS_WITH`` is a case-sensitive ``prefix`` query.
* Every negation returns a self-contained ``{"bool": {"must_not": …}}`` clause,
  so callers only ever wrap the returned clauses in ``bool``/``must``.
* An unsupported operator raises ``ValueError`` rather than silently matching
  every document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..query import Operator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..query import Filter
    from ..query_logic import Condition


# Operators that use the ``.keyword`` sub-field for exact matching on strings.
_KEYWORD_EQUALITY_OPS = frozenset(
    {Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN}
)
_KEYWORD_PATTERN_OPS = frozenset(
    {Operator.LIKE, Operator.NOT_LIKE, Operator.STARTS_WITH}
)


def _is_string_value(value: Any) -> bool:
    """Whether a value (or the first element of a list) is a string."""
    if isinstance(value, str):
        return True
    return bool(value) and isinstance(value, list) and isinstance(value[0], str)


def _field_path(filter_obj: Filter) -> str:
    """The document path an operator's clause should target for this filter.

    ``id`` resolves to the top-level ``id`` keyword; other fields to
    ``data.<field>`` with the ``.keyword`` suffix appended where exact matching
    on a string applies.
    """
    if filter_obj.field == "id":
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
    """Translate SQL ``LIKE`` wildcards to Elasticsearch wildcard syntax."""
    return pattern.replace("%", "*").replace("_", "?")


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
        return {"regexp": {field_path: value}}
    if op == Operator.STARTS_WITH:
        # Literal, case-sensitive prefix — no case_insensitive flag.
        return {"prefix": {field_path: value}}
    if op == Operator.BETWEEN:
        lower, upper = _bounds(value)
        return {"range": {field_path: {"gte": lower, "lte": upper}}}
    if op == Operator.NOT_BETWEEN:
        lower, upper = _bounds(value)
        return {"bool": {"must_not": {"range": {field_path: {"gte": lower, "lte": upper}}}}}

    raise ValueError(f"Unsupported operator: {op}")


def _bounds(value: Any) -> tuple[Any, Any]:
    """Unpack a two-element BETWEEN bound, raising on a malformed value."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(
        f"BETWEEN/NOT_BETWEEN requires a two-element bound, got: {value!r}"
    )


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
        clauses = [
            build_complex_es_query(sub) for sub in condition.conditions
        ]
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

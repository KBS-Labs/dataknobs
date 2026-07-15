"""Offline unit tests for the shared Elasticsearch filter-to-DSL translator.

These pin the exact Query-DSL emitted for every operator without a running
Elasticsearch — pure ``dict`` assertions that run in ordinary CI. The sync and
async backends and the vector-pre-filter mixin all route through the functions
under test, so this suite is the guard that keeps the three translation sites
from drifting in coverage or semantics.

Contract pinned here:

* ``Filter("id", …)`` targets the top-level ``id`` keyword (a full query
  target: term/terms/range/prefix/wildcard/regexp/exists), never ``_id`` or
  ``data.id``.
* Other string fields use the ``.keyword`` sub-field where exact matching
  applies (equality, membership, wildcard, prefix); numeric/range/exists/regex
  use the analyzed ``data.<field>`` path.
* ``LIKE``/``NOT_LIKE`` translate SQL wildcards (``%``→``*``, ``_``→``?``) and
  match case-insensitively, matching the in-memory and SQL backends.
* ``STARTS_WITH`` is a case-sensitive ``prefix`` (no ``case_insensitive`` flag).
* Every negation returns a self-contained ``{"bool": {"must_not": …}}`` clause.
* An unsupported operator raises rather than silently matching everything.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dataknobs_data import Filter, Operator
from dataknobs_data.backends.elasticsearch_query import (
    build_bool_query,
    build_complex_es_query,
    build_filter_es_query,
)
from dataknobs_data.query_logic import FilterCondition, LogicCondition, LogicOperator


# --- equality / inequality -------------------------------------------------

def test_eq_string_uses_keyword() -> None:
    assert build_filter_es_query(Filter("name", Operator.EQ, "alice")) == {
        "term": {"data.name.keyword": "alice"}
    }


def test_eq_int_uses_base_field() -> None:
    assert build_filter_es_query(Filter("age", Operator.EQ, 30)) == {
        "term": {"data.age": 30}
    }


def test_eq_bool_passes_native_value() -> None:
    # No str(v).lower() stringification — a native bool term matches the
    # dynamically-mapped ES boolean field.
    assert build_filter_es_query(Filter("active", Operator.EQ, True)) == {
        "term": {"data.active": True}
    }


def test_neq_wraps_in_must_not() -> None:
    assert build_filter_es_query(Filter("name", Operator.NEQ, "alice")) == {
        "bool": {"must_not": {"term": {"data.name.keyword": "alice"}}}
    }


# --- range -----------------------------------------------------------------

@pytest.mark.parametrize(
    ("op", "es_op"),
    [
        (Operator.GT, "gt"),
        (Operator.GTE, "gte"),
        (Operator.LT, "lt"),
        (Operator.LTE, "lte"),
    ],
)
def test_range_operators(op: Operator, es_op: str) -> None:
    assert build_filter_es_query(Filter("age", op, 21)) == {
        "range": {"data.age": {es_op: 21}}
    }


def test_between() -> None:
    assert build_filter_es_query(Filter("age", Operator.BETWEEN, [18, 65])) == {
        "range": {"data.age": {"gte": 18, "lte": 65}}
    }


def test_not_between_wraps_in_must_not() -> None:
    assert build_filter_es_query(Filter("age", Operator.NOT_BETWEEN, [18, 65])) == {
        "bool": {"must_not": {"range": {"data.age": {"gte": 18, "lte": 65}}}}
    }


# --- LIKE: SQL-wildcard AND case-insensitive -------------------------------

def test_like_translates_sql_wildcards_case_insensitive() -> None:
    # % -> *, _ -> ?, on the .keyword sub-field, case-insensitive — matching
    # the in-memory (re.IGNORECASE, anchored) and SQL LIKE backends. NOT the
    # async substring form (*val*), NOT case-sensitive.
    assert build_filter_es_query(Filter("f", Operator.LIKE, "a_b%")) == {
        "wildcard": {"data.f.keyword": {"value": "a?b*", "case_insensitive": True}}
    }


def test_not_like_wraps_case_insensitive_wildcard() -> None:
    assert build_filter_es_query(Filter("f", Operator.NOT_LIKE, "a_b%")) == {
        "bool": {
            "must_not": {
                "wildcard": {
                    "data.f.keyword": {"value": "a?b*", "case_insensitive": True}
                }
            }
        }
    }


# --- membership ------------------------------------------------------------

def test_in_string_list_uses_keyword() -> None:
    assert build_filter_es_query(Filter("status", Operator.IN, ["a", "b"])) == {
        "terms": {"data.status.keyword": ["a", "b"]}
    }


def test_in_int_list_uses_base_field() -> None:
    assert build_filter_es_query(Filter("n", Operator.IN, [1, 2])) == {
        "terms": {"data.n": [1, 2]}
    }


def test_not_in_wraps_in_must_not() -> None:
    assert build_filter_es_query(Filter("status", Operator.NOT_IN, ["a", "b"])) == {
        "bool": {"must_not": {"terms": {"data.status.keyword": ["a", "b"]}}}
    }


# --- existence -------------------------------------------------------------

def test_exists() -> None:
    assert build_filter_es_query(Filter("f", Operator.EXISTS)) == {
        "exists": {"field": "data.f"}
    }


def test_not_exists_wraps_in_must_not() -> None:
    assert build_filter_es_query(Filter("f", Operator.NOT_EXISTS)) == {
        "bool": {"must_not": {"exists": {"field": "data.f"}}}
    }


# --- regex / prefix --------------------------------------------------------

def test_regex_uses_base_field_case_sensitive() -> None:
    assert build_filter_es_query(Filter("f", Operator.REGEX, "a.*z")) == {
        "regexp": {"data.f": "a.*z"}
    }


def test_starts_with_is_case_sensitive_prefix() -> None:
    # prefix on the .keyword sub-field, NO case_insensitive flag.
    assert build_filter_es_query(Filter("f", Operator.STARTS_WITH, "pre")) == {
        "prefix": {"data.f.keyword": "pre"}
    }


# --- the id field is a full first-class query target -----------------------

@pytest.mark.parametrize(
    ("filter_obj", "expected"),
    [
        (Filter("id", Operator.EQ, "orders/1"), {"term": {"id": "orders/1"}}),
        (
            Filter("id", Operator.NEQ, "orders/1"),
            {"bool": {"must_not": {"term": {"id": "orders/1"}}}},
        ),
        (
            Filter("id", Operator.IN, ["orders/1", "orders/2"]),
            {"terms": {"id": ["orders/1", "orders/2"]}},
        ),
        (
            Filter("id", Operator.STARTS_WITH, "orders/"),
            {"prefix": {"id": "orders/"}},
        ),
        (
            Filter("id", Operator.LIKE, "orders/%"),
            {"wildcard": {"id": {"value": "orders/*", "case_insensitive": True}}},
        ),
        (
            Filter("id", Operator.GT, "orders/1"),
            {"range": {"id": {"gt": "orders/1"}}},
        ),
        (Filter("id", Operator.REGEX, "orders/.*"), {"regexp": {"id": "orders/.*"}}),
        (Filter("id", Operator.EXISTS), {"exists": {"field": "id"}}),
    ],
)
def test_id_targets_top_level_keyword(
    filter_obj: Filter, expected: dict
) -> None:
    # Never _id, never data.id — the top-level ``id`` keyword uniformly.
    assert build_filter_es_query(filter_obj) == expected


# --- outer bool wrapper ----------------------------------------------------

def test_bool_query_empty_is_match_all() -> None:
    assert build_bool_query([]) == {"match_all": {}}


def test_bool_query_wraps_clauses_in_must() -> None:
    filters = [Filter("a", Operator.EQ, "x"), Filter("age", Operator.GT, 1)]
    assert build_bool_query(filters) == {
        "bool": {
            "must": [
                {"term": {"data.a.keyword": "x"}},
                {"range": {"data.age": {"gt": 1}}},
            ]
        }
    }


# --- ComplexQuery condition tree -------------------------------------------

def test_complex_and_or_not_tree() -> None:
    condition = LogicCondition(
        operator=LogicOperator.AND,
        conditions=[
            FilterCondition(Filter("a", Operator.EQ, "x")),
            LogicCondition(
                operator=LogicOperator.OR,
                conditions=[
                    FilterCondition(Filter("age", Operator.GT, 1)),
                    FilterCondition(Filter("age", Operator.LT, 100)),
                ],
            ),
            LogicCondition(
                operator=LogicOperator.NOT,
                conditions=[FilterCondition(Filter("b", Operator.EQ, "y"))],
            ),
        ],
    )
    assert build_complex_es_query(condition) == {
        "bool": {
            "must": [
                {"term": {"data.a.keyword": "x"}},
                {
                    "bool": {
                        "should": [
                            {"range": {"data.age": {"gt": 1}}},
                            {"range": {"data.age": {"lt": 100}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                {"bool": {"must_not": {"term": {"data.b.keyword": "y"}}}},
            ]
        }
    }


def test_complex_single_filter_condition() -> None:
    condition = FilterCondition(Filter("id", Operator.STARTS_WITH, "orders/"))
    assert build_complex_es_query(condition) == {"prefix": {"id": "orders/"}}


# --- unsupported operator fails loud ---------------------------------------

def test_unsupported_operator_raises() -> None:
    # A translator that cannot express a filter must fail loud, not silently
    # fall back to match_all (which is how a dropped filter returns everything).
    bogus = SimpleNamespace(field="f", operator="not-an-operator", value=1)
    with pytest.raises(ValueError, match="Unsupported operator"):
        build_filter_es_query(bogus)  # type: ignore[arg-type]

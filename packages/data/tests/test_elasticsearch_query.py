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
* Other string fields use the ``.keyword`` sub-field wherever matching is
  against the full un-analyzed value (equality, membership, wildcard, prefix,
  regex); numeric range/exists use the analyzed ``data.<field>`` path.
* ``LIKE``/``NOT_LIKE`` translate SQL wildcards (``%``→``*``, ``_``→``?``),
  escape literal Lucene metacharacters (``*``/``?``/backslash), and match
  case-insensitively, matching the in-memory and SQL backends.
* ``REGEX`` targets the ``.keyword`` sub-field so the pattern matches the full
  value, not a single analyzed token.
* ``STARTS_WITH`` is a case-sensitive ``prefix`` (no ``case_insensitive`` flag).
* Every negation returns a self-contained ``{"bool": {"must_not": …}}`` clause.
* An unsupported operator raises rather than silently matching everything.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dataknobs_data import Filter, Operator
from dataknobs_data.backends.elasticsearch_query import (
    _field_path,
    _sql_wildcard_to_es,
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


@pytest.mark.parametrize(
    ("sql_pattern", "es_value"),
    [
        # Literal Lucene metacharacters in the SQL pattern must be escaped so
        # they match verbatim — only % and _ are wildcards in SQL LIKE.
        ("a*b", "a\\*b"),  # literal * -> \*  (not an ES wildcard)
        ("a?b", "a\\?b"),  # literal ? -> \?  (not an ES single-char wildcard)
        ("a\\b", "a\\\\b"),  # literal backslash -> \\
        # SQL wildcards still map through, and a mix escapes only the literals.
        ("a%_*", "a*?\\*"),  # % -> * , _ -> ? , literal * -> \*
    ],
)
def test_like_escapes_literal_lucene_metacharacters(
    sql_pattern: str, es_value: str
) -> None:
    # Reproduce-first for the escaping gap: without escaping, a literal '*' in
    # a SQL LIKE pattern became an ES wildcard matching anything.
    assert build_filter_es_query(Filter("f", Operator.LIKE, sql_pattern)) == {
        "wildcard": {"data.f.keyword": {"value": es_value, "case_insensitive": True}}
    }


def test_like_non_string_pattern_raises() -> None:
    # A non-string LIKE pattern fails loud with a clear message rather than an
    # opaque AttributeError from .replace on a non-str.
    with pytest.raises(ValueError, match="must be a string"):
        build_filter_es_query(Filter("f", Operator.LIKE, 123))  # type: ignore[arg-type]


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

def test_regex_uses_keyword_full_value() -> None:
    # REGEX targets the .keyword sub-field (full, un-analyzed value), NOT the
    # analyzed base path where a regexp would match a single lowercased token.
    assert build_filter_es_query(Filter("f", Operator.REGEX, "a.*z")) == {
        "regexp": {"data.f.keyword": "a.*z"}
    }


def test_regex_multi_token_pattern_targets_full_value() -> None:
    # Reproduce-first for the tokenization divergence: a pattern spanning a
    # space (``alice.*smith``) is meaningless against per-token analyzed text
    # but matches the full value ``"alice smith"`` on the keyword sub-field.
    # Pin that REGEX resolves to .keyword so cross-token patterns work like the
    # in-memory (re.search) and SQL backends, not the old per-token data.name.
    assert build_filter_es_query(Filter("name", Operator.REGEX, "alice.*smith")) == {
        "regexp": {"data.name.keyword": "alice.*smith"}
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


def test_complex_single_clause_and_collapses() -> None:
    # A one-condition AND collapses to the bare clause (no bool/must wrapper).
    condition = LogicCondition(
        operator=LogicOperator.AND,
        conditions=[FilterCondition(Filter("a", Operator.EQ, "x"))],
    )
    assert build_complex_es_query(condition) == {"term": {"data.a.keyword": "x"}}


def test_complex_single_clause_or_collapses() -> None:
    # A one-condition OR collapses too (no should/minimum_should_match wrapper).
    condition = LogicCondition(
        operator=LogicOperator.OR,
        conditions=[FilterCondition(Filter("a", Operator.EQ, "x"))],
    )
    assert build_complex_es_query(condition) == {"term": {"data.a.keyword": "x"}}


@pytest.mark.parametrize("logic_op", [LogicOperator.AND, LogicOperator.OR])
def test_complex_empty_branch_is_match_all(logic_op: LogicOperator) -> None:
    # An empty AND/OR branch is match_all — a no-constraint branch matches all.
    assert build_complex_es_query(
        LogicCondition(operator=logic_op, conditions=[])
    ) == {"match_all": {}}


def test_complex_empty_not_is_match_all() -> None:
    # A NOT with no inner condition has nothing to negate -> match_all.
    assert build_complex_es_query(
        LogicCondition(operator=LogicOperator.NOT, conditions=[])
    ) == {"match_all": {}}


# --- unsupported operator / malformed value fail loud ----------------------

def test_unsupported_operator_raises() -> None:
    # A translator that cannot express a filter must fail loud, not silently
    # fall back to match_all (which is how a dropped filter returns everything).
    bogus = SimpleNamespace(field="f", operator="not-an-operator", value=1)
    with pytest.raises(ValueError, match="Unsupported operator"):
        build_filter_es_query(bogus)  # type: ignore[arg-type]


@pytest.mark.parametrize("op", [Operator.BETWEEN, Operator.NOT_BETWEEN])
@pytest.mark.parametrize("bad_value", [[1], [1, 2, 3], "not-a-pair", 5])
def test_between_malformed_bounds_raise(op: Operator, bad_value: object) -> None:
    # BETWEEN/NOT_BETWEEN require exactly a two-element bound; anything else
    # fails loud rather than emitting a malformed range clause.
    with pytest.raises(ValueError, match="two-element bound"):
        build_filter_es_query(Filter("age", op, bad_value))


# --- helper functions in isolation -----------------------------------------
# The operators above pin these transitively; these pin them directly so a
# regression in the escaping or field-path logic points at the helper, not a
# downstream operator clause.

@pytest.mark.parametrize(
    ("sql_pattern", "es_value"),
    [
        ("abc", "abc"),  # no wildcards -> unchanged
        ("a%b", "a*b"),  # % -> *
        ("a_b", "a?b"),  # _ -> ?
        ("a%_b", "a*?b"),  # both SQL wildcards
        ("a*b", "a\\*b"),  # literal * escaped
        ("a?b", "a\\?b"),  # literal ? escaped
        ("a\\b", "a\\\\b"),  # literal backslash escaped
        ("*_%", "\\*?*"),  # literal * escaped; _ -> ? ; % -> *
        ("\\%", "\\\\*"),  # backslash escaped first, then % -> *
    ],
)
def test_sql_wildcard_to_es_translates_and_escapes(
    sql_pattern: str, es_value: str
) -> None:
    assert _sql_wildcard_to_es(sql_pattern) == es_value


@pytest.mark.parametrize("bad", [123, None, ["a"], 4.5])
def test_sql_wildcard_to_es_non_string_raises(bad: object) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        _sql_wildcard_to_es(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "op",
    [
        Operator.EQ,
        Operator.STARTS_WITH,
        Operator.REGEX,
        Operator.GT,
        Operator.IN,
        Operator.EXISTS,
    ],
)
def test_field_path_id_is_always_unsuffixed(op: Operator) -> None:
    # The id short-circuit runs before any operator check, so every operator
    # resolves id to the bare top-level keyword — never id.keyword, never _id.
    value = ["x"] if op in (Operator.IN,) else "x"
    assert _field_path(Filter("id", op, value)) == "id"


@pytest.mark.parametrize(
    "op",
    [Operator.LIKE, Operator.NOT_LIKE, Operator.STARTS_WITH, Operator.REGEX],
)
def test_field_path_pattern_ops_use_keyword_regardless_of_value(
    op: Operator,
) -> None:
    # Pattern operators always match the full un-analyzed value, so they target
    # .keyword unconditionally (patterns are strings anyway).
    assert _field_path(Filter("f", op, "p")) == "data.f.keyword"


def test_field_path_string_equality_uses_keyword() -> None:
    assert _field_path(Filter("f", Operator.EQ, "s")) == "data.f.keyword"
    assert _field_path(Filter("f", Operator.IN, ["a", "b"])) == "data.f.keyword"


@pytest.mark.parametrize(
    ("op", "value"),
    [
        (Operator.EQ, 30),  # non-string equality stays on the analyzed path
        (Operator.IN, [1, 2]),  # non-string membership too
        (Operator.GT, 5),  # range never uses .keyword
        (Operator.EXISTS, None),  # existence targets the base field
    ],
)
def test_field_path_non_string_and_range_use_base(
    op: Operator, value: object
) -> None:
    assert _field_path(Filter("f", op, value)) == "data.f"

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
* A field mapped as analyzed text is matched through its keyword sub-field by
  every operator but existence; any other field is matched directly. Which it is
  comes from the index mapping, as :class:`FieldTypes` has learned it.
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

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from dataknobs_data import Filter, Operator, Query
from dataknobs_data.backends.elasticsearch_query import (
    FieldTypes,
    _field_path,
    _sql_wildcard_to_es,
    build_bool_query,
    build_complex_es_query,
    build_filter_es_query,
    build_query,
    build_sort,
    query_fields,
)
from dataknobs_data.query import SortOrder, SortSpec
from dataknobs_data.query_logic import (
    ComplexQuery,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)


def _learned(**mapped: str) -> FieldTypes:
    """What a field-capabilities response over ``mapped`` teaches.

    ``text`` is what dynamic mapping makes of a string: with a ``.keyword``.
    """
    capabilities: dict[str, dict[str, Any]] = {"data": {"object": {}}}
    for name, kind in mapped.items():
        capabilities[f"data.{name}"] = {kind: {}}
        if kind == "text":
            capabilities[f"data.{name}.keyword"] = {"keyword": {}}
    types = FieldTypes()
    types.learn(mapped, capabilities)
    return types


TYPES = _learned(
    **dict.fromkeys(("name", "f", "status", "a", "b"), "text"),
    age="long",
    n="long",
    active="boolean",
)


def _has_value_but_not(path: str, clause: dict[str, Any]) -> dict[str, Any]:
    return {"bool": {"filter": {"exists": {"field": path}}, "must_not": clause}}


NOTHING = {"match_none": {}}


# --- equality / inequality -------------------------------------------------


def test_eq_string_uses_keyword() -> None:
    assert build_filter_es_query(Filter("name", Operator.EQ, "alice"), TYPES) == {
        "term": {"data.name.keyword": "alice"}
    }


def test_eq_int_uses_base_field() -> None:
    assert build_filter_es_query(Filter("age", Operator.EQ, 30), TYPES) == {
        "term": {"data.age": 30}
    }


def test_eq_bool_passes_native_value() -> None:
    # No str(v).lower() stringification — a native bool term matches the
    # dynamically-mapped ES boolean field.
    assert build_filter_es_query(Filter("active", Operator.EQ, True), TYPES) == {
        "term": {"data.active": True}
    }


def test_neq_wraps_in_must_not() -> None:
    assert build_filter_es_query(
        Filter("name", Operator.NEQ, "alice"), TYPES
    ) == _has_value_but_not("data.name", {"term": {"data.name.keyword": "alice"}})


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
    assert build_filter_es_query(Filter("age", op, 21), TYPES) == {
        "range": {"data.age": {es_op: 21}}
    }


def test_between() -> None:
    assert build_filter_es_query(Filter("age", Operator.BETWEEN, [18, 65]), TYPES) == {
        "range": {"data.age": {"gte": 18, "lte": 65}}
    }


def test_not_between_wraps_in_must_not() -> None:
    assert build_filter_es_query(
        Filter("age", Operator.NOT_BETWEEN, [18, 65]), TYPES
    ) == _has_value_but_not("data.age", {"range": {"data.age": {"gte": 18, "lte": 65}}})


# --- LIKE: SQL-wildcard AND case-insensitive -------------------------------


def test_like_translates_sql_wildcards_case_insensitive() -> None:
    # % -> *, _ -> ?, on the .keyword sub-field, case-insensitive — matching
    # the in-memory (re.IGNORECASE, anchored) and SQL LIKE backends. NOT the
    # async substring form (*val*), NOT case-sensitive.
    assert build_filter_es_query(Filter("f", Operator.LIKE, "a_b%"), TYPES) == {
        "wildcard": {"data.f.keyword": {"value": "a?b*", "case_insensitive": True}}
    }


def test_not_like_wraps_case_insensitive_wildcard() -> None:
    assert build_filter_es_query(
        Filter("f", Operator.NOT_LIKE, "a_b%"), TYPES
    ) == _has_value_but_not(
        "data.f", {"wildcard": {"data.f.keyword": {"value": "a?b*", "case_insensitive": True}}}
    )


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
def test_like_escapes_literal_lucene_metacharacters(sql_pattern: str, es_value: str) -> None:
    # Reproduce-first for the escaping gap: without escaping, a literal '*' in
    # a SQL LIKE pattern became an ES wildcard matching anything.
    assert build_filter_es_query(Filter("f", Operator.LIKE, sql_pattern), TYPES) == {
        "wildcard": {"data.f.keyword": {"value": es_value, "case_insensitive": True}}
    }


def test_like_non_string_pattern_raises() -> None:
    # A non-string LIKE pattern fails loud with a clear message rather than an
    # opaque AttributeError from .replace on a non-str.
    with pytest.raises(ValueError, match="must be a string"):
        build_filter_es_query(Filter("f", Operator.LIKE, 123), TYPES)  # type: ignore[arg-type]


# --- membership ------------------------------------------------------------


def test_in_string_list_uses_keyword() -> None:
    assert build_filter_es_query(Filter("status", Operator.IN, ["a", "b"]), TYPES) == {
        "terms": {"data.status.keyword": ["a", "b"]}
    }


def test_in_int_list_uses_base_field() -> None:
    assert build_filter_es_query(Filter("n", Operator.IN, [1, 2]), TYPES) == {
        "terms": {"data.n": [1, 2]}
    }


def test_not_in_wraps_in_must_not() -> None:
    assert build_filter_es_query(
        Filter("status", Operator.NOT_IN, ["a", "b"]), TYPES
    ) == _has_value_but_not("data.status", {"terms": {"data.status.keyword": ["a", "b"]}})


# --- existence -------------------------------------------------------------


def test_exists() -> None:
    assert build_filter_es_query(Filter("f", Operator.EXISTS), TYPES) == {
        "exists": {"field": "data.f"}
    }


def test_not_exists_wraps_in_must_not() -> None:
    assert build_filter_es_query(Filter("f", Operator.NOT_EXISTS), TYPES) == {
        "bool": {"must_not": {"exists": {"field": "data.f"}}}
    }


# --- regex / prefix --------------------------------------------------------


def test_regex_uses_keyword_full_value() -> None:
    # REGEX targets the .keyword sub-field (full, un-analyzed value), NOT the
    # analyzed base path where a regexp would match a single lowercased token.
    assert build_filter_es_query(Filter("f", Operator.REGEX, "a.*z"), TYPES) == {
        "regexp": {"data.f.keyword": "a.*z"}
    }


def test_regex_multi_token_pattern_targets_full_value() -> None:
    # Reproduce-first for the tokenization divergence: a pattern spanning a
    # space (``alice.*smith``) is meaningless against per-token analyzed text
    # but matches the full value ``"alice smith"`` on the keyword sub-field.
    # Pin that REGEX resolves to .keyword so cross-token patterns work like the
    # in-memory (re.search) and SQL backends, not the old per-token data.name.
    assert build_filter_es_query(Filter("name", Operator.REGEX, "alice.*smith"), TYPES) == {
        "regexp": {"data.name.keyword": "alice.*smith"}
    }


def test_starts_with_is_case_sensitive_prefix() -> None:
    # prefix on the .keyword sub-field, NO case_insensitive flag.
    assert build_filter_es_query(Filter("f", Operator.STARTS_WITH, "pre"), TYPES) == {
        "prefix": {"data.f.keyword": "pre"}
    }


# --- the id field is a full first-class query target -----------------------


@pytest.mark.parametrize(
    ("filter_obj", "expected"),
    [
        (Filter("id", Operator.EQ, "orders/1"), {"term": {"id": "orders/1"}}),
        (
            Filter("id", Operator.NEQ, "orders/1"),
            _has_value_but_not("id", {"term": {"id": "orders/1"}}),
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
def test_id_targets_top_level_keyword(filter_obj: Filter, expected: dict) -> None:
    # Never _id, never data.id — the top-level ``id`` keyword uniformly.
    assert build_filter_es_query(filter_obj, TYPES) == expected


# --- outer bool wrapper ----------------------------------------------------


def test_bool_query_empty_is_match_all() -> None:
    assert build_bool_query([], TYPES) == {"match_all": {}}


def test_bool_query_wraps_clauses_in_must() -> None:
    filters = [Filter("a", Operator.EQ, "x"), Filter("age", Operator.GT, 1)]
    assert build_bool_query(filters, TYPES) == {
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
    assert build_complex_es_query(condition, TYPES) == {
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
                {"bool": {"must_not": [{"term": {"data.b.keyword": "y"}}]}},
            ]
        }
    }


def test_complex_single_filter_condition() -> None:
    condition = FilterCondition(Filter("id", Operator.STARTS_WITH, "orders/"))
    assert build_complex_es_query(condition, TYPES) == {"prefix": {"id": "orders/"}}


def test_complex_single_clause_and_collapses() -> None:
    # A one-condition AND collapses to the bare clause (no bool/must wrapper).
    condition = LogicCondition(
        operator=LogicOperator.AND,
        conditions=[FilterCondition(Filter("a", Operator.EQ, "x"))],
    )
    assert build_complex_es_query(condition, TYPES) == {"term": {"data.a.keyword": "x"}}


def test_complex_single_clause_or_collapses() -> None:
    # A one-condition OR collapses too (no should/minimum_should_match wrapper).
    condition = LogicCondition(
        operator=LogicOperator.OR,
        conditions=[FilterCondition(Filter("a", Operator.EQ, "x"))],
    )
    assert build_complex_es_query(condition, TYPES) == {"term": {"data.a.keyword": "x"}}


@pytest.mark.parametrize(
    ("logic_op", "expected"),
    [
        (LogicOperator.AND, {"match_all": {}}),
        (LogicOperator.OR, {"match_none": {}}),
    ],
)
def test_complex_empty_branch_answers_as_matches_does(
    logic_op: LogicOperator, expected: dict[str, Any]
) -> None:
    # Every condition of an empty AND holds; none of an empty OR does, so it
    # matches nothing -- and constrains a parent AND rather than vanishing.
    assert (
        build_complex_es_query(LogicCondition(operator=logic_op, conditions=[]), TYPES) == expected
    )


def test_complex_not_negates_every_condition() -> None:
    # NOT over several conditions matches when none of them does.
    condition = LogicCondition(
        operator=LogicOperator.NOT,
        conditions=[
            FilterCondition(Filter("a", Operator.EQ, "x")),
            FilterCondition(Filter("b", Operator.EQ, "y")),
        ],
    )
    assert build_complex_es_query(condition, TYPES) == {
        "bool": {
            "must_not": [
                {"term": {"data.a.keyword": "x"}},
                {"term": {"data.b.keyword": "y"}},
            ]
        }
    }


def test_complex_empty_not_is_match_all() -> None:
    # A NOT with no inner condition has nothing to negate -> match_all.
    assert build_complex_es_query(
        LogicCondition(operator=LogicOperator.NOT, conditions=[]), TYPES
    ) == {"match_all": {}}


# --- unsupported operator / malformed value fail loud ----------------------


def test_unsupported_operator_raises() -> None:
    # A translator that cannot express a filter must fail loud, not silently
    # fall back to match_all (which is how a dropped filter returns everything).
    bogus = SimpleNamespace(field="f", operator="not-an-operator", value=1)
    with pytest.raises(ValueError, match="Unsupported operator"):
        build_filter_es_query(bogus, TYPES)  # type: ignore[arg-type]


@pytest.mark.parametrize("op", [Operator.BETWEEN, Operator.NOT_BETWEEN])
@pytest.mark.parametrize("bad_value", [[1], [1, 2, 3], "not-a-pair", 5])
def test_between_malformed_bounds_raise(op: Operator, bad_value: object) -> None:
    # BETWEEN/NOT_BETWEEN require exactly two bounds; anything else fails loud
    # rather than emitting a malformed range clause. The refusal is the
    # Filter's own, so no translator ever receives such a value.
    with pytest.raises(ValueError, match="needs two bounds"):
        build_filter_es_query(Filter("age", op, bad_value), TYPES)


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
def test_sql_wildcard_to_es_translates_and_escapes(sql_pattern: str, es_value: str) -> None:
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
    assert _field_path(Filter("id", op, value), TYPES) == "id"


ALL_OPERATORS = [
    (Operator.EQ, "s"),
    (Operator.EQ, 30),
    (Operator.IN, ["a", "b"]),
    (Operator.GT, "s"),
    (Operator.BETWEEN, ["a", "b"]),
    (Operator.LIKE, "p"),
    (Operator.STARTS_WITH, "p"),
    (Operator.REGEX, "p"),
]


@pytest.mark.parametrize(("op", "value"), ALL_OPERATORS)
def test_a_text_field_is_reached_through_its_keyword(op: Operator, value: object) -> None:
    assert _field_path(Filter("f", op, value), TYPES) == "data.f.keyword"


@pytest.mark.parametrize("kind", ["keyword", "wildcard", "flattened", "constant_keyword"])
@pytest.mark.parametrize(("op", "value"), ALL_OPERATORS)
def test_a_field_holding_whole_strings_is_reached_directly(
    op: Operator, value: object, kind: str
) -> None:
    assert _field_path(Filter("k", op, value), _learned(k=kind)) == "data.k"


@pytest.mark.parametrize("op", [Operator.EXISTS, Operator.NOT_EXISTS])
def test_existence_asks_the_field_itself(op: Operator) -> None:
    assert _field_path(Filter("f", op), TYPES) == "data.f"


@pytest.mark.parametrize("kind", ["long", "double", "date", "boolean"])
class TestAFieldThatHoldsNoStrings:
    @pytest.mark.parametrize(
        ("op", "value"), [(Operator.EQ, 5), (Operator.GT, 5), (Operator.GT, "5")]
    )
    def test_is_compared_directly(self, kind: str, op: Operator, value: object) -> None:
        assert _field_path(Filter("x", op, value), _learned(x=kind)) == "data.x"

    @pytest.mark.parametrize("op", [Operator.LIKE, Operator.STARTS_WITH, Operator.REGEX])
    def test_matches_no_pattern(self, kind: str, op: Operator) -> None:
        # Elasticsearch refuses these on the field itself.
        assert build_filter_es_query(Filter("x", op, "p"), _learned(x=kind)) == NOTHING


# --- a bound relates to its own kind ---------------------------------------


NUMBER_TYPES = [
    "byte",
    "short",
    "integer",
    "long",
    "unsigned_long",
    "half_float",
    "float",
    "double",
    "scaled_float",
]
NAN = float("nan")
NOON = datetime(2024, 1, 1, 12)


def _positives(bound: Any) -> list[Filter]:
    return [
        Filter("x", Operator.EQ, bound),
        Filter("x", Operator.IN, [bound]),
        Filter("x", Operator.GT, bound),
        Filter("x", Operator.GTE, bound),
        Filter("x", Operator.LT, bound),
        Filter("x", Operator.LTE, bound),
        Filter("x", Operator.BETWEEN, [bound, bound]),
    ]


def _negatives(bound: Any) -> list[Filter]:
    return [
        Filter("x", Operator.NEQ, bound),
        Filter("x", Operator.NOT_IN, [bound]),
        Filter("x", Operator.NOT_BETWEEN, [bound, bound]),
    ]


OTHER_KINDS = [
    *((kind, bound) for kind in NUMBER_TYPES for bound in ("3", "red", True, NOON)),
    *(("boolean", bound) for bound in ("true", 1, 0, NOON)),
    *((kind, bound) for kind in ("long", "boolean", "date", "text") for bound in (None, NAN)),
]


@pytest.mark.parametrize(("kind", "bound"), OTHER_KINDS)
def test_a_bound_that_relates_to_no_value_matches_none(kind: str, bound: Any) -> None:
    types = _learned(x=kind)
    for spec in _positives(bound):
        assert build_filter_es_query(spec, types) == NOTHING, spec
    for spec in _negatives(bound):
        assert build_filter_es_query(spec, types) == {"exists": {"field": "data.x"}}, spec


@pytest.mark.parametrize("bound", ["3", True, None, NAN])
def test_a_member_that_relates_to_no_value_is_dropped(bound: Any) -> None:
    for members in ([bound, 5], [5, bound]):
        assert build_filter_es_query(Filter("n", Operator.IN, members), TYPES) == {
            "terms": {"data.n": [5]}
        }


@pytest.mark.parametrize("members", [(5,), {5}, {5: 1}.keys()])
def test_any_collection_of_members_is_sent_as_a_list(members: Any) -> None:
    clause = {"terms": {"data.n": [5]}}
    assert build_filter_es_query(Filter("n", Operator.IN, members), TYPES) == clause
    assert build_filter_es_query(
        Filter("n", Operator.NOT_IN, members), TYPES
    ) == _has_value_but_not("data.n", clause)


def test_an_empty_list() -> None:
    assert build_filter_es_query(Filter("n", Operator.IN, []), TYPES) == NOTHING
    assert build_filter_es_query(Filter("n", Operator.NOT_IN, []), TYPES) == {
        "exists": {"field": "data.n"}
    }


@pytest.mark.parametrize("kind", ["date", "date_nanos", "ip", "version", "date_range"])
def test_a_string_is_how_other_values_are_written(kind: str) -> None:
    types = _learned(x=kind)
    assert build_filter_es_query(Filter("x", Operator.EQ, "s"), types) == {"term": {"data.x": "s"}}
    assert build_filter_es_query(Filter("x", Operator.IN, ["s"]), types) == {
        "terms": {"data.x": ["s"]}
    }
    assert build_filter_es_query(Filter("x", Operator.GT, "s"), types) == {
        "range": {"data.x": {"gt": "s"}}
    }


def test_a_range_mixing_a_string_and_a_number_matches_none() -> None:
    assert build_filter_es_query(Filter("f", Operator.BETWEEN, [0, "z"]), TYPES) == NOTHING


@pytest.mark.parametrize("bounds", [[5, NOON], [NOON, 5]])
def test_a_range_with_one_bound_of_another_kind_matches_none(bounds: list[Any]) -> None:
    assert build_filter_es_query(Filter("n", Operator.BETWEEN, bounds), TYPES) == NOTHING


def test_a_range_may_give_one_time_as_a_string() -> None:
    assert build_filter_es_query(
        Filter("x", Operator.BETWEEN, ["2024-01-01", NOON]), _learned(x="date")
    ) == {"range": {"data.x": {"gte": "2024-01-01", "lte": NOON}}}


@pytest.mark.parametrize("kind", ["text", "match_only_text"])
def test_no_string_equals_text_with_no_full_value(kind: str) -> None:
    types = FieldTypes()
    types.learn(["x"], {"data.x": {kind: {}}})
    assert build_filter_es_query(Filter("x", Operator.EQ, "s"), types) == NOTHING
    assert build_filter_es_query(Filter("x", Operator.IN, ["s"]), types) == NOTHING
    assert build_filter_es_query(Filter("x", Operator.NEQ, "s"), types) == {
        "exists": {"field": "data.x"}
    }
    # A range is over the words.
    assert build_filter_es_query(Filter("x", Operator.GT, "s"), types) == {
        "range": {"data.x": {"gt": "s"}}
    }


@pytest.mark.parametrize("kind", [*NUMBER_TYPES, "boolean"])
def test_a_negated_pattern_asks_for_a_string(kind: str) -> None:
    assert build_filter_es_query(Filter("x", Operator.NOT_LIKE, "p"), _learned(x=kind)) == NOTHING


def test_an_alias_over_indices_of_two_kinds_relates_to_either() -> None:
    types = FieldTypes()
    types.learn(["x"], {"data.x": {"long": {}, "boolean": {}}})
    for bound in (1, True):
        assert build_filter_es_query(Filter("x", Operator.EQ, bound), types) == {
            "term": {"data.x": bound}
        }
    assert build_filter_es_query(Filter("x", Operator.EQ, "1"), types) == NOTHING


def test_an_alias_over_indices_that_map_a_field_both_ways_asks_the_field() -> None:
    types = FieldTypes()
    types.learn(["x"], {"data.x": {"long": {}, "date": {}}})
    assert _field_path(Filter("x", Operator.EQ, "2024"), types) == "data.x"


@pytest.mark.parametrize(("op", "value"), ALL_OPERATORS)
def test_a_field_nothing_maps_is_reached_directly(op: Operator, value: object) -> None:
    assert _field_path(Filter("zz", op, value), TYPES) == "data.zz"


# --- learning the mapping --------------------------------------------------


def test_a_keyword_sub_field_of_any_name_serves() -> None:
    types = FieldTypes()
    types.learn(
        ["t"],
        {"data.t": {"text": {}}, "data.t.words": {"text": {}}, "data.t.raw": {"keyword": {}}},
    )
    assert types.value_path("t") == "data.t.raw"


def test_a_sub_field_called_keyword_is_preferred() -> None:
    types = FieldTypes()
    types.learn(
        ["t"],
        {"data.t": {"text": {}}, "data.t.a": {"keyword": {}}, "data.t.keyword": {"keyword": {}}},
    )
    assert types.value_path("t") == "data.t.keyword"


def test_a_field_whose_name_starts_anothers_is_not_its_sub_field() -> None:
    types = FieldTypes()
    types.learn(["t"], {"data.t": {"text": {}}, "data.tag": {"keyword": {}}})
    assert types.value_path("t") == "data.t"


def test_an_unmapped_field_is_asked_about_again() -> None:
    types = FieldTypes()
    types.learn(["a", "b", "c"], {"data.a": {"long": {}}, "data.c": {"unmapped": {}}})
    assert types.unknown(["id", "a", "b", "c"]) == ["b", "c"]


def test_a_query_names_the_fields_it_filters_and_sorts_by() -> None:
    plain = Query(filters=[Filter("a", Operator.EQ, 1)]).sort_by("b")
    assert query_fields(plain) == {"a", "b"}
    nested = LogicCondition(
        operator=LogicOperator.NOT,
        conditions=[
            LogicCondition(
                operator=LogicOperator.OR,
                conditions=[
                    FilterCondition(Filter("c", Operator.EQ, 1)),
                    FilterCondition(Filter("d", Operator.EQ, 1)),
                ],
            )
        ],
    )
    assert query_fields(ComplexQuery(condition=nested, sort_specs=[SortSpec("e")])) == {
        "c",
        "d",
        "e",
    }
    assert query_fields(None) == set()


def test_a_complex_query_with_no_condition_matches_everything() -> None:
    assert build_query(ComplexQuery(), TYPES) == {"match_all": {}}


def test_a_sort_reaches_each_fields_full_value() -> None:
    specs = [SortSpec("name", SortOrder.DESC), SortSpec("age"), SortSpec("id")]
    assert [
        (path, body["order"]) for c in build_sort(specs, TYPES) for path, body in c.items()
    ] == [
        ("data.name.keyword", "desc"),
        ("data.age", "asc"),
        ("id", "asc"),
    ]

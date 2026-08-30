"""One vocabulary for operators and sort orders, on every path that takes one.

``Query.filter`` mapped an unrecognized operator string to equality and
``Query.sort_by`` mapped an unrecognized order string to descending, neither
raising --- so a typo inverted a query instead of failing it. The sharpest
case needs no typo at all: ``Operator.NOT_LIKE`` is a member of the enum and
was absent from the fluent path's private alias table, so
``filter("name", "not_like", "A%")`` meant ``name = "A%"``.

The same class also rejected those strings on its *other* path:
``Filter.from_dict`` builds ``Operator(...)`` directly and raises. So the
deserialisation path was strict and the fluent path was silent, for one
vocabulary --- which is what makes a shared coercion the fix rather than a
second table.
"""

from __future__ import annotations

import pytest

from dataknobs_data.query import (
    Filter,
    Operator,
    Query,
    SortOrder,
    SortSpec,
    coerce_operator,
    coerce_sort_order,
)
from dataknobs_data.query_logic import QueryBuilder


class TestAnOperatorTheEnumHasIsReachableFromTheFluentPath:
    """The drift that produced the bug, guarded at its source."""

    @pytest.mark.parametrize("member", list(Operator), ids=lambda m: m.name)
    def test_every_member_is_reachable_by_its_own_value(self, member: Operator) -> None:
        query = Query().filter("field", member.value, "x")

        assert query.filters[0].operator is member

    @pytest.mark.parametrize("member", list(Operator), ids=lambda m: m.name)
    def test_every_member_is_accepted_as_itself(self, member: Operator) -> None:
        query = Query().filter("field", member, "x")

        assert query.filters[0].operator is member

    def test_not_like_does_not_silently_mean_equals(self) -> None:
        """The member the private alias table forgot.

        No typo, no unusual spelling --- ``not_like`` is what
        ``Operator.NOT_LIKE.value`` is, and it produced an equality filter.
        """
        query = Query().filter("name", "not_like", "A%")

        assert query.filters[0].operator is Operator.NOT_LIKE

    @pytest.mark.parametrize("member", list(Operator), ids=lambda m: m.name)
    def test_the_fluent_path_and_from_dict_agree(self, member: Operator) -> None:
        """One vocabulary, not two, for the two ways to name an operator."""
        fluent = Query().filter("field", member.value, "x").filters[0]
        deserialized = Filter.from_dict({"field": "field", "operator": member.value, "value": "x"})

        assert fluent.operator is deserialized.operator is member


class TestTheQueryBuilderReadsTheSameVocabulary:
    """``QueryBuilder.where`` was the third construction of an operator.

    It built ``Operator(operator)`` directly, which accepts only an exact
    member value --- so every alias the fluent builder derives by
    normalisation raised there instead, for the same argument, in the same
    package. That two paths disagreed is the whole reason this coercion
    exists rather than a second table; a third path had to join them.
    """

    @pytest.mark.parametrize("member", list(Operator), ids=lambda m: m.name)
    def test_every_member_is_reachable(self, member: Operator) -> None:
        builder = QueryBuilder().where("field", member.value, "x")

        assert builder.root_condition.filter.operator is member  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("==", Operator.EQ),
            ("IN", Operator.IN),
            ("NOT IN", Operator.NOT_IN),
            ("NOT BETWEEN", Operator.NOT_BETWEEN),
            ("STARTS_WITH", Operator.STARTS_WITH),
        ],
    )
    def test_the_aliases_the_fluent_path_takes(self, spelling: str, expected: Operator) -> None:
        builder = QueryBuilder().where("field", spelling, "x")

        assert builder.root_condition.filter.operator is expected  # type: ignore[union-attr]

    def test_it_agrees_with_query_filter(self) -> None:
        fluent = Query().filter("f", "==", 1).filters[0]
        built = QueryBuilder().where("f", "==", 1).root_condition

        assert built.filter.operator is fluent.operator  # type: ignore[union-attr]

    def test_an_unknown_spelling_is_refused_here_too(self) -> None:
        with pytest.raises(ValueError, match="operator"):
            QueryBuilder().where("colour", "contains", "blue")


class TestAnUnknownSpellingIsRefused:
    @pytest.mark.parametrize(
        "spelling",
        [
            "ne",  # the mongo/elasticsearch spelling of !=
            "eq",  # ...and of =
            "gt",
            "contains",
            "",
            "  ",
        ],
    )
    def test_an_unknown_operator_raises(self, spelling: str) -> None:
        with pytest.raises(ValueError, match="operator"):
            Query().filter("colour", spelling, "blue")

    @pytest.mark.parametrize("spelling", ["ascending", "descending", "up", "", "1"])
    def test_an_unknown_sort_order_raises(self, spelling: str) -> None:
        with pytest.raises(ValueError, match="sort order"):
            Query().sort_by("score", spelling)

    def test_the_message_names_what_would_have_worked(self) -> None:
        """A refusal a caller cannot act on is barely better than silence."""
        with pytest.raises(ValueError) as excinfo:
            Query().filter("colour", "ne", "blue")

        assert "!=" in str(excinfo.value)


class TestTheSpellingsThatAlwaysWorkedStillDo:
    """The aliases the private table carried, kept rather than dropped."""

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("=", Operator.EQ),
            ("==", Operator.EQ),
            ("!=", Operator.NEQ),
            (">", Operator.GT),
            (">=", Operator.GTE),
            ("<", Operator.LT),
            ("<=", Operator.LTE),
            ("in", Operator.IN),
            ("IN", Operator.IN),
            ("not_in", Operator.NOT_IN),
            ("NOT IN", Operator.NOT_IN),
            ("like", Operator.LIKE),
            ("LIKE", Operator.LIKE),
            ("regex", Operator.REGEX),
            ("starts_with", Operator.STARTS_WITH),
            ("STARTS_WITH", Operator.STARTS_WITH),
            ("exists", Operator.EXISTS),
            ("not_exists", Operator.NOT_EXISTS),
            ("between", Operator.BETWEEN),
            ("BETWEEN", Operator.BETWEEN),
            ("not_between", Operator.NOT_BETWEEN),
            ("NOT BETWEEN", Operator.NOT_BETWEEN),
        ],
    )
    def test_alias(self, spelling: str, expected: Operator) -> None:
        assert Query().filter("f", spelling, 1).filters[0].operator is expected

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("asc", SortOrder.ASC),
            ("ASC", SortOrder.ASC),
            ("desc", SortOrder.DESC),
            ("DESC", SortOrder.DESC),
        ],
    )
    def test_sort_alias(self, spelling: str, expected: SortOrder) -> None:
        assert Query().sort_by("score", spelling).sort_specs[0].order is expected

    def test_sort_by_still_defaults_to_ascending(self) -> None:
        assert Query().sort_by("score").sort_specs[0].order is SortOrder.ASC


class TestTheCoercionIsOneFunctionPerVocabulary:
    """Exported, because a backend or a consumer building a Filter by hand
    needs the same reading of a string that the fluent path gets.
    """

    def test_an_operator_passes_through(self) -> None:
        assert coerce_operator(Operator.GTE) is Operator.GTE

    def test_a_sort_order_passes_through(self) -> None:
        assert coerce_sort_order(SortOrder.DESC) is SortOrder.DESC

    def test_a_non_string_operator_is_refused(self) -> None:
        with pytest.raises(ValueError, match="operator"):
            coerce_operator(7)  # type: ignore[arg-type]

    def test_a_non_string_sort_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sort order"):
            coerce_sort_order(None)  # type: ignore[arg-type]

    def test_sort_spec_from_dict_shares_it(self) -> None:
        assert SortSpec.from_dict({"field": "s", "order": "DESC"}).order is SortOrder.DESC

    def test_sort_spec_from_dict_refuses_an_unknown_order(self) -> None:
        with pytest.raises(ValueError, match="sort order"):
            SortSpec.from_dict({"field": "s", "order": "ascending"})

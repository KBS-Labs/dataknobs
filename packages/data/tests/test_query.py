"""Tests for Query system."""

from dataknobs_data import Filter, Operator, Query, SortOrder, SortSpec


class TestFilter:
    """Test Filter class."""

    def test_filter_creation(self):
        """Test creating filters."""
        filter = Filter(field="age", operator=Operator.GT, value=25)
        assert filter.field == "age"
        assert filter.operator == Operator.GT
        assert filter.value == 25

    def test_filter_matches(self):
        """Test filter matching logic."""
        # Equality
        filter = Filter("name", Operator.EQ, "John")
        assert filter.matches("John") is True
        assert filter.matches("Jane") is False
        assert filter.matches(None) is False

        # Inequality
        filter = Filter("age", Operator.NEQ, 30)
        assert filter.matches(25) is True
        assert filter.matches(30) is False

        # Greater than
        filter = Filter("score", Operator.GT, 50)
        assert filter.matches(60) is True
        assert filter.matches(50) is False
        assert filter.matches(40) is False

        # Greater than or equal
        filter = Filter("score", Operator.GTE, 50)
        assert filter.matches(60) is True
        assert filter.matches(50) is True
        assert filter.matches(40) is False

        # Less than
        filter = Filter("score", Operator.LT, 50)
        assert filter.matches(40) is True
        assert filter.matches(50) is False
        assert filter.matches(60) is False

        # Less than or equal
        filter = Filter("score", Operator.LTE, 50)
        assert filter.matches(40) is True
        assert filter.matches(50) is True
        assert filter.matches(60) is False

        # In list
        filter = Filter("status", Operator.IN, ["active", "pending"])
        assert filter.matches("active") is True
        assert filter.matches("pending") is True
        assert filter.matches("inactive") is False

        # Not in list
        filter = Filter("status", Operator.NOT_IN, ["deleted", "archived"])
        assert filter.matches("active") is True
        assert filter.matches("deleted") is False

        # Like pattern
        filter = Filter("name", Operator.LIKE, "John%")
        assert filter.matches("John") is True
        assert filter.matches("Johnny") is True
        assert filter.matches("Jane") is False
        assert filter.matches(123) is False  # Non-string

        filter = Filter("code", Operator.LIKE, "A_C")
        assert filter.matches("ABC") is True
        assert filter.matches("AXC") is True
        assert filter.matches("AC") is False

        # Regex
        filter = Filter("email", Operator.REGEX, r".*@example\.com$")
        assert filter.matches("john@example.com") is True
        assert filter.matches("test@example.com") is True
        assert filter.matches("user@other.com") is False
        assert filter.matches(123) is False  # Non-string

        # Exists
        filter = Filter("field", Operator.EXISTS)
        assert filter.matches("value") is True
        assert filter.matches(0) is True
        assert filter.matches(None) is False

        # Not exists
        filter = Filter("field", Operator.NOT_EXISTS)
        assert filter.matches(None) is True
        assert filter.matches("value") is False

    def test_filter_to_from_dict(self):
        """Test filter serialization."""
        filter = Filter("age", Operator.GT, 25)

        dict_repr = filter.to_dict()
        assert dict_repr == {"field": "age", "operator": ">", "value": 25}

        restored = Filter.from_dict(dict_repr)
        assert restored.field == filter.field
        assert restored.operator == filter.operator
        assert restored.value == filter.value


class TestSortSpec:
    """Test SortSpec class."""

    def test_sort_spec_creation(self):
        """Test creating sort specifications."""
        sort = SortSpec("name", SortOrder.ASC)
        assert sort.field == "name"
        assert sort.order == SortOrder.ASC

        sort_default = SortSpec("age")
        assert sort_default.order == SortOrder.ASC

    def test_sort_spec_to_from_dict(self):
        """Test sort spec serialization."""
        sort = SortSpec("score", SortOrder.DESC)

        dict_repr = sort.to_dict()
        assert dict_repr == {"field": "score", "order": "desc"}

        restored = SortSpec.from_dict(dict_repr)
        assert restored.field == sort.field
        assert restored.order == sort.order


class TestQuery:
    """Test Query class."""

    def test_query_creation(self):
        """Test creating a query."""
        query = Query()
        assert query.filters == []
        assert query.sort_specs == []
        assert query.limit_value is None
        assert query.offset_value is None
        assert query.fields is None

    def test_query_fluent_interface(self):
        """Test fluent interface for building queries."""
        query = (
            Query()
            .filter("age", ">=", 25)
            .filter("status", "in", ["active", "pending"])
            .sort_by("age", "desc")
            .sort_by("name", "asc")
            .set_limit(10)
            .set_offset(20)
            .select("name", "age", "email")
        )

        assert len(query.filters) == 2
        assert query.filters[0].field == "age"
        assert query.filters[0].operator == Operator.GTE
        assert query.filters[1].field == "status"
        assert query.filters[1].operator == Operator.IN

        assert len(query.sort_specs) == 2
        assert query.sort_specs[0].field == "age"
        assert query.sort_specs[0].order == SortOrder.DESC
        assert query.sort_specs[1].field == "name"
        assert query.sort_specs[1].order == SortOrder.ASC

        assert query.limit_value == 10
        assert query.offset_value == 20
        assert query.fields == ["name", "age", "email"]

    def test_query_filter_operator_mapping(self):
        """Test string operator mapping in filter method."""
        query = Query()

        # Test various operator strings
        operators = [
            ("=", Operator.EQ),
            ("==", Operator.EQ),
            ("!=", Operator.NEQ),
            (">", Operator.GT),
            (">=", Operator.GTE),
            ("<", Operator.LT),
            ("<=", Operator.LTE),
            ("in", Operator.IN),
            ("not_in", Operator.NOT_IN),
            ("like", Operator.LIKE),
            ("LIKE", Operator.LIKE),
            ("regex", Operator.REGEX),
            ("exists", Operator.EXISTS),
            ("not_exists", Operator.NOT_EXISTS),
        ]

        for str_op, enum_op in operators:
            query.clear_filters()
            query.filter("field", str_op, "value")
            assert query.filters[0].operator == enum_op

    def test_query_clear_methods(self):
        """Test clearing filters and sorts."""
        query = Query().filter("a", "=", 1).filter("b", "=", 2).sort_by("c").sort_by("d")

        assert len(query.filters) == 2
        assert len(query.sort_specs) == 2

        query.clear_filters()
        assert len(query.filters) == 0
        assert len(query.sort_specs) == 2

        query.clear_sort()
        assert len(query.sort_specs) == 0

    def test_query_to_from_dict(self):
        """Test query serialization."""
        query = (
            Query()
            .filter("age", ">=", 25)
            .sort_by("name", "asc")
            .set_limit(10)
            .set_offset(5)
            .select("name", "age")
        )

        dict_repr = query.to_dict()
        assert "filters" in dict_repr
        assert "sort" in dict_repr
        assert dict_repr["limit"] == 10
        assert dict_repr["offset"] == 5
        assert dict_repr["fields"] == ["name", "age"]

        restored = Query.from_dict(dict_repr)
        assert len(restored.filters) == 1
        assert restored.filters[0].field == "age"
        assert len(restored.sort_specs) == 1
        assert restored.sort_specs[0].field == "name"
        assert restored.limit_value == 10
        assert restored.offset_value == 5
        assert restored.fields == ["name", "age"]

    def test_query_copy(self):
        """Test copying a query."""
        original = Query().filter("age", ">", 25).sort_by("name").set_limit(10)

        copy = original.copy()

        # Modify copy
        copy.filter("status", "=", "active")
        copy.set_limit(20)

        # Original should be unchanged
        assert len(original.filters) == 1
        assert original.limit_value == 10

        # Copy should have changes
        assert len(copy.filters) == 2
        assert copy.limit_value == 20

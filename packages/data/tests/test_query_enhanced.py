"""Enhanced tests for Query system covering edge cases and recent fixes."""

import pytest
from dataknobs_data import Filter, Operator, Query, SortOrder, SortSpec


class TestQueryOperatorMapping:
    """Test operator string mapping including uppercase variants."""
    
    def test_uppercase_operators(self):
        """Test that uppercase operator strings are correctly mapped."""
        query = Query()
        
        # Test uppercase IN
        query.filter("name", "IN", ["Alice", "Bob"])
        assert query.filters[0].operator == Operator.IN
        assert query.filters[0].value == ["Alice", "Bob"]
        
        # Test uppercase NOT IN
        query.clear_filters()
        query.filter("status", "NOT IN", ["deleted", "archived"])
        assert query.filters[0].operator == Operator.NOT_IN
        
        # Test uppercase LIKE (already in original tests but let's be explicit)
        query.clear_filters()
        query.filter("name", "LIKE", "C%")
        assert query.filters[0].operator == Operator.LIKE
        assert query.filters[0].value == "C%"
    
    def test_mixed_case_operators(self):
        """Test mixed case operator strings default to EQ."""
        query = Query()
        
        # Unknown operator should default to EQ
        query.filter("field", "Unknown", "value")
        assert query.filters[0].operator == Operator.EQ
        
        # Mixed case not in mapping defaults to EQ
        query.clear_filters()
        query.filter("field", "In", "value")  # lowercase 'n'
        assert query.filters[0].operator == Operator.EQ
    
    def test_all_operator_mappings(self):
        """Comprehensive test of all operator mappings."""
        mappings = [
            # Equality
            ("=", Operator.EQ),
            ("==", Operator.EQ),
            
            # Inequality
            ("!=", Operator.NEQ),
            
            # Comparison
            (">", Operator.GT),
            (">=", Operator.GTE),
            ("<", Operator.LT),
            ("<=", Operator.LTE),
            
            # Membership
            ("in", Operator.IN),
            ("IN", Operator.IN),
            ("not_in", Operator.NOT_IN),
            ("NOT IN", Operator.NOT_IN),
            
            # Pattern matching
            ("like", Operator.LIKE),
            ("LIKE", Operator.LIKE),
            ("regex", Operator.REGEX),
            
            # Existence
            ("exists", Operator.EXISTS),
            ("not_exists", Operator.NOT_EXISTS),
        ]
        
        for str_op, expected_op in mappings:
            query = Query().filter("field", str_op, "value")
            assert query.filters[0].operator == expected_op, f"Failed for operator '{str_op}'"


class TestQueryFluentInterface:
    """Test the fluent interface methods."""
    
    def test_method_aliases(self):
        """Test that method aliases work correctly."""
        # Test sort() as alias for sort_by()
        q1 = Query().sort("name", "DESC")
        assert len(q1.sort_specs) == 1
        assert q1.sort_specs[0].field == "name"
        assert q1.sort_specs[0].order == SortOrder.DESC
        
        # Test limit() as alias for set_limit()
        q2 = Query().limit(10)
        assert q2.limit_value == 10
        
        # Test offset() as alias for set_offset()
        q3 = Query().offset(20)
        assert q3.offset_value == 20
    
    def test_method_chaining(self):
        """Test complex method chaining."""
        query = (Query()
                .filter("age", ">", 18)
                .filter("age", "<", 65)
                .filter("status", "IN", ["active", "pending"])
                .sort("age", "ASC")
                .sort("name", "DESC")
                .limit(50)
                .offset(100)
                .select("id", "name", "age", "status"))
        
        assert len(query.filters) == 3
        assert len(query.sort_specs) == 2
        assert query.limit_value == 50
        assert query.offset_value == 100
        assert query.fields == ["id", "name", "age", "status"]
    
    def test_property_access(self):
        """Test backward compatibility properties."""
        query = Query()
        query.set_limit(10)
        query.set_offset(20)
        query.sort_by("name")
        
        # Test properties
        assert query.limit_property == 10  # Property access
        assert query.offset_property == 20  # Property access
        assert query.sort_property == query.sort_specs  # Property access
        
        # Also verify direct attribute access still works
        assert query.limit_value == 10
        assert query.offset_value == 20


class TestQueryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_query(self):
        """Test behavior with empty query."""
        query = Query()
        
        # Empty query should have sensible defaults
        assert query.filters == []
        assert query.sort_specs == []
        assert query.limit_value is None
        assert query.offset_value is None
        assert query.fields is None
        
        # to_dict should handle empty query
        dict_repr = query.to_dict()
        assert dict_repr["filters"] == []
        assert dict_repr["sort"] == []
        assert "limit" not in dict_repr
        assert "offset" not in dict_repr
        assert "fields" not in dict_repr
    
    def test_none_values(self):
        """Test handling of None values."""
        # Filter with None value
        query = Query().filter("field", "=", None)
        assert query.filters[0].value is None
        
        # Select with no fields
        query2 = Query().select()
        assert query2.fields is None
    
    def test_special_characters_in_patterns(self):
        """Test LIKE patterns with special characters."""
        query = Query()
        
        # Pattern with multiple wildcards
        query.filter("path", "LIKE", "%/docs/%/*.md")
        assert query.filters[0].value == "%/docs/%/*.md"
        
        # Pattern with underscore
        query.filter("code", "LIKE", "US_R_%")
        assert query.filters[1].value == "US_R_%"
    
    def test_large_in_list(self):
        """Test IN operator with large list."""
        large_list = list(range(1000))
        query = Query().filter("id", "IN", large_list)
        
        assert query.filters[0].operator == Operator.IN
        assert query.filters[0].value == large_list
        assert len(query.filters[0].value) == 1000
    
    def test_complex_sorting(self):
        """Test multiple sort fields."""
        query = (Query()
                .sort("priority", "DESC")
                .sort("created_at", "ASC")
                .sort("name", "ASC"))
        
        assert len(query.sort_specs) == 3
        assert query.sort_specs[0].field == "priority"
        assert query.sort_specs[0].order == SortOrder.DESC
        assert query.sort_specs[1].field == "created_at"
        assert query.sort_specs[1].order == SortOrder.ASC
        assert query.sort_specs[2].field == "name"
        assert query.sort_specs[2].order == SortOrder.ASC


class TestQuerySerialization:
    """Test query serialization and deserialization."""
    
    def test_complex_query_roundtrip(self):
        """Test serializing and deserializing a complex query."""
        original = (Query()
                   .filter("age", ">=", 18)
                   .filter("age", "<=", 65)
                   .filter("department", "IN", ["Engineering", "Sales"])
                   .filter("name", "LIKE", "John%")
                   .sort("department", "ASC")
                   .sort("salary", "DESC")
                   .limit(100)
                   .offset(200)
                   .select("id", "name", "department", "salary"))
        
        # Serialize
        dict_repr = original.to_dict()
        
        # Deserialize
        restored = Query.from_dict(dict_repr)
        
        # Verify all fields are restored correctly
        assert len(restored.filters) == 4
        assert restored.filters[0].field == "age"
        assert restored.filters[0].operator == Operator.GTE
        assert restored.filters[0].value == 18
        
        assert restored.filters[3].field == "name"
        assert restored.filters[3].operator == Operator.LIKE
        assert restored.filters[3].value == "John%"
        
        assert len(restored.sort_specs) == 2
        assert restored.sort_specs[0].field == "department"
        assert restored.sort_specs[1].field == "salary"
        
        assert restored.limit_value == 100
        assert restored.offset_value == 200
        assert restored.fields == ["id", "name", "department", "salary"]
    
    def test_partial_deserialization(self):
        """Test deserializing queries with missing fields."""
        # Minimal dict
        minimal = {"filters": [], "sort": []}
        query = Query.from_dict(minimal)
        assert query.filters == []
        assert query.sort_specs == []
        assert query.limit_value is None
        
        # Dict with only filters
        filters_only = {
            "filters": [{"field": "age", "operator": ">", "value": 25}]
        }
        query2 = Query.from_dict(filters_only)
        assert len(query2.filters) == 1
        assert query2.sort_specs == []


class TestFilterMatching:
    """Test the filter matching logic."""
    
    def test_like_pattern_matching(self):
        """Test LIKE pattern matching with various patterns."""
        # Start with pattern
        filter1 = Filter("name", Operator.LIKE, "John%")
        assert filter1.matches("John") is True
        assert filter1.matches("Johnny") is True
        assert filter1.matches("John Smith") is True
        assert filter1.matches("Jim") is False
        
        # End with pattern
        filter2 = Filter("email", Operator.LIKE, "%@example.com")
        assert filter2.matches("user@example.com") is True
        assert filter2.matches("admin@example.com") is True
        assert filter2.matches("user@other.com") is False
        
        # Contains pattern
        filter3 = Filter("description", Operator.LIKE, "%important%")
        assert filter3.matches("This is important") is True
        assert filter3.matches("important note") is True
        assert filter3.matches("very important message") is True
        assert filter3.matches("not relevant") is False
        
        # Single character wildcard
        filter4 = Filter("code", Operator.LIKE, "A_C")
        assert filter4.matches("ABC") is True
        assert filter4.matches("A1C") is True
        assert filter4.matches("AC") is False
        assert filter4.matches("ABBC") is False
    
    def test_in_operator_with_different_types(self):
        """Test IN operator with different value types."""
        # String list
        filter1 = Filter("status", Operator.IN, ["active", "pending"])
        assert filter1.matches("active") is True
        assert filter1.matches("inactive") is False
        
        # Number list
        filter2 = Filter("age", Operator.IN, [25, 30, 35])
        assert filter2.matches(25) is True
        assert filter2.matches(30) is True
        assert filter2.matches(28) is False
        
        # Mixed type list
        filter3 = Filter("value", Operator.IN, [1, "one", True])
        assert filter3.matches(1) is True
        assert filter3.matches("one") is True
        assert filter3.matches(True) is True
        assert filter3.matches("two") is False
    
    def test_comparison_with_nulls(self):
        """Test comparison operators with null values."""
        filter_gt = Filter("age", Operator.GT, 25)
        assert filter_gt.matches(None) is False
        assert filter_gt.matches(30) is True
        
        filter_lt = Filter("score", Operator.LT, 100)
        assert filter_lt.matches(None) is False
        assert filter_lt.matches(50) is True


class TestQueryIntegration:
    """Integration tests for Query with actual use cases."""
    
    def test_user_search_query(self):
        """Test a typical user search query."""
        # Simulate searching for active adult users in specific departments
        query = (Query()
                .filter("age", ">=", 18)
                .filter("status", "=", "active")
                .filter("department", "IN", ["Engineering", "Product", "Design"])
                .filter("email", "LIKE", "%@company.com")
                .sort("department", "ASC")
                .sort("joined_date", "DESC")
                .limit(20)
                .offset(0))
        
        # Verify the query is built correctly
        assert len(query.filters) == 4
        assert query.filters[2].operator == Operator.IN
        assert query.filters[3].operator == Operator.LIKE
        assert query.limit_value == 20
    
    def test_pagination_query(self):
        """Test building pagination queries."""
        page_size = 25
        
        # Page 1
        page1 = Query().limit(page_size).offset(0)
        assert page1.limit_value == 25
        assert page1.offset_value == 0
        
        # Page 2
        page2 = Query().limit(page_size).offset(page_size)
        assert page2.limit_value == 25
        assert page2.offset_value == 25
        
        # Page 3
        page3 = Query().limit(page_size).offset(page_size * 2)
        assert page3.limit_value == 25
        assert page3.offset_value == 50
    
    def test_dynamic_query_building(self):
        """Test building queries dynamically based on conditions."""
        def build_search_query(filters=None, sort_field=None, page=1, page_size=10):
            query = Query()
            
            if filters:
                for field, op, value in filters:
                    query.filter(field, op, value)
            
            if sort_field:
                query.sort(sort_field, "ASC")
            
            # Pagination
            query.limit(page_size).offset((page - 1) * page_size)
            
            return query
        
        # Test with various parameters
        query1 = build_search_query(
            filters=[("status", "=", "active"), ("age", ">", 25)],
            sort_field="name",
            page=2,
            page_size=15
        )
        
        assert len(query1.filters) == 2
        assert len(query1.sort_specs) == 1
        assert query1.limit_value == 15
        assert query1.offset_value == 15  # page 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
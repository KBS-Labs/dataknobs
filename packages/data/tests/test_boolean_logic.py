"""Tests for boolean logic operators in queries."""

import pytest
import pytest_asyncio
from dataknobs_data import (
    Record,
    Query,
    Filter,
    Operator,
    ComplexQuery,
    QueryBuilder,
    LogicOperator,
    FilterCondition,
    LogicCondition,
)
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase


class TestBooleanLogicQueries:
    """Test boolean logic operations in queries."""
    
    @pytest.fixture
    def sample_records(self):
        """Create sample records for testing."""
        return [
            Record(id="1", data={"name": "Alice", "age": 30, "city": "New York", "active": True}),
            Record(id="2", data={"name": "Bob", "age": 25, "city": "Los Angeles", "active": False}),
            Record(id="3", data={"name": "Charlie", "age": 35, "city": "New York", "active": True}),
            Record(id="4", data={"name": "David", "age": 28, "city": "Chicago", "active": True}),
            Record(id="5", data={"name": "Eve", "age": 32, "city": "Los Angeles", "active": False}),
        ]
    
    @pytest.fixture
    def db_with_data(self, sample_records):
        """Create a database with sample data."""
        db = SyncMemoryDatabase()
        for record in sample_records:
            db.create(record)
        return db
    
    def test_simple_or_query(self, db_with_data):
        """Test simple OR query."""
        # Find records where city is New York OR Los Angeles
        query = Query().or_(
            Filter("city", Operator.EQ, "New York"),
            Filter("city", Operator.EQ, "Los Angeles")
        )
        
        results = db_with_data.search(query)
        assert len(results) == 4  # Alice, Bob, Charlie, Eve
        cities = {r.get_value("city") for r in results}
        assert cities == {"New York", "Los Angeles"}
    
    def test_and_with_or_query(self, db_with_data):
        """Test combining AND and OR conditions."""
        # Find active users in New York OR Los Angeles
        base_query = Query().filter("active", Operator.EQ, True)
        complex_query = base_query.or_(
            Filter("city", Operator.EQ, "New York"),
            Filter("city", Operator.EQ, "Los Angeles")
        )
        
        # This creates: active=True AND (city="New York" OR city="Los Angeles")
        results = db_with_data.search(complex_query)
        assert len(results) == 2  # Alice and Charlie (active in NY)
        names = {r.get_value("name") for r in results}
        assert names == {"Alice", "Charlie"}
    
    def test_not_query(self, db_with_data):
        """Test NOT query."""
        # Find users who are active but NOT in New York
        query = Query().filter("active", Operator.EQ, True).not_(
            Filter("city", Operator.EQ, "New York")
        )
        
        results = db_with_data.search(query)
        assert len(results) == 1  # David (active in Chicago)
        assert results[0].get_value("name") == "David"
    
    def test_complex_nested_conditions(self, db_with_data):
        """Test complex nested boolean conditions."""
        # Find: (age > 30 AND city="New York") OR (age < 30 AND active=True)
        builder = QueryBuilder()
        
        # First condition group: age > 30 AND city="New York"
        group1 = QueryBuilder().where("age", Operator.GT, 30).where("city", Operator.EQ, "New York")
        
        # Second condition group: age < 30 AND active=True
        group2 = QueryBuilder().where("age", Operator.LT, 30).where("active", Operator.EQ, True)
        
        # Combine with OR
        builder.or_(group1, group2)
        
        query = builder.build()
        results = db_with_data.search(query)
        assert len(results) == 2  # Charlie (35, NY) and David (28, Chicago, active)
        names = {r.get_value("name") for r in results}
        assert names == {"Charlie", "David"}
    
    def test_query_builder_fluent_api(self, db_with_data):
        """Test QueryBuilder fluent API."""
        # This creates: age >= 30 OR city="Chicago" OR active=False
        query = (
            QueryBuilder()
            .or_(
                Filter("age", Operator.GTE, 30),
                Filter("city", Operator.EQ, "Chicago"),
                Filter("active", Operator.EQ, False)
            )
            .sort_by("name", "asc")
            .limit(3)
            .build()
        )
        
        results = db_with_data.search(query)
        assert len(results) == 3
        # Should get Alice (30), Bob (inactive), Charlie (35) - sorted by name, limited to 3
        names = [r.get_value("name") for r in results]
        assert names == ["Alice", "Bob", "Charlie"]  # Sorted by name
    
    def test_between_with_or(self, db_with_data):
        """Test BETWEEN operator with OR logic."""
        # Find users aged 25-30 OR in Chicago
        query = Query().or_(
            Filter("age", Operator.BETWEEN, (25, 30)),
            Filter("city", Operator.EQ, "Chicago")
        )
        
        results = db_with_data.search(query)
        assert len(results) == 3  # Bob (25), David (28), Alice (30)
        names = {r.get_value("name") for r in results}
        assert names == {"Bob", "David", "Alice"}
    
    def test_complex_query_serialization(self):
        """Test serialization of complex queries."""
        builder = QueryBuilder()
        builder.where("status", Operator.EQ, "active")
        builder.or_(
            Filter("priority", Operator.GT, 5),
            Filter("urgent", Operator.EQ, True)
        )
        
        query = builder.build()
        
        # Convert to dict and back
        query_dict = query.to_dict()
        assert "condition" in query_dict
        
        # Recreate from dict
        query2 = ComplexQuery.from_dict(query_dict)
        assert query2.condition is not None
    
    def test_simple_query_conversion(self, db_with_data):
        """Test conversion between simple and complex queries."""
        # Simple AND query can be converted
        simple = Query().filter("age", Operator.GT, 30).filter("active", Operator.EQ, True)
        
        # Convert to ComplexQuery and back
        complex = simple.or_()  # Creates ComplexQuery with current filters
        
        # Should work the same
        simple_results = db_with_data.search(simple)
        complex_results = db_with_data.search(complex)
        
        assert len(simple_results) == len(complex_results)
        simple_ids = {r.id for r in simple_results}
        complex_ids = {r.id for r in complex_results}
        assert simple_ids == complex_ids
    
    def test_empty_or_conditions(self, db_with_data):
        """Test OR with empty conditions."""
        query = Query().or_()  # No additional conditions
        
        # Should return all records since no filters
        results = db_with_data.search(query)
        assert len(results) == 5
    
    def test_multiple_not_conditions(self, db_with_data):
        """Test multiple NOT conditions."""
        # Find users NOT in New York AND NOT inactive
        builder = QueryBuilder()
        builder.not_(Filter("city", Operator.EQ, "New York"))
        builder.not_(Filter("active", Operator.EQ, False))
        
        query = builder.build()
        results = db_with_data.search(query)
        
        # Should get David (Chicago, active)
        assert len(results) == 1
        assert results[0].get_value("name") == "David"


class TestAsyncBooleanLogic:
    """Test boolean logic with async database."""
    
    @pytest.fixture
    def sample_records(self):
        """Create sample records for testing."""
        return [
            Record(id="1", data={"type": "sensor", "value": 25.5, "location": "room1"}),
            Record(id="2", data={"type": "sensor", "value": 30.2, "location": "room2"}),
            Record(id="3", data={"type": "actuator", "value": 15.0, "location": "room1"}),
            Record(id="4", data={"type": "sensor", "value": 28.7, "location": "room3"}),
        ]
    
    @pytest_asyncio.fixture
    async def async_db_with_data(self, sample_records):
        """Create async database with sample data."""
        db = AsyncMemoryDatabase()
        await db.connect()  # Ensure database is connected
        for record in sample_records:
            await db.create(record)
        return db
    
    @pytest.mark.asyncio
    async def test_async_or_query(self, async_db_with_data):
        """Test OR query with async database."""
        # Find sensors OR items in room1
        query = Query().or_(
            Filter("type", Operator.EQ, "sensor"),
            Filter("location", Operator.EQ, "room1")
        )
        
        results = await async_db_with_data.search(query)
        assert len(results) == 4  # All records match
    
    @pytest.mark.asyncio
    async def test_async_complex_conditions(self, async_db_with_data):
        """Test complex conditions with async database."""
        # Find: (type="sensor" AND value > 26) OR location="room1"
        sensor_high = Query().filter("type", Operator.EQ, "sensor").filter("value", Operator.GT, 26)
        
        query = Query().or_(
            sensor_high,
            Filter("location", Operator.EQ, "room1")
        )
        
        results = await async_db_with_data.search(query)
        # Should match: 1 (room1), 2 (sensor >26), 3 (room1), 4 (sensor >26)
        assert len(results) == 4
        ids = {r.id for r in results}
        assert ids == {"1", "2", "3", "4"}


class TestLogicConditions:
    """Test individual logic condition classes."""
    
    def test_filter_condition(self):
        """Test FilterCondition matching."""
        condition = FilterCondition(Filter("status", Operator.EQ, "active"))
        
        record = Record(data={"status": "active"})
        assert condition.matches(record)
        
        record2 = Record(data={"status": "inactive"})
        assert not condition.matches(record2)
    
    def test_and_logic_condition(self):
        """Test AND logic condition."""
        condition = LogicCondition(
            operator=LogicOperator.AND,
            conditions=[
                FilterCondition(Filter("age", Operator.GT, 25)),
                FilterCondition(Filter("active", Operator.EQ, True))
            ]
        )
        
        record1 = Record(data={"age": 30, "active": True})
        assert condition.matches(record1)
        
        record2 = Record(data={"age": 30, "active": False})
        assert not condition.matches(record2)
        
        record3 = Record(data={"age": 20, "active": True})
        assert not condition.matches(record3)
    
    def test_or_logic_condition(self):
        """Test OR logic condition."""
        condition = LogicCondition(
            operator=LogicOperator.OR,
            conditions=[
                FilterCondition(Filter("priority", Operator.EQ, "high")),
                FilterCondition(Filter("urgent", Operator.EQ, True))
            ]
        )
        
        record1 = Record(data={"priority": "high", "urgent": False})
        assert condition.matches(record1)
        
        record2 = Record(data={"priority": "low", "urgent": True})
        assert condition.matches(record2)
        
        record3 = Record(data={"priority": "low", "urgent": False})
        assert not condition.matches(record3)
    
    def test_not_logic_condition(self):
        """Test NOT logic condition."""
        condition = LogicCondition(
            operator=LogicOperator.NOT,
            conditions=[FilterCondition(Filter("blocked", Operator.EQ, True))]
        )
        
        record1 = Record(data={"blocked": False})
        assert condition.matches(record1)
        
        record2 = Record(data={"blocked": True})
        assert not condition.matches(record2)
    
    def test_nested_logic_conditions(self):
        """Test nested logic conditions."""
        # (age > 30 OR vip=True) AND active=True
        or_condition = LogicCondition(
            operator=LogicOperator.OR,
            conditions=[
                FilterCondition(Filter("age", Operator.GT, 30)),
                FilterCondition(Filter("vip", Operator.EQ, True))
            ]
        )
        
        and_condition = LogicCondition(
            operator=LogicOperator.AND,
            conditions=[
                or_condition,
                FilterCondition(Filter("active", Operator.EQ, True))
            ]
        )
        
        # Should match: age > 30 and active
        record1 = Record(data={"age": 35, "vip": False, "active": True})
        assert and_condition.matches(record1)
        
        # Should match: vip and active
        record2 = Record(data={"age": 25, "vip": True, "active": True})
        assert and_condition.matches(record2)
        
        # Should not match: age > 30 but not active
        record3 = Record(data={"age": 35, "vip": False, "active": False})
        assert not and_condition.matches(record3)
        
        # Should not match: neither age > 30 nor vip
        record4 = Record(data={"age": 25, "vip": False, "active": True})
        assert not and_condition.matches(record4)
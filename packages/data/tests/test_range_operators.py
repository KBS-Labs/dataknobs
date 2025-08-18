"""Test range operators (BETWEEN, type-aware comparisons)."""
import pytest
from datetime import datetime, date, timedelta
from dataknobs_data import Record, Query, Filter, Operator
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase


class TestRangeOperators:
    """Test BETWEEN and NOT_BETWEEN operators with various data types."""
    
    def test_numeric_between(self):
        """Test BETWEEN operator with numeric values."""
        db = SyncMemoryDatabase()
        
        # Create test records with numeric values
        records = [
            Record(id="1", data={"temperature": 15.5, "humidity": 30}),
            Record(id="2", data={"temperature": 25.0, "humidity": 50}),
            Record(id="3", data={"temperature": 35.5, "humidity": 70}),
            Record(id="4", data={"temperature": 45.0, "humidity": 90}),
        ]
        
        for record in records:
            db.create(record)
        
        # Test BETWEEN with floats
        query = Query(filters=[Filter("temperature", Operator.BETWEEN, (20, 40))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test BETWEEN with integers
        query = Query(filters=[Filter("humidity", Operator.BETWEEN, (40, 80))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test NOT_BETWEEN
        query = Query(filters=[Filter("temperature", Operator.NOT_BETWEEN, (20, 40))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"1", "4"}
    
    def test_datetime_between(self):
        """Test BETWEEN operator with datetime values."""
        db = SyncMemoryDatabase()
        
        base_time = datetime(2025, 1, 15, 12, 0, 0)
        
        # Create test records with datetime values
        records = [
            Record(id="1", data={"timestamp": (base_time - timedelta(days=2)).isoformat()}),
            Record(id="2", data={"timestamp": base_time.isoformat()}),
            Record(id="3", data={"timestamp": (base_time + timedelta(days=2)).isoformat()}),
            Record(id="4", data={"timestamp": (base_time + timedelta(days=5)).isoformat()}),
        ]
        
        for record in records:
            db.create(record)
        
        # Test BETWEEN with datetime objects
        start = base_time - timedelta(days=1)
        end = base_time + timedelta(days=3)
        query = Query(filters=[Filter("timestamp", Operator.BETWEEN, (start, end))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test BETWEEN with ISO strings
        query = Query(filters=[
            Filter("timestamp", Operator.BETWEEN, 
                  (start.isoformat(), end.isoformat()))
        ])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test NOT_BETWEEN
        query = Query(filters=[Filter("timestamp", Operator.NOT_BETWEEN, (start, end))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"1", "4"}
    
    def test_string_between(self):
        """Test BETWEEN operator with string values."""
        db = SyncMemoryDatabase()
        
        # Create test records with string values
        records = [
            Record(id="1", data={"name": "Alice", "code": "A100"}),
            Record(id="2", data={"name": "Bob", "code": "B200"}),
            Record(id="3", data={"name": "Charlie", "code": "C300"}),
            Record(id="4", data={"name": "David", "code": "D400"}),
        ]
        
        for record in records:
            db.create(record)
        
        # Test BETWEEN with strings (alphabetical)
        query = Query(filters=[Filter("name", Operator.BETWEEN, ("Bob", "David"))])
        results = db.search(query)
        assert len(results) == 3
        assert set(r.id for r in results) == {"2", "3", "4"}
        
        # Test with codes
        query = Query(filters=[Filter("code", Operator.BETWEEN, ("B000", "C999"))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
    
    def test_mixed_type_comparisons(self):
        """Test comparisons with mixed types."""
        db = SyncMemoryDatabase()
        
        # Create records with mixed types
        records = [
            Record(id="1", data={"value": 10}),      # int
            Record(id="2", data={"value": 20.5}),    # float
            Record(id="3", data={"value": "30"}),    # string number
            Record(id="4", data={"value": 40}),      # int
        ]
        
        for record in records:
            db.create(record)
        
        # Test GT with mixed numeric types
        query = Query(filters=[Filter("value", Operator.GT, 15)])
        results = db.search(query)
        # Should match numeric values > 15
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "4"}
        
        # Test BETWEEN with mixed types (won't match string "30")
        query = Query(filters=[Filter("value", Operator.BETWEEN, (15, 35))])
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "2"
    
    def test_nested_field_between(self):
        """Test BETWEEN operator on nested fields."""
        db = SyncMemoryDatabase()
        
        # Create records with nested structures
        records = [
            Record(
                id="1",
                data={"metrics": {"cpu": 25, "memory": 40}},
                metadata={"priority": 1}
            ),
            Record(
                id="2",
                data={"metrics": {"cpu": 50, "memory": 60}},
                metadata={"priority": 2}
            ),
            Record(
                id="3",
                data={"metrics": {"cpu": 75, "memory": 80}},
                metadata={"priority": 3}
            ),
            Record(
                id="4",
                data={"metrics": {"cpu": 90, "memory": 95}},
                metadata={"priority": 4}
            ),
        ]
        
        for record in records:
            db.create(record)
        
        # Test BETWEEN on nested field
        query = Query(filters=[Filter("metrics.cpu", Operator.BETWEEN, (40, 80))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test on metadata
        query = Query(filters=[Filter("metadata.priority", Operator.BETWEEN, (2, 3))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
    
    def test_between_edge_cases(self):
        """Test BETWEEN operator edge cases."""
        db = SyncMemoryDatabase()
        
        records = [
            Record(id="1", data={"value": 10}),
            Record(id="2", data={"value": 20}),
            Record(id="3", data={"value": None}),
            Record(id="4", data={}),  # Missing field
        ]
        
        for record in records:
            db.create(record)
        
        # Test BETWEEN with None values (should not match)
        query = Query(filters=[Filter("value", Operator.BETWEEN, (5, 25))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"1", "2"}
        
        # Test BETWEEN with invalid range format
        query = Query(filters=[Filter("value", Operator.BETWEEN, 15)])  # Not a tuple
        results = db.search(query)
        assert len(results) == 0
        
        # Test NOT_BETWEEN with invalid range
        query = Query(filters=[Filter("value", Operator.NOT_BETWEEN, 15)])
        results = db.search(query)
        assert len(results) == 2  # All records with values
        assert set(r.id for r in results) == {"1", "2"}
        
        # Test BETWEEN inclusive boundaries
        query = Query(filters=[Filter("value", Operator.BETWEEN, (10, 20))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"1", "2"}  # Both boundaries included
    
    @pytest.mark.asyncio
    async def test_async_between_operator(self):
        """Test BETWEEN operator with async database."""
        db = AsyncMemoryDatabase()
        
        # Create test records
        records = [
            Record(id="1", data={"score": 60}),
            Record(id="2", data={"score": 75}),
            Record(id="3", data={"score": 85}),
            Record(id="4", data={"score": 95}),
        ]
        
        for record in records:
            await db.create(record)
        
        # Test BETWEEN
        query = Query(filters=[Filter("score", Operator.BETWEEN, (70, 90))])
        results = await db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test NOT_BETWEEN
        query = Query(filters=[Filter("score", Operator.NOT_BETWEEN, (70, 90))])
        results = await db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"1", "4"}
    
    def test_fluent_interface_between(self):
        """Test using BETWEEN through the fluent interface."""
        db = SyncMemoryDatabase()
        
        records = [
            Record(id="1", data={"price": 10.99}),
            Record(id="2", data={"price": 25.50}),
            Record(id="3", data={"price": 35.00}),
            Record(id="4", data={"price": 50.00}),
        ]
        
        for record in records:
            db.create(record)
        
        # Test using string operator
        query = Query().filter("price", "between", (20, 40))
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test using Operator enum
        query = Query().filter("price", Operator.BETWEEN, (20, 40))
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test NOT_BETWEEN with string
        query = Query().filter("price", "not_between", (20, 40))
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"1", "4"}
    
    def test_combined_filters_with_between(self):
        """Test BETWEEN combined with other filters."""
        db = SyncMemoryDatabase()
        
        base_time = datetime(2025, 1, 15, 12, 0, 0)
        
        records = [
            Record(
                id="1",
                data={
                    "temperature": 22.5,
                    "timestamp": (base_time - timedelta(hours=2)).isoformat()
                },
                metadata={"sensor": "A"}
            ),
            Record(
                id="2",
                data={
                    "temperature": 25.0,
                    "timestamp": base_time.isoformat()
                },
                metadata={"sensor": "A"}
            ),
            Record(
                id="3",
                data={
                    "temperature": 27.5,
                    "timestamp": (base_time + timedelta(hours=2)).isoformat()
                },
                metadata={"sensor": "B"}
            ),
            Record(
                id="4",
                data={
                    "temperature": 30.0,
                    "timestamp": (base_time + timedelta(hours=4)).isoformat()
                },
                metadata={"sensor": "B"}
            ),
        ]
        
        for record in records:
            db.create(record)
        
        # Combine BETWEEN with EQ filter
        time_range = (
            (base_time - timedelta(hours=1)).isoformat(),
            (base_time + timedelta(hours=3)).isoformat()
        )
        
        query = Query(filters=[
            Filter("timestamp", Operator.BETWEEN, time_range),
            Filter("metadata.sensor", Operator.EQ, "A")
        ])
        
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "2"
        
        # Combine temperature and time ranges
        query = Query(filters=[
            Filter("temperature", Operator.BETWEEN, (24, 28)),
            Filter("timestamp", Operator.BETWEEN, time_range)
        ])
        
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}


class TestImprovedComparisons:
    """Test improved type-aware comparisons for existing operators."""
    
    def test_datetime_string_comparisons(self):
        """Test comparing datetime strings with datetime objects."""
        db = SyncMemoryDatabase()
        
        base_time = datetime(2025, 1, 15, 12, 0, 0)
        
        records = [
            Record(id="1", data={"timestamp": (base_time - timedelta(days=1)).isoformat()}),
            Record(id="2", data={"timestamp": base_time.isoformat()}),
            Record(id="3", data={"timestamp": (base_time + timedelta(days=1)).isoformat()}),
        ]
        
        for record in records:
            db.create(record)
        
        # Test GT with datetime object against ISO strings
        query = Query(filters=[Filter("timestamp", Operator.GT, base_time)])
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "3"
        
        # Test LTE with datetime
        query = Query(filters=[Filter("timestamp", Operator.LTE, base_time)])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"1", "2"}
    
    def test_numeric_type_mixing(self):
        """Test comparisons between int and float types."""
        db = SyncMemoryDatabase()
        
        records = [
            Record(id="1", data={"value": 10}),     # int
            Record(id="2", data={"value": 10.5}),   # float
            Record(id="3", data={"value": 11}),     # int
        ]
        
        for record in records:
            db.create(record)
        
        # Test with float threshold against mixed types
        query = Query(filters=[Filter("value", Operator.GT, 10.2)])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test with int threshold
        query = Query(filters=[Filter("value", Operator.GTE, 11)])
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "3"
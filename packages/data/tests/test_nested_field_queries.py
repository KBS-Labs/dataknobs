"""Test nested field query functionality."""
import pytest
from dataknobs_data import Record, Query, Filter, Operator
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase


class TestNestedFieldQueries:
    """Test that nested field queries work correctly."""
    
    def test_record_get_nested_value(self):
        """Test Record.get_nested_value() method."""
        # Create a record with nested metadata and fields
        record = Record(
            id="test-1",
            data={
                "temperature": 25.5,
                "sensor_config": {
                    "interval": 60,
                    "thresholds": {
                        "min": 10,
                        "max": 40
                    }
                }
            },
            metadata={
                "type": "sensor_reading",
                "location": "room_a",
                "config": {
                    "timeout": 30,
                    "retries": 3
                }
            }
        )
        
        # Test metadata access
        assert record.get_value("metadata.type") == "sensor_reading"
        assert record.get_value("metadata.location") == "room_a"
        assert record.get_value("metadata.config.timeout") == 30
        assert record.get_value("metadata.config.retries") == 3
        
        # Test field access
        assert record.get_value("temperature") == 25.5
        assert record.get_value("fields.temperature") == 25.5
        
        # Test nested field dict access
        assert record.get_value("sensor_config.interval") == 60
        assert record.get_value("sensor_config.thresholds.min") == 10
        assert record.get_value("sensor_config.thresholds.max") == 40
        
        # Test non-existent paths
        assert record.get_value("metadata.missing") is None
        assert record.get_value("metadata.missing", "default") == "default"
        assert record.get_value("sensor_config.missing.nested") is None
    
    def test_memory_database_nested_queries(self):
        """Test that memory database can query nested fields."""
        db = SyncMemoryDatabase()
        
        # Create test records
        records = [
            Record(
                id="sensor-1",
                data={"temperature": 25.0, "humidity": 60},
                metadata={"type": "sensor_reading", "location": "room_a"}
            ),
            Record(
                id="sensor-2",
                data={"temperature": 30.0, "humidity": 45},
                metadata={"type": "sensor_reading", "location": "room_b"}
            ),
            Record(
                id="config-1",
                data={"setting": "value"},
                metadata={"type": "config", "location": "room_a"}
            )
        ]
        
        for record in records:
            db.create(record)
        
        # Query by nested metadata field
        query = Query(filters=[Filter("metadata.type", Operator.EQ, "sensor_reading")])
        results = db.search(query)
        assert len(results) == 2
        assert all(r.metadata["type"] == "sensor_reading" for r in results)
        
        # Query by nested metadata with specific value
        query = Query(filters=[Filter("metadata.location", Operator.EQ, "room_a")])
        results = db.search(query)
        assert len(results) == 2
        assert all(r.metadata["location"] == "room_a" for r in results)
        
        # Combined nested queries
        query = Query(filters=[
            Filter("metadata.type", Operator.EQ, "sensor_reading"),
            Filter("metadata.location", Operator.EQ, "room_b")
        ])
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "sensor-2"
    
    def test_complex_nested_structures(self):
        """Test querying deeply nested structures."""
        db = SyncMemoryDatabase()
        
        # Create a record with complex nesting
        record = Record(
            id="complex-1",
            data={
                "device": {
                    "info": {
                        "manufacturer": "ACME",
                        "model": "TH-100",
                        "specs": {
                            "accuracy": 0.1,
                            "range": {"min": -50, "max": 100}
                        }
                    }
                }
            },
            metadata={
                "deployment": {
                    "site": "factory-1",
                    "building": "A",
                    "floor": 2
                }
            }
        )
        
        db.create(record)
        
        # Query deeply nested fields
        query = Query(filters=[Filter("device.info.manufacturer", Operator.EQ, "ACME")])
        results = db.search(query)
        assert len(results) == 1
        
        query = Query(filters=[Filter("device.info.specs.accuracy", Operator.EQ, 0.1)])
        results = db.search(query)
        assert len(results) == 1
        
        query = Query(filters=[Filter("metadata.deployment.site", Operator.EQ, "factory-1")])
        results = db.search(query)
        assert len(results) == 1
        
        query = Query(filters=[Filter("metadata.deployment.floor", Operator.EQ, 2)])
        results = db.search(query)
        assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_async_database_nested_queries(self):
        """Test that async database also supports nested queries."""
        db = AsyncMemoryDatabase()
        
        # Create test records
        records = [
            Record(
                id="async-1",
                data={"value": 100},
                metadata={"type": "measurement", "source": "sensor_a"}
            ),
            Record(
                id="async-2",
                data={"value": 200},
                metadata={"type": "measurement", "source": "sensor_b"}
            ),
            Record(
                id="async-3",
                data={"value": 300},
                metadata={"type": "config", "source": "sensor_a"}
            )
        ]
        
        for record in records:
            await db.create(record)
        
        # Query by nested metadata
        query = Query(filters=[Filter("metadata.type", Operator.EQ, "measurement")])
        results = await db.search(query)
        assert len(results) == 2
        
        query = Query(filters=[Filter("metadata.source", Operator.EQ, "sensor_a")])
        results = await db.search(query)
        assert len(results) == 2
        
        # Combined filters
        query = Query(filters=[
            Filter("metadata.type", Operator.EQ, "measurement"),
            Filter("metadata.source", Operator.EQ, "sensor_b")
        ])
        results = await db.search(query)
        assert len(results) == 1
        assert results[0].id == "async-2"
    
    def test_nested_queries_with_operators(self):
        """Test nested queries with different operators."""
        db = SyncMemoryDatabase()
        
        # Create records with numeric nested values
        records = [
            Record(
                id="reading-1",
                data={"metrics": {"cpu": 45, "memory": 60}},
                metadata={"priority": 1}
            ),
            Record(
                id="reading-2",
                data={"metrics": {"cpu": 75, "memory": 80}},
                metadata={"priority": 2}
            ),
            Record(
                id="reading-3",
                data={"metrics": {"cpu": 30, "memory": 40}},
                metadata={"priority": 3}
            )
        ]
        
        for record in records:
            db.create(record)
        
        # Test GT operator on nested field
        query = Query(filters=[Filter("metrics.cpu", Operator.GT, 50)])
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "reading-2"
        
        # Test LTE operator on nested field
        query = Query(filters=[Filter("metrics.memory", Operator.LTE, 60)])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"reading-1", "reading-3"}
        
        # Test on metadata
        query = Query(filters=[Filter("metadata.priority", Operator.LT, 3)])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"reading-1", "reading-2"}
    def test_metadata_as_field_not_attribute(self):
        """Test that 'metadata' field is accessed before metadata attribute.
        
        This is a regression test for a bug where get_nested_value() would
        always check record.metadata attribute for paths like 'metadata.key',
        ignoring when 'metadata' was actually a field in the data.
        """
        db = SyncMemoryDatabase()
        
        # Create a record where 'metadata' is a FIELD (not the attribute)
        record = Record(
            id="test-field-metadata",
            data={
                "conversation_id": "conv-123",
                "metadata": {"user_id": "alice", "session": "abc"}
            },
            storage_id="test-field-metadata"
        )
        
        db.create(record)
        
        # Test direct field access
        loaded = db.read("test-field-metadata")
        assert loaded.get_value("metadata") == {"user_id": "alice", "session": "abc"}
        
        # Test nested access - this was broken before the fix
        assert loaded.get_value("metadata.user_id") == "alice"
        assert loaded.get_value("metadata.session") == "abc"
        
        # Test querying by nested field
        query = Query(filters=[Filter("metadata.user_id", Operator.EQ, "alice")])
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "test-field-metadata"
        
        # Query with different value should return nothing
        query = Query(filters=[Filter("metadata.user_id", Operator.EQ, "bob")])
        results = db.search(query)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_metadata_as_field_async(self):
        """Test metadata field access with async database."""
        import uuid
        db = AsyncMemoryDatabase()

        # Create records with metadata as a field
        for idx, user in enumerate(["alice", "bob", "alice"]):
            record = Record(
                data={
                    "conversation_id": f"conv-{user}-{idx}",
                    "metadata": {"user_id": user, "active": True}
                },
                storage_id=f"conv-{user}-{uuid.uuid4()}"
            )
            await db.create(record)

        # Query by nested metadata field
        query = Query(filters=[Filter("metadata.user_id", Operator.EQ, "alice")])
        results = await db.search(query)

        # Should find records with metadata field containing user_id=alice
        assert len(results) == 2
        for result in results:
            assert result.get_value("metadata.user_id") == "alice"

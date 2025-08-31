"""Integration tests for Elasticsearch backend with real service."""

import asyncio
import concurrent.futures
import os
import time
import uuid

import pytest

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data.query import Filter, Operator, SortOrder

#pytestmark = pytest.mark.integration

# Skip all tests if Elasticsearch is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_ELASTICSEARCH", "").lower() == "true",
    reason="Elasticsearch tests require TEST_ELASTICSEARCH=true and a running Elasticsearch instance"
)


class TestElasticsearchIntegration:
    """Integration tests for Elasticsearch backend."""

    def test_connection_and_index_creation(self, elasticsearch_test_index):
        """Test that we can connect and create indices."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # The index should be created automatically
        # Try to create a record to verify
        record = Record({"test": "data"})
        id = db.create(record)
        
        assert id is not None
        assert db.exists(id)
        
        # Cleanup
        db.delete(id)
        db.close()

    def test_full_crud_cycle(self, elasticsearch_test_index):
        """Test complete CRUD operations with real Elasticsearch."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create
        record = Record({
            "name": "Test User",
            "age": 30,
            "active": True,
            "balance": 1234.56,
            "tags": ["test", "integration"],
            "nested": {"key": "value", "number": 42},
        })
        id = db.create(record)
        assert id is not None
        
        # Small delay for indexing (even with refresh=true)
        time.sleep(0.5)
        
        # Read
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Test User"
        assert retrieved.get_value("age") == 30
        assert retrieved.get_value("active") is True
        assert retrieved.get_value("balance") == 1234.56
        assert retrieved.get_value("tags") == ["test", "integration"]
        assert retrieved.get_value("nested") == {"key": "value", "number": 42}
        
        # Update
        updated_record = Record({
            "name": "Updated User",
            "age": 31,
            "active": False,
            "balance": 5678.90,
            "tags": ["updated"],
            "nested": {"key": "updated_value", "number": 84},
        })
        success = db.update(id, updated_record)
        assert success is True
        
        time.sleep(0.5)
        
        retrieved = db.read(id)
        assert retrieved.get_value("name") == "Updated User"
        assert retrieved.get_value("age") == 31
        assert retrieved.get_value("active") is False
        assert retrieved.get_value("nested")["number"] == 84
        
        # Delete
        success = db.delete(id)
        assert success is True
        
        time.sleep(0.5)
        assert db.read(id) is None
        
        db.close()

    def test_batch_operations_with_sample_data(self, elasticsearch_test_index, sample_records):
        """Test batch operations with sample dataset."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create batch
        ids = db.create_batch(sample_records)
        assert len(ids) == len(sample_records)
        
        # Wait for indexing
        time.sleep(1)
        
        # Verify all created
        for id in ids:
            assert db.exists(id)
        
        # Read batch
        retrieved = db.read_batch(ids)
        assert len(retrieved) == len(ids)
        assert all(r is not None for r in retrieved)
        
        # Verify data integrity
        for i, record in enumerate(retrieved):
            original = sample_records[i]
            assert record.get_value("name") == original.get_value("name")
            assert record.get_value("age") == original.get_value("age")
            
            # Compare metadata, but ignore Elasticsearch-specific metadata
            expected_metadata = original.metadata
            actual_metadata = {k: v for k, v in record.metadata.items() 
                             if k in expected_metadata}
            assert actual_metadata == expected_metadata
        
        # Delete batch
        results = db.delete_batch(ids)
        assert all(results)
        
        time.sleep(0.5)
        
        # Verify all deleted
        retrieved = db.read_batch(ids)
        assert all(r is None for r in retrieved)
        
        db.close()

    def test_complex_queries(self, elasticsearch_test_index, sample_records):
        """Test complex query operations."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Insert sample data
        ids = db.create_batch(sample_records)
        
        # Wait for indexing
        time.sleep(1)
        
        # Test 1: Filter by department (term query)
        query = Query().filter("department", Operator.EQ, "Engineering")
        results = db.search(query)
        assert len(results) == 3
        names = {r.get_value("name") for r in results}
        assert names == {"Alice Johnson", "Charlie Brown", "Eve Anderson"}
        
        # Test 2: Range query on salary
        query = Query().filter("salary", Operator.GT, 100000)
        results = db.search(query)
        assert len(results) == 2
        names = {r.get_value("name") for r in results}
        assert names == {"Charlie Brown", "Eve Anderson"}
        
        # Test 3: Combined filters
        query = (Query()
            .filter("department", Operator.EQ, "Engineering")
            .filter("active", Operator.EQ, True)
            .filter("salary", Operator.GTE, 100000))
        results = db.search(query)
        assert len(results) == 1
        assert results[0].get_value("name") == "Eve Anderson"
        
        # Test 4: Wildcard pattern matching (LIKE)
        query = Query().filter("email", Operator.LIKE, "*@example.com")
        results = db.search(query)
        assert len(results) == 5  # All have @example.com
        
        # Test 5: IN operator (terms query)
        query = Query().filter("department", Operator.IN, ["Engineering", "HR"])
        results = db.search(query)
        assert len(results) == 4
        
        # Test 6: NOT_IN operator
        query = Query().filter("department", Operator.NOT_IN, ["Engineering", "HR"])
        results = db.search(query)
        assert len(results) == 1
        assert results[0].get_value("department") == "Marketing"
        
        # Test 7: EXISTS filter
        # Add a record without email
        no_email_record = Record({"name": "No Email", "department": "IT"})
        no_email_id = db.create(no_email_record)
        time.sleep(0.5)
        
        query = Query().filter("email", Operator.EXISTS)
        results = db.search(query)
        assert len(results) == 5  # Only original sample records have email
        
        query = Query().filter("email", Operator.NOT_EXISTS)
        results = db.search(query)
        assert len(results) == 1
        assert results[0].get_value("name") == "No Email"
        
        # Cleanup
        db.delete(no_email_id)
        db.delete_batch(ids)
        db.close()

    def test_sorting_and_pagination(self, elasticsearch_test_index, sample_records):
        """Test sorting and pagination features."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Insert sample data
        ids = db.create_batch(sample_records)
        
        # Wait for indexing
        time.sleep(1)
        
        # Test 1: Sort by age ascending
        query = Query().sort("age", SortOrder.ASC)
        results = db.search(query)
        ages = [r.get_value("age") for r in results]
        assert ages == sorted(ages)
        
        # Test 2: Sort by salary descending
        query = Query().sort("salary", SortOrder.DESC)
        results = db.search(query)
        salaries = [r.get_value("salary") for r in results]
        assert salaries == sorted(salaries, reverse=True)
        
        # Test 3: Pagination with limit
        query = Query().sort("name.keyword", SortOrder.ASC).limit(3)
        results = db.search(query)
        assert len(results) == 3
        
        # Test 4: Pagination with offset
        query = Query().sort("name.keyword", SortOrder.ASC).offset(2).limit(2)
        results = db.search(query)
        assert len(results) == 2
        
        # Test 5: Deep pagination
        query = Query().sort("age", SortOrder.ASC)
        all_results = db.search(query)
        
        # Get results in pages
        page1 = db.search(Query().sort("age", SortOrder.ASC).limit(2))
        page2 = db.search(Query().sort("age", SortOrder.ASC).offset(2).limit(2))
        page3 = db.search(Query().sort("age", SortOrder.ASC).offset(4).limit(2))
        
        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1  # Only 5 records total
        
        # Verify no overlap
        page1_ages = {r.get_value("age") for r in page1}
        page2_ages = {r.get_value("age") for r in page2}
        page3_ages = {r.get_value("age") for r in page3}
        assert page1_ages.isdisjoint(page2_ages)
        assert page2_ages.isdisjoint(page3_ages)
        
        # Cleanup
        db.delete_batch(ids)
        db.close()

    def test_full_text_search_capabilities(self, elasticsearch_test_index):
        """Test Elasticsearch full-text search features."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create documents with text content
        docs = [
            Record({
                "title": "Introduction to Elasticsearch",
                "content": "Elasticsearch is a distributed, RESTful search and analytics engine.",
                "tags": ["search", "distributed", "analytics"],
            }),
            Record({
                "title": "Getting Started with Python",
                "content": "Python is a high-level programming language with dynamic semantics.",
                "tags": ["programming", "python", "tutorial"],
            }),
            Record({
                "title": "Advanced Elasticsearch Queries",
                "content": "Learn how to write complex queries in Elasticsearch using the Query DSL.",
                "tags": ["search", "advanced", "queries"],
            }),
        ]
        
        ids = db.create_batch(docs)
        time.sleep(1)
        
        # Test wildcard search
        query = Query().filter("title", Operator.LIKE, "*Elasticsearch*")
        results = db.search(query)
        assert len(results) == 2
        
        # Test regex search
        query = Query().filter("content", Operator.REGEX, ".*distributed.*")
        results = db.search(query)
        assert len(results) == 1
        assert "distributed" in results[0].get_value("content")
        
        # Cleanup
        db.delete_batch(ids)
        db.close()

    def test_metadata_persistence(self, elasticsearch_test_index):
        """Test that metadata is properly stored and retrieved."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create record with complex metadata
        record = Record(
            data={"name": "Test", "value": 123},
            metadata={
                "created_by": "integration_test",
                "version": 2,
                "tags": ["test", "elasticsearch"],
                "nested": {
                    "level1": {
                        "level2": {
                            "key": "deeply_nested_value"
                        }
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        )
        id = db.create(record)
        
        time.sleep(0.5)
        
        # Retrieve and verify metadata
        retrieved = db.read(id)
        
        # Compare metadata, but ignore Elasticsearch-specific metadata
        expected_metadata = record.metadata
        actual_metadata = {k: v for k, v in retrieved.metadata.items() 
                         if k in expected_metadata}
        assert actual_metadata == expected_metadata
        assert retrieved.metadata["created_by"] == "integration_test"
        assert retrieved.metadata["version"] == 2
        assert retrieved.metadata["tags"] == ["test", "elasticsearch"]
        assert retrieved.metadata["nested"]["level1"]["level2"]["key"] == "deeply_nested_value"
        
        # Update with new metadata
        updated_record = Record(
            data={"name": "Updated", "value": 456},
            metadata={"version": 3, "updated": True, "updater": "test_suite"}
        )
        db.update(id, updated_record)
        
        time.sleep(0.5)
        
        retrieved = db.read(id)
        assert retrieved.metadata["version"] == 3
        assert retrieved.metadata["updated"] is True
        assert retrieved.metadata["updater"] == "test_suite"
        
        # Cleanup
        db.delete(id)
        db.close()

    def test_concurrent_operations(self, elasticsearch_test_index):
        """Test concurrent database operations."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        def create_record(index):
            record = Record({"index": index, "data": f"concurrent_{index}"})
            return db.create(record)
        
        def read_record(id):
            return db.read(id)
        
        # Create records concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_record, i) for i in range(10)]
            ids = [f.result() for f in futures]
        
        assert len(ids) == 10
        assert all(id is not None for id in ids)
        
        time.sleep(1)
        
        # Read records concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_record, id) for id in ids]
            records = [f.result() for f in futures]
        
        assert len(records) == 10
        assert all(r is not None for r in records)
        
        # Verify data integrity
        indices = {r.get_value("index") for r in records}
        assert indices == set(range(10))
        
        # Cleanup
        db.delete_batch(ids)
        db.close()

    def test_special_characters_and_unicode(self, elasticsearch_test_index):
        """Test handling of special characters and Unicode."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create record with special characters
        record = Record({
            "name": "Test's \"Special\" Name",
            "description": "Line 1\nLine 2\tTabbed",
            "unicode": "Hello 世界 🌍 مرحبا мир",
            "symbols": "!@#$%^&*(){}[]|\\:;\"'<>,.?/",
            "json_field": {"nested": {"key": "value with 'quotes'"}},
            "html": "<div>HTML & entities</div>",
        })
        id = db.create(record)
        
        time.sleep(0.5)
        
        # Retrieve and verify
        retrieved = db.read(id)
        assert retrieved.get_value("name") == "Test's \"Special\" Name"
        assert retrieved.get_value("description") == "Line 1\nLine 2\tTabbed"
        assert retrieved.get_value("unicode") == "Hello 世界 🌍 مرحبا мир"
        assert retrieved.get_value("symbols") == "!@#$%^&*(){}[]|\\:;\"'<>,.?/"
        assert retrieved.get_value("json_field")["nested"]["key"] == "value with 'quotes'"
        assert retrieved.get_value("html") == "<div>HTML & entities</div>"
        
        # Search with special characters
        query = Query().filter("name", Operator.LIKE, "*Special*")
        results = db.search(query)
        assert len(results) == 1
        
        # Cleanup
        db.delete(id)
        db.close()

    def test_count_operations(self, elasticsearch_test_index, sample_records):
        """Test count operations."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Initially empty
        assert db.count() == 0
        
        # Insert sample data
        ids = db.create_batch(sample_records)
        
        time.sleep(1)
        
        # Count all
        assert db.count() == len(sample_records)
        
        # Count with query
        query = Query().filter("department", Operator.EQ, "Engineering")
        assert db.count(query) == 3
        
        query = Query().filter("active", Operator.EQ, True)
        assert db.count(query) == 4
        
        # Clear and verify
        deleted = db.clear()
        assert deleted == len(sample_records)
        
        time.sleep(0.5)
        assert db.count() == 0
        
        db.close()

    def test_array_field_operations(self, elasticsearch_test_index):
        """Test operations on array fields."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create records with array fields
        records = [
            Record({
                "name": "Product A",
                "tags": ["electronics", "mobile", "smartphone"],
                "ratings": [4.5, 4.2, 4.8, 4.6],
            }),
            Record({
                "name": "Product B",
                "tags": ["electronics", "laptop", "computer"],
                "ratings": [3.9, 4.1, 4.0],
            }),
            Record({
                "name": "Product C",
                "tags": ["books", "fiction", "novel"],
                "ratings": [4.9, 5.0, 4.8],
            }),
        ]
        
        ids = db.create_batch(records)
        time.sleep(1)
        
        # Search by array element
        query = Query().filter("tags", Operator.EQ, "electronics")
        results = db.search(query)
        assert len(results) == 2
        names = {r.get_value("name") for r in results}
        assert names == {"Product A", "Product B"}
        
        # IN query with array field
        query = Query().filter("tags", Operator.IN, ["mobile", "laptop"])
        results = db.search(query)
        assert len(results) == 2
        
        # Cleanup
        db.delete_batch(ids)
        db.close()


@pytest.mark.asyncio
class TestElasticsearchAsyncIntegration:
    """Async integration tests for Elasticsearch backend."""

    async def test_async_crud_operations(self, elasticsearch_test_index):
        """Test async CRUD operations."""
        db = await AsyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create
        record = Record({
            "name": "Async Test",
            "value": 42,
            "async": True,
        })
        id = await db.create(record)
        assert id is not None
        
        # Small delay for indexing
        await asyncio.sleep(0.5)
        
        # Read
        retrieved = await db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Async Test"
        assert retrieved.get_value("async") is True
        
        # Update
        updated = Record({
            "name": "Async Updated",
            "value": 84,
            "async": False,
        })
        success = await db.update(id, updated)
        assert success is True
        
        await asyncio.sleep(0.5)
        
        retrieved = await db.read(id)
        assert retrieved.get_value("value") == 84
        
        # Delete
        success = await db.delete(id)
        assert success is True
        
        await asyncio.sleep(0.5)
        assert await db.read(id) is None
        
        await db.close()

    async def test_async_batch_operations(self, elasticsearch_test_index, sample_records):
        """Test async batch operations."""
        db = await AsyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create batch
        ids = await db.create_batch(sample_records)
        assert len(ids) == len(sample_records)
        
        await asyncio.sleep(1)
        
        # Read batch
        retrieved = await db.read_batch(ids)
        assert all(r is not None for r in retrieved)
        
        # Delete batch
        results = await db.delete_batch(ids)
        assert all(results)
        
        await db.close()

    async def test_async_concurrent_operations(self, elasticsearch_test_index):
        """Test concurrent async operations."""
        db = await AsyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Create multiple records concurrently
        tasks = []
        for i in range(10):
            record = Record({"index": i, "type": "async_concurrent"})
            tasks.append(db.create(record))
        
        ids = await asyncio.gather(*tasks)
        assert len(ids) == 10
        
        await asyncio.sleep(1)
        
        # Read all concurrently
        tasks = [db.read(id) for id in ids]
        records = await asyncio.gather(*tasks)
        assert all(r is not None for r in records)
        
        # Verify data
        indices = {r.get_value("index") for r in records}
        assert indices == set(range(10))
        
        # Cleanup
        await db.delete_batch(ids)
        await db.close()

    async def test_async_search_operations(self, elasticsearch_test_index, sample_records):
        """Test async search operations."""
        db = await AsyncDatabase.from_backend("elasticsearch", elasticsearch_test_index)
        
        # Insert data
        ids = await db.create_batch(sample_records)
        
        await asyncio.sleep(1)
        
        # Search
        query = Query().filter("department", Operator.EQ, "Engineering")
        results = await db.search(query)
        assert len(results) == 3
        
        # Count
        count = await db.count(query)
        assert count == 3
        
        # Cleanup
        await db.clear()
        await db.close()

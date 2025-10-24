"""Unit tests for dataknobs backend resource adapters."""

import pytest
from typing import List, Optional, Any
from collections import OrderedDict
from dataknobs_llm.prompts import (
    DataknobsBackendAdapter,
    AsyncDataknobsBackendAdapter,
)


# Mock dataknobs data structures for testing
class MockField:
    """Mock Field class from dataknobs_data."""

    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


class MockRecord:
    """Mock Record class from dataknobs_data."""

    def __init__(self, data: dict, storage_id: Optional[str] = None):
        self.fields = OrderedDict()
        self.metadata = {}
        self.storage_id = storage_id

        for key, value in data.items():
            if key == "_metadata":
                self.metadata = value
            else:
                self.fields[key] = MockField(key, value)

    def get_value(self, name: str, default: Any = None) -> Any:
        """Get field value with dot-notation support."""
        if '.' in name:
            parts = name.split('.')
            value = self
            for part in parts:
                if hasattr(value, 'fields') and part in value.fields:
                    value = value.fields[part].value
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        else:
            if name in self.fields:
                return self.fields[name].value
            return default

    def to_dict(self, include_metadata: bool = False) -> dict:
        """Convert record to dictionary."""
        result = {name: field.value for name, field in self.fields.items()}
        if include_metadata:
            result["_metadata"] = self.metadata
        return result


class MockQuery:
    """Mock Query class from dataknobs_data."""

    def __init__(self, query_text: str, limit: int = 5, **kwargs):
        self.query_text = query_text
        self.limit = limit
        self.kwargs = kwargs


class MockSyncDatabase:
    """Mock SyncDatabase for testing."""

    def __init__(self):
        self._records = {}

    def add_record(self, record_id: str, record: MockRecord):
        """Add a record to the mock database."""
        self._records[record_id] = record

    def read(self, id: str) -> Optional[MockRecord]:
        """Read a record by ID."""
        return self._records.get(id)

    def search(self, query: MockQuery) -> List[MockRecord]:
        """Search records matching query."""
        results = []
        for record in self._records.values():
            # Simple text search across all field values
            record_text = " ".join(
                str(field.value) for field in record.fields.values()
            )
            if query.query_text.lower() in record_text.lower():
                results.append(record)
                if len(results) >= query.limit:
                    break
        return results


class MockAsyncDatabase:
    """Mock AsyncDatabase for testing."""

    def __init__(self):
        self._records = {}

    def add_record(self, record_id: str, record: MockRecord):
        """Add a record to the mock database."""
        self._records[record_id] = record

    async def read(self, id: str) -> Optional[MockRecord]:
        """Read a record by ID (async)."""
        return self._records.get(id)

    async def search(self, query: MockQuery) -> List[MockRecord]:
        """Search records matching query (async)."""
        results = []
        for record in self._records.values():
            # Simple text search across all field values
            record_text = " ".join(
                str(field.value) for field in record.fields.values()
            )
            if query.query_text.lower() in record_text.lower():
                results.append(record)
                if len(results) >= query.limit:
                    break
        return results


# Patch the imports in the adapter module
import sys
from unittest.mock import MagicMock

# Create mock modules for dataknobs_data
mock_query_module = MagicMock()
mock_query_module.Query = MockQuery

sys.modules['dataknobs_data'] = MagicMock()
sys.modules['dataknobs_data.database'] = MagicMock()
sys.modules['dataknobs_data.records'] = MagicMock()
sys.modules['dataknobs_data.query'] = mock_query_module


class TestDataknobsBackendAdapter:
    """Test suite for DataknobsBackendAdapter (sync)."""

    def test_initialization(self):
        """Test basic initialization."""
        db = MockSyncDatabase()
        adapter = DataknobsBackendAdapter(db, name="test_db")
        assert adapter.name == "test_db"
        assert adapter._database == db
        assert adapter._text_field == "content"

    def test_initialization_with_custom_fields(self):
        """Test initialization with custom field names."""
        db = MockSyncDatabase()
        adapter = DataknobsBackendAdapter(
            db,
            name="custom_db",
            text_field="text",
            metadata_field="meta"
        )
        assert adapter._text_field == "text"
        assert adapter._metadata_field == "meta"

    def test_get_value_full_record(self):
        """Test getting full record by ID."""
        db = MockSyncDatabase()
        record = MockRecord({
            "content": "Hello world",
            "author": "Alice"
        }, storage_id="rec_1")
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(db)
        result = adapter.get_value("rec_1")

        assert result is not None
        assert result["content"] == "Hello world"
        assert result["author"] == "Alice"

    def test_get_value_specific_field(self):
        """Test getting specific field from record."""
        db = MockSyncDatabase()
        record = MockRecord({
            "content": "Hello world",
            "author": "Alice",
            "score": 95
        }, storage_id="rec_1")
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(db)

        # Get specific fields using dot notation
        assert adapter.get_value("rec_1.author") == "Alice"
        assert adapter.get_value("rec_1.score") == 95

    def test_get_value_missing_record(self):
        """Test getting value for non-existent record."""
        db = MockSyncDatabase()
        adapter = DataknobsBackendAdapter(db)

        result = adapter.get_value("missing_id", default="not_found")
        assert result == "not_found"

    def test_get_value_missing_field(self):
        """Test getting missing field from record."""
        db = MockSyncDatabase()
        record = MockRecord({"content": "Hello"}, storage_id="rec_1")
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(db)
        result = adapter.get_value("rec_1.missing_field", default="default")
        assert result == "default"

    def test_search_basic(self):
        """Test basic search functionality."""
        db = MockSyncDatabase()

        # Add test records
        record1 = MockRecord(
            {"content": "Alice in Wonderland"},
            storage_id="rec_1"
        )
        record1.metadata["score"] = 0.9

        record2 = MockRecord(
            {"content": "Bob's adventure"},
            storage_id="rec_2"
        )
        record2.metadata["score"] = 0.8

        record3 = MockRecord(
            {"content": "Alice in Paris"},
            storage_id="rec_3"
        )
        record3.metadata["score"] = 0.85

        db.add_record("rec_1", record1)
        db.add_record("rec_2", record2)
        db.add_record("rec_3", record3)

        adapter = DataknobsBackendAdapter(db)

        # Search for "Alice"
        results = adapter.search("Alice")
        assert len(results) == 2
        assert all("Alice" in r["content"] for r in results)

    def test_search_with_k_limit(self):
        """Test search with k parameter limiting results."""
        db = MockSyncDatabase()

        # Add multiple records
        for i in range(10):
            record = MockRecord(
                {"content": f"test item {i}"},
                storage_id=f"rec_{i}"
            )
            record.metadata["score"] = 1.0
            db.add_record(f"rec_{i}", record)

        adapter = DataknobsBackendAdapter(db)

        results = adapter.search("test", k=3)
        assert len(results) == 3

    def test_search_result_structure(self):
        """Test that search results have correct structure."""
        db = MockSyncDatabase()

        record = MockRecord(
            {"content": "Hello world"},
            storage_id="rec_1"
        )
        record.metadata["score"] = 0.95
        record.metadata["author"] = "Alice"
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(db)
        results = adapter.search("Hello")

        assert len(results) == 1
        result = results[0]

        assert "content" in result
        assert result["content"] == "Hello world"
        assert "score" in result
        assert result["score"] == 0.95
        assert "metadata" in result
        assert result["metadata"]["author"] == "Alice"
        assert result["metadata"]["record_id"] == "rec_1"

    def test_search_with_custom_text_field(self):
        """Test search using custom text field."""
        db = MockSyncDatabase()

        record = MockRecord(
            {"text": "Custom text field", "content": "Ignored"},
            storage_id="rec_1"
        )
        record.metadata["score"] = 1.0
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(db, text_field="text")
        results = adapter.search("Custom")

        assert len(results) == 1
        assert results[0]["content"] == "Custom text field"

    def test_search_with_metadata_field(self):
        """Test search with metadata field extraction."""
        db = MockSyncDatabase()

        record = MockRecord(
            {"content": "Hello", "category": "greeting"},
            storage_id="rec_1"
        )
        record.metadata["score"] = 1.0
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(
            db,
            metadata_field="category"
        )
        results = adapter.search("Hello")

        assert len(results) == 1
        assert "metadata_field" in results[0]["metadata"]
        assert results[0]["metadata"]["metadata_field"] == "greeting"

    def test_search_with_min_score(self):
        """Test search with minimum score filter."""
        db = MockSyncDatabase()

        record1 = MockRecord({"content": "High score"}, storage_id="rec_1")
        record1.metadata["score"] = 0.9

        record2 = MockRecord({"content": "Low score"}, storage_id="rec_2")
        record2.metadata["score"] = 0.3

        db.add_record("rec_1", record1)
        db.add_record("rec_2", record2)

        adapter = DataknobsBackendAdapter(db)

        # Without min_score
        results = adapter.search("score")
        assert len(results) == 2

        # With min_score=0.5
        results = adapter.search("score", min_score=0.5)
        assert len(results) == 1
        assert results[0]["content"] == "High score"

    def test_batch_get_values(self):
        """Test batch getting multiple record fields."""
        db = MockSyncDatabase()

        record1 = MockRecord({"content": "Value 1"}, storage_id="rec_1")
        record2 = MockRecord({"content": "Value 2"}, storage_id="rec_2")

        db.add_record("rec_1", record1)
        db.add_record("rec_2", record2)

        adapter = DataknobsBackendAdapter(db)

        results = adapter.batch_get_values(
            ["rec_1.content", "rec_2.content", "rec_3.content"],
            default="Missing"
        )

        assert results["rec_1.content"] == "Value 1"
        assert results["rec_2.content"] == "Value 2"
        assert results["rec_3.content"] == "Missing"


class TestAsyncDataknobsBackendAdapter:
    """Test suite for AsyncDataknobsBackendAdapter."""

    def test_initialization(self):
        """Test basic initialization."""
        db = MockAsyncDatabase()
        adapter = AsyncDataknobsBackendAdapter(db, name="test_async_db")
        assert adapter.name == "test_async_db"
        assert adapter._database == db

    @pytest.mark.asyncio
    async def test_get_value_full_record(self):
        """Test getting full record by ID (async)."""
        db = MockAsyncDatabase()
        record = MockRecord({
            "content": "Hello async",
            "author": "Bob"
        }, storage_id="rec_1")
        db.add_record("rec_1", record)

        adapter = AsyncDataknobsBackendAdapter(db)
        result = await adapter.get_value("rec_1")

        assert result is not None
        assert result["content"] == "Hello async"
        assert result["author"] == "Bob"

    @pytest.mark.asyncio
    async def test_get_value_specific_field(self):
        """Test getting specific field from record (async)."""
        db = MockAsyncDatabase()
        record = MockRecord({
            "content": "Hello",
            "author": "Charlie"
        }, storage_id="rec_1")
        db.add_record("rec_1", record)

        adapter = AsyncDataknobsBackendAdapter(db)
        assert await adapter.get_value("rec_1.author") == "Charlie"

    @pytest.mark.asyncio
    async def test_get_value_missing_record(self):
        """Test getting value for non-existent record (async)."""
        db = MockAsyncDatabase()
        adapter = AsyncDataknobsBackendAdapter(db)

        result = await adapter.get_value("missing", default="not_found")
        assert result == "not_found"

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search functionality (async)."""
        db = MockAsyncDatabase()

        record1 = MockRecord(
            {"content": "Async search test"},
            storage_id="rec_1"
        )
        record1.metadata["score"] = 1.0

        record2 = MockRecord(
            {"content": "Another test"},
            storage_id="rec_2"
        )
        record2.metadata["score"] = 0.9

        db.add_record("rec_1", record1)
        db.add_record("rec_2", record2)

        adapter = AsyncDataknobsBackendAdapter(db)
        results = await adapter.search("test")

        assert len(results) == 2
        assert all("test" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_with_k_limit(self):
        """Test search with k limit (async)."""
        db = MockAsyncDatabase()

        for i in range(10):
            record = MockRecord(
                {"content": f"async item {i}"},
                storage_id=f"rec_{i}"
            )
            record.metadata["score"] = 1.0
            db.add_record(f"rec_{i}", record)

        adapter = AsyncDataknobsBackendAdapter(db)
        results = await adapter.search("async", k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_get_values(self):
        """Test async batch getting multiple values."""
        db = MockAsyncDatabase()

        record1 = MockRecord({"content": "Async 1"}, storage_id="rec_1")
        record2 = MockRecord({"content": "Async 2"}, storage_id="rec_2")

        db.add_record("rec_1", record1)
        db.add_record("rec_2", record2)

        adapter = AsyncDataknobsBackendAdapter(db)

        results = await adapter.batch_get_values(
            ["rec_1.content", "rec_2.content", "rec_3.content"],
            default="Missing"
        )

        assert results["rec_1.content"] == "Async 1"
        assert results["rec_2.content"] == "Async 2"
        assert results["rec_3.content"] == "Missing"


class TestDataknobsAdapterEdgeCases:
    """Test edge cases for dataknobs backend adapters."""

    def test_empty_database(self):
        """Test adapter with empty database."""
        db = MockSyncDatabase()
        adapter = DataknobsBackendAdapter(db)

        assert adapter.get_value("any_id") is None
        assert adapter.search("any_query") == []

    def test_search_with_no_results(self):
        """Test search that matches no records."""
        db = MockSyncDatabase()

        record = MockRecord({"content": "Hello"}, storage_id="rec_1")
        record.metadata["score"] = 1.0
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(db)
        results = adapter.search("NonExistent")

        assert results == []

    def test_record_with_missing_score(self):
        """Test handling records without score in metadata."""
        db = MockSyncDatabase()

        record = MockRecord({"content": "No score"}, storage_id="rec_1")
        # No score in metadata
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(db)
        results = adapter.search("score")

        assert len(results) == 1
        # Should default to 1.0
        assert results[0]["score"] == 1.0

    def test_record_with_alternative_score_field(self):
        """Test using _score field as alternative to score."""
        db = MockSyncDatabase()

        record = MockRecord({"content": "Alternative score"}, storage_id="rec_1")
        record.metadata["_score"] = 0.75
        db.add_record("rec_1", record)

        adapter = DataknobsBackendAdapter(db)
        results = adapter.search("score")

        assert len(results) == 1
        assert results[0]["score"] == 0.75

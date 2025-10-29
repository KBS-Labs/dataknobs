"""Unit tests for dataknobs backend resource adapters.

Tests use MemoryDatabase backends which support text search via LIKE operator.
The adapter uses Filter(field=text_field, operator=Operator.LIKE, value="%query%")
for text search, which works with all dataknobs backends.
"""

import pytest
from dataknobs_llm.prompts import (
    DataknobsBackendAdapter,
    AsyncDataknobsBackendAdapter,
)
from dataknobs_data import Record
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase


class TestDataknobsBackendAdapter:
    """Test suite for DataknobsBackendAdapter (sync)."""

    def test_initialization(self):
        """Test basic initialization."""
        db = SyncMemoryDatabase()
        adapter = DataknobsBackendAdapter(db, name="test_db")
        assert adapter.name == "test_db"
        assert adapter._database == db
        assert adapter._text_field == "content"

    def test_initialization_with_custom_fields(self):
        """Test initialization with custom field names."""
        db = SyncMemoryDatabase()
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
        db = SyncMemoryDatabase()
        record = Record(
            data={"content": "Hello world", "author": "Alice"},
            storage_id="rec_1"
        )
        db.create(record)

        adapter = DataknobsBackendAdapter(db)
        result = adapter.get_value("rec_1")

        assert result is not None
        assert result["content"] == "Hello world"
        assert result["author"] == "Alice"

    def test_get_value_specific_field(self):
        """Test getting specific field from record."""
        db = SyncMemoryDatabase()
        record = Record(
            data={"content": "Hello world", "author": "Alice", "score": 95},
            storage_id="rec_1"
        )
        db.create(record)

        adapter = DataknobsBackendAdapter(db)

        # Get specific fields using dot notation
        assert adapter.get_value("rec_1.author") == "Alice"
        assert adapter.get_value("rec_1.score") == 95

    def test_get_value_missing_record(self):
        """Test getting value for non-existent record."""
        db = SyncMemoryDatabase()
        adapter = DataknobsBackendAdapter(db)

        result = adapter.get_value("missing_id", default="not_found")
        assert result == "not_found"

    def test_get_value_missing_field(self):
        """Test getting missing field from record."""
        db = SyncMemoryDatabase()
        record = Record(data={"content": "Hello"}, storage_id="rec_1")
        db.create(record)

        adapter = DataknobsBackendAdapter(db)
        result = adapter.get_value("rec_1.missing_field", default="default")
        assert result == "default"

    def test_search_basic(self):
        """Test basic search functionality."""
        db = SyncMemoryDatabase()

        # Add test records
        db.create(Record(
            data={"content": "Alice in Wonderland"},
            metadata={"score": 0.9},
            storage_id="rec_1"
        ))
        db.create(Record(
            data={"content": "Bob's adventure"},
            metadata={"score": 0.8},
            storage_id="rec_2"
        ))
        db.create(Record(
            data={"content": "Alice in Paris"},
            metadata={"score": 0.85},
            storage_id="rec_3"
        ))

        adapter = DataknobsBackendAdapter(db)

        # Search for "Alice"
        results = adapter.search("Alice")
        assert len(results) == 2
        assert all("Alice" in r["content"] for r in results)

    def test_search_with_k_limit(self):
        """Test search with k parameter limiting results."""
        db = SyncMemoryDatabase()

        # Add multiple records
        for i in range(10):
            db.create(Record(
                data={"content": f"test item {i}"},
                metadata={"score": 1.0},
                storage_id=f"rec_{i}"
            ))

        adapter = DataknobsBackendAdapter(db)

        results = adapter.search("test", k=3)
        assert len(results) == 3

    def test_search_result_structure(self):
        """Test that search results have correct structure."""
        db = SyncMemoryDatabase()

        db.create(Record(
            data={"content": "Hello world"},
            metadata={"score": 0.95, "author": "Alice"},
            storage_id="rec_1"
        ))

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
        db = SyncMemoryDatabase()

        db.create(Record(
            data={"text": "Custom text field", "content": "Ignored"},
            metadata={"score": 1.0},
            storage_id="rec_1"
        ))

        adapter = DataknobsBackendAdapter(db, text_field="text")
        results = adapter.search("Custom")

        assert len(results) == 1
        assert results[0]["content"] == "Custom text field"

    def test_search_with_metadata_field(self):
        """Test search with metadata field extraction."""
        db = SyncMemoryDatabase()

        db.create(Record(
            data={"content": "Hello", "category": "greeting"},
            metadata={"score": 1.0},
            storage_id="rec_1"
        ))

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
        db = SyncMemoryDatabase()

        db.create(Record(
            data={"content": "High score"},
            metadata={"score": 0.9},
            storage_id="rec_1"
        ))
        db.create(Record(
            data={"content": "Low score"},
            metadata={"score": 0.3},
            storage_id="rec_2"
        ))

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
        db = SyncMemoryDatabase()

        db.create(Record(data={"content": "Value 1"}, storage_id="rec_1"))
        db.create(Record(data={"content": "Value 2"}, storage_id="rec_2"))

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
        db = AsyncMemoryDatabase()
        adapter = AsyncDataknobsBackendAdapter(db, name="test_async_db")
        assert adapter.name == "test_async_db"
        assert adapter._database == db

    @pytest.mark.asyncio
    async def test_get_value_full_record(self):
        """Test getting full record by ID (async)."""
        db = AsyncMemoryDatabase()
        record = Record(
            data={"content": "Hello async", "author": "Bob"},
            storage_id="rec_1"
        )
        await db.create(record)

        adapter = AsyncDataknobsBackendAdapter(db)
        result = await adapter.get_value("rec_1")

        assert result is not None
        assert result["content"] == "Hello async"
        assert result["author"] == "Bob"

    @pytest.mark.asyncio
    async def test_get_value_specific_field(self):
        """Test getting specific field from record (async)."""
        db = AsyncMemoryDatabase()
        record = Record(
            data={"content": "Hello", "author": "Charlie"},
            storage_id="rec_1"
        )
        await db.create(record)

        adapter = AsyncDataknobsBackendAdapter(db)
        assert await adapter.get_value("rec_1.author") == "Charlie"

    @pytest.mark.asyncio
    async def test_get_value_missing_record(self):
        """Test getting value for non-existent record (async)."""
        db = AsyncMemoryDatabase()
        adapter = AsyncDataknobsBackendAdapter(db)

        result = await adapter.get_value("missing", default="not_found")
        assert result == "not_found"

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search functionality (async)."""
        db = AsyncMemoryDatabase()

        await db.create(Record(
            data={"content": "Async search test"},
            metadata={"score": 1.0},
            storage_id="rec_1"
        ))
        await db.create(Record(
            data={"content": "Another test"},
            metadata={"score": 0.9},
            storage_id="rec_2"
        ))

        adapter = AsyncDataknobsBackendAdapter(db)
        results = await adapter.search("test")

        assert len(results) == 2
        assert all("test" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_with_k_limit(self):
        """Test search with k limit (async)."""
        db = AsyncMemoryDatabase()

        for i in range(10):
            await db.create(Record(
                data={"content": f"async item {i}"},
                metadata={"score": 1.0},
                storage_id=f"rec_{i}"
            ))

        adapter = AsyncDataknobsBackendAdapter(db)
        results = await adapter.search("async", k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_get_values(self):
        """Test async batch getting multiple values."""
        db = AsyncMemoryDatabase()

        await db.create(Record(data={"content": "Async 1"}, storage_id="rec_1"))
        await db.create(Record(data={"content": "Async 2"}, storage_id="rec_2"))

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
        db = SyncMemoryDatabase()
        adapter = DataknobsBackendAdapter(db)

        assert adapter.get_value("any_id") is None
        assert adapter.search("any_query") == []

    def test_search_with_no_results(self):
        """Test search that matches no records."""
        db = SyncMemoryDatabase()

        db.create(Record(
            data={"content": "Hello"},
            metadata={"score": 1.0},
            storage_id="rec_1"
        ))

        adapter = DataknobsBackendAdapter(db)
        results = adapter.search("NonExistent")

        assert results == []

    def test_record_with_missing_score(self):
        """Test handling records without score in metadata."""
        db = SyncMemoryDatabase()

        db.create(Record(
            data={"content": "No score"},
            storage_id="rec_1"
        ))

        adapter = DataknobsBackendAdapter(db)
        results = adapter.search("score")

        assert len(results) == 1
        # Should default to 1.0
        assert results[0]["score"] == 1.0

    def test_record_with_alternative_score_field(self):
        """Test using _score field as alternative to score."""
        db = SyncMemoryDatabase()

        db.create(Record(
            data={"content": "Alternative score"},
            metadata={"_score": 0.75},
            storage_id="rec_1"
        ))

        adapter = DataknobsBackendAdapter(db)
        results = adapter.search("score")

        assert len(results) == 1
        assert results[0]["score"] == 0.75

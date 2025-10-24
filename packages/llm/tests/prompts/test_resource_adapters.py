"""Unit tests for resource adapters."""

import pytest
from dataknobs_llm.prompts import (
    ResourceAdapter,
    AsyncResourceAdapter,
    ResourceAdapterBase,
    BaseSearchLogic,
)


# Concrete implementations for testing

class MockSyncAdapter(ResourceAdapter):
    """Mock synchronous adapter for testing."""

    def __init__(self, data: dict, name: str = "mock_sync"):
        super().__init__(name=name)
        self._data = data

    def get_value(self, key: str, default=None, context=None):
        """Get value from mock data."""
        return self._data.get(key, default)

    def search(self, query: str, k: int = 5, filters=None, **kwargs):
        """Mock search that returns items containing query."""
        results = []
        for key, value in self._data.items():
            if query.lower() in str(value).lower():
                results.append({
                    "content": value,
                    "key": key,
                    "score": 1.0
                })
                if len(results) >= k:
                    break
        return results


class MockAsyncAdapter(AsyncResourceAdapter):
    """Mock asynchronous adapter for testing."""

    def __init__(self, data: dict, name: str = "mock_async"):
        super().__init__(name=name)
        self._data = data

    async def get_value(self, key: str, default=None, context=None):
        """Get value from mock data (async)."""
        return self._data.get(key, default)

    async def search(self, query: str, k: int = 5, filters=None, **kwargs):
        """Mock search that returns items containing query (async)."""
        results = []
        for key, value in self._data.items():
            if query.lower() in str(value).lower():
                results.append({
                    "content": value,
                    "key": key,
                    "score": 1.0
                })
                if len(results) >= k:
                    break
        return results


class TestResourceAdapterBase:
    """Test suite for ResourceAdapterBase."""

    def test_initialization(self):
        """Test adapter base initialization."""
        adapter = MockSyncAdapter({}, name="test_adapter")
        assert adapter.name == "test_adapter"

    def test_default_name(self):
        """Test adapter uses default name if not provided."""
        adapter = MockSyncAdapter({})
        assert adapter.name == "mock_sync"

    def test_get_metadata(self):
        """Test get_metadata returns correct information."""
        adapter = MockSyncAdapter(
            {},
            name="test"
        )
        metadata = adapter.get_metadata()
        assert metadata["name"] == "test"
        assert metadata["type"] == "sync"
        assert metadata["class"] == "MockSyncAdapter"

    def test_repr(self):
        """Test string representation of adapter."""
        adapter = MockSyncAdapter({}, name="test")
        repr_str = repr(adapter)
        assert "MockSyncAdapter" in repr_str
        assert "test" in repr_str
        assert "sync" in repr_str


class TestResourceAdapter:
    """Test suite for synchronous ResourceAdapter."""

    def test_is_async_false(self):
        """Test that sync adapters return False for is_async()."""
        adapter = MockSyncAdapter({})
        assert adapter.is_async() is False

    def test_get_value(self):
        """Test getting values from adapter."""
        adapter = MockSyncAdapter({"name": "Alice", "age": 30})
        assert adapter.get_value("name") == "Alice"
        assert adapter.get_value("age") == 30

    def test_get_value_default(self):
        """Test default value for missing keys."""
        adapter = MockSyncAdapter({"name": "Alice"})
        assert adapter.get_value("age", default=0) == 0

    def test_get_value_none_default(self):
        """Test None default for missing keys."""
        adapter = MockSyncAdapter({})
        assert adapter.get_value("missing") is None

    def test_search(self):
        """Test search functionality."""
        adapter = MockSyncAdapter({
            "user1": "Alice in NYC",
            "user2": "Bob in LA",
            "user3": "Alice in Paris"
        })
        results = adapter.search("Alice")
        assert len(results) == 2
        assert all("Alice" in r["content"] for r in results)

    def test_search_with_k_limit(self):
        """Test search respects k parameter."""
        adapter = MockSyncAdapter({
            f"item{i}": f"test item {i}"
            for i in range(10)
        })
        results = adapter.search("test", k=3)
        assert len(results) == 3

    def test_batch_get_values(self):
        """Test batch getting multiple values."""
        adapter = MockSyncAdapter({
            "name": "Alice",
            "age": 30,
            "city": "NYC"
        })
        results = adapter.batch_get_values(["name", "age", "country"], default="Unknown")
        assert results["name"] == "Alice"
        assert results["age"] == 30
        assert results["country"] == "Unknown"


class TestAsyncResourceAdapter:
    """Test suite for asynchronous AsyncResourceAdapter."""

    def test_is_async_true(self):
        """Test that async adapters return True for is_async()."""
        adapter = MockAsyncAdapter({})
        assert adapter.is_async() is True

    def test_get_metadata_shows_async(self):
        """Test that metadata indicates async type."""
        adapter = MockAsyncAdapter({})
        metadata = adapter.get_metadata()
        assert metadata["type"] == "async"

    @pytest.mark.asyncio
    async def test_get_value(self):
        """Test getting values from async adapter."""
        adapter = MockAsyncAdapter({"name": "Alice", "age": 30})
        assert await adapter.get_value("name") == "Alice"
        assert await adapter.get_value("age") == 30

    @pytest.mark.asyncio
    async def test_get_value_default(self):
        """Test default value for missing keys."""
        adapter = MockAsyncAdapter({"name": "Alice"})
        assert await adapter.get_value("age", default=0) == 0

    @pytest.mark.asyncio
    async def test_search(self):
        """Test async search functionality."""
        adapter = MockAsyncAdapter({
            "user1": "Alice in NYC",
            "user2": "Bob in LA",
            "user3": "Alice in Paris"
        })
        results = await adapter.search("Alice")
        assert len(results) == 2
        assert all("Alice" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_batch_get_values(self):
        """Test async batch getting multiple values."""
        adapter = MockAsyncAdapter({
            "name": "Alice",
            "age": 30,
            "city": "NYC"
        })
        results = await adapter.batch_get_values(["name", "age", "country"], default="Unknown")
        assert results["name"] == "Alice"
        assert results["age"] == 30
        assert results["country"] == "Unknown"


class TestBaseSearchLogic:
    """Test suite for BaseSearchLogic helper class."""

    def test_format_search_result_from_string(self):
        """Test formatting a string result."""
        result = BaseSearchLogic.format_search_result(
            "Hello world",
            score=0.95
        )
        assert result["content"] == "Hello world"
        assert result["score"] == 0.95

    def test_format_search_result_from_dict(self):
        """Test formatting a dict result."""
        result = BaseSearchLogic.format_search_result(
            {"content": "Hello", "author": "Alice"},
            score=0.8
        )
        assert result["content"] == "Hello"
        assert result["score"] == 0.8
        assert result["metadata"]["author"] == "Alice"

    def test_format_search_result_dict_with_text_key(self):
        """Test formatting dict with 'text' key instead of 'content'."""
        result = BaseSearchLogic.format_search_result(
            {"text": "Hello world"}
        )
        assert result["content"] == "Hello world"

    def test_filter_results_by_score(self):
        """Test filtering results by minimum score."""
        results = [
            {"content": "A", "score": 0.9},
            {"content": "B", "score": 0.5},
            {"content": "C", "score": 0.8},
        ]
        filtered = BaseSearchLogic.filter_results(results, min_score=0.7)
        assert len(filtered) == 2
        assert all(r["score"] >= 0.7 for r in filtered)

    def test_filter_results_by_field(self):
        """Test filtering results by field value."""
        results = [
            {"content": "A", "type": "user"},
            {"content": "B", "type": "system"},
            {"content": "C", "type": "user"},
        ]
        filtered = BaseSearchLogic.filter_results(
            results,
            filters={"type": "user"}
        )
        assert len(filtered) == 2
        assert all(r["type"] == "user" for r in filtered)

    def test_filter_results_by_metadata_field(self):
        """Test filtering results by metadata field."""
        results = [
            {"content": "A", "metadata": {"category": "tech"}},
            {"content": "B", "metadata": {"category": "sports"}},
            {"content": "C", "metadata": {"category": "tech"}},
        ]
        filtered = BaseSearchLogic.filter_results(
            results,
            filters={"category": "tech"}
        )
        assert len(filtered) == 2

    def test_deduplicate_results(self):
        """Test deduplication of results."""
        results = [
            {"content": "A", "score": 0.9},
            {"content": "B", "score": 0.8},
            {"content": "A", "score": 0.7},  # Duplicate
            {"content": "C", "score": 0.6},
        ]
        deduped = BaseSearchLogic.deduplicate_results(results)
        assert len(deduped) == 3
        contents = [r["content"] for r in deduped]
        assert contents == ["A", "B", "C"]
        # Should keep first occurrence (higher score)
        assert deduped[0]["score"] == 0.9

    def test_deduplicate_by_custom_key(self):
        """Test deduplication by custom key."""
        results = [
            {"content": "Message 1", "id": "1"},
            {"content": "Message 2", "id": "2"},
            {"content": "Message 3", "id": "1"},  # Duplicate ID
        ]
        deduped = BaseSearchLogic.deduplicate_results(results, key="id")
        assert len(deduped) == 2
        ids = [r["id"] for r in deduped]
        assert ids == ["1", "2"]

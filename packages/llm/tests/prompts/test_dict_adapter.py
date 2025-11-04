"""Unit tests for dictionary resource adapters."""

import pytest
from dataknobs_llm.prompts import (
    DictResourceAdapter,
    AsyncDictResourceAdapter,
)


class TestDictResourceAdapter:
    """Test suite for DictResourceAdapter (sync)."""

    def test_initialization(self):
        """Test basic initialization."""
        data = {"key": "value"}
        adapter = DictResourceAdapter(data, name="test")
        assert adapter.name == "test"
        assert adapter._data == data

    def test_initialization_with_case_sensitivity(self):
        """Test initialization with case sensitivity option."""
        data = {"key": "value"}
        adapter = DictResourceAdapter(data, case_sensitive=True)
        assert adapter._case_sensitive is True

        adapter2 = DictResourceAdapter(data, case_sensitive=False)
        assert adapter2._case_sensitive is False

    def test_get_value_simple(self):
        """Test getting simple top-level values."""
        data = {
            "name": "Alice",
            "age": 30,
            "active": True
        }
        adapter = DictResourceAdapter(data)

        assert adapter.get_value("name") == "Alice"
        assert adapter.get_value("age") == 30
        assert adapter.get_value("active") is True

    def test_get_value_nested(self):
        """Test getting nested values with dot notation."""
        data = {
            "user": {
                "name": "Bob",
                "profile": {
                    "age": 25,
                    "city": "NYC"
                }
            }
        }
        adapter = DictResourceAdapter(data)

        assert adapter.get_value("user.name") == "Bob"
        assert adapter.get_value("user.profile.age") == 25
        assert adapter.get_value("user.profile.city") == "NYC"

    def test_get_value_default(self):
        """Test default value for missing keys."""
        data = {"key1": "value1"}
        adapter = DictResourceAdapter(data)

        assert adapter.get_value("missing", default="default") == "default"
        assert adapter.get_value("missing") is None

    def test_get_value_nested_missing(self):
        """Test default value for missing nested keys."""
        data = {"user": {"name": "Alice"}}
        adapter = DictResourceAdapter(data)

        # Missing nested path
        assert adapter.get_value("user.age", default=0) == 0

        # Missing parent path
        assert adapter.get_value("settings.theme", default="dark") == "dark"

    def test_search_case_insensitive(self):
        """Test case-insensitive search (default)."""
        data = {
            "user1": "Alice in Wonderland",
            "user2": "Bob in NYC",
            "user3": "alice in Paris"
        }
        adapter = DictResourceAdapter(data, case_sensitive=False)

        results = adapter.search("alice")
        assert len(results) == 2
        assert all("alice" in r["content"].lower() for r in results)

    def test_search_case_sensitive(self):
        """Test case-sensitive search."""
        data = {
            "user1": "Alice in Wonderland",
            "user2": "Bob in NYC",
            "user3": "alice in Paris"
        }
        adapter = DictResourceAdapter(data, case_sensitive=True)

        results = adapter.search("Alice")
        assert len(results) == 1
        assert "Alice" in results[0]["content"]

        results = adapter.search("alice")
        assert len(results) == 1
        assert "alice" in results[0]["content"]

    def test_search_with_k_limit(self):
        """Test search with k parameter limiting results."""
        data = {f"item{i}": f"test value {i}" for i in range(10)}
        adapter = DictResourceAdapter(data)

        results = adapter.search("test", k=3)
        assert len(results) == 3

    def test_search_scoring(self):
        """Test search scoring (exact vs contains)."""
        data = {
            "exact": "alice",
            "contains": "alice in wonderland"
        }
        adapter = DictResourceAdapter(data, case_sensitive=False)

        results = adapter.search("alice")
        assert len(results) == 2

        # Find exact match
        exact_result = next(r for r in results if r["content"] == "alice")
        assert exact_result["score"] == 1.0

        # Find contains match
        contains_result = next(r for r in results if "wonderland" in r["content"])
        assert contains_result["score"] == 0.8

    def test_search_includes_key(self):
        """Test that search results include the key."""
        data = {
            "user.name": "Alice",
            "user.age": "30"
        }
        adapter = DictResourceAdapter(data)

        results = adapter.search("Alice")
        assert len(results) == 1
        assert results[0]["key"] == "user.name"

    def test_search_nested_data(self):
        """Test searching through nested dictionary data."""
        data = {
            "users": {
                "alice": {
                    "email": "alice@example.com",
                    "role": "admin"
                },
                "bob": {
                    "email": "bob@example.com",
                    "role": "user"
                }
            }
        }
        adapter = DictResourceAdapter(data)

        results = adapter.search("alice")
        assert len(results) >= 1

        # Search should find nested values
        results = adapter.search("admin")
        assert len(results) >= 1

    def test_search_with_min_score(self):
        """Test search with minimum score filter."""
        data = {
            "exact": "test",
            "contains": "this is a test value"
        }
        adapter = DictResourceAdapter(data)

        # Without min_score, both results returned
        results = adapter.search("test")
        assert len(results) == 2

        # With min_score=1.0, only exact match
        results = adapter.search("test", min_score=1.0)
        assert len(results) == 1
        assert results[0]["content"] == "test"

    def test_search_with_deduplication(self):
        """Test search with deduplication enabled."""
        data = {
            "item1": "duplicate value",
            "item2": "duplicate value",
            "item3": "unique value"
        }
        adapter = DictResourceAdapter(data)

        # Without deduplication
        results = adapter.search("value")
        assert len(results) == 3

        # With deduplication
        results = adapter.search("value", deduplicate=True)
        assert len(results) == 2  # duplicate and unique

    def test_flatten_dict(self):
        """Test dictionary flattening."""
        adapter = DictResourceAdapter({})

        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }

        flattened = adapter._flatten_dict(nested)
        assert flattened == {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3
        }

    def test_batch_get_values(self):
        """Test batch getting multiple values."""
        data = {
            "name": "Alice",
            "age": 30,
            "city": "NYC"
        }
        adapter = DictResourceAdapter(data)

        results = adapter.batch_get_values(
            ["name", "age", "country"],
            default="Unknown"
        )

        assert results["name"] == "Alice"
        assert results["age"] == 30
        assert results["country"] == "Unknown"


class TestAsyncDictResourceAdapter:
    """Test suite for AsyncDictResourceAdapter."""

    def test_initialization(self):
        """Test basic initialization."""
        data = {"key": "value"}
        adapter = AsyncDictResourceAdapter(data, name="test_async")
        assert adapter.name == "test_async"
        assert adapter._data == data

    @pytest.mark.asyncio
    async def test_get_value_simple(self):
        """Test getting simple values (async)."""
        data = {"name": "Alice", "age": 30}
        adapter = AsyncDictResourceAdapter(data)

        assert await adapter.get_value("name") == "Alice"
        assert await adapter.get_value("age") == 30

    @pytest.mark.asyncio
    async def test_get_value_nested(self):
        """Test getting nested values (async)."""
        data = {
            "user": {
                "name": "Bob",
                "profile": {
                    "city": "NYC"
                }
            }
        }
        adapter = AsyncDictResourceAdapter(data)

        assert await adapter.get_value("user.name") == "Bob"
        assert await adapter.get_value("user.profile.city") == "NYC"

    @pytest.mark.asyncio
    async def test_get_value_default(self):
        """Test default value for missing keys (async)."""
        data = {"key1": "value1"}
        adapter = AsyncDictResourceAdapter(data)

        assert await adapter.get_value("missing", default="default") == "default"
        assert await adapter.get_value("missing") is None

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search (async)."""
        data = {
            "user1": "Alice in Wonderland",
            "user2": "Bob in NYC",
            "user3": "Alice in Paris"
        }
        adapter = AsyncDictResourceAdapter(data)

        results = await adapter.search("Alice")
        assert len(results) == 2
        assert all("Alice" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_search_with_k_limit(self):
        """Test search with k limit (async)."""
        data = {f"item{i}": f"test value {i}" for i in range(10)}
        adapter = AsyncDictResourceAdapter(data)

        results = await adapter.search("test", k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_get_values(self):
        """Test async batch getting multiple values."""
        data = {
            "name": "Alice",
            "age": 30,
            "city": "NYC"
        }
        adapter = AsyncDictResourceAdapter(data)

        results = await adapter.batch_get_values(
            ["name", "age", "country"],
            default="Unknown"
        )

        assert results["name"] == "Alice"
        assert results["age"] == 30
        assert results["country"] == "Unknown"


class TestDictAdapterEdgeCases:
    """Test edge cases for dict adapters."""

    def test_empty_dictionary(self):
        """Test adapter with empty dictionary."""
        adapter = DictResourceAdapter({})

        assert adapter.get_value("any_key") is None
        assert adapter.search("any_query") == []

    def test_none_values(self):
        """Test handling of None values in dictionary."""
        data = {"key": None}
        adapter = DictResourceAdapter(data)

        # Getting None value should return None
        assert adapter.get_value("key") is None

        # With explicit default, None is still None (not replaced)
        result = adapter.get_value("key", default="default")
        # Since key exists with None value, return None
        assert result is None

    def test_numeric_values_in_search(self):
        """Test searching for numeric values."""
        data = {
            "age": 30,
            "score": 95.5,
            "count": 0
        }
        adapter = DictResourceAdapter(data)

        results = adapter.search("30")
        assert len(results) == 1
        assert "30" in results[0]["content"]

        results = adapter.search("95.5")
        assert len(results) == 1

    def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        data = {
            "user-name": "Alice",
            "user_email": "alice@example.com",
            "user/id": "123"
        }
        adapter = DictResourceAdapter(data)

        assert adapter.get_value("user-name") == "Alice"
        assert adapter.get_value("user_email") == "alice@example.com"
        assert adapter.get_value("user/id") == "123"

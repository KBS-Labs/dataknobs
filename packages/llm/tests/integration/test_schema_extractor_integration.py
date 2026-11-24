"""Integration tests for SchemaExtractor with real LLM providers.

A real model answers one question the scripted tests in
``test_schema_extractor.py`` cannot: whether it follows the extraction prompt.
That takes one test per shape of schema. What the extractor does with a
response (parsing, validation, confidence) is tested there.

These tests require a running Ollama instance and are skipped by default.
Run with: TEST_OLLAMA=true uv run pytest packages/llm/tests/integration/test_schema_extractor_integration.py
"""

import os

import pytest

from dataknobs_llm.extraction import SchemaExtractor

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TEST_OLLAMA", "").lower() != "true",
        reason="Ollama integration tests require TEST_OLLAMA=true and a running Ollama instance",
    ),
    pytest.mark.integration,
    pytest.mark.ollama,
]


class TestSchemaExtractorOllamaIntegration:
    """Integration tests for SchemaExtractor with Ollama."""

    @pytest.mark.asyncio
    async def test_extract_simple_fields(self, ollama_extractor_config: dict) -> None:
        """Test extracting simple string and integer fields."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
            },
            "required": ["name"],
        }

        result = await extractor.extract(
            text="My name is Alice and I am 28 years old.",
            schema=schema,
        )

        await extractor._provider.close()

        assert result.data.get("name") == "Alice"
        assert result.data.get("age") == 28
        assert result.confidence >= 0.8
        assert result.is_confident

    @pytest.mark.asyncio
    async def test_extract_with_enum_constraint(self, ollama_extractor_config: dict) -> None:
        """Test extracting with enum constraints."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "size": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "description": "T-shirt size",
                },
                "quantity": {"type": "integer", "description": "Number of items"},
            },
            "required": ["size", "quantity"],
        }

        result = await extractor.extract(
            text="I'd like to order 3 large t-shirts please.",
            schema=schema,
        )

        await extractor._provider.close()

        assert result.data.get("size") == "large"
        assert result.data.get("quantity") == 3
        assert result.is_confident

    @pytest.mark.asyncio
    async def test_extract_nested_object(self, ollama_extractor_config: dict) -> None:
        """Test extracting a nested object and an array."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "email"],
                },
                "hobbies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of hobbies",
                },
            },
            "required": ["user", "hobbies"],
        }

        result = await extractor.extract(
            text="I'm John, my email is john@example.com, and I enjoy hiking and reading.",
            schema=schema,
        )

        await extractor._provider.close()

        assert result.data["user"]["name"] == "John"
        assert result.data["user"]["email"] == "john@example.com"
        assert sorted(h.lower() for h in result.data["hobbies"]) == ["hiking", "reading"]
        assert result.is_confident

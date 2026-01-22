"""Integration tests for SchemaExtractor with real LLM providers.

These tests require a running Ollama instance and are skipped by default.
Run with: TEST_OLLAMA=true uv run pytest packages/llm/tests/integration/test_schema_extractor_integration.py
"""

import os

import pytest

from dataknobs_llm.extraction import ExtractionResult, SchemaExtractor

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("TEST_OLLAMA", "").lower() == "true",
        reason="Ollama integration tests require TEST_OLLAMA=true and a running Ollama instance",
    ),
    pytest.mark.integration,
    pytest.mark.ollama,
]


class TestSchemaExtractorOllamaIntegration:
    """Integration tests for SchemaExtractor with Ollama."""

    @pytest.mark.asyncio
    async def test_extract_simple_fields(
        self, ollama_extractor_config: dict
    ) -> None:
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
    async def test_extract_with_enum_constraint(
        self, ollama_extractor_config: dict
    ) -> None:
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
    async def test_extract_array_field(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test extracting array fields."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "hobbies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of hobbies",
                },
            },
            "required": ["name"],
        }

        result = await extractor.extract(
            text="I'm Bob and I enjoy hiking, reading, and cooking.",
            schema=schema,
        )

        await extractor._provider.close()

        assert result.data.get("name") == "Bob"
        hobbies = result.data.get("hobbies", [])
        assert isinstance(hobbies, list)
        assert len(hobbies) >= 2  # Should extract at least some hobbies
        assert result.is_confident

    @pytest.mark.asyncio
    async def test_extract_bot_configuration(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test extracting bot configuration (ConfigBot use case)."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "domain_id": {
                    "type": "string",
                    "description": "Unique bot identifier (slug format)",
                },
                "domain_name": {
                    "type": "string",
                    "description": "Human-readable bot name",
                },
                "bot_description": {
                    "type": "string",
                    "description": "Description of what the bot does",
                },
                "persona": {
                    "type": "string",
                    "enum": ["tutor", "assistant", "expert"],
                    "description": "Bot personality type",
                },
            },
            "required": ["domain_id", "domain_name"],
        }

        result = await extractor.extract(
            text="I want to create a math tutor bot called math-helper. "
            "It should help students learn algebra and geometry.",
            schema=schema,
            context={"stage": "configure", "prompt": "Configure your new bot"},
        )

        await extractor._provider.close()

        assert result.data.get("domain_id") is not None
        assert result.data.get("domain_name") is not None
        # Verify reasonable values
        domain_id = result.data.get("domain_id", "")
        assert "math" in domain_id.lower() or "helper" in domain_id.lower()
        assert result.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_extract_partial_information(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test extraction when only partial information is provided."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "domain_id": {"type": "string"},
                "domain_name": {"type": "string"},
                "bot_description": {"type": "string"},
            },
            "required": ["domain_id", "domain_name"],
        }

        # Only provides name, not description
        result = await extractor.extract(
            text="I want to call my bot study-buddy",
            schema=schema,
        )

        await extractor._provider.close()

        # Should extract what's available
        assert result.data.get("domain_id") is not None or result.data.get("domain_name") is not None
        # Confidence should reflect partial extraction
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_extract_nested_object(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test extracting nested objects."""
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
                },
                "preferences": {
                    "type": "object",
                    "properties": {
                        "theme": {"type": "string", "enum": ["light", "dark"]},
                        "language": {"type": "string"},
                    },
                },
            },
        }

        result = await extractor.extract(
            text="Set up account for John at john@example.com with dark theme in English",
            schema=schema,
        )

        await extractor._provider.close()

        # Check nested extraction
        user = result.data.get("user", {})
        if user:
            assert user.get("name") == "John" or "john" in str(user.get("name", "")).lower()

        prefs = result.data.get("preferences", {})
        if prefs:
            assert prefs.get("theme") == "dark"

    @pytest.mark.asyncio
    async def test_extract_with_context(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test that context influences extraction."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "description": "User's intent"},
                "entities": {"type": "array", "items": {"type": "string"}},
            },
        }

        result = await extractor.extract(
            text="Can you help me with Python?",
            schema=schema,
            context={
                "stage": "intent_detection",
                "prompt": "Identify the user's programming intent and any mentioned technologies",
            },
        )

        await extractor._provider.close()

        assert result.data.get("intent") is not None
        # Should mention help or assistance
        intent = result.data.get("intent", "").lower()
        assert any(word in intent for word in ["help", "assist", "learn", "question"])

    @pytest.mark.asyncio
    async def test_handles_ambiguous_input(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test handling of ambiguous or unclear input."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["product", "quantity"],
        }

        # Ambiguous input
        result = await extractor.extract(
            text="maybe some stuff",
            schema=schema,
        )

        await extractor._provider.close()

        # Should have low confidence or errors for ambiguous input
        # Either extraction fails or confidence is low
        if result.data:
            # If it extracted something, it might have errors or low confidence
            pass  # Some models might still extract
        else:
            assert result.confidence < 0.8 or len(result.errors) > 0


class TestSchemaExtractorExtractionQuality:
    """Tests focused on extraction quality and edge cases."""

    @pytest.mark.asyncio
    async def test_handles_json_in_natural_language(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test that extractor works when user mentions JSON-like content."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "config_value": {"type": "string"},
                "description": {"type": "string"},
            },
        }

        # User mentions JSON in their text
        result = await extractor.extract(
            text='Set the config_value to "production" for the deployment environment',
            schema=schema,
        )

        await extractor._provider.close()

        assert result.data.get("config_value") == "production"

    @pytest.mark.asyncio
    async def test_maintains_data_types(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test that extracted data maintains correct types."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "active": {"type": "boolean"},
                "name": {"type": "string"},
            },
        }

        result = await extractor.extract(
            text="Order 5 items at $19.99 each, make it active for user John",
            schema=schema,
        )

        await extractor._provider.close()

        if result.data.get("count") is not None:
            assert isinstance(result.data["count"], int)
        if result.data.get("price") is not None:
            assert isinstance(result.data["price"], (int, float))
        if result.data.get("active") is not None:
            assert isinstance(result.data["active"], bool)
        if result.data.get("name") is not None:
            assert isinstance(result.data["name"], str)

    @pytest.mark.asyncio
    async def test_extraction_is_deterministic(
        self, ollama_extractor_config: dict
    ) -> None:
        """Test that extraction with temperature=0 is reasonably consistent."""
        extractor = SchemaExtractor.from_config(ollama_extractor_config)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "color": {"type": "string"},
            },
            "required": ["name", "color"],
        }

        text = "The product is called Rainbow and it's blue."

        # Run extraction twice
        result1 = await extractor.extract(text=text, schema=schema)
        result2 = await extractor.extract(text=text, schema=schema)

        await extractor._provider.close()

        # With temperature=0, results should be identical or very similar
        assert result1.data.get("name") == result2.data.get("name")
        assert result1.data.get("color") == result2.data.get("color")

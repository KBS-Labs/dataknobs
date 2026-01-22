"""Tests for SchemaExtractor."""

import pytest

from dataknobs_llm.extraction import ExtractionResult, SchemaExtractor
from dataknobs_llm.llm.providers.echo import EchoProvider


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values for ExtractionResult."""
        result = ExtractionResult()

        assert result.data == {}
        assert result.confidence == 0.0
        assert result.errors == []
        assert result.raw_response == ""
        assert result.is_confident is False

    def test_confident_extraction(self) -> None:
        """Test is_confident with high confidence and no errors."""
        result = ExtractionResult(
            data={"name": "Alice"},
            confidence=0.9,
            errors=[],
        )

        assert result.is_confident is True

    def test_not_confident_low_confidence(self) -> None:
        """Test is_confident returns False with low confidence."""
        result = ExtractionResult(
            data={"name": "Alice"},
            confidence=0.5,
            errors=[],
        )

        assert result.is_confident is False

    def test_not_confident_with_errors(self) -> None:
        """Test is_confident returns False with errors."""
        result = ExtractionResult(
            data={"name": "Alice"},
            confidence=0.9,
            errors=["Missing required field: age"],
        )

        assert result.is_confident is False

    def test_confidence_threshold(self) -> None:
        """Test confidence threshold is exactly 0.8."""
        # At threshold
        result_at = ExtractionResult(data={"x": 1}, confidence=0.8, errors=[])
        assert result_at.is_confident is True

        # Just below threshold
        result_below = ExtractionResult(data={"x": 1}, confidence=0.79, errors=[])
        assert result_below.is_confident is False


class TestSchemaExtractor:
    """Tests for SchemaExtractor."""

    @pytest.fixture
    def echo_provider(self) -> EchoProvider:
        """Create an EchoProvider for testing."""
        return EchoProvider(
            {"provider": "echo", "model": "test", "options": {"echo_prefix": ""}}
        )

    @pytest.fixture
    def extractor(self, echo_provider: EchoProvider) -> SchemaExtractor:
        """Create a SchemaExtractor with EchoProvider."""
        return SchemaExtractor(provider=echo_provider)

    @pytest.fixture
    def simple_schema(self) -> dict:
        """Create a simple test schema."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

    @pytest.mark.asyncio
    async def test_extract_valid_json(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extracting valid JSON from response."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Script a JSON response
        echo_provider.set_responses(['{"name": "Alice"}'])

        result = await extractor.extract(
            text="My name is Alice",
            schema=schema,
        )

        assert result.data == {"name": "Alice"}
        assert result.confidence >= 0.8
        assert result.is_confident

    @pytest.mark.asyncio
    async def test_extract_with_required_fields(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with required fields."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        }

        # Response has all required fields
        echo_provider.set_responses(['{"name": "Alice", "email": "alice@example.com"}'])

        result = await extractor.extract(text="I'm Alice, email alice@example.com", schema=schema)

        assert result.data["name"] == "Alice"
        assert result.data["email"] == "alice@example.com"
        assert result.is_confident

    @pytest.mark.asyncio
    async def test_extract_missing_required_field(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with missing required field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        }

        # Response missing required email
        echo_provider.set_responses(['{"name": "Alice"}'])

        result = await extractor.extract(text="I'm Alice", schema=schema)

        assert result.data == {"name": "Alice"}
        assert "Missing required field: email" in result.errors
        assert not result.is_confident

    @pytest.mark.asyncio
    async def test_extract_enum_validation(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with enum constraint."""
        schema = {
            "type": "object",
            "properties": {
                "size": {"type": "string", "enum": ["small", "medium", "large"]},
            },
        }

        # Invalid enum value
        echo_provider.set_responses(['{"size": "huge"}'])

        result = await extractor.extract(text="I want a huge pizza", schema=schema)

        assert result.data == {"size": "huge"}
        assert any("Invalid value for size" in e for e in result.errors)
        assert not result.is_confident

    @pytest.mark.asyncio
    async def test_extract_valid_enum(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with valid enum value."""
        schema = {
            "type": "object",
            "properties": {
                "size": {"type": "string", "enum": ["small", "medium", "large"]},
            },
        }

        echo_provider.set_responses(['{"size": "large"}'])

        result = await extractor.extract(text="I want a large pizza", schema=schema)

        assert result.data == {"size": "large"}
        assert result.errors == []
        assert result.is_confident

    @pytest.mark.asyncio
    async def test_extract_empty_response(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with empty JSON response."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        echo_provider.set_responses(["{}"])

        result = await extractor.extract(text="Hello", schema=schema)

        assert result.data == {}
        assert result.confidence == 0.0
        assert not result.is_confident

    @pytest.mark.asyncio
    async def test_extract_malformed_json(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with malformed JSON that can be repaired."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Malformed JSON (missing closing brace)
        echo_provider.set_responses(['{"name": "Alice"'])

        result = await extractor.extract(text="My name is Alice", schema=schema)

        # JSONExtractor should repair it
        assert result.data == {"name": "Alice"}
        assert any("repaired" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_extract_no_json(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction when response has no JSON."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        echo_provider.set_responses(["I couldn't understand that."])

        result = await extractor.extract(text="xyz", schema=schema)

        assert result.data == {}
        assert any("Could not parse JSON" in e for e in result.errors)
        assert not result.is_confident

    @pytest.mark.asyncio
    async def test_extract_json_in_text(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction when JSON is embedded in text."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # JSON embedded in surrounding text
        echo_provider.set_responses(['Here is the data: {"name": "Bob"} Hope this helps!'])

        result = await extractor.extract(text="I'm Bob", schema=schema)

        assert result.data == {"name": "Bob"}
        assert result.is_confident

    @pytest.mark.asyncio
    async def test_extract_with_context(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with context passed to prompt."""
        schema = {"type": "object", "properties": {"intent": {"type": "string"}}}

        echo_provider.set_responses(['{"intent": "greeting"}'])

        result = await extractor.extract(
            text="Hello there",
            schema=schema,
            context={"stage": "welcome", "prompt": "What does the user want?"},
        )

        assert result.data == {"intent": "greeting"}
        # Verify the prompt was built with context
        last_call = echo_provider.get_last_call()
        assert last_call is not None

    @pytest.mark.asyncio
    async def test_extract_with_model_override(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with model override."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        echo_provider.set_responses(['{"name": "Test"}'])

        result = await extractor.extract(
            text="test",
            schema=schema,
            model="different-model",
        )

        assert result.data == {"name": "Test"}
        # Verify model override was passed
        last_call = echo_provider.get_last_call()
        assert last_call is not None
        assert last_call.get("config_overrides", {}).get("model") == "different-model"

    @pytest.mark.asyncio
    async def test_confidence_calculation_full(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test confidence calculation with full data."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "age"],
        }

        # All fields present
        echo_provider.set_responses(
            ['{"name": "Alice", "age": 30, "email": "alice@test.com"}']
        )

        result = await extractor.extract(text="Alice, 30, alice@test.com", schema=schema)

        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_confidence_calculation_partial(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test confidence calculation with partial data."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        # Only one required field present (should fail validation)
        echo_provider.set_responses(['{"name": "Alice"}'])

        result = await extractor.extract(text="Alice", schema=schema)

        # Has errors so confidence is capped at 0.5
        assert result.confidence == 0.5
        assert not result.is_confident


class TestSchemaExtractorFromConfig:
    """Tests for SchemaExtractor.from_config()."""

    def test_from_config_echo_provider(self) -> None:
        """Test creating extractor with echo provider config."""
        config = {
            "provider": "echo",
            "model": "test-model",
            "temperature": 0.0,
        }

        extractor = SchemaExtractor.from_config(config)

        assert extractor._provider is not None
        assert isinstance(extractor._provider, EchoProvider)

    def test_from_config_missing_provider(self) -> None:
        """Test error when provider is missing."""
        config = {"model": "test-model"}

        with pytest.raises(ValueError, match="provider is required"):
            SchemaExtractor.from_config(config)

    def test_from_config_unsupported_provider(self) -> None:
        """Test error when provider is unsupported."""
        config = {"provider": "unsupported", "model": "test"}

        with pytest.raises(ValueError, match="Unsupported provider"):
            SchemaExtractor.from_config(config)

    def test_from_config_custom_prompt(self) -> None:
        """Test creating extractor with custom extraction prompt."""
        custom_prompt = "Custom extraction: {schema} {context} {text}"
        config = {
            "provider": "echo",
            "model": "test",
            "extraction_prompt": custom_prompt,
        }

        extractor = SchemaExtractor.from_config(config)

        assert extractor._extraction_prompt == custom_prompt

    @pytest.mark.asyncio
    async def test_from_config_functional(self) -> None:
        """Test that from_config creates a functional extractor."""
        config = {
            "provider": "echo",
            "model": "test",
            "options": {"echo_prefix": ""},
        }

        extractor = SchemaExtractor.from_config(config)

        # Set up response
        extractor._provider.set_responses(['{"value": 42}'])  # type: ignore

        result = await extractor.extract(
            text="The answer is 42",
            schema={"type": "object", "properties": {"value": {"type": "integer"}}},
        )

        assert result.data == {"value": 42}
        assert result.is_confident

    def test_from_env_config_is_alias_for_from_config(self) -> None:
        """Test that from_env_config() is an alias for from_config().

        WizardReasoning.from_config() uses from_env_config() to create
        the SchemaExtractor, so this verifies the expected API.
        """
        config = {
            "provider": "echo",
            "model": "test-model",
            "temperature": 0.1,
        }

        extractor = SchemaExtractor.from_env_config(config)

        assert extractor._provider is not None
        assert isinstance(extractor._provider, EchoProvider)


class TestSchemaExtractorProviderCreation:
    """Tests for provider creation via _create_provider()."""

    def test_create_provider_ollama(self) -> None:
        """Test creating Ollama provider."""
        from dataknobs_llm.llm.providers.ollama import OllamaProvider

        config = {
            "provider": "ollama",
            "model": "qwen3-coder",
            "temperature": 0.0,
        }

        provider = SchemaExtractor._create_provider(config)

        assert isinstance(provider, OllamaProvider)

    def test_create_provider_openai(self) -> None:
        """Test creating OpenAI provider."""
        from dataknobs_llm.llm.providers.openai import OpenAIProvider

        config = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.0,
        }

        provider = SchemaExtractor._create_provider(config)

        assert isinstance(provider, OpenAIProvider)

    def test_create_provider_anthropic(self) -> None:
        """Test creating Anthropic provider."""
        from dataknobs_llm.llm.providers.anthropic import AnthropicProvider

        config = {
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307",
            "temperature": 0.0,
        }

        provider = SchemaExtractor._create_provider(config)

        assert isinstance(provider, AnthropicProvider)

    def test_create_provider_echo(self) -> None:
        """Test creating Echo provider."""
        config = {
            "provider": "echo",
            "model": "test",
            "temperature": 0.0,
        }

        provider = SchemaExtractor._create_provider(config)

        assert isinstance(provider, EchoProvider)


class TestSchemaExtractorPromptBuilding:
    """Tests for extraction prompt building."""

    @pytest.fixture
    def extractor(self) -> SchemaExtractor:
        """Create extractor with EchoProvider."""
        provider = EchoProvider(
            {"provider": "echo", "model": "test", "options": {"echo_prefix": ""}}
        )
        return SchemaExtractor(provider=provider)

    def test_build_prompt_basic(self, extractor: SchemaExtractor) -> None:
        """Test basic prompt building."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        prompt = extractor._build_prompt(
            text="Hello Alice",
            schema=schema,
            context=None,
        )

        assert "Hello Alice" in prompt
        assert '"name"' in prompt
        assert "string" in prompt

    def test_build_prompt_with_context(self, extractor: SchemaExtractor) -> None:
        """Test prompt building with context."""
        schema = {"type": "object", "properties": {"intent": {"type": "string"}}}
        context = {"stage": "welcome", "prompt": "What do they want?"}

        prompt = extractor._build_prompt(
            text="Help me",
            schema=schema,
            context=context,
        )

        assert "welcome" in prompt
        assert "What do they want?" in prompt


class TestSchemaExtractorJSONParsing:
    """Tests for JSON parsing functionality."""

    @pytest.fixture
    def extractor(self) -> SchemaExtractor:
        """Create extractor for testing."""
        provider = EchoProvider(
            {"provider": "echo", "model": "test", "options": {"echo_prefix": ""}}
        )
        return SchemaExtractor(provider=provider)

    def test_parse_valid_json(self, extractor: SchemaExtractor) -> None:
        """Test parsing valid JSON."""
        data, errors = extractor._parse_json('{"name": "Alice", "age": 30}')

        assert data == {"name": "Alice", "age": 30}
        assert errors == []

    def test_parse_json_with_whitespace(self, extractor: SchemaExtractor) -> None:
        """Test parsing JSON with surrounding whitespace."""
        data, errors = extractor._parse_json('  {"name": "Bob"}  \n')

        assert data == {"name": "Bob"}
        assert errors == []

    def test_parse_json_embedded_in_text(self, extractor: SchemaExtractor) -> None:
        """Test parsing JSON embedded in text."""
        data, errors = extractor._parse_json(
            'Here is the result: {"value": 123} That is all.'
        )

        assert data == {"value": 123}
        assert errors == []

    def test_parse_malformed_json_repaired(self, extractor: SchemaExtractor) -> None:
        """Test parsing malformed JSON that can be repaired."""
        # Missing closing brace
        data, errors = extractor._parse_json('{"name": "Test"')

        assert data == {"name": "Test"}
        assert any("repaired" in e.lower() for e in errors)

    def test_parse_invalid_json(self, extractor: SchemaExtractor) -> None:
        """Test parsing completely invalid JSON."""
        data, errors = extractor._parse_json("not json at all")

        assert data == {}
        assert any("Could not parse" in e for e in errors)

    def test_parse_json_array_returns_empty(self, extractor: SchemaExtractor) -> None:
        """Test that JSON arrays return empty dict (we expect objects)."""
        data, errors = extractor._parse_json("[1, 2, 3]")

        # Direct parse would fail type check, JSONExtractor finds no objects
        assert data == {}


class TestSchemaExtractorValidation:
    """Tests for schema validation functionality."""

    @pytest.fixture
    def extractor(self) -> SchemaExtractor:
        """Create extractor for testing."""
        provider = EchoProvider(
            {"provider": "echo", "model": "test", "options": {"echo_prefix": ""}}
        )
        return SchemaExtractor(provider=provider)

    def test_validate_all_required_present(self, extractor: SchemaExtractor) -> None:
        """Test validation with all required fields present."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }

        errors = extractor._validate_schema({"a": "x", "b": "y"}, schema)

        assert errors == []

    def test_validate_missing_required(self, extractor: SchemaExtractor) -> None:
        """Test validation with missing required field."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }

        errors = extractor._validate_schema({"a": "x"}, schema)

        assert len(errors) == 1
        assert "Missing required field: b" in errors

    def test_validate_enum_valid(self, extractor: SchemaExtractor) -> None:
        """Test validation with valid enum value."""
        schema = {
            "type": "object",
            "properties": {"size": {"type": "string", "enum": ["s", "m", "l"]}},
        }

        errors = extractor._validate_schema({"size": "m"}, schema)

        assert errors == []

    def test_validate_enum_invalid(self, extractor: SchemaExtractor) -> None:
        """Test validation with invalid enum value."""
        schema = {
            "type": "object",
            "properties": {"size": {"type": "string", "enum": ["s", "m", "l"]}},
        }

        errors = extractor._validate_schema({"size": "xl"}, schema)

        assert len(errors) == 1
        assert "Invalid value for size" in errors[0]

    def test_validate_extra_fields_allowed(self, extractor: SchemaExtractor) -> None:
        """Test that extra fields don't cause errors."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        errors = extractor._validate_schema({"name": "Alice", "extra": "ignored"}, schema)

        assert errors == []


class TestSchemaExtractorWithTracking:
    """Tests for SchemaExtractor with extraction tracking enabled."""

    @pytest.fixture
    def echo_provider(self) -> EchoProvider:
        """Create an EchoProvider for testing."""
        return EchoProvider(
            {"provider": "echo", "model": "test-model", "options": {"echo_prefix": ""}}
        )

    @pytest.fixture
    def extractor(self, echo_provider: EchoProvider) -> SchemaExtractor:
        """Create a SchemaExtractor with EchoProvider."""
        return SchemaExtractor(provider=echo_provider)

    @pytest.mark.asyncio
    async def test_extract_with_tracker(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction with tracking enabled records the extraction."""
        from dataknobs_llm.extraction import ExtractionTracker

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        tracker = ExtractionTracker()

        echo_provider.set_responses(['{"name": "Alice"}'])

        result = await extractor.extract(
            text="My name is Alice",
            schema=schema,
            tracker=tracker,
        )

        # Extraction should succeed
        assert result.data == {"name": "Alice"}
        assert result.is_confident

        # Tracker should have one record
        assert len(tracker) == 1
        records = tracker.query()
        assert records[0].input_text == "My name is Alice"
        assert records[0].extracted_data == {"name": "Alice"}
        assert records[0].success is True

    @pytest.mark.asyncio
    async def test_extract_without_tracker(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test extraction without tracker still works normally."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        echo_provider.set_responses(['{"name": "Bob"}'])

        result = await extractor.extract(
            text="My name is Bob",
            schema=schema,
            # No tracker parameter
        )

        assert result.data == {"name": "Bob"}
        assert result.is_confident

    @pytest.mark.asyncio
    async def test_tracker_records_failed_extraction(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test that failed extractions are recorded in tracker."""
        from dataknobs_llm.extraction import ExtractionTracker

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        tracker = ExtractionTracker()

        # Response missing required field
        echo_provider.set_responses(['{"name": "Alice"}'])

        result = await extractor.extract(
            text="My name is Alice",
            schema=schema,
            tracker=tracker,
        )

        # Extraction has errors
        assert not result.is_confident
        assert "Missing required field: age" in result.errors

        # Tracker should record the failure
        assert len(tracker) == 1
        records = tracker.query()
        assert records[0].success is False
        assert "Missing required field: age" in records[0].validation_errors

    @pytest.mark.asyncio
    async def test_tracker_records_schema_info(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test that tracker records schema name and hash."""
        from dataknobs_llm.extraction import ExtractionTracker

        schema = {
            "title": "PersonSchema",
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        tracker = ExtractionTracker()

        echo_provider.set_responses(['{"name": "Alice"}'])

        await extractor.extract(
            text="My name is Alice",
            schema=schema,
            tracker=tracker,
        )

        records = tracker.query()
        assert records[0].schema_name == "PersonSchema"
        assert records[0].schema_hash is not None

    @pytest.mark.asyncio
    async def test_tracker_records_context(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test that tracker records context if provided."""
        from dataknobs_llm.extraction import ExtractionTracker

        schema = {"type": "object", "properties": {"intent": {"type": "string"}}}
        tracker = ExtractionTracker()

        echo_provider.set_responses(['{"intent": "greeting"}'])

        await extractor.extract(
            text="Hello",
            schema=schema,
            context={"stage": "welcome", "prompt": "What do they want?"},
            tracker=tracker,
        )

        records = tracker.query()
        assert records[0].context == {"stage": "welcome", "prompt": "What do they want?"}

    @pytest.mark.asyncio
    async def test_tracker_records_duration(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test that tracker records extraction duration."""
        from dataknobs_llm.extraction import ExtractionTracker

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        tracker = ExtractionTracker()

        echo_provider.set_responses(['{"name": "Test"}'])

        await extractor.extract(
            text="Test",
            schema=schema,
            tracker=tracker,
        )

        records = tracker.query()
        # Duration should be positive (extraction takes some time)
        assert records[0].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_tracker_records_raw_response(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test that tracker records raw LLM response."""
        from dataknobs_llm.extraction import ExtractionTracker

        schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
        tracker = ExtractionTracker()

        echo_provider.set_responses(['{"value": 42}'])

        await extractor.extract(
            text="The answer is 42",
            schema=schema,
            tracker=tracker,
        )

        records = tracker.query()
        assert records[0].raw_response == '{"value": 42}'

    @pytest.mark.asyncio
    async def test_tracker_with_multiple_extractions(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test tracker accumulates multiple extractions."""
        from dataknobs_llm.extraction import ExtractionTracker

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        tracker = ExtractionTracker()

        echo_provider.set_responses(['{"name": "Alice"}', '{"name": "Bob"}', '{"name": "Charlie"}'])

        for text in ["Alice", "Bob", "Charlie"]:
            await extractor.extract(
                text=f"My name is {text}",
                schema=schema,
                tracker=tracker,
            )

        assert len(tracker) == 3
        stats = tracker.get_stats()
        assert stats.total_extractions == 3
        assert stats.successful_extractions == 3

    @pytest.mark.asyncio
    async def test_tracker_stats_after_mixed_results(
        self, extractor: SchemaExtractor, echo_provider: EchoProvider
    ) -> None:
        """Test tracker stats with mix of success and failure."""
        from dataknobs_llm.extraction import ExtractionTracker

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        tracker = ExtractionTracker()

        # Good response, then bad response
        echo_provider.set_responses(['{"name": "Alice"}', '{}'])

        await extractor.extract(text="Alice", schema=schema, tracker=tracker)
        await extractor.extract(text="???", schema=schema, tracker=tracker)

        stats = tracker.get_stats()
        assert stats.total_extractions == 2
        assert stats.successful_extractions == 1
        assert stats.failed_extractions == 1
        assert stats.success_rate == 0.5

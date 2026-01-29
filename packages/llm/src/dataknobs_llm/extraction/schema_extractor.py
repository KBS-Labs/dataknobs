"""Schema-based data extraction using LLM providers.

This module provides SchemaExtractor for extracting structured data
from user input using LLM-based extraction with JSON Schema validation.

Example:
    ```python
    from dataknobs_llm.extraction import SchemaExtractor, ExtractionResult
    from dataknobs_llm.llm.providers.echo import EchoProvider

    # Create extractor with any LLM provider
    provider = EchoProvider({"provider": "echo", "model": "test"})
    extractor = SchemaExtractor(provider=provider)

    # Define schema for extraction
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    # Extract structured data from text
    result = await extractor.extract(
        text="My name is Alice and I'm 30 years old",
        schema=schema,
        context={"stage": "gather_info"}
    )

    if result.is_confident:
        print(result.data)  # {"name": "Alice", "age": 30}
    ```
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataknobs_utils.json_extractor import JSONExtractor

if TYPE_CHECKING:
    from dataknobs_llm.extraction.observability import ExtractionTracker
    from dataknobs_llm.llm.base import AsyncLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ExtractedAssumption:
    """An assumption identified during extraction.

    Captures assumptions the extractor made about the user's intent
    or meaning, enabling downstream validation and confirmation.

    Attributes:
        content: The assumption text
        field: The field this assumption relates to (if any)
        confidence: How confident the extractor is in this assumption (0.0-1.0)
        source: Where the assumption came from (e.g., "inferred", "default")
    """

    content: str
    field: str | None = None
    confidence: float = 0.5
    source: str = "inferred"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "field": self.field,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedAssumption":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            field=data.get("field"),
            confidence=data.get("confidence", 0.5),
            source=data.get("source", "inferred"),
        )


@dataclass
class ExtractionResult:
    """Result from schema extraction.

    Contains extracted data, confidence score, and any validation errors.
    Used by WizardReasoning to determine if extraction was successful.

    Attributes:
        data: Extracted structured data as dict
        confidence: Confidence score from 0.0 to 1.0
        errors: List of extraction/validation error messages
        raw_response: Raw LLM response text (for debugging)
        assumptions: List of assumptions made during extraction

    Example:
        ```python
        result = ExtractionResult(
            data={"name": "Alice", "age": 30},
            confidence=0.95,
            errors=[],
            assumptions=[
                ExtractedAssumption(
                    content="User's age is exactly 30",
                    field="age",
                    confidence=0.9,
                )
            ]
        )

        if result.is_confident:
            process(result.data)
        else:
            ask_clarification(result.errors)
        ```
    """

    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    errors: list[str] = field(default_factory=list)
    raw_response: str = ""
    assumptions: list[ExtractedAssumption] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        """Check if extraction is confident enough to proceed.

        Returns True if confidence >= 0.8 and no errors occurred.
        """
        return self.confidence >= 0.8 and not self.errors

    @property
    def has_assumptions(self) -> bool:
        """Check if any assumptions were made during extraction."""
        return len(self.assumptions) > 0

    @property
    def unconfirmed_assumptions(self) -> list[ExtractedAssumption]:
        """Get assumptions with low confidence that may need confirmation."""
        return [a for a in self.assumptions if a.confidence < 0.8]


# Default extraction prompt template
DEFAULT_EXTRACTION_PROMPT = """Extract structured data from the user's message.

## Schema
Extract data matching this JSON Schema:
```json
{schema}
```

## Context
{context}

## Instructions
1. Parse the user's message and extract relevant information
2. Return ONLY a valid JSON object matching the schema
3. If information is missing, omit the field (don't use null unless appropriate)
4. If you cannot extract the required information, return an empty object {{}}
5. Do not include explanations - only return the JSON object

## User Message
{text}

## Extracted Data (JSON only):"""

# Extraction prompt template with assumption tracking
EXTRACTION_WITH_ASSUMPTIONS_PROMPT = """Extract structured data from the user's message and identify any assumptions made.

## Schema
Extract data matching this JSON Schema:
```json
{schema}
```

## Context
{context}

## Instructions
1. Parse the user's message and extract relevant information
2. Identify any assumptions you make about:
   - Ambiguous terms or references
   - Implied but not explicitly stated information
   - Default values used when information is missing
   - Interpretations that could have multiple meanings
3. Return a JSON object with two keys:
   - "data": The extracted data matching the schema
   - "assumptions": Array of assumption objects with:
     - "content": Description of the assumption
     - "field": Which field this relates to (or null if general)
     - "confidence": How confident you are (0.0-1.0)

## Example Output
```json
{{
  "data": {{"name": "John", "age": 30}},
  "assumptions": [
    {{"content": "Assumed 'John' refers to a person's first name", "field": "name", "confidence": 0.9}},
    {{"content": "Age of 30 was mentioned in a different context but likely applies here", "field": "age", "confidence": 0.7}}
  ]
}}
```

## User Message
{text}

## Extracted Data and Assumptions (JSON only):"""


class SchemaExtractor:
    """LLM-based structured data extractor with JSON Schema validation.

    Uses an LLM provider to extract structured data from natural language
    text, validating against a JSON Schema. Supports multiple provider
    configurations for cost/performance optimization.

    Features:
        - Works with any AsyncLLMProvider implementation
        - JSON Schema validation for extracted data
        - Confidence scoring based on extraction quality
        - Error collection for debugging
        - Custom extraction prompts
        - Model override for per-extraction provider selection

    Configuration:
        The extractor can be configured via from_config() which supports:
        - provider: LLM provider name (openai, anthropic, ollama, echo)
        - model: Model to use for extraction
        - temperature: Sampling temperature (default: 0.0 for deterministic)
        - Additional provider-specific options

    Example:
        ```python
        from dataknobs_llm.extraction import SchemaExtractor
        from dataknobs_llm.llm.providers.ollama import OllamaProvider

        # Direct instantiation
        provider = OllamaProvider({
            "provider": "ollama",
            "model": "qwen3-coder"
        })
        extractor = SchemaExtractor(provider=provider)

        # Or from config
        extractor = SchemaExtractor.from_config({
            "provider": "ollama",
            "model": "qwen3-coder",
            "temperature": 0.0
        })

        # Extract with schema
        result = await extractor.extract(
            text="I want to order 3 large pizzas",
            schema={
                "type": "object",
                "properties": {
                    "quantity": {"type": "integer"},
                    "size": {"type": "string", "enum": ["small", "medium", "large"]}
                },
                "required": ["quantity", "size"]
            }
        )
        ```

    Attributes:
        _provider: LLM provider for extraction
        _extraction_prompt: Template for extraction prompts
        _json_extractor: Utility for parsing JSON from responses
    """

    def __init__(
        self,
        provider: "AsyncLLMProvider",
        extraction_prompt: str | None = None,
    ):
        """Initialize SchemaExtractor.

        Args:
            provider: LLM provider for extraction
            extraction_prompt: Custom prompt template (uses default if None)
        """
        self._provider = provider
        self._extraction_prompt = extraction_prompt or DEFAULT_EXTRACTION_PROMPT
        self._json_extractor = JSONExtractor()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SchemaExtractor":
        """Create SchemaExtractor from configuration dict.

        Supports creating the LLM provider from configuration, enabling
        flexible provider selection for extraction (e.g., using a smaller/
        faster model for extraction vs. generation).

        Args:
            config: Configuration dict with:
                - provider: Provider name (openai, anthropic, ollama, echo)
                - model: Model name
                - temperature: Sampling temperature (default: 0.0)
                - api_key: API key (if required)
                - api_base: Custom API endpoint (optional)
                - extraction_prompt: Custom prompt template (optional)

        Returns:
            Configured SchemaExtractor instance

        Raises:
            ValueError: If provider is not specified or unsupported

        Example:
            ```python
            extractor = SchemaExtractor.from_config({
                "provider": "ollama",
                "model": "qwen3-coder",
                "temperature": 0.0
            })
            ```
        """
        provider_name = config.get("provider")
        if not provider_name:
            raise ValueError("provider is required in extraction config")

        # Build LLM config
        llm_config = {
            "provider": provider_name,
            "model": config.get("model", "default"),
            "temperature": config.get("temperature", 0.0),
        }

        # Add optional config values
        if config.get("api_key"):
            llm_config["api_key"] = config["api_key"]
        if config.get("api_base"):
            llm_config["api_base"] = config["api_base"]

        # Create provider
        provider = cls._create_provider(llm_config)

        return cls(
            provider=provider,
            extraction_prompt=config.get("extraction_prompt"),
        )

    @classmethod
    def from_env_config(cls, config: dict[str, Any]) -> "SchemaExtractor":
        """Create SchemaExtractor from environment-aware configuration.

        Alias for from_config() that follows the dataknobs convention
        of environment-aware configuration loading.

        Args:
            config: Configuration dict (same as from_config)

        Returns:
            Configured SchemaExtractor instance
        """
        return cls.from_config(config)

    @staticmethod
    def _create_provider(llm_config: dict[str, Any]) -> "AsyncLLMProvider":
        """Create LLM provider from config.

        Args:
            llm_config: LLM configuration dict

        Returns:
            AsyncLLMProvider instance

        Raises:
            ValueError: If provider type is unsupported
        """
        provider_name = llm_config["provider"]

        if provider_name == "echo":
            from dataknobs_llm.llm.providers.echo import EchoProvider

            return EchoProvider(llm_config)
        elif provider_name == "ollama":
            from dataknobs_llm.llm.providers.ollama import OllamaProvider

            return OllamaProvider(llm_config)
        elif provider_name == "openai":
            from dataknobs_llm.llm.providers.openai import OpenAIProvider

            return OpenAIProvider(llm_config)
        elif provider_name == "anthropic":
            from dataknobs_llm.llm.providers.anthropic import AnthropicProvider

            return AnthropicProvider(llm_config)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    async def extract(
        self,
        text: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
        model: str | None = None,
        tracker: "ExtractionTracker | None" = None,
        track_assumptions: bool = False,
    ) -> ExtractionResult:
        """Extract structured data from text using the schema.

        Sends the text to the LLM with schema context and parses the
        JSON response. Validates extracted data against the schema
        and calculates a confidence score.

        Args:
            text: User input text to extract from
            schema: JSON Schema defining expected data structure
            context: Optional context dict (stage name, prompt, etc.)
            model: Optional model override for this extraction
            tracker: Optional ExtractionTracker for recording extraction history
            track_assumptions: If True, identify and return assumptions made

        Returns:
            ExtractionResult with extracted data, confidence, errors, and assumptions

        Example:
            ```python
            result = await extractor.extract(
                text="I'd like a large pepperoni pizza",
                schema={
                    "type": "object",
                    "properties": {
                        "size": {"type": "string"},
                        "toppings": {"type": "array", "items": {"type": "string"}}
                    }
                },
                context={"stage": "order_details"}
            )
            # result.data = {"size": "large", "toppings": ["pepperoni"]}

            # With tracking enabled
            from dataknobs_llm.extraction import ExtractionTracker
            tracker = ExtractionTracker()
            result = await extractor.extract(text, schema, tracker=tracker)
            stats = tracker.get_stats()

            # With assumption tracking
            result = await extractor.extract(
                text="Set up a math tutor",
                schema=bot_schema,
                track_assumptions=True,
            )
            for assumption in result.assumptions:
                print(f"Assumed: {assumption.content}")
            ```
        """
        start_time = time.time()

        # Build extraction prompt
        prompt = self._build_prompt(text, schema, context, track_assumptions)

        # Determine model being used
        model_used = model or getattr(self._provider, "_model", None)
        provider_name = getattr(self._provider, "_provider_name", None) or getattr(
            self._provider.__class__, "__name__", "unknown"
        ).lower().replace("provider", "")

        # Call LLM
        config_overrides = {}
        if model:
            config_overrides["model"] = model

        try:
            response = await self._provider.complete(
                prompt,
                config_overrides=config_overrides if config_overrides else None,
            )
            raw_response = response.content
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            result = ExtractionResult(
                data={},
                confidence=0.0,
                errors=[f"LLM extraction failed: {e}"],
                raw_response="",
            )

            # Record failed extraction if tracking
            if tracker is not None:
                self._record_extraction(
                    tracker=tracker,
                    text=text,
                    schema=schema,
                    result=result,
                    duration_ms=(time.time() - start_time) * 1000,
                    model_used=model_used,
                    provider=provider_name,
                    context=context,
                )

            return result

        # Parse JSON from response
        parsed_data, parse_errors = self._parse_json(raw_response)

        # Extract assumptions if tracking enabled
        assumptions: list[ExtractedAssumption] = []
        if track_assumptions and isinstance(parsed_data, dict):
            # Check if response has the expected format with data and assumptions
            if "data" in parsed_data and "assumptions" in parsed_data:
                data = parsed_data.get("data", {})
                raw_assumptions = parsed_data.get("assumptions", [])
                assumptions = self._parse_assumptions(raw_assumptions)
            else:
                # Fallback: assume the entire response is data
                data = parsed_data
        else:
            data = parsed_data

        # Validate against schema
        validation_errors = self._validate_schema(data, schema) if data else []

        # Calculate confidence
        all_errors = parse_errors + validation_errors
        confidence = self._calculate_confidence(data, schema, all_errors)

        result = ExtractionResult(
            data=data,
            confidence=confidence,
            errors=all_errors,
            raw_response=raw_response,
            assumptions=assumptions,
        )

        # Record extraction if tracking
        if tracker is not None:
            self._record_extraction(
                tracker=tracker,
                text=text,
                schema=schema,
                result=result,
                duration_ms=(time.time() - start_time) * 1000,
                model_used=model_used,
                provider=provider_name,
                context=context,
            )

        return result

    def _record_extraction(
        self,
        tracker: "ExtractionTracker",
        text: str,
        schema: dict[str, Any],
        result: ExtractionResult,
        duration_ms: float,
        model_used: str | None,
        provider: str | None,
        context: dict[str, Any] | None,
    ) -> None:
        """Record an extraction operation to the tracker.

        Args:
            tracker: ExtractionTracker to record to
            text: Input text
            schema: JSON Schema used
            result: Extraction result
            duration_ms: Duration in milliseconds
            model_used: Model that performed extraction
            provider: Provider type
            context: Context dict
        """
        from dataknobs_llm.extraction.observability import create_extraction_record

        record = create_extraction_record(
            input_text=text,
            extracted_data=result.data,
            confidence=result.confidence,
            validation_errors=result.errors,
            duration_ms=duration_ms,
            schema=schema,
            model_used=model_used,
            provider=provider,
            context=context,
            raw_response=result.raw_response,
        )
        tracker.record(record)

    def _build_prompt(
        self,
        text: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None,
        track_assumptions: bool = False,
    ) -> str:
        """Build extraction prompt from template.

        Args:
            text: User input text
            schema: JSON Schema
            context: Optional context dict
            track_assumptions: Whether to use assumption-tracking prompt

        Returns:
            Formatted extraction prompt
        """
        context_str = ""
        if context:
            context_parts = []
            if context.get("stage"):
                context_parts.append(f"Stage: {context['stage']}")
            if context.get("prompt"):
                context_parts.append(f"What we're looking for: {context['prompt']}")
            context_str = "\n".join(context_parts) if context_parts else "None"
        else:
            context_str = "None"

        # Choose prompt template based on assumption tracking
        if track_assumptions:
            prompt_template = EXTRACTION_WITH_ASSUMPTIONS_PROMPT
        else:
            prompt_template = self._extraction_prompt

        return prompt_template.format(
            schema=json.dumps(schema, indent=2),
            context=context_str,
            text=text,
        )

    def _parse_assumptions(
        self,
        raw_assumptions: list[dict[str, Any]],
    ) -> list[ExtractedAssumption]:
        """Parse raw assumption dicts into ExtractedAssumption objects.

        Args:
            raw_assumptions: List of assumption dicts from LLM response

        Returns:
            List of ExtractedAssumption objects
        """
        assumptions: list[ExtractedAssumption] = []

        for raw in raw_assumptions:
            if not isinstance(raw, dict):
                continue

            content = raw.get("content", "")
            if not content:
                continue

            assumptions.append(ExtractedAssumption(
                content=content,
                field=raw.get("field"),
                confidence=raw.get("confidence", 0.5),
                source="inferred",
            ))

        return assumptions

    def _parse_json(self, response: str) -> tuple[dict[str, Any], list[str]]:
        """Parse JSON from LLM response.

        Uses JSONExtractor to find and parse JSON objects from the
        response, handling malformed JSON when possible.

        Args:
            response: Raw LLM response text

        Returns:
            Tuple of (extracted data dict, list of error messages)
        """
        errors: list[str] = []

        # Try direct JSON parse first
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict):
                return data, []
        except json.JSONDecodeError:
            pass

        # Use JSONExtractor for more robust parsing
        self._json_extractor.extract_jsons(response)

        if self._json_extractor.complete_jsons:
            return self._json_extractor.complete_jsons[0], []

        if self._json_extractor.fixed_jsons:
            errors.append("JSON was malformed but repaired")
            return self._json_extractor.fixed_jsons[0], errors

        errors.append("Could not parse JSON from response")
        return {}, errors

    def _validate_schema(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
    ) -> list[str]:
        """Validate extracted data against JSON Schema.

        Performs basic validation of required fields and enum constraints.

        Args:
            data: Extracted data to validate
            schema: JSON Schema to validate against

        Returns:
            List of validation error messages
        """
        errors: list[str] = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field_name in required:
            if field_name not in data or data[field_name] is None:
                errors.append(f"Missing required field: {field_name}")

        # Check enum constraints
        for name, value in data.items():
            if name in properties:
                prop = properties[name]
                if "enum" in prop and value not in prop["enum"]:
                    errors.append(
                        f"Invalid value for {name}: must be one of {prop['enum']}"
                    )

        return errors

    def _calculate_confidence(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        errors: list[str],
    ) -> float:
        """Calculate confidence score for extraction.

        Confidence is based on:
        - Whether any data was extracted
        - How many required fields are present
        - Whether there are validation errors

        Args:
            data: Extracted data
            schema: JSON Schema
            errors: List of errors encountered

        Returns:
            Confidence score from 0.0 to 1.0
        """
        if not data:
            return 0.0

        if errors:
            # Some errors but data extracted - lower confidence
            return 0.5

        # Check how many required fields are present
        required = schema.get("required", [])
        if required:
            present = sum(1 for f in required if f in data and data[f] is not None)
            required_ratio = present / len(required)
        else:
            required_ratio = 1.0

        # Check how many properties we extracted
        properties = schema.get("properties", {})
        if properties:
            extracted = sum(1 for k in data if k in properties)
            property_ratio = extracted / len(properties)
        else:
            property_ratio = 1.0 if data else 0.0

        # Combine factors
        confidence = (required_ratio * 0.7) + (property_ratio * 0.3)

        return min(1.0, confidence)

    async def close(self) -> None:
        """Close the extractor and release resources.

        Closes the underlying LLM provider, releasing HTTP connections
        and other resources. Should be called when the extractor is no
        longer needed.

        Example:
            ```python
            extractor = SchemaExtractor.from_config(config)
            try:
                result = await extractor.extract(text, schema)
            finally:
                await extractor.close()
            ```
        """
        if self._provider and hasattr(self._provider, "close"):
            await self._provider.close()

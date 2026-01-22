# Schema Extraction

This guide covers LLM-based structured data extraction with JSON Schema validation and observability tracking.

## Overview

The extraction module provides:

- **SchemaExtractor**: LLM-powered structured data extraction
- **ExtractionTracker**: Recording and querying extraction operations
- **ExtractionStats**: Aggregated metrics for monitoring

All types are in `dataknobs_llm.extraction`.

## SchemaExtractor

`SchemaExtractor` uses LLM providers to extract structured data from natural language text, validating against JSON Schema.

### Basic Usage

```python
from dataknobs_llm.extraction import SchemaExtractor, ExtractionResult
from dataknobs_llm.llm.providers.ollama import OllamaProvider

# Create provider
provider = OllamaProvider({
    "provider": "ollama",
    "model": "qwen3-coder",
    "temperature": 0.0,  # Deterministic for extraction
})

# Create extractor
extractor = SchemaExtractor(provider=provider)

# Define schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name"],
}

# Extract data
result = await extractor.extract(
    text="My name is Alice, I'm 30 years old, reach me at alice@example.com",
    schema=schema,
)

# Check result
if result.is_confident:
    print(result.data)
    # {"name": "Alice", "age": 30, "email": "alice@example.com"}
else:
    print(f"Low confidence: {result.confidence}")
    print(f"Errors: {result.errors}")
```

### Creating from Configuration

```python
# From config dict
extractor = SchemaExtractor.from_config({
    "provider": "ollama",
    "model": "qwen3-coder",
    "temperature": 0.0,
})

# Or using from_env_config alias
extractor = SchemaExtractor.from_env_config({
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "${OPENAI_API_KEY}",
})
```

### ExtractionResult

The `extract()` method returns an `ExtractionResult`:

```python
@dataclass
class ExtractionResult:
    data: dict[str, Any]      # Extracted structured data
    confidence: float         # Score from 0.0 to 1.0
    errors: list[str]         # Validation errors
    raw_response: str         # Raw LLM response

    @property
    def is_confident(self) -> bool:
        """True if confidence >= 0.8 and no errors."""
```

### Custom Extraction Prompts

Override the default extraction prompt:

```python
custom_prompt = """Extract information from the user's message.

## Schema
{schema}

## Context
{context}

## Message
{text}

## Response (JSON only):"""

extractor = SchemaExtractor(
    provider=provider,
    extraction_prompt=custom_prompt,
)
```

### Context for Extraction

Provide context to guide extraction:

```python
result = await extractor.extract(
    text="Set it to large",
    schema=size_schema,
    context={
        "stage": "product_selection",
        "prompt": "Collecting size preference",
    },
)
```

## Extraction Tracking

### ExtractionTracker

`ExtractionTracker` records extraction operations for observability and debugging:

```python
from dataknobs_llm.extraction import SchemaExtractor, ExtractionTracker

# Create tracker with default settings
tracker = ExtractionTracker(max_history=100)

# Or with custom text truncation (for memory efficiency)
tracker = ExtractionTracker(
    max_history=100,
    truncate_text_at=200,  # Truncate long text fields in records
)

# Extract with tracking
result = await extractor.extract(
    text="I'm Alice, 30 years old",
    schema=person_schema,
    tracker=tracker,  # Enable tracking
)

# Query history
all_records = tracker.query()
recent = tracker.get_recent(10)

# Get statistics
stats = tracker.get_stats()
print(f"Total: {stats.total_extractions}")
print(f"Success rate: {stats.success_rate:.1%}")
print(f"Avg confidence: {stats.avg_confidence:.2f}")
print(f"Avg duration: {stats.avg_duration_ms:.1f}ms")

# Clear history when needed
tracker.clear()

# Check history size
print(f"Records in history: {len(tracker)}")
```

### ExtractionRecord

Each extraction creates an `ExtractionRecord`:

```python
from dataknobs_llm.extraction import ExtractionRecord

record = ExtractionRecord(
    timestamp=1700000000.0,
    input_text="My name is Alice",
    extracted_data={"name": "Alice"},
    confidence=0.95,
    validation_errors=[],
    duration_ms=150.5,
    success=True,
    schema_name="Person",
    schema_hash="abc123",
    model_used="qwen3-coder",
    provider="ollama",
    context={"stage": "greeting"},
    raw_response='{"name": "Alice"}',
    input_length=17,
    truncated=False,
)
```

### Querying History

Filter extraction history with `ExtractionHistoryQuery`:

```python
from dataknobs_llm.extraction import ExtractionHistoryQuery

# Filter by schema
query = ExtractionHistoryQuery(schema_name="Person")
person_extractions = tracker.query(query)

# Filter by model
query = ExtractionHistoryQuery(model="gpt-4")
gpt4_extractions = tracker.query(query)

# Filter by success/failure
query = ExtractionHistoryQuery(success_only=True, min_confidence=0.9)
high_confidence = tracker.query(query)

# Filter by time range
query = ExtractionHistoryQuery(
    since=time.time() - 3600,  # Last hour
    until=time.time(),
    limit=50,
)
recent = tracker.query(query)
```

### ExtractionStats

Get aggregated statistics:

```python
stats = tracker.get_stats()

# Basic counts
print(f"Total: {stats.total_extractions}")
print(f"Successful: {stats.successful_extractions}")
print(f"Failed: {stats.failed_extractions}")
print(f"Success rate: {stats.success_rate:.1%}")

# Performance metrics
print(f"Avg confidence: {stats.avg_confidence:.2f}")
print(f"Avg duration: {stats.avg_duration_ms:.1f}ms")

# Time range
print(f"First: {stats.first_extraction}")
print(f"Last: {stats.last_extraction}")

# Breakdown by schema
for schema, count in stats.by_schema.items():
    print(f"  {schema}: {count}")

# Breakdown by model
for model, count in stats.by_model.items():
    print(f"  {model}: {count}")

# Most common errors
for error, count in stats.most_common_errors:
    print(f"  {error}: {count}")
```

### Factory Function

Use `create_extraction_record()` for convenience:

```python
from dataknobs_llm.extraction import create_extraction_record

record = create_extraction_record(
    input_text="My name is Alice and I'm 30 years old",
    extracted_data={"name": "Alice", "age": 30},
    confidence=0.95,
    validation_errors=[],
    duration_ms=150.5,
    schema=person_schema,  # Extracts title, computes hash
    model_used="qwen3-coder",
    provider="ollama",
    context={"stage": "greeting"},
    raw_response='{"name": "Alice", "age": 30}',
    truncate_at=200,  # Truncate long text fields
)

tracker.record(record)
```

## Integration with Wizard Reasoning

Extraction tracking integrates with wizard reasoning for wizard-based data collection:

```python
from dataknobs_bots.reasoning import WizardReasoning
from dataknobs_llm.extraction import ExtractionTracker

# Create tracker
tracker = ExtractionTracker()

# Configure wizard with extraction tracking
config = {
    "reasoning": {
        "strategy": "wizard",
        "wizard_config": "wizard.yaml",
        "extraction_config": {
            "provider": "ollama",
            "model": "qwen3-coder",
        },
    },
}

# Pass tracker to extract calls
# (WizardReasoning passes tracker to SchemaExtractor internally)
```

## Best Practices

### 1. Use Low Temperature for Extraction

Set temperature to 0 for deterministic extraction:

```python
extractor = SchemaExtractor.from_config({
    "provider": "ollama",
    "model": "qwen3-coder",
    "temperature": 0.0,  # Deterministic
})
```

### 2. Provide Clear Schema Titles

Include titles in schemas for better tracking:

```python
schema = {
    "title": "PersonInfo",  # Used in ExtractionRecord.schema_name
    "type": "object",
    "properties": {...},
}
```

### 3. Monitor Success Rates

Track extraction success over time:

```python
stats = tracker.get_stats()

if stats.success_rate < 0.8:
    log.warning(
        "Low extraction success rate: %.1f%% (%d/%d)",
        stats.success_rate * 100,
        stats.successful_extractions,
        stats.total_extractions,
    )
```

### 4. Analyze Common Errors

Use error statistics to improve prompts/schemas:

```python
stats = tracker.get_stats()

for error, count in stats.most_common_errors[:5]:
    if "required field" in error.lower():
        log.info("Missing required field errors: %d", count)
    elif "json" in error.lower():
        log.info("JSON parsing errors: %d", count)
```

### 5. Set Appropriate History Limits

Configure tracker capacity based on needs:

```python
# For debugging (short history)
debug_tracker = ExtractionTracker(max_history=20)

# For production monitoring
prod_tracker = ExtractionTracker(max_history=1000)
```

### 6. Manage Memory with Text Truncation

Long input texts and raw responses are truncated to save memory:

```python
# Default truncation at 200 characters
tracker = ExtractionTracker(truncate_text_at=200)

# Store more context if needed
tracker = ExtractionTracker(truncate_text_at=500)

# Records indicate if truncation occurred
record = tracker.get_recent(1)[0]
if record.truncated:
    print(f"Original input was {record.input_length} chars")
```

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `SchemaExtractor` | LLM-powered structured data extraction |
| `ExtractionResult` | Result from schema extraction |
| `ExtractionTracker` | Manages extraction history with queries |
| `ExtractionRecord` | Record of a single extraction operation |
| `ExtractionStats` | Aggregated extraction statistics |
| `ExtractionHistoryQuery` | Query parameters for filtering history |

### Factory Functions

| Function | Description |
|----------|-------------|
| `create_extraction_record()` | Create ExtractionRecord with auto-timestamp |

### ExtractionTracker Methods

| Method | Description |
|--------|-------------|
| `record(extraction)` | Add an extraction record to history |
| `query(query)` | Filter history with ExtractionHistoryQuery |
| `get_stats()` | Get aggregated ExtractionStats |
| `get_recent(count)` | Get most recent N records |
| `clear()` | Clear all extraction history |
| `__len__()` | Return number of records in history |

### ExtractionTracker Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_history` | int | 100 | Maximum records to retain |
| `truncate_text_at` | int | 200 | Max length for text fields |

### Configuration Options

| Option | Type | Description |
|--------|------|-------------|
| `provider` | string | LLM provider name (required) |
| `model` | string | Model identifier (required) |
| `temperature` | float | Sampling temperature (default: 0.0) |
| `api_key` | string | API key if required |
| `api_base` | string | Custom API endpoint |
| `extraction_prompt` | string | Custom prompt template |

## See Also

- [Prompt Engineering](prompts.md) - Prompt templates and rendering
- [Config Overrides](config-overrides.md) - Per-request configuration
- [Performance](performance.md) - Benchmarking and optimization

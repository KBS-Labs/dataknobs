"""Extraction observability for tracking and auditing schema extraction operations.

This module provides data structures for recording extraction operations,
enabling observability, debugging, and auditing of LLM-based data extraction.

Example:
    ```python
    from dataknobs_llm.extraction import SchemaExtractor, ExtractionTracker

    # Create extractor with tracking enabled
    extractor = SchemaExtractor.from_config(config)
    tracker = ExtractionTracker(max_history=100)

    # Extract with tracking
    result = await extractor.extract(
        text="My name is Alice and I'm 30 years old",
        schema=person_schema,
        tracker=tracker,  # Enable tracking
    )

    # Query extraction history
    history = tracker.get_history()
    stats = tracker.get_stats()
    print(f"Total extractions: {stats.total_extractions}")
    print(f"Success rate: {stats.success_rate:.1%}")
    ```
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExtractionRecord:
    """Record of a single schema extraction operation.

    Captures all relevant information about an extraction including
    timing, input/output, confidence, and validation results.

    Attributes:
        timestamp: Unix timestamp when extraction started
        input_text: The text that was processed (may be truncated)
        schema_name: Name/title from the schema, if provided
        schema_hash: Hash of the schema for identification
        extracted_data: The data that was extracted
        confidence: Confidence score (0.0 to 1.0)
        validation_errors: List of validation error messages
        duration_ms: Time taken for extraction in milliseconds
        success: Whether extraction was successful (confident with no errors)
        model_used: The model that performed the extraction
        provider: The provider type (ollama, openai, anthropic, etc.)
        context: Optional context dict that was provided
        raw_response: The raw LLM response (may be truncated)
        input_length: Length of the original input text
        truncated: Whether input_text or raw_response were truncated

    Example:
        ```python
        record = ExtractionRecord(
            timestamp=time.time(),
            input_text="Call my bot MathHelper with ID math-helper",
            schema_name="BotIdentity",
            extracted_data={"name": "MathHelper", "id": "math-helper"},
            confidence=0.95,
            validation_errors=[],
            duration_ms=150.5,
            success=True,
            model_used="claude-3-haiku-20240307",
            provider="anthropic",
        )
        ```
    """

    timestamp: float
    input_text: str
    extracted_data: dict[str, Any]
    confidence: float
    validation_errors: list[str]
    duration_ms: float
    success: bool
    schema_name: str | None = None
    schema_hash: str | None = None
    model_used: str | None = None
    provider: str | None = None
    context: dict[str, Any] | None = None
    raw_response: str | None = None
    input_length: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary.

        Returns:
            Dictionary representation of the record
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionRecord":
        """Create record from dictionary.

        Args:
            data: Dictionary containing record fields

        Returns:
            ExtractionRecord instance
        """
        return cls(**data)


@dataclass
class ExtractionStats:
    """Aggregated statistics for extraction operations.

    Attributes:
        total_extractions: Total number of extraction operations
        successful_extractions: Number of successful extractions
        failed_extractions: Number of failed extractions
        avg_confidence: Average confidence score across all extractions
        avg_duration_ms: Average extraction duration in milliseconds
        success_rate: Ratio of successful to total extractions
        most_common_errors: List of (error_message, count) tuples
        first_extraction: Timestamp of first extraction
        last_extraction: Timestamp of last extraction
        by_schema: Extraction counts per schema name
        by_model: Extraction counts per model
    """

    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    avg_confidence: float = 0.0
    avg_duration_ms: float = 0.0
    success_rate: float = 0.0
    most_common_errors: list[tuple[str, int]] = field(default_factory=list)
    first_extraction: float | None = None
    last_extraction: float | None = None
    by_schema: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractionHistoryQuery:
    """Query parameters for filtering extraction history.

    Attributes:
        schema_name: Filter by schema name
        model: Filter by model used
        success_only: Only include successful extractions
        failed_only: Only include failed extractions
        min_confidence: Minimum confidence threshold
        since: Filter to records after this timestamp
        until: Filter to records before this timestamp
        limit: Maximum number of records to return
    """

    schema_name: str | None = None
    model: str | None = None
    success_only: bool = False
    failed_only: bool = False
    min_confidence: float | None = None
    since: float | None = None
    until: float | None = None
    limit: int | None = None


class ExtractionTracker:
    """Tracks extraction history with query and statistics capabilities.

    Manages a bounded history of extraction operations and provides
    methods for querying and aggregating extraction data.

    Attributes:
        max_history: Maximum number of records to retain
        truncate_text_at: Maximum length for stored text fields

    Example:
        ```python
        tracker = ExtractionTracker(max_history=100)

        # Record an extraction (usually done automatically by SchemaExtractor)
        tracker.record(ExtractionRecord(
            timestamp=time.time(),
            input_text="User said: create a math tutor bot",
            extracted_data={"intent": "create", "type": "tutor"},
            confidence=0.92,
            validation_errors=[],
            duration_ms=125.0,
            success=True,
        ))

        # Query history
        recent = tracker.query(ExtractionHistoryQuery(
            success_only=True,
            min_confidence=0.8,
        ))

        # Get statistics
        stats = tracker.get_stats()
        print(f"Success rate: {stats.success_rate:.1%}")
        ```
    """

    def __init__(
        self,
        max_history: int = 100,
        truncate_text_at: int = 200,
    ):
        """Initialize tracker.

        Args:
            max_history: Maximum records to retain (default 100)
            truncate_text_at: Max length for text fields (default 200)
        """
        self._history: list[ExtractionRecord] = []
        self._max_history = max_history
        self._truncate_at = truncate_text_at

    def record(self, extraction: ExtractionRecord) -> None:
        """Record an extraction operation.

        Args:
            extraction: The extraction record to store
        """
        self._history.append(extraction)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def query(
        self, query: ExtractionHistoryQuery | None = None
    ) -> list[ExtractionRecord]:
        """Query extraction history.

        Args:
            query: Query parameters, or None for all records

        Returns:
            List of matching extraction records
        """
        if query is None:
            return list(self._history)

        results = self._history

        if query.schema_name:
            results = [r for r in results if r.schema_name == query.schema_name]

        if query.model:
            results = [r for r in results if r.model_used == query.model]

        if query.success_only:
            results = [r for r in results if r.success]

        if query.failed_only:
            results = [r for r in results if not r.success]

        if query.min_confidence is not None:
            results = [r for r in results if r.confidence >= query.min_confidence]

        if query.since:
            results = [r for r in results if r.timestamp >= query.since]

        if query.until:
            results = [r for r in results if r.timestamp <= query.until]

        if query.limit:
            results = results[-query.limit:]

        return results

    def get_stats(self) -> ExtractionStats:
        """Get aggregated extraction statistics.

        Returns:
            ExtractionStats with aggregated metrics
        """
        if not self._history:
            return ExtractionStats()

        # Basic counts
        total = len(self._history)
        successful = sum(1 for r in self._history if r.success)
        failed = total - successful

        # Averages
        total_confidence = sum(r.confidence for r in self._history)
        total_duration = sum(r.duration_ms for r in self._history)

        # Error frequency
        error_counts: dict[str, int] = {}
        for record in self._history:
            for error in record.validation_errors:
                error_counts[error] = error_counts.get(error, 0) + 1

        most_common_errors = sorted(
            error_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # By schema
        by_schema: dict[str, int] = {}
        for record in self._history:
            name = record.schema_name or "unknown"
            by_schema[name] = by_schema.get(name, 0) + 1

        # By model
        by_model: dict[str, int] = {}
        for record in self._history:
            model = record.model_used or "unknown"
            by_model[model] = by_model.get(model, 0) + 1

        return ExtractionStats(
            total_extractions=total,
            successful_extractions=successful,
            failed_extractions=failed,
            avg_confidence=total_confidence / total,
            avg_duration_ms=total_duration / total,
            success_rate=successful / total if total > 0 else 0.0,
            most_common_errors=most_common_errors,
            first_extraction=self._history[0].timestamp,
            last_extraction=self._history[-1].timestamp,
            by_schema=by_schema,
            by_model=by_model,
        )

    def get_recent(self, count: int = 10) -> list[ExtractionRecord]:
        """Get the most recent extraction records.

        Args:
            count: Number of records to return

        Returns:
            List of most recent extraction records
        """
        return self._history[-count:]

    def clear(self) -> None:
        """Clear all extraction history."""
        self._history.clear()

    def __len__(self) -> int:
        """Return number of records in history."""
        return len(self._history)


def create_extraction_record(
    input_text: str,
    extracted_data: dict[str, Any],
    confidence: float,
    validation_errors: list[str],
    duration_ms: float,
    schema: dict[str, Any] | None = None,
    model_used: str | None = None,
    provider: str | None = None,
    context: dict[str, Any] | None = None,
    raw_response: str | None = None,
    truncate_at: int = 200,
) -> ExtractionRecord:
    """Factory function to create an extraction record.

    Convenience function that automatically sets timestamp and handles
    text truncation.

    Args:
        input_text: The text that was processed
        extracted_data: The data that was extracted
        confidence: Confidence score
        validation_errors: List of validation errors
        duration_ms: Extraction duration in milliseconds
        schema: The JSON schema used (extracts name and computes hash)
        model_used: The model that performed extraction
        provider: The provider type
        context: Optional context dict
        raw_response: The raw LLM response
        truncate_at: Maximum length for text fields

    Returns:
        ExtractionRecord with current timestamp
    """
    import hashlib

    # Extract schema name and compute hash
    schema_name = schema.get("title") if schema else None
    schema_hash = None
    if schema:
        schema_str = str(sorted(schema.items()))
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]

    # Determine if truncation occurred
    input_length = len(input_text)
    truncated = False

    # Truncate text fields
    if len(input_text) > truncate_at:
        input_text = input_text[:truncate_at] + "..."
        truncated = True

    if raw_response and len(raw_response) > truncate_at:
        raw_response = raw_response[:truncate_at] + "..."
        truncated = True

    return ExtractionRecord(
        timestamp=time.time(),
        input_text=input_text,
        schema_name=schema_name,
        schema_hash=schema_hash,
        extracted_data=extracted_data,
        confidence=confidence,
        validation_errors=validation_errors,
        duration_ms=duration_ms,
        success=confidence >= 0.8 and not validation_errors,
        model_used=model_used,
        provider=provider,
        context=context,
        raw_response=raw_response,
        input_length=input_length,
        truncated=truncated,
    )


__all__ = [
    "ExtractionRecord",
    "ExtractionStats",
    "ExtractionHistoryQuery",
    "ExtractionTracker",
    "create_extraction_record",
]

"""Extraction utilities for structured data extraction from text.

This module provides SchemaExtractor for extracting structured data
from user input using LLM-based extraction with JSON Schema validation.

Observability support includes ExtractionTracker for recording and
querying extraction history.
"""

from dataknobs_llm.extraction.observability import (
    ExtractionHistoryQuery,
    ExtractionRecord,
    ExtractionStats,
    ExtractionTracker,
    create_extraction_record,
)
from dataknobs_llm.extraction.schema_extractor import (
    ExtractedAssumption,
    ExtractionResult,
    SchemaExtractor,
    SimpleExtractionResult,
)

__all__ = [
    # Core extraction
    "ExtractedAssumption",
    "ExtractionResult",
    "SchemaExtractor",
    # Stable re-export from testing (used as a production dependency by
    # consumers for lightweight extraction results without a full
    # SchemaExtractor pipeline).  The is_confident threshold (0.8) is a
    # stable contract.
    "SimpleExtractionResult",
    # Observability
    "ExtractionRecord",
    "ExtractionStats",
    "ExtractionHistoryQuery",
    "ExtractionTracker",
    "create_extraction_record",
]

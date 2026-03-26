"""Extraction utilities for structured data extraction from text.

This module provides SchemaExtractor for extracting structured data
from user input using LLM-based extraction with JSON Schema validation,
and SimpleExtractionResult for lightweight extraction results without
a full SchemaExtractor pipeline.

Grounding support validates extracted values against the user's actual
message to prevent hallucinated extractions.

Observability support includes ExtractionTracker for recording and
querying extraction history.
"""

from dataknobs_llm.extraction.grounding import (
    DEFAULT_NEGATION_KEYWORDS,
    DEFAULT_STOPWORDS,
    FieldGroundingResult,
    GroundingConfig,
    detect_boolean_signal,
    field_keywords,
    ground_extraction,
    has_negation,
    is_field_grounded,
    significant_words,
)
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
    # Lightweight extraction result (canonical definition in
    # schema_extractor.py alongside ExtractionResult).  Also re-exported
    # from dataknobs_llm.testing for backward compatibility.
    # The is_confident threshold (0.8) is a stable contract.
    "SimpleExtractionResult",
    # Grounding
    "DEFAULT_NEGATION_KEYWORDS",
    "DEFAULT_STOPWORDS",
    "FieldGroundingResult",
    "GroundingConfig",
    "detect_boolean_signal",
    "field_keywords",
    "ground_extraction",
    "has_negation",
    "is_field_grounded",
    "significant_words",
    # Observability
    "ExtractionRecord",
    "ExtractionStats",
    "ExtractionHistoryQuery",
    "ExtractionTracker",
    "create_extraction_record",
]

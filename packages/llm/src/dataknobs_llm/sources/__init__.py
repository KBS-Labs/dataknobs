"""Intent schema composition for grounded retrieval.

Bridges GroundedSource schema declarations (from dataknobs-data)
with SchemaExtractor (from dataknobs-llm).
"""

from .intent import compose_intent_schema, parse_intent

__all__ = [
    "compose_intent_schema",
    "parse_intent",
]

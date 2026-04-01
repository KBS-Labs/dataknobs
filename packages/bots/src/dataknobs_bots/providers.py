"""Provider creation utilities for dataknobs-bots.

Shared helpers for creating and initializing LLM providers used across
bot subsystems (memory, knowledge base, reasoning).

The canonical ``create_embedding_provider()`` implementation lives in
``dataknobs_llm`` and is re-exported here for backward compatibility.
"""

from __future__ import annotations

# Re-export from the canonical location in dataknobs-llm.
from dataknobs_llm import create_embedding_provider

__all__ = ["create_embedding_provider"]

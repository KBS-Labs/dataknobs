"""Provider creation utilities for dataknobs-bots.

Shared helpers for creating and initializing LLM providers used across
bot subsystems (memory, knowledge base, reasoning).

The canonical ``create_embedding_provider()`` implementation lives in
``dataknobs_llm`` and is re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import Any

# Re-export from the canonical location in dataknobs-llm.
from dataknobs_llm import create_embedding_provider


def build_embedding_config(
    *,
    embedding: dict[str, Any] | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Project typed embedding fields onto a ``create_embedding_provider`` dict.

    ``create_embedding_provider`` consumes a config dict, reading either
    a nested ``embedding`` sub-dict (preferred) or the legacy flat
    ``embedding_provider`` / ``embedding_model`` keys. For the legacy flat
    form it also reads top-level ``api_base`` / ``api_key`` / ``dimensions``
    as passthroughs. Subsystem consumers that hold these values as typed
    config fields call this to build the minimal dict — only set
    (non-``None``) keys are included, so the result matches the sparse raw
    dict the helper saw before structured-config adoption (forwarding
    ``dimensions=None`` etc. is avoided).

    The passthrough keys (``api_base`` / ``api_key`` / ``dimensions``) are
    only consumed by the helper's legacy-flat branch; when a nested
    ``embedding`` sub-dict is supplied the helper reads endpoint/key/dims
    from inside it, so any top-level values projected here are ignored —
    matching the pre-adoption whole-dict behavior exactly.

    Args:
        embedding: Nested embedding-provider config, when present.
        embedding_provider: Legacy flat provider key.
        embedding_model: Legacy flat model key.
        dimensions: Embedder dimension (plural), forwarded to the
            provider as a legacy passthrough.
        api_base: Legacy flat custom embedder endpoint passthrough.
        api_key: Legacy flat embedder credential passthrough.

    Returns:
        A dict containing only the keys whose values are not ``None``.
    """
    config: dict[str, Any] = {}
    if embedding is not None:
        config["embedding"] = embedding
    if embedding_provider is not None:
        config["embedding_provider"] = embedding_provider
    if embedding_model is not None:
        config["embedding_model"] = embedding_model
    if dimensions is not None:
        config["dimensions"] = dimensions
    if api_base is not None:
        config["api_base"] = api_base
    if api_key is not None:
        config["api_key"] = api_key
    return config


__all__ = ["build_embedding_config", "create_embedding_provider"]

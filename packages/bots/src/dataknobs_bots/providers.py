# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Provider creation utilities for dataknobs-bots.

Shared helpers for creating and initializing LLM providers used across
bot subsystems (memory, knowledge base, reasoning).

The canonical ``create_embedding_provider()`` implementation lives in
``dataknobs_llm`` and is re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import Any

# Re-export from the canonical location in dataknobs-llm.
from dataknobs_llm import create_embedding_provider, reads_nested_embedding


def build_embedding_config(
    *,
    embedding: dict[str, Any] | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    store_dimensions: int | None = None,
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

    ``store_dimensions`` closes the gap between a subsystem's two width
    knobs. A caller that builds **both** a vector store and an embedder —
    :class:`~dataknobs_bots.knowledge.rag.RAGKnowledgeBase` and
    :class:`~dataknobs_bots.memory.vector.VectorMemory` both do — writes the
    embedder's output straight into the store, so the two widths are one
    number. Stating it twice is how they come apart, and nothing compared
    them: a config naming only the store's width left the embedder on its
    own default, and the store accepted whatever arrived. Passing the
    store's declared width here supplies the embedder's when the config gave
    it none, so a number written once reaches both halves.

    It **supplies** rather than overrules. A width stated for the embedder —
    top-level for the flat form, inside the section for the nested one —
    wins, because stating both is deliberate and the caller may mean it. The
    rule is the one ``OllamaProvider.embed`` states for its own layer: a
    stated width is never ignored.

    Args:
        embedding: Nested embedding-provider config, when present.
        embedding_provider: Legacy flat provider key.
        embedding_model: Legacy flat model key.
        dimensions: Embedder dimension (plural), forwarded to the
            provider as a legacy passthrough.
        store_dimensions: Width the caller's own vector store was built
            with, used as the embedder's when none was stated for it. Omit
            it when the caller does not build a store.
        api_base: Legacy flat custom embedder endpoint passthrough.
        api_key: Legacy flat embedder credential passthrough.

    Returns:
        A dict containing only the keys whose values are not ``None``.
    """
    config: dict[str, Any] = {}
    # Which branch `create_embedding_provider` will take decides where a
    # supplied width goes, so it is *asked* rather than predicted. Writing
    # the condition out a second time here is what let the two drift: this
    # kept the truthy half and dropped the `isinstance` half, which is the
    # only half that can disagree, and the width then went where nothing
    # would read it -- or the copy below raised on a section that is not a
    # mapping, which the helper itself accepts and ignores.
    nested = reads_nested_embedding(embedding)
    if embedding is not None:
        if nested and store_dimensions is not None and "dimensions" not in embedding:
            # Copied, not mutated: this section is a live field on the
            # caller's config object, not a scratch dict.
            embedding = {**embedding, "dimensions": store_dimensions}
        config["embedding"] = embedding
    if embedding_provider is not None:
        config["embedding_provider"] = embedding_provider
    if embedding_model is not None:
        config["embedding_model"] = embedding_model
    if dimensions is None and not nested:
        dimensions = store_dimensions
    if dimensions is not None:
        config["dimensions"] = dimensions
    if api_base is not None:
        config["api_base"] = api_base
    if api_key is not None:
        config["api_key"] = api_key
    return config


__all__ = ["build_embedding_config", "create_embedding_provider"]

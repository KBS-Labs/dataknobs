# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A width stated once is the width both halves get.

``RAGKnowledgeBase`` and ``VectorMemory`` each build **both** a vector store
and an embedding provider, and each used to take the width twice: once for the
store (``vector_store.dimensions`` / ``dimension``) and once for the embedder
(``dimensions``, or ``dimensions`` inside the nested ``embedding`` dict). The
two were never compared, and ``_embed_and_store_chunks`` writes the embedder's
output straight into the store -- so a config that stated one and not the other
produced vectors of a width the store had declared it would not hold.

Nothing reported that. ``MemoryVectorStore.add_vectors`` accepted any width it
was handed, so a store declaring 384 and holding 768-wide rows searched fine as
long as every row and every query came from the same embedder, and the declared
number was simply a lie. Thirty-six test files in this repository stated 384
against an embedder producing 768 for as long as they had existed.

The rule these tests pin is the one ``OllamaProvider.embed`` already states for
its own layer -- *"a stated width is never ignored"*. Here that means the store's
declared width reaches the embedder when the config did not give the embedder
one of its own, so the single number a consumer wrote is the number both halves
use. An embedder width stated explicitly still wins: this supplies a missing
value, it does not overrule a given one.

The two subsystems share ``build_embedding_config``, which is where the rule
lives; the end-to-end cases below are what make it true of the surfaces a
consumer actually calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge.rag import RAGKnowledgeBase
from dataknobs_bots.memory.vector import VectorMemory
from dataknobs_bots.providers import build_embedding_config

WIDTH = 384
"""A width the echo provider does not default to, so agreement cannot be luck.

``EchoProvider`` falls back to 768. Any assertion below that reads 384 is
therefore reading a number that travelled, not one that happened to match.
"""


# ---------------------------------------------------------------------------
# The helper both subsystems call
# ---------------------------------------------------------------------------


def test_the_store_width_is_supplied_when_the_flat_form_states_none() -> None:
    """Legacy flat keys: the embedder gets the store's width as its own."""
    built = build_embedding_config(
        embedding_provider="echo",
        embedding_model="test",
        store_dimensions=WIDTH,
    )

    assert built["dimensions"] == WIDTH


def test_the_store_width_is_supplied_inside_the_nested_form() -> None:
    """Nested ``embedding``: the width has to go *inside* it to be read.

    ``create_embedding_provider`` reads the top-level ``dimensions`` only on
    its legacy-flat branch; with a nested section present it reads endpoint,
    key and width from inside that section. A top-level value would be
    accepted here and silently dropped there, which is the failure mode this
    case exists to rule out.
    """
    built = build_embedding_config(
        embedding={"provider": "echo", "model": "test"},
        store_dimensions=WIDTH,
    )

    assert built["embedding"]["dimensions"] == WIDTH


def test_an_empty_nested_section_is_treated_as_the_flat_form() -> None:
    """An empty ``embedding`` is not the nested form, and must not become one.

    ``create_embedding_provider`` takes its nested branch only for a truthy
    section, so an empty one falls through to the flat keys and the width
    belongs at the top level. Writing it inside instead would do worse than
    miss: a section holding ``{"dimensions": N}`` is truthy, so the nested
    branch *would* be taken, with no ``provider`` or ``model`` in it -- and
    the caller's echo embedder would silently become the ollama default.
    """
    built = build_embedding_config(
        embedding={},
        embedding_provider="echo",
        embedding_model="test",
        store_dimensions=WIDTH,
    )

    assert built["embedding"] == {}
    assert built["dimensions"] == WIDTH


def test_the_nested_form_takes_the_width_only_where_it_is_read() -> None:
    """No top-level twin beside the section's own width.

    The top-level key is dead on the nested branch. Emitting one anyway
    would put two widths in one dict with only one of them consulted, which
    is the shape this whole fallback exists to remove.
    """
    built = build_embedding_config(
        embedding={"provider": "echo", "model": "test"},
        store_dimensions=WIDTH,
    )

    assert "dimensions" not in built


def test_a_stated_embedder_width_is_not_overruled() -> None:
    """The store's width supplies a missing value; it does not win a contest.

    A consumer who states both has said something deliberate -- most usefully
    that the two differ and they know it. Overruling that would trade one
    silent disagreement for another.
    """
    flat = build_embedding_config(
        embedding_provider="echo",
        embedding_model="test",
        dimensions=16,
        store_dimensions=WIDTH,
    )
    nested = build_embedding_config(
        embedding={"provider": "echo", "model": "test", "dimensions": 16},
        store_dimensions=WIDTH,
    )

    assert flat["dimensions"] == 16
    assert nested["embedding"]["dimensions"] == 16


def test_the_nested_section_the_caller_passed_is_not_mutated() -> None:
    """The config dict a caller holds is theirs, and comes back unchanged.

    ``RAGKnowledgeBaseConfig.embedding`` is a live field on a config object a
    consumer may read afterwards or reuse; writing the store's width into it
    would make the projection a side effect on its own input.
    """
    section: dict[str, Any] = {"provider": "echo", "model": "test"}

    build_embedding_config(embedding=section, store_dimensions=WIDTH)

    assert section == {"provider": "echo", "model": "test"}


def test_nothing_is_added_when_no_store_width_is_known() -> None:
    """Sparse-dict parity: an absent width stays absent, not ``None``.

    The helper's contract is that only set keys appear, so a store config
    naming no width leaves the embedder exactly as it was before this rule
    existed.
    """
    assert build_embedding_config(embedding_provider="echo") == {
        "embedding_provider": "echo",
    }


# ---------------------------------------------------------------------------
# The surfaces a consumer calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rag_base_writes_the_width_its_store_declared() -> None:
    """The reproducer: 384 declared once, and the write used to be refused.

    Against the unfixed code the embedder produced 768, and
    ``load_markdown_text`` raised ``ValueError: Vector dimension mismatch:
    expected 384, got 768`` from inside ``add_vectors`` -- one config, two
    widths, and the store the one that noticed.
    """
    kb = await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": WIDTH},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    try:
        assert kb.embedding_provider.embedding_dim == WIDTH

        stored = await kb.load_markdown_text("# T\n\nBody.", source="t.md")

        assert stored > 0, "the ingest stored nothing, so nothing was written"
        assert kb.vector_store.dimensions == WIDTH
    finally:
        await kb.close()


@pytest.mark.asyncio
async def test_a_rag_base_using_the_nested_form_does_the_same() -> None:
    """The preferred config shape reaches the embedder by the other route."""
    kb = await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": WIDTH},
            "embedding": {"provider": "echo", "model": "test"},
        }
    )
    try:
        assert kb.embedding_provider.embedding_dim == WIDTH
        assert await kb.load_markdown_text("# T\n\nBody.", source="t.md") > 0
    finally:
        await kb.close()


@pytest.mark.asyncio
async def test_vector_memory_writes_the_width_its_store_declared() -> None:
    """The same defect on the other subsystem, whose two keys differ by an ``s``.

    ``VectorMemory`` spells the store's width ``dimension`` and the embedder's
    ``dimensions``. Its own docstring warns that the plural one is *"forwarded
    to the embedding provider, not the vector store"* -- a warning that exists
    because the pair is easy to get wrong, and that nothing enforced.
    """
    memory = await VectorMemory.from_config(
        {
            "backend": "memory",
            "dimension": WIDTH,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    try:
        assert memory.embedding_provider.embedding_dim == WIDTH

        await memory.add_message("Test message", "user")

        assert memory.vector_store.dimensions == WIDTH
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_a_width_stated_for_the_embedder_alone_still_reaches_it() -> None:
    """The pre-existing passthrough is untouched by the new fallback.

    A config naming the embedder's width and *not* the store's was already
    correct, and stays so: the fallback has nothing to supply, and the store
    takes the factory's own default rather than this value.
    """
    memory = await VectorMemory.from_config(
        {
            "backend": "memory",
            "dimension": 16,
            "dimensions": 16,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    try:
        assert memory.embedding_provider.embedding_dim == 16
        await memory.add_message("Test message", "user")
    finally:
        await memory.close()

"""RAGKnowledgeBase close-cascade ownership.

``RAGKnowledgeBase.close()`` cascades to its vector store and embedding
provider. When those collaborators are *built from config*, the knowledge
base owns them and closing it must close them. When they are *injected*
via :meth:`RAGKnowledgeBase.from_components` — the pattern a consumer uses
to share one store/provider across several knowledge bases — they are
caller-owned and must survive one base's close, or a shared backing
resource (e.g. an asyncpg pool behind a ``PgVectorStore``) gets torn down
out from under the other holders.

Reproduce-first: inject a shared store + provider, close the base, and
assert the injected collaborators are untouched and still usable. Before
the ownership gate, ``close()`` cascaded unconditionally — the injected
store's ``_initialized`` flipped to ``False`` and the injected provider's
``close_count`` reached ``1`` — so these assertions fail. The owned-path
test pins that a config-built store/provider IS still closed.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm.llm import LLMProviderFactory


async def _make_shared_store() -> Any:
    store = VectorStoreFactory().create(backend="memory", dimensions=384)
    await store.initialize()
    return store


async def _make_shared_provider() -> Any:
    provider = LLMProviderFactory(is_async=True).create(
        {"provider": "echo", "model": "test"}
    )
    await provider.initialize()
    return provider


@pytest.mark.asyncio
async def test_close_leaves_injected_collaborators_open() -> None:
    """Injected store + provider survive the knowledge base's close."""
    store = await _make_shared_store()
    provider = await _make_shared_provider()

    kb = RAGKnowledgeBase.from_components(
        {},
        vector_store=store,
        embedding_provider=provider,
    )

    await kb.close()

    # The injected collaborators were NOT closed.
    assert store._initialized is True
    assert provider.close_count == 0

    # ...and remain usable by the caller / other holders.
    assert await store.count() == 0
    embedding = await provider.embed("still works")
    assert embedding is not None


@pytest.mark.asyncio
async def test_close_leaves_shared_store_usable_for_second_kb() -> None:
    """Closing one knowledge base does not break a sibling sharing the store.

    The canonical consumer pattern: two knowledge bases over one shared
    store/provider. Closing the first must leave the second fully working.
    """
    store = await _make_shared_store()
    provider = await _make_shared_provider()

    kb_a = RAGKnowledgeBase.from_components(
        {}, vector_store=store, embedding_provider=provider
    )
    kb_b = RAGKnowledgeBase.from_components(
        {}, vector_store=store, embedding_provider=provider
    )

    await kb_a.close()

    # kb_b's shared store + provider are still live.
    assert store._initialized is True
    assert provider.close_count == 0
    assert await kb_b.vector_store.count() == 0


@pytest.mark.asyncio
async def test_close_closes_owned_collaborators() -> None:
    """Config-built store + provider ARE closed (owned path preserved)."""
    kb = await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding": {"provider": "echo", "model": "test"},
        }
    )
    owned_store = kb.vector_store
    owned_provider = kb.embedding_provider
    assert owned_store._initialized is True
    assert owned_provider.close_count == 0

    await kb.close()

    # The owned collaborators were closed by the cascade.
    assert owned_store._initialized is False
    assert owned_provider.close_count == 1

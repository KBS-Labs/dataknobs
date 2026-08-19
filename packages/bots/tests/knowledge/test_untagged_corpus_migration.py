"""The documented cleanup for a corpus written before the binding existed.

Binding a knowledge base changes what ``count()`` means, and that is the
count ``check_needs_ingestion`` reads. For the population this binding
exists to repair — several domains over one *unscoped* shared store —
the rows already there were written before anything stamped ``domain_id``
into chunk metadata. A newly-bound knowledge base counts none of them,
concludes it has never been ingested, and ingests: the corpus is stored a
second time, and the untagged copy is invisible to every scoped read.

The part that makes it a migration rather than a wrinkle is that a bound
``clear()`` cannot remove the old copy either — it composes a tag those
rows do not carry. The escape hatch is the one the binding always
documented: an **unbound** knowledge base over the same store.

Nothing here is automatic, and that is deliberate. Untagged rows on a
shared store belong to no single domain — several domains wrote them and
collided on the same ids, which is the defect being repaired — so
adopting them into whichever binding looked first would invent an
answer. Re-ingest is the repair. These tests pin the recipe that makes
it complete, so a future change cannot quietly take the escape hatch
away.
"""

from __future__ import annotations

from typing import Any

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_bots.knowledge.service import KnowledgeIngestionService
from dataknobs_bots.providers import build_embedding_config, create_embedding_provider
from dataknobs_data.vector.stores import VectorStoreFactory


async def _shared_unscoped_store() -> Any:
    store = VectorStoreFactory().create(backend="memory", dimensions=768)
    await store.initialize()
    return store


async def _kb(store: Any, config: dict[str, Any] | None) -> RAGKnowledgeBase:
    embedder = await create_embedding_provider(
        build_embedding_config(embedding_provider="echo", embedding_model="test")
    )
    return RAGKnowledgeBase.from_components(config, vector_store=store, embedding_provider=embedder)


async def _seed_untagged(store: Any) -> None:
    """The pre-binding corpus: ids from the source stem, no domain tag."""
    await store.add_vectors(
        [[0.1] * 768, [0.2] * 768],
        ids=["overview_0", "guide_0"],
        metadata=[{"source": "overview.md"}, {"source": "guide.md"}],
    )


async def test_a_binding_hides_the_untagged_corpus_from_its_own_count() -> None:
    """The condition the migration note exists for, stated as a fact."""
    store = await _shared_unscoped_store()
    await _seed_untagged(store)
    kb = await _kb(store, {"domain_id": "bot-a"})

    assert await store.count() == 2, "precondition: the store is not empty"
    assert await kb.count() == 0, "a bound count sees only tagged rows"
    assert await KnowledgeIngestionService().check_needs_ingestion(kb) is True


async def test_a_bound_clear_cannot_remove_the_untagged_corpus() -> None:
    """Why the migration needs a recipe rather than a re-ingest."""
    store = await _shared_unscoped_store()
    await _seed_untagged(store)
    kb = await _kb(store, {"domain_id": "bot-a"})

    await kb.clear()

    assert await store.count() == 2, (
        "a bound clear reached rows it does not tag — or the escape hatch moved"
    )


async def test_an_unbound_kb_over_the_same_store_removes_it() -> None:
    """The documented recipe, pinned so it cannot be taken away."""
    store = await _shared_unscoped_store()
    await _seed_untagged(store)

    await (await _kb(store, None)).clear()

    assert await store.count() == 0


async def test_the_recipe_leaves_each_domain_correct() -> None:
    """End to end: clear unbound, then let each domain re-ingest itself."""
    store = await _shared_unscoped_store()
    await _seed_untagged(store)

    await (await _kb(store, None)).clear()

    kb_a = await _kb(store, {"domain_id": "bot-a"})
    kb_b = await _kb(store, {"domain_id": "bot-b"})
    await kb_a.load_markdown_text("# Overview\n\nAlpha.\n", source="overview.md")
    await kb_b.load_markdown_text("# Overview\n\nBeta.\n", source="overview.md")

    assert await kb_a.count() == 1
    assert await kb_b.count() == 1
    assert await store.count() == 2, "the two domains' overviews no longer collide"
    assert sorted(store.metadata_store) == [
        "bot-a\x1foverview\x1f0",
        "bot-b\x1foverview\x1f0",
    ]

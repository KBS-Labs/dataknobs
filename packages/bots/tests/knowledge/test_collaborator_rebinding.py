"""Rebinding a knowledge base's collaborators must not corrupt derived state.

``vector_store`` and ``embedding_provider`` are public attributes, and
two things are derived from them that a plain assignment silently
invalidates.

**The chunk-id namespace.** ``_domain_id`` falls back to the store's own
scope, and the effective domain folds into every chunk id
(``_CHUNK_ID_PREFIX_KEYS``). Swap in a store with a different scope and
every id already written is under a prefix the knowledge base will never
compose again: ``count()`` stops seeing them, the skip-if-populated gate
re-ingests over rows it can no longer see, and ``clear()`` cannot reach
them. Nothing raises.

**The ownership flags.** ``close()`` tears down only what this instance
owns. Replacing a config-built collaborator without clearing the flag
leaks the one being replaced *and* closes the caller's replacement — the
exact inversion the ownership gate exists to prevent.
``QueryTransformer.set_provider`` already clears its flag on injection;
these are the siblings that did not.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm.llm import LLMProviderFactory

_DIM = 384


async def _store(**kwargs: Any) -> Any:
    store = VectorStoreFactory().create(backend="memory", dimensions=_DIM, **kwargs)
    await store.initialize()
    return store


async def _provider() -> Any:
    provider = LLMProviderFactory(is_async=True).create({"provider": "echo", "model": "test"})
    await provider.initialize()
    return provider


async def _kb(**config: Any) -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": _DIM},
            "embedding_provider": "echo",
            "embedding_model": "test",
            **config,
        }
    )


class TestTheBindingCannotChangeUnderWrittenIds:
    async def test_swapping_to_a_differently_scoped_store_is_refused(self) -> None:
        """The corruption case: ids already written become unreachable."""
        kb = await _kb(vector_store={"backend": "memory", "dimensions": _DIM, "domain_id": "bot-a"})
        assert kb._domain_id == "bot-a"

        with pytest.raises(ConfigurationError) as excinfo:
            kb.vector_store = await _store(domain_id="bot-b")

        assert "bot-a" in str(excinfo.value) and "bot-b" in str(excinfo.value)
        assert kb._domain_id == "bot-a", "the refused swap took effect anyway"
        await kb.close()

    async def test_swapping_away_from_a_scoped_store_is_refused(self) -> None:
        """Losing the binding orphans the ids just as thoroughly as changing it."""
        kb = await _kb(vector_store={"backend": "memory", "dimensions": _DIM, "domain_id": "bot-a"})

        with pytest.raises(ConfigurationError):
            kb.vector_store = await _store()

        assert kb._domain_id == "bot-a"
        await kb.close()

    async def test_acquiring_a_binding_is_refused_too(self) -> None:
        """Unscoped -> scoped splits the corpus the same way.

        Chunks written while unbound carry no domain in their prefix;
        after the swap every id this instance composes carries one, so
        the earlier chunks are just as unreachable as in the other
        direction. Asserted separately because "the binding changed"
        reads as a narrower condition than it is.
        """
        kb = await _kb()
        with pytest.raises(ConfigurationError):
            kb.vector_store = await _store(domain_id="bot-a")
        assert kb._domain_id is None
        await kb.close()

    async def test_a_binding_preserving_swap_is_allowed(self) -> None:
        """Swapping backends under the same binding keeps every id valid.

        This is the shape the in-repo swap uses — a different store
        implementation, same scope — so refusing it would refuse the
        legitimate case along with the corrupting one.
        """
        kb = await _kb()
        replacement = await _store()
        kb.vector_store = replacement
        assert kb.vector_store is replacement
        await kb.close()

    async def test_an_explicit_config_domain_pins_the_binding(self) -> None:
        """A configured ``domain_id`` outranks the store's, so any store fits.

        The binding cannot move, so there is nothing to orphan — the
        check must not refuse on the store's scope alone.
        """
        kb = await _kb(domain_id="pinned")
        kb.vector_store = await _store()
        assert kb._domain_id == "pinned"
        await kb.close()

    async def test_construction_is_not_a_swap(self) -> None:
        """The first bind has no prior ids to orphan, whatever its scope."""
        kb = await RAGKnowledgeBase.from_config(
            {
                "vector_store": {"backend": "memory", "dimensions": _DIM, "domain_id": "bot-a"},
                "embedding_provider": "echo",
                "embedding_model": "test",
            }
        )
        assert kb._domain_id == "bot-a"
        await kb.close()

    async def test_from_components_is_not_a_swap(self) -> None:
        """Neither is the injection entry point."""
        store = await _store(domain_id="bot-a")
        kb = RAGKnowledgeBase.from_components(
            vector_store=store, embedding_provider=await _provider()
        )
        assert kb._domain_id == "bot-a"
        await kb.close()


class TestRebindingHandsOwnershipBack:
    async def test_a_replaced_store_is_no_longer_ours_to_close(self) -> None:
        """The incoming store came from outside, so we must not close it."""
        kb = await _kb()
        assert kb._owns_vector_store is True

        replacement = await _store()
        kb.vector_store = replacement
        await kb.close()

        assert replacement._initialized is True, "close() tore down the caller's store"

    async def test_set_provider_hands_back_the_embedder(self) -> None:
        """``set_provider`` had the same hole, without even an assignment.

        A config-built knowledge base owns its embedder. Injecting a
        replacement left the flag True, so ``close()`` closed the
        caller's provider — and the owned one it replaced was never
        closed at all.
        """
        from dataknobs_bots.bot.base import PROVIDER_ROLE_KB_EMBEDDING

        kb = await _kb()
        assert kb._owns_embedding_provider is True

        injected = await _provider()
        assert kb.set_provider(PROVIDER_ROLE_KB_EMBEDDING, injected) is True
        assert kb._owns_embedding_provider is False
        await kb.close()

        assert injected.close_count == 0, "close() closed the injected provider"

    async def test_an_unrecognised_role_changes_nothing(self) -> None:
        kb = await _kb()
        assert kb.set_provider("some.other.role", await _provider()) is False
        assert kb._owns_embedding_provider is True
        await kb.close()

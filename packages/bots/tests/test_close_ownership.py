"""Collaborator-ownership close gating across the bots close-cascade.

A holder that *builds* a collaborator from config owns its lifecycle and
closes it; a holder handed a pre-built collaborator (shared across several
holders) does NOT own it and must leave it open. These tests inject a
shared collaborator into one holder, close that holder, and assert the
shared collaborator is still usable — the reproduce-first guard for the
"close() tears down an injected collaborator" bug class.

Real constructs only (no mocks): ``EchoProvider`` (built-in
``close_count``), a thin close-counting ``SyncMemoryDatabase`` /
``AsyncMemoryDatabase`` subclass that still exercises the real backend,
and ``MemoryVectorStore``.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.memory.base import Memory
from dataknobs_data.backends.memory import (
    AsyncMemoryDatabase,
    SyncMemoryDatabase,
)
from dataknobs_llm.conversations import DataknobsConversationStorage


class CountingKnowledgeBase(KnowledgeBase):
    """Real ``KnowledgeBase`` that records ``close()`` invocations.

    A genuine implementation of the abstract interface (query + close);
    the only instrumentation is a close counter. Not a mock.
    """

    def __init__(self) -> None:
        self.close_count = 0
        self.closed = False

    async def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.closed:
            raise RuntimeError("knowledge base is closed")
        return [{"text": f"result for {query}"}]

    async def close(self) -> None:
        self.close_count += 1
        self.closed = True


class CountingSyncDB(SyncMemoryDatabase):
    """Real ``SyncMemoryDatabase`` that records ``close()`` invocations.

    Exercises the real in-memory backend; the only addition is a counter
    so a test can assert whether a holder closed this db. Not a mock — every
    method runs the genuine code path.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()


class CountingAsyncDB(AsyncMemoryDatabase):
    """Async sibling of :class:`CountingSyncDB`."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1
        await super().close()


# =====================================================================
# MemoryBank — sync db ownership
# =====================================================================


class TestMemoryBankDbOwnership:
    """``MemoryBank`` closes its db only when it owns it."""

    def test_injected_db_not_closed(self) -> None:
        """A caller-supplied db survives the bank's close()."""
        from dataknobs_bots.memory.bank import MemoryBank

        shared = CountingSyncDB()
        bank = MemoryBank(name="b", schema={}, db=shared)

        bank.close()

        assert shared.close_count == 0, "injected db must not be closed"
        # The shared db is still usable by another holder.
        other = MemoryBank(name="b2", schema={}, db=shared)
        other.add({"x": 1})
        assert other.count() == 1

    def test_two_banks_one_shared_db(self) -> None:
        """Closing one bank does not tear down a db shared with another."""
        from dataknobs_bots.memory.bank import MemoryBank

        shared = CountingSyncDB()
        bank_a = MemoryBank(name="a", schema={}, db=shared)
        bank_b = MemoryBank(name="b", schema={}, db=shared)
        bank_a.add({"v": 1})

        bank_a.close()

        assert shared.close_count == 0
        bank_b.add({"v": 2})  # still works
        assert bank_b.count() >= 1

    def test_owned_db_is_closed(self) -> None:
        """An owned db is closed by the bank's close()."""
        from dataknobs_bots.memory.bank import MemoryBank

        owned = CountingSyncDB()
        bank = MemoryBank(name="b", schema={}, db=owned, owns_db=True)

        bank.close()

        assert owned.close_count == 1, "owned db must be closed"

    def test_from_dict_builds_owned_db(self) -> None:
        """from_dict with db=None builds a fresh db the bank owns."""
        from dataknobs_bots.memory.bank import MemoryBank

        bank = MemoryBank.from_dict({"name": "b", "schema": {}})
        assert isinstance(bank._db, SyncMemoryDatabase)
        assert bank._owns_db is True

    def test_from_dict_injected_db_not_owned(self) -> None:
        """from_dict with an explicit db treats it as caller-owned."""
        from dataknobs_bots.memory.bank import MemoryBank

        shared = CountingSyncDB()
        bank = MemoryBank.from_dict({"name": "b", "schema": {}}, db=shared)

        bank.close()

        assert bank._owns_db is False
        assert shared.close_count == 0


# =====================================================================
# VectorKnowledgeSource — injected KB is never owned
# =====================================================================


class TestVectorSourceKbOwnership:
    """``VectorKnowledgeSource`` always wraps a caller-supplied KB and
    must never close it."""

    @pytest.mark.asyncio
    async def test_injected_kb_not_closed(self) -> None:
        from dataknobs_bots.knowledge.sources.vector import (
            VectorKnowledgeSource,
        )

        shared = CountingKnowledgeBase()
        source = VectorKnowledgeSource(shared, name="kb")

        await source.close()

        assert shared.close_count == 0, "injected KB must not be closed"
        # The shared KB is still usable directly and by a second source
        # wrapping it; that source's close() is also a no-op for the KB.
        assert await shared.query("hi")
        other = VectorKnowledgeSource(shared, name="kb2")
        await other.close()
        assert shared.close_count == 0
        assert await shared.query("again")

    @pytest.mark.asyncio
    async def test_owns_kb_flag_default_false(self) -> None:
        from dataknobs_bots.knowledge.sources.vector import (
            VectorKnowledgeSource,
        )

        source = VectorKnowledgeSource(CountingKnowledgeBase(), name="kb")
        assert source._owns_kb is False


# =====================================================================
# CompositeMemory — audit verdict: owns its sub-strategies (both paths)
# =====================================================================
#
# A CompositeMemory's sub-strategies are dedicated to it, not shared across
# composites, so it closes all of them. Any genuinely shared *backing*
# resource lives inside a child (e.g. a VectorMemory's vector store) and is
# protected by that child's own ownership gate — so no composite-level gate
# is needed. This is the "audit-only — gate if so" verdict resolving to
# "not so".


class CountingMemory(Memory):
    """Real ``Memory`` implementation that records ``close()`` calls."""

    def __init__(self) -> None:
        self.close_count = 0
        self._messages: list[dict[str, Any]] = []

    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._messages.append({"content": content, "role": role})

    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        return list(self._messages)

    async def clear(self) -> None:
        self._messages.clear()

    async def close(self) -> None:
        self.close_count += 1


class TestCompositeMemoryClosesChildren:
    """``CompositeMemory`` owns and closes its sub-strategies."""

    @pytest.mark.asyncio
    async def test_children_closed(self) -> None:
        from dataknobs_bots.memory.composite import CompositeMemory

        m1, m2 = CountingMemory(), CountingMemory()
        composite = CompositeMemory.from_components(strategies=[m1, m2])

        await composite.close()

        assert m1.close_count == 1 and m2.close_count == 1

    @pytest.mark.asyncio
    async def test_child_protects_its_own_injected_backing_resource(self) -> None:
        """A VectorMemory child leaves its injected vector store open even
        though the composite closes the child."""
        from dataknobs_bots.memory.composite import CompositeMemory
        from dataknobs_bots.memory.vector import VectorMemory
        from dataknobs_data.vector.stores.memory import MemoryVectorStore
        from dataknobs_llm import EchoProvider

        store = MemoryVectorStore(dimensions=8)
        await store.initialize()
        embedder = EchoProvider({"provider": "echo", "model": "test"})
        vec = VectorMemory.from_components(
            vector_store=store, embedding_provider=embedder
        )
        composite = CompositeMemory.from_components(strategies=[vec])

        await composite.close()

        # The composite closed its VectorMemory child, but the child left
        # the injected store + embedder open (its own ownership gate).
        assert embedder.close_count == 0
        await store.add_vectors([[0.0] * 8], ["m"], [{}])  # still usable


# =====================================================================
# GroundedReasoning — provider / extractor / source ownership
# =====================================================================


class CountingSource:
    """Real ``GroundedSource``-shaped stub recording ``close()`` calls."""

    def __init__(self, name: str = "stub") -> None:
        self._name = name
        self.close_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return "stub"

    async def query(
        self, intent: Any, *, top_k: int = 5, score_threshold: float = 0.0
    ) -> list[Any]:
        return []

    async def close(self) -> None:
        self.close_count += 1


class CountingExtractor:
    """Minimal extractor stub recording ``close()`` calls."""

    def __init__(self) -> None:
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


class TestGroundedReasoningOwnership:
    """``GroundedReasoning`` closes only the collaborators it built."""

    def _strategy(self) -> Any:
        from dataknobs_bots.reasoning.grounded import GroundedReasoning
        from dataknobs_bots.reasoning.grounded_config import (
            GroundedReasoningConfig,
        )

        return GroundedReasoning(config=GroundedReasoningConfig())

    @pytest.mark.asyncio
    async def test_injected_source_not_closed(self) -> None:
        from dataknobs_bots.reasoning.grounded import GroundedReasoning
        from dataknobs_bots.reasoning.grounded_config import (
            GroundedReasoningConfig,
        )

        injected = CountingSource("shared")
        strategy = GroundedReasoning.from_config(
            GroundedReasoningConfig(), sources=[injected]
        )

        await strategy.close()

        assert injected.close_count == 0, "injected source must not be closed"

    @pytest.mark.asyncio
    async def test_added_source_is_owned_and_closed(self) -> None:
        strategy = self._strategy()
        owned = CountingSource("owned")
        strategy.add_source(owned)

        await strategy.close()

        assert owned.close_count == 1, "config-added source must be closed"

    @pytest.mark.asyncio
    async def test_add_source_owns_false_not_closed(self) -> None:
        strategy = self._strategy()
        shared = CountingSource("shared")
        strategy.add_source(shared, owns=False)

        await strategy.close()

        assert shared.close_count == 0

    @pytest.mark.asyncio
    async def test_injected_extractor_not_closed(self) -> None:
        strategy = self._strategy()
        ext = CountingExtractor()
        strategy.set_extractor(ext)

        await strategy.close()

        assert ext.close_count == 0, "injected extractor must not be closed"

    @pytest.mark.asyncio
    async def test_injected_query_provider_not_closed(self) -> None:
        from dataknobs_llm import EchoProvider

        strategy = self._strategy()
        provider = EchoProvider({"provider": "echo", "model": "test"})
        strategy.set_provider("grounded_query", provider)

        await strategy.close()

        assert provider.close_count == 0, "injected provider must not be closed"


# =====================================================================
# DynaBot — cascade collaborator ownership
# =====================================================================


class CountingConversationStorage(DataknobsConversationStorage):
    """Real conversation storage that records ``close()`` invocations."""

    def __init__(self, db: Any) -> None:
        super().__init__(db)
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1
        await super().close()


def _prompt_builder() -> Any:
    from dataknobs_llm.prompts import AsyncPromptBuilder
    from dataknobs_llm.prompts.implementations import CompositePromptLibrary

    return AsyncPromptBuilder(CompositePromptLibrary())


class TestDynaBotCascadeOwnership:
    """``DynaBot`` closes injected cascade collaborators only when owned."""

    @pytest.mark.asyncio
    async def test_injected_kb_storage_memory_not_closed(self) -> None:
        from dataknobs_bots.bot.base import DynaBot
        from dataknobs_llm import EchoProvider

        shared_kb = CountingKnowledgeBase()
        shared_storage = CountingConversationStorage(AsyncMemoryDatabase())
        shared_memory = CountingMemory()
        provider = EchoProvider({"provider": "echo", "model": "test"})

        bot = DynaBot.from_components(
            llm=provider,
            prompt_builder=_prompt_builder(),
            conversation_storage=shared_storage,
            knowledge_base=shared_kb,
            memory=shared_memory,
        )
        assert bot._owns_knowledge_base is False
        assert bot._owns_conversation_storage is False
        assert bot._owns_memory is False
        assert bot._owns_reasoning_strategy is False

        await bot.close()

        assert shared_kb.close_count == 0, "injected KB must not be closed"
        assert shared_storage.close_count == 0, "injected storage not closed"
        assert shared_memory.close_count == 0, "injected memory not closed"
        # The shared KB still serves a second bot over the same instance.
        bot2 = DynaBot.from_components(
            llm=EchoProvider({"provider": "echo", "model": "test"}),
            prompt_builder=_prompt_builder(),
            conversation_storage=CountingConversationStorage(AsyncMemoryDatabase()),
            knowledge_base=shared_kb,
        )
        assert await shared_kb.query("still works")
        await bot2.close()
        assert shared_kb.close_count == 0

    @pytest.mark.asyncio
    async def test_config_built_collaborators_owned(self) -> None:
        from dataknobs_bots.bot.base import DynaBot

        bot = await DynaBot.from_config({
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 5},
            "reasoning": {"strategy": "simple"},
        })
        assert bot._owns_conversation_storage is True
        assert bot._owns_memory is True
        assert bot._owns_reasoning_strategy is True
        await bot.close()

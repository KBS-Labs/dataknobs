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

import asyncio
from typing import Any

import pytest

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.memory.base import Memory
from dataknobs_data.backends.memory import (
    AsyncMemoryDatabase,
    SyncMemoryDatabase,
)
from dataknobs_data.sources.base import GroundedSource
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

    def test_from_dict_built_db_owned_despite_explicit_owns_false(self) -> None:
        """An internally-built db is owned even if owns_db=False is passed.

        The bank built the db itself, so the caller holds no reference to
        close it; honoring an explicit owns_db=False would leak it. The
        contradictory input warns and is forced to ownership rather than
        leaked.
        """
        import warnings

        from dataknobs_bots.memory.bank import MemoryBank

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank = MemoryBank.from_dict(
                {"name": "b", "schema": {}}, owns_db=False
            )

        assert isinstance(bank._db, SyncMemoryDatabase)
        assert bank._owns_db is True
        assert any(
            issubclass(w.category, UserWarning)
            and "owns_db=False is ignored" in str(w.message)
            for w in caught
        ), "contradictory owns_db=False with db=None should warn"


# =====================================================================
# AsyncMemoryBank — async db ownership (sync parity)
# =====================================================================


class TestAsyncMemoryBankDbOwnership:
    """``AsyncMemoryBank`` closes its db only when it owns it — the async
    mirror of :class:`TestMemoryBankDbOwnership`.

    Teardown-reaching tests are parametrized over ``close`` and ``aclose``
    so both public teardown methods are proven to reach the owned db.
    Uses the real ``CountingAsyncDB`` (an ``AsyncMemoryDatabase`` subclass),
    not a mock.
    """

    @pytest.mark.asyncio
    async def test_injected_db_not_closed(self) -> None:
        """A caller-supplied db survives the bank's close() (default owns_db=False)."""
        from dataknobs_bots.memory.bank import AsyncMemoryBank

        shared = CountingAsyncDB()
        bank = AsyncMemoryBank(name="b", schema={}, db=shared)

        await bank.close()

        assert shared.close_count == 0, "injected db must not be closed"
        # The shared db is still usable by another holder.
        other = AsyncMemoryBank(name="b2", schema={}, db=shared)
        await other.add({"x": 1})
        assert await other.count() == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["close", "aclose"])
    async def test_owned_db_is_closed(self, method: str) -> None:
        """An owned db is closed by the bank's close()/aclose()."""
        from dataknobs_bots.memory.bank import AsyncMemoryBank

        owned = CountingAsyncDB()
        bank = AsyncMemoryBank(name="b", schema={}, db=owned, owns_db=True)

        await getattr(bank, method)()

        assert owned.close_count == 1, f"owned db must be closed via {method}()"

    @pytest.mark.asyncio
    async def test_from_dict_builds_owned_db(self) -> None:
        """from_dict with db=None builds a fresh db the bank owns."""
        from dataknobs_bots.memory.bank import AsyncMemoryBank

        bank = await AsyncMemoryBank.from_dict({"name": "b", "schema": {}})

        assert isinstance(bank._db, AsyncMemoryDatabase)
        assert bank._owns_db is True
        await bank.close()  # does not raise

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["close", "aclose"])
    async def test_from_dict_owned_self_built_db_is_closed(
        self, method: str
    ) -> None:
        """The from_dict leak site: an owned db routed through from_dict is
        torn down. Routes an instrumentable owned db through the new
        ``from_dict`` db param so the close is observable.
        """
        from dataknobs_bots.memory.bank import AsyncMemoryBank

        owned = CountingAsyncDB()
        bank = await AsyncMemoryBank.from_dict(
            {"name": "b", "schema": {}}, db=owned, owns_db=True
        )

        await getattr(bank, method)()

        assert owned.close_count == 1, "from_dict-owned db must be closed"

    @pytest.mark.asyncio
    async def test_from_dict_injected_db_not_owned(self) -> None:
        """from_dict with an explicit db treats it as caller-owned."""
        from dataknobs_bots.memory.bank import AsyncMemoryBank

        shared = CountingAsyncDB()
        bank = await AsyncMemoryBank.from_dict(
            {"name": "b", "schema": {}}, db=shared
        )

        await bank.close()

        assert bank._owns_db is False
        assert shared.close_count == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["close", "aclose"])
    async def test_from_dict_built_db_owned_despite_explicit_owns_false(
        self, method: str
    ) -> None:
        """An internally-built db is owned even if owns_db=False is passed.

        The bank built the db itself, so the caller holds no reference to
        close it; honoring an explicit owns_db=False would leak it. The
        contradictory input warns and is forced to ownership.
        """
        import warnings

        from dataknobs_bots.memory.bank import AsyncMemoryBank

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank = await AsyncMemoryBank.from_dict(
                {"name": "b", "schema": {}}, owns_db=False
            )

        assert isinstance(bank._db, AsyncMemoryDatabase)
        assert bank._owns_db is True
        assert any(
            issubclass(w.category, UserWarning)
            and "owns_db=False is ignored" in str(w.message)
            for w in caught
        ), "contradictory owns_db=False with db=None should warn"
        await getattr(bank, method)()  # does not raise

    @pytest.mark.asyncio
    async def test_protocol_conformance(self) -> None:
        """The bank satisfies AsyncBankProtocol and exposes both teardowns."""
        from dataknobs_bots.memory.bank import (
            AsyncBankProtocol,
            AsyncMemoryBank,
        )

        bank = AsyncMemoryBank(name="b", schema={}, db=CountingAsyncDB())

        assert isinstance(bank, AsyncBankProtocol)
        assert hasattr(bank, "close") and hasattr(bank, "aclose")


# =====================================================================
# VectorKnowledgeSource — injected KB is never owned
# =====================================================================


class TestVectorSourceKbOwnership:
    """``VectorKnowledgeSource`` always wraps a caller-supplied KB and
    must never close it.
    """

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
        though the composite closes the child.
        """
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


class CountingSource(GroundedSource):
    """Real ``GroundedSource`` implementation recording ``close()`` calls.

    Subclasses the genuine ABC and implements its abstract surface
    (``name`` / ``source_type`` / ``query``); ``close()`` counts and then
    chains to the base no-op. Not a mock — a config-built owned source is
    structurally identical to this.
    """

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
        await super().close()


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


# =====================================================================
# ArtifactBank — section-db ownership (delegates to MemoryBank.close)
# =====================================================================


class RaisingSyncDB(SyncMemoryDatabase):
    """Real ``SyncMemoryDatabase`` whose ``close()`` raises.

    Used to prove ``ArtifactBank.close()`` isolates one failing section's
    teardown from its siblings. Not a mock — the backend is genuine; only
    ``close()`` is overridden to fail.
    """

    def close(self) -> None:
        raise RuntimeError("section close failure")


def _artifact_from_config_with_counting_dbs(
    section_names: list[str],
) -> tuple[Any, dict[str, CountingSyncDB]]:
    """Build an ``ArtifactBank`` via ``from_config`` handing each section a
    ``CountingSyncDB`` (owned), so teardown of the owned section dbs is
    observable. Returns ``(artifact, {section_name: db})``.
    """
    from dataknobs_bots.memory.artifact_bank import ArtifactBank

    dbs: dict[str, CountingSyncDB] = {}

    def factory(name: str, cfg: dict[str, Any]) -> tuple[Any, str]:
        db = CountingSyncDB()
        dbs[name] = db
        return db, "external"

    config = {
        "name": "recipe",
        "fields": {"recipe_name": {"required": True}},
        "sections": {name: {"schema": {}} for name in section_names},
    }
    artifact = ArtifactBank.from_config(config, db_factory=factory)
    return artifact, dbs


class TestArtifactBankClose:
    """``ArtifactBank.close()`` releases the dbs its sections own and leaves
    caller-injected section dbs open.
    """

    def test_close_closes_owned_section_dbs(self) -> None:
        """The §2.3 leak: a from_config-built artifact closes every owned
        section db.
        """
        artifact, dbs = _artifact_from_config_with_counting_dbs(
            ["ingredients", "instructions"]
        )

        artifact.close()

        assert len(dbs) == 2
        assert all(db.close_count == 1 for db in dbs.values()), (
            "every owned section db must be closed"
        )

    def test_close_leaves_injected_section_db_open(self) -> None:
        """A section handed a caller-owned db must leave it open — the
        leak's inverse (a shared backing store survives).
        """
        from dataknobs_bots.memory.artifact_bank import ArtifactBank
        from dataknobs_bots.memory.bank import MemoryBank

        shared = CountingSyncDB()
        section = MemoryBank(name="s", schema={}, db=shared)  # owns_db=False
        artifact = ArtifactBank(
            name="a", field_defs={}, sections={"s": section}
        )

        artifact.close()

        assert shared.close_count == 0, "injected section db must not be closed"
        section.add({"x": 1})  # still usable
        assert section.count() == 1

    def test_close_twice_is_safe(self) -> None:
        """Closing an artifact twice does not raise.

        There is no dedup layer — ``close()`` delegates to each section on
        every call — but the backend's ``close()`` is safe to repeat, so the
        second call is harmless.
        """
        artifact, dbs = _artifact_from_config_with_counting_dbs(["s"])

        artifact.close()
        artifact.close()  # must not raise

        assert dbs["s"].close_count == 2

    def test_close_isolates_section_failure(self) -> None:
        """One section whose close() raises does not prevent the sibling
        section's db from being closed (the per-section try/except).
        """
        from dataknobs_bots.memory.artifact_bank import ArtifactBank
        from dataknobs_bots.memory.bank import MemoryBank

        good = CountingSyncDB()
        bad_section = MemoryBank(
            name="bad", schema={}, db=RaisingSyncDB(), owns_db=True
        )
        good_section = MemoryBank(
            name="good", schema={}, db=good, owns_db=True
        )
        artifact = ArtifactBank(
            name="a",
            field_defs={},
            sections={"bad": bad_section, "good": good_section},
        )

        artifact.close()  # must not raise despite the failing section

        assert good.close_count == 1, "sibling section must still be closed"


# =====================================================================
# ArtifactBankCatalog — owned-vs-injected db close
# =====================================================================


class TestArtifactBankCatalogClose:
    """``ArtifactBankCatalog`` closes its db only when it owns it."""

    def test_from_config_owns_and_closes_db(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The §2.2 leak: a from_config-built catalog owns its db (built +
        connected by the factory) and closes it.
        """
        import dataknobs_data
        from dataknobs_bots.memory.catalog import ArtifactBankCatalog

        created: list[CountingSyncDB] = []

        def fake_create(**kwargs: Any) -> CountingSyncDB:
            db = CountingSyncDB()
            created.append(db)
            return db

        monkeypatch.setattr(
            dataknobs_data.database_factory, "create", fake_create
        )

        catalog = ArtifactBankCatalog.from_config({"backend": "memory"})

        assert catalog._owns_db is True
        assert len(created) == 1
        assert created[0].close_count == 0

        catalog.close()

        assert created[0].close_count == 1, "owned catalog db must be closed"

    def test_injected_db_not_closed(self) -> None:
        """A caller-supplied catalog db survives close() (default owns_db)."""
        from dataknobs_bots.memory.catalog import ArtifactBankCatalog

        shared = CountingSyncDB()
        catalog = ArtifactBankCatalog(shared)

        assert catalog._owns_db is False
        catalog.close()

        assert shared.close_count == 0, "injected catalog db must not be closed"
        # Still usable by its owner after the catalog's close().
        assert catalog.count() == 0

    def test_close_twice_is_safe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing an owned catalog twice does not raise.

        There is no ``_closed`` guard and ``_owns_db`` is not flipped after
        close, so a second ``close()`` re-invokes the owned db's ``close()``;
        the memory backend tolerates repetition, so the second call is
        harmless. Mirrors ``ArtifactBank.test_close_twice_is_safe``.
        """
        import dataknobs_data
        from dataknobs_bots.memory.catalog import ArtifactBankCatalog

        created: list[CountingSyncDB] = []

        def fake_create(**kwargs: Any) -> CountingSyncDB:
            db = CountingSyncDB()
            created.append(db)
            return db

        monkeypatch.setattr(
            dataknobs_data.database_factory, "create", fake_create
        )

        catalog = ArtifactBankCatalog.from_config({"backend": "memory"})

        catalog.close()
        catalog.close()  # must not raise

        assert created[0].close_count == 2


# =====================================================================
# WizardReasoning — artifact-catalog close cascade
# =====================================================================


class TestWizardCatalogCascade:
    """``WizardReasoning.close()`` closes the catalog it creates, and the
    banks loop still closes the section dbs.

    Constructs ``WizardReasoning`` directly (a legitimate internal-lifecycle
    unit test per the DynaBot testing mandate's exception) and drives the
    real ``_init_artifact`` wiring, monkeypatching the database factory so
    both the section db and the catalog db are observable ``CountingSyncDB``
    instances.
    """

    @pytest.mark.asyncio
    async def test_wizard_close_closes_catalog_and_sections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dataknobs_data
        from dataknobs_bots.reasoning.wizard import WizardReasoning
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        created: list[CountingSyncDB] = []

        def fake_create(**kwargs: Any) -> CountingSyncDB:
            db = CountingSyncDB()
            created.append(db)
            return db

        monkeypatch.setattr(
            dataknobs_data.database_factory, "create", fake_create
        )

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict({
            "name": "w",
            "version": "1.0",
            "settings": {},
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "t",
                },
            ],
        })
        strategy = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # A non-memory section backend routes _create_bank_db through the
        # (patched) factory; the catalog's from_config always does. Both
        # dbs are owned (section owns_db=True, catalog owns_db=True).
        strategy._init_artifact({
            "name": "recipe",
            "fields": {"recipe_name": {"required": True}},
            "sections": {"ingredients": {"backend": "sqlite", "schema": {}}},
            "catalog": {"backend": "memory"},
        })

        assert len(created) == 2, "one section db + one catalog db"
        assert all(db.close_count == 0 for db in created)

        await strategy.close()

        assert all(db.close_count == 1 for db in created), (
            "wizard close() must reach every owned db (section + catalog)"
        )

    @pytest.mark.asyncio
    async def test_wizard_close_isolates_failing_section_still_closes_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A section db whose close() raises must not orphan the catalog db.

        M1: the wizard close-cascade closes the section banks, then the
        catalog. Without per-step error isolation, a section bank that
        raises on close propagates out of the banks loop and the
        ``catalog.close()`` beneath it never runs — leaking exactly the
        owned db connection the cascade exists to release. The failing
        section is isolated (logged) so the catalog is still closed.
        """
        import dataknobs_data
        from dataknobs_bots.reasoning.wizard import WizardReasoning
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        catalog_dbs: list[CountingSyncDB] = []

        def fake_create(**kwargs: Any) -> SyncMemoryDatabase:
            # The section routes through the factory with a ``table`` kwarg
            # (set by ``_create_bank_db``); the catalog's from_config does
            # not. So the section db raises on close and the catalog db is
            # an observable CountingSyncDB.
            if "table" in kwargs:
                return RaisingSyncDB()
            db = CountingSyncDB()
            catalog_dbs.append(db)
            return db

        monkeypatch.setattr(
            dataknobs_data.database_factory, "create", fake_create
        )

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict({
            "name": "w",
            "version": "1.0",
            "settings": {},
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "t",
                },
            ],
        })
        strategy = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        strategy._init_artifact({
            "name": "recipe",
            "fields": {"recipe_name": {"required": True}},
            "sections": {"ingredients": {"backend": "sqlite", "schema": {}}},
            "catalog": {"backend": "memory"},
        })

        assert len(catalog_dbs) == 1, "one catalog db created"

        # Must not raise despite the section bank's close() failing.
        await strategy.close()

        assert catalog_dbs[0].close_count == 1, (
            "catalog db must still be closed after a section close() raises"
        )

    @pytest.mark.asyncio
    async def test_wizard_close_twice_is_safe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing the wizard cascade twice does not raise.

        ``close()`` clears the per-conversation bank-state map after tearing
        down every resident conversation, so a second ``close()`` finds no
        slots and is a clean idempotent no-op — each owned db is closed exactly
        once (``close_count == 1``), not re-delegated.
        """
        import dataknobs_data
        from dataknobs_bots.reasoning.wizard import WizardReasoning
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        created: list[CountingSyncDB] = []

        def fake_create(**kwargs: Any) -> CountingSyncDB:
            db = CountingSyncDB()
            created.append(db)
            return db

        monkeypatch.setattr(
            dataknobs_data.database_factory, "create", fake_create
        )

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict({
            "name": "w",
            "version": "1.0",
            "settings": {},
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "t",
                },
            ],
        })
        strategy = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        strategy._init_artifact({
            "name": "recipe",
            "fields": {"recipe_name": {"required": True}},
            "sections": {"ingredients": {"backend": "sqlite", "schema": {}}},
            "catalog": {"backend": "memory"},
        })

        await strategy.close()
        await strategy.close()  # must not raise

        assert all(db.close_count == 1 for db in created), (
            "close() clears the slot map, so a second close is a no-op"
        )


# =====================================================================
# WizardReasoning restore — release prior-turn banks (no cross-turn leak)
# =====================================================================


async def _real_conversation_manager() -> Any:
    """Build a real ``ConversationManager`` for driving save→restore.

    Mirrors the wizard-test conftest fixture inline (this module lives
    outside ``tests/unit/`` and cannot use that fixture). Real constructs
    only — ``EchoProvider``, ``ConfigPromptLibrary``, in-memory storage.
    """
    from dataknobs_llm.conversations import ConversationManager
    from dataknobs_llm.llm import LLMConfig
    from dataknobs_llm.llm.providers.echo import EchoProvider
    from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary

    provider = EchoProvider(
        LLMConfig(
            provider="echo", model="echo-test", options={"echo_prefix": ""}
        )
    )
    library = ConfigPromptLibrary({
        "system": {"assistant": {"template": "You are a helpful assistant."}},
    })
    builder = AsyncPromptBuilder(library=library)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    return await ConversationManager.create(
        llm=provider,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="assistant",
    )


class TestWizardRestoreReleasesPriorBanks:
    """``WizardReasoning`` restore closes the prior turn's owned bank dbs
    before rebuilding them, so a persistent-backend wizard does not orphan a
    connection on every turn.

    The strategy object outlives a single turn (``self._banks`` /
    ``self._artifact`` are strategy-level, not per-conversation), so every
    ``_get_wizard_state`` restore rebuilds them. A non-memory section/bank
    backend opens a real connection each rebuild; without release-before-
    rebuild the prior connection is orphaned. Constructs ``WizardReasoning``
    directly (a legitimate internal-lifecycle unit test per the DynaBot
    testing mandate's exception) and drives a real save→restore.
    """

    @pytest.mark.asyncio
    async def test_restore_closes_prior_artifact_section_db(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A save→restore cycle closes the init-built section db before the
        artifact rebuild opens a fresh one.
        """
        import dataknobs_data
        from dataknobs_bots.reasoning.wizard import WizardReasoning
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        created: list[CountingSyncDB] = []

        def fake_create(**kwargs: Any) -> CountingSyncDB:
            db = CountingSyncDB()
            created.append(db)
            return db

        monkeypatch.setattr(
            dataknobs_data.database_factory, "create", fake_create
        )

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict({
            "name": "w",
            "version": "1.0",
            "settings": {},
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "t",
                },
            ],
        })
        strategy = WizardReasoning(wizard_fsm=fsm, strict_validation=False)
        # A non-memory section routes _create_bank_db through the patched
        # factory and opens a (fake) connection the section owns.
        strategy._init_artifact({
            "name": "recipe",
            "fields": {"recipe_name": {"required": True}},
            "sections": {"ingredients": {"backend": "sqlite", "schema": {}}},
        })

        assert len(created) == 1, "init built one owned section db"
        init_section_db = created[0]
        assert init_section_db.close_count == 0

        manager = await _real_conversation_manager()
        # First access: fresh state (no persisted fsm_state) — banks untouched.
        state = strategy._get_wizard_state(manager)
        assert len(created) == 1
        # Persist banks + artifact + fsm_state to the manager's metadata.
        await strategy._save_wizard_state(manager, state)

        # Second access: restore branch rebuilds the artifact, opening a new
        # section connection. The prior (init) section db must be closed first.
        strategy._get_wizard_state(manager)

        assert init_section_db.close_count == 1, (
            "restore must close the prior turn's section db before rebuilding"
        )
        # The rebuild really did open a fresh connection (the leak window is
        # real — the old db was not merely reused in place).
        assert len(created) >= 2, "restore rebuilt the section with a new db"

        await strategy.close()

    @pytest.mark.asyncio
    async def test_restore_rebuilds_config_added_bank_open_not_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bank added to config after the last save is rebuilt fresh and
        OPEN on restore — not left referencing a just-closed connection.

        The restore releases the prior turn's banks, then rebuilds. A bank the
        persisted snapshot omits (config drift: added to the wizard config
        after that save) must be self-healed into a fresh open bank; without
        the fix it stayed in ``self._banks`` still referencing the connection
        the release had just closed (closed-but-referenced → use-after-close).
        """
        import dataknobs_data
        from dataknobs_bots.reasoning.wizard import WizardReasoning
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        created: list[CountingSyncDB] = []

        def fake_create(**kwargs: Any) -> CountingSyncDB:
            db = CountingSyncDB()
            created.append(db)
            return db

        monkeypatch.setattr(
            dataknobs_data.database_factory, "create", fake_create
        )

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict({
            "name": "w",
            "version": "1.0",
            "settings": {
                "banks": {
                    "alpha": {"backend": "sqlite", "schema": {}},
                    "beta": {"backend": "sqlite", "schema": {}},
                }
            },
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "t",
                },
            ],
        })
        strategy = WizardReasoning(wizard_fsm=fsm, strict_validation=False)
        assert set(strategy._banks) == {"alpha", "beta"}

        manager = await _real_conversation_manager()
        state = strategy._get_wizard_state(manager)
        await strategy._save_wizard_state(manager, state)

        # Simulate config drift: 'beta' was added to the wizard config after
        # this conversation's last save, so the persisted snapshot omits it.
        del manager.metadata["wizard"]["banks"]["beta"]

        strategy._get_wizard_state(manager)  # restore

        # 'beta' must be present and OPEN — rebuilt fresh, not left referencing
        # the connection the restore's release just closed.
        assert "beta" in strategy._banks, (
            "config-added bank must survive restore"
        )
        assert strategy._banks["beta"]._db.close_count == 0, (
            "config-added bank must be rebuilt with a fresh open db, not left "
            "referencing the closed connection"
        )
        # It is genuinely usable after restore (write + read round-trips).
        strategy._banks["beta"].add({"v": 1})
        assert strategy._banks["beta"].count() == 1

        await strategy.close()


# =====================================================================
# WizardReasoning — per-conversation bank scoping (concurrency isolation)
# =====================================================================


def _sqlite_bank_wizard(
    *, ephemeral_keys: list[str] | None = None
) -> Any:
    """Build a ``WizardReasoning`` with one persistent (sqlite) memory bank.

    A non-memory backend routes ``_create_bank_db`` through the (patched-in
    each test) ``database_factory.create``, so the section's owned db is an
    observable ``CountingSyncDB``. Constructs the strategy directly — a
    legitimate internal-lifecycle unit test per the DynaBot testing mandate's
    exception.
    """
    from dataknobs_bots.reasoning.wizard import WizardReasoning
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    settings: dict[str, Any] = {
        "banks": {"ledger": {"backend": "sqlite", "schema": {}}}
    }
    if ephemeral_keys:
        settings["ephemeral_keys"] = ephemeral_keys
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict({
        "name": "w",
        "version": "1.0",
        "settings": settings,
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "is_end": True,
                "prompt": "t",
            },
        ],
    })
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


def _patch_counting_factory(
    monkeypatch: pytest.MonkeyPatch, created: list[CountingSyncDB]
) -> None:
    """Route ``database_factory.create`` to append a ``CountingSyncDB``."""
    import dataknobs_data

    def fake_create(**kwargs: Any) -> CountingSyncDB:
        db = CountingSyncDB()
        created.append(db)
        return db

    monkeypatch.setattr(dataknobs_data.database_factory, "create", fake_create)


class TestWizardPerConversationScoping:
    """``WizardReasoning`` scopes its live memory banks per conversation, so
    concurrent conversations served by one strategy neither clobber nor tear
    down each other's bank database connections.

    Every test builds ONE strategy with a persistent (sqlite) bank and drives
    two real ``ConversationManager`` instances (distinct conversation ids)
    through it. RED at ``main`` @ ``70569b63`` (single shared ``self._banks``
    slot), GREEN after per-conversation scoping.
    """

    @pytest.mark.asyncio
    async def test_concurrent_restore_does_not_close_other_conversations_bank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The post-#526 worsened mode: restoring conversation A must not
        close conversation B's live bank db connection.

        With a single shared slot, A's restore-path ``_close_banks()`` closes
        whatever bank is in the slot — B's, if B touched it last (post-#526
        loud-closed-connection). Per-conversation scoping re-scopes that
        teardown to A's slot only.
        """
        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard()

        mgr_a = await _real_conversation_manager()
        mgr_b = await _real_conversation_manager()

        # A: first turn (adopts the construction slot), persist so A restores.
        state_a = strategy._get_wizard_state(mgr_a)
        await strategy._save_wizard_state(mgr_a, state_a)

        # B: first turn builds B's own bank. Capture B's live section db while
        # B is the active conversation in this task.
        state_b = strategy._get_wizard_state(mgr_b)
        b_bank = strategy._banks["ledger"]
        b_db = b_bank._db
        await strategy._save_wizard_state(mgr_b, state_b)

        # A restore → re-scoped _close_banks closes only A's bank.
        strategy._get_wizard_state(mgr_a)

        assert b_db.close_count == 0, (
            "restoring conversation A must not close conversation B's bank db"
        )
        # B's bank is still open and usable.
        b_bank.add({"v": 1})
        assert b_bank.count() == 1

        await strategy.close()

    @pytest.mark.asyncio
    async def test_concurrent_state_isolation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After A's restore, resolving B's banks (under B's key) still yields
        B's own bank objects — not A's rebuilt ones.
        """
        from dataknobs_bots.reasoning.wizard import _active_conversation

        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard()

        mgr_a = await _real_conversation_manager()
        mgr_b = await _real_conversation_manager()

        state_a = strategy._get_wizard_state(mgr_a)
        await strategy._save_wizard_state(mgr_a, state_a)
        strategy._get_wizard_state(mgr_b)
        b_bank = strategy._banks["ledger"]

        # A restore rebinds A's slot; B's slot must be untouched.
        strategy._get_wizard_state(mgr_a)

        _active_conversation.set(mgr_b.conversation_id)
        assert strategy._banks["ledger"] is b_bank, (
            "B's banks must be isolated from A's restore"
        )

        await strategy.close()

    @pytest.mark.asyncio
    async def test_fresh_conversation_gets_own_banks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh conversation builds its own banks rather than inheriting
        the previous conversation's shared-slot banks.
        """
        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard()

        mgr_a = await _real_conversation_manager()
        mgr_b = await _real_conversation_manager()

        strategy._get_wizard_state(mgr_a)
        a_bank = strategy._banks["ledger"]

        strategy._get_wizard_state(mgr_b)
        b_bank = strategy._banks["ledger"]

        assert b_bank is not a_bank, (
            "a fresh conversation must not inherit another's banks"
        )

        await strategy.close()

    @pytest.mark.asyncio
    async def test_eviction_closes_only_the_evicted_conversation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``on_conversation_evicted`` closes only the evicted conversation's
        owned bank db; a concurrent conversation's bank stays open.
        """
        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard()

        mgr_a = await _real_conversation_manager()
        mgr_b = await _real_conversation_manager()

        strategy._get_wizard_state(mgr_a)
        a_db = strategy._banks["ledger"]._db
        strategy._get_wizard_state(mgr_b)
        b_bank = strategy._banks["ledger"]
        b_db = b_bank._db

        strategy.on_conversation_evicted(mgr_a.conversation_id)

        assert a_db.close_count == 1, "evicted conversation's db must close"
        assert b_db.close_count == 0, "other conversation's db must stay open"
        b_bank.add({"v": 1})
        assert b_bank.count() == 1

        await strategy.close()

    @pytest.mark.asyncio
    async def test_close_tears_down_all_resident_conversations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strategy ``close()`` tears down every resident conversation's banks
        — not merely the most recently accessed one — and clears the slot map.
        """
        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard()

        mgr_a = await _real_conversation_manager()
        mgr_b = await _real_conversation_manager()

        strategy._get_wizard_state(mgr_a)
        a_db = strategy._banks["ledger"]._db
        strategy._get_wizard_state(mgr_b)
        b_db = strategy._banks["ledger"]._db

        await strategy.close()

        assert a_db.close_count == 1, "conversation A's db must be closed"
        assert b_db.close_count == 1, "conversation B's db must be closed"
        assert strategy._conv_state == {}, "slot map must be cleared on close"

    @pytest.mark.asyncio
    async def test_ephemeral_task_cancellation_is_per_conversation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``close()`` cancels pending ephemeral asyncio tasks for EVERY
        resident conversation, not just the last-accessed one.
        """
        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard(ephemeral_keys=["_bg_task"])
        assert "_bg_task" in strategy._ephemeral_keys

        mgr_a = await _real_conversation_manager()
        mgr_b = await _real_conversation_manager()

        state_a = strategy._get_wizard_state(mgr_a)
        task_a = asyncio.ensure_future(asyncio.sleep(3600))
        state_a.data["_bg_task"] = task_a

        state_b = strategy._get_wizard_state(mgr_b)
        task_b = asyncio.ensure_future(asyncio.sleep(3600))
        state_b.data["_bg_task"] = task_b

        await strategy.close()

        assert task_a.cancelled(), "conversation A's ephemeral task must cancel"
        assert task_b.cancelled(), "conversation B's ephemeral task must cancel"

    @pytest.mark.asyncio
    async def test_single_conversation_uses_exactly_one_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parity guard: a single conversation adopts the construction slot
        (byte-identical single-conversation build) — no slot proliferation.
        """
        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard()

        # Construction built exactly one owned bank db.
        assert len(created) == 1

        mgr = await _real_conversation_manager()
        # First turn adopts the construction slot — no second db built.
        state = strategy._get_wizard_state(mgr)
        assert len(created) == 1, "first turn must adopt, not rebuild"
        assert len(strategy._conv_state) == 1, "exactly one conversation slot"

        await strategy._save_wizard_state(mgr, state)
        await strategy.close()

    @pytest.mark.asyncio
    async def test_asyncio_task_locality(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two interleaved asyncio tasks — each running a turn for a different
        conversation with an ``await`` between the active-key set and the bank
        read — each resolve their OWN conversation's banks (ContextVar is
        copied per task, so one task's active conversation cannot leak into
        the other across the await).
        """
        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard()

        mgr_a = await _real_conversation_manager()
        mgr_b = await _real_conversation_manager()

        async def run_turn(manager: Any, hold: float) -> Any:
            strategy._get_wizard_state(manager)  # sets the task-local key
            await asyncio.sleep(hold)  # yield so the other task interleaves
            return strategy._banks["ledger"]

        bank_a, bank_b = await asyncio.gather(
            run_turn(mgr_a, 0.02),
            run_turn(mgr_b, 0.0),
        )

        assert bank_a is not bank_b, (
            "each task must resolve its own conversation's banks across await"
        )

        await strategy.close()

    @pytest.mark.asyncio
    async def test_undo_reverts_active_conversation_banks_from_fresh_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Undo back through the first turn reverts THE UNDOING conversation's
        banks even when the undo runs in a fresh request task.

        A server runs each request in its own asyncio task: the chat that
        populated the banks ran in one task, a later ``undo_last_turn`` runs in
        another. The chat task's ``_active_conversation.set(<conv>)`` does NOT
        leak into the undo task (context is copied per task), so the undo
        starts at the default key. Undoing through the *first* turn is the
        empty-anchor sentinel (``checkpoint_node_id`` = ``None``) — the path
        where the bot skips ``restore_from_checkpoint`` entirely, so
        ``undo_to_checkpoint`` is the only place the active key can be set.

        RED before the manager-keyed fix (regression from per-conversation
        scoping, issue #527): ``undo_to_checkpoint`` read the default slot and
        left the conversation's real banks untouched, so the conversation
        reused stale turn-1 records after an undo. GREEN: keying the revert off
        ``manager`` reverts this conversation's slot regardless of entry key.
        """
        created: list[CountingSyncDB] = []
        _patch_counting_factory(monkeypatch, created)
        strategy = _sqlite_bank_wizard()
        mgr = await _real_conversation_manager()

        async def first_turn() -> None:
            # Own task, mirroring a chat request: its active-key .set() stays
            # local to this task and does not leak to the root test task.
            state = strategy._get_wizard_state(mgr)
            strategy._banks["ledger"].add({"v": "turn-1"})
            await strategy._save_wizard_state(mgr, state)

        await asyncio.ensure_future(first_turn())

        # From the root task, reach the conversation's live bank directly by id
        # (the slot dict is not ContextVar-scoped) to assert across the undo.
        conv_bank = strategy._conv_state[mgr.conversation_id].banks["ledger"]
        assert conv_bank.count() == 1

        async def undo_first_turn() -> None:
            # Fresh request task: default active-conversation key (the chat
            # task's .set did not leak here). Empty-anchor undo → id None.
            strategy.undo_to_checkpoint(mgr, None)

        await asyncio.ensure_future(undo_first_turn())

        assert conv_bank.count() == 0, (
            "undo through the first turn must revert THIS conversation's banks "
            "to empty, not operate on the default slot"
        )

        await strategy.close()

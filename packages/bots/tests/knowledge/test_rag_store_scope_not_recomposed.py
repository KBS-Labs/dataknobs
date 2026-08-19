"""A knowledge base must not re-compose the scope its store already enforces.

``RAGKnowledgeBase`` resolves its ``domain_id`` from the bound vector
store when the config does not set one, which is what gives a consumer
who scoped the *store* a namespaced chunk-id prefix without configuring
the same value twice. Composing that resolved value back into the
**filter** is a different act, and it is not free.

A configured store scope is an *isolation* guarantee the store delivers
uniformly: it confines every read, count, clear and update to that
domain on every backend, whether by ``_effective_filter``, by
``_in_configured_domain``, or — on ``PgVectorStore`` — by a predicate on
a dedicated ``domain_id`` column. A caller *explicitly* naming
``domain_id`` in a filter is the surface the store layer documents as
deliberately **not** uniform: pgvector stores caller metadata JSONB
verbatim, so an explicit key is a containment probe against a key the
column consumed, orthogonal to the column scope and answering zero for
the configured domain as readily as for another one.

So composing the store's own scope moves the knowledge base off the
contract that holds everywhere and onto the one that does not, buying
nothing — the store was already enforcing exactly that equality. The
rows it costs are the ones written before the knowledge base began
stamping ``domain_id`` into chunk metadata: on pgvector they carry the
domain in the column and nowhere else, so the probe excludes the whole
pre-existing corpus, `count()` answers 0, and the skip-if-populated
check re-ingests over the top of a corpus it can no longer see.

An **explicitly configured** ``domain_id`` over an *unscoped* store is
the opposite case and still composes: nothing else is enforcing it.
"""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_bots.knowledge.service import KnowledgeIngestionService
from dataknobs_bots.providers import build_embedding_config, create_embedding_provider
from dataknobs_data.vector.stores import VectorStoreFactory


async def _embedder() -> Any:
    return await create_embedding_provider(
        build_embedding_config(embedding_provider="echo", embedding_model="test")
    )


async def _kb_over(store: Any, config: dict[str, Any] | None = None) -> RAGKnowledgeBase:
    return RAGKnowledgeBase.from_components(
        config, vector_store=store, embedding_provider=await _embedder()
    )


class TestTheStoreScopeIsNotComposedIntoFilters:
    """Backend-independent: the resolved scope must not reach the filter.

    Asserted on the composed filter rather than on retrieval, because
    the rows this costs are the ones a *backend* declines to match on a
    metadata key it keeps in a column — and the in-process backends put
    ``domain_id`` in metadata on write, so retrieval against them cannot
    show the difference. The behavioural proof is the pgvector test
    below; this one has teeth wherever it runs, including where no
    Postgres is reachable.
    """

    async def test_a_store_derived_scope_is_left_to_the_store(self) -> None:
        """The store enforces its own scope; naming it again is the defect."""
        store = VectorStoreFactory().create(backend="memory", dimensions=384, domain_id="bot-a")
        await store.initialize()
        kb = await _kb_over(store)

        assert kb._domain_id == "bot-a", "the binding still resolves for chunk ids"
        assert kb._resolve_read_filter(None) is None, (
            "the store's own scope was composed into the read filter"
        )
        assert kb._scope_for_write(None) is None, (
            "the store's own scope was composed into a write filter"
        )

    async def test_an_explicit_scope_over_an_unscoped_store_still_composes(self) -> None:
        """Nothing else enforces this one, so the knowledge base must."""
        store = VectorStoreFactory().create(backend="memory", dimensions=384)
        await store.initialize()
        kb = await _kb_over(store, {"domain_id": "bot-a"})

        assert kb._resolve_read_filter(None) == {"domain_id": "bot-a"}
        assert kb._scope_for_write(None) == {"domain_id": "bot-a"}

    async def test_a_tenant_binding_is_unaffected(self) -> None:
        """``tenant_id`` has no store to defer to and always composes."""
        store = VectorStoreFactory().create(backend="memory", dimensions=384, domain_id="bot-a")
        await store.initialize()
        kb = await _kb_over(store, {"tenant_id": "acme"})

        assert kb._resolve_read_filter(None) == {"tenant_id": "acme"}
        assert kb._scope_for_write(None) == {"tenant_id": "acme"}


@requires_postgres
class TestAgainstARealDomainScopedPgVectorStore:
    """The behavioural half, on the one backend where it is visible.

    ``PgVectorStore`` is the only backend that keeps ``domain_id`` in a
    column instead of in the metadata blob, so it is the only one on
    which composing the key can exclude a row that is genuinely in
    scope. Skips without a reachable server, ``TEST_POSTGRES=true`` and
    asyncpg; the structural tests above cover the same invariant
    everywhere else.
    """

    @pytest.fixture
    def pgvector_config(self, make_pgvector_test_table: Any) -> Any:
        yield from make_pgvector_test_table("test_kb_domain_scope_", dimensions=768)

    async def test_rows_the_store_scoped_stay_readable(self, pgvector_config: Any) -> None:
        """A corpus written without a metadata tag is still this domain's.

        Seeded through the store directly, which is what every ingest
        before the chunk-metadata stamp looked like: the domain reaches
        the column from the store's own config and appears nowhere in
        the JSONB. A knowledge base over that store must still read,
        count and clear it.
        """
        store = VectorStoreFactory().create(**{**pgvector_config, "domain_id": "bot-a"})
        await store.initialize()
        try:
            await store.add_vectors(
                [[0.1] * 768, [0.2] * 768],
                ids=["legacy_0", "legacy_1"],
                metadata=[{"source": "overview.md"}, {"source": "guide.md"}],
            )

            kb = await _kb_over(store)
            service = KnowledgeIngestionService()

            assert await kb.count() == 2, "the store's own rows were filtered out"
            assert await service.check_needs_ingestion(kb) is False, (
                "a populated knowledge base was told to ingest over itself"
            )

            results = await kb.query("overview", k=5)
            assert results, "a domain-scoped query returned none of its own rows"

            await kb.clear()
            assert await kb.count() == 0, "clear() could not reach the rows it owns"
        finally:
            await store.close()

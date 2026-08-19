"""Behavioral tests for AutoIngestionMixin auto-ingestion entry points.

Pins the ``completed_at`` invariant on the mixin's two terminal-state
construction sites — KB-disabled skip and exception handler — and
backfills missing behavioral coverage for ``_ensure_knowledge_base_ingested``.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from dataknobs_bots.knowledge.config import RAGKnowledgeBaseConfig
from dataknobs_bots.knowledge.registry_mixin import AutoIngestionMixin
from dataknobs_bots.knowledge.service import KnowledgeIngestionService
from dataknobs_data.vector.stores import VectorStoreFactory


class _MinimalMixinUser(AutoIngestionMixin):
    """Minimal class wiring AutoIngestionMixin's required attrs."""

    def __init__(self) -> None:
        self._auto_ingest = True
        self._ingestion_service = KnowledgeIngestionService()


class TestAutoIngestionMixinDisabledSkip:
    async def test_disabled_kb_returns_skipped_with_completed_at(self) -> None:
        """KB-disabled skip path constructs a result with completed_at populated."""
        mixin = _MinimalMixinUser()
        result = await mixin._ensure_knowledge_base_ingested(
            domain_id="x",
            config={"bot": {"knowledge_base": {"enabled": False}}},
        )
        assert result.skipped is True
        assert result.reason == "knowledge_base_disabled"
        assert result.completed_at is not None

    async def test_missing_knowledge_base_section_returns_skipped(self) -> None:
        """Absent knowledge_base config defaults to disabled and populates completed_at."""
        mixin = _MinimalMixinUser()
        result = await mixin._ensure_knowledge_base_ingested(
            domain_id="x",
            config={"bot": {}},
        )
        assert result.skipped is True
        assert result.reason == "knowledge_base_disabled"
        assert result.completed_at is not None


class TestAutoIngestionMixinErrorPath:
    async def test_rag_construction_failure_returns_error_with_completed_at(
        self,
    ) -> None:
        """Exception in RAG construction returns error result with completed_at."""
        mixin = _MinimalMixinUser()
        # An unknown embedding_provider drives RAGKnowledgeBase.from_config to raise
        config: dict[str, Any] = {
            "bot": {
                "knowledge_base": {
                    "enabled": True,
                    "documents_path": "/tmp/whatever",
                    "vector_store": {"backend": "memory", "dimensions": 4},
                    "embedding_provider": "__nonexistent__",
                }
            }
        }
        result = await mixin._ensure_knowledge_base_ingested(domain_id="x", config=config)
        assert result.error is not None
        assert result.completed_at is not None


class TestBuildRagConfigForwarding:
    """``_build_rag_config`` forwards the knowledge-base config it is given.

    It used to hand-copy a six-key whitelist, so the ingest knowledge
    base and the bot's own knowledge base — built from the *whole*
    section — disagreed about everything the whitelist did not name.
    ``tenant_id`` was one of them, which meant the ingest wrote untagged
    chunks that the bot's tenant-scoped reads could never match.

    The projection is now a pass-through with a named exclusion set, and
    these tests are the recurrence guard: the first compares the two
    knowledge bases field-for-field, so a field added to
    ``RAGKnowledgeBaseConfig`` later is covered without editing this
    file.
    """

    @staticmethod
    def _kb_config() -> dict[str, Any]:
        return {
            "enabled": True,
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding": {"provider": "echo", "model": "test"},
            "tenant_id": "acme",
            "domain_id": "bot-a",
            "documents_path": "/tmp/docs",
            "document_pattern": "**/*.md",
            "chunking": {"max_chunk_size": 250},
        }

    def test_forwards_every_field_the_read_side_knowledge_base_gets(self) -> None:
        """The ingest KB's config matches the bot's, bar the excluded keys.

        Compared through the typed config rather than key-by-key on the
        dict, so this asserts about what ``RAGKnowledgeBase`` actually
        receives — and covers a newly-added field for free.
        """
        kb_config = self._kb_config()
        ingest = RAGKnowledgeBaseConfig.from_dict(_MinimalMixinUser()._build_rag_config(kb_config))
        read_side = RAGKnowledgeBaseConfig.from_dict(kb_config)

        excluded = {"documents_path", "document_pattern"}
        for f in fields(RAGKnowledgeBaseConfig):
            if f.name in excluded:
                continue
            assert getattr(ingest, f.name) == getattr(read_side, f.name), (
                f"{f.name} differs between the ingest and read-side knowledge bases"
            )

        assert ingest.tenant_id == "acme"
        assert ingest.domain_id == "bot-a"
        assert ingest.embedding == {"provider": "echo", "model": "test"}

    def test_excludes_documents_path(self) -> None:
        """``documents_path`` must not reach ``RAGKnowledgeBase.from_config``.

        ``_ainit`` ingests it during construction — before
        ``ensure_ingested`` runs its skip-if-populated check and without
        consulting ``force`` — so forwarding it would ingest the corpus
        twice. Excluded deliberately, not by omission.
        """
        rag_config = _MinimalMixinUser()._build_rag_config(self._kb_config())

        assert "documents_path" not in rag_config
        assert "document_pattern" not in rag_config
        assert RAGKnowledgeBaseConfig.from_dict(rag_config).documents_path is None

    def test_embedder_defaults_apply_only_when_nothing_is_configured(self) -> None:
        """The historical Ollama defaults fill a gap; they do not override."""
        bare = _MinimalMixinUser()._build_rag_config(
            {"vector_store": {"backend": "memory", "dimensions": 384}}
        )
        assert bare["embedding_provider"] == "ollama"
        assert bare["embedding_model"] == "nomic-embed-text"

        nested = _MinimalMixinUser()._build_rag_config(self._kb_config())
        assert "embedding_provider" not in nested
        assert "embedding_model" not in nested

        flat = _MinimalMixinUser()._build_rag_config(
            {
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_provider": "echo",
            }
        )
        assert flat["embedding_provider"] == "echo"
        assert "embedding_model" not in flat

    def test_a_configured_embedding_model_survives_the_defaults(self) -> None:
        """A flat ``embedding_model`` alone is a configured model, not a gap.

        The gap test asks about ``embedding`` and ``embedding_provider``
        while the fill wrote ``embedding_model`` too, so configuring
        only the model — legal, and the common shape, since the default
        provider is Ollama anyway — had it replaced by the default. The
        ingest then embedded with one model while the bot's own
        knowledge base, built from the whole section and applying no
        defaults of its own, queried with the other: two vector spaces,
        a dimension error or silent garbage retrieval.
        """
        model_only = _MinimalMixinUser()._build_rag_config(
            {
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_model": "mxbai-embed-large",
            }
        )
        assert model_only["embedding_model"] == "mxbai-embed-large"
        assert "embedding_provider" not in model_only, (
            "the defaults are a pair: half of one and half of the other is the "
            "divergence, not the fix"
        )

    def test_embedding_base_url_arrives_as_api_base(self) -> None:
        """A legacy ``embedding_base_url`` reaches the config it was meant for.

        The key was read by the mixin and forwarded under a name no
        config field carries, so ``from_dict`` discarded it — it had
        never worked. It is now an alias for ``api_base``.
        """
        config = RAGKnowledgeBaseConfig.from_dict(
            _MinimalMixinUser()._build_rag_config(
                {
                    "vector_store": {"backend": "memory", "dimensions": 384},
                    "embedding_base_url": "http://embedder.internal:11434",
                }
            )
        )

        assert config.api_base == "http://embedder.internal:11434"

    def test_explicit_api_base_beats_the_legacy_alias(self) -> None:
        """The canonical key wins when both are present."""
        config = RAGKnowledgeBaseConfig.from_dict(
            {
                "api_base": "http://canonical:11434",
                "embedding_base_url": "http://legacy:11434",
            }
        )

        assert config.api_base == "http://canonical:11434"


class TestEnsureIngestedDomainScoping:
    async def test_two_domains_over_one_shared_store_both_survive(self, tmp_path: Path) -> None:
        """The consumer-visible statement of the whole fix.

        Two domains registered through the mixin, each with its own
        corpus, sharing one physical vector store. Both ingests report
        the file they were given and both domains' chunks are readable
        afterwards.
        """
        mixin = _MinimalMixinUser()
        store_path = tmp_path / "shared.pkl"

        for domain, body in (
            ("bot-a", "Alpha content about photosynthesis."),
            ("bot-b", "Beta content about the French revolution."),
        ):
            docs = tmp_path / domain
            docs.mkdir()
            (docs / "overview.md").write_text(f"# Overview\n\n{body}\n")
            result = await mixin._ensure_knowledge_base_ingested(
                domain_id=domain,
                config={
                    "bot": {
                        "knowledge_base": {
                            "enabled": True,
                            "documents_path": str(docs),
                            "vector_store": {
                                "backend": "memory",
                                "dimensions": 384,
                                "persist_path": str(store_path),
                                "domain_id": domain,
                            },
                            "embedding_provider": "echo",
                            "embedding_model": "test",
                        }
                    }
                },
            )
            assert result.error is None, result.error
            assert result.errors == []
            assert result.total_files == 1, f"{domain} ingested no files"

        store = VectorStoreFactory().create(
            backend="memory", dimensions=384, persist_path=str(store_path)
        )
        await store.initialize()
        assert await store.count({"domain_id": "bot-a"}) == 1
        assert await store.count({"domain_id": "bot-b"}) == 1

"""Mixin for adding auto-ingestion to registry managers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from dataknobs_bots.knowledge.service import (
        EnsureIngestionResult,
        KnowledgeIngestionService,
    )

logger = logging.getLogger(__name__)


class AutoIngestionMixin:
    """Mixin that adds auto-ingestion capability to registry managers.

    This mixin provides the `_ensure_knowledge_base_ingested()` method that
    can be called during bot registration to automatically populate knowledge
    bases from configured document paths.

    Requires the class to have:
    - _auto_ingest: bool attribute
    - _ingestion_service: KnowledgeIngestionService attribute

    Usage:
        ```python
        from dataknobs_bots.registry import CachingRegistryManager, InMemoryBackend
        from dataknobs_bots.knowledge import (
            AutoIngestionMixin,
            get_ingestion_service,
        )

        class MyBotManager(CachingRegistryManager[MyBot], AutoIngestionMixin):
            def __init__(self, auto_ingest: bool = False, **kwargs):
                super().__init__(**kwargs)
                self._auto_ingest = auto_ingest
                self._ingestion_service = get_ingestion_service()

            async def register(self, domain_id, config, ingest=None):
                await super().register(domain_id, config)
                should_ingest = ingest if ingest is not None else self._auto_ingest
                if should_ingest:
                    await self._ensure_knowledge_base_ingested(domain_id, config)
        ```
    """

    _auto_ingest: bool
    _ingestion_service: KnowledgeIngestionService

    # Keys the ingestion layer consumes itself, which must not reach
    # ``RAGKnowledgeBase.from_config``. Everything else in the
    # knowledge-base section is forwarded verbatim — see
    # :meth:`_build_rag_config` for why that direction is the safe one.
    _MIXIN_ONLY_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            # Gate read by _ensure_knowledge_base_ingested.
            "enabled",
            # MUST stay excluded. RAGKnowledgeBase._ainit ingests
            # documents_path during construction, which would run the
            # ingest before ensure_ingested's skip-if-populated check
            # and without consulting ``force`` — every registration
            # would re-ingest the whole corpus. The ingestion service
            # reads it from kb_config directly, which is the path that
            # honours both.
            "documents_path",
            # Travels with documents_path; meaningless without it.
            "document_pattern",
        }
    )

    async def _ensure_knowledge_base_ingested(
        self,
        domain_id: str,
        config: dict[str, Any],
        force: bool = False,
    ) -> EnsureIngestionResult:
        """Ensure knowledge base is ingested for a domain.

        Creates a temporary RAGKnowledgeBase from config, runs ingestion,
        and closes it. The bot's own knowledge base will be created
        separately when the bot is instantiated.

        Args:
            domain_id: Domain identifier
            config: Domain configuration (full config with bot.knowledge_base)
            force: Force re-ingestion even if already populated

        Returns:
            EnsureIngestionResult with operation details
        """
        from dataknobs_bots.knowledge.rag import RAGKnowledgeBase
        from dataknobs_bots.knowledge.service import EnsureIngestionResult

        # Extract knowledge_base config
        bot_config = config.get("bot", {})
        kb_config = bot_config.get("knowledge_base", {})

        if not kb_config.get("enabled", False):
            logger.debug("Knowledge base not enabled for %s", domain_id)
            return EnsureIngestionResult(skipped=True, reason="knowledge_base_disabled")

        logger.info("Ensuring knowledge base ingested for %s", domain_id)

        try:
            # Create RAGKnowledgeBase from config. The registration's
            # own ``domain_id`` becomes the knowledge base's binding
            # unless the config names one, so an adopter gets
            # namespaced chunk ids over a shared store with no config
            # change at all — this mixin is a manager and already holds
            # the value.
            rag_config = self._build_rag_config(kb_config, domain_id=domain_id)
            knowledge_base = await RAGKnowledgeBase.from_config(rag_config)

            try:
                result = await self._ingestion_service.ensure_ingested(
                    knowledge_base, kb_config, force=force
                )

                if result.skipped:
                    logger.debug(
                        "Knowledge base ingestion skipped for %s: %s",
                        domain_id,
                        result.reason,
                    )
                else:
                    logger.info(
                        "Knowledge base ingested for %s: %d files, %d chunks",
                        domain_id,
                        result.total_files,
                        result.total_chunks,
                    )

                return result

            finally:
                # Always close the temporary knowledge base
                if hasattr(knowledge_base, "close"):
                    await knowledge_base.close()

        except Exception as e:
            logger.error("Failed to ingest knowledge base for %s: %s", domain_id, e)
            return EnsureIngestionResult(error=str(e))

    def _build_rag_config(
        self,
        kb_config: dict[str, Any],
        *,
        domain_id: str | None = None,
    ) -> dict[str, Any]:
        """Project the knowledge-base config onto the ingest knowledge base.

        Forwards everything except :attr:`_MIXIN_ONLY_KEYS`. The
        direction matters: a bot's own knowledge base is built from the
        *whole* section (``create_knowledge_base_from_config``), so a
        projection that enumerated what to keep guaranteed the two
        would disagree about anything it had not thought of — and it
        did, silently. ``tenant_id`` was one, which meant the ingest
        wrote untagged chunks that the bot's tenant-scoped reads could
        never match: a total retrieval blackout reported as a
        successful ingest. Excluding a named few, and saying why each
        is excluded, fails in the harmless direction instead — a field
        added to ``RAGKnowledgeBaseConfig`` later arrives here on its
        own.

        Unknown keys are safe to forward: ``StructuredConfig.from_dict``
        ignores what matches no field.

        Args:
            kb_config: The knowledge_base section of bot config
            domain_id: The registration's domain, used as the knowledge
                base's binding when the config does not set one. The
                resulting precedence is ``kb_config["domain_id"]`` →
                this argument → the vector store's own ``domain_id``.

        Returns:
            Configuration dict for RAGKnowledgeBase.from_config()
        """
        rag_config: dict[str, Any] = {
            key: value for key, value in kb_config.items() if key not in self._MIXIN_ONLY_KEYS
        }

        if domain_id is not None and rag_config.get("domain_id") is None:
            rag_config["domain_id"] = domain_id

        # The historical embedder defaults, applied only when nothing is
        # configured either way. Applying them unconditionally is what
        # made a configured nested ``embedding`` section fall back to
        # Ollama — landing ingest and query in different vector spaces.
        if "embedding" not in kb_config and "embedding_provider" not in kb_config:
            rag_config["embedding_provider"] = "ollama"
            rag_config["embedding_model"] = "nomic-embed-text"

        return rag_config

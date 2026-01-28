"""Mixin for adding auto-ingestion to registry managers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
            # Create RAGKnowledgeBase from config
            rag_config = self._build_rag_config(kb_config)
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

    def _build_rag_config(self, kb_config: dict[str, Any]) -> dict[str, Any]:
        """Build RAGKnowledgeBase config from knowledge_base config.

        Extracts the relevant fields from knowledge_base config and
        formats them for RAGKnowledgeBase.from_config().

        Args:
            kb_config: The knowledge_base section of bot config

        Returns:
            Configuration dict for RAGKnowledgeBase.from_config()
        """
        rag_config: dict[str, Any] = {
            "vector_store": kb_config.get("vector_store", {}),
            "embedding_provider": kb_config.get("embedding_provider", "ollama"),
            "embedding_model": kb_config.get("embedding_model", "nomic-embed-text"),
            "chunking": kb_config.get("chunking", {}),
            "merger": kb_config.get("merger", {}),
            "formatter": kb_config.get("formatter", {}),
        }

        # Add optional embedding base URL (for Ollama, vLLM, etc.)
        if "embedding_base_url" in kb_config:
            rag_config["embedding_base_url"] = kb_config["embedding_base_url"]

        return rag_config

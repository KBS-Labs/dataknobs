"""Behavioral tests for AutoIngestionMixin auto-ingestion entry points.

Pins the ``completed_at`` invariant on the mixin's two terminal-state
construction sites — KB-disabled skip and exception handler — and
backfills missing behavioral coverage for ``_ensure_knowledge_base_ingested``.
"""

from __future__ import annotations

from typing import Any

from dataknobs_bots.knowledge.registry_mixin import AutoIngestionMixin
from dataknobs_bots.knowledge.service import KnowledgeIngestionService


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
        result = await mixin._ensure_knowledge_base_ingested(
            domain_id="x", config=config
        )
        assert result.error is not None
        assert result.completed_at is not None

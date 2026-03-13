"""Tests for pluggable conversation storage in DynaBot.from_config().

Verifies that DynaBot supports custom ConversationStorage implementations
via the ``storage_class`` config key, while maintaining backward compatibility
with the default DataknobsConversationStorage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.config.builder import DynaBotConfigBuilder
from dataknobs_llm.conversations import ConversationStorage, DataknobsConversationStorage
from dataknobs_llm.conversations.storage import ConversationState


class InMemoryTestStorage(ConversationStorage):
    """Minimal real ConversationStorage for testing pluggable storage."""

    _created_with_config: dict[str, Any] | None = None

    def __init__(self) -> None:
        self._conversations: dict[str, ConversationState] = {}

    @classmethod
    async def create(cls, config: dict[str, Any]) -> InMemoryTestStorage:
        instance = cls()
        instance._created_with_config = config
        return instance

    async def save_conversation(self, state: ConversationState) -> None:
        self._conversations[state.conversation_id] = state

    async def load_conversation(
        self, conversation_id: str
    ) -> ConversationState | None:
        return self._conversations.get(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> bool:
        return self._conversations.pop(conversation_id, None) is not None

    async def list_conversations(
        self,
        filter_metadata: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str | None = None,
        sort_order: str = "desc",
    ) -> list[ConversationState]:
        return list(self._conversations.values())[offset : offset + limit]

    async def search_conversations(
        self,
        content_contains: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        filter_metadata: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> list[ConversationState]:
        return list(self._conversations.values())[offset : offset + limit]

    async def delete_conversations(
        self,
        content_contains: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        ids = list(self._conversations.keys())
        self._conversations.clear()
        return ids


@pytest.mark.asyncio
async def test_default_storage_uses_dataknobs() -> None:
    """from_config() without storage_class uses DataknobsConversationStorage."""
    config = (
        DynaBotConfigBuilder()
        .set_llm(provider="echo", model="echo-test")
        .set_conversation_storage("memory")
        .build()
    )
    bot = await DynaBot.from_config(config)
    assert isinstance(bot.conversation_storage, DataknobsConversationStorage)
    await bot.close()


@pytest.mark.asyncio
async def test_custom_storage_class_via_config() -> None:
    """from_config() with storage_class instantiates the custom class."""
    config = (
        DynaBotConfigBuilder()
        .set_llm(provider="echo", model="echo-test")
        .set_conversation_storage_class(
            "tests.unit.test_pluggable_conversation_storage:InMemoryTestStorage",
            custom_param="value",
        )
        .build()
    )
    bot = await DynaBot.from_config(config)
    storage = bot.conversation_storage
    assert isinstance(storage, InMemoryTestStorage)
    # Verify config was passed through (storage_class key is popped)
    assert storage._created_with_config is not None
    assert storage._created_with_config.get("custom_param") == "value"
    assert "storage_class" not in storage._created_with_config
    await bot.close()


def test_builder_storage_class_config() -> None:
    """DynaBotConfigBuilder.set_conversation_storage_class() emits correct config."""
    config = (
        DynaBotConfigBuilder()
        .set_llm(provider="echo", model="echo-test")
        .set_conversation_storage_class(
            "myapp.storage:MyStorage",
            db_url="postgres://...",
        )
        .build()
    )
    sc = config["conversation_storage"]
    assert sc["storage_class"] == "myapp.storage:MyStorage"
    assert sc["db_url"] == "postgres://..."
    assert "backend" not in sc


@pytest.mark.asyncio
async def test_invalid_storage_class_raises() -> None:
    """from_config() with bad storage_class raises ImportError."""
    config = (
        DynaBotConfigBuilder()
        .set_llm(provider="echo", model="echo-test")
        .set_conversation_storage_class("nonexistent.module:FakeClass")
        .build()
    )
    with pytest.raises((ImportError, ModuleNotFoundError)):
        await DynaBot.from_config(config)


def test_existing_builder_method_unchanged() -> None:
    """set_conversation_storage() still produces backend-only config."""
    config = (
        DynaBotConfigBuilder()
        .set_llm(provider="echo", model="echo-test")
        .set_conversation_storage("sqlite", path="/tmp/test.db")
        .build()
    )
    sc = config["conversation_storage"]
    assert sc["backend"] == "sqlite"
    assert sc["path"] == "/tmp/test.db"
    assert "storage_class" not in sc

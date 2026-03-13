"""Tests for ConversationStorage.count_conversations().

Verifies that:
- count_conversations() returns correct totals
- Metadata filtering works with count
- Empty storage returns 0
- Default implementation on the ABC works via list_conversations fallback
"""

from __future__ import annotations

import pytest
from dataknobs_data.backends import AsyncMemoryDatabase
from dataknobs_structures.tree import Tree

from dataknobs_llm.conversations import (
    ConversationNode,
    ConversationState,
    ConversationStorage,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm.base import LLMMessage


def _make_state(
    conversation_id: str,
    metadata: dict | None = None,
) -> ConversationState:
    """Build a minimal ConversationState for testing."""
    root = ConversationNode(
        message=LLMMessage(role="system", content="System"),
        node_id="",
    )
    return ConversationState(
        conversation_id=conversation_id,
        message_tree=Tree(root),
        metadata=metadata or {},
    )


class TestDataknobsConversationStorageCount:
    """Tests for DataknobsConversationStorage.count_conversations()."""

    @pytest.fixture()
    def storage(self) -> DataknobsConversationStorage:
        return DataknobsConversationStorage(AsyncMemoryDatabase())

    async def test_empty_storage_returns_zero(
        self, storage: DataknobsConversationStorage
    ) -> None:
        assert await storage.count_conversations() == 0

    async def test_count_all(
        self, storage: DataknobsConversationStorage
    ) -> None:
        for i in range(4):
            await storage.save_conversation(_make_state(f"conv-{i}"))

        assert await storage.count_conversations() == 4

    async def test_count_with_metadata_filter(
        self, storage: DataknobsConversationStorage
    ) -> None:
        await storage.save_conversation(
            _make_state("c1", metadata={"user_id": "alice"})
        )
        await storage.save_conversation(
            _make_state("c2", metadata={"user_id": "bob"})
        )
        await storage.save_conversation(
            _make_state("c3", metadata={"user_id": "alice"})
        )

        assert await storage.count_conversations(
            filter_metadata={"user_id": "alice"}
        ) == 2
        assert await storage.count_conversations(
            filter_metadata={"user_id": "bob"}
        ) == 1
        assert await storage.count_conversations(
            filter_metadata={"user_id": "nobody"}
        ) == 0

    async def test_count_unaffected_by_delete(
        self, storage: DataknobsConversationStorage
    ) -> None:
        for i in range(3):
            await storage.save_conversation(_make_state(f"conv-{i}"))

        assert await storage.count_conversations() == 3

        await storage.delete_conversation("conv-1")

        assert await storage.count_conversations() == 2


class TestConversationStorageDefaultCount:
    """Verify that the ABC default count_conversations() works.

    The default implementation delegates to list_conversations(limit=100_000).
    We test it via a minimal concrete subclass that only implements the
    abstract methods.
    """

    async def test_default_count_delegates_to_list(self) -> None:
        """Default count uses list_conversations under the hood."""

        class _MinimalStorage(ConversationStorage):
            """Concrete storage backed by an in-memory list."""

            def __init__(self) -> None:
                self._states: list[ConversationState] = []

            @classmethod
            async def create(cls, config: dict) -> "_MinimalStorage":
                return cls()

            async def save_conversation(self, state: ConversationState) -> None:
                self._states = [
                    s for s in self._states
                    if s.conversation_id != state.conversation_id
                ]
                self._states.append(state)

            async def load_conversation(self, conversation_id: str) -> ConversationState | None:
                for s in self._states:
                    if s.conversation_id == conversation_id:
                        return s
                return None

            async def delete_conversation(self, conversation_id: str) -> bool:
                before = len(self._states)
                self._states = [
                    s for s in self._states
                    if s.conversation_id != conversation_id
                ]
                return len(self._states) < before

            async def list_conversations(
                self,
                filter_metadata: dict | None = None,
                limit: int = 100,
                offset: int = 0,
                sort_by: str | None = None,
                sort_order: str = "desc",
            ) -> list[ConversationState]:
                results = self._states
                if filter_metadata:
                    results = [
                        s for s in results
                        if all(
                            s.metadata.get(k) == v
                            for k, v in filter_metadata.items()
                        )
                    ]
                return results[offset:offset + limit]

            async def search_conversations(self, **kwargs) -> list[ConversationState]:  # type: ignore[override]
                return []

            async def delete_conversations(self, **kwargs) -> list[str]:  # type: ignore[override]
                return []

        storage = _MinimalStorage()
        assert await storage.count_conversations() == 0

        await storage.save_conversation(_make_state("a"))
        await storage.save_conversation(_make_state("b"))
        assert await storage.count_conversations() == 2

        await storage.save_conversation(
            _make_state("c", metadata={"user_id": "alice"})
        )
        assert await storage.count_conversations(
            filter_metadata={"user_id": "alice"}
        ) == 1

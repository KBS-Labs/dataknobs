"""What a conversation store's error is allowed to say.

The backend is a database, so what it raises is a driver's text: a connection
URL, a host, a column. The store's error names the exception's type and keeps
the original as its cause. What the store words itself keeps its words.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations.storage import (
    ConversationNode,
    ConversationState,
    DataknobsConversationStorage,
)
from dataknobs_llm.exceptions import SchemaVersionError, StorageError
from dataknobs_llm.llm.base import LLMMessage
from dataknobs_structures.tree import Tree

_DSN = "postgresql://svc:hunter2@db.internal/chat"


class UnreachableDatabase(AsyncMemoryDatabase):
    """A memory database whose operations fail as a driver's do."""

    error_type: type[Exception] = ConnectionError

    async def _unreachable(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error_type(f"could not connect to {_DSN}")

    upsert = read = delete = search = count = _unreachable


class _UnwritableStorage(DataknobsConversationStorage):
    """Loads a conversation, then fails outside the store's own methods."""

    async def load_conversation(self, conversation_id: str) -> ConversationState | None:
        raise ConnectionError(f"could not connect to {_DSN}")


def _state() -> ConversationState:
    root = ConversationNode(message=LLMMessage(role="user", content="hi"), node_id="")
    return ConversationState(conversation_id="c1", message_tree=Tree(root))


_OPERATIONS = {
    "save conversation": lambda s: s.save_conversation(_state()),
    "load conversation": lambda s: s.load_conversation("c1"),
    "delete conversation": lambda s: s.delete_conversation("c1"),
    "delete conversations": lambda s: s.delete_conversations(filter_metadata={"a": 1}),
    "list conversations": lambda s: s.list_conversations(),
    "count conversations": lambda s: s.count_conversations(),
    "search conversations": lambda s: s.search_conversations(),
}


@pytest.mark.parametrize("error_type", [ConnectionError, ValueError])
@pytest.mark.parametrize("operation", _OPERATIONS)
async def test_what_the_driver_said_is_the_cause_and_not_the_message(operation, error_type):
    database = UnreachableDatabase()
    database.error_type = error_type
    storage = DataknobsConversationStorage(database)

    with pytest.raises(StorageError) as excinfo:
        await _OPERATIONS[operation](storage)

    assert str(excinfo.value) == f"Failed to {operation} ({error_type.__name__})"
    assert _DSN in str(excinfo.value.__cause__)


class RefusingDatabase(AsyncMemoryDatabase):
    """Refuses each operation in the store's own error type."""

    async def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise StorageError("the store is read-only today")

    upsert = read = delete = search = count = _refuse


@pytest.mark.parametrize("operation", _OPERATIONS)
async def test_a_storage_error_passes_through_unwrapped(operation):
    storage = DataknobsConversationStorage(RefusingDatabase())
    with pytest.raises(StorageError, match=r"^the store is read-only today$"):
        await _OPERATIONS[operation](storage)


async def test_deleting_with_no_filter_is_the_callers_error():
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    with pytest.raises(ValueError, match="At least one filter parameter"):
        await storage.delete_conversations()


async def test_updating_metadata_is_bounded_too():
    storage = _UnwritableStorage(AsyncMemoryDatabase())

    with pytest.raises(StorageError) as excinfo:
        await storage.update_metadata("c1", {})

    assert str(excinfo.value) == "Failed to update metadata (ConnectionError)"
    assert _DSN in str(excinfo.value.__cause__)


async def test_a_conversation_that_is_not_there_is_named():
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    with pytest.raises(StorageError, match=r"^Conversation not found: c1$"):
        await storage.update_metadata("c1", {})


async def _holding(payload: dict[str, Any]) -> DataknobsConversationStorage:
    database = AsyncMemoryDatabase()
    await database.upsert("c1", Record(data=payload, storage_id="c1", metadata={"a": 1}))
    return DataknobsConversationStorage(database)


_READS = ["load conversation", "delete conversations", "list conversations", "search conversations"]


@pytest.mark.parametrize("operation", _READS)
async def test_a_record_written_by_a_newer_schema_says_so(operation):
    storage = await _holding(
        {**_state().to_dict(), "schema_version": "2.0.0", "metadata": {"a": 1}}
    )

    with pytest.raises(StorageError) as excinfo:
        await _OPERATIONS[operation](storage)

    assert str(excinfo.value).startswith(
        f"Failed to {operation}: Cannot downgrade from schema 2.0.0"
    )
    assert isinstance(excinfo.value.__cause__, SchemaVersionError)


@pytest.mark.parametrize("operation", _READS)
async def test_a_record_whose_message_cannot_be_used_names_the_node(operation):
    payload = {**_state().to_dict(), "metadata": {"a": 1}}
    payload["nodes"][0]["message"]["tool_calls"] = [{"name": "t", "parameters": "{not json"}]
    storage = await _holding(payload)

    with pytest.raises(StorageError) as excinfo:
        await _OPERATIONS[operation](storage)

    assert str(excinfo.value).startswith(
        f"Failed to {operation}: Stored conversation node '' cannot be read"
    )


async def test_a_record_missing_a_field_is_bounded():
    payload = _state().to_dict()
    del payload["current_node_id"]
    storage = await _holding(payload)
    with pytest.raises(StorageError) as excinfo:
        await storage.load_conversation("c1")
    assert str(excinfo.value) == "Failed to load conversation (KeyError)"

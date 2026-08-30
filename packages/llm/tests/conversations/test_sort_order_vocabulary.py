"""A sort order this package accepts is one the query layer accepts.

``list_conversations`` and ``search_conversations`` take ``sort_order`` as a
bare ``str``, with no ``Literal``, no validation and no constraint stated in
the docstring, and forward it straight to ``Query.sort_by``. That layer used
to map every string but ``"asc"`` to ``DESC`` --- so ``"descending"``,
``"DESCENDING"`` and ``"newest"`` all produced the intended order by
accident, and matched this method's own ``"desc"`` default.

It now refuses a spelling it does not recognise, which is the right call
there and leaves this package with a parameter whose accepted values are
narrower than its type. Two consequences these cells pin:

* the refusal must reach the caller as the ``ValueError`` it is, rather than
  being laundered into a ``StorageError`` by the broad ``except Exception``
  wrapped around the whole body --- a caller cannot act on "storage failed"
  when the fault is in its own argument;
* the spellings the query layer *does* accept must keep working here, in
  either case, on both methods.
"""

from __future__ import annotations

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationNode,
    ConversationState,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm.base import LLMMessage
from dataknobs_structures.tree import Tree


async def _storage_with(count: int = 3) -> DataknobsConversationStorage:
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    for i in range(count):
        root = ConversationNode(
            message=LLMMessage(role="system", content=f"System {i}"), node_id=""
        )
        await storage.save_conversation(
            ConversationState(
                conversation_id=f"conv-{i}",
                message_tree=Tree(root),
                metadata={"updated_at": f"2026-01-0{i + 1}T00:00:00"},
            )
        )
    return storage


class TestAnUnknownSortOrderIsRefusedAsAValueError:
    """Not as a `StorageError`: the storage is fine, the argument is not."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spelling", ["descending", "DESCENDING", "newest", "", "1"])
    async def test_list_conversations(self, spelling: str) -> None:
        storage = await _storage_with()

        with pytest.raises(ValueError, match="sort order"):
            await storage.list_conversations(sort_by="metadata.updated_at", sort_order=spelling)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spelling", ["descending", "ascending", "up"])
    async def test_search_conversations(self, spelling: str) -> None:
        """This one forwards unconditionally --- it has no ``if sort_by``."""
        storage = await _storage_with()

        with pytest.raises(ValueError, match="sort order"):
            await storage.search_conversations(sort_order=spelling)

    @pytest.mark.asyncio
    async def test_the_refusal_is_not_wrapped(self) -> None:
        from dataknobs_llm.exceptions import StorageError

        storage = await _storage_with()

        with pytest.raises(ValueError) as excinfo:
            await storage.list_conversations(sort_by="metadata.updated_at", sort_order="newest")

        assert not isinstance(excinfo.value, StorageError)

    @pytest.mark.asyncio
    async def test_an_unknown_order_is_refused_even_without_a_sort_field(self) -> None:
        """``list_conversations`` only sorts ``if sort_by``, so an invalid
        order used to pass silently when no field was named. A caller that
        gets no complaint reasonably concludes the spelling is fine.
        """
        storage = await _storage_with()

        with pytest.raises(ValueError, match="sort order"):
            await storage.list_conversations(sort_order="descending")


class TestTheSpellingsTheQueryLayerAcceptsStillWork:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("spelling", ["asc", "ASC", "desc", "DESC"])
    async def test_list_conversations(self, spelling: str) -> None:
        storage = await _storage_with()

        conversations = await storage.list_conversations(
            sort_by="metadata.updated_at", sort_order=spelling
        )

        assert len(conversations) == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spelling", ["asc", "ASC", "desc", "DESC"])
    async def test_search_conversations(self, spelling: str) -> None:
        storage = await _storage_with()

        conversations = await storage.search_conversations(sort_order=spelling)

        assert len(conversations) == 3

    @pytest.mark.asyncio
    async def test_a_sort_order_member_is_accepted_as_itself(self) -> None:
        """A caller holding the enum should not have to stringify it."""
        from dataknobs_data.query import SortOrder

        storage = await _storage_with()

        conversations = await storage.list_conversations(
            sort_by="metadata.updated_at", sort_order=SortOrder.ASC
        )

        assert len(conversations) == 3

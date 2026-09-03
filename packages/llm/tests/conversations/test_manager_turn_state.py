"""The state a turn runs against, named once.

A completion is three steps — prepare, call the provider, finalize — and the
last two read the state the first one validated. Nothing stops a caller from
clearing that state in between: ``stream_complete`` yields to its consumer
between the two, and ``reset()`` drops the tree.

The middleware contract says what should happen. ``process_request`` and
``process_response`` both take a ``ConversationState``, not an optional one,
so a turn with no state is over — it is not a turn to run with ``None``.
"""

import tempfile
from pathlib import Path

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary


@pytest.fixture
async def manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        (prompt_dir / "system").mkdir(parents=True)
        (prompt_dir / "system" / "helpful.yaml").write_text("template: You are helpful\n")

        llm = EchoProvider(LLMConfig(provider="echo", model="echo", options={"echo_prefix": ""}))
        builder = AsyncPromptBuilder(library=FileSystemPromptLibrary(prompt_dir))
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        mgr = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="helpful",
        )
        yield mgr
        await llm.close()


@pytest.mark.asyncio
async def test_a_reset_during_a_stream_ends_the_turn_with_its_reason(manager):
    """Resetting between chunks reports the missing state, not an attribute.

    The consumer holds the generator open across ``reset()``, so finalization
    runs against a manager whose state is gone. It says so, rather than
    reaching through ``None`` for a tree node.
    """
    await manager.add_message(role="user", content="What is Python?")

    stream = manager.stream_complete()
    await anext(stream)
    await manager.reset()

    with pytest.raises(ValueError, match="no messages in conversation"):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_completing_an_empty_conversation_says_so(manager):
    """The same reason, from the step that has always checked."""
    await manager.reset()

    with pytest.raises(ValueError, match="no messages in conversation"):
        await manager.complete()

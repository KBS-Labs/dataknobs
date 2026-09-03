"""Cost accounting over the conversation tree.

``Tree.children`` is ``None`` for a leaf, not an empty list, so every walk
of the tree has to say what it does at the bottom. Two of the manager's
walks did not, which made cost tracking raise ``TypeError`` on the first
leaf it reached -- always, since every branch ends in one.

The failure was invisible because :meth:`ConversationManager.complete` calls
the tracker inside ``except Exception: logger.warning(...)``, so a completion
recorded no cost and reported success.
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
    """A manager over a priced model, so the cost path is exercised end to end.

    ``gpt-4`` is in ``CostCalculator.PRICING``; EchoProvider reports token
    usage, so ``complete()`` resolves a real cost rather than ``None``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        (prompt_dir / "system").mkdir(parents=True)
        (prompt_dir / "system" / "helpful.yaml").write_text("template: You are helpful\n")

        llm = EchoProvider(LLMConfig(provider="echo", model="gpt-4", options={"echo_prefix": ""}))
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
async def test_get_total_cost_walks_to_the_leaves(manager):
    """The tree walk reaches a leaf, whose ``children`` is None, not []."""
    await manager.add_message(role="user", content="What is Python?")

    assert manager.get_total_cost() == 0.0


@pytest.mark.asyncio
async def test_get_total_cost_sums_every_recorded_cost(manager):
    """Costs from all branches are summed, not just the current path."""
    await manager.add_message(role="user", content="first")
    await manager.complete()
    first_branch = manager.current_node_id

    await manager.switch_to_node("0.0")
    await manager.add_message(role="user", content="second")
    await manager.complete()

    assert manager.current_node_id != first_branch
    assert manager.get_total_cost() > 0.0


@pytest.mark.asyncio
async def test_completion_records_the_cost_it_calculated(manager):
    """``complete()`` stores cost on the response and the assistant node.

    The tracker swallows its own exceptions, so a broken tree walk showed up
    only as absent metadata and a log line.
    """
    await manager.add_message(role="user", content="What is Python?")
    response = await manager.complete()

    assert response.cost_usd is not None
    assert response.cumulative_cost_usd is not None

    assistant_node = manager.state.get_current_nodes()[-1]
    assert assistant_node.message.role == "assistant"
    assert "cost_usd" in assistant_node.metadata
    assert "cumulative_cost_usd" in assistant_node.metadata


@pytest.mark.asyncio
async def test_get_cost_by_branch_stops_at_the_bottom_of_the_tree(manager):
    """A node id deeper than the tree runs out of children rather than raising.

    Consistent with the method's existing leniency: an index past the end of
    a level is already skipped rather than reported.
    """
    await manager.add_message(role="user", content="What is Python?")

    assert manager.get_cost_by_branch("0.0.0.0") == 0.0


@pytest.mark.asyncio
async def test_get_cost_by_branch_sums_the_path(manager):
    """The current branch's recorded costs are summed."""
    await manager.add_message(role="user", content="What is Python?")
    await manager.complete()

    assert manager.get_cost_by_branch() > 0.0

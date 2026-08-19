"""``set_provider`` must hand ownership back with the object.

Every subsystem that can hold a config-built collaborator gates its
teardown on an ``_owns_*`` flag, so ``close()`` releases what it built
and leaves an injected one alone. ``set_provider`` — the documented
injection point ``inject_providers`` drives — replaced the collaborator
without touching that flag, which inverts the gate precisely: the
provider handed in by the caller gets closed, and the one this instance
built and replaced is never closed at all.

``QueryTransformer.set_provider`` already clears its flag on injection
and says why in its docstring; these are the siblings that did not.
"""

from __future__ import annotations

from typing import Any

from dataknobs_bots.bot.base import PROVIDER_ROLE_MEMORY_EMBEDDING, PROVIDER_ROLE_SUMMARY_LLM
from dataknobs_bots.memory.summary import SummaryMemory
from dataknobs_bots.memory.vector import VectorMemory
from dataknobs_llm.llm import LLMProviderFactory


async def _provider() -> Any:
    provider = LLMProviderFactory(is_async=True).create({"provider": "echo", "model": "test"})
    await provider.initialize()
    return provider


class TestVectorMemory:
    async def test_injecting_an_embedder_hands_ownership_back(self) -> None:
        memory = await VectorMemory.from_config(
            {"dimension": 384, "embedding": {"provider": "echo", "model": "test"}}
        )
        assert memory._owns_embedding_provider is True

        injected = await _provider()
        assert memory.set_provider(PROVIDER_ROLE_MEMORY_EMBEDDING, injected) is True
        assert memory._owns_embedding_provider is False

        await memory.close()
        assert injected.close_count == 0, "close() closed the injected provider"

    async def test_an_unrecognised_role_leaves_ownership_alone(self) -> None:
        memory = await VectorMemory.from_config(
            {"dimension": 384, "embedding": {"provider": "echo", "model": "test"}}
        )
        assert memory.set_provider("some.other.role", await _provider()) is False
        assert memory._owns_embedding_provider is True
        await memory.close()


class TestSummaryMemory:
    async def test_injecting_a_summary_llm_hands_ownership_back(self) -> None:
        memory = await SummaryMemory.from_config({"llm": {"provider": "echo", "model": "test"}})
        assert memory._owns_llm_provider is True

        injected = await _provider()
        assert memory.set_provider(PROVIDER_ROLE_SUMMARY_LLM, injected) is True
        assert memory._owns_llm_provider is False

        await memory.close()
        assert injected.close_count == 0, "close() closed the injected provider"

    async def test_an_unrecognised_role_leaves_ownership_alone(self) -> None:
        memory = await SummaryMemory.from_config({"llm": {"provider": "echo", "model": "test"}})
        assert memory.set_provider("some.other.role", await _provider()) is False
        assert memory._owns_llm_provider is True
        await memory.close()

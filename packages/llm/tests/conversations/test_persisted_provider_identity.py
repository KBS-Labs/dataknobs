"""Persisted conversation metadata records the canonical provider family.

The provider-identity contract exists so that every consumer of "which
provider served this?" agrees on the answer. Conversation storage is the
consumer where disagreement is most expensive: turn logs and cost buckets are
rewritten on the next deploy, but a node's metadata is **durable** — a
mismatched value there outlives the process that wrote it, and any analytics
joining stored conversations to cost or telemetry silently splits into two
populations.

``LLMConfig.provider`` is stored **verbatim** (the factory canonicalizes only
the registry lookup), so writing it raw persists whatever the config author
typed. A deployment configured ``provider: OpenAI`` recorded ``"openai"`` to
the cost bucket and ``"OpenAI"`` to the conversation node, in the same turn.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary


def _create_prompts(prompt_dir: Path) -> None:
    system_dir = prompt_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "assistant.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
    )


@pytest.fixture
async def manager_for_spelling():
    """Build a manager whose provider is configured with a given spelling."""
    providers: list[EchoProvider] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        _create_prompts(prompt_dir)
        builder = AsyncPromptBuilder(library=FileSystemPromptLibrary(prompt_dir))
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        async def _make(spelling: str) -> ConversationManager:
            llm = EchoProvider(
                LLMConfig(
                    provider=spelling,
                    model="echo-model",
                    options={"echo_prefix": ""},
                )
            )
            providers.append(llm)
            return await ConversationManager.create(
                llm=llm,
                prompt_builder=builder,
                storage=storage,
                system_prompt_name="assistant",
            )

        yield _make

        for llm in providers:
            await llm.close()


def _assistant_metadata(manager: ConversationManager) -> dict[str, Any]:
    node = manager.state.get_current_node()
    assert node is not None
    return node.data.metadata


class TestPersistedProviderIdentity:
    """The node records the family key, not the configured spelling."""

    @pytest.mark.asyncio
    async def test_lowercase_config_persists_the_family(self, manager_for_spelling) -> None:
        manager = await manager_for_spelling("echo")
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        assert _assistant_metadata(manager)["provider"] == "echo"

    @pytest.mark.asyncio
    async def test_capitalized_config_persists_the_same_family(self, manager_for_spelling) -> None:
        """The regression guard.

        Writing ``config.provider`` raw persists ``"Echo"`` here while the
        same turn's cost bucket and turn log say ``"echo"`` — the identity
        disagreement the contract was introduced to end, surviving in the one
        place it cannot be corrected after the fact.
        """
        manager = await manager_for_spelling("Echo")
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        assert _assistant_metadata(manager)["provider"] == "echo"

    @pytest.mark.asyncio
    async def test_two_spellings_persist_one_key(self, manager_for_spelling) -> None:
        """Two deployments of one provider must not split an analytics join."""
        lower = await manager_for_spelling("echo")
        await lower.add_message(role="user", content="Hello")
        await lower.complete()

        upper = await manager_for_spelling("ECHO")
        await upper.add_message(role="user", content="Hello")
        await upper.complete()

        assert _assistant_metadata(lower)["provider"] == _assistant_metadata(upper)["provider"]

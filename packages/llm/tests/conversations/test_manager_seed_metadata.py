"""Tests for ConversationManager seed-metadata API.

Pins the new ``seed_metadata`` / ``update_seed_metadata`` /
``remove_seed_metadata`` / ``get_seed_metadata`` public methods, which
cross the pre-/post-state boundary by writing to both the live state
metadata (when state exists) and the initial-metadata seed bucket
(always). Also pins the pre-state branch of the existing metadata
methods — those branches had no test coverage before this module landed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary


@pytest.fixture
async def test_components():
    """Echo provider, file-system prompt library, in-memory storage.

    Same shape as the fixture in ``test_manager.py``; duplicated here so
    this module is self-contained.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        (prompt_dir / "system").mkdir(parents=True)
        (prompt_dir / "user").mkdir(parents=True)
        (prompt_dir / "system" / "helpful.yaml").write_text(
            yaml.dump({"template": "You are a helpful assistant"})
        )

        llm = EchoProvider(
            LLMConfig(
                provider="echo",
                model="echo-model",
                options={"echo_prefix": ""},
            )
        )
        library = FileSystemPromptLibrary(prompt_dir)
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        yield {"llm": llm, "builder": builder, "storage": storage}

        await llm.close()


async def _new_manager(components, **kwargs):
    return await ConversationManager.create(
        llm=components["llm"],
        prompt_builder=components["builder"],
        storage=components["storage"],
        **kwargs,
    )


class TestSeedMetadataNewAPI:
    """Behavioural pins for the new seed-* methods."""

    @pytest.mark.asyncio
    async def test_seed_metadata_pre_state_survives_materialization(
        self, test_components
    ):
        """Reproducing pin from the brief — fails against HEAD before the
        new method lands.

        Pre-state write must (a) be readable pre-state via
        ``get_seed_metadata`` and (b) land on ``state.metadata`` once the
        first message materializes state.
        """
        manager = await _new_manager(test_components)

        manager.seed_metadata("active_project_id", "proj-42")

        assert manager.state is None
        assert manager.get_seed_metadata("active_project_id") == "proj-42"

        await manager.add_message(role="user", content="hi")

        assert manager.state is not None
        assert manager.state.metadata["active_project_id"] == "proj-42"

    @pytest.mark.asyncio
    async def test_seed_metadata_post_state_writes_both_buckets_under_resume(
        self, test_components
    ):
        """Pin the two-bucket post-state semantics on the path where the
        buckets diverge.

        After ``resume()``, the manager's ``_initial_metadata`` is ``{}``
        (the resume path does not re-seed it) and ``state.metadata`` is
        the loaded dict — distinct dicts. A post-state ``seed_metadata``
        write must reach both so that a subsequent
        rematerialization-from-seed (a future code path; the contract
        the seed bucket exposes) picks up the write cleanly.
        """
        manager = await _new_manager(
            test_components, metadata={"user_id": "alice"}
        )
        await manager.add_message(role="user", content="hi")
        conv_id = manager.conversation_id

        resumed = await ConversationManager.resume(
            conversation_id=conv_id,
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )
        # Sanity: the buckets are distinct dict instances under resume.
        assert resumed.state is not None
        assert resumed.state.metadata is not resumed._initial_metadata

        resumed.seed_metadata("active_project_id", "proj-7")

        assert resumed.state.metadata["active_project_id"] == "proj-7"
        # Seed bucket is also updated even though it started empty.
        assert resumed._initial_metadata["active_project_id"] == "proj-7"

    @pytest.mark.asyncio
    async def test_update_seed_metadata_batch_pre_state(self, test_components):
        """Bulk pre-state write of multiple keys lands on the seed and
        survives materialization."""
        manager = await _new_manager(test_components)

        manager.update_seed_metadata(
            {"active_project_id": "proj-1", "tenant_id": "tenant-a"}
        )

        assert manager.state is None
        assert manager.get_seed_metadata("active_project_id") == "proj-1"
        assert manager.get_seed_metadata("tenant_id") == "tenant-a"

        await manager.add_message(role="user", content="hi")

        assert manager.state.metadata["active_project_id"] == "proj-1"
        assert manager.state.metadata["tenant_id"] == "tenant-a"

    @pytest.mark.asyncio
    async def test_remove_seed_metadata_removes_from_both_buckets(
        self, test_components
    ):
        """Pre-state remove → key absent post-materialization. Post-state
        remove (under resume, where the buckets diverge) → key absent
        from both the live dict and the seed bucket."""
        # Pre-state path.
        manager = await _new_manager(test_components)
        manager.seed_metadata("active_project_id", "proj-1")
        manager.remove_seed_metadata("active_project_id")

        assert manager.get_seed_metadata("active_project_id") is None
        await manager.add_message(role="user", content="hi")
        assert "active_project_id" not in manager.state.metadata

        # Post-state-with-divergent-buckets (resume) path.
        manager2 = await _new_manager(
            test_components, metadata={"active_project_id": "proj-2"}
        )
        await manager2.add_message(role="user", content="hi")
        conv_id = manager2.conversation_id

        resumed = await ConversationManager.resume(
            conversation_id=conv_id,
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )
        # Seed the resume-path seed bucket so we can prove removal touches it.
        resumed._initial_metadata["active_project_id"] = "proj-2"

        resumed.remove_seed_metadata("active_project_id")

        assert "active_project_id" not in resumed.state.metadata
        assert "active_project_id" not in resumed._initial_metadata

    @pytest.mark.asyncio
    async def test_remove_seed_metadata_missing_key_is_silent(
        self, test_components
    ):
        """``remove_seed_metadata`` on a missing key must not raise — same
        idempotent shape as ``set``/``update``."""
        manager = await _new_manager(test_components)

        # Pre-state.
        manager.remove_seed_metadata("never_set")

        # Post-state.
        await manager.add_message(role="user", content="hi")
        manager.remove_seed_metadata("never_set")  # still no raise

    @pytest.mark.asyncio
    async def test_get_seed_metadata_returns_copy_pre_state(
        self, test_components
    ):
        """Pre-state whole-dict return is a copy: mutating it must not
        write back into the seed bucket."""
        manager = await _new_manager(test_components)
        manager.seed_metadata("a", 1)

        result = manager.get_seed_metadata()
        result["b"] = 2

        assert manager.get_seed_metadata("b") is None
        assert "b" not in manager._initial_metadata

    @pytest.mark.asyncio
    async def test_get_seed_metadata_returns_live_dict_post_state(
        self, test_components
    ):
        """Post-state whole-dict return is the live ``state.metadata``
        (consistent with the existing ``metadata`` property)."""
        manager = await _new_manager(test_components)
        await manager.add_message(role="user", content="hi")

        result = manager.get_seed_metadata()
        result["new_key"] = "new_value"

        assert manager.state.metadata["new_key"] == "new_value"
        assert manager.get_seed_metadata("new_key") == "new_value"

    @pytest.mark.asyncio
    async def test_get_seed_metadata_returns_default_for_missing_key(
        self, test_components
    ):
        """Both pre- and post-state, ``get_seed_metadata(key, default)``
        returns ``default`` when the key is absent."""
        manager = await _new_manager(test_components)

        assert manager.get_seed_metadata("missing", default="fallback") == "fallback"
        assert manager.get_seed_metadata("missing") is None

        await manager.add_message(role="user", content="hi")

        assert manager.get_seed_metadata("missing", default="fallback") == "fallback"
        assert manager.get_seed_metadata("missing") is None


class TestExistingMetadataMethodsPreStatePins:
    """Regression pins for the pre-state branch of the existing methods.

    These methods existed without pre-state test coverage. The seed-*
    methods are an additive sibling family, not a replacement, so the
    silent-no-op / raise contracts of the existing methods must be pinned
    against a future PR that "helpfully fixes" them in a way that breaks
    callers depending on the no-op shape.
    """

    @pytest.mark.asyncio
    async def test_set_metadata_silent_no_op_pre_state(self, test_components):
        manager = await _new_manager(test_components)

        manager.set_metadata("k", "v")

        assert manager.state is None
        assert manager.get_metadata("k") is None
        # Write did not leak into the seed either.
        assert manager.get_seed_metadata("k") is None

    @pytest.mark.asyncio
    async def test_update_metadata_silent_no_op_pre_state(self, test_components):
        manager = await _new_manager(test_components)

        manager.update_metadata({"k1": "v1", "k2": "v2"})

        assert manager.state is None
        assert manager.get_metadata("k1") is None
        assert manager.get_seed_metadata("k1") is None

    @pytest.mark.asyncio
    async def test_remove_metadata_silent_no_op_pre_state(self, test_components):
        manager = await _new_manager(
            test_components, metadata={"existing": "value"}
        )

        # ``remove_metadata`` pre-state must not raise and must not touch
        # the seed bucket either — its silent-no-op contract is total.
        manager.remove_metadata("existing")

        assert manager.state is None
        assert manager.get_seed_metadata("existing") == "value"

    @pytest.mark.asyncio
    async def test_add_metadata_raises_pre_state(self, test_components):
        manager = await _new_manager(test_components)

        with pytest.raises(ValueError, match="No conversation state"):
            await manager.add_metadata("k", "v")

    @pytest.mark.asyncio
    async def test_metadata_property_empty_pre_state(self, test_components):
        """The ``metadata`` property deliberately returns ``{}`` pre-state
        regardless of the ``metadata=`` seed. Pins the asymmetric-property
        trap the design rejected — a future PR that "fixes" the property
        to return the seed would silently break this test.
        """
        manager = await _new_manager(
            test_components, metadata={"seeded": "value"}
        )

        assert manager.state is None
        assert manager.metadata == {}
        # The seed is still readable through the explicit accessor.
        assert manager.get_seed_metadata("seeded") == "value"

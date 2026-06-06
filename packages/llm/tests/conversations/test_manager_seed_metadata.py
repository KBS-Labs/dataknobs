"""Tests for ConversationManager seed-metadata API.

Pins the new ``seed_metadata`` / ``update_seed_metadata`` /
``remove_seed_metadata`` / ``get_seed_metadata`` / ``add_seed_metadata``
public methods, which cross the pre-/post-state boundary by writing to
both the live state metadata (when state exists) and the initial-metadata
seed bucket (always). Also pins the pre-state branch of the existing
metadata methods, the public ``save()`` entry point that the seed-*
docstrings reference, the resume-from-seed alias contract, and the
strict-orphan-``default`` contract on ``get_metadata`` /
``get_seed_metadata``.
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
    async def test_resume_aliases_seed_bucket_to_state_metadata(
        self, test_components
    ):
        """Pin the resume-from-seed alias contract.

        Before the alias landed, ``resume()`` left ``_initial_metadata``
        empty (``{}``) and decoupled from the loaded ``state.metadata``
        — so a post-resume ``seed_metadata`` write would touch the seed
        bucket but the seed bucket was never consumed again, making the
        write effectively dead. The alias makes the seed bucket BE the
        live ``state.metadata`` dict on resume, mirroring the
        post-first-materialization shape (where ``add_message`` passes
        ``_initial_metadata`` to ``ConversationState`` by reference).
        Effect: the seed-aware family's contract is uniform across the
        post-state lifecycle.

        Reproducing pin — fails on the pre-fix tree (``is not`` was the
        pinned invariant); passes once ``resume()`` aliases the buckets.
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

        assert resumed.state is not None
        # The buckets are the SAME dict object after resume (alias).
        assert resumed._initial_metadata is resumed.state.metadata
        # Seed bucket already reflects the loaded state's metadata —
        # ``_initial_metadata`` IS the loaded dict, not a fresh ``{}``.
        assert resumed.get_seed_metadata("user_id") == "alice"

    @pytest.mark.asyncio
    async def test_seed_metadata_post_state_writes_propagate_under_resume(
        self, test_components
    ):
        """Post-resume ``seed_metadata`` writes show up in both buckets.

        With the alias in place, the two-bucket abstraction collapses
        post-resume — a single write lands on the shared dict. This
        replaces the original ``test_seed_metadata_post_state_writes_both_buckets_under_resume``
        (which pinned the now-removed divergence). The behaviour the
        seed family promises (writes durable through state, observable
        via both accessors) is what we pin here.
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

        resumed.seed_metadata("active_project_id", "proj-7")

        assert resumed.state.metadata["active_project_id"] == "proj-7"
        assert resumed._initial_metadata["active_project_id"] == "proj-7"
        # Single observation — the buckets are the same dict.
        assert (
            resumed.state.metadata["active_project_id"]
            is resumed._initial_metadata["active_project_id"]
        )

    @pytest.mark.asyncio
    async def test_update_seed_metadata_batch_pre_state(self, test_components):
        """Bulk pre-state write of multiple keys lands on the seed and
        survives materialization.
        """
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
        remove (under resume) → key absent from both accessors.

        With the resume-from-seed alias the two buckets reference the
        same dict post-resume, so removal once strips it from both
        observation paths. We still assert both to pin the contract a
        consumer relies on (``state.metadata`` and ``_initial_metadata``
        agree).
        """
        # Pre-state path.
        manager = await _new_manager(test_components)
        manager.seed_metadata("active_project_id", "proj-1")
        manager.remove_seed_metadata("active_project_id")

        assert manager.get_seed_metadata("active_project_id") is None
        await manager.add_message(role="user", content="hi")
        assert "active_project_id" not in manager.state.metadata

        # Post-resume path — buckets are aliased, removal touches one dict.
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
        # The seed bucket starts out aliased to ``state.metadata`` and
        # already contains the loaded value — no manual reseeding needed.
        assert resumed.get_seed_metadata("active_project_id") == "proj-2"

        resumed.remove_seed_metadata("active_project_id")

        assert "active_project_id" not in resumed.state.metadata
        assert "active_project_id" not in resumed._initial_metadata

    @pytest.mark.asyncio
    async def test_remove_seed_metadata_missing_key_is_silent(
        self, test_components
    ):
        """``remove_seed_metadata`` on a missing key must not raise — same
        idempotent shape as ``set``/``update``.
        """
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
        write back into the seed bucket.
        """
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
        (consistent with the existing ``metadata`` property).
        """
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
        returns ``default`` when the key is absent.
        """
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


class TestPublicSave:
    """Behavioural pins for the public ``save()`` method.

    The metadata-method docstrings (existing AND seed-aware) reference
    ``save()`` as the public escape hatch for durable persistence — but
    no public ``save()`` existed pre-fix; consumers had to either trigger
    another turn or call the private ``_save_state()``. The public
    method makes the documented affordance real.
    """

    @pytest.mark.asyncio
    async def test_save_persists_state_so_resume_sees_writes(
        self, test_components
    ):
        """Reproducing pin — fails on pre-fix HEAD (``AttributeError`` —
        no ``save`` method). Once the public ``save()`` lands, a sync
        ``seed_metadata`` write followed by ``await manager.save()``
        becomes observable on a resumed manager.
        """
        manager = await _new_manager(test_components)
        await manager.add_message(role="user", content="hi")
        manager.seed_metadata("active_project_id", "proj-9")

        # The seed-* writers do not auto-persist — that is documented as
        # a precondition for this call.
        await manager.save()

        conv_id = manager.conversation_id
        resumed = await ConversationManager.resume(
            conversation_id=conv_id,
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        assert resumed.get_seed_metadata("active_project_id") == "proj-9"

    @pytest.mark.asyncio
    async def test_save_pre_state_is_silent_no_op(self, test_components):
        """Calling ``save()`` before the first turn must not raise.

        Pre-state there is no ``ConversationState`` to persist; ``save``
        delegates to ``_save_state`` which short-circuits when ``state``
        is ``None``. The contract matches the existing private method.
        """
        manager = await _new_manager(test_components)

        assert manager.state is None
        await manager.save()  # no raise, no persistence


class TestAddSeedMetadata:
    """Behavioural pins for ``add_seed_metadata`` — the async, persisting
    seed analogue of ``add_metadata``.
    """

    @pytest.mark.asyncio
    async def test_add_seed_metadata_pre_state_writes_seed_no_raise(
        self, test_components
    ):
        """Pre-state, the call writes to the seed bucket and does not
        raise (unlike ``add_metadata`` which raises pre-state).

        There is no ``state`` to persist pre-state, so the persistence
        step is a no-op — but the seed-bucket write survives state
        materialization, satisfying the "durable" half of the contract.
        """
        manager = await _new_manager(test_components)

        await manager.add_seed_metadata("active_project_id", "proj-9")

        assert manager.state is None
        assert manager.get_seed_metadata("active_project_id") == "proj-9"

        await manager.add_message(role="user", content="hi")
        assert manager.state.metadata["active_project_id"] == "proj-9"

    @pytest.mark.asyncio
    async def test_add_seed_metadata_post_state_persists_immediately(
        self, test_components
    ):
        """Post-state, the call writes to both buckets AND persists.

        A subsequent ``resume()`` observes the value without any
        intervening turn — that's the difference between ``seed_metadata``
        (must wait for next turn or explicit ``save``) and the async
        persisting variant.
        """
        manager = await _new_manager(test_components)
        await manager.add_message(role="user", content="hi")

        await manager.add_seed_metadata("active_project_id", "proj-9")

        assert manager.state.metadata["active_project_id"] == "proj-9"

        conv_id = manager.conversation_id
        resumed = await ConversationManager.resume(
            conversation_id=conv_id,
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )
        # Visible without an intervening turn — proof the persist
        # happened inside ``add_seed_metadata``.
        assert resumed.get_seed_metadata("active_project_id") == "proj-9"

    @pytest.mark.asyncio
    async def test_add_seed_metadata_post_resume_persists(
        self, test_components
    ):
        """After ``resume()``, the call still writes both buckets (now
        aliased) and persists. Pins the symmetry across the post-state
        lifecycle.
        """
        manager = await _new_manager(test_components)
        await manager.add_message(role="user", content="hi")
        conv_id = manager.conversation_id

        resumed = await ConversationManager.resume(
            conversation_id=conv_id,
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        await resumed.add_seed_metadata("tenant_id", "tenant-x")

        # Re-resume to see the persisted value.
        re_resumed = await ConversationManager.resume(
            conversation_id=conv_id,
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )
        assert re_resumed.get_seed_metadata("tenant_id") == "tenant-x"


class TestStrictOrphanDefault:
    """Pin the strict-orphan-``default`` contract on both ``get_metadata``
    and ``get_seed_metadata``.

    Pre-fix, passing ``default=`` without ``key=`` silently discarded the
    default — the call returned the whole bucket dict. A consumer writing
    ``manager.get_seed_metadata(default={"fallback": True})`` (thinking
    "give me the bucket, or this fallback if it's empty") got the empty
    dict, not the fallback, with no error. The strict contract rejects
    this orphan-default shape at the call site.
    """

    @pytest.mark.asyncio
    async def test_get_seed_metadata_rejects_orphan_default(
        self, test_components
    ):
        manager = await _new_manager(test_components)

        with pytest.raises(TypeError, match="default"):
            manager.get_seed_metadata(default="fallback")

        await manager.add_message(role="user", content="hi")

        with pytest.raises(TypeError, match="default"):
            manager.get_seed_metadata(default="fallback")

    @pytest.mark.asyncio
    async def test_get_metadata_rejects_orphan_default(
        self, test_components
    ):
        """The existing ``get_metadata`` family shares the orphan-default
        quirk; the strict contract applies to both for symmetry.
        """
        manager = await _new_manager(test_components)

        with pytest.raises(TypeError, match="default"):
            manager.get_metadata(default="fallback")

        await manager.add_message(role="user", content="hi")

        with pytest.raises(TypeError, match="default"):
            manager.get_metadata(default="fallback")

    @pytest.mark.asyncio
    async def test_get_metadata_default_with_key_still_works(
        self, test_components
    ):
        """The strict contract MUST NOT regress the normal
        ``(key, default)`` shape — that is the documented use of the
        ``default`` parameter and is exercised throughout the codebase.
        """
        manager = await _new_manager(test_components)
        await manager.add_message(role="user", content="hi")

        assert manager.get_metadata("missing", default="x") == "x"
        assert manager.get_seed_metadata("missing", default="x") == "x"
        # Positional form too.
        assert manager.get_metadata("missing", "y") == "y"
        assert manager.get_seed_metadata("missing", "y") == "y"

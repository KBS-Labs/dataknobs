"""Tests for registry backends."""

import asyncio

import pytest

from dataknobs_bots.registry import InMemoryBackend
from dataknobs_data import SortOrder, SortSpec


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    @pytest.fixture
    def backend(self):
        """Create a fresh backend for each test."""
        return InMemoryBackend()

    @pytest.mark.asyncio
    async def test_initialize(self, backend):
        """Test backend initialization."""
        await backend.initialize()
        assert backend._initialized is True

    @pytest.mark.asyncio
    async def test_close(self, backend):
        """Test backend close clears data."""
        await backend.initialize()
        await backend.register("test-bot", {"llm": {}})

        await backend.close()

        assert backend._initialized is False
        assert len(backend._registrations) == 0

    @pytest.mark.asyncio
    async def test_register_new(self, backend):
        """Test registering a new bot."""
        await backend.initialize()

        reg = await backend.register("new-bot", {"llm": {"provider": "echo"}})

        assert reg.bot_id == "new-bot"
        assert reg.config == {"llm": {"provider": "echo"}}
        assert reg.status == "active"
        assert reg.created_at is not None

    @pytest.mark.asyncio
    async def test_register_update(self, backend):
        """Test updating an existing registration."""
        await backend.initialize()

        # Initial registration
        reg1 = await backend.register("update-bot", {"version": 1})
        original_created = reg1.created_at

        # Wait a tiny bit to ensure timestamps differ
        await asyncio.sleep(0.001)

        # Update
        reg2 = await backend.register("update-bot", {"version": 2})

        assert reg2.bot_id == "update-bot"
        assert reg2.config == {"version": 2}
        # created_at should be preserved
        assert reg2.created_at == original_created
        # updated_at should change
        assert reg2.updated_at >= reg1.updated_at

    @pytest.mark.asyncio
    async def test_register_with_status(self, backend):
        """Test registering with custom status."""
        await backend.initialize()

        reg = await backend.register("status-bot", {}, status="inactive")

        assert reg.status == "inactive"

    @pytest.mark.asyncio
    async def test_get_existing(self, backend):
        """Test getting an existing registration."""
        await backend.initialize()
        await backend.register("get-bot", {"found": True})

        reg = await backend.get("get-bot")

        assert reg is not None
        assert reg.bot_id == "get-bot"
        assert reg.config == {"found": True}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, backend):
        """Test getting a non-existent registration."""
        await backend.initialize()

        reg = await backend.get("nonexistent")

        assert reg is None

    @pytest.mark.asyncio
    async def test_get_updates_access_time(self, backend):
        """Test that get updates last_accessed_at."""
        await backend.initialize()
        reg1 = await backend.register("access-bot", {})
        original_access = reg1.last_accessed_at

        await asyncio.sleep(0.001)

        reg2 = await backend.get("access-bot")

        assert reg2.last_accessed_at > original_access

    @pytest.mark.asyncio
    async def test_get_config(self, backend):
        """Test getting just the config."""
        await backend.initialize()
        await backend.register("config-bot", {"key": "value"})

        config = await backend.get_config("config-bot")

        assert config == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_config_nonexistent(self, backend):
        """Test get_config for non-existent bot."""
        await backend.initialize()

        config = await backend.get_config("nonexistent")

        assert config is None

    @pytest.mark.asyncio
    async def test_exists_active(self, backend):
        """Test exists for active registration."""
        await backend.initialize()
        await backend.register("exists-bot", {})

        assert await backend.exists("exists-bot") is True

    @pytest.mark.asyncio
    async def test_exists_inactive(self, backend):
        """Test exists returns False for inactive registration."""
        await backend.initialize()
        await backend.register("inactive-bot", {}, status="inactive")

        assert await backend.exists("inactive-bot") is False

    @pytest.mark.asyncio
    async def test_exists_nonexistent(self, backend):
        """Test exists for non-existent registration."""
        await backend.initialize()

        assert await backend.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_unregister(self, backend):
        """Test hard delete."""
        await backend.initialize()
        await backend.register("delete-bot", {})

        result = await backend.unregister("delete-bot")

        assert result is True
        assert await backend.get("delete-bot") is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, backend):
        """Test unregister returns False for non-existent."""
        await backend.initialize()

        result = await backend.unregister("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate(self, backend):
        """Test soft delete."""
        await backend.initialize()
        await backend.register("deactivate-bot", {})

        result = await backend.deactivate("deactivate-bot")

        assert result is True
        # Still retrievable but inactive
        reg = await backend.get("deactivate-bot")
        assert reg is not None
        assert reg.status == "inactive"
        # Not returned by exists
        assert await backend.exists("deactivate-bot") is False

    @pytest.mark.asyncio
    async def test_deactivate_nonexistent(self, backend):
        """Test deactivate returns False for non-existent."""
        await backend.initialize()

        result = await backend.deactivate("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_active(self, backend):
        """Test listing active registrations."""
        await backend.initialize()
        await backend.register("active1", {})
        await backend.register("active2", {})
        await backend.register("inactive1", {}, status="inactive")

        active = await backend.list_active()

        assert len(active) == 2
        ids = [r.bot_id for r in active]
        assert "active1" in ids
        assert "active2" in ids
        assert "inactive1" not in ids

    @pytest.mark.asyncio
    async def test_list_all(self, backend):
        """Test listing all registrations."""
        await backend.initialize()
        await backend.register("bot1", {})
        await backend.register("bot2", {}, status="inactive")

        all_regs = await backend.list_all()

        assert len(all_regs) == 2

    @pytest.mark.asyncio
    async def test_list_ids(self, backend):
        """Test listing active bot IDs."""
        await backend.initialize()
        await backend.register("id1", {})
        await backend.register("id2", {})
        await backend.register("inactive", {}, status="inactive")

        ids = await backend.list_ids()

        assert len(ids) == 2
        assert "id1" in ids
        assert "id2" in ids
        assert "inactive" not in ids

    @pytest.mark.asyncio
    async def test_count(self, backend):
        """Test counting active registrations."""
        await backend.initialize()
        await backend.register("bot1", {})
        await backend.register("bot2", {})
        await backend.register("inactive", {}, status="inactive")

        count = await backend.count()

        assert count == 2

    @pytest.mark.asyncio
    async def test_clear(self, backend):
        """Test clearing all registrations."""
        await backend.initialize()
        await backend.register("bot1", {})
        await backend.register("bot2", {})

        await backend.clear()

        assert await backend.count() == 0
        assert len(await backend.list_all()) == 0

    def test_repr(self, backend):
        """Test string representation."""
        repr_str = repr(backend)
        assert "InMemoryBackend" in repr_str
        assert "count=" in repr_str


class TestInMemoryBackendPeekConfig:
    """Tests for non-mutating peek_config."""

    @pytest.fixture
    def backend(self):
        return InMemoryBackend()

    @pytest.mark.asyncio
    async def test_peek_config_returns_config(self, backend):
        """peek_config returns the same config dict as get_config."""
        await backend.initialize()
        await backend.register("peek-bot", {"key": "value"})

        config = await backend.peek_config("peek-bot")

        assert config == {"key": "value"}

    @pytest.mark.asyncio
    async def test_peek_config_nonexistent_returns_none(self, backend):
        """peek_config returns None for missing bot."""
        await backend.initialize()

        config = await backend.peek_config("missing")

        assert config is None

    @pytest.mark.asyncio
    async def test_peek_config_does_not_bump_last_accessed_at(self, backend):
        """peek_config leaves last_accessed_at unchanged (contract guarantee)."""
        await backend.initialize()
        reg = await backend.register("peek-bot", {"key": "value"})
        original_access = reg.last_accessed_at

        await asyncio.sleep(0.001)
        await backend.peek_config("peek-bot")

        # Re-read via list_all (also non-mutating in-memory) to inspect timestamp
        all_regs = await backend.list_all()
        peek_bot = next(r for r in all_regs if r.bot_id == "peek-bot")
        assert peek_bot.last_accessed_at == original_access

    @pytest.mark.asyncio
    async def test_peek_config_does_not_bump_while_get_config_does(self, backend):
        """Pin the asymmetry: peek does not touch, get does."""
        await backend.initialize()
        await backend.register("compare-bot", {"k": "v"})

        # Snapshot baseline access time via list_all
        baseline = next(
            r for r in await backend.list_all() if r.bot_id == "compare-bot"
        ).last_accessed_at

        await asyncio.sleep(0.001)
        await backend.peek_config("compare-bot")
        after_peek = next(
            r for r in await backend.list_all() if r.bot_id == "compare-bot"
        ).last_accessed_at
        assert after_peek == baseline  # peek did NOT touch

        await asyncio.sleep(0.001)
        await backend.get_config("compare-bot")
        after_get = next(
            r for r in await backend.list_all() if r.bot_id == "compare-bot"
        ).last_accessed_at
        assert after_get > baseline  # get DID touch

    @pytest.mark.asyncio
    async def test_get_config_updates_last_accessed_at(self, backend):
        """Regression-pin: get_config bumps last_accessed_at.

        Guards against silent drift toward making get_config non-touching
        — the touching contract is the user-facing read's defining
        property; that's why peek_config exists as the explicit non-
        touching sibling.
        """
        await backend.initialize()
        reg1 = await backend.register("touch-bot", {"k": "v"})
        original_access = reg1.last_accessed_at

        await asyncio.sleep(0.001)
        await backend.get_config("touch-bot")

        # get() also touches; we use it here to read the current timestamp.
        reg2 = await backend.get("touch-bot")
        assert reg2.last_accessed_at > original_access


class TestInMemoryBackendConcurrency:
    """Tests for InMemoryBackend thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_registers(self):
        """Test concurrent registration is thread-safe."""
        backend = InMemoryBackend()
        await backend.initialize()

        async def register_bot(i: int):
            await backend.register(f"bot-{i}", {"index": i})

        # Register 100 bots concurrently
        await asyncio.gather(*[register_bot(i) for i in range(100)])

        assert await backend.count() == 100

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        backend = InMemoryBackend()
        await backend.initialize()

        # Pre-populate
        for i in range(10):
            await backend.register(f"bot-{i}", {"index": i})

        async def read_and_write(i: int):
            # Read existing
            await backend.get(f"bot-{i % 10}")
            # Write new
            await backend.register(f"new-bot-{i}", {"new": True})

        await asyncio.gather(*[read_and_write(i) for i in range(50)])

        # Should have original 10 + 50 new = 60
        assert await backend.count() == 60


class TestInMemoryBackendMetadata:
    """Metadata channel + filter_metadata coverage."""

    @pytest.fixture
    def backend(self):
        return InMemoryBackend()

    @pytest.mark.asyncio
    async def test_register_persists_metadata(self, backend):
        await backend.initialize()

        reg = await backend.register(
            "alpha", {"llm": {}}, metadata={"tenant_id": "acme"}
        )

        assert reg.metadata == {"tenant_id": "acme"}
        fetched = await backend.get("alpha")
        assert fetched is not None
        assert fetched.metadata == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_register_default_metadata_is_empty_dict(self, backend):
        await backend.initialize()
        reg = await backend.register("alpha", {"llm": {}})
        assert reg.metadata == {}

    @pytest.mark.asyncio
    async def test_register_metadata_is_copied(self, backend):
        """Mutating the caller's dict after register must not leak into stored state."""
        await backend.initialize()
        meta = {"tenant_id": "acme"}
        await backend.register("alpha", {}, metadata=meta)
        meta["tenant_id"] = "mutated"
        fetched = await backend.get("alpha")
        assert fetched is not None
        assert fetched.metadata == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_re_register_replaces_metadata(self, backend):
        """register() updates metadata wholesale, like config and status."""
        await backend.initialize()
        await backend.register("alpha", {}, metadata={"tenant_id": "t1"})
        await backend.register("alpha", {}, metadata={"tenant_id": "t2", "k": "v"})
        fetched = await backend.get("alpha")
        assert fetched is not None
        assert fetched.metadata == {"tenant_id": "t2", "k": "v"}

    @pytest.mark.asyncio
    async def test_deactivate_preserves_metadata(self, backend):
        await backend.initialize()
        await backend.register("alpha", {}, metadata={"tenant_id": "t1"})
        await backend.deactivate("alpha")
        # list_all (no filter) still returns the deactivated reg
        all_regs = await backend.list_all()
        assert len(all_regs) == 1
        assert all_regs[0].metadata == {"tenant_id": "t1"}

    @pytest.mark.asyncio
    async def test_list_active_filter_metadata(self, backend):
        await backend.initialize()
        await backend.register("a", {}, metadata={"tenant_id": "t1"})
        await backend.register("b", {}, metadata={"tenant_id": "t2"})
        await backend.register("c", {}, metadata={"tenant_id": "t1"})

        t1 = await backend.list_active(filter_metadata={"tenant_id": "t1"})
        assert {r.bot_id for r in t1} == {"a", "c"}

        t2 = await backend.list_active(filter_metadata={"tenant_id": "t2"})
        assert {r.bot_id for r in t2} == {"b"}

    @pytest.mark.asyncio
    async def test_list_all_filter_metadata_includes_inactive(self, backend):
        await backend.initialize()
        await backend.register("a", {}, metadata={"tenant_id": "t1"})
        await backend.register("b", {}, metadata={"tenant_id": "t1"})
        await backend.deactivate("b")

        all_t1 = await backend.list_all(filter_metadata={"tenant_id": "t1"})
        assert {r.bot_id for r in all_t1} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_count_filter_metadata(self, backend):
        await backend.initialize()
        await backend.register("a", {}, metadata={"tenant_id": "t1"})
        await backend.register("b", {}, metadata={"tenant_id": "t2"})
        await backend.register("c", {}, metadata={"tenant_id": "t1"})

        assert await backend.count(filter_metadata={"tenant_id": "t1"}) == 2
        assert await backend.count(filter_metadata={"tenant_id": "t2"}) == 1
        assert await backend.count() == 3

    @pytest.mark.asyncio
    async def test_empty_filter_metadata_is_no_filter(self, backend):
        """Empty mapping must be treated identically to None (no-filter)."""
        await backend.initialize()
        await backend.register("a", {}, metadata={"tenant_id": "t1"})
        await backend.register("b", {}, metadata={"tenant_id": "t2"})

        assert await backend.count(filter_metadata={}) == 2
        assert await backend.count(filter_metadata=None) == 2

    @pytest.mark.asyncio
    async def test_filter_metadata_multi_key_is_and(self, backend):
        await backend.initialize()
        await backend.register(
            "a", {}, metadata={"tenant_id": "t1", "tier": "gold"}
        )
        await backend.register(
            "b", {}, metadata={"tenant_id": "t1", "tier": "silver"}
        )
        await backend.register(
            "c", {}, metadata={"tenant_id": "t2", "tier": "gold"}
        )

        result = await backend.list_active(
            filter_metadata={"tenant_id": "t1", "tier": "gold"}
        )
        assert {r.bot_id for r in result} == {"a"}


class TestInMemoryBackendSurfaceCompletion:
    """Backend surface — status kwarg, list_inactive, sort/limit/offset, stream."""

    @pytest.fixture
    def backend(self):
        return InMemoryBackend()

    @pytest.mark.asyncio
    async def test_list_all_no_status_returns_active_and_inactive(self, backend):
        """No ``status`` kwarg ⇒ all statuses returned."""
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")
        await backend.register("c", {})

        regs = await backend.list_all()
        assert {r.bot_id for r in regs} == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_list_all_status_kwarg_filters_active(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")
        await backend.register("c", {})

        regs = await backend.list_all(status="active")
        assert {r.bot_id for r in regs} == {"a", "c"}

    @pytest.mark.asyncio
    async def test_list_all_status_kwarg_filters_inactive(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")

        regs = await backend.list_all(status="inactive")
        assert {r.bot_id for r in regs} == {"b"}

    @pytest.mark.asyncio
    async def test_list_inactive_returns_only_inactive(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")
        await backend.register("c", {}, status="inactive")

        regs = await backend.list_inactive()
        assert {r.bot_id for r in regs} == {"b", "c"}

    @pytest.mark.asyncio
    async def test_list_inactive_filter_metadata(self, backend):
        await backend.initialize()
        await backend.register(
            "a", {}, status="inactive", metadata={"tenant_id": "t1"}
        )
        await backend.register(
            "b", {}, status="inactive", metadata={"tenant_id": "t2"}
        )
        await backend.register(
            "c", {}, metadata={"tenant_id": "t1"}  # active — excluded
        )

        regs = await backend.list_inactive(filter_metadata={"tenant_id": "t1"})
        assert {r.bot_id for r in regs} == {"a"}

    @pytest.mark.asyncio
    async def test_list_active_sort_by_bot_id_asc(self, backend):
        await backend.initialize()
        await backend.register("c", {})
        await backend.register("a", {})
        await backend.register("b", {})

        regs = await backend.list_active(sort=[SortSpec("bot_id", SortOrder.ASC)])
        assert [r.bot_id for r in regs] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_list_all_sort_desc(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {})
        await backend.register("c", {})

        regs = await backend.list_all(sort=[SortSpec("bot_id", SortOrder.DESC)])
        assert [r.bot_id for r in regs] == ["c", "b", "a"]

    @pytest.mark.asyncio
    async def test_list_all_sort_by_metadata_field(self, backend):
        """Sort by ``metadata.X`` resolves through the metadata channel."""
        await backend.initialize()
        await backend.register("c", {}, metadata={"priority": 3})
        await backend.register("a", {}, metadata={"priority": 1})
        await backend.register("b", {}, metadata={"priority": 2})

        regs = await backend.list_all(
            sort=[SortSpec("metadata.priority", SortOrder.ASC)]
        )
        assert [r.bot_id for r in regs] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_list_active_limit(self, backend):
        await backend.initialize()
        for i in range(5):
            await backend.register(f"bot-{i}", {})

        regs = await backend.list_active(
            sort=[SortSpec("bot_id", SortOrder.ASC)], limit=2
        )
        assert [r.bot_id for r in regs] == ["bot-0", "bot-1"]

    @pytest.mark.asyncio
    async def test_list_active_offset(self, backend):
        await backend.initialize()
        for i in range(5):
            await backend.register(f"bot-{i}", {})

        regs = await backend.list_active(
            sort=[SortSpec("bot_id", SortOrder.ASC)], offset=2
        )
        assert [r.bot_id for r in regs] == ["bot-2", "bot-3", "bot-4"]

    @pytest.mark.asyncio
    async def test_list_active_limit_and_offset(self, backend):
        """Offset is applied before limit (standard SQL semantics)."""
        await backend.initialize()
        for i in range(5):
            await backend.register(f"bot-{i}", {})

        regs = await backend.list_active(
            sort=[SortSpec("bot_id", SortOrder.ASC)], offset=1, limit=2
        )
        assert [r.bot_id for r in regs] == ["bot-1", "bot-2"]

    @pytest.mark.asyncio
    async def test_pagination_combines_with_filter_metadata(self, backend):
        await backend.initialize()
        for i in range(6):
            await backend.register(
                f"bot-{i}", {}, metadata={"tenant_id": "t1" if i < 3 else "t2"}
            )

        regs = await backend.list_active(
            filter_metadata={"tenant_id": "t1"},
            sort=[SortSpec("bot_id", SortOrder.ASC)],
            limit=2,
        )
        assert [r.bot_id for r in regs] == ["bot-0", "bot-1"]

    @pytest.mark.asyncio
    async def test_count_all_counts_all_statuses(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")

        assert await backend.count_all() == 2

    @pytest.mark.asyncio
    async def test_count_all_status_filter(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")
        await backend.register("c", {}, status="inactive")

        assert await backend.count_all(status="active") == 1
        assert await backend.count_all(status="inactive") == 2

    @pytest.mark.asyncio
    async def test_count_active_preserves_back_compat(self, backend):
        """Plain ``count()`` still counts active only."""
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")

        assert await backend.count() == 1

    @pytest.mark.asyncio
    async def test_count_inactive(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")
        await backend.register("c", {}, status="inactive")

        assert await backend.count_inactive() == 2

    @pytest.mark.asyncio
    async def test_count_all_combines_status_and_filter_metadata(self, backend):
        await backend.initialize()
        await backend.register("a", {}, metadata={"tenant_id": "t1"})
        await backend.register(
            "b", {}, status="inactive", metadata={"tenant_id": "t1"}
        )
        await backend.register("c", {}, metadata={"tenant_id": "t2"})

        assert (
            await backend.count_all(
                status="active", filter_metadata={"tenant_id": "t1"}
            )
            == 1
        )
        assert (
            await backend.count_all(
                status="inactive", filter_metadata={"tenant_id": "t1"}
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_stream_yields_all_matching(self, backend):
        await backend.initialize()
        await backend.register("a", {}, metadata={"tenant_id": "t1"})
        await backend.register("b", {}, metadata={"tenant_id": "t1"})
        await backend.register("c", {}, metadata={"tenant_id": "t2"})

        seen = [reg async for reg in backend.stream(filter_metadata={"tenant_id": "t1"})]
        assert {r.bot_id for r in seen} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_stream_status_filter(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")

        active = [reg async for reg in backend.stream(status="active")]
        assert {r.bot_id for r in active} == {"a"}

        inactive = [reg async for reg in backend.stream(status="inactive")]
        assert {r.bot_id for r in inactive} == {"b"}

    @pytest.mark.asyncio
    async def test_stream_no_filters_yields_everything(self, backend):
        await backend.initialize()
        await backend.register("a", {})
        await backend.register("b", {}, status="inactive")

        seen = [reg async for reg in backend.stream()]
        assert {r.bot_id for r in seen} == {"a", "b"}

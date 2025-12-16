"""Tests for registry backends."""

from datetime import datetime, timezone

import pytest

from dataknobs_bots.registry import InMemoryBackend, Registration


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
        import asyncio

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

        import asyncio

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


class TestInMemoryBackendConcurrency:
    """Tests for InMemoryBackend thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_registers(self):
        """Test concurrent registration is thread-safe."""
        import asyncio

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
        import asyncio

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

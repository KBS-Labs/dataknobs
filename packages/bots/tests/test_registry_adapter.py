"""Tests for DataKnobsRegistryAdapter."""

import asyncio
from datetime import datetime, timezone

import pytest

from dataknobs_bots.registry import DataKnobsRegistryAdapter, Registration
from dataknobs_data.backends.memory import AsyncMemoryDatabase


class TestDataKnobsRegistryAdapter:
    """Tests for DataKnobsRegistryAdapter with memory backend."""

    @pytest.fixture
    async def adapter(self):
        """Create a fresh adapter for each test."""
        adapter = DataKnobsRegistryAdapter(backend_type="memory")
        await adapter.initialize()
        yield adapter
        await adapter.close()

    @pytest.fixture
    async def adapter_with_db(self):
        """Create adapter with pre-configured database."""
        db = AsyncMemoryDatabase()
        await db.connect()
        adapter = DataKnobsRegistryAdapter(database=db)
        await adapter.initialize()
        yield adapter
        # Don't close - adapter shouldn't close db it didn't create
        await db.close()

    @pytest.mark.asyncio
    async def test_initialize(self, adapter):
        """Test adapter initialization."""
        assert adapter._initialized is True
        assert adapter._db is not None

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, adapter):
        """Test that initialize can be called multiple times."""
        db_before = adapter._db
        await adapter.initialize()
        assert adapter._db is db_before

    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating adapter from config dict."""
        config = {"backend": "memory"}
        adapter = DataKnobsRegistryAdapter.from_config(config)
        await adapter.initialize()

        assert adapter._backend_type == "memory"
        assert adapter._initialized is True

        await adapter.close()

    @pytest.mark.asyncio
    async def test_from_config_preserves_options(self):
        """Test that from_config passes backend options."""
        config = {"backend": "memory", "some_option": "value"}
        adapter = DataKnobsRegistryAdapter.from_config(config)

        assert adapter._backend_type == "memory"
        assert adapter._backend_config == {"some_option": "value"}

    @pytest.mark.asyncio
    async def test_close_only_owned_db(self, adapter_with_db):
        """Test that close doesn't close external database."""
        # Get reference to the underlying db
        db = adapter_with_db._db

        await adapter_with_db.close()

        # Adapter should be uninitialized but db should still work
        assert adapter_with_db._initialized is False
        # External db is still usable (we close it in fixture cleanup)

    @pytest.mark.asyncio
    async def test_register_new(self, adapter):
        """Test registering a new bot."""
        reg = await adapter.register("new-bot", {"llm": {"provider": "echo"}})

        assert reg.bot_id == "new-bot"
        assert reg.config == {"llm": {"provider": "echo"}}
        assert reg.status == "active"
        assert isinstance(reg.created_at, datetime)
        assert isinstance(reg.updated_at, datetime)

    @pytest.mark.asyncio
    async def test_register_update(self, adapter):
        """Test updating an existing registration."""
        # Initial registration
        reg1 = await adapter.register("update-bot", {"version": 1})
        original_created = reg1.created_at

        # Small delay to ensure timestamp changes
        await asyncio.sleep(0.01)

        # Update
        reg2 = await adapter.register("update-bot", {"version": 2})

        assert reg2.bot_id == "update-bot"
        assert reg2.config == {"version": 2}
        # created_at should be preserved
        assert reg2.created_at == original_created
        # updated_at should change
        assert reg2.updated_at > reg1.updated_at

    @pytest.mark.asyncio
    async def test_register_with_status(self, adapter):
        """Test registering with custom status."""
        reg = await adapter.register("status-bot", {}, status="inactive")

        assert reg.status == "inactive"

    @pytest.mark.asyncio
    async def test_get_existing(self, adapter):
        """Test getting an existing registration."""
        await adapter.register("get-bot", {"found": True})

        reg = await adapter.get("get-bot")

        assert reg is not None
        assert reg.bot_id == "get-bot"
        assert reg.config == {"found": True}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, adapter):
        """Test getting a non-existent registration."""
        reg = await adapter.get("nonexistent")

        assert reg is None

    @pytest.mark.asyncio
    async def test_get_updates_last_accessed(self, adapter):
        """Test that get updates last_accessed_at."""
        await adapter.register("access-bot", {})
        reg1 = await adapter.get("access-bot")

        await asyncio.sleep(0.01)

        reg2 = await adapter.get("access-bot")

        assert reg2.last_accessed_at > reg1.last_accessed_at

    @pytest.mark.asyncio
    async def test_get_config(self, adapter):
        """Test getting just the config."""
        config = {"llm": {"provider": "anthropic"}, "memory": {"type": "buffer"}}
        await adapter.register("config-bot", config)

        result = await adapter.get_config("config-bot")

        assert result == config

    @pytest.mark.asyncio
    async def test_get_config_nonexistent(self, adapter):
        """Test getting config for non-existent bot."""
        result = await adapter.get_config("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_exists_active(self, adapter):
        """Test exists for active registration."""
        await adapter.register("exists-bot", {})

        assert await adapter.exists("exists-bot") is True

    @pytest.mark.asyncio
    async def test_exists_inactive(self, adapter):
        """Test exists returns False for inactive registration."""
        await adapter.register("inactive-bot", {}, status="inactive")

        assert await adapter.exists("inactive-bot") is False

    @pytest.mark.asyncio
    async def test_exists_nonexistent(self, adapter):
        """Test exists for non-existent bot."""
        assert await adapter.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_unregister(self, adapter):
        """Test hard delete."""
        await adapter.register("delete-bot", {})

        result = await adapter.unregister("delete-bot")

        assert result is True
        assert await adapter.get("delete-bot") is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, adapter):
        """Test unregister on non-existent bot."""
        result = await adapter.unregister("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate(self, adapter):
        """Test soft delete."""
        await adapter.register("soft-delete-bot", {})

        result = await adapter.deactivate("soft-delete-bot")

        assert result is True

        # Should still exist but be inactive
        reg = await adapter.get("soft-delete-bot")
        assert reg is not None
        assert reg.status == "inactive"

        # Should not appear in exists check
        assert await adapter.exists("soft-delete-bot") is False

    @pytest.mark.asyncio
    async def test_deactivate_nonexistent(self, adapter):
        """Test deactivate on non-existent bot."""
        result = await adapter.deactivate("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_active(self, adapter):
        """Test listing active registrations."""
        await adapter.register("active-1", {"n": 1})
        await adapter.register("active-2", {"n": 2})
        await adapter.register("inactive-1", {"n": 3}, status="inactive")

        active = await adapter.list_active()

        assert len(active) == 2
        ids = [r.bot_id for r in active]
        assert "active-1" in ids
        assert "active-2" in ids
        assert "inactive-1" not in ids

    @pytest.mark.asyncio
    async def test_list_all(self, adapter):
        """Test listing all registrations."""
        await adapter.register("bot-1", {})
        await adapter.register("bot-2", {}, status="inactive")

        all_regs = await adapter.list_all()

        assert len(all_regs) == 2

    @pytest.mark.asyncio
    async def test_list_ids(self, adapter):
        """Test listing active bot IDs."""
        await adapter.register("id-1", {})
        await adapter.register("id-2", {})
        await adapter.register("id-3", {}, status="inactive")

        ids = await adapter.list_ids()

        assert len(ids) == 2
        assert "id-1" in ids
        assert "id-2" in ids
        assert "id-3" not in ids

    @pytest.mark.asyncio
    async def test_count(self, adapter):
        """Test counting active registrations."""
        await adapter.register("count-1", {})
        await adapter.register("count-2", {})
        await adapter.register("count-3", {}, status="inactive")

        count = await adapter.count()

        assert count == 2

    @pytest.mark.asyncio
    async def test_clear(self, adapter):
        """Test clearing all registrations."""
        await adapter.register("clear-1", {})
        await adapter.register("clear-2", {})

        await adapter.clear()

        assert await adapter.count() == 0
        assert await adapter.list_all() == []


class TestDataKnobsRegistryAdapterProtocolCompliance:
    """Test that adapter implements RegistryBackend protocol."""

    @pytest.fixture
    async def adapter(self):
        """Create adapter for protocol tests."""
        adapter = DataKnobsRegistryAdapter(backend_type="memory")
        await adapter.initialize()
        yield adapter
        await adapter.close()

    @pytest.mark.asyncio
    async def test_has_all_protocol_methods(self, adapter):
        """Verify adapter has all required protocol methods."""
        protocol_methods = [
            "initialize",
            "close",
            "register",
            "get",
            "get_config",
            "exists",
            "unregister",
            "deactivate",
            "list_active",
            "list_all",
            "list_ids",
            "count",
            "clear",
        ]

        for method in protocol_methods:
            assert hasattr(adapter, method), f"Missing method: {method}"
            assert callable(getattr(adapter, method)), f"Not callable: {method}"

    @pytest.mark.asyncio
    async def test_register_returns_registration(self, adapter):
        """Test that register returns a Registration instance."""
        result = await adapter.register("proto-bot", {})
        assert isinstance(result, Registration)

    @pytest.mark.asyncio
    async def test_get_returns_registration_or_none(self, adapter):
        """Test get return type."""
        await adapter.register("get-proto", {})

        found = await adapter.get("get-proto")
        assert isinstance(found, Registration)

        not_found = await adapter.get("nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_list_returns_list_of_registrations(self, adapter):
        """Test list methods return proper types."""
        await adapter.register("list-proto", {})

        active = await adapter.list_active()
        assert isinstance(active, list)
        assert all(isinstance(r, Registration) for r in active)

        all_regs = await adapter.list_all()
        assert isinstance(all_regs, list)
        assert all(isinstance(r, Registration) for r in all_regs)


class TestDataKnobsRegistryAdapterEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    async def adapter(self):
        """Create adapter for edge case tests."""
        adapter = DataKnobsRegistryAdapter(backend_type="memory")
        await adapter.initialize()
        yield adapter
        await adapter.close()

    @pytest.mark.asyncio
    async def test_empty_config(self, adapter):
        """Test registering with empty config."""
        reg = await adapter.register("empty-config", {})
        assert reg.config == {}

    @pytest.mark.asyncio
    async def test_nested_config(self, adapter):
        """Test registering with deeply nested config."""
        config = {
            "bot": {
                "llm": {
                    "$resource": "default",
                    "type": "llm_providers",
                },
                "memory": {
                    "type": "buffer",
                    "config": {"max_messages": 100},
                },
            },
            "features": ["chat", "search"],
        }

        reg = await adapter.register("nested-config", config)
        retrieved = await adapter.get_config("nested-config")

        assert retrieved == config

    @pytest.mark.asyncio
    async def test_special_characters_in_bot_id(self, adapter):
        """Test bot IDs with special characters."""
        special_ids = [
            "bot-with-dashes",
            "bot_with_underscores",
            "bot.with.dots",
            "bot:with:colons",
        ]

        for bot_id in special_ids:
            reg = await adapter.register(bot_id, {"id": bot_id})
            retrieved = await adapter.get(bot_id)
            assert retrieved is not None
            assert retrieved.bot_id == bot_id

    @pytest.mark.asyncio
    async def test_reactivate_inactive(self, adapter):
        """Test reactivating an inactive registration."""
        await adapter.register("reactivate-bot", {})
        await adapter.deactivate("reactivate-bot")

        assert await adapter.exists("reactivate-bot") is False

        # Re-register with active status
        reg = await adapter.register("reactivate-bot", {}, status="active")

        assert reg.status == "active"
        assert await adapter.exists("reactivate-bot") is True

    @pytest.mark.asyncio
    async def test_timestamps_are_utc(self, adapter):
        """Test that timestamps are UTC."""
        reg = await adapter.register("utc-bot", {})

        assert reg.created_at.tzinfo is not None
        assert reg.updated_at.tzinfo is not None
        assert reg.last_accessed_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_multiple_operations_sequence(self, adapter):
        """Test a realistic sequence of operations."""
        # Register
        reg = await adapter.register("seq-bot", {"version": 1})
        assert reg.status == "active"

        # Get config
        config = await adapter.get_config("seq-bot")
        assert config["version"] == 1

        # Update
        reg = await adapter.register("seq-bot", {"version": 2})
        config = await adapter.get_config("seq-bot")
        assert config["version"] == 2

        # Deactivate
        await adapter.deactivate("seq-bot")
        assert await adapter.exists("seq-bot") is False

        # Reactivate
        reg = await adapter.register("seq-bot", {"version": 3})
        assert await adapter.exists("seq-bot") is True

        # Unregister
        await adapter.unregister("seq-bot")
        assert await adapter.get("seq-bot") is None

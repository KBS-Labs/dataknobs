"""Tests for DataKnobsRegistryAdapter."""

import asyncio
from datetime import datetime

import pytest

from dataknobs_bots.registry import DataKnobsRegistryAdapter, Registration
from dataknobs_data import SortOrder, SortSpec
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
            "peek_config",
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


class TestDataKnobsRegistryAdapterPeekConfig:
    """Tests for non-mutating peek_config on the adapter."""

    @pytest.fixture
    async def adapter(self):
        adapter = DataKnobsRegistryAdapter(backend_type="memory")
        await adapter.initialize()
        yield adapter
        await adapter.close()

    @pytest.mark.asyncio
    async def test_peek_config_returns_config(self, adapter):
        """peek_config returns the same config dict as get_config."""
        config = {"llm": {"provider": "anthropic"}}
        await adapter.register("peek-bot", config)

        result = await adapter.peek_config("peek-bot")

        assert result == config

    @pytest.mark.asyncio
    async def test_peek_config_nonexistent_returns_none(self, adapter):
        """peek_config returns None for missing bot."""
        result = await adapter.peek_config("missing")

        assert result is None

    @pytest.mark.asyncio
    async def test_peek_config_does_not_bump_last_accessed_at(self, adapter):
        """peek_config leaves last_accessed_at unchanged (contract guarantee).

        Verifies via list_all() — which the adapter does not mutate.
        """
        await adapter.register("peek-bot", {"k": "v"})

        baseline = next(
            r for r in await adapter.list_all() if r.bot_id == "peek-bot"
        ).last_accessed_at

        await asyncio.sleep(0.01)
        await adapter.peek_config("peek-bot")

        after_peek = next(
            r for r in await adapter.list_all() if r.bot_id == "peek-bot"
        ).last_accessed_at
        assert after_peek == baseline

    @pytest.mark.asyncio
    async def test_peek_config_does_not_bump_while_get_config_does(self, adapter):
        """Pin the asymmetry: peek leaves the timestamp; get bumps it."""
        await adapter.register("compare-bot", {"k": "v"})

        baseline = next(
            r for r in await adapter.list_all() if r.bot_id == "compare-bot"
        ).last_accessed_at

        await asyncio.sleep(0.01)
        await adapter.peek_config("compare-bot")
        after_peek = next(
            r for r in await adapter.list_all() if r.bot_id == "compare-bot"
        ).last_accessed_at
        assert after_peek == baseline

        await asyncio.sleep(0.01)
        await adapter.get_config("compare-bot")
        after_get = next(
            r for r in await adapter.list_all() if r.bot_id == "compare-bot"
        ).last_accessed_at
        assert after_get > baseline

    @pytest.mark.asyncio
    async def test_get_config_updates_last_accessed_at(self, adapter):
        """Regression-pin: get_config bumps last_accessed_at.

        Guards against silent drift toward making get_config non-touching
        — the touching contract is the user-facing read's defining
        property; that's why peek_config exists as the explicit non-
        touching sibling.
        """
        reg1 = await adapter.register("touch-bot", {"k": "v"})
        original_access = reg1.last_accessed_at

        await asyncio.sleep(0.01)
        await adapter.get_config("touch-bot")

        reg2 = await adapter.get("touch-bot")
        assert reg2.last_accessed_at > original_access


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

        await adapter.register("nested-config", config)
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
            await adapter.register(bot_id, {"id": bot_id})
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


class TestDataKnobsRegistryAdapterMetadata:
    """Metadata channel routes through AsyncKeyedRecordStore.

    These tests pin the registry-adapter metadata-preservation contract:
    caller-supplied ``metadata`` must reach the underlying record's
    metadata column so SQL/JSONB backends can route ``metadata.X`` filters
    into the indexable channel (rather than scanning every row).

    Coverage axes:

    - Round-trip: ``register(..., metadata=...)`` → ``list``/``get`` returns metadata
    - Structural separation: metadata column is populated, no leakage into ``data``
    - ``filter_metadata={}`` is equivalent to ``filter_metadata=None`` (no filter)
    - ``filter_metadata`` AND-combines with the implicit ``status='active'`` filter
    - ``count`` honors ``filter_metadata``
    - ``register`` update semantics replace metadata (caller-controlled, not merged)
    """

    @pytest.fixture
    async def adapter(self):
        """Fresh memory-backed adapter per test."""
        adapter = DataKnobsRegistryAdapter(backend_type="memory")
        await adapter.initialize()
        yield adapter
        await adapter.close()

    @pytest.mark.asyncio
    async def test_register_with_metadata_preserved(self, adapter):
        """``register(..., metadata=...)`` round-trips through ``get``."""
        reg = await adapter.register(
            "tenant-bot",
            {"llm": {"provider": "echo"}},
            metadata={"tenant_id": "acme", "correlation_id": "corr-1"},
        )

        assert reg.metadata == {"tenant_id": "acme", "correlation_id": "corr-1"}

        retrieved = await adapter.get("tenant-bot")
        assert retrieved is not None
        assert retrieved.metadata == {
            "tenant_id": "acme",
            "correlation_id": "corr-1",
        }

    @pytest.mark.asyncio
    async def test_register_without_metadata_defaults_empty(self, adapter):
        """Omitting ``metadata=`` stores an empty dict — no None leakage."""
        reg = await adapter.register("plain-bot", {})

        assert reg.metadata == {}

        retrieved = await adapter.get("plain-bot")
        assert retrieved is not None
        assert retrieved.metadata == {}

    @pytest.mark.asyncio
    async def test_metadata_persisted_to_metadata_column_not_data(self, adapter):
        """Structural fix: metadata column is populated, data column is unchanged.

        Pre-migration, the adapter built ``Record(data=...)`` inline and
        the metadata channel was silently dropped.  This pins the
        structural separation that ``AsyncKeyedRecordStore[Registration]``
        enforces by construction.
        """
        await adapter.register(
            "split-bot",
            {"llm": {"provider": "echo"}},
            metadata={"tenant_id": "acme"},
        )

        raw = await adapter._db.read("split-bot")
        assert raw is not None
        assert raw.metadata == {"tenant_id": "acme"}
        # Structural fields live in data.
        assert raw.data["bot_id"] == "split-bot"
        assert raw.data["status"] == "active"
        # And metadata fields are not duplicated into data.
        assert "tenant_id" not in raw.data

    @pytest.mark.asyncio
    async def test_list_active_filter_metadata_matches(self, adapter):
        """``filter_metadata`` selects only matching active registrations."""
        await adapter.register("a", {}, metadata={"tenant_id": "t1"})
        await adapter.register("b", {}, metadata={"tenant_id": "t2"})
        await adapter.register("c", {}, metadata={"tenant_id": "t1"})

        t1 = await adapter.list_active(filter_metadata={"tenant_id": "t1"})
        assert {r.bot_id for r in t1} == {"a", "c"}

        t2 = await adapter.list_active(filter_metadata={"tenant_id": "t2"})
        assert {r.bot_id for r in t2} == {"b"}

    @pytest.mark.asyncio
    async def test_list_active_empty_filter_metadata_is_no_filter(self, adapter):
        """``filter_metadata={}`` is equivalent to ``filter_metadata=None``."""
        await adapter.register("a", {}, metadata={"tenant_id": "t1"})
        await adapter.register("b", {}, metadata={"tenant_id": "t2"})

        with_empty = await adapter.list_active(filter_metadata={})
        with_none = await adapter.list_active()
        assert {r.bot_id for r in with_empty} == {r.bot_id for r in with_none}
        assert len(with_empty) == 2

    @pytest.mark.asyncio
    async def test_list_active_filter_metadata_and_combine_with_status(self, adapter):
        """``filter_metadata`` AND-combines with the implicit ``status='active'`` filter.

        list_active applies ``filter_data={"status": "active"}`` under the hood;
        inactive registrations matching ``filter_metadata`` MUST NOT be returned.
        """
        await adapter.register("active-acme", {}, metadata={"tenant_id": "acme"})
        await adapter.register(
            "inactive-acme",
            {},
            status="inactive",
            metadata={"tenant_id": "acme"},
        )
        await adapter.register("active-globex", {}, metadata={"tenant_id": "globex"})

        results = await adapter.list_active(filter_metadata={"tenant_id": "acme"})
        ids = {r.bot_id for r in results}
        assert ids == {"active-acme"}
        # Inactive-acme is filtered out by status; active-globex by tenant.
        assert "inactive-acme" not in ids
        assert "active-globex" not in ids

    @pytest.mark.asyncio
    async def test_list_all_filter_metadata_includes_inactive(self, adapter):
        """``list_all`` honors ``filter_metadata`` but does NOT filter on status."""
        await adapter.register("a-active", {}, metadata={"tenant_id": "acme"})
        await adapter.register(
            "a-inactive",
            {},
            status="inactive",
            metadata={"tenant_id": "acme"},
        )
        await adapter.register("b-active", {}, metadata={"tenant_id": "globex"})

        acme = await adapter.list_all(filter_metadata={"tenant_id": "acme"})
        assert {r.bot_id for r in acme} == {"a-active", "a-inactive"}

    @pytest.mark.asyncio
    async def test_count_with_filter_metadata(self, adapter):
        """``count(filter_metadata=...)`` returns the matching active count."""
        await adapter.register("a", {}, metadata={"tenant_id": "t1"})
        await adapter.register("b", {}, metadata={"tenant_id": "t1"})
        await adapter.register("c", {}, metadata={"tenant_id": "t2"})
        await adapter.register(
            "d",
            {},
            status="inactive",
            metadata={"tenant_id": "t1"},
        )

        # t1 has two active rows (a, b) plus one inactive (d).
        assert await adapter.count(filter_metadata={"tenant_id": "t1"}) == 2
        assert await adapter.count(filter_metadata={"tenant_id": "t2"}) == 1

    @pytest.mark.asyncio
    async def test_count_empty_filter_metadata_is_no_filter(self, adapter):
        """``count(filter_metadata={})`` is equivalent to ``count()``."""
        await adapter.register("a", {}, metadata={"tenant_id": "t1"})
        await adapter.register("b", {}, metadata={"tenant_id": "t2"})

        with_empty = await adapter.count(filter_metadata={})
        with_none = await adapter.count()
        assert with_empty == with_none == 2

    @pytest.mark.asyncio
    async def test_re_register_replaces_metadata(self, adapter):
        """Re-registering with a new ``metadata=`` replaces the prior dict.

        Pins the caller-controlled semantics: the registry does not merge
        metadata across registrations.  Callers needing a merge perform it
        explicitly via ``get`` → mutate → ``register``.
        """
        await adapter.register("evolve-bot", {}, metadata={"tenant_id": "acme"})
        await adapter.register(
            "evolve-bot",
            {},
            metadata={"tenant_id": "acme", "audit": "v2"},
        )

        retrieved = await adapter.get("evolve-bot")
        assert retrieved is not None
        assert retrieved.metadata == {"tenant_id": "acme", "audit": "v2"}

    @pytest.mark.asyncio
    async def test_re_register_without_metadata_clears_prior(self, adapter):
        """Re-registering without ``metadata=`` clears prior metadata (replace, not preserve).

        Companion to ``test_re_register_replaces_metadata``: the caller-
        controlled semantics are *replace*, not *merge or preserve*.  If
        callers want to preserve metadata across re-registration, they
        must read the prior registration first and pass it back.
        """
        await adapter.register("clear-bot", {}, metadata={"tenant_id": "acme"})
        await adapter.register("clear-bot", {})

        retrieved = await adapter.get("clear-bot")
        assert retrieved is not None
        assert retrieved.metadata == {}


class TestDataKnobsRegistryAdapterSurfaceCompletion:
    """Adapter surface — status kwarg, list_inactive, sort/limit/offset, stream."""

    @pytest.fixture
    async def adapter(self):
        db = AsyncMemoryDatabase()
        adapter = DataKnobsRegistryAdapter(database=db)
        await adapter.initialize()
        try:
            yield adapter
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_list_all_no_status_returns_all(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {}, status="inactive")
        await adapter.register("c", {})

        regs = await adapter.list_all()
        assert {r.bot_id for r in regs} == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_list_all_status_active_matches_list_active(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {}, status="inactive")

        all_active = await adapter.list_all(status="active")
        list_active = await adapter.list_active()
        assert {r.bot_id for r in all_active} == {r.bot_id for r in list_active} == {"a"}

    @pytest.mark.asyncio
    async def test_list_inactive_returns_only_inactive(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {}, status="inactive")
        await adapter.register("c", {}, status="inactive")

        regs = await adapter.list_inactive()
        assert {r.bot_id for r in regs} == {"b", "c"}

    @pytest.mark.asyncio
    async def test_list_inactive_filter_metadata_and_combines(self, adapter):
        await adapter.register(
            "a", {}, status="inactive", metadata={"tenant_id": "t1"}
        )
        await adapter.register(
            "b", {}, status="inactive", metadata={"tenant_id": "t2"}
        )
        await adapter.register("c", {}, metadata={"tenant_id": "t1"})

        regs = await adapter.list_inactive(filter_metadata={"tenant_id": "t1"})
        assert {r.bot_id for r in regs} == {"a"}

    @pytest.mark.asyncio
    async def test_list_active_sort_by_bot_id(self, adapter):
        await adapter.register("c", {})
        await adapter.register("a", {})
        await adapter.register("b", {})

        regs = await adapter.list_active(sort=[SortSpec("bot_id", SortOrder.ASC)])
        assert [r.bot_id for r in regs] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_list_active_sort_desc(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {})
        await adapter.register("c", {})

        regs = await adapter.list_active(sort=[SortSpec("bot_id", SortOrder.DESC)])
        assert [r.bot_id for r in regs] == ["c", "b", "a"]

    @pytest.mark.asyncio
    async def test_list_active_limit(self, adapter):
        for i in range(5):
            await adapter.register(f"bot-{i}", {})

        regs = await adapter.list_active(
            sort=[SortSpec("bot_id", SortOrder.ASC)], limit=2
        )
        assert [r.bot_id for r in regs] == ["bot-0", "bot-1"]

    @pytest.mark.asyncio
    async def test_list_active_offset_and_limit(self, adapter):
        for i in range(5):
            await adapter.register(f"bot-{i}", {})

        regs = await adapter.list_active(
            sort=[SortSpec("bot_id", SortOrder.ASC)], offset=1, limit=2
        )
        assert [r.bot_id for r in regs] == ["bot-1", "bot-2"]

    @pytest.mark.asyncio
    async def test_pagination_combines_with_filter_metadata(self, adapter):
        for i in range(6):
            await adapter.register(
                f"bot-{i}", {}, metadata={"tenant_id": "t1" if i < 3 else "t2"}
            )

        regs = await adapter.list_active(
            filter_metadata={"tenant_id": "t1"},
            sort=[SortSpec("bot_id", SortOrder.ASC)],
            limit=2,
        )
        assert [r.bot_id for r in regs] == ["bot-0", "bot-1"]

    @pytest.mark.asyncio
    async def test_count_all_no_status(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {}, status="inactive")

        assert await adapter.count_all() == 2

    @pytest.mark.asyncio
    async def test_count_all_status_filter(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {}, status="inactive")
        await adapter.register("c", {}, status="inactive")

        assert await adapter.count_all(status="active") == 1
        assert await adapter.count_all(status="inactive") == 2

    @pytest.mark.asyncio
    async def test_count_preserves_back_compat_active_only(self, adapter):
        """Bare ``count()`` continues to count active only — no behavior change."""
        await adapter.register("a", {})
        await adapter.register("b", {}, status="inactive")

        assert await adapter.count() == 1

    @pytest.mark.asyncio
    async def test_count_inactive(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {}, status="inactive")
        await adapter.register("c", {}, status="inactive")

        assert await adapter.count_inactive() == 2

    @pytest.mark.asyncio
    async def test_count_all_combines_status_and_filter_metadata(self, adapter):
        await adapter.register("a", {}, metadata={"tenant_id": "t1"})
        await adapter.register(
            "b", {}, status="inactive", metadata={"tenant_id": "t1"}
        )
        await adapter.register("c", {}, metadata={"tenant_id": "t2"})

        assert (
            await adapter.count_all(
                status="active", filter_metadata={"tenant_id": "t1"}
            )
            == 1
        )
        assert (
            await adapter.count_all(
                status="inactive", filter_metadata={"tenant_id": "t1"}
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_stream_yields_matching(self, adapter):
        for i in range(3):
            await adapter.register(f"bot-{i}", {}, metadata={"tenant_id": "t1"})
        await adapter.register("other", {}, metadata={"tenant_id": "t2"})

        seen = [
            reg async for reg in adapter.stream(filter_metadata={"tenant_id": "t1"})
        ]
        assert {r.bot_id for r in seen} == {"bot-0", "bot-1", "bot-2"}

    @pytest.mark.asyncio
    async def test_stream_status_filter(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {}, status="inactive")

        active = [reg async for reg in adapter.stream(status="active")]
        assert {r.bot_id for r in active} == {"a"}

    @pytest.mark.asyncio
    async def test_empty_filter_metadata_is_no_filter_on_count_all(self, adapter):
        await adapter.register("a", {})
        await adapter.register("b", {})

        assert await adapter.count_all(filter_metadata={}) == 2
        assert await adapter.count_all(filter_metadata=None) == 2

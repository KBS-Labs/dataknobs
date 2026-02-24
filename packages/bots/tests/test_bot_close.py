"""Tests for DynaBot.close() resilience.

Verifies that close() continues cleaning up remaining resources
even when individual resource closes raise exceptions.
"""

import pytest

from dataknobs_bots import DynaBot


class TestDynaBotClose:
    """Tests for DynaBot.close() method."""

    @pytest.mark.asyncio
    async def test_close_all_resources(self):
        """Test that close cleanly shuts down all resources."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "memory": {"type": "buffer", "max_messages": 5},
        }
        bot = await DynaBot.from_config(config)

        # Verify resources exist before close
        assert bot.llm is not None
        assert bot.conversation_storage is not None
        assert bot.memory is not None

        # Should not raise
        await bot.close()

    @pytest.mark.asyncio
    async def test_close_continues_past_llm_failure(self):
        """Test that storage is still closed when LLM close raises."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)

        # Make the LLM provider's close() raise
        async def failing_close():
            raise RuntimeError("LLM close failed")

        bot.llm.close = failing_close  # type: ignore[assignment]

        # Track whether storage backend close was called
        backend = bot.conversation_storage.backend
        backend_closed = False
        original_backend_close = backend.close

        async def tracking_close():
            nonlocal backend_closed
            backend_closed = True
            await original_backend_close()

        backend.close = tracking_close  # type: ignore[assignment]

        # close() should not raise despite LLM failure
        await bot.close()

        # Storage backend should still have been closed
        assert backend_closed, "Storage backend was not closed after LLM failure"

    @pytest.mark.asyncio
    async def test_close_continues_past_storage_failure(self):
        """Test that close completes when storage close raises."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)

        # Make the storage backend's close() raise
        backend = bot.conversation_storage.backend

        async def failing_close():
            raise RuntimeError("Storage close failed")

        backend.close = failing_close  # type: ignore[assignment]

        # Track whether LLM close was called (comes before storage)
        llm_closed = False
        original_llm_close = bot.llm.close

        async def tracking_llm_close():
            nonlocal llm_closed
            llm_closed = True
            await original_llm_close()

        bot.llm.close = tracking_llm_close  # type: ignore[assignment]

        # close() should not raise despite storage failure
        await bot.close()

        # LLM should have been closed (before the failure)
        assert llm_closed, "LLM was not closed"

    @pytest.mark.asyncio
    async def test_double_close_is_safe(self):
        """Test that calling close() twice does not raise."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)

        await bot.close()
        # Second close should also succeed without errors
        await bot.close()

"""Tests for BotManager."""

import pytest

from dataknobs_bots.bot.manager import BotManager


@pytest.fixture
def echo_bot_config() -> dict:
    """Create a test bot configuration with echo LLM."""
    return {
        "llm": {
            "provider": "echo",
            "model": "test",
            "temperature": 0.7,
        },
        "conversation_storage": {
            "backend": "memory",
        },
    }


@pytest.fixture
def config_loader(echo_bot_config: dict):
    """Create a simple config loader function."""

    def loader(bot_id: str) -> dict:
        return echo_bot_config

    return loader


class TestBotManager:
    """Tests for BotManager."""

    @pytest.mark.asyncio
    async def test_get_or_create_with_config(self, echo_bot_config: dict):
        """Test creating a bot with inline configuration."""
        manager = BotManager()

        bot = await manager.get_or_create("test-bot", config=echo_bot_config)

        assert bot is not None
        assert manager.get_bot_count() == 1
        assert "test-bot" in manager.list_bots()

    @pytest.mark.asyncio
    async def test_get_or_create_returns_cached(self, echo_bot_config: dict):
        """Test that get_or_create returns cached bot instance."""
        manager = BotManager()

        bot1 = await manager.get_or_create("test-bot", config=echo_bot_config)
        bot2 = await manager.get_or_create("test-bot")

        assert bot1 is bot2
        assert manager.get_bot_count() == 1

    @pytest.mark.asyncio
    async def test_get_or_create_with_loader(self, config_loader):
        """Test creating a bot using config loader."""
        manager = BotManager(config_loader=config_loader)

        bot = await manager.get_or_create("test-bot")

        assert bot is not None
        assert manager.get_bot_count() == 1

    @pytest.mark.asyncio
    async def test_get_or_create_no_config_raises(self):
        """Test that get_or_create raises without config or loader."""
        manager = BotManager()

        with pytest.raises(ValueError, match="No configuration provided"):
            await manager.get_or_create("test-bot")

    @pytest.mark.asyncio
    async def test_get_existing_bot(self, echo_bot_config: dict):
        """Test getting an existing bot."""
        manager = BotManager()

        await manager.get_or_create("test-bot", config=echo_bot_config)
        bot = await manager.get("test-bot")

        assert bot is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_bot(self):
        """Test getting a nonexistent bot returns None."""
        manager = BotManager()

        bot = await manager.get("nonexistent-bot")

        assert bot is None

    @pytest.mark.asyncio
    async def test_remove_bot(self, echo_bot_config: dict):
        """Test removing a bot."""
        manager = BotManager()

        await manager.get_or_create("test-bot", config=echo_bot_config)
        removed = await manager.remove("test-bot")

        assert removed is True
        assert manager.get_bot_count() == 0
        assert await manager.get("test-bot") is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_bot(self):
        """Test removing a nonexistent bot returns False."""
        manager = BotManager()

        removed = await manager.remove("nonexistent-bot")

        assert removed is False

    @pytest.mark.asyncio
    async def test_reload_bot(self, config_loader):
        """Test reloading a bot."""
        manager = BotManager(config_loader=config_loader)

        bot1 = await manager.get_or_create("test-bot")
        bot2 = await manager.reload("test-bot")

        # Should be different instances
        assert bot1 is not bot2
        assert manager.get_bot_count() == 1

    @pytest.mark.asyncio
    async def test_reload_without_loader_raises(self, echo_bot_config: dict):
        """Test that reload raises without config loader."""
        manager = BotManager()

        await manager.get_or_create("test-bot", config=echo_bot_config)

        with pytest.raises(ValueError, match="Cannot reload without config_loader"):
            await manager.reload("test-bot")

    @pytest.mark.asyncio
    async def test_list_bots(self, echo_bot_config: dict):
        """Test listing active bots."""
        manager = BotManager()

        await manager.get_or_create("bot-1", config=echo_bot_config)
        await manager.get_or_create("bot-2", config=echo_bot_config)

        bots = manager.list_bots()

        assert len(bots) == 2
        assert "bot-1" in bots
        assert "bot-2" in bots

    @pytest.mark.asyncio
    async def test_get_bot_count(self, echo_bot_config: dict):
        """Test getting bot count."""
        manager = BotManager()

        assert manager.get_bot_count() == 0

        await manager.get_or_create("bot-1", config=echo_bot_config)
        assert manager.get_bot_count() == 1

        await manager.get_or_create("bot-2", config=echo_bot_config)
        assert manager.get_bot_count() == 2

    @pytest.mark.asyncio
    async def test_clear_all(self, echo_bot_config: dict):
        """Test clearing all bots."""
        manager = BotManager()

        await manager.get_or_create("bot-1", config=echo_bot_config)
        await manager.get_or_create("bot-2", config=echo_bot_config)

        await manager.clear_all()

        assert manager.get_bot_count() == 0
        assert manager.list_bots() == []

    @pytest.mark.asyncio
    async def test_async_config_loader(self, echo_bot_config: dict):
        """Test using an async config loader."""

        async def async_loader(bot_id: str) -> dict:
            return echo_bot_config

        manager = BotManager(config_loader=async_loader)

        bot = await manager.get_or_create("test-bot")

        assert bot is not None
        assert manager.get_bot_count() == 1

    @pytest.mark.asyncio
    async def test_class_based_config_loader(self, echo_bot_config: dict):
        """Test using a class-based config loader."""

        class MyConfigLoader:
            def load(self, bot_id: str) -> dict:
                return echo_bot_config

        manager = BotManager(config_loader=MyConfigLoader())

        bot = await manager.get_or_create("test-bot")

        assert bot is not None
        assert manager.get_bot_count() == 1

    def test_repr(self, echo_bot_config: dict):
        """Test string representation."""
        manager = BotManager()

        repr_str = repr(manager)
        assert "BotManager" in repr_str
        assert "count=0" in repr_str

"""Tests for DynaBot provider registry.

Verifies that DynaBot tracks all subsystem LLM/embedding providers via
a central registry, enabling comprehensive shutdown, observability,
and testing injection.
"""

import logging

import pytest

from dataknobs_bots import DynaBot
from dataknobs_bots.bot.base import (
    PROVIDER_ROLE_EXTRACTION,
    PROVIDER_ROLE_KB_EMBEDDING,
    PROVIDER_ROLE_MAIN,
    PROVIDER_ROLE_MEMORY_EMBEDDING,
    PROVIDER_ROLE_SUMMARY_LLM,
)
from dataknobs_bots.testing import inject_providers
from dataknobs_llm import EchoProvider


class TestProviderRegistryBasics:
    """Core registry operations: register, get, all_providers."""

    @pytest.mark.asyncio
    async def test_all_providers_includes_main(self):
        """all_providers always contains the 'main' provider."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            providers = bot.all_providers
            assert PROVIDER_ROLE_MAIN in providers
            assert providers[PROVIDER_ROLE_MAIN] is bot.llm
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_register_and_get_provider(self):
        """register_provider + get_provider round-trip works."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            extra = EchoProvider({"provider": "echo", "model": "extra"})
            bot.register_provider(PROVIDER_ROLE_MEMORY_EMBEDDING, extra)

            assert bot.get_provider(PROVIDER_ROLE_MEMORY_EMBEDDING) is extra
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_all_providers_includes_registered(self):
        """Registered providers appear in all_providers."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            extra = EchoProvider({"provider": "echo", "model": "extra"})
            bot.register_provider(PROVIDER_ROLE_MEMORY_EMBEDDING, extra)

            providers = bot.all_providers
            assert PROVIDER_ROLE_MAIN in providers
            assert PROVIDER_ROLE_MEMORY_EMBEDDING in providers
            assert len(providers) == 2
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_all_providers_returns_snapshot(self):
        """all_providers returns a fresh dict, not a mutable view."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            snap1 = bot.all_providers
            snap1["rogue"] = EchoProvider({"provider": "echo", "model": "x"})  # type: ignore[assignment]

            # The mutation should not affect the bot's actual providers
            snap2 = bot.all_providers
            assert "rogue" not in snap2
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_register_main_role_rejected(self, caplog):
        """Registering the reserved 'main' role is rejected with a warning."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            original_llm = bot.llm
            rogue = EchoProvider({"provider": "echo", "model": "rogue"})

            with caplog.at_level(logging.WARNING, logger="dataknobs_bots.bot.base"):
                bot.register_provider(PROVIDER_ROLE_MAIN, rogue)

            # Main should still be the original
            assert bot.get_provider(PROVIDER_ROLE_MAIN) is original_llm
            assert "reserved role" in caplog.text
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_get_provider_unknown_role(self):
        """get_provider returns None for an unregistered role."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            assert bot.get_provider("nonexistent") is None
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_get_provider_main_returns_llm(self):
        """get_provider('main') returns self.llm."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            assert bot.get_provider(PROVIDER_ROLE_MAIN) is bot.llm
        finally:
            await bot.close()


class TestProviderRegistryClose:
    """Verify close() follows originator-owns-lifecycle.

    The provider registry is a catalog — it does NOT close providers.
    DynaBot closes self.llm (the main provider it owns). Subsystems
    close their own providers in their own close() methods.
    """

    @pytest.mark.asyncio
    async def test_close_closes_main_provider(self):
        """close() closes the main LLM provider (owned by DynaBot)."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)

        main_llm = bot.llm
        assert isinstance(main_llm, EchoProvider)

        await bot.close()

        assert main_llm.close_count == 1, "Main provider should be closed exactly once"

    @pytest.mark.asyncio
    async def test_close_does_not_close_catalog_only_providers(self):
        """Providers only in the registry catalog are NOT closed by bot.

        The registry is for observability. Lifecycle is the originator's
        responsibility.
        """
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)

        extra = EchoProvider({"provider": "echo", "model": "extra"})
        await extra.initialize()
        bot.register_provider(PROVIDER_ROLE_MEMORY_EMBEDDING, extra)

        await bot.close()

        assert extra.close_count == 0, (
            "Catalog-only provider should not be closed by bot — "
            "originator owns lifecycle"
        )

    @pytest.mark.asyncio
    async def test_close_handles_main_provider_error(self):
        """Main provider close error doesn't prevent subsystem cleanup."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)

        # Make main provider's close raise
        async def failing_close():
            raise RuntimeError("Provider close failed")

        bot.llm.close = failing_close  # type: ignore[assignment]

        # close() should not raise despite main provider failure
        await bot.close()


class TestInjectProvidersRegistry:
    """Verify inject_providers() uses the registry."""

    @pytest.mark.asyncio
    async def test_inject_main_provider(self):
        """inject_providers replaces bot.llm."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            new_main = EchoProvider({"provider": "echo", "model": "injected"})
            inject_providers(bot, main_provider=new_main)

            assert bot.llm is new_main
            assert bot.get_provider(PROVIDER_ROLE_MAIN) is new_main
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_inject_role_providers(self):
        """inject_providers registers additional role-based providers."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            mem_embed = EchoProvider({"provider": "echo", "model": "mem"})
            inject_providers(bot, memory_embedding=mem_embed)

            assert bot.get_provider(PROVIDER_ROLE_MEMORY_EMBEDDING) is mem_embed
        finally:
            await bot.close()


class TestFromConfigRegistersProviders:
    """Verify _build_from_config() registers subsystem providers."""

    @pytest.mark.asyncio
    async def test_from_config_registers_extraction_provider(self):
        """Config with wizard reasoning registers extraction provider."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": {
                    "name": "test-wizard",
                    "stages": [
                        {
                            "name": "start",
                            "is_start": True,
                            "prompt": "Hello",
                            "transitions": [{"target": "end"}],
                        },
                        {
                            "name": "end",
                            "is_end": True,
                            "prompt": "Done",
                        },
                    ],
                },
                "extraction_config": {
                    "provider": "echo",
                    "model": "extract",
                },
            },
        }
        bot = await DynaBot.from_config(config)
        try:
            extraction = bot.get_provider(PROVIDER_ROLE_EXTRACTION)
            assert extraction is not None
            assert PROVIDER_ROLE_EXTRACTION in bot.all_providers
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_minimal_config_has_only_main(self):
        """Minimal config (no memory/KB/reasoning) has only main provider."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        bot = await DynaBot.from_config(config)
        try:
            providers = bot.all_providers
            assert list(providers.keys()) == [PROVIDER_ROLE_MAIN]
        finally:
            await bot.close()

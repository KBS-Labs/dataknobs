"""Tests for from_config() error-path cleanup (dk-29).

When DynaBot.from_config() fails **after** the LLM provider has been
created and initialized, the provider must be closed before the error
propagates.  Without proper cleanup the aiohttp SSL transport callbacks
prevent the event loop from shutting down, causing asyncio.run() to hang
indefinitely.

These tests use EchoProvider's instance tracking to verify that
close() is actually called on the internally-created provider when
_build_from_config() raises.
"""

from __future__ import annotations

import asyncio

import pytest
from dataknobs_common.exceptions import ConfigurationError

from dataknobs_bots.bot.base import DynaBot
from dataknobs_llm import EchoProvider


# ---------------------------------------------------------------------------
# Error-path cleanup tests
# ---------------------------------------------------------------------------

class TestFromConfigErrorCleanup:
    """from_config() closes internally-created provider on build failure."""

    def setup_method(self) -> None:
        """Clear EchoProvider instance tracking between tests."""
        EchoProvider.reset_tracking()

    @pytest.mark.asyncio
    async def test_provider_closed_on_reasoning_error(self) -> None:
        """Provider.close() is called when _build_from_config() raises.

        This is the structural fix for dk-29: if close() is called on
        error, the aiohttp drain sleep in _close_client() prevents the
        event loop from hanging on shutdown.

        Uses EchoProvider instance tracking to inspect the provider that
        from_config() created internally via LLMProviderFactory.
        """
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "nonexistent_strategy"},
        }

        with EchoProvider.track_instances() as instances:
            with pytest.raises(ValueError, match="Unknown reasoning strategy"):
                await DynaBot.from_config(config)

        assert len(instances) == 1
        provider = instances[0]
        assert provider.close_count == 1, (
            "Provider must be closed on error path to allow aiohttp "
            "SSL transport callbacks to drain (dk-29)"
        )

    @pytest.mark.asyncio
    async def test_provider_closed_on_storage_error(self) -> None:
        """Provider.close() is called when storage config is invalid."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {},  # missing 'backend' and 'storage_class'
        }

        with EchoProvider.track_instances() as instances:
            with pytest.raises(ConfigurationError, match="conversation_storage requires"):
                await DynaBot.from_config(config)

        assert len(instances) == 1
        assert instances[0].close_count == 1

    @pytest.mark.asyncio
    async def test_error_path_completes_promptly(self) -> None:
        """from_config() error path completes within 5 seconds.

        The original dk-29 bug caused asyncio.run() to hang
        indefinitely.  This test guards against the hang by wrapping
        the error path in asyncio.wait_for() with a generous timeout.
        """
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "nonexistent_strategy"},
        }

        async def attempt_from_config() -> None:
            await DynaBot.from_config(config)

        with pytest.raises(ValueError, match="Unknown reasoning strategy"):
            await asyncio.wait_for(attempt_from_config(), timeout=5.0)


# ---------------------------------------------------------------------------
# Tool resolution — loud failure by default
# ---------------------------------------------------------------------------

class TestToolResolutionFailure:
    """Tool resolution errors raise ConfigurationError by default."""

    def setup_method(self) -> None:
        EchoProvider.reset_tracking()

    @pytest.mark.asyncio
    async def test_bad_tool_class_raises(self) -> None:
        """A bad tool class path raises ConfigurationError."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {"class": "nonexistent.module.BadTool", "params": {}},
            ],
        }

        with EchoProvider.track_instances() as instances:
            with pytest.raises(ConfigurationError, match="Failed to import tool class"):
                await DynaBot.from_config(config)

        # Provider must be closed on error path
        assert len(instances) == 1
        assert instances[0].close_count == 1

    @pytest.mark.asyncio
    async def test_optional_tool_skipped_gracefully(self) -> None:
        """A bad tool class with optional: true is skipped with a warning."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {
                    "class": "nonexistent.module.BadTool",
                    "params": {},
                    "optional": True,
                },
            ],
        }

        bot = await DynaBot.from_config(config)
        async with bot:
            assert len(bot.tool_registry.list_tools()) == 0

    @pytest.mark.asyncio
    async def test_optional_xref_dict_skipped_gracefully(self) -> None:
        """An optional xref dict with a bad target is skipped, not raised."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {},
            "tools": [
                {"xref": "xref:tools[nonexistent]", "optional": True},
            ],
        }

        bot = await DynaBot.from_config(config)
        async with bot:
            assert len(bot.tool_registry.list_tools()) == 0

    @pytest.mark.asyncio
    async def test_optional_xref_propagates_to_resolved_definition(self) -> None:
        """optional flag propagates through xref to the resolved definition."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "bad_tool": {
                    "class": "nonexistent.module.BadTool",
                    "params": {},
                },
            },
            "tools": [
                {"xref": "xref:tools[bad_tool]", "optional": True},
            ],
        }

        # The definition itself is not marked optional, but the xref
        # reference is — the optional flag must propagate through.
        bot = await DynaBot.from_config(config)
        async with bot:
            assert len(bot.tool_registry.list_tools()) == 0

    @pytest.mark.asyncio
    async def test_required_tool_fails_even_with_optional_sibling(self) -> None:
        """A required tool failure raises even when an optional sibling was skipped."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {
                    "class": "nonexistent.module.OptionalTool",
                    "optional": True,
                },
                {
                    "class": "also.nonexistent.RequiredTool",
                },
            ],
        }

        with pytest.raises(ConfigurationError, match="Failed to import tool class"):
            await DynaBot.from_config(config)


# ---------------------------------------------------------------------------
# Middleware resolution — loud failure by default
# ---------------------------------------------------------------------------

class TestMiddlewareResolutionFailure:
    """Middleware resolution errors raise ConfigurationError by default."""

    def setup_method(self) -> None:
        EchoProvider.reset_tracking()

    @pytest.mark.asyncio
    async def test_bad_middleware_class_raises(self) -> None:
        """A bad middleware class path raises ConfigurationError."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "middleware": [
                {"class": "nonexistent.module.BadMiddleware"},
            ],
        }

        with EchoProvider.track_instances() as instances:
            with pytest.raises(ConfigurationError, match="Failed to create middleware"):
                await DynaBot.from_config(config)

        assert len(instances) == 1
        assert instances[0].close_count == 1

    @pytest.mark.asyncio
    async def test_optional_middleware_skipped_gracefully(self) -> None:
        """A bad middleware with optional: true is skipped with a warning."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "middleware": [
                {
                    "class": "nonexistent.module.BadMiddleware",
                    "optional": True,
                },
            ],
        }

        bot = await DynaBot.from_config(config)
        async with bot:
            assert bot.middleware == []

    @pytest.mark.asyncio
    async def test_valid_middleware_still_loads(self) -> None:
        """Valid middleware continues to load normally."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "middleware": [
                {"class": "dataknobs_bots.middleware.logging.LoggingMiddleware"},
            ],
        }

        bot = await DynaBot.from_config(config)
        async with bot:
            from dataknobs_bots.middleware.logging import LoggingMiddleware

            assert len(bot.middleware) == 1
            assert isinstance(bot.middleware[0], LoggingMiddleware)

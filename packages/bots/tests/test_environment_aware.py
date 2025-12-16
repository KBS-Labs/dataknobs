"""Tests for environment-aware configuration in DynaBot and BotManager."""

import pytest

from dataknobs_bots import DynaBot, BotManager
from dataknobs_config import EnvironmentAwareConfig, EnvironmentConfig


class TestDynaBotEnvironmentAware:
    """Tests for DynaBot.from_environment_aware_config()."""

    @pytest.fixture
    def env_config(self):
        """Create a test environment configuration."""
        return EnvironmentConfig(
            name="test",
            resources={
                "llm_providers": {
                    "default": {
                        "provider": "echo",
                        "model": "test",
                        "temperature": 0.5,
                    },
                },
                "databases": {
                    "conversations": {
                        "backend": "memory",
                    },
                },
            },
            settings={"log_level": "DEBUG"},
        )

    @pytest.fixture
    def portable_config(self):
        """Create a portable configuration with $resource references."""
        return {
            "bot": {
                "llm": {
                    "$resource": "default",
                    "type": "llm_providers",
                    "max_tokens": 500,  # Override
                },
                "conversation_storage": {
                    "$resource": "conversations",
                    "type": "databases",
                },
            }
        }

    @pytest.mark.asyncio
    async def test_from_environment_aware_config_with_dict(
        self, portable_config, env_config
    ):
        """Test creating DynaBot from dict with $resource references."""
        bot = await DynaBot.from_environment_aware_config(
            portable_config,
            environment=env_config,
        )

        assert bot is not None
        # Temperature comes from environment config's llm_providers.default
        assert bot.default_temperature == 0.5
        # max_tokens comes from portable config override
        assert bot.default_max_tokens == 500

    @pytest.mark.asyncio
    async def test_from_environment_aware_config_with_env_aware_config(
        self, portable_config, env_config
    ):
        """Test creating DynaBot from EnvironmentAwareConfig."""
        env_aware = EnvironmentAwareConfig(
            config=portable_config,
            environment=env_config,
        )

        bot = await DynaBot.from_environment_aware_config(env_aware)

        assert bot is not None
        assert bot.default_temperature == 0.5

    @pytest.mark.asyncio
    async def test_from_environment_aware_config_with_custom_config_key(
        self, env_config
    ):
        """Test using custom config_key."""
        config = {
            "my_bot_config": {
                "llm": {
                    "$resource": "default",
                    "type": "llm_providers",
                },
                "conversation_storage": {
                    "$resource": "conversations",
                    "type": "databases",
                },
            }
        }

        bot = await DynaBot.from_environment_aware_config(
            config,
            environment=env_config,
            config_key="my_bot_config",
        )

        assert bot is not None

    @pytest.mark.asyncio
    async def test_from_environment_aware_config_with_none_config_key(
        self, env_config
    ):
        """Test with config_key=None (use root config)."""
        # Config without nesting
        config = {
            "llm": {
                "$resource": "default",
                "type": "llm_providers",
            },
            "conversation_storage": {
                "$resource": "conversations",
                "type": "databases",
            },
        }

        bot = await DynaBot.from_environment_aware_config(
            config,
            environment=env_config,
            config_key=None,
        )

        assert bot is not None

    @pytest.mark.asyncio
    async def test_resource_default_merging(self, env_config):
        """Test that portable config values fill in missing resource values.

        Environment config values take precedence over portable config defaults.
        Portable config values only fill in keys missing from the environment.
        """
        config = {
            "bot": {
                "llm": {
                    "$resource": "default",
                    "type": "llm_providers",
                    "temperature": 0.9,  # Won't override env's 0.5
                    "max_tokens": 2000,  # Will be used (not in env config)
                },
                "conversation_storage": {
                    "$resource": "conversations",
                    "type": "databases",
                },
            }
        }

        bot = await DynaBot.from_environment_aware_config(
            config,
            environment=env_config,
        )

        # Environment config value wins for temperature (env has 0.5)
        assert bot.default_temperature == 0.5
        # Portable config value used for max_tokens (not in env config)
        assert bot.default_max_tokens == 2000


class TestDynaBotGetPortableConfig:
    """Tests for DynaBot.get_portable_config()."""

    def test_get_portable_config_from_dict(self):
        """Test extracting portable config from dict."""
        config = {"bot": {"llm": {"$resource": "default"}}}

        portable = DynaBot.get_portable_config(config)

        assert portable == config
        assert portable is config  # Should be same reference for dicts

    def test_get_portable_config_from_env_aware_config(self):
        """Test extracting portable config from EnvironmentAwareConfig."""
        original = {"bot": {"llm": {"$resource": "default"}}}
        env_config = EnvironmentConfig(name="test")
        env_aware = EnvironmentAwareConfig(config=original, environment=env_config)

        portable = DynaBot.get_portable_config(env_aware)

        assert portable == original
        # Should be a copy, not the same reference
        assert portable is not original

    def test_get_portable_config_preserves_resource_refs(self):
        """Test that $resource references are preserved."""
        original = {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
                "database": {"$resource": "main", "type": "databases"},
            }
        }
        env_config = EnvironmentConfig(
            name="test",
            resources={
                "llm_providers": {"default": {"provider": "openai"}},
                "databases": {"main": {"backend": "postgres"}},
            },
        )
        env_aware = EnvironmentAwareConfig(config=original, environment=env_config)

        portable = DynaBot.get_portable_config(env_aware)

        # Resource references should be intact, not resolved
        assert portable["bot"]["llm"]["$resource"] == "default"
        assert portable["bot"]["database"]["$resource"] == "main"


class TestBotManagerEnvironmentAware:
    """Tests for BotManager with environment support."""

    @pytest.fixture
    def env_config(self):
        """Create a test environment configuration."""
        return EnvironmentConfig(
            name="test",
            resources={
                "llm_providers": {
                    "default": {
                        "provider": "echo",
                        "model": "test",
                    },
                },
                "databases": {
                    "conversations": {
                        "backend": "memory",
                    },
                },
            },
        )

    @pytest.fixture
    def portable_config(self):
        """Create a portable configuration."""
        return {
            "bot": {
                "llm": {
                    "$resource": "default",
                    "type": "llm_providers",
                },
                "conversation_storage": {
                    "$resource": "conversations",
                    "type": "databases",
                },
            }
        }

    @pytest.fixture
    def resolved_config(self):
        """Create a resolved (non-portable) configuration."""
        return {
            "llm": {
                "provider": "echo",
                "model": "test",
            },
            "conversation_storage": {
                "backend": "memory",
            },
        }

    def test_init_with_environment_config(self, env_config):
        """Test initializing BotManager with EnvironmentConfig."""
        manager = BotManager(environment=env_config)

        assert manager.environment_name == "test"
        assert manager.environment is env_config

    def test_init_without_environment(self):
        """Test initializing BotManager without environment."""
        manager = BotManager()

        assert manager.environment_name is None
        assert manager.environment is None

    @pytest.mark.asyncio
    async def test_get_or_create_with_environment(
        self, env_config, portable_config
    ):
        """Test creating bot with environment resolution."""
        manager = BotManager(environment=env_config)

        bot = await manager.get_or_create("test-bot", config=portable_config)

        assert bot is not None
        assert manager.get_bot_count() == 1

    @pytest.mark.asyncio
    async def test_get_or_create_without_environment(self, resolved_config):
        """Test creating bot without environment (traditional path)."""
        manager = BotManager()

        bot = await manager.get_or_create(
            "test-bot",
            config=resolved_config,
            use_environment=False,
        )

        assert bot is not None

    @pytest.mark.asyncio
    async def test_get_or_create_explicit_use_environment_true(
        self, env_config, portable_config
    ):
        """Test explicit use_environment=True."""
        manager = BotManager(environment=env_config)

        bot = await manager.get_or_create(
            "test-bot",
            config=portable_config,
            use_environment=True,
        )

        assert bot is not None

    @pytest.mark.asyncio
    async def test_get_or_create_explicit_use_environment_false(
        self, env_config, resolved_config
    ):
        """Test explicit use_environment=False skips resolution."""
        manager = BotManager(environment=env_config)

        # Even with environment configured, use_environment=False
        # should use config as-is
        bot = await manager.get_or_create(
            "test-bot",
            config=resolved_config,
            use_environment=False,
        )

        assert bot is not None

    @pytest.mark.asyncio
    async def test_get_or_create_auto_detect_with_env_aware_config(
        self, env_config, portable_config
    ):
        """Test auto-detection with EnvironmentAwareConfig."""
        manager = BotManager()  # No environment configured

        env_aware = EnvironmentAwareConfig(
            config=portable_config,
            environment=env_config,
        )

        # Should auto-detect that env resolution is needed
        bot = await manager.get_or_create("test-bot", config=env_aware)

        assert bot is not None

    def test_get_portable_config(self, portable_config):
        """Test BotManager.get_portable_config()."""
        manager = BotManager()

        # Dict passes through
        portable = manager.get_portable_config(portable_config)
        assert portable == portable_config

    def test_get_portable_config_from_env_aware(self, env_config, portable_config):
        """Test get_portable_config from EnvironmentAwareConfig."""
        manager = BotManager(environment=env_config)

        env_aware = EnvironmentAwareConfig(
            config=portable_config,
            environment=env_config,
        )

        portable = manager.get_portable_config(env_aware)

        assert portable == portable_config

    def test_repr_with_environment(self, env_config):
        """Test repr includes environment name."""
        manager = BotManager(environment=env_config)

        repr_str = repr(manager)

        assert "BotManager" in repr_str
        assert "test" in repr_str  # Environment name

    def test_repr_without_environment(self):
        """Test repr without environment."""
        manager = BotManager()

        repr_str = repr(manager)

        assert "BotManager" in repr_str
        assert "environment" not in repr_str


class TestEnvironmentSwitching:
    """Tests for switching environments."""

    @pytest.fixture
    def dev_env(self):
        """Development environment config."""
        return EnvironmentConfig(
            name="development",
            resources={
                "llm_providers": {
                    "default": {
                        "provider": "echo",
                        "model": "dev-model",
                    },
                },
                "databases": {
                    "conversations": {"backend": "memory"},
                },
            },
        )

    @pytest.fixture
    def prod_env(self):
        """Production environment config."""
        return EnvironmentConfig(
            name="production",
            resources={
                "llm_providers": {
                    "default": {
                        "provider": "echo",
                        "model": "prod-model",
                    },
                },
                "databases": {
                    "conversations": {"backend": "memory"},
                },
            },
        )

    @pytest.fixture
    def portable_config(self):
        """Portable config that works in any environment."""
        return {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
                "conversation_storage": {"$resource": "conversations", "type": "databases"},
            }
        }

    @pytest.mark.asyncio
    async def test_same_config_different_environments(
        self, portable_config, dev_env, prod_env
    ):
        """Test same portable config resolves differently per environment."""
        env_aware = EnvironmentAwareConfig(
            config=portable_config,
            environment=dev_env,
        )

        # Resolve for dev
        dev_resolved = env_aware.resolve_for_build("bot")
        assert dev_resolved["llm"]["model"] == "dev-model"

        # Switch to prod and resolve
        prod_aware = env_aware.with_environment(prod_env)
        prod_resolved = prod_aware.resolve_for_build("bot")
        assert prod_resolved["llm"]["model"] == "prod-model"

        # Original should be unchanged
        dev_resolved_again = env_aware.resolve_for_build("bot")
        assert dev_resolved_again["llm"]["model"] == "dev-model"

    @pytest.mark.asyncio
    async def test_bot_creation_per_environment(
        self, portable_config, dev_env, prod_env
    ):
        """Test creating bots with different environments."""
        # Create bot for dev
        dev_bot = await DynaBot.from_environment_aware_config(
            portable_config,
            environment=dev_env,
        )

        # Create bot for prod
        prod_bot = await DynaBot.from_environment_aware_config(
            portable_config,
            environment=prod_env,
        )

        # Both should be valid bots
        assert dev_bot is not None
        assert prod_bot is not None
        # They should be different instances
        assert dev_bot is not prod_bot

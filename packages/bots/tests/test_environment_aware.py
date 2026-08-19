"""Tests for environment-aware configuration in DynaBot and its registries."""

import pytest

from dataknobs_bots import DynaBot, BotManager
from dataknobs_bots.bot import InMemoryBotRegistry
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
    async def test_from_environment_aware_config_with_dict(self, portable_config, env_config):
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
    async def test_from_environment_aware_config_with_custom_config_key(self, env_config):
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
    async def test_from_environment_aware_config_with_none_config_key(self, env_config):
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


@pytest.mark.filterwarnings("ignore:BotManager is deprecated:DeprecationWarning")
class TestBotManagerEnvironmentAware:
    """Tests for BotManager with environment support (deprecated)."""

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
    async def test_get_or_create_with_environment(self, env_config, portable_config):
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
    async def test_get_or_create_explicit_use_environment_true(self, env_config, portable_config):
        """Test explicit use_environment=True."""
        manager = BotManager(environment=env_config)

        bot = await manager.get_or_create(
            "test-bot",
            config=portable_config,
            use_environment=True,
        )

        assert bot is not None

    @pytest.mark.asyncio
    async def test_get_or_create_explicit_use_environment_false(self, env_config, resolved_config):
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
    async def test_same_config_different_environments(self, portable_config, dev_env, prod_env):
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
    async def test_bot_creation_per_environment(self, portable_config, dev_env, prod_env):
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


class TestResolutionThroughAnAlreadyExpandedEnvironment:
    """The arm the production entry point actually takes.

    ``DynaBot.from_environment_aware_config`` builds its environment with
    ``EnvironmentConfig.load()``, which substitutes — so the resolution it
    drives must *not* substitute again. Every other test in this file
    constructs an environment directly, which is the opposite provenance, so
    the arm the shipped call path uses had no consumer-side coverage.

    The value below is chosen so a second expansion is visible: one pass
    yields the secret, and a second re-reads that secret as a template.
    """

    @pytest.fixture
    def env(self, monkeypatch):
        monkeypatch.setenv("CONV_DSN", "postgres://u:p${x}ss@h/db")
        monkeypatch.setenv("x", "INJECTED")
        return EnvironmentConfig.from_dict(
            {
                "name": "production",
                "resources": {
                    "databases": {
                        "conversations": {
                            "backend": "postgres",
                            "connection_string": "${CONV_DSN}",
                        }
                    }
                },
            }
        )

    def test_a_resource_value_reaches_the_factory_expanded_exactly_once(self, env):
        assert env.substituted is True

        config = EnvironmentAwareConfig(
            config={
                "bot": {
                    "conversation_storage": {
                        "$resource": "conversations",
                        "type": "databases",
                    }
                }
            },
            environment=env,
        )

        resolved = config.resolve_for_build("bot")

        assert resolved["conversation_storage"]["connection_string"] == "postgres://u:p${x}ss@h/db"

    def test_a_nested_reference_carried_by_the_environment_too(self, monkeypatch):
        """A `$resource` block inside a resource the environment supplies.

        Built with the nesting in place rather than written in afterwards:
        an amended config's ``substituted`` flag does not update, and
        ``substituted_view()`` is a no-op once the flag is already ``True``,
        so a post-construction write would leave the added value raw.
        """
        monkeypatch.setenv("CONV_DSN", "postgres://u:p${x}ss@h/db")
        monkeypatch.setenv("x", "INJECTED")
        env = EnvironmentConfig.from_dict(
            {
                "name": "production",
                "resources": {
                    "databases": {
                        "conversations": {
                            "backend": "postgres",
                            "replica": {
                                "$resource": "absent",
                                "type": "databases",
                                "connection_string": "${CONV_DSN}",
                            },
                        }
                    }
                },
            }
        )

        config = EnvironmentAwareConfig(
            config={"bot": {"db": {"$resource": "conversations", "type": "databases"}}},
            environment=env,
        )

        resolved = config.resolve_for_build("bot")

        assert resolved["db"]["replica"]["connection_string"] == "postgres://u:p${x}ss@h/db"


class TestMissingResourcePolicyPassThrough:
    """The missing-resource policy is declarable at the bot entry points.

    ``dataknobs-config`` resolves a reference naming a resource the
    environment does not define leniently by default -- warn, degrade to
    the reference's inline defaults -- and takes a ``strict_resources``
    argument to say otherwise. Every level of that chain was reachable
    except the two that live in code, because the argument stopped at
    ``resolve_for_build`` and none of this package's entry points
    forwarded it. A caller handing in a plain dict could not reach them
    at all: the ``EnvironmentAwareConfig`` those levels live on is built
    inside the entry point, so its constructor is not the caller's to
    pass.

    The consequence is the one the policy exists for. A degraded
    ``conversation_storage`` binding is an in-memory database, which
    holds state perfectly until the process restarts -- so the bot below
    builds either way, and only the policy decides whether the operator
    hears about it before or after deployment.
    """

    @staticmethod
    def _environment(settings: dict | None = None) -> EnvironmentConfig:
        """An environment defining the LLM binding but not the storage one."""
        return EnvironmentConfig(
            name="test",
            resources={
                "llm_providers": {"default": {"provider": "echo", "model": "test"}},
            },
            settings=settings or {},
        )

    @staticmethod
    def _portable() -> dict:
        return {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
                "conversation_storage": {
                    "$resource": "conversations",
                    "type": "databases",
                    # The inline default the lenient path degrades to.
                    "backend": "memory",
                },
            }
        }

    async def test_the_default_still_degrades(self):
        """Unset means unset: the lenient default is unchanged."""
        bot = await DynaBot.from_environment_aware_config(
            self._portable(), environment=self._environment()
        )
        assert bot is not None

    async def test_a_dict_caller_can_now_ask_to_fail(self):
        """The dict branch builds the config here, so this is the only level it has."""
        from dataknobs_config import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError, match="conversations"):
            await DynaBot.from_environment_aware_config(
                self._portable(),
                environment=self._environment(),
                strict_resources=True,
            )

    async def test_it_reaches_the_prebuilt_config_branch_too(self):
        """A caller handing in an EnvironmentAwareConfig never touches the constructor.

        Forwarding the policy there rather than to ``resolve_for_build``
        would have covered only half the signature's accepted input, and
        silently: the argument would be accepted and ignored.
        """
        from dataknobs_config import ResourceNotFoundError

        env_aware = EnvironmentAwareConfig(config=self._portable(), environment=self._environment())

        with pytest.raises(ResourceNotFoundError, match="conversations"):
            await DynaBot.from_environment_aware_config(env_aware, strict_resources=True)

    async def test_the_call_level_outranks_the_config_s_own(self):
        """A config carrying a policy is overridden by the caller, per the chain.

        The call level is more specific than the instance level, so a
        caller who says ``False`` gets the degrade even though the config
        was built strict. This is the documented precedence rather than a
        rule this package adds, and it is why the default is ``None``.
        """
        env_aware = EnvironmentAwareConfig(
            config=self._portable(),
            environment=self._environment(),
            strict_resources=True,
        )

        bot = await DynaBot.from_environment_aware_config(env_aware, strict_resources=False)
        assert bot is not None

    async def test_none_leaves_the_config_s_own_policy_alone(self):
        """Not passing it must not read as passing ``False``."""
        from dataknobs_config import ResourceNotFoundError

        env_aware = EnvironmentAwareConfig(
            config=self._portable(),
            environment=self._environment(),
            strict_resources=True,
        )

        with pytest.raises(ResourceNotFoundError, match="conversations"):
            await DynaBot.from_environment_aware_config(env_aware)

    async def test_the_operator_level_still_decides_when_nothing_else_does(self):
        """The environment setting was already reachable; it must stay so."""
        from dataknobs_config import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError, match="conversations"):
            await DynaBot.from_environment_aware_config(
                self._portable(),
                environment=self._environment({"strict_resources": True}),
            )

    async def test_a_reference_may_still_opt_out_of_a_strict_call(self):
        """``$required: false`` is the most specific level and outranks the call."""
        portable = self._portable()
        portable["bot"]["conversation_storage"]["$required"] = False

        bot = await DynaBot.from_environment_aware_config(
            portable, environment=self._environment(), strict_resources=True
        )
        assert bot is not None


class TestMissingResourcePolicyThroughTheRegistries:
    """The same policy, reachable from the paths consumers are told to use.

    ``BotRegistry`` is the recommended entry point -- ``BotManager``'s
    deprecation message names it as the replacement -- and it always
    takes the dict branch, since a config comes back from its backend as
    a mapping. So without a registry-level parameter the two code levels
    of the chain were unreachable for exactly the callers on the path
    this package recommends.

    Registry-wide rather than per-call because both classes cache: a
    policy passed to one ``get_bot`` call would silently decide what
    every later caller gets back.
    """

    @staticmethod
    def _environment() -> EnvironmentConfig:
        return EnvironmentConfig(
            name="test",
            resources={
                "llm_providers": {"default": {"provider": "echo", "model": "test"}},
            },
            settings={},
        )

    @staticmethod
    def _portable() -> dict:
        return {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
                "conversation_storage": {
                    "$resource": "conversations",
                    "type": "databases",
                    "backend": "memory",
                },
            }
        }

    async def test_registry_default_still_degrades(self):
        registry = InMemoryBotRegistry(environment=self._environment(), validate_on_register=False)
        await registry.initialize()
        try:
            await registry.register("b", self._portable())
            assert await registry.get_bot("b") is not None
        finally:
            await registry.close()

    async def test_registry_can_declare_the_binding_must_exist(self):
        from dataknobs_config import ResourceNotFoundError

        registry = InMemoryBotRegistry(
            environment=self._environment(),
            validate_on_register=False,
            strict_resources=True,
        )
        await registry.initialize()
        try:
            await registry.register("b", self._portable())
            with pytest.raises(ResourceNotFoundError, match="conversations"):
                await registry.get_bot("b")
        finally:
            await registry.close()

    async def test_the_in_memory_subclass_forwards_it_too(self):
        """It re-declares the whole signature rather than inheriting it.

        Which is why the parameter has to be added twice, and why a test
        that only drove the base class would have reported the in-process
        registry -- the one most consumers reach for first -- as covered.
        """
        import inspect

        from dataknobs_bots.bot import BotRegistry

        for cls in (BotRegistry, InMemoryBotRegistry):
            assert "strict_resources" in inspect.signature(cls.__init__).parameters

    async def test_manager_can_declare_it_as_well(self):
        from dataknobs_config import ResourceNotFoundError

        manager = BotManager(environment=self._environment(), strict_resources=True)
        with pytest.raises(ResourceNotFoundError, match="conversations"):
            await manager.get_or_create("b", config=self._portable())

    async def test_manager_default_still_degrades(self):
        manager = BotManager(environment=self._environment())
        assert await manager.get_or_create("b", config=self._portable()) is not None

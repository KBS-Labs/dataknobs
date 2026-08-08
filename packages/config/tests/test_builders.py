"""Tests for object construction and builders."""

import pytest

from dataknobs_config import Config, ConfigError
from dataknobs_config.examples import (
    AsyncWidget,
    Cache,
    Database,
    PlainWidget,
    SyncWidget,
)


class TestClassLoadingDisclosure:
    """``_load_class`` imports the module, and importing runs it."""

    def test_a_module_that_fails_to_import_does_not_leak_its_message(
        self, tmp_path, monkeypatch
    ):
        """The non-``ImportError`` branch catches arbitrary module-level code.

        A dotted class path resolves through ``importlib.import_module``, which
        *executes* the target module. Anything that module does at import time
        — opening a connection, reading a secret — can raise with text this
        project does not control, and ``ConfigError`` is a
        ``ConfigurationError``, which the bots API layer renders at the HTTP
        boundary.

        The class path survives, because it comes from the config rather than
        from the exception, and so does the exception type, because a class
        name is bounded. The rest travels on ``__cause__``.
        """
        package = tmp_path / "leakypkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "boom.py").write_text(
            'raise ValueError("connect failed: '
            'postgresql://svc:hunter2@db.internal:5432/prod")\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        config = Config(
            {"widget": [{"name": "w", "class": "leakypkg.boom.Widget"}]}
        )

        with pytest.raises(ConfigError) as excinfo:
            config.build_object("xref:widget[w]")

        message = str(excinfo.value)
        assert "hunter2" not in message
        assert "leakypkg.boom.Widget" in message
        assert "ValueError" in message


class TestClassPathResolution:
    """``class:`` is a dotted path, resolved the way every other one is.

    ``_load_class`` used to import and ``getattr`` for itself, which cost it
    the ``:`` separator the rest of the workspace accepts and cost its callers
    two messages it took care to write — see the double-wrap test below.
    """

    @pytest.mark.parametrize("separator", [":", "."], ids=["colon", "dot"])
    def test_a_class_path_resolves_by_either_separator(self, separator):
        """``module:Name`` and ``module.Name`` name the same class.

        This site accepted only ``.``, so a ``class:`` value valid under one
        spelling was invalid under the other for no reason a config author
        could see.
        """
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": f"dataknobs_config.examples{separator}PlainWidget",
                    }
                ]
            }
        )

        assert isinstance(config.build_object("xref:widget[w]"), PlainWidget)

    @pytest.mark.parametrize(
        ("class_path", "expected"),
        [
            ("NoSeparatorAtAll", "NoSeparatorAtAll"),
            ("dataknobs_config.examples.NoSuchWidget", "NoSuchWidget"),
        ],
        ids=["malformed", "missing attribute"],
    )
    def test_a_resolution_failure_says_what_was_wrong(self, class_path, expected):
        """The specific diagnosis reaches the caller, not a generic wrapper.

        Both failures were raised *inside* the method's own ``try``, so its
        trailing ``except Exception`` caught them and replaced each with
        ``"Failed to load class <path> (<type>)"``. The messages existed and
        were never seen: a caller asking why got told the class failed to
        load, which they knew, and the exception type of the report they were
        already reading.
        """
        config = Config({"widget": [{"name": "w", "class": class_path}]})

        with pytest.raises(ConfigError) as excinfo:
            config.build_object("xref:widget[w]")

        message = str(excinfo.value)
        assert expected in message
        assert "Failed to load class" not in message, (
            f"the generic wrapper replaced the diagnosis: {message}"
        )


class TestConfigFileDisclosure:
    """A config file's path and contents are the server's, not the caller's.

    ``ConfigFileNotFoundError`` is a ``NotFoundError`` and the parse failure is
    a ``ValidationError``, so the bots API layer renders both with their
    message disclosed — a 404 and a 422. Config loading is reachable from a
    request because bots are built lazily on the request path.
    """

    def test_a_missing_file_does_not_disclose_the_resolved_path(self, tmp_path):
        """The path is resolved to an absolute one before the message is built.

        Which turns "the config you asked for is missing" into a disclosure of
        the server's directory layout. The name the caller referred to is the
        actionable half and stays; the resolved location moves to ``context``,
        which this type does not disclose.
        """
        from dataknobs_config.exceptions import FileNotFoundError as ConfigFileNotFound

        missing = tmp_path / "deploy" / "secrets" / "bots.yaml"

        with pytest.raises(ConfigFileNotFound) as excinfo:
            Config(str(missing))

        message = str(excinfo.value)
        assert "bots.yaml" in message
        assert str(tmp_path) not in message
        assert excinfo.value.context["path"] == str(missing.resolve())

    def test_a_parse_failure_does_not_echo_the_file_contents(self, tmp_path):
        """A YAML syntax error quotes the line it choked on.

        That line is config, which is where credentials live. An unterminated
        quote on an ``api_key`` puts the key itself in the parser's message,
        and relaying that message put it in a 422 response body.
        """
        from dataknobs_config.exceptions import ValidationError

        broken = tmp_path / "bots.yaml"
        broken.write_text('api_key: "sk-live-do-not-disclose\nother: 1\n')

        with pytest.raises(ValidationError) as excinfo:
            Config(str(broken))

        message = str(excinfo.value)
        assert "sk-live-do-not-disclose" not in message
        assert "bots.yaml" in message


class TestObjectConstruction:
    """Test object construction from configurations."""

    def test_build_with_class(self):
        """Test building object with class attribute."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "class": "dataknobs_config.examples.Database",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        obj = config.build_object("xref:database[primary]")

        assert isinstance(obj, Database)
        assert obj.host == "localhost"
        assert obj.port == 5432

    def test_build_with_configurable_base(self):
        """Test building object that inherits from ConfigurableBase."""
        config = Config(
            {
                "database": [
                    {
                        "name": "test",
                        "class": "dataknobs_config.examples.Database",
                        "host": "testhost",
                        "port": 3306,
                        "extra_param": "value",
                    }
                ]
            }
        )

        obj = config.build_object("xref:database[test]")

        assert isinstance(obj, Database)
        assert obj.host == "testhost"
        assert obj.extra["extra_param"] == "value"

    def test_build_with_factory(self):
        """Test building object with factory attribute."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "dataknobs_config.examples.DatabaseFactory",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        obj = config.build_object("xref:database[primary]")

        assert isinstance(obj, Database)
        assert obj.host == "localhost"
        assert obj.port == 5432
        assert obj.pool_size == 10  # Added by factory

    def test_build_with_callable_factory(self):
        """Test building with callable factory."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "factory": "dataknobs_config.examples.CacheFactory",
                        "host": "localhost",
                        "port": 6379,
                        "ttl": 7200,
                    }
                ]
            }
        )

        obj = config.build_object("xref:cache[redis]")

        assert isinstance(obj, Cache)
        assert obj.host == "localhost"
        assert obj.ttl == 7200

    def test_build_without_class_or_factory(self):
        """Test that building without class or factory raises error."""
        config = Config({"database": [{"name": "test", "host": "localhost"}]})

        with pytest.raises(ConfigError):
            config.build_object("xref:database[test]")

    def test_invalid_class_path(self):
        """Test that invalid class path raises error."""
        config = Config({"database": [{"name": "test", "class": "nonexistent.module.Class"}]})

        with pytest.raises(ConfigError):
            config.build_object("xref:database[test]")

    def test_build_with_kwargs(self):
        """Test building with additional kwargs."""
        config = Config(
            {
                "database": [
                    {
                        "name": "test",
                        "class": "dataknobs_config.examples.Database",
                        "host": "localhost",
                    }
                ]
            }
        )

        obj = config.build_object("xref:database[test]", port=3306, extra_param="added")

        assert obj.host == "localhost"
        assert obj.port == 3306
        assert obj.extra["extra_param"] == "added"


class TestObjectCaching:
    """Test object caching functionality."""

    def test_cache_enabled(self):
        """Test that objects are cached by default."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "class": "dataknobs_config.examples.Cache",
                        "host": "localhost",
                        "port": 6379,
                    }
                ]
            }
        )

        obj1 = config.build_object("xref:cache[redis]")
        obj2 = config.build_object("xref:cache[redis]")

        assert obj1 is obj2  # Same object instance

    def test_cache_disabled(self):
        """Test building without caching."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "class": "dataknobs_config.examples.Cache",
                        "host": "localhost",
                        "port": 6379,
                    }
                ]
            }
        )

        obj1 = config.build_object("xref:cache[redis]", cache=False)
        obj2 = config.build_object("xref:cache[redis]", cache=False)

        assert obj1 is not obj2  # Different instances
        assert obj1.host == obj2.host  # But same config

    def test_clear_cache(self):
        """Test clearing object cache."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "class": "dataknobs_config.examples.Cache",
                        "host": "localhost",
                        "port": 6379,
                    }
                ]
            }
        )

        obj1 = config.build_object("xref:cache[redis]")
        config.clear_object_cache("xref:cache[redis]")
        obj2 = config.build_object("xref:cache[redis]")

        assert obj1 is not obj2  # Different instances after cache clear

    def test_clear_all_cache(self):
        """Test clearing all cached objects."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis1",
                        "class": "dataknobs_config.examples.Cache",
                        "host": "host1",
                        "port": 6379,
                    },
                    {
                        "name": "redis2",
                        "class": "dataknobs_config.examples.Cache",
                        "host": "host2",
                        "port": 6380,
                    },
                ]
            }
        )

        obj1a = config.build_object("xref:cache[redis1]")
        obj2a = config.build_object("xref:cache[redis2]")

        config.clear_object_cache()  # Clear all

        obj1b = config.build_object("xref:cache[redis1]")
        obj2b = config.build_object("xref:cache[redis2]")

        assert obj1a is not obj1b
        assert obj2a is not obj2b


class TestAsyncObjectConstruction:
    """``build_object_async`` prefers async entry points."""

    @pytest.mark.asyncio
    async def test_prefers_from_config_async(self) -> None:
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        obj = await config.build_object_async("xref:widget[w]")
        assert isinstance(obj, AsyncWidget)
        # ``_ainit`` ran — only the async path sets this.
        assert obj.warmed is True

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_from_config(self) -> None:
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.SyncWidget",
                        "extra_field": "ignored-by-from_config",
                    }
                ]
            }
        )
        obj = await config.build_object_async("xref:widget[w]")
        assert isinstance(obj, SyncWidget)
        assert obj.name == "sync"

    @pytest.mark.asyncio
    async def test_falls_back_to_direct_instantiation(self) -> None:
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.PlainWidget",
                    }
                ]
            }
        )
        obj = await config.build_object_async("xref:widget[w]")
        assert isinstance(obj, PlainWidget)
        assert obj.name == "plain"

    @pytest.mark.asyncio
    async def test_async_build_caches(self) -> None:
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        a = await config.build_object_async("xref:widget[w]")
        b = await config.build_object_async("xref:widget[w]")
        assert a is b

    def test_sync_build_object_does_not_run_ainit(self) -> None:
        """The sync path leaves the async-only flag unset."""
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        obj = config.build_object("xref:widget[w]")
        assert isinstance(obj, AsyncWidget)
        assert obj.warmed is False

    @pytest.mark.asyncio
    async def test_factory_path_runs_consumer_async_init(self) -> None:
        """A consumer referenced via ``factory:`` runs ``_ainit`` (async).

        A genuine factory's ``create``/``build``/``__call__`` is preferred;
        the consumer protocol (``from_config_async``) is the last-resort
        path, so a consumer class used as a factory builds correctly
        instead of erroring.
        """
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "factory": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        obj = await config.build_object_async("xref:widget[w]")
        assert isinstance(obj, AsyncWidget)
        assert obj.warmed is True

    def test_factory_path_builds_consumer_sync(self) -> None:
        """Sync ``factory:`` path builds a consumer via ``from_config``."""
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "factory": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        obj = config.build_object("xref:widget[w]")
        assert isinstance(obj, AsyncWidget)
        assert obj.warmed is False

"""What a registry can be asked about itself, and what it says when it guesses.

Four gaps are pinned here, each written to fail before the code answering it
existed.

1. **A resolved key carried no record of where it came from.**
   ``_resolve_factory`` reads ``config[config_key]`` and falls back to
   ``config_key_default``, then returns the factory -- by which point an
   absent key and an explicit one naming the same value are the same value.
   Nothing was logged at any level. Ten registries share the construct and
   three of them default to ``memory``, so ``create_lock({})`` returned an
   in-process lock that every process believed it held alone.

2. **Aliases were counted as separate plugins.** ``list_keys()`` reports
   every accepted spelling, which is right for the lookup it serves and
   wrong for a list shown to someone choosing between plugins.

3. **Metadata was reachable only by the canonical spelling**, and only
   shallowly copied -- so a nested value handed to a caller was the live
   registry dict, and mutating it changed what every later caller saw.

4. **A plugin whose optional dependency is missing was indistinguishable
   from a typo.** Both were simply unregistered, so nothing could say what
   to install: the one question whose answer is only needed while the
   answer is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_common import NotFoundError, PluginRegistry


REGISTRY_LOGGER = "dataknobs_common.registry"


class _Plugin:
    """A factory result with a ``from_config`` classmethod, as a backend has."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> _Plugin:
        return cls(config)


class _OtherPlugin(_Plugin):
    """A second, distinct factory.

    Grouping is by factory identity, so a registration meant to form its own
    group has to be a different object -- registering ``_Plugin`` twice
    would make one group under two names, which is the case
    :meth:`list_canonical_keys` is designed to collapse.
    """


def _registry_with_default() -> PluginRegistry[Any]:
    registry: PluginRegistry[Any] = PluginRegistry(
        "defaulting",
        config_key="backend",
        config_key_default="memory",
        not_found_kind="test backend",
        not_found_exception=ValueError,
    )
    registry.register("memory", _Plugin)
    registry.register("postgres", _Plugin)
    return registry


def _registry_with_aliases() -> PluginRegistry[Any]:
    """One factory under three spellings, the canonical one holding metadata.

    The registration convention this package uses everywhere: the canonical
    key takes the metadata and the aliases take none.
    """
    registry: PluginRegistry[Any] = PluginRegistry("aliased")
    registry.register(
        "postgres",
        _Plugin,
        metadata={
            "description": "PostgreSQL",
            "requires_install": "pip install dataknobs-data[postgres]",
            "config_options": {"host": "Server host"},
        },
    )
    registry.register("postgresql", _Plugin)
    registry.register("pg", _Plugin)
    registry.register("memory", _OtherPlugin, metadata={"description": "Memory"})
    return registry


# ---------------------------------------------------------------------------
# 1. A defaulted key is reported as a default
# ---------------------------------------------------------------------------


class TestADefaultedKeyIsReported:
    """An absent routing key and an explicit one are different events."""

    def test_an_absent_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = _registry_with_default()

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            registry.create(config={})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, (
            f"expected one WARNING, got {[r.message for r in caplog.records]}"
        )
        assert "memory" in warnings[0].getMessage()
        assert "backend" in warnings[0].getMessage()

    def test_an_explicit_key_naming_the_same_value_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The whole point: same resolved value, different provenance."""
        registry = _registry_with_default()

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            registry.create(config={"backend": "memory"})

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_an_explicit_key_argument_is_not_a_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = _registry_with_default()

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            registry.create("memory", config={})

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_a_registry_can_say_what_its_default_costs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A generic sentence cannot say why the fallback matters here."""
        registry: PluginRegistry[Any] = PluginRegistry(
            "costly",
            config_key="backend",
            config_key_default="memory",
            default_warning=(
                "No '%(config_key)s' key in this lock config; falling back to "
                "'%(key)s', which is in-process and coordinates nothing "
                "between processes."
            ),
        )
        registry.register("memory", _Plugin)

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            registry.create(config={})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "coordinates nothing between processes" in warnings[0].getMessage()
        assert "'memory'" in warnings[0].getMessage()

    def test_create_async_reports_the_default_too(self, caplog: pytest.LogCaptureFixture) -> None:
        """The sync and async paths share ``_resolve_factory``; so must this."""
        import asyncio

        registry = _registry_with_default()

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            asyncio.run(registry.create_async(config={}))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "memory" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# 2. Canonical keys
# ---------------------------------------------------------------------------


class TestCanonicalKeys:
    """A list of plugins rather than a list of spellings."""

    def test_aliases_collapse_to_the_key_carrying_metadata(self) -> None:
        registry = _registry_with_aliases()

        assert registry.list_canonical_keys() == ["memory", "postgres"]

    def test_list_keys_still_reports_every_spelling(self) -> None:
        """The lookup surface is unchanged; only the reporting one is new."""
        registry = _registry_with_aliases()

        assert sorted(registry.list_keys()) == ["memory", "pg", "postgres", "postgresql"]

    def test_a_group_with_no_metadata_still_yields_one_name(self) -> None:
        """A custom registration need not follow the metadata convention."""
        registry: PluginRegistry[Any] = PluginRegistry("undescribed")
        registry.register("primary", _Plugin)
        registry.register("alias", _Plugin)

        assert registry.list_canonical_keys() == ["primary"]


# ---------------------------------------------------------------------------
# 3. Metadata by any spelling, and isolated from the registry
# ---------------------------------------------------------------------------


class TestMetadataAccess:
    def test_an_alias_answers_for_its_canonical_key(self) -> None:
        registry = _registry_with_aliases()

        assert registry.get_metadata("pg", follow_alias=True) == registry.get_metadata("postgres")

    def test_an_alias_without_following_still_answers_empty(self) -> None:
        """The historical shape is preserved; following is opt-in."""
        registry = _registry_with_aliases()

        assert registry.get_metadata("pg") == {}

    def test_a_nested_value_is_not_the_live_registry_dict(self) -> None:
        registry = _registry_with_aliases()

        handed_out = registry.get_metadata("postgres")
        handed_out["config_options"]["host"] = "mutated"

        assert registry.get_metadata("postgres")["config_options"]["host"] == "Server host"


# ---------------------------------------------------------------------------
# 4. Known, but not creatable here
# ---------------------------------------------------------------------------


class TestKnownButUnavailable:
    """The question "what would I install?" is only asked while uninstalled."""

    def _registry(self) -> PluginRegistry[Any]:
        registry: PluginRegistry[Any] = PluginRegistry(
            "partial",
            not_found_kind="test backend",
            not_found_exception=ValueError,
        )
        registry.register("memory", _Plugin, metadata={"description": "Memory"})
        registry.declare_unavailable(
            "postgres",
            metadata={
                "description": "PostgreSQL",
                "requires_install": "pip install dataknobs-data[postgres]",
            },
            reason="psycopg2 is not installed",
        )
        return registry

    def test_it_is_not_registered(self) -> None:
        assert self._registry().is_registered("postgres") is False

    def test_it_still_says_what_to_install(self) -> None:
        registry = self._registry()

        info = registry.get_metadata("postgres")

        assert info["requires_install"] == "pip install dataknobs-data[postgres]"

    def test_creating_it_names_the_reason_not_a_typo(self) -> None:
        registry = self._registry()

        with pytest.raises(ValueError) as excinfo:
            registry.create("postgres", config={})

        message = str(excinfo.value)
        assert "psycopg2 is not installed" in message
        assert "Unknown" not in message

    def test_an_actual_typo_still_reads_as_one(self) -> None:
        registry = self._registry()

        with pytest.raises(ValueError) as excinfo:
            registry.create("postgrez", config={})

        assert "Unknown test backend" in str(excinfo.value)

    def test_list_keys_excludes_it_and_known_keys_includes_it(self) -> None:
        registry = self._registry()

        assert registry.list_keys() == ["memory"]
        assert registry.list_known_keys() == ["memory", "postgres"]

    def test_registering_it_later_clears_the_mark(self) -> None:
        """Order of registration must not decide the outcome."""
        registry = self._registry()

        registry.register("postgres", _Plugin, metadata={"description": "PostgreSQL"})

        assert registry.is_registered("postgres") is True
        assert registry.list_known_keys() == ["memory", "postgres"]
        assert isinstance(registry.create("postgres", config={}), _Plugin)

    def test_declaring_over_a_registration_removes_it(self) -> None:
        registry = self._registry()

        registry.declare_unavailable("memory", metadata={}, reason="withdrawn")

        assert registry.is_registered("memory") is False
        with pytest.raises(ValueError, match="withdrawn"):
            registry.create("memory", config={})

    def test_the_default_not_found_exception_shape_is_kept(self) -> None:
        """A registry that did not opt into ValueError keeps NotFoundError."""
        registry: PluginRegistry[Any] = PluginRegistry("plain")
        registry.declare_unavailable("gone", metadata={}, reason="not installed")

        with pytest.raises(NotFoundError, match="not installed"):
            registry.create("gone", config={})


# ---------------------------------------------------------------------------
# The three shipped registries whose default has consequences
# ---------------------------------------------------------------------------


class TestTheDefaultsThatCostSomething:
    """A generic sentence cannot say what these particular defaults lose.

    Each of these three defaults to ``memory``, and in each case the
    in-process variant is not a smaller version of the real thing but a
    different thing that fails silently: a lock every process holds at once,
    a rate limit enforced N times over, a bus whose events reach nobody.
    """

    def test_an_empty_lock_config_says_what_in_process_means(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dataknobs_common.locks import create_lock

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            create_lock({})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "all of them believe they hold it" in warnings[0].getMessage()

    def test_an_explicit_memory_lock_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_common.locks import create_lock

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            create_lock({"backend": "memory"})

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_an_empty_event_bus_config_says_the_events_go_nowhere(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dataknobs_common.events import create_event_bus

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            create_event_bus({})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "never reaches another process" in warnings[0].getMessage()

    def test_a_rate_limiter_config_naming_no_backend_says_the_rate_multiplies(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rates are required, so the reachable case is rates without a backend."""
        from dataknobs_common.ratelimit import create_rate_limiter

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            create_rate_limiter({"rates": [{"limit": 60, "interval": 60}]})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "N times the configured rate" in warnings[0].getMessage()

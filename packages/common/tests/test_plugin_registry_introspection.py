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
    """A registry whose default costs something, so it says so.

    Passing ``default_warning`` is what makes the fallback a WARNING;
    a registry that declares none is reporting a default it does not
    consider consequential, and :class:`TestADefaultNobodyCalledCostly`
    covers that half.
    """
    registry: PluginRegistry[Any] = PluginRegistry(
        "defaulting",
        config_key="backend",
        config_key_default="memory",
        not_found_kind="test backend",
        not_found_exception=ValueError,
        default_warning=(
            "No '%(config_key)s' key in this %(registry)s config; falling back "
            "to '%(key)s', which keeps nothing once this process exits."
        ),
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


class TestADefaultNobodyCalledCostly:
    """A registry that declares no warning text does not raise its voice.

    ``default_warning`` is a registry saying what its own fallback costs.
    Most defaults cost nothing worth interrupting anyone for -- ``simple``
    reasoning, ``buffer`` memory, ``rag`` knowledge, ``null`` partitioning
    -- and the documented config for each of those omits the key on
    purpose, because the default is the recommended answer.

    Reporting those at WARNING inverts the level's meaning. It fires on
    configurations this repository's own documentation prints, it fires
    per turn where the resolve is per turn, and it buries the three
    registries whose default really does cost something: a lock that
    coordinates nothing, a bus whose events reach nobody, a limiter that
    multiplies the rate by the number of processes.

    So the provenance is still recorded -- an absent key and an explicit
    one are still different events, which is the whole point -- but at
    DEBUG until a registry claims otherwise.
    """

    @staticmethod
    def _quiet() -> PluginRegistry[Any]:
        registry: PluginRegistry[Any] = PluginRegistry(
            "quiet",
            config_key="backend",
            config_key_default="memory",
        )
        registry.register("memory", _Plugin)
        return registry

    def test_it_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = self._quiet()

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            registry.create(config={})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [], (
            "a registry that declared no consequence should not report one: "
            f"{[r.getMessage() for r in warnings]}"
        )

    def test_the_provenance_is_still_recorded_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Quieter, not silent -- the distinction is still findable."""
        registry = self._quiet()

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            registry.create(config={})

        debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("memory" in r.getMessage() and "backend" in r.getMessage() for r in debug), (
            f"expected the fallback recorded at DEBUG, got {[r.getMessage() for r in debug]}"
        )

    def test_create_async_is_quiet_too(self, caplog: pytest.LogCaptureFixture) -> None:
        import asyncio

        registry = self._quiet()

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            asyncio.run(registry.create_async(config={}))

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


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

    def test_a_failed_lazy_init_leaves_no_mark_behind(self) -> None:
        """The rollback has to cover every dict a populator can write to.

        ``_ensure_initialized`` snapshots the registry before running the
        populator so a partial failure cannot leave half a registration
        behind -- ``_initialized`` is reset, the next access re-runs from
        the top, and anything left over from the abandoned run is now
        indistinguishable from something the new run put there.

        ``_unavailable`` was added to the class without being added to that
        snapshot. So a populator that declared a backend unavailable and
        then failed left the mark, while the metadata that came with it was
        rolled back -- leaving a key ``list_known_keys`` reports,
        ``get_metadata`` knows nothing about, and ``create`` refuses as
        "not available here" with a reason nobody can look up.

        Reproduced with a populator that fails once and then succeeds along
        a path that never mentions ``postgres``, which is the shape that
        makes the leftover visible: a driver probe is an environment read,
        and an environment can differ between two runs.
        """
        attempts: list[int] = []

        def populate(registry: PluginRegistry[Any]) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                registry.declare_unavailable(
                    "postgres",
                    metadata={"description": "PostgreSQL"},
                    reason="psycopg2 is not installed",
                )
                raise RuntimeError("the populator's own failure")
            registry.register("memory", _Plugin, metadata={"description": "Memory"})

        registry: PluginRegistry[Any] = PluginRegistry("lazy", on_first_access=populate)

        with pytest.raises(RuntimeError, match="the populator's own failure"):
            registry.list_known_keys()

        assert registry.list_known_keys() == ["memory"], (
            "the abandoned run's unavailable mark survived the rollback"
        )
        assert attempts == [1, 1]

    def test_an_alias_of_it_answers_for_the_canonical_key(self) -> None:
        """``follow_alias`` was dead for anything not creatable.

        It groups by shared factory, and an unavailable key has no factory
        -- so the one state in which ``requires_install`` is ever read was
        the one state in which an alias could not reach it. The only
        consumer of this API worked around it by writing the same metadata
        dict under every spelling, which is a fix in the wrong place: it
        makes alias metadata asymmetric between the two states, and every
        later consumer has to rediscover the need for it.
        """
        registry: PluginRegistry[Any] = PluginRegistry("aliased-unavailable")
        registry.declare_unavailable(
            "postgres",
            metadata={"requires_install": "pip install dataknobs-data[postgres]"},
            reason="psycopg2 is not installed",
            aliases=("pg", "postgresql"),
        )

        for spelling in ("pg", "postgresql"):
            assert registry.get_metadata(spelling, follow_alias=True) == registry.get_metadata(
                "postgres", follow_alias=True
            ), spelling

    def test_an_alias_of_it_is_not_creatable_either(self) -> None:
        registry: PluginRegistry[Any] = PluginRegistry(
            "aliased-unavailable", not_found_exception=ValueError
        )
        registry.declare_unavailable(
            "postgres",
            metadata={},
            reason="psycopg2 is not installed",
            aliases=("pg",),
        )

        assert registry.is_registered("pg") is False
        with pytest.raises(ValueError, match="psycopg2 is not installed"):
            registry.create("pg", config={})

    def test_declaring_it_unavailable_withdraws_its_aliases_too(self) -> None:
        """An alias registered earlier must not outlive the withdrawal."""
        registry: PluginRegistry[Any] = PluginRegistry("withdrawing")
        registry.register("postgres", _Plugin)
        registry.register("pg", _Plugin)

        registry.declare_unavailable("postgres", metadata={}, reason="withdrawn", aliases=("pg",))

        assert registry.is_registered("pg") is False

    def test_it_can_be_unregistered(self) -> None:
        """``unregister`` looked only at ``_factories``.

        So a declared-unavailable key could not be removed at all: the
        method raised ``NotFoundError`` for a key the registry was
        simultaneously reporting through ``list_known_keys``, and
        ``register`` was the only way to clear a mark.
        """
        registry: PluginRegistry[Any] = PluginRegistry("removable")
        registry.declare_unavailable(
            "postgres", metadata={"description": "PostgreSQL"}, reason="absent"
        )

        registry.unregister("postgres")

        assert registry.list_known_keys() == []
        assert registry.get_metadata("postgres") == {}

    def test_unregistering_something_it_never_knew_still_raises(self) -> None:
        registry: PluginRegistry[Any] = PluginRegistry("removable")

        with pytest.raises(NotFoundError):
            registry.unregister("never-heard-of-it")

    def test_a_leftover_mark_cannot_outlive_its_metadata(self) -> None:
        """The inconsistency the rollback exists to prevent, stated directly.

        Every key this registry admits to knowing has to be one it can say
        something about -- either a factory or a reason.
        """
        attempts: list[int] = []

        def populate(registry: PluginRegistry[Any]) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                registry.declare_unavailable(
                    "postgres", metadata={"description": "PostgreSQL"}, reason="absent"
                )
                raise RuntimeError("boom")
            registry.register("memory", _Plugin, metadata={"description": "Memory"})

        registry: PluginRegistry[Any] = PluginRegistry("lazy", on_first_access=populate)

        with pytest.raises(RuntimeError, match="boom"):
            registry.list_known_keys()

        for key in registry.list_known_keys():
            assert registry.is_registered(key) or registry.get_metadata(key), (
                f"{key!r} is reported as known but has neither a factory nor metadata"
            )


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


class TestAPopulatorThatAbandonsItsRun:
    """Rollback covers *state*, not the dicts someone remembered.

    ``_ensure_initialized`` snapshots so a populator that fails partway
    leaves nothing behind for the retry to trip over. The list it snapshots
    was written as "every dict", and has twice been wrong: ``_unavailable``
    was added to the class without being added to it, and
    ``_default_factory`` is not a dict at all, so a populator calling the
    public :meth:`set_default_factory` escaped the rollback entirely.
    """

    def test_a_default_factory_set_before_the_failure_is_rolled_back(self) -> None:
        def populate(registry: PluginRegistry[Any]) -> None:
            registry.set_default_factory(_Plugin)
            raise RuntimeError("populator failed")

        registry = PluginRegistry[Any]("t", on_first_access=populate)

        with pytest.raises(RuntimeError, match="populator failed"):
            registry.list_keys()

        # Observable rather than asserted on the attribute: with the
        # default left behind, a second run's create() for an unregistered
        # key silently succeeds off the abandoned run's factory.
        def populate_again(registry: PluginRegistry[Any]) -> None:
            registry.register("real", _Plugin)

        registry._initializer = populate_again
        with pytest.raises(NotFoundError):
            registry.create("never-registered")

    def test_a_registered_key_from_the_failed_run_is_rolled_back(self) -> None:
        """The case the snapshot was written for, kept covered."""

        def populate(registry: PluginRegistry[Any]) -> None:
            registry.register("half", _Plugin)
            raise RuntimeError("populator failed")

        registry = PluginRegistry[Any]("t", on_first_access=populate)
        with pytest.raises(RuntimeError):
            registry.list_keys()

        def populate_again(registry: PluginRegistry[Any]) -> None:
            registry.register("half", _Plugin)

        registry._initializer = populate_again
        assert registry.list_keys() == ["half"]


class TestConsumerSuppliedWarningText:
    """``default_warning`` is interpolated by ``logging``, against a dict.

    A literal ``%`` in it is a format spec to the handler, so the failure
    lands inside ``logging`` at the first fallback -- on the branch nobody
    exercises, long after the registry was built, in a traceback that names
    neither the registry nor the text. Proved where the text is authored
    instead.
    """

    def test_a_literal_percent_is_refused_where_it_is_written(self) -> None:
        with pytest.raises(ValueError, match="must be written"):
            PluginRegistry[Any](
                "t",
                config_key="kind",
                config_key_default="d",
                default_warning="the default costs 50% of throughput",
            )

    def test_the_error_names_the_registry_and_the_placeholders(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            PluginRegistry[Any](
                "locks",
                config_key="kind",
                config_key_default="d",
                default_warning="100% in-process",
            )
        message = str(excinfo.value)
        assert "'locks'" in message
        assert "%(config_key)s" in message

    def test_an_escaped_percent_is_accepted_and_renders(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = PluginRegistry[Any](
            "t",
            config_key="kind",
            config_key_default="d",
            default_warning="%(key)s coordinates 0%% of processes",
        )
        registry.register("d", _Plugin)

        with caplog.at_level(logging.WARNING, logger=REGISTRY_LOGGER):
            registry.create(config={})

        assert "d coordinates 0% of processes" in caplog.records[0].getMessage()


class TestAskingWhetherAKeyIsKnown:
    """``is_known`` asks the registry; truthy metadata only guesses.

    Every caller distinguishing "misspelled" from "not installed here" was
    reaching for :meth:`get_metadata` and testing it for truth, because
    that was the only public surface that answered at all. It answers a
    different question: :meth:`declare_unavailable` accepts ``metadata=None``,
    and a plugin declared without any then reads as unknown -- reported as a
    typo, which is the one thing the declaration existed to prevent.
    """

    def test_a_key_declared_without_metadata_is_still_known(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable("acme", reason="acme-sdk is not installed")

        assert registry.get_metadata("acme") == {}, "precondition: no metadata to read"
        assert registry.is_known("acme") is True
        assert registry.is_registered("acme") is False

    def test_a_name_nobody_declared_is_not_known(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable("acme", reason="acme-sdk is not installed")

        assert registry.is_known("acme-typo") is False

    def test_a_creatable_key_is_known_too(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.register("real", _Plugin)

        assert registry.is_known("real") is True

    def test_an_alias_of_a_declared_key_is_known(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable(
            "postgres", reason="psycopg2 is not installed", aliases=("pg",)
        )

        assert registry.is_known("pg") is True


class TestReadingTheClassOfAPluginThatCannotBeBuilt:
    """``load_declared_type`` serves the caller that wants to *read*.

    A plugin whose optional driver is missing has no factory -- and must
    not, since ``is_registered`` means "creatable". Its class may still be
    importable, though, and something on it (a typed config schema, here)
    may be exactly what a caller needs. Without this the caller keeps its
    own second key-to-class table, which is the drift the registry exists
    to prevent.
    """

    def test_it_returns_the_class_when_the_module_imports(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable(
            "acme", reason="acme-sdk is not installed", type_loader=lambda: _Plugin
        )

        assert registry.load_declared_type("acme") is _Plugin
        assert registry.get_factory("acme") is None, "still not creatable"

    def test_it_returns_none_when_the_module_cannot_import(self) -> None:
        """The other idiom: a driver imported at module top level."""

        def explode() -> type:
            raise ImportError("No module named 'acme_sdk'")

        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable(
            "acme", reason="acme-sdk is not installed", type_loader=explode
        )

        assert registry.load_declared_type("acme") is None

    def test_it_returns_none_when_no_loader_was_declared(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable("acme", reason="acme-sdk is not installed")

        assert registry.load_declared_type("acme") is None

    def test_an_alias_reaches_the_same_class(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable(
            "postgres",
            reason="psycopg2 is not installed",
            aliases=("pg",),
            type_loader=lambda: _Plugin,
        )

        assert registry.load_declared_type("pg") is _Plugin


class TestUnregisteringAKeyOtherSpellingsDependOn:
    """An unavailable alias answers through the key being dropped.

    It carries no metadata of its own -- ``describes_key`` points at the
    canonical key, whose metadata ``unregister`` removes. Dropping only the
    one spelling therefore strands the aliases: still reported by
    :meth:`list_known_keys`, and now answering ``{}`` to the
    ``requires_install`` question that is the whole reason a withdrawn
    plugin stays visible.
    """

    def test_its_aliases_go_with_it(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable(
            "postgres",
            metadata={"requires_install": "pip install x[postgres]"},
            reason="psycopg2 is not installed",
            aliases=("pg", "postgresql"),
        )

        registry.unregister("postgres")

        assert registry.list_known_keys() == []
        assert registry.is_known("pg") is False

    def test_an_alias_dropped_on_its_own_leaves_the_canonical_key(self) -> None:
        registry = PluginRegistry[Any]("t")
        registry.declare_unavailable(
            "postgres",
            metadata={"requires_install": "pip install x[postgres]"},
            reason="psycopg2 is not installed",
            aliases=("pg",),
        )

        registry.unregister("pg")

        assert registry.is_known("pg") is False
        assert registry.is_known("postgres") is True
        assert registry.get_metadata("postgres")["requires_install"]

    def test_unregistering_a_creatable_key_leaves_no_marks(self) -> None:
        """The canonical key's own metadata still goes, as it always did."""
        registry = PluginRegistry[Any]("t")
        registry.register("real", _Plugin, metadata={"a": 1})

        registry.unregister("real")

        assert registry.list_known_keys() == []
        assert registry.get_metadata("real") == {}

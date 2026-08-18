"""Backend selection: what a factory did, and what it can be asked.

Three defects are pinned here, each written to fail before the code that
answers it existed:

1. **A config with no ``backend`` key was indistinguishable from one asking
   for memory.** Both produced the same INFO line naming the same backend --
   or, on the async factory, no line at all. The two are not the same event:
   one was asked for, and the other is what is left when a config arrived
   empty. An empty config handed to a factory does not fail; it produces an
   in-process store that answers every query with zero results and loses
   everything on restart.

2. **The documented backend-information API was two-thirds absent.** Three
   documents describe ``get_available_backends()`` and
   ``is_backend_available()`` as factory methods. Neither existed, so a
   reader following any of them got an ``AttributeError``.

3. **``get_backend_info`` was on two of the three factories.** The async one
   had none, which is the same surface gap discovered from the other side.

The last group is not a defect but the guard that keeps the first two from
drifting apart: the list a factory reports and the list its unknown-backend
error prints have to be the same list.

The provenance group asserts on a set of candidate loggers rather than on
one. Which of them currently emits is an implementation detail these tests
should survive; that a factory says out loud when nothing asked for a
backend is not. The set is not "any logger" -- moving the decision outside
it would fail, which is the point at which the choice has become a
different design rather than the same one relocated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from dataknobs_data import (
    DEFAULT_BACKEND,
    AsyncDatabaseFactory,
    DatabaseFactory,
    is_default_backend,
)
from dataknobs_data.backends import async_backends, sync_backends
from dataknobs_data.vector.stores import vector_backends
from dataknobs_data.vector.stores.factory import VectorStoreFactory


#: The default this package has always applied when a config names no
#: backend. Spelled out rather than imported, so the provenance tests below
#: pin the documented value rather than agreeing with whatever the code says.
MEMORY = "memory"

#: Every module that could reasonably log the backend decision -- the three
#: factories that made it inline, and the module they now share. Filtering to
#: these keeps an unrelated ``dataknobs_data`` log line out of the counts
#: without the test having to know which one currently emits.
_SELECTION_LOGGERS = frozenset(
    {
        "dataknobs_data.factory",
        "dataknobs_data.vector.stores.factory",
        "dataknobs_data.backend_selection",
    }
)

#: ``(label, factory, registry)`` for the three factories that read a
#: ``backend`` key. The bug being pinned existed in three places because the
#: block did, so every case below runs against all three.
FACTORIES: list[tuple[str, Any, Any]] = [
    ("database", DatabaseFactory(), sync_backends),
    ("async database", AsyncDatabaseFactory(), async_backends),
    ("vector store", VectorStoreFactory(), vector_backends),
]

_IDS = [label for label, _, _ in FACTORIES]


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    """Records at exactly ``level`` from whichever module decided the backend."""
    return [
        record
        for record in caplog.records
        if record.levelno == level and record.name in _SELECTION_LOGGERS
    ]


@pytest.fixture
def selection_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture every level from all modules that may log the decision."""
    for name in _SELECTION_LOGGERS:
        caplog.set_level(logging.DEBUG, logger=name)
    return caplog


class TestProvenanceIsVisibleInTheLog:
    """An absent ``backend`` key and an explicit one are different events."""

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_absent_backend_key_warns(
        self,
        label: str,
        factory: Any,
        _registry: Any,
        selection_log: pytest.LogCaptureFixture,
    ) -> None:
        """No ``backend`` key at all is a WARNING, naming what it fell back to."""
        factory.create()

        warnings = _records(selection_log, logging.WARNING)
        assert len(warnings) == 1, f"{label}: expected one WARNING, got {warnings}"
        message = warnings[0].getMessage()
        assert "backend" in message
        assert MEMORY in message

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_explicit_memory_does_not_warn(
        self,
        label: str,
        factory: Any,
        _registry: Any,
        selection_log: pytest.LogCaptureFixture,
    ) -> None:
        """``backend: memory`` was asked for, so it is INFO -- not WARNING.

        This is the half that makes the WARNING mean something. A factory
        warning on every memory store would be warning about a choice the
        config made deliberately.
        """
        factory.create(backend=MEMORY)

        assert _records(selection_log, logging.WARNING) == []
        infos = _records(selection_log, logging.INFO)
        assert len(infos) == 1, f"{label}: expected one INFO, got {infos}"
        assert MEMORY in infos[0].getMessage()

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_both_paths_build_the_same_backend(
        self, label: str, factory: Any, _registry: Any
    ) -> None:
        """Only the log level changes. The object built does not."""
        assert type(factory.create()) is type(factory.create(backend=MEMORY))

    def test_an_explicit_non_default_backend_does_not_warn(
        self, selection_log: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """A backend that is neither absent nor the default is ordinary.

        ``path`` is never opened -- construction is disk-free and this
        asserts on the log, not on storage -- but it names a real temporary
        file rather than ``:memory:``, which means nothing to the file
        backend and would become a file of that literal name in the cwd if
        construction ever did touch the disk.
        """
        DatabaseFactory().create(backend="file", path=str(tmp_path / "records.json"))

        assert _records(selection_log, logging.WARNING) == []

    def test_an_alias_for_the_default_does_not_warn(
        self, selection_log: pytest.LogCaptureFixture
    ) -> None:
        """``mem`` is a spelling of a request, not the absence of one."""
        DatabaseFactory().create(backend="mem")

        assert _records(selection_log, logging.WARNING) == []


class TestUnknownBackendMessages:
    """The three lead sentences differ deliberately, and stay differing."""

    def test_sync_message_is_unchanged(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend type: invalid"):
            DatabaseFactory().create(backend="invalid")

    def test_vector_message_is_unchanged(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend type: invalid"):
            VectorStoreFactory().create(backend="invalid")

    def test_async_message_is_unchanged(self) -> None:
        """The async factory says something different, and it means it.

        An unrecognised key there usually means the backend exists but has
        no async variant, which is a different thing to tell a reader than
        "you typed it wrong".
        """
        with pytest.raises(ValueError, match="does not support async operations yet") as excinfo:
            AsyncDatabaseFactory().create(backend="invalid")
        assert "Available async backends:" in str(excinfo.value)

    def test_an_unknown_backend_does_not_warn_about_the_default(
        self, selection_log: pytest.LogCaptureFixture
    ) -> None:
        """The key was present. That it names nothing is the error's business."""
        with pytest.raises(ValueError):
            DatabaseFactory().create(backend="invalid")

        assert _records(selection_log, logging.WARNING) == []

    def test_a_present_but_empty_backend_key_is_an_error_not_a_default(
        self, selection_log: pytest.LogCaptureFixture
    ) -> None:
        """``backend: null`` named a backend and named nothing.

        Treating it as absent would silently produce the in-process default
        from a config that did try to choose, which is the failure this
        module exists to make visible. It raises instead -- previously an
        ``AttributeError`` from calling ``.lower()`` on ``None``.

        The message says which of the two happened. ``Unknown backend type:
        none`` reads as a backend literally named ``none`` and sends the
        reader looking for a spelling mistake.
        """
        with pytest.raises(ValueError, match="present but null"):
            DatabaseFactory().create(backend=None)

        assert _records(selection_log, logging.WARNING) == []

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "present but null"),
            ("", "present but empty"),
            ("   ", "present but empty"),
            (5, "must be a string"),
            (["memory"], "must be a string"),
        ],
        ids=["null", "empty", "whitespace", "int", "list"],
    )
    def test_a_discriminator_that_names_nothing_says_so_specifically(
        self, value: Any, expected: str, selection_log: pytest.LogCaptureFixture
    ) -> None:
        """Every one of these used to render into the unknown-name message."""
        with pytest.raises(ValueError, match=expected):
            DatabaseFactory().create(backend=value)

        assert _records(selection_log, logging.WARNING) == []

    def test_a_backend_name_is_stripped_before_lookup(self) -> None:
        """Whitespace around a name is a config artefact, not a new backend."""
        assert DatabaseFactory().create(backend="  MEMORY  ") is not None


class TestAskingWhetherAConfigWantsTheDefault:
    """For the caller that branches before any factory is involved.

    Four sites decided "in-process store, or build one?" by writing
    ``cfg.get("backend", "memory") == "memory"`` -- the default's name
    twice per site, eight copies of a constant that
    :data:`DEFAULT_BACKEND` already holds, in three different spellings
    that no longer had to agree.

    They are not laundering an absent key into an explicit choice, which is
    the defect the factory callers had: nothing is built through a factory
    on the branch they take, so there is no provenance to lose. They are
    asking one question in four ways.
    """

    def test_a_config_naming_nothing_wants_it(self) -> None:
        assert is_default_backend({}) is True

    def test_a_config_naming_it_wants_it(self) -> None:
        assert is_default_backend({"backend": DEFAULT_BACKEND}) is True

    def test_a_config_naming_something_else_does_not(self) -> None:
        assert is_default_backend({"backend": "postgres"}) is False

    @pytest.mark.parametrize("spelling", ["MEMORY", "  memory  ", "Memory"])
    def test_it_reads_the_name_the_way_the_factory_does(self, spelling: str) -> None:
        """A comparison against a literal missed every one of these."""
        assert is_default_backend({"backend": spelling}) is True

    @pytest.mark.parametrize("value", [None, "", "   ", 3])
    def test_a_present_but_unusable_key_is_an_error_not_the_default(self, value: object) -> None:
        """``.get(key, "memory")`` returned the value, so ``None`` compared
        unequal to ``"memory"`` and the config went to a factory that could
        not use it either.
        """
        with pytest.raises(ValueError):
            is_default_backend({"backend": value})


class TestABackendThatCannotBeBuiltFromAConfig:
    """The registries accept any callable; the factories require a class.

    ``PluginRegistry`` stores ``type[T] | Callable[..., T]``, and the
    database factories call ``from_config`` on whatever comes back. A
    consumer registering a plain function -- which ``register`` accepts,
    and which ``PluginRegistry.create`` supports -- got an
    ``AttributeError`` naming ``'function' object``, from inside the
    factory, with nothing pointing at the registration that caused it.
    """

    def _registry_with_a_bare_callable(self) -> Any:
        from dataknobs_common.registry import PluginRegistry

        registry: PluginRegistry[Any] = PluginRegistry("bare", canonicalize_keys=True)
        registry.register("plain", lambda config: {"built": config})
        return registry

    def test_it_says_what_is_wrong_with_the_registration(self) -> None:
        from dataknobs_data.backend_selection import build_backend, select_backend

        registry = self._registry_with_a_bare_callable()
        backend_class, backend_type, options = select_backend(
            {"backend": "plain"}, registry, kind="database"
        )

        with pytest.raises(ValueError) as excinfo:
            build_backend(backend_class, options, kind="database", backend_type=backend_type)

        message = str(excinfo.value)
        assert "plain" in message
        assert "from_config" in message

    def test_a_real_backend_class_still_builds(self) -> None:
        """The guard is a narrowing, not a new restriction."""
        assert DatabaseFactory().create(backend="memory") is not None


class TestTheValidationPathReadsTheConfigTheSameWay:
    """The resolver mirrors the factory, and used to only nearly.

    ``_resolve_vector_store_config_cls`` re-implemented "read the backend
    key" rather than sharing it, and the two disagreed on every input the
    happy path does not cover: the factory raised a ``ValueError`` naming
    the problem while the resolver called ``.lower()`` on the value and
    raised ``AttributeError`` -- a crash, from the path whose whole job is
    to report configuration problems.
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "present but null"),
            ("", "present but empty"),
            (7, "must be a string"),
        ],
        ids=["null", "empty", "int"],
    )
    def test_an_unusable_discriminator_is_a_configuration_error(
        self, value: Any, expected: str
    ) -> None:
        from dataknobs_common.exceptions import ConfigurationError

        from dataknobs_data.vector.stores import _resolve_vector_store_config_cls

        with pytest.raises(ConfigurationError, match=expected):
            _resolve_vector_store_config_cls({"backend": value})

    def test_it_still_reports_a_genuine_typo_as_unresolvable(self) -> None:
        """``None`` is the resolver's "no variant matches" answer."""
        from dataknobs_data.vector.stores import _resolve_vector_store_config_cls

        assert _resolve_vector_store_config_cls({"backend": "nope"}) is None

    def test_it_reads_an_absent_key_as_the_shared_default(self) -> None:
        from dataknobs_data.vector.stores import _resolve_vector_store_config_cls

        assert _resolve_vector_store_config_cls({"dimensions": 8}) is not None

    def test_it_normalises_case_and_whitespace_as_the_factory_does(self) -> None:
        from dataknobs_data.vector.stores import _resolve_vector_store_config_cls

        assert _resolve_vector_store_config_cls(
            {"backend": " MEMORY "}
        ) is _resolve_vector_store_config_cls({"backend": "memory"})


class TestBackendInformationApi:
    """The surface three documents describe, on all three factories."""

    @pytest.mark.parametrize(("label", "factory", "registry"), FACTORIES, ids=_IDS)
    def test_get_available_backends_agrees_with_the_registry(
        self, label: str, factory: Any, registry: Any
    ) -> None:
        from dataknobs_data.backend_selection import available_backends

        assert factory.get_available_backends() == available_backends(registry)

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_the_default_backend_is_reported_available(
        self, label: str, factory: Any, _registry: Any
    ) -> None:
        assert MEMORY in factory.get_available_backends()
        assert factory.is_backend_available(MEMORY) is True

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_is_backend_available_rejects_an_unregistered_name(
        self, label: str, factory: Any, _registry: Any
    ) -> None:
        assert factory.is_backend_available("no-such-backend") is False

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_is_backend_available_accepts_an_alias(
        self, label: str, factory: Any, _registry: Any
    ) -> None:
        """An alias is a usable spelling even though it is not listed.

        ``get_available_backends()`` reports canonical names, but
        ``create(backend=<alias>)`` works, so the availability question has
        to answer for the alias too or the two disagree about one backend.
        """
        alias = {
            "database": "mem",
            "async database": "mem",
            "vector store": "chromadb",
        }[label]
        assert factory.is_backend_available(alias) is True

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_get_backend_info_exists_on_every_factory(
        self, label: str, factory: Any, _registry: Any
    ) -> None:
        """Including the async one, which had no such method."""
        info = factory.get_backend_info(MEMORY)
        assert "description" in info
        assert "error" not in info

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_get_backend_info_reports_an_unknown_backend(
        self, label: str, factory: Any, _registry: Any
    ) -> None:
        info = factory.get_backend_info("no-such-backend")
        assert info["description"] == "Unknown backend"
        assert "no-such-backend" in info["error"]

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_the_three_accessors_agree_with_each_other(
        self, label: str, factory: Any, _registry: Any
    ) -> None:
        """Every listed backend is available and has real metadata.

        Three methods answering about one registry can each be right alone
        and still contradict each other, which is the failure a reader hits
        and no single-method test sees.
        """
        for backend in factory.get_available_backends():
            assert factory.is_backend_available(backend) is True, backend
            assert "error" not in factory.get_backend_info(backend), backend


class TestAliasesCollapse:
    """A registry lists every spelling. The reported list names each once."""

    @pytest.mark.parametrize(("label", "_factory", "registry"), FACTORIES, ids=_IDS)
    def test_reported_names_are_sorted_and_unique(
        self, label: str, _factory: Any, registry: Any
    ) -> None:
        from dataknobs_data.backend_selection import available_backends

        reported = available_backends(registry)
        assert reported == sorted(set(reported))

    @pytest.mark.parametrize(("label", "_factory", "registry"), FACTORIES, ids=_IDS)
    def test_every_reported_name_is_registered(
        self, label: str, _factory: Any, registry: Any
    ) -> None:
        from dataknobs_data.backend_selection import available_backends

        for name in available_backends(registry):
            assert registry.is_registered(name), name

    @pytest.mark.parametrize(("label", "_factory", "registry"), FACTORIES, ids=_IDS)
    def test_each_registered_backend_is_reported_exactly_once(
        self, label: str, _factory: Any, registry: Any
    ) -> None:
        """Two keys resolving to one class contribute one name, not two."""
        from dataknobs_data.backend_selection import available_backends

        reported = available_backends(registry)
        reported_factories = {id(registry.get_factory(name)) for name in reported}
        assert len(reported_factories) == len(reported)

        every_factory = {id(registry.get_factory(key)) for key in registry.list_keys()}
        assert reported_factories == every_factory

    def test_the_known_sync_aliases_are_not_listed_separately(self) -> None:
        """Named explicitly, so a regression says which alias came back."""
        from dataknobs_data.backend_selection import available_backends

        reported = available_backends(sync_backends)
        for alias, canonical in (
            ("mem", "memory"),
            ("sqlite3", "sqlite"),
            ("postgresql", "postgres"),
            ("pg", "postgres"),
            ("es", "elasticsearch"),
        ):
            if sync_backends.is_registered(canonical):
                assert canonical in reported
                assert alias not in reported

    def test_the_known_vector_aliases_are_not_listed_separately(self) -> None:
        from dataknobs_data.backend_selection import available_backends

        reported = available_backends(vector_backends)
        for alias, canonical in (("chromadb", "chroma"), ("postgresql", "pgvector")):
            if vector_backends.is_registered(canonical):
                assert canonical in reported
                assert alias not in reported


class TestTheErrorAndTheAccessorShareOneList:
    """The guard against the two answers drifting back apart."""

    @pytest.mark.parametrize(("label", "factory", "_registry"), FACTORIES, ids=_IDS)
    def test_the_unknown_backend_message_prints_the_reported_list(
        self, label: str, factory: Any, _registry: Any
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            factory.create(backend="no-such-backend")

        rendered = ", ".join(factory.get_available_backends())
        assert rendered in str(excinfo.value)


class TestOneDefaultBackendTable:
    """The validation path mirrors the construction path by construction."""

    def test_the_shared_default_is_memory(self) -> None:
        from dataknobs_data.backend_selection import DEFAULT_BACKEND

        assert DEFAULT_BACKEND == MEMORY

    def test_the_vector_config_resolver_uses_the_shared_default(self) -> None:
        """``_resolve_vector_store_config_cls`` defaults where the factory does.

        Holding no independent table is what keeps the validator from
        refusing a config the factory would happily build.
        """
        from dataknobs_data.backend_selection import DEFAULT_BACKEND
        from dataknobs_data.vector.stores import _resolve_vector_store_config_cls

        assert _resolve_vector_store_config_cls({}) is _resolve_vector_store_config_cls(
            {"backend": DEFAULT_BACKEND}
        )


class TestRecognisingTheDefaultUnderItsOtherNames:
    """``mem`` and ``memory`` name one backend; a caller branching on the
    answer must not route them differently.

    Without a registry this compares names, which is right for a caller
    choosing a log line and wrong for one choosing a code path -- the
    factory resolves both spellings to the same class, so a config saying
    ``mem`` took the non-default branch and got a *different storage mode*
    for a backend that was the default all along.
    """

    def test_an_alias_of_the_default_is_recognised_with_a_registry(self) -> None:
        from dataknobs_data.backends import sync_backends

        assert is_default_backend({"backend": "mem"}, sync_backends) is True

    def test_case_and_padding_still_normalise_around_the_alias(self) -> None:
        from dataknobs_data.backends import sync_backends

        assert is_default_backend({"backend": "  MEM  "}, sync_backends) is True

    def test_a_real_choice_is_still_a_real_choice(self) -> None:
        from dataknobs_data.backends import sync_backends

        assert is_default_backend({"backend": "file"}, sync_backends) is False

    def test_without_a_registry_it_compares_names(self) -> None:
        """The documented narrower contract, pinned so it cannot drift."""
        assert is_default_backend({"backend": "memory"}) is True
        assert is_default_backend({"backend": "mem"}) is False

    def test_the_absent_key_answer_does_not_depend_on_the_registry(self) -> None:
        from dataknobs_data.backends import sync_backends

        assert is_default_backend({}) is True
        assert is_default_backend({}, sync_backends) is True

    def test_a_null_backend_is_still_refused_with_a_registry(self) -> None:
        from dataknobs_data.backends import sync_backends

        with pytest.raises(ValueError, match="present but null"):
            is_default_backend({"backend": None}, sync_backends)

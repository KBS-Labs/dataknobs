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

The provenance group deliberately names no module. Which file decides the
backend is an implementation detail the tests should survive; that a factory
says out loud when nothing asked for one is not.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_data import AsyncDatabaseFactory, DatabaseFactory
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
        self, selection_log: pytest.LogCaptureFixture
    ) -> None:
        """A backend that is neither absent nor the default is ordinary."""
        DatabaseFactory().create(backend="file", path=":memory:")

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
        """
        with pytest.raises(ValueError, match="Unknown backend type"):
            DatabaseFactory().create(backend=None)

        assert _records(selection_log, logging.WARNING) == []


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

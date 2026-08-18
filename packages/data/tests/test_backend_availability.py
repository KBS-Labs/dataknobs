"""Available means installed -- and what to install when it is not.

Two registration idioms lived in this package and meant opposite things.
``postgres``, ``duckdb`` and async ``sqlite`` import their driver at module
top level, so a missing driver failed the import and the backend went
unregistered. ``faiss``, ``chroma``, ``pgvector``, ``s3`` and async
``elasticsearch`` swallow their own ``ImportError`` and defer the raise to
construction, so they registered whether or not the driver was there.

``is_backend_available`` was ``is_registered``, so it answered honestly for
the first group and dishonestly for the second: a consumer following the
documented guard-before-offer flow on a machine without ``faiss-cpu`` was
told the backend was available and then handed the exact exception the
guard exists to prevent.

The two idioms also broke the other half of the flow, in mirror image. The
question "what would I install?" is only ever asked while the answer is
unavailable -- and the metadata carrying ``requires_install`` went
unregistered along with the factory, so it could be read only for backends
that did not need it.

Both are pinned here. The environment cannot supply a missing driver on
demand -- every optional dependency is installed in this repo's dev env --
so the registration functions take the "is this module importable?"
predicate as a parameter and these tests pass their own. The registry, the
registration code and the factories are all real; the only substituted
thing is the environment fact that cannot be varied for real.
"""

from __future__ import annotations

import logging

from typing import Any, Callable

import pytest

from dataknobs_common.registry import PluginRegistry

from dataknobs_data import AsyncDatabaseFactory, DatabaseFactory
from dataknobs_data.backend_selection import (
    available_backends,
    backend_available,
    backend_info,
    select_backend,
)
from dataknobs_data.backends import (
    _register_async_backends,
    _register_sync_backends,
    async_backends,
    sync_backends,
)
from dataknobs_data.vector.stores import _register_vector_backends, vector_backends
from dataknobs_data.vector.stores.factory import VectorStoreFactory


def _without(*absent: str) -> Callable[[str], bool]:
    """An environment in which exactly ``absent`` is not installed."""
    missing = frozenset(absent)
    return lambda module: module not in missing


def _sync_registry(installed: Callable[[str], bool]) -> PluginRegistry[Any]:
    registry: PluginRegistry[Any] = PluginRegistry("probe_sync", canonicalize_keys=True)
    _register_sync_backends(registry, installed=installed)
    return registry


def _async_registry(installed: Callable[[str], bool]) -> PluginRegistry[Any]:
    registry: PluginRegistry[Any] = PluginRegistry("probe_async", canonicalize_keys=True)
    _register_async_backends(registry, installed=installed)
    return registry


def _vector_registry(installed: Callable[[str], bool]) -> PluginRegistry[Any]:
    registry: PluginRegistry[Any] = PluginRegistry("probe_vector", canonicalize_keys=True)
    _register_vector_backends(registry, installed=installed)
    return registry


# ---------------------------------------------------------------------------
# A backend behind a missing driver
# ---------------------------------------------------------------------------


class TestABackendBehindAMissingDriver:
    """The first idiom: the driver import fails, so the backend never loads."""

    def test_it_is_not_reported_available(self) -> None:
        registry = _sync_registry(_without("psycopg2"))

        assert backend_available(registry, "postgres") is False

    def test_it_is_absent_from_the_reported_list(self) -> None:
        registry = _sync_registry(_without("psycopg2"))

        assert "postgres" not in available_backends(registry)

    def test_memory_is_unaffected(self) -> None:
        """A backend with no optional dependency does not get gated."""
        registry = _sync_registry(_without("psycopg2"))

        assert backend_available(registry, "memory") is True

    def test_it_still_says_what_to_install(self) -> None:
        registry = _sync_registry(_without("psycopg2"))

        info = backend_info(registry, "postgres")

        assert info["requires_install"] == "pip install dataknobs-data[postgres]"

    def test_an_alias_of_it_says_the_same(self) -> None:
        registry = _sync_registry(_without("psycopg2"))

        assert backend_info(registry, "pg") == backend_info(registry, "postgres")

    def test_selecting_it_says_what_to_install_rather_than_reporting_a_typo(
        self,
    ) -> None:
        """Through ``select_backend``, which is what the factories call."""
        registry = _sync_registry(_without("psycopg2"))

        with pytest.raises(ValueError) as excinfo:
            select_backend({"backend": "postgres"}, registry, kind="database")

        message = str(excinfo.value)
        assert "pip install dataknobs-data[postgres]" in message
        assert "Unknown" not in message

    def test_a_real_typo_still_reads_as_one(self) -> None:
        registry = _sync_registry(_without("psycopg2"))

        with pytest.raises(ValueError, match="Unknown backend type"):
            select_backend({"backend": "postgrez"}, registry, kind="database")


class TestABackendThatDefersItsImportError:
    """The second idiom: the module loads, and raises only on construction.

    This is the group ``is_registered`` answered dishonestly about.
    """

    def test_faiss_is_not_reported_available_without_its_driver(self) -> None:
        registry = _vector_registry(_without("faiss"))

        assert backend_available(registry, "faiss") is False

    def test_memory_vector_store_is_unaffected(self) -> None:
        registry = _vector_registry(_without("faiss"))

        assert backend_available(registry, "memory") is True

    def test_chroma_answers_for_its_own_driver(self) -> None:
        registry = _vector_registry(_without("chromadb"))

        assert backend_available(registry, "chroma") is False
        assert backend_available(registry, "faiss") is True

    def test_a_chroma_alias_is_gated_with_it(self) -> None:
        registry = _vector_registry(_without("chromadb"))

        assert backend_available(registry, "chromadb") is False

    def test_it_still_says_what_to_install(self) -> None:
        registry = _vector_registry(_without("faiss"))

        assert backend_info(registry, "faiss")["requires_install"] == "pip install faiss-cpu"

    def test_s3_is_gated_on_boto3(self) -> None:
        registry = _sync_registry(_without("boto3"))

        assert backend_available(registry, "s3") is False

    def test_async_elasticsearch_is_gated_on_its_client(self) -> None:
        registry = _async_registry(_without("elasticsearch"))

        assert backend_available(registry, "elasticsearch") is False


# ---------------------------------------------------------------------------
# The flow the documentation describes
# ---------------------------------------------------------------------------


class TestTheDocumentedFlow:
    """Guard, then say what to install -- as one sequence, as written."""

    def test_guard_then_install_hint(self) -> None:
        registry = _sync_registry(_without("psycopg2"))

        printed = None
        if not backend_available(registry, "postgres"):
            printed = backend_info(registry, "postgres")["requires_install"]

        assert printed == "pip install dataknobs-data[postgres]"

    def test_the_hint_is_absent_for_a_backend_that_needs_none(self) -> None:
        """``memory`` reports ``False``, not a pip command."""
        registry = _sync_registry(_without("psycopg2"))

        assert backend_info(registry, "memory")["requires_install"] is False


class TestTheOtherWayIn:
    """``from_backend`` reads the same registries and must answer alike.

    ``AsyncDatabase.from_backend`` / ``SyncDatabase.from_backend`` are the
    same four steps the factories take -- read a name, look it up, raise if
    it is missing, build -- written a second time. They were not migrated
    with the factories, so a correctly spelled backend whose driver is
    absent came back as ``Unknown backend: postgres``: the "go and look for
    a typo in a name that is spelled right" answer this whole change exists
    to remove, still reachable through a public classmethod that a
    resource adapter in another package calls.
    """

    @staticmethod
    def _withdrawn(registry: Any, backend: str) -> tuple[Any, dict[str, Any]]:
        """Declare a real backend unavailable, as a driverless machine would."""
        backend_class = registry.get_factory(backend)
        if backend_class is None:
            pytest.skip(f"{backend} is not installed here, so it cannot be withdrawn")
        metadata = registry.get_metadata(backend)
        registry.declare_unavailable(
            backend,
            metadata=metadata,
            reason="psycopg2 is not installed. Install with: pip install dataknobs-data[postgres]",
        )
        return backend_class, metadata

    @staticmethod
    def _restore(registry: Any, backend: str, backend_class: Any, metadata: Any) -> None:
        registry.register(backend, backend_class, metadata=metadata, override=True)

    @pytest.mark.asyncio
    async def test_async_says_what_to_install_rather_than_reporting_a_typo(self) -> None:
        from dataknobs_data.database import AsyncDatabase

        backend_class, metadata = self._withdrawn(async_backends, "postgres")
        try:
            with pytest.raises(ValueError) as excinfo:
                await AsyncDatabase.from_backend("postgres", {})
        finally:
            self._restore(async_backends, "postgres", backend_class, metadata)

        message = str(excinfo.value)
        assert "pip install dataknobs-data[postgres]" in message
        assert "Unknown" not in message

    def test_sync_says_what_to_install_rather_than_reporting_a_typo(self) -> None:
        from dataknobs_data.database import SyncDatabase

        backend_class, metadata = self._withdrawn(sync_backends, "postgres")
        try:
            with pytest.raises(ValueError) as excinfo:
                SyncDatabase.from_backend("postgres", {})
        finally:
            self._restore(sync_backends, "postgres", backend_class, metadata)

        message = str(excinfo.value)
        assert "pip install dataknobs-data[postgres]" in message
        assert "Unknown" not in message

    @pytest.mark.asyncio
    async def test_a_real_typo_lists_canonical_names_not_every_spelling(self) -> None:
        """It printed ``list_keys()``, so aliases read as separate backends."""
        from dataknobs_data.database import AsyncDatabase

        with pytest.raises(ValueError) as excinfo:
            await AsyncDatabase.from_backend("postgrez", {})

        message = str(excinfo.value)
        assert "postgres" in message
        # Parsed into names rather than substring-tested: `es` is an alias
        # of elasticsearch and also a substring of "postgres", so a
        # substring test reports a passing message as failing. ("pg," was
        # asserted here before and could not fail either way -- the pre-fix
        # message was a list repr, where the substring was "pg',".)
        listed = message.split("Available backends: ")[1].split(", ")
        aliases = {"postgresql", "pg", "sqlite3", "es", "mem"}
        assert not aliases & set(listed), (
            f"aliases listed as backends: {sorted(aliases & set(listed))}"
        )
        assert "postgres" in listed

    @pytest.mark.asyncio
    async def test_it_still_builds_a_usable_connected_database(self) -> None:
        """The behaviour every existing caller depends on.

        Asserted through a round-trip rather than a connection flag,
        because moving construction onto ``from_config`` is the part of
        this change that could plausibly build something different.
        """
        from dataknobs_data import Record
        from dataknobs_data.database import AsyncDatabase

        db = await AsyncDatabase.from_backend("memory", {})

        await db.upsert("k", Record({"v": 1}))
        stored = await db.read("k")

        assert stored is not None
        assert stored.data["v"] == 1


# ---------------------------------------------------------------------------
# What each deferred loader actually loads
# ---------------------------------------------------------------------------


#: Canonical backend name -> ``(module suffix, class name)``, one row per
#: deferred loader. Spelled out rather than derived, because deriving it from
#: the loaders would agree with them by construction and check nothing.
EXPECTED_SYNC = {
    "memory": ("backends.memory", "SyncMemoryDatabase"),
    "file": ("backends.file", "SyncFileDatabase"),
    "sqlite": ("backends.sqlite", "SyncSQLiteDatabase"),
    "postgres": ("backends.postgres", "SyncPostgresDatabase"),
    "elasticsearch": ("backends.elasticsearch", "SyncElasticsearchDatabase"),
    "s3": ("backends.s3", "SyncS3Database"),
    "duckdb": ("backends.duckdb", "SyncDuckDBDatabase"),
}

EXPECTED_ASYNC = {
    "memory": ("backends.memory", "AsyncMemoryDatabase"),
    "file": ("backends.file", "AsyncFileDatabase"),
    "sqlite": ("backends.sqlite_async", "AsyncSQLiteDatabase"),
    "postgres": ("backends.postgres", "AsyncPostgresDatabase"),
    "elasticsearch": ("backends.elasticsearch_async", "AsyncElasticsearchDatabase"),
    "s3": ("backends.s3_async", "AsyncS3Database"),
    "duckdb": ("backends.duckdb", "AsyncDuckDBDatabase"),
}

EXPECTED_VECTOR = {
    "memory": ("vector.stores.memory", "MemoryVectorStore"),
    "faiss": ("vector.stores.faiss", "FaissVectorStore"),
    "chroma": ("vector.stores.chroma", "ChromaVectorStore"),
    "pgvector": ("vector.stores.pgvector", "PgVectorStore"),
}

EXPECTATIONS = [
    ("sync", sync_backends, EXPECTED_SYNC),
    ("async", async_backends, EXPECTED_ASYNC),
    ("vector", vector_backends, EXPECTED_VECTOR),
]


class TestEachLoaderLoadsWhatItSays:
    """A deferred loader that names the wrong thing fails quietly.

    Registration cannot import its backend classes at module scope -- one
    missing optional driver would take down the package import -- so each
    is deferred behind a loader called after the driver probe passes.
    Whether that loader is written as ``import_module(name)`` or as an
    import statement, a wrong module or a wrong class name raises
    ``ImportError``, which ``register_backend`` treats exactly as it treats
    an absent driver: the backend is declared unavailable and the process
    carries on.

    That is the right handling of a genuinely missing dependency and the
    wrong handling of a typo, and nothing distinguishes them at runtime.
    The failure is silent in the worst way -- the backend simply is not
    there, and the message says to install something that is already
    installed.

    ``sqlite`` is the row most exposed: the sync class lives in
    ``sqlite.py`` and the async one in ``sqlite_async.py``, so the two
    loaders differ by a suffix and each would raise ``ImportError`` against
    the other's module rather than returning the wrong class.
    """

    @pytest.mark.parametrize(
        ("label", "registry", "expected"),
        EXPECTATIONS,
        ids=[label for label, _, _ in EXPECTATIONS],
    )
    def test_every_expected_backend_is_known(
        self, label: str, registry: Any, expected: dict[str, tuple[str, str]]
    ) -> None:
        """Known, not creatable -- the difference is the local install set.

        Asserting *registered* here made this a test that every optional
        driver is installed, which fails on a lean environment with a
        message about a loader typo. A backend whose driver is absent is
        still `is_known`, so that is the invariant which holds everywhere;
        whether it is creatable is checked below, per backend, where it can
        be skipped honestly.
        """
        unknown = sorted(key for key in expected if not registry.is_known(key))

        assert unknown == [], (
            f"{label}: the registry has never heard of {unknown} -- a loader "
            "naming a module or class that does not exist raises ImportError, "
            "which registration reports as a missing driver"
        )

    @pytest.mark.parametrize(
        ("label", "registry", "expected"),
        EXPECTATIONS,
        ids=[label for label, _, _ in EXPECTATIONS],
    )
    def test_each_one_resolves_to_the_named_class(
        self, label: str, registry: Any, expected: dict[str, tuple[str, str]]
    ) -> None:
        skipped = []
        for key, (module_suffix, class_name) in expected.items():
            loaded = registry.get_factory(key)
            if loaded is None:
                # Declared unavailable: its driver is absent here. The
                # loader may still be reachable (a store guarding its
                # driver behind a flag imports fine without it) -- take
                # that where it is offered, and record the rest rather
                # than reporting a lean environment as a typo.
                loaded = registry.load_declared_type(key)
                if loaded is None:
                    skipped.append(key)
                    continue
            assert loaded.__name__ == class_name, f"{label}: {key}"
            assert loaded.__module__ == f"dataknobs_data.{module_suffix}", (
                f"{label}: {key} resolved to a class from the wrong module"
            )

        if skipped:
            # Named rather than silently passing over: a green run that
            # checked four of seven loaders should say which three it did
            # not reach.
            pytest.skip(f"{label}: driver absent, loader unverifiable for {sorted(skipped)}")

    @pytest.mark.parametrize(
        ("label", "registry", "expected"),
        EXPECTATIONS,
        ids=[label for label, _, _ in EXPECTATIONS],
    )
    def test_no_two_backends_share_a_class(
        self, label: str, registry: Any, expected: dict[str, tuple[str, str]]
    ) -> None:
        """A copy-paste slip between two loaders reads as an alias.

        Two canonical keys resolving to one class is how
        ``list_canonical_keys`` decides they are the same backend, so the
        symptom of the slip is a backend quietly vanishing from the list
        rather than an error.
        """
        loaded = {key: registry.get_factory(key) for key in expected}
        by_class: dict[int, list[str]] = {}
        for key, cls in loaded.items():
            by_class.setdefault(id(cls), []).append(key)

        shared = {tuple(keys) for keys in by_class.values() if len(keys) > 1}
        assert shared == set(), f"{label}: these backends resolved to one class: {shared}"


# ---------------------------------------------------------------------------
# The shipped registries, against the environment actually running
# ---------------------------------------------------------------------------


FACTORIES: list[tuple[str, Any, Any]] = [
    ("database", DatabaseFactory(), sync_backends),
    ("async database", AsyncDatabaseFactory(), async_backends),
    ("vector store", VectorStoreFactory(), vector_backends),
]
_IDS = [label for label, _, _ in FACTORIES]


class TestTheShippedRegistries:
    @pytest.mark.parametrize(("label", "factory", "registry"), FACTORIES, ids=_IDS)
    def test_every_reported_backend_is_actually_available(
        self, label: str, factory: Any, registry: Any
    ) -> None:
        """The list a factory reports is a list of things it can build."""
        unavailable = [
            name
            for name in factory.get_available_backends()
            if not factory.is_backend_available(name)
        ]

        assert unavailable == []

    @pytest.mark.parametrize(("label", "factory", "registry"), FACTORIES, ids=_IDS)
    def test_every_reported_backend_describes_itself(
        self, label: str, factory: Any, registry: Any
    ) -> None:
        for name in factory.get_available_backends():
            assert factory.get_backend_info(name).get("description"), name

    @pytest.mark.parametrize(("label", "factory", "registry"), FACTORIES, ids=_IDS)
    def test_an_optional_backend_declares_the_module_to_probe(
        self, label: str, factory: Any, registry: Any
    ) -> None:
        """``requires_install`` says what to type; ``requires_module`` is
        what makes availability answerable without typing it.
        """
        for name in factory.get_available_backends():
            info = factory.get_backend_info(name)
            if info.get("requires_install"):
                assert info.get("requires_module"), name

    @pytest.mark.parametrize(("label", "factory", "registry"), FACTORIES, ids=_IDS)
    def test_the_accessor_is_not_reimplemented_per_factory(
        self, label: str, factory: Any, registry: Any
    ) -> None:
        """One implementation, reached three ways."""
        for name in registry.list_known_keys():
            assert factory.is_backend_available(name) == backend_available(registry, name)


class TestDescribingABackendWithNoMetadata:
    """``backend_info`` answers "what is this?", including "not a thing".

    It read metadata and treated a falsy result as "never heard of it",
    which is wrong for a backend declared unavailable without any --
    ``declare_unavailable`` accepts ``metadata=None``. The one state this
    function exists to describe was reported as unrecognised.
    """

    def test_a_declared_backend_without_metadata_is_not_called_unknown(self) -> None:
        sync_backends.declare_unavailable("acme_db", reason="acme-sdk is not installed")
        try:
            info = backend_info(sync_backends, "acme_db")
            assert "error" not in info, f"reported as unrecognised: {info}"
        finally:
            sync_backends.unregister("acme_db")

    def test_a_name_nobody_declared_is_still_unknown(self) -> None:
        info = backend_info(sync_backends, "no_such_backend")

        assert info["error"] == "Backend 'no_such_backend' not recognized"

    def test_a_declared_backend_with_metadata_still_answers_with_it(self) -> None:
        """The path that already worked, kept covered."""
        sync_backends.declare_unavailable(
            "acme_db",
            metadata={"requires_install": "pip install acme-sdk"},
            reason="acme-sdk is not installed",
        )
        try:
            assert backend_info(sync_backends, "acme_db")["requires_install"] == (
                "pip install acme-sdk"
            )
        finally:
            sync_backends.unregister("acme_db")


class TestAUserStateStoreThatNamedNoBackend:
    """The config spelled the default itself, so the factory never knew.

    ``UserStateStoreConfig.backend`` defaulted to ``"memory"`` and was
    forwarded unconditionally, so a config that named nothing arrived at
    the factory as an explicit choice and the absence was consumed one
    frame above the only code positioned to report it. Same shape as the
    three sites migrated earlier, in the typed-dataclass spelling rather
    than ``.get(key, default)``.
    """

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
            and record.name == "dataknobs_data.backend_selection"
        ]

    @pytest.mark.asyncio
    async def test_an_absent_backend_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_data.user.config import UserStateStoreConfig
        from dataknobs_data.user.store import AsyncUserStateStore

        config = UserStateStoreConfig.from_dict({"namespace": "u"})
        with caplog.at_level(logging.DEBUG, logger="dataknobs_data.backend_selection"):
            await AsyncUserStateStore.from_config_async(config)

        assert len(self._warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_a_named_backend_is_not(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_data.user.config import UserStateStoreConfig
        from dataknobs_data.user.store import AsyncUserStateStore

        config = UserStateStoreConfig.from_dict({"namespace": "u", "backend": "memory"})
        with caplog.at_level(logging.DEBUG, logger="dataknobs_data.backend_selection"):
            await AsyncUserStateStore.from_config_async(config)

        assert self._warnings(caplog) == []

    def test_the_sync_twin_reports_it_too(self, caplog: pytest.LogCaptureFixture) -> None:
        """Both ``_setup`` and ``_ainit`` held the same line."""
        from dataknobs_data.user.config import UserStateStoreConfig
        from dataknobs_data.user.store import UserStateStore

        config = UserStateStoreConfig.from_dict({"namespace": "u"})
        with caplog.at_level(logging.DEBUG, logger="dataknobs_data.backend_selection"):
            UserStateStore.from_config(config)

        assert len(self._warnings(caplog)) == 1

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
        assert backend_class is not None, f"{backend} is not installed in this env"
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
        assert "pg," not in message and "postgresql" not in message

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
    def test_every_expected_backend_is_registered(
        self, label: str, registry: Any, expected: dict[str, tuple[str, str]]
    ) -> None:
        missing = sorted(set(expected) - set(registry.list_canonical_keys()))

        assert missing == [], (
            f"{label}: registered nothing for {missing} -- a loader naming a "
            "module or class that does not exist raises ImportError, which "
            "registration reports as a missing driver"
        )

    @pytest.mark.parametrize(
        ("label", "registry", "expected"),
        EXPECTATIONS,
        ids=[label for label, _, _ in EXPECTATIONS],
    )
    def test_each_one_resolves_to_the_named_class(
        self, label: str, registry: Any, expected: dict[str, tuple[str, str]]
    ) -> None:
        for key, (module_suffix, class_name) in expected.items():
            loaded = registry.get_factory(key)
            assert loaded is not None, f"{label}: {key} is not registered"
            assert loaded.__name__ == class_name, f"{label}: {key}"
            assert loaded.__module__ == f"dataknobs_data.{module_suffix}", (
                f"{label}: {key} resolved to a class from the wrong module"
            )

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

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
        what makes availability answerable without typing it."""
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

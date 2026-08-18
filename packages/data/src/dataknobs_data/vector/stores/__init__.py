"""Specialized vector store implementations."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Type

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    ConfigClassResolution,
    StructuredConfig,
    config_registries,
)

from dataknobs_data.backend_selection import (
    DEFAULT_BACKEND,
    module_installed,
    normalize_backend,
    register_backend,
)

from .base import VectorStore

logger = logging.getLogger(__name__)


# --- Deferred loaders ------------------------------------------------
#
# Each imports its store class on first call rather than at module scope,
# so a store whose optional driver is absent fails its own registration
# instead of the package import.
#
# Spelled as an import *statement* rather than ``import_module(name)``
# because the module and the attribute are literals in this source, not a
# dotted path arriving from configuration: there is no separator to pick,
# no typo that a user could make, and nothing for
# ``dataknobs_common.imports`` to own. Written statically the class stays
# a symbol -- mypy checks it, a rename updates it, and a misspelling fails
# at authoring time rather than at registration.


def _memory_vector_store() -> type[VectorStore]:
    from .memory import MemoryVectorStore

    return MemoryVectorStore


def _faiss_vector_store() -> type[VectorStore]:
    from .faiss import FaissVectorStore

    return FaissVectorStore


def _chroma_vector_store() -> type[VectorStore]:
    from .chroma import ChromaVectorStore

    return ChromaVectorStore


def _pgvector_store() -> type[VectorStore]:
    from .pgvector import PgVectorStore

    return PgVectorStore


def _register_vector_backends(
    registry: PluginRegistry[Type[VectorStore]],
    *,
    installed: Callable[[str], bool] = module_installed,
) -> None:
    """Register the built-in vector backends this installation can build.

    Every store here defers its driver's ``ImportError`` to construction --
    the module sets an ``*_AVAILABLE`` flag and raises from ``_setup`` --
    so all three used to register whether or not the driver was present.
    Probing the declared driver at registration is what makes the answer to
    "is this available?" the same as the answer to "will ``create`` work?"

    Args:
        registry: The registry to populate.
        installed: The "is this module importable?" predicate, forwarded to
            :func:`register_backend`. Injectable so a test can describe an
            environment the one it runs in is not.
    """
    register_backend(
        registry,
        "memory",
        _memory_vector_store,
        metadata={
            "description": "In-memory vector storage for testing",
            "persistent": False,
            "requires_install": False,
            "config_options": {
                "dimensions": "Vector dimensions (required)",
                "metric": "Distance metric: cosine, euclidean, dot_product",
            },
        },
        installed=installed,
    )

    register_backend(
        registry,
        "faiss",
        _faiss_vector_store,
        metadata={
            "description": "Facebook AI Similarity Search - efficient vector search",
            "persistent": True,
            "requires_install": "pip install faiss-cpu",
            "requires_module": "faiss",
            "config_options": {
                "dimensions": "Vector dimensions (required)",
                "metric": "Distance metric: cosine, euclidean, dot_product",
                "index_type": "Index type: flat, ivfflat, hnsw, auto",
                "persist_path": "Path to save/load index",
                "nlist": "Number of clusters for IVF index",
                "m": "Number of connections for HNSW",
            },
        },
        installed=installed,
    )

    register_backend(
        registry,
        "chroma",
        _chroma_vector_store,
        metadata={
            "description": "ChromaDB - AI-native vector database",
            "persistent": True,
            "requires_install": "pip install chromadb",
            "requires_module": "chromadb",
            "config_options": {
                "collection_name": "Name of the collection",
                "persist_path": "Path for persistent storage",
                "embedding_function": "Embedding function name or object",
                "metric": "Distance metric: cosine, euclidean, dot_product",
            },
        },
        aliases=("chromadb",),
        installed=installed,
    )

    register_backend(
        registry,
        "pgvector",
        _pgvector_store,
        metadata={
            "description": "PostgreSQL with pgvector extension - production vector database",
            "persistent": True,
            "requires_install": "pip install asyncpg",
            "requires_module": "asyncpg",
            "config_options": {
                "connection_string": "PostgreSQL connection URL (or use DATABASE_URL env)",
                "dimensions": "Vector dimensions (required)",
                "metric": "Distance metric: cosine, euclidean, inner_product",
                "schema": "Database schema (default: public)",
                "table_name": "Table name (default: knowledge_embeddings)",
                "domain_id": "Domain ID for multi-tenant isolation (optional)",
                "pool_min_size": "Min connection pool size (default: 2)",
                "pool_max_size": "Max connection pool size (default: 10)",
                "columns": "Column name mappings dict (optional)",
                "auto_create_table": "Create table if missing (default: True)",
                "id_type": "ID column type: uuid or text (default: text)",
            },
        },
        aliases=("postgresql",),
        installed=installed,
    )


# Create singleton instance BEFORE importing factory to avoid circular import
vector_backends: PluginRegistry[Type[VectorStore]] = PluginRegistry(
    "vector_backends",
    canonicalize_keys=True,
    on_first_access=_register_vector_backends,
)

# Keep VectorBackendRegistry as alias for backward compat.
# Use the unparameterized class so isinstance() checks still work at runtime.
VectorBackendRegistry = PluginRegistry

# Now import factory (which will import vector_backends from this module)
from .factory import VectorStoreFactory  # noqa: E402


def _resolve_vector_store_config_cls(
    raw: Mapping[str, Any],
) -> ConfigClassResolution:
    """Resolve a ``vector_store`` section's dict to its config class.

    The resolver registered for the ``"vector_store"`` binding in
    :data:`~dataknobs_common.structured_config.config_registries`, used by
    :meth:`StructuredConfig.validate
    <dataknobs_common.structured_config.StructuredConfig.validate>` to
    validate a raw ``vector_store`` config without constructing the store.

    Delegates to ``vector_backends`` — the same registry the construction
    path uses — by reading ``CONFIG_CLS`` off the registered store class
    for the ``"backend"`` discriminator (defaulting to
    :data:`~dataknobs_data.backend_selection.DEFAULT_BACKEND`, the constant
    the factory itself falls back to). Holding no independent
    backend→config-class table, and no second spelling of the default, is
    the no-drift guarantee. Returns ``None`` for an unknown
    backend, which ``validate`` surfaces as a ``ConfigurationError``.

    Three outcomes, because the discriminator has three states and only two
    of them are the same event:

    - **A creatable backend** resolves to its ``CONFIG_CLS``.
    - **A backend this installation cannot build** — known to the registry,
      driver absent — resolves to its ``CONFIG_CLS`` when the store class
      is still importable, and to :data:`SKIP_VALIDATION` when it is not.
      Whether a config is well-formed is a property of the config, not of
      the machine reading it, so an uninstalled driver must not fail a
      valid section; and it is not a typo, so it must not be reported as
      one. ``create()`` is the call that cares whether the driver is
      present, and it says so by name.
    - **A backend nobody registered** returns ``None``, which ``validate``
      surfaces as a ``ConfigurationError``. This is the genuine typo.

    A backend that is *registered* but exposes no ``StructuredConfig``
    ``CONFIG_CLS`` also returns :data:`SKIP_VALIDATION`, logged at WARNING
    rather than DEBUG: unlike a missing driver, that one is a gap someone
    can close. No built-in backend is in this state today (the parity guard
    ``test_resolver_agrees_with_construction_registry_for_all_backends``
    keeps it so); the branch covers a custom bare-callable backend
    registered out of band.

    The middle case is split rather than skipped wholesale because the two
    halves are genuinely different. A store guarding its optional driver
    behind a module-level flag — which every optional store here does —
    imports without it, so its schema is readable and the section is
    checked exactly as it would be on a machine that has the driver. Only
    a store whose module raises on import has nothing to read, and that is
    what :data:`SKIP_VALIDATION` is for. The schema comes from the same
    loader the construction path uses, via
    :meth:`~dataknobs_common.registry.PluginRegistry.load_declared_type`,
    so this holds no second key-to-class table to drift out of step.

    What ``validate()`` does with the returned class is its own question:
    today ``from_dict`` is permissive, so the class it gets back and
    :data:`SKIP_VALIDATION` reject the same configs — everything except an
    unrecognised discriminator. Returning the class anyway is what keeps
    *which check runs* independent of the local install set, so tightening
    ``from_dict`` later does not silently tighten it only on the machines
    that happen to have every driver.
    """
    if "backend" in raw:
        # Normalised by the same function the construction path uses. The
        # two used to normalise separately and disagreed about a present
        # but unusable discriminator: this path called ``.lower()`` on it
        # and raised ``AttributeError`` where the factory raised a
        # ``ValueError`` naming the problem. Same reading of the config,
        # each path's own exception type.
        try:
            backend = normalize_backend(raw["backend"])
        except ValueError as exc:
            raise ConfigurationError(
                f"vector_store: {exc}",
                context={"binding": "vector_store", "backend": raw["backend"]},
            ) from exc
    else:
        backend = DEFAULT_BACKEND
    store_cls = vector_backends.get_factory(backend)
    if store_cls is None:
        # Asked of the registry directly. Truthy metadata used to stand in
        # for "known", which is a different question with a different
        # answer for a backend declared unavailable without any -- it was
        # reported as a typo, the exact failure this branch exists to
        # prevent, just under a narrower precondition.
        if not vector_backends.is_known(backend):
            # Unknown discriminator — the legitimate typo path; validate()
            # raises ConfigurationError. Silent here so a real typo is
            # reported by validate(), not pre-empted by a misleading
            # WARNING.
            return None
        # Known, but not creatable on this machine. Whether a config is
        # well-formed does not depend on which optional drivers happen to
        # be installed where it is being checked, so reporting the section
        # as matching no variant is wrong twice over: it fails a valid
        # config, and it sends the reader to look for a typo in a name that
        # is spelled correctly. `create()` is the call that cares about the
        # driver, and it names it.
        store_cls = vector_backends.load_declared_type(backend)
        if store_cls is None:
            logger.debug(
                "Vector-store backend %r is known but neither creatable nor "
                "importable here, so there is no store class to read a typed "
                "schema off; skipping validation of this section.",
                backend,
            )
            return SKIP_VALIDATION
    config_cls = getattr(store_cls, "CONFIG_CLS", None)
    if isinstance(config_cls, type) and issubclass(config_cls, StructuredConfig):
        return config_cls
    logger.warning(
        "Vector-store backend %r is registered but exposes no StructuredConfig "
        "CONFIG_CLS; validate() has no typed schema to check its config "
        "section against and will skip it. Give the backend a CONFIG_CLS to "
        "make its section validatable.",
        backend,
    )
    return SKIP_VALIDATION


# Eager registration (not on_first_access): importing this package is what
# makes the ``vector_store`` binding resolvable, and any parent config that
# holds a vector-store section already depends on this package. ``override``
# keeps re-import idempotent.
config_registries.register("vector_store", _resolve_vector_store_config_cls, allow_overwrite=True)


__all__ = [
    "VectorStore",
    "VectorStoreFactory",
    "VectorBackendRegistry",
    "vector_backends",
]

# Import specialized stores when available
try:
    from .faiss import FaissVectorStore

    __all__ += ["FaissVectorStore"]
except ImportError:
    pass

try:
    from .chroma import ChromaVectorStore

    __all__ += ["ChromaVectorStore"]
except ImportError:
    pass

try:
    from .pgvector import PgVectorStore

    __all__ += ["PgVectorStore"]
except ImportError:
    pass

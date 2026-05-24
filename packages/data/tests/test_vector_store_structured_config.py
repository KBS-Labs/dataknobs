"""Construction-path tests for the vector-store structured-config refactor.

The four vector stores (memory, faiss, chroma, pgvector) are constructed
from a typed ``<Backend>VectorStoreConfig`` consumed via
:class:`~dataknobs_common.structured_config.StructuredConfigConsumer`,
mirroring the ``dataknobs-data`` database backends.

These tests pin the unified contract: the config dataclass hierarchy
projects documented keys and round-trips; dict, typed-config, and
``from_config`` paths reach identical state; mixing a typed config with
loose kwargs raises ``TypeError``; ``store.config`` is the typed config
(not a dict); per-backend validation surfaces ``ValueError`` /
``ImportError`` at construction; and the parity guard
(:func:`assert_structured_config_consumer`) holds, including the
MRO-ordering check.

Construction only — no external service is required. Optional-dependency
backends are gated by ``@requires_*`` markers.
"""

from __future__ import annotations

import pytest
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_common.testing import (
    assert_structured_config_consumer,
    assert_structured_config_roundtrip,
    requires_chromadb,
    requires_faiss,
    requires_package,
)

from dataknobs_data.vector.stores.config import (
    ChromaVectorStoreConfig,
    FaissVectorStoreConfig,
    MemoryVectorStoreConfig,
    PgVectorStoreConfig,
    VectorStoreConfig,
    VectorStoreTimestampConfig,
)
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.vector.types import DistanceMetric

_PG_DSN = "postgresql://user:pass@host:5432/db"

_POSTGRES_ENV_KEYS = (
    "DATABASE_URL",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


@pytest.fixture
def clear_postgres_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from ambient postgres env vars and dotenv files.

    ``PgVectorStoreConfig`` resolves the connection through
    ``normalize_postgres_connection_config``, which reads ``POSTGRES_*`` /
    ``DATABASE_URL`` env vars and dotenv fallbacks. The dev environment
    (``bin/dk up`` / project dotenv) sets these, so the "no connection ⇒
    ValueError" assertion must run with them cleared — otherwise the
    normalizer resolves an ambient connection and no error is raised.
    """
    for key in _POSTGRES_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "dataknobs_common.postgres_config._load_dotenv_fallbacks",
        lambda _start_path=None: {},
    )


# ---------------------------------------------------------------------------
# Config dataclass hierarchy
# ---------------------------------------------------------------------------


class TestConfigHierarchy:
    """The config dataclasses inherit the shared key set correctly."""

    def test_leaf_configs_inherit_base(self) -> None:
        for leaf in (
            MemoryVectorStoreConfig,
            FaissVectorStoreConfig,
            ChromaVectorStoreConfig,
            PgVectorStoreConfig,
        ):
            assert issubclass(leaf, VectorStoreConfig)

    def test_base_defaults(self) -> None:
        cfg = VectorStoreConfig()
        assert cfg.dimensions == 0
        assert cfg.metric == "cosine"
        assert cfg.persist_path is None
        assert cfg.batch_size == 100
        assert cfg.index_params == {}
        assert cfg.domain_id is None
        assert cfg.timestamps is None

    def test_from_dict_projects_documented_keys(self) -> None:
        cfg = VectorStoreConfig.from_dict(
            {"dimensions": 768, "metric": "euclidean", "domain_id": "acme"}
        )
        assert cfg.dimensions == 768
        assert cfg.metric == "euclidean"
        assert cfg.domain_id == "acme"

    def test_from_dict_ignores_routing_keys(self) -> None:
        # The factory passes the whole config dict, including the
        # ``backend`` routing key — unknown keys must pass through.
        cfg = MemoryVectorStoreConfig.from_dict(
            {"backend": "memory", "dimensions": 16}
        )
        assert cfg.dimensions == 16

    def test_nested_timestamp_config_composes(self) -> None:
        cfg = VectorStoreConfig.from_dict(
            {"timestamps": {"format": "epoch", "created_key": "made_at"}}
        )
        assert isinstance(cfg.timestamps, VectorStoreTimestampConfig)
        assert cfg.timestamps.format == "epoch"
        assert cfg.timestamps.created_key == "made_at"

    def test_timestamp_format_validation(self) -> None:
        with pytest.raises(ValueError, match=r"timestamps\.format"):
            VectorStoreTimestampConfig(format="nanoseconds")

    def test_roundtrip_flat_and_nested(self) -> None:
        assert_structured_config_roundtrip(MemoryVectorStoreConfig())
        assert_structured_config_roundtrip(
            VectorStoreConfig(
                dimensions=8,
                metric="dot_product",
                timestamps=VectorStoreTimestampConfig(format="datetime"),
            )
        )


# ---------------------------------------------------------------------------
# Memory backend — construction parity
# ---------------------------------------------------------------------------


class TestMemoryConstructionParity:
    """Dict, typed-config, and ``from_config`` paths agree."""

    def test_is_structured_config_consumer(self) -> None:
        assert issubclass(MemoryVectorStore, StructuredConfigConsumer)
        assert MemoryVectorStore.CONFIG_CLS is MemoryVectorStoreConfig

    def test_parity_guard(self) -> None:
        assert_structured_config_consumer(MemoryVectorStore)

    def test_self_config_is_typed_not_dict(self) -> None:
        store = MemoryVectorStore({"dimensions": 8})
        assert isinstance(store.config, MemoryVectorStoreConfig)

    def test_dict_and_typed_reach_identical_state(self) -> None:
        from_dict = MemoryVectorStore({"dimensions": 8, "metric": "euclidean"})
        from_typed = MemoryVectorStore(
            MemoryVectorStoreConfig(dimensions=8, metric="euclidean")
        )
        for store in (from_dict, from_typed):
            assert store.dimensions == 8
            assert store.metric == DistanceMetric.EUCLIDEAN
            assert store.vectors == {}
            assert store.metadata_store == {}

    def test_from_config_path(self) -> None:
        store = MemoryVectorStore.from_config({"dimensions": 8})
        assert isinstance(store, MemoryVectorStore)
        assert store.dimensions == 8

    def test_mixing_typed_config_with_kwargs_raises(self) -> None:
        with pytest.raises(TypeError):
            MemoryVectorStore(
                MemoryVectorStoreConfig(dimensions=8), dimensions=16
            )

    def test_derived_metric_enum(self) -> None:
        store = MemoryVectorStore({"dimensions": 4, "metric": "dot_product"})
        assert store.metric == DistanceMetric.DOT_PRODUCT

    def test_timestamp_keys_from_nested_config(self) -> None:
        store = MemoryVectorStore(
            {"dimensions": 4, "timestamps": {"format": "epoch", "created_key": "c"}}
        )
        assert store.timestamps_format == "epoch"
        assert store.timestamps_created_key == "c"
        assert store.timestamps_updated_key == "_updated_at"


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------


class TestFaissConfig:
    def test_parity_guard(self) -> None:
        from dataknobs_data.vector.stores.faiss import FaissVectorStore

        assert_structured_config_consumer(FaissVectorStore)
        assert FaissVectorStore.CONFIG_CLS is FaissVectorStoreConfig

    def test_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            FaissVectorStoreConfig(dimensions=8, index_type="ivfflat")
        )

    @requires_faiss
    def test_index_type_explicit_wins(self) -> None:
        from dataknobs_data.vector.stores.faiss import FaissVectorStore

        store = FaissVectorStore(
            {"dimensions": 8, "index_type": "hnsw", "index_params": {"type": "flat"}}
        )
        assert store.index_type == "hnsw"

    @requires_faiss
    def test_index_type_falls_back_to_index_params(self) -> None:
        from dataknobs_data.vector.stores.faiss import FaissVectorStore

        store = FaissVectorStore(
            {"dimensions": 8, "index_params": {"type": "ivfflat", "nlist": 50}}
        )
        assert store.index_type == "ivfflat"
        assert store.nlist == 50

    @requires_faiss
    def test_index_type_default_auto(self) -> None:
        from dataknobs_data.vector.stores.faiss import FaissVectorStore

        store = FaissVectorStore({"dimensions": 8})
        assert store.index_type == "auto"


# ---------------------------------------------------------------------------
# Chroma backend
# ---------------------------------------------------------------------------


class TestChromaConfig:
    def test_parity_guard(self) -> None:
        from dataknobs_data.vector.stores.chroma import ChromaVectorStore

        assert_structured_config_consumer(ChromaVectorStore)
        assert ChromaVectorStore.CONFIG_CLS is ChromaVectorStoreConfig

    def test_dimensions_default_384(self) -> None:
        cfg = ChromaVectorStoreConfig.from_dict({"collection_name": "docs"})
        assert cfg.dimensions == 384

    def test_explicit_dimensions_preserved(self) -> None:
        cfg = ChromaVectorStoreConfig.from_dict({"dimensions": 512})
        assert cfg.dimensions == 512

    def test_scalar_metadata_keys_normalized_to_frozenset(self) -> None:
        cfg = ChromaVectorStoreConfig.from_dict(
            {"scalar_metadata_keys": ["domain_id", "tenant"]}
        )
        assert cfg.scalar_metadata_keys == frozenset({"domain_id", "tenant"})

    def test_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            ChromaVectorStoreConfig.from_dict(
                {"dimensions": 384, "scalar_metadata_keys": ["domain_id"]}
            )
        )

    @requires_chromadb
    def test_construct_sets_collection_and_metric(self) -> None:
        from dataknobs_data.vector.stores.chroma import ChromaVectorStore

        store = ChromaVectorStore({"collection_name": "docs", "metric": "euclidean"})
        assert store.collection_name == "docs"
        assert store.dimensions == 384
        assert store.chroma_metric == "l2"


# ---------------------------------------------------------------------------
# pgvector backend
# ---------------------------------------------------------------------------


class TestPgVectorConfig:
    def test_parity_guard(self) -> None:
        from dataknobs_data.vector.stores.pgvector import PgVectorStore

        assert_structured_config_consumer(PgVectorStore)
        assert PgVectorStore.CONFIG_CLS is PgVectorStoreConfig

    def test_connection_string_resolved(self) -> None:
        cfg = PgVectorStoreConfig.from_dict(
            {"connection_string": _PG_DSN, "dimensions": 768}
        )
        assert cfg.connection_string == _PG_DSN

    def test_missing_connection_raises_value_error(
        self, clear_postgres_env: None
    ) -> None:
        with pytest.raises(ValueError, match="requires a postgres connection"):
            PgVectorStoreConfig.from_dict({"dimensions": 768})

    def test_invalid_id_type_raises(self) -> None:
        with pytest.raises(ValueError, match="id_type"):
            PgVectorStoreConfig.from_dict(
                {"connection_string": _PG_DSN, "id_type": "bogus"}
            )

    def test_invalid_index_type_raises(self) -> None:
        with pytest.raises(ValueError, match="index_type"):
            PgVectorStoreConfig.from_dict(
                {"connection_string": _PG_DSN, "index_type": "bogus"}
            )

    def test_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            PgVectorStoreConfig.from_dict(
                {"connection_string": _PG_DSN, "dimensions": 768, "id_type": "uuid"}
            )
        )

    @requires_package("asyncpg")
    def test_construct_merges_columns_and_quotes_identifiers(self) -> None:
        from dataknobs_data.vector.stores.pgvector import PgVectorStore

        store = PgVectorStore(
            {
                "connection_string": _PG_DSN,
                "dimensions": 768,
                "table_name": "embeds",
                "schema": "rag",
                "columns": {"id": "pk"},
            }
        )
        # User override merged over DEFAULT_COLUMNS.
        assert store.columns["id"] == "pk"
        assert store.columns["embedding"] == "embedding"
        assert store._q_qualified == '"rag"."embeds"'

    @requires_package("asyncpg")
    def test_dict_and_typed_reach_identical_state(self) -> None:
        from dataknobs_data.vector.stores.pgvector import PgVectorStore

        from_dict = PgVectorStore({"connection_string": _PG_DSN, "dimensions": 768})
        from_typed = PgVectorStore(
            PgVectorStoreConfig.from_dict(
                {"connection_string": _PG_DSN, "dimensions": 768}
            )
        )
        for store in (from_dict, from_typed):
            assert store.connection_string == _PG_DSN
            assert store.dimensions == 768
            assert store.index_type == "none"
            assert store.id_type == "text"

"""Construction-path tests for the structured-config backend refactor.

The 14 ``SyncDatabase`` / ``AsyncDatabase`` backends are migrated, phase
by phase, from the legacy dict-as-``self.config`` construction
(``ConfigurableBase`` + hand-rolled ``__init__``/``from_config``) to a
typed ``<Backend>DatabaseConfig`` consumed via
:class:`~dataknobs_common.structured_config.StructuredConfigConsumer`.

These tests pin the unified contract per migrated backend: typed-config,
dict, ``from_config``, and factory paths all reach identical state;
mixing a typed config with loose kwargs raises ``TypeError``;
``self.config`` is the typed config (not a dict); the public
``Database(config=..., schema=...)`` kwarg still lands on the typed
config's ``schema`` field; and the parity guard
(:func:`assert_structured_config_consumer`) holds — including the
MRO-ordering check (consumer mixin must precede other bases).

No external service is required — construction only. Each backend
family is added to this module as its migration phase lands.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_common.testing import (
    assert_structured_config_consumer,
    assert_structured_config_roundtrip,
)

from dataknobs_data import Record, async_database_factory, database_factory
from dataknobs_data.backends.config import (
    AsyncDuckDBDatabaseConfig,
    AsyncElasticsearchDatabaseConfig,
    AsyncS3DatabaseConfig,
    AsyncSQLiteDatabaseConfig,
    DatabaseConfig,
    DuckDBDatabaseConfigBase,
    ElasticsearchDatabaseConfigBase,
    FileDatabaseConfig,
    MemoryDatabaseConfig,
    PostgresDatabaseConfig,
    S3DatabaseConfigBase,
    SQLiteDatabaseConfigBase,
    SyncDuckDBDatabaseConfig,
    SyncElasticsearchDatabaseConfig,
    SyncS3DatabaseConfig,
    SyncSQLiteDatabaseConfig,
    VectorBackendConfig,
)
from dataknobs_data.backends.duckdb import (
    AsyncDuckDBDatabase,
    SyncDuckDBDatabase,
)
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import (
    AsyncElasticsearchDatabase,
)
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import (
    AsyncMemoryDatabase,
    SyncMemoryDatabase,
)
from dataknobs_data.backends.postgres import (
    AsyncPostgresDatabase,
    SyncPostgresDatabase,
)
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.schema import DatabaseSchema
from dataknobs_data.vector.types import DistanceMetric

# ---------------------------------------------------------------------------
# Config dataclass hierarchy
# ---------------------------------------------------------------------------


class TestConfigHierarchy:
    """The config dataclasses inherit shared key sets correctly."""

    def test_memory_config_inherits_vector_and_schema_fields(self) -> None:
        assert issubclass(MemoryDatabaseConfig, VectorBackendConfig)
        assert issubclass(VectorBackendConfig, DatabaseConfig)
        cfg = MemoryDatabaseConfig()
        assert cfg.schema is None
        assert cfg.vector_enabled is False
        assert cfg.vector_metric == "cosine"

    def test_from_dict_projects_documented_keys(self) -> None:
        cfg = MemoryDatabaseConfig.from_dict(
            {"vector_enabled": True, "vector_metric": "euclidean"}
        )
        assert cfg.vector_enabled is True
        assert cfg.vector_metric == "euclidean"

    def test_from_dict_ignores_routing_keys(self) -> None:
        # The factory passes the whole config dict, including the
        # ``backend`` routing key — unknown keys must pass through.
        cfg = MemoryDatabaseConfig.from_dict(
            {"backend": "memory", "vector_enabled": True}
        )
        assert cfg.vector_enabled is True

    def test_schema_dict_normalized_to_database_schema(self) -> None:
        cfg = MemoryDatabaseConfig.from_dict(
            {"schema": {"fields": {"age": "integer"}}}
        )
        assert isinstance(cfg.schema, DatabaseSchema)
        assert "age" in cfg.schema.fields

    def test_schema_instance_passes_through(self) -> None:
        sch = DatabaseSchema.create(name=FieldType.STRING)
        cfg = MemoryDatabaseConfig.from_dict({"schema": sch})
        assert cfg.schema is sch

    def test_roundtrip_holds(self) -> None:
        assert_structured_config_roundtrip(MemoryDatabaseConfig())
        assert_structured_config_roundtrip(
            MemoryDatabaseConfig(vector_enabled=True, vector_metric="dot_product")
        )


# ---------------------------------------------------------------------------
# Memory backends — construction parity
# ---------------------------------------------------------------------------

_MEMORY_BACKENDS = [SyncMemoryDatabase, AsyncMemoryDatabase]


class TestMemoryConstructionParity:
    """Dict, typed-config, ``from_config``, and factory paths agree."""

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_is_structured_config_consumer(self, backend_cls: type) -> None:
        assert issubclass(backend_cls, StructuredConfigConsumer)
        assert backend_cls.CONFIG_CLS is MemoryDatabaseConfig

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_parity_guard(self, backend_cls: type) -> None:
        # Pins CONFIG_CLS, field-set match, and the MRO-ordering
        # check (StructuredConfigConsumer first among the bases).
        assert_structured_config_consumer(backend_cls)

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_self_config_is_typed_not_dict(self, backend_cls: type) -> None:
        db = backend_cls({"vector_enabled": True})
        assert isinstance(db.config, MemoryDatabaseConfig)

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_dict_and_typed_reach_identical_state(
        self, backend_cls: type
    ) -> None:
        from_dict = backend_cls(
            {"vector_enabled": True, "vector_metric": "euclidean"}
        )
        from_typed = backend_cls(
            MemoryDatabaseConfig(vector_enabled=True, vector_metric="euclidean")
        )
        assert from_dict.config == from_typed.config
        assert from_dict.vector_enabled == from_typed.vector_enabled is True
        assert (
            from_dict.vector_metric
            == from_typed.vector_metric
            == DistanceMetric.EUCLIDEAN
        )

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_typed_config_passes_through_identity(
        self, backend_cls: type
    ) -> None:
        cfg = MemoryDatabaseConfig(vector_enabled=True)
        db = backend_cls(cfg)
        assert db.config is cfg

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_from_config_matches_direct(self, backend_cls: type) -> None:
        db = backend_cls.from_config({"vector_enabled": True})
        assert db.vector_enabled is True
        assert isinstance(db.config, MemoryDatabaseConfig)

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_mixing_typed_config_with_kwargs_raises(
        self, backend_cls: type
    ) -> None:
        with pytest.raises(TypeError):
            backend_cls(MemoryDatabaseConfig(), vector_enabled=True)

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_empty_construction_uses_defaults(
        self, backend_cls: type
    ) -> None:
        db = backend_cls()
        assert db.config.vector_enabled is False
        assert db.vector_metric == DistanceMetric.COSINE

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_invalid_metric_falls_back_to_cosine(
        self, backend_cls: type
    ) -> None:
        db = backend_cls({"vector_metric": "not-a-metric"})
        assert db.vector_metric == DistanceMetric.COSINE


class TestMemorySchemaHandling:
    """The ``Database(config=..., schema=...)`` kwarg is preserved."""

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_schema_kwarg_lands_on_config_field(
        self, backend_cls: type
    ) -> None:
        sch = DatabaseSchema.create(name=FieldType.STRING)
        db = backend_cls(config={"vector_enabled": True}, schema=sch)
        assert db.config.schema is sch
        assert db.schema is sch

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_schema_dict_in_config_builds_schema(
        self, backend_cls: type
    ) -> None:
        db = backend_cls({"schema": {"fields": {"age": "integer"}}})
        assert isinstance(db.schema, DatabaseSchema)
        assert "age" in db.schema.fields

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_no_schema_yields_empty_schema(self, backend_cls: type) -> None:
        db = backend_cls()
        assert isinstance(db.schema, DatabaseSchema)


class TestMemoryCooperativeInit:
    """The cooperative construction chain wires every collaborator."""

    @pytest.mark.parametrize("backend_cls", _MEMORY_BACKENDS)
    def test_storage_lock_and_vector_state_initialized(
        self, backend_cls: type
    ) -> None:
        db = backend_cls({"vector_enabled": True})
        # ``_setup`` ran: storage + lock present.
        assert hasattr(db, "_storage")
        assert hasattr(db, "_lock")
        # Vector state (SQLiteVectorSupport / VectorConfigMixin) ran.
        assert hasattr(db, "_vector_dimensions")
        assert hasattr(db, "_vector_fields")
        # Base construction branch ran: schema set, ``_initialize`` done.
        assert db.schema is not None


class TestMemoryFactory:
    """Registry factories build the migrated backends unchanged."""

    def test_sync_factory_builds_memory(self) -> None:
        db = database_factory.create(backend="memory")
        assert isinstance(db, SyncMemoryDatabase)
        assert isinstance(db.config, MemoryDatabaseConfig)

    def test_async_factory_builds_memory(self) -> None:
        db = async_database_factory.create(backend="memory")
        assert isinstance(db, AsyncMemoryDatabase)
        assert isinstance(db.config, MemoryDatabaseConfig)


class TestMemoryBehaviorRegression:
    """A migrated backend still performs basic CRUD."""

    def test_sync_crud(self) -> None:
        db = SyncMemoryDatabase()
        rid = db.create(Record({"name": "Alice"}))
        assert db.read(rid).get_value("name") == "Alice"

    @pytest.mark.asyncio
    async def test_async_crud(self) -> None:
        db = AsyncMemoryDatabase()
        rid = await db.create(Record({"name": "Bob"}))
        result = await db.read(rid)
        assert result.get_value("name") == "Bob"

    @pytest.mark.asyncio
    async def test_async_from_config_async_lazy_noop(self) -> None:
        # Async memory keeps lazy connect semantics: ``from_config_async``
        # is available but ``_ainit`` is a no-op (nothing connects eagerly).
        db = await AsyncMemoryDatabase.from_config_async({"vector_enabled": True})
        assert isinstance(db.config, MemoryDatabaseConfig)
        assert db.vector_enabled is True


# ---------------------------------------------------------------------------
# SQLite config dataclass hierarchy
# ---------------------------------------------------------------------------


class TestSQLiteConfigHierarchy:
    """Sync/async SQLite configs share a base and diverge on connection knobs."""

    def test_both_configs_share_base(self) -> None:
        assert issubclass(SyncSQLiteDatabaseConfig, SQLiteDatabaseConfigBase)
        assert issubclass(AsyncSQLiteDatabaseConfig, SQLiteDatabaseConfigBase)
        assert issubclass(SQLiteDatabaseConfigBase, VectorBackendConfig)

    def test_shared_defaults(self) -> None:
        for cfg in (SyncSQLiteDatabaseConfig(), AsyncSQLiteDatabaseConfig()):
            assert cfg.path == ":memory:"
            assert cfg.table == "records"
            assert cfg.timeout == 5.0
            assert cfg.journal_mode is None
            assert cfg.auto_create_table is True
            assert cfg.vector_enabled is False

    def test_divergent_fields(self) -> None:
        sync = SyncSQLiteDatabaseConfig()
        # ``synchronous`` defaults differ: sync leaves it unset, async NORMAL.
        assert sync.synchronous is None
        assert sync.check_same_thread is False
        async_cfg = AsyncSQLiteDatabaseConfig()
        assert async_cfg.synchronous == "NORMAL"
        assert async_cfg.pool_size == 5
        # The divergent knobs do not bleed across the sibling configs.
        assert not hasattr(sync, "pool_size")
        assert not hasattr(async_cfg, "check_same_thread")

    def test_from_dict_projects_keys(self) -> None:
        cfg = SyncSQLiteDatabaseConfig.from_dict(
            {
                "path": "/tmp/db.sqlite",
                "table": "items",
                "timeout": 1.5,
                "check_same_thread": True,
                "journal_mode": "WAL",
                "synchronous": "FULL",
                "vector_enabled": True,
            }
        )
        assert cfg.path == "/tmp/db.sqlite"
        assert cfg.table == "items"
        assert cfg.timeout == 1.5
        assert cfg.check_same_thread is True
        assert cfg.journal_mode == "WAL"
        assert cfg.synchronous == "FULL"
        assert cfg.vector_enabled is True

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("false", False), ("0", False), ("no", False), ("true", True)],
    )
    def test_auto_create_table_string_coercion(
        self, raw: str, expected: bool
    ) -> None:
        # YAML/env string values coerce as the legacy ``__init__`` did.
        cfg = SyncSQLiteDatabaseConfig.from_dict({"auto_create_table": raw})
        assert cfg.auto_create_table is expected

    def test_roundtrip_holds(self) -> None:
        assert_structured_config_roundtrip(SyncSQLiteDatabaseConfig())
        assert_structured_config_roundtrip(
            AsyncSQLiteDatabaseConfig(path="/tmp/x.db", pool_size=3)
        )


# ---------------------------------------------------------------------------
# SQLite backends — construction parity
# ---------------------------------------------------------------------------

# (backend_cls, config_cls) pairs — each SQLite backend has its own config.
_SQLITE_BACKENDS = [
    (SyncSQLiteDatabase, SyncSQLiteDatabaseConfig),
    (AsyncSQLiteDatabase, AsyncSQLiteDatabaseConfig),
]


class TestSQLiteConstructionParity:
    """Dict, typed-config, ``from_config``, and factory paths agree."""

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_is_structured_config_consumer(
        self, backend_cls: type, config_cls: type
    ) -> None:
        assert issubclass(backend_cls, StructuredConfigConsumer)
        assert backend_cls.CONFIG_CLS is config_cls

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_parity_guard(self, backend_cls: type, config_cls: type) -> None:
        # Pins CONFIG_CLS, field-set match, and the MRO-ordering
        # check (StructuredConfigConsumer first among the bases).
        assert_structured_config_consumer(backend_cls)

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_self_config_is_typed_not_dict(
        self, backend_cls: type, config_cls: type
    ) -> None:
        db = backend_cls({"table": "items"})
        assert isinstance(db.config, config_cls)
        assert db.config.table == "items"

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_dict_and_typed_reach_identical_state(
        self, backend_cls: type, config_cls: type
    ) -> None:
        spec = {"table": "items", "timeout": 2.0, "vector_enabled": True}
        from_dict = backend_cls(spec)
        from_typed = backend_cls(config_cls(**spec))
        assert from_dict.config == from_typed.config
        assert from_dict.table_name == from_typed.table_name == "items"
        assert from_dict.timeout == from_typed.timeout == 2.0
        assert from_dict.vector_metric == from_typed.vector_metric

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_typed_config_passes_through_identity(
        self, backend_cls: type, config_cls: type
    ) -> None:
        cfg = config_cls(table="items")
        db = backend_cls(cfg)
        assert db.config is cfg

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_from_config_matches_direct(
        self, backend_cls: type, config_cls: type
    ) -> None:
        db = backend_cls.from_config({"table": "items"})
        assert isinstance(db.config, config_cls)
        assert db.table_name == "items"

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_mixing_typed_config_with_kwargs_raises(
        self, backend_cls: type, config_cls: type
    ) -> None:
        with pytest.raises(TypeError):
            backend_cls(config_cls(), table="items")

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_invalid_metric_falls_back_to_cosine(
        self, backend_cls: type, config_cls: type
    ) -> None:
        db = backend_cls({"vector_metric": "not-a-metric"})
        assert db.vector_metric == DistanceMetric.COSINE

    @pytest.mark.parametrize(("backend_cls", "config_cls"), _SQLITE_BACKENDS)
    def test_schema_kwarg_lands_on_config_field(
        self, backend_cls: type, config_cls: type
    ) -> None:
        sch = DatabaseSchema.create(name=FieldType.STRING)
        db = backend_cls(config={"table": "items"}, schema=sch)
        assert db.config.schema is sch
        assert db.schema is sch


class TestSQLiteAsyncJournalDefault:
    """The async backend's path-dependent ``journal_mode`` default holds."""

    def test_file_path_defaults_to_wal(self) -> None:
        db = AsyncSQLiteDatabase({"path": "/tmp/dk-test.sqlite"})
        assert db.journal_mode == "WAL"

    def test_memory_path_defaults_to_none(self) -> None:
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        assert db.journal_mode is None

    def test_explicit_journal_mode_honored(self) -> None:
        db = AsyncSQLiteDatabase(
            {"path": "/tmp/dk-test.sqlite", "journal_mode": "DELETE"}
        )
        assert db.journal_mode == "DELETE"


class TestSQLiteFactory:
    """Registry factories build the migrated SQLite backends unchanged."""

    def test_sync_factory_builds_sqlite(self) -> None:
        db = database_factory.create(backend="sqlite")
        assert isinstance(db, SyncSQLiteDatabase)
        assert isinstance(db.config, SyncSQLiteDatabaseConfig)

    def test_async_factory_builds_sqlite(self) -> None:
        db = async_database_factory.create(backend="sqlite")
        assert isinstance(db, AsyncSQLiteDatabase)
        assert isinstance(db.config, AsyncSQLiteDatabaseConfig)


class TestSQLiteBehaviorRegression:
    """A migrated SQLite backend still performs basic CRUD."""

    def test_sync_crud(self) -> None:
        db = SyncSQLiteDatabase()
        db.connect()
        try:
            rid = db.create(Record({"name": "Alice"}))
            assert db.read(rid).get_value("name") == "Alice"
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_async_crud(self) -> None:
        db = AsyncSQLiteDatabase()
        await db.connect()
        try:
            rid = await db.create(Record({"name": "Bob"}))
            result = await db.read(rid)
            assert result.get_value("name") == "Bob"
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Postgres backends — unified config (sync + async share one config class)
# ---------------------------------------------------------------------------

_POSTGRES_BACKENDS = [SyncPostgresDatabase, AsyncPostgresDatabase]


class TestPostgresConfigHierarchy:
    """One ``PostgresDatabaseConfig`` is the union shared by both backends."""

    def test_both_backends_share_one_config(self) -> None:
        assert issubclass(PostgresDatabaseConfig, VectorBackendConfig)
        assert SyncPostgresDatabase.CONFIG_CLS is PostgresDatabaseConfig
        assert AsyncPostgresDatabase.CONFIG_CLS is PostgresDatabaseConfig

    def test_defaults(self) -> None:
        cfg = PostgresDatabaseConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.database == "postgres"
        assert cfg.schema_name == "public"
        assert cfg.table == "records"
        assert cfg.ssl is None
        assert cfg.min_pool_size == 2
        assert cfg.max_pool_size == 5
        # The inherited structural-schema field is distinct from schema_name.
        assert cfg.schema is None

    def test_schema_key_maps_to_schema_name(self) -> None:
        # Postgres overloads `schema` as the SQL schema NAME (string).
        cfg = PostgresDatabaseConfig.from_dict({"schema": "myschema"})
        assert cfg.schema_name == "myschema"
        assert cfg.schema is None  # structural-schema field untouched

    def test_schema_wins_over_schema_name(self) -> None:
        cfg = PostgresDatabaseConfig.from_dict(
            {"schema": "winner", "schema_name": "loser"}
        )
        assert cfg.schema_name == "winner"

    def test_table_name_alias(self) -> None:
        assert PostgresDatabaseConfig.from_dict(
            {"table_name": "items"}
        ).table == "items"
        # `table` wins over the alias.
        assert PostgresDatabaseConfig.from_dict(
            {"table": "a", "table_name": "b"}
        ).table == "a"

    def test_connection_string_resolved_into_keys(self) -> None:
        cfg = PostgresDatabaseConfig.from_dict(
            {"connection_string": "postgresql://u:p@h:5433/db"}
        )
        assert cfg.host == "h"
        assert cfg.port == 5433
        assert cfg.database == "db"
        assert cfg.user == "u"

    def test_non_string_schema_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            PostgresDatabaseConfig.from_dict({"schema": object()})

    def test_invalid_table_identifier_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            PostgresDatabaseConfig.from_dict({"table": "bad name"})

    def test_bool_string_coercion(self) -> None:
        cfg = PostgresDatabaseConfig.from_dict(
            {"ensure_database": "false", "auto_create_table": "0"}
        )
        assert cfg.ensure_database is False
        assert cfg.auto_create_table is False

    def test_roundtrip_holds(self) -> None:
        assert_structured_config_roundtrip(PostgresDatabaseConfig())
        assert_structured_config_roundtrip(
            PostgresDatabaseConfig(
                schema_name="s", table="t", ssl="require", max_pool_size=9
            )
        )


class TestPostgresConstructionParity:
    """Both backends construct from the typed config (no server needed)."""

    @pytest.mark.parametrize("backend_cls", _POSTGRES_BACKENDS)
    def test_is_structured_config_consumer(self, backend_cls: type) -> None:
        assert issubclass(backend_cls, StructuredConfigConsumer)
        assert backend_cls.CONFIG_CLS is PostgresDatabaseConfig

    @pytest.mark.parametrize("backend_cls", _POSTGRES_BACKENDS)
    def test_parity_guard(self, backend_cls: type) -> None:
        assert_structured_config_consumer(backend_cls)

    @pytest.mark.parametrize("backend_cls", _POSTGRES_BACKENDS)
    def test_self_config_is_typed(self, backend_cls: type) -> None:
        db = backend_cls({"schema": "myschema", "table": "items"})
        assert isinstance(db.config, PostgresDatabaseConfig)
        assert db.schema_name == "myschema"
        assert db.table_name == "items"

    @pytest.mark.parametrize("backend_cls", _POSTGRES_BACKENDS)
    def test_from_config_matches_direct(self, backend_cls: type) -> None:
        db = backend_cls.from_config({"table": "items"})
        assert isinstance(db.config, PostgresDatabaseConfig)
        assert db.table_name == "items"

    @pytest.mark.parametrize("backend_cls", _POSTGRES_BACKENDS)
    def test_mixing_typed_config_with_kwargs_raises(
        self, backend_cls: type
    ) -> None:
        with pytest.raises(TypeError):
            backend_cls(PostgresDatabaseConfig(), table="items")

    @pytest.mark.parametrize("backend_cls", _POSTGRES_BACKENDS)
    def test_non_string_schema_raises_at_construction(
        self, backend_cls: type
    ) -> None:
        with pytest.raises(ConfigurationError):
            backend_cls({"schema": object()})


class TestPostgresSslUnification:
    """Both backends accept ``ssl`` — the corrected sync/async drift."""

    def test_async_passes_ssl_to_pool_config(self) -> None:
        db = AsyncPostgresDatabase({"ssl": "require"})
        assert db._pool_config.ssl == "require"

    def test_async_pool_sizing(self) -> None:
        db = AsyncPostgresDatabase({"min_pool_size": 3, "max_pool_size": 9})
        assert db._pool_config.min_size == 3
        assert db._pool_config.max_size == 9

    @pytest.mark.parametrize(
        ("ssl_value", "expected"),
        [("require", "require"), (True, "require"), (False, "disable"), (None, None)],
    )
    def test_sync_translates_ssl_to_sslmode(
        self, ssl_value: object, expected: str | None
    ) -> None:
        db = SyncPostgresDatabase({"ssl": ssl_value})
        assert db._sslmode == expected

    def test_sync_rejects_untranslatable_ssl(self) -> None:
        # An asyncpg-only value (e.g. an SSLContext stand-in) that psycopg2
        # cannot accept fails loud rather than silently degrading.
        with pytest.raises(ConfigurationError):
            SyncPostgresDatabase({"ssl": object()})


class TestPostgresFactory:
    """Registry factories build the migrated Postgres backends unchanged."""

    def test_sync_factory(self) -> None:
        db = database_factory.create(backend="postgres")
        assert isinstance(db, SyncPostgresDatabase)
        assert isinstance(db.config, PostgresDatabaseConfig)

    def test_async_factory(self) -> None:
        db = async_database_factory.create(backend="postgres")
        assert isinstance(db, AsyncPostgresDatabase)
        assert isinstance(db.config, PostgresDatabaseConfig)


# ---------------------------------------------------------------------------
# Elasticsearch (sync + async)
# ---------------------------------------------------------------------------
#
# The two ES backends use different connection mechanisms (sync:
# ``SimplifiedElasticsearchIndex`` over a single host/port; async: a pooled
# ``AsyncElasticsearch`` with full auth/TLS options), so — unlike Postgres —
# they keep sibling config dataclasses under a shared base rather than one
# unified config. Construction only; no ES server is contacted.


class TestElasticsearchConfigHierarchy:
    """Sibling configs share ``index`` / ``refresh`` via a common base."""

    def test_hierarchy(self) -> None:
        assert issubclass(ElasticsearchDatabaseConfigBase, VectorBackendConfig)
        assert issubclass(
            SyncElasticsearchDatabaseConfig, ElasticsearchDatabaseConfigBase
        )
        assert issubclass(
            AsyncElasticsearchDatabaseConfig, ElasticsearchDatabaseConfigBase
        )
        assert (
            SyncElasticsearchDatabase.CONFIG_CLS
            is SyncElasticsearchDatabaseConfig
        )
        assert (
            AsyncElasticsearchDatabase.CONFIG_CLS
            is AsyncElasticsearchDatabaseConfig
        )

    def test_sync_defaults(self) -> None:
        cfg = SyncElasticsearchDatabaseConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 9200
        assert cfg.index == "records"
        assert cfg.refresh is True
        assert cfg.default_vector_field == "embedding"
        assert cfg.vector_dimensions == 1536
        assert cfg.mappings is None and cfg.settings is None

    def test_async_defaults(self) -> None:
        cfg = AsyncElasticsearchDatabaseConfig()
        assert cfg.hosts is None
        assert cfg.host is None and cfg.port is None
        assert cfg.verify_certs is True
        assert cfg.ssl_show_warn is True
        assert cfg.index == "records"

    def test_roundtrip_holds(self) -> None:
        assert_structured_config_roundtrip(SyncElasticsearchDatabaseConfig())
        assert_structured_config_roundtrip(
            SyncElasticsearchDatabaseConfig(
                host="h", port=9201, index="i", vector_enabled=True
            )
        )
        assert_structured_config_roundtrip(AsyncElasticsearchDatabaseConfig())
        assert_structured_config_roundtrip(
            AsyncElasticsearchDatabaseConfig(
                hosts=["http://a:9200"], api_key="k", verify_certs=False
            )
        )


class TestElasticsearchConstructionParity:
    """Both ES backends construct from their typed config (no server)."""

    @pytest.mark.parametrize(
        ("backend_cls", "config_cls"),
        [
            (SyncElasticsearchDatabase, SyncElasticsearchDatabaseConfig),
            (AsyncElasticsearchDatabase, AsyncElasticsearchDatabaseConfig),
        ],
    )
    def test_parity_guard(self, backend_cls: type, config_cls: type) -> None:
        assert issubclass(backend_cls, StructuredConfigConsumer)
        assert backend_cls.CONFIG_CLS is config_cls
        assert_structured_config_consumer(backend_cls)

    def test_sync_self_config_is_typed(self) -> None:
        db = SyncElasticsearchDatabase({"host": "h", "index": "ix"})
        assert isinstance(db.config, SyncElasticsearchDatabaseConfig)
        assert db.host == "h"
        assert db.index_name == "ix"
        assert db.vector_fields == {}

    def test_sync_from_config_matches_direct(self) -> None:
        db = SyncElasticsearchDatabase.from_config({"index": "ix"})
        assert db.index_name == "ix"

    def test_sync_vector_metric_parsed(self) -> None:
        db = SyncElasticsearchDatabase(
            {"vector_enabled": True, "vector_metric": "euclidean"}
        )
        assert db._vector_enabled is True
        assert db.vector_metric is DistanceMetric.EUCLIDEAN

    def test_async_self_config_is_typed(self) -> None:
        db = AsyncElasticsearchDatabase({"index": "ix"})
        assert isinstance(db.config, AsyncElasticsearchDatabaseConfig)
        assert db.index_name == "ix"
        assert db.vector_enabled is False

    def test_async_host_port_derives_single_url(self) -> None:
        db = AsyncElasticsearchDatabase({"host": "es1", "port": 9202})
        assert db._pool_config.hosts == ["http://es1:9202"]

    def test_async_explicit_hosts_win(self) -> None:
        db = AsyncElasticsearchDatabase(
            {"hosts": ["https://a:9200", "https://b:9200"]}
        )
        assert db._pool_config.hosts == ["https://a:9200", "https://b:9200"]

    def test_async_auth_and_tls_forwarded(self) -> None:
        db = AsyncElasticsearchDatabase(
            {"api_key": "k", "verify_certs": False, "ca_certs": "/ca.pem"}
        )
        assert db._pool_config.api_key == "k"
        assert db._pool_config.verify_certs is False
        assert db._pool_config.ca_certs == "/ca.pem"

    @pytest.mark.parametrize(
        ("backend_cls", "config_cls"),
        [
            (SyncElasticsearchDatabase, SyncElasticsearchDatabaseConfig),
            (AsyncElasticsearchDatabase, AsyncElasticsearchDatabaseConfig),
        ],
    )
    def test_mixing_typed_config_with_kwargs_raises(
        self, backend_cls: type, config_cls: type
    ) -> None:
        with pytest.raises(TypeError):
            backend_cls(config_cls(), index="ix")


class TestElasticsearchFactory:
    """Registry factories build the migrated ES backends unchanged."""

    def test_sync_factory(self) -> None:
        db = database_factory.create(backend="elasticsearch", host="h")
        assert isinstance(db, SyncElasticsearchDatabase)
        assert isinstance(db.config, SyncElasticsearchDatabaseConfig)

    def test_async_factory(self) -> None:
        db = async_database_factory.create(backend="es", host="h")
        assert isinstance(db, AsyncElasticsearchDatabase)
        assert isinstance(db.config, AsyncElasticsearchDatabaseConfig)


# ---------------------------------------------------------------------------
# S3 (sync + async)
# ---------------------------------------------------------------------------
#
# Both S3 backends share ``S3DatabaseConfigBase`` (bucket + region/credential/
# endpoint surface, alias mapping, and the non-empty-``bucket`` invariant).
# ``prefix`` defaults differ (sync ``"records/"`` with trailing-slash
# normalization; async ``""``), so it lives on the subclasses. Construction
# only; no S3 service is contacted.


class TestS3ConfigHierarchy:
    """Shared base owns the connection surface + bucket validation."""

    def test_hierarchy(self) -> None:
        assert issubclass(S3DatabaseConfigBase, VectorBackendConfig)
        assert issubclass(SyncS3DatabaseConfig, S3DatabaseConfigBase)
        assert issubclass(AsyncS3DatabaseConfig, S3DatabaseConfigBase)
        assert SyncS3Database.CONFIG_CLS is SyncS3DatabaseConfig
        assert AsyncS3Database.CONFIG_CLS is AsyncS3DatabaseConfig

    def test_bucket_required(self) -> None:
        with pytest.raises(ValueError, match="requires 'bucket'"):
            SyncS3DatabaseConfig()
        with pytest.raises(ValueError, match="requires 'bucket'"):
            AsyncS3DatabaseConfig()

    def test_region_alias(self) -> None:
        cfg = SyncS3DatabaseConfig.from_dict({"bucket": "b", "region": "r"})
        assert cfg.region_name == "r"
        # Canonical wins over the alias.
        cfg2 = SyncS3DatabaseConfig.from_dict(
            {"bucket": "b", "region": "alias", "region_name": "canon"}
        )
        assert cfg2.region_name == "canon"

    def test_credential_aliases(self) -> None:
        cfg = SyncS3DatabaseConfig.from_dict(
            {
                "bucket": "b",
                "access_key_id": "AK",
                "secret_access_key": "SK",
                "session_token": "ST",
            }
        )
        assert cfg.aws_access_key_id == "AK"
        assert cfg.aws_secret_access_key == "SK"
        assert cfg.aws_session_token == "ST"

    def test_sync_prefix_normalization(self) -> None:
        assert SyncS3DatabaseConfig(bucket="b").prefix == "records/"
        assert SyncS3DatabaseConfig(bucket="b", prefix="recs").prefix == "recs/"
        assert (
            SyncS3DatabaseConfig(bucket="b", prefix="recs/").prefix == "recs/"
        )

    def test_async_prefix_default_empty(self) -> None:
        assert AsyncS3DatabaseConfig(bucket="b").prefix == ""

    def test_sync_pool_retry_aliases_and_coercion(self) -> None:
        cfg = SyncS3DatabaseConfig.from_dict(
            {"bucket": "b", "max_workers": "7", "max_retries": "4"}
        )
        assert cfg.max_pool_connections == 7
        assert cfg.max_attempts == 4

    def test_roundtrip_holds(self) -> None:
        assert_structured_config_roundtrip(SyncS3DatabaseConfig(bucket="b"))
        assert_structured_config_roundtrip(
            SyncS3DatabaseConfig(
                bucket="b", prefix="p/", region_name="r", max_pool_connections=4
            )
        )
        assert_structured_config_roundtrip(AsyncS3DatabaseConfig(bucket="b"))
        assert_structured_config_roundtrip(
            AsyncS3DatabaseConfig(bucket="b", prefix="p", region_name="r")
        )


class TestS3ConstructionParity:
    """Both S3 backends construct from their typed config (no service)."""

    @pytest.mark.parametrize(
        ("backend_cls", "config_cls"),
        [
            (SyncS3Database, SyncS3DatabaseConfig),
            (AsyncS3Database, AsyncS3DatabaseConfig),
        ],
    )
    def test_parity_guard(self, backend_cls: type, config_cls: type) -> None:
        assert issubclass(backend_cls, StructuredConfigConsumer)
        assert backend_cls.CONFIG_CLS is config_cls
        assert_structured_config_consumer(backend_cls)

    @pytest.mark.parametrize(
        "backend_cls", [SyncS3Database, AsyncS3Database]
    )
    def test_bucket_required_at_construction(self, backend_cls: type) -> None:
        with pytest.raises(ValueError, match="requires 'bucket'"):
            backend_cls({})

    def test_sync_self_config_is_typed(self) -> None:
        db = SyncS3Database({"bucket": "b", "region": "us-west-2", "prefix": "recs"})
        assert isinstance(db.config, SyncS3DatabaseConfig)
        assert db.bucket == "b"
        assert db.prefix == "recs/"
        assert db.region == "us-west-2"

    def test_sync_max_workers_alias_to_attribute(self) -> None:
        db = SyncS3Database({"bucket": "b", "max_workers": "7"})
        assert db.max_workers == 7
        assert db._session_config.max_pool_connections == 7

    def test_sync_from_config_matches_direct(self) -> None:
        db = SyncS3Database.from_config({"bucket": "b", "prefix": "p"})
        assert db.bucket == "b"
        assert db.prefix == "p/"

    def test_async_self_config_is_typed(self) -> None:
        db = AsyncS3Database(
            {"bucket": "b", "prefix": "p", "region": "eu-1", "access_key_id": "AK"}
        )
        assert isinstance(db.config, AsyncS3DatabaseConfig)
        assert db._pool_config.bucket == "b"
        assert db._pool_config.prefix == "p"
        assert db._pool_config.region_name == "eu-1"
        assert db._pool_config.aws_access_key_id == "AK"
        assert db.region == "eu-1"

    @pytest.mark.parametrize(
        ("backend_cls", "config_cls"),
        [
            (SyncS3Database, SyncS3DatabaseConfig),
            (AsyncS3Database, AsyncS3DatabaseConfig),
        ],
    )
    def test_mixing_typed_config_with_kwargs_raises(
        self, backend_cls: type, config_cls: type
    ) -> None:
        with pytest.raises(TypeError):
            backend_cls(config_cls(bucket="b"), prefix="p")


class TestS3Factory:
    """Registry factories build the migrated S3 backends unchanged."""

    def test_sync_factory(self) -> None:
        db = database_factory.create(backend="s3", bucket="b")
        assert isinstance(db, SyncS3Database)
        assert isinstance(db.config, SyncS3DatabaseConfig)

    def test_async_factory(self) -> None:
        db = async_database_factory.create(backend="s3", bucket="b")
        assert isinstance(db, AsyncS3Database)
        assert isinstance(db.config, AsyncS3DatabaseConfig)


# ---------------------------------------------------------------------------
# DuckDB (sync + async)
# ---------------------------------------------------------------------------
#
# DuckDB has no vector support, so its configs inherit ``DatabaseConfig``
# directly (NOT ``VectorBackendConfig``). The async backend adds a
# ``max_workers`` thread-pool knob. Construction only (``:memory:``).


class TestDuckDBConfigHierarchy:
    """DuckDB configs inherit DatabaseConfig (no vector knobs)."""

    def test_hierarchy(self) -> None:
        assert issubclass(DuckDBDatabaseConfigBase, DatabaseConfig)
        # DuckDB has no vector support — it must NOT pick up VectorBackendConfig.
        assert not issubclass(DuckDBDatabaseConfigBase, VectorBackendConfig)
        assert issubclass(SyncDuckDBDatabaseConfig, DuckDBDatabaseConfigBase)
        assert issubclass(AsyncDuckDBDatabaseConfig, DuckDBDatabaseConfigBase)
        assert SyncDuckDBDatabase.CONFIG_CLS is SyncDuckDBDatabaseConfig
        assert AsyncDuckDBDatabase.CONFIG_CLS is AsyncDuckDBDatabaseConfig

    def test_no_vector_fields(self) -> None:
        # vector_enabled / vector_metric live on VectorBackendConfig only.
        field_names = {
            f.name for f in dataclasses.fields(SyncDuckDBDatabaseConfig)
        }
        assert "vector_enabled" not in field_names
        assert "vector_metric" not in field_names

    def test_defaults(self) -> None:
        cfg = SyncDuckDBDatabaseConfig()
        assert cfg.path == ":memory:"
        assert cfg.table == "records"
        assert cfg.timeout == 5.0
        assert cfg.read_only is False
        assert cfg.auto_create_table is True
        assert AsyncDuckDBDatabaseConfig().max_workers == 4

    def test_auto_create_table_bool_coercion(self) -> None:
        assert SyncDuckDBDatabaseConfig.from_dict(
            {"auto_create_table": "false"}
        ).auto_create_table is False
        assert AsyncDuckDBDatabaseConfig.from_dict(
            {"auto_create_table": "0"}
        ).auto_create_table is False

    def test_roundtrip_holds(self) -> None:
        assert_structured_config_roundtrip(SyncDuckDBDatabaseConfig())
        assert_structured_config_roundtrip(
            SyncDuckDBDatabaseConfig(path="/tmp/x.duckdb", read_only=True)
        )
        assert_structured_config_roundtrip(
            AsyncDuckDBDatabaseConfig(max_workers=8)
        )


class TestDuckDBConstructionParity:
    """Both DuckDB backends construct from the typed config."""

    @pytest.mark.parametrize(
        ("backend_cls", "config_cls"),
        [
            (SyncDuckDBDatabase, SyncDuckDBDatabaseConfig),
            (AsyncDuckDBDatabase, AsyncDuckDBDatabaseConfig),
        ],
    )
    def test_parity_guard(self, backend_cls: type, config_cls: type) -> None:
        assert issubclass(backend_cls, StructuredConfigConsumer)
        assert backend_cls.CONFIG_CLS is config_cls
        assert_structured_config_consumer(backend_cls)

    def test_sync_self_config_is_typed(self) -> None:
        db = SyncDuckDBDatabase({"table": "items", "read_only": True})
        assert isinstance(db.config, SyncDuckDBDatabaseConfig)
        assert db.table_name == "items"
        assert db.read_only is True
        assert db.db_path == ":memory:"

    def test_sync_from_config_matches_direct(self) -> None:
        db = SyncDuckDBDatabase.from_config({"table": "items"})
        assert db.table_name == "items"

    def test_async_max_workers_builds_executor(self) -> None:
        db = AsyncDuckDBDatabase({"max_workers": 3})
        try:
            assert db.max_workers == 3
            assert db.executor._max_workers == 3
        finally:
            db.executor.shutdown(wait=False)

    @pytest.mark.parametrize(
        ("backend_cls", "config_cls"),
        [
            (SyncDuckDBDatabase, SyncDuckDBDatabaseConfig),
            (AsyncDuckDBDatabase, AsyncDuckDBDatabaseConfig),
        ],
    )
    def test_mixing_typed_config_with_kwargs_raises(
        self, backend_cls: type, config_cls: type
    ) -> None:
        with pytest.raises(TypeError):
            backend_cls(config_cls(), table="items")


class TestDuckDBFactory:
    """Registry factories build the migrated DuckDB backends unchanged."""

    def test_sync_factory(self) -> None:
        db = database_factory.create(backend="duckdb")
        assert isinstance(db, SyncDuckDBDatabase)
        assert isinstance(db.config, SyncDuckDBDatabaseConfig)

    def test_async_factory(self) -> None:
        db = async_database_factory.create(backend="duckdb")
        assert isinstance(db, AsyncDuckDBDatabase)
        assert isinstance(db.config, AsyncDuckDBDatabaseConfig)
        db.executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# File (sync + async)
# ---------------------------------------------------------------------------
#
# The two File backends share one ``FileDatabaseConfig`` (identical surface —
# only the temp-file prefix and lock primitive differ, which is backend
# logic). ``path=None`` selects a temp file; ``format`` is auto-detected.


class TestFileConfigHierarchy:
    """A single config backs both File backends."""

    def test_both_backends_share_one_config(self) -> None:
        assert issubclass(FileDatabaseConfig, VectorBackendConfig)
        assert SyncFileDatabase.CONFIG_CLS is FileDatabaseConfig
        assert AsyncFileDatabase.CONFIG_CLS is FileDatabaseConfig

    def test_defaults(self) -> None:
        cfg = FileDatabaseConfig()
        assert cfg.path is None
        assert cfg.format is None
        assert cfg.compression is None

    def test_roundtrip_holds(self) -> None:
        assert_structured_config_roundtrip(FileDatabaseConfig())
        assert_structured_config_roundtrip(
            FileDatabaseConfig(path="/tmp/x.json", format="json")
        )
        assert_structured_config_roundtrip(
            FileDatabaseConfig(
                path="/tmp/x.csv", format="csv", vector_enabled=True
            )
        )


class TestFileConstructionParity:
    """Both File backends construct from the shared typed config."""

    @pytest.mark.parametrize(
        "backend_cls", [SyncFileDatabase, AsyncFileDatabase]
    )
    def test_parity_guard(self, backend_cls: type) -> None:
        assert issubclass(backend_cls, StructuredConfigConsumer)
        assert backend_cls.CONFIG_CLS is FileDatabaseConfig
        assert_structured_config_consumer(backend_cls)

    def test_sync_temp_file_when_no_path(self) -> None:
        db = SyncFileDatabase({})
        try:
            assert isinstance(db.config, FileDatabaseConfig)
            assert db._is_temp_file is True
            assert "dataknobs_sync_db_" in db.filepath
        finally:
            db.close()

    def test_async_temp_prefix_differs(self) -> None:
        db = AsyncFileDatabase({})
        try:
            assert "dataknobs_async_db_" in db.filepath
        finally:
            asyncio.run(db.close())

    def test_sync_explicit_path_and_format_detection(self, tmp_path) -> None:
        target = tmp_path / "data.csv"
        db = SyncFileDatabase({"path": str(target)})
        try:
            assert db._is_temp_file is False
            assert db.format == "csv"
        finally:
            db.close()

    def test_sync_gzip_compression_detected(self, tmp_path) -> None:
        target = tmp_path / "data.json.gz"
        db = SyncFileDatabase({"path": str(target)})
        try:
            assert db.compression == "gzip"
            assert db.filepath.endswith(".gz")
        finally:
            db.close()

    def test_sync_vector_metric_parsed(self) -> None:
        db = SyncFileDatabase(
            {"vector_enabled": True, "vector_metric": "euclidean"}
        )
        try:
            assert db._vector_enabled is True
            assert db.vector_metric is DistanceMetric.EUCLIDEAN
        finally:
            db.close()

    def test_sync_from_config_matches_direct(self, tmp_path) -> None:
        target = tmp_path / "x.json"
        db = SyncFileDatabase.from_config({"path": str(target)})
        try:
            assert db.filepath == str(target)
        finally:
            db.close()

    @pytest.mark.parametrize(
        "backend_cls", [SyncFileDatabase, AsyncFileDatabase]
    )
    def test_mixing_typed_config_with_kwargs_raises(
        self, backend_cls: type
    ) -> None:
        with pytest.raises(TypeError):
            backend_cls(FileDatabaseConfig(), format="json")


class TestFileFactory:
    """Registry factories build the migrated File backends unchanged."""

    def test_sync_factory(self, tmp_path) -> None:
        db = database_factory.create(
            backend="file", path=str(tmp_path / "f.json")
        )
        try:
            assert isinstance(db, SyncFileDatabase)
            assert isinstance(db.config, FileDatabaseConfig)
        finally:
            db.close()

    def test_async_factory(self, tmp_path) -> None:
        db = async_database_factory.create(
            backend="file", path=str(tmp_path / "f.json")
        )
        try:
            assert isinstance(db, AsyncFileDatabase)
            assert isinstance(db.config, FileDatabaseConfig)
        finally:
            asyncio.run(db.close())

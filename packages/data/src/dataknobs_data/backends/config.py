"""Structured configuration dataclasses for database backends.

Every backend's documented config key is a typed dataclass field; the
auto-derived :meth:`StructuredConfig.from_dict
<dataknobs_common.structured_config.StructuredConfig.from_dict>`
classmethod is the single source of truth for translating a config dict
to typed construction. Backends mix in
:class:`~dataknobs_common.structured_config.StructuredConfigConsumer`
parameterized by their config dataclass, so the registry factories
collapse to one-line wrappers over ``<Backend>.from_config(config)`` and
drift between the backend's documented surface and its construction path
becomes structurally impossible.

The dataclasses are ``frozen=True`` so ``db.config`` is a safe read-only
window onto the construction parameters.

The hierarchy mirrors the shared key sets:

``DatabaseConfig`` (``schema``) is the root every backend config inherits.
``VectorBackendConfig`` adds the ``vector_enabled`` / ``vector_metric``
knobs shared by every backend except DuckDB (which has no vector support).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataknobs_common import normalize_postgres_connection_config
from dataknobs_common.structured_config import StructuredConfig

from ..database import extract_schema_from_config
from ..schema import DatabaseSchema
from .postgres_mixins import validate_pg_identifier
from .sql_base import SQLTableManager


@dataclass(frozen=True)
class DatabaseConfig(StructuredConfig):
    """Base configuration for every ``SyncDatabase`` / ``AsyncDatabase`` backend.

    The ``schema`` field carries the ``DatabaseSchema`` the base
    ``Database`` accepts today. ``_normalize_dict`` routes a dict-shaped
    ``schema`` through :func:`extract_schema_from_config` so it becomes a
    ``DatabaseSchema`` before field projection; a ``DatabaseSchema``
    instance passes through unchanged. This preserves the public
    ``Database(config=..., schema=...)`` kwarg: the consumer mixin merges
    ``schema=`` into the dict and this field captures it.

    Attributes:
        schema: Optional database schema. A dict is converted to a
            ``DatabaseSchema`` at construction; ``None`` yields an empty
            schema in the backend.
    """

    schema: DatabaseSchema | None = None

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert a dict-shaped ``schema`` to a ``DatabaseSchema``.

        Subclasses that override this for backend-specific normalization
        (e.g. Postgres connection assembly) must call
        ``super()._normalize_dict(raw)`` so the shared schema handling
        still runs.
        """
        if "schema" in raw and not isinstance(raw["schema"], DatabaseSchema):
            raw["schema"] = extract_schema_from_config(raw["schema"])
        return raw


@dataclass(frozen=True)
class VectorBackendConfig(DatabaseConfig):
    """Base configuration for backends with Python-side vector support.

    Shared by every backend except DuckDB. ``vector_metric`` is kept as
    the raw string (``"cosine"``, ``"euclidean"``, ...) the documented
    config accepts; the backend converts it to a
    :class:`~dataknobs_data.vector.types.DistanceMetric` during ``_setup``
    (an unrecognized value falls back to cosine with a warning, matching
    the legacy ``_parse_vector_config`` behavior).

    Attributes:
        vector_enabled: Whether vector operations are enabled.
        vector_metric: Distance-metric name for vector similarity.
    """

    vector_enabled: bool = False
    vector_metric: str = "cosine"


@dataclass(frozen=True)
class MemoryDatabaseConfig(VectorBackendConfig):
    """Configuration for ``SyncMemoryDatabase`` / ``AsyncMemoryDatabase``.

    The in-memory backends have no construction parameters beyond the
    shared schema + vector knobs; the dataclass exists for structural
    symmetry so every backend exposes the same ``config`` /
    ``from_config`` surface.
    """


@dataclass(frozen=True)
class SQLiteDatabaseConfigBase(VectorBackendConfig):
    """Shared SQLite configuration for the sync and async backends.

    The two SQLite backends diverge on connection management — the sync
    backend exposes ``check_same_thread`` (a stdlib ``sqlite3`` knob)
    while the async backend (aiosqlite) exposes ``pool_size`` and
    defaults ``synchronous`` to ``"NORMAL"``. Everything they share lives
    here; the divergent fields live on the sibling subclasses so each
    backend's documented surface is exactly its config dataclass.

    ``auto_create_table`` is coerced through
    :meth:`SQLTableManager.coerce_bool` in ``__post_init__`` so YAML/env
    string values (``"false"``, ``"0"``) behave as they did when the
    legacy ``__init__`` coerced them.

    Attributes:
        path: Database file path (``":memory:"`` for in-memory).
        table: Records table name.
        timeout: Connection timeout in seconds.
        journal_mode: SQLite journal mode (``WAL``, ``DELETE``, ...).
        synchronous: SQLite synchronous mode (``NORMAL``, ``FULL``, ``OFF``).
        auto_create_table: Create the records table on connect if missing.
    """

    path: str = ":memory:"
    table: str = "records"
    timeout: float = 5.0
    journal_mode: str | None = None
    synchronous: str | None = None
    auto_create_table: bool = True

    def __post_init__(self) -> None:
        # Match the legacy ``__init__`` which always coerced this knob,
        # so YAML/env string values ("false", "0", "no") behave correctly.
        object.__setattr__(
            self,
            "auto_create_table",
            SQLTableManager.coerce_bool(self.auto_create_table),
        )


@dataclass(frozen=True)
class SyncSQLiteDatabaseConfig(SQLiteDatabaseConfigBase):
    """Configuration for ``SyncSQLiteDatabase``.

    Adds ``check_same_thread`` — the stdlib ``sqlite3`` knob controlling
    whether a connection may be shared across threads — which has no
    aiosqlite equivalent.

    Attributes:
        check_same_thread: Allow the connection to be used across threads.
    """

    check_same_thread: bool = False


@dataclass(frozen=True)
class AsyncSQLiteDatabaseConfig(SQLiteDatabaseConfigBase):
    """Configuration for ``AsyncSQLiteDatabase``.

    Adds ``pool_size`` (aiosqlite connection pool depth) and defaults
    ``synchronous`` to ``"NORMAL"``, matching the legacy async ``__init__``.
    The async backend also defaults ``journal_mode`` to ``"WAL"`` for
    file-based databases; because that default depends on ``path`` it is
    resolved in the backend's ``_setup`` (a static field default cannot
    express it), so the field itself stays ``None`` here.

    Attributes:
        pool_size: Number of pooled aiosqlite connections.
    """

    synchronous: str | None = "NORMAL"
    pool_size: int = 5


@dataclass(frozen=True)
class PostgresDatabaseConfig(VectorBackendConfig):
    """Unified configuration for ``SyncPostgresDatabase`` / ``AsyncPostgresDatabase``.

    A single config class backs **both** Postgres backends — the union of
    their parameters — so the two backends accept an identical connection
    surface (correcting prior sync/async drift where only the async
    backend honored ``ssl``).

    **Connection layer.** ``host`` / ``port`` / ``database`` / ``user`` /
    ``password`` are resolved in ``_normalize_dict`` through
    :func:`normalize_postgres_connection_config`, which folds in a
    ``connection_string`` and the ``POSTGRES_*`` / ``DATABASE_URL`` env
    vars (the single env-var contract; ``require=False`` so resolvability
    is deferred to ``connect()`` as the backends historically did).

    **SSL.** ``ssl`` keeps asyncpg-native semantics (``bool`` / ``str`` /
    ``ssl.SSLContext``). The async backend passes it straight to asyncpg;
    the sync backend translates it to a psycopg2 ``sslmode`` in
    ``connect()`` (``str`` → that mode, ``True`` → ``"require"``,
    ``False`` → ``"disable"``, and an unsupported value such as an
    ``SSLContext`` raises rather than silently degrading).

    **Schema-name overload.** Postgres uses the ``schema`` config key for
    the SQL *schema name* (a string), which collides with the base
    :attr:`DatabaseConfig.schema` (a ``DatabaseSchema``). ``_normalize_dict``
    maps a non-``None`` ``schema`` to ``schema_name`` (``schema`` wins over
    an explicit ``schema_name``, matching legacy precedence); a ``None``
    ``schema`` is the inherited structural-schema default and is left for
    the base. Identifiers are validated in ``__post_init__``.

    Attributes:
        host/port/database/user/password: Connection parameters
            (resolved from env / ``connection_string`` / explicit keys).
        ssl: SSL configuration (asyncpg-native; sync translates to ``sslmode``).
        command_timeout: asyncpg command timeout in seconds. **Async-only**
            (psycopg2 has no equivalent connect-time knob).
        min_pool_size/max_pool_size: asyncpg pool bounds. **Async-only**
            (inapplicable to the single synchronous psycopg2 connection).
        table: Records table name.
        schema_name: SQL schema name.
        ensure_database: Auto-create the database if missing.
        auto_create_table: Create the records table on connect if missing.
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"
    user: str = "postgres"
    password: str = ""
    ssl: Any | None = None
    command_timeout: float | None = None
    min_pool_size: int = 2
    max_pool_size: int = 5
    table: str = "records"
    schema_name: str = "public"
    ensure_database: bool = True
    auto_create_table: bool = True

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        # Map the overloaded ``schema`` key (SQL schema name) to
        # ``schema_name`` before the base routes ``schema`` through
        # ``extract_schema_from_config``. A ``None`` ``schema`` is the
        # inherited structural-schema default emitted by ``to_dict`` —
        # leave it for the base (keeps round-trip stable). A non-``None``
        # ``schema`` wins over an explicit ``schema_name`` (legacy
        # precedence); a non-identifier value is caught in ``__post_init__``.
        if "schema" in raw and raw["schema"] is not None:
            raw["schema_name"] = raw.pop("schema")
        # ``table`` wins over the ``table_name`` alias (legacy precedence).
        if "table_name" in raw:
            if "table" not in raw:
                raw["table"] = raw["table_name"]
            del raw["table_name"]

        raw = super()._normalize_dict(raw)

        # Resolve the connection layer (env / connection_string / explicit
        # keys) into canonical individual keys via the shared normalizer.
        conn_keys = (
            "host",
            "port",
            "database",
            "user",
            "password",
            "connection_string",
        )
        conn_input = {k: raw[k] for k in conn_keys if k in raw}
        normalized = normalize_postgres_connection_config(conn_input, require=False)
        if normalized is not None:
            for k in ("host", "port", "database", "user", "password"):
                if k in normalized:
                    raw[k] = normalized[k]
        # ``connection_string`` is an input alias fully resolved into the
        # individual keys above; drop it so it doesn't linger as an
        # unknown key.
        raw.pop("connection_string", None)
        return raw

    def __post_init__(self) -> None:
        # Validate identifiers early (a non-string ``schema``/``table`` —
        # e.g. a DatabaseSchema injected via the key collision — fails
        # fast with a clear ConfigurationError rather than emitting broken
        # DDL at first query).
        object.__setattr__(
            self, "table", validate_pg_identifier(self.table, "table")
        )
        object.__setattr__(
            self, "schema_name", validate_pg_identifier(self.schema_name, "schema")
        )
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(
            self,
            "ensure_database",
            SQLTableManager.coerce_bool(self.ensure_database, default=True),
        )
        object.__setattr__(
            self,
            "auto_create_table",
            SQLTableManager.coerce_bool(self.auto_create_table, default=True),
        )


@dataclass(frozen=True)
class ElasticsearchDatabaseConfigBase(VectorBackendConfig):
    """Shared Elasticsearch configuration for the sync and async backends.

    The two Elasticsearch backends have diverged on connection management
    (and that drift is preserved here rather than papered over): the sync
    backend talks to a single ``host``/``port`` through
    ``SimplifiedElasticsearchIndex`` and supports custom ``mappings`` /
    ``settings`` and a default vector field; the async backend connects
    through a pooled ``AsyncElasticsearch`` client with full auth/TLS
    options (``hosts``, ``api_key``, ``basic_auth``, ``verify_certs``, …).
    Only ``index`` and ``refresh`` are genuinely shared; everything else
    lives on the sibling subclasses so each backend's documented surface
    is exactly its config dataclass.

    Attributes:
        index: Elasticsearch index name.
        refresh: Whether to refresh the index after write operations.
    """

    index: str = "records"
    refresh: bool = True


@dataclass(frozen=True)
class SyncElasticsearchDatabaseConfig(ElasticsearchDatabaseConfigBase):
    """Configuration for ``SyncElasticsearchDatabase``.

    The sync backend connects to a single ``host``/``port`` and builds a
    ``SimplifiedElasticsearchIndex``. ``vector_dimensions`` /
    ``default_vector_field`` seed a default dense-vector field at connect
    time when vector support is enabled but no fields have been observed
    yet; ``mappings`` / ``settings`` override the derived index config.

    Attributes:
        host: Elasticsearch host.
        port: Elasticsearch port.
        vector_dimensions: Dimensions for the default vector field.
        default_vector_field: Name of the default vector field.
        mappings: Custom index mappings (overrides the derived mappings).
        settings: Custom index settings (overrides the derived settings).
    """

    host: str = "localhost"
    port: int = 9200
    vector_dimensions: int = 1536
    default_vector_field: str = "embedding"
    mappings: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None


@dataclass(frozen=True)
class AsyncElasticsearchDatabaseConfig(ElasticsearchDatabaseConfigBase):
    """Configuration for ``AsyncElasticsearchDatabase``.

    The async backend connects through a pooled ``AsyncElasticsearch``
    client. The connection fields mirror
    :class:`~dataknobs_data.pooling.elasticsearch.ElasticsearchPoolConfig`
    (the pool config is derived from them in the backend's ``_setup``):
    ``hosts`` wins when set, otherwise ``host`` / ``port`` are composed
    into a single URL.

    Attributes:
        hosts: Explicit list of Elasticsearch host URLs.
        host: Single host (composed with ``port`` when ``hosts`` is unset).
        port: Port for the single-host form.
        api_key: API key for authentication.
        basic_auth: ``(user, password)`` tuple for basic auth.
        verify_certs: Verify TLS certificates.
        ca_certs: Path to a CA bundle.
        client_cert: Path to a client certificate.
        client_key: Path to a client key.
        ssl_show_warn: Show TLS warnings.
    """

    hosts: list[str] | None = None
    host: str | None = None
    port: int | None = None
    api_key: str | None = None
    basic_auth: tuple | None = None
    verify_certs: bool = True
    ca_certs: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    ssl_show_warn: bool = True


@dataclass(frozen=True)
class S3DatabaseConfigBase(VectorBackendConfig):
    """Shared S3 configuration for the sync and async backends.

    Both S3 backends route region/credential/endpoint resolution through
    :class:`~dataknobs_data.pooling.s3.S3SessionConfig` /
    :class:`~dataknobs_data.pooling.s3.S3PoolConfig`. This base captures
    the connection surface those normalizers consume and maps the legacy
    aliases (``region``, ``access_key_id`` / ``secret_access_key`` /
    ``session_token``) onto the canonical keys in ``_normalize_dict`` so
    the typed config is canonical. ``bucket`` is validated non-empty in
    ``__post_init__`` (it has no usable default — every S3 backend
    requires it).

    Attributes:
        bucket: S3 bucket name (required).
        region_name: AWS region (alias: ``region``).
        aws_access_key_id: AWS access key (alias: ``access_key_id``).
        aws_secret_access_key: AWS secret key (alias: ``secret_access_key``).
        aws_session_token: AWS session token (alias: ``session_token``).
        endpoint_url: Custom S3 endpoint (LocalStack / MinIO / etc.).
    """

    bucket: str | None = None
    region_name: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    endpoint_url: str | None = None

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        # Map the legacy region/credential aliases onto the canonical keys
        # so the typed config is canonical (the canonical key wins when
        # both are present, matching ``S3SessionConfig.from_dict``).
        for alias, canonical in (
            ("region", "region_name"),
            ("access_key_id", "aws_access_key_id"),
            ("secret_access_key", "aws_secret_access_key"),
            ("session_token", "aws_session_token"),
        ):
            if alias in raw:
                if canonical not in raw:
                    raw[canonical] = raw[alias]
                del raw[alias]
        return super()._normalize_dict(raw)

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("S3 backend requires 'bucket' in configuration")


@dataclass(frozen=True)
class SyncS3DatabaseConfig(S3DatabaseConfigBase):
    """Configuration for ``SyncS3Database``.

    Adds the sync-only client tuning knobs consumed by
    :class:`~dataknobs_data.pooling.s3.S3SessionConfig` (``max_workers`` /
    ``max_retries`` are accepted as aliases for ``max_pool_connections`` /
    ``max_attempts``) plus the multipart thresholds. ``prefix`` defaults
    to ``"records/"`` and is normalized to a single trailing slash, matching
    the legacy backend.

    Attributes:
        prefix: Object key prefix (normalized to end with ``/``).
        multipart_threshold: Size threshold for multipart uploads.
        multipart_chunksize: Chunk size for multipart uploads.
        max_pool_connections: boto3 connection-pool size (alias:
            ``max_workers``). Also bounds the search/write thread pool.
        max_attempts: boto3 retry attempts (alias: ``max_retries``).
        retry_mode: boto3 retry mode.
        extra_client_kwargs: Extra kwargs forwarded to the boto3 client.
    """

    prefix: str = "records/"
    multipart_threshold: int = 8 * 1024 * 1024
    multipart_chunksize: int = 8 * 1024 * 1024
    max_pool_connections: int = 10
    max_attempts: int = 3
    retry_mode: str = "standard"
    extra_client_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        # Sync-only pool/retry aliases, then the shared region/credential
        # aliases handled by the base.
        for alias, canonical in (
            ("max_workers", "max_pool_connections"),
            ("max_retries", "max_attempts"),
        ):
            if alias in raw:
                if canonical not in raw:
                    raw[canonical] = raw[alias]
                del raw[alias]
        return super()._normalize_dict(raw)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Normalize the prefix to a single trailing slash (legacy behavior).
        object.__setattr__(self, "prefix", self.prefix.rstrip("/") + "/")
        # Coerce the int knobs so YAML/env string values behave as they did
        # when ``S3SessionConfig.from_dict`` applied ``int(...)``.
        object.__setattr__(
            self, "max_pool_connections", int(self.max_pool_connections)
        )
        object.__setattr__(self, "max_attempts", int(self.max_attempts))


@dataclass(frozen=True)
class AsyncS3DatabaseConfig(S3DatabaseConfigBase):
    """Configuration for ``AsyncS3Database``.

    Mirrors :class:`~dataknobs_data.pooling.s3.S3PoolConfig` (the pool
    config is constructed directly from these fields in the backend's
    ``_setup``). ``prefix`` defaults to ``""`` and keys are joined as
    ``"{prefix}/{id}.json"``, matching the legacy async backend.

    Attributes:
        prefix: Object key prefix (empty by default).
    """

    prefix: str = ""


@dataclass(frozen=True)
class DuckDBDatabaseConfigBase(DatabaseConfig):
    """Shared DuckDB configuration for the sync and async backends.

    DuckDB has no vector support, so this inherits :class:`DatabaseConfig`
    directly (not ``VectorBackendConfig``). The sync and async backends
    share everything except the async-only thread-pool size, which lives
    on :class:`AsyncDuckDBDatabaseConfig`.

    ``auto_create_table`` is coerced through
    :meth:`SQLTableManager.coerce_bool` in ``__post_init__`` so YAML/env
    string values behave as they did when the legacy ``__init__`` coerced
    them.

    Attributes:
        path: Database file path (``":memory:"`` for in-memory).
        table: Records table name.
        timeout: Connection timeout in seconds.
        read_only: Open the database in read-only mode.
        auto_create_table: Create the records table on connect if missing.
    """

    path: str = ":memory:"
    table: str = "records"
    timeout: float = 5.0
    read_only: bool = False
    auto_create_table: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "auto_create_table",
            SQLTableManager.coerce_bool(self.auto_create_table),
        )


@dataclass(frozen=True)
class SyncDuckDBDatabaseConfig(DuckDBDatabaseConfigBase):
    """Configuration for ``SyncDuckDBDatabase`` (no extra fields)."""


@dataclass(frozen=True)
class AsyncDuckDBDatabaseConfig(DuckDBDatabaseConfigBase):
    """Configuration for ``AsyncDuckDBDatabase``.

    Adds ``max_workers`` — the size of the thread pool the async backend
    uses to run DuckDB's synchronous API off the event loop.

    Attributes:
        max_workers: Number of threads in the executor pool.
    """

    max_workers: int = 4


@dataclass(frozen=True)
class FileDatabaseConfig(VectorBackendConfig):
    """Configuration for ``SyncFileDatabase`` / ``AsyncFileDatabase``.

    A single config backs both file backends — their documented surface is
    identical (only the temp-file name prefix and lock primitive differ,
    which is backend logic, not config). ``path`` is optional: when unset
    (``None``) the backend writes to a unique temp file. ``format`` is
    auto-detected from the file extension when unset.

    Attributes:
        path: File path; ``None`` selects a temp file.
        format: File format (``json``, ``csv``, ``parquet``, …);
            auto-detected from the extension when unset.
        compression: Compression scheme (``"gzip"`` or ``None``).
    """

    path: str | None = None
    format: str | None = None
    compression: str | None = None


__all__ = [
    "AsyncDuckDBDatabaseConfig",
    "AsyncElasticsearchDatabaseConfig",
    "AsyncS3DatabaseConfig",
    "AsyncSQLiteDatabaseConfig",
    "DatabaseConfig",
    "DuckDBDatabaseConfigBase",
    "ElasticsearchDatabaseConfigBase",
    "FileDatabaseConfig",
    "MemoryDatabaseConfig",
    "PostgresDatabaseConfig",
    "S3DatabaseConfigBase",
    "SQLiteDatabaseConfigBase",
    "SyncDuckDBDatabaseConfig",
    "SyncElasticsearchDatabaseConfig",
    "SyncS3DatabaseConfig",
    "SyncSQLiteDatabaseConfig",
    "VectorBackendConfig",
]

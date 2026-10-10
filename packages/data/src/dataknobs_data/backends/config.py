# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from dataknobs_common import normalize_postgres_connection_config
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.structured_config import StructuredConfig

from ..database import extract_schema_from_config
from ..query import Filter
from ..schema import FIELD_KEYS, NATIVE_FIELD_KEYS, DatabaseSchema
from .postgres_mixins import validate_pg_identifier
from .sql_base import SQLTableManager


@dataclass(frozen=True)
class DatabaseConfig(StructuredConfig):
    """Base configuration for every ``SyncDatabase`` / ``AsyncDatabase`` backend.

    The ``schema`` field carries the ``DatabaseSchema`` the base
    ``Database`` accepts today. ``__post_init__`` routes any other
    ``schema`` value through :func:`extract_schema_from_config` so it
    becomes a ``DatabaseSchema`` -- a mapping, or a list of field rows -- or
    is refused by name, whether the config was read from a mapping or built
    by calling the class; a ``DatabaseSchema`` instance passes through
    unchanged. This preserves the public
    ``Database(config=..., schema=...)`` kwarg: the consumer mixin merges
    ``schema=`` into the dict and this field captures it.

    Attributes:
        schema: Optional database schema. A mapping or a list of field
            rows is converted to a ``DatabaseSchema`` at construction, and
            any other value is refused; ``None`` yields an empty schema in
            the backend.
    """

    schema: DatabaseSchema | None = None

    #: Reject an input key no backend field claims, rather than dropping it.
    #:
    #: Declared once here so all fourteen backends -- seven sync, seven
    #: async -- inherit it, and so a backend added later inherits it too.
    #: The default (``"ignore"``) is wrong specifically for this family:
    #: every connection field has a working default, so a config made of
    #: misspelled keys does not fail, it succeeds against the wrong store.
    #: A Postgres config carrying ``hosst`` used to connect to
    #: ``localhost`` and log nothing, because the "synthesized default
    #: values" warning fires only when *recognized* explicit keys mix with
    #: defaults -- an unrecognized key enters neither bucket, so the config
    #: reads as "nothing was configured" and the one case most in need of
    #: the warning is the one case it cannot see.
    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"

    def _schema_field_keys(self) -> frozenset[str]:
        """The keys a field declared in ``schema`` takes under this configuration."""
        return FIELD_KEYS

    def __post_init__(self) -> None:
        """Read ``schema`` into a ``DatabaseSchema``, or refuse it.

        The end of the ``__post_init__`` chain, so every subclass can call
        ``super()``.
        """
        object.__setattr__(
            self,
            "schema",
            extract_schema_from_config(self.schema, keys=self._schema_field_keys()),
        )


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

    def __post_init__(self) -> None:
        super().__post_init__()
        # YAML and environment substitution hand over "false" as a string,
        # which is truthy: coerced here, as every other flag on these configs
        # is, so a subclass's ``__post_init__`` must call this one.
        object.__setattr__(
            self,
            "vector_enabled",
            SQLTableManager.coerce_bool(self.vector_enabled, default=False),
        )


@dataclass(frozen=True)
class ColumnLayoutConfig(DatabaseConfig):
    """The keys a SQL backend reads a table it does not own by, shared by every such backend.

    ``layout: native`` reads a table with its own typed columns rather than
    the one this package creates. The declared fields are then the table's
    columns, and may name a ``sql_type``; ``id_column`` names its key, and
    ``scope`` (a list of filters) fixes which rows it is. See
    :func:`~dataknobs_data.backends.column_layout.read_layout_config`, which
    reads the three keys; this class only carries them.

    **A native table is read-only and is never created.** Each switch a
    backend names in :attr:`_CREATE_SWITCHES` is resolved from the layout when
    it is left out, or given with no value: on under the JSON layout, off under
    the native one. Set true under the native one, it is refused. A backend
    with further switches of its own refuses them in its ``__post_init__``,
    through :meth:`_refuse_under_native`.

    Attributes:
        layout: ``"jsonb"`` (the table this package creates, the default) or
            ``"native"`` (a table with its own typed columns).
        id_column: Under ``layout: native``, the column holding the key.
        scope: Under ``layout: native``, the filters fixing which rows the
            table is, each written as a ``{field, operator, value}`` mapping.
    """

    layout: str = "jsonb"
    id_column: str | None = None
    #: Filters or ``{field, operator, value}`` mappings; held as mappings.
    scope: list[Any] | None = None

    #: The backend's name, as a refusal says it.
    _BACKEND: ClassVar[str] = "This backend"
    #: The backend's switches that create or add something: resolved from the
    #: layout when unset, and refused when set true under ``layout: native``.
    _CREATE_SWITCHES: ClassVar[tuple[str, ...]] = ()

    @property
    def native(self) -> bool:
        """Whether the table is read through its own columns."""
        return self.layout == "native"

    def _schema_field_keys(self) -> frozenset[str]:
        # A native table's fields may name a ``sql_type``, which only that
        # layout reads; the JSON layout refuses one.
        return NATIVE_FIELD_KEYS if self.native else super()._schema_field_keys()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Unset -- left out, or given with no value (YAML's ``null``) -- means
        # the layout's default: a native table is never created, so off; and
        # refused below if given true. Resolved here rather than when a mapping
        # is read, so a config built directly gets the same default.
        for key in self._CREATE_SWITCHES:
            object.__setattr__(
                self, key, SQLTableManager.coerce_bool(getattr(self, key), default=not self.native)
            )
        if isinstance(self.scope, (list, tuple)):
            # Held as mappings, so the config is what a config file holds and
            # reads back from its own ``to_dict``.
            object.__setattr__(
                self,
                "scope",
                [spec.to_dict() if isinstance(spec, Filter) else spec for spec in self.scope],
            )
        for key in self._CREATE_SWITCHES:
            if getattr(self, key):
                self._refuse_under_native(
                    f"`{key}: true`",
                    "a native table belongs to someone else and is read-only, so nothing "
                    f"creates it or adds to it. Leave `{key}` out",
                    key=key,
                )

    def _refuse_under_native(self, setting: str, why: str, *, key: str) -> None:
        """Refuse ``setting`` when the layout is native; say ``why``."""
        if self.native:
            raise ConfigurationError(
                f"{self._BACKEND} {setting} under `layout: native`: {why}",
                context={"table": getattr(self, "table", None), "key": key},
            )


@dataclass(frozen=True)
class MemoryDatabaseConfig(VectorBackendConfig):
    """Configuration for ``SyncMemoryDatabase`` / ``AsyncMemoryDatabase``.

    The in-memory backends have no construction parameters beyond the
    shared schema + vector knobs; the dataclass exists for structural
    symmetry so every backend exposes the same ``config`` /
    ``from_config`` surface.
    """


#: The paths that name no file: SQLite's in-memory and temporary databases.
_NO_FILE = (":memory:", "")


@dataclass(frozen=True)
class SQLiteDatabaseConfigBase(ColumnLayoutConfig, VectorBackendConfig):
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

    **A table in somebody else's file** is read with ``layout: native`` (see
    :class:`ColumnLayoutConfig`). The file is opened read-only, so ``path``
    must name it: an in-memory or temporary database holds no table somebody
    else made. ``auto_create_table`` is off there, and setting it,
    ``vector_enabled`` or ``journal_mode`` -- which is written into the file
    -- is refused.

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
    #: ``None`` until ``__post_init__`` resolves it from ``layout``; a bool after.
    auto_create_table: bool | None = None

    _BACKEND: ClassVar[str] = "SQLite"
    _CREATE_SWITCHES: ClassVar[tuple[str, ...]] = ("auto_create_table", "vector_enabled")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.path in _NO_FILE:
            self._refuse_under_native(
                f"`path: {self.path!r}`",
                "a new database holds no table somebody else made. Name the file that "
                "holds the table in `path`",
                key="path",
            )
        if self.journal_mode is not None:
            self._refuse_under_native(
                f"`journal_mode: {self.journal_mode}`",
                "the journal mode is written into the file, which is somebody else's and "
                "is opened read-only. Leave `journal_mode` out",
                key="journal_mode",
            )


@dataclass(frozen=True)
class SyncSQLiteDatabaseConfig(SQLiteDatabaseConfigBase):
    """Configuration for ``SyncSQLiteDatabase``.

    Adds ``check_same_thread`` — the stdlib ``sqlite3`` knob controlling
    whether a connection may be shared across threads — which has no
    aiosqlite equivalent.

    Attributes:
        check_same_thread: Passed straight to ``sqlite3.connect``. Note the
            stdlib polarity: ``True`` *restricts* the connection to its
            creating thread (raising on cross-thread use); ``False`` (the
            default here) *permits* cross-thread sharing.
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


#: Sentinel for "this side gave no namespace as ``schema``" -- ``None`` is a
#: value a side may hold.
_NO_NAMESPACE: Any = object()


@dataclass(frozen=True)
class PostgresDatabaseConfig(ColumnLayoutConfig, VectorBackendConfig):
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

    **What ``schema`` means is decided by its type.** Postgres also reads
    the ``schema`` key as its SQL namespace. A *string* is the namespace and
    maps to ``schema_name`` (winning over an explicit ``schema_name``,
    matching legacy precedence). A mapping, a list of field rows or a
    ``DatabaseSchema`` is the declared fields, read as every other backend
    reads them. When a mapping and keyword arguments both carry ``schema``,
    :meth:`merge_inputs` sorts each side first, so a namespace in one and
    declared fields in the other are both kept, and a namespace given as
    ``schema`` wins over ``schema_name`` whichever side each arrives on.
    Identifiers are validated in
    ``__post_init__``.

    **A table this package did not create** is read with ``layout: native``
    (see :class:`ColumnLayoutConfig`). A native table is read-only and is
    never created, so ``auto_create_table`` and ``ensure_database`` default
    to ``False`` there, and setting either, or ``vector_enabled``, to
    ``True`` is refused.

    Attributes:
        host/port/database/user/password: Connection parameters
            (resolved from env / ``connection_string`` / explicit keys).
            ``password`` is redacted from ``repr``.
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
    #: ``None`` until ``__post_init__`` resolves it from ``layout``; a bool after.
    ensure_database: bool | None = None
    #: ``None`` until ``__post_init__`` resolves it from ``layout``; a bool after.
    auto_create_table: bool | None = None

    _BACKEND: ClassVar[str] = "Postgres"
    _CREATE_SWITCHES: ClassVar[tuple[str, ...]] = (
        "auto_create_table",
        "ensure_database",
        "vector_enabled",
    )

    # Redacted from ``repr`` by the StructuredConfig base.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"password"})

    #: Accepted inputs ``_normalize_dict`` resolves away rather than
    #: keeping: ``connection_string`` is decomposed into the individual
    #: connection keys and popped, ``table_name`` is the legacy spelling
    #: of ``table``. Declared so a caller who wrote ``connection`` is
    #: pointed at ``connection_string`` instead of reading a field list
    #: that does not contain it.
    _INPUT_KEYS: ClassVar[frozenset[str]] = frozenset({"connection_string", "table_name"})

    @classmethod
    def merge_inputs(cls, config: Mapping[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Merge after sorting each side's ``schema`` into namespace or fields.

        A namespace in the mapping and declared fields as a keyword (or the
        other way round) land on different fields, so both are kept. A
        namespace given as ``schema`` wins over ``schema_name`` whichever side
        each arrives on, as it does within one mapping; given as ``schema`` on
        both sides, the keyword's wins. Every other key merges as in the
        default: the keyword wins.
        """
        config_rest, config_ns = cls._namespace_apart(config)
        kwargs_rest, kwargs_ns = cls._namespace_apart(kwargs)
        merged = {**config_rest, **kwargs_rest}
        namespace = kwargs_ns if kwargs_ns is not _NO_NAMESPACE else config_ns
        if namespace is not _NO_NAMESPACE:
            # Onto ``schema_name``, over whatever either side gave there, and
            # off ``schema``, where the other side's declared fields may be.
            merged["schema_name"] = namespace
        return merged

    @staticmethod
    def _names_namespace(value: Any) -> bool:
        """Whether a ``schema`` value is the SQL namespace rather than declared fields.

        Declared fields are a mapping, a list of field rows or a
        ``DatabaseSchema``; ``None`` is the structural default ``to_dict``
        writes. Anything else is the namespace, where a value that is not a
        string fails identifier validation by name.
        """
        return not isinstance(value, (DatabaseSchema, Mapping, list, tuple, type(None)))

    @classmethod
    def _namespace_apart(cls, side: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        """One side's keys without a namespace ``schema``, and that namespace."""
        out = dict(side)
        if "schema" in out and cls._names_namespace(out["schema"]):
            return out, out.pop("schema")
        return out, _NO_NAMESPACE

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        # Disambiguate the overloaded ``schema`` key by type. A mapping, a
        # list of rows or a ``DatabaseSchema`` is the declared fields, left for
        # ``DatabaseConfig.__post_init__`` to read (a native table's fields may
        # also name a ``sql_type``). ``None`` is the
        # structural default ``to_dict`` writes.
        # Anything else is the SQL namespace: routed to ``schema_name`` (it
        # wins over an explicit ``schema_name``, matching legacy precedence),
        # where a value that is not a string fails identifier validation in
        # ``__post_init__`` with the error it always has.
        if "schema" in raw and cls._names_namespace(raw["schema"]):
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
        super().__post_init__()
        # Validate identifiers early (a non-string ``schema``/``table`` —
        # e.g. a DatabaseSchema injected via the key collision — fails
        # fast with a clear ConfigurationError rather than emitting broken
        # DDL at first query).
        object.__setattr__(self, "table", validate_pg_identifier(self.table, "table"))
        object.__setattr__(self, "schema_name", validate_pg_identifier(self.schema_name, "schema"))
        object.__setattr__(self, "port", int(self.port))


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
        max_result_window: The index's ``index.max_result_window``. A search
            whose ``offset + limit`` fits inside it is one ``from``/``size``
            request; anything larger, and any search with no ``limit``, pages
            with ``search_after``. Set it when the index's own setting differs
            from Elasticsearch's default.
        search_page_size: Hits per request when a search pages with
            ``search_after``. At most ``max_result_window``.
    """

    index: str = "records"
    refresh: bool = True
    max_result_window: int = 10_000
    search_page_size: int = 1_000

    def __post_init__(self) -> None:
        super().__post_init__()
        # A value from YAML or the environment may arrive as a string.
        for name in ("max_result_window", "search_page_size"):
            value = getattr(self, name)
            try:
                number = int(value) if not isinstance(value, bool) else None
            except (TypeError, ValueError):
                number = None
            if number is None or number < 1:
                raise ValueError(
                    f"Elasticsearch '{name}' must be a positive integer, got {value!r}"
                )
            object.__setattr__(self, name, number)
        if self.search_page_size > self.max_result_window:
            raise ValueError(
                f"Elasticsearch 'search_page_size' ({self.search_page_size}) must not exceed "
                f"'max_result_window' ({self.max_result_window})"
            )


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
        api_key: API key for authentication. Redacted from ``repr``.
        basic_auth: ``(user, password)`` tuple for basic auth. Redacted
            from ``repr``.
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

    # Redacted from ``repr`` by the StructuredConfig base. ``basic_auth``
    # is a ``(user, password)`` tuple — masking the whole field hides the
    # password (and the username) in one shot.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key", "basic_auth"})


@dataclass(frozen=True)
class S3DatabaseConfigBase(VectorBackendConfig):
    """Shared S3 configuration for the sync and async backends.

    Both S3 backends route region/credential/endpoint resolution through
    :class:`~dataknobs_common.aws.AwsSessionConfig` /
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
            Redacted from ``repr``.
        aws_secret_access_key: AWS secret key (alias: ``secret_access_key``).
            Redacted from ``repr``.
        aws_session_token: AWS session token (alias: ``session_token``).
            Redacted from ``repr``.
        endpoint_url: Custom S3 endpoint (LocalStack / MinIO / etc.).
    """

    bucket: str | None = None
    region_name: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    endpoint_url: str | None = None

    # Redacted from ``repr`` by the StructuredConfig base. Declared on the
    # shared base so both Sync/AsyncS3DatabaseConfig inherit the masking.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"aws_access_key_id", "aws_secret_access_key", "aws_session_token"}
    )

    # The legacy spellings ``_normalize_dict`` maps onto the canonical
    # fields below and then deletes. Declared so the unknown-key error
    # describes the surface a caller may actually write, not only the
    # names that survive normalization.
    _INPUT_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"region", "access_key_id", "secret_access_key", "session_token"}
    )

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        # Map the legacy region/credential aliases onto the canonical keys
        # so the typed config is canonical (the canonical key wins when
        # both are present, matching ``AwsSessionConfig.from_dict``).
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
        super().__post_init__()
        if not self.bucket:
            raise ValueError("S3 backend requires 'bucket' in configuration")


@dataclass(frozen=True)
class SyncS3DatabaseConfig(S3DatabaseConfigBase):
    """Configuration for ``SyncS3Database``.

    Adds the sync-only client tuning knobs consumed by
    :class:`~dataknobs_common.aws.AwsSessionConfig` (``max_workers`` /
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

    # Only the two this class adds -- ``_accepted_keys`` unions
    # ``_INPUT_KEYS`` across the MRO, so the base's credential aliases
    # stay accepted without being restated here.
    _INPUT_KEYS: ClassVar[frozenset[str]] = frozenset({"max_workers", "max_retries"})

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
        # when ``AwsSessionConfig.from_dict`` applied ``int(...)``.
        object.__setattr__(self, "max_pool_connections", int(self.max_pool_connections))
        object.__setattr__(self, "max_attempts", int(self.max_attempts))


@dataclass(frozen=True)
class AsyncS3DatabaseConfig(S3DatabaseConfigBase):
    """Configuration for ``AsyncS3Database``.

    Mirrors :class:`~dataknobs_data.pooling.s3.S3PoolConfig` (the pool
    config is constructed directly from these fields in the backend's
    ``_setup``). ``prefix`` defaults to ``""``; the backend joins keys as
    ``"{prefix}/{id}.json"`` when a prefix is set and ``"{id}.json"`` when
    it is empty (the default), matching the legacy async backend.

    Attributes:
        prefix: Object key prefix (empty by default).
    """

    prefix: str = ""


@dataclass(frozen=True)
class DuckDBDatabaseConfigBase(ColumnLayoutConfig):
    """Shared DuckDB configuration for the sync and async backends.

    DuckDB has no vector support, so this inherits :class:`DatabaseConfig`
    directly (not ``VectorBackendConfig``). The sync and async backends
    share everything except the async-only thread-pool size, which lives
    on :class:`AsyncDuckDBDatabaseConfig`.

    ``auto_create_table`` and ``read_only`` are coerced through
    :meth:`SQLTableManager.coerce_bool` in ``__post_init__`` so YAML/env
    string values behave as they did when the legacy ``__init__`` coerced
    them.

    **A table in somebody else's file** is read with ``layout: native`` (see
    :class:`ColumnLayoutConfig`). ``path`` must name the file, and it is
    opened ``read_only``: unset, ``read_only`` is on there, and
    ``read_only: false`` is refused, as is ``auto_create_table: true``.

    Attributes:
        path: Database file path (``":memory:"`` for in-memory).
        table: Records table name.
        timeout: Connection timeout in seconds.
        read_only: Open the database in read-only mode. Off by default, and
            on under ``layout: native``.
        auto_create_table: Create the records table on connect if missing.
    """

    path: str = ":memory:"
    table: str = "records"
    timeout: float = 5.0
    #: ``None`` until ``__post_init__`` resolves it from ``layout``; a bool after.
    read_only: bool | None = None
    #: ``None`` until ``__post_init__`` resolves it from ``layout``; a bool after.
    auto_create_table: bool | None = None

    _BACKEND: ClassVar[str] = "DuckDB"
    _CREATE_SWITCHES: ClassVar[tuple[str, ...]] = ("auto_create_table",)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "read_only", SQLTableManager.coerce_bool(self.read_only, default=self.native)
        )
        if not self.read_only:
            self._refuse_under_native(
                "`read_only: false`",
                "the file is somebody else's and is only read. Leave `read_only` out",
                key="read_only",
            )
        if self.path in _NO_FILE:
            self._refuse_under_native(
                f"`path: {self.path!r}`",
                "a new database holds no table somebody else made. Name the file that "
                "holds the table in `path`",
                key="path",
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
    "ColumnLayoutConfig",
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

"""Structured configuration dataclasses for vector store backends.

Every vector store's documented config key is a typed dataclass field;
the auto-derived :meth:`StructuredConfig.from_dict
<dataknobs_common.structured_config.StructuredConfig.from_dict>`
classmethod is the single source of truth for translating a config dict
into typed construction. The stores mix in
:class:`~dataknobs_common.structured_config.StructuredConfigConsumer`
parameterized by their config dataclass, so the registry factory passes a
dict straight through and drift between a store's documented surface and
its construction path becomes structurally impossible.

The dataclasses are ``frozen=True`` so ``store.config`` is a safe
read-only window onto the construction parameters.

The hierarchy mirrors the shared key sets: :class:`VectorStoreConfig`
holds the keys common to every backend (dimensions, metric, persistence,
batch size, parameter sub-dicts, multi-tenant ``domain_id``, and the
nested timestamp-exposure config). Each backend's leaf config adds only
its own keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from dataknobs_common import normalize_postgres_connection_config
from dataknobs_common.structured_config import StructuredConfig

from ...backends.postgres_mixins import validate_pg_identifier

_VALID_TIMESTAMP_FORMATS = ("iso", "epoch", "datetime")


@dataclass(frozen=True)
class VectorStoreTimestampConfig(StructuredConfig):
    """Timestamp-exposure config nested under ``VectorStoreConfig.timestamps``.

    All vector stores expose ``created_at`` / ``updated_at`` metadata via
    ``include_timestamps=True`` on ``get_vectors()`` and ``search()``. The
    format and key names are configurable; the defaults are consistent
    across backends so a runtime backend swap produces an identical
    metadata surface.

    Attributes:
        format: Timestamp rendering — ``"iso"`` (ISO-8601 string),
            ``"epoch"`` (seconds as ``float``), or ``"datetime"`` (native
            ``datetime`` object).
        created_key: Metadata key under which the created timestamp is
            injected.
        updated_key: Metadata key under which the updated timestamp is
            injected.
    """

    format: str = "iso"
    created_key: str = "_created_at"
    updated_key: str = "_updated_at"

    def __post_init__(self) -> None:
        if self.format not in _VALID_TIMESTAMP_FORMATS:
            raise ValueError(
                "timestamps.format must be 'iso', 'epoch', or 'datetime'; "
                f"got {self.format!r}"
            )


@dataclass(frozen=True)
class VectorStoreConfig(StructuredConfig):
    """Base configuration for every vector store backend.

    The shared keys extracted by the legacy ``_parse_common_config``.
    ``metric`` is kept as the raw string the documented config accepts
    (``"cosine"``, ``"euclidean"``, ...); the store converts it to a
    :class:`~dataknobs_data.vector.types.DistanceMetric` during ``_setup``.
    ``timestamps`` is a nested :class:`VectorStoreTimestampConfig` that
    composes automatically via ``StructuredConfig.from_dict`` recursion;
    ``None`` means "use the timestamp defaults".

    Attributes:
        dimensions: Vector dimensions.
        metric: Distance-metric name for vector similarity.
        persist_path: Filesystem path for persistent storage; ``~`` is
            expanded by the store during ``_setup``.
        batch_size: Batch size for bulk operations.
        index_params: Backend-specific index parameters.
        search_params: Backend-specific search parameters.
        metadata: Arbitrary store-level metadata.
        domain_id: Multi-tenant scoping. When set, every read/count/clear
            is implicitly scoped to this domain and ``add_vectors``
            defaults a row's ``domain_id`` to it.
        timestamps: Nested timestamp-exposure config; ``None`` uses
            defaults.
    """

    dimensions: int = 0
    metric: str = "cosine"
    persist_path: str | None = None
    batch_size: int = 100
    index_params: dict[str, Any] = field(default_factory=dict)
    search_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    domain_id: str | None = None
    timestamps: VectorStoreTimestampConfig | None = None


@dataclass(frozen=True)
class MemoryVectorStoreConfig(VectorStoreConfig):
    """Configuration for ``MemoryVectorStore`` (no extra keys).

    The in-memory store has no construction parameters beyond the shared
    set; the dataclass exists for structural symmetry so every backend
    exposes the same ``config`` / ``from_config`` surface.
    """


@dataclass(frozen=True)
class FaissVectorStoreConfig(VectorStoreConfig):
    """Configuration for ``FaissVectorStore``.

    The IVF/HNSW tuning knobs (``nlist`` / ``m`` / ``ef_construction`` /
    ``ef_search`` / ``nprobe``) are not flat keys — they live inside
    ``index_params`` / ``search_params`` and are derived by the store in
    ``_setup``.

    Attributes:
        index_type: Index type (``"flat"``, ``"ivfflat"``, ``"hnsw"``,
            ``"auto"``). When ``None``, the store falls back to
            ``index_params["type"]`` (default ``"auto"``), preserving the
            legacy dual-source precedence.
    """

    index_type: str | None = None


@dataclass(frozen=True)
class ChromaVectorStoreConfig(VectorStoreConfig):
    """Configuration for ``ChromaVectorStore``.

    ``scalar_metadata_keys`` is normalized to a ``frozenset`` so
    ``store.config`` is canonical and round-trips. ``openai_api_key`` is a
    credential, so this config lists it in ``_SENSITIVE_FIELDS``; the
    ``StructuredConfig`` base then redacts it from ``repr`` automatically,
    keeping the key out of any log that interpolates ``repr(config)``.

    Attributes:
        dimensions: Vector dimensions. ``0`` is a sentinel meaning "use the
            384-dimension sentence-transformers default"; it is resolved to
            384 in ``__post_init__`` (matching the legacy backend), so an
            explicit ``dimensions=0`` cannot be used to request a
            zero-dimension store.
        collection_name: Chroma collection name.
        scalar_metadata_keys: Opt-in set of metadata keys whose values are
            always scalar; lets the store push a Chroma-native ``$eq``
            predicate instead of post-filtering.
        embedding_function: Embedding-function name (``"default"`` /
            ``"openai"``) or a pre-built Chroma embedding function object.
        openai_api_key: API key used when ``embedding_function="openai"``.
            Redacted from ``repr``.
    """

    collection_name: str = "vectors"
    scalar_metadata_keys: frozenset[str] | None = None
    embedding_function: Any = None
    openai_api_key: str | None = None

    # Redaction is automatic: the StructuredConfig base installs a
    # repr that masks every field named here (see _redacted_repr).
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"openai_api_key"})

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        # Normalize a list/tuple/set ``scalar_metadata_keys`` to a
        # ``frozenset`` before field projection so ``self.config`` is
        # canonical and round-trips. Done here (where the value is still
        # untyped) rather than in ``__post_init__`` so the field's declared
        # ``frozenset`` type stays honest.
        keys = raw.get("scalar_metadata_keys")
        if keys is not None and not isinstance(keys, frozenset):
            raw["scalar_metadata_keys"] = frozenset(keys)
        return raw

    def __post_init__(self) -> None:
        if self.dimensions == 0:
            object.__setattr__(self, "dimensions", 384)


@dataclass(frozen=True)
class PgVectorStoreConfig(VectorStoreConfig):
    """Configuration for ``PgVectorStore`` (the heaviest backend).

    The connection layer is resolved in :meth:`_normalize_dict` through
    :func:`normalize_postgres_connection_config`, which folds in a
    ``connection_string`` and the ``POSTGRES_*`` / ``DATABASE_URL`` env
    vars. A connection that cannot be resolved raises ``ValueError`` at
    construction (the public contract consumers rely on). Identifiers are
    validated in ``__post_init__``.

    The resolved ``connection_string`` embeds the database password, so
    this config lists it in ``_SENSITIVE_FIELDS``; the ``StructuredConfig``
    base then redacts it from ``repr`` automatically, keeping the DSN out
    of any log that interpolates ``repr(config)``.

    Attributes:
        connection_string: Resolved PostgreSQL connection URL (populated by
            ``_normalize_dict``). Embeds the password; redacted from
            ``repr``.
        table_name: Embeddings table name.
        schema: SQL schema name.
        pool_min_size/pool_max_size: asyncpg pool bounds.
        columns: Logical→physical column-name overrides; merged over the
            store's ``DEFAULT_COLUMNS`` in ``_setup``.
        auto_create_table: Create the table on connect if missing.
        id_type: ID column type — ``"uuid"`` or ``"text"``.
        index_type: Vector index type — ``"none"``, ``"hnsw"``, or
            ``"ivfflat"``.
        auto_create_index: Auto-create the index when conditions are met.
        min_rows_for_index: Minimum rows before auto-creating an IVFFlat
            index.
    """

    connection_string: str = ""
    table_name: str = "knowledge_embeddings"
    schema: str = "public"
    pool_min_size: int = 2
    pool_max_size: int = 10
    columns: dict[str, str] = field(default_factory=dict)
    auto_create_table: bool = True
    id_type: str = "text"
    index_type: str = "none"
    auto_create_index: bool = False
    min_rows_for_index: int = 1000

    # Redaction is automatic: the StructuredConfig base installs a
    # repr that masks every field named here (see _redacted_repr).
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"connection_string"})

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        # Route through the shared normalizer so pgvector accepts every
        # input shape the rest of dataknobs supports (connection_string,
        # individual host/port/database/user/password keys, DATABASE_URL
        # env var, POSTGRES_* env vars). Call with require=False + a manual
        # ValueError on None to preserve the public ValueError contract
        # (the normalizer itself raises ConfigurationError); consumers and
        # tests rely on the ValueError type for this failure mode.
        normalized = normalize_postgres_connection_config(raw, require=False)
        if normalized is None:
            raise ValueError(
                "PgVectorStore requires a postgres connection. Provide one of: "
                "'connection_string', individual host/port/database/user/password "
                "keys, 'DATABASE_URL' env var, or POSTGRES_HOST/POSTGRES_PORT/"
                "POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD env vars."
            )
        raw["connection_string"] = normalized["connection_string"]
        return raw

    def __post_init__(self) -> None:
        # Validate identifiers early so misconfiguration surfaces at
        # construction rather than as a syntax error at first DDL.
        object.__setattr__(
            self, "table_name", validate_pg_identifier(self.table_name, "table_name")
        )
        object.__setattr__(
            self, "schema", validate_pg_identifier(self.schema, "schema")
        )
        if self.id_type not in ("uuid", "text"):
            raise ValueError(
                f"id_type must be 'uuid' or 'text', got: {self.id_type}"
            )
        if self.index_type not in ("none", "hnsw", "ivfflat"):
            raise ValueError(
                "index_type must be 'none', 'hnsw', or 'ivfflat', got: "
                f"{self.index_type}"
            )


__all__ = [
    "ChromaVectorStoreConfig",
    "FaissVectorStoreConfig",
    "MemoryVectorStoreConfig",
    "PgVectorStoreConfig",
    "VectorStoreConfig",
    "VectorStoreTimestampConfig",
]

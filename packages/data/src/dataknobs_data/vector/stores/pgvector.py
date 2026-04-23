"""PostgreSQL pgvector backend implementation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from dataknobs_common import normalize_postgres_connection_config

from ..types import DistanceMetric
from .base import VectorStore

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


class PgVectorStore(VectorStore):
    """PostgreSQL pgvector backend for vector similarity search.

    Uses PostgreSQL with the pgvector extension for efficient vector storage
    and similarity search. Supports IVFFlat and HNSW indexes.

    Schema contract (auto-created when ``auto_create_table=True``):
    ``id``, ``domain_id``, ``document_id``, ``chunk_index``,
    ``content``, ``embedding``, ``metadata``, ``created_at``,
    ``updated_at``. On upsert (same-ID ``add_vectors``), all content
    columns are refreshed, ``updated_at = NOW()``, and ``created_at``
    is preserved. Pre-existing tables gain ``updated_at`` via an
    idempotent migration during ``initialize()`` — legacy rows keep
    ``NULL`` until re-ingested.

    Configuration:
        connection_string: PostgreSQL connection URL
        table_name: Table name (default: knowledge_embeddings)
        schema: Database schema (default: edubot)
        dimensions: Vector dimensions (required)
        metric: Distance metric (cosine, euclidean, inner_product)
        pool_min_size: Minimum connection pool size (default: 2)
        pool_max_size: Maximum connection pool size (default: 10)
        columns: Column name mappings (optional)
        auto_create_table: Create table if missing (default: True)
        id_type: ID column type - 'uuid' or 'text' (default: 'text')

        .. note::
            The ``id_type`` default is ``"text"`` so that RAG consumers
            passing chunk ids such as ``"01-fundamentals_0"`` work
            out-of-the-box. Set ``id_type: "uuid"`` explicitly to
            opt into server-generated UUID columns.

        Index configuration:
        index_type: Type of vector index - 'none', 'hnsw', or 'ivfflat' (default: 'none')
        auto_create_index: Automatically create index when conditions are met (default: False)
        min_rows_for_index: Minimum rows before auto-creating IVFFlat index (default: 1000)
        index_params: Parameters for index creation (optional dict)
            - For HNSW: m (default: 16), ef_construction (default: 64)
            - For IVFFlat: lists (default: 100)

    Example - Default schema:
        ```python
        store = PgVectorStore({
            "connection_string": "postgresql://user:pass@host:5432/db",
            "dimensions": 768,
            "metric": "cosine",
            "schema": "edubot",
        })
        ```

    Example - With HNSW index (created immediately, works with any data size):
        ```python
        store = PgVectorStore({
            "connection_string": "postgresql://user:pass@host:5432/db",
            "dimensions": 768,
            "index_type": "hnsw",
            "auto_create_index": True,
            "index_params": {"m": 16, "ef_construction": 64},
        })
        ```

    Example - With IVFFlat index (auto-created when data exceeds threshold):
        ```python
        store = PgVectorStore({
            "connection_string": "postgresql://user:pass@host:5432/db",
            "dimensions": 768,
            "index_type": "ivfflat",
            "auto_create_index": True,
            "min_rows_for_index": 1000,
            "index_params": {"lists": 100},
        })
        ```

    Example - Custom table with column mappings:
        ```python
        store = PgVectorStore({
            "connection_string": "postgresql://user:pass@host:5432/db",
            "dimensions": 768,
            "table_name": "product_embeddings",
            "columns": {
                "id": "product_id",
                "embedding": "vector_data",
                "content": "description",
                "metadata": "attributes",
                "domain_id": "category",
            },
            "id_type": "text",
            "auto_create_table": True,
        })
        ```
    """

    # Default column mappings
    DEFAULT_COLUMNS = {
        "id": "id",
        "embedding": "embedding",
        "content": "content",
        "metadata": "metadata",
        "domain_id": "domain_id",
        "document_id": "document_id",
        "chunk_index": "chunk_index",
        "created_at": "created_at",
        "updated_at": "updated_at",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize pgvector store."""
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is not installed. Install with: pip install asyncpg"
            )

        super().__init__(config)
        self._pool: asyncpg.Pool | None = None

    def _parse_backend_config(self) -> None:
        """Parse pgvector-specific configuration."""
        # Route through the shared normalizer so pgvector accepts every
        # input shape the rest of dataknobs supports (``connection_string``,
        # individual host/port/database/user/password keys, ``DATABASE_URL``
        # env var, ``POSTGRES_*`` env vars, and ``.env`` / ``.project_vars``
        # files). We call with ``require=False`` + manual ``ValueError``
        # on ``None`` to preserve the public ``ValueError`` contract (the
        # normalizer itself raises ``ConfigurationError``); consumers and
        # tests rely on the ``ValueError`` type for this construction
        # failure mode.
        normalized = normalize_postgres_connection_config(
            self.config, require=False,
        )
        if normalized is None:
            raise ValueError(
                "PgVectorStore requires a postgres connection. Provide one of: "
                "'connection_string', individual host/port/database/user/password "
                "keys, 'DATABASE_URL' env var, or POSTGRES_HOST/POSTGRES_PORT/"
                "POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD env vars."
            )
        self.connection_string = normalized["connection_string"]

        self.table_name = self.config.get("table_name", "knowledge_embeddings")
        self.schema = self.config.get("schema", "edubot")
        self.pool_min_size = self.config.get("pool_min_size", 2)
        self.pool_max_size = self.config.get("pool_max_size", 10)

        # Domain filtering (optional - for multi-tenant isolation)
        self.domain_id = self.config.get("domain_id")

        # Column mappings - merge user config with defaults
        user_columns = self.config.get("columns", {})
        self.columns = {**self.DEFAULT_COLUMNS, **user_columns}

        # Table creation options
        self.auto_create_table = self.config.get("auto_create_table", True)
        # Defect A: default is ``"text"`` so RAG consumers passing chunk ids
        # like ``"01-fundamentals_0"`` work without any config override.
        # Pre-flip consumers using server-generated UUIDs must set
        # ``id_type="uuid"`` explicitly.
        self.id_type = self.config.get("id_type", "text")
        if self.id_type not in ("uuid", "text"):
            raise ValueError(f"id_type must be 'uuid' or 'text', got: {self.id_type}")

        # Index configuration
        self.index_type = self.config.get("index_type", "none")
        if self.index_type not in ("none", "hnsw", "ivfflat"):
            raise ValueError(
                f"index_type must be 'none', 'hnsw', or 'ivfflat', got: {self.index_type}"
            )
        self.auto_create_index = self.config.get("auto_create_index", False)
        self.min_rows_for_index = self.config.get("min_rows_for_index", 1000)
        self.index_params = self.config.get("index_params", {})

    def _col(self, name: str) -> str:
        """Get the actual column name for a logical field name."""
        return self.columns.get(name, name)

    async def _exec_with_id_type_guard(
        self,
        conn: asyncpg.Connection,
        method: str,
        query: str,
        *args: Any,
        vec_id: Any = None,
    ) -> Any:
        """Run a pgvector query, wrapping id-type mismatch errors.

        Wraps raw ``asyncpg.DataError`` for the common failure modes
        where the configured ``id_type`` disagrees with the actual
        table column type or the passed id value:

        * ``id_type="uuid"`` but caller passed a non-UUID id —
          Postgres rejects the bind with "invalid input syntax for
          type uuid". The guided ValueError tells the caller to
          either flip the config to ``"text"`` or supply a UUID.
        * ``id_type="text"`` but the table column is actually UUID
          (e.g., a pre-flip consumer who did not add
          ``id_type: "uuid"`` to config after the default flip) —
          Postgres rejects the bind the same way. The guided
          ValueError points at adding ``id_type: "uuid"`` and notes
          that the schema cannot be changed in place.

        Detection prefers ``asyncpg.exceptions.InvalidTextRepresentation``
        (SQLSTATE ``22P02``) — the specific error raised for
        "invalid input syntax for type uuid" — and falls back to a
        case-insensitive string match on the message for older
        asyncpg versions or unusual error subclasses.

        Args:
            conn: Active asyncpg connection.
            method: One of ``"execute"``, ``"fetchrow"``, or ``"fetch"``.
            query: SQL statement to run.
            *args: Parameters bound to ``$1``..``$N``.
            vec_id: The id value being bound; included verbatim in the
                guided error message. For bulk operations
                (``delete_vectors``) this is a list — callers that can
                identify the specific offending id should validate
                client-side and pass the single bad id.

        Returns:
            Whatever the underlying asyncpg method returns.

        Raises:
            ValueError: When the id-type mismatch is detected.
            asyncpg.DataError: For any other data error (unchanged).
        """
        try:
            callable_ = getattr(conn, method)
            return await callable_(query, *args)
        except asyncpg.DataError as e:
            if self._is_uuid_parse_error(e):
                raise self._guided_id_type_error(vec_id, e) from e
            raise

    def _is_uuid_parse_error(self, exc: BaseException) -> bool:
        """Return True when ``exc`` is Postgres's UUID parse error.

        Prefers the specific ``InvalidTextRepresentationError`` class
        (SQLSTATE ``22P02``), falling back to a string-match heuristic
        for forward-compat with asyncpg versions that restructure the
        exception hierarchy.
        """
        invalid_text_cls = getattr(
            asyncpg.exceptions, "InvalidTextRepresentationError", None,
        )
        if invalid_text_cls is not None and isinstance(exc, invalid_text_cls):
            return True
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate == "22P02":
            return True
        msg = str(exc).lower()
        return "invalid" in msg and "uuid" in msg

    def _guided_id_type_error(
        self, vec_id: Any, cause: BaseException,
    ) -> ValueError:
        """Build the guided ValueError for an id-type mismatch.

        The direction of the mismatch drives the remediation hint:

        * ``id_type="uuid"`` + non-UUID id → flip to text or supply UUID.
        * ``id_type="text"`` + UUID-typed column → add ``id_type: "uuid"``
          and note that swapping an existing UUID table to text
          requires a DROP + re-ingest (``CREATE TABLE IF NOT EXISTS``
          is a no-op on the existing schema).
        """
        location = f"table={self.schema}.{self.table_name}"
        if self.id_type == "uuid":
            return ValueError(
                f"PgVectorStore id_type={self.id_type!r} but received "
                f"non-UUID id {vec_id!r}. Either set `id_type: \"text\"` "
                f"in the vector store config, or pass UUID-formatted "
                f"string ids. Note: flipping to ``text`` only affects "
                f"new deployments — existing UUID-typed tables require "
                f"DROP + re-ingest to accept text ids. "
                f"({location})"
            )
        # id_type="text" but the table column is UUID — this is the
        # common post-flip migration case.
        return ValueError(
            f"PgVectorStore id_type={self.id_type!r} but the underlying "
            f"table column is UUID. Add `id_type: \"uuid\"` to the vector "
            f"store config so id values are cast to uuid, or migrate the "
            f"table (DROP + re-ingest) to use a text id column. "
            f"Offending id: {vec_id!r}. ({location})"
        )

    def _get_operator_class(self) -> str:
        """Get the pgvector operator class for the configured metric."""
        if self.metric == DistanceMetric.COSINE:
            return "vector_cosine_ops"
        elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            return "vector_l2_ops"
        elif self.metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            return "vector_ip_ops"
        else:
            return "vector_cosine_ops"  # Default

    async def _check_index_exists(self) -> bool:
        """Check if a vector index exists on the embedding column.

        Queries PostgreSQL's pg_indexes catalog to check for any index
        on the embedding column. Works reliably in distributed environments.
        """
        if not self._pool:
            return False

        col_embedding = self._col("embedding")
        async with self._pool.acquire() as conn:
            # Check pg_indexes for any index on our table that includes the embedding column
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = $1
                    AND tablename = $2
                    AND indexdef LIKE $3
                )
                """,
                self.schema,
                self.table_name,
                f"%{col_embedding}%",
            )
        return bool(exists)

    async def create_index(
        self,
        index_type: str | None = None,
        params: dict[str, Any] | None = None,
        if_not_exists: bool = True,
    ) -> bool:
        """Create a vector index on the embedding column.

        Args:
            index_type: Type of index - 'hnsw' or 'ivfflat'. Defaults to configured index_type.
            params: Index parameters. Defaults to configured index_params.
                - For HNSW: m (connections per layer), ef_construction (build quality)
                - For IVFFlat: lists (number of clusters)
            if_not_exists: Skip creation if index already exists (default: True)

        Returns:
            True if index was created, False if skipped (already exists)

        Raises:
            ValueError: If index_type is invalid or 'none'
            RuntimeError: If store not initialized

        Example:
            ```python
            # Create HNSW index with custom parameters
            await store.create_index("hnsw", {"m": 32, "ef_construction": 128})

            # Create IVFFlat index (requires sufficient data)
            await store.create_index("ivfflat", {"lists": 200})
            ```
        """
        if not self._initialized:
            raise RuntimeError("Store must be initialized before creating index")

        # Use configured values as defaults
        idx_type = index_type or self.index_type
        idx_params = params if params is not None else self.index_params

        if idx_type == "none":
            raise ValueError("Cannot create index with index_type='none'")
        if idx_type not in ("hnsw", "ivfflat"):
            raise ValueError(f"index_type must be 'hnsw' or 'ivfflat', got: {idx_type}")

        # Check if index already exists
        if if_not_exists and await self._check_index_exists():
            logger.info(f"Index already exists on {self.schema}.{self.table_name}")
            return False

        col_embedding = self._col("embedding")
        operator_class = self._get_operator_class()
        index_name = f"idx_{self.table_name}_{col_embedding}_{idx_type}"

        async with self._pool.acquire() as conn:
            if idx_type == "hnsw":
                m = idx_params.get("m", 16)
                ef_construction = idx_params.get("ef_construction", 64)
                await conn.execute(f"""
                    CREATE INDEX {"IF NOT EXISTS" if if_not_exists else ""} {index_name}
                    ON {self.schema}.{self.table_name}
                    USING hnsw ({col_embedding} {operator_class})
                    WITH (m = {m}, ef_construction = {ef_construction})
                """)
            else:  # ivfflat
                lists = idx_params.get("lists", 100)
                await conn.execute(f"""
                    CREATE INDEX {"IF NOT EXISTS" if if_not_exists else ""} {index_name}
                    ON {self.schema}.{self.table_name}
                    USING ivfflat ({col_embedding} {operator_class})
                    WITH (lists = {lists})
                """)

        logger.info(
            f"Created {idx_type} index on {self.schema}.{self.table_name}.{col_embedding}"
        )
        return True

    async def _maybe_create_index(self) -> None:
        """Conditionally create index based on configuration and data size.

        Called during search() to auto-create IVFFlat index when:
        - auto_create_index is True
        - index_type is 'ivfflat'
        - Row count exceeds min_rows_for_index
        - No index exists yet
        """
        if not self.auto_create_index:
            return
        if self.index_type != "ivfflat":
            return  # HNSW is created at table creation time

        # Check if index already exists (distributed-safe)
        if await self._check_index_exists():
            return

        # Check row count
        row_count = await self.count()
        if row_count < self.min_rows_for_index:
            return

        logger.info(
            f"Auto-creating IVFFlat index: {row_count} rows >= {self.min_rows_for_index} threshold"
        )
        await self.create_index("ivfflat", self.index_params, if_not_exists=True)

    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if self._initialized:
            return

        logger.info(f"Initializing pgvector store: {self.schema}.{self.table_name}")

        # Create connection pool
        self._pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            command_timeout=30,
        )

        # Verify pgvector extension and table exist
        async with self._pool.acquire() as conn:
            # Check pgvector extension
            has_pgvector = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            if not has_pgvector:
                raise RuntimeError(
                    "pgvector extension not installed. Run: CREATE EXTENSION vector;"
                )

            # Check table exists
            table_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = $1 AND table_name = $2
                )
                """,
                self.schema,
                self.table_name,
            )
            if not table_exists:
                if self.auto_create_table:
                    logger.info(
                        f"Table {self.schema}.{self.table_name} does not exist. "
                        "Creating with configured schema."
                    )
                    await self._create_table(conn)
                else:
                    raise RuntimeError(
                        f"Table {self.schema}.{self.table_name} does not exist "
                        "and auto_create_table is disabled."
                    )

            # Apply additive schema migrations for existing tables. Only
            # runs when auto_create_table=True — consumers managing their
            # own DDL are expected to apply migrations themselves. Every
            # step uses IF NOT EXISTS so repeated calls are safe.
            if self.auto_create_table:
                await self._migrate_schema(conn)

        self._initialized = True
        logger.info("pgvector store initialized successfully")

    async def _create_table(self, conn: asyncpg.Connection) -> None:
        """Create the embeddings table using configured column names."""
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

        # Build ID column definition based on id_type
        if self.id_type == "uuid":
            id_def = f"{self._col('id')} UUID PRIMARY KEY DEFAULT gen_random_uuid()"
        else:
            id_def = f"{self._col('id')} TEXT PRIMARY KEY"

        # Build CREATE TABLE with configured column names
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.{self.table_name} (
                {id_def},
                {self._col('domain_id')} VARCHAR(100),
                {self._col('document_id')} VARCHAR(255),
                {self._col('chunk_index')} INTEGER,
                {self._col('content')} TEXT,
                {self._col('embedding')} vector({self.dimensions}),
                {self._col('metadata')} JSONB DEFAULT '{{}}',
                {self._col('created_at')} TIMESTAMP DEFAULT NOW(),
                {self._col('updated_at')} TIMESTAMP DEFAULT NOW()
            )
        """)

        # Create HNSW index immediately if configured (HNSW works with empty tables)
        if self.auto_create_index and self.index_type == "hnsw":
            col_embedding = self._col("embedding")
            operator_class = self._get_operator_class()
            index_name = f"idx_{self.table_name}_{col_embedding}_hnsw"
            m = self.index_params.get("m", 16)
            ef_construction = self.index_params.get("ef_construction", 64)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {self.schema}.{self.table_name}
                USING hnsw ({col_embedding} {operator_class})
                WITH (m = {m}, ef_construction = {ef_construction})
            """)
            logger.info(f"Created HNSW index on {self.schema}.{self.table_name}")

        # Note: IVFFlat index is not created here because it requires existing data.
        # It will be auto-created during search() if auto_create_index=True and
        # row count exceeds min_rows_for_index. Or use create_index() explicitly.

        logger.info(
            f"Created table {self.schema}.{self.table_name} with columns: {self.columns}"
        )

    async def _migrate_schema(self, conn: asyncpg.Connection) -> None:
        """Apply additive schema migrations to an existing table.

        Runs unconditionally in the ``auto_create_table=True`` path —
        both after creation (no-op when the column is already present)
        and when the table pre-existed without the column. Uses
        ``ADD COLUMN IF NOT EXISTS`` so every step is idempotent.

        All steps here must remain additive (new columns, new indexes);
        destructive or type-changing migrations require consumer
        coordination and are out of scope.

        Current steps:

        * Add ``updated_at TIMESTAMP`` (Item 36). The column is added
          without a default so Postgres does not rewrite the table to
          backfill existing rows; ``NOW()`` is a volatile default and
          Postgres would otherwise evaluate it for every pre-existing
          row, tying up a large table for the duration of the rewrite
          and producing a misleading "written at migration time"
          timestamp on rows that were never actually re-ingested.
          Instead, pre-existing rows keep ``NULL`` in the new column —
          an honest "not re-ingested since the column was added"
          signal. A separate ``ALTER COLUMN ... SET DEFAULT NOW()``
          step then wires up the default for future inserts
          (catalog-only change, no rewrite). Fresh tables created via
          ``_create_table`` already have the default inline, so this
          step is a no-op for them.
        """
        col_updated_at = self._col('updated_at')
        await conn.execute(
            f"ALTER TABLE {self.schema}.{self.table_name} "
            f"ADD COLUMN IF NOT EXISTS {col_updated_at} TIMESTAMP"
        )
        await conn.execute(
            f"ALTER TABLE {self.schema}.{self.table_name} "
            f"ALTER COLUMN {col_updated_at} SET DEFAULT NOW()"
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("pgvector store closed")

    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to the store, upserting on ID conflict.

        When a vector with the same ID already exists, all content
        columns are updated (embedding, metadata, content, domain_id,
        document_id, chunk_index). The ``created_at`` timestamp is
        preserved to retain the original insertion time, and the
        ``updated_at`` timestamp is refreshed to ``NOW()`` to record
        the re-ingestion.

        Args:
            vectors: Vector data as numpy array(s).
            ids: Optional IDs for the vectors. Generated if not provided.
            metadata: Optional metadata dicts. Keys ``source_text``,
                ``document_id``, ``chunk_index``, and ``domain_id`` are
                extracted and stored in dedicated columns.

        Returns:
            List of vector IDs (provided or generated).
        """
        if not self._initialized:
            await self.initialize()


        # Prepare vectors
        vectors = self._prepare_vector(
            vectors, normalize=(self.metric == DistanceMetric.COSINE)
        )

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        # Generate metadata if not provided
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]

        # Build ID type cast
        id_cast = "::uuid" if self.id_type == "uuid" else ""

        async with self._pool.acquire() as conn:
            # Batch insert
            for i, (vec, vec_id, meta) in enumerate(zip(vectors, ids, metadata)):
                # Extract document info from metadata if available
                document_id = meta.get("document_id", meta.get("source"))
                chunk_index = meta.get("chunk_index", i)
                content = meta.get("source_text", meta.get("content", ""))
                domain_id = meta.get("domain_id", self.domain_id)

                # Upsert: refresh all content columns and updated_at on
                # conflict; preserve created_at (original insertion
                # timestamp). Uses explicit NOW() for updated_at rather
                # than EXCLUDED.updated_at — same wall-clock value, but
                # intent is unambiguous at the SQL level.
                await self._exec_with_id_type_guard(
                    conn,
                    "execute",
                    f"""
                    INSERT INTO {self.schema}.{self.table_name}
                        ({self._col('id')}, {self._col('domain_id')},
                         {self._col('document_id')}, {self._col('chunk_index')},
                         {self._col('content')}, {self._col('embedding')},
                         {self._col('metadata')})
                    VALUES ($1{id_cast}, $2, $3, $4, $5, $6::vector, $7::jsonb)
                    ON CONFLICT ({self._col('id')}) DO UPDATE SET
                        {self._col('embedding')} = EXCLUDED.{self._col('embedding')},
                        {self._col('metadata')} = EXCLUDED.{self._col('metadata')},
                        {self._col('content')} = EXCLUDED.{self._col('content')},
                        {self._col('domain_id')} = EXCLUDED.{self._col('domain_id')},
                        {self._col('document_id')} = EXCLUDED.{self._col('document_id')},
                        {self._col('chunk_index')} = EXCLUDED.{self._col('chunk_index')},
                        {self._col('updated_at')} = NOW()
                    """,
                    vec_id,
                    domain_id,
                    document_id,
                    chunk_index,
                    content,
                    f"[{','.join(str(x) for x in vec.tolist())}]",
                    json.dumps(meta),
                    vec_id=vec_id,
                )

        logger.debug(f"Added {len(ids)} vectors to pgvector")
        return ids

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Retrieve vectors by ID.

        Args:
            ids: Vector IDs to retrieve.
            include_metadata: Whether to include metadata dicts.
            include_timestamps: When True, inject ``_created_at`` and
                ``_updated_at`` (or configured keys) into each returned
                metadata dict, formatted per ``timestamps.format``
                config. Requires ``include_metadata=True`` — silently
                no-op otherwise. Legacy rows with ``updated_at IS NULL``
                surface as ``None`` (see vector-timestamps docs).
        """
        if not self._initialized:
            await self.initialize()

        import numpy as np

        id_cast = "::uuid" if self.id_type == "uuid" else ""
        col_embedding = self._col("embedding")
        col_metadata = self._col("metadata")
        col_id = self._col("id")
        col_created_at = self._col("created_at")
        col_updated_at = self._col("updated_at")

        ts_select = ""
        fetch_timestamps = include_timestamps and include_metadata
        if fetch_timestamps:
            ts_select = (
                f",\n                           "
                f"{col_created_at} as _ts_created, "
                f"{col_updated_at} as _ts_updated"
            )

        results: list[tuple[np.ndarray | None, dict[str, Any] | None]] = []
        async with self._pool.acquire() as conn:
            for vec_id in ids:
                row = await self._exec_with_id_type_guard(
                    conn,
                    "fetchrow",
                    f"""
                    SELECT {col_embedding}::text as embedding,
                           {col_metadata} as metadata{ts_select}
                    FROM {self.schema}.{self.table_name}
                    WHERE {col_id} = $1{id_cast}
                    """,
                    vec_id,
                    vec_id=vec_id,
                )

                if row is None:
                    results.append((None, None))
                else:
                    # Parse vector from PostgreSQL format
                    vec_str = row["embedding"]
                    vec = np.array(
                        [float(x) for x in vec_str.strip("[]").split(",")],
                        dtype=np.float32,
                    )
                    # asyncpg returns JSONB as dict or str depending on version
                    meta = None
                    if include_metadata and row["metadata"] is not None:
                        raw_meta = row["metadata"]
                        if isinstance(raw_meta, dict):
                            meta = raw_meta
                        elif isinstance(raw_meta, str):
                            meta = json.loads(raw_meta)
                        else:
                            meta = dict(raw_meta)
                    if fetch_timestamps:
                        meta = self._inject_timestamps(
                            meta,
                            created=row["_ts_created"],
                            updated=row["_ts_updated"],
                        )
                    results.append((vec, meta))

        return results

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        id_array_cast = "::uuid[]" if self.id_type == "uuid" else "::text[]"
        col_id = self._col("id")

        # When id_type="uuid", validate client-side so bulk errors
        # name the specific offending id. Without this, asyncpg's
        # error surface is "invalid input for array element at index
        # N" on a potentially large array — the full list would leak
        # into the guided error message and offer no way to identify
        # which id is malformed.
        if self.id_type == "uuid":
            self._validate_uuid_ids(ids)

        async with self._pool.acquire() as conn:
            result = await self._exec_with_id_type_guard(
                conn,
                "execute",
                f"""
                DELETE FROM {self.schema}.{self.table_name}
                WHERE {col_id} = ANY($1{id_array_cast})
                """,
                ids,
                vec_id=ids,
            )
            # Parse "DELETE n" to get count
            count = int(result.split()[-1])

        logger.debug(f"Deleted {count} vectors from pgvector")
        return count

    def _validate_uuid_ids(self, ids: list[str]) -> None:
        """Validate ids are UUID-formatted when ``id_type="uuid"``.

        Called before bulk operations so the guided error names the
        specific offending id(s) instead of dumping the whole list.
        Collects up to three bad ids so a batch with multiple
        problems surfaces meaningful context.
        """
        bad: list[str] = []
        for candidate in ids:
            try:
                UUID(str(candidate))
            except (ValueError, AttributeError, TypeError):
                bad.append(str(candidate))
                if len(bad) >= 3:
                    break
        if bad:
            sample = ", ".join(repr(b) for b in bad)
            more = "" if len(bad) < 3 else " (and possibly more)"
            raise ValueError(
                f"PgVectorStore id_type={self.id_type!r} but received "
                f"non-UUID id(s) in bulk operation: {sample}{more}. "
                f"Either set `id_type: \"text\"` in the vector store "
                f"config, or supply UUID-formatted string ids. "
                f"(table={self.schema}.{self.table_name})"
            )

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors using pgvector.

        Args:
            query_vector: Query vector.
            k: Number of results.
            filter: Optional metadata filter.
            include_metadata: Whether to include metadata dicts.
            include_timestamps: When True, inject ``_created_at`` and
                ``_updated_at`` (or configured keys) into each returned
                metadata dict, formatted per ``timestamps.format``
                config. Requires ``include_metadata=True`` — silently
                no-op otherwise.
        """
        if not self._initialized:
            await self.initialize()

        # Auto-create IVFFlat index if conditions are met
        await self._maybe_create_index()

        # Prepare query vector
        query = self._prepare_vector(
            query_vector, normalize=(self.metric == DistanceMetric.COSINE)
        )
        query_str = f"[{','.join(str(x) for x in query[0].tolist())}]"

        # Get column names
        col_id = self._col("id")
        col_embedding = self._col("embedding")
        col_metadata = self._col("metadata")
        col_content = self._col("content")
        col_domain_id = self._col("domain_id")
        col_created_at = self._col("created_at")
        col_updated_at = self._col("updated_at")

        fetch_timestamps = include_timestamps and include_metadata
        ts_select = ""
        if fetch_timestamps:
            ts_select = (
                f",\n                    "
                f"{col_created_at} as _ts_created, "
                f"{col_updated_at} as _ts_updated"
            )

        # Build distance operator based on metric
        if self.metric == DistanceMetric.COSINE:
            distance_op = "<=>"  # Cosine distance
            # Convert to similarity
            score_expr = f"1 - ({col_embedding} <=> $1::vector)"
        elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            distance_op = "<->"  # L2 distance
            score_expr = f"1.0 / (1.0 + ({col_embedding} <-> $1::vector))"
        elif self.metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            distance_op = "<#>"  # Negative inner product
            # Negate to get actual inner product
            score_expr = f"-({col_embedding} <#> $1::vector)"
        else:
            distance_op = "<=>"
            score_expr = f"1 - ({col_embedding} <=> $1::vector)"

        # Build WHERE clause for filters
        where_clauses = []
        params: list[Any] = [query_str]
        param_idx = 2

        # Add domain filter if configured
        if self.domain_id:
            where_clauses.append(f"{col_domain_id} = ${param_idx}")
            params.append(self.domain_id)
            param_idx += 1

        # Add metadata filters
        if filter:
            for key, value in filter.items():
                where_clauses.append(f"{col_metadata}->>'{key}' = ${param_idx}")
                params.append(str(value))
                param_idx += 1

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Execute search query
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    {col_id}::text as id,
                    {score_expr} as score,
                    {col_metadata} as metadata,
                    {col_content} as content{ts_select}
                FROM {self.schema}.{self.table_name}
                {where_sql}
                ORDER BY {col_embedding} {distance_op} $1::vector
                LIMIT {k}
                """,
                *params,
            )

        results = []
        for row in rows:
            meta = None
            if include_metadata and row["metadata"] is not None:
                raw_meta = row["metadata"]
                if isinstance(raw_meta, dict):
                    meta = raw_meta.copy()
                elif isinstance(raw_meta, str):
                    meta = json.loads(raw_meta)
                else:
                    meta = dict(raw_meta)
                # Add content to metadata for convenience
                meta["content"] = row["content"]
            if fetch_timestamps:
                meta = self._inject_timestamps(
                    meta,
                    created=row["_ts_created"],
                    updated=row["_ts_updated"],
                )
            results.append((row["id"], float(row["score"]), meta))

        return results

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for existing vectors.

        Refreshes ``updated_at = NOW()`` alongside the metadata change,
        mirroring the upsert semantics in ``add_vectors``: any write to
        a row is a re-ingestion signal.
        """
        if not self._initialized:
            await self.initialize()

        id_cast = "::uuid" if self.id_type == "uuid" else ""
        col_id = self._col("id")
        col_metadata = self._col("metadata")
        col_updated_at = self._col("updated_at")

        updated = 0
        async with self._pool.acquire() as conn:
            for vec_id, meta in zip(ids, metadata):
                result = await self._exec_with_id_type_guard(
                    conn,
                    "execute",
                    f"""
                    UPDATE {self.schema}.{self.table_name}
                    SET {col_metadata} = $2::jsonb,
                        {col_updated_at} = NOW()
                    WHERE {col_id} = $1{id_cast}
                    """,
                    vec_id,
                    json.dumps(meta),
                    vec_id=vec_id,
                )
                if result == "UPDATE 1":
                    updated += 1

        return updated

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Count vectors in the store."""
        if not self._initialized:
            await self.initialize()

        col_domain_id = self._col("domain_id")
        col_metadata = self._col("metadata")

        where_clauses = []
        params: list[Any] = []
        param_idx = 1

        if self.domain_id:
            where_clauses.append(f"{col_domain_id} = ${param_idx}")
            params.append(self.domain_id)
            param_idx += 1

        if filter:
            for key, value in filter.items():
                where_clauses.append(f"{col_metadata}->>'{key}' = ${param_idx}")
                params.append(str(value))
                param_idx += 1

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.schema}.{self.table_name} {where_sql}",
                *params,
            )

        return int(count or 0)

    async def metadata_fields(self) -> set[str]:
        """Discover metadata field names across all stored vectors.

        Uses PostgreSQL's ``jsonb_object_keys`` to extract the union of
        all top-level keys from the JSONB metadata column.
        """
        if not self._initialized:
            await self.initialize()

        col_metadata = self._col("metadata")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT key
                FROM {self.schema}.{self.table_name},
                     jsonb_object_keys({col_metadata}) AS key
                """,
            )

        return {row["key"] for row in rows}

    async def clear(self) -> None:
        """Clear all vectors from the store."""
        if not self._initialized:
            await self.initialize()

        col_domain_id = self._col("domain_id")

        async with self._pool.acquire() as conn:
            if self.domain_id:
                await conn.execute(
                    f"DELETE FROM {self.schema}.{self.table_name} "
                    f"WHERE {col_domain_id} = $1",
                    self.domain_id,
                )
            else:
                await conn.execute(f"TRUNCATE {self.schema}.{self.table_name}")

        logger.info("Cleared pgvector store")

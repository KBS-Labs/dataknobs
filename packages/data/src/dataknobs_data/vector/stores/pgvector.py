"""PostgreSQL pgvector backend implementation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

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
        id_type: ID column type - 'uuid' or 'text' (default: 'uuid')

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
        import os

        self.connection_string = self.config.get("connection_string")
        if not self.connection_string:
            self.connection_string = os.environ.get("DATABASE_URL")

        if not self.connection_string:
            raise ValueError(
                "connection_string required for pgvector backend. "
                "Set in config or DATABASE_URL environment variable."
            )

        # Normalize connection string format
        if self.connection_string.startswith("postgresql+asyncpg://"):
            self.connection_string = self.connection_string.replace(
                "postgresql+asyncpg://", "postgresql://"
            )

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
        self.id_type = self.config.get("id_type", "uuid")
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
                {self._col('created_at')} TIMESTAMP DEFAULT NOW()
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
        """Add vectors to the store."""
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

                await conn.execute(
                    f"""
                    INSERT INTO {self.schema}.{self.table_name}
                        ({self._col('id')}, {self._col('domain_id')},
                         {self._col('document_id')}, {self._col('chunk_index')},
                         {self._col('content')}, {self._col('embedding')},
                         {self._col('metadata')})
                    VALUES ($1{id_cast}, $2, $3, $4, $5, $6::vector, $7::jsonb)
                    ON CONFLICT ({self._col('id')}) DO UPDATE SET
                        {self._col('embedding')} = EXCLUDED.{self._col('embedding')},
                        {self._col('metadata')} = EXCLUDED.{self._col('metadata')}
                    """,
                    vec_id,
                    domain_id,
                    document_id,
                    chunk_index,
                    content,
                    f"[{','.join(str(x) for x in vec.tolist())}]",
                    json.dumps(meta),
                )

        logger.debug(f"Added {len(ids)} vectors to pgvector")
        return ids

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Retrieve vectors by ID."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        id_cast = "::uuid" if self.id_type == "uuid" else ""
        col_embedding = self._col("embedding")
        col_metadata = self._col("metadata")
        col_id = self._col("id")

        results: list[tuple[np.ndarray | None, dict[str, Any] | None]] = []
        async with self._pool.acquire() as conn:
            for vec_id in ids:
                row = await conn.fetchrow(
                    f"""
                    SELECT {col_embedding}::text as embedding, {col_metadata} as metadata
                    FROM {self.schema}.{self.table_name}
                    WHERE {col_id} = $1{id_cast}
                    """,
                    vec_id,
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
                    results.append((vec, meta))

        return results

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        id_array_cast = "::uuid[]" if self.id_type == "uuid" else "::text[]"
        col_id = self._col("id")

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.schema}.{self.table_name}
                WHERE {col_id} = ANY($1{id_array_cast})
                """,
                ids,
            )
            # Parse "DELETE n" to get count
            count = int(result.split()[-1])

        logger.debug(f"Deleted {count} vectors from pgvector")
        return count

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors using pgvector."""
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
                    {col_content} as content
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
            results.append((row["id"], float(row["score"]), meta))

        return results

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for existing vectors."""
        if not self._initialized:
            await self.initialize()

        id_cast = "::uuid" if self.id_type == "uuid" else ""
        col_id = self._col("id")
        col_metadata = self._col("metadata")

        updated = 0
        async with self._pool.acquire() as conn:
            for vec_id, meta in zip(ids, metadata):
                result = await conn.execute(
                    f"""
                    UPDATE {self.schema}.{self.table_name}
                    SET {col_metadata} = $2::jsonb
                    WHERE {col_id} = $1{id_cast}
                    """,
                    vec_id,
                    json.dumps(meta),
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

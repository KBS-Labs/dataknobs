"""Tests for PostgreSQL pgvector vector store.

These tests require a running PostgreSQL instance with pgvector extension.
Set TEST_POSTGRES=true to enable these tests.
"""

import os
import uuid

import numpy as np
import pytest

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="pgvector tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

# Check if asyncpg is available
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Only import if available to avoid import errors
if ASYNCPG_AVAILABLE:
    from dataknobs_data.vector.stores.pgvector import PgVectorStore
    from dataknobs_data.vector.types import DistanceMetric


def get_test_connection_string() -> str:
    """Build connection string from environment variables."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    database = os.environ.get("POSTGRES_DB", "test_dataknobs")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture(scope="session")
def ensure_pgvector_extension():
    """Ensure pgvector extension is available in test database."""
    if not os.environ.get("TEST_POSTGRES", "").lower() == "true":
        return

    if not ASYNCPG_AVAILABLE:
        pytest.skip("asyncpg not installed")

    import asyncio

    async def setup_extension():
        conn_str = get_test_connection_string()
        conn = await asyncpg.connect(conn_str)
        try:
            # Try to create the extension (may already exist)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as e:
            print(f"Warning: Could not create vector extension: {e}")
        finally:
            await conn.close()

    # Use asyncio.run for cleaner event loop handling
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If there's already a running loop, create a new one in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(asyncio.run, setup_extension()).result()
    else:
        asyncio.run(setup_extension())

    yield


@pytest.fixture
def pgvector_config(ensure_pgvector_extension):
    """Generate unique pgvector configuration for each test."""
    table_name = f"test_vectors_{uuid.uuid4().hex[:8]}"
    return {
        "connection_string": get_test_connection_string(),
        "dimensions": 128,
        "metric": "cosine",
        "schema": "public",
        "table_name": table_name,
        "auto_create_table": True,
    }


@pytest.fixture
async def pgvector_store(pgvector_config):
    """Create a pgvector store for testing with cleanup."""
    store = PgVectorStore(pgvector_config)
    await store.initialize()

    yield store

    # Cleanup: drop the test table
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}"
            )
    except Exception:
        pass
    await store.close()


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
class TestPgVectorStoreConfiguration:
    """Test PgVectorStore configuration options."""

    def test_config_from_connection_string(self, pgvector_config):
        """Test configuration with explicit connection string."""
        store = PgVectorStore(pgvector_config)
        assert store.connection_string == pgvector_config["connection_string"]
        assert store.dimensions == 128
        assert store.table_name == pgvector_config["table_name"]

    def test_config_default_values(self, ensure_pgvector_extension):
        """Test default configuration values."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 768,
        }
        store = PgVectorStore(config)

        assert store.table_name == "knowledge_embeddings"
        assert store.schema == "edubot"
        assert store.pool_min_size == 2
        assert store.pool_max_size == 10
        assert store.auto_create_table is True
        assert store.id_type == "uuid"
        assert store.domain_id is None

    def test_config_custom_values(self, ensure_pgvector_extension):
        """Test custom configuration values."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 768,
            "table_name": "custom_table",
            "schema": "public",
            "pool_min_size": 5,
            "pool_max_size": 20,
            "auto_create_table": False,
            "id_type": "text",
            "domain_id": "my-domain",
        }
        store = PgVectorStore(config)

        assert store.table_name == "custom_table"
        assert store.schema == "public"
        assert store.pool_min_size == 5
        assert store.pool_max_size == 20
        assert store.auto_create_table is False
        assert store.id_type == "text"
        assert store.domain_id == "my-domain"

    def test_config_invalid_id_type(self, pgvector_config):
        """Test that invalid id_type raises ValueError."""
        pgvector_config["id_type"] = "invalid"
        with pytest.raises(ValueError, match="id_type must be 'uuid' or 'text'"):
            PgVectorStore(pgvector_config)

    def test_config_missing_connection_string(self):
        """Test that missing connection string raises ValueError."""
        # Clear DATABASE_URL if set
        with pytest.MonkeyPatch().context() as m:
            m.delenv("DATABASE_URL", raising=False)
            with pytest.raises(ValueError, match="connection_string required"):
                PgVectorStore({"dimensions": 768})

    def test_config_normalizes_asyncpg_url(self, ensure_pgvector_extension):
        """Test that postgresql+asyncpg:// URLs are normalized."""
        conn_str = get_test_connection_string()
        asyncpg_url = conn_str.replace("postgresql://", "postgresql+asyncpg://")
        store = PgVectorStore({
            "connection_string": asyncpg_url,
            "dimensions": 768,
        })
        assert store.connection_string == conn_str

    def test_column_mappings_default(self, pgvector_config):
        """Test default column mappings."""
        store = PgVectorStore(pgvector_config)

        assert store._col("id") == "id"
        assert store._col("embedding") == "embedding"
        assert store._col("content") == "content"
        assert store._col("metadata") == "metadata"
        assert store._col("domain_id") == "domain_id"

    def test_column_mappings_custom(self, pgvector_config):
        """Test custom column mappings."""
        pgvector_config["columns"] = {
            "id": "item_id",
            "embedding": "vec",
            "content": "text_data",
        }
        store = PgVectorStore(pgvector_config)

        assert store._col("id") == "item_id"
        assert store._col("embedding") == "vec"
        assert store._col("content") == "text_data"
        # Non-overridden columns use defaults
        assert store._col("metadata") == "metadata"


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
@pytest.mark.asyncio
class TestPgVectorStoreBasicOperations:
    """Test basic vector store operations."""

    async def test_initialize(self, pgvector_store):
        """Test store initialization."""
        assert pgvector_store._initialized is True
        assert pgvector_store._pool is not None

    async def test_add_vectors(self, pgvector_store):
        """Test adding vectors."""
        vectors = np.random.rand(5, 128).astype(np.float32)
        ids = [str(uuid.uuid4()) for _ in range(5)]
        metadata = [{"index": i} for i in range(5)]

        result_ids = await pgvector_store.add_vectors(
            vectors, ids=ids, metadata=metadata
        )

        assert result_ids == ids
        count = await pgvector_store.count()
        assert count == 5

    async def test_add_vectors_generates_ids(self, pgvector_store):
        """Test that add_vectors generates IDs when not provided."""
        vectors = np.random.rand(3, 128).astype(np.float32)
        result_ids = await pgvector_store.add_vectors(vectors)

        assert len(result_ids) == 3
        # IDs should be valid UUIDs (36 chars with dashes)
        for id_ in result_ids:
            assert len(id_) == 36

    async def test_get_vectors(self, pgvector_store):
        """Test retrieving vectors by ID."""
        vectors = np.random.rand(3, 128).astype(np.float32)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        metadata = [{"key": f"value{i}"} for i in range(3)]

        await pgvector_store.add_vectors(vectors, ids=ids, metadata=metadata)

        # Get vectors
        results = await pgvector_store.get_vectors(ids, include_metadata=True)

        assert len(results) == 3
        for i, (vec, meta) in enumerate(results):
            assert vec is not None
            assert meta is not None
            assert meta["key"] == f"value{i}"

    async def test_get_vectors_not_found(self, pgvector_store):
        """Test retrieving non-existent vectors."""
        results = await pgvector_store.get_vectors(
            [str(uuid.uuid4())], include_metadata=True
        )
        assert len(results) == 1
        assert results[0] == (None, None)

    async def test_delete_vectors(self, pgvector_store):
        """Test deleting vectors by ID."""
        vectors = np.random.rand(5, 128).astype(np.float32)
        ids = [str(uuid.uuid4()) for _ in range(5)]

        await pgvector_store.add_vectors(vectors, ids=ids)
        assert await pgvector_store.count() == 5

        # Delete some vectors
        deleted = await pgvector_store.delete_vectors(ids[:3])
        assert deleted == 3
        assert await pgvector_store.count() == 2

    async def test_update_metadata(self, pgvector_store):
        """Test updating vector metadata."""
        vectors = np.random.rand(2, 128).astype(np.float32)
        ids = [str(uuid.uuid4()) for _ in range(2)]
        metadata = [{"version": 1}, {"version": 1}]

        await pgvector_store.add_vectors(vectors, ids=ids, metadata=metadata)

        # Update metadata
        new_metadata = [{"version": 2, "updated": True}, {"version": 3}]
        updated = await pgvector_store.update_metadata(ids, new_metadata)
        assert updated == 2

        # Verify updates
        results = await pgvector_store.get_vectors(ids)
        assert results[0][1]["version"] == 2
        assert results[0][1]["updated"] is True
        assert results[1][1]["version"] == 3

    async def test_count(self, pgvector_store):
        """Test counting vectors."""
        assert await pgvector_store.count() == 0

        vectors = np.random.rand(10, 128).astype(np.float32)
        metadata = [{"type": "A" if i < 6 else "B"} for i in range(10)]
        await pgvector_store.add_vectors(vectors, metadata=metadata)

        assert await pgvector_store.count() == 10
        assert await pgvector_store.count(filter={"type": "A"}) == 6
        assert await pgvector_store.count(filter={"type": "B"}) == 4

    async def test_clear(self, pgvector_store):
        """Test clearing all vectors."""
        vectors = np.random.rand(5, 128).astype(np.float32)
        await pgvector_store.add_vectors(vectors)
        assert await pgvector_store.count() == 5

        await pgvector_store.clear()
        assert await pgvector_store.count() == 0


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
@pytest.mark.asyncio
class TestPgVectorStoreSearch:
    """Test vector similarity search operations."""

    async def test_search_cosine(self, ensure_pgvector_extension):
        """Test vector search with cosine similarity."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "metric": "cosine",
            "schema": "public",
            "table_name": f"test_cosine_{uuid.uuid4().hex[:8]}",
            "id_type": "text",  # Use text IDs for readable test assertions
            "auto_create_table": True,
            # Use HNSW index for reliable results with small datasets
            "index_type": "hnsw",
            "auto_create_index": True,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            # Add random vectors with different similarities to query
            np.random.seed(42)  # For reproducibility
            base_vector = np.random.rand(128).astype(np.float32)
            similar_vector = base_vector + np.random.rand(128).astype(np.float32) * 0.1
            different_vector = np.random.rand(128).astype(np.float32)

            vectors = np.array([
                base_vector,
                similar_vector,
                different_vector,
            ], dtype=np.float32)
            ids = ["base", "similar", "different"]
            metadata = [{"type": name} for name in ids]

            await store.add_vectors(vectors, ids=ids, metadata=metadata)

            # Search for vector similar to base
            results = await store.search(base_vector, k=2)

            assert len(results) == 2
            # First result should be base (exact match)
            assert results[0][0] == "base"
            assert results[0][1] > 0.99  # Near-perfect match
            # Second result should be similar
            assert results[1][0] == "similar"
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()

    async def test_search_with_filter(self, pgvector_store):
        """Test vector search with metadata filter."""
        vectors = np.random.rand(10, 128).astype(np.float32)
        ids = [str(uuid.uuid4()) for _ in range(10)]
        metadata = [{"category": "A" if i < 5 else "B", "index": i} for i in range(10)]

        await pgvector_store.add_vectors(vectors, ids=ids, metadata=metadata)

        # Search with filter for category B only
        query = vectors[0]
        results = await pgvector_store.search(query, k=10, filter={"category": "B"})

        # Should only return category B vectors
        assert all(r[2]["category"] == "B" for r in results)
        assert len(results) == 5  # Should only return the 5 category B vectors

    async def test_search_returns_k_results(self, pgvector_store):
        """Test that search returns exactly k results."""
        vectors = np.random.rand(20, 128).astype(np.float32)
        await pgvector_store.add_vectors(vectors)

        query = np.random.rand(128).astype(np.float32)

        results = await pgvector_store.search(query, k=5)
        assert len(results) == 5

        results = await pgvector_store.search(query, k=10)
        assert len(results) == 10

    async def test_search_includes_metadata(self, pgvector_store):
        """Test that search includes metadata when requested."""
        vectors = np.random.rand(5, 128).astype(np.float32)
        metadata = [{"key": f"value{i}"} for i in range(5)]
        await pgvector_store.add_vectors(vectors, metadata=metadata)

        query = vectors[0]
        results = await pgvector_store.search(query, k=5, include_metadata=True)

        for id_, score, meta in results:
            assert meta is not None
            assert "key" in meta


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
@pytest.mark.asyncio
class TestPgVectorStoreDistanceMetrics:
    """Test different distance metrics."""

    async def test_euclidean_metric(self, ensure_pgvector_extension):
        """Test Euclidean distance metric."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "metric": "euclidean",
            "schema": "public",
            "table_name": f"test_euclidean_{uuid.uuid4().hex[:8]}",
            "id_type": "text",  # Use text IDs for readable test assertions
            "auto_create_table": True,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            # Add vectors
            vectors = np.array([
                [0.0, 0.0] + [0.0] * 126,
                [1.0, 0.0] + [0.0] * 126,
                [0.0, 1.0] + [0.0] * 126,
                [3.0, 4.0] + [0.0] * 126,
            ], dtype=np.float32)
            ids = ["origin", "x1", "y1", "far"]
            await store.add_vectors(vectors, ids=ids)

            # Search from near origin - should find origin first
            query = np.array([0.1, 0.1] + [0.0] * 126, dtype=np.float32)
            results = await store.search(query, k=2)

            assert len(results) == 2
            # Origin should be closest
            assert results[0][0] == "origin"
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()

    async def test_inner_product_metric(self, ensure_pgvector_extension):
        """Test inner product distance metric."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "metric": "inner_product",
            "schema": "public",
            "table_name": f"test_ip_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            vectors = np.random.rand(10, 128).astype(np.float32)
            await store.add_vectors(vectors)

            query = vectors[0]
            results = await store.search(query, k=5)

            # Should return results
            assert len(results) == 5
            # First result should have highest score
            assert results[0][1] >= results[1][1]
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
@pytest.mark.asyncio
class TestPgVectorStoreDomainIsolation:
    """Test multi-tenant domain isolation."""

    async def test_domain_isolation(self, ensure_pgvector_extension):
        """Test that domains are properly isolated."""
        table_name = f"test_domains_{uuid.uuid4().hex[:8]}"
        conn_str = get_test_connection_string()

        # Create stores for two domains
        store1 = PgVectorStore({
            "connection_string": conn_str,
            "dimensions": 128,
            "schema": "public",
            "table_name": table_name,
            "domain_id": "domain-1",
            "auto_create_table": True,
        })
        store2 = PgVectorStore({
            "connection_string": conn_str,
            "dimensions": 128,
            "schema": "public",
            "table_name": table_name,
            "domain_id": "domain-2",
            "auto_create_table": False,  # Table already created by store1
        })

        await store1.initialize()
        await store2.initialize()

        try:
            # Add vectors to each domain
            vectors1 = np.random.rand(5, 128).astype(np.float32)
            vectors2 = np.random.rand(3, 128).astype(np.float32)

            await store1.add_vectors(vectors1)
            await store2.add_vectors(vectors2)

            # Each domain should only see its own vectors
            assert await store1.count() == 5
            assert await store2.count() == 3

            # Search should only return domain-specific results
            query = np.random.rand(128).astype(np.float32)
            results1 = await store1.search(query, k=10)
            results2 = await store2.search(query, k=10)

            assert len(results1) == 5
            assert len(results2) == 3

            # Clear domain-1
            await store1.clear()
            assert await store1.count() == 0
            assert await store2.count() == 3  # domain-2 unaffected
        finally:
            # Cleanup - drop the shared table
            async with store1._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS public.{table_name}")
            await store1.close()
            await store2.close()


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
@pytest.mark.asyncio
class TestPgVectorStoreTextIds:
    """Test text-based IDs instead of UUIDs."""

    async def test_text_id_type(self, ensure_pgvector_extension):
        """Test using text IDs instead of UUIDs."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "schema": "public",
            "table_name": f"test_text_ids_{uuid.uuid4().hex[:8]}",
            "id_type": "text",
            "auto_create_table": True,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            vectors = np.random.rand(3, 128).astype(np.float32)
            ids = ["item-001", "item-002", "item-003"]
            metadata = [{"name": f"Item {i}"} for i in range(3)]

            result_ids = await store.add_vectors(vectors, ids=ids, metadata=metadata)
            assert result_ids == ids

            # Retrieve by text ID
            results = await store.get_vectors(["item-001", "item-002"])
            assert results[0][0] is not None
            assert results[1][0] is not None

            # Delete by text ID
            deleted = await store.delete_vectors(["item-001"])
            assert deleted == 1
            assert await store.count() == 2
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
@pytest.mark.asyncio
class TestPgVectorStoreCustomColumns:
    """Test custom column mappings."""

    async def test_custom_column_names(self, ensure_pgvector_extension):
        """Test creating table with custom column names."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "schema": "public",
            "table_name": f"test_custom_cols_{uuid.uuid4().hex[:8]}",
            "id_type": "text",
            "columns": {
                "id": "item_id",
                "embedding": "vec_data",
                "content": "description",
                "metadata": "attributes",
                "domain_id": "category",
            },
            "auto_create_table": True,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            vectors = np.random.rand(3, 128).astype(np.float32)
            ids = ["prod-1", "prod-2", "prod-3"]
            metadata = [{"type": "widget"}, {"type": "gadget"}, {"type": "widget"}]

            await store.add_vectors(vectors, ids=ids, metadata=metadata)
            assert await store.count() == 3

            # Search and filter should work with custom columns
            results = await store.search(vectors[0], k=3, filter={"type": "widget"})
            assert len(results) == 2
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
@pytest.mark.asyncio
class TestPgVectorStoreEdgeCases:
    """Test edge cases and error handling."""

    async def test_empty_search(self, pgvector_store):
        """Test searching an empty store."""
        query = np.random.rand(128).astype(np.float32)
        results = await pgvector_store.search(query, k=5)
        assert len(results) == 0

    async def test_single_vector(self, pgvector_store):
        """Test operations with single vector."""
        vector = np.random.rand(128).astype(np.float32)
        ids = await pgvector_store.add_vectors(vector)

        assert len(ids) == 1
        results = await pgvector_store.search(vector, k=1)
        assert len(results) == 1
        assert results[0][1] > 0.99  # Near-perfect match

    async def test_upsert_behavior(self, pgvector_store):
        """Test that adding vectors with same ID updates them."""
        vector1 = np.random.rand(128).astype(np.float32)
        vector2 = np.random.rand(128).astype(np.float32)
        id_ = str(uuid.uuid4())

        # Add first vector
        await pgvector_store.add_vectors(vector1, ids=[id_])
        assert await pgvector_store.count() == 1

        # Add second vector with same ID - should upsert
        await pgvector_store.add_vectors(
            vector2, ids=[id_], metadata=[{"updated": True}]
        )
        assert await pgvector_store.count() == 1

        # Verify the vector was updated
        results = await pgvector_store.get_vectors([id_])
        assert results[0][1]["updated"] is True

    async def test_large_batch(self, pgvector_store):
        """Test adding a large batch of vectors."""
        vectors = np.random.rand(100, 128).astype(np.float32)
        metadata = [{"index": i} for i in range(100)]

        ids = await pgvector_store.add_vectors(vectors, metadata=metadata)
        assert len(ids) == 100
        assert await pgvector_store.count() == 100

        # Search should work
        query = vectors[50]
        results = await pgvector_store.search(query, k=10)
        assert len(results) == 10


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
class TestPgVectorStoreIndexConfiguration:
    """Test index configuration options."""

    def test_config_invalid_index_type(self, pgvector_config):
        """Test that invalid index_type raises ValueError."""
        pgvector_config["index_type"] = "invalid"
        with pytest.raises(ValueError, match="index_type must be"):
            PgVectorStore(pgvector_config)

    def test_config_valid_index_types(self, ensure_pgvector_extension):
        """Test valid index_type options."""
        conn_str = get_test_connection_string()

        for idx_type in ("none", "hnsw", "ivfflat"):
            store = PgVectorStore({
                "connection_string": conn_str,
                "dimensions": 128,
                "index_type": idx_type,
            })
            assert store.index_type == idx_type

    def test_config_index_defaults(self, pgvector_config):
        """Test default index configuration values."""
        store = PgVectorStore(pgvector_config)

        assert store.index_type == "none"
        assert store.auto_create_index is False
        assert store.min_rows_for_index == 1000
        assert store.index_params == {}

    def test_config_custom_index_params(self, pgvector_config):
        """Test custom index configuration values."""
        pgvector_config["index_type"] = "hnsw"
        pgvector_config["auto_create_index"] = True
        pgvector_config["min_rows_for_index"] = 500
        pgvector_config["index_params"] = {"m": 32, "ef_construction": 128}

        store = PgVectorStore(pgvector_config)

        assert store.index_type == "hnsw"
        assert store.auto_create_index is True
        assert store.min_rows_for_index == 500
        assert store.index_params["m"] == 32
        assert store.index_params["ef_construction"] == 128


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
@pytest.mark.asyncio
class TestPgVectorStoreIndexOperations:
    """Test index creation and management operations."""

    async def test_check_index_exists_no_index(self, pgvector_store):
        """Test _check_index_exists returns False when no index exists."""
        exists = await pgvector_store._check_index_exists()
        assert exists is False

    async def test_create_index_hnsw_explicit(self, ensure_pgvector_extension):
        """Test explicit HNSW index creation."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "schema": "public",
            "table_name": f"test_hnsw_explicit_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            # No index initially
            assert await store._check_index_exists() is False

            # Create HNSW index explicitly
            created = await store.create_index("hnsw", {"m": 16, "ef_construction": 64})
            assert created is True

            # Index should now exist
            assert await store._check_index_exists() is True

            # Creating again should return False (already exists)
            created = await store.create_index("hnsw")
            assert created is False
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()

    async def test_create_index_ivfflat_explicit(self, ensure_pgvector_extension):
        """Test explicit IVFFlat index creation."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "schema": "public",
            "table_name": f"test_ivf_explicit_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            # Add some data first (IVFFlat works better with data)
            vectors = np.random.rand(50, 128).astype(np.float32)
            await store.add_vectors(vectors)

            # Create IVFFlat index
            created = await store.create_index("ivfflat", {"lists": 10})
            assert created is True
            assert await store._check_index_exists() is True
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()

    async def test_create_index_invalid_type(self, pgvector_store):
        """Test create_index with invalid type raises error."""
        with pytest.raises(ValueError, match="index_type must be"):
            await pgvector_store.create_index("invalid")

    async def test_create_index_none_type(self, pgvector_store):
        """Test create_index with 'none' type raises error."""
        with pytest.raises(ValueError, match="Cannot create index"):
            await pgvector_store.create_index("none")

    async def test_hnsw_auto_created_on_table_creation(self, ensure_pgvector_extension):
        """Test HNSW index is auto-created when table is created."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "schema": "public",
            "table_name": f"test_hnsw_auto_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
            "index_type": "hnsw",
            "auto_create_index": True,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            # HNSW index should exist immediately after initialization
            assert await store._check_index_exists() is True
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()

    async def test_ivfflat_not_auto_created_below_threshold(self, ensure_pgvector_extension):
        """Test IVFFlat index is NOT auto-created when below threshold."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "schema": "public",
            "table_name": f"test_ivf_threshold_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
            "index_type": "ivfflat",
            "auto_create_index": True,
            "min_rows_for_index": 100,  # Threshold of 100 rows
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            # Add only 50 vectors (below threshold)
            vectors = np.random.rand(50, 128).astype(np.float32)
            await store.add_vectors(vectors)

            # Search should trigger _maybe_create_index but not create index
            query = np.random.rand(128).astype(np.float32)
            await store.search(query, k=5)

            # Index should NOT exist (below threshold)
            assert await store._check_index_exists() is False
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()

    async def test_ivfflat_auto_created_above_threshold(self, ensure_pgvector_extension):
        """Test IVFFlat index IS auto-created when above threshold."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "schema": "public",
            "table_name": f"test_ivf_auto_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
            "index_type": "ivfflat",
            "auto_create_index": True,
            "min_rows_for_index": 50,  # Low threshold for testing
            "index_params": {"lists": 10},
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            # Add 100 vectors (above threshold)
            vectors = np.random.rand(100, 128).astype(np.float32)
            await store.add_vectors(vectors)

            # No index yet (not searched)
            assert await store._check_index_exists() is False

            # Search should trigger auto-creation
            query = np.random.rand(128).astype(np.float32)
            await store.search(query, k=5)

            # Index should now exist
            assert await store._check_index_exists() is True
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()

    async def test_no_auto_create_when_disabled(self, ensure_pgvector_extension):
        """Test index is NOT auto-created when auto_create_index=False."""
        config = {
            "connection_string": get_test_connection_string(),
            "dimensions": 128,
            "schema": "public",
            "table_name": f"test_no_auto_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
            "index_type": "ivfflat",
            "auto_create_index": False,  # Disabled
            "min_rows_for_index": 10,
        }
        store = PgVectorStore(config)
        await store.initialize()

        try:
            # Add vectors above threshold
            vectors = np.random.rand(50, 128).astype(np.float32)
            await store.add_vectors(vectors)

            # Search
            query = np.random.rand(128).astype(np.float32)
            await store.search(query, k=5)

            # Index should NOT exist (auto_create_index=False)
            assert await store._check_index_exists() is False
        finally:
            async with store._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
            await store.close()

    async def test_index_with_different_metrics(self, ensure_pgvector_extension):
        """Test index creation with different distance metrics."""
        for metric in ("cosine", "euclidean", "inner_product"):
            config = {
                "connection_string": get_test_connection_string(),
                "dimensions": 64,
                "metric": metric,
                "schema": "public",
                "table_name": f"test_metric_{metric}_{uuid.uuid4().hex[:8]}",
                "auto_create_table": True,
            }
            store = PgVectorStore(config)
            await store.initialize()

            try:
                # Create HNSW index - should use correct operator class
                created = await store.create_index("hnsw")
                assert created is True
                assert await store._check_index_exists() is True
            finally:
                async with store._pool.acquire() as conn:
                    await conn.execute(f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}")
                await store.close()

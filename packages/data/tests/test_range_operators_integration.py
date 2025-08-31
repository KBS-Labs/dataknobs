"""Integration tests for range operators with real backends."""
import os
import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime, timedelta
from dataknobs_data import Record, Query, Filter, Operator


# Skip these tests if backends are not available
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def ensure_postgres_test_db():
    """Ensure the test database exists for integration tests."""
    if not os.environ.get("TEST_POSTGRES", "").lower() == "true":
        return
    
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", 5432))
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    db_name = os.environ.get("POSTGRES_DB", "test_dataknobs")
    
    # Connect to postgres database to create test database
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        
        if not exists:
            # Create the database
            cur.execute(f"CREATE DATABASE {db_name}")
        
        cur.close()
        conn.close()
    except Exception as e:
        # If we can't create the database, the tests will fail anyway
        print(f"Warning: Could not ensure test database exists: {e}")


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true"
)
class TestPostgresRangeOperators:
    """Test BETWEEN operators with real PostgreSQL backend."""
    
    @pytest.fixture
    def postgres_db(self, ensure_postgres_test_db):
        """Create a PostgreSQL database connection."""
        from dataknobs_data.backends.postgres import SyncPostgresDatabase
        
        import uuid
        # Use public schema with unique table name to avoid conflicts
        table_name = f"test_range_{uuid.uuid4().hex[:8]}"
        
        db = SyncPostgresDatabase(config={
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "test_dataknobs"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "schema": "public",  # Use public schema which always exists
            "table": table_name
        })
        
        # Connect
        db.connect()
        
        # Clean up any existing data
        try:
            db.db.execute(f"TRUNCATE TABLE {db.schema_name}.{db.table_name}")
        except:
            pass  # Table might not exist yet
        
        yield db
        
        # Cleanup
        try:
            db.db.execute(f"DROP TABLE IF EXISTS {db.schema_name}.{db.table_name}")
        except:
            pass
        finally:
            db.close()
    
    def test_numeric_between_postgres(self, postgres_db):
        """Test BETWEEN with numeric values in PostgreSQL."""
        # Insert test data
        records = [
            Record(id="1", data={"price": 99.99, "quantity": 10}),
            Record(id="2", data={"price": 149.99, "quantity": 25}),
            Record(id="3", data={"price": 199.99, "quantity": 50}),
            Record(id="4", data={"price": 299.99, "quantity": 100}),
        ]
        
        for record in records:
            postgres_db.create(record)
        
        # Test BETWEEN with floats
        query = Query(filters=[Filter("price", Operator.BETWEEN, (100, 200))])
        results = postgres_db.search(query)
        assert len(results) == 2
        prices = [r.get_value("price") for r in results]
        assert all(100 <= p <= 200 for p in prices)
        
        # Test BETWEEN with integers
        query = Query(filters=[Filter("quantity", Operator.BETWEEN, (20, 60))])
        results = postgres_db.search(query)
        assert len(results) == 2
        quantities = [r.get_value("quantity") for r in results]
        assert all(20 <= q <= 60 for q in quantities)
        
        # Test NOT_BETWEEN
        query = Query(filters=[Filter("price", Operator.NOT_BETWEEN, (100, 200))])
        results = postgres_db.search(query)
        assert len(results) == 2
        prices = [r.get_value("price") for r in results]
        assert all(p < 100 or p > 200 for p in prices)
    
    def test_datetime_between_postgres(self, postgres_db):
        """Test BETWEEN with datetime values in PostgreSQL."""
        base_time = datetime(2025, 1, 15, 12, 0, 0)
        
        records = [
            Record(id="1", data={"created_at": (base_time - timedelta(days=5)).isoformat()}),
            Record(id="2", data={"created_at": (base_time - timedelta(days=1)).isoformat()}),
            Record(id="3", data={"created_at": base_time.isoformat()}),
            Record(id="4", data={"created_at": (base_time + timedelta(days=3)).isoformat()}),
        ]
        
        for record in records:
            postgres_db.create(record)
        
        # Test BETWEEN with datetime objects
        start = base_time - timedelta(days=2)
        end = base_time + timedelta(days=1)
        query = Query(filters=[Filter("created_at", Operator.BETWEEN, (start, end))])
        results = postgres_db.search(query)
        assert len(results) == 2
        # Check the actual timestamps are in range
        for r in results:
            ts = r.get_value("created_at")
            ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
            assert start <= ts_dt <= end
    
    def test_string_between_postgres(self, postgres_db):
        """Test BETWEEN with string values in PostgreSQL."""
        records = [
            Record(id="1", data={"product_code": "AA100"}),
            Record(id="2", data={"product_code": "AB200"}),
            Record(id="3", data={"product_code": "AC300"}),
            Record(id="4", data={"product_code": "BA100"}),
        ]
        
        for record in records:
            postgres_db.create(record)
        
        # Test BETWEEN with strings
        query = Query(filters=[Filter("product_code", Operator.BETWEEN, ("AB000", "AC999"))])
        results = postgres_db.search(query)
        assert len(results) == 2
        codes = [r.get_value("product_code") for r in results]
        assert all("AB000" <= c <= "AC999" for c in codes)
    
    @pytest.mark.asyncio
    async def test_async_postgres_between(self, ensure_postgres_test_db):
        """Test BETWEEN with async PostgreSQL backend."""
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase
        
        import uuid
        # Use public schema with unique table name to avoid conflicts
        table_name = f"test_async_range_{uuid.uuid4().hex[:8]}"
        
        db = AsyncPostgresDatabase(config={
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "test_dataknobs"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "schema": "public",  # Use public schema which always exists
            "table": table_name
        })
        
        await db.connect()
        
        try:
            # Insert test data
            records = [
                Record(id="1", data={"score": 50}),
                Record(id="2", data={"score": 70}),
                Record(id="3", data={"score": 85}),
                Record(id="4", data={"score": 95}),
            ]
            
            for record in records:
                await db.create(record)
            
            # Test BETWEEN
            query = Query(filters=[Filter("score", Operator.BETWEEN, (60, 90))])
            results = await db.search(query)
            assert len(results) == 2
            scores = [r.get_value("score") for r in results]
            assert all(60 <= s <= 90 for s in scores)
            
            # Test NOT_BETWEEN
            query = Query(filters=[Filter("score", Operator.NOT_BETWEEN, (60, 90))])
            results = await db.search(query)
            assert len(results) == 2
            scores = [r.get_value("score") for r in results]
            assert all(s < 60 or s > 90 for s in scores)
            
        finally:
            # Cleanup
            try:
                if db._pool:
                    await db._pool.execute(f"DROP TABLE IF EXISTS {db.schema_name}.{db.table_name}")
            except:
                pass
            finally:
                await db.close()


@pytest.mark.skipif(
    not os.getenv("ELASTICSEARCH_HOST"),
    reason="Elasticsearch not configured"
)
class TestElasticsearchRangeOperators:
    """Test BETWEEN operators with real Elasticsearch backend."""
    
    @pytest.fixture
    def es_db(self):
        """Create an Elasticsearch database connection."""
        from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
        
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost:9200")
        if ":" in es_host:
            host, port = es_host.split(":")
            port = int(port)
        else:
            host = es_host
            port = 9200
        
        db = SyncElasticsearchDatabase(config={
            "host": host,
            "port": port,
            "index": "test_range_index",
            "refresh": True  # Immediate refresh for testing
        })
        
        # Connect
        db.connect()
        
        # Clean up any existing index
        try:
            db.es_index.delete()
        except:
            pass
        
        # Create fresh index
        db.es_index.create()
        
        yield db
        
        # Cleanup
        try:
            db.es_index.delete()
        except:
            pass
        db.disconnect()
    
    def test_numeric_between_elasticsearch(self, es_db):
        """Test BETWEEN with numeric values in Elasticsearch."""
        # Insert test data
        records = [
            Record(id="1", data={"temperature": 15.5}),
            Record(id="2", data={"temperature": 22.0}),
            Record(id="3", data={"temperature": 28.5}),
            Record(id="4", data={"temperature": 35.0}),
        ]
        
        for record in records:
            es_db.create(record)
        
        # Test BETWEEN
        query = Query(filters=[Filter("temperature", Operator.BETWEEN, (20, 30))])
        results = es_db.search(query)
        assert len(results) == 2
        # The IDs returned may not be the ones we set - check the actual values
        result_temps = [r.get_value("temperature") for r in results]
        assert all(20 <= temp <= 30 for temp in result_temps)
        
        # Test NOT_BETWEEN
        query = Query(filters=[Filter("temperature", Operator.NOT_BETWEEN, (20, 30))])
        results = es_db.search(query)
        assert len(results) == 2
        result_temps = [r.get_value("temperature") for r in results]
        assert all(temp < 20 or temp > 30 for temp in result_temps)
    
    def test_datetime_between_elasticsearch(self, es_db):
        """Test BETWEEN with datetime values in Elasticsearch."""
        base_time = datetime(2025, 1, 15, 12, 0, 0)
        
        records = [
            Record(id="1", data={"timestamp": (base_time - timedelta(hours=3)).isoformat()}),
            Record(id="2", data={"timestamp": (base_time - timedelta(hours=1)).isoformat()}),
            Record(id="3", data={"timestamp": base_time.isoformat()}),
            Record(id="4", data={"timestamp": (base_time + timedelta(hours=2)).isoformat()}),
        ]
        
        for record in records:
            es_db.create(record)
        
        # Test BETWEEN with ISO strings
        start = (base_time - timedelta(hours=2)).isoformat()
        end = (base_time + timedelta(hours=1)).isoformat()
        query = Query(filters=[Filter("timestamp", Operator.BETWEEN, (start, end))])
        results = es_db.search(query)
        assert len(results) == 2
        # Check the actual timestamps are in range
        for r in results:
            ts = r.get_value("timestamp")
            assert start <= ts <= end
    
    def test_string_between_elasticsearch(self, es_db):
        """Test BETWEEN with string values in Elasticsearch."""
        records = [
            Record(id="1", data={"username": "alice"}),
            Record(id="2", data={"username": "bob"}),
            Record(id="3", data={"username": "charlie"}),
            Record(id="4", data={"username": "david"}),
        ]
        
        for record in records:
            es_db.create(record)
        
        # Test BETWEEN with strings (lexicographic)
        query = Query(filters=[Filter("username", Operator.BETWEEN, ("bob", "david"))])
        results = es_db.search(query)
        # Note: This should match bob, charlie, david
        assert len(results) == 3
        usernames = [r.get_value("username") for r in results]
        assert all("bob" <= u <= "david" for u in usernames)
    
    def test_nested_field_between_elasticsearch(self, es_db):
        """Test BETWEEN on nested fields in Elasticsearch."""
        records = [
            Record(
                id="1",
                data={"metrics": {"cpu": 25}},
                metadata={"priority": 1}
            ),
            Record(
                id="2",
                data={"metrics": {"cpu": 50}},
                metadata={"priority": 2}
            ),
            Record(
                id="3",
                data={"metrics": {"cpu": 75}},
                metadata={"priority": 3}
            ),
            Record(
                id="4",
                data={"metrics": {"cpu": 90}},
                metadata={"priority": 4}
            ),
        ]
        
        for record in records:
            es_db.create(record)
        
        # Test BETWEEN on nested field
        query = Query(filters=[Filter("metrics.cpu", Operator.BETWEEN, (40, 80))])
        results = es_db.search(query)
        assert len(results) == 2
        cpu_values = [r.get_value("metrics.cpu") for r in results]
        assert all(40 <= cpu <= 80 for cpu in cpu_values)
    
    @pytest.mark.asyncio
    async def test_async_elasticsearch_between(self):
        """Test BETWEEN with async Elasticsearch backend."""
        from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
        
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost:9200")
        if ":" in es_host:
            host, port = es_host.split(":")
            port = int(port)
        else:
            host = es_host
            port = 9200
        
        db = AsyncElasticsearchDatabase(config={
            "host": host,
            "port": port,
            "index": "test_async_range_index",
            "refresh": True
        })
        
        await db.connect()
        
        try:
            # Clean and recreate index
            try:
                await db._client.indices.delete(index=db.index_name)
            except:
                pass
            
            await db._client.indices.create(index=db.index_name)
            
            # Insert test data
            records = [
                Record(id="1", data={"value": 10}),
                Record(id="2", data={"value": 25}),
                Record(id="3", data={"value": 40}),
                Record(id="4", data={"value": 55}),
            ]
            
            for record in records:
                await db.create(record)
            
            # Test BETWEEN
            query = Query(filters=[Filter("value", Operator.BETWEEN, (20, 50))])
            results = await db.search(query)
            assert len(results) == 2
            values = [r.get_value("value") for r in results]
            assert all(20 <= v <= 50 for v in values)
            
            # Test NOT_BETWEEN
            query = Query(filters=[Filter("value", Operator.NOT_BETWEEN, (20, 50))])
            results = await db.search(query)
            assert len(results) == 2
            values = [r.get_value("value") for r in results]
            assert all(v < 20 or v > 50 for v in values)
            
        finally:
            # Cleanup
            try:
                await db._client.indices.delete(index=db.index_name)
            except:
                pass
            await db.disconnect()


class TestCrossBackendConsistency:
    """Test that BETWEEN behaves consistently across all backends."""
    
    @pytest.fixture
    def test_data(self):
        """Common test data for all backends."""
        base_time = datetime(2025, 1, 15, 12, 0, 0)
        return [
            Record(
                id="1",
                data={
                    "price": 50.0,
                    "name": "Alpha",
                    "timestamp": (base_time - timedelta(days=2)).isoformat()
                }
            ),
            Record(
                id="2",
                data={
                    "price": 150.0,
                    "name": "Beta",
                    "timestamp": base_time.isoformat()
                }
            ),
            Record(
                id="3",
                data={
                    "price": 250.0,
                    "name": "Gamma",
                    "timestamp": (base_time + timedelta(days=2)).isoformat()
                }
            ),
            Record(
                id="4",
                data={
                    "price": 350.0,
                    "name": "Delta",
                    "timestamp": (base_time + timedelta(days=4)).isoformat()
                }
            ),
        ]
    
    def test_memory_backend_baseline(self, test_data):
        """Establish baseline behavior with memory backend."""
        from dataknobs_data.backends.memory import SyncMemoryDatabase
        
        db = SyncMemoryDatabase()
        for record in test_data:
            db.create(record)
        
        # Test numeric BETWEEN
        query = Query(filters=[Filter("price", Operator.BETWEEN, (100, 300))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
        
        # Test string BETWEEN (Note: "Delta" is between "Beta" and "Gamma" alphabetically)
        # So we use "Alpha" to "Beta" to get just 2 results
        query = Query(filters=[Filter("name", Operator.BETWEEN, ("Alpha", "Beta"))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"1", "2"}
        
        # Test datetime BETWEEN
        base_time = datetime(2025, 1, 15, 12, 0, 0)
        start = (base_time - timedelta(days=1)).isoformat()
        end = (base_time + timedelta(days=3)).isoformat()
        query = Query(filters=[Filter("timestamp", Operator.BETWEEN, (start, end))])
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
    
    @pytest.mark.skipif(
        not os.getenv("POSTGRES_HOST"),
        reason="PostgreSQL not configured"
    )
    def test_postgres_consistency(self, test_data):
        """Verify PostgreSQL backend matches memory backend behavior."""
        from dataknobs_data.backends.postgres import SyncPostgresDatabase
        
        import uuid
        # Use public schema with unique table name to avoid conflicts
        table_name = f"test_consistency_{uuid.uuid4().hex[:8]}"
        
        db = SyncPostgresDatabase(config={
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "test_db"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "schema": "public",  # Use public schema which always exists
            "table": table_name
        })
        
        db.connect()
        
        try:
            for record in test_data:
                db.create(record)
            
            # Test numeric BETWEEN
            query = Query(filters=[Filter("price", Operator.BETWEEN, (100, 300))])
            results = db.search(query)
            assert len(results) == 2
            prices = [r.get_value("price") for r in results]
            assert all(100 <= p <= 300 for p in prices)
            
            # Test string BETWEEN (use same range as memory backend test)
            query = Query(filters=[Filter("name", Operator.BETWEEN, ("Alpha", "Beta"))])
            results = db.search(query)
            assert len(results) == 2
            names = [r.get_value("name") for r in results]
            assert all("Alpha" <= n <= "Beta" for n in names)
            
        finally:
            try:
                db.db.execute(f"DROP TABLE IF EXISTS {db.schema_name}.{db.table_name}")
            except:
                pass
            finally:
                db.close()
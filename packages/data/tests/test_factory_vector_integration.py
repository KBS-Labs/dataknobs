"""Test DatabaseFactory integration with vector-enabled backends."""

import os
import tempfile
import pytest
import numpy as np
from collections import OrderedDict

from dataknobs_data.factory import DatabaseFactory
from dataknobs_data.records import Record
from dataknobs_data.fields import VectorField


class TestFactoryVectorIntegration:
    """Test that factory correctly handles vector-enabled backends."""
    
    @pytest.fixture
    def factory(self):
        """Create a DatabaseFactory instance."""
        return DatabaseFactory()
    
    def test_sqlite_vector_enabled(self, factory):
        """Test creating SQLite database with vector support via factory."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
            db_path = f.name
            
            # Create database via factory with vector enabled
            db = factory.create(
                backend="sqlite",
                path=db_path,
                table="test_vectors",
                vector_enabled=True,
                vector_metric="cosine"
            )
            
            try:
                db.connect()
                
                # Verify vector support is enabled
                assert db.vector_enabled is True
                assert db.enable_vector_support() is True
                
                # Test vector operations
                vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
                vector_field = VectorField(
                    name="embedding",
                    value=vec,
                    dimensions=3
                )
                
                record = Record(
                    data=OrderedDict({"embedding": vector_field}),
                    metadata={"test": "factory"}
                )
                
                # Create record with vector
                record_id = db.create(record)
                assert record_id is not None
                
                # Read it back
                retrieved = db.read(record_id)
                assert retrieved is not None
                assert "embedding" in retrieved.fields
                
                # Test vector search
                results = db.vector_search(
                    query_vector=vec,
                    field_name="embedding",
                    k=1
                )
                assert len(results) == 1
                assert results[0].record.id == record_id
                
            finally:
                db.close()
    
    def test_vector_enabled_all_backends(self, factory):
        """Test that vector_enabled works for all backends now."""
        # Memory backend should work with vectors
        db = factory.create(
            backend="memory",
            vector_enabled=True
        )
        assert db is not None
        db.connect()
        
        # File backend should work with vectors (using temp file)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            db2 = factory.create(
                backend="file",
                path=tmp.name,
                vector_enabled=True
            )
            assert db2 is not None
            db2.connect()
            db2.close()
            import os
            os.unlink(tmp.name)
    
    def test_sqlite_without_vector_enabled(self, factory):
        """Test SQLite works normally without vector_enabled."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
            db = factory.create(
                backend="sqlite",
                path=f.name,
                table="test_normal"
            )
            
            try:
                db.connect()
                
                # Should work as normal database
                from dataknobs_data.fields import Field
                record = Record(
                    data=OrderedDict({"text": Field(name="text", value="test")}),
                    metadata={"type": "normal"}
                )
                
                record_id = db.create(record)
                assert record_id is not None
                
                retrieved = db.read(record_id)
                assert retrieved is not None
                assert retrieved.fields["text"].value == "test"
                
            finally:
                db.close()
    
    def test_backend_info_shows_vector_support(self, factory):
        """Test that backend info correctly shows vector support."""
        # Check PostgreSQL info
        pg_info = factory.get_backend_info("postgres")
        assert pg_info["vector_support"] is True
        assert "pgvector" in pg_info["description"]
        assert "vector_enabled" in pg_info["config_options"]
        
        # Check Elasticsearch info
        es_info = factory.get_backend_info("elasticsearch")
        assert es_info["vector_support"] is True
        assert "KNN" in es_info["description"]
        assert "vector_enabled" in es_info["config_options"]
        
        # Check SQLite info
        sqlite_info = factory.get_backend_info("sqlite")
        assert sqlite_info["vector_support"] is True
        assert "Python-based" in sqlite_info["description"]
        assert "vector_enabled" in sqlite_info["config_options"]
        
        # Check that memory doesn't have vector support
        memory_info = factory.get_backend_info("memory")
        assert "vector_support" not in memory_info
    
    @pytest.mark.skipif(
        not os.environ.get("TEST_POSTGRES", "").lower() == "true",
        reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance"
    )
    def test_postgres_vector_enabled(self, factory):
        """Test PostgreSQL with vector support (requires actual database)."""
        import uuid
        # Use a unique table name to avoid conflicts
        table_name = f"test_factory_vectors_{uuid.uuid4().hex[:8]}"
        
        # Get PostgreSQL configuration from environment
        db = factory.create(
            backend="postgres",
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            database=os.environ.get("POSTGRES_DB", "dataknobs_test"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            table=table_name,
            vector_enabled=True
        )
        
        try:
            db.connect()
            
            # Verify vector support is enabled
            assert db.vector_enabled is True
            assert db.has_vector_support() is True  # PostgreSQL with pgvector
            
            # Test basic vector operations
            vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            vector_field = VectorField(
                name="embedding",
                value=vec,
                dimensions=4
            )
            
            record = Record(
                data=OrderedDict({"embedding": vector_field}),
                metadata={"test": "postgres_factory"}
            )
            
            # Create record with vector
            record_id = db.create(record)
            assert record_id is not None
            
            # Test vector search
            results = db.vector_search(
                query_vector=vec,
                field_name="embedding",
                k=1
            )
            assert len(results) == 1
            assert results[0].record.id == record_id
            
            # Cleanup
            db.delete(record_id)
            
        finally:
            # Clean up the table
            try:
                import psycopg2
                # Detect if we're running in Docker container
                if os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
                    postgres_host = os.getenv('POSTGRES_HOST', 'postgres')
                else:
                    postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
                
                conn = psycopg2.connect(
                    host=postgres_host,
                    port=int(os.environ.get("POSTGRES_PORT", 5432)),
                    database=os.environ.get("POSTGRES_DB", "dataknobs_test"),
                    user=os.environ.get("POSTGRES_USER", "postgres"),
                    password=os.environ.get("POSTGRES_PASSWORD", "postgres")
                )
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS public.{table_name}")
                conn.commit()
                cursor.close()
                conn.close()
            except Exception:
                pass  # Ignore cleanup errors
            
            db.close()
    
    @pytest.mark.skipif(
        not os.environ.get("TEST_ELASTICSEARCH", "").lower() == "true",
        reason="Elasticsearch tests require TEST_ELASTICSEARCH=true and a running Elasticsearch instance"
    )
    def test_elasticsearch_vector_enabled(self, factory):
        """Test Elasticsearch with vector support (requires actual instance)."""
        import uuid
        # Use a unique index name to avoid conflicts
        index_name = f"test_factory_vectors_{uuid.uuid4().hex[:8]}"
        
        # Detect if we're running in Docker container
        if os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
            elasticsearch_host = os.getenv('ELASTICSEARCH_HOST', 'elasticsearch')
        else:
            elasticsearch_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
        
        # Get Elasticsearch configuration from environment
        port = int(os.environ.get("ELASTICSEARCH_PORT", "9200"))
        
        db = factory.create(
            backend="elasticsearch",
            host=elasticsearch_host,
            port=port,
            index=index_name,
            vector_enabled=True,
            vector_dimensions=4  # Specify dimensions for test vectors
        )
        
        try:
            db.connect()
            
            # Verify vector support is enabled
            assert db.vector_enabled is True
            
            # Test basic vector operations
            vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            vector_field = VectorField(
                name="embedding",
                value=vec,
                dimensions=4
            )
            
            record = Record(
                data=OrderedDict({"embedding": vector_field}),
                metadata={"test": "es_factory"}
            )
            
            # Create record with vector
            record_id = db.create(record)
            assert record_id is not None
            
            # Give Elasticsearch time to index
            import time
            time.sleep(1)
            
            # Test vector search
            results = db.vector_search(
                query_vector=vec,
                field_name="embedding",
                k=1
            )
            assert len(results) == 1
            assert results[0].record.id == record_id
            
            # Cleanup
            db.delete(record_id)
            
        finally:
            # Clean up the index
            try:
                from elasticsearch import Elasticsearch
                es = Elasticsearch(hosts)
                if es.indices.exists(index=index_name):
                    es.indices.delete(index=index_name)
            except Exception:
                pass  # Ignore cleanup errors
            
            db.close()
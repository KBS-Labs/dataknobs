"""Integration test configuration and fixtures."""

import os
import time
from typing import Generator

import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import requests

from dataknobs_utils.elasticsearch_utils import SimplifiedElasticsearchIndex
from dataknobs_utils.requests_utils import RequestHelper


def wait_for_postgres(host: str, port: int, user: str, password: str, max_retries: int = 30):
    """Wait for PostgreSQL to be ready."""
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database="postgres"
            )
            conn.close()
            return True
        except psycopg2.OperationalError:
            if i == max_retries - 1:
                raise
            time.sleep(1)
    return False


def wait_for_elasticsearch(host: str, port: int, max_retries: int = 30):
    """Wait for Elasticsearch to be ready."""
    import socket
    
    # First check if port is open
    for i in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            # Port is open, now check if Elasticsearch is responding
            helper = RequestHelper(host, port, timeout=5)
            
            try:
                response = helper.get("_cluster/health")
                if response.succeeded:
                    # Check cluster health status
                    if response.json and "status" in response.json:
                        status = response.json["status"]
                        # Yellow or green status is acceptable
                        if status in ["yellow", "green"]:
                            return True
                        else:
                            print(f"Elasticsearch cluster status is {status}, waiting...")
                    else:
                        return True
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error to Elasticsearch at {host}:{port}: {e}")
            except requests.exceptions.Timeout as e:
                print(f"Timeout connecting to Elasticsearch at {host}:{port}: {e}")
            except Exception as e:
                print(f"Unexpected error connecting to Elasticsearch at {host}:{port}: {type(e).__name__}: {e}")
        else:
            print(f"Port {port} on {host} is not open yet (attempt {i+1}/{max_retries})")
        
        if i == max_retries - 1:
            raise ConnectionError(
                f"Could not connect to Elasticsearch at {host}:{port} after {max_retries} attempts. "
                f"Please ensure Elasticsearch is running and accessible."
            )
        time.sleep(1)
    
    return False


@pytest.fixture(scope="session")
def postgres_connection_params():
    """PostgreSQL connection parameters for integration tests."""
    # Detect if we're running in Docker container
    if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
        default_host = 'postgres'
    else:
        default_host = 'localhost'
    
    return {
        "host": os.environ.get("POSTGRES_HOST", default_host),
        "port": int(os.environ.get("POSTGRES_PORT", 5432)),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
        "database": os.environ.get("POSTGRES_DB", "dataknobs_test"),
    }


@pytest.fixture(scope="session")
def elasticsearch_connection_params():
    """Elasticsearch connection parameters for integration tests."""
    # Detect if we're running in Docker container
    if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
        default_host = 'elasticsearch'
    else:
        default_host = 'localhost'
    
    return {
        "host": os.environ.get("ELASTICSEARCH_HOST", default_host),
        "port": int(os.environ.get("ELASTICSEARCH_PORT", 9200)),
    }


@pytest.fixture(scope="session")
def ensure_postgres_ready(postgres_connection_params):
    """Ensure PostgreSQL is ready before running tests."""
    wait_for_postgres(
        host=postgres_connection_params["host"],
        port=postgres_connection_params["port"],
        user=postgres_connection_params["user"],
        password=postgres_connection_params["password"],
    )
    
    # Create test database if it doesn't exist
    conn = psycopg2.connect(
        host=postgres_connection_params["host"],
        port=postgres_connection_params["port"],
        user=postgres_connection_params["user"],
        password=postgres_connection_params["password"],
        database="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    cursor = conn.cursor()
    try:
        # Check if database exists first
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{postgres_connection_params['database']}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {postgres_connection_params['database']}")
    except psycopg2.errors.DuplicateDatabase:
        pass
    finally:
        cursor.close()
        conn.close()


@pytest.fixture(scope="session")
def ensure_elasticsearch_ready(elasticsearch_connection_params):
    """Ensure Elasticsearch is ready before running tests."""
    wait_for_elasticsearch(
        host=elasticsearch_connection_params["host"],
        port=elasticsearch_connection_params["port"],
    )


@pytest.fixture
def postgres_test_db(ensure_postgres_ready, postgres_connection_params) -> Generator[dict, None, None]:
    """Provide a clean PostgreSQL database for each test."""
    import uuid
    
    # Generate unique table name for this test
    test_id = uuid.uuid4().hex[:8]
    config = postgres_connection_params.copy()
    config["table"] = f"test_records_{test_id}"
    config["schema"] = "public"
    
    yield config
    
    # Cleanup: Drop the test table
    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"]
    )
    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {config['schema']}.{config['table']} CASCADE")
        conn.commit()
    finally:
        cursor.close()
        conn.close()


@pytest.fixture
def elasticsearch_test_index(ensure_elasticsearch_ready, elasticsearch_connection_params) -> Generator[dict, None, None]:
    """Provide a clean Elasticsearch index for each test."""
    import uuid
    
    # Generate unique index name for this test
    test_id = uuid.uuid4().hex[:8]
    config = elasticsearch_connection_params.copy()
    config["index"] = f"test_records_{test_id}"
    config["refresh"] = True  # Immediate refresh for testing
    
    yield config
    
    # Cleanup: Delete the test index
    try:
        es_index = SimplifiedElasticsearchIndex(
            index_name=config["index"],
            host=config["host"],
            port=config["port"],
        )
        if es_index.exists():
            es_index.delete()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def sample_records():
    """Provide sample records for testing."""
    from dataknobs_data import Record
    
    return [
        Record({
            "name": "Alice Johnson",
            "age": 28,
            "email": "alice@example.com",
            "department": "Engineering",
            "salary": 95000.50,
            "active": True,
            "joined_date": "2021-03-15",
            "skills": ["Python", "PostgreSQL", "Docker"],
        }, metadata={"source": "test", "version": 1}),
        
        Record({
            "name": "Bob Smith",
            "age": 35,
            "email": "bob@example.com",
            "department": "Marketing",
            "salary": 82000.00,
            "active": True,
            "joined_date": "2019-07-22",
            "skills": ["SEO", "Content Marketing", "Analytics"],
        }, metadata={"source": "test", "version": 1}),
        
        Record({
            "name": "Charlie Brown",
            "age": 42,
            "email": "charlie@example.com",
            "department": "Engineering",
            "salary": 120000.75,
            "active": False,
            "joined_date": "2018-01-10",
            "skills": ["Java", "Kubernetes", "AWS"],
        }, metadata={"source": "test", "version": 2}),
        
        Record({
            "name": "Diana Prince",
            "age": 31,
            "email": "diana@example.com",
            "department": "HR",
            "salary": 78000.00,
            "active": True,
            "joined_date": "2020-09-05",
            "skills": ["Recruitment", "Training", "Compliance"],
        }, metadata={"source": "test", "version": 1}),
        
        Record({
            "name": "Eve Anderson",
            "age": 29,
            "email": "eve@example.com",
            "department": "Engineering",
            "salary": 105000.25,
            "active": True,
            "joined_date": "2022-02-28",
            "skills": ["React", "Node.js", "MongoDB"],
        }, metadata={"source": "test", "version": 1}),
    ]
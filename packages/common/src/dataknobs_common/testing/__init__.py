"""Test utilities for dataknobs packages.

This package provides pytest utilities for service availability checking,
test configuration factories, fixture helpers, and shared pytest11 fixture
plugins for Postgres and Elasticsearch integration tests.

The pytest11 entry points are registered in ``packages/common/pyproject.toml``
so that any package depending on ``dataknobs-common`` automatically gets the
shared fixtures via pytest's plugin discovery — no explicit imports needed
in consumer ``conftest.py`` files. Consumers wrap the factory fixtures
(``make_postgres_test_db`` / ``make_elasticsearch_test_index``) with their
own thin per-prefix fixtures.

Example:
    ```python
    import pytest
    from dataknobs_common.testing import (
        is_ollama_available,
        requires_ollama,
        get_test_bot_config,
        safe_sql_ident,
    )

    # Skip test if Ollama not available
    @pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
    def test_with_ollama():
        ...

    # Or use the marker
    @requires_ollama
    def test_with_ollama_marker():
        ...

    # Get test configuration
    config = get_test_bot_config(use_echo_llm=True)

    # Validate SQL identifiers built from env vars / hardcoded defaults /
    # uuid suffixes before f-string interpolation in test fixtures
    cursor.execute(f"DROP TABLE IF EXISTS {safe_sql_ident(table)}")
    ```
"""

from dataknobs_common.testing._core import (
    create_test_json_files,
    create_test_markdown_files,
    get_test_bot_config,
    get_test_rag_config,
    is_chromadb_available,
    is_faiss_available,
    is_localstack_available,
    is_ollama_available,
    is_ollama_model_available,
    is_package_available,
    is_postgres_available,
    is_redis_available,
    requires_chromadb,
    requires_faiss,
    requires_localstack,
    requires_ollama,
    requires_ollama_model,
    requires_package,
    requires_postgres,
    requires_real_postgres,
    requires_redis,
    safe_sql_ident,
)
from dataknobs_common.testing.elasticsearch_fixtures import (
    elasticsearch_connection_params,
    ensure_elasticsearch_ready,
    make_elasticsearch_test_index,
    wait_for_elasticsearch,
)
from dataknobs_common.testing.postgres_fixtures import (
    ensure_postgres_ready,
    make_postgres_test_db,
    postgres_connection_params,
    wait_for_postgres,
)

__all__ = [
    "create_test_json_files",
    "create_test_markdown_files",
    "elasticsearch_connection_params",
    "ensure_elasticsearch_ready",
    "ensure_postgres_ready",
    "get_test_bot_config",
    "get_test_rag_config",
    "is_chromadb_available",
    "is_faiss_available",
    "is_localstack_available",
    "is_ollama_available",
    "is_ollama_model_available",
    "is_package_available",
    "is_postgres_available",
    "is_redis_available",
    "make_elasticsearch_test_index",
    "make_postgres_test_db",
    "postgres_connection_params",
    "requires_chromadb",
    "requires_faiss",
    "requires_localstack",
    "requires_ollama",
    "requires_ollama_model",
    "requires_package",
    "requires_postgres",
    "requires_real_postgres",
    "requires_redis",
    "safe_sql_ident",
    "wait_for_elasticsearch",
    "wait_for_postgres",
]

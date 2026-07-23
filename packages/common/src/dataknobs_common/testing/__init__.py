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
    ensure_localstack_s3_bucket,
    get_localstack_endpoint,
    get_test_bot_config,
    get_test_rag_config,
    is_bedrock_available,
    is_chromadb_available,
    is_elasticsearch_available,
    is_faiss_available,
    is_localstack_available,
    is_ollama_available,
    is_ollama_model_available,
    is_package_available,
    is_postgres_available,
    is_redis_available,
    requires_bedrock,
    requires_chromadb,
    requires_elasticsearch,
    requires_faiss,
    requires_localstack,
    requires_localstack_service,
    requires_ollama,
    requires_ollama_model,
    requires_package,
    requires_postgres,
    requires_real_postgres,
    requires_redis,
    safe_sql_ident,
)
from dataknobs_common.testing.blocking import (
    assert_no_blocking,
    blocking_error_type,
    is_blockbuster_available,
    no_blocking,
    requires_blockbuster,
)
from dataknobs_common.testing.elasticsearch_fixtures import (
    elasticsearch_connection_params,
    ensure_elasticsearch_ready,
    make_elasticsearch_test_index,
    sweep_stale_test_indices,
    wait_for_elasticsearch,
)
from dataknobs_common.testing.factory_parity import (
    assert_config_attribute_access_matches_dataclass,
    assert_ctor_reads_documented_keys,
    assert_dataclass_config_matches_ctor,
    assert_factory_kwargs_match_ctor,
    assert_polymorphic_bindings_resolve,
    assert_structured_config_consumer,
    assert_structured_config_roundtrip,
)
from dataknobs_common.testing.localstack_fixtures import (
    localstack_endpoint,
    make_localstack_s3_bucket,
)
from dataknobs_common.testing.postgres_fixtures import (
    ensure_postgres_ready,
    make_postgres_test_db,
    postgres_connection_params,
    wait_for_postgres,
)

__all__ = [
    "assert_config_attribute_access_matches_dataclass",
    "assert_ctor_reads_documented_keys",
    "assert_dataclass_config_matches_ctor",
    "assert_factory_kwargs_match_ctor",
    "assert_no_blocking",
    "assert_polymorphic_bindings_resolve",
    "assert_structured_config_consumer",
    "assert_structured_config_roundtrip",
    "blocking_error_type",
    "create_test_json_files",
    "create_test_markdown_files",
    "elasticsearch_connection_params",
    "ensure_elasticsearch_ready",
    "ensure_localstack_s3_bucket",
    "ensure_postgres_ready",
    "get_localstack_endpoint",
    "get_test_bot_config",
    "get_test_rag_config",
    "is_bedrock_available",
    "is_blockbuster_available",
    "is_chromadb_available",
    "is_elasticsearch_available",
    "is_faiss_available",
    "is_localstack_available",
    "is_ollama_available",
    "is_ollama_model_available",
    "is_package_available",
    "is_postgres_available",
    "is_redis_available",
    "localstack_endpoint",
    "make_elasticsearch_test_index",
    "make_localstack_s3_bucket",
    "make_postgres_test_db",
    "no_blocking",
    "postgres_connection_params",
    "requires_bedrock",
    "requires_blockbuster",
    "requires_chromadb",
    "requires_elasticsearch",
    "requires_faiss",
    "requires_localstack",
    "requires_localstack_service",
    "requires_ollama",
    "requires_ollama_model",
    "requires_package",
    "requires_postgres",
    "requires_real_postgres",
    "requires_redis",
    "safe_sql_ident",
    "sweep_stale_test_indices",
    "wait_for_elasticsearch",
    "wait_for_postgres",
]

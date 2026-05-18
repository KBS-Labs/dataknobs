"""Integration test configuration and fixtures.

Postgres and Elasticsearch infrastructure fixtures
(``postgres_connection_params``, ``ensure_postgres_ready``,
``wait_for_postgres``, ``elasticsearch_connection_params``,
``ensure_elasticsearch_ready``, ``wait_for_elasticsearch``) come from the
``dataknobs_common.testing`` pytest11 plugin — no duplication here. The
two thin wrappers below use the ``dataknobs-data`` package's
``test_records_`` table/index prefix.
"""

from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture
def isolate_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub ambient AWS env so ``moto.mock_aws()`` is truly hermetic.

    botocore 1.34+ honors ``AWS_ENDPOINT_URL`` / ``AWS_ENDPOINT_URL_S3`` /
    ``LOCALSTACK_ENDPOINT`` as global default endpoints **even inside
    ``mock_aws()``**. ``bin/test.sh`` exports these for the real-LocalStack
    integration tests; without scrubbing them, a moto-based fixture
    silently routes to the shared persistent LocalStack container and
    picks up cross-test / cross-run state (e.g. ``test_list_all`` seeing
    leftover objects from an interrupted prior run -> ``assert 25 == 5``).

    Single source of truth: every moto fixture and every
    region-resolution test class depends on this instead of duplicating
    the scrub list.
    """
    for key in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "LOCALSTACK_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.fixture
def postgres_test_db(make_postgres_test_db) -> Generator[dict[str, Any], None, None]:
    """Provide a clean PostgreSQL table per test, using the ``test_records_`` prefix."""
    yield from make_postgres_test_db("test_records_")


@pytest.fixture
def elasticsearch_test_index(
    make_elasticsearch_test_index,
) -> Generator[dict[str, Any], None, None]:
    """Provide a clean Elasticsearch index per test, using the ``test_records_`` prefix."""
    yield from make_elasticsearch_test_index("test_records_")


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

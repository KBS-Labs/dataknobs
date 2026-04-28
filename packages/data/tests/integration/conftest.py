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

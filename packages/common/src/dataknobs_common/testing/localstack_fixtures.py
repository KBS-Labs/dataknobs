# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Shared LocalStack pytest fixtures for dataknobs integration tests.

This module is a pytest11 plugin (registered in
``packages/common/pyproject.toml``) so any package depending on
``dataknobs-common`` automatically gets these fixtures via pytest's
plugin discovery — no explicit ``conftest.py`` imports required.

The bucket-creation helper is async (``aioboto3``-backed) but the
factory fixture wraps it in :func:`asyncio.run` so both sync and async
tests can consume it without ceremony. Fixture bodies run during
pytest setup, outside any per-test event loop, so ``asyncio.run`` is
safe here.

Consumers wrap :func:`make_localstack_s3_bucket` with a thin
per-bucket fixture:

    @pytest.fixture
    def s3_test_bucket(make_localstack_s3_bucket):
        yield from make_localstack_s3_bucket("test-bucket")

Unlike the Postgres / Elasticsearch fixtures, the bucket is **not**
torn down on test exit. LocalStack persists the bucket across the
session and tests are expected to wipe their own object contents (via
``db.clear()`` or equivalent). This matches the LocalStack
data-volume model and keeps reruns fast.

The factory is session-scoped and ensures each bucket once per session
(per xdist worker): a later test asking for the same bucket makes no
request. Ensuring costs a new event loop, a new ``aioboto3`` session and
a ``HEAD`` request, about 40 ms against a warm LocalStack, which every S3
test used to pay. Isolate tests with a per-test key prefix inside the
bucket. A test that deletes a bucket should use a name no other test
asks for, since the session will not recreate it.

Environment variables (resolved via
:func:`~dataknobs_common.testing.get_localstack_endpoint`):

- ``LOCALSTACK_ENDPOINT`` (full URL)
- ``AWS_ENDPOINT_URL`` (full URL, fallback)
- ``LOCALSTACK_HOST`` / ``LOCALSTACK_PORT``
- ``DOCKER_CONTAINER`` / ``/.dockerenv`` (Docker-aware default host)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator
from typing import Any

from dataknobs_common.testing._core import (
    ensure_localstack_s3_bucket,
    get_localstack_endpoint,
)

logger = logging.getLogger(__name__)


try:
    import pytest

    @pytest.fixture(scope="session")
    def localstack_endpoint() -> str:
        """The resolved LocalStack edge endpoint URL for this session.

        Delegates to :func:`get_localstack_endpoint`; see that
        function's docstring for the resolution chain. Cached at
        session scope so all consumers share one URL.
        """
        return get_localstack_endpoint()

    @pytest.fixture(scope="session")
    def make_localstack_s3_bucket(
        localstack_endpoint: str,
    ) -> Callable[[str], Iterator[dict[str, Any]]]:
        """Factory fixture: idempotently ensure an S3 bucket on LocalStack.

        Returns a callable ``factory(bucket)`` that yields a config
        dict shaped for the dataknobs S3 backends (sync and async):

            @pytest.fixture
            def s3_test_bucket(make_localstack_s3_bucket):
                yield from make_localstack_s3_bucket("test-bucket")

        The yielded config dict contains:

        - ``bucket``: the bucket name (as passed)
        - ``endpoint_url``: the LocalStack edge URL
        - ``region``: ``"us-east-1"`` (LocalStack default)
        - ``aws_access_key_id`` / ``aws_secret_access_key``: ``"test"``
          / ``"test"`` (LocalStack accepts any credentials). Canonical
          boto form — ``AwsSessionConfig.from_dict`` also accepts the
          legacy short aliases ``access_key_id`` / ``secret_access_key``.

        No teardown — the bucket persists for the LocalStack session.
        Tests should wipe their own object contents on teardown (e.g.
        ``await db.clear()``).

        Each bucket is ensured once per session: the factory remembers
        the buckets it has made and does not ask LocalStack again. A
        bucket whose ensure raised is not remembered, so the next ask
        tries again.

        Args:
            localstack_endpoint: Session-scoped endpoint URL.

        Returns:
            A callable that, given a ``bucket`` name, yields a config
            dict suitable for spread into a dataknobs S3 backend
            constructor.
        """
        ensured: set[str] = set()

        def factory(bucket: str) -> Iterator[dict[str, Any]]:
            if bucket not in ensured:
                asyncio.run(ensure_localstack_s3_bucket(bucket, endpoint=localstack_endpoint))
                ensured.add(bucket)
            yield {
                "bucket": bucket,
                "endpoint_url": localstack_endpoint,
                "region": "us-east-1",
                "aws_access_key_id": "test",
                "aws_secret_access_key": "test",
            }

        return factory

except ImportError:
    # pytest not installed — fixture decorators unavailable. The
    # ensure_localstack_s3_bucket / get_localstack_endpoint helpers
    # imported above remain usable from non-pytest contexts.
    localstack_endpoint = None  # type: ignore[assignment]
    make_localstack_s3_bucket = None  # type: ignore[assignment]

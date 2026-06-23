"""Integration test for shared Elasticsearch-client ownership (real ES).

``ConnectionPoolManager`` shares one pooled client per pool key on an
event loop. ``ElasticsearchPoolConfig.to_hash_key()`` is
``(tuple(hosts), index)`` — so two ``AsyncElasticsearchDatabase``
instances on the **same hosts and the same index** share one client.
(This differs from Postgres, whose key is table-independent
``(host, port, database, user)``; for ES the index participates in the
key. The refcount/``release_pool`` contract is correct regardless of
what the key includes — it simply scopes sharing to the key.)

173-B-es is a real behavior change: a holder's ``close()`` now
*releases* its claim via ``release_pool``, and the registered close-func
runs only when the last holder releases. Before the fix the client was
dropped locally but never released, so it was leaked until
``close_all()``/``atexit``. Because Elasticsearch wraps an optional
dependency, the project mandates a behavioral test with the real library
— this proves the ``release_pool`` wiring (``_pool_config`` -> client
manager) is correct end-to-end, which the backend-agnostic unit tests
cannot see.
"""

import os

import pytest

from dataknobs_data import AsyncDatabase, Record
from dataknobs_data.backends.elasticsearch_async import _client_manager

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_ELASTICSEARCH", "").lower() == "true",
    reason="Elasticsearch tests require TEST_ELASTICSEARCH=true and a running Elasticsearch instance",
)


class TestElasticsearchPoolSharing:
    """Shared-client ownership across sibling AsyncElasticsearchDatabase holders."""

    @pytest.mark.asyncio
    async def test_es_sibling_close_releases_not_destroys(self, elasticsearch_test_index):
        """A sibling close releases its claim; the last release closes the client.

        Both holders target the same hosts + index, so they share one
        pooled client. Counts are asserted relative to a baseline because
        ``_client_manager`` is a module-global shared across tests.
        """
        # Same config for both holders -> same (hosts, index) pool key.
        config = dict(elasticsearch_test_index)

        baseline = _client_manager.get_pool_count()

        a = await AsyncDatabase.from_backend("elasticsearch", config)
        b = await AsyncDatabase.from_backend("elasticsearch", config)
        try:
            # Same DSN + index -> one shared client, two holders.
            assert a._client is b._client
            assert _client_manager.get_pool_count() == baseline + 1

            rec_id = await a.create(Record({"v": "before"}))
            await a._client.indices.refresh(index=config["index"])
            assert await a.read(rec_id) is not None

            # Sibling release: the shared client must NOT be closed.
            await b.close()
            assert _client_manager.get_pool_count() == baseline + 1

            # a still issues a live request against the shared client.
            assert await a.read(rec_id) is not None
            await a.delete(rec_id)

            # Last holder release: client closed, entry evicted.
            await a.close()
            assert _client_manager.get_pool_count() == baseline
        finally:
            # Idempotent — both already released on the happy path.
            await a.close()
            await b.close()

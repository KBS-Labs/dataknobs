"""Cross-construct parity tests for the shared S3 session factory.

Verifies that one config dict — using either ``region`` or
``region_name`` — produces equivalent region resolution for both
``SyncS3Database`` (sync boto3 path) and ``AsyncS3Database`` (aioboto3
path via ``S3PoolConfig``). This is the contract the shared factory
exists to enforce.
"""

from __future__ import annotations

import pytest
from moto import mock_aws

from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database
from dataknobs_data.pooling.s3 import S3PoolConfig


@pytest.fixture(autouse=True)
def _isolate_aws_env(isolate_aws_env: None) -> None:
    """Apply the shared ambient-AWS-env scrub (conftest) to this module.

    Cross-construct parity assertions require a hermetic moto
    environment; the scrub logic lives once in the ``isolate_aws_env``
    conftest fixture (see its docstring for the botocore-1.34
    rationale).
    """
    # Intentionally no body: this fixture exists only to make pytest
    # request ``isolate_aws_env`` (autouse) for every test in this
    # module. Do not "tidy up" by deleting it — the scrub would stop
    # being applied.


@pytest.mark.parametrize(
    "region_key", ["region", "region_name"], ids=["legacy", "boto-native"]
)
def test_sync_and_async_accept_same_region_key(region_key: str) -> None:
    """Same config dict — using either region key — wires both constructs.

    The factory normalizes both keys to ``region_name`` internally, so
    a config authored for one path works against the other without
    rename. This is the cross-construct parity guarantee.
    """
    cfg = {"bucket": "parity-bucket", region_key: "eu-west-1"}

    with mock_aws():
        sync_db = SyncS3Database(cfg)
        sync_db.connect()
        try:
            assert sync_db.s3_client.meta.region_name == "eu-west-1"
        finally:
            sync_db.close()

    # Async path uses S3PoolConfig — its from_dict must accept the
    # same key shapes. Verify the projection without spinning up a
    # full aioboto3 session (the kwarg-shaping is exercised in the
    # unit tests).
    pool_cfg = S3PoolConfig.from_dict(cfg)
    assert pool_cfg.region_name == "eu-west-1"
    sess_cfg = pool_cfg.to_session_config()
    assert sess_cfg.region_name == "eu-west-1"

    # AsyncS3Database accepts the same dict and sets up its pool config.
    async_db = AsyncS3Database(cfg)
    assert async_db.region == "eu-west-1"


def test_sync_and_async_share_default_chain_when_region_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No region in config → both constructs defer to boto's chain."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    cfg = {"bucket": "default-chain-bucket"}

    with mock_aws():
        sync_db = SyncS3Database(cfg)
        sync_db.connect()
        try:
            assert sync_db.s3_client.meta.region_name == "ap-southeast-2"
        finally:
            sync_db.close()

    # Async pool config preserves None → projects to None on session
    # config → the eventual aioboto3 session inherits boto's chain.
    pool_cfg = S3PoolConfig.from_dict(cfg)
    assert pool_cfg.region_name is None
    assert pool_cfg.to_session_config().region_name is None

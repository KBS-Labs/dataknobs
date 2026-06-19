"""Region-handling tests for S3KnowledgeBackend.

The discovery defect: ``S3KnowledgeBackend`` hardcoded its region to
``us-east-1``, silently overriding ``AWS_DEFAULT_REGION`` env / IAM-role
metadata for any consumer deployed elsewhere. The fix routes client
construction through the shared :class:`S3SessionConfig` factory so the
region defaults to ``None`` and boto's resolution chain takes over.

These assertions target the **normalized session config and the client
kwargs it produces** — the values that flow into the aioboto3 session
(:func:`create_aioboto3_session`) and into every per-operation
``session.client("s3", **client_kwargs)`` call. After the move to
aioboto3 there is no persistent ``_client`` to inspect; the region
contract lives entirely in :class:`S3SessionConfig`, so that is where
the regression guard belongs. No moto / LocalStack needed — region
*resolution* (env var → client) is botocore's job and is covered by the
LocalStack integration suite; these tests pin only that the backend
does not re-hardcode a region.

The reproduce-first guard is
:func:`test_no_region_defers_to_boto_chain`: against the unfixed
backend ``region_name`` was forced to ``"us-east-1"``; post-fix it is
``None`` so boto's chain resolves it.
"""

from __future__ import annotations

from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_data.pooling.s3 import S3SessionConfig


def test_no_region_defers_to_boto_chain() -> None:
    """Reproducer: omit region kwarg → ``region_name`` is None, not hardcoded.

    Pre-fix the backend forced ``region="us-east-1"`` so the resolved
    region ignored the environment / IAM metadata. Post-fix the session
    config carries ``region_name=None``, and the client kwargs omit
    ``region_name`` entirely, leaving boto's resolution chain
    (``AWS_DEFAULT_REGION`` env, ``~/.aws/config``, instance metadata,
    then the ``us-east-1`` terminal fallback) to decide at client time.
    """
    backend = S3KnowledgeBackend(bucket="reproducer-bucket")
    assert backend._session_config.region_name is None
    assert "region_name" not in backend._session_config.to_client_kwargs()


def test_region_kwarg_threads_through() -> None:
    """Explicit region= kwarg pins the region on the session + client kwargs."""
    backend = S3KnowledgeBackend(
        bucket="explicit-bucket", region="eu-west-1"
    )
    assert backend._session_config.region_name == "eu-west-1"
    assert backend._session_config.to_client_kwargs()["region_name"] == (
        "eu-west-1"
    )


def test_region_name_via_from_config() -> None:
    """``region_name`` (boto-native key) accepted in from_config dict."""
    backend = S3KnowledgeBackend.from_config(
        {"bucket": "from-config-bucket", "region_name": "eu-west-1"}
    )
    assert backend._session_config.region_name == "eu-west-1"


def test_region_via_from_config() -> None:
    """``region`` (legacy key) accepted in from_config dict."""
    backend = S3KnowledgeBackend.from_config(
        {"bucket": "legacy-key-bucket", "region": "eu-west-1"}
    )
    assert backend._session_config.region_name == "eu-west-1"


def test_session_config_injection_wins_over_kwargs() -> None:
    """A pre-built session_config beats individual region/credential kwargs."""
    session_config = S3SessionConfig(region_name="eu-west-1")
    backend = S3KnowledgeBackend(
        bucket="injected-bucket",
        region="us-east-2",  # ignored: session_config takes precedence
        session_config=session_config,
    )
    assert backend._session_config.region_name == "eu-west-1"


def test_aws_session_token_kwarg_threads_through_to_session_config() -> None:
    """``aws_session_token`` kwarg must flow into the underlying session config.

    Prior to exposing it on ``__init__``, callers with temporary
    credentials had to drop down to ``session_config=`` to pass a
    session token — an interface gap relative to the other credential
    kwargs (``aws_access_key_id``, ``aws_secret_access_key``).
    """
    backend = S3KnowledgeBackend(
        bucket="token-bucket",
        aws_access_key_id="AK",
        aws_secret_access_key="SK",
        aws_session_token="ST",
    )
    assert backend._session_config.aws_access_key_id == "AK"
    assert backend._session_config.aws_secret_access_key == "SK"
    assert backend._session_config.aws_session_token == "ST"

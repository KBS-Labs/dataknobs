"""make_localstack_s3_bucket ensures a bucket once per session, not once per test.

Ensuring costs a new event loop, a new aioboto3 session and a HEAD request:
about 40 ms against a warm LocalStack, paid before every S3 test that asks for
a bucket the session has already made. The tests here run an inner pytest
session against the real LocalStack, because the property is about what one
test leaves for the next.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import boto3
import pytest
from botocore.exceptions import ClientError

from dataknobs_common.testing import requires_localstack

pytest_plugins = ("pytester",)

# The inner session has no pyproject of its own, so it states the asyncio
# settings the workspace's sessions run under.
_INNER_INI = """
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
"""


@pytest.fixture
def bucket(localstack_endpoint: str) -> Iterator[str]:
    """A bucket name no other test uses, removed afterwards if it exists."""
    name = f"dk-ensured-once-{uuid.uuid4().hex[:10]}"
    yield name
    client = boto3.client(
        "s3",
        endpoint_url=localstack_endpoint,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    try:
        client.delete_bucket(Bucket=name)
    except ClientError:
        pass  # already gone, which is what the passing case leaves


@requires_localstack
def test_a_later_test_asking_for_the_same_bucket_makes_no_request(
    pytester: pytest.Pytester, bucket: str
) -> None:
    """The second ask does not reach LocalStack, so it cannot recreate the bucket.

    The second test deletes the bucket behind the fixture's back and asks again.
    If the factory re-ensured it, the bucket would exist once more.
    """
    pytester.makeini(_INNER_INI)
    pytester.makepyfile(
        f"""
        import boto3

        BUCKET = {bucket!r}


        def _client(endpoint):
            return boto3.client(
                "s3", endpoint_url=endpoint, region_name="us-east-1",
                aws_access_key_id="test", aws_secret_access_key="test",
            )


        def _exists(endpoint):
            names = [b["Name"] for b in _client(endpoint).list_buckets()["Buckets"]]
            return BUCKET in names


        def test_1_first_ask_creates(make_localstack_s3_bucket, localstack_endpoint):
            for config in make_localstack_s3_bucket(BUCKET):
                assert config["bucket"] == BUCKET
            assert _exists(localstack_endpoint)


        def test_2_second_ask_is_remembered(make_localstack_s3_bucket, localstack_endpoint):
            _client(localstack_endpoint).delete_bucket(Bucket=BUCKET)
            for config in make_localstack_s3_bucket(BUCKET):
                assert config["bucket"] == BUCKET
            assert not _exists(localstack_endpoint)
        """
    )
    result = pytester.runpytest_inprocess("-p", "no:randomly", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=2)


@requires_localstack
def test_a_bucket_whose_ensure_failed_is_ensured_again(
    pytester: pytest.Pytester, localstack_endpoint: str
) -> None:
    """Only a bucket that was made is remembered.

    An invalid name fails to ensure; asking again must try again rather than
    hand back a config for a bucket that was never created.
    """
    pytester.makeini(_INNER_INI)
    pytester.makepyfile(
        """
        import pytest
        from botocore.exceptions import ClientError

        BAD = "Not_A_Valid_Bucket_Name"


        def test_both_asks_fail(make_localstack_s3_bucket):
            for _ in range(2):
                with pytest.raises((ClientError, ValueError)):
                    for _config in make_localstack_s3_bucket(BAD):
                        pass
        """
    )
    result = pytester.runpytest_inprocess("-p", "no:randomly", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)

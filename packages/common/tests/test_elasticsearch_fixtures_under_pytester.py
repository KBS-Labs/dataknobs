"""The Elasticsearch fixtures keep working after a pytester test in the same worker.

The ``pytester`` fixture snapshots ``sys.modules`` when it is set up and restores
the snapshot at teardown, so a module first imported during such a test is
dropped afterwards. ``requests`` does not survive that: pytest has already
imported the ``http`` package, so ``http`` stays while ``http.cookiejar`` goes,
and the re-imported ``requests`` takes the stale ``http.cookiejar`` still bound
on ``http`` while a fresh ``import http.cookiejar`` makes a new one. The two
``CookieJar`` classes then disagree, and the next request the fixtures make
fails with "You can only merge into CookieJar".

The plugin's helpers import ``requests`` lazily, so a pytester test that is the
first in its worker to call one is exactly the test that drops it. The tests
here run an inner session in a fresh interpreter, where nothing has imported
``requests`` yet, which is the condition the failure needs.
"""

from __future__ import annotations

import pytest

from dataknobs_common.testing import requires_elasticsearch

pytest_plugins = ("pytester",)

_INNER_INI = """
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
"""


def _run(pytester: pytest.Pytester, source: str) -> pytest.RunResult:
    pytester.makeini(_INNER_INI)
    pytester.makepyfile(test_inner=source)
    return pytester.runpytest_subprocess("-p", "no:randomly", "-p", "no:cacheprovider")


def test_requests_is_loaded_before_the_first_test(pytester: pytest.Pytester) -> None:
    """Loaded before any test, ``requests`` is in every pytester snapshot."""
    result = _run(
        pytester,
        """
import sys


def test_first():
    assert "requests" in sys.modules
""",
    )
    result.assert_outcomes(passed=1)


@requires_elasticsearch
def test_a_request_after_a_pytester_test_succeeds(pytester: pytest.Pytester) -> None:
    """The first request in a worker made inside a pytester test leaves the next one working."""
    result = _run(
        pytester,
        """
from dataknobs_common.testing import wait_for_elasticsearch

pytest_plugins = ("pytester",)


def test_first_request_inside_pytester(pytester, elasticsearch_connection_params):
    wait_for_elasticsearch(**elasticsearch_connection_params, max_retries=1)


def test_next_request(elasticsearch_connection_params):
    wait_for_elasticsearch(**elasticsearch_connection_params, max_retries=1)
""",
    )
    result.assert_outcomes(passed=2)

"""Shared fixtures for ``dataknobs-common`` tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import pytest

from dataknobs_common.testing import live_dk_daemon_threads


@pytest.fixture
def new_dk_daemon_threads() -> Iterator[Callable[..., list[str]]]:
    """Report dataknobs daemon threads *this test* created and left alive.

    Thread assertions in this package used to compare against an absolute
    zero, which quietly made them a report on the whole process: a thread
    leaked by any other test in a multi-package run turned them red and
    named the wrong file as the culprit. Measuring against a per-test
    baseline scopes each assertion to the test that owns it.

    The baseline is captured over *every* watched name, so the returned
    callable can narrow to one name per call without needing a matching
    baseline per name::

        def test_something(new_dk_daemon_threads):
            bridge = SyncLoopBridge()
            assert new_dk_daemon_threads(DK_SYNC_BRIDGE_THREAD)
            bridge.close()
            assert new_dk_daemon_threads(DK_SYNC_BRIDGE_THREAD) == []

    Lives here rather than in each test module because the same eight-line
    idiom had been copied into three of them — which is the duplication
    ``dataknobs_common.testing.threads`` was extracted to end. Prefer
    ``assert_no_leaked_bridge_threads`` when a whole block should leak
    nothing; reach for this only when a test needs to assert *mid-run* that
    a thread does or does not exist.
    """
    baseline = set(live_dk_daemon_threads())

    def _still_alive(names: Iterable[str] | str | None = None) -> list[str]:
        watched = [names] if isinstance(names, str) else names
        return sorted(t.name for t in live_dk_daemon_threads(watched) if t not in baseline)

    yield _still_alive

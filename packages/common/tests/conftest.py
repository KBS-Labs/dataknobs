"""Shared fixtures for ``dataknobs-common`` tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import pytest

from dataknobs_common.testing import declare_import_root, live_dk_daemon_threads

# This directory, so its shared helper modules (``_dataclass_sweep``,
# ``_vocabularies``) import by bare name. pytest's prepend import mode already
# inserts a rootdir without an ``__init__.py``, which is how the fsm suite's
# helpers were reached until an ``__init__.py`` there made the mechanism
# ambiguous and broke them. Declaring it says out loud what the imports below
# depend on.
#
# The declaration precedes the import it enables, which is why that import
# carries an E402 directive rather than sitting with the others: the ordering
# is the dependency, and moving the import up would break the file rather than
# tidy it.
declare_import_root(__file__)

from _vocabularies import (  # noqa: E402 - must follow the declare_import_root above
    MAMMALS_DOCUMENT,
    MAMMALS_V11_DOCUMENT,
    MATERIALIZED_CONTENT_DOCUMENT,
    MATERIALIZED_STRUCTURE_DOCUMENT,
)


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


@pytest.fixture
def mammals_path(tmp_path: Path) -> Path:
    """:data:`MAMMALS_DOCUMENT` written to disk."""
    path = tmp_path / "mammals.yaml"
    path.write_text(MAMMALS_DOCUMENT)
    return path


@pytest.fixture
def mammals_v11_path(tmp_path: Path) -> Path:
    """:data:`MAMMALS_V11_DOCUMENT` written to disk."""
    path = tmp_path / "mammals.yaml"
    path.write_text(MAMMALS_V11_DOCUMENT)
    return path


@pytest.fixture
def materialized_content_path(tmp_path: Path) -> Path:
    """:data:`MATERIALIZED_CONTENT_DOCUMENT` written to disk.

    Its own filename, because ``tmp_path`` is per *test* and not per fixture:
    a test requesting two of these fixtures under one name would get whichever
    wrote last, twice, and every assertion comparing them would hold.
    """
    path = tmp_path / "materialized-content.yaml"
    path.write_text(MATERIALIZED_CONTENT_DOCUMENT)
    return path


@pytest.fixture
def materialized_structure_path(tmp_path: Path) -> Path:
    """:data:`MATERIALIZED_STRUCTURE_DOCUMENT` written to disk.

    Its own filename, for the reason :func:`materialized_content_path` gives.
    """
    path = tmp_path / "materialized-structure.yaml"
    path.write_text(MATERIALIZED_STRUCTURE_DOCUMENT)
    return path

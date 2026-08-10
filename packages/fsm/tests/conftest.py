"""Pytest configuration and shared fixtures for dataknobs_fsm tests."""

import pytest
from collections.abc import Iterator
from pathlib import Path

from dataknobs_common.testing import (
    assert_no_leaked_bridge_threads,
    declare_import_root,
)

# Add src and the examples directory to path for testing.
#
# ``examples`` rather than the package root, which is what carried it before:
# a root exposes every immediate child as a top-level name, and this package's
# children include ``tests`` — a name the repo's own ``tests/`` already
# supplies. The two merged into one namespace package whose search order
# followed ``sys.path``, so a module added here under a name the workspace
# guards import (``_workspace``) would have shadowed theirs, silently, in
# exactly the runs those guards exist to protect. The examples directory holds
# ten modules and no subdirectories, so it has nothing to leak.
_pkg_root = Path(__file__).parent.parent
declare_import_root(_pkg_root / "src")
declare_import_root(_pkg_root / "examples")

# This directory too: its shared fixture modules (``_resource_fixtures``,
# ``custom_fns_fixture``) are imported by bare name. They were reached through
# the top-level name ``tests`` until an ``__init__.py`` here made that name
# ambiguous with ``packages/bots/tests`` and the repo's own ``tests/``.
declare_import_root(__file__)


@pytest.fixture(autouse=True)
def _no_leaked_daemon_threads() -> Iterator[None]:
    """Fail any test that leaks a dataknobs daemon thread.

    The synchronous entry points that belong to an explicit-lifecycle object
    — ``SimpleFSM``, ``AdvancedFSM.execute_step_sync`` — drive the async
    engine through a shared bridge, and allocating one costs a daemon
    event-loop thread that lives until ``close()``. A test that steps an FSM
    and drops it leaks that thread for the rest of the session.

    The leak is silent where it happens: daemon threads never delay exit, the
    FSM keeps working, nothing raises. It surfaced instead as *another
    package's* thread assertions failing depending on test order, which named
    the wrong culprit entirely. Per-test and delta-based, this names the
    right one — and an earlier leak cannot cascade into every later test.

    The reliable way to satisfy it is ``with create_advanced_fsm(...) as
    fsm:`` — every API class here supports the context-manager form.
    """
    with assert_no_leaked_bridge_threads():
        yield


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "id": "test-123",
        "name": "Test Item",
        "value": 42,
        "metadata": {
            "created": "2024-01-01",
            "source": "test"
        }
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for file-based tests."""
    return tmp_path

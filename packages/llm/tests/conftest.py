"""Pytest configuration for ``dataknobs-llm`` tests.

This suite shares scaffolding between modules — the aiohttp transport stubs in
``_aiohttp_error_stub``, the Anthropic SDK stand-ins in ``_anthropic_stubs``,
the Bedrock boundary stubs in ``_bedrock_stubs`` — and imports each by bare
name. That resolved only because pytest's ``prepend`` import mode inserts each
collected file's directory onto ``sys.path`` as a side effect of collecting it,
which the root configuration's ``importlib`` mode does not do. So the imports
worked under ``pytest packages/llm/tests`` and failed under the same command
with any second package named after it.

Declaring the root here states what was being relied on, and holds under both
modes and every invocation: pytest loads this file before collecting anything
beside it.

The declared root is ``_support/`` rather than this directory. Both make the
three modules importable by bare name, but a root exposes *every* immediate
child as a top-level name, and this directory has ten subdirectories — two of
which, ``prompts`` and ``integration``, are also names ``packages/bots/tests``
supplies. A regular package beats a namespace portion no matter which came
first on ``sys.path``, so a later ``from prompts import ...`` here would have
bound to bots' package while still passing under ``pytest packages/llm/tests``:
the same defect this file exists to close, one directory further down. A
directory holding nothing but scaffolding has no such children to leak.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from dataknobs_common.testing import assert_no_leaked_bridge_threads, declare_import_root

declare_import_root(Path(__file__).parent / "_support")


@pytest.fixture(autouse=True)
def _no_leaked_daemon_threads() -> Iterator[None]:
    """Fail the test that leaves a dataknobs daemon thread behind.

    ``SyncProviderAdapter`` owns a :class:`SyncLoopBridge`, so every sync
    provider this suite builds holds an event loop on a daemon thread until
    its ``close()``. A daemon thread never delays exit and the object goes on
    working, so a missing ``close()`` is silent here and surfaces somewhere
    else --- ``data``, ``bots`` and ``fsm`` all carry this same guard, and the
    reason they do is that an unrelated suite's leak once broke their
    thread-teardown assertions depending on test order.

    Per **test**, not per session, and for the same reason ``bots`` gives: a
    session-scoped guard reports a count with no test identity. The guard
    measures a delta, so one leak cannot cascade into every later test.

    **It under-reports, and knowing how is the point.** ``SyncLoopBridge``
    closes itself from ``__del__``, so an adapter whose last reference dies
    with the test's frame is already torn down before this check runs. What
    the guard actually names is a leak the test is still *holding* --- under
    ``pytest.raises(...) as excinfo``, whose captured traceback keeps the
    frame alive, or in a fixture. The rest emit a ``ResourceWarning`` that
    Python's default filters drop, and pass here. ``-W always::ResourceWarning``
    is what surfaces those; it found five in ``test_resources.py`` that this
    guard reported clean.
    """
    with assert_no_leaked_bridge_threads():
        yield

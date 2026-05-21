"""Drift guards for the distributed-lock registry factories.

Structural mirror of :mod:`test_event_bus_factory_parity`. The lock
factory uses the whole-dict ``cls(config=config)`` shape rather than
allowlist enumeration, so the SQS-style allowlist-drops-a-knob drift
mode doesn't apply directly. Two checks still pin the contract:

1. Each registered factory passes only kwargs that exist on the
   target ctor (``assert_factory_kwargs_match_ctor``).
2. ``PostgresAdvisoryLock.__init__`` reads the documented postgres
   connection keys from its config dict
   (``assert_ctor_reads_documented_keys``).

If a future factory regresses to per-kwarg enumeration without a
matching ctor change, check #1 surfaces it. If the postgres lock
silently stops reading a documented key (so consumers passing it via
``create_lock({...})`` silently get a different resolution), check
#2 surfaces it.
"""

from __future__ import annotations

import pytest

from dataknobs_common.locks import lock_backends
from dataknobs_common.locks.factory import (
    _create_in_process_lock,
    _create_postgres_lock,
)
from dataknobs_common.locks.memory import InProcessLock
from dataknobs_common.locks.postgres import PostgresAdvisoryLock
from dataknobs_common.testing import (
    assert_factory_kwargs_match_ctor,
)


def test_in_process_lock_factory_signature() -> None:
    """``_create_in_process_lock`` calls ``InProcessLock()`` with no kwargs."""
    assert_factory_kwargs_match_ctor(_create_in_process_lock, InProcessLock)


def test_postgres_lock_factory_signature() -> None:
    """``_create_postgres_lock`` passes ``config=config`` (whole-dict path)."""
    assert_factory_kwargs_match_ctor(
        _create_postgres_lock,
        PostgresAdvisoryLock,
        # The factory uses the keyword-only ``config=`` path, so it
        # does not forward ``connection_string`` — that's a legacy
        # positional for direct construction.
        ignore_kwargs={"connection_string"},
    )


def test_postgres_lock_ctor_delegates_to_normalizer() -> None:
    """``PostgresAdvisoryLock.__init__`` delegates to the shared normalizer.

    The whole-dict-pass pattern: every postgres-connection key (host,
    port, user, password, connection_string, ...) is resolved by
    :func:`normalize_postgres_connection_config`. The leaf ctor's
    contract is "forward the dict to the normalizer" — so the parity
    guarantee is "the normalizer is called on the merged dict", not
    "each key is read by name". This AST-walks the ctor for a call to
    the normalizer; absence indicates the lock has stopped routing
    through the canonical resolution path.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(PostgresAdvisoryLock.__init__))
    tree = ast.parse(src)
    found_normalize_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "normalize_postgres_connection_config"
        for node in ast.walk(tree)
    )
    assert found_normalize_call, (
        "PostgresAdvisoryLock.__init__ no longer calls "
        "normalize_postgres_connection_config — postgres-connection key "
        "resolution is documented to use this single source of truth. "
        "Either restore the call or update this parity test to reflect "
        "the new canonical resolution path."
    )


def test_registered_lock_backends_are_audited() -> None:
    """The audit matrix above must cover every registered built-in.

    New built-in backend → add a row to the per-backend tests above
    so the parity guards continue to cover the registry.
    """
    builtin = {"memory", "postgres"}
    registered = set(lock_backends.list_keys())
    missing_audit = builtin - registered
    new_unaudited = registered - builtin
    assert not missing_audit, (
        f"Built-in lock backends no longer registered: {missing_audit}"
    )
    if new_unaudited:
        pytest.fail(
            f"New lock backends registered without parity coverage: "
            f"{new_unaudited}. Add a per-backend signature test above."
        )

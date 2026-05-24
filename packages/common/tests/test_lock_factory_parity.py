"""Drift guards for the distributed-lock registry factories.

Structural mirror of :mod:`test_event_bus_factory_parity`. The lock
factory routes through ``PostgresAdvisoryLock.from_config(config)``
(the structured-config path) rather than allowlist enumeration, so the
SQS-style allowlist-drops-a-knob drift mode doesn't apply directly.
Three checks pin the contract:

1. Each registered factory passes only kwargs that exist on the
   target ctor (``assert_factory_kwargs_match_ctor``).
2. ``PostgresAdvisoryLock`` correctly applies the structured-config
   pattern: declares ``CONFIG_CLS``, its ctor params match the
   ``PostgresLockConfig`` field set, and its sync ``from_config``
   override routes through ``_coerce_config``
   (``assert_structured_config_consumer``).
3. ``PostgresLockConfig._normalize_dict`` resolves the documented
   postgres connection keys via the shared normalizer (AST guard).

If a future factory regresses to per-kwarg enumeration without a
matching ctor change, check #1 surfaces it. If the lock's construction
surface drifts from its config dataclass, check #2 surfaces it. If the
config silently stops routing through the canonical resolution path (so
consumers passing keys via ``create_lock({...})`` get a different
resolution), check #3 surfaces it.
"""

from __future__ import annotations

import pytest

from dataknobs_common.locks import lock_backends
from dataknobs_common.locks.config import PostgresLockConfig
from dataknobs_common.locks.factory import (
    _create_in_process_lock,
    _create_postgres_lock,
)
from dataknobs_common.locks.memory import InProcessLock
from dataknobs_common.locks.postgres import PostgresAdvisoryLock
from dataknobs_common.testing import (
    assert_factory_kwargs_match_ctor,
    assert_structured_config_consumer,
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


def test_postgres_lock_uses_structured_config_consumer() -> None:
    """``PostgresAdvisoryLock`` correctly applies the structured-config pattern.

    Unified guard combining the CONFIG_CLS declaration, the
    dataclass-field ↔ ctor-param match, and the entry-point symmetry
    (the sync ``from_config`` override routes through ``_coerce_config``).
    The lock keeps a thin ``__init__`` override accepting
    ``connection_string`` positionally for back-compat; it IS the sole
    ``PostgresLockConfig`` field, so it matches the mixin's contract
    without an ignore.
    """
    assert_structured_config_consumer(PostgresAdvisoryLock)


def test_postgres_lock_config_delegates_to_normalizer() -> None:
    """``PostgresLockConfig._normalize_dict`` delegates to the shared normalizer.

    The whole-dict-pass pattern: every postgres-connection key (host,
    port, user, password, connection_string, ...) is resolved by
    :func:`normalize_postgres_connection_config`. After the
    structured-config refactor this resolution lives at the config layer
    (``_normalize_dict``), not the ctor. This AST-walks ``_normalize_dict``
    for a call to the normalizer; absence indicates the lock config has
    stopped routing through the canonical resolution path.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(
        inspect.getsource(PostgresLockConfig._normalize_dict)
    )
    tree = ast.parse(src)
    found_normalize_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "normalize_postgres_connection_config"
        for node in ast.walk(tree)
    )
    assert found_normalize_call, (
        "PostgresLockConfig._normalize_dict no longer calls "
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

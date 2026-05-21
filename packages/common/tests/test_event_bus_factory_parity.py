"""Drift guards for the event-bus registry factories.

The original failure mode: ``_create_sqs_bus`` enumerated 8 of the
9 ctor kwargs ``SqsEventBus`` exposes, silently dropping one knob.
The structural refactor moved construction through a typed
:class:`<Backend>EventBusConfig` dataclass — but the same drift mode
could recur the next time anyone adds a ctor kwarg without a matching
dataclass field. These tests pin both directions:

1. Every config dataclass field must be an ``__init__`` kwarg on its
   bus (``assert_dataclass_config_matches_ctor``).
2. Every registered ``_create_*_bus`` factory must construct its bus
   through the dataclass — so adding a ctor knob propagates to the
   config-driven entry point automatically.

If either check fails, drift has been introduced. The check runs
without service dependencies (LocalStack / Postgres / Redis) — pure
AST + signature inspection.
"""

from __future__ import annotations

import pytest

from dataknobs_common.events import (
    MemoryEventBusConfig,
    PostgresEventBusConfig,
    RedisEventBusConfig,
    SqsEventBusConfig,
    event_bus_backends,
)
from dataknobs_common.events.memory import InMemoryEventBus
from dataknobs_common.events.postgres import PostgresEventBus
from dataknobs_common.events.redis import RedisEventBus
from dataknobs_common.events.registry import (
    _create_memory_bus,
    _create_postgres_bus,
    _create_redis_bus,
    _create_sqs_bus,
)
from dataknobs_common.events.sqs import SqsEventBus
from dataknobs_common.testing import (
    assert_dataclass_config_matches_ctor,
    assert_factory_kwargs_match_ctor,
)


@pytest.mark.parametrize(
    "config_cls, target_cls",
    [
        (MemoryEventBusConfig, InMemoryEventBus),
        (RedisEventBusConfig, RedisEventBus),
        (PostgresEventBusConfig, PostgresEventBus),
        (SqsEventBusConfig, SqsEventBus),
    ],
    ids=["memory", "redis", "postgres", "sqs"],
)
def test_event_bus_dataclass_matches_ctor(
    config_cls: type, target_cls: type
) -> None:
    """Every config dataclass field is a ctor kwarg on its bus.

    The original drift mode was the inverse — a ctor kwarg missing
    from the factory allowlist. Now that the factory consumes the
    dataclass via ``from_dict``, the dataclass IS the allowlist; this
    parity check makes the allowlist-vs-ctor symmetry structurally
    enforced.
    """
    # PostgresEventBus.__init__ takes ``connection_string``,
    # ``channel_prefix`` as positionals for legacy compat; both are
    # also dataclass fields, so they match without an ignore.
    assert_dataclass_config_matches_ctor(config_cls, target_cls)


@pytest.mark.parametrize(
    "factory, target_cls",
    [
        (_create_memory_bus, InMemoryEventBus),
        (_create_redis_bus, RedisEventBus),
        (_create_postgres_bus, PostgresEventBus),
        (_create_sqs_bus, SqsEventBus),
    ],
    ids=["memory", "redis", "postgres", "sqs"],
)
def test_factory_uses_from_config(
    factory: object, target_cls: type
) -> None:
    """Every registry factory routes through ``Bus.from_config``.

    The structural correction: factories no longer enumerate kwargs
    by hand. If a factory regresses to a per-kwarg allowlist, this
    test flags it because the AST walk will find kwargs that aren't
    on the ctor's named-parameter list (the only valid kwarg on the
    new path is ``config=``).
    """
    # ``from_config`` is the recommended path; allow no other named
    # kwargs on the call site. ``config`` is implicit (passed as the
    # only positional or via the dispatch shape).
    assert_factory_kwargs_match_ctor(
        factory,
        target_cls,
        # ``connection_string`` / ``channel_prefix`` on Postgres are
        # legacy positionals; factories don't have to forward them
        # since they're already represented as dataclass fields the
        # config dict flows through.
        ignore_kwargs={"connection_string", "channel_prefix"},
    )


def test_registered_backends_are_audited() -> None:
    """The parity matrix above must cover every registered built-in.

    If a new backend is added (e.g. SNS) and registered in
    ``event_bus_backends`` but not paired with a dataclass + parity
    test row, this test fails so the contributor remembers to add the
    guard. Consumer-registered backends are exempt — they're outside
    the built-in audit surface.
    """
    builtin_backends = {"memory", "postgres", "redis", "sqs"}
    registered = set(event_bus_backends.list_keys())
    missing_audit = builtin_backends - registered
    new_unaudited = registered - builtin_backends
    assert not missing_audit, (
        f"Built-in backends no longer registered: {missing_audit}"
    )
    if new_unaudited:
        pytest.fail(
            f"New backends registered without parity coverage: "
            f"{new_unaudited}. Add a dataclass + add rows to the two "
            "parametrized tests above."
        )

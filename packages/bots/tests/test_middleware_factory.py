"""Tests for the public middleware spec -> instance factories.

These functions were promoted out of ``DynaBot`` so anything that
assembles middleware declaratively can build instances without going
through a bot. The tests below pin the two semantics that had to survive
that move — ``optional: true`` skipping and the always-raising class-shape
check — against the **public** entry points, and pin that the private
aliases still delegate to them.

Real middleware classes throughout (``LoggingMiddleware``,
``HistoryRedactionMiddleware``); no mocks are involved, because the
subject is dotted-path resolution of real classes.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.middleware import (
    build_conversation_middleware,
    build_middleware,
    resolve_middleware_from_spec,
)
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.middleware.logging import LoggingMiddleware
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_llm.conversations import (
    ConversationMiddleware,
    HistoryRedactionMiddleware,
)

_BOT_MIDDLEWARE_CLASS = "dataknobs_bots.middleware.logging.LoggingMiddleware"
_CONVERSATION_MIDDLEWARE_CLASS = (
    "dataknobs_llm.conversations.HistoryRedactionMiddleware"
)
_FACTORY_LOGGER = "dataknobs_bots.middleware.factory"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_build_middleware_returns_live_instances() -> None:
    """The spec shape the bot config uses, resolved without a bot."""
    built = build_middleware([{"class": _BOT_MIDDLEWARE_CLASS}])

    assert len(built) == 1
    assert isinstance(built[0], LoggingMiddleware)
    assert isinstance(built[0], Middleware)


def test_build_middleware_preserves_spec_order() -> None:
    """Order is behavior for middleware, so the list order is contractual."""
    built = build_middleware(
        [
            {"class": _BOT_MIDDLEWARE_CLASS, "params": {"log_level": "DEBUG"}},
            {"class": "dataknobs_bots.middleware.cost.CostTrackingMiddleware"},
            {"class": _BOT_MIDDLEWARE_CLASS},
        ]
    )

    assert [type(mw).__name__ for mw in built] == [
        "LoggingMiddleware",
        "CostTrackingMiddleware",
        "LoggingMiddleware",
    ]


def test_build_middleware_accepts_an_empty_sequence() -> None:
    """The no-middleware case is a plain empty list, not an error."""
    assert build_middleware([]) == []
    assert build_conversation_middleware([]) == []


def test_build_conversation_middleware_passes_params_to_the_ctor() -> None:
    """``params`` reaches the constructor as keyword arguments."""
    built = build_conversation_middleware(
        [
            {
                "class": _CONVERSATION_MIDDLEWARE_CLASS,
                "params": {
                    "redactions": [{"pattern": r"\d{3}", "replacement": "###"}]
                },
            }
        ]
    )

    assert len(built) == 1
    assert isinstance(built[0], HistoryRedactionMiddleware)
    assert isinstance(built[0], ConversationMiddleware)


def test_resolve_accepts_a_base_neither_wrapper_covers() -> None:
    """The general entry point works for any expected base + label.

    ``LoggingMiddleware`` is resolved against its own concrete class
    rather than either built-in flavor, which is the documented reason
    the general function is public alongside the two wrappers.
    """
    mw = resolve_middleware_from_spec(
        {"class": _BOT_MIDDLEWARE_CLASS}, LoggingMiddleware, label="custom"
    )

    assert isinstance(mw, LoggingMiddleware)


# ---------------------------------------------------------------------------
# ``optional`` semantics — must survive the move
# ---------------------------------------------------------------------------


def test_optional_true_skips_a_resolution_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing class is skipped with a warning, not raised."""
    spec = {"class": "nonexistent.module.NoSuchClass", "optional": True}

    with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
        assert build_middleware([spec]) == []
        assert build_conversation_middleware([spec]) == []

    assert "nonexistent.module.NoSuchClass" in caplog.text
    assert "Skipping optional" in caplog.text
    # The warning is attributed to the factory module, not to the bot.
    assert any(record.name == _FACTORY_LOGGER for record in caplog.records)


def test_a_skipped_optional_spec_does_not_shift_the_others() -> None:
    """The skip is a removal, not a ``None`` hole in the result.

    This is the whole reason the builders take a sequence: a caller passing
    the result to ``platform_middleware=`` must never receive a list with a
    ``None`` in it, which would surface much later as an opaque
    ``AttributeError`` at turn time.
    """
    built = build_middleware(
        [
            {"class": _BOT_MIDDLEWARE_CLASS},
            {"class": "nonexistent.module.NoSuchClass", "optional": True},
            {"class": "dataknobs_bots.middleware.cost.CostTrackingMiddleware"},
        ]
    )

    assert [type(mw).__name__ for mw in built] == [
        "LoggingMiddleware",
        "CostTrackingMiddleware",
    ]
    assert all(mw is not None for mw in built)


def test_optional_true_skips_an_instantiation_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ctor that raises (required param omitted) is skipped too."""
    spec = {"class": _CONVERSATION_MIDDLEWARE_CLASS, "optional": True}

    with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
        assert build_conversation_middleware([spec]) == []

    assert "Skipping optional" in caplog.text


def test_resolution_failure_raises_without_optional() -> None:
    """Without ``optional``, an unresolvable class is a config error."""
    with pytest.raises(ConfigurationError) as excinfo:
        build_middleware([{"class": "nonexistent.module.NoSuchClass"}])

    assert "Failed to resolve middleware" in str(excinfo.value)


_LEAKY_DSN = "postgresql://svc:hunter2@db.internal:5432/prod"


class LeakyCtorMiddleware(Middleware):
    """A middleware whose constructor fails the way a real driver does.

    Not a mock: the point is that an arbitrary third-party constructor runs
    inside the factory's ``except Exception``, and the text it raises with is
    outside DataKnobs' control. Database and cache clients routinely put the
    connection URL in the message.
    """

    def __init__(self) -> None:
        raise ValueError(f"Could not parse URL from string {_LEAKY_DSN!r}")


_LEAKY_CLASS = "tests.test_middleware_factory.LeakyCtorMiddleware"


def test_a_failing_ctor_does_not_leak_its_message_into_the_config_error() -> None:
    """The funnel catches ``Exception``, so ``{e}`` here is unbounded text.

    ``ConfigurationError`` is a diagnostic type whose messages are otherwise
    authored — key names, class paths, sorted key lists — and this package
    renders it at the HTTP boundary. A message built from an arbitrary
    constructor's failure breaks that property, and the constructor in
    question is the consumer's, reached through their config.

    The class path stays: it names the spec that failed, which is what the
    deployment has to fix, and it comes from the config rather than from the
    exception. The underlying error stays reachable through ``__cause__``.
    """
    with pytest.raises(ConfigurationError) as excinfo:
        build_middleware([{"class": _LEAKY_CLASS}])

    message = str(excinfo.value)
    assert "hunter2" not in message
    assert _LEAKY_DSN not in message
    assert _LEAKY_CLASS in message
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert _LEAKY_DSN in str(excinfo.value.__cause__)


# ---------------------------------------------------------------------------
# Class-shape check — always raises, never optional
# ---------------------------------------------------------------------------


def test_wrong_flavor_raises_in_both_directions() -> None:
    """Each wrapper rejects the other's class."""
    with pytest.raises(ConfigurationError) as bot_exc:
        build_middleware([{"class": _CONVERSATION_MIDDLEWARE_CLASS}])
    assert "must subclass" in str(bot_exc.value)

    with pytest.raises(ConfigurationError) as conv_exc:
        build_conversation_middleware([{"class": _BOT_MIDDLEWARE_CLASS}])
    assert "must subclass" in str(conv_exc.value)


@pytest.mark.parametrize(
    ("build", "class_path"),
    [
        (build_middleware, _CONVERSATION_MIDDLEWARE_CLASS),
        (build_conversation_middleware, _BOT_MIDDLEWARE_CLASS),
    ],
)
def test_optional_true_does_not_silence_a_shape_mismatch(build, class_path) -> None:
    """``optional`` covers transient failures, never a misplaced spec."""
    with pytest.raises(ConfigurationError, match="must subclass"):
        build([{"class": class_path, "optional": True}])


def test_a_shape_mismatch_raises_even_behind_a_valid_spec() -> None:
    """The sequence form must not swallow a later spec's shape error."""
    with pytest.raises(ConfigurationError, match="must subclass"):
        build_middleware(
            [
                {"class": _BOT_MIDDLEWARE_CLASS},
                {"class": _CONVERSATION_MIDDLEWARE_CLASS},
            ]
        )


# ---------------------------------------------------------------------------
# Delegation — the private aliases still route here
# ---------------------------------------------------------------------------


def test_bot_module_binds_the_factory_functions_themselves() -> None:
    """``bot.base`` holds the factory's own function objects.

    Identity, not equivalence: a re-inlined second copy of the resolution
    body in ``bot.base`` would still behave identically (so the behavioral
    guard below would pass) while reintroducing exactly the duplication
    this move removed.
    """
    from dataknobs_bots.bot import base as bot_base
    from dataknobs_bots.middleware import factory

    assert bot_base.build_middleware is factory.build_middleware
    assert (
        bot_base.build_conversation_middleware
        is factory.build_conversation_middleware
    )
    assert (
        bot_base.resolve_middleware_from_spec is factory.resolve_middleware_from_spec
    )


def test_private_aliases_match_the_public_functions() -> None:
    """``DynaBot``'s private single-spec helpers agree with the builders."""
    bot_spec = {"class": _BOT_MIDDLEWARE_CLASS}
    conv_spec = {
        "class": _CONVERSATION_MIDDLEWARE_CLASS,
        "params": {"redactions": []},
    }

    assert type(DynaBot._create_bot_middleware(bot_spec)) is type(
        build_middleware([bot_spec])[0]
    )
    assert type(DynaBot._create_conversation_middleware(conv_spec)) is type(
        build_conversation_middleware([conv_spec])[0]
    )
    assert type(
        DynaBot._resolve_middleware_from_spec(bot_spec, Middleware, label="middleware")
    ) is type(build_middleware([bot_spec])[0])


def test_private_aliases_propagate_the_shape_error() -> None:
    """Delegation preserves the raise, not just the happy path."""
    with pytest.raises(ConfigurationError, match="must subclass"):
        DynaBot._create_bot_middleware({"class": _CONVERSATION_MIDDLEWARE_CLASS})


async def test_the_bot_builds_its_configured_lists_through_the_factory() -> None:
    """``DynaBot`` uses the public builders rather than its own loop.

    The bot previously collected each of its two configured middleware
    lists with its own inline "resolve, skip ``None``, append" loop — the
    same six lines twice. Both now route through the builders, so a
    behavior change to spec resolution lands in one place. Asserted through
    a real bot build rather than by reading the source: the observable
    contract is that a configured spec becomes a live instance and an
    unresolvable ``optional`` one is dropped.
    """
    from dataknobs_bots.testing import BotTestHarness

    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
            "middleware": [
                {"class": _BOT_MIDDLEWARE_CLASS},
                {"class": "nonexistent.module.NoSuchClass", "optional": True},
            ],
        },
        main_responses=["ok"],
    ) as harness:
        names = [type(mw).__name__ for mw in harness.bot.middleware]
        assert "LoggingMiddleware" in names
        assert all(mw is not None for mw in harness.bot.middleware)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_exported_from_the_top_level_package() -> None:
    """All three reach ``dataknobs_bots`` itself, not just the subpackage.

    A consumer assembling middleware declaratively should not have to know
    which subpackage the factories live in — the same accessibility the
    ``Middleware`` base class already has. Identity-checked so a future
    re-binding (a wrapper, a shim) is a deliberate choice rather than an
    accident.
    """
    import dataknobs_bots
    from dataknobs_bots.middleware import factory

    assert dataknobs_bots.build_middleware is factory.build_middleware
    assert (
        dataknobs_bots.build_conversation_middleware
        is factory.build_conversation_middleware
    )
    assert (
        dataknobs_bots.resolve_middleware_from_spec
        is factory.resolve_middleware_from_spec
    )

    for name in (
        "build_middleware",
        "build_conversation_middleware",
        "resolve_middleware_from_spec",
    ):
        assert name in dataknobs_bots.__all__, f"{name} missing from __all__"


# ---------------------------------------------------------------------------
# Import direction
# ---------------------------------------------------------------------------


def test_factory_imports_standalone_without_a_cycle() -> None:
    """Importing the factory first in a fresh interpreter must succeed.

    The module deliberately depends only on ``.base`` and
    ``dataknobs_llm.conversations`` — never on ``bot.base``. Reaching back
    into the bot would create a cycle, which surfaces as an ``ImportError``
    only when the factory is the entry point, not when a bot import has
    already primed the module graph.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import dataknobs_bots.middleware.factory"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

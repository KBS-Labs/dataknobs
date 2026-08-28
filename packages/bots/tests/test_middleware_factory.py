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
from collections.abc import Iterator

import pytest
from dataknobs_common.imports import dotted_path

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.middleware import (
    build_conversation_middleware,
    build_middleware,
    resolve_middleware_class,
    resolve_middleware_from_spec,
)
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.middleware.logging import LoggingMiddleware
from dataknobs_common.exceptions import (
    ConfigurationError,
    DottedPathError,
    DottedPathReason,
)
from dataknobs_llm.conversations import (
    ConversationMiddleware,
    HistoryRedactionMiddleware,
)

_BOT_MIDDLEWARE_CLASS = "dataknobs_bots.middleware.logging.LoggingMiddleware"
_CONVERSATION_MIDDLEWARE_CLASS = "dataknobs_llm.conversations.HistoryRedactionMiddleware"
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
                "params": {"redactions": [{"pattern": r"\d{3}", "replacement": "###"}]},
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


_LEAKY_CLASS = dotted_path(LeakyCtorMiddleware)


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
# The check half — every rule above, with no constructor run
# ---------------------------------------------------------------------------

_CONSTRUCTED: list[str] = []


class ConstructionRecordingMiddleware(Middleware):
    """A real ``Middleware`` whose constructor is observable.

    Not a mock, and not decoration: "constructed nothing" cannot be
    asserted from the absence of an error, because a check that built an
    instance and threw it away would pass that assertion while doing the
    exact thing the caller is avoiding. The side effect is the only way to
    tell the two apart from outside.
    """

    def __init__(self, tag: str = "untagged") -> None:
        _CONSTRUCTED.append(tag)
        self.tag = tag


_RECORDING_CLASS = dotted_path(ConstructionRecordingMiddleware)


@pytest.fixture
def constructed() -> Iterator[list[str]]:
    """The tags of every ``ConstructionRecordingMiddleware`` built so far."""
    _CONSTRUCTED.clear()
    yield _CONSTRUCTED
    _CONSTRUCTED.clear()


def test_the_check_half_runs_no_constructor(constructed: list[str]) -> None:
    """The item's whole point: same spec, class back, ctor unrun.

    Both calls are made on one spec so the difference between them is the
    construction and nothing else.
    """
    spec = {"class": _RECORDING_CLASS, "params": {"tag": "checked"}}

    resolved = resolve_middleware_class(spec, Middleware, label="middleware")

    assert resolved is not None
    cls, params = resolved
    assert cls is ConstructionRecordingMiddleware
    assert dict(params) == {"tag": "checked"}
    assert constructed == []

    built = resolve_middleware_from_spec(spec, Middleware, label="middleware")

    assert isinstance(built, ConstructionRecordingMiddleware)
    assert constructed == ["checked"]


@pytest.mark.parametrize(
    ("class_path", "expected_base", "label"),
    [
        (_CONVERSATION_MIDDLEWARE_CLASS, Middleware, "middleware"),
        (_BOT_MIDDLEWARE_CLASS, ConversationMiddleware, "conversation_middleware"),
    ],
)
def test_the_check_half_rejects_a_wrong_rail_class_even_when_optional(
    class_path: str, expected_base: type, label: str
) -> None:
    """``optional`` covers resolution failure, never a misplaced spec.

    The one semantic the split had to carry across unchanged: a checker
    that skipped a shape mismatch under ``optional`` would accept configs
    the builder rejects, which is the divergence the shared body exists to
    prevent.
    """
    with pytest.raises(ConfigurationError, match="must subclass"):
        resolve_middleware_class(
            {"class": class_path, "optional": True}, expected_base, label=label
        )


def test_the_check_half_skips_where_the_build_half_skips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unresolvable path: ``None`` under ``optional``, raise without it.

    Same point, same exception, same lifted ``reason`` and ``label`` — a
    caller that checks with one and builds with the other cannot disagree
    about which specs are installable.
    """
    spec = {"class": "nonexistent.module.NoSuchClass"}

    with pytest.raises(DottedPathError) as excinfo:
        resolve_middleware_class(spec, Middleware, label="middleware")

    assert excinfo.value.ref == "nonexistent.module.NoSuchClass"
    assert excinfo.value.reason == DottedPathReason.MODULE_NOT_FOUND
    assert excinfo.value.context["label"] == "middleware"

    with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
        skipped = resolve_middleware_class(
            {**spec, "optional": True}, Middleware, label="middleware"
        )

    assert skipped is None
    assert "Skipping optional" in caplog.text
    assert any(record.name == _FACTORY_LOGGER for record in caplog.records)


def test_the_check_half_reports_a_spec_with_no_class_key() -> None:
    """The ``KeyError`` clause moved with the rest, ``optional`` included."""
    with pytest.raises(ConfigurationError, match="no 'class' key"):
        resolve_middleware_class({}, Middleware, label="middleware")

    assert resolve_middleware_class({"optional": True}, Middleware, label="middleware") is None


def test_a_spec_the_ctor_rejects_passes_the_check_and_fails_the_build() -> None:
    """What a check cannot catch, pinned as a test rather than a docstring.

    ``resolve_middleware_class`` is strictly weaker than the build path,
    and inherently so: detecting that a constructor rejects its ``params``
    means running it. A linter built on this answers "is this spec
    installable?", never "will this bot start?" — and the asymmetry is
    tested so the next reader meets it as behavior rather than as a claim.
    """
    spec = {"class": _BOT_MIDDLEWARE_CLASS, "params": {"no_such_param": True}}

    resolved = resolve_middleware_class(spec, Middleware, label="middleware")

    assert resolved is not None
    assert resolved[0] is LoggingMiddleware

    with pytest.raises(ConfigurationError, match="Failed to instantiate"):
        resolve_middleware_from_spec(spec, Middleware, label="middleware")


def test_the_build_half_routes_through_the_check_half() -> None:
    """Identity, not equivalence — one resolution body, not two.

    A re-inlined copy of the resolve clauses inside
    ``resolve_middleware_from_spec`` would satisfy every behavioral test
    above while reintroducing the drift the split exists to remove, so the
    delegation is asserted directly.

    Read off the compiled code object rather than the source text, so a
    docstring that *mentions* ``resolve_class`` (the build half's does,
    cross-referencing where the rules now live) cannot fail this.
    """
    from dataknobs_bots.middleware import factory

    called = factory.resolve_middleware_from_spec.__code__.co_names

    assert "resolve_middleware_class" in called
    assert "resolve_class" not in called


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
    assert bot_base.build_conversation_middleware is factory.build_conversation_middleware
    assert bot_base.resolve_middleware_from_spec is factory.resolve_middleware_from_spec


def test_private_aliases_match_the_public_functions() -> None:
    """``DynaBot``'s private single-spec helpers agree with the builders."""
    bot_spec = {"class": _BOT_MIDDLEWARE_CLASS}
    conv_spec = {
        "class": _CONVERSATION_MIDDLEWARE_CLASS,
        "params": {"redactions": []},
    }

    assert type(DynaBot._create_bot_middleware(bot_spec)) is type(build_middleware([bot_spec])[0])
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
    """All four reach ``dataknobs_bots`` itself, not just the subpackage.

    A consumer assembling middleware declaratively should not have to know
    which subpackage the factories live in — the same accessibility the
    ``Middleware`` base class already has. Identity-checked so a future
    re-binding (a wrapper, a shim) is a deliberate choice rather than an
    accident.

    The check half is held to the same reach as the build half: a caller
    that lints with one and installs with the other should import them
    from the same place, or the pairing is something it has to discover.
    """
    import dataknobs_bots
    from dataknobs_bots.middleware import factory

    assert dataknobs_bots.build_middleware is factory.build_middleware
    assert dataknobs_bots.build_conversation_middleware is factory.build_conversation_middleware
    assert dataknobs_bots.resolve_middleware_from_spec is factory.resolve_middleware_from_spec
    assert dataknobs_bots.resolve_middleware_class is factory.resolve_middleware_class

    for name in (
        "build_middleware",
        "build_conversation_middleware",
        "resolve_middleware_from_spec",
        "resolve_middleware_class",
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

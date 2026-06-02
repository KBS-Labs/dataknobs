"""Tests for from_config() dependency injection (Gap 6).

Tests the ``llm`` and ``middleware`` override kwargs on
``DynaBot.from_config()``.  These are factory method tests — direct
``from_config()`` calls are appropriate here (not BotTestHarness).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.bot.turn import TurnState
from dataknobs_bots.middleware.base import Middleware
from dataknobs_llm import EchoProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _echo_provider() -> EchoProvider:
    """Create a pre-built EchoProvider (simulates shared provider)."""
    return EchoProvider({"provider": "echo", "model": "shared-echo"})


class TrackingMiddleware(Middleware):
    """Middleware that records hook calls for verification."""

    def __init__(self, name: str = "tracker") -> None:
        self.name = name
        self.turns: list[TurnState] = []

    async def after_turn(self, turn: TurnState) -> None:
        self.turns.append(turn)


# ---------------------------------------------------------------------------
# LLM injection tests
# ---------------------------------------------------------------------------

class TestFromConfigLLMInjection:
    """from_config() accepts a pre-built LLM provider via the ``llm`` kwarg."""

    @pytest.mark.asyncio
    async def test_injected_llm_is_used(self) -> None:
        """Injected provider is the one the bot uses for generation."""
        shared = _echo_provider()
        shared.set_responses(["Hello from shared!"])

        bot = await DynaBot.from_config(
            {"conversation_storage": {"backend": "memory"}},
            llm=shared,
        )
        async with bot:
            ctx = BotContext(conversation_id="c1", client_id="t1")
            result = await bot.chat("hi", ctx)

        assert result == "Hello from shared!"
        assert shared.call_count == 1
        assert bot.llm is shared

    @pytest.mark.asyncio
    async def test_config_llm_is_optional_when_injected(self) -> None:
        """config["llm"] is not required when llm kwarg is provided."""
        shared = _echo_provider()

        # No "llm" key in config at all
        bot = await DynaBot.from_config(
            {"conversation_storage": {"backend": "memory"}},
            llm=shared,
        )
        async with bot:
            assert bot.llm is shared

    @pytest.mark.asyncio
    async def test_config_llm_ignored_when_injected(self) -> None:
        """config["llm"] is ignored when llm kwarg is provided."""
        shared = _echo_provider()
        shared.set_responses(["from shared"])

        # Config has a different provider config — should be ignored
        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "config-echo"},
                "conversation_storage": {"backend": "memory"},
            },
            llm=shared,
        )
        async with bot:
            ctx = BotContext(conversation_id="c1", client_id="t1")
            result = await bot.chat("hi", ctx)

        assert result == "from shared"
        assert bot.llm is shared

    @pytest.mark.asyncio
    async def test_injected_llm_not_closed_on_bot_close(self) -> None:
        """Caller-owned provider is NOT closed when bot is closed.

        The originator-owns-lifecycle principle: from_config did not
        create the provider, so it must not close it.
        """
        shared = _echo_provider()
        shared.set_responses(["response"])

        bot = await DynaBot.from_config(
            {"conversation_storage": {"backend": "memory"}},
            llm=shared,
        )
        async with bot:
            ctx = BotContext(conversation_id="c1", client_id="t1")
            await bot.chat("hi", ctx)

        # After bot.close(), the shared provider should still be usable
        # (EchoProvider doesn't really "close" but the contract is clear)
        shared.set_responses(["still alive"])
        bot2 = await DynaBot.from_config(
            {"conversation_storage": {"backend": "memory"}},
            llm=shared,
        )
        async with bot2:
            ctx2 = BotContext(conversation_id="c2", client_id="t1")
            result = await bot2.chat("hi", ctx2)
        assert result == "still alive"

    @pytest.mark.asyncio
    async def test_shared_provider_across_instances(self) -> None:
        """Multiple bot instances can share a single provider."""
        shared = _echo_provider()
        shared.set_responses(["r1", "r2"])

        config = {"conversation_storage": {"backend": "memory"}}

        bot1 = await DynaBot.from_config(config, llm=shared)
        bot2 = await DynaBot.from_config(config, llm=shared)

        async with bot1, bot2:
            ctx1 = BotContext(conversation_id="c1", client_id="t1")
            ctx2 = BotContext(conversation_id="c2", client_id="t1")

            r1 = await bot1.chat("hi", ctx1)
            r2 = await bot2.chat("hi", ctx2)

        assert r1 == "r1"
        assert r2 == "r2"
        assert shared.call_count == 2
        assert bot1.llm is bot2.llm is shared


# ---------------------------------------------------------------------------
# Middleware injection tests
# ---------------------------------------------------------------------------

class TestFromConfigMiddlewareInjection:
    """from_config() accepts pre-built middleware via the ``middleware`` kwarg."""

    @pytest.mark.asyncio
    async def test_injected_middleware_is_used(self) -> None:
        """Injected middleware list replaces config-driven middleware."""
        tracker = TrackingMiddleware("injected")

        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
            middleware=[tracker],
        )
        async with bot:
            ctx = BotContext(conversation_id="c1", client_id="t1")
            await bot.chat("hi", ctx)

        assert len(tracker.turns) == 1
        assert bot.middleware == [tracker]

    @pytest.mark.asyncio
    async def test_middleware_override_replaces_config(self) -> None:
        """Config-defined middleware is ignored when middleware kwarg is provided."""
        tracker = TrackingMiddleware("injected")

        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "middleware": [{
                    "class": "dataknobs_bots.middleware.logging.LoggingMiddleware",
                }],
            },
            middleware=[tracker],
        )
        async with bot:
            # Only our injected middleware should be present
            assert len(bot.middleware) == 1
            assert bot.middleware[0] is tracker

    @pytest.mark.asyncio
    async def test_empty_middleware_override(self) -> None:
        """Empty middleware list explicitly disables all middleware."""
        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "middleware": [{
                    "class": "dataknobs_bots.middleware.logging.LoggingMiddleware",
                }],
            },
            middleware=[],
        )
        async with bot:
            assert bot.middleware == []

    @pytest.mark.asyncio
    async def test_no_middleware_kwarg_uses_config(self) -> None:
        """Without middleware kwarg, config-driven middleware is created."""
        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "middleware": [{
                    "class": "dataknobs_bots.middleware.logging.LoggingMiddleware",
                }],
            },
        )
        async with bot:
            from dataknobs_bots.middleware.logging import LoggingMiddleware

            assert len(bot.middleware) == 1
            assert isinstance(bot.middleware[0], LoggingMiddleware)


# ---------------------------------------------------------------------------
# Combined injection tests
# ---------------------------------------------------------------------------

class TestFromConfigCombinedInjection:
    """Both llm and middleware can be injected together."""

    @pytest.mark.asyncio
    async def test_both_llm_and_middleware_injected(self) -> None:
        """Injecting both llm and middleware works together."""
        shared = _echo_provider()
        shared.set_responses(["combined"])
        tracker = TrackingMiddleware("combined")

        bot = await DynaBot.from_config(
            {"conversation_storage": {"backend": "memory"}},
            llm=shared,
            middleware=[tracker],
        )
        async with bot:
            ctx = BotContext(conversation_id="c1", client_id="t1")
            result = await bot.chat("hi", ctx)

        assert result == "combined"
        assert bot.llm is shared
        assert bot.middleware == [tracker]
        assert len(tracker.turns) == 1


# ---------------------------------------------------------------------------
# ConversationMiddleware injection tests (LLM-call wraps, distinct from
# the bot-turn `middleware` channel above).
# ---------------------------------------------------------------------------


class TestFromConfigConversationMiddleware:
    """from_config() supports ConversationMiddleware via both config and kwarg.

    These guard:
      - the config-driven path (``conversation_middleware:`` list of specs)
        actually builds and forwards LLM-call wraps to every
        ``ConversationManager`` the bot creates,
      - the ``conversation_middleware=`` kwarg replaces the config-driven
        list symmetrically with the existing ``middleware=`` kwarg, and
      - a type-mismatched spec (a bot-turn ``Middleware`` listed under
        ``conversation_middleware:``) is rejected at config-load with a
        clear error rather than crashing on first message.
    """

    @pytest.mark.asyncio
    async def test_config_driven_conversation_middleware_forwarded(self) -> None:
        """A spec under ``conversation_middleware:`` reaches the ConversationManager.

        The PR's load-bearing claim is that listing a ConversationMiddleware
        under ``conversation_middleware:`` ends up wrapping the LLM call
        — this verifies the plumbing end-to-end via the manager's
        ``middleware`` list.
        """
        from dataknobs_llm.conversations import HistoryRedactionMiddleware

        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "conversation_middleware": [
                    {
                        "class": (
                            "dataknobs_llm.conversations."
                            "HistoryRedactionMiddleware"
                        ),
                        "params": {
                            "redactions": [
                                {"pattern": r"\bbib:\d+\b", "replacement": "[x]"},
                            ],
                        },
                    },
                ],
            },
        )
        async with bot:
            ctx = BotContext(conversation_id="conv-1", client_id="t1")
            # First chat creates the manager; we then inspect its middleware.
            await bot.chat("hello", ctx)
            manager = bot.get_conversation_manager(ctx.conversation_id)
            assert any(
                isinstance(mw, HistoryRedactionMiddleware)
                for mw in manager.middleware
            ), (
                "HistoryRedactionMiddleware from conversation_middleware: "
                "should be wired onto the ConversationManager"
            )

    @pytest.mark.asyncio
    async def test_conversation_middleware_kwarg_overrides_config(self) -> None:
        """``conversation_middleware=`` kwarg replaces config-driven middleware."""
        from dataknobs_llm.conversations import HistoryRedactionMiddleware

        injected = HistoryRedactionMiddleware(
            redactions=[{"pattern": r"\bsecret:\d+\b", "replacement": "[s]"}],
        )

        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "conversation_middleware": [
                    {
                        "class": (
                            "dataknobs_llm.conversations."
                            "HistoryRedactionMiddleware"
                        ),
                        "params": {
                            "redactions": [
                                {"pattern": r"\bbib:\d+\b", "replacement": "[b]"},
                            ],
                        },
                    },
                ],
            },
            conversation_middleware=[injected],
        )
        async with bot:
            ctx = BotContext(conversation_id="conv-2", client_id="t1")
            await bot.chat("hello", ctx)
            manager = bot.get_conversation_manager(ctx.conversation_id)
            # Only the injected instance — config-driven list is replaced.
            convo_mws = [
                m for m in manager.middleware
                if isinstance(m, HistoryRedactionMiddleware)
            ]
            assert len(convo_mws) == 1
            assert convo_mws[0] is injected

    @pytest.mark.asyncio
    async def test_empty_conversation_middleware_kwarg_disables(self) -> None:
        """Empty kwarg list explicitly disables config-driven middleware."""
        from dataknobs_llm.conversations import HistoryRedactionMiddleware

        bot = await DynaBot.from_config(
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "conversation_middleware": [
                    {
                        "class": (
                            "dataknobs_llm.conversations."
                            "HistoryRedactionMiddleware"
                        ),
                        "params": {
                            "redactions": [
                                {"pattern": r"\bbib:\d+\b", "replacement": "[b]"},
                            ],
                        },
                    },
                ],
            },
            conversation_middleware=[],
        )
        async with bot:
            ctx = BotContext(conversation_id="conv-3", client_id="t1")
            await bot.chat("hello", ctx)
            manager = bot.get_conversation_manager(ctx.conversation_id)
            convo_mws = [
                m for m in manager.middleware
                if isinstance(m, HistoryRedactionMiddleware)
            ]
            assert convo_mws == []

    @pytest.mark.asyncio
    async def test_type_mismatch_rejected_at_config_load(self) -> None:
        """A bot-turn Middleware listed under conversation_middleware: is rejected.

        Without the up-front isinstance check, the misplaced spec would
        crash at first message with ``AttributeError`` (no ``process_request``
        on ``Middleware``). The fix raises ``ConfigurationError`` at
        config-load time with a message that names the misplacement.
        """
        from dataknobs_common.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="conversation_middleware"):
            await DynaBot.from_config(
                {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                    "conversation_middleware": [
                        # A bot-turn Middleware in the LLM-call slot.
                        {
                            "class": (
                                "dataknobs_bots.middleware.logging."
                                "LoggingMiddleware"
                            ),
                        },
                    ],
                },
            )


# ---------------------------------------------------------------------------
# Middleware-spec class-shape resolution (the split _create_*_middleware
# helpers). These call the static helpers directly — no bot construction
# needed — to pin the pre-construction issubclass validation and the
# optional-flag semantics.
# ---------------------------------------------------------------------------


_BOT_MIDDLEWARE_CLASS = "dataknobs_bots.middleware.logging.LoggingMiddleware"
_CONVERSATION_MIDDLEWARE_CLASS = (
    "dataknobs_llm.conversations.HistoryRedactionMiddleware"
)


# Module-level fixture for the no-ctor-side-effects guarantee. Imported by
# dotted path through the resolver, so it must live at module scope. NOT a
# subclass of either ``Middleware`` or ``ConversationMiddleware`` — the spec
# is intentionally misplaced under both fields to exercise the issubclass
# rejection from both helpers. The class-level counter records every ctor
# call so a test can prove the ctor never ran.
class _SideEffectyNonMiddleware:
    """Records every ``__init__`` call; never a middleware.

    A test that asserts the counter is 0 after a misplaced-spec
    rejection proves the rejection happened BEFORE the ctor ran — the
    structural guarantee that lets us safely resolve specs without
    accidentally triggering network reads / file opens / log writes a
    misplaced class's initializer might perform. This counter does not
    rely on incidental properties (no required args, no raising ctor);
    it would still record a ``__init__`` call even if the rejection
    were moved post-instantiation, so a regression cannot be silently
    masked.
    """

    instances_created: int = 0

    def __init__(self, **_kwargs: object) -> None:
        type(self).instances_created += 1

    @classmethod
    def reset(cls) -> None:
        cls.instances_created = 0


_SIDE_EFFECTY_NON_MIDDLEWARE_CLASS = (
    "tests.test_from_config_di._SideEffectyNonMiddleware"
)


class TestMiddlewareSpecResolution:
    """Class-shape validation in _create_bot/_create_conversation_middleware.

    The split helpers reject a spec whose resolved class does not subclass
    the expected base — BEFORE instantiation (``issubclass``), so a
    wrong-shape spec never runs its ctor. Type-mismatch always raises;
    ``optional: true`` only covers transient resolution failures.
    """

    def test_create_conversation_middleware_rejects_bot_middleware_class(
        self,
    ) -> None:
        """A bot-turn Middleware listed as a ConversationMiddleware is rejected."""
        from dataknobs_common.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc:
            DynaBot._create_conversation_middleware(
                {"class": _BOT_MIDDLEWARE_CLASS}
            )

        message = str(exc.value)
        # Names both the resolved class and the expected base.
        assert _BOT_MIDDLEWARE_CLASS in message
        assert "ConversationMiddleware" in message

    def test_create_bot_middleware_rejects_conversation_middleware_class(
        self,
    ) -> None:
        """A ConversationMiddleware listed as a bot-turn Middleware is rejected."""
        from dataknobs_common.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc:
            DynaBot._create_bot_middleware(
                {"class": _CONVERSATION_MIDDLEWARE_CLASS}
            )

        message = str(exc.value)
        assert _CONVERSATION_MIDDLEWARE_CLASS in message
        assert "must subclass" in message

    def test_misplaced_spec_rejection_does_not_instantiate_ctor(self) -> None:
        """Misplaced spec is rejected BEFORE the ctor runs (both helpers).

        Pins the no-side-effects guarantee structurally. The other
        rejection tests use real middleware classes whose ctors happen
        to take no required args (``LoggingMiddleware``) — they would
        still pass if the helper regressed to instantiate-first +
        post-check, because the post-check would still raise. This test
        uses a fixture class that records every ``__init__`` call, so a
        nonzero counter after a rejection proves the ordering broke.
        Tests both helpers in one shot — the issubclass-before-ctor
        ordering is a property of the shared ``_resolve_middleware_from_spec``
        body, not of either wrapper individually.
        """
        from dataknobs_common.exceptions import ConfigurationError

        _SideEffectyNonMiddleware.reset()

        spec = {"class": _SIDE_EFFECTY_NON_MIDDLEWARE_CLASS}

        with pytest.raises(ConfigurationError, match="must subclass"):
            DynaBot._create_bot_middleware(spec)
        with pytest.raises(ConfigurationError, match="must subclass"):
            DynaBot._create_conversation_middleware(spec)

        assert _SideEffectyNonMiddleware.instances_created == 0, (
            "Wrong-shape class was instantiated before the shape check — "
            "the issubclass-before-ctor ordering in "
            "_resolve_middleware_from_spec has regressed."
        )

    def test_optional_true_logs_and_returns_none_on_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """optional: true skips a resolution failure (missing class) for both."""
        spec = {
            "class": "nonexistent.module.NoSuchClass",
            "optional": True,
        }

        with caplog.at_level(logging.WARNING, logger="dataknobs_bots.bot.base"):
            assert DynaBot._create_bot_middleware(spec) is None
            assert DynaBot._create_conversation_middleware(spec) is None

        assert "nonexistent.module.NoSuchClass" in caplog.text
        assert "Skipping optional" in caplog.text

    def test_optional_true_does_not_silence_type_mismatch(self) -> None:
        """A class-shape mismatch always raises, even with optional: true."""
        from dataknobs_common.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="conversation_middleware"):
            DynaBot._create_conversation_middleware(
                {"class": _BOT_MIDDLEWARE_CLASS, "optional": True}
            )

    def test_instantiation_failure_raises_with_instantiate_label(self) -> None:
        """A good-shape class whose ctor raises surfaces the instantiation branch.

        ``HistoryRedactionMiddleware`` is a real ``ConversationMiddleware``
        with a REQUIRED ``redactions`` parameter — omitting ``params``
        passes the issubclass check but raises ``TypeError`` from
        ``__init__``. The resolver's third branch (instantiation
        failure) is then exercised end-to-end, including the exact
        ``"Failed to instantiate {label}"`` error-message format the
        helper emits.

        Without this test the format string is unverified by direct
        coverage — only the first two branches (resolution failure +
        class-shape mismatch) have direct tests.
        """
        from dataknobs_common.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc:
            DynaBot._create_conversation_middleware(
                {"class": _CONVERSATION_MIDDLEWARE_CLASS}
            )

        message = str(exc.value)
        assert "Failed to instantiate" in message
        assert "conversation_middleware" in message
        assert _CONVERSATION_MIDDLEWARE_CLASS in message
        # The chained-from underlying error should be the ctor's TypeError.
        assert isinstance(exc.value.__cause__, TypeError)

    def test_optional_true_silences_instantiation_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """optional: true skips an instantiation failure (ctor raised).

        Same trigger as the previous test (omitted required param), but
        with ``optional: true`` the helper logs a warning and returns
        ``None`` instead of raising. Pins the documented contract that
        ``optional`` covers BOTH resolution and instantiation failures,
        not only the class-import path.
        """
        spec = {
            "class": _CONVERSATION_MIDDLEWARE_CLASS,
            "optional": True,
        }

        with caplog.at_level(logging.WARNING, logger="dataknobs_bots.bot.base"):
            assert DynaBot._create_conversation_middleware(spec) is None

        assert _CONVERSATION_MIDDLEWARE_CLASS in caplog.text
        assert "Skipping optional" in caplog.text

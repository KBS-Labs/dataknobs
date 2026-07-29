"""Tests for the additive ``platform_middleware`` injection channel.

``DynaBot.from_config`` grows two additive pre-built middleware channels
(``platform_middleware`` / ``platform_conversation_middleware``) that are
*appended* to the resolved list rather than *replacing* it (the existing
``middleware=`` / ``conversation_middleware=`` kwargs replace).

These are factory-method / construction-wiring tests — direct
``from_config()`` calls are appropriate here (they assert the composed
instance lists, which ``BotTestHarness``'s post-append would obscure).
Test #10 exercises the mandated ``BotTestHarness`` pass-through (the
blessed consumer path).

Real constructs only — a recording ``Middleware`` / ``ConversationMiddleware``
appended to a shared list, ``EchoProvider``, and memory storage. The
config-block specs reference the module-scope recording classes by dotted
path (the ``params`` dict carries the live sink object directly, since the
config is a Python dict passed straight to ``from_config``).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.bot.turn import TurnState
from dataknobs_bots.middleware.base import Middleware
from dataknobs_llm import EchoProvider
from dataknobs_llm.conversations.middleware import ConversationMiddleware


# ---------------------------------------------------------------------------
# Recording middleware (module-scope so config specs can resolve them by
# dotted path; the live sink travels through the spec's ``params`` dict).
# ---------------------------------------------------------------------------


class RecordingMiddleware(Middleware):
    """Bot-turn middleware that records its tag in ``after_turn``."""

    def __init__(self, tag: str, sink: list[str]) -> None:
        self.tag = tag
        self.sink = sink

    async def after_turn(self, turn: TurnState) -> None:
        self.sink.append(self.tag)


class RecordingConvMiddleware(ConversationMiddleware):
    """LLM-call middleware that records its tag on both onion legs."""

    def __init__(
        self, tag: str, req_sink: list[str], resp_sink: list[str]
    ) -> None:
        self.tag = tag
        self.req_sink = req_sink
        self.resp_sink = resp_sink

    async def process_request(self, messages: Any, state: Any) -> Any:
        self.req_sink.append(self.tag)
        return messages

    async def process_response(self, response: Any, state: Any) -> Any:
        self.resp_sink.append(self.tag)
        return response


_MW_CLASS = "tests.test_platform_middleware_injection.RecordingMiddleware"
_CONV_MW_CLASS = (
    "tests.test_platform_middleware_injection.RecordingConvMiddleware"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _echo(*responses: str) -> EchoProvider:
    """Pre-built EchoProvider with an optional scripted response queue."""
    provider = EchoProvider({"provider": "echo", "model": "test"})
    if responses:
        provider.set_responses(list(responses))
    return provider


def _base_config(**extra: Any) -> dict[str, Any]:
    return {"conversation_storage": {"backend": "memory"}, **extra}


def _ctx(conversation_id: str = "conv-1") -> BotContext:
    return BotContext(conversation_id=conversation_id, client_id="t1")


# ---------------------------------------------------------------------------
# Bot-turn ``platform_middleware`` (additive over config / over replace)
# ---------------------------------------------------------------------------


class TestPlatformMiddlewareBotTurn:
    """``platform_middleware`` appends to the resolved bot-turn list."""

    @pytest.mark.asyncio
    async def test_platform_middleware_additive_over_config(self) -> None:
        """Config middleware + platform middleware both run (neither dropped)."""
        sink: list[str] = []
        probe = RecordingMiddleware("P", sink)

        bot = await DynaBot.from_config(
            _base_config(
                middleware=[
                    {"class": _MW_CLASS, "params": {"tag": "A", "sink": sink}}
                ]
            ),
            llm=_echo("ok"),
            platform_middleware=[probe],
        )
        async with bot:
            # Both middleware present; platform appended last.
            assert len(bot.middleware) == 2
            assert all(
                isinstance(mw, RecordingMiddleware) for mw in bot.middleware
            )
            assert bot.middleware[-1] is probe

            await bot.chat("hi", _ctx())

        # Config middleware fired first, then platform.
        assert sink == ["A", "P"]

    @pytest.mark.asyncio
    async def test_platform_middleware_order_after_config(self) -> None:
        """Platform middleware occupies the last position and runs last."""
        sink: list[str] = []
        config_mw = RecordingMiddleware("A", sink)  # identity check target
        probe = RecordingMiddleware("P", sink)

        # Inject the config middleware as a replace-override standing in for
        # a config-resolved instance, then append the platform probe.
        bot = await DynaBot.from_config(
            _base_config(),
            llm=_echo("ok"),
            middleware=[config_mw],
            platform_middleware=[probe],
        )
        async with bot:
            assert bot.middleware == [config_mw, probe]
            await bot.chat("hi", _ctx())

        assert sink == ["A", "P"]

    @pytest.mark.asyncio
    async def test_platform_middleware_additive_over_replace_override(
        self,
    ) -> None:
        """``middleware=[X]`` still drops config; ``platform=[Y]`` appends."""
        sink: list[str] = []
        x = RecordingMiddleware("X", sink)
        y = RecordingMiddleware("Y", sink)

        bot = await DynaBot.from_config(
            _base_config(
                # A config block that MUST be dropped by the replace-override.
                middleware=[
                    {
                        "class": _MW_CLASS,
                        "params": {"tag": "CONFIG", "sink": sink},
                    }
                ]
            ),
            llm=_echo("ok"),
            middleware=[x],
            platform_middleware=[y],
        )
        async with bot:
            # Replace dropped the config block; platform appended after X.
            assert bot.middleware == [x, y]
            await bot.chat("hi", _ctx())

        assert sink == ["X", "Y"]
        assert "CONFIG" not in sink

    @pytest.mark.asyncio
    async def test_platform_middleware_only_no_config(self) -> None:
        """Platform middleware alone (no config middleware block)."""
        sink: list[str] = []
        probe = RecordingMiddleware("P", sink)

        bot = await DynaBot.from_config(
            _base_config(),
            llm=_echo("ok"),
            platform_middleware=[probe],
        )
        async with bot:
            assert bot.middleware == [probe]
            await bot.chat("hi", _ctx())

        assert sink == ["P"]

    @pytest.mark.asyncio
    async def test_no_platform_params_is_noop(self) -> None:
        """Omitting the platform params is byte-identical to today.

        Regression guard for consumers whose configs declare a
        ``middleware:`` block — the resolved list must be exactly the
        config-built middleware, with nothing appended.
        """
        sink: list[str] = []
        bot = await DynaBot.from_config(
            _base_config(
                middleware=[
                    {"class": _MW_CLASS, "params": {"tag": "A", "sink": sink}}
                ]
            ),
            llm=_echo("ok"),
        )
        async with bot:
            assert len(bot.middleware) == 1
            assert isinstance(bot.middleware[0], RecordingMiddleware)
            await bot.chat("hi", _ctx())

        assert sink == ["A"]

    @pytest.mark.asyncio
    async def test_platform_middleware_additive_on_stream_chat(self) -> None:
        """The streaming path honors the additive channel too.

        Middleware wiring lands on ``self.middleware`` at build time and both
        the buffered (``chat``) and streaming (``stream_chat``) paths iterate
        the same list, so parity holds by construction. This is the
        belt-and-suspenders regression guard against a future path-specific
        divergence — the appended platform probe's ``after_turn`` must fire on
        ``stream_chat`` exactly as it does on ``chat``.
        """
        sink: list[str] = []
        probe = RecordingMiddleware("P", sink)

        bot = await DynaBot.from_config(
            _base_config(
                middleware=[
                    {"class": _MW_CLASS, "params": {"tag": "A", "sink": sink}}
                ]
            ),
            llm=_echo("ok"),
            platform_middleware=[probe],
        )
        async with bot:
            assert bot.middleware[-1] is probe
            async for _ in bot.stream_chat("hi", _ctx()):
                pass

        # Config middleware fired first, then the appended platform probe —
        # identical to the buffered path.
        assert sink == ["A", "P"]


# ---------------------------------------------------------------------------
# LLM-call ``platform_conversation_middleware`` (additive + onion order)
# ---------------------------------------------------------------------------


class TestPlatformConversationMiddleware:
    """``platform_conversation_middleware`` appends to the LLM-call list."""

    @pytest.mark.asyncio
    async def test_additive_over_config(self) -> None:
        """Config conv-middleware + platform conv-middleware both forwarded."""
        req: list[str] = []
        resp: list[str] = []
        probe = RecordingConvMiddleware("P", req, resp)

        bot = await DynaBot.from_config(
            _base_config(
                conversation_middleware=[
                    {
                        "class": _CONV_MW_CLASS,
                        "params": {
                            "tag": "A",
                            "req_sink": req,
                            "resp_sink": resp,
                        },
                    }
                ]
            ),
            llm=_echo("ok"),
            platform_conversation_middleware=[probe],
        )
        async with bot:
            ctx = _ctx("conv-cm-1")
            await bot.chat("hi", ctx)
            manager = bot.get_conversation_manager(ctx.conversation_id)
            convo_mws = [
                m
                for m in manager.middleware
                if isinstance(m, RecordingConvMiddleware)
            ]
            assert len(convo_mws) == 2
            assert convo_mws[-1] is probe

        # Both request hooks fired.
        assert set(req) == {"A", "P"}

    @pytest.mark.asyncio
    async def test_additive_over_replace(self) -> None:
        """``conversation_middleware=[X]`` replaces config; platform appends."""
        req: list[str] = []
        resp: list[str] = []
        x = RecordingConvMiddleware("X", req, resp)
        y = RecordingConvMiddleware("Y", req, resp)

        bot = await DynaBot.from_config(
            _base_config(
                conversation_middleware=[
                    {
                        "class": _CONV_MW_CLASS,
                        "params": {
                            "tag": "CONFIG",
                            "req_sink": req,
                            "resp_sink": resp,
                        },
                    }
                ]
            ),
            llm=_echo("ok"),
            conversation_middleware=[x],
            platform_conversation_middleware=[y],
        )
        async with bot:
            ctx = _ctx("conv-cm-2")
            await bot.chat("hi", ctx)
            manager = bot.get_conversation_manager(ctx.conversation_id)
            convo_mws = [
                m
                for m in manager.middleware
                if isinstance(m, RecordingConvMiddleware)
            ]
            assert convo_mws == [x, y]

        assert "CONFIG" not in req

    @pytest.mark.asyncio
    async def test_onion_order(self) -> None:
        """Appended platform conv-middleware wraps innermost-request /
        outermost-response.

        ``ConversationManager`` runs middleware onion-style:
        ``process_request`` forward, ``process_response`` reversed. So for
        list ``[A, P]`` the request order is ``[A, P]`` (P innermost / last
        before the LLM) and the response order is ``[P, A]`` (P outermost /
        first after the LLM). Locks the documented ordering nuance.
        """
        req: list[str] = []
        resp: list[str] = []
        probe = RecordingConvMiddleware("P", req, resp)

        bot = await DynaBot.from_config(
            _base_config(
                conversation_middleware=[
                    {
                        "class": _CONV_MW_CLASS,
                        "params": {
                            "tag": "A",
                            "req_sink": req,
                            "resp_sink": resp,
                        },
                    }
                ]
            ),
            llm=_echo("ok"),
            platform_conversation_middleware=[probe],
        )
        async with bot:
            await bot.chat("hi", _ctx("conv-cm-3"))

        assert req == ["A", "P"]
        assert resp == ["P", "A"]


# ---------------------------------------------------------------------------
# Pre-built path (out of scope — documents the boundary)
# ---------------------------------------------------------------------------


class TestPrebuiltPathUnaffected:
    """The pre-built ``from_components`` path has no platform channel.

    A pre-built caller already holds a fully-resolved middleware list and
    concatenates its own platform instances before passing them — there is
    no config list to be additive over, so ``from_config``'s platform
    params correctly do not apply here.
    """

    @pytest.mark.asyncio
    async def test_prebuilt_middleware_list_untouched(self) -> None:
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage
        from dataknobs_llm.prompts import AsyncPromptBuilder
        from dataknobs_llm.prompts.implementations import (
            CompositePromptLibrary,
        )

        sink: list[str] = []
        caller_mw = RecordingMiddleware("caller", sink)

        bot = DynaBot.from_components(
            llm=_echo("ok"),
            prompt_builder=AsyncPromptBuilder(CompositePromptLibrary()),
            conversation_storage=DataknobsConversationStorage(
                AsyncMemoryDatabase()
            ),
            middleware=[caller_mw],
        )
        # The caller's list is used verbatim — no additive channel involved.
        assert bot.middleware == [caller_mw]


# ---------------------------------------------------------------------------
# BotTestHarness pass-through (the blessed consumer path — Change D)
# ---------------------------------------------------------------------------


class TestHarnessPlatformPassThrough:
    """``BotTestHarness.create`` routes platform middleware through
    ``from_config``."""

    @pytest.mark.asyncio
    async def test_harness_platform_pass_through(self) -> None:
        from dataknobs_bots.testing import BotTestHarness

        sink: list[str] = []
        probe = RecordingMiddleware("P", sink)

        async with await BotTestHarness.create(
            bot_config=_base_config(
                llm={"provider": "echo", "model": "test"},
                reasoning={"strategy": "simple"},
                middleware=[
                    {"class": _MW_CLASS, "params": {"tag": "A", "sink": sink}}
                ],
            ),
            main_responses=["done"],
            platform_middleware=[probe],
        ) as harness:
            # Config middleware preserved AND platform probe appended.
            assert len(harness.bot.middleware) == 2
            assert harness.bot.middleware[-1] is probe

            await harness.chat("hi")

        assert sink == ["A", "P"]

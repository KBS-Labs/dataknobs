"""Cost tracking keys on the provider family, and says so when it cannot price.

Three defects shipped together because they share one symptom — ``$0.00``
recorded for paid traffic — and one root cause: nothing ever agreed on what
string identifies a provider.

* The lookup key was the provider *class* name, which matches no rate table.
* ``bedrock`` is a family DK ships and the table omitted.
* A miss returned ``0.0`` silently, which is what let both survive.

The end-to-end test here is the one that would have caught the original
defect. Every pre-existing cost test synthesizes a ``TurnState`` and passes
``provider_name=`` in by hand, so none of them ever exercised the code that
computes it.
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_bots.middleware import CostTrackingMiddleware
from dataknobs_bots.testing import BotTestHarness


# ---------------------------------------------------------------------------
# The end-to-end guard
# ---------------------------------------------------------------------------


class TestRealTurnRecordsTheFamily:
    """A real bot turn attributes usage to the family key."""

    @pytest.mark.asyncio
    async def test_turn_records_by_provider_family_not_class_name(self) -> None:
        """Regression guard for the original defect.

        Before the ``provider_name`` contract, a real turn recorded
        ``by_provider["EchoProvider"]`` — a key no rate table can match, so
        every paid provider priced at ``$0.00``.
        """
        tracker = CostTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test-model"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "simple"},
            },
            main_responses=["Hi there!"],
            middleware=[tracker],
        ) as harness:
            await harness.chat("Hello")

        stats = tracker.get_client_stats(harness.context.client_id)
        assert stats is not None
        assert "echo" in stats["by_provider"]
        assert "EchoProvider" not in stats["by_provider"]

    @pytest.mark.asyncio
    async def test_capitalized_config_records_the_same_family(self) -> None:
        """``provider: Echo`` and ``provider: echo`` land in one bucket.

        Canonicalization happens on the provider, so a config author's shift
        key cannot split one family's spend across two rows.
        """
        tracker = CostTrackingMiddleware()

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "Echo", "model": "test-model"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "simple"},
            },
            main_responses=["Hi there!"],
            middleware=[tracker],
        ) as harness:
            await harness.chat("Hello")

        stats = tracker.get_client_stats(harness.context.client_id)
        assert stats is not None
        assert "echo" in stats["by_provider"]
        assert "Echo" not in stats["by_provider"]


class TestLegacyKwargsShimIsNotWidened:
    """``middleware_kwargs`` stays exactly as wide as it was.

    It is the backward-compatible dict handed to legacy ``after_message``
    consumers, and adding a key to a compatibility shim is how compatibility
    shims stop being compatible. Consumers wanting the implementation
    identity read ``turn.provider_impl`` off the ``TurnState`` directly.
    """

    def test_provider_impl_is_not_added_to_the_legacy_kwargs(self) -> None:
        from dataknobs_bots.bot.context import BotContext
        from dataknobs_bots.bot.turn import TurnMode, TurnState

        turn = TurnState(
            mode=TurnMode.CHAT,
            message="Hello",
            context=BotContext(conversation_id="c", client_id="cl"),
        )
        turn.provider_name = "echo"
        turn.provider_impl = "EchoProvider"
        turn.model = "test-model"

        kwargs = turn.middleware_kwargs()

        assert kwargs["provider"] == "echo"
        assert "provider_impl" not in kwargs


# ---------------------------------------------------------------------------
# The rate table
# ---------------------------------------------------------------------------


class TestBedrockFamily:
    """Bedrock resells Anthropic models and needs its own row."""

    def test_prices_a_qualified_bedrock_model_id(self) -> None:
        """Bedrock's fully-qualified IDs resolve through the substring fallback."""
        mw = CostTrackingMiddleware()

        cost = mw._calculate_cost(
            "bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0", 1000, 1000
        )

        assert cost == pytest.approx(0.018)

    def test_is_not_an_alias_of_the_anthropic_table(self) -> None:
        """The two families are spelled separately so pricing can diverge.

        Sharing one dict object would let a ``cost_rates={"bedrock": ...}``
        override silently rewrite Anthropic's rates, since the merge mutates
        in place.
        """
        mw = CostTrackingMiddleware()

        assert mw.cost_rates["bedrock"] is not mw.cost_rates["anthropic"]

    def test_echo_is_priced_at_zero(self) -> None:
        """Echo performs no inference, so zero is correct rather than a placeholder."""
        mw = CostTrackingMiddleware()

        assert mw._calculate_cost("echo", "test-model", 1000, 1000) == 0.0


# ---------------------------------------------------------------------------
# The miss warning
# ---------------------------------------------------------------------------


class TestMissWarning:
    """Traffic priced at zero because nothing matched must say so."""

    def test_warns_on_an_unknown_provider_family(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mw = CostTrackingMiddleware()

        with caplog.at_level(logging.WARNING):
            cost = mw._calculate_cost("acme-llm", "acme-1", 1000, 1000)

        assert cost == 0.0
        assert any(
            "acme-llm" in r.getMessage() and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_warns_on_an_unknown_model_within_a_known_family(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mw = CostTrackingMiddleware()

        with caplog.at_level(logging.WARNING):
            cost = mw._calculate_cost("openai", "some-unreleased-model", 1000, 1000)

        assert cost == 0.0
        assert any(
            "some-unreleased-model" in r.getMessage() for r in caplog.records
        )

    def test_warns_once_per_provider_model_pair(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An undeduplicated per-turn warning is a flood that gets filtered out.

        Filtering it restores exactly the silence the warning exists to break.
        """
        mw = CostTrackingMiddleware()

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                mw._calculate_cost("acme-llm", "acme-1", 1000, 1000)

        misses = [r for r in caplog.records if "acme-llm" in r.getMessage()]
        assert len(misses) == 1

    def test_distinct_pairs_each_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dedup is per ``(provider, model)``, not a global once-only latch."""
        mw = CostTrackingMiddleware()

        with caplog.at_level(logging.WARNING):
            mw._calculate_cost("acme-llm", "acme-1", 1000, 1000)
            mw._calculate_cost("acme-llm", "acme-2", 1000, 1000)

        misses = [r for r in caplog.records if "acme-llm" in r.getMessage()]
        assert len(misses) == 2

    @pytest.mark.parametrize("family", ["ollama", "echo"])
    def test_priced_zero_families_never_warn(
        self, family: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A real zero price is not a miss.

        ``echo`` is the default provider across DK's own suite, so a warning
        here would fire in every harness test that installs cost tracking.
        """
        mw = CostTrackingMiddleware()

        with caplog.at_level(logging.WARNING):
            cost = mw._calculate_cost(family, "some-model", 1000, 1000)

        assert cost == 0.0
        assert [r for r in caplog.records if "no rate entry" in r.getMessage()] == []


# ---------------------------------------------------------------------------
# Rate-table isolation
# ---------------------------------------------------------------------------


class TestCustomRatesDoNotLeak:
    """A per-instance override must not rewrite the class-level defaults."""

    def test_override_does_not_mutate_the_class_default(self) -> None:
        """Regression guard for a shallow ``DEFAULT_RATES.copy()``.

        With a shallow copy, ``self.cost_rates["openai"]`` *is* the class
        attribute's dict, so one instance's ``cost_rates=`` permanently
        rewrote pricing for every instance built afterwards in the process —
        including instances belonging to other tenants.
        """
        original = CostTrackingMiddleware.DEFAULT_RATES["openai"]["gpt-4o"].copy()

        CostTrackingMiddleware(
            cost_rates={"openai": {"gpt-4o": {"input": 99.0, "output": 99.0}}}
        )

        assert CostTrackingMiddleware.DEFAULT_RATES["openai"]["gpt-4o"] == original

    def test_a_later_instance_sees_the_untouched_defaults(self) -> None:
        CostTrackingMiddleware(
            cost_rates={"openai": {"gpt-4o": {"input": 99.0, "output": 99.0}}}
        )
        fresh = CostTrackingMiddleware()

        assert fresh.cost_rates["openai"]["gpt-4o"]["input"] == pytest.approx(0.0025)

    def test_instances_do_not_share_nested_rate_dicts(self) -> None:
        first = CostTrackingMiddleware()
        second = CostTrackingMiddleware()

        assert first.cost_rates["openai"] is not second.cost_rates["openai"]

    def test_a_custom_family_is_still_accepted(self) -> None:
        """Isolation must not break the documented way to price a new family."""
        mw = CostTrackingMiddleware(
            cost_rates={"huggingface": {"input": 0.001, "output": 0.002}}
        )

        assert mw._calculate_cost("huggingface", "any-model", 1000, 1000) == (
            pytest.approx(0.003)
        )

    def test_instances_built_from_one_shared_dict_do_not_alias(self) -> None:
        """The caller's dict is the other half of the isolation problem.

        Deep-copying the *defaults* stops one instance rewriting the class
        attribute, but the merge then inserts the **caller's** nested dicts by
        reference. The realistic shape is a module-level rate constant reused
        across per-tenant middleware instances, which puts every tenant back
        on one shared dict — the same cross-tenant corruption, arriving
        through the parameter instead of through the class.
        """
        shared = {"openai": {"gpt-4o": {"input": 0.0025, "output": 0.01}}}

        first = CostTrackingMiddleware(cost_rates=shared)
        second = CostTrackingMiddleware(cost_rates=shared)

        assert (
            first.cost_rates["openai"]["gpt-4o"]
            is not second.cost_rates["openai"]["gpt-4o"]
        )

    def test_the_callers_own_dict_is_never_mutated(self) -> None:
        """A config constant handed in must come back out unchanged."""
        shared = {"openai": {"gpt-4o": {"input": 0.0025, "output": 0.01}}}

        mw = CostTrackingMiddleware(cost_rates=shared)
        mw.cost_rates["openai"]["gpt-4o"]["input"] = 99.0

        assert shared["openai"]["gpt-4o"]["input"] == pytest.approx(0.0025)

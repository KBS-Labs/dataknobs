"""Cost tracking prices from the provider's catalog, and its fallback is sane.

Repairing the provider-family lookup made a previously-**dead** code path live:
while the lookup key was a class name, ``_calculate_cost`` returned ``0.0``
before ever reaching the rate table, so nothing in that table had been
exercised in production. Two defects were waiting there.

* The table is a hand-maintained duplicate of pricing ``dataknobs-llm``
  already resolves from dated catalogs, and it has drifted — ``o1-mini`` is
  priced at nearly 3x the real rate, on an exact match.
* Its model lookup falls back to a first-match-wins substring scan over dict
  insertion order, so ``gpt-4o-mini-2024-07-18`` — the id OpenAI actually
  returns — resolves to ``gpt-4o`` and is billed at 16x.

The second is the more dangerous of the two, because ``$0.00`` is visibly
broken and ``$25.00`` against a true ``$1.50`` is not. Neither is caught by
the miss warning: a *wrong* match is still a match.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.bot.turn import TurnMode, TurnState
from dataknobs_bots.middleware import CostTrackingMiddleware
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm import LLMConfig, LLMResponse
from dataknobs_llm.llm.model_profile import ModelPricing
from dataknobs_llm.llm.providers.openai import OpenAIProvider


def _openai(model: str) -> OpenAIProvider:
    return OpenAIProvider(LLMConfig(provider="openai", model=model, api_key="x"))


# ---------------------------------------------------------------------------
# The provider's own pricing is authoritative
# ---------------------------------------------------------------------------


class TestPricingComesFromTheProvider:
    """``dataknobs-llm`` already owns pricing; the middleware must consume it.

    ``LLMProvider.get_pricing`` resolves per-model USD rates through the
    layered model-profile substrate — dated catalogs, config-overridable,
    per-provider isolated. A second hand-maintained table in the middleware is
    a duplicate that can only drift, and has.
    """

    def test_turn_captures_the_providers_pricing(self) -> None:
        """The provider is in hand at populate time; the pricing must be kept.

        ``TurnState`` discards the provider object after reading a name, so a
        rate not captured here is unrecoverable downstream — which is why the
        middleware grew its own table in the first place.
        """
        turn = TurnState(
            mode=TurnMode.CHAT,
            message="hi",
            context=None,  # type: ignore[arg-type]
        )
        turn.populate_from_response(
            LLMResponse(content="ok", model="gpt-4o-mini", usage={}),
            _openai("gpt-4o-mini"),
        )

        assert turn.pricing == ModelPricing(
            input_per_mtok=0.15,
            output_per_mtok=0.6,
            cached_input_per_mtok=0.075,
        )

    def test_catalog_pricing_beats_a_stale_table_entry(self) -> None:
        """``o1-mini`` is the exact-match case, so no substring scan is involved.

        The middleware table says ``$3.00``/Mtok in, ``$12.00`` out. The
        catalog says ``$1.10`` / ``$4.40``. Both are exact matches on the
        model id — this is pure duplication drift, and the provider's answer
        is the one with a verification date on it.
        """
        mw = CostTrackingMiddleware()
        pricing = ModelPricing(input_per_mtok=1.1, output_per_mtok=4.4)

        cost = mw._calculate_cost(
            "openai", "o1-mini", 1_000_000, 1_000_000, pricing=pricing
        )

        assert cost == pytest.approx(5.5)

    def test_a_model_absent_from_the_table_is_still_priced(self) -> None:
        """``gpt-5`` is the current flagship and the table has never had it.

        Without profile pricing this warns and returns ``$0.00`` — the
        original defect, unfixed, for the models most likely to be in use.
        """
        mw = CostTrackingMiddleware()
        pricing = ModelPricing(input_per_mtok=1.25, output_per_mtok=10.0)

        cost = mw._calculate_cost(
            "openai", "gpt-5", 1_000_000, 0, pricing=pricing
        )

        assert cost == pytest.approx(1.25)

    def test_an_explicit_consumer_rate_still_wins(self) -> None:
        """``cost_rates=`` is a negotiated price, not a guess to be overridden.

        Precedence is explicit-override, then the provider's catalog, then the
        built-in table. A consumer who supplied a rate has stated what they
        are billed; nothing derived may outrank it.
        """
        mw = CostTrackingMiddleware(
            cost_rates={"openai": {"gpt-4o-mini": {"input": 0.001, "output": 0.0}}}
        )
        catalog = ModelPricing(input_per_mtok=0.15, output_per_mtok=0.6)

        cost = mw._calculate_cost(
            "openai", "gpt-4o-mini", 1000, 0, pricing=catalog
        )

        assert cost == pytest.approx(0.001)

    def test_no_pricing_and_no_table_entry_still_warns(self) -> None:
        """Routing through the provider must not silence the miss warning."""
        mw = CostTrackingMiddleware()

        cost = mw._calculate_cost("acme", "some-model", 1000, 1000, pricing=None)

        assert cost == 0.0
        assert ("acme", "some-model") in mw._warned_misses

    @pytest.mark.asyncio
    async def test_end_to_end_turn_prices_from_the_catalog(self) -> None:
        """The whole path: real turn, real provider, catalog-sourced rate.

        ``EchoProvider`` has no profile pricing, so this pins the *other*
        half — a family the catalog does not price must fall through to the
        table rather than crash or silently zero out a priced family.
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
        assert stats["by_provider"]["echo"]["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# The fallback table's substring scan
# ---------------------------------------------------------------------------


class TestSubstringFallbackPicksTheClosestModel:
    """The table remains the fallback, so its lookup still has to be right.

    Reached whenever a provider sources no profile pricing — a consumer's
    out-of-tree provider, a self-hosted gateway, a family DK does not ship a
    catalog for. First-match-wins over insertion order makes the answer depend
    on how the table happens to be written, which is not a pricing rule.
    """

    @pytest.fixture
    def mw(self) -> CostTrackingMiddleware:
        return CostTrackingMiddleware(
            cost_rates={
                "acme": {
                    "gpt-4o": {"input": 2.5, "output": 0.0},
                    "gpt-4o-mini": {"input": 0.15, "output": 0.0},
                    "o1": {"input": 15.0, "output": 0.0},
                    "o1-mini": {"input": 1.1, "output": 0.0},
                }
            }
        )

    @pytest.mark.parametrize(
        ("model", "expected_per_1k"),
        [
            ("gpt-4o-mini-2024-07-18", 0.15),
            ("gpt-4o-2024-08-06", 2.5),
            ("o1-mini-2024-09-12", 1.1),
            ("o1-preview", 15.0),
        ],
    )
    def test_dated_model_id_resolves_to_the_longest_matching_key(
        self, mw: CostTrackingMiddleware, model: str, expected_per_1k: float
    ) -> None:
        """OpenAI returns the dated snapshot id, not the alias you requested.

        ``gpt-4o`` is a prefix of ``gpt-4o-mini-2024-07-18``, so a scan that
        stops at the first hit bills the mini model at the full model's rate.
        The longest matching key is the most specific one, and specificity is
        the only defensible tie-break here.
        """
        cost = mw._calculate_cost("acme", model, 1000, 0)

        assert cost == pytest.approx(expected_per_1k)

    def test_a_shorter_id_does_not_match_a_longer_table_key(
        self, mw: CostTrackingMiddleware
    ) -> None:
        """The reverse direction is not a containment rule at all.

        Matching ``model in model_key`` lets a request for ``gpt-4`` be priced
        as ``gpt-4o`` — a different model — purely because one id is a
        substring of the other. Absent a real entry this must miss and warn,
        not guess.
        """
        cost = mw._calculate_cost("acme", "gpt-4", 1000, 0)

        assert cost == 0.0
        assert ("acme", "gpt-4") in mw._warned_misses

    def test_an_exact_match_is_unaffected(
        self, mw: CostTrackingMiddleware
    ) -> None:
        assert mw._calculate_cost("acme", "gpt-4o-mini", 1000, 0) == (
            pytest.approx(0.15)
        )


# ---------------------------------------------------------------------------
# Malformed table entries
# ---------------------------------------------------------------------------


class TestMalformedRateEntries:
    """A bad rate must warn and price zero, never raise mid-turn.

    Cost tracking is observability; a typo in a rate constant must not take
    down the conversation it is measuring.
    """

    def test_a_scalar_under_a_family_warns(self) -> None:
        mw = CostTrackingMiddleware(cost_rates={"acme": "free"})

        assert mw._calculate_cost("acme", "any", 1000, 1000) == 0.0
        assert ("acme", "any") in mw._warned_misses

    def test_a_scalar_under_a_model_warns_instead_of_raising(self) -> None:
        """``rates.get(...)`` on a string is an ``AttributeError`` mid-turn."""
        mw = CostTrackingMiddleware(cost_rates={"acme": {"m": "free"}})

        assert mw._calculate_cost("acme", "m", 1000, 1000) == 0.0
        assert ("acme", "m") in mw._warned_misses

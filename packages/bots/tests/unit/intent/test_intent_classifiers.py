"""Tests for the pluggable IntentClassifier protocol.

Covers each built-in classifier in isolation plus the composite +
negation-filter composition that reproduces the v2 default
"rule-first + optional LLM fallback" behaviour.

The pluggable shape lets consumers swap the classification strategy
(I18N tokenizer, embedding-similarity, LLM-first, ensemble vote)
without forking — directly answers the user feedback question
"could we swap this for an I18N-aware version or an LLM-only
classifier."
"""
from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.intent import (
    DEFAULT_VOCABULARY,
    CompositeIntentClassifier,
    IntentClassifier,
    IntentMatchResult,
    IntentSpec,
    KeywordIntentClassifier,
    LLMIntentClassifier,
    NegationFilter,
    intent_classifier_backends,
)

# ---------------------------------------------------------------------------
# KeywordIntentClassifier — vocabulary + tokenizer injection contract
# ---------------------------------------------------------------------------


class TestKeywordIntentClassifier:
    @pytest.mark.asyncio
    async def test_default_tokenizer_word_boundary_match(self) -> None:
        clf = KeywordIntentClassifier(vocabulary={"accept": frozenset({"yes"})})
        result = await clf.classify(
            "yes please", [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is not None
        assert result.intent.name == "accept"

    @pytest.mark.asyncio
    async def test_default_tokenizer_does_not_match_substring(self) -> None:
        """Pre-fix substring foot-gun: ``"yesterday"`` matched ``"yes"``.

        Lives at the classifier layer in v3 — fix benefits BOTH the
        wizard's ``intent_detection:`` block and the public API.
        """
        clf = KeywordIntentClassifier(vocabulary={"accept": frozenset({"yes"})})
        result = await clf.classify(
            "yesterday I was reading",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is None

    @pytest.mark.asyncio
    async def test_per_intent_keywords_replace_default_vocabulary(self) -> None:
        clf = KeywordIntentClassifier(vocabulary=DEFAULT_VOCABULARY)
        result = await clf.classify(
            "affirm",
            [IntentSpec(name="accept", target="t", keywords=("affirm",))],
        )
        assert result.intent is not None
        assert result.intent.name == "accept"

        # 'yes' should NOT match — per-intent keywords REPLACE default
        result_yes = await clf.classify(
            "yes",
            [IntentSpec(name="accept", target="t", keywords=("affirm",))],
        )
        assert result_yes.intent is None

    @pytest.mark.asyncio
    async def test_custom_tokenizer_injection(self) -> None:
        """Consumer can swap the tokenizer for an I18N / fuzzy / N-gram
        variant — the central use case justifying the protocol.
        """
        # Toy tokenizer: strip "extra-" prefixes before word matching
        def stripped_tokenizer(keyword: str, message: str) -> bool:
            return keyword in message.lower().replace("extra-", "")

        clf = KeywordIntentClassifier(
            vocabulary={"accept": frozenset({"yes"})},
            tokenizer=stripped_tokenizer,
        )
        result = await clf.classify(
            "extra-yes please",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is not None
        assert result.intent.name == "accept"

    @pytest.mark.asyncio
    async def test_first_matching_intent_wins(self) -> None:
        clf = KeywordIntentClassifier(vocabulary=DEFAULT_VOCABULARY)
        result = await clf.classify(
            "yes I decline",
            [
                IntentSpec(name="accept", target="t"),
                IntentSpec(name="decline", target="t"),
            ],
        )
        assert result.intent is not None
        assert result.intent.name == "accept"


# ---------------------------------------------------------------------------
# LLMIntentClassifier — prompt template + extract field contract
# ---------------------------------------------------------------------------


class TestLLMIntentClassifier:
    @pytest.mark.asyncio
    async def test_default_prompt_classifies_via_llm(self) -> None:
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(
            [text_response('{"intent": "accept", "extracted": null}')],
        )
        clf = LLMIntentClassifier(llm=provider)
        result = await clf.classify(
            "ambiguous reply",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is not None
        assert result.intent.name == "accept"
        assert result.rule_based is False

    @pytest.mark.asyncio
    async def test_custom_prompt_template_used(self) -> None:
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(
            [text_response('{"intent": "accept", "extracted": null}')],
        )
        custom = (
            "DOMAIN-CLASSIFY: '{message}' into {intent_list}. "
            'Output JSON {{"intent": <name|null>, "extracted": null}}.'
        )
        clf = LLMIntentClassifier(llm=provider, prompt_template=custom)
        await clf.classify(
            "ambiguous reply",
            [IntentSpec(name="accept", target="t")],
        )
        last_msg = provider.get_last_call()["messages"][-1].content
        assert "DOMAIN-CLASSIFY" in last_msg

    @pytest.mark.asyncio
    async def test_extract_field_payload_captured(self) -> None:
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            text_response('{"intent": "alternative", "extracted": "AIAM"}'),
        ])
        clf = LLMIntentClassifier(llm=provider)
        result = await clf.classify(
            "use AIAM instead",
            [IntentSpec(
                name="alternative", target="t", extract="framework_name",
            )],
        )
        assert result.intent is not None
        assert result.intent.name == "alternative"
        assert result.extracted == "AIAM"


# ---------------------------------------------------------------------------
# CompositeIntentClassifier — chains backends; reproduces v2 default
# ---------------------------------------------------------------------------


class TestCompositeIntentClassifier:
    @pytest.mark.asyncio
    async def test_first_match_strategy_keyword_then_llm(self) -> None:
        """The v2 default — rule-first, optional LLM fallback —
        reproduced as an explicit composition in v3.
        """
        from dataknobs_llm import EchoProvider

        keyword = KeywordIntentClassifier(vocabulary=DEFAULT_VOCABULARY)
        llm_provider = EchoProvider(  # never called — keyword wins
            {"provider": "echo", "model": "test"},
        )
        llm = LLMIntentClassifier(llm=llm_provider)
        composite = CompositeIntentClassifier(
            classifiers=[keyword, llm],
            strategy="first_match",
        )

        result = await composite.classify(
            "yes",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is not None
        assert result.intent.name == "accept"
        assert result.rule_based is True
        assert llm_provider.call_count == 0   # short-circuited

    @pytest.mark.asyncio
    async def test_first_match_strategy_falls_through_to_llm(self) -> None:
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(
            [text_response('{"intent": "accept", "extracted": null}')],
        )
        keyword = KeywordIntentClassifier(
            vocabulary={"accept": frozenset({"yes"})},  # narrow vocab — won't match
        )
        llm = LLMIntentClassifier(llm=provider)
        composite = CompositeIntentClassifier(
            classifiers=[keyword, llm], strategy="first_match",
        )

        result = await composite.classify(
            "I would like to proceed",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is not None
        assert result.intent.name == "accept"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_llm_first_strategy_is_just_reordering(self) -> None:
        """A consumer wanting LLM-first / keyword-fallback just
        reorders the composite — no new code, no fork.
        """
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses(
            [text_response('{"intent": "accept", "extracted": null}')],
        )
        keyword = KeywordIntentClassifier(vocabulary=DEFAULT_VOCABULARY)
        llm = LLMIntentClassifier(llm=provider)
        composite = CompositeIntentClassifier(
            classifiers=[llm, keyword],   # LLM first
            strategy="first_match",
        )
        await composite.classify("yes", [IntentSpec(name="accept", target="t")])
        # LLM called even though keyword would have matched
        assert provider.call_count == 1


# ---------------------------------------------------------------------------
# NegationFilter — composes existing has_negation helper
# ---------------------------------------------------------------------------


class TestNegationFilter:
    @pytest.mark.asyncio
    async def test_negation_suppresses_accept_match(self) -> None:
        """``"no, I don't want to accept"`` matches the accept vocab
        but the negation context flips the result to no-match.

        Closes the foot-gun for any IntentClassifier consumer that
        opts in via ``NegationFilter(...)`` or the wizard's
        ``intent_detection: {negation_filter: true}`` flag.
        """
        inner = KeywordIntentClassifier(vocabulary=DEFAULT_VOCABULARY)
        clf = NegationFilter(inner)
        result = await clf.classify(
            "no, I don't want to accept that",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is None or result.intent.name != "accept"

    @pytest.mark.asyncio
    async def test_no_negation_passes_through_unchanged(self) -> None:
        inner = KeywordIntentClassifier(vocabulary=DEFAULT_VOCABULARY)
        clf = NegationFilter(inner)
        result = await clf.classify(
            "yes please proceed",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is not None
        assert result.intent.name == "accept"

    @pytest.mark.asyncio
    async def test_custom_negation_keywords(self) -> None:
        """Consumer overrides the negation set for their domain."""
        inner = KeywordIntentClassifier(
            vocabulary={"accept": frozenset({"engage"})},
        )
        clf = NegationFilter(inner, negation_keywords=frozenset({"halt"}))
        result = await clf.classify(
            "halt — do not engage",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is None


# ---------------------------------------------------------------------------
# Classifier-backend registry — consumer-extensible
# ---------------------------------------------------------------------------


class TestClassifierBackendRegistry:
    def test_built_in_backends_registered(self) -> None:
        for name in ("keyword", "llm", "composite"):
            assert intent_classifier_backends.has(name), name

    def test_consumer_can_register_own_backend(self) -> None:
        """Pins the contract for consumer-defined backends —
        embedding-similarity classifier, fuzzy-match, locale-specific
        keyword variants, etc.
        """
        class _FuzzyMatch(IntentClassifier):
            async def classify(
                self, message: str, intents: Any, **_: Any,
            ) -> IntentMatchResult:
                # toy: matches if any intent name is a substring of message
                for spec in intents:
                    if spec.name in message.lower():
                        return IntentMatchResult(
                            intent=spec, extracted=None,
                            rule_based=True, raw_reply=message,
                        )
                return IntentMatchResult(
                    intent=None, extracted=None, rule_based=False,
                    raw_reply=message,
                )

        def _factory(config: dict[str, Any]) -> IntentClassifier:
            return _FuzzyMatch()

        intent_classifier_backends.register(
            "fuzzy_match", _factory, allow_overwrite=True,
        )
        try:
            clf = intent_classifier_backends.get("fuzzy_match")({})
            assert isinstance(clf, _FuzzyMatch)
        finally:
            intent_classifier_backends.unregister("fuzzy_match")

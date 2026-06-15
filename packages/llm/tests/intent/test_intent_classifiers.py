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

from dataknobs_llm.intent import (
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

    @pytest.mark.asyncio
    async def test_provider_error_returns_no_match(self) -> None:
        """Provider exception is absorbed; caller sees ``intent=None``.

        An LLM classifier is typically one signal among many
        (composite chain, optional fallback). A provider outage
        should NOT crash the wizard turn — the classifier returns
        no-match and logs a warning to make the absorption auditable.
        """
        from dataknobs_llm import EchoProvider

        class _RaisingProvider(EchoProvider):
            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("simulated provider outage")

        provider = _RaisingProvider({"provider": "echo", "model": "test"})
        clf = LLMIntentClassifier(llm=provider)
        result = await clf.classify(
            "yes please",
            [IntentSpec(name="accept", target="t")],
        )
        assert result.intent is None
        assert result.extracted is None
        assert result.raw_reply == "yes please"

    @pytest.mark.asyncio
    async def test_provider_cancelled_error_is_not_swallowed(self) -> None:
        """Cancellation must propagate. asyncio.CancelledError is a
        cooperative signal — swallowing it breaks task cancellation.
        """
        import asyncio

        from dataknobs_llm import EchoProvider

        class _CancellingProvider(EchoProvider):
            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                raise asyncio.CancelledError()

        provider = _CancellingProvider({"provider": "echo", "model": "test"})
        clf = LLMIntentClassifier(llm=provider)
        with pytest.raises(asyncio.CancelledError):
            await clf.classify(
                "yes",
                [IntentSpec(name="accept", target="t")],
            )

    @pytest.mark.asyncio
    async def test_non_string_extracted_is_normalized(self) -> None:
        """Models occasionally return non-string ``extracted`` payloads
        (list, int, bool). The documented shape is ``str | None``;
        non-string values are coerced or dropped so the wizard never
        writes a list/dict into a schema-declared ``string`` field.
        """
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        cases: list[tuple[str, str | None]] = [
            ('{"intent": "alt", "extracted": "AIAM"}', "AIAM"),
            ('{"intent": "alt", "extracted": ["AIAM"]}', "AIAM"),
            ('{"intent": "alt", "extracted": 42}', "42"),
            ('{"intent": "alt", "extracted": true}', "True"),
            ('{"intent": "alt", "extracted": null}', None),
            ('{"intent": "alt", "extracted": ""}', None),
            (
                '{"intent": "alt", "extracted": ["a", "b"]}', None,
            ),
            ('{"intent": "alt", "extracted": {"k": 1}}', None),
        ]
        for body, expected in cases:
            provider = EchoProvider({"provider": "echo", "model": "test"})
            provider.set_responses([text_response(body)])
            clf = LLMIntentClassifier(llm=provider)
            result = await clf.classify(
                "use AIAM instead",
                [IntentSpec(
                    name="alt", target="t", extract="framework_name",
                )],
            )
            assert result.intent is not None and result.intent.name == "alt"
            assert result.extracted == expected, (
                f"extracted normalization mismatch for {body!r}: "
                f"got {result.extracted!r}, expected {expected!r}"
            )

    @pytest.mark.asyncio
    async def test_prompt_intent_list_preserves_caller_order(self) -> None:
        """Intent-list ordering in the rendered prompt follows caller
        order — not set iteration order. Stabilizes prompt-cache hit
        rates and LLM-eval reproducibility across runs.
        """
        from dataknobs_llm import EchoProvider
        from dataknobs_llm.testing import text_response

        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([
            text_response('{"intent": null, "extracted": null}'),
        ])
        clf = LLMIntentClassifier(llm=provider)
        await clf.classify(
            "hello",
            [
                IntentSpec(name="alpha", target="a"),
                IntentSpec(name="bravo", target="b"),
                IntentSpec(name="charlie", target="c"),
            ],
        )
        call = provider.get_last_call()
        assert call is not None
        rendered = call["messages"][0].content
        # The names should appear in caller order in the prompt.
        i_alpha = rendered.index("alpha")
        i_bravo = rendered.index("bravo")
        i_charlie = rendered.index("charlie")
        assert i_alpha < i_bravo < i_charlie


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
            # ``PluginRegistry.get_factory`` returns the stored factory
            # callable; calling it with ``({})`` mirrors the pre-
            # consolidation ``Registry.get(name)(config)`` shape and
            # confirms the registration round-trips.
            factory = intent_classifier_backends.get_factory("fuzzy_match")
            assert factory is not None
            clf = factory({})
            assert isinstance(clf, _FuzzyMatch)
        finally:
            intent_classifier_backends.unregister("fuzzy_match")

    def test_create_intent_classifier_resolves_registered_name(self) -> None:
        from dataknobs_llm.intent import create_intent_classifier

        clf = create_intent_classifier("keyword", {})
        assert isinstance(clf, KeywordIntentClassifier)

    def test_create_intent_classifier_unknown_name_raises_with_list(
        self,
    ) -> None:
        from dataknobs_llm.intent import create_intent_classifier

        with pytest.raises(ValueError) as exc_info:
            create_intent_classifier("does_not_exist", {})
        msg = str(exc_info.value)
        assert "does_not_exist" in msg
        # Lists every registered backend for self-diagnosis
        assert "keyword" in msg
        assert "llm" in msg
        assert "composite" in msg

    def test_create_intent_classifier_none_config_defaults_empty(
        self,
    ) -> None:
        from dataknobs_llm.intent import create_intent_classifier

        clf = create_intent_classifier("keyword", None)
        assert isinstance(clf, KeywordIntentClassifier)

    def test_composite_missing_classifier_field_raises(self) -> None:
        """Typo in ``classifier:`` (e.g. ``classifer:``) must raise
        with the offending spec, not silently produce an empty
        composite that then fails with a misleading "requires at
        least one inner classifier" message.
        """
        from dataknobs_llm.intent import create_intent_classifier

        with pytest.raises(ValueError) as exc_info:
            create_intent_classifier("composite", {
                "classifiers": [
                    {"classifer": "keyword"},  # typo
                ],
            })
        assert "missing required 'classifier'" in str(exc_info.value)
        assert "classifer" in str(exc_info.value)

    def test_composite_non_mapping_child_raises(self) -> None:
        from dataknobs_llm.intent import create_intent_classifier

        with pytest.raises(ValueError) as exc_info:
            create_intent_classifier("composite", {
                "classifiers": ["keyword"],
            })
        assert "must be a mapping" in str(exc_info.value)

    def test_composite_unknown_child_name_raises(self) -> None:
        from dataknobs_llm.intent import create_intent_classifier

        with pytest.raises(ValueError) as exc_info:
            create_intent_classifier("composite", {
                "classifiers": [
                    {"classifier": "does_not_exist"},
                ],
            })
        assert "does_not_exist" in str(exc_info.value)

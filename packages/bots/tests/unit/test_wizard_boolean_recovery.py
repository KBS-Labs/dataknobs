"""Tests for boolean extraction recovery.

When LLM extraction fails to produce a value for a boolean field,
boolean recovery scans the user's message for affirmative/negative
signal words and fills the field deterministically.

Unit tests exercise ``detect_boolean_signal()`` directly.
Integration tests use ``BotTestHarness`` for the full pipeline.
"""

import pytest

from dataknobs_bots.reasoning.wizard import (
    RECOVERY_BOOLEAN,
    RECOVERY_DERIVATION,
    _DEFAULT_AFFIRMATIVE_PHRASES,
    _DEFAULT_AFFIRMATIVE_SIGNALS,
    _DEFAULT_NEGATIVE_PHRASES,
    _DEFAULT_NEGATIVE_SIGNALS,
)
from dataknobs_bots.reasoning.wizard_grounding import detect_boolean_signal
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_llm.extraction.grounding import (
    DEFAULT_NEGATION_KEYWORDS,
    _word_in_text,
    has_negation,
)
from dataknobs_llm.intent import IntentSpec, KeywordIntentClassifier
from dataknobs_llm.testing import text_response


# ---------------------------------------------------------------------------
# Unit tests: detect_boolean_signal
# ---------------------------------------------------------------------------


class TestDetectBooleanSignal:
    """Test the core signal detection function."""

    def _detect(
        self,
        msg: str,
        *,
        aff: frozenset[str] | None = None,
        aff_phrases: tuple[str, ...] | None = None,
        neg: frozenset[str] | None = None,
        neg_phrases: tuple[str, ...] | None = None,
    ) -> bool | None:
        return detect_boolean_signal(
            msg.lower(),
            affirmative_signals=aff or _DEFAULT_AFFIRMATIVE_SIGNALS,
            affirmative_phrases=aff_phrases or _DEFAULT_AFFIRMATIVE_PHRASES,
            negative_signals=neg or _DEFAULT_NEGATIVE_SIGNALS,
            negative_phrases=neg_phrases or _DEFAULT_NEGATIVE_PHRASES,
        )

    def test_clear_affirmative_yes(self) -> None:
        assert self._detect("Yes, save it!") is True

    def test_clear_affirmative_confirm(self) -> None:
        assert self._detect("I confirm.") is True

    def test_clear_affirmative_looks_good(self) -> None:
        assert self._detect("Looks good, let's go!") is True

    def test_clear_affirmative_sounds_good(self) -> None:
        assert self._detect("Sounds good to me.") is True

    def test_clear_affirmative_go_ahead(self) -> None:
        assert self._detect("Go ahead and save.") is True

    def test_clear_negative_no(self) -> None:
        assert self._detect("No, don't do that.") is False

    def test_clear_negative_wait(self) -> None:
        assert self._detect("Wait, I need to change something.") is False

    def test_clear_negative_cancel(self) -> None:
        assert self._detect("Cancel this.") is False

    def test_negative_phrase_not_yet(self) -> None:
        assert self._detect("Not yet, I'm not ready.") is False

    def test_negative_phrase_hold_on(self) -> None:
        assert self._detect("Hold on, let me check.") is False

    def test_negative_phrase_start_over(self) -> None:
        assert self._detect("Let's start over.") is False

    def test_no_signals_returns_none(self) -> None:
        assert self._detect("I like pizza.") is None

    def test_unrelated_message_returns_none(self) -> None:
        assert self._detect("What time is it?") is None

    def test_both_single_word_signals_returns_none(self) -> None:
        """Both affirmative and negative single words → ambiguous."""
        # "confirm" (affirmative) + "no" (negative) — both present
        assert self._detect("No, I don't confirm.") is None

    def test_negative_phrase_overrides_affirmative_word(self) -> None:
        """Negative phrase + affirmative word → negative phrase wins."""
        assert self._detect("Not yet, but ok maybe.") is False

    def test_affirmative_phrase_overrides_negative_word(self) -> None:
        """Affirmative phrase beats negative single word."""
        result = self._detect("Looks good, no changes needed.")
        # "looks good" (affirmative phrase) + "no" (negative word).
        # Affirmative phrase is stronger than single negative word.
        assert result is True

    def test_idiomatic_no_with_affirmative(self) -> None:
        """'sure, no worries' — 'no' is idiomatic, not a rejection."""
        assert self._detect("Sure, no worries!") is None

    def test_dont_save_phrase_matches_negative(self) -> None:
        """'don't save' is a negative phrase — overrides 'save' word."""
        assert self._detect("Don't save it.") is False

    def test_not_correct_known_limitation(self) -> None:
        """'that's not correct' — 'correct' is affirmative, 'not' is
        not a negative signal or negation keyword.  Returns True.
        This is a known limitation of single-word signal matching."""
        assert self._detect("That's not correct.") is True

    def test_custom_affirmative_signals(self) -> None:
        custom = frozenset({"proceed", "ship"})
        assert self._detect("Ship it!", aff=custom) is True

    def test_custom_negative_signals(self) -> None:
        custom = frozenset({"abort", "reject"})
        assert self._detect("Abort!", neg=custom) is False

    def test_custom_affirmative_phrases(self) -> None:
        custom = ("all systems go",)
        assert self._detect(
            "All systems go!",
            aff_phrases=custom,
        ) is True

    def test_custom_negative_phrases(self) -> None:
        custom = ("take it back",)
        assert self._detect(
            "Take it back!",
            neg_phrases=custom,
        ) is False

    def test_empty_message_returns_none(self) -> None:
        assert self._detect("") is None

    def test_yep_affirmative(self) -> None:
        assert self._detect("Yep!") is True

    def test_nope_negative(self) -> None:
        assert self._detect("Nope.") is False


# ---------------------------------------------------------------------------
# Drift guard: wrapper and classifier path agree on the canonical spec.
#
# ``detect_boolean_signal`` dispatches through
# ``KeywordIntentClassifier(phrase_priority=True)`` wrapped in
# ``NegationFilter(flip_when_alone={"affirmative": "negative"})``.
# This class re-derives the wrapper's verdict from the classifier
# primitives directly and asserts equivalence over the same set of
# canonical messages exercised by ``TestDetectBooleanSignal``. A
# future change to either path that produces a different ``bool |
# None`` for any case fails this parity check before the consumer
# sees it.
# ---------------------------------------------------------------------------


_PARITY_AFF_NEG_SPECS: tuple[IntentSpec, ...] = (
    IntentSpec(name="affirmative", target="_boolean_true"),
    IntentSpec(name="negative", target="_boolean_false"),
)


_PARITY_DEFAULT_CASES: tuple[str, ...] = (
    "Yes, save it!",
    "I confirm.",
    "Looks good, let's go!",
    "Sounds good to me.",
    "Go ahead and save.",
    "No, don't do that.",
    "Wait, I need to change something.",
    "Cancel this.",
    "Not yet, I'm not ready.",
    "Hold on, let me check.",
    "Let's start over.",
    "I like pizza.",
    "What time is it?",
    "No, I don't confirm.",
    "Not yet, but ok maybe.",
    "Looks good, no changes needed.",
    "Sure, no worries!",
    "Don't save it.",
    "That's not correct.",
    "",
    "Yep!",
    "Nope.",
)


_PARITY_CUSTOM_CASES: tuple[tuple[str, dict], ...] = (
    (
        "Ship it!",
        {"aff": frozenset({"proceed", "ship"})},
    ),
    (
        "Abort!",
        {"neg": frozenset({"abort", "reject"})},
    ),
    (
        "All systems go!",
        {"aff_phrases": ("all systems go",)},
    ),
    (
        "Take it back!",
        {"neg_phrases": ("take it back",)},
    ),
)


def _classifier_signal(
    msg: str,
    *,
    aff: frozenset[str] | None = None,
    aff_phrases: tuple[str, ...] | None = None,
    neg: frozenset[str] | None = None,
    neg_phrases: tuple[str, ...] | None = None,
) -> bool | None:
    """Re-derive the boolean signal directly from the classifier
    primitives, mirroring ``detect_boolean_signal``'s construction.
    """
    aff_signals = aff or _DEFAULT_AFFIRMATIVE_SIGNALS
    aff_phr = aff_phrases or _DEFAULT_AFFIRMATIVE_PHRASES
    neg_signals = neg or _DEFAULT_NEGATIVE_SIGNALS
    neg_phr = neg_phrases or _DEFAULT_NEGATIVE_PHRASES

    msg_lower = msg.lower()
    classifier = KeywordIntentClassifier(
        vocabulary={
            "affirmative": aff_signals,
            "negative": neg_signals,
        },
        phrases={
            "affirmative": frozenset(aff_phr),
            "negative": frozenset(neg_phr),
        },
        phrase_priority=True,
    )
    result = classifier._classify_sync(msg_lower, _PARITY_AFF_NEG_SPECS)
    if result.intent is None:
        return None
    if result.intent.name == "negative":
        return False

    # Affirmative won — apply the same flip-when-alone semantic that
    # ``detect_boolean_signal`` applies locally.
    has_neg_match = (
        any(_word_in_text(w, msg_lower) for w in neg_signals)
        or any(_word_in_text(p, msg_lower) for p in neg_phr)
    )
    if not has_neg_match:
        check_neg = DEFAULT_NEGATION_KEYWORDS - neg_signals
        if check_neg and has_negation(msg_lower, check_neg):
            return False
    return True


class TestDetectBooleanSignalParityWithClassifier:
    """Drift guard: wrapper and classifier path return the same verdict.

    Parametrized over the canonical spec exercised by
    ``TestDetectBooleanSignal``. Without this guard, the wrapper
    could quietly diverge from the classifier primitives after a
    future change to either ``KeywordIntentClassifier`` or
    ``NegationFilter`` — exactly the drift this consolidation
    eliminates.
    """

    @pytest.mark.parametrize("message", _PARITY_DEFAULT_CASES)
    def test_default_vocabulary_parity(self, message: str) -> None:
        wrapper_signal = detect_boolean_signal(
            message.lower(),
            affirmative_signals=_DEFAULT_AFFIRMATIVE_SIGNALS,
            affirmative_phrases=_DEFAULT_AFFIRMATIVE_PHRASES,
            negative_signals=_DEFAULT_NEGATIVE_SIGNALS,
            negative_phrases=_DEFAULT_NEGATIVE_PHRASES,
        )
        classifier_signal = _classifier_signal(message)
        assert wrapper_signal == classifier_signal, (
            f"Drift for message {message!r}: "
            f"wrapper={wrapper_signal!r} classifier={classifier_signal!r}"
        )

    @pytest.mark.parametrize(("message", "overrides"), _PARITY_CUSTOM_CASES)
    def test_custom_vocabulary_parity(
        self, message: str, overrides: dict,
    ) -> None:
        wrapper_signal = detect_boolean_signal(
            message.lower(),
            affirmative_signals=overrides.get(
                "aff", _DEFAULT_AFFIRMATIVE_SIGNALS,
            ),
            affirmative_phrases=overrides.get(
                "aff_phrases", _DEFAULT_AFFIRMATIVE_PHRASES,
            ),
            negative_signals=overrides.get(
                "neg", _DEFAULT_NEGATIVE_SIGNALS,
            ),
            negative_phrases=overrides.get(
                "neg_phrases", _DEFAULT_NEGATIVE_PHRASES,
            ),
        )
        classifier_signal = _classifier_signal(message, **overrides)
        assert wrapper_signal == classifier_signal, (
            f"Drift for message {message!r} overrides={overrides!r}: "
            f"wrapper={wrapper_signal!r} classifier={classifier_signal!r}"
        )


# ---------------------------------------------------------------------------
# Shared config builders for integration tests
# ---------------------------------------------------------------------------


def _confirm_stage_config(
    *,
    boolean_recovery: bool = True,
    per_field_recovery: bool | None = None,
    pipeline: list[str] | None = None,
    custom_signals: dict | None = None,
) -> dict:
    """Wizard with a review stage that has a boolean ``confirmed`` field."""
    builder = WizardConfigBuilder("bool-recovery-test")

    # Gather stage: collect a name
    builder.stage(
        "gather", is_start=True, prompt="Tell me your name.",
    ).field("name", field_type="string", required=True).transition(
        "review", "data.get('name')",
    )

    # Review stage: boolean confirmed field
    x_extraction: dict = {}
    if per_field_recovery is not None:
        x_extraction["boolean_recovery"] = per_field_recovery
    if custom_signals:
        x_extraction.update(custom_signals)

    builder.stage("review", prompt="Confirm to save.")
    if x_extraction:
        builder.field(
            "confirmed",
            field_type="boolean",
            required=True,
            description="User confirms the configuration is correct.",
            x_extraction=x_extraction,
        )
    else:
        builder.field(
            "confirmed",
            field_type="boolean",
            required=True,
            description="User confirms the configuration is correct.",
        )
    builder.transition("done", "data.get('confirmed') == True")

    builder.stage("done", is_end=True, prompt="Saved!")

    config = builder.build()

    # Apply extraction hints
    settings = config.setdefault("settings", {})
    hints = settings.setdefault("extraction_hints", {})
    hints["boolean_recovery"] = boolean_recovery

    # Apply recovery pipeline
    if pipeline is not None:
        recovery = settings.setdefault("recovery", {})
        recovery["pipeline"] = pipeline

    return config


def _multi_bool_config() -> dict:
    """Wizard with two boolean fields for scope restriction testing."""
    builder = WizardConfigBuilder("multi-bool-test")
    builder.stage("gather", is_start=True, prompt="Confirm settings.")
    builder.field(
        "save_confirmed",
        field_type="boolean",
        required=True,
        description="User confirms they want to save.",
        x_extraction={"boolean_recovery": True},
    )
    builder.field(
        "email_notifications",
        field_type="boolean",
        required=True,
        description="User wants email notifications enabled.",
        x_extraction={"boolean_recovery": True},
    )
    builder.transition(
        "done",
        "has('save_confirmed') and has('email_notifications')",
    )
    builder.stage("done", is_end=True, prompt="Done!")

    config = builder.build()
    settings = config.setdefault("settings", {})
    settings.setdefault("extraction_hints", {})["boolean_recovery"] = True
    recovery = settings.setdefault("recovery", {})
    recovery["pipeline"] = [RECOVERY_BOOLEAN]
    return config


# ---------------------------------------------------------------------------
# Integration tests: BotTestHarness
# ---------------------------------------------------------------------------


class TestBooleanRecoveryIntegration:
    """Integration tests for boolean recovery via BotTestHarness."""

    @pytest.mark.asyncio
    async def test_affirmative_recovery_fills_confirmed(self) -> None:
        """'Yes, save it!' recovers confirmed=True when extraction
        returns nothing for the field."""
        config = _confirm_stage_config(
            pipeline=[RECOVERY_DERIVATION, RECOVERY_BOOLEAN],
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Please confirm."),
                text_response("Saved!"),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],  # Extraction fails to produce confirmed
                [],
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            assert harness.wizard_data.get("name") is not None
            assert harness.wizard_stage == "review"

            await harness.chat("Yes, save it!")
            assert harness.wizard_data.get("confirmed") is True
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_negative_recovery_fills_false(self) -> None:
        """'No, wait.' recovers confirmed=False — wizard stays."""
        config = _confirm_stage_config(
            pipeline=[RECOVERY_BOOLEAN],
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("OK, what would you like to change?"),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],  # Extraction fails
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("No, wait.")
            # confirmed=False doesn't satisfy transition (needs True)
            assert harness.wizard_data.get("confirmed") is False
            assert harness.wizard_stage == "review"

    @pytest.mark.asyncio
    async def test_ambiguous_message_no_recovery(self) -> None:
        """Ambiguous message leaves field unset."""
        config = _confirm_stage_config(
            pipeline=[RECOVERY_BOOLEAN],
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Could you clarify?"),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("I like pizza")
            assert "confirmed" not in harness.wizard_data
            assert harness.wizard_stage == "review"

    @pytest.mark.asyncio
    async def test_recovery_disabled_via_class_level_flag(self) -> None:
        """With boolean_recovery=False, no recovery happens even when
        the strategy is in the pipeline."""
        config = _confirm_stage_config(
            boolean_recovery=False,
            pipeline=[RECOVERY_BOOLEAN],
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Please confirm."),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("Yes, save it!")
            # Recovery disabled — field stays unset
            assert "confirmed" not in harness.wizard_data
            assert harness.wizard_stage == "review"

    @pytest.mark.asyncio
    async def test_per_field_override_enables(self) -> None:
        """Per-field x-extraction.boolean_recovery=True overrides
        class-level False."""
        config = _confirm_stage_config(
            boolean_recovery=False,
            per_field_recovery=True,
            pipeline=[RECOVERY_BOOLEAN],
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Saved!"),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("Yes, save it!")
            assert harness.wizard_data.get("confirmed") is True

    @pytest.mark.asyncio
    async def test_per_field_override_disables(self) -> None:
        """Per-field x-extraction.boolean_recovery=False overrides
        class-level True."""
        config = _confirm_stage_config(
            boolean_recovery=True,
            per_field_recovery=False,
            pipeline=[RECOVERY_BOOLEAN],
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Please confirm."),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("Yes, save it!")
            assert "confirmed" not in harness.wizard_data

    @pytest.mark.asyncio
    async def test_extraction_succeeds_no_recovery_needed(self) -> None:
        """When extraction produces the boolean, recovery is not needed."""
        config = _confirm_stage_config(
            pipeline=[RECOVERY_BOOLEAN],
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Saved!"),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{"confirmed": True}],  # Extraction succeeds
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("Yes, save it!")
            assert harness.wizard_data.get("confirmed") is True
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_pipeline_ordering_boolean_before_escalation(self) -> None:
        """Boolean recovery in pipeline runs before scope escalation.
        When it fills the field, escalation is skipped."""
        config = _confirm_stage_config(
            pipeline=[RECOVERY_BOOLEAN, "scope_escalation"],
        )
        # If boolean recovery works, scope escalation should never fire.
        # We can verify this by not providing extraction for escalation
        # (which would fail if called).
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Saved!"),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],  # Only initial extraction, no escalation result needed
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("Yes, confirm!")
            assert harness.wizard_data.get("confirmed") is True
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_custom_signal_words(self) -> None:
        """Per-field custom signal words are used."""
        config = _confirm_stage_config(
            pipeline=[RECOVERY_BOOLEAN],
            custom_signals={
                "boolean_recovery": True,
                "affirmative_signals": ["proceed", "ship"],
                "negative_signals": ["abort"],
            },
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Saved!"),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("Ship it!")
            assert harness.wizard_data.get("confirmed") is True

    @pytest.mark.asyncio
    async def test_custom_signal_negative(self) -> None:
        """Per-field custom negative signal words are used."""
        config = _confirm_stage_config(
            pipeline=[RECOVERY_BOOLEAN],
            custom_signals={
                "boolean_recovery": True,
                "affirmative_signals": ["proceed"],
                "negative_signals": ["abort"],
            },
        )
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What's your name?"),
                text_response("Aborted."),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{}],
            ],
        ) as harness:
            await harness.chat("My name is Alice")
            await harness.chat("Abort!")
            assert harness.wizard_data.get("confirmed") is False

    @pytest.mark.asyncio
    async def test_non_boolean_field_not_recovered(self) -> None:
        """Boolean recovery only applies to boolean-type fields.
        A missing string field remains missing after boolean recovery."""
        builder = WizardConfigBuilder("non-bool-test")
        builder.stage("gather", is_start=True, prompt="Tell me.")
        builder.field(
            "confirmed", field_type="boolean", required=True,
            description="User confirmation.",
        )
        builder.field("name", field_type="string", required=True)
        builder.transition("done", "data.get('name') and data.get('confirmed')")
        builder.stage("done", is_end=True, prompt="Done!")
        config = builder.build()
        settings = config.setdefault("settings", {})
        settings["extraction_hints"] = {"boolean_recovery": True}
        recovery = settings.setdefault("recovery", {})
        recovery["pipeline"] = [RECOVERY_BOOLEAN]

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Tell me your name."),
            ],
            extraction_results=[
                [{}],
            ],
        ) as harness:
            await harness.chat("Yes!")
            # "confirmed" is boolean → should be recovered
            assert harness.wizard_data.get("confirmed") is True
            # "name" is string → boolean recovery doesn't apply, still missing
            assert harness.wizard_stage == "gather"


class TestMultiBooleanScopeRestriction:
    """Test scope restriction when multiple boolean fields are missing."""

    @pytest.mark.asyncio
    async def test_scope_restriction_requires_field_keywords(self) -> None:
        """With 2 missing boolean fields, only the one whose keywords
        appear in the message gets recovered."""
        config = _multi_bool_config()
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What about notifications?"),
            ],
            extraction_results=[
                [{}],
            ],
        ) as harness:
            # "save" keyword maps to save_confirmed field
            await harness.chat("Yes, save it please!")
            assert harness.wizard_data.get("save_confirmed") is True
            # email_notifications should NOT be filled (no keywords)
            assert "email_notifications" not in harness.wizard_data

    @pytest.mark.asyncio
    async def test_scope_restriction_other_field(self) -> None:
        """The other boolean field can be recovered when its keywords
        appear."""
        config = _multi_bool_config()
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("OK, noted."),
            ],
            extraction_results=[
                [{}],
            ],
        ) as harness:
            await harness.chat("Yes, enable email notifications!")
            assert harness.wizard_data.get("email_notifications") is True
            assert "save_confirmed" not in harness.wizard_data

    @pytest.mark.asyncio
    async def test_single_missing_boolean_no_scope_restriction(self) -> None:
        """When only one boolean field is missing, scope restriction
        is relaxed — any affirmative signal fills it."""
        config = _multi_bool_config()
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("And notifications?"),
            ],
            extraction_results=[
                # First extraction fills one field
                [{"save_confirmed": True}],
            ],
        ) as harness:
            # "Yes!" has no field keywords but only one bool is missing
            await harness.chat("Yes!")
            assert harness.wizard_data.get("email_notifications") is True

    @pytest.mark.asyncio
    async def test_empty_keywords_skipped_with_warning(self) -> None:
        """A boolean field with no extractable keywords (short name,
        no description) is skipped in multi-boolean scope restriction."""
        builder = WizardConfigBuilder("empty-kw-test")
        builder.stage("gather", is_start=True, prompt="Confirm.")
        # "ok" → all words ≤ 2 chars → significant_words returns empty set
        builder.field(
            "ok",
            field_type="boolean",
            required=True,
            x_extraction={"boolean_recovery": True},
        )
        builder.field(
            "save_confirmed",
            field_type="boolean",
            required=True,
            description="User confirms they want to save.",
            x_extraction={"boolean_recovery": True},
        )
        builder.transition(
            "done",
            "data.get('ok') and data.get('save_confirmed')",
        )
        builder.stage("done", is_end=True, prompt="Done!")
        config = builder.build()
        settings = config.setdefault("settings", {})
        settings["extraction_hints"] = {"boolean_recovery": True}
        recovery = settings.setdefault("recovery", {})
        recovery["pipeline"] = [RECOVERY_BOOLEAN]

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[text_response("Please confirm.")],
            extraction_results=[[{}]],
        ) as harness:
            await harness.chat("Yes, save it!")
            # save_confirmed has keywords → recovered
            assert harness.wizard_data.get("save_confirmed") is True
            # "ok" has no extractable keywords → skipped (logged warning)
            assert "ok" not in harness.wizard_data

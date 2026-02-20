"""Tests for wizard transform flow: extraction, confirmation, transition, and data propagation.

These tests exercise the end-to-end generate() flow that the quiz wizard uses:
  1. User provides data → extraction → confirmation template shown (no transition)
  2. User confirms → transition fires → transforms run → data propagates to template
  3. User selects action → transition fires (no confirmation block)
  4. LLM and artifact registry are available in transform context

Issues covered:
  - Settings confirmation being skipped on first message (immediate transition)
  - Transform data not propagating back to wizard state for template rendering
  - _-prefixed keys (like _questions) being filtered from template context
  - Action selection at post-transition stages blocked by confirmation logic
  - LLM not available in TransformContext for question polishing
"""

from typing import Any
from unittest.mock import sentinel

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.extraction.schema_extractor import SchemaExtractor
from dataknobs_llm.llm.providers.echo import EchoProvider

from .conftest import WizardTestManager


# ── Test transform functions ──────────────────────────────────────────


def generate_test_questions(
    data: dict, context: object = None, **kwargs: object
) -> None:
    """Simulate quiz question generation.

    Sets _questions on data dict (in-place mutation, returns None),
    matching the convention used by real quiz transforms.
    """
    topic = data.get("topic", "unknown")
    count = data.get("batch_size", 3)
    data["_questions"] = [
        {
            "question_text": f"Question {i + 1} about {topic}",
            "options": [
                {"id": "A", "text": "Option A", "correct": i == 0},
                {"id": "B", "text": "Option B", "correct": i != 0},
            ],
        }
        for i in range(count)
    ]


def initialize_test_bank(
    data: dict, context: object = None, **kwargs: object
) -> None:
    """Simulate bank initialization (in-place mutation, returns None)."""
    data.setdefault("_bank_questions", [])
    data.setdefault("_bank_reviewed", [])


_captured_contexts: list[Any] = []
"""Module-level list to capture TransformContext from transforms."""


def submit_test_review(
    data: dict, context: object = None, **kwargs: object
) -> None:
    """Simulate review submission.  Captures context for test inspection."""
    _captured_contexts.append(context)
    questions = data.get("_questions", [])
    data["_question_evaluations"] = [
        {
            "question_index": i,
            "passed": True,
            "score": 0.85,
            "feedback": f"Question {i + 1} meets quality standards.",
        }
        for i in range(len(questions))
    ]
    data["_review_passed"] = len(questions)
    data["_review_failed"] = 0


def context_capturing_transform(
    data: dict, context: object = None, **kwargs: object
) -> None:
    """Transform that only captures its TransformContext for inspection."""
    _captured_contexts.append(context)


# ── Wizard config fixtures ───────────────────────────────────────────


QUIZ_WIZARD_CONFIG = {
    "name": "test-quiz-wizard",
    "version": "1.0",
    "settings": {
        "extraction_scope": "current_message",
    },
    "stages": [
        {
            "name": "define_topic",
            "is_start": True,
            "prompt": "What topic should the quiz cover?",
            "response_template": (
                "{%- if topic and difficulty -%}\n"
                "Great — here's what I have:\n"
                "- **Topic:** {{ topic }}\n"
                "- **Difficulty:** {{ difficulty }}\n"
                "- **Batch size:** {{ batch_size | default(3) }}\n"
                "\n"
                "Ready to generate? Say 'looks good' to proceed.\n"
                "{%- elif topic -%}\n"
                "Got it — topic is **{{ topic }}**.\n"
                "Difficulty defaults to medium, batch size to 3.\n"
                "Say 'looks good' to proceed or provide more details.\n"
                "{%- else -%}\n"
                "What topic should the quiz cover?\n"
                "{%- endif -%}"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Subject topic for the quiz",
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["topic"],
            },
            "transitions": [
                {
                    "target": "generate_questions",
                    "condition": "data.get('topic')",
                    "transform": [
                        "initialize_test_bank",
                        "generate_test_questions",
                    ],
                }
            ],
        },
        {
            "name": "generate_questions",
            "prompt": "Here are the generated questions.",
            "response_template": (
                "Generated {{ _questions | length }} questions"
                " on {{ topic }}.\n"
                "\n"
                "{% for q in _questions %}"
                "### Question {{ loop.index }}\n"
                "{{ q.question_text }}\n"
                "{% endfor %}\n"
                "Would you like to **Review** or **Regenerate**?"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["review", "regenerate"],
                    },
                },
            },
            "transitions": [
                {
                    "target": "review_questions",
                    "condition": "data.get('action') == 'review'",
                    "transform": "submit_test_review",
                },
                {
                    "target": "generate_questions",
                    "condition": "data.get('action') == 'regenerate'",
                    "transform": "generate_test_questions",
                },
            ],
        },
        {
            "name": "review_questions",
            "is_end": True,
            "prompt": "Quality review results.",
            "response_template": (
                "## Quality Review\n"
                "**Passed:** {{ _review_passed }}"
                " / {{ _review_passed + _review_failed }}\n"
                "\n"
                "{% for ev in _question_evaluations %}"
                "- Q{{ ev.question_index + 1 }}:"
                " {{ 'PASS' if ev.passed else 'FAIL' }}"
                " ({{ ev.score }})\n"
                "{% endfor %}"
            ),
        },
    ],
}


# ── Helper to build wizard reasoning with extraction ─────────────────


def _build_wizard(
    extraction_responses: list[str],
    artifact_registry: Any = None,
) -> tuple[WizardReasoning, EchoProvider]:
    """Build a WizardReasoning with scripted extraction responses.

    Returns:
        Tuple of (WizardReasoning, extraction_provider) so tests can
        inspect extraction call counts.
    """
    extraction_provider = EchoProvider(
        {"provider": "echo", "model": "echo-extraction"}
    )
    extraction_provider.set_responses(extraction_responses)
    extractor = SchemaExtractor(provider=extraction_provider)

    custom_fns: dict[str, Any] = {
        "generate_test_questions": generate_test_questions,
        "initialize_test_bank": initialize_test_bank,
        "submit_test_review": submit_test_review,
        "context_capturing_transform": context_capturing_transform,
    }

    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(QUIZ_WIZARD_CONFIG, custom_fns)

    reasoning = WizardReasoning(
        wizard_fsm=wizard_fsm,
        extractor=extractor,
        strict_validation=False,
        extraction_scope="current_message",
        artifact_registry=artifact_registry,
    )

    return reasoning, extraction_provider


def _advance_to_generate_questions(
    extraction_responses: list[str] | None = None,
    artifact_registry: Any = None,
) -> tuple[WizardReasoning, WizardTestManager, list[str]]:
    """Helper: build wizard and prepare extraction for a 2-turn advance
    to generate_questions.  Returns (reasoning, manager, remaining_responses)
    where remaining_responses should be passed as additional set_responses.
    """
    all_responses = [
        '{"topic": "English grammar"}',
        "{}",
        *(extraction_responses or []),
    ]
    reasoning, _ = _build_wizard(
        all_responses, artifact_registry=artifact_registry
    )
    return reasoning, WizardTestManager(), all_responses


# ── Tests ─────────────────────────────────────────────────────────────


class TestWizardConfirmation:
    """Tests for the first-render confirmation logic."""

    @pytest.mark.asyncio
    async def test_first_message_shows_confirmation_not_transition(
        self,
    ) -> None:
        """First user message providing data should show confirmation
        template, NOT immediately transition to the next stage.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=['{"topic": "English grammar"}']
        )

        manager = WizardTestManager()
        manager.add_user_message("Create English grammar questions")

        response = await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "define_topic", (
            f"Expected to stay at define_topic for confirmation, "
            f"but moved to {state['current_stage']}"
        )
        assert state["data"].get("topic") == "English grammar"
        assert "English grammar" in response.content

    @pytest.mark.asyncio
    async def test_confirmation_message_triggers_transition(self) -> None:
        """Second message ('Looks good') should trigger the transition
        because no new data is extracted — it's a confirmation.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=[
                '{"topic": "English grammar"}',
                "{}",
            ]
        )

        manager = WizardTestManager()

        # Turn 1: provide topic → confirmation
        manager.add_user_message("Create English grammar questions")
        response1 = await reasoning.generate(manager, llm=None)
        assert manager.metadata["wizard"]["fsm_state"]["current_stage"] == "define_topic"

        # Turn 2: confirm → transition
        manager.add_assistant_message(response1.content)
        manager.add_user_message("Looks good, generate")
        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "generate_questions", (
            f"Expected transition to generate_questions on confirmation, "
            f"but stayed at {state['current_stage']}"
        )

    @pytest.mark.asyncio
    async def test_data_update_after_confirmation_proceeds(self) -> None:
        """After the confirmation template has been shown once, subsequent
        messages with new data proceed to transition evaluation rather
        than re-showing confirmation.  This prevents action-selection
        stages from being blocked.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=[
                '{"topic": "English grammar"}',
                '{"difficulty": "hard"}',
            ]
        )

        manager = WizardTestManager()

        # Turn 1: topic extracted → confirmation shown
        manager.add_user_message("Create English grammar questions")
        response1 = await reasoning.generate(manager, llm=None)
        assert manager.metadata["wizard"]["fsm_state"]["current_stage"] == "define_topic"

        # Turn 2: difficulty changed — template already shown once,
        # so this proceeds to transition evaluation.
        manager.add_assistant_message(response1.content)
        manager.add_user_message("Change difficulty to hard")
        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["data"].get("difficulty") == "hard"
        # Transition fires because data.get('topic') is True and
        # the confirmation was already shown.
        assert state["current_stage"] == "generate_questions"

    @pytest.mark.asyncio
    async def test_missing_required_fields_shows_clarification(self) -> None:
        """When extraction can't satisfy required fields, a clarification
        response should be returned.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=['{"difficulty": "hard"}']
        )

        manager = WizardTestManager()
        manager.add_user_message("Make it hard difficulty")
        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "define_topic"
        assert state.get("data", {}).get("topic") is None


class TestWizardTransformDataPropagation:
    """Tests for transform data propagation and template rendering."""

    @pytest.mark.asyncio
    async def test_transform_data_in_wizard_state(self) -> None:
        """Transform outputs (_questions, _bank_questions) should be in
        wizard_state.data after the FSM transition.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=['{"topic": "English grammar"}', "{}"]
        )
        manager = WizardTestManager()

        manager.add_user_message("Create English grammar questions")
        r1 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r1.content)
        manager.add_user_message("Looks good")
        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert "_questions" in state["data"]
        assert len(state["data"]["_questions"]) == 3
        assert "_bank_questions" in state["data"]

    @pytest.mark.asyncio
    async def test_underscore_keys_in_template(self) -> None:
        """_-prefixed keys (set by transforms) should be accessible in
        response templates.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=['{"topic": "English grammar"}', "{}"]
        )
        manager = WizardTestManager()

        manager.add_user_message("Create English grammar questions")
        r1 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r1.content)
        manager.add_user_message("Looks good")
        r2 = await reasoning.generate(manager, llm=None)

        assert "Generated 3 questions" in r2.content, (
            f"Expected 'Generated 3 questions', got: {r2.content!r}"
        )
        assert "English grammar" in r2.content

    @pytest.mark.asyncio
    async def test_transform_none_return_preserves_data(self) -> None:
        """Transforms that return None should not break the chain."""
        reasoning, _ = _build_wizard(
            extraction_responses=['{"topic": "Physics"}', "{}"]
        )
        manager = WizardTestManager()

        manager.add_user_message("Create Physics quiz")
        r1 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r1.content)
        manager.add_user_message("Go ahead")
        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert "_bank_questions" in state["data"]
        assert "_questions" in state["data"]
        assert state["data"]["topic"] == "Physics"


class TestWizardReviewTransition:
    """Tests for the review action triggering a stage transition."""

    @pytest.mark.asyncio
    async def test_review_action_transitions_to_review_stage(self) -> None:
        """At generate_questions, saying 'review' should transition to
        review_questions — NOT re-render the generate_questions template.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=[
                '{"topic": "English grammar"}',
                "{}",
                '{"action": "review"}',
            ]
        )
        manager = WizardTestManager()

        # Turn 1: topic → confirmation
        manager.add_user_message("Create English grammar questions")
        r1 = await reasoning.generate(manager, llm=None)

        # Turn 2: confirm → transition to generate_questions
        manager.add_assistant_message(r1.content)
        manager.add_user_message("Looks good")
        r2 = await reasoning.generate(manager, llm=None)
        assert "Generated 3 questions" in r2.content

        # Turn 3: review → should transition to review_questions
        manager.add_assistant_message(r2.content)
        manager.add_user_message("Review these questions")
        r3 = await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "review_questions", (
            f"Expected transition to review_questions, "
            f"but stayed at {state['current_stage']}"
        )

    @pytest.mark.asyncio
    async def test_review_template_shows_quality_results(self) -> None:
        """The review_questions template should show pass/fail counts
        and per-question evaluations, not re-display the questions.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=[
                '{"topic": "English grammar"}',
                "{}",
                '{"action": "review"}',
            ]
        )
        manager = WizardTestManager()

        manager.add_user_message("Create English grammar questions")
        r1 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r1.content)
        manager.add_user_message("Looks good")
        r2 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r2.content)
        manager.add_user_message("Review these questions")
        r3 = await reasoning.generate(manager, llm=None)

        # Review template should show quality results
        assert "Quality Review" in r3.content, (
            f"Expected review results, got: {r3.content!r}"
        )
        assert "PASS" in r3.content
        assert "3" in r3.content  # 3 passed

    @pytest.mark.asyncio
    async def test_review_transform_receives_questions(self) -> None:
        """The submit_test_review transform should receive _questions in
        its data dict and produce _question_evaluations.
        """
        _captured_contexts.clear()
        reasoning, _ = _build_wizard(
            extraction_responses=[
                '{"topic": "English grammar"}',
                "{}",
                '{"action": "review"}',
            ]
        )
        manager = WizardTestManager()

        manager.add_user_message("Create English grammar questions")
        r1 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r1.content)
        manager.add_user_message("Looks good")
        r2 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r2.content)
        manager.add_user_message("Review these questions")
        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert "_question_evaluations" in state["data"]
        evals = state["data"]["_question_evaluations"]
        assert len(evals) == 3
        assert all(e["passed"] for e in evals)


class TestWizardTransformContext:
    """Tests for LLM and artifact registry availability in transforms."""

    @pytest.mark.asyncio
    async def test_llm_available_in_transform_context(self) -> None:
        """When an LLM is passed to generate(), it should be accessible
        in the TransformContext.config['llm'] for transforms that need it
        (e.g., question polishing).
        """
        _captured_contexts.clear()

        # Use a config with context_capturing_transform to inspect
        config = {
            "name": "context-test",
            "version": "1.0",
            "settings": {"extraction_scope": "current_message"},
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Go",
                    "schema": {
                        "type": "object",
                        "properties": {"go": {"type": "string"}},
                        "required": ["go"],
                    },
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('go')",
                            "transform": "context_capturing_transform",
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        extraction_provider = EchoProvider(
            {"provider": "echo", "model": "echo-extraction"}
        )
        extraction_provider.set_responses(['{"go": "yes"}'])
        extractor = SchemaExtractor(provider=extraction_provider)

        custom_fns = {
            "context_capturing_transform": context_capturing_transform,
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config, custom_fns)

        fake_llm = sentinel.llm_provider
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            extractor=extractor,
            strict_validation=False,
            extraction_scope="current_message",
        )

        manager = WizardTestManager()
        manager.add_user_message("go")
        # No confirmation stage (no response_template), so transition
        # fires on first message.
        await reasoning.generate(manager, llm=fake_llm)

        assert len(_captured_contexts) == 1
        ctx = _captured_contexts[0]
        assert hasattr(ctx, "config")
        assert ctx.config.get("llm") is fake_llm

    @pytest.mark.asyncio
    async def test_artifact_registry_in_transform_context(self) -> None:
        """When artifact_registry is set on WizardReasoning, it should
        be available in the TransformContext passed to transforms.
        """
        _captured_contexts.clear()
        fake_registry = sentinel.artifact_registry

        config = {
            "name": "registry-test",
            "version": "1.0",
            "settings": {"extraction_scope": "current_message"},
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Go",
                    "schema": {
                        "type": "object",
                        "properties": {"go": {"type": "string"}},
                        "required": ["go"],
                    },
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('go')",
                            "transform": "context_capturing_transform",
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        extraction_provider = EchoProvider(
            {"provider": "echo", "model": "echo-extraction"}
        )
        extraction_provider.set_responses(['{"go": "yes"}'])
        extractor = SchemaExtractor(provider=extraction_provider)

        custom_fns = {
            "context_capturing_transform": context_capturing_transform,
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config, custom_fns)

        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            extractor=extractor,
            strict_validation=False,
            extraction_scope="current_message",
            artifact_registry=fake_registry,
        )

        manager = WizardTestManager()
        manager.add_user_message("go")
        await reasoning.generate(manager, llm=None)

        assert len(_captured_contexts) == 1
        ctx = _captured_contexts[0]
        assert ctx.artifact_registry is fake_registry


class TestWizardTransitionRecords:
    """Tests for transition audit trail."""

    @pytest.mark.asyncio
    async def test_transition_records_created(self) -> None:
        """Transition records should be created when the wizard moves
        between stages.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=['{"topic": "Biology"}', "{}"]
        )
        manager = WizardTestManager()

        manager.add_user_message("Create Biology quiz")
        r1 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r1.content)
        manager.add_user_message("Looks good")
        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        transitions = state.get("transitions", [])
        assert len(transitions) >= 1
        assert transitions[-1]["from_stage"] == "define_topic"
        assert transitions[-1]["to_stage"] == "generate_questions"

    @pytest.mark.asyncio
    async def test_full_three_stage_flow(self) -> None:
        """Full flow: define_topic → generate_questions → review_questions
        should produce two transition records.
        """
        reasoning, _ = _build_wizard(
            extraction_responses=[
                '{"topic": "Math"}',
                "{}",
                '{"action": "review"}',
            ]
        )
        manager = WizardTestManager()

        manager.add_user_message("Create Math quiz")
        r1 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r1.content)
        manager.add_user_message("Looks good")
        r2 = await reasoning.generate(manager, llm=None)
        manager.add_assistant_message(r2.content)
        manager.add_user_message("Review")
        await reasoning.generate(manager, llm=None)

        state = manager.metadata["wizard"]["fsm_state"]
        assert state["current_stage"] == "review_questions"
        transitions = state.get("transitions", [])
        assert len(transitions) == 2
        assert transitions[0]["from_stage"] == "define_topic"
        assert transitions[0]["to_stage"] == "generate_questions"
        assert transitions[1]["from_stage"] == "generate_questions"
        assert transitions[1]["to_stage"] == "review_questions"

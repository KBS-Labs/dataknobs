"""Tests for wizard condition evaluation and exec() scope handling.

These tests verify that wizard inline conditions can properly access the 'data'
variable, which requires correct handling of Python's exec() globals/locals.

Bug context (2026-02-02):
- Inline condition functions created via exec() couldn't access 'data' variable
- This was because exec() was called with {} as globals and local_vars as locals
- The inner function _test() couldn't access 'data' from the outer scope
- Fix: Pass data in globals dict so _test() can access it
"""

from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from typing import Any
import logging
import pytest


class TestWizardConditionEvaluation:
    """Tests for wizard condition function creation and evaluation."""

    def test_simple_data_access(self, wizard_loader: WizardConfigLoader):
        """Verify condition can access data.get() pattern."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('ready') == True",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Without ready flag - should stay at start
        fsm.step({"ready": False})
        assert fsm.current_stage == "start"

        # With ready flag - should transition
        fsm.restart()
        fsm.step({"ready": True})
        assert fsm.current_stage == "end"

    def test_missing_key_returns_false(self, wizard_loader: WizardConfigLoader):
        """Verify condition handles missing keys gracefully."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('nonexistent')",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Missing key should not cause error, just evaluate to False
        fsm.step({})
        assert fsm.current_stage == "start"

    def test_nested_data_access(self, wizard_loader: WizardConfigLoader):
        """Verify condition can access nested data structures."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('user', {}).get('confirmed') == True",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Without nested confirmed - stay at start
        fsm.step({"user": {}})
        assert fsm.current_stage == "start"

        # With nested confirmed=True - transition
        fsm.restart()
        fsm.step({"user": {"confirmed": True}})
        assert fsm.current_stage == "end"

    def test_condition_with_boolean_comparison(self, wizard_loader: WizardConfigLoader):
        """Verify condition handles boolean comparisons correctly."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            # Explicit boolean comparison
                            "condition": "data.get('confirmed') == True",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # confirmed=False should not transition
        fsm.step({"confirmed": False})
        assert fsm.current_stage == "start"

        # confirmed=True should transition
        fsm.restart()
        fsm.step({"confirmed": True})
        assert fsm.current_stage == "end"

    def test_condition_with_truthy_check(self, wizard_loader: WizardConfigLoader):
        """Verify condition handles truthy checks (without explicit comparison)."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            # Just checking truthiness, no == True
                            "condition": "data.get('value')",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Empty string is falsy
        fsm.step({"value": ""})
        assert fsm.current_stage == "start"

        # Non-empty string is truthy
        fsm.restart()
        fsm.step({"value": "something"})
        assert fsm.current_stage == "end"

    def test_condition_with_numeric_check(self, wizard_loader: WizardConfigLoader):
        """Verify condition handles numeric comparisons."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('count', 0) > 5",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # count=3 should not transition (3 > 5 is False)
        fsm.step({"count": 3})
        assert fsm.current_stage == "start"

        # count=10 should transition (10 > 5 is True)
        fsm.restart()
        fsm.step({"count": 10})
        assert fsm.current_stage == "end"

    def test_condition_with_in_operator(self, wizard_loader: WizardConfigLoader):
        """Verify condition handles 'in' operator checks."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('status') in ['approved', 'confirmed']",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # status=pending should not transition
        fsm.step({"status": "pending"})
        assert fsm.current_stage == "start"

        # status=approved should transition
        fsm.restart()
        fsm.step({"status": "approved"})
        assert fsm.current_stage == "end"

    def test_multiple_transitions_with_conditions(self, wizard_loader: WizardConfigLoader):
        """Verify multiple transitions with different conditions work correctly."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "approved",
                            "condition": "data.get('status') == 'approved'",
                            "priority": 0,
                        },
                        {
                            "target": "rejected",
                            "condition": "data.get('status') == 'rejected'",
                            "priority": 1,
                        },
                    ],
                },
                {
                    "name": "approved",
                    "is_end": True,
                    "prompt": "Approved!",
                },
                {
                    "name": "rejected",
                    "is_end": True,
                    "prompt": "Rejected",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Test approved path
        fsm.step({"status": "approved"})
        assert fsm.current_stage == "approved"

        # Test rejected path
        fsm.restart()
        fsm.step({"status": "rejected"})
        assert fsm.current_stage == "rejected"

        # Test no match - should stay at start
        fsm.restart()
        fsm.step({"status": "pending"})
        assert fsm.current_stage == "start"

    def test_condition_error_returns_false(self, wizard_loader: WizardConfigLoader):
        """Verify condition errors are caught and return False."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            # This will raise an error (calling int on non-int)
                            "condition": "int(data.get('value', 'not_a_number')) > 5",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Should not crash, just return False and stay at start
        fsm.step({})  # Will try int('not_a_number') which raises ValueError
        assert fsm.current_stage == "start"

    def test_data_variable_is_accessible(self, wizard_loader: WizardConfigLoader):
        """Explicitly test that the 'data' variable is accessible in conditions.

        This was the core bug - exec() scope issues prevented 'data' from being
        accessible inside the dynamically created condition function.
        """
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            # Simple data access - will fail if data not in scope
                            "condition": "data is not None and 'key' in data",
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Should transition because data dict has 'key'
        fsm.step({"key": "value"})
        assert fsm.current_stage == "end", (
            "'data' variable should be accessible in condition. "
            "If this fails, check exec() scope handling."
        )


class TestWizardTransitionLogic:
    """Tests for wizard state transition logic."""

    def test_stay_at_stage_without_matching_condition(self, wizard_loader: WizardConfigLoader):
        """Verify FSM stays at current stage if no conditions match."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "review",
                    "is_start": True,
                    "prompt": "Review your settings",
                    "transitions": [
                        {
                            "target": "save",
                            "condition": "data.get('confirmed') == True",
                        }
                    ],
                },
                {
                    "name": "save",
                    "is_end": True,
                    "prompt": "Saving...",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Step multiple times without confirmed - should stay at review
        fsm.step({"some_data": "value"})
        assert fsm.current_stage == "review"

        fsm.step({"more_data": "value2"})
        assert fsm.current_stage == "review"

        fsm.step({"confirmed": False})
        assert fsm.current_stage == "review"

        # Finally confirm - should transition
        fsm.step({"confirmed": True})
        assert fsm.current_stage == "save"

    def test_unconditional_transition(self, wizard_loader: WizardConfigLoader):
        """Verify unconditional transitions work."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            # No condition - always transitions
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # Should immediately transition
        fsm.step({})
        assert fsm.current_stage == "end"


class TestYamlBooleanLiterals:
    """Tests for YAML/JSON boolean literal support in conditions.

    Config authors naturally write ``condition: "true"`` (lowercase) since
    that's the YAML/JSON convention. The condition evaluator must accept
    both Python (True/False/None) and YAML/JSON (true/false/null)
    conventions.
    """

    @pytest.mark.parametrize(
        "condition,expected_stage",
        [
            ("True", "end"),
            ("true", "end"),
            ("False", "start"),
            ("false", "start"),
        ],
    )
    def test_boolean_literal_variants(
        self, condition, expected_stage, wizard_loader: WizardConfigLoader
    ):
        """Both Python and YAML boolean literals work in conditions."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [{"target": "end", "condition": condition}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)
        fsm.step({})
        assert fsm.current_stage == expected_stage

    @pytest.mark.parametrize(
        "condition,expected_stage",
        [
            ("null", "start"),  # null is None, which is falsy
            ("none", "start"),  # none is None, which is falsy
            ("None", "start"),  # Python None, which is falsy
        ],
    )
    def test_null_none_literal_variants(
        self, condition, expected_stage, wizard_loader: WizardConfigLoader
    ):
        """null/none/None all evaluate to Python None (falsy)."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [{"target": "end", "condition": condition}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)
        fsm.step({})
        assert fsm.current_stage == expected_stage

    def test_lowercase_true_with_data_expression(self, wizard_loader: WizardConfigLoader):
        """Lowercase true works in compound expressions."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('ready') == true",
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        # ready=False should not match `== true`
        fsm.step({"ready": False})
        assert fsm.current_stage == "start"

        # ready=True should match `== true`
        fsm.restart()
        fsm.step({"ready": True})
        assert fsm.current_stage == "end"

    def test_no_collision_with_variable_names(self, wizard_loader: WizardConfigLoader):
        """Variables like 'is_true' are not affected by the aliases."""
        wizard_config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Start",
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('is_true') == 'yes'",
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        fsm = wizard_loader.load_from_dict(wizard_config)

        fsm.step({"is_true": "yes"})
        assert fsm.current_stage == "end"


# ---------------------------------------------------------------------------
# Load-time reporting of conditions the expression engine will refuse
# ---------------------------------------------------------------------------

LOADER_LOGGER = "dataknobs_bots.reasoning.wizard_loader"

#: A condition ``safe_eval`` refuses statically, before evaluating anything.
#: Multiline is the realistic case: conditions live in YAML, and both a ``|``
#: literal block and a ``>`` folded block containing a blank line produce a
#: newline that survives ``.strip()``.
UNEVALUABLE_CONDITION = "(data.get('a')\n and data.get('b'))"

#: A condition the engine accepts and that fails at evaluation, on this data.
#: The distinction this file is about: "not satisfied yet", not "will never
#: run".
RUNTIME_FAILING_CONDITION = "data['absent']"


def _wizard_with_condition(condition: str) -> dict[str, Any]:
    """A two-stage wizard whose only transition carries ``condition``.

    Every stage is terminal or has a transition, and the start stage
    carries a ``response_template``, so none of the loader's *other*
    config warnings can fire. That lets the assertions below be made
    against every WARNING the loader emits rather than against a
    hand-picked subset — which is what makes "no warning" mean it.
    """
    return {
        "name": "test-wizard",
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "prompt": "Start",
                "response_template": "Start",
                "transitions": [{"target": "end", "condition": condition}],
            },
            {"name": "end", "is_end": True, "prompt": "Done"},
        ],
    }


RESPONDER_LOGGER = "dataknobs_bots.reasoning.wizard_response"


def _warnings(
    caplog: pytest.LogCaptureFixture,
    logger: str = LOADER_LOGGER,
) -> list[str]:
    """Every WARNING (or worse) ``logger`` emitted, as rendered text.

    Scoped to one logger on purpose. ``caplog`` collects everything that
    propagates, and a whole-bot build emits warnings from several modules;
    an unscoped "nothing was warned" assertion would then be reporting on
    code the test is not about.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name == logger
    ]


class TestConditionLoadTimeReport:
    """A condition the engine will refuse is reported once, at load.

    A refusal and an unmet condition are the same value to the FSM —
    ``default=False`` — so a refused transition is indistinguishable from
    one that is simply not satisfied. Reporting it per evaluation says so
    after deploy, N times. Reporting it at load says so once, while the
    author is still in the build loop, which is the only moment the
    information is actionable.
    """

    def test_unevaluable_condition_is_reported_at_load(
        self,
        wizard_loader: WizardConfigLoader,
        caplog: pytest.LogCaptureFixture,
    ):
        """The warning names the stage, the target and the reason.

        All three are needed to act on it: the reason alone does not say
        which of a config's conditions to go and fix.
        """
        with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
            wizard_loader.load_from_dict(_wizard_with_condition(UNEVALUABLE_CONDITION))

        messages = _warnings(caplog)
        assert len(messages) == 1, messages
        assert "start" in messages[0]
        assert "end" in messages[0]
        assert "Multiline expressions are not allowed" in messages[0]

    def test_runtime_failure_is_not_reported_at_load(
        self,
        wizard_loader: WizardConfigLoader,
        caplog: pytest.LogCaptureFixture,
    ):
        """The anti-test: a missing key must not be reported as a refusal.

        ``data['absent']`` raises ``KeyError`` and degrades to False, which
        is a legitimate "not satisfied yet" — the state every wizard
        condition is in before its data arrives. Reporting it at load would
        warn about every condition in every config, and a warning that
        fires on correct input is one consumers learn to ignore.
        """
        with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
            wizard_loader.load_from_dict(_wizard_with_condition(RUNTIME_FAILING_CONDITION))

        assert _warnings(caplog) == []

    def test_unevaluable_condition_still_loads(
        self,
        wizard_loader: WizardConfigLoader,
    ):
        """The load-time pass reports; it does not reject.

        One unusable transition must not take the whole wizard down —
        every other stage still works, and the refused condition behaves
        exactly as it did before the check existed.
        """
        fsm = wizard_loader.load_from_dict(_wizard_with_condition(UNEVALUABLE_CONDITION))

        fsm.step({"a": 1, "b": 2})
        assert fsm.current_stage == "start"


class TestConditionEvaluationLogLevels:
    """WARNING means an author must act; DEBUG means "not satisfied yet"."""

    def test_runtime_failure_is_debug_at_evaluation(
        self,
        wizard_loader: WizardConfigLoader,
        caplog: pytest.LogCaptureFixture,
    ):
        """A condition that can run but did not succeed is not a warning.

        This one warned on every turn before: a wizard sitting on a stage
        whose guard reads a key that has not arrived yet produced a
        WARNING per turn, for a config that is entirely correct.
        """
        fsm = wizard_loader.load_from_dict(_wizard_with_condition(RUNTIME_FAILING_CONDITION))

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=LOADER_LOGGER):
            fsm.step({})

        assert fsm.current_stage == "start"
        assert _warnings(caplog) == []
        assert any(
            RUNTIME_FAILING_CONDITION in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.DEBUG
        ), [r.getMessage() for r in caplog.records]

    def test_unevaluable_condition_warns_at_evaluation(
        self,
        wizard_loader: WizardConfigLoader,
        caplog: pytest.LogCaptureFixture,
    ):
        """A refusal keeps WARNING, at load and at every evaluation.

        Unlike the runtime case this does not go away on its own: the
        expression cannot run on any data, so every turn it is asked is a
        turn the stage silently cannot advance.
        """
        fsm = wizard_loader.load_from_dict(_wizard_with_condition(UNEVALUABLE_CONDITION))

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=LOADER_LOGGER):
            fsm.step({"a": 1, "b": 2})

        assert len(_warnings(caplog)) == 1, _warnings(caplog)
        assert "Multiline expressions are not allowed" in _warnings(caplog)[0]


class TestConditionReturnPrefix:
    """The engine owns the ``return`` wrap; the loader no longer copies it.

    These pass both before and after the loader's copy is deleted — which
    is the point. They are what makes the deletion checkable rather than
    merely asserted.
    """

    def test_explicit_return_prefix_is_accepted(
        self,
        wizard_loader: WizardConfigLoader,
    ):
        """``return <expr>`` is a supported spelling for a condition.

        This is the case the loader's own prefix wrap existed to handle,
        and the case that breaks if the engine's wrap is ever re-derived
        with a ``mode="eval"`` parse instead of reused.
        """
        fsm = wizard_loader.load_from_dict(_wizard_with_condition("return data.get('ready')"))

        fsm.step({"ready": True})
        assert fsm.current_stage == "end"

    def test_name_beginning_with_return_is_reported_not_silently_false(
        self,
        wizard_loader: WizardConfigLoader,
        caplog: pytest.LogCaptureFixture,
    ):
        """An identifier merely *starting* with ``return`` is an expression.

        ``return_code`` is not in the condition scope, so this is a
        NameError — a runtime failure, reported as one. What matters is
        that it is reported at all: while the engine's prefix rule was a
        substring test this exact string was left unwrapped, so the
        generated body was a bare expression statement, the function
        returned None, and ``coerce_bool=True`` turned that into False
        with ``success=True`` and nothing logged anywhere.
        """
        fsm = wizard_loader.load_from_dict(_wizard_with_condition("return_code == 0"))

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=LOADER_LOGGER):
            fsm.step({"return_code": 0})

        assert fsm.current_stage == "start"
        assert any(
            "return_code" in r.getMessage() and "not defined" in r.getMessage()
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# The same two reports, across the whole bot-construction boundary
# ---------------------------------------------------------------------------


def _harness_wizard(condition: str) -> dict[str, Any]:
    """A three-stage wizard whose second transition carries ``condition``.

    The shape is what it takes to reach
    :meth:`WizardResponder.evaluate_condition` from an ordinary turn.
    That method runs from the auto-advance gate, which the wizard consults
    *after* a transition — so a config whose only conditional transition is
    the one under test never reaches it, because the condition under test
    is the thing preventing the transition. ``gather`` therefore advances
    on its own satisfiable condition, and the gate then evaluates
    ``hold``'s.
    """
    return (
        WizardConfigBuilder("condition-report")
        .stage("gather", is_start=True, prompt="Tell me your name.")
        .field("name", field_type="string", required=True)
        .transition("hold", "data.get('name')")
        .stage(
            "hold",
            prompt="Holding.",
            response_template="Holding.",
            auto_advance=True,
        )
        .transition("done", condition)
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


class TestConditionReportsThroughABuiltBot:
    """The reports have to reach a bot built from config, not just a loader.

    The tests above drive ``load_from_dict`` directly, which is the
    production path but not the whole one. These pin that a consumer who
    only ever calls ``from_config`` sees both reports — the load-time one
    from the loader, and the per-evaluation one from the responder, whose
    conditions never pass through the loader at all.
    """

    async def test_unevaluable_condition_is_reported_when_the_bot_is_built(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        """Building the bot is enough — no turn required.

        This is the claim that makes the check worth having: the author
        learns the condition is unusable at build time, rather than from a
        wizard that silently declines to advance in production.
        """
        with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
            async with await BotTestHarness.create(
                wizard_config=_harness_wizard(UNEVALUABLE_CONDITION),
                main_responses=["Got it!"],
            ):
                pass

        messages = _warnings(caplog)
        assert len(messages) == 1, messages
        assert "Multiline expressions are not allowed" in messages[0]
        assert "hold" in messages[0]
        assert "done" in messages[0]

    async def test_responder_warns_on_a_condition_it_cannot_evaluate(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        """``WizardResponder`` conditions never pass through the loader.

        ``evaluate_condition`` is reached from the auto-advance gate and
        from subflow guards, neither of which has a load-time moment — so
        the responder runs the static check itself, on the failure path.
        """
        async with await BotTestHarness.create(
            wizard_config=_harness_wizard(UNEVALUABLE_CONDITION),
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice"}]],
        ) as harness:
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger=RESPONDER_LOGGER):
                await harness.chat("My name is Alice")

        assert harness.wizard_stage == "hold"
        messages = _warnings(caplog, RESPONDER_LOGGER)
        assert any("Multiline expressions are not allowed" in m for m in messages), messages

    async def test_responder_stays_quiet_when_the_data_has_not_arrived(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        """The anti-test, at the responder.

        A guard reading a key that is not there yet is the ordinary state
        of a wizard mid-flow. If that warned, every wizard would warn on
        every turn, and the warning above would be invisible in the noise.
        """
        async with await BotTestHarness.create(
            wizard_config=_harness_wizard(RUNTIME_FAILING_CONDITION),
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice"}]],
        ) as harness:
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger=RESPONDER_LOGGER):
                await harness.chat("My name is Alice")

        assert harness.wizard_stage == "hold"
        assert _warnings(caplog, RESPONDER_LOGGER) == []

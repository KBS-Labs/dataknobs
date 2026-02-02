"""Tests for wizard condition evaluation and exec() scope handling.

These tests verify that wizard inline conditions can properly access the 'data'
variable, which requires correct handling of Python's exec() globals/locals.

Bug context (2026-02-02):
- Inline condition functions created via exec() couldn't access 'data' variable
- This was because exec() was called with {} as globals and local_vars as locals
- The inner function _test() couldn't access 'data' from the outer scope
- Fix: Pass data in globals dict so _test() can access it
"""

import pytest

from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


class TestWizardConditionEvaluation:
    """Tests for wizard condition function creation and evaluation."""

    def test_simple_data_access(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # Without ready flag - should stay at start
        fsm.step({"ready": False})
        assert fsm.current_stage == "start"

        # With ready flag - should transition
        fsm.restart()
        fsm.step({"ready": True})
        assert fsm.current_stage == "end"

    def test_missing_key_returns_false(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # Missing key should not cause error, just evaluate to False
        fsm.step({})
        assert fsm.current_stage == "start"

    def test_nested_data_access(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # Without nested confirmed - stay at start
        fsm.step({"user": {}})
        assert fsm.current_stage == "start"

        # With nested confirmed=True - transition
        fsm.restart()
        fsm.step({"user": {"confirmed": True}})
        assert fsm.current_stage == "end"

    def test_condition_with_boolean_comparison(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # confirmed=False should not transition
        fsm.step({"confirmed": False})
        assert fsm.current_stage == "start"

        # confirmed=True should transition
        fsm.restart()
        fsm.step({"confirmed": True})
        assert fsm.current_stage == "end"

    def test_condition_with_truthy_check(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # Empty string is falsy
        fsm.step({"value": ""})
        assert fsm.current_stage == "start"

        # Non-empty string is truthy
        fsm.restart()
        fsm.step({"value": "something"})
        assert fsm.current_stage == "end"

    def test_condition_with_numeric_check(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # count=3 should not transition (3 > 5 is False)
        fsm.step({"count": 3})
        assert fsm.current_stage == "start"

        # count=10 should transition (10 > 5 is True)
        fsm.restart()
        fsm.step({"count": 10})
        assert fsm.current_stage == "end"

    def test_condition_with_in_operator(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # status=pending should not transition
        fsm.step({"status": "pending"})
        assert fsm.current_stage == "start"

        # status=approved should transition
        fsm.restart()
        fsm.step({"status": "approved"})
        assert fsm.current_stage == "end"

    def test_multiple_transitions_with_conditions(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

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

    def test_condition_error_returns_false(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # Should not crash, just return False and stay at start
        fsm.step({})  # Will try int('not_a_number') which raises ValueError
        assert fsm.current_stage == "start"

    def test_data_variable_is_accessible(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # Should transition because data dict has 'key'
        fsm.step({"key": "value"})
        assert fsm.current_stage == "end", (
            "'data' variable should be accessible in condition. "
            "If this fails, check exec() scope handling."
        )


class TestWizardTransitionLogic:
    """Tests for wizard state transition logic."""

    def test_stay_at_stage_without_matching_condition(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

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

    def test_unconditional_transition(self):
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

        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config)

        # Should immediately transition
        fsm.step({})
        assert fsm.current_stage == "end"

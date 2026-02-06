"""Tests for truthy/falsy condition evaluation edge cases.

These tests verify that FSM condition evaluation handles various return types
correctly, especially tuples like (bool, reason) that are truthy even when
the boolean component is False.

Bug context (2026-02-02):
- Condition functions returning (False, None) were evaluated as truthy
- This caused FSM to always transition because bool((False, None)) == True
- The fix extracts result[0] when result is a tuple before boolean evaluation
"""

import pytest

from dataknobs_fsm.api.advanced import create_advanced_fsm


def make_simple_config(condition_name: str) -> dict:
    """Create a simple two-state FSM config with a condition."""
    return {
        "name": "test",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "end",
                        "condition": {
                            "type": "registered",
                            "name": condition_name,
                        },
                    }
                ],
            }
        ],
    }


def make_inline_condition_config(condition_code: str) -> dict:
    """Create a simple two-state FSM config with an inline condition."""
    return {
        "name": "test",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "end",
                        "condition": {
                            "type": "inline",
                            "name": "check",
                            "code": condition_code,
                        },
                    }
                ],
            }
        ],
    }


class TestTupleReturnHandling:
    """Tests for condition functions that return tuples."""

    def test_tuple_false_none_should_not_transition(self):
        """Verify (False, None) tuple is treated as False, not truthy.

        This was the original bug: bool((False, None)) == True, so
        conditions returning (False, None) would always pass.
        """
        config = make_simple_config("return_false_tuple")

        # Function that returns (False, None) - should NOT transition
        def return_false_tuple(data, context=None):
            return (False, None)

        fsm = create_advanced_fsm(
            config, custom_functions={"return_false_tuple": return_false_tuple}
        )
        ctx = fsm.create_context({})

        # Execute step - should stay at start because condition is False
        fsm.execute_step_sync(ctx)

        assert ctx.current_state == "start", (
            "FSM should stay at 'start' when condition returns (False, None). "
            f"Got: {ctx.current_state}"
        )

    def test_tuple_true_none_should_transition(self):
        """Verify (True, None) tuple correctly triggers transition."""
        config = make_simple_config("return_true_tuple")

        def return_true_tuple(data, context=None):
            return (True, "condition met")

        fsm = create_advanced_fsm(
            config, custom_functions={"return_true_tuple": return_true_tuple}
        )
        ctx = fsm.create_context({})

        fsm.execute_step_sync(ctx)

        assert ctx.current_state == "end", (
            "FSM should transition to 'end' when condition returns (True, ...). "
            f"Got: {ctx.current_state}"
        )

    def test_plain_false_should_not_transition(self):
        """Verify plain False return does not trigger transition."""
        config = make_simple_config("return_false")

        def return_false(data, context=None):
            return False

        fsm = create_advanced_fsm(
            config, custom_functions={"return_false": return_false}
        )
        ctx = fsm.create_context({})

        fsm.execute_step_sync(ctx)

        assert ctx.current_state == "start"

    def test_plain_true_should_transition(self):
        """Verify plain True return triggers transition."""
        config = make_simple_config("return_true")

        def return_true(data, context=None):
            return True

        fsm = create_advanced_fsm(
            config, custom_functions={"return_true": return_true}
        )
        ctx = fsm.create_context({})

        fsm.execute_step_sync(ctx)

        assert ctx.current_state == "end"

    def test_empty_tuple_should_not_transition(self):
        """Verify empty tuple () is treated as falsy."""
        config = make_simple_config("return_empty_tuple")

        def return_empty_tuple(data, context=None):
            return ()  # Empty tuple is falsy

        fsm = create_advanced_fsm(
            config, custom_functions={"return_empty_tuple": return_empty_tuple}
        )
        ctx = fsm.create_context({})

        fsm.execute_step_sync(ctx)

        # Empty tuple should be falsy, so no transition
        assert ctx.current_state == "start"

    def test_none_should_not_transition(self):
        """Verify None return does not trigger transition."""
        config = make_simple_config("return_none")

        def return_none(data, context=None):
            return None

        fsm = create_advanced_fsm(
            config, custom_functions={"return_none": return_none}
        )
        ctx = fsm.create_context({})

        fsm.execute_step_sync(ctx)

        assert ctx.current_state == "start"

    def test_zero_should_not_transition(self):
        """Verify 0 return does not trigger transition."""
        config = make_simple_config("return_zero")

        def return_zero(data, context=None):
            return 0

        fsm = create_advanced_fsm(
            config, custom_functions={"return_zero": return_zero}
        )
        ctx = fsm.create_context({})

        fsm.execute_step_sync(ctx)

        assert ctx.current_state == "start"

    def test_empty_string_should_not_transition(self):
        """Verify empty string return does not trigger transition."""
        config = make_simple_config("return_empty_string")

        def return_empty_string(data, context=None):
            return ""

        fsm = create_advanced_fsm(
            config, custom_functions={"return_empty_string": return_empty_string}
        )
        ctx = fsm.create_context({})

        fsm.execute_step_sync(ctx)

        assert ctx.current_state == "start"

    def test_tuple_with_false_first_element_should_not_transition(self):
        """Verify tuple with False as first element does not transition.

        This tests the specific pattern where conditions return (bool, reason)
        and we need to extract the boolean to evaluate truthiness.
        """
        config = make_simple_config("check_condition")

        # This function returns a tuple with reason - common pattern
        def check_condition(data, context=None):
            if data.get("approved"):
                return (True, "User approved")
            return (False, "User has not approved")

        fsm = create_advanced_fsm(
            config, custom_functions={"check_condition": check_condition}
        )

        # Test without approval - should not transition
        ctx = fsm.create_context({"approved": False})
        fsm.execute_step_sync(ctx)
        assert ctx.current_state == "start", "Should not transition without approval"

        # Test with approval - should transition
        ctx2 = fsm.create_context({"approved": True})
        fsm.execute_step_sync(ctx2)
        assert ctx2.current_state == "end", "Should transition with approval"


class TestDataAccessInConditions:
    """Tests for proper data access within inline condition functions.

    These tests verify that inline conditions can properly access the 'data'
    variable passed to them, which requires correct exec() scope handling.
    """

    def test_inline_condition_can_access_data(self):
        """Verify inline conditions can access the data dict."""
        config = make_inline_condition_config("return data.get('value') == 'correct'")

        fsm = create_advanced_fsm(config)

        # Test with wrong value - should not transition
        ctx = fsm.create_context({"value": "wrong"})
        fsm.execute_step_sync(ctx)
        assert ctx.current_state == "start"

        # Test with correct value - should transition
        ctx2 = fsm.create_context({"value": "correct"})
        fsm.execute_step_sync(ctx2)
        assert ctx2.current_state == "end"

    def test_inline_condition_with_missing_data_key(self):
        """Verify inline conditions handle missing data keys gracefully."""
        # Using .get() should handle missing keys
        config = make_inline_condition_config(
            "return data.get('nonexistent') is not None"
        )

        fsm = create_advanced_fsm(config)
        ctx = fsm.create_context({})  # Empty data

        # Should not raise an error, just return False
        fsm.execute_step_sync(ctx)
        assert ctx.current_state == "start"

    def test_inline_condition_with_nested_data(self):
        """Verify inline conditions can access nested data structures."""
        config = make_inline_condition_config(
            "return data.get('user', {}).get('confirmed') == True"
        )

        fsm = create_advanced_fsm(config)

        # Test without nested data - should not transition
        ctx = fsm.create_context({"user": {}})
        fsm.execute_step_sync(ctx)
        assert ctx.current_state == "start"

        # Test with confirmed=False - should not transition
        ctx2 = fsm.create_context({"user": {"confirmed": False}})
        fsm.execute_step_sync(ctx2)
        assert ctx2.current_state == "start"

        # Test with confirmed=True - should transition
        ctx3 = fsm.create_context({"user": {"confirmed": True}})
        fsm.execute_step_sync(ctx3)
        assert ctx3.current_state == "end"


class TestEdgeCaseTruthyValues:
    """Tests for edge cases in truthy/falsy evaluation."""

    @pytest.mark.parametrize(
        "return_value,should_transition",
        [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            (-1, True),  # Non-zero numbers are truthy
            ("yes", True),
            ("", False),
            ([], False),
            ([1], True),
            ({}, False),
            ({"key": "value"}, True),
            (None, False),
            ((True,), True),  # Single-element tuple with True
            ((False,), False),  # Single-element tuple with False
            ((True, "reason"), True),  # Tuple with True first
            ((False, "reason"), False),  # Tuple with False first - THE BUG CASE
            ((0,), False),  # Tuple with falsy first element
            ((1,), True),  # Tuple with truthy first element
        ],
    )
    def test_various_return_values(self, return_value, should_transition):
        """Parametrized test for various return value truthiness."""
        config = make_simple_config("test_func")

        def test_func(data, context=None):
            return return_value

        fsm = create_advanced_fsm(config, custom_functions={"test_func": test_func})
        ctx = fsm.create_context({})

        fsm.execute_step_sync(ctx)

        expected_state = "end" if should_transition else "start"
        assert ctx.current_state == expected_state, (
            f"Return value {return_value!r} (truthy={bool(return_value)}) "
            f"should {'transition' if should_transition else 'not transition'}. "
            f"Got state: {ctx.current_state}"
        )

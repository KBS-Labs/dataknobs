"""Tests for dataknobs_common.transitions module."""

import pytest

from dataknobs_common.exceptions import OperationError
from dataknobs_common.transitions import InvalidTransitionError, TransitionValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def order_validator():
    """A typical order lifecycle graph."""
    return TransitionValidator("order", {
        "draft":     {"submitted"},
        "submitted": {"approved", "rejected"},
        "approved":  {"shipped"},
        "shipped":   {"delivered"},
        "rejected":  set(),
        "delivered":  set(),
    })


@pytest.fixture
def cyclic_validator():
    """A graph with a cycle (failed -> pending -> running -> failed)."""
    return TransitionValidator("job", {
        "pending":   {"running", "cancelled"},
        "running":   {"completed", "failed"},
        "failed":    {"pending"},
        "completed": set(),
        "cancelled": set(),
    })


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------


class TestValidTransitions:

    def test_valid_transition_does_not_raise(self, order_validator):
        order_validator.validate("draft", "submitted")

    def test_multiple_valid_targets(self, order_validator):
        order_validator.validate("submitted", "approved")
        order_validator.validate("submitted", "rejected")

    def test_none_current_skips_validation(self, order_validator):
        # Should not raise for any target when current is None
        order_validator.validate(None, "submitted")
        order_validator.validate(None, "nonexistent")


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


class TestInvalidTransitions:

    def test_disallowed_transition_raises(self, order_validator):
        with pytest.raises(InvalidTransitionError) as exc_info:
            order_validator.validate("draft", "approved")

        err = exc_info.value
        assert err.entity == "order"
        assert err.current_status == "draft"
        assert err.target_status == "approved"
        assert err.allowed == {"submitted"}

    def test_terminal_status_rejects_all_targets(self, order_validator):
        with pytest.raises(InvalidTransitionError) as exc_info:
            order_validator.validate("delivered", "draft")

        err = exc_info.value
        assert err.allowed == set()

    def test_unknown_current_status_raises(self, order_validator):
        with pytest.raises(InvalidTransitionError) as exc_info:
            order_validator.validate("nonexistent", "draft")

        err = exc_info.value
        assert err.current_status == "nonexistent"
        assert err.allowed is None

    def test_backward_transition_raises(self, order_validator):
        with pytest.raises(InvalidTransitionError):
            order_validator.validate("shipped", "submitted")


# ---------------------------------------------------------------------------
# Error message content
# ---------------------------------------------------------------------------


class TestErrorMessages:

    def test_error_message_includes_entity_and_statuses(self, order_validator):
        with pytest.raises(InvalidTransitionError, match="order"):
            order_validator.validate("draft", "delivered")

    def test_error_message_for_terminal_shows_none(self, order_validator):
        with pytest.raises(InvalidTransitionError, match="none — terminal"):
            order_validator.validate("delivered", "draft")

    def test_error_message_for_unknown_status(self, order_validator):
        with pytest.raises(InvalidTransitionError, match="unknown current status"):
            order_validator.validate("bogus", "draft")

    def test_context_dict_populated(self, order_validator):
        with pytest.raises(InvalidTransitionError) as exc_info:
            order_validator.validate("draft", "shipped")

        ctx = exc_info.value.context
        assert ctx["entity"] == "order"
        assert ctx["current_status"] == "draft"
        assert ctx["target_status"] == "shipped"
        assert ctx["allowed"] == ["submitted"]  # sorted list


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:

    def test_inherits_from_operation_error(self):
        assert issubclass(InvalidTransitionError, OperationError)

    def test_caught_by_operation_error_handler(self, order_validator):
        with pytest.raises(OperationError):
            order_validator.validate("delivered", "draft")


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


class TestIntrospection:

    def test_allowed_transitions_returns_copy(self, order_validator):
        graph = order_validator.allowed_transitions
        assert graph["draft"] == {"submitted"}
        assert graph["submitted"] == {"approved", "rejected"}
        assert graph["delivered"] == set()

        # Verify it's a copy
        graph["draft"].add("hacked")
        assert "hacked" not in order_validator.allowed_transitions["draft"]

    def test_statuses_includes_all(self, order_validator):
        expected = {"draft", "submitted", "approved", "rejected", "shipped", "delivered"}
        assert order_validator.statuses == expected

    def test_name_property(self, order_validator):
        assert order_validator.name == "order"

    def test_repr(self, order_validator):
        r = repr(order_validator)
        assert "order" in r
        assert "6 statuses" in r


# ---------------------------------------------------------------------------
# get_reachable — transitive closure
# ---------------------------------------------------------------------------


class TestGetReachable:

    def test_reachable_from_start(self, order_validator):
        reachable = order_validator.get_reachable("draft")
        assert reachable == {"submitted", "approved", "rejected", "shipped", "delivered"}

    def test_reachable_from_terminal_is_empty(self, order_validator):
        reachable = order_validator.get_reachable("delivered")
        assert reachable == set()

    def test_reachable_from_mid_graph(self, order_validator):
        reachable = order_validator.get_reachable("approved")
        assert reachable == {"shipped", "delivered"}

    def test_reachable_with_cycle(self, cyclic_validator):
        reachable = cyclic_validator.get_reachable("pending")
        # pending -> running -> {completed, failed}
        # failed -> pending (cycle) -> running -> ...
        assert reachable == {"running", "completed", "failed", "pending", "cancelled"}

    def test_reachable_from_unknown_raises(self, order_validator):
        with pytest.raises(InvalidTransitionError, match="unknown current status"):
            order_validator.get_reachable("nonexistent")


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------


class TestPackageImport:

    def test_import_from_dataknobs_common(self):
        from dataknobs_common import InvalidTransitionError as ITE
        from dataknobs_common import TransitionValidator as TV

        assert ITE is InvalidTransitionError
        assert TV is TransitionValidator

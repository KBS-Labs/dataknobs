"""Stateless transition validation for declarative status graphs.

This module provides a lightweight transition validator for systems that
need to enforce valid status transitions without a full state machine
framework. It is suitable for guarding database writes, API updates, or
any operation where an entity moves between named statuses.

This is **not** a state machine — it does not manage state, execute
actions, or track lifecycle. For full FSM capabilities, see
``dataknobs_fsm``.

Example:
    ```python
    from dataknobs_common.transitions import TransitionValidator

    RUN_STATUS = TransitionValidator(
        "run_status",
        {
            "pending":   {"running", "cancelled"},
            "running":   {"completed", "failed", "cancelled"},
            "failed":    {"pending"},       # allow retry
            "completed": set(),             # terminal
            "cancelled": set(),             # terminal
        },
    )

    RUN_STATUS.validate("pending", "running")       # ok
    RUN_STATUS.validate("completed", "pending")     # raises InvalidTransitionError
    RUN_STATUS.validate(None, "running")            # ok (skip when current unknown)
    ```
"""

from __future__ import annotations

import logging

from dataknobs_common.exceptions import OperationError

logger = logging.getLogger(__name__)


class InvalidTransitionError(OperationError):
    """Raised when a status transition is not allowed.

    Attributes:
        entity: Name of the entity or transition graph (e.g. ``"run_status"``).
        current_status: The current status that was being transitioned from.
        target_status: The target status that was rejected.
        allowed: The set of statuses that are valid targets from ``current_status``,
            or ``None`` if the current status itself is unknown.
    """

    def __init__(
        self,
        entity: str,
        current_status: str,
        target_status: str,
        allowed: set[str] | None = None,
    ) -> None:
        self.entity = entity
        self.current_status = current_status
        self.target_status = target_status
        self.allowed = allowed

        if allowed is not None:
            allowed_str = ", ".join(sorted(allowed)) if allowed else "(none — terminal)"
            message = (
                f"{entity}: cannot transition from '{current_status}' to '{target_status}'. "
                f"Allowed targets: {allowed_str}"
            )
        else:
            message = (
                f"{entity}: unknown current status '{current_status}'"
            )

        super().__init__(
            message,
            context={
                "entity": entity,
                "current_status": current_status,
                "target_status": target_status,
                "allowed": sorted(allowed) if allowed else [],
            },
        )


class TransitionValidator:
    """Stateless validator for declarative transition graphs.

    Declares which status transitions are allowed and validates proposed
    transitions. Does not manage or store state — the caller owns the
    current status.

    Args:
        name: Human-readable name for this transition graph, used in
            error messages (e.g. ``"run_status"``, ``"order_state"``).
        transitions: Mapping from each status to the set of statuses it
            may transition to. Statuses that appear only as targets (not
            as keys) are implicitly terminal (no outgoing transitions).

    Example:
        ```python
        ORDER = TransitionValidator("order", {
            "draft":     {"submitted"},
            "submitted": {"approved", "rejected"},
            "approved":  {"shipped"},
            "shipped":   {"delivered"},
            "rejected":  set(),
            "delivered":  set(),
        })

        ORDER.validate("draft", "submitted")  # ok
        ORDER.validate("shipped", "draft")    # raises InvalidTransitionError
        ```
    """

    def __init__(self, name: str, transitions: dict[str, set[str]]) -> None:
        self._name = name
        self._transitions = {k: set(v) for k, v in transitions.items()}

    @property
    def name(self) -> str:
        """The name of this transition graph."""
        return self._name

    @property
    def allowed_transitions(self) -> dict[str, set[str]]:
        """Return a copy of the full transition graph."""
        return {k: set(v) for k, v in self._transitions.items()}

    @property
    def statuses(self) -> set[str]:
        """Return all known statuses (sources and targets)."""
        all_statuses: set[str] = set(self._transitions.keys())
        for targets in self._transitions.values():
            all_statuses.update(targets)
        return all_statuses

    def validate(self, current_status: str | None, target_status: str) -> None:
        """Validate a proposed transition.

        Args:
            current_status: The current status. If ``None``, validation is
                skipped (useful for callers that don't yet know the current
                state).
            target_status: The desired target status.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
        """
        if current_status is None:
            return

        allowed = self._transitions.get(current_status)
        if allowed is None:
            raise InvalidTransitionError(
                entity=self._name,
                current_status=current_status,
                target_status=target_status,
                allowed=None,
            )

        if target_status not in allowed:
            raise InvalidTransitionError(
                entity=self._name,
                current_status=current_status,
                target_status=target_status,
                allowed=allowed,
            )

    def get_reachable(self, from_status: str) -> set[str]:
        """Compute all statuses reachable from a given status (transitive closure).

        Args:
            from_status: The starting status.

        Returns:
            Set of all statuses reachable via one or more transitions.
            Does not include ``from_status`` itself unless there is a cycle.

        Raises:
            InvalidTransitionError: If ``from_status`` is not a known status.
        """
        if from_status not in self._transitions:
            raise InvalidTransitionError(
                entity=self._name,
                current_status=from_status,
                target_status="",
                allowed=None,
            )

        reachable: set[str] = set()
        frontier = set(self._transitions.get(from_status, set()))

        while frontier:
            status = frontier.pop()
            if status not in reachable:
                reachable.add(status)
                next_targets = self._transitions.get(status, set())
                frontier.update(next_targets - reachable)

        return reachable

    def __repr__(self) -> str:
        status_count = len(self.statuses)
        return f"TransitionValidator({self._name!r}, {status_count} statuses)"

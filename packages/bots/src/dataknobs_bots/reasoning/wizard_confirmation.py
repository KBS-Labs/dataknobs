"""Confirmation decision logic for wizard stages.

Owns the confirmation gate (should we confirm?) and the snapshot
lifecycle (save, compare, diff) so that ``process_input`` delegates
to a flat result rather than embedding deeply nested if/elif logic.

Extracted as part of item 87 — see the revised implementation plan
for the full design rationale.
"""

from __future__ import annotations

import logging
from typing import Any

from .wizard_types import ConfirmationEvaluation, StageSchema, WizardState

logger = logging.getLogger(__name__)


class ConfirmationEvaluator:
    """Stateless evaluator for wizard confirmation decisions.

    Pure decision logic — no LLM calls, no conversation manager
    interaction.  Designed for easy unit testing: construct a
    ``WizardState`` and stage dict, call :meth:`evaluate`, assert on
    the returned :class:`ConfirmationEvaluation`.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        stage: dict[str, Any],
        wizard_state: WizardState,
        new_data_keys: set[str],
    ) -> ConfirmationEvaluation:
        """Decide whether to show a confirmation and with which keys.

        Decision matrix (``cfr`` = ``confirm_first_render``):

        +-------+------------------+---------------+----------+----------------+---------------+
        | count | confirm_on_new   | new_data_keys | snap     | should_confirm | save_snapshot |
        +=======+==================+===============+==========+================+===============+
        | 0     | any              | non-empty     | n/a      | cfr != False   | has_cond      |
        +-------+------------------+---------------+----------+----------------+---------------+
        | 0     | any              | empty         | n/a      | False          | has_cond      |
        +-------+------------------+---------------+----------+----------------+---------------+
        | >0    | True             | any           | non-empty| True           | True          |
        +-------+------------------+---------------+----------+----------------+---------------+
        | >0    | True             | any           | empty    | False          | True          |
        +-------+------------------+---------------+----------+----------------+---------------+
        | >0    | False            | any           | any      | False          | False         |
        +-------+------------------+---------------+----------+----------------+---------------+

        ``has_cond`` = ``confirm_on_new_data`` is set on the stage.

        Args:
            stage: Current stage metadata dict.
            wizard_state: Current wizard state.
            new_data_keys: Keys produced by the extraction pipeline
                (may be empty).

        Returns:
            :class:`ConfirmationEvaluation` with the decision.
        """
        no_confirm = ConfirmationEvaluation(
            should_confirm=False,
            confirm_keys=frozenset(),
            should_save_snapshot=False,
            snapshot_diff_keys=frozenset(),
        )

        # Gate: response_template is required for any confirmation path.
        if not stage.get("response_template"):
            return no_confirm

        stage_name = self._stage_name(stage)
        has_confirm_on_new_data = bool(stage.get("confirm_on_new_data"))
        render_count = wizard_state.get_render_count(stage_name)

        if render_count == 0:
            # First render — confirm when new data exists, unless the
            # stage explicitly opts out via confirm_first_render=False.
            if new_data_keys and stage.get("confirm_first_render", True) is not False:
                return ConfirmationEvaluation(
                    should_confirm=True,
                    confirm_keys=frozenset(new_data_keys),
                    should_save_snapshot=has_confirm_on_new_data,
                    snapshot_diff_keys=frozenset(),
                )
            # No confirmation, but save a baseline snapshot if
            # confirm_on_new_data is set so that the next turn's
            # diff compares against actual values, not an empty dict.
            return ConfirmationEvaluation(
                should_confirm=False,
                confirm_keys=frozenset(),
                should_save_snapshot=has_confirm_on_new_data,
                snapshot_diff_keys=frozenset(),
            )

        # render_count > 0 — template was already shown at least once.
        if has_confirm_on_new_data:
            diff_keys = self.compute_snapshot_diff(stage, wizard_state)
            if diff_keys:
                return ConfirmationEvaluation(
                    should_confirm=True,
                    confirm_keys=frozenset(new_data_keys | diff_keys),
                    should_save_snapshot=True,
                    snapshot_diff_keys=frozenset(diff_keys),
                )
            # Snapshot unchanged — no re-confirmation needed.
            return ConfirmationEvaluation(
                should_confirm=False,
                confirm_keys=frozenset(),
                should_save_snapshot=True,
                snapshot_diff_keys=frozenset(),
            )

        # No confirm_on_new_data — previously rendered stages skip
        # confirmation so the user's response triggers a transition.
        return no_confirm

    def compute_snapshot_diff(
        self,
        stage: dict[str, Any],
        wizard_state: WizardState,
    ) -> set[str]:
        """Return schema property names whose values changed since the last snapshot.

        Detects both value changes (key present in both snapshots with
        different values) and field deletions (key present in prior
        snapshot but absent or ``None`` in current state).

        Args:
            stage: Current stage metadata dict.
            wizard_state: Current wizard state.

        Returns:
            Set of property names where the current value differs from
            the saved snapshot.  Empty if snapshots match or no schema.
        """
        stage_name = self._stage_name(stage)
        ss_props = StageSchema.from_stage(stage).property_names
        current = {
            k: wizard_state.data[k]
            for k in ss_props
            if k in wizard_state.data and wizard_state.data[k] is not None
        }
        prior = wizard_state.get_stage_snapshot(stage_name)
        # Keys whose values changed
        changed = {
            k
            for k in current
            if current.get(k) != (prior or {}).get(k)
        }
        # Keys present in prior snapshot but now absent/None (cleared)
        cleared = set((prior or {}).keys()) - set(current.keys())
        return changed | cleared

    def save_snapshot(
        self,
        stage: dict[str, Any],
        wizard_state: WizardState,
    ) -> None:
        """Save current schema property values as the stage snapshot.

        Single entry point for snapshot persistence — callers should
        not call ``wizard_state.save_stage_snapshot`` directly.

        Args:
            stage: Current stage metadata dict.
            wizard_state: Current wizard state.
        """
        stage_name = self._stage_name(stage)
        ss_props = StageSchema.from_stage(stage).property_names
        wizard_state.save_stage_snapshot(stage_name, ss_props)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_name(stage: dict[str, Any]) -> str:
        """Extract stage name with fallback."""
        return stage.get("name", "unknown")

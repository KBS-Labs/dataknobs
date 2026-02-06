"""Tests for wizard stages roadmap building."""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import SubflowContext, WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


@pytest.fixture
def labeled_wizard_config() -> dict[str, Any]:
    """Wizard config with stage labels."""
    return {
        "name": "labeled-wizard",
        "version": "1.0",
        "stages": [
            {
                "name": "welcome",
                "is_start": True,
                "label": "Welcome",
                "prompt": "Hello!",
                "transitions": [{"target": "configure"}],
            },
            {
                "name": "configure",
                "label": "Configuration",
                "prompt": "Set up...",
                "transitions": [{"target": "review"}],
            },
            {
                "name": "review",
                "label": "Review",
                "prompt": "Review...",
                "transitions": [{"target": "complete"}],
            },
            {
                "name": "complete",
                "is_end": True,
                "label": "Done",
                "prompt": "All done!",
            },
        ],
    }


@pytest.fixture
def labeled_wizard_reasoning(
    labeled_wizard_config: dict[str, Any],
) -> WizardReasoning:
    """WizardReasoning with labeled stages."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(labeled_wizard_config)
    return WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)


class TestBuildStagesRoadmap:
    """Tests for _build_stages_roadmap."""

    def test_at_start(
        self, labeled_wizard_reasoning: WizardReasoning
    ) -> None:
        """At start, only first stage is current, rest are pending."""
        state = WizardState(
            current_stage="welcome",
            history=["welcome"],
        )

        roadmap = labeled_wizard_reasoning._build_stages_roadmap(state)

        assert len(roadmap) == 4
        assert roadmap[0] == {"name": "welcome", "label": "Welcome", "status": "current"}
        assert roadmap[1]["status"] == "pending"
        assert roadmap[2]["status"] == "pending"
        assert roadmap[3]["status"] == "pending"

    def test_linear_progression(
        self, labeled_wizard_reasoning: WizardReasoning
    ) -> None:
        """Visited stages are completed, current is current, rest pending."""
        state = WizardState(
            current_stage="review",
            history=["welcome", "configure", "review"],
        )

        roadmap = labeled_wizard_reasoning._build_stages_roadmap(state)

        assert roadmap[0] == {"name": "welcome", "label": "Welcome", "status": "completed"}
        assert roadmap[1] == {"name": "configure", "label": "Configuration", "status": "completed"}
        assert roadmap[2] == {"name": "review", "label": "Review", "status": "current"}
        assert roadmap[3] == {"name": "complete", "label": "Done", "status": "pending"}

    def test_stage_labels_included(
        self, labeled_wizard_reasoning: WizardReasoning
    ) -> None:
        """Labels from config are present in roadmap entries."""
        state = WizardState(
            current_stage="welcome",
            history=["welcome"],
        )

        roadmap = labeled_wizard_reasoning._build_stages_roadmap(state)

        labels = [s["label"] for s in roadmap]
        assert labels == ["Welcome", "Configuration", "Review", "Done"]

    def test_label_falls_back_to_name(
        self, wizard_reasoning: WizardReasoning
    ) -> None:
        """When no label is set, name is used as label."""
        state = WizardState(
            current_stage="welcome",
            history=["welcome"],
        )

        roadmap = wizard_reasoning._build_stages_roadmap(state)

        # simple_wizard_config has no labels, so name is used
        assert roadmap[0]["label"] == "welcome"

    def test_subflow_marks_parent_as_current(
        self, labeled_wizard_reasoning: WizardReasoning
    ) -> None:
        """During subflow, parent stage is shown as current."""
        state = WizardState(
            current_stage="sub_step_1",  # inside subflow
            history=["sub_step_1"],  # subflow history (reset at push)
            subflow_stack=[
                SubflowContext(
                    parent_stage="configure",
                    parent_data={},
                    parent_history=["welcome", "configure"],
                    return_stage="review",
                    result_mapping={},
                    subflow_network="child_flow",
                )
            ],
        )

        roadmap = labeled_wizard_reasoning._build_stages_roadmap(state)

        # welcome was visited before subflow → completed
        assert roadmap[0] == {"name": "welcome", "label": "Welcome", "status": "completed"}
        # configure is parent stage → current (subflow in progress)
        assert roadmap[1] == {"name": "configure", "label": "Configuration", "status": "current"}
        # review and complete are pending
        assert roadmap[2]["status"] == "pending"
        assert roadmap[3]["status"] == "pending"

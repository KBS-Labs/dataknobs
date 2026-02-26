"""Tests for multi-bank wizard flows and advanced operations.

Covers Phase 5: multiple banks, cross-bank conditions, bank access
in TransformContext, rich template rendering.
"""

from __future__ import annotations

from typing import Any

import jinja2
import pytest

from dataknobs_bots.artifacts.transforms import TransformContext
from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_data.backends.memory import SyncMemoryDatabase


# =====================================================================
# Helpers
# =====================================================================

def _make_multi_bank_wizard() -> WizardReasoning:
    """Create a wizard with 3 banks: team, milestones, budget."""
    config: dict[str, Any] = {
        "name": "project-wizard",
        "version": "1.0",
        "settings": {
            "banks": {
                "team": {
                    "schema": {"required": ["name", "role"]},
                    "max_records": 20,
                },
                "milestones": {
                    "schema": {"required": ["title"]},
                },
                "budget": {
                    "schema": {"required": ["item", "cost"]},
                },
            },
        },
        "stages": [
            {
                "name": "add_team",
                "is_start": True,
                "prompt": "Add a team member",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                    },
                    "required": ["name", "role"],
                },
                "transitions": [
                    {
                        "target": "add_milestones",
                        "condition": "bank('team').count() > 0",
                    },
                ],
            },
            {
                "name": "add_milestones",
                "prompt": "Add milestones",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                    },
                    "required": ["title"],
                },
                "transitions": [
                    {
                        "target": "review",
                        "condition": (
                            "bank('team').count() > 0 "
                            "and bank('milestones').count() > 0"
                        ),
                    },
                ],
            },
            {
                "name": "review",
                "is_end": True,
                "prompt": "Project review",
                "response_template": (
                    "Team: {{ bank('team').count() }} members\n"
                    "Milestones: {{ bank('milestones').count() }}\n"
                    "Budget items: {{ bank('budget').count() }}"
                ),
            },
        ],
    }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


# =====================================================================
# Multi-bank initialisation
# =====================================================================

class TestMultiBankInit:

    def test_all_banks_created(self) -> None:
        reasoning = _make_multi_bank_wizard()
        assert "team" in reasoning._banks
        assert "milestones" in reasoning._banks
        assert "budget" in reasoning._banks

    def test_banks_are_independent(self) -> None:
        reasoning = _make_multi_bank_wizard()
        reasoning._banks["team"].add({"name": "Alice", "role": "Lead"})
        reasoning._banks["milestones"].add({"title": "Sprint 1"})
        assert reasoning._banks["team"].count() == 1
        assert reasoning._banks["milestones"].count() == 1
        assert reasoning._banks["budget"].count() == 0

    def test_bank_schemas_differ(self) -> None:
        reasoning = _make_multi_bank_wizard()
        assert reasoning._banks["team"].schema == {
            "required": ["name", "role"]
        }
        assert reasoning._banks["milestones"].schema == {
            "required": ["title"]
        }


# =====================================================================
# Cross-bank conditions
# =====================================================================

class TestCrossBankConditions:

    def test_cross_bank_condition_false_when_empty(self) -> None:
        reasoning = _make_multi_bank_wizard()
        result = reasoning._evaluate_condition(
            "bank('team').count() > 0 and bank('milestones').count() > 0",
            {},
        )
        assert result is False

    def test_cross_bank_condition_false_when_partial(self) -> None:
        reasoning = _make_multi_bank_wizard()
        reasoning._banks["team"].add({"name": "Alice", "role": "Lead"})
        result = reasoning._evaluate_condition(
            "bank('team').count() > 0 and bank('milestones').count() > 0",
            {},
        )
        assert result is False

    def test_cross_bank_condition_true_when_both_populated(self) -> None:
        reasoning = _make_multi_bank_wizard()
        reasoning._banks["team"].add({"name": "Alice", "role": "Lead"})
        reasoning._banks["milestones"].add({"title": "Sprint 1"})
        result = reasoning._evaluate_condition(
            "bank('team').count() > 0 and bank('milestones').count() > 0",
            {},
        )
        assert result is True


# =====================================================================
# Template rendering with banks
# =====================================================================

class TestBankTemplateRendering:

    def test_multi_bank_count_in_template(self) -> None:
        reasoning = _make_multi_bank_wizard()
        reasoning._banks["team"].add({"name": "Alice", "role": "Lead"})
        reasoning._banks["team"].add({"name": "Bob", "role": "Dev"})
        reasoning._banks["milestones"].add({"title": "Sprint 1"})

        accessor = reasoning._make_bank_accessor()
        env = jinja2.Environment(undefined=jinja2.Undefined)
        template = env.from_string(
            "Team: {{ bank('team').count() }} members\n"
            "Milestones: {{ bank('milestones').count() }}\n"
            "Budget items: {{ bank('budget').count() }}"
        )
        result = template.render(bank=accessor)
        assert "Team: 2 members" in result
        assert "Milestones: 1" in result
        assert "Budget items: 0" in result

    def test_iterate_bank_records_in_template(self) -> None:
        reasoning = _make_multi_bank_wizard()
        reasoning._banks["team"].add({"name": "Alice", "role": "Lead"})
        reasoning._banks["team"].add({"name": "Bob", "role": "Dev"})

        accessor = reasoning._make_bank_accessor()
        env = jinja2.Environment(undefined=jinja2.Undefined)
        template = env.from_string(
            "{% for m in bank('team').all() %}"
            "- {{ m.data.name }}: {{ m.data.role }}\n"
            "{% endfor %}"
        )
        result = template.render(bank=accessor)
        assert "- Alice: Lead" in result
        assert "- Bob: Dev" in result


# =====================================================================
# Banks in TransformContext
# =====================================================================

class TestBanksInTransformContext:

    def test_transform_context_has_banks(self) -> None:
        reasoning = _make_multi_bank_wizard()
        reasoning._banks["team"].add({"name": "Alice", "role": "Lead"})

        ctx = TransformContext(banks=reasoning._banks)
        assert "team" in ctx.banks
        assert ctx.banks["team"].count() == 1

    def test_transform_can_read_from_bank(self) -> None:
        reasoning = _make_multi_bank_wizard()
        reasoning._banks["team"].add({"name": "Alice", "role": "Lead"})

        ctx = TransformContext(banks=reasoning._banks)
        records = ctx.banks["team"].all()
        assert len(records) == 1
        assert records[0].data["name"] == "Alice"

    def test_transform_can_write_to_bank(self) -> None:
        reasoning = _make_multi_bank_wizard()
        ctx = TransformContext(banks=reasoning._banks)

        # Simulate a transform writing to a bank
        ctx.banks["budget"].add({"item": "Server", "cost": "500"})
        assert reasoning._banks["budget"].count() == 1

    def test_empty_banks_default(self) -> None:
        ctx = TransformContext()
        assert ctx.banks == {}


# =====================================================================
# Per-stage bank targeting
# =====================================================================

class TestPerStageBankTargeting:

    def test_different_stages_target_different_banks(self) -> None:
        reasoning = _make_multi_bank_wizard()
        # Stage 1 targets "team"
        reasoning._banks["team"].add(
            {"name": "Alice", "role": "Lead"},
            source_stage="add_team",
        )
        # Stage 2 targets "milestones"
        reasoning._banks["milestones"].add(
            {"title": "Sprint 1"},
            source_stage="add_milestones",
        )

        team_records = reasoning._banks["team"].all()
        milestone_records = reasoning._banks["milestones"].all()
        assert all(r.source_stage == "add_team" for r in team_records)
        assert all(
            r.source_stage == "add_milestones" for r in milestone_records
        )

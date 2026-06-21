"""Smoke tests for the shipped example wizard configs.

Each YAML under ``examples/configs/wizards/`` must survive the full
load pipeline (synthesize -> validate -> translate to FSM). This guards
the examples from rotting as the wizard config surface evolves — a
broken example would otherwise only surface when a reader tried to run
it. ``WizardConfigLoader().load`` exercises the same path a consumer
hits, including the ``intent_confirm:`` synthesizer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.reasoning.wizard_fsm import WizardFSM
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

_EXAMPLE_DIR = (
    Path(__file__).parent.parent / "examples" / "configs" / "wizards"
)


def _example_wizards() -> list[Path]:
    return sorted(_EXAMPLE_DIR.glob("*.yaml"))


def test_example_dir_is_populated() -> None:
    """Guard against the glob silently matching nothing."""
    assert _example_wizards(), f"No example wizards found in {_EXAMPLE_DIR}"


@pytest.mark.parametrize(
    "config_path", _example_wizards(), ids=lambda p: p.name
)
def test_example_wizard_loads(config_path: Path) -> None:
    """Each example loads end-to-end without raising."""
    fsm = WizardConfigLoader().load(config_path)
    assert isinstance(fsm, WizardFSM)

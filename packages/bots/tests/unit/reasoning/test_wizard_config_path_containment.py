"""A ``wizard_config`` path is bounded by the ``config_base_path`` beside it.

``WizardReasoning.from_config`` composes ``config_base_path / wizard_config``
and hands the result to ``WizardConfigLoader.load``, which opens it. Both
operands come out of the bot's typed config, so the provenance is identical to
the ``subflow.network`` name one call further down — which *is* bounded. This
is that same boundary, one frame up: it is where the tree the loader threads is
established, not a separate question about a different kind of name.

Declaring no ``config_base_path`` declares no tree, and nothing is bounded.
That is the honest spelling of "this path is not inside anything" — and it is
the migration for a deployment that genuinely wants an absolute
``wizard_config``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dataknobs_common.paths import PathEscapeError

from dataknobs_bots.reasoning.wizard import WizardReasoning

_WIZARD_YAML = """\
name: reachable-wizard
version: '1.0'
stages:
  - name: only
    is_start: true
    is_end: true
    prompt: 'from the wizard'
"""

# Matches the containment failure specifically. `pydantic.ValidationError` is
# also a `ValueError`, so a bare `ValueError` assertion here would pass on an
# unrelated malformed config.
_ESCAPE = "outside"


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """The declared tree — what a relative ``wizard_config`` is bounded to."""
    d = tmp_path / "bots"
    d.mkdir()
    (d / "inside.yaml").write_text(_WIZARD_YAML)
    return d


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A loadable wizard the bot config must not reach."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "other-wizard.yaml").write_text(_WIZARD_YAML)
    return other / "other-wizard.yaml"


def test_a_wizard_config_may_not_walk_out_of_the_declared_base(base: Path, outside: Path) -> None:
    with pytest.raises(PathEscapeError, match=_ESCAPE):
        WizardReasoning.from_config(
            {
                "wizard_config": "../elsewhere/other-wizard.yaml",
                "config_base_path": str(base),
            }
        )


def test_an_absolute_wizard_config_may_not_discard_the_declared_base(
    base: Path, outside: Path
) -> None:
    """The branch that never consulted the base at all.

    ``is_absolute()`` skipped the composition entirely, so a deployment that
    declared a tree could still be pointed at any file on the volume by a
    config value. The provenance is the same in both spellings, and #571 bounds
    the absolute spelling of ``extends:`` for exactly this reason.
    """
    with pytest.raises(PathEscapeError, match=_ESCAPE):
        WizardReasoning.from_config({"wizard_config": str(outside), "config_base_path": str(base)})


def test_a_relative_wizard_config_inside_the_base_still_loads(base: Path) -> None:
    reasoning = WizardReasoning.from_config(
        {"wizard_config": "inside.yaml", "config_base_path": str(base)}
    )

    assert reasoning is not None


def test_a_wizard_config_in_a_subdirectory_still_loads(base: Path) -> None:
    """Descending is legal; only leaving the tree is not."""
    (base / "flows").mkdir()
    (base / "flows" / "nested.yaml").write_text(_WIZARD_YAML)

    reasoning = WizardReasoning.from_config(
        {"wizard_config": "flows/nested.yaml", "config_base_path": str(base)}
    )

    assert reasoning is not None


def test_with_no_declared_base_an_absolute_path_is_unbounded(outside: Path) -> None:
    """No ``config_base_path`` means no tree, so there is nothing to leave.

    Pinned as the deliberate reading, not an oversight: bounding a path
    against a base nobody declared would have to invent one, and the only
    candidates (the process CWD, the package root) are worse than no boundary
    because they vary with how the bot happens to be launched.
    """
    reasoning = WizardReasoning.from_config({"wizard_config": str(outside)})

    assert reasoning is not None


def test_the_refusal_names_the_value_the_config_supplied(base: Path, outside: Path) -> None:
    with pytest.raises(PathEscapeError) as excinfo:
        WizardReasoning.from_config(
            {
                "wizard_config": "../elsewhere/other-wizard.yaml",
                "config_base_path": str(base),
            }
        )

    assert "wizard_config" in str(excinfo.value)
    assert "../elsewhere/other-wizard.yaml" in str(excinfo.value)


def test_the_declared_base_becomes_the_tree_the_loader_threads(base: Path, outside: Path) -> None:
    """S9 establishes the root; the subflow guard below inherits it.

    The wizard sits in a subdirectory of the declared base, and names a
    subflow above itself. That is inside the tree and must load — which it
    only does if this frame passed its own base down as the root rather than
    letting the loader re-anchor on the wizard file's own directory.
    """
    (base / "shared.yaml").write_text(_WIZARD_YAML)
    (base / "flows").mkdir()
    (base / "flows" / "host.yaml").write_text(
        """\
name: host-wizard
version: '1.0'
stages:
  - name: start
    is_start: true
    prompt: 'go'
    response_template: 'go'
    transitions:
      - target: _subflow
        subflow:
          network: '../shared'
  - name: done
    is_end: true
    prompt: 'done'
    response_template: 'done'
"""
    )

    reasoning = WizardReasoning.from_config(
        {"wizard_config": "flows/host.yaml", "config_base_path": str(base)}
    )

    assert reasoning is not None


def test_a_subflow_still_may_not_leave_the_declared_base(base: Path, outside: Path) -> None:
    (base / "flows").mkdir()
    (base / "flows" / "host.yaml").write_text(
        """\
name: host-wizard
version: '1.0'
stages:
  - name: start
    is_start: true
    prompt: 'go'
    response_template: 'go'
    transitions:
      - target: _subflow
        subflow:
          network: '../../elsewhere/other-wizard'
  - name: done
    is_end: true
    prompt: 'done'
    response_template: 'done'
"""
    )

    with pytest.raises(PathEscapeError, match=_ESCAPE):
        WizardReasoning.from_config(
            {"wizard_config": "flows/host.yaml", "config_base_path": str(base)}
        )

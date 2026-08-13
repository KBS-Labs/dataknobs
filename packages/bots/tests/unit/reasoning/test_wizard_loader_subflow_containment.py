"""A subflow name must not address a config outside the wizard's own tree.

``_load_single_subflow`` composes a ``subflow_name`` onto
``config_base_path`` twice — ``{base}/{name}.yaml`` and
``{base}/subflows/{name}.yaml`` — and loads whichever exists.

The name is not a caller argument. It is read out of config *content*:
either a ``subflows:`` key, or a transition's ``subflow.network`` value.
So the provenance is a YAML file, and a wizard config that names
``../../elsewhere/other-wizard`` pulls a state machine in from outside
the directory the wizard was loaded from — with its transitions,
transforms and function references.

``load()`` sets ``config_base_path`` to the loaded file's parent, so the
tests below go through the public entry point rather than reaching for
the private composer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

_SUBFLOW_YAML = """\
name: reachable-subflow
version: '1.0'
stages:
  - name: only
    is_start: true
    is_end: true
    prompt: 'from the subflow'
"""

# Matches the containment failure specifically. Without it these tests
# also pass on pydantic's ValidationError (a ValueError subclass) raised
# for an unrelated malformed config -- which is how the first draft of
# this file reported green against unguarded code.
_ESCAPE = "outside"


def _wizard_naming(subflow: str) -> str:
    """A wizard whose single transition targets ``subflow`` by name."""
    return f"""\
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
          network: {subflow!r}
  - name: done
    is_end: true
    prompt: 'done'
    response_template: 'done'
"""


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Where the wizard lives — the boundary a subflow name must respect."""
    cfg = tmp_path / "wizards"
    cfg.mkdir()
    return cfg


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A loadable wizard config the host wizard must not reach."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "other-wizard.yaml").write_text(_SUBFLOW_YAML)
    return other


def test_a_subflow_name_cannot_walk_out_of_the_config_directory(
    config_dir: Path, outside: Path
) -> None:
    host = config_dir / "host.yaml"
    host.write_text(_wizard_naming("../elsewhere/other-wizard"))

    with pytest.raises(ValueError, match=_ESCAPE):
        WizardConfigLoader().load(host)


def test_an_absolute_subflow_name_cannot_replace_the_config_directory(
    config_dir: Path, outside: Path
) -> None:
    """An absolute name discards the base; rejecting ``..`` alone misses it."""
    host = config_dir / "host.yaml"
    host.write_text(_wizard_naming(str(outside / "other-wizard")))

    with pytest.raises(ValueError, match=_ESCAPE):
        WizardConfigLoader().load(host)


def test_the_subflows_subdirectory_probe_is_bounded_too(config_dir: Path, outside: Path) -> None:
    """The second composition inserts ``subflows/``, which a ``..`` undoes.

    ``{base}/subflows/../../elsewhere/x`` leaves the base just as the
    first probe does, so guarding only the first would leave a live path.
    The name is spelled with one more ``..`` than the first probe needs,
    so only this second composition resolves onto the outside file — and
    ``subflows/`` has to exist for the kernel to walk through it, which
    is why the directory is created here.
    """
    (config_dir / "subflows").mkdir()
    host = config_dir / "host.yaml"
    host.write_text(_wizard_naming("../../elsewhere/other-wizard"))

    with pytest.raises(ValueError, match=_ESCAPE):
        WizardConfigLoader().load(host)


def test_a_subflow_in_the_subflows_subdirectory_still_loads(config_dir: Path) -> None:
    """The nested layout the second probe exists to serve."""
    (config_dir / "subflows").mkdir()
    (config_dir / "subflows" / "nested.yaml").write_text(_SUBFLOW_YAML)
    host = config_dir / "host.yaml"
    host.write_text(_wizard_naming("nested"))

    fsm = WizardConfigLoader().load(host)

    assert fsm is not None


def test_a_subflow_beside_the_wizard_still_loads(config_dir: Path) -> None:
    """The first probe's normal case."""
    (config_dir / "sibling.yaml").write_text(_SUBFLOW_YAML)
    host = config_dir / "host.yaml"
    host.write_text(_wizard_naming("sibling"))

    fsm = WizardConfigLoader().load(host)

    assert fsm is not None


def test_an_inline_subflow_definition_is_unaffected(config_dir: Path) -> None:
    """An inline ``subflows:`` entry never composes a path at all."""
    host = config_dir / "host.yaml"
    host.write_text(
        _wizard_naming("inline")
        + """
subflows:
  inline:
    name: inline-subflow
    version: '1.0'
    stages:
      - name: only
        is_start: true
        is_end: true
        prompt: 'inline'
"""
    )

    fsm = WizardConfigLoader().load(host)

    assert fsm is not None

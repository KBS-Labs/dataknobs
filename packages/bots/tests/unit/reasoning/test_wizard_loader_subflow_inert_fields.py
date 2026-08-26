"""A pushed subflow's ``settings:`` block is parsed, held, and never read.

``WizardReasoning`` hoists every setting once, off the **top-level** FSM,
at construction: each one is captured into a collaborator that outlives
any push -- the extractor holds ``extraction_scope``, the navigator holds
the merged navigation config, the banks are built from ``banks:``.
Nothing re-reads ``.settings`` off the FSM a push made active, so a
subflow's own block is in force at no point, including while its own
stage is current.

Honouring it would mean rebuilding that collaborator graph on every push
and pop, which is out of proportion to the gap, so the config is answered
where it is authored instead: the loader says so at load time, and points
at the per-stage fields that *are* read from the active flow.

The check has to know it is looking at a subflow, which the config cannot
say -- the same file loaded on its own is a wizard, and its settings are
honoured. So ``is_subflow`` is set by the loader at its own two recursion
sites and appears in no schema; these tests drive both of them, and the
counterfactuals pin that the flag is not simply always on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

# Matches the subflow-settings warning specifically. A bare "settings"
# also matches the unrecognized-field warning, which fires for a
# different reason and would report green against unguarded code.
_INERT = "never read"


def _subflow(**overrides: Any) -> dict[str, Any]:
    """A loadable one-stage subflow, with ``settings:`` unless told otherwise."""
    config: dict[str, Any] = {
        "name": "helper",
        "settings": {"extraction_scope": "current_message"},
        "stages": [
            {
                "name": "only",
                "is_start": True,
                "is_end": True,
                "prompt": "from the subflow",
                "response_template": "from the subflow",
            }
        ],
    }
    config.update(overrides)
    return config


def _host(subflows: dict[str, Any] | None = None) -> dict[str, Any]:
    """A wizard whose single transition pushes the subflow named ``helper``."""
    config: dict[str, Any] = {
        "name": "host",
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "prompt": "go",
                "response_template": "go",
                "transitions": [
                    {"target": "_subflow", "subflow": {"network": "helper"}},
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "done",
                "response_template": "done",
            },
        ],
    }
    if subflows is not None:
        config["subflows"] = subflows
    return config


def _yaml(config: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(config)


def test_a_subflow_declaring_settings_warns_at_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The inline recursion site -- a ``subflows:`` key in the host config."""
    loader = WizardConfigLoader()

    with caplog.at_level(logging.WARNING):
        loader.load_from_dict(_host({"helper": _subflow()}))

    assert any(_INERT in r.message and "settings" in r.message for r in caplog.records)


def test_a_file_backed_subflow_warns_too(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The second recursion site, reached through ``load``.

    It goes through ``load`` rather than ``load_from_dict``, so the flag
    has to survive one more hop than the inline case does.
    """
    (tmp_path / "helper.yaml").write_text(_yaml(_subflow()))
    host = tmp_path / "host.yaml"
    host.write_text(_yaml(_host()))
    loader = WizardConfigLoader()

    with caplog.at_level(logging.WARNING):
        loader.load(host)

    assert any(_INERT in r.message and "settings" in r.message for r in caplog.records)


def test_a_subflow_of_a_subflow_warns_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Depth does not exempt: everything below the top level is pushed."""
    inner = _subflow(name="inner")
    middle = _subflow(name="middle")
    middle.pop("settings")
    middle["stages"][0]["is_end"] = False
    middle["stages"][0]["transitions"] = [
        {"target": "_subflow", "subflow": {"network": "inner"}},
    ]
    middle["subflows"] = {"inner": inner}
    loader = WizardConfigLoader()

    with caplog.at_level(logging.WARNING):
        loader.load_from_dict(_host({"helper": middle}))

    assert any(_INERT in r.message and "inner" in r.message for r in caplog.records), (
        "the innermost subflow's settings are as unread as the first level's"
    )


def test_a_top_level_wizard_declaring_settings_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The counterfactual that makes the check mean something.

    The same block, in the same file, loaded as a wizard rather than
    pushed: every setting is honoured, so saying it is unread would be
    false. This is why the flag exists rather than a plain check.
    """
    loader = WizardConfigLoader()

    with caplog.at_level(logging.WARNING):
        loader.load_from_dict(_subflow())

    assert not any(_INERT in r.message for r in caplog.records)


def test_a_subflow_without_settings_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A subflow that declares nothing has nothing discarded."""
    without = _subflow()
    without.pop("settings")
    loader = WizardConfigLoader()

    with caplog.at_level(logging.WARNING):
        loader.load_from_dict(_host({"helper": without}))

    assert not any(_INERT in r.message for r in caplog.records)


def test_a_settings_block_with_non_string_keys_still_only_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The keys are authored, so their types are not ours to assume.

    A ``settings:`` block whose key is written ``1:`` is ordinary YAML --
    an unquoted numeric key is an ``int``, not a string -- and naming the
    keys in the warning
    sorted and joined them. Both steps refuse a non-string key, and the
    exception does not stop at this check: ``_load_subflow_networks``
    catches it and re-raises, so the whole wizard fails to load. A check
    that exists to advise about a config must not be the thing that
    refuses it.
    """
    loader = WizardConfigLoader()

    with caplog.at_level(logging.WARNING):
        loader.load_from_dict(_host({"helper": _subflow(settings={1: "foo", "b": 2})}))

    assert any(_INERT in r.message for r in caplog.records)


def test_a_wrong_typed_settings_block_still_only_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ``settings:`` that is not a mapping at all is shown, not iterated."""
    loader = WizardConfigLoader()

    with caplog.at_level(logging.WARNING):
        loader.load_from_dict(_host({"helper": _subflow(settings="extraction_scope")}))

    assert any(_INERT in r.message for r in caplog.records)

"""``SaveConfigTool`` writes through the manager's guard, not beside it.

``_persist_config`` reached into ``draft_manager.output_dir`` and composed
``output_dir / f"{name}.yaml"`` itself, then re-checked the result with
``resolve().is_relative_to``. Two problems followed from the composition
being local:

* the re-check guarded only this method's own ``open()``. The
  ``finalize()`` call above it writes through the draft manager, which
  had no guard at all — so the check ran *after* a write it could not
  see, which is the ordering the containment-check row is about;
* it was a third containment idiom, alongside ``_is_safe_config_name``
  and the manager's own composition, with no shared definition of what
  "inside" means.

Both close by having the manager own the composition. The entry-point
check on :meth:`SaveConfigTool.execute_with_context` stays, but as a
*naming policy* that returns a structured tool error the LLM can act on
— not as the containment boundary. These tests exercise the path that
policy does not cover: a direct ``_persist_config`` call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.config.drafts import ConfigDraftManager
from dataknobs_bots.tools.config_tools import SaveConfigTool


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out


@pytest.fixture
def tool(output_dir: Path) -> SaveConfigTool:
    return SaveConfigTool(draft_manager=ConfigDraftManager(output_dir=output_dir))


def test_persist_config_refuses_a_name_that_walks_out(tool: SaveConfigTool, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tool._persist_config("../escaped", None, {"bot": {"name": "x"}})

    assert not (tmp_path / "escaped.yaml").exists()


def test_persist_config_refuses_an_absolute_name(tool: SaveConfigTool, tmp_path: Path) -> None:
    """The re-check this replaces used ``resolve()``, which follows symlinks.

    ``safe_join`` is lexical, so it also answers for a path that does not
    exist yet — which is every path about to be written.
    """
    target = tmp_path / "escaped-absolute"

    with pytest.raises(ValueError):
        tool._persist_config(str(target), None, {"bot": {"name": "x"}})

    assert not target.with_suffix(".yaml").exists()


def test_persist_config_refuses_an_escaping_name_before_finalizing_the_draft(
    tool: SaveConfigTool, output_dir: Path, tmp_path: Path
) -> None:
    """The draft write happens first, so the guard has to precede it.

    ``_persist_config`` calls ``finalize(draft_id, final_name=name)``
    before composing its own path. A guard that only covered the local
    composition would let the manager's write land outside first.
    """
    draft_id = tool._draft_manager.create_draft({"bot": {"name": "x"}})

    with pytest.raises(ValueError):
        tool._persist_config("../escaped-via-finalize", draft_id, {"bot": {"name": "x"}})

    assert not (tmp_path / "escaped-via-finalize.yaml").exists()
    # The draft is untouched: the call failed closed rather than part-way.
    assert (output_dir / f"_draft-{draft_id}.yaml").exists()


def test_persist_config_writes_an_ordinary_name(tool: SaveConfigTool, output_dir: Path) -> None:
    written = tool._persist_config("my-bot", None, {"bot": {"name": "x"}})

    assert written == output_dir / "my-bot.yaml"
    assert written.exists()

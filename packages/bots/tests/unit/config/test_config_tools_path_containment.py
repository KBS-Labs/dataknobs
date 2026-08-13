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

**Which of these reproduce, and which guard.** Only
``test_..._before_finalizing_the_draft`` reproduced the ordering defect:
the deleted re-check did refuse a bare escaping ``name`` before its own
``open()``, so the first two tests would have passed against the
unfixed tree on containment alone. What they discriminate now is the
error *type* — the old sites raised a bare ``ValueError``, which a
consumer cannot tell from an unrelated one on the same call, and this
suite was itself caught by that ambiguity elsewhere. Asserting
:class:`~dataknobs_common.paths.PathEscapeError` fails against the
unfixed tree and keeps the containment coverage as a regression guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dataknobs_common.paths import PathEscapeError

from dataknobs_bots.config.drafts import ConfigDraftManager
from dataknobs_bots.tools.config_tools import SaveConfigTool

from .test_config_tools import _make_context


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out


@pytest.fixture
def tool(output_dir: Path) -> SaveConfigTool:
    return SaveConfigTool(draft_manager=ConfigDraftManager(output_dir=output_dir))


def test_persist_config_refuses_a_name_that_walks_out(tool: SaveConfigTool, tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        tool._persist_config("../escaped", None, {"bot": {"name": "x"}})

    assert not (tmp_path / "escaped.yaml").exists()


def test_persist_config_refuses_an_absolute_name(tool: SaveConfigTool, tmp_path: Path) -> None:
    """The re-check this replaces used ``resolve()``, which follows symlinks.

    ``safe_join`` is lexical, so it also answers for a path that does not
    exist yet — which is every path about to be written.
    """
    target = tmp_path / "escaped-absolute"

    with pytest.raises(PathEscapeError):
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

    with pytest.raises(PathEscapeError):
        tool._persist_config("../escaped-via-finalize", draft_id, {"bot": {"name": "x"}})

    assert not (tmp_path / "escaped-via-finalize.yaml").exists()
    # The draft is untouched: the call failed closed rather than part-way.
    assert (output_dir / f"_draft-{draft_id}.yaml").exists()


def test_persist_config_writes_an_ordinary_name(tool: SaveConfigTool, output_dir: Path) -> None:
    written = tool._persist_config("my-bot", None, {"bot": {"name": "x"}})

    assert written == output_dir / "my-bot.yaml"
    assert written.exists()


async def test_an_escaping_draft_id_returns_a_tool_error_rather_than_raising(
    tool: SaveConfigTool, output_dir: Path, tmp_path: Path
) -> None:
    """Every other refusal in this tool returns; the guard raises.

    ``_is_safe_config_name`` covers ``name`` at the entry point and
    returns ``{"success": False, "error": ...}`` the model can act on.
    ``_draft_id`` comes from wizard data and reaches the manager
    unchecked, so the manager's guard is what catches it — and a raise
    out of ``execute_with_context`` is a tool-call crash, not something
    the model can correct on its next turn. One condition, one contract.
    """
    (output_dir / "_draft-a").mkdir()
    context = _make_context(
        {"_draft_id": "a/../../outside/y", "bot": {"name": "x"}},
    )

    result = await tool.execute_with_context(context, config_name="fine")

    assert result["success"] is False
    assert "draft id" in result["error"]
    assert not (tmp_path / "outside" / "y.yaml").exists()


async def test_an_ordinary_save_still_reports_success(
    tool: SaveConfigTool, output_dir: Path
) -> None:
    """The translation must not swallow the happy path."""
    context = _make_context({"bot": {"name": "x"}})

    result = await tool.execute_with_context(context, config_name="my-bot")

    assert result["success"] is True
    assert (output_dir / "my-bot.yaml").exists()

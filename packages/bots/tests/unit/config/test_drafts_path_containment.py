"""A config name or draft id must not address outside the output directory.

:class:`ConfigDraftManager` composes three paths from caller-supplied
identifiers — a final config name in :meth:`finalize`, an alias name in
``_write_named_file``, and a ``draft_id`` in ``_draft_path`` — and each
reaches a write or an ``unlink``.

The names are not merely caller-supplied: ``finalize()`` called without
``final_name`` reads the name back out of the draft file's own YAML
metadata, so a draft on disk carries one. ``SaveConfigTool`` feeds this
manager from LLM tool arguments and wizard data.

``draft_id`` is the subtler of the two. ``_draft_path`` composes
``f"{draft_prefix}{draft_id}.yaml"``, so the prefix anchors the first
path segment and a *leading* ``..`` becomes the literal directory name
``_draft-..`` — which does not escape. An *interior* ``..`` does, once
the first segment exists as a directory. Both spellings are pinned
below so neither the accident nor the escape is mistaken for the rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.config.drafts import ConfigDraftManager


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "configs"
    out.mkdir()
    return out


@pytest.fixture
def manager(output_dir: Path) -> ConfigDraftManager:
    return ConfigDraftManager(output_dir=output_dir)


# --- final config name (finalize / _write_named_file) --------------------


def test_finalize_refuses_a_final_name_that_walks_out(
    manager: ConfigDraftManager, tmp_path: Path
) -> None:
    draft_id = manager.create_draft({"bot": {"name": "x"}})

    with pytest.raises(ValueError):
        manager.finalize(draft_id, final_name="../escaped")

    assert not (tmp_path / "escaped.yaml").exists()


def test_finalize_refuses_an_absolute_final_name(
    manager: ConfigDraftManager, tmp_path: Path
) -> None:
    """The extension is appended before composing, so that is what is guarded."""
    draft_id = manager.create_draft({"bot": {"name": "x"}})
    target = tmp_path / "escaped-absolute"

    with pytest.raises(ValueError):
        manager.finalize(draft_id, final_name=str(target))

    assert not target.with_suffix(".yaml").exists()


def test_finalize_refuses_a_name_read_back_out_of_the_draft(
    manager: ConfigDraftManager, output_dir: Path, tmp_path: Path
) -> None:
    """With no ``final_name`` the name comes from the draft file on disk."""
    draft_id = manager.create_draft({"bot": {"name": "x"}})
    draft_file = output_dir / f"_draft-{draft_id}.yaml"
    draft_file.write_text(
        draft_file.read_text().replace("_draft:", "_draft:\n  config_name: ../escaped-readback", 1)
    )

    with pytest.raises(ValueError):
        manager.finalize(draft_id)

    assert not (tmp_path / "escaped-readback.yaml").exists()


def test_update_draft_refuses_an_alias_name_that_walks_out(
    manager: ConfigDraftManager, tmp_path: Path
) -> None:
    """``config_name`` reaches ``_write_named_file``, a second composition."""
    draft_id = manager.create_draft({"bot": {"name": "x"}})

    with pytest.raises(ValueError):
        manager.update_draft(draft_id, {"bot": {"name": "x"}}, config_name="../escaped-alias")

    assert not (tmp_path / "escaped-alias.yaml").exists()


def test_a_config_name_in_a_subdirectory_still_works(
    manager: ConfigDraftManager, output_dir: Path
) -> None:
    """Containment is not a ``/``-rejecting character class."""
    (output_dir / "team").mkdir()
    draft_id = manager.create_draft({"bot": {"name": "x"}})

    manager.finalize(draft_id, final_name="team/alpha")

    assert (output_dir / "team" / "alpha.yaml").exists()


# --- draft_id (_draft_path) ----------------------------------------------


def test_discard_refuses_a_draft_id_with_an_interior_parent_ref(
    manager: ConfigDraftManager, output_dir: Path, tmp_path: Path
) -> None:
    """The escape that the draft prefix does *not* absorb.

    ``_draft-a/../../outside/y.yaml`` normalizes out of the output dir
    once ``_draft-a`` exists as a directory — and ``discard`` unlinks
    whatever it lands on.
    """
    victim_dir = tmp_path / "outside"
    victim_dir.mkdir()
    victim = victim_dir / "y.yaml"
    victim.write_text("keep me")
    (output_dir / "_draft-a").mkdir()

    with pytest.raises(ValueError):
        manager.discard("a/../../outside/y")

    assert victim.read_text() == "keep me"


def test_finalize_refuses_a_draft_id_with_an_interior_parent_ref(
    manager: ConfigDraftManager, output_dir: Path, tmp_path: Path
) -> None:
    """``finalize`` unlinks the draft too, via the same composition."""
    victim_dir = tmp_path / "outside"
    victim_dir.mkdir()
    (victim_dir / "y.yaml").write_text("keep me")
    (output_dir / "_draft-a").mkdir()

    with pytest.raises(ValueError):
        manager.finalize("a/../../outside/y", final_name="ok")

    assert (victim_dir / "y.yaml").read_text() == "keep me"


def test_a_leading_parent_ref_in_a_draft_id_is_contained_by_the_prefix(
    manager: ConfigDraftManager, tmp_path: Path
) -> None:
    """Pinned as an accident, not a defence.

    ``_draft-`` prepends, so ``../outside/x`` yields the literal segment
    ``_draft-..`` and never leaves. This is why the guard cannot be a
    leading-``..`` check — it would report success on the spelling that
    was never dangerous while missing the one that is.
    """
    victim_dir = tmp_path / "outside"
    victim_dir.mkdir()
    victim = victim_dir / "_draft-x.yaml"
    victim.write_text("keep me")

    assert manager.discard("../outside/x") is False
    assert victim.read_text() == "keep me"


def test_an_ordinary_draft_round_trips(manager: ConfigDraftManager) -> None:
    """The guard must not disturb the normal lifecycle."""
    draft_id = manager.create_draft({"bot": {"name": "x"}}, stage="gather")

    assert manager.get_draft(draft_id) is not None
    manager.update_draft(draft_id, {"bot": {"name": "y"}}, stage="review")
    assert manager.discard(draft_id) is True

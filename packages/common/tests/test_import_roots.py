"""Tests for :func:`dataknobs_common.testing.declare_import_root`.

The behaviour that matters is not "does it append to ``sys.path``" but the two
properties a hand-rolled version keeps getting wrong: it must not append the
same directory twice, and the comparison that decides that must be against the
form ``sys.path`` actually holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dataknobs_common.testing import declare_import_root


@pytest.fixture(autouse=True)
def _restore_sys_path() -> object:
    """Leave ``sys.path`` exactly as found — this is process-global state."""
    saved = list(sys.path)
    yield
    sys.path[:] = saved


def test_a_file_anchor_declares_its_directory(tmp_path: Path) -> None:
    """The normal call is ``declare_import_root(__file__)`` from a conftest."""
    anchor = tmp_path / "conftest.py"
    anchor.write_text("")

    declared = declare_import_root(anchor)

    assert declared == tmp_path.resolve()
    assert sys.path[0] == str(tmp_path.resolve())


def test_a_directory_anchor_is_used_as_is(tmp_path: Path) -> None:
    """A caller already holding the directory need not fabricate a file in it."""
    assert declare_import_root(tmp_path) == tmp_path.resolve()
    assert sys.path[0] == str(tmp_path.resolve())


def test_declaring_the_same_root_twice_does_not_grow_sys_path(tmp_path: Path) -> None:
    """The property a hand-rolled version misses.

    A conftest is imported once per session, but the same directory is reached
    by more than one anchor — and a re-insert on every call leaves ``sys.path``
    growing for the life of the process, with the earliest entry shadowing
    whatever a later declaration meant to take precedence.
    """
    anchor = tmp_path / "conftest.py"
    anchor.write_text("")

    declare_import_root(anchor)
    before = list(sys.path)
    declare_import_root(anchor)
    declare_import_root(tmp_path)

    assert sys.path == before


def test_an_unresolved_anchor_matches_the_resolved_entry(tmp_path: Path) -> None:
    """Resolution happens before the comparison, not after.

    ``Path`` comparison is the trap: ``sys.path`` holds strings, so testing a
    ``Path`` for membership never matches and the entry is inserted every time.
    Reaching the same directory by a ``..`` detour must be recognised too.
    """
    nested = tmp_path / "pkg" / "tests"
    nested.mkdir(parents=True)
    (tmp_path / "pkg" / "other").mkdir()

    declare_import_root(nested)
    before = list(sys.path)
    declare_import_root(tmp_path / "pkg" / "other" / ".." / "tests")

    assert sys.path == before


def test_an_anchor_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    """A declaration that cannot work must say so rather than no-op.

    ``sys.path`` accepts a nonexistent entry without complaint, so a typo'd
    anchor leaves every import it was meant to enable failing while the
    declaration itself reads as correct — the failure then surfaces as a
    ``ModuleNotFoundError`` somewhere else entirely.
    """
    before = list(sys.path)

    with pytest.raises(ValueError, match="not a directory"):
        declare_import_root(tmp_path / "no_such_directory" / "conftest.py")

    assert sys.path == before


def test_a_declared_root_makes_a_sibling_importable(tmp_path: Path) -> None:
    """End to end: the reason the function exists."""
    (tmp_path / "_dk_declared_root_fixture.py").write_text("VALUE = 41\n")
    declare_import_root(tmp_path)
    sys.modules.pop("_dk_declared_root_fixture", None)

    # Imported here rather than at module scope: the import is the behaviour
    # under test, and it cannot succeed before the declaration above runs.
    import _dk_declared_root_fixture

    try:
        assert _dk_declared_root_fixture.VALUE == 41
    finally:
        sys.modules.pop("_dk_declared_root_fixture", None)

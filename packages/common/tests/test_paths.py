"""Tests for ``dataknobs_common.paths``.

Containment is a lexical property of the composed path, so every test
here works on paths that need not exist -- the helper does no
filesystem I/O and the suite proves it by never creating a file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dataknobs_common.paths import safe_join


class TestSafeJoinAllows:
    """What a contained name composes to."""

    def test_a_plain_name(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "config.yaml") == tmp_path / "config.yaml"

    def test_a_subdirectory_name(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "domains/child.yaml") == tmp_path / "domains" / "child.yaml"

    def test_several_parts(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "domains", "child.yaml") == tmp_path / "domains" / "child.yaml"

    def test_no_parts_is_the_base_itself(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path) == tmp_path

    def test_interior_parent_segment_that_stays_inside(self, tmp_path: Path) -> None:
        """``a/../b`` never leaves the base, so it is contained."""
        assert safe_join(tmp_path, "a/../b.yaml") == tmp_path / "b.yaml"

    def test_curdir_segments_are_collapsed(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "./a/./b.yaml") == tmp_path / "a" / "b.yaml"

    def test_an_absolute_part_that_lands_inside_the_base(self, tmp_path: Path) -> None:
        """Containment is judged on where the path lands, not on how it was spelled."""
        assert safe_join(tmp_path, str(tmp_path / "a.yaml")) == tmp_path / "a.yaml"

    def test_a_relative_base(self) -> None:
        assert safe_join("configs", "child.yaml") == Path("configs/child.yaml")

    def test_a_curdir_base(self) -> None:
        """``normpath('.')`` erases the base, which a string-prefix test gets wrong."""
        assert safe_join(".", "child.yaml") == Path("child.yaml")

    def test_a_str_base(self, tmp_path: Path) -> None:
        assert safe_join(str(tmp_path), "child.yaml") == tmp_path / "child.yaml"

    def test_nothing_needs_to_exist(self, tmp_path: Path) -> None:
        composed = safe_join(tmp_path, "not/created/yet.yaml")
        assert composed == tmp_path / "not" / "created" / "yet.yaml"
        assert not composed.exists()


class TestSafeJoinRejects:
    """Every way a name can address outside the base."""

    def test_a_parent_segment(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "../escaped.yaml") is None

    def test_a_parent_segment_after_a_subdirectory(self, tmp_path: Path) -> None:
        """``sub/../..`` escapes even though no single segment does."""
        assert safe_join(tmp_path, "sub/../../escaped.yaml") is None

    def test_parent_segments_split_across_parts(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "..", "escaped.yaml") is None

    def test_the_parent_directory_itself(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "..") is None

    def test_an_absolute_part(self, tmp_path: Path) -> None:
        """``Path.__truediv__`` discards the base for an absolute operand."""
        assert safe_join(tmp_path, "/etc/passwd") is None

    def test_an_absolute_part_against_a_relative_base(self) -> None:
        assert safe_join("configs", "/etc/passwd") is None

    def test_a_parent_segment_against_a_relative_base(self) -> None:
        assert safe_join("configs", "../escaped.yaml") is None

    def test_a_parent_segment_against_a_curdir_base(self) -> None:
        assert safe_join(".", "../escaped.yaml") is None

    def test_a_sibling_that_shares_the_bases_prefix(self, tmp_path: Path) -> None:
        """``/base-other`` is not inside ``/base`` despite the string prefix."""
        base = tmp_path / "base"
        assert safe_join(base, f"..{os.sep}base-other{os.sep}x.yaml") is None

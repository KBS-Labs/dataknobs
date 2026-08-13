"""Tests for ``dataknobs_common.paths``.

Containment is a lexical property of the composed path, so every test
here works on paths that need not exist -- the helper does no
filesystem I/O and the suite proves it by never creating a file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dataknobs_common.paths import PathEscapeError, safe_join, safe_join_or_raise


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

    def test_a_root_base(self) -> None:
        """Everything is inside ``/``.

        The base's own component tuple is empty here, which is the case a
        string-prefix test gets wrong: ``str(Path("/")) + os.sep`` is
        ``"//"``, a prefix of nothing.
        """
        assert safe_join("/", "etc/passwd") == Path("/etc/passwd")

    def test_a_parent_segment_against_a_root_base(self) -> None:
        """``/..`` is ``/`` on a real filesystem, so this stays contained."""
        assert safe_join("/", "../etc/passwd") == Path("/etc/passwd")

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

    def test_a_prefix_sharing_sibling_reached_without_a_parent_segment(
        self, tmp_path: Path
    ) -> None:
        """The same boundary, pinned on the component comparison alone.

        The ``..`` spelling above is rejected by the pardir branch, so it
        never reaches the prefix test. An absolute part carries no ``..``
        and exercises the comparison directly -- the case a naive
        ``startswith`` would wave through.
        """
        base = tmp_path / "base"
        assert safe_join(base, str(tmp_path / "base-other" / "x.yaml")) is None
        assert safe_join(base, str(tmp_path / "basement" / "x.yaml")) is None

    def test_a_directory_named_like_the_base_but_nested_is_still_inside(
        self, tmp_path: Path
    ) -> None:
        """The comparison bounds the base, it does not blocklist the name."""
        base = tmp_path / "base"
        assert safe_join(base, "base-other/x.yaml") == base / "base-other" / "x.yaml"


class TestSafeJoinOrRaise:
    """The raising spelling, and why the sentinel one is not enough.

    Four call sites in this repo turned ``None`` into an exception by
    hand, and between them they used three different exception types and
    worded the same refusal four ways. A caller could not write one
    ``except`` for one condition. These pin the collapsed form.
    """

    def test_it_returns_the_collapsed_path_when_contained(self) -> None:
        result = safe_join_or_raise(
            Path("/srv/configs"),
            "domains/../child.yaml",
            what="config name",
            outside="the config directory",
        )

        assert result == Path("/srv/configs/child.yaml")

    def test_it_raises_rather_than_returning_none(self) -> None:
        with pytest.raises(PathEscapeError):
            safe_join_or_raise(
                Path("/srv/configs"),
                "../../etc/passwd",
                what="config name",
                outside="the config directory",
            )

    def test_the_message_names_the_input_not_the_deployment_layout(self) -> None:
        """A caller learns what to correct without learning where the
        deployment keeps its files.
        """
        with pytest.raises(PathEscapeError) as excinfo:
            safe_join_or_raise(
                Path("/srv/secret-location/configs"),
                "../../etc/passwd",
                what="config name",
                outside="the config directory",
            )

        message = str(excinfo.value)
        assert "config name" in message
        assert "'../../etc/passwd'" in message
        assert "/srv/secret-location" not in message

    def test_supplied_overrides_what_is_quoted_back(self) -> None:
        """The last part is often derived — a name with a suffix appended,
        or a prefixed draft filename. The message should quote what the
        caller actually passed, not what we built from it.
        """
        with pytest.raises(PathEscapeError) as excinfo:
            safe_join_or_raise(
                Path("/base"),
                "_draft-a/../../x.yaml",
                what="draft id",
                outside="the output directory",
                supplied="a/../../x",
            )

        assert "'a/../../x'" in str(excinfo.value)
        assert "_draft-" not in str(excinfo.value)

    def test_it_is_still_a_value_error(self) -> None:
        """Narrowing, not a breaking change: the sites that raised bare
        ``ValueError`` before this type existed stay catchable that way.
        """
        with pytest.raises(ValueError):
            safe_join_or_raise(Path("/base"), "/etc/passwd", what="path", outside="the base")


class TestNulIsRefused:
    """A NUL is not a containment question, but it arrives the same way.

    ``open()`` rejects an embedded NUL with its own ``ValueError``, so
    nothing was exploitable — but the width measured by the guard is not
    the string the C library sees, and the refusal arrived as a fourth
    error surface for the same class of bad name. It is refused here so
    every rejection of a name comes from one place as one type.
    """

    def test_safe_join_refuses_a_nul(self) -> None:
        assert safe_join(Path("/base"), "a\x00b.yaml") is None

    def test_safe_join_or_raise_refuses_a_nul(self) -> None:
        with pytest.raises(PathEscapeError):
            safe_join_or_raise(Path("/base"), "a\x00b.yaml", what="config name", outside="the base")

    def test_a_nul_in_any_part_is_refused(self) -> None:
        assert safe_join(Path("/base"), "sub", "a\x00b") is None

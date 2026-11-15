"""Behavioural tests for ``filepath_generator`` and ``write_lines``."""

import gzip
import os
import sys
import types
from pathlib import Path

import pytest

from dataknobs_utils.file_utils import filepath_generator, write_lines


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """root/{a.txt, s1/{b.txt, s2/{c.txt}}, s3/{d.txt}, empty/}"""
    for rel in ("a.txt", "s1/b.txt", "s1/s2/c.txt", "s3/d.txt"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    return tmp_path


def _rel(paths: list[str], root: Path) -> list[str]:
    return sorted(os.path.relpath(p, root) for p in paths)


class TestFilepathGenerator:
    def test_is_lazy_generator(self, tree: Path) -> None:
        assert isinstance(filepath_generator(str(tree)), types.GeneratorType)

    def test_descend_files_only(self, tree: Path) -> None:
        assert _rel(list(filepath_generator(str(tree))), tree) == [
            "a.txt",
            str(Path("s1", "b.txt")),
            str(Path("s1", "s2", "c.txt")),
            str(Path("s3", "d.txt")),
        ]

    def test_descend_with_directories(self, tree: Path) -> None:
        assert _rel(list(filepath_generator(str(tree), files_only=False)), tree) == sorted(
            [
                "a.txt",
                "empty",
                "s1",
                str(Path("s1", "b.txt")),
                str(Path("s1", "s2")),
                str(Path("s1", "s2", "c.txt")),
                "s3",
                str(Path("s3", "d.txt")),
            ]
        )

    def test_no_descend_with_directories(self, tree: Path) -> None:
        got = _rel(list(filepath_generator(str(tree), descend=False, files_only=False)), tree)
        assert got == ["a.txt", "empty", "s1", "s3"]

    def test_no_descend_files_only(self, tree: Path) -> None:
        assert _rel(list(filepath_generator(str(tree), descend=False)), tree) == ["a.txt"]

    def test_root_never_yielded_and_no_duplicates(self, tree: Path) -> None:
        got = list(filepath_generator(str(tree), files_only=False))
        assert str(tree) not in got
        assert len(got) == len(set(got))

    def test_path_form(self, tree: Path) -> None:
        got = sorted(filepath_generator(str(tree) + os.sep))
        assert got[0] == str(tree / "a.txt")
        assert all(os.sep + os.sep not in p for p in got)

    def test_relative_root_gives_relative_paths(
        self, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tree)
        assert sorted(filepath_generator("s1")) == [
            str(Path("s1", "b.txt")),
            str(Path("s1", "s2", "c.txt")),
        ]

    def test_missing_root_yields_nothing(self, tmp_path: Path) -> None:
        assert list(filepath_generator(str(tmp_path / "nope"))) == []

    def test_file_root_yields_nothing(self, tree: Path) -> None:
        assert list(filepath_generator(str(tree / "a.txt"))) == []

    def test_seen_paths_are_not_yielded_but_directories_are_traversed(self, tree: Path) -> None:
        seen = {str(tree / "s1"), str(tree / "a.txt")}
        got = _rel(list(filepath_generator(str(tree), seen=seen, files_only=False)), tree)
        assert "s1" not in got
        assert "a.txt" not in got
        assert str(Path("s1", "b.txt")) in got
        assert str(Path("s1", "s2")) in got

    def test_seen_is_updated(self, tree: Path) -> None:
        seen: set[str] = set()
        got = list(filepath_generator(str(tree), seen=seen))
        assert seen == set(got) | {str(tree)}

    def test_shared_seen_avoids_repeats_across_calls(self, tree: Path) -> None:
        seen: set[str] = set()
        first = list(filepath_generator(str(tree), seen=seen))
        second = list(filepath_generator(str(tree / "s1"), seen=seen))
        assert len(first) == 4
        assert second == []

    def test_root_in_seen_does_not_stop_traversal(self, tree: Path) -> None:
        assert len(list(filepath_generator(str(tree), seen={str(tree)}))) == 4

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_symlinks(self, tree: Path) -> None:
        (tree / "link_to_dir").symlink_to(tree / "s1", target_is_directory=True)
        (tree / "link_to_file").symlink_to(tree / "a.txt")
        files = _rel(list(filepath_generator(str(tree))), tree)
        assert "link_to_file" in files
        assert not any(p.startswith("link_to_dir" + os.sep) for p in files)
        with_dirs = _rel(list(filepath_generator(str(tree), files_only=False)), tree)
        assert "link_to_dir" in with_dirs
        assert not any(p.startswith("link_to_dir" + os.sep) for p in with_dirs)

    @pytest.mark.skipif(
        sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="needs POSIX permissions and a non-root user",
    )
    def test_unreadable_subdirectory_is_skipped(self, tree: Path) -> None:
        locked = tree / "s3"
        locked.chmod(0)
        try:
            got = _rel(list(filepath_generator(str(tree))), tree)
        finally:
            locked.chmod(0o755)
        assert str(Path("s3", "d.txt")) not in got
        assert "a.txt" in got


class TestWriteLines:
    def test_sorted_with_trailing_newlines(self, tmp_path: Path) -> None:
        write_lines("o.txt", ["b", "a", "c b", ""], rootdir=str(tmp_path))
        assert (tmp_path / "o.txt").read_text(encoding="utf-8") == "\na\nb\nc b\n"

    def test_without_rootdir(self, tmp_path: Path) -> None:
        target = tmp_path / "o.txt"
        write_lines(str(target), ["z", "y"])
        assert target.read_text(encoding="utf-8") == "y\nz\n"

    def test_empty_list_gives_empty_file(self, tmp_path: Path) -> None:
        write_lines("o.txt", [], rootdir=str(tmp_path))
        assert (tmp_path / "o.txt").read_text(encoding="utf-8") == ""

    def test_overwrites(self, tmp_path: Path) -> None:
        write_lines("o.txt", ["old", "older"], rootdir=str(tmp_path))
        write_lines("o.txt", ["new"], rootdir=str(tmp_path))
        assert (tmp_path / "o.txt").read_text(encoding="utf-8") == "new\n"

    def test_gzip_by_suffix(self, tmp_path: Path) -> None:
        write_lines("o.txt.gz", ["z", "y\nw"], rootdir=str(tmp_path))
        raw = (tmp_path / "o.txt.gz").read_bytes()
        assert raw[:2] == b"\x1f\x8b"
        with gzip.open(tmp_path / "o.txt.gz", "rt", encoding="utf-8") as f:
            assert f.read() == "y\nw\nz\n"

    def test_gzip_suffix_is_case_sensitive(self, tmp_path: Path) -> None:
        write_lines("o.GZ", ["a"], rootdir=str(tmp_path))
        assert (tmp_path / "o.GZ").read_bytes() == b"a\n"

    def test_utf8(self, tmp_path: Path) -> None:
        write_lines("o.txt", ["é"], rootdir=str(tmp_path))
        assert (tmp_path / "o.txt").read_bytes() == "é\n".encode()

    def test_items_written_with_str(self, tmp_path: Path) -> None:
        write_lines("o.txt", [3, 1, 2], rootdir=str(tmp_path))  # type: ignore[list-item]
        assert (tmp_path / "o.txt").read_text(encoding="utf-8") == "1\n2\n3\n"

    def test_missing_parent_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            write_lines("x/y.txt", ["a"], rootdir=str(tmp_path))

    def test_returns_none(self, tmp_path: Path) -> None:
        assert write_lines("o.txt", ["a"], rootdir=str(tmp_path)) is None  # type: ignore[func-returns-value]

    def test_unsortable_items_raise_type_error(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            write_lines("o.txt", [1, "a"], rootdir=str(tmp_path))  # type: ignore[list-item]

    def test_failed_sort_leaves_target_untouched(self, tmp_path: Path) -> None:
        target = tmp_path / "o.txt"
        target.write_text("keep\n", encoding="utf-8")
        with pytest.raises(TypeError):
            write_lines("o.txt", [1, "a"], rootdir=str(tmp_path))  # type: ignore[list-item]
        assert target.read_text(encoding="utf-8") == "keep\n"

    def test_failed_sort_creates_no_file(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            write_lines("new.txt", [1, "a"], rootdir=str(tmp_path))  # type: ignore[list-item]
        assert not (tmp_path / "new.txt").exists()

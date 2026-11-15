# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""File system utility functions for traversing, reading, and writing files.

Provides generators and helpers for working with files, directories,
and compressed file formats.
"""

import gzip
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import List, Set, TextIO, Tuple

logger = logging.getLogger(__name__)


def _list_directory(dirpath: str) -> List[Tuple[str, bool, bool]]:
    """List a directory's entries as (path, is_dir, is_symlink), sorted by name.

    A directory that cannot be read gives an empty list.
    """
    entries: List[Tuple[str, bool, bool]] = []
    try:
        with os.scandir(dirpath) as it:
            for entry in sorted(it, key=lambda e: e.name):
                entries.append(
                    (str(Path(dirpath) / entry.name), entry.is_dir(), entry.is_symlink())
                )
    except OSError as err:
        logger.debug("Skipping unreadable directory %s: %s", dirpath, err)
        return []
    return entries


def filepath_generator(
    rootpath: str, descend: bool = True, seen: Set[str] | None = None, files_only: bool = True
) -> Generator[str, None, None]:
    """Lazily generate the paths of entries under the root path.

    Each path is ``rootpath`` joined to the entry's location, rendered as
    ``str(pathlib.Path(parent) / name)``. ``rootpath`` itself is never
    yielded, and no path is yielded twice in one call. The order is stable for
    an unchanged tree but otherwise unspecified.

    A symbolic link to a file is yielded as a file. A symbolic link to a
    directory is treated as a directory (yielded when directories are) but is
    never descended into. Any entry that is not a directory counts as a file.

    A ``rootpath`` that does not exist or is not a directory yields nothing,
    and a subdirectory that cannot be read is skipped; neither raises.

    Args:
        rootpath: The root path under which to find entries.
        descend: True to yield entries at any depth below the root; False to
            yield only the root's immediate children. Defaults to True.
        seen: Set of path strings already reported. A path in ``seen`` is
            never yielded, but a directory in ``seen`` is still traversed. The
            set is updated as the generator runs: ``rootpath`` is added at the
            start and every yielded path as it is yielded, so one set can be
            shared across calls. None uses a fresh set. Defaults to None.
        files_only: True to yield only files; False to also yield
            directories. Defaults to True.

    Yields:
        str: Each path found under the root path.
    """
    if seen is None:
        seen = set()
    seen.add(rootpath)
    if not Path(rootpath).is_dir():
        return

    stack = [iter(_list_directory(rootpath))]
    while stack:
        entry = next(stack[-1], None)
        if entry is None:
            stack.pop()
            continue
        path, is_dir, is_symlink = entry
        if (not is_dir or not files_only) and path not in seen:
            seen.add(path)
            yield path
        if is_dir and descend and not is_symlink:
            stack.append(iter(_list_directory(path)))


def fileline_generator(filename: str, rootdir: str | None = None) -> Generator[str, None, None]:
    """Generate lines from the file.

    Automatically handles both plain text and gzip-compressed files based on
    the .gz extension. All lines are stripped of leading/trailing whitespace.

    Args:
        filename: The name of the file to read.
        rootdir: Optional directory path to prepend to filename. Defaults to None.

    Yields:
        str: Each stripped line from the file.
    """
    if rootdir is not None:
        filename = str(Path(rootdir) / filename)
    if filename.endswith(".gz"):
        with gzip.open(filename, mode="rt", encoding="utf-8") as f:
            for line in f:
                yield line.strip()
    else:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                yield line.strip()


def _open_text_for_writing(target: str) -> TextIO:
    """Open ``target`` for UTF-8 text writing, gzip-compressed when it ends with ".gz"."""
    if target.endswith(".gz"):
        return gzip.open(target, mode="wt", encoding="utf-8")
    return open(target, mode="w", encoding="utf-8")


def write_lines(outfile: str, lines: List[str], rootdir: str | None = None) -> None:
    """Write lines to a file in sorted order.

    The lines are sorted in ascending natural order and each is written as
    ``str(item)`` followed by a newline, as UTF-8 text. The target is created
    or overwritten; it is gzip-compressed when its path ends with ".gz"
    (case-sensitive). The sort happens before the target is opened, so a
    failed sort leaves an existing target untouched and creates none.

    Args:
        outfile: The name of the output file.
        lines: Lines to write to the file.
        rootdir: Optional directory path to prepend to outfile. Defaults to None.

    Raises:
        TypeError: If the lines cannot be compared with each other.
        FileNotFoundError: If the target's parent directory does not exist.
    """
    target = str(Path(rootdir) / outfile) if rootdir is not None else outfile
    ordered = sorted(lines)
    with _open_text_for_writing(target) as out:
        for item in ordered:
            out.write(f"{item}\n")


def is_gzip_file(filepath: str) -> bool:
    """Determine whether a file is gzip-compressed.

    Checks the file's magic number (first 3 bytes) to identify gzip format.

    Args:
        filepath: Path to the file to check.

    Returns:
        bool: True if the file is gzip-compressed, False otherwise.
    """
    is_gzip = False
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            b = f.read(3)
            is_gzip = b == b"\x1f\x8b\x08"
    return is_gzip

"""File system utility functions for traversing, reading, and writing files.

Provides generators and helpers for working with files, directories,
and compressed file formats.
"""

import gzip
import os
from collections.abc import Generator
from pathlib import Path
from typing import List, Set


def filepath_generator(
    rootpath: str,
    descend: bool = True,
    seen: Set[str] | None = None,
    files_only: bool = True,
) -> Generator[str, None, None]:
    """Generate all filepaths under the root path.

    Args:
        rootpath: The root path under which to find files.
        descend: True to descend into subdirectories. Defaults to True.
        seen: Set of filepaths and/or directories to ignore. Defaults to None.
        files_only: True to generate only paths to files; False to include
            paths to directories. Defaults to True.

    Yields:
        str: Each file path found under the root path.
    """
    if seen is None:
        seen = set()
    seen.add(rootpath)
    for root, dirs, files in os.walk(rootpath, topdown=True):
        if not descend and root != rootpath and root in seen:
            break
        for name in files:
            fpath = str(Path(root) / name)
            if fpath not in seen:
                seen.add(fpath)
                yield fpath
        if not descend or not files_only:
            for name in dirs:
                next_root = str(Path(root) / name)
                if next_root not in seen:
                    seen.add(next_root)
                    yield next_root


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


def write_lines(outfile: str, lines: List[str], rootdir: str | None = None) -> None:
    """Write lines to a file in sorted order.

    Automatically handles both plain text and gzip-compressed files based on
    the .gz extension. Lines are sorted before writing.

    Args:
        outfile: The name of the output file.
        lines: List of lines to write to the file.
        rootdir: Optional directory path to prepend to outfile. Defaults to None.
    """
    if rootdir is not None:
        outfile = str(Path(rootdir) / outfile)
    if outfile.endswith(".gz"):
        with gzip.open(outfile, mode="wt", encoding="utf-8") as f:
            for line in sorted(lines):
                print(line, file=f)
    else:
        with open(outfile, mode="w", encoding="utf-8") as f:
            for line in sorted(lines):
                print(line, file=f)


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

import gzip
import os
from typing import Set, Optional, Generator, List, Any, IO, Union


def filepath_generator(
    rootpath: str,
    descend: bool = True,
    seen: Optional[Set[str]] = None,
    files_only: bool = True,
) -> Generator[str, None, None]:
    """Generate all filepaths under the root path.

    :param rootpath: The root path under which to find files.
    :param descend: True to descend into subdirectories
    :param seen: Set of filepaths and/or directories to ignore
    :param files_only: True to generate only paths to files; False to
        include paths to directories.
    :yield: Each file path
    """
    if seen is None:
        seen = set()
    seen.add(rootpath)
    for root, dirs, files in os.walk(rootpath, topdown=True):
        if not descend and root != rootpath and root in seen:
            break
        for name in files:
            fpath = os.path.join(root, name)
            if fpath not in seen:
                seen.add(fpath)
                yield fpath
        if not descend or not files_only:
            for name in dirs:
                next_root = os.path.join(root, name)
                if next_root not in seen:
                    seen.add(next_root)
                    yield next_root


def fileline_generator(filename: str, rootdir: Optional[str] = None) -> Generator[str, None, None]:
    """Generate lines from the file.
    :param filename: The name of the file.
    :param rootdir: (optional) The directory of the file.
    :yield: Each stripped file line
    """
    if rootdir is not None:
        filename = os.path.join(rootdir, filename)
    if filename.endswith(".gz"):
        with gzip.open(filename, mode="rt", encoding="utf-8") as f:
            for line in f:
                yield line.strip()
    else:
        with open(filename, mode="r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()


def write_lines(outfile: str, lines: List[str], rootdir: Optional[str] = None) -> None:
    """Write the lines to the file.
    :param outfile: The name of the file.
    :param rootdir: (optional) The directory of the file.
    """
    if rootdir is not None:
        outfile = os.path.join(rootdir, outfile)
    if outfile.endswith(".gz"):
        with gzip.open(outfile, mode="wt", encoding="utf-8") as f:
            for line in sorted(lines):
                print(line, file=f)
    else:
        with open(outfile, mode="w", encoding="utf-8") as f:
            for line in sorted(lines):
                print(line, file=f)


def is_gzip_file(filepath: str) -> bool:
    """Determine whether the file at filepath is gzipped.
    :param filepath: The path to the file
    :return: True if the file is gzipped
    """
    is_gzip = False
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            b = f.read(3)
            is_gzip = b == b"\x1f\x8b\x08"
    return is_gzip

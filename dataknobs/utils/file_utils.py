import gzip
import os
from typing import Set


def filepath_generator(
        rootpath: str,
        descend: bool = True,
        seen: Set[str] = None
):  # yields -> str
    '''
    Generate all filepaths under the root path.

    :param rootpath: The root path under which to find files.
    :param descend: True to descend into subdirectories
    :param seen: Set of filepaths and/or directories to ignore
    :yield: Each file path
    '''
    if seen is None:
        seen = set()
    seen.add(rootpath)
    for root, dirs, files in os.walk(rootpath, topdown=True):
        for name in files:
            fpath = os.path.join(root, name)
            if fpath not in seen:
                seen.add(fpath)
                yield fpath
        if descend:
            for name in dirs:
                next_root = os.path.join(root, name)
                if next_root not in seen:
                    seen.add(next_root)
                    filepath_generator(next_root, descend=True, seen=seen)


def fileline_generator(filename: str, rootdir: str = None):
    '''
    Generate lines from the file.
    :param filename: The name of the file.
    :param rootdir: (optional) The directory of the file.
    :yield: Each stripped file line
    '''
    if filename.endswith('.gz'):
        open_fn = gzip.open
        mode = 'rt'
    else:
        open_fn = open
        mode = 'r'
    if rootdir is not None:
        filename = os.path.join(rootdir, filename)
    with open_fn(filename, mode=mode, encoding='utf-8') as f:
        for line in f:
            yield line.strip()


def write_lines(outfile, lines, rootdir=None):
    '''
    Write the lines to the file.
    :param outfile: The name of the file.
    :param rootdir: (optional) The directory of the file.
    '''
    if rootdir is not None:
        outfile = os.path.join(rootdir, outfile)
    with open(outfile, 'w', encoding='utf-8') as f:
        for line in sorted(lines):
            print(line, file=f)

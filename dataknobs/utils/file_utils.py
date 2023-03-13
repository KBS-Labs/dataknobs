import os
from typing import Set


def split_wget_fpath(fpath: str):
    '''
    Split the filepath into parts in cases where files may be named from e.g.,
    wget and may have query strings in the name.

    :param fpath: The file path to split
    :return: (dirpath, basename, ext, query)
    '''
    query = ''
    dirpath, fname = os.path.split(fpath)
    basename, ext = os.path.splitext(fname)
    if ext.startswith('.gz?'):
        query = '.' + ext[3:]
        ext = '.gz'
    if ext == '.gz':
        basename, ext = os.path.splitext(basename)
    return (dirpath, basename, ext, query)


def get_norm_ext(fpath: str):
    '''
    Get the normalized (lowercase) extension of the file, accounting for files
    named from e.g., wget having query strings in the name.
    '''
    return split_wget_fpath(fpath)[2].lower()


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

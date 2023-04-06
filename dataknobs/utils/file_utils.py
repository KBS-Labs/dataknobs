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

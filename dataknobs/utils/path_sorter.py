import json
import os
import re
import dataknobs.utils.json_paths as jpaths
import dataknobs.utils.json_utils as jutils
import dataknobs.utils.file_utils as file_utils
from typing import Any, Callable, Dict, List, Set, Tuple, Union


class Path:
    '''
    Container for a path.
    '''
    def __init__(self, jq_path: str, item: Any, line_num: int = -1):
        '''
        :param jq_path: A fully-qualified indexed path.
        :param item: The path's item (value)
        '''
        self.jq_path = jq_path
        self.item = item
        self.line_num = line_num
        self._path_elts = None  # jq_path.split('.')
        self._len = None  # Number of path elements

    def __repr__(self) -> str:
        lnstr = f'{self.line_num}: ' if self.line_num >= 0 else ''
        return f'{lnstr}{self.jq_path}: {self.item}'

    def __key(self):
        return (self.jq_path, self.item) if self.line_num < 0 else self.line_num

    def __lt__(self, other: Path) -> bool:
        if self.line_num < 0 or other.line_num < 0:
            return self.jq_path < other.jq_path
        else:
            return self.line_num < other.line_num

    def __hash__(self) -> int:
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Path):
            return self.__key() == other.__key()
        return NotImplemented

    @property
    def path_elts(self) -> List[str]:
        ''' Get this path's (index-qualified) elements '''
        if self._path_elts is None:
            self._path_elts = self.jq_path.split('.')
        return self._path_elts

    @property
    def size(self) -> int:
        ''' Get the number of path_elements in this path. '''
        return len(self.path_elts)


class PathGroup:
    '''
    Container for a consistent group of paths.
    '''
    def __init__(self, first_path: Path = None):
        self._paths = {first_path} if first_path is not None else None
        self.distributed_paths = None  # Set[Path]
        self._longest_path = first_path
        self._all_paths = None

    @property
    def num_paths(self) -> int:
        return len(self._paths) if self._paths is not None else 0

    @property
    def num_distributed_paths(self) -> int:
        return (
            len(self.distributed_paths)
            if self.distributed_paths is not None
            else 0
        )

    @property
    def size(self) -> int:
        ''' Get the number of paths in this group. '''
        return self.num_paths + self.num_distributed_paths

    @property
    def paths(self) -> List[Path]:
        ''' Get all paths (including distributed) '''
        if self._all_paths is None:
            if self._paths is not None:
                self._all_paths = self._paths.copy()
                if self.distributed_paths is not None:
                    self._all_paths.update(self.distributed_paths)
                self._all_paths = sorted(self._all_paths)
            elif self.distributed_paths is not None:
                self._all_paths = sorted(self.distributed_paths)
        return self._all_paths

    @property
    def longest_path(self) -> Path:
        ''' Get this group's longest path '''
        return self._longest_path

    def as_dict(self) -> Dict[str, str]:
        ''' Reconstruct the object from the paths '''
        d = dict()
        if self.paths is not None:
            for path in self.paths:
                jutils.path_to_dict(path.jq_path, path.item, result=d)
        return d

    def accept(self, path: Path) -> bool:
        '''
        Add the path if it belongs in this group.
        :param path: The path to (potentially) add.
        :return: True if the path was accepted and added.
        '''
        return self._do_accept(path, distribute=False)

    def accept_distributed(self, path: Path):
        '''
        Add the path as a distributed path, if accepted.
        '''
        self._do_accept(path, distribute=True)

    def incorporate_paths(self, group: PathGroup):
        '''
        Incorporate (distribute) the group's appliccable paths into this group.
        '''
        for path in group.paths:
            self.accept_distributed(path)

    def _do_accept(self, path: Path, distribute: bool = False):
        result = True
        if self._paths is None:
            self._paths = {path}
            self._longest_path = path
            self._all_paths = None
        elif self.path_aligns(path):
            if distribute:
                # add to self.distributed_paths
                if self.distributed_paths is None:
                    self.distributed_paths = {path}
                else:
                    self.distributed_paths.add(path)
            else:
                # add to (primary) paths
                self._paths.add(path)
                if path.size > self._longest_path.size:
                    self._longest_path = path
            self._all_paths = None
        else:
            result = False
        return result

    def path_aligns(self, path: Path) -> bool:
        '''
        Determine whether the path aligns with this group's paths.
        '''
        if path.size > self._longest_path.size:
            lpath = path
            opath = self._longest_path
        else:
            lpath = self._longest_path
            opath = path
        return self._paths_align(lpath, opath)

    def _paths_align(self, longer_path: Path, other_path: Path) -> bool:
        '''
        Determine whether the other path aligns with the longer path.
        '''
        aligns = True
        for idx in range(other_path.size):
            lelt = longer_path.path_elts[idx]
            oelt = other_path.path_elts[idx]
            if lelt != oelt:
                # Doesn't align if matches up to [] (because idx differs
                if (
                        lelt[-1] == ']' and oelt[-1] == ']' and
                        lelt[:lelt.find('[')] == oelt[:oelt.find('[')]
                ):
                    aligns = False
                    break
        return aligns


class PathSorter:
    '''
    Container for sorting paths belonging together into groups.
    '''
    def __init__(self, group_size: int = 0):
        '''
        :param group_size: A size constraint/expectation for groups such that
           any group not reaching this size (if > 0) is silently "dropped".
        '''
        self.group_size = group_size
        self.groups = None  # List[PathGroup]

    @property
    def num_groups(self) -> int:
        ''' Get the number of groups '''
        return len(self.groups) if self.groups is not None else 0

    def add_path(self, path: Path):
        '''
        Add the path to an existing group, or create a new group.
        '''
        if self.groups is None or len(self.groups) == 0:
            self.groups = [PathGroup(first_path=path)]
        else:
            # Assume always "incremental", such that new paths will belong in
            # the latest group
            latest_group = self.groups[-1]
            if not latest_group.accept(path):
                # Time to add a new group
                self.close_group()

                # Start a new group
                self.groups.append(PathGroup(first_path=path))

    def close_group(self, idx: int = -1, check_size: bool = True):
        '''
        Close the (last) group by
          * Adding distributable lines from the prior group
          * Checking size constraints and dropping if warranted
        '''
        if self.groups is None or len(self.groups) == 0:
            return

        if idx == -1:
            idx = len(self.groups) - 1
        latest_group = self.groups[idx]

        # Add distributable lines from the prior group
        if idx > 0:
            latest_group.incorporate_paths(self.groups[idx-1])

        # Check size constraints
        if check_size and self.group_size > 0 and len(self.groups) > 0:
            if latest_group.size < self.group_size:
                # Last group not "filled" ... need to drop
                print('POP!')
                self.groups.pop()

    def accept_path(self, path: Path):
        '''
        Add the path only if accepted by an existing group.
        :param path: The path to add
        :return: True if accepted
        '''
        for group in self.groups:
            if group.accept(path):
                return True
        return False

    def all_groups_have_size(self, group_size: int) -> bool:
        '''
        :param group_size: The group size to test for.
        :return: True if all groups have the group_size
        '''
        if self.groups is not None:
            for group in self.groups:
                if group.size != group_size:
                    return False
            return True
        return False


def sort_matching_lines(
        jdo: jpaths.JsonDataOrganizer,
        path2value: Dict[str, str],
        max_groups: int = 1000,
) -> PathSorter:
    '''
    Find and sort lines matching all of the path2value specs.
    :param jdo: The JsonDataOrganizer around the paths
    :param path2value: A dict of {<path>: <value>}s, to find in records
        where each spec's:
            "path" is considered a substring of fully qualified jq_paths
            "value" must fully (str) match
    :param max_groups: The maximum number of groups to collect
    :return: The PathSorter with collected lines
    '''
    sorter = PathSorter(group_size=len(path2value))

    def match_fn(jq_path, item):
        for path, value in path2value.items():
            if value == item and path in jq_path:
                return True
        return False

    # Collect all matching lines, sorting into groups
    for lnum, (jq_path, item) in enumerate(jdo.path_line_generator(None)):
        # NOTE: call match_fn inside for consistent line_nums
        if match_fn(jq_path, item):
            sorter.add_path(Path(jq_path, item, line_num=lnum))
            if max_groups > 0 and sorter.num_groups >= max_groups:
                break
        #TODO: add mode to continue streaming to outfile while not collecting

    # Check that the last group has all specs
    sorter.close_group()

    return sorter


def filter_into_groups(
        jdo: jpaths.JsonDataOrganizer,
        sorter: PathSorter,
        outfile = None,
):
    '''
    Walk through the paths once more, adding lines to the sorter to fill out
    the groups.
    '''
    group_idx = 0
    if sorter.num_groups == 0:
        return
    group = sorter.groups[group_idx]
    next_group = (
        sorter.groups[group_idx+1]
        if group_idx + 1 < sorter.num_groups
        else None
    )
    gotone = False

    for lnum, (jq_path, item) in enumerate(jdo.path_line_generator(None)):
        path = Path(jq_path, item, line_num=lnum)
        if group.accept(path):
            gotone = True
        else:
            if gotone:
                # Move up to the next group
                if next_group is not None:
                    if next_group.accept(path):
                        # Close current group
                        sorter.close_group(idx=group_idx, check_size=False)
                        # Increment to the next group
                        group_idx += 1
                        group = next_group
                        next_group = (
                            sorter.groups[group_idx+1]
                            if group_idx + 1 < sorter.num_groups
                            else None
                        )
                else:
                    # No more groups ... can exit
                    break
            #else -- keep going

    if gotone:  # close the final group
        sorter.close_group(idx=group_idx, check_size=False)

import gzip
import os
import dataknobs.utils.json_utils as jutils
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Set, Tuple, Union


class Formatter(ABC):
    '''
    Class for formatting jq_path, item pairs into file lines and reversing.
    '''
    def __init__(self, delim: str = '\t'):
        '''
        :param delim: The line field delimiter
        '''
        self.delim = delim

    def __call__(self, jq_path: str, item: Any, delim='\t') -> str:
        '''
        Build the formatted line.
        :param jq_path: A fully indexed path
        :param item: The path's item value
        '''
        return self.format_path(jq_path, item)

    @abstractmethod
    def format_path(self, jq_path: str, item: Any, delim='\t') -> str:
        '''
        Format the jq_path and item as a line.

        :param jq_path: A fully indexed path
        :param item: The path's item value
        '''
        raise NotImplementedError

    def split_line(self, line: str) -> List[str]:
        '''
        Split a formatted line into its parts.
        '''
        return [v.strip() for v in line.split(self.delim)]

    @abstractmethod
    def reverse_line(self, line: str) -> Tuple[str, str]:
        '''
        Reverse a line built by this formatter back to the original jq_path
        and its item.
        :param line: A line produced by this formatter.
        :return: The (jq_path, item) tuple
        '''
        raise NotImplementedError


class PathFormatter(Formatter):
    '''
    A simple path formatter for lines with fully indexed paths and items.
    '''

    def __init__(self, delim='\t'):
        '''
        :param delim: The line field delimiter
        '''
        super().__init__(delim)

    def format_path(self, jq_path: str, item: Any) -> str:
        '''
        Format the jq_path and item as a delimitted line of the form:
        item \t jq_path

        :param jq_path: A fully indexed path
        :param item: The path's item value
        '''
        return f'{jq_path}{self.delim}{item}'

    def reverse_line(self, line: str) -> Tuple[str, str]:
        '''
        Reverse a line built by this formatter back to the original jq_path
        and its item.
        :param line: A line produced by this formatter.
        :return: The (jq_path, item) tuple
        '''
        jq_path, item = self.split_line(line)
        return (jq_path, item)


class CPathParts:
    '''
    A structure to access CPath line information.
    '''
    def __init__(self, cpath_parts: List[str]):
        self.parts = cpath_parts
        self.item = self.parts[0]
        self.cjq_path = self.parts[1]
        self._cjq_parts = None
        self._idxstr = None
        self._idxs = None

    @property
    def idxstr(self) -> str:
        if self._idxstr is None:
            self._idxstr = self.parts[2] if len(self.parts) > 2 else ''
        return self._idxstr

    @property
    def idxs(self) -> List[int]:
        if self._idxs is None:
            self._idxs = [int(i) for i in self.idxstr.split(', ')]
        return self._idxs

    @property
    def cjq_parts(self) -> List[str]:
        if self._cjq_parts is None:
            self._cjq_parts = self.cjq_path.split('.')
        return self._cjq_parts

    def rebuild_path(self, idx_count: int = 0) -> str:
        '''
        Rebuild the fully indexed path up through idx_count indexes, or for
        the entire path if idx_count is 0.
        '''
        cur_ind = 0
        jq_parts = list()
        for part in self.cjq_parts:
            if part.endswith('[]'):
                part = f'{part[:-1]}{self.idxs[cur_ind]}]'
                cur_ind += 1
            jq_parts.append(part)
            if idx_count > 0 and cur_ind >= idx_count:
                break
        return '.'.join(jq_parts)
                

class CPathFormatter(Formatter):
    '''
    A path formatter for items, collapsed paths, and path indexes.
    '''

    def __init__(self, delim='\t'):
        '''
        :param delim: The line field delimiter
        '''
        super().__init__(delim)

    def format_path(self, jq_path: str, item: Any) -> str:
        '''
        Format the jq_path and item as a collapsed path (cjq_path) with indexes:
            item \t cjq_path \t indexes

        Where
          * cjq_path is the jq_path with "[]" for each indexed element
          * indexes is formatted as comma-delimitted:
                "<idx-1>, <idx-2>, ..., <idx-N>"
          * such that idx-i corresponds to the i-th jq_path indexed element

        :param jq_path: A fully indexed path
        :param item: The path's item value
        '''
        idxs = ', '.join(jutils.FLATTEN_IDX_RE.findall(jq_path))
        cjq_path = jutils.FLATTEN_IDX_RE.sub('[]', jq_path)
        return f'{item}\t{cjq_path}\t{idxs}'

    def get_cpath_parts(self, line: str) -> CPathParts:
        '''
        Get a CPathParts instance for the line.
        '''
        return CPathParts(self.split_line(line))

    def reverse_line(self, line: str) -> Tuple[str, str]:
        '''
        Reverse a line built by this formatter back to the original jq_path
        and its item.
        :param line: A line produced by this formatter.
        :return: The (jq_path, item) tuple
        '''
        cpath_parts = get_cpath_parts(line)
        idxs = cpath_parts.idxstr.split(', ')  # ok to leave as strs
        jq_path = cpath_parts.rebuild_path()
        return (jq_path, cpath_parts.item)


class CPathValueLocator:

    def __init__(self, path2value: Dict[str, str]):
        self.path2value = path2value
        self._finding = dict()
        self._found = None

    def __call__(self, cpath_parts: CPathParts) -> bool:
        matches = False
        for vidx, (path, value) in enumerate(self.path2value.items()):
            matches = (
                cpath_parts.item == value and
                cpath_parts.cjq_path.find(path) >= 0
            )
            if matches:
                # Found a match
                if vidx in self._finding:
                    # Found another before having found all ...start over
                    self._finding = dict()
                self._finding[vidx] = cpath_parts
        if matches and len(self._finding) == len(self.path2value):
            # Found all!
            self._found = self._finding
            self._finding = dict()
        else:
            matches = False
        return matches

    def has_values(self) -> bool:
        return self._found is not None

    @property
    def values(self) -> List[CPathParts]:
        '''
        Return the cpath_parts that have been found matching the values.
        '''
        return list(self._found.values()) if self._found else None
        

class JsonDataOrganizer:
    '''
    A method for organizing high volume structured json data for random access.

    The data is organized in two structures:
        (1) A "paths" structure associating items with their fully indexed paths
        (2) A "cpaths" structure with items and paths collapsed and indexes separated

    This illustrative class organizes these structures as two files:
        (1) A "paths" file, holding items with their fully indexed paths
        (2) A "cpaths" file holding collapsed paths, indexes and items
    '''
    def __init__(
            self,
            jdata_path: str,
            paths_sfx: str = '.paths.tsv',
            cpaths_sfx: str = '.cpaths.tsv',
            path_formatter: PathFormatter = None,
            cpath_formatter: CPathFormatter = None,
            paths_path: str = None,
            cpaths_path: str = None,
            lazy_build: bool = False,
            overwrite: bool = False,
            lockdown: bool = False,
    ):
        '''
        :param jdata_path: The path to the json data
        :param paths_sfx: The suffix for the paths file
        :param cpaths_sfx: The suffix for the cpaths file
        :param path_formatter: A PathFormatter for formatting path lines
        :param cpath_format_fn: A CPathFormatter for formatting cpath lines
        :param paths_path: Override for paths file location
        :param cpaths_path: Override for cpaths file location
        :param lazy_build: If True, only build files when needed
        :param overwrite: If True force re-build of files
        :param lockdown: If True, don't create any files
        '''
        self.lazy_build = lazy_build
        self.overwrite = overwrite
        self.lockdown = lockdown
        self.jdata_path = jdata_path
        self.base_path = os.path.splitext(self.jdata_path)[0]
        self.paths_sfx = paths_sfx
        self.cpaths_sfx = cpaths_sfx
        self.paths_path = self._build_path(paths_path, self.paths_sfx)
        self.cpaths_path = self._build_path(cpaths_path, self.cpaths_sfx)
        self.path_formatter = path_formatter or PathFormatter()
        self.cpath_formatter = cpath_formatter or CPathFormatter()
        if not self.lazy_build:
            self._build_path_files()

    def _build_path(self, override: str, sfx: str) -> str:
        '''
        Build a self.base_path + sfx path, adding '.gz' if it exists.
        '''
        path = override or self.base_path + sfx
        if os.path.exists(path + '.gz'):
            path = path + '.gz'
        return path

    def write_paths(self):
        '''
        Stream the json data to the outfile as fully indexed paths to each item,
        one per line, as:
            jq_path \t item

        If the file exists (and not overwrite) assume it is correct.
        '''
        if self.lockdown:
            return
        if self.overwrite or not os.path.exists(self.paths_path):
            self.overwrite = False
            with open(self.paths_path, 'w', encoding='utf-8') as outfile:
                jutils.write_squashed(outfile, self.jdata)

    def write_cpaths(self):
        '''
        Stream the json data to the outfile according to the format_fn, using the
        cpath_formatter function (above) by default.

        If the file exists (and not overwrite) assume it is correct.
        '''
        if self.lockdown:
            return
        if self.overwrite or not os.path.exists(self.cpaths_path):
            self.overwrite = False
            with open(self.cpaths_path, 'w', encoding='utf-8') as outfile:
                jutils.write_squashed(
                    outfile, self.jdata, format_fn=self.cpath_format_fn
                )

    def _build_path_files(self):
        '''
        Ensure both path and cpath files are built, rebuilding if overwrite.
        '''
        if self.lockdown:
            return
        files = list()
        if self.overwrite or not os.path.exists(self.paths_path):
            files.append(
                (
                    open(self.paths_path, 'w', encoding='utf-8'),
                    self.path_formatter
                )
            )
        if self.overwrite or not os.path.exists(self.cpaths_path):
            files.append(
                (
                    open(self.cpaths_path, 'w', encoding='utf-8'),
                    self.cpath_formatter
                )
            )
        if len(files) > 0:
            self.overwrite = False
            def format_fn(jq_path, item):
                for f, ffn in files:
                    print(ffn(jq_path, item), file=f)
            jutils.squash_data(format_fn, self.jdata_path)
            # Close the files
            for f, _ in files:
                f.close()

    def cpath_line_generator(
            self,
            match_fn: Callable[[CPathParts], bool]
    ):  # yields List[CPathParts]:
        '''
        Generate CPathParts instances where the match_fn is True.
        :param match_fn: A fn(CPathParts) that returns True on cpath lines to
            yield
        '''
        self.write_cpaths()  # ensure cpaths exists
        if not os.path.exists(self.cpaths_path):
            return
        open_fn, mode = (
            (gzip.open, 'rt')
            if self.cpaths_path.endswith('.gz')
            else (open, 'r')
        )
        with open_fn(self.cpaths_path, mode, encoding='utf-8') as f:
            for line in f:
                cpath_parts = self.cpath_formatter.get_cpath_parts(line)
                if match_fn(cpath_parts):
                    yield cpath_parts

    def path_line_generator(
            self,
            match_fn: Callable[[str, str], bool]
    ):  # yields List[Tuple[str, str]]:
        '''
        Generate (jq_path, item) tuples where match_fn(jq_path, item) is True.
        :param match_fn: A fn(jq_path, item) that returns true on path lines to
            yield
        '''
        self.write_paths()  # ensure paths exists
        if not os.path.exists(self.paths_path):
            return
        open_fn = gzip.open if self.paths_path.endswith('.gz') else open
        with open_fn(self.paths_path, 'r', encoding='utf-8') as f:
            for line in f:
                jq_path, item = self.path_formatter.split_line(line)
                if match_fn is None or match_fn(jq_path, item):
                    yield (jq_path, item)

    def cpaths_with_values_generator(
            self,
            path2value: Dict[str, str],
    ):  # yields List[CPathParts]:
        '''
        Generator for a List of CPathParts with matching path_values.
        
        :param path2value: A dictionary of {<path>:<value>}, to find cpath lines
           where "path" is contained in the cjq_path with an equal value for all
           of the path+values in the dict.
        '''
        value_locator = CPathValueLocator(path2value)
        for cpath_parts in self.cpath_line_generator(value_locator):
            values = value_locator.values
            if values is not None:
                yield values

    def records_with_values_generator(
            self,
            path2value: Dict[str, str],
    ):  # yields List[Tuple[str, Any]]
        '''
        Generate "path" records having all of the specified values at the
        jq_paths, where records are comprised of a sequence of (jq_path, item)
        tuples.
        
        :param path2value: A dictionary of {<path>:<value>}, to find path lines
           where "path" is contained in the cjq_path with an equal value for all
           of the path+values in the dict.
        '''
        self._build_path_files()  # make sure both path files exist

        rows = list()
        record_generator = self.path_line_generator(None)
        prev_pathline = None

        # Find cpath lines with matching values
        for cpaths in self.cpaths_with_values_generator(path2value):
            jq_path_pfx = self._get_jq_path_pfx(cpaths)
            is_in_block = False
            if jq_path_pfx is not None:
                if (
                        prev_pathline is not None and
                        prev_pathline[0].startswith(jq_path_pfx)
                ):
                    is_in_block = True
                    rows.append(prev_pathline)
                # increment up through matching records
                for jq_path, item in record_generator:
                    if jq_path.startswith(jq_path_pfx):
                        # path is in record block being collected
                        rows.append((jq_path, item))
                        is_in_block = True
                    elif is_in_block:
                        # just exited the block
                        prev_pathline = (jq_path, item)
                        is_in_block = False
                        yield rows
                        rows = list()
                        break
                    # else -- not yet in or beyond the block. keep going.

    def records_generator(
            self,
            accept_rec_fn: Callable[[str, str], bool] = None,
    ):  # yields List[Tuple[str, Any]]
        '''
        Generate (accepted) "path" records, where records are comprised of a
        sequence of (jq_path, item) tuples.
        
        :param accept_rec_fn: A fn(jq_path, item) returning True for records
            to accept. If None, all records are accepted.
        '''
        self._build_path_files()  # make sure both path files exist

        rows = list()
        record_generator = self.path_line_generator(None)
        cur_pfx = None

        # increment up through acceptable records
        for jq_path, item in self.path_line_generator(accept_rec_fn):
            first_arraypos = jq_path.find('].')
            jq_path_pfx = (
                jq_path
                if first_arraypos < 0
                else jq_path[:first_arraypos+1]
            )
            if cur_pfx is None or cur_pfx == jq_path_pfx:
                # Is in block
                rows.append((jq_path, item))
            else:
                # Starting a new block
                if len(rows) > 0:
                    yield rows
                rows = [(jq_path, item)]
            cur_pfx = jq_path_pfx

        # yield the final set of rows
        if len(rows) > 0:
            yield rows

    def _get_jq_path_pfx(self, cpps: List[CPathParts]) -> str:
        '''
        Get the fully indexed jq_path prefix common to the cpaths.
        '''
        pfx = None
        cpath_parts = None
        path_idx = None
        for cpp in cpps:
            if cpp.idxstr:
                cur_path_idx = cpp.idxs[0]
                if path_idx is None:
                    path_idx = cur_path_idx
                    cpath_parts = cpp
                elif path_idx != cur_path_idx:
                    # discordant paths
                    return None
        if cpath_parts is not None:
            pfx = cpath_parts.rebuild_path(1)
        return pfx

import gzip
import io
import json_stream.requests
import os
import pandas as pd
import re
import requests
import dataknobs.structures.tree as dk_tree
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple, Union


ELT_IDX_RE = re.compile(r'^(.*)\[(.*)\]$')
FLATTEN_IDX_RE = re.compile(r'\[(\d+)\]')
URL_RE = re.compile(r'^\s*https?://.*$', flags=re.IGNORECASE)
TIMEOUT = 10  # 10 seconds


def stream_json_data(
        json_data: str,
        visitor_fn: Callable[Any, str],
        timeout: int = TIMEOUT,
):
    '''
    Stream the json data, calling the visitor_fn at each value.
    :param json_data: The json data (url, file_path, or str)
    :param visitor_fn: The visitor_fn(item, path) to call, where
        item is each json item's value and path is the tuple of
        elements identifying the path to the item.
    :param timeout: The requests timeout (in seconds)
    '''
    if os.path.exists(json_data):
        if json_data.endswith('.gz') or '.gz?' in json_data:
            with gzip.open(json_data, 'rt', encoding='utf-8') as f:
                json_stream.visit(f, visitor_fn)
        else:
            with open(json_data, 'r', encoding='utf-8') as f:
                json_stream.visit(f, visitor_fn)
    elif json_data.startswith('http'):
        with requests.get(json_data, stream=True, timeout=timeout) as response:
            json_stream.requests.visit(response, visitor_fn)
    elif isinstance(json_data, str):
        f = io.StringIO(json_data)
        json_stream.visit(f, visitor_fn)


def build_jq_path(path: Tuple[Any], keep_list_idxs=True) -> str:
    '''
    Build a jq path string from the json_stream path tuple.
    :param path: A tuple with json_stream path components.
    :param keep_list_idxs: True to keep the exact path index values;
        False to emit a generic "[]" for the list.
    '''
    jq_path = ''
    for elt in path:
        if isinstance(elt, int):
            jq_path += f'[{elt}]' if keep_list_idxs and elt>=0 else '[]'
        else:
            jq_path += f'.{elt}'
    return jq_path


def build_path_tuple(jq_path: str, any_list_idx: int = -1) -> Tuple:
    '''
    Build a json_stream tuple path from a jq_path (reverse of build_jq_path).
    :param jq_path: The jq_path whose values to extract.
    :param any_list_idx: The index value to give for a generic list
    :return: The json_stream tuple form of the path
    '''
    path = list()
    for part in jq_path.split('.'):
        if part == '':
            continue
        m = ELT_IDX_RE.match(part)
        if m:
            path.append(m.group(1))
            idx = m.group(2)
            idxval = any_list_idx if idx == '' else int(idx)
            path.append(idxval)
        else:
            path.append(part)
    return tuple(path)


def stream_jq_paths(
        json_data: str,
        output_stream,
        line_builder_fn: Callable[[str, Any], str] = (
            lambda jq_path,item: f'{jq_path}\t{item}'
        ),
        keep_list_idxs: bool = True,
        timeout: int = TIMEOUT,
):
    '''
    Write built lines from (jq_path, item) tuples from the json_data.
    :param json_data: The json_data to copy
    :param output_stream: The output stream to write lines to
    :param line_builder_fn: The function for turning record tuples into a line
        string.
    :param keep_list_idxs: True to keep the exact path index values;
        False to emit a generic "[]" for the list.
    :param timeout: The requests timeout (in seconds)
    '''
    def visitor(item, path):
        jq_path = build_jq_path(path, keep_list_idxs=keep_list_idxs)
        line = line_builder_fn(jq_path, item)
        print(line, file=output_stream)

    stream_json_data(json_data, visitor, timeout=timeout)


def squash_data(
        builder_fn: Callable[str, Any],
        json_data: str,
        prune_at: List[Union[str, Tuple[str, int]]] = None,
        timeout: int = TIMEOUT,
):
    '''
    Squash the json_data, optionally pruning branches by id'd path elements.
    Where "squashed" means the paths are compressed to a single level
    with complex jq keys.

    Each path to prune is identified by:
        * (path_element_name:str, path_index:int)
            -- or paths where path[path_index] == path_element_name
        * path_element_name:str or (path_element_name, None)
            -- or any path element whose name is path_element_name
        * path_index:int or (None, path_index)
            -- or any path at path_index depth

    :param builder_fn: fn(jq_path, item) for building a result
    :param json_data: The json_data to copy
    :param prune_at: The names and optional path indexes of path elements
        to ignore
    :param timeout: The requests timeout (in seconds)
    '''
    raw_depths = set()
    raw_elts = set()
    elt_depths = dict()
    depth_elts = defaultdict(set)

    def decode_item(item):
        if item is not None:
            if isinstance(item, str):
                raw_elts.add(item)
            elif isinstance(item, int):
                raw_depths.add(item)
            elif isinstance(item, tuple) and len(item) == 2:
                elt = item[0]
                depth = item[1]
                if elt is None:
                    if depth is not None:
                        raw_depths.add(depth)
                else:
                    if depth is not None:
                        depth_elts[depth].add(elt)
                        elt_depths[elt] = depth
                    else:
                        raw_elts.add(elt)
            elif isinstance(item, list) or isinstance(item, tuple):
                for i in item:
                    decode_item(i)

    decode_item(prune_at)
    has_raw_depths = len(raw_depths) > 0
    has_raw_elts = len(raw_elts) > 0
    has_elts = len(elt_depths) > 0
    has_depth_elts = len(depth_elts) > 0
    do_prune = has_raw_depths or has_raw_elts or has_elts or has_depth_elts

    def visitor(item, path):
        if do_prune:
            cur_depth = len(path)
            if has_raw_depths and cur_depth in raw_depths:
                return
            if has_raw_elts:
                if len(raw_elts.intersection(path)) > 0:
                    return
            cur_elt = path[-1]
            if has_elts and cur_elt in elt_depths:
                if cur_depth == elt_depths[cur_elt]:
                    return
            if has_depth_elts:
                for depth, elts in depth_elts.items():
                    if depth < cur_depth and path[depth] in elts:
                        return
        # Add squashed element
        jq_path = build_jq_path(path, keep_list_idxs=True)
        builder_fn(jq_path, item)

    stream_json_data(json_data, visitor, timeout=timeout)


def collect_squashed(
        jdata: str,
        prune_at: List[Union[str, Tuple[str, int]]] = None,
        timeout: int = TIMEOUT,
        result: Dict = None,
) -> Dict:
    '''
    Collected squashed data in a dictionary
    :param json_data: The json_data to copy
    :param prune_at: The names and optional path indexes of path elements
        to ignore
    :param timeout: The requests timeout (in seconds)
    :param result: (optional) The dictionary in which to collect items
    '''
    if result is None:
        result = dict()
    def collector_fn(jq_path, item):
        result[jq_path] = item
    squash_data(
        collector_fn,
        jdata,
        prune_at=prune_at,
        timeout=timeout,
    )
    return result


def indexing_format_fn(jq_path: str, item: Any) -> str:
    '''
    A formatting function for writing a csv with the columns:
        item (value) \t field \t flat_jq \t idxs

    Where
        * value -- is the item value
        * field -- is the last flat_jq path element
        * flat_jq -- is the jq_path with all indexes flattened "[.*]" -> "[]"
                     up to the last path element
        * idxs -- is a comma-delimitted string with the indices
    '''
    idxs = ', '.join(FLATTEN_IDX_RE.findall(jq_path))
    flat_jq = FLATTEN_IDX_RE.sub('[]', jq_path)
    dotpos = flat_jq.rindex('.')
    field = flat_jq[dotpos+1:]
    flat_jq = flat_jq[:dotpos]
    return f'{item}\t{field}\t{flat_jq}\t{idxs}'


def indexing_format_splitter(fileline: str) -> Tuple[str, str, str, str]:
    '''
    Reversal of the indexing_format_fn to extract:
        (value, field, flat_jq, idxs)
    Where
        * value -- is the item value
        * field -- is the last flat_jq path element
        * flat_jq -- is the jq_path with all indexes flattened "[.*]" -> "[]"
                     up to the last path element
        * idxs -- is a comma-delimitted string with the indices
    '''
    line = fileline.strip()
    value = None
    field = None
    flat_jq = None
    idxs = None
    if line:
        parts = fileline.split('\t')
        value = parts[0]
        if len(parts) > 1:
            field = parts[1] 
            if len(parts) > 2:
                flat_jq = parts[2] 
                if len(parts) > 3:
                    idxs = parts[3] 
    return (value, field, flat_jq, idxs)


def write_squashed(
        dest_file: str,
        jdata: str,
        prune_at: List[Union[str, Tuple[str, int]]] = None,
        timeout: int = TIMEOUT,
        format_fn: Callable[[str, Any], str] = lambda jq_path, item: f'{jq_path}\t{item}',
):
    '''
    Write squashed data to the file.
    :param json_data: The json_data to copy
    :param prune_at: The names and optional path indexes of path elements
        to ignore
    :param timeout: The requests timeout (in seconds)
    :param format_fn: A function for formatting each output line. Default is
        tab-delimitted: jq_path <tab> item.
    '''
    needs_close = False
    if isinstance(dest_file, str):
        f = open(dest_file, 'w', encoding='utf-8')
        needs_close = True
    else:
        f = dest_file
    squash_data(
        lambda jq_path, item: print(format_fn(jq_path, item), file=f),
        jdata,
        prune_at=prune_at,
        timeout=timeout,
    )
    if needs_close:
        f.close()


def path_to_dict(path: Union[tuple, str], value: Any, result=None) -> dict:
    '''
    Convert the jq_path (if a string) or path (if a tuple) to a dict.
    :param path: The path to convert
    :param value: The path's value
    :param result: The dictionary to add the result to
    :return: the result
    '''
    if result is None:
        result = dict()
    if isinstance(path, str):
        path = build_path_tuple(path, any_list_idx=-1)

    def do_it(cur_dict, path, path_idx, pathlen, value):
        if path_idx >= pathlen:
            return
        path_elt = path[path_idx]
        path_idx += 1
        list_idx = None
        if path_idx < pathlen:
            next_elt = path[path_idx]
            if isinstance(next_elt, int):
                list_idx = next_elt
                path_idx += 1

        if path_elt in cur_dict:
            if list_idx is None:
                if path_idx < pathlen:
                    cur_dict = cur_dict[path_elt]
                else:
                    cur_dict[path_elt] = value
            else:
                cur_list = cur_dict[path_elt]
                if len(cur_list) <= list_idx:
                    if path_idx < pathlen:
                        cur_dict = dict()
                        # simplifying assumption: idxs are in consecutive order fm 0
                        cur_list.append(cur_dict)
                    else:
                        cur_list.append(value)
                else:
                    if path_idx < pathlen:
                        cur_dict = cur_list[-1]
                    else:
                        cur_list.append(value)
        else:
            if list_idx is None:
                if path_idx < pathlen:
                    elt_dict = dict()
                    cur_dict[path_elt] = elt_dict
                    cur_dict = elt_dict
                else:
                    cur_dict[path_elt] = value
            else:
                cur_list = list()
                cur_dict[path_elt] = cur_list
                if path_idx < pathlen:
                    cur_dict = dict()
                    cur_list.append(cur_dict)
                else:
                    cur_list.append(value)
        # recurse to keep moving along the path
        do_it(cur_dict, path, path_idx, pathlen, value)


    do_it(result, path, 0, len(path), value)

    return result


def explode(squashed: Dict) -> Dict:
    '''
    Explode a "squashed" json with jq paths as keys.
    '''
    result = dict()
    for jq_path, value in squashed.items():
        path_to_dict(jq_path, value, result)
    return result


class ValuePath:
    '''
    Structure to hold (compressed) information about the jq_path indices
    leading to a unique value.

    Essentially, this the tree of path indices leading to the value.
    '''
    def __init__(self, jq_path: str, value: Any):
        '''
        :param jq_path: The jq_path (key)
        :param value: The path's value
        '''
        self.jq_path = jq_path
        self.value = value
        self._indices = dk_tree.Tree(0).as_string()  # root data will hold total path count

    @property
    def indices(self) -> dk_tree:
        return dk_tree.build_tree_from_string(self._indices)

    def add(self, path: tuple):
        '''
        Add the path (with the samem structure as jq_path) to the tree.
        '''
        root = self.indices
        node = root
        node.data = int(node.data) + 1  # keep track of total
        if path is None:
            self._indices = root.as_string()
            return
        for elt in path:
            if isinstance(elt, int):
                found = False
                if node.has_children():
                    # simplifying assumption: idxs are in consecutive order fm 0
                    child = node.children[-1]
                    if str(elt) == child.data:
                        node = child
                        found = True
                if not found:
                    node = node.add_child(elt)
        self._indices = root.as_string()

    @property
    def path_count(self) -> int:
        ''' Get the number of jq_paths to the value. '''
        return int(self.indices.data)

    def path_generator(self, result_type='jq_path'):
        '''
        Generate value paths.
        :param result_type: 'jq_path', 'path', or 'idx'
            'jq_path' to generate jq_path strings;
            'path' to generate path tuples
            'idx' to generate index tuples.
        '''
        path = build_path_tuple(self.jq_path, any_list_idx=-1)
        for node in self.indices.collect_terminal_nodes():
            node_path = node.get_path()
            node_idx = 0
            gen_path = list()
            for elt in path:
                if isinstance(elt, int):
                    gen_path.append(int(node_path[node_idx].data))
                    node_idx += 1
                elif result_type != 'idx':
                    gen_path.append(elt)
            if result_type == 'jq_path':
                yield build_jq_path(gen_path, keep_list_idxs=True)
            else:
                yield gen_path


class ValuesIndex:
    '''
    An index of unique values by jpath and (optionally) an index of values to
    their paths for each jpath.

    The values to paths index is compressed such that each unique value mapped
    to the tree of path indices leading to it, if path information is available.
    '''
    def __init__(self):
        self.path_values = dict()  # Dict[jq_path, Dict[value, ValuePath]]

    def add(self, value: Any, jq_path: str, path: tuple = None):
        if jq_path in self.path_values:
            value_paths = self.path_values[jq_path]
        else:
            value_paths = dict()
            self.path_values[jq_path] = value_paths

        if value == []:
            value = '_EMPTY_LIST_'
        elif value == {}:
            value = '_EMPTY_DICT_'
            
        if value in value_paths:
            value_path = value_paths[value]
        else:
            value_path = ValuePath(jq_path, value)
            value_paths[value] = value_path

        value_path.add(path)

    def has_jqpath(self, jq_path: str) -> bool:
        ''' Determine whether there are any values for jq_path '''
        return jq_path in self.path_values

    def get_values(self, jq_path: str) -> Set[Any]:
        ''' Get the set of values for jq_path '''
        return set(self.path_values.get(jq_path, {}).keys())

    def num_values(self, jq_path: str) -> int:
        ''' Get the number of values for jq_path '''
        return len(self.path_values.get(jq_path, {}))


class JsonSchema:
    '''
    Container for a schema view of a json object of the form:

    {
        <jq_path>: {
            <value_type>: <value_count>,
        }
        ...,
    }

    Where,
        * value_type is the type of value at the path (e.g., int, float,
            str, etc.)
        * value_count is the number of types the value_type occurs in an object.
    '''

    def __init__(
            self,
            schema: Dict[str, Any] = None,
            values: ValuesIndex = None,
            values_limit: int = 0,  # max number of unique values to keep
    ):
        self.schema = schema if schema is not None else dict()
        self.values = ValuesIndex() if values is None else values
        self._values_limit = values_limit
        self._df = None

    def add_path(
            self,
            jq_path: str,
            value_type: str,
            value: Any = None,
            path: tuple = None,
    ):
        '''
        Add an instance of the jq_path/value_type
        :param jq_path: The "key" path for grouping/squashing values
        :param value_type: The type of value with this path
        :param value: (optional) The value for tracking unique values (if not
            None)
        :param path: (optional) The path tuple for inverting paths to uniques
            (if not None and if value is also not None)
        '''
        if jq_path not in self.schema:
            self.schema[jq_path] = {value_type: 1}
        else:
            if value_type not in self.schema[jq_path]:
                self.schema[jq_path][value_type] = 1
            else:
                self.schema[jq_path][value_type] += 1
        if value is not None:
            if (
                    self._values_limit == 0 or
                    self.values.num_values(jq_path) < self._values_limit
            ):
                self.values.add(value, jq_path, path=path)
        self._df = None
        
    @property
    def df(self) -> pd.DataFrame:
        '''
        Get schema information as a DataFrame with columns:

            jq_path    value_type    value_count
        '''
        if self._df is None:
            self._df = self._build_df()
        return self._df

    def _build_df(self) -> pd.DataFrame:
        '''
        Get schema information as a DataFrame with columns:

            jq_path    value_type    value_count    [unique_count]
        '''
        data = list()
        has_value = False
        for k1, v1 in self.schema.items():  # jq_path -> [value_type -> value_count]
            for k2, v2 in v1.items():  # value_type -> value_count
                if self.values.has_jqpath(k1):
                    row = (k1, k2, v2, self.values.num_values(k1))
                    has_value = True
                else:
                    row = (k1, k2, v2)
                data.append(row)
        columns = ['jq_path', 'value_type', 'value_count']
        if has_value:
            columns.append('unique_count')
        return pd.DataFrame(data, columns=columns)

    def extract_values(
            self,
            jq_path: str,
            json_data: str,
            unique: bool = True,
            timeout: int = TIMEOUT,
    ) -> Union[List[Any], Set[Any]]:
        '''
        Extract values from the json_data's jq_path.
        :param jq_path: The jq_path whose values to extract.
        :param json_data: The json data (url, file_path, or str)
        :param unique: True to collect only unique values
        :param timeout: The requests timeout (in seconds)
        :return: The list (or set if unique) of values.
        '''
        keep_list_idxs = False if '[]' in jq_path else False
        sresult = set()
        lresult = list()
        def visitor(item, path):
            cur_jq_path = build_jq_path(path, keep_list_idxs=keep_list_idxs)
            if jq_path == cur_jq_path:
                if unique:
                    sresult.add(item)
                else:
                    lresult.append(item)
        stream_json_data(json_data, visitor, timeout=timeout)
        return sresult if unique else lresult


class JsonSchemaBuilder:
    '''
    Create a schema view of a json object.
    '''

    def __init__(
            self,
            json_data: str,
            value_typer: Callable[[Any], str] = None,
            keep_unique_values: bool = False,
            invert_uniques: bool = False,
            keep_list_idxs: bool = False,
            timeout: int = TIMEOUT,
            empty_dict_type: str = '_EMPTY_DICT_',
            empty_list_type: str = '_EMPTY_LIST_',
            unk_value_type: str = '_UNKNOWN_',
            int_value_type: str = 'int',
            float_value_type: str = 'float',
            str_value_type: str = 'str',
            url_value_type: str = 'URL',
    ):
        '''
        :param json_data: The json data (url, file_path, or str)
        :param value_typer: A fn(atomic_value) that returns the type of the
            value to override the default typing of "int", "float", and "str"
        :param keep_unique_values: True to keep unique values for each path or
            an integer for a maximum number of unique values to keep
        :param invert_uniques: True to keep value paths for unique values
        :param keep_list_idxs: True to keep the list indexes in the dictionary
            paths. When False, all list indexes will be generalized to "[]".
        :param timeout: The requests timeout (in seconds)
        :param empty_dict_type: The type of an empty dictionary
        :param empty_list_type: The type of an empty list
        :param unk_value_type: The type of an unknown/unclassified value
        :param int_value_type: The type of an int
        :param float_value_type: The type of a float
        :param str_value_type: The type of a string
        :param url_value_type: The value type of a URL if not None
        '''
        self.json_data = json_data
        self.value_typer = value_typer
        self.keep_uniques = keep_unique_values
        self.values_limit = (
            0
            if isinstance(keep_unique_values, bool)
            else int(keep_unique_values)
        )
        self.invert_uniques = invert_uniques
        self.keep_list_idxs = keep_list_idxs
        self.timeout = timeout
        self.empty_dict_type = empty_dict_type
        self.empty_list_type = empty_list_type
        self.unk_value_type = unk_value_type
        self.int_value_type = int_value_type
        self.float_value_type = float_value_type
        self.str_value_type = str_value_type
        self.url_value_type = url_value_type
        self._schema = None

    @property
    def schema(self) -> JsonSchema:
        ''' Get the schema for the json data '''
        if self._schema is None:
            self._schema = self._build_schema()
        return self._schema

    def _build_schema(self) -> JsonSchema:
        '''
        Stream the json data and build the schema.
        '''
        schema = JsonSchema(values_limit = self.values_limit)

        def visitor(item, path):
            self._visit_item(schema, item, path)

        stream_json_data(self.json_data, visitor, timeout=self.timeout)
        return schema

    def _visit_item(self, schema: JsonSchema, item: Any, path: Tuple):
        '''
        The visitor function for processing items and paths while streaming
        the json data.
        :param schema: The schema being built
        :param item: The next json item (value) encountered
        :param path: The path of the item as a tuple of the form:
            (<key1>, <list1-idx>, <key2>, ...)
        '''
        jq_path = build_jq_path(path, keep_list_idxs = self.keep_list_idxs)
        value_type = None
        if isinstance(item, dict):
            value_type = self.empty_dict_type
        elif isinstance(item, list):
            value_type = self.empty_list_type
        elif self.value_typer is not None:
            value_type = self.value_typer(item)
        if value_type is None:
            if isinstance(item, str):
                if self.url_value_type and URL_RE.match(item):
                    value_type = self.url_value_type
                else:
                    value_type = self.str_value_type
            elif isinstance(item, float):
                value_type = self.float_value_type
            elif isinstance(item, int):
                value_type = self.int_value_type
            else:
                value_type = self.unk_value_type
        schema.add_path(
            jq_path, value_type,
            value=(item if self.keep_uniques else None),
            path=(path if self.invert_uniques else None),
        )


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

    def __lt__(self, other: 'Path') -> bool:
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


class GroupAcceptStrategy(ABC):
    '''
    Add a Path to a Group if it belongs.
    '''
    @abstractmethod
    def accept_path(
            self,
            path: Path,
            group: 'PathGroup',
            distribute: bool = False
    ) -> str:
        '''
        Determine whether the Path belongs in the group.
        :param path: The path to potentially add
        :param group: The group to which to add the path
        :param distribute: True if the path is proposed as a distributed path
            instead of as a main path.
        :return: 'main', 'distributed', or None if the path belongs as a main
            path, a distributed path, or not at all in the group
        '''
        raise NotImplementedError


class PathGroup:
    '''
    Container for a group of related paths.
    '''
    def __init__(
            self,
            accept_strategy: GroupAcceptStrategy,
            first_path: Path = None
    ):
        self._all_paths = None
        self.main_paths = None  # Set[Path]
        self.distributed_paths = None  # Set[Path]
        self.accept_strategy = accept_strategy
        if first_path is not None:
            self.accept(first_path, distribute=False)

    @property
    def num_main_paths(self) -> int:
        ''' Get the number of main paths in this group '''
        return len(self.main_paths) if self.main_paths is not None else 0

    @property
    def num_distributed_paths(self) -> int:
        ''' Get the number of distributed paths in this group '''
        return (
            len(self.distributed_paths)
            if self.distributed_paths is not None
            else 0
        )

    @property
    def size(self) -> int:
        ''' Get the total number of paths in this group. '''
        return self.num_main_paths + self.num_distributed_paths

    @property
    def paths(self) -> List[Path]:
        ''' Get all paths (both main and distributed) '''
        if self._all_paths is None:
            if self.main_paths is not None:
                self._all_paths = self.main_paths.copy()
                if self.distributed_paths is not None:
                    self._all_paths.update(self.distributed_paths)
                self._all_paths = sorted(self._all_paths)
            elif self.distributed_paths is not None:
                self._all_paths = sorted(self.distributed_paths)
        return self._all_paths

    def as_dict(self) -> Dict[str, str]:
        ''' Reconstruct the object from the paths '''
        d = dict()
        if self.paths is not None:
            for path in self.paths:
                path_to_dict(path.jq_path, path.item, result=d)
        return d

    def accept(self, path: Path, distribute: bool = False) -> bool:
        '''
        Add the path if it belongs in this group.
        :param path: The path to (potentially) add.
        :param distribute: True to propose the path as a distributed path
        :return: True if the path was accepted and added.
        '''
        added = False
        add_type = self.accept_strategy.accept_path(
            path, self, distribute=distribute
        )
        if add_type is not None:
            if add_type == 'main':
                if self.main_paths is None:
                    self.main_paths = {path}
                else:
                    self.main_paths.add(path)
            else:  # 'distributed'
                if self.distributed_paths is None:
                    self.distributed_paths = {path}
                else:
                    self.distributed_paths.add(path)
            added = True
            self._all_paths = None
        return added

    def incorporate_paths(self, group: 'PathGroup'):
        '''
        Incorporate (distribute) the group's appliccable paths into this group.
        '''
        for path in group.paths:
            self.accept(path, distribute=True)


class ArrayElementAcceptStrategy(GroupAcceptStrategy):
    '''
    Container for a consistent group of paths built around each array.
    '''
    def __init__(self, max_array_level: int = -1):
        '''
        :param max_array_level: -1 to ignore, 0 to force new record at the
            first (top) array level, 1 at the 2nd, etc.
        '''
        self.max_array_level = max_array_level
        self.ref_path = None

    def accept_path(
            self,
            path: Path,
            group: PathGroup,
            distribute: bool = False
    ):
        if distribute or not '[' in path.jq_path:
            return 'distributed'

        if group.num_main_paths == 0:
            # Accept first path with an array
            self.ref_path = path
            return 'main'
        else:
            if self.ref_path is None:
                self.ref_path = list(group.main_paths)[0]
            # All elements up through max_array_level must fully match
            cur_array_level = -1
            for idx in range(1, min(self.ref_path.size, path.size)):
                ref_elt = self.ref_path.path_elts[idx]
                path_elt = path.path_elts[idx]
                if ref_elt != path_elt:
                    return None
                elif ref_elt[-1] == ']':
                    cur_array_level += 1
                    if cur_array_level >= self.max_array_level:
                        break
        return 'main'


class PathSorter:
    '''
    Container for sorting paths belonging together into groups.
    '''
    def __init__(
            self,
            accept_strategy: GroupAcceptStrategy,
            group_size: int = 0,
            max_groups: int = 0,
    ):
        '''
        :param accept_strategy: The group accept strategy to use
        :param group_size: A size constraint/expectation for groups such that
           any group not reaching this size (if > 0) is silently "dropped".
        :param max_groups: The maximum number of groups to keep in memory where
            all groups are kept if <= 0
        '''
        self.accept_strategy = accept_strategy
        self.group_size = group_size
        #NOTE: Must keep at least 3 groups if keeping any for propagating
        #      distributed paths.
        self.max_groups = max_groups if max_groups <= 0 else max(3, max_groups)
        self.groups = None  # List[PathGroup]

    @property
    def num_groups(self) -> int:
        ''' Get the number of groups '''
        return len(self.groups) if self.groups is not None else 0

    def add_path(self, path: Path) -> PathGroup:
        '''
        Add the path to an existing group, or create a new group.
        :param path: The path to add
        :return: each closed PathGroup, otherwise None.
        '''
        result = None
        if self.groups is None or len(self.groups) == 0:
            self.groups = [
                PathGroup(
                    accept_strategy = self.accept_strategy,
                    first_path=path,
                )
            ]
        else:
            # Assume always "incremental", such that new paths will belong in
            # the latest group
            latest_group = self.groups[-1]
            if not latest_group.accept(path):
                # Time to add a new group
                self.close_group()
                result = latest_group

                # Start a new group
                self.groups.append(
                    PathGroup(
                        accept_strategy = self.accept_strategy,
                        first_path=path,
                    )
                )

                # Enforce max_group limit by removing groups from the front
                if self.max_groups > 0 and len(self.groups) >= self.max_groups:
                    while len(self.groups) >= self.max_groups:
                        self.groups.pop(0)
        return result

    def close_group(self, idx: int = -1, check_size: bool = True) -> PathGroup:
        '''
        Close the (last) group by
          * Adding distributable lines from the prior group
          * Checking size constraints and dropping if warranted
        :return: The closed PathGroup
        '''
        if self.groups is None or len(self.groups) == 0:
            return None

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

        return latest_group

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


class RecordPathBuilder:
    '''
    Class for building record paths from json_data.
    '''
    def __init__(
            self,
            json_data: str,
            output_stream,
            line_builder_fn: Callable[[int, int, str, Any], str],
            timeout: int = TIMEOUT,
    ):
        '''
        :param json_data: The json data (url, file_path, or str)
        :param output_stream: The output stream to write lines to
        :param line_builder_fn: The function for turning record tuples into a line
            string.
        :param timeout: The requests timeout (if json_data is a url)
        '''
        self.jdata = json_data
        self.output_stream = output_stream
        self.builder_fn = line_builder_fn
        self.timeout = timeout
        self.rec_id = 0
        self.inum = 0
        self.sorter = None

    def write_group(self, group: PathGroup):
        for path in group.paths:
            line = self.builder_fn(
                self.rec_id, path.line_num, path.jq_path, path.item
            )
            print(line, file=self.output_stream)

    def visitor(self, item, path):
        jq_path = build_jq_path(path, keep_list_idxs=True)
        group = self.sorter.add_path(Path(jq_path, item, line_num=self.inum))
        if group is not None:
            self.write_group(group)
            self.rec_id += 1
        self.inum += 1

    def stream_record_paths(self):
        self.rec_id = 0
        self.inum = 0
        self.sorter = PathSorter(
            ArrayElementAcceptStrategy(max_array_level=0),
            max_groups=2,
        )

        stream_json_data(self.jdata, self.visitor, timeout=self.timeout)

        last_group = self.sorter.close_group(check_size=False)
        if last_group is not None:
            self.write_group(last_group)
        

def stream_record_paths(
        json_data: str,
        output_stream,
        line_builder_fn: Callable[[int, int, str, Any], str],
        timeout: int = TIMEOUT,
):
    '''
    Write built lines from (rec_id, item_num, jq_path, item) tuples of the
    top-level json "records" to the output stream where:
        * rec_id is a 0-based integer identifying unique records
        * line_num is a 0-based integer identifying the original item number
        * jq_path is the fully-qualified (indexed) path (a record attribute)
        * item is the item value at the path.
    :param json_data: The json data (url, file_path, or str)
    :param output_stream: The output stream to write lines to
    :param line_builder_fn: The function for turning record tuples into a line
        string.
    :param timeout: The requests timeout (if json_data is a url)
    '''
    rpb = RecordPathBuilder(
        json_data, output_stream, line_builder_fn, timeout=timeout
    )
    rpb.stream_record_paths()


def get_records_df(
        json_data: str,
        timeout: int = TIMEOUT
) -> pd.DataFrame:
    '''
    Convenience function to collect the (top-level) record lines from a json
    stream as a DataFrame with columns:
        record_id, line_num, jq_path, item

    WARNING: Don't use this method with very large streams.
    :param json_data: The source json data
    :param timeout: The requests timeout (if json_data is a url)
    :return: The record lines as a dataframe
    '''
    s = io.StringIO()
    stream_record_paths(
        json_data,
        s,
        lambda rid, lid, jqp, val: f'{rid}\t{lid}\t{jqp}\t{val}',
        timeout=timeout,
    )
    s.seek(0)
    df = pd.read_csv(s, sep='\t', names=['rec_id', 'line_num', 'jq_path', 'item'])
    return df

import gzip
import io
import json
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
    Build a jq path string from the path tuple.
    :param path: A tuple with path components.
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
    Build a tuple path from a jq_path (reverse of build_jq_path).
    :param jq_path: The jq_path whose values to extract.
    :param any_list_idx: The index value to give for a generic list
    :return: The tuple form of the path
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


def squash_data(
        builder_fn: Callable[str, Any],
        json_data: str,
        prune_at: List[Union[str, Tuple[str, int]]] = None,
        timeout: int = TIMEOUT,
):
    '''
    Squash the json_data, pruning branches with path the given names.
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


class BlockCollector:
    '''
    A class for collecting json blocks surrounding a matching path and/or value
    '''

    def __init__(
            self,
            jq_path: str,
            item_value: Any = None,
            block_path_idx: int = None,
            timeout: int = TIMEOUT,
    ):
        '''
        :param jq_path: The path at which to find the item.
        :param item_value: The item value to find to identify the block
        :param block_path_idx: The index in the path tuple to be the "top" of
            block extracted.
        :param timeout: The requests timeout (in seconds)
        '''
        self.jq_path = jq_path
        self.timeout = timeout
        self.keep_list_idxs = False if '[]' in self.jq_path else False
        self.path = build_path_tuple(self.jq_path)
        self.list_idxs = [
            idx for idx in range(len(self.path))
            if isinstance(self.path[idx], int)
        ]
        self.item_value = item_value
        self._path_idx = (
            block_path_idx if block_path_idx is not None
            else max(self.list_idxs) if len(self.list_idxs) > 0
            else -1
        )
        self.path = build_path_tuple(jq_path)
        self._jqpp = None
        self._jqpi = None
        self._jqpf = None
        if self._path_idx >= 0:
            self._jqpp = build_jq_path(self.path[:self._path_idx], keep_list_idxs=False)
            self._jqpi = build_jq_path(self.path[:self._path_idx+1], keep_list_idxs=False)
            self._jqpf = build_jq_path(self.path, keep_list_idxs=False)

        self._keeper = False
        self._cur_path = None
        self._cur_path_idx = -1
        self._cur_block = None

    def _reset(self):
        self._keeper = False
        self._cur_path = None
        self._cur_path_idx = -1
        self._cur_block = None

    def _update(self, path: str, item: Any) -> bool:
        result = None

        pidx = self._path_idx
        idx = pidx + 1
        if self._cur_path is not None:
            # Check if moved out of block
            p1 = None
            if len(path) >= idx:
                p1 = build_jq_path(path[:idx], keep_list_idxs=False)
            if (
                    len(path) < idx or
                    p1 != self._jqpi or
                    (self._cur_path_idx >= 0 and self._cur_path_idx != path[pidx])
            ):
                # block is done
                if self._keeper:
                    result = self._cur_block
                self._cur_block = None
                self._cur_path = None
                self._keeper = False
            #else still in block

        elif len(path) > pidx:
            # Check if moved into block
            p1 = None
            if pidx >= 0:
                p1 = build_jq_path(path[:pidx], keep_list_idxs=False)
            if (
                    pidx < 0 or
                    p1 == self._jqpp
            ):
                self._cur_path = path
                self._cur_path_idx = path[pidx] if pidx >= 0 else 0
                self._cur_block = dict()

        if self._cur_path is not None:
            # In a capture block ... add block value
            jq_path = build_jq_path(path, keep_list_idxs=True)
            self._cur_block[jq_path] = item
            p1 = build_jq_path(path, keep_list_idxs=False)
            if (
                    p1 == self._jqpf and (
                        self.item_value is None or item == self.item_value
                    )
            ):
                self._keeper = True

        return result

    def collect_blocks(self, json_data: str, max_count: int = 0):
        self._reset()
        result = list()
        def visitor(item, path):
            if max_count == 0 or len(result) < max_count:
                block = self._update(path, item)
                if block is not None:
                    result.append(block)
        stream_json_data(json_data, visitor, timeout=self.timeout)
        if self._keeper:  # pop the last item
            result.append(self._cur_block)
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

    def collect_value_blocks(
            self,
            jq_path: str,
            item_value: Any,
            json_data: str,
            max_count: int = 0,
    ) -> List[Dict]:
        '''
        Collect blocks from json_data where the item_value is found at the
        jq_path.
        :param jq_path: The path at which to find the item.
        :param item_value: The item value to find to identify the block
        :param json_data: The json_data from which to collect blocks
        :param max_count: The maximum number of blocks to collect (0 for no
            limit)
        :return: The json block data.
        '''
        collector = BlockCollector(jq_path, item_value=item_value)
        return collector.collect_blocks(json_data, max_count=max_count)


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


def clean_jq_style_records(jq_rec: Dict, lower: bool=False) -> Dict:
    '''
    Clean the record's attributes from jq-style to db-style, by keeping
    (and lowercasing) only the last jq path component and dropping square
    brackets.
    Clean the record's values by converting lists of values to a comma-
    plus-space-delimitted string of values.
    :param jq_rec: The record to clean.
    :param lower: True to lowercase the attributes
    :return: The cleaned record
    '''
    def clean_attr(key: str) -> str:
        key = key.split('.')[-1]
        if lower:
            key = key.lower()
        if key.endswith('[]'):
            key = key[:-2]
        return key

    def clean_value(value: Any) -> Any:
        if isinstance(value, list):
            value = ', '.join([str(v) for v in value])
        return value

    return {
        clean_attr(key): clean_value(value)
        for key, value in jq_rec.items()
    }


class FlatRecordBuilder:
    '''
    Build "flat" (simple, shallow) records from squashed rows.
    '''
    def __init__(
            self,
            full_path_attrs: bool = False,
            pivot_pfx: str = None,
            ignore_pfxs: Set[str] = None,
            jq_clean: Callable[[Dict], Dict] = None,
            add_idx_attrs: bool = False,
    ):
        '''
        :param full_path_attrs: True to use the full path for record attributes
        :param pivot_pfx: The prefix at which to "pivot" for grouping values
        :param ignore_pfxs: The path prefixes to ignore -- None to not ignore any,
            an empty set (or string) to ignore everything other than pivot_pfx,
            or the set of path prefixes to ignore. Note that the records built
            from selectively ignoring paths include paths from broader scopes,
            while those build ignoring all other paths do not.
        :param jq_clean: Function to clean the final record's attributes
        :param add_idx_attrs: True to add "link__" attributes to recs for array
            indeces
        '''
        self.full_path_attrs = full_path_attrs
        self.jq_clean = jq_clean
        self.add_idx_attrs = add_idx_attrs
        self.pivot_pfx = pivot_pfx
        self.ignore_pfxs = ignore_pfxs
        self.cur_rec = dict()
        self.cols = list()
        self.fld2flatjq = dict()
        self.cur_flatjq = None
        self.cur_idxs = defaultdict(set)  # Dict[attr, Set[idx]]

    def get_clean_rec(self) -> Dict:
        ''' Get the final record '''
        currec = self.cur_rec
        if self.add_idx_attrs:
            currec = currec.copy()
            # add idx attrs
            for k, v in self.cur_idxs.items():
                currec[f'link__{k}'] = ', '.join([str(x) for x in v])
        if self.jq_clean is not None:
            currec = self.jq_clean(currec)
        return currec

    def add_flatpath(
            self,
            value: Any,
            field: str,
            flat_jq: str = None,
            idxs: str = None,
    ) -> Dict:
        '''
        Add each flat path in order, incrementally constructing a record.
        A full record has been collected and a new starts when a field repeats.
        :param value: The value portion of a flat path
        :param field: The field portion of the flat path
        :param flat_jq: The remaining portion of the flat path
        :param idxs: The flat path indexes
        :return: The built record after a new record starts; otherwise None

        NOTE: The final record after the input is exhausted is in self.cur_rec.
        '''

        # Check for immediate exit conditions
        if value is None or field is None:
            return self.get_clean_rec() if len(self.cur_rec) > 0 else None
        if flat_jq is not None and self.ignore_pfxs is not None:
            if (
                    len(self.ignore_pfxs) == 0 and
                    self.pivot_pfx is not None and
                    not flat_jq.startswith(self.pivot_pfx)
            ):
                # Ignore all non-pivot paths
                return None
            # Ignore specific paths
            for pfx in self.ignore_pfxs:
                if flat_jq.startswith(pfx):
                    return None

        fullrec = None
        setval = False
        islist = False
        col = field
        if field.endswith('[]'):
            islist = True
            col = field[:-2]  # strip off the "[]"
        ffjq = flat_jq  # full flat_jq
        if self.pivot_pfx is not None and flat_jq is not None:
            if flat_jq.startswith(self.pivot_pfx):
                col = f"{flat_jq[1+len(self.pivot_pfx):]}.{col}"
                flat_jq = self.pivot_pfx
        if self.full_path_attrs:
            col = f"{flat_jq if flat_jq is not None else ''}.{col}"
        if self.add_idx_attrs or islist:
            idxs = [
                int(x.strip()) for x in idxs.split(',') if x.strip()
            ] if idxs else [0]


        if (
                flat_jq and self.cur_flatjq and
                not flat_jq.startswith(self.cur_flatjq) and (
                    self.pivot_pfx is None or (
                        flat_jq.startswith(self.pivot_pfx) and
                        flat_jq != self.pivot_pfx
                    )
                )
        ):
            # Find where flat_jq and self.cur_flatjq diverge
            fullrec = self.get_clean_rec()
            currec = dict()
            curidxs = dict()
            colidx = 0
            for idx, c in enumerate(self.cols):
                fld_flatjq = self.fld2flatjq[c]
                if flat_jq.startswith(fld_flatjq):
                    currec[c] = self.cur_rec[c]
                    if self.add_idx_attrs:
                        for k,v in self.cur_idxs.items():
                            if k in fld_flatjq:
                                curidxs[k] = v
                else:
                    colidx = idx
                    break
            self.cur_rec = currec
            self.cur_idxs = curidxs
            self.cols = self.cols[:colidx]

        if islist:  # terminal path node is a list
            field_idx = idxs[-1]
            idxs = idxs[:-1]
            if col not in self.cur_rec:
                # 1st time list value start
                self.cur_rec[col] = [value]
                self.cols.append(col)
                self.fld2flatjq[col] = flat_jq
                self.cur_flatjq = flat_jq
                setval = True
            elif field_idx != 0:
                # continuation of list
                self.cur_rec[col].append(value)
                setval = True
            #else field_idx == 0 ==> another start ... pop cur_rec

            if setval and self.add_idx_attrs and ffjq and len(idxs) > 0:
                self._add_idxs(ffjq, idxs)
                

        if not setval and col in self.cols:
            # Field repeat ==> pop cur_rec, reset to field pos in cols

            if fullrec is None:
                fullrec = self.get_clean_rec()
            currec = dict()
            self.cols = self.cols[:self.cols.index(col)]
            for c in self.cols:
                currec[c] = self.cur_rec[c]
            self.cur_rec = currec
            self.cur_idxs = defaultdict(set)

        if not setval:
            # Set value, track columns, add/track index attrs
            if islist:
                self.cur_rec[col] = [value]
            else:
                self.cur_rec[col] = value
            if (
                    self.add_idx_attrs and
                    ffjq and '[]' in ffjq and
                    len(idxs) > 0
            ):
                self._add_idxs(ffjq, idxs)
            self.cols.append(col)
            self.fld2flatjq[col] = flat_jq
            self.cur_flatjq = flat_jq

        return fullrec

    def _add_idxs(self, flat_jq, idxs):
        ''' add idxs to self.cur_idxs '''
        i = 0
        for part in flat_jq.split('.'):
            if part.endswith('[]'):
                self.cur_idxs[part[:-2]].add(idxs[i])
                i += 1


class RecordMetaInfo:
    '''
    Structure to hold meta-info for a set of records.
    '''
    def __init__(
            self,
            filepath: str,
            pivot_pfx: str = None,
            ignore_pfxs: Set[str] = None,
            jq_clean: Callable[[Dict], Dict] = None,
            add_idx_attrs: bool = False,
            rec_builder: FlatRecordBuilder = None,
            full_path_attrs: bool = False,
            file_obj = None,
    ):
        '''
        :param filepath: The path to the records file
        :param pivot_pfx: The prefix at which to "pivot" for grouping values
        :param ignore_pfxs: The path prefixes to ignore -- None to not ignore any,
            an empty set (or string) to ignore everything other than pivot_pfx,
            or the set of path prefixes to ignore. Note that the records built
            from selectively ignoring paths include paths from broader scopes,
            while those build ignoring all other paths do not.
        :param jq_clean: Function to clean the final record's attributes
        :param add_idx_attrs: True to add attributes to recs for array indeces
        :param rec_builder: (Optional) The record builder to use
        :param full_path_attrs: True to use the full path for record attributes
        :param file_obj: The output file object to use instead of opening
            filepath for write
        '''
        self.filepath = filepath
        self.pivot_pfx = pivot_pfx
        self.ignore_pfxs = ignore_pfxs
        self.jq_clean = jq_clean
        self.add_idx_attrs = add_idx_attrs
        self._recbuilder = rec_builder
        self.full_path_attrs = full_path_attrs
        self._file = file_obj
        self._opened = False

    @property
    def rec_builder(self) -> FlatRecordBuilder:
        if self._recbuilder is None:
            self._recbuilder = FlatRecordBuilder(
                full_path_attrs = self.full_path_attrs,
                pivot_pfx = self.pivot_pfx,
                ignore_pfxs = self.ignore_pfxs,
                jq_clean = self.jq_clean,
                add_idx_attrs = self.add_idx_attrs,
            )
        return self._recbuilder

    @property
    def file(self):
        ''' Get the (output) file handle to the filepath. '''
        if self._file == None:
            # open filepath for writing
            self._file = open(self.filepath, 'w', encoding='utf-8')
            self._opened = True
        return self._file

    def close(self):
        ''' Close the file (if opened here) '''
        if self._file is not None and self._opened:
            self._file.close()
            self._opened = False


class FlatRecordsBuilder:
    '''
    Generate flat records from squashed lines, where each type
    of record is streamed to its own file.
    '''
    def __init__(
            self,
            split_line_fn: Callable[[str], Tuple[str, str, str, str]],
            metainfos: List[RecordMetaInfo],
    ):
        self.split_line_fn = split_line_fn
        self.metainfos = metainfos

    def process_flatfile(self, flatfile):
        '''
        Process lines from the flatfile.
        :param flatfile: Either a filename (str whose path exists) or a fileobj
        '''
        opened = False
        fileobj = flatfile
        if isinstance(flatfile, str) and os.path.exists(flatfile):
            opened = True
            if flatfile.endswith('.gz'):
                fileobj = gzip.open(flatfile, 'rt', encoding='utf-8')
            else:
                fileobj = open(flatfile, 'r', encoding='utf-8')
        for line in fileobj:
            (value, field, flat_jq, idxs) = self.split_line_fn(line)
            for minfo in self.metainfos:
                rec = minfo.rec_builder.add_flatpath(
                    value, field, flat_jq=flat_jq, idxs=idxs
                )
                if rec is not None:
                    print(json.dumps(rec), file=minfo.file)
        if opened:
            fileobj.close()
        for minfo in self.metainfos:
            last_rec = minfo.rec_builder.get_clean_rec()
            if len(last_rec) > 0:
                print(json.dumps(last_rec), file=minfo.file)
            minfo.close()


def flat_record_generator(
        file_obj,
        split_line_fn: Callable[[str], Tuple[str, str, str, str]],
        builder: FlatRecordBuilder = None,
        pivot_pfx: str = None,
        ignore_pfxs: Set[str] = None,
        jq_clean: Callable[[Dict], Dict] = None,
        add_idx_attrs: bool = False,
):
    '''
    Generate flat records (dictionaries) from the file_obj's lines.
    :param file_obj: The file object supplying the flat lines
    :param split_line_fn: The function for splitting each line into
        (value, field, flat_jq, idxs)
    :param builder: The builder to use if not the default
    :param pivot_pfx: The prefix at which to "pivot" for grouping values
    :param ignore_pfxs: The path prefixes to ignore -- None to not ignore any,
        an empty set (or string) to ignore everything other than pivot_pfx,
        or the set of path prefixes to ignore. Note that the records built
        from selectively ignoring paths include paths from broader scopes,
        while those build ignoring all other paths do not.
    :param jq_clean: Function to clean the final record's attributes
    :param add_idx_attrs: True to add attributes to recs for array indeces
    '''
    if builder is None:
        builder = FlatRecordBuilder(
            pivot_pfx=pivot_pfx, ignore_pfxs=ignore_pfxs,
            jq_clean=jq_clean, add_idx_attrs=add_idx_attrs,
        )
    for line in file_obj:
        (value, field, flat_jq, idxs) = split_line_fn(line)
        rec = builder.add_flatpath(value, field, flat_jq=flat_jq, idxs=idxs)
        if rec is not None:
            yield rec
    # Generate the final record being built
    if len(builder.cur_rec) > 0:
        yield builder.get_clean_rec()


class LineFormatter(ABC):
    '''
    Class for formatting a record (dictionary) as a json string.
    '''
    def __init__(
            self,
            field_formatting_fn: Callable[[str], str] = None,
            value_formatting_fn: Callable[[str, Any], str] = None,
    ):
        '''
        :param field_formatting_fn: A fn(cjq_path) that returns a revised,
            formatted field name. Default is to return the cjq_path itself.
        :param value_formatting_fn: A fn(cjq_path, item_value) that returns
            a (potentially) revised formatted value for the path (key). Default
            is to return the value itself.
        '''
        self.field_fn = field_formatting_fn or (lambda f: f)
        self.value_fn = value_formatting_fn or (lambda _f, v: v)

    def __call__(self, record: Dict[str, Any]) -> str:
        return self.format_record(record)

    def _revise_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Apply the field and value formatting functions to the record.
        '''
        return {
            self.field_fn(k): self.value_fn(k, v)
            for k, v in record.items()
        }

    def flush(self, fileobj):
        '''
        Hook for flushing any data to a file after having processed all records.
        (default is a no-op.)
        '''
        pass

    @abstractmethod
    def format_record(self, record: Dict[str, Any]) -> str:
        '''
        Format the record as a string.
        '''
        raise NotImplementedError


class JsonLineFormatter(LineFormatter):
    '''
    Class for formatting a record (dictionary) as a json string.
    '''
    def __init__(
            self,
            field_formatting_fn: Callable[[str], str] = None,
            value_formatting_fn: Callable[[str, Any], str] = None,
    ):
        '''
        :param field_formatting_fn: A fn(cjq_path) that returns a revised,
            formatted field name. For example, formatting the field name
            for use as a DB column name would entail removing non-word chars
            from the string (which is the default.)
        :param value_formatting_fn: A fn(cjq_path, item_value) that returns
            a (potentially) revised formatted value for the path (key). Default
            is to return the value itself.
        '''
        super().__init__(
            field_formatting_fn=(
                field_formatting_fn or (lambda f: re.sub(r'\W+', '_', f))
            ),
            value_formatting_fn=value_formatting_fn,
        )

    def format_record(self, record: Dict[str, Any]) -> str:
        '''
        Format the record as a string.
        '''
        return json.dumps(self._revise_record(record))


class TsvLineFormatter(LineFormatter):
    '''
    Class for formatting a record (dictionary) as a TSV file line.
    '''
    def __init__(
            self,
            field_formatting_fn: Callable[[str], str] = None,
            value_formatting_fn: Callable[[str, Any], str] = None,
    ):
        '''
        :param field_formatting_fn: A fn(cjq_path) that returns a revised,
            formatted field name. Default is to return the cjq_path itself.
        :param value_formatting_fn: A fn(cjq_path, item_value) that returns
            a (potentially) revised formatted value for the path (key). Default
            is to return the value itself.
        '''
        super().__init__(
            field_formatting_fn=field_formatting_fn,
            value_formatting_fn=value_formatting_fn,
        )
        self._header = None  # List[str]

    @property
    def header(self) -> List[str]:
        '''
        Get the TSV header (field/attribute/column names).

        NOTE: The header values for the TSV can be retrieved from this object
              only *AFTER* all records have been processed because all fields
              will not necessarily be present until all record processed.
        '''
        return self._header

    def flush(self, fileobj):
        '''
        Hook for flushing any data to a file after having processed all records.
        In this case, write the header line as the last row of the TSV file.
        '''
        if (
                self._header is not None and
                len(self._header) > 0 and
                fileobj is not None
        ):
            print('\t'.join(self._header), file=fileobj)

    def format_record(self, record: Dict[str, Any]) -> str:
        '''
        Format the record as a string.
        '''
        revrec = self._revise_record(record)
        if self._header is None or len(revrec) > len(self._header):
            self._header = list(revrec.keys())
        return '\t'.join([str(v) for v in revrec.values()])
        

class RecordCache:
    '''
    Structure to keep track of record fields at distinct levels.
    '''
    def __init__(
            self,
            idx_level: int,
            idx_value: int,
            record_formatting_fn: Callable[[Dict[str, Any]], str],
            split_terminal_arrays: bool = False,
    ):
        '''
        Starting at index level -1 (pre-indices),
        build records as index level N,
        holding non-array items at level N
        and, for each array element, the list of RecordCache instances
        for records at level N+1

        :param idx_level: The idx level (index into the idxs array) for this
            cache
        :param idx_value: The index of the element to capture/cache
        :param record_formatting_fn: A fn(rec_dict) that returns the record
            as a line of text for writing. Examples: write the record as a json
            line or as a tab-delimited file line.
        :param split_terminal_arrays: When True, items in terminal arrays yield
            new rows; otherwise, they are glommed into a list value

        NOTE: Currently hardwired to "pop" top-level elements, or one deeper
              than the root.
        '''
        self.idx_level = idx_level
        self.idx_value = idx_value
        self.format_fn = record_formatting_fn
        self.split_terminal_arrays = split_terminal_arrays
        self.rcs = dict()  # Dict[elt, List[RecordCache]] for array elts at idx_level+1
        self._record = dict()  # non-array items at idx_level

    def add_item(
            self,
            cjq_path: str,
            idxs: List[Tuple[str, str]],
            item: Any,
            fileobj,
    ) -> bool:
        '''
        Add an item.
        :param cjq_path: The "common" jq_path (no indexes)
        :param idxs: The list of extracted (path_elt, index) tuples.
        :param item: The path's value
        :param fileobj: The file object to write flushed records to
        :return: True if flushed
        '''
        flushed = False
        completed_old = False
        islist = False
        num_idxs = len(idxs)
        if not self.split_terminal_arrays and cjq_path.endswith('[]'):
            islist = True
            num_idxs -= 1

        if num_idxs > self.idx_level + 1:
            # Add deeper item
            idx_elt, idx_value = idxs[self.idx_level+1]
            rc = None
            make_new_rc = False
            if idx_elt in self.rcs:
                rc = self.rcs[idx_elt][-1]
                if rc.idx_value < idx_value:
                    # adding a repeat value to or starting anew in *any* deep
                    # item triggers popping *all* deep items
                    make_new_rc = True
                    completed_old = True

            if (completed_old and self.idx_level < 0):
                # build and pop records (NOTE: idx_level < 0 pops top-level recs)
                for rec in self._records_generator():
                    print(self.format_fn(rec), file=fileobj)
                self.rcs.clear()
                flushed = True

            if idx_elt not in self.rcs:
                self.rcs[idx_elt] = list()
                make_new_rc = True

            if make_new_rc:
                rc = RecordCache(
                    self.idx_level+1,
                    idx_value,
                    self.format_fn,
                    split_terminal_arrays=self.split_terminal_arrays
                )
                self.rcs[idx_elt].append(rc)
            rc.add_item(cjq_path, idxs, item, fileobj)
        else:
            # Add local item
            if islist:
                if cjq_path not in self._record:
                    self._record[cjq_path] = [item]
                else:
                    self._record[cjq_path].append(item)
            else:
                self._record[cjq_path] = item
        return flushed

    def _records_generator(self):
        '''
        Generate the current records.
        '''
        if len(self.rcs) == 0:
            if len(self._record) > 0:
                yield self._record.copy()
        else:
            for rc_list in self.rcs.values():
                rec = self._record.copy()
                for rc in rc_list:
                    recc = rec.copy()
                    for srec in rc._records_generator():
                        recc.update(srec)
                        if len(recc) > 0:
                            yield recc

    def flush(self, fileobj):
        '''
        Flush any remaining records to the fileobj.
        :param fileobj: The file object to flush the final record(s) to.
        '''
        if len(self._record) > 0 or len(self.rcs) > 0:
            for rec in self._records_generator():
                print(self.format_fn(rec), file=fileobj)
            self._record.clear()
            self.rcs.clear()


class RecordBuilder:
    '''
    Structure to capture, build, and flush records.
    '''
    def __init__(
            self,
            name: str,
            recfile: str,
            record_formatting_fn: Callable[[Dict[str, Any]], str],
            split_terminal_arrays: bool = False,
    ):
        '''
        :param name: A name for the record set
        :param recfile: The output path (str) or obj to send records
        :param record_formatting_fn: A fn(rec_dict) that returns the record
            as a line of text for writing. Examples: write the record as a json
            line or as a tab-delimited file line.
        :param split_terminal_arrays: When True, items in terminal arrays yield
            new rows; otherwise, they are glommed into a list
        '''
        self.name = name
        self.recfile = recfile
        self.format_fn = record_formatting_fn
        self.rc = RecordCache(
            -1,
            0,
            self.format_fn,
            split_terminal_arrays=split_terminal_arrays
        )
        self._file = None
        self._opened = False

    @property
    def file(self):
        ''' Get the (output) file handle to the recfile. '''
        if self._file == None:
            if isinstance(self.recfile, str):
                # open recfile for writing
                self._file = open(self.recfile, 'w', encoding='utf-8')
                self._opened = True
            else:
                self._file = self.recfile
        return self._file

    def close(self):
        ''' Close the file (if opened here) '''
        if self._file is not None and self._opened:
            self._file.close()
            self._opened = False

    def add_item(
            self,
            cjq_path: str,
            idxs: List[Tuple[str, str]],
            item: Any
    ) -> bool:
        '''
        Add an item.
        :param cjq_path: The "common" jq_path (no indexes).
        :param idxs: The list of extracted (path_elt, index) tuples.
        :param item: The path's value
        :return: True if flushed
        '''
        return self.rc.add_item(cjq_path, idxs, item, self.file)

    def cleanup(self):
        self.rc.flush(self.file)
        if 'flush' in dir(self.format_fn):
            # Hack to write the header row to the end of a TSV file
            self.format_fn.flush(self.file)
        self.close()


class RecordsBuilder:
    '''
    Class for building single records from 2nd-tier (just under the root)
    blocks while streaming a json file.
    '''
    def __init__(
            self,
            record_builder: RecordBuilder,
            timeout: int = TIMEOUT,
    ):
        '''
        :param record_builder: The RecordBuilder to use
        :param timeout: The requests timeout (in seconds)
        '''
        self.builder = record_builder
        self.timeout = timeout

    def build_records(self, jdata: str):
        '''
        :param jdata: The source json_data from which to build records.
        '''
        squash_data(
            self._builder_fn, jdata, timeout=self.timeout,
        )
        # Cleanup
        self.builder.cleanup()

    def _builder_fn(self, jq_path: str, item: Any):
        '''
        Build the records from each streamed jq_path and its item.
        :param jq_path: The jq_path (with indexes)
        :param item: The item value for the path
        '''
        # format for RecordInfo.add params and call each
        cjq_path_elts = list()
        idxs = list()
        for path_elt in jq_path.split('.'):
            if path_elt.endswith(']'):
                left_bracket = path_elt.rindex('[')
                elt = path_elt[:left_bracket]
                idx = int(path_elt[left_bracket+1:len(path_elt)-1])
                idxs.append((f'{elt}', idx))
                cjq_path_elts.append(f'{elt}[]')
            else:
                cjq_path_elts.append(path_elt)

        cjq_path = '.'.join(cjq_path_elts)
        self.builder.add_item(cjq_path, idxs, item)

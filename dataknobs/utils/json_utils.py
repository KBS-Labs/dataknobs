import gzip
import io
import json_stream.requests
import os
import pandas as pd
import re
import requests
import dataknobs.structures.tree as dk_tree
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple, Union


ELT_IDX_RE = re.compile(r'^(.*)\[(.*)\]$')
FLATTEN_IDX_RE = re.compile(r'\[(\d+)\]')
URL_RE = re.compile(r'^https?://.*$', flags=re.IGNORECASE)
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
        if json_data.endswith('.gz'):
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
    ):
        self.schema = schema if schema is not None else dict()
        self.values = ValuesIndex() if values is None else values
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
            self.schema[jq_path][value_type] += 1
        if value is not None:
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
        :param keep_unique_values: True to keep unique values for each path
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
        schema = JsonSchema()

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

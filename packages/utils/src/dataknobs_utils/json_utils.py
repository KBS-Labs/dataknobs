"""Utility functions for JSON processing, streaming, and manipulation.

Provides functions for working with JSON data including nested value access,
streaming from files and URLs, and tree-based JSON structure operations.
"""

import gzip
import io
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Dict, List, Set, TextIO, Tuple, Union

import json_stream.requests
import pandas as pd
import requests

import dataknobs_utils.file_utils as dk_futils
from dataknobs_structures.tree import Tree, build_tree_from_string

ELT_IDX_RE = re.compile(r"^(.*)\[(.*)\]$")
FLATTEN_IDX_RE = re.compile(r"\[(\d+)\]")
URL_RE = re.compile(r"^\s*https?://.*$", flags=re.IGNORECASE)
TIMEOUT = 10  # 10 seconds


def get_value(
    json_obj: Dict[str, Any], key_path: str, default: Any | None = None
) -> Union[Any, List[Any]]:
    """Get a value from a JSON object using a key path in indexed dot notation.

    Args:
        json_obj: The JSON object to search in
        key_path: Dot-delimited string with optional [index] notation
        default: Value to return if path not found (default: None)

    Returns:
        Single value or list of values depending on key_path
    """
    # Split key path into individual segments
    pattern = r"([^.\[]+)(?:\[([^\]]+)\])?"
    segments = [
        (match.group(1), match.group(2))
        for match in re.finditer(pattern, key_path)
        if match.group(1)
    ]

    def traverse(obj: Any, seg_idx: int = 0) -> Union[Any, List[Any]]:
        # Base case: reached end of segments
        if seg_idx >= len(segments):
            return obj

        key, index = segments[seg_idx]

        # Handle case where object is None or doesn't have the key
        if not isinstance(obj, (dict, list)) or (isinstance(obj, dict) and key not in obj):
            return [] if any(s[1] in ("*", "?") for s in segments[seg_idx:]) else default

        # Get the next level value
        if isinstance(obj, dict):
            next_obj = obj.get(key)
        else:  # list
            return [] if any(s[1] in ("*", "?") for s in segments[seg_idx:]) else default

        # Handle different index cases
        if index is None:
            return traverse(next_obj, seg_idx + 1)

        elif index == "*":
            if not isinstance(next_obj, list):
                return []
            results = [traverse(item, seg_idx + 1) for item in next_obj]
            # Flatten single-item lists if no more wildcards ahead
            if not any(s[1] in ("*", "?") for s in segments[seg_idx + 1 :]):
                return [r for r in results if r != default]
            return results

        elif index == "?":
            if not isinstance(next_obj, list):
                return default
            for item in next_obj:
                result = traverse(item, seg_idx + 1)
                if result != default:
                    return result
            return default

        else:  # numeric index
            try:
                idx = int(index)
                if not isinstance(next_obj, list) or idx >= len(next_obj):
                    return default
                return traverse(next_obj[idx], seg_idx + 1)
            except ValueError:
                return default

    result = traverse(json_obj)
    # Return default value if result is None and no wildcards were used
    if result is None and not any(seg[1] in ("*", "?") for seg in segments):
        return default
    return result


def stream_json_data(
    json_data: str,
    visitor_fn: Callable[[Any, Tuple[Any, ...]], None],
    timeout: int = TIMEOUT,
) -> None:
    """Stream JSON data and call a visitor function for each value.

    Supports multiple input formats: file paths (including .gz), URLs, or JSON strings.
    Automatically detects and handles gzip-compressed files.

    Args:
        json_data: The JSON data source - can be a file path, URL (starting with
            'http'), or JSON string.
        visitor_fn: Function called for each JSON value with signature
            visitor_fn(item, path) where item is the value and path is a tuple
            of elements identifying the path to the item.
        timeout: Request timeout in seconds for URL sources. Defaults to 10.
    """
    if os.path.exists(json_data):
        if dk_futils.is_gzip_file(json_data):
            with gzip.open(json_data, "rt", encoding="utf-8") as f:
                json_stream.visit(f, visitor_fn)
        else:
            with open(json_data, encoding="utf-8") as f:
                json_stream.visit(f, visitor_fn)
    elif json_data.startswith("http"):
        with requests.get(json_data, stream=True, timeout=timeout) as response:
            json_stream.requests.visit(response, visitor_fn)
    elif isinstance(json_data, str):
        string_io = io.StringIO(json_data)
        json_stream.visit(string_io, visitor_fn)


def build_jq_path(path: Tuple[Any, ...], keep_list_idxs: bool = True) -> str:
    """Build a jq path string from a json_stream path tuple.

    Converts a path tuple like ('data', 0, 'name') to a jq-style path
    like '.data[0].name'.

    Args:
        path: Tuple of json_stream path components, with integers representing
            array indices and strings representing object keys.
        keep_list_idxs: If True, keeps exact array index values (e.g., [0], [1]).
            If False, emits generic '[]' for all array indices. Defaults to True.

    Returns:
        str: A jq-style path string.
    """
    jq_path = ""
    for elt in path:
        if isinstance(elt, int):
            jq_path += f"[{elt}]" if keep_list_idxs and elt >= 0 else "[]"
        else:
            jq_path += f".{elt}"
    return jq_path


def build_path_tuple(jq_path: str, any_list_idx: int = -1) -> Tuple[Any, ...]:
    """Build a json_stream tuple path from a jq path string.

    Reverses the operation of build_jq_path, converting a jq-style path like
    '.data[0].name' back to a tuple like ('data', 0, 'name').

    Args:
        jq_path: The jq-style path string to convert (e.g., '.data[0].name').
        any_list_idx: Index value to use for generic '[]' array notation.
            Defaults to -1.

    Returns:
        Tuple[Any, ...]: Path tuple in json_stream format alternating between
            keys (str) and array indices (int).
    """
    path = []
    for part in jq_path.split("."):
        if part == "":
            continue
        m = ELT_IDX_RE.match(part)
        if m:
            path.append(m.group(1))
            idx = m.group(2)
            idxval = any_list_idx if idx == "" else int(idx)
            path.append(idxval)
        else:
            path.append(part)
    return tuple(path)


def stream_jq_paths(
    json_data: str,
    output_stream: TextIO,
    line_builder_fn: Callable[[str, Any], str] = (lambda jq_path, item: f"{jq_path}\t{item}"),
    keep_list_idxs: bool = True,
    timeout: int = TIMEOUT,
) -> None:
    """Stream JSON data and write formatted lines for each (jq_path, item) pair.

    Args:
        json_data: JSON data source (file path, URL, or JSON string).
        output_stream: Text stream to write formatted lines to.
        line_builder_fn: Function that takes (jq_path, item) and returns a
            formatted string. Defaults to tab-separated format.
        keep_list_idxs: If True, keeps exact array index values.
            If False, uses generic '[]'. Defaults to True.
        timeout: Request timeout in seconds for URL sources. Defaults to 10.
    """

    def visitor(item: Any, path: Tuple[Any, ...]) -> None:
        jq_path = build_jq_path(path, keep_list_idxs=keep_list_idxs)
        line = line_builder_fn(jq_path, item)
        print(line, file=output_stream)

    stream_json_data(json_data, visitor, timeout=timeout)


def squash_data(
    builder_fn: Callable[[str, Any], None],
    json_data: str,
    prune_at: List[Union[str, Tuple[str, int]]] | None = None,
    timeout: int = TIMEOUT,
) -> None:
    """Squash JSON data into single-level structure with jq-style keys.

    Compresses nested JSON paths into a flat structure where each path becomes
    a jq-style key. Optionally prunes specific branches from the output.

    Pruning specification formats:
        - (path_element_name, path_index): Paths where path[path_index] == path_element_name
        - path_element_name or (path_element_name, None): Any path containing this element name
        - path_index or (None, path_index): Any path at this depth

    Args:
        builder_fn: Callback function with signature fn(jq_path, item) called for
            each path/value pair to build results.
        json_data: JSON data source (file path, URL, or JSON string).
        prune_at: List of path specifications identifying branches to skip.
            Defaults to None (no pruning).
        timeout: Request timeout in seconds for URL sources. Defaults to 10.
    """
    raw_depths = set()
    raw_elts = set()
    elt_depths = {}
    depth_elts = defaultdict(set)

    def decode_item(item: Any) -> None:
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
                elif depth is not None:
                    depth_elts[depth].add(elt)
                    elt_depths[elt] = depth
                else:
                    raw_elts.add(elt)
            elif isinstance(item, (list, tuple)):
                for i in item:
                    decode_item(i)

    decode_item(prune_at)
    has_raw_depths = len(raw_depths) > 0
    has_raw_elts = len(raw_elts) > 0
    has_elts = len(elt_depths) > 0
    has_depth_elts = len(depth_elts) > 0
    do_prune = has_raw_depths or has_raw_elts or has_elts or has_depth_elts

    def visitor(item: Any, path: Tuple[Any, ...]) -> None:
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
    prune_at: List[Union[str, Tuple[str, int]]] | None = None,
    timeout: int = TIMEOUT,
    result: Dict[Any, Any] | None = None,
) -> Dict[Any, Any]:
    """Collect squashed JSON data into a dictionary.

    Convenience function that squashes JSON into a flat dictionary where keys
    are jq-style paths and values are the leaf values from the JSON.

    Args:
        jdata: JSON data source (file path, URL, or JSON string).
        prune_at: List of path specifications identifying branches to skip.
            Defaults to None. See squash_data() for format details.
        timeout: Request timeout in seconds for URL sources. Defaults to 10.
        result: Optional dictionary to populate. If None, creates a new dict.

    Returns:
        Dict[Any, Any]: Dictionary mapping jq-style paths to their values.
    """
    if result is None:
        result = {}

    def collector_fn(jq_path: str, item: Any) -> None:
        result[jq_path] = item

    squash_data(
        collector_fn,
        jdata,
        prune_at=prune_at,
        timeout=timeout,
    )
    return result


def indexing_format_fn(jq_path: str, item: Any) -> str:
    """Format a (jq_path, item) pair for indexing purposes.

    Creates a tab-separated format optimized for indexing with columns:
    value, field, flat_jq, idxs.

    Args:
        jq_path: The jq-style path to the item.
        item: The value at the path.

    Returns:
        str: Tab-separated string with format:
            - value: The item value
            - field: The last path element (field name)
            - flat_jq: The path with array indices flattened to '[]'
            - idxs: Comma-separated list of array indices from the path
    """
    idxs = ", ".join(FLATTEN_IDX_RE.findall(jq_path))
    flat_jq = FLATTEN_IDX_RE.sub("[]", jq_path)
    dotpos = flat_jq.rindex(".")
    field = flat_jq[dotpos + 1 :]
    flat_jq = flat_jq[:dotpos]
    return f"{item}\t{field}\t{flat_jq}\t{idxs}"


def indexing_format_splitter(
    fileline: str,
) -> Tuple[str | None, str | None, str | None, str | None]:
    """Parse a line formatted by indexing_format_fn.

    Reverses indexing_format_fn to extract the original components.

    Args:
        fileline: Tab-separated line from indexing_format_fn.

    Returns:
        Tuple[str | None, str | None, str | None, str | None]: A tuple containing:
            - value: The item value
            - field: The last path element (field name)
            - flat_jq: The path with array indices flattened to '[]'
            - idxs: Comma-separated list of array indices
            All values are None if line is empty.
    """
    line = fileline.strip()
    value = None
    field = None
    flat_jq = None
    idxs = None
    if line:
        parts = fileline.split("\t")
        value = parts[0]
        if len(parts) > 1:
            field = parts[1]
            if len(parts) > 2:
                flat_jq = parts[2]
                if len(parts) > 3:
                    idxs = parts[3]
    return (value, field, flat_jq, idxs)


def write_squashed(
    dest_file: Union[str, TextIO],
    jdata: str,
    prune_at: List[Union[str, Tuple[str, int]]] | None = None,
    timeout: int = TIMEOUT,
    format_fn: Callable[[str, Any], str] = lambda jq_path, item: f"{jq_path}\t{item}",
) -> None:
    """Write squashed JSON data to a file.

    Args:
        dest_file: Output file path (str) or open text stream (TextIO).
        jdata: JSON data source (file path, URL, or JSON string).
        prune_at: List of path specifications identifying branches to skip.
            Defaults to None. See squash_data() for format details.
        timeout: Request timeout in seconds for URL sources. Defaults to 10.
        format_fn: Function to format each (jq_path, item) pair as a line.
            Defaults to tab-separated format.
    """
    needs_close = False
    f: TextIO
    if isinstance(dest_file, str):
        f = open(dest_file, "w", encoding="utf-8")
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


def path_to_dict(
    path: Union[Tuple[Any, ...], str], value: Any, result: Dict[Any, Any] | None = None
) -> Dict[Any, Any]:
    """Convert a jq path and value into a nested dictionary structure.

    Takes a path (jq string or tuple) and reconstructs the nested dictionary
    structure it represents, setting the value at the leaf.

    Args:
        path: Path to the value - either a jq-style string (e.g., '.data[0].name')
            or a path tuple (e.g., ('data', 0, 'name')).
        value: The value to set at the path.
        result: Optional dictionary to populate. If None, creates a new dict.

    Returns:
        Dict[Any, Any]: Dictionary with nested structure representing the path.
    """
    if result is None:
        result = {}
    if isinstance(path, str):
        path = build_path_tuple(path, any_list_idx=-1)

    def do_it(
        cur_dict: Dict[Any, Any], path: Tuple[Any, ...], path_idx: int, pathlen: int, value: Any
    ) -> None:
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
                        cur_dict = {}
                        # simplifying assumption: idxs are in consecutive order fm 0
                        cur_list.append(cur_dict)
                    else:
                        cur_list.append(value)
                elif path_idx < pathlen:
                    cur_dict = cur_list[-1]
                else:
                    cur_list.append(value)
        elif list_idx is None:
            if path_idx < pathlen:
                elt_dict: Dict[Any, Any] = {}
                cur_dict[path_elt] = elt_dict
                cur_dict = elt_dict
            else:
                cur_dict[path_elt] = value
        else:
            cur_list = []
            cur_dict[path_elt] = cur_list
            if path_idx < pathlen:
                cur_dict = {}
                cur_list.append(cur_dict)
            else:
                cur_list.append(value)
        # recurse to keep moving along the path
        do_it(cur_dict, path, path_idx, pathlen, value)

    do_it(result, path, 0, len(path), value)

    return result


def explode(squashed: Dict[Any, Any]) -> Dict[Any, Any]:
    """Explode a squashed JSON dictionary back into nested structure.

    Reverses the squashing operation by converting a flat dictionary with
    jq-style paths as keys back into a nested JSON-like structure.

    Args:
        squashed: Dictionary with jq-style paths as keys and leaf values.

    Returns:
        Dict[Any, Any]: Nested dictionary reconstructed from the paths.
    """
    result: Dict[Any, Any] = {}
    for jq_path, value in squashed.items():
        path_to_dict(jq_path, value, result)
    return result


class ValuePath:
    """Structure to hold compressed information about paths to a unique value.

    Stores the tree of array indices leading to each occurrence of a value in
    JSON data, enabling efficient tracking of where values appear.

    Attributes:
        jq_path: The jq-style path template with generic array indices.
        value: The value found at this path.
    """

    def __init__(self, jq_path: str, value: Any):
        """Initialize a ValuePath for tracking occurrences of a value.

        Args:
            jq_path: The jq-style path template (key).
            value: The value found at this path.
        """
        self.jq_path = jq_path
        self.value = value
        self._indices = Tree(0).as_string()  # root data will hold total path count

    @property
    def indices(self) -> Tree:
        return build_tree_from_string(self._indices)

    def add(self, path: Tuple[Any, ...] | None) -> None:
        """Add a path occurrence to the index tree.

        Args:
            path: Path tuple with the same structure as jq_path, or None if
                no path information is available.
        """
        root = self.indices
        node = root
        node.data = int(node.data) + 1  # keep track of total
        if path is None:
            self._indices = root.as_string()
            return
        for elt in path:
            if isinstance(elt, int):
                found = False
                if node.has_children() and node.children is not None:
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
        """Get the number of jq_paths to the value."""
        return int(self.indices.data)

    def path_generator(self, result_type: str = "jq_path") -> Any:
        """Generate all concrete paths to this value.

        Args:
            result_type: Format for generated paths:
                - 'jq_path': Generate jq-style path strings (default)
                - 'path': Generate full path tuples with keys and indices
                - 'idx': Generate index-only tuples

        Yields:
            Paths in the requested format for each occurrence of the value.
        """
        path = build_path_tuple(self.jq_path, any_list_idx=-1)
        for node in self.indices.collect_terminal_nodes():
            node_path = node.get_path()
            node_idx = 0
            gen_path = []
            for elt in path:
                if isinstance(elt, int):
                    gen_path.append(int(node_path[node_idx].data))
                    node_idx += 1
                elif result_type != "idx":
                    gen_path.append(elt)
            if result_type == "jq_path":
                yield build_jq_path(tuple(gen_path), keep_list_idxs=True)
            else:
                yield gen_path


class ValuesIndex:
    """Index of unique values organized by their jq paths.

    Maintains a compressed index mapping each jq path to its unique values,
    with optional tracking of all occurrences via path trees.

    Attributes:
        path_values: Nested dict mapping jq_path -> value -> ValuePath.
    """

    def __init__(self) -> None:
        self.path_values: Dict[str, Dict[Any, ValuePath]] = (
            {}
        )  # Dict[jq_path, Dict[value, ValuePath]]

    def add(self, value: Any, jq_path: str, path: Tuple[Any, ...] | None = None) -> None:
        """Add a value occurrence to the index.

        Args:
            value: The value to index.
            jq_path: The jq-style path where the value was found.
            path: Optional full path tuple for tracking specific occurrences.
        """
        if jq_path in self.path_values:
            value_paths = self.path_values[jq_path]
        else:
            value_paths = {}
            self.path_values[jq_path] = value_paths

        if value == []:
            value = "_EMPTY_LIST_"
        elif value == {}:
            value = "_EMPTY_DICT_"

        if value in value_paths:
            value_path = value_paths[value]
        else:
            value_path = ValuePath(jq_path, value)
            value_paths[value] = value_path

        value_path.add(path)

    def has_jqpath(self, jq_path: str) -> bool:
        """Check if any values exist for the given jq path.

        Args:
            jq_path: The jq-style path to check.

        Returns:
            bool: True if values exist for this path, False otherwise.
        """
        return jq_path in self.path_values

    def get_values(self, jq_path: str) -> Set[Any]:
        """Get all unique values for a jq path.

        Args:
            jq_path: The jq-style path to query.

        Returns:
            Set[Any]: Set of unique values found at this path.
        """
        return set(self.path_values.get(jq_path, {}).keys())

    def num_values(self, jq_path: str) -> int:
        """Get count of unique values for a jq path.

        Args:
            jq_path: The jq-style path to query.

        Returns:
            int: Number of unique values at this path.
        """
        return len(self.path_values.get(jq_path, {}))


class JsonSchema:
    """Schema representation of JSON structure with type statistics.

    Maintains a mapping of jq paths to value types and their occurrence counts,
    with optional tracking of unique values at each path.

    Schema format:
        {
            <jq_path>: {
                <value_type>: <value_count>,
                ...
            },
            ...
        }

    Where value_type is the type of value (e.g., 'int', 'float', 'str') and
    value_count is the number of times that type occurs at the path.

    Attributes:
        schema: Mapping of jq_path to type counts.
        values: Optional ValuesIndex for tracking unique values.
    """

    def __init__(
        self,
        schema: Dict[str, Any] | None = None,
        values: ValuesIndex | None = None,
        values_limit: int = 0,  # max number of unique values to keep
    ):
        """Initialize a JsonSchema.

        Args:
            schema: Pre-existing schema to reconstruct from. Defaults to None.
            values: Pre-existing ValuesIndex to include. Defaults to None.
            values_limit: Maximum unique values to track per path. If 0, tracks
                all unique values. Defaults to 0.
        """
        self.schema = schema if schema is not None else {}
        self.values = ValuesIndex() if values is None else values
        self._values_limit = values_limit
        self._df: pd.DataFrame | None = None

    def add_path(
        self,
        jq_path: str,
        value_type: str,
        value: Any = None,
        path: Tuple[Any, ...] | None = None,
    ) -> None:
        """Add an occurrence of a value type at a jq path.

        Args:
            jq_path: The jq-style path identifying the location.
            value_type: The type of value at this path (e.g., 'int', 'str').
            value: Optional actual value for unique value tracking.
            path: Optional full path tuple for tracking value occurrences.
        """
        if jq_path not in self.schema:
            self.schema[jq_path] = {value_type: 1}
        elif value_type not in self.schema[jq_path]:
            self.schema[jq_path][value_type] = 1
        else:
            self.schema[jq_path][value_type] += 1
        if value is not None:
            if self._values_limit == 0 or self.values.num_values(jq_path) < self._values_limit:
                self.values.add(value, jq_path, path=path)
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Get schema information as a DataFrame with columns:

        jq_path    value_type    value_count
        """
        if self._df is None:
            self._df = self._build_df()
        return self._df

    def _build_df(self) -> pd.DataFrame:
        """Build a DataFrame representation of the schema.

        Returns:
            pd.DataFrame: Schema with columns:
                - jq_path: The jq-style path
                - value_type: Type of value at this path
                - value_count: Number of occurrences of this type
                - unique_count: (optional) Number of unique values if tracked
        """
        data = []
        has_value = False
        for k1, v1 in self.schema.items():  # jq_path -> [value_type -> value_count]
            for k2, v2 in v1.items():  # value_type -> value_count
                if self.values.has_jqpath(k1):
                    row = (k1, k2, v2, self.values.num_values(k1))
                    has_value = True
                else:
                    row = (k1, k2, v2)  # type: ignore[assignment]
                data.append(row)
        columns = ["jq_path", "value_type", "value_count"]
        if has_value:
            columns.append("unique_count")
        return pd.DataFrame(data, columns=columns)

    def extract_values(
        self,
        jq_path: str,
        json_data: str,
        unique: bool = True,
        timeout: int = TIMEOUT,
    ) -> Union[List[Any], Set[Any]]:
        """Extract all values at a jq path from JSON data.

        Args:
            jq_path: The jq-style path to extract values from.
            json_data: JSON data source (file path, URL, or JSON string).
            unique: If True, returns only unique values as a set.
                If False, returns all values as a list. Defaults to True.
            timeout: Request timeout in seconds for URL sources. Defaults to 10.

        Returns:
            Union[List[Any], Set[Any]]: Set of unique values if unique=True,
                otherwise list of all values.
        """
        sresult = set()
        lresult = []

        def visitor(item: Any, path: Tuple[Any, ...]) -> None:
            cur_jq_path = build_jq_path(path, keep_list_idxs=False)
            if jq_path == cur_jq_path:
                if unique:
                    sresult.add(item)
                else:
                    lresult.append(item)

        stream_json_data(json_data, visitor, timeout=timeout)
        return sresult if unique else lresult


class JsonSchemaBuilder:
    """Build a schema view of JSON data by streaming and analyzing structure.

    Processes JSON data to extract type information, value statistics, and
    optionally track unique values at each path.
    """

    def __init__(
        self,
        json_data: str,
        value_typer: Callable[[Any], str] | None = None,
        keep_unique_values: bool = False,
        invert_uniques: bool = False,
        keep_list_idxs: bool = False,
        timeout: int = TIMEOUT,
        empty_dict_type: str = "_EMPTY_DICT_",
        empty_list_type: str = "_EMPTY_LIST_",
        unk_value_type: str = "_UNKNOWN_",
        int_value_type: str = "int",
        float_value_type: str = "float",
        str_value_type: str = "str",
        url_value_type: str = "URL",
        on_add: Callable[[str], bool] | None = None,
    ):
        """Initialize a JsonSchemaBuilder to analyze JSON structure.

        Args:
            json_data: JSON data source (file path, URL, or JSON string).
            value_typer: Optional custom function that takes a value and returns
                its type string, overriding default type detection.
            keep_unique_values: If True or an integer, tracks unique values.
                If int, limits tracking to that many unique values per path.
            invert_uniques: If True, maintains reverse index from values to paths.
            keep_list_idxs: If True, preserves exact array indices in paths.
                If False, generalizes all indices to '[]'.
            timeout: Request timeout in seconds for URL sources. Defaults to 10.
            empty_dict_type: Type name for empty dictionaries.
            empty_list_type: Type name for empty lists.
            unk_value_type: Type name for unclassified values.
            int_value_type: Type name for integers.
            float_value_type: Type name for floats.
            str_value_type: Type name for strings.
            url_value_type: Type name for URL strings, or None to treat as regular strings.
            on_add: Optional filter function called before adding each path.
                Takes jq_path and returns True to include or False to skip.
        """
        self.json_data = json_data
        self.value_typer = value_typer
        self.keep_uniques = keep_unique_values
        self.values_limit = 0 if isinstance(keep_unique_values, bool) else int(keep_unique_values)
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
        self._on_add = on_add
        self._schema = JsonSchema(values_limit=self.values_limit)
        self._built_schema = False

    @property
    def schema(self) -> JsonSchema:
        """Get the schema for the json data"""
        if not self._built_schema:
            self._built_schema = self._build_schema()
        return self._schema

    @property
    def partial_schema(self) -> JsonSchema:
        """Get the current, possibly incomplete, schema"""
        return self._schema

    def _build_schema(self) -> bool:
        """Stream the json data and build the schema."""

        def visitor(item: Any, path: Tuple[Any, ...]) -> None:
            self._visit_item(self._schema, item, path)

        stream_json_data(self.json_data, visitor, timeout=self.timeout)
        return True

    def _visit_item(self, schema: JsonSchema, item: Any, path: Tuple[Any, ...]) -> None:
        """The visitor function for processing items and paths while streaming
        the json data.
        :param schema: The schema being built
        :param item: The next json item (value) encountered
        :param path: The path of the item as a tuple of the form:
            (<key1>, <list1-idx>, <key2>, ...)
        """
        jq_path = build_jq_path(path, keep_list_idxs=self.keep_list_idxs)
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
        do_add = True
        if self._on_add is not None:
            do_add = self._on_add(jq_path)
        if do_add:
            schema.add_path(
                jq_path,
                value_type,
                value=(item if self.keep_uniques else None),
                path=(path if self.invert_uniques else None),
            )


class Path:
    """Container for a jq path with its value and optional line number.

    Represents a single path/value pair from JSON data, with optional tracking
    of the original line number for streaming scenarios.

    Attributes:
        jq_path: Fully-qualified jq-style path (e.g., '.data[0].name').
        item: The value at this path.
        line_num: Optional line number from streaming (-1 if not tracked).
    """

    def __init__(self, jq_path: str, item: Any, line_num: int = -1):
        """Initialize a Path.

        Args:
            jq_path: Fully-qualified jq-style path with indices.
            item: The value at this path.
            line_num: Optional line number for ordering. Defaults to -1.
        """
        self.jq_path = jq_path
        self.item = item
        self.line_num = line_num
        self._path_elts: List[str] | None = None  # jq_path.split('.')
        self._len: int | None = None  # Number of path elements

    def __repr__(self) -> str:
        lnstr = f"{self.line_num}: " if self.line_num >= 0 else ""
        return f"{lnstr}{self.jq_path}: {self.item}"

    def __key(self) -> Union[Tuple[str, Any], int]:
        return (self.jq_path, self.item) if self.line_num < 0 else self.line_num

    def __lt__(self, other: "Path") -> bool:
        if self.line_num < 0 or other.line_num < 0:
            return self.jq_path < other.jq_path
        else:
            return self.line_num < other.line_num

    def __hash__(self) -> int:
        return hash(self.__key())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            return self.__key() == other.__key()
        return NotImplemented

    @property
    def path_elts(self) -> List[str]:
        """Get this path's (index-qualified) elements"""
        if self._path_elts is None:
            self._path_elts = self.jq_path.split(".")
        return self._path_elts

    @property
    def size(self) -> int:
        """Get the number of path_elements in this path."""
        return len(self.path_elts)


class GroupAcceptStrategy(ABC):
    """Abstract strategy for determining if a Path belongs in a PathGroup.

    Defines the logic for accepting paths as either main paths (defining the
    record structure) or distributed paths (shared across records).
    """

    @abstractmethod
    def accept_path(self, path: Path, group: "PathGroup", distribute: bool = False) -> str | None:
        """Determine if and how a path should be added to the group.

        Args:
            path: The path to evaluate for acceptance.
            group: The group that would receive the path.
            distribute: If True, path is proposed as a distributed (shared) path
                rather than a main (record-defining) path.

        Returns:
            str | None: One of:
                - 'main': Accept as main path (record structure)
                - 'distributed': Accept as distributed path (shared data)
                - None: Reject the path
        """
        raise NotImplementedError


class PathGroup:
    """Container for a group of related paths."""

    def __init__(self, accept_strategy: GroupAcceptStrategy, first_path: Path | None = None):
        self._all_paths: List[Path] | None = None
        self.main_paths: Set[Path] | None = None
        self.distributed_paths: Set[Path] | None = None
        self.accept_strategy = accept_strategy
        if first_path is not None:
            self.accept(first_path, distribute=False)

    @property
    def num_main_paths(self) -> int:
        """Get the number of main paths in this group"""
        return len(self.main_paths) if self.main_paths is not None else 0

    @property
    def num_distributed_paths(self) -> int:
        """Get the number of distributed paths in this group"""
        return len(self.distributed_paths) if self.distributed_paths is not None else 0

    @property
    def size(self) -> int:
        """Get the total number of paths in this group."""
        return self.num_main_paths + self.num_distributed_paths

    @property
    def paths(self) -> List[Path]:
        """Get all paths (both main and distributed)"""
        if self._all_paths is None:
            if self.main_paths is not None:
                all_paths_set = self.main_paths.copy()
                if self.distributed_paths is not None:
                    all_paths_set.update(self.distributed_paths)
                self._all_paths = sorted(all_paths_set)
            elif self.distributed_paths is not None:
                self._all_paths = sorted(self.distributed_paths)
            else:
                self._all_paths = []
        return self._all_paths

    def as_dict(self) -> Dict[str, Any]:
        """Reconstruct the object from the paths"""
        d: Dict[str, Any] = {}
        if self.paths is not None:
            for path in self.paths:
                path_to_dict(path.jq_path, path.item, result=d)
        return d

    def accept(self, path: Path, distribute: bool = False) -> bool:
        """Add the path if it belongs in this group.
        :param path: The path to (potentially) add.
        :param distribute: True to propose the path as a distributed path
        :return: True if the path was accepted and added.
        """
        added = False
        add_type = self.accept_strategy.accept_path(path, self, distribute=distribute)
        if add_type is not None:
            if add_type == "main":
                if self.main_paths is None:
                    self.main_paths = {path}
                else:
                    self.main_paths.add(path)
            elif self.distributed_paths is None:
                self.distributed_paths = {path}
            else:
                self.distributed_paths.add(path)
            added = True
            self._all_paths = None
        return added

    def incorporate_paths(self, group: "PathGroup") -> None:
        """Incorporate (distribute) the group's appliccable paths into this group."""
        for path in group.paths:
            self.accept(path, distribute=True)


class ArrayElementAcceptStrategy(GroupAcceptStrategy):
    """Strategy that groups paths by array element at a specific nesting level.

    Creates record boundaries at array elements, treating each element as a
    distinct record with paths that share the same array indices up to the
    specified level.

    Attributes:
        max_array_level: Array nesting level at which to create records.
        ref_path: Reference path for matching subsequent paths.
    """

    def __init__(self, max_array_level: int = -1):
        """Initialize the array element grouping strategy.

        Args:
            max_array_level: Array nesting depth for record boundaries:
                - -1: Ignore array levels (accept all)
                - 0: New record at first (top-level) array
                - 1: New record at second array level
                - etc.
        """
        self.max_array_level = max_array_level
        self.ref_path: Path | None = None

    def accept_path(self, path: Path, group: PathGroup, distribute: bool = False) -> str | None:
        if distribute or "[" not in path.jq_path:
            return "distributed"

        if group.num_main_paths == 0:
            # Accept first path with an array
            self.ref_path = path
            return "main"
        else:
            if self.ref_path is None:
                if group.main_paths is not None:
                    self.ref_path = next(iter(group.main_paths))
            # All elements up through max_array_level must fully match
            cur_array_level = -1
            if self.ref_path is not None:
                for idx in range(1, min(self.ref_path.size, path.size)):
                    ref_elt = self.ref_path.path_elts[idx]
                    path_elt = path.path_elts[idx]
                    if ref_elt != path_elt:
                        return None
                    elif ref_elt[-1] == "]":
                        cur_array_level += 1
                        if cur_array_level >= self.max_array_level:
                            break
        return "main"


class PathSorter:
    """Sort and group paths into records based on an acceptance strategy.

    Manages the incremental grouping of paths as they're streamed, creating
    record boundaries according to the acceptance strategy and enforcing
    size constraints.

    Attributes:
        accept_strategy: Strategy for determining path membership.
        group_size: Minimum group size (groups below this are dropped).
        max_groups: Maximum groups kept in memory.
        groups: List of active PathGroup instances.
    """

    def __init__(
        self,
        accept_strategy: GroupAcceptStrategy,
        group_size: int = 0,
        max_groups: int = 0,
    ):
        """Initialize a PathSorter.

        Args:
            accept_strategy: Strategy for determining how paths are grouped.
            group_size: Minimum required group size. Groups smaller than this
                are silently dropped when closed. 0 means no minimum. Defaults to 0.
            max_groups: Maximum groups kept in memory. Older groups are dropped
                when limit is reached. 0 means unlimited. Defaults to 0.
        """
        self.accept_strategy = accept_strategy
        self.group_size = group_size
        # NOTE: Must keep at least 3 groups if keeping any for propagating
        #      distributed paths.
        self.max_groups = max_groups if max_groups <= 0 else max(3, max_groups)
        self.groups: List[PathGroup] | None = None

    @property
    def num_groups(self) -> int:
        """Get the number of groups"""
        return len(self.groups) if self.groups is not None else 0

    def add_path(self, path: Path) -> PathGroup | None:
        """Add a path to the appropriate group, creating new groups as needed.

        Paths are added to the most recent group if accepted. When a path is
        rejected, the current group is closed and a new group is created.

        Args:
            path: The path to add to a group.

        Returns:
            PathGroup | None: The most recently closed group if one was closed,
                otherwise None.
        """
        result = None
        if self.groups is None or len(self.groups) == 0:
            self.groups = [
                PathGroup(
                    accept_strategy=self.accept_strategy,
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
                        accept_strategy=self.accept_strategy,
                        first_path=path,
                    )
                )

                # Enforce max_group limit by removing groups from the front
                if self.max_groups > 0 and len(self.groups) >= self.max_groups:
                    while len(self.groups) >= self.max_groups:
                        self.groups.pop(0)
        return result

    def close_group(self, idx: int = -1, check_size: bool = True) -> PathGroup | None:
        """Close a group, finalizing its contents.

        Closing a group involves:
        1. Incorporating distributed paths from the previous group
        2. Optionally checking size constraints and dropping undersized groups

        Args:
            idx: Index of group to close (-1 for last). Defaults to -1.
            check_size: If True, enforces group_size constraint. Defaults to True.

        Returns:
            PathGroup | None: The closed group, or None if dropped for size.
        """
        if self.groups is None or len(self.groups) == 0:
            return None

        if idx == -1:
            idx = len(self.groups) - 1
        latest_group = self.groups[idx]

        # Add distributable lines from the prior group
        if idx > 0:
            latest_group.incorporate_paths(self.groups[idx - 1])

        # Check size constraints
        if check_size and self.group_size > 0 and len(self.groups) > 0:
            if latest_group.size < self.group_size:
                # Last group not "filled" ... need to drop
                self.groups.pop()

        return latest_group

    def accept_path(self, path: Path) -> bool:
        """Try to add a path to any existing group.

        Args:
            path: The path to add.

        Returns:
            bool: True if the path was accepted by a group, False otherwise.
        """
        if self.groups is not None:
            for group in self.groups:
                if group.accept(path):
                    return True
        return False

    def all_groups_have_size(self, group_size: int) -> bool:
        """Check if all groups have a specific size.

        Args:
            group_size: The size to test for.

        Returns:
            bool: True if all groups have exactly this size, False otherwise.
        """
        if self.groups is not None:
            for group in self.groups:
                if group.size != group_size:
                    return False
            return True
        return False


class RecordPathBuilder:
    """Build and output records from JSON by grouping paths.

    Streams JSON data, groups paths into records using a PathSorter, and
    writes formatted output for each record.

    Attributes:
        jdata: JSON data source.
        output_stream: Where to write formatted output.
        builder_fn: Function to format record lines.
        timeout: Request timeout for URLs.
        rec_id: Current record ID counter.
        inum: Current item number counter.
        sorter: PathSorter for grouping paths.
    """

    def __init__(
        self,
        json_data: str,
        output_stream: TextIO,
        line_builder_fn: Callable[[int, int, str, Any], str],
        timeout: int = TIMEOUT,
    ):
        """Initialize a RecordPathBuilder.

        Args:
            json_data: JSON data source (file path, URL, or JSON string).
            output_stream: Text stream to write formatted lines to.
            line_builder_fn: Function with signature fn(rec_id, line_num, jq_path, item)
                that returns a formatted string.
            timeout: Request timeout in seconds for URL sources. Defaults to 10.
        """
        self.jdata = json_data
        self.output_stream = output_stream
        self.builder_fn = line_builder_fn
        self.timeout = timeout
        self.rec_id = 0
        self.inum = 0
        self.sorter: PathSorter | None = None

    def write_group(self, group: PathGroup) -> None:
        for path in group.paths:
            line = self.builder_fn(self.rec_id, path.line_num, path.jq_path, path.item)
            print(line, file=self.output_stream)

    def visitor(self, item: Any, path: Tuple[Any, ...]) -> None:
        jq_path = build_jq_path(path, keep_list_idxs=True)
        if self.sorter is not None:
            group = self.sorter.add_path(Path(jq_path, item, line_num=self.inum))
        else:
            group = None
        if group is not None:
            self.write_group(group)
            self.rec_id += 1
        self.inum += 1

    def stream_record_paths(self) -> None:
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
    output_stream: TextIO,
    line_builder_fn: Callable[[int, int, str, Any], str],
    timeout: int = TIMEOUT,
) -> None:
    """Stream JSON and write formatted records grouped by top-level structure.

    Identifies top-level JSON records (typically array elements) and writes
    formatted lines for each path within each record.

    Each output line represents a (rec_id, line_num, jq_path, item) tuple where:
        - rec_id: 0-based record identifier
        - line_num: 0-based original item number from stream
        - jq_path: Fully-qualified path within the record
        - item: Value at the path

    Args:
        json_data: JSON data source (file path, URL, or JSON string).
        output_stream: Text stream to write formatted lines to.
        line_builder_fn: Function with signature fn(rec_id, line_num, jq_path, item)
            that returns a formatted string.
        timeout: Request timeout in seconds for URL sources. Defaults to 10.
    """
    rpb = RecordPathBuilder(json_data, output_stream, line_builder_fn, timeout=timeout)
    rpb.stream_record_paths()


def get_records_df(json_data: str, timeout: int = TIMEOUT) -> pd.DataFrame:
    """Collect top-level JSON records into a pandas DataFrame.

    Convenience function that streams JSON and collects all records into memory
    as a DataFrame with columns: rec_id, line_num, jq_path, item.

    Warning:
        Not suitable for large JSON files as all data is loaded into memory.

    Args:
        json_data: JSON data source (file path, URL, or JSON string).
        timeout: Request timeout in seconds for URL sources. Defaults to 10.

    Returns:
        pd.DataFrame: Records with columns rec_id, line_num, jq_path, item.
    """
    s = io.StringIO()
    stream_record_paths(
        json_data,
        s,
        lambda rid, lid, jqp, val: f"{rid}\t{lid}\t{jqp}\t{val}",
        timeout=timeout,
    )
    s.seek(0)
    df = pd.read_csv(s, sep="\t", names=["rec_id", "line_num", "jq_path", "item"])
    return df

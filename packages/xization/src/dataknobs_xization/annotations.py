"""Text annotation data structures and interfaces.

Provides classes for managing text annotations with metadata, including
position tracking, annotation types, and derived annotation columns.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

import dataknobs_structures.document as dk_doc

# Key annotations column name constants for use across annotation interfaces
KEY_START_POS_COL = "start_pos"
KEY_END_POS_COL = "end_pos"
KEY_TEXT_COL = "text"
KEY_ANN_TYPE_COL = "ann_type"


class AnnotationsMetaData(dk_doc.MetaData):
    """Container for annotations meta-data, identifying key column names.

    NOTE: this object contains only information about annotation column names
    and not annotation table values.
    """

    def __init__(
        self,
        start_pos_col: str = KEY_START_POS_COL,
        end_pos_col: str = KEY_END_POS_COL,
        text_col: str = KEY_TEXT_COL,
        ann_type_col: str = KEY_ANN_TYPE_COL,
        sort_fields: List[str] = (KEY_START_POS_COL, KEY_END_POS_COL),
        sort_fields_ascending: List[bool] = (True, False),
        **kwargs: Any
    ):
        """Initialize with key (and more) column names and info.

        Key column types:
          * start_pos
          * end_pos
          * text
          * ann_type

        Note:
            Actual table columns can be named arbitrarily, BUT interactions
            through annotations classes and interfaces relating to the "key"
            columns must use the key column constants.

        Args:
            start_pos_col: Col name for the token starting position.
            end_pos_col: Col name for the token ending position.
            text_col: Col name for the token text.
            ann_type_col: Col name for the annotation types.
            sort_fields: The col types relevant for sorting annotation rows.
            sort_fields_ascending: To specify sort order of sort_fields.
            **kwargs: More column types mapped to column names.
        """
        super().__init__(
            {
                KEY_START_POS_COL: start_pos_col,
                KEY_END_POS_COL: end_pos_col,
                KEY_TEXT_COL: text_col,
                KEY_ANN_TYPE_COL: ann_type_col,
            },
            **kwargs,
        )
        self.sort_fields = list(sort_fields)
        self.ascending = sort_fields_ascending

    @property
    def start_pos_col(self) -> str:
        """Get the column name for the token starting postition"""
        return self.data[KEY_START_POS_COL]

    @property
    def end_pos_col(self) -> str:
        """Get the column name for the token ending position"""
        return self.data[KEY_END_POS_COL]

    @property
    def text_col(self) -> str:
        """Get the column name for the token text"""
        return self.data[KEY_TEXT_COL]

    @property
    def ann_type_col(self) -> str:
        """Get the column name for the token annotation type"""
        return self.data[KEY_ANN_TYPE_COL]

    def get_col(self, col_type: str, missing: str = None) -> str:
        """Get the name of the column having the given type (including key column
        types but not derived,) or get the missing value.

        Args:
            col_type: The type of column name to get.
            missing: The value to return for unknown column types.

        Returns:
            The column name or the missing value.
        """
        return self.get_value(col_type, missing)

    def sort_df(self, an_df: pd.DataFrame) -> pd.DataFrame:
        """Sort an annotations dataframe according to this metadata.

        Args:
            an_df: An annotations dataframe.

        Returns:
            The sorted annotations dataframe.
        """
        if self.sort_fields is not None:
            an_df = an_df.sort_values(self.sort_fields, ascending=self.ascending)
        return an_df


class DerivedAnnotationColumns(ABC):
    """Interface for injecting derived columns into AnnotationsMetaData."""

    @abstractmethod
    def get_col_value(
        self,
        metadata: AnnotationsMetaData,
        col_type: str,
        row: pd.Series,
        missing: str = None,
    ) -> str:
        """Get the value of the column in the given row derived from col_type.

        Args:
            metadata: The AnnotationsMetaData.
            col_type: The type of column value to derive.
            row: A row from which to get the value.
            missing: The value to return for unknown or missing column.

        Returns:
            The row value or the missing value.
        """
        raise NotImplementedError


class AnnotationsRowAccessor:
    """A class that accesses row data according to the metadata and derived cols."""

    def __init__(
        self, metadata: AnnotationsMetaData, derived_cols: DerivedAnnotationColumns = None
    ):
        """Initialize AnnotationsRowAccessor.

        Args:
            metadata: The metadata for annotation columns.
            derived_cols: A DerivedAnnotationColumns instance for injecting
                derived columns.
        """
        self.metadata = metadata
        self.derived_cols = derived_cols

    def get_col_value(
        self,
        col_type: str,
        row: pd.Series,
        missing: str = None,
    ) -> str:
        """Get the value of the column in the given row with the given type.

        This gets the value from the first existing column in the row from:
          * The metadata.get_col(col_type) column
          * col_type itself
          * The columns derived from col_type

        Args:
            col_type: The type of column value to get.
            row: A row from which to get the value.
            missing: The value to return for unknown or missing column.

        Returns:
            The row value or the missing value.
        """
        value = missing
        col = self.metadata.get_col(col_type, None)
        if col is None or col not in row.index:
            if col_type in self.metadata.data:
                value = row[col_type]
            elif self.derived_cols is not None:
                value = self.derived_cols.get_col_value(self.metadata, col_type, row, missing)
        else:
            value = row[col]
        return value


class Annotations:
    """DAO for collecting and managing a table of annotations, where each row
    carries annotation information for an input token.

    The data in this class is maintained either as a list of dicts, each dict
    representing a "row," or as a pandas DataFrame, depending on the latest
    access. Changes in either the lists or dataframe will be reflected in the
    alternate data structure.
    """

    def __init__(
        self,
        metadata: AnnotationsMetaData,
        df: pd.DataFrame = None,
    ):
        """Construct as empty or initialize with the dataframe form.

        Args:
            metadata: The annotations metadata.
            df: A dataframe with annotation records.
        """
        self.metadata = metadata
        self._annotations_list = None
        self._df = df

    @property
    def ann_row_dicts(self) -> List[Dict[str, Any]]:
        """Get the annotations as a list of dictionaries."""
        if self._annotations_list is None:
            self._annotations_list = self._build_list()
        return self._annotations_list

    @property
    def df(self) -> pd.DataFrame:
        """Get the annotations as a pandas dataframe."""
        if self._df is None:
            self._df = self._build_df()
        return self._df

    def clear(self) -> pd.DataFrame:
        """Clear/empty out all annotations, returning the annotations df"""
        rv = self.df
        self._df = None
        self._annotations_list = None
        return rv

    def is_empty(self) -> bool:
        return (self._df is None or len(self._df) == 0) and (
            self._annotations_list is None or len(self._annotations_list) == 0
        )

    def add_dict(self, annotation: Dict[str, Any]):
        """Add the annotation dict."""
        self.ann_row_dicts.append(annotation)

    def add_dicts(self, annotations: List[Dict[str, Any]]):
        """Add the annotation dicts."""
        self.ann_row_dicts.extend(annotations)

    def add_df(self, an_df: pd.DataFrame):
        """Add (concatentate) the annotation dataframe to the current annotations."""
        df = self.metadata.sort_df(pd.concat([self.df, an_df]))
        self.set_df(df)

    def _build_list(self) -> List[Dict[str, Any]]:
        """Build the annotations list from the dataframe."""
        alist = None
        if self._df is not None:
            alist = self._df.to_dict(orient="records")
            self._df = None
        return alist if alist is not None else []

    def _build_df(self) -> pd.DataFrame:
        """Get the annotations as a df."""
        df = None
        if self._annotations_list is not None:
            if len(self._annotations_list) > 0:
                df = self.metadata.sort_df(pd.DataFrame(self._annotations_list))
            self._annotations_list = None
        return df

    def set_df(self, df: pd.DataFrame):
        """Set (or reset) this annotation's dataframe.

        Args:
            df: The new annotations dataframe.
        """
        self._df = df
        self._annotations_list = None


class AnnotationsBuilder:
    """A class for building annotations."""

    def __init__(
        self,
        metadata: AnnotationsMetaData,
        data_defaults: Dict[str, Any],
    ):
        """Initialize AnnotationsBuilder.

        Args:
            metadata: The annotations metadata.
            data_defaults: Dict[ann_colname, default_value] with default
                values for annotation columns.
        """
        self.metadata = metadata if metadata is not None else AnnotationsMetaData()
        self.data_defaults = data_defaults

    def build_annotation_row(
        self, start_pos: int, end_pos: int, text: str, ann_type: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Build an annotation row with the mandatory key values and those from
        the remaining keyword arguments.

        For those kwargs whose names match metadata column names, override the
        data_defaults and add remaining data_default attributes.

        Args:
            start_pos: The token start position.
            end_pos: The token end position.
            text: The token text.
            ann_type: The annotation type.
            **kwargs: Additional keyword arguments for extra annotation fields.

        Returns:
            The result row dictionary.
        """
        return self.do_build_row(
            {
                self.metadata.start_pos_col: start_pos,
                self.metadata.end_pos_col: end_pos,
                self.metadata.text_col: text,
                self.metadata.ann_type_col: ann_type,
            },
            **kwargs,
        )

    def do_build_row(self, key_fields: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Do the row building with the key fields, followed by data defaults,
        followed by any extra kwargs.

        Args:
            key_fields: The dictionary of key fields.
            **kwargs: Any extra fields to add.

        Returns:
            The constructed row dictionary.
        """
        result = {}
        result.update(key_fields)
        if self.data_defaults is not None:
            # Add data_defaults
            result.update(self.data_defaults)
        if kwargs is not None:
            # Override with extra kwargs
            result.update(kwargs)
        return result


class RowData:
    """A wrapper for an annotation row (pd.Series) to facilitate e.g., grouping."""

    def __init__(
        self,
        metadata: AnnotationsMetaData,
        row: pd.Series,
    ):
        self.metadata = metadata
        self.row = row

    @property
    def loc(self):
        return self.row.name

    def __repr__(self) -> str:
        return f'[{self.start_pos}:{self.end_pos})"{self.text}"'

    @property
    def start_pos(self) -> int:
        return self.row[self.metadata.start_pos_col]

    @property
    def end_pos(self) -> int:
        return self.row[self.metadata.end_pos_col]

    @property
    def text(self) -> str:
        return self.row[self.metadata.text_col]

    def is_subset(self, other_row: "RowData") -> bool:
        """Determine whether this row's span is a subset of the other.

        Args:
            other_row: The other row.

        Returns:
            True if this row's span is a subset of the other row's span.
        """
        return self.start_pos >= other_row.start_pos and self.end_pos <= other_row.end_pos

    def is_subset_of_any(self, other_rows: List["RowData"]) -> bool:
        """Determine whether this row is a subset of any of the others
        according to text span coverage.

        Args:
            other_rows: The rows to test for this to be a subset of any.

        Returns:
            True if this row is a subset of any of the other rows.
        """
        result = False
        for other_row in other_rows:
            if self.is_subset(other_row):
                result = True
                break
        return result


class AnnotationsGroup:
    """Container for annotation rows that belong together as a (consistent) group.

    NOTE: An instance will only accept rows on condition of consistency per its
    acceptance function.
    """

    def __init__(
        self,
        row_accessor: AnnotationsRowAccessor,
        field_col_type: str,
        accept_fn: Callable[["AnnotationsGroup", RowData], bool],
        group_type: str = None,
        group_num: int = None,
        valid: bool = True,
        autolock: bool = False,
    ):
        """Initialize AnnotationsGroup.

        Args:
            row_accessor: The annotations row_accessor.
            field_col_type: The col_type for the group field_type for retrieval
                using the annotations row accessor.
            accept_fn: A fn(g, row_data) that returns True to accept the row
                data into this group g, or False to reject the row. If None, then
                all rows are always accepted.
            group_type: An optional (override) type for identifying this group.
            group_num: An optional number for identifying this group.
            valid: True if the group is valid, or False if not.
            autolock: True to automatically lock this group when (1) at
                least one row has been added and (2) a row is rejected.
        """
        self.rows = []  # List[RowData]
        self.row_accessor = row_accessor
        self.field_col_type = field_col_type
        self.accept_fn = accept_fn
        self._group_type = group_type
        self._group_num = group_num
        self._valid = valid
        self._autolock = autolock
        self._locked = False
        self._locs = None  # track loc's for recognizing dupes
        self._key = None  # a hash key using the _locs
        self._df = None
        self._ann_type = None

    @property
    def is_locked(self) -> bool:
        """Get whether this group is locked from adding more rows."""
        return self._locked

    @is_locked.setter
    def is_locked(self, value: bool):
        """Set this group as locked (value=True) or unlocked (value=False) to
        allow or disallow more rows from being added regardless of the accept
        function.

        Note that while unlocked only rows that pass the accept function will
        be added.

        Args:
            value: True to lock or False to unlock this group.
        """
        self._locked = value

    @property
    def is_valid(self) -> bool:
        """Get whether this group is currently marked as valid."""
        return self._valid

    @is_valid.setter
    def is_valid(self, value: bool):
        """Mark this group as valid (value=True) or invalid (value=False).

        Args:
            value: True for valid or False for invalid.
        """
        self._valid = value

    @property
    def autolock(self) -> bool:
        """Get whether this group is currently set to autolock."""
        return self._autolock

    @autolock.setter
    def autolock(self, value: bool):
        """Set this group to autolock (True) or not (False).

        Args:
            value: True to autolock or False to not autolock.
        """
        self._autolock = value

    def __repr__(self):
        return json.dumps(self.to_dict())

    @property
    def size(self) -> int:
        """Get the number of rows in this group."""
        return len(self.rows)

    @property
    def group_type(self) -> str:
        """Get this group's type, which is either an "override" value that has
        been set, or the "ann_type" value of the first row added.
        """
        return self._group_type if self._group_type is not None else self.ann_type

    @group_type.setter
    def group_type(self, value: str):
        """Set this group's type"""
        self._group_type = value

    @property
    def group_num(self) -> int:
        """Get this group's number"""
        return self._group_num

    @group_num.setter
    def group_num(self, value: int):
        """Set this group's num"""
        self._group_num = value

    @property
    def df(self) -> pd.DataFrame:
        """Get this group as a dataframe"""
        if self._df is None:
            self._df = pd.DataFrame([r.row for r in self.rows])
        return self._df

    @property
    def ann_type(self) -> str:
        """Get this record's annotation type"""
        return self._ann_type

    @property
    def text(self) -> str:
        return " ".join([row.text for row in self.rows])

    @property
    def locs(self) -> List[int]:
        if self._locs is None:
            self._locs = [r.loc for r in self.rows]
        return self._locs

    @property
    def key(self) -> str:
        """A hash key for this group."""
        if self._key is None:
            self._key = "_".join([str(x) for x in sorted(self.locs)])
        return self._key

    def copy(self) -> "AnnotationsGroup":
        result = AnnotationsGroup(
            self.row_accessor,
            self.field_col_type,
            self.accept_fn,
            group_type=self.group_type,
            group_num=self.group_num,
            valid=self.is_valid,
            autolock=self.autolock,
        )
        result.rows = self.rows.copy()
        result._locked = self._locked  # pylint: disable=protected-access
        result._ann_type = self._ann_type  # pylint: disable=protected-access

    def add(self, rowdata: RowData) -> bool:
        """Add the row if the group is not locked and the row belongs in this
        group, or return False.

        If autolock is True and a row fails to be added (after the first
        row has been added,) "lock" the group and refuse to accept any more
        rows.

        Args:
            rowdata: The row to add.

        Returns:
            True if the row belongs and was added; otherwise, False.
        """
        result = False
        if self._locked:
            return result

        if self.accept_fn is None or self.accept_fn(self, rowdata):
            self.rows.append(rowdata)
            self._df = None
            self._locs = None
            self._key = None
            if self._ann_type is None:
                self._ann_type = self.row_accessor.get_col_value(
                    KEY_ANN_TYPE_COL,
                    rowdata.row,
                    missing=None,
                )
            result = True

        if not result and self.size > 0 and self.autolock:
            self._locked = True

        return result

    def to_dict(self) -> Dict[str, str]:
        """Get this group (record) as a dictionary of field type to text values."""
        return {self.row_accessor.get_col_value(self.field_col_type): row.text for row in self.rows}

    def is_subset(self, other: "AnnotationsGroup") -> bool:
        """Determine whether the this group's text is contained within the others.

        Args:
            other: The other group.

        Returns:
            True if this group's text is contained within the other group.
        """
        result = True
        for my_row in self.rows:
            if not my_row.is_subset_of_any(other.rows):
                result = False
                break
        return result

    def is_subset_of_any(self, groups: List["AnnotationsGroup"]) -> "AnnotationsGroup":
        """Determine whether this group is a subset of any of the given groups.

        Args:
            groups: List of annotation groups.

        Returns:
            The first AnnotationsGroup that this group is a subset of, or None.
        """
        result = None
        for other_group in groups:
            if self.is_subset(other_group):
                result = other_group
                break
        return result

    def remove_row(
        self,
        row_idx: int,
    ) -> RowData:
        """Remove the row from this group and optionally update the annotations
        accordingly.

        Args:
            row_idx: The positional index of the row to remove.

        Returns:
            The removed row data instance.
        """
        rowdata = self.rows.pop(row_idx)

        # Reset cached values
        self._df = None
        self._locs = None
        self._key = None

        return rowdata


class MergeStrategy(ABC):
    """A merge strategy to be injected based on entity types being merged."""

    @abstractmethod
    def merge(self, group: AnnotationsGroup) -> List[Dict[str, Any]]:
        """Process the annotations in the given annotations group, returning the
        group's merged annotation dictionaries.
        """
        raise NotImplementedError


class PositionalAnnotationsGroup(AnnotationsGroup):
    """Container for annotations that either overlap with each other or don't."""

    def __init__(self, overlap: bool, rectype: str = None, gnum: int = -1):
        """Initialize PositionalAnnotationsGroup.

        Args:
            overlap: If False, then only accept rows that don't overlap; else
                only accept rows that do overlap.
            rectype: The record type.
            gnum: The group number.
        """
        super().__init__(None, None, None, group_type=rectype, group_num=gnum)
        self.overlap = overlap
        self.start_pos = -1
        self.end_pos = -1

    def __repr__(self) -> str:
        return f'nrows={len(self.rows)}[{self.start_pos},{self.end_pos})"{self.entity_text}"'

    @property
    def entity_text(self) -> str:
        jstr = " | " if self.overlap else " "
        return jstr.join(r.entity_text for r in self.rows)

    def belongs(self, rowdata: RowData) -> bool:
        """Determine if the row belongs in this instance based on its overlap
        or not.

        Args:
            rowdata: The rowdata to test.

        Returns:
            True if the rowdata belongs in this instance.
        """
        result = True  # Anything belongs to an empty group
        if len(self.rows) > 0:
            start_overlaps = self._is_in_bounds(rowdata.start_pos)
            end_overlaps = self._is_in_bounds(rowdata.end_pos - 1)
            result = start_overlaps or end_overlaps
            if not self.overlap:
                result = not result
        if result:
            if self.start_pos < 0:
                self.start_pos = rowdata.start_pos
                self.end_pos = rowdata.end_pos
            else:
                self.start_pos = min(self.start_pos, rowdata.start_pos)
                self.end_pos = max(self.end_pos, rowdata.end_pos)
        return result

    def _is_in_bounds(self, char_pos):
        return char_pos >= self.start_pos and char_pos < self.end_pos

    def copy(self) -> "PositionalAnnotationsGroup":
        result = PositionalAnnotationsGroup(self.overlap)
        result.start_pos = self.start_pos
        result.end_pos = self.end_pos
        result.rows = self.rows.copy()
        return result

    # TODO: Add comparison and merge functions


class OverlapGroupIterator:
    """Given:
      * annotation rows (dataframe)
        * in order sorted by
          * start_pos (increasing for input order), and
          * end_pos (decreasing for longest spans first)
    Collect:
      * overlapping consecutive annotations
      * for processing
    """

    def __init__(self, an_df: pd.DataFrame):
        """Initialize OverlapGroupIterator.

        Args:
            an_df: An annotations.as_df DataFrame, sliced and sorted.
        """
        self.an_df = an_df
        self._cur_iter = None
        self._queued_row_data = None
        self.cur_group = None
        self.reset()

    def next_group(self) -> AnnotationsGroup:
        group = None
        if self.has_next:
            group = PositionalAnnotationsGroup(True)
            while self.has_next and group.belongs(self._queued_row_data):
                self._queue_next()
            self.cur_group = group
        return group

    def reset(self):
        self._cur_iter = self.an_df.iterrows()
        self._queue_next()
        self.cur_group = None

    @property
    def has_next(self) -> bool:
        return self._queued_row_data is not None

    def _queue_next(self):
        try:
            _loc, row = next(self._cur_iter)
            self._queued_row_data = RowData(None, row)  # TODO: add metadata
        except StopIteration:
            self._queued_row_data = None


def merge(
    annotations: Annotations,
    merge_strategy: MergeStrategy,
) -> Annotations:
    """Merge the overlapping groups according to the given strategy."""
    og_iter = OverlapGroupIterator(annotations.as_df)
    result = Annotations(annotations.metadata)
    while og_iter.has_next:
        og = og_iter.next_group()
        result.add_dicts(merge_strategy.merge(og))
    return result


class AnnotationsGroupList:
    """Container for a list of annotation groups."""

    def __init__(
        self,
        groups: List[AnnotationsGroup] = None,
        accept_fn: Callable[["AnnotationsGroupList", AnnotationsGroup], bool] = lambda lst, g: lst.size
        == 0
        or not g.is_subset_of_any(lst.groups),
    ):
        """Initialize AnnotationsGroupList.

        Args:
            groups: The initial groups for this list.
            accept_fn: A fn(lst, g) that returns True to accept the group, g,
                into this list, lst, or False to reject the group. If None, then all
                groups are always accepted. The default function will reject any
                group that is a subset of any existing group in the list.
        """
        self.groups = groups if groups is not None else []
        self.accept_fn = accept_fn
        self._coverage = None

    def __repr__(self) -> str:
        return str(self.groups)

    @property
    def size(self) -> int:
        """Get the number of groups in this list"""
        return len(self.groups)

    @property
    def coverage(self) -> int:
        """Get the total number of (token) rows covered by the groups"""
        if self._coverage is None:
            locs = set()
            for group in self.groups:
                locs.update(set(group.locs))
            self._coverage = len(locs)
        return self._coverage

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([r.row for g in self.groups for r in g.rows])

    def copy(self) -> "AnnotationsGroupList":
        result = AnnotationsGroupList(self.groups.copy(), accept_fn=self.accept_fn)
        result._coverage = self._coverage  # pylint: disable=protected-access
        return result

    def add(self, group: AnnotationsGroup) -> bool:
        """Add the group if it belongs in this group list or return False.

        Args:
            group: The group to add.

        Returns:
            True if the group belongs and was added; otherwise, False.
        """
        result = False
        if self.accept_fn is None or self.accept_fn(self, group):
            self.groups.append(group)
            self._coverage = None
            result = True
        return result

    def is_subset(self, other: "AnnotationsGroupList") -> bool:
        """Determine whether the this group's text spans are contained within all
        of the other's.

        Args:
            other: The other group list.

        Returns:
            True if this group list is a subset of the other group list.
        """
        result = True
        for my_group in self.groups:
            if not my_group.is_subset_of_any(other.groups):
                result = False
                break
        return result


class AnnotatedText(dk_doc.Text):
    """A Text object that manages its own annotations."""

    def __init__(
        self,
        text_str: str,
        metadata: dk_doc.TextMetaData = None,
        annots: Annotations = None,
        bookmarks: Dict[str, pd.DataFrame] = None,
        text_obj: dk_doc.Text = None,
        annots_metadata: AnnotationsMetaData = None,
    ):
        """Initialize AnnotatedText.

        Args:
            text_str: The text string.
            metadata: The text's metadata.
            annots: The annotations.
            bookmarks: The annotation bookmarks.
            text_obj: A text_obj to override text_str and metadata initialization.
            annots_metadata: Override for default annotations metadata
                (NOTE: ineffectual if an annots instance is provided.)
        """
        super().__init__(
            text_obj.text if text_obj is not None else text_str,
            text_obj.metadata if text_obj is not None else metadata,
        )
        self._annots = annots
        self._bookmarks = bookmarks
        self._annots_metadata = annots_metadata

    @property
    def annotations(self) -> Annotations:
        """Get the this object's annotations"""
        if self._annots is None:
            self._annots = Annotations(self._annots_metadata or AnnotationsMetaData())
        return self._annots

    @property
    def bookmarks(self) -> Dict[str, pd.DataFrame]:
        """Get this object's bookmarks"""
        if self._bookmarks is None:
            self._bookmarks = {}
        return self._bookmarks

    def get_text(
        self,
        annot2mask: Dict[str, str] = None,
        annot_df: pd.DataFrame = None,
        text: str = None,
    ) -> str:
        """Get the text object's string, masking if indicated.

        Args:
            annot2mask: Mapping from annotation column (e.g., _num or
                _recsnum) to the replacement character(s) in the input text
                for masking already managed input.
            annot_df: Override annotations dataframe.
            text: Override text.

        Returns:
            The (masked) text.
        """
        if annot2mask is None:
            return self.text
        # Apply the mask
        text_s = self.get_text_series(text=text)  # no padding
        if annot2mask is not None:
            annot_df = self.annotations.as_df
            text_s = self._apply_mask(text_s, annot2mask, annot_df)
        return "".join(text_s)

    def get_text_series(
        self,
        pad_len: int = 0,
        text: str = None,
    ) -> pd.Series:
        """Get the input text as a (padded) pandas series.

        Args:
            pad_len: The number of spaces to pad both front and back.
            text: Override text.

        Returns:
            The (padded) pandas series of input characters.
        """
        if text is None:
            text = self.text
        return pd.Series(list(" " * pad_len + text + " " * pad_len))

    def get_annot_mask(
        self,
        annot_col: str,
        pad_len: int = 0,
        annot_df: pd.DataFrame = None,
        text: str = None,
    ) -> pd.Series:
        """Get a True/False series for the input such that start to end positions
        for rows where the the annotation column is non-null and non-empty are
        True.

        Args:
            annot_col: The annotation column identifying chars to mask.
            pad_len: The number of characters to pad the mask with False
                values at both the front and back.
            annot_df: Override annotations dataframe.
            text: Override text.

        Returns:
            A pandas Series where annotated input character positions
            are True and non-annotated positions are False.
        """
        if annot_df is None:
            annot_df = self.annotations.as_df
        if text is None:
            text = self.text
        textlen = len(text)
        return self._get_annot_mask(annot_df, textlen, annot_col, pad_len=pad_len)

    @staticmethod
    def _get_annot_mask(
        annot_df: pd.DataFrame,
        textlen: int,
        annot_col: str,
        pad_len: int = 0,
    ) -> pd.Series:
        """Get a True/False series for the input such that start to end positions
        for rows where the the annotation column is non-null and non-empty are
        True.

        Args:
            annot_df: The annotations dataframe.
            textlen: The length of the input text.
            annot_col: The annotation column identifying chars to mask.
            pad_len: The number of characters to pad the mask with False
                values at both the front and back.

        Returns:
            A pandas Series where annotated input character positions
            are True and non-annotated positions are False.
        """
        mask = None
        df = annot_df
        if annot_col in df.columns:
            df = df[np.logical_and(df[annot_col].notna(), df[annot_col] != "")]
            mask = pd.Series([False] * textlen)
            for _, row in df.iterrows():
                mask.loc[row["start_pos"] + pad_len : row["end_pos"] - 1 + pad_len] = True
        return mask

    def _apply_mask(
        self,
        text_s: pd.Series,
        annot2mask: Dict[str, str],
        annot_df: pd.DataFrame,
    ) -> str:
        if len(text_s) > 0 and annot2mask is not None and annot_df is not None:
            cols = set(annot_df.columns).intersection(annot2mask.keys())
            if len(cols) > 0:
                for col in cols:
                    text_s = self._substitute(
                        text_s,
                        col,
                        annot2mask[col],
                        annot_df,
                    )
        return text_s

    def _substitute(
        self,
        text_s: pd.Series,
        col: str,
        repl_mask: str,
        annot_df: pd.DataFrame,
    ) -> str:
        """Substitute the "mask" char for "text" chars at "col"-annotated positions.

        Args:
            text_s: The text series to revise.
            col: The annotation col identifying positions to mask.
            repl_mask: The mask character to inject at annotated positions.
            annot_df: The annotations dataframe.

        Returns:
            The masked text.
        """
        annot_mask = self._get_annot_mask(annot_df, len(text_s), col)
        text_s = text_s.mask(annot_mask, repl_mask)
        return text_s

    def add_annotations(self, annotations: Annotations):
        """Add the annotations to this instance.

        Args:
            annotations: The annotations to add.
        """
        if annotations is not None and not annotations.is_empty():
            df = annotations.df
            if self._annots is None:
                self._annots = annotations
            elif self._annots.is_empty():
                if df is not None:
                    self._annots.set_df(df.copy())
            elif df is not None:
                self._annots.add_df(df)


class Annotator(ABC):
    """Class for annotating text"""

    def __init__(
        self,
        name: str,
    ):
        """Initialize Annotator.

        Args:
            name: The name of this annotator.
        """
        self.name = name

    @abstractmethod
    def annotate_input(
        self,
        text_obj: AnnotatedText,
        **kwargs: Any
    ) -> Annotations:
        """Annotate this instance's text, additively updating its annotations.

        Args:
            text_obj: The text object to annotate.
            **kwargs: Additional keyword arguments.

        Returns:
            The annotations added.
        """
        raise NotImplementedError


class BasicAnnotator(Annotator):
    """Class for extracting basic (possibly multi -level or -part) entities."""

    def annotate_input(
        self,
        text_obj: AnnotatedText,
        **kwargs: Any
    ) -> Annotations:
        """Annotate the text obj, additively updating the annotations.

        Args:
            text_obj: The text to annotate.
            **kwargs: Additional keyword arguments.

        Returns:
            The annotations added to the text.
        """
        # Get new annotation with just the syntax
        annots = self.annotate_text(text_obj.text)

        # Add syntactic annotations only as a bookmark
        text_obj.annotations.add_df(annots.as_df)

        return annots

    @abstractmethod
    def annotate_text(self, text_str: str) -> Annotations:
        """Build annotations for the text string.

        Args:
            text_str: The text string to annotate.

        Returns:
            Annotations for the text.
        """
        raise NotImplementedError


# TODO: remove this if unused -- stanza_annotator isa Authority -vs- stanza_annotator isa SyntacticParser
class SyntacticParser(BasicAnnotator):
    """Class for creating syntactic annotations for an input."""

    def annotate_input(
        self,
        text_obj: AnnotatedText,
        **kwargs: Any
    ) -> Annotations:
        """Annotate the text, additively updating the annotations.

        Args:
            text_obj: The text to annotate.
            **kwargs: Additional keyword arguments.

        Returns:
            The annotations added to the text.
        """
        # Get new annotation with just the syntax
        annots = self.annotate_text(text_obj.text)

        # Add syntactic annotations only as a bookmark
        text_obj.bookmarks[self.name] = annots.as_df

        return annots


class EntityAnnotator(BasicAnnotator):
    """Class for extracting single (possibly multi-level or -part) entities."""

    def __init__(
        self,
        name: str,
        mask_char: str = " ",
    ):
        """Initialize EntityAnnotator.

        Args:
            name: The name of this annotator.
            mask_char: The character to use to mask out previously annotated
                spans of this annotator's text.
        """
        super().__init__(name)
        self.mask_char = mask_char

    @property
    @abstractmethod
    def annotation_cols(self) -> Set[str]:
        """Report the (final group or record) annotation columns that are filled
        by this annotator when its entities are annotated.
        """
        raise NotImplementedError

    @abstractmethod
    def mark_records(self, annotations: Annotations, largest_only: bool = True):
        """Collect and mark annotation records.

        Args:
            annotations: The annotations.
            largest_only: True to only mark (keep) the largest records.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_records(
        self,
        annotations: Annotations,
    ):
        """Validate annotated records.

        Args:
            annotations: The annotations.
        """
        raise NotImplementedError

    @abstractmethod
    def compose_groups(self, annotations: Annotations) -> Annotations:
        """Compose annotation rows into groups.

        Args:
            annotations: The annotations.

        Returns:
            The composed annotations.
        """
        raise NotImplementedError

    def annotate_input(
        self,
        text_obj: AnnotatedText,
        annot_mask_cols: Set[str] = None,
        merge_strategies: Dict[str, MergeStrategy] = None,
        largest_only: bool = True,
        **kwargs: Any
    ) -> Annotations:
        """Annotate the text object (optionally) after masking out previously
        annotated spans, additively updating the annotations in the text
        object.

        Args:
            text_obj: The text object to annotate.
            annot_mask_cols: The (possible) previous annotations whose
                spans to ignore in the text.
            merge_strategies: A dictionary of each input annotation bookmark
                tag mapped to a merge strategy for merging this annotator's
                annotations with the bookmarked dataframe. This is useful, for
                example, when merging syntactic information to refine ambiguities.
            largest_only: True to only mark largest records.
            **kwargs: Additional keyword arguments.

        Returns:
            The annotations added to the text object.
        """
        # TODO: Use annot_mask_cols to mask annotations
        # annot2mask = (
        #     None
        #     if annot_mask_cols is None
        #     else {
        #         col: self.mask_char for col in annot_mask_cols
        #     }
        # )

        annots = self.annotate_text(text_obj.text)
        if annots is None:
            return annots

        if merge_strategies is not None:
            bookmarks = text_obj.bookmarks
            if bookmarks is not None and len(bookmarks) > 0:
                for tag, merge_strategy in merge_strategies.items():
                    if tag in bookmarks:
                        text_obj.bookmarks[f"{self.name}.pre-merge:{tag}"] = annots.df
                        annots.add_df(bookmarks[tag])
                        annots = merge(annots, merge_strategy)

        annots = self.compose_groups(annots)

        self.mark_records(annots, largest_only=largest_only)
        # NOTE: don't pass "text" here because it may be masked
        self.validate_records(annots)
        text_obj.annotations.add_df(annots.df)
        return annots

    @property
    @abstractmethod
    def highlight_fieldstyles(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Get highlight field styles for this annotator's annotations of the form:
        {
            <field_col>: {
                <field_value>: {
                    <css-attr>: <css-value>
                }
            }
        }
        For css-attr's like 'background-color', 'foreground-color', etc.
        """
        raise NotImplementedError


class HtmlHighlighter:
    """Helper class to add HTML markup for highlighting spans of text."""

    def __init__(
        self,
        field2style: Dict[str, Dict[str, str]],
        tooltip_class: str = "tooltip",
        tooltiptext_class: str = "tooltiptext",
    ):
        """Initialize HtmlHighlighter.

        Args:
            field2style: The annotation column to highlight with its
                associated style, for example:
                    {
                        'car_model_field': {
                            'year': {'background-color': 'lightyellow'},
                            'make': {'background-color': 'lightgreen'},
                            'model': {'background-color': 'cyan'},
                            'style': {'background-color': 'magenta'},
                        },
                    }
            tooltip_class: The css tooltip class.
            tooltiptext_class: The css tooltiptext class.
        """
        self.field2style = field2style
        self.tooltip_class = tooltip_class
        self.tooltiptext_class = tooltiptext_class

    def highlight(
        self,
        text_obj: AnnotatedText,
    ) -> str:
        """Return an html string with the given fields (annotation columns)
        highlighted with the associated styles.

        Args:
            text_obj: The annotated text to markup.

        Returns:
            HTML string with highlighted annotations.
        """
        result = ["<p>"]
        anns = text_obj.annotations
        an_df = anns.df
        for field, styles in self.field2style.items():
            # NOTE: the following line relies on an_df already being sorted
            df = an_df[an_df[field].isin(styles)]
            cur_pos = 0
            for _loc, row in df.iterrows():
                enttype = row[field]
                style = styles[enttype]
                style_str = " ".join([f"{key}: {value};" for key, value in style.items()])
                start_pos = row[anns.metadata.start_pos_col]
                if start_pos > cur_pos:
                    result.append(text_obj.text[cur_pos:start_pos])
                end_pos = row[anns.metadata.end_pos_col]
                result.append(f'<mark class="{self.tooltip_class}" style="{style_str}">')
                result.append(text_obj.text[start_pos:end_pos])
                result.append(f'<span class="{self.tooltiptext_class}">{enttype}</span>')
                result.append("</mark>")
                cur_pos = end_pos
        result.append("</p>")
        return "\n".join(result)


class AnnotatorKernel(ABC):
    """Class for encapsulating core annotation logic for multiple annotators"""

    @property
    @abstractmethod
    def annotators(self) -> List[EntityAnnotator]:
        """Get the entity annotators"""
        raise NotImplementedError

    @abstractmethod
    def annotate_input(self, text_obj: AnnotatedText) -> Annotations:
        """Execute all annotations on the text_obj"""
        raise NotImplementedError


class CompoundAnnotator(Annotator):
    """Class to apply a series of annotators through an AnnotatorKernel"""

    def __init__(
        self,
        kernel: AnnotatorKernel,
        name: str = "entity",
    ):
        """Initialize with the annotators and this extractor's name.

        Args:
            kernel: The annotations kernel to use.
            name: The name of this information extractor to be the
                annotations base column name for <name>_num and <name>_recsnum.
        """
        super().__init__(name=name)
        self.kernel = kernel

    def annotate_input(
        self,
        text_obj: AnnotatedText,
        reset: bool = True,
        **kwargs: Any
    ) -> Annotations:
        """Annotate the text.

        Args:
            text_obj: The AnnotatedText object to annotate.
            reset: When True, reset and rebuild any existing annotations.
            **kwargs: Additional keyword arguments.

        Returns:
            The annotations added to the text_obj.
        """
        if reset:
            text_obj.annotations.clear()
        annots = self.kernel.annotate_input(text_obj)
        return annots

    def get_html_highlighted_text(
        self,
        text_obj: AnnotatedText,
        annotator_names: List[str] = None,
    ) -> str:
        """Get html-hilighted text for the identified input's annotations
        from the given annotators (or all).

        Args:
            text_obj: The input text to highlight.
            annotator_names: The subset of annotators to highlight.

        Returns:
            HTML string with highlighted text.
        """
        if annotator_names is None:
            annotator_names = [ann.name for ann in self.kernel.annotators]
        hfs = {
            ann.name: ann.highlight_fieldstyles
            for ann in self.kernel.annotators
            if ann.name in annotator_names
        }
        hh = HtmlHighlighter(hfs)
        return hh.highlight(text_obj)

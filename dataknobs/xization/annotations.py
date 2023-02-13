import json
import pandas as pd
import dataknobs.structures.document as dk_doc
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


# Key annotations column name constants for use across annotation interfaces
KEY_START_POS_COL = 'start_pos'
KEY_END_POS_COL = 'end_pos'
KEY_TEXT_COL = 'text'
KEY_ANN_TYPE_COL = 'ann_type'


class AnnotationsMetaData(dk_doc.MetaData):
    '''
    Container for annotations meta-data, identifying key column names.

    NOTE: this object contains only information about annotation column names
    and not annotation table values.
    '''
    def __init__(
            self,
            start_pos_col: str = KEY_START_POS_COL,
            end_pos_col: str = KEY_END_POS_COL,
            text_col: str = KEY_TEXT_COL,
            ann_type_col: str = KEY_ANN_TYPE_COL,
            sort_fields: List[str] = (KEY_START_POS_COL, KEY_END_POS_COL),
            sort_fields_ascending: List[bool] = (True, False),
            **kwargs
    ):
        '''
        Initialize with key (and more) column names and info.

        Key column types:
          * start_pos
          * end_pos
          * text
          * ann_type

        NOTEs:
          * Actual table columns can be named arbitrarily
             * BUT: interactions through annotations classes and interfaces
               relating to the "key" columns must use the key column constants

        :param start_pos_col: Col name for the token starting position
        :param end_pos_col: Col name for the token ending position
        :param text_col: Col name for the token text
        :param ann_type_col: Col name for the annotation types
        :param sort_fields: The col types relevant for sorting annotation rows
        :param sort_fields_ascending: To specify sort order of sort_fields
        :param **kwargs: More column types mapped to column names
        '''
        super().__init__(
            {
                KEY_START_POS_COL: start_pos_col,
                KEY_END_POS_COL: end_pos_col,
                KEY_TEXT_COL: text_col,
                KEY_ANN_TYPE_COL: ann_type_col,
            },
            **kwargs
        )
        self.sort_fields = list(sort_fields)
        self.ascending = sort_fields_ascending

    @property
    def start_pos_col(self) -> str:
        ''' Get the column name for the token starting postition '''
        return self.data[KEY_START_POS_COL]

    @property
    def end_pos_col(self) -> str:
        ''' Get the column name for the token ending position '''
        return self.data[KEY_END_POS_COL]

    @property
    def text_col(self) -> str:
        ''' Get the column name for the token text '''
        return self.data[KEY_TEXT_COL]

    @property
    def ann_type_col(self) -> str:
        ''' Get the column name for the token annotation type '''
        return self.data[KEY_ANN_TYPE_COL]

    def get_col(self, col_type: str, missing: str = None) -> str:
        '''
        Get the name of the column having the given type (including key column
        types but not derived,) or get the missing value.

        :param col_type: The type of column name to get
        :param missing: The value to return for unknown column types
        :return: The column name or the missing value
        '''
        return self.get_value(col_type, missing)

    def sort_df(self, an_df: pd.DataFrame):
        '''
        Sort an annotations dataframe according to this metadata.
        :param an_df: An annotations dataframe
        :return: The sorted annotations dataframe.
        '''
        if self.sort_fields is not None:
            an_df = an_df.sort_values(
                self.sort_fields, ascending=self.ascending
            )
        return an_df


class DerivedAnnotationColumns(ABC):
    '''
    Interface for injecting derived columns into AnnotationsMetaData.
    '''
    @abstractmethod
    def get_col_value(
            self,
            metadata: AnnotationsMetaData,
            col_type: str,
            row: pd.Series,
            missing: str = None,
    ) -> str:
        '''
        Get the value of the column in the given row derived from col_type.

        :param metadata: The AnnotationsMetaData
        :param col_type: The type of column value to derive
        :param row: A row from which to get the value.
        :param missing: The value to return for unknown or missing column
        :return: The row value or the missing value
        '''
        raise NotImplementedError


class AnnotationsRowAccessor:
    '''
    A class that accesses row data according to the metadata and derived cols.
    '''
    def __init__(
            self,
            metadata: AnnotationsMetaData,
            derived_cols: DerivedAnnotationColumns = None
    ):
        '''
        :param metadata: The metadata for annotation columns
        :param derived_cols: A DerivedAnnotationColumns instance for injecting
            derived columns.
        '''
        self.metadata = metadata
        self.derived_cols = derived_cols

    def get_col_value(
            self,
            col_type: str,
            row: pd.Series,
            missing: str = None,
    ) -> str:
        '''
        Get the value of the column in the given row with the given type.

        This gets the value from the first existing column in the row from:
          * The metadata.get_col(col_type) column
          * col_type itself
          * The columns derived from col_type

        :param col_type: The type of column value to get
        :param row: A row from which to get the value.
        :param missing: The value to return for unknown or missing column
        :return: The row value or the missing value
        '''
        value = missing
        col = self.metadata.get_col(col_type, None)
        if col is None or col not in row.index:
            if col_type in self.metadata.data:
                value = row[col_type]
            else:
                if self.derived_cols is not None:
                    value = self.derived_cols.get_col_value(
                        self.metadata, col_type, row, missing
                    )
        else:
            value = row[col]
        return value


class Annotations:
    '''
    DAO for collecting and managing a table of annotations, where each row
    carries annotation information for an input token.

    The data in this class is maintained either as a list of dicts, each dict
    representing a "row," or as a pandas DataFrame, depending on the latest
    access. Changes in either the lists or dataframe will be reflected in the
    alternate data structure.
    '''
    def __init__(
            self,
            metadata: AnnotationsMetaData,
            df: pd.DataFrame = None,
    ):
        '''
        Construct as empty or initialize with the dataframe form.
        :param df: A dataframe with annotation records.
        '''
        self.metadata = metadata
        self._annotations_list = None
        self._df = df

    @property
    def ann_row_dicts(self) -> List[Dict[str, Any]]:
        '''
        Get the annotations as a list of dictionaries.
        '''
        if self._annotations_list is None:
            self._annotations_list = self._build_list()
        return self._annotations_list

    @property
    def df(self) -> pd.DataFrame:
        '''
        Get the annotations as a pandas dataframe.
        '''
        if self._df is None:
            self._df = self._build_df()
        return self._df

    def add_dict(self, annotation: Dict[str, Any]):
        '''
        Add the annotation dict.
        '''
        self.ann_row_dicts.append(annotation)

    def add_dicts(self, annotations: List[Dict[str, Any]]):
        '''
        Add the annotation dicts.
        '''
        self.ann_row_dicts.extend(annotations)

    def add_df(self, an_df: pd.DataFrame):
        '''
        Add (concatentate) the annotation dataframe to the current annotations.
        '''
        df = self.metadata.sort_df(
            pd.concat([self.df, an_df])
        )
        self.set_df(df)

    def _build_list(self) -> List[Dict[str, Any]]:
        '''
        Build the annotations list from the dataframe.
        '''
        alist = None
        if self._df is not None:
            alist = self._df.to_dict(orient='records')
            self._df = None
        return alist if alist is not None else list()

    def _build_df(self) -> pd.DataFrame:
        '''
        Get the annotations as a df.
        '''
        df = None
        if self._annotations_list is not None:
            if len(self._annotations_list) > 0:
                df = self.metadata.sort_df(
                    pd.DataFrame(self._annotations_list)
                )
            self._annotations_list = None
        return df

    def set_df(self, df: pd.DataFrame):
        '''
        Set (or reset) this annotation's dataframe.
        :param df: The new annotations dataframe.
        '''
        self._df = df
        self._annotations_list = None


class AnnotationsBuilder:
    '''
    A class for building annotations.
    '''

    def __init__(
            self,
            metadata: AnnotationsMetaData,
            data_defaults: Dict[str, Any],
    ):
        '''
        :param metadata: The annotations metadata
        :param data_defaults: Dict[ann_colname, default_value] with default
            values for annotation columns
        '''
        self.metadata = metadata if metadata is not None else AnnotationsMetaData()
        self.data_defaults = data_defaults

    def build_annotation_row(
            self,
            start_pos: int,
            end_pos: int,
            text: str,
            ann_type: str,
            **kwargs
    ) -> Dict[str, Any]:
        '''
        Build an annotation row with the mandatory key values and those from
        the remaining keyword arguments.

        For those kwargs whose names match metadata column names, override the
        data_defaults and add remaining data_default attributes.

        :param result_row_dict: The result row dictionary being built
        :param start_pos: The token start position
        :param end_pos: The token end position
        :param text: The token text
        :param ann_type: The annotation type
        :return: The result_row_dict
        '''
        return self.do_build_row({
            self.metadata.start_pos_col: start_pos,
            self.metadata.end_pos_col: end_pos,
            self.metadata.text_col: text,
            self.metadata.ann_type_col: ann_type,
        }, **kwargs)

    def do_build_row(
            self,
            key_fields: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        '''
        Do the row building with the key fields, followed by data defaults,
        followed by any extra kwargs.
        :param key_fields: The dictionary of key fields
        :param kwargs: Any extra fields to add
        '''
        result = dict()
        result.update(key_fields)
        if self.data_defaults is not None:
            # Add data_defaults
            result.update(self.data_defaults)
        if kwargs is not None:
            # Override with extra kwargs
            result.update(kwargs)
        return result


class RowData:
    '''
    A wrapper for an annotation row (pd.Series) to facilitate e.g., grouping.
    '''

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
        
    def is_subset(self, other_row: 'RowData') -> bool:
        '''
        Determine whether this row's span is a subset of the other.
        :param other_row: The other row
        '''
        return (
            self.start_pos >= other_row.start_pos and
            self.end_pos <= other_row.end_pos
        )

    def is_subset_of_any(self, other_rows: List['RowData']) -> bool:
        '''
        Determine whether this row is a subset of any of the others
        according to text span coverage.
        :param other_rows: The rows to test for this to be a subset of any
        '''
        result = False
        for other_row in other_rows:
            if self.is_subset(other_row):
                result = True
                break
        return result


class AnnotationsGroup:
    '''
    Container for annotation rows that belong together as a (consistent) group.

    NOTE: An instance will only accept rows on condition of consistency per its
    acceptance function.
    '''
    def __init__(
            self,
            row_accessor: AnnotationsRowAccessor,
            field_col_type: str,
            accept_fn: Callable[['AnnotationsGroup', RowData], bool],
            group_type: str = None,
            group_num: int = None,
            valid: bool = True,
            autolock: bool = False,
    ):
        '''
        :param row_accessor: The annotations row_accessor
        :param field_col_type: The col_type for the group field_type for retrieval
           using the annotations row accessor
        :param accept_fn: A fn(g, row_data) that returns True to accept the row
            data into this group g, or False to reject the row. If None, then
            all rows are always accepted.
        :param group_type: An optional (override) type for identifying this group.
        :param group_num: An optional number for identifying this group.
        :param valid: True if the group is valid, or False if not
        :param autolock: True to automatically lock this group when (1) at
            least one row has been added and (2) a row is rejected.
        '''
        self.rows = list()   # List[RowData]
        self.row_accessor = row_accessor
        self.field_col_type = field_col_type
        self.accept_fn = accept_fn
        self._group_type = group_type
        self._group_num = group_num
        self._valid = valid
        self._autolock = autolock
        self._locked = False
        self._locs = None    # track loc's for recognizing dupes
        self._key = None     # a hash key using the _locs
        self._df = None
        self._ann_type = None

    @property
    def is_locked(self) -> bool:
        '''
        Get whether this group is locked from adding more rows.
        '''
        return self._locked

    @is_locked.setter
    def is_locked(self, value: bool):
        '''
        Set this group as locked (value=True) or unlocked (value=False) to
        allow or disallow more rows from being added regardless of the accept
        function.

        Note that while unlocked only rows that pass the accept function will
        be added.

        :param value: True to lock or False to unlock this group.
        '''
        self._locked = value

    @property
    def is_valid(self) -> bool:
        '''
        Get whether this group is currently marked as valid.
        '''
        return self._valid

    @is_valid.setter
    def is_valid(self, value: bool):
        '''
        Mark this group as valid (value=True) or invalid (value=False).
        :param value: True for valid or False for invalid.
        '''
        self._valid = value

    @property
    def autolock(self) -> bool:
        '''
        Get whether this group is currently set to autolock.
        '''
        return self._autolock

    @autolock.setter
    def autolock(self, value: bool):
        '''
        Set this group to autolock (True) or not (False).
        :param value: True for False to autolock or not.
        '''
        self._autolock = value

    def __repr__(self):
        return json.dumps(self.to_dict())

    @property
    def size(self) -> int:
        '''
        Get the number of rows in this group.
        '''
        return len(self.rows)

    @property
    def group_type(self) -> str:
        '''
        Get this group's type, which is either an "override" value that has
        been set, or the "ann_type" value of the first row added.
        '''
        return (
            self._group_type
            if self._group_type is not None
            else self.ann_type
        )

    @group_type.setter
    def group_type(self, value: str):
        ''' Set this group's type '''
        self._group_type = value

    @property
    def group_num(self) -> int:
        ''' Get this group's number '''
        return self._group_num

    @group_num.setter
    def group_num(self, value: int):
        ''' Set this group's num '''
        self._group_num = value

    @property
    def df(self) -> pd.DataFrame:
        ''' Get this group as a dataframe '''
        if self._df is None:
            self._df = pd.DataFrame([r.row for r in self.rows])
        return self._df

    @property
    def ann_type(self) -> str:
        ''' Get this record's annotation type '''
        return self._ann_type

    @property
    def text(self) -> str:
        return ' '.join([
            row.text for row in self.rows
        ])

    @property
    def locs(self) -> List[int]:
        if self._locs is None:
            self._locs = [r.loc for r in self.rows]
        return self._locs

    @property
    def key(self) -> str:
        '''
        A hash key for this group.
        '''
        if self._key is None:
            self._key = '_'.join([str(x) for x in sorted(self.locs)])
        return self._key

    def copy(self) -> 'AnnotationsGroup':
        result = AnnotationsGroup(
            self.row_accessor, self.field_col_type, self.accept_fn,
            group_type=self.group_type,
            group_num=self.group_num,
            valid=self.is_valid,
            autolock=self.autolock
        )
        result.rows = self.rows.copy()
        result._locked = self._locked  # pylint: disable=protected-access
        result._ann_type = self._ann_type  # pylint: disable=protected-access

    def add(self, rowdata: RowData) -> bool:
        '''
        Add the row if the group is not locked and the row belongs in this
        group, or return False.

        If autolock is True and a row fails to be added (after the first
        row has been added,) "lock" the group and refuse to accept any more
        rows.
        
        :param rowdata: The row to add
        :return: True if the row belongs and was added; otherwise, False
        '''
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
                    KEY_ANN_TYPE_COL, rowdata.row, missing=None,
                )
            result = True

        if not result and self.size > 0 and self.autolock:
            self._locked = True

        return result

    def to_dict(self) -> Dict[str, str]:
        '''
        Get this group (record) as a dictionary of field type to text values.
        '''
        return {
            self.row_accessor.get_col_value(self.field_col_type): row.text
            for row in self.rows
        }

    def is_subset(self, other: 'AnnotationsGroup') -> bool:
        '''
        Determine whether the this group's text is contained within the others.
        :param other: The other group
        '''
        result = True
        for my_row in self.rows:
            if not my_row.is_subset_of_any(other.rows):
                result = False
                break
        return result

    def is_subset_of_any(
            self,
            groups: List['AnnotationsGroup']
    ) -> 'AnnotationsGroup':
        '''
        Determine whether this group is a subset of any of the given groups.
        :param groups: List of annotation groups
        :return: The first AnnotationsGroup that this group is a subset of, or
            None
        '''
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
        '''
        Remove the row from this group and optionally update the annotations
        accordingly.

        :param row_idx: The positional index of the row to remove
        :return: The removed row data instance
        '''
        rowdata = self.rows.pop(row_idx)

        # Reset cached values
        self._df = None
        self._locs = None
        self._key = None
        
        return rowdata
        

class AnnotationsGroupList:
    '''
    Container for a list of annotation groups.
    '''
    def __init__(
            self,
            groups: List[AnnotationsGroup] = None,
            accept_fn: Callable[
                ['AnnotationsGroupList', AnnotationsGroup], bool
            ] = lambda l, g: l.size == 0 or not g.is_subset_of_any(l.groups)
    ):
        '''
        :param groups: The initial groups for this list
        :param accept_fn: A fn(l, g) that returns True to accept the group, g,
            into this list, l, or False to reject the group. If None, then all
            groups are always accepted. The default function will reject any
            group that is a subset of any existing group in the list.
        '''
        self.groups = groups if groups is not None else list()
        self.accept_fn = accept_fn
        self._coverage = None

    def __repr__(self) -> str:
        return str(self.groups)

    @property
    def size(self) -> int:
        ''' Get the number of groups in this list '''
        return len(self.groups)

    @property
    def coverage(self) -> int:
        '''
        Get the total number of (token) rows covered by the groups
        '''
        if self._coverage is None:
            locs = set()
            for group in self.groups:
                locs.update(set(group.locs))
            self._coverage = len(locs)
        return self._coverage

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([r.row for g in self.groups for r in g.rows])

    def copy(self) -> 'AnnotationsGroupList':
        result = AnnotationsGroupList(
            self.groups.copy(), accept_fn=self.accept_fn
        )
        result._coverage = self._coverage  # pylint: disable=protected-access
        return result

    def add(self, group: AnnotationsGroup) -> bool:
        '''
        Add the group if it belongs in this group list or return False.
        :param group: The group to add
        :return: True if the group belongs and was added; otherwise, False
        '''
        result = False
        if self.accept_fn is None or self.accept_fn(self, group):
            self.groups.append(group)
            self._coverage = None
            result = True
        return result

    def is_subset(self, other: 'AnnotationsGroupList') -> bool:
        '''
        Determine whether the this group's text spans are contained within all
        of the other's.
        :param other: The other group list
        '''
        result = True
        for my_group in self.groups:
            if not my_group.is_subset_of_any(other.groups):
                result = False
                break
        return result

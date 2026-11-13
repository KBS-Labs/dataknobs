# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The deprecated half of :mod:`dataknobs_xization.annotations`.

Removed in the next major release. Nothing imports this module directly:
:mod:`~dataknobs_xization.annotations` reaches each name here through its
``__getattr__`` and warns as it does, so a subclass or an ``isinstance`` check
written against ``annotations.<name>`` is warned too, and removing the half is
deleting this file and that ``__getattr__``.

Several of these do not work, and are deprecated rather than fixed: ``merge``,
``BasicAnnotator``, ``SyntacticParser`` and ``EntityAnnotator`` read
``Annotations.as_df``, which does not exist, and ``HtmlHighlighter`` drops the
text after its last highlight, repeats the text once per highlighted column,
and does not escape HTML. Each warning says so.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Set

import pandas as pd

from dataknobs_xization.annotations import (
    KEY_ANN_TYPE_COL,
    AnnotatedText,
    Annotations,
    AnnotationsMetaData,
    AnnotationsRowAccessor,
    Annotator,
)


class RowData:
    """A wrapper for an annotation row (pd.Series) to facilitate e.g., grouping.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

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

    .. deprecated::
        Removed in the next major release. It has no replacement.
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
    """A merge strategy to be injected based on entity types being merged.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

    @abstractmethod
    def merge(self, group: AnnotationsGroup) -> List[Dict[str, Any]]:
        """Process the annotations in the given annotations group, returning the
        group's merged annotation dictionaries.
        """
        raise NotImplementedError


class PositionalAnnotationsGroup(AnnotationsGroup):
    """Container for annotations that either overlap with each other or don't.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

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

    .. deprecated::
        Removed in the next major release. It has no replacement.
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
    """Merge the overlapping groups according to the given strategy.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """
    og_iter = OverlapGroupIterator(annotations.as_df)
    result = Annotations(annotations.metadata)
    while og_iter.has_next:
        og = og_iter.next_group()
        result.add_dicts(merge_strategy.merge(og))
    return result


class AnnotationsGroupList:
    """Container for a list of annotation groups.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

    def __init__(
        self,
        groups: List[AnnotationsGroup] = None,
        accept_fn: Callable[["AnnotationsGroupList", AnnotationsGroup], bool] = lambda lst, g: (
            lst.size == 0 or not g.is_subset_of_any(lst.groups)
        ),
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


class BasicAnnotator(Annotator):
    """Class for extracting basic (possibly multi -level or -part) entities.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

    def annotate_input(self, text_obj: AnnotatedText, **kwargs: Any) -> Annotations:
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
    """Class for creating syntactic annotations for an input.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

    def annotate_input(self, text_obj: AnnotatedText, **kwargs: Any) -> Annotations:
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
    """Class for extracting single (possibly multi-level or -part) entities.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

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
        **kwargs: Any,
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
    """Helper class to add HTML markup for highlighting spans of text.

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

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
    """Class for encapsulating core annotation logic for multiple annotators

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

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
    """Class to apply a series of annotators through an AnnotatorKernel

    .. deprecated::
        Removed in the next major release. It has no replacement.
    """

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
        self, text_obj: AnnotatedText, reset: bool = True, **kwargs: Any
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

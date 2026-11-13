# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Text annotation data structures and interfaces.

Provides classes for managing text annotations with metadata, including
position tracking, annotation types, and derived annotation columns:
:class:`Annotations` holds the rows, :class:`AnnotationsMetaData` names their
columns, :class:`AnnotatedText` is a text that carries its own annotations, and
:class:`Annotator` is the extension point for producing them.

**Deprecated, and removed in the next major release:** ``RowData``,
``AnnotationsGroup``, ``MergeStrategy``, ``PositionalAnnotationsGroup``,
``OverlapGroupIterator``, ``merge``, ``AnnotationsGroupList``,
``BasicAnnotator``, ``SyntacticParser``, ``EntityAnnotator``,
``HtmlHighlighter``, ``AnnotatorKernel`` and ``CompoundAnnotator``. They are
still reached as ``annotations.<name>``, and each warns when it is.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dataknobs_xization import _deprecation

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
        **kwargs: Any,
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

        ``sort_fields`` holds col **types**, as its own parameter documents,
        and a frame's columns are col **names** -- so each field is
        translated through :meth:`get_col` on the way to pandas. Skipping the
        translation asked for ``start_pos`` on a frame whose column is
        whatever ``start_pos_col`` was configured to, and raised ``KeyError``
        from inside the ``df`` accessor.

        Invisible until something configured a name, because the two
        vocabularies agree for every default: the feature the parameter
        exists for was the only thing that could reach the defect.

        A field matching no col type is passed through unchanged, so a caller
        who put an actual column name there -- indistinguishable from a col
        type while the two agreed -- keeps sorting by it.

        Args:
            an_df: An annotations dataframe.

        Returns:
            The sorted annotations dataframe.
        """
        if self.sort_fields is not None:
            columns = [self.get_col(field, field) for field in self.sort_fields]
            an_df = an_df.sort_values(columns, ascending=self.ascending)
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

        The second clause asks whether the *row* has a column named
        ``col_type``, so a column carried under exactly the name asked for
        resolves to itself.

        Args:
            col_type: The type of column value to get.
            row: A row from which to get the value.
            missing: The value to return for unknown or missing column.

        Returns:
            The row value or the missing value.

        Raises:
            KeyError: If ``col_type`` is one of the metadata's column types and
                the row has neither its configured column nor a column named
                ``col_type``. That is a row written in another vocabulary --
                read through the wrong metadata -- and answering ``missing``
                would hide it.
        """
        col = self.metadata.get_col(col_type, None)
        if col is not None and col in row.index:
            return row[col]
        if col_type in row.index:
            return row[col_type]
        if col is not None:
            raise KeyError(
                f"the row has no {col!r} column for col type {col_type!r}, and no "
                f"{col_type!r} column either"
            )
        if self.derived_cols is not None:
            return self.derived_cols.get_col_value(self.metadata, col_type, row, missing)
        return missing


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
        if annot_df is None:
            annot_df = self.annotations.df
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
            are True and non-annotated positions are False; all False when
            there are no annotations, and None when there are annotations but
            none has ``annot_col``.
        """
        if annot_df is None:
            annot_df = self.annotations.df
        if text is None:
            text = self.text
        textlen = len(text)
        if annot_df is None:
            return pd.Series([False] * (textlen + 2 * pad_len))
        metadata = self.annotations.metadata
        return self._get_annot_mask(
            annot_df,
            textlen,
            annot_col,
            pad_len=pad_len,
            start_pos_col=metadata.start_pos_col,
            end_pos_col=metadata.end_pos_col,
        )

    @staticmethod
    def _get_annot_mask(
        annot_df: pd.DataFrame,
        textlen: int,
        annot_col: str,
        pad_len: int = 0,
        start_pos_col: str = KEY_START_POS_COL,
        end_pos_col: str = KEY_END_POS_COL,
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
            start_pos_col: The frame's start position column.
            end_pos_col: The frame's end position column.

        Returns:
            A pandas Series of ``textlen + 2 * pad_len`` values where annotated
            input character positions are True and all others are False, or
            None if the frame has no ``annot_col``.
        """
        mask = None
        df = annot_df
        if annot_col in df.columns:
            df = df[np.logical_and(df[annot_col].notna(), df[annot_col] != "")]
            mask = pd.Series([False] * (textlen + 2 * pad_len))
            for _, row in df.iterrows():
                start = row[start_pos_col] + pad_len
                mask.loc[start : row[end_pos_col] - 1 + pad_len] = True
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
        metadata = self.annotations.metadata
        annot_mask = self._get_annot_mask(
            annot_df,
            len(text_s),
            col,
            start_pos_col=metadata.start_pos_col,
            end_pos_col=metadata.end_pos_col,
        )
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
    def annotate_input(self, text_obj: AnnotatedText, **kwargs: Any) -> Annotations:
        """Annotate this instance's text, additively updating its annotations.

        Args:
            text_obj: The text object to annotate.
            **kwargs: Additional keyword arguments.

        Returns:
            The annotations added.
        """
        raise NotImplementedError


def __getattr__(name: str) -> Any:
    """Reach a deprecated name, warning that it is deprecated.

    The deprecated half of this module lives in ``_annotations_deprecated``,
    and every way of naming one of its members through this module -- an
    attribute, a ``from`` import, a base class, an ``isinstance`` check --
    passes through here.
    """
    if name in _deprecation.DEPRECATED_ANNOTATION_NAMES:
        _deprecation.warn_annotation_name(name)
        from dataknobs_xization import _annotations_deprecated

        return getattr(_annotations_deprecated, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

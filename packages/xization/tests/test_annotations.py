"""The half of :mod:`dataknobs_xization.annotations` that stays.

``Annotations`` and the classes around it are how a consumer holds annotation
rows -- one row per annotated span, in a pandas frame or as dicts -- and how
:class:`AnnotatedText` masks a text by them. This module had no tests of its
own, which is how ``AnnotatedText.get_text`` and ``get_annot_mask`` stayed
broken from the migration until now: both read ``Annotations.as_df``, a member
``Annotations`` has never had.

The other half of the module (the groups, ``merge``, the annotators and
``HtmlHighlighter``) is deprecated, and is not covered here; its deprecation is
asserted in ``test_stack_deprecation.py``.
"""

import pandas as pd
import pytest

from dataknobs_structures.document import TextMetaData
from dataknobs_xization.annotations import (
    KEY_ANN_TYPE_COL,
    KEY_END_POS_COL,
    KEY_START_POS_COL,
    KEY_TEXT_COL,
    AnnotatedText,
    Annotations,
    AnnotationsBuilder,
    AnnotationsMetaData,
    AnnotationsRowAccessor,
    Annotator,
    DerivedAnnotationColumns,
)

TEXT = "a beagle met a collie"


def _row(start: int, end: int, ann_type: str = "dog", **extra: object) -> dict:
    return {
        KEY_START_POS_COL: start,
        KEY_END_POS_COL: end,
        KEY_TEXT_COL: TEXT[start:end],
        KEY_ANN_TYPE_COL: ann_type,
        **extra,
    }


def _dogs() -> Annotations:
    """``beagle`` and ``collie``, added out of order to show the sort."""
    annots = Annotations(AnnotationsMetaData())
    annots.add_dicts([_row(15, 21, breed="collie"), _row(2, 8, breed="beagle")])
    return annots


# --- AnnotationsMetaData ------------------------------------------------------


def test_metadata_names_the_key_columns_and_maps_types_to_names() -> None:
    meta = AnnotationsMetaData(start_pos_col="s", end_pos_col="e", entity_id="eid")

    assert (meta.start_pos_col, meta.end_pos_col) == ("s", "e")
    assert (meta.text_col, meta.ann_type_col) == (KEY_TEXT_COL, KEY_ANN_TYPE_COL)
    assert meta.get_col(KEY_START_POS_COL) == "s"
    assert meta.get_col("entity_id") == "eid"
    assert meta.get_col("absent", "fallback") == "fallback"


def test_sort_df_sorts_by_the_configured_column_names() -> None:
    meta = AnnotationsMetaData(start_pos_col="s", end_pos_col="e")
    frame = pd.DataFrame({"s": [9, 0, 0], "e": [12, 3, 5]})

    ordered = meta.sort_df(frame)

    # Ascending start, and the longer span first at a shared start.
    assert list(zip(ordered["s"], ordered["e"], strict=True)) == [(0, 5), (0, 3), (9, 12)]


# --- Annotations and AnnotationsBuilder ---------------------------------------


def test_rows_added_as_dicts_come_back_as_a_sorted_frame() -> None:
    frame = _dogs().df

    assert list(frame["breed"]) == ["beagle", "collie"]
    assert list(frame[KEY_TEXT_COL]) == ["beagle", "collie"]


def test_the_frame_and_the_row_dicts_are_one_store() -> None:
    annots = _dogs()

    assert len(annots.df) == 2
    annots.add_dict(_row(9, 12, ann_type="verb"))

    assert [r[KEY_TEXT_COL] for r in annots.ann_row_dicts] == ["beagle", "collie", "met"]
    assert list(annots.df[KEY_TEXT_COL]) == ["beagle", "met", "collie"]


def test_add_df_concatenates_and_resorts() -> None:
    annots = _dogs()
    annots.add_df(pd.DataFrame([_row(9, 12, ann_type="verb")]))

    assert list(annots.df[KEY_TEXT_COL]) == ["beagle", "met", "collie"]


def test_set_df_replaces_and_clear_empties_returning_what_was_there() -> None:
    annots = _dogs()
    annots.set_df(pd.DataFrame([_row(9, 12, ann_type="verb")]))
    assert list(annots.df[KEY_TEXT_COL]) == ["met"]

    cleared = annots.clear()

    assert list(cleared[KEY_TEXT_COL]) == ["met"]
    assert annots.is_empty()
    assert annots.df is None


def test_an_empty_annotations_has_no_frame() -> None:
    annots = Annotations(AnnotationsMetaData())

    assert annots.is_empty()
    assert annots.df is None
    assert annots.ann_row_dicts == []


def test_identical_spans_with_different_ids_are_two_rows() -> None:
    """One mention with two readings is two rows, and neither is dropped.

    This is the shape an ambiguous form takes as rows: the same span, text and
    type, told apart only by the id each reading carries.
    """
    annots = Annotations(AnnotationsMetaData())
    annots.add_dicts([_row(2, 8, entity_id="dog:1"), _row(2, 8, entity_id="dog:2")])

    frame = annots.df

    assert len(frame) == 2
    assert sorted(frame["entity_id"]) == ["dog:1", "dog:2"]
    assert frame[[KEY_START_POS_COL, KEY_END_POS_COL]].drop_duplicates().shape[0] == 1


def test_builder_writes_key_columns_under_their_configured_names() -> None:
    builder = AnnotationsBuilder(
        AnnotationsMetaData(start_pos_col="s", text_col="t"),
        {"source": "default", "confidence": 0.5},
    )

    row = builder.build_annotation_row(2, 8, "beagle", "dog", confidence=0.9)

    assert row == {
        "s": 2,
        KEY_END_POS_COL: 8,
        "t": "beagle",
        KEY_ANN_TYPE_COL: "dog",
        "source": "default",
        "confidence": 0.9,
    }


def test_builder_defaults_its_metadata() -> None:
    builder = AnnotationsBuilder(None, None)

    assert builder.build_annotation_row(0, 1, "a", "x") == {
        KEY_START_POS_COL: 0,
        KEY_END_POS_COL: 1,
        KEY_TEXT_COL: "a",
        KEY_ANN_TYPE_COL: "x",
    }


# --- AnnotationsRowAccessor and DerivedAnnotationColumns ----------------------


class _Upper(DerivedAnnotationColumns):
    """Derives ``upper`` from the text column, and nothing else."""

    def get_col_value(self, metadata, col_type, row, missing=None):
        if col_type == "upper":
            return row[metadata.text_col].upper()
        return missing


def test_accessor_reads_a_key_type_through_its_configured_name() -> None:
    accessor = AnnotationsRowAccessor(AnnotationsMetaData(text_col="surface"))
    row = pd.Series({"surface": "beagle"})

    assert accessor.get_col_value(KEY_TEXT_COL, row) == "beagle"


def test_accessor_reads_a_column_the_row_carries_under_the_name_asked_for() -> None:
    """The documented second clause: ``col_type`` itself, as a column.

    It answered only for a name among the metadata's column *types*, and
    otherwise went straight to the derived columns, so a column the row carried
    under exactly the name asked for read as the missing value.
    """
    accessor = AnnotationsRowAccessor(AnnotationsMetaData())
    row = pd.Series({KEY_TEXT_COL: "07/04/1776", "date_field": "day"})

    assert accessor.get_col_value("date_field", row, missing="?") == "day"


def test_accessor_prefers_the_configured_column_to_the_type_name() -> None:
    accessor = AnnotationsRowAccessor(AnnotationsMetaData(text_col="surface"))
    row = pd.Series({"surface": "beagle", KEY_TEXT_COL: "other"})

    assert accessor.get_col_value(KEY_TEXT_COL, row) == "beagle"


def test_accessor_falls_back_to_the_type_name_for_a_key_type() -> None:
    accessor = AnnotationsRowAccessor(AnnotationsMetaData(text_col="surface"))
    row = pd.Series({KEY_TEXT_COL: "beagle"})

    assert accessor.get_col_value(KEY_TEXT_COL, row) == "beagle"


def test_accessor_derives_what_the_row_does_not_carry() -> None:
    accessor = AnnotationsRowAccessor(AnnotationsMetaData(), _Upper())
    row = pd.Series({KEY_TEXT_COL: "beagle"})

    assert accessor.get_col_value("upper", row) == "BEAGLE"
    assert accessor.get_col_value("absent", row, missing="?") == "?"


def test_accessor_prefers_the_rows_own_column_to_a_derived_one() -> None:
    """The docstring's order: ``col_type`` as a column, then derived columns."""
    accessor = AnnotationsRowAccessor(AnnotationsMetaData(), _Upper())
    row = pd.Series({KEY_TEXT_COL: "beagle", "upper": "carried"})

    assert accessor.get_col_value("upper", row) == "carried"


def test_accessor_returns_missing_for_an_unknown_column() -> None:
    accessor = AnnotationsRowAccessor(AnnotationsMetaData())

    assert accessor.get_col_value("absent", pd.Series({KEY_TEXT_COL: "x"}), "?") == "?"


def test_a_key_type_whose_column_is_absent_still_raises() -> None:
    """A row written in another vocabulary is a loud failure, not a miss.

    Reading rows through the wrong metadata leaves every key type's column
    absent. Answering ``missing`` there would hide the mismatch, so the
    accessor raises as it always has, and names both spellings it tried.
    """
    accessor = AnnotationsRowAccessor(AnnotationsMetaData(text_col="surface"))

    with pytest.raises(KeyError, match="surface"):
        accessor.get_col_value(KEY_TEXT_COL, pd.Series({"other": "beagle"}))


# --- AnnotatedText -------------------------------------------------------------


def _annotated() -> AnnotatedText:
    doc = AnnotatedText(TEXT, TextMetaData("doc1"))
    doc.add_annotations(_dogs())
    return doc


def test_annotated_text_creates_its_annotations_and_bookmarks_on_demand() -> None:
    doc = AnnotatedText(
        TEXT, TextMetaData("doc1"), annots_metadata=AnnotationsMetaData(text_col="t")
    )

    assert doc.annotations.is_empty()
    assert doc.annotations.metadata.text_col == "t"
    assert doc.bookmarks == {}
    doc.bookmarks["saved"] = pd.DataFrame()
    assert "saved" in doc.bookmarks


def test_annotated_text_takes_its_text_from_a_text_obj() -> None:
    doc = AnnotatedText("ignored", None, text_obj=AnnotatedText(TEXT, TextMetaData("doc9")))

    assert doc.text == TEXT
    assert doc.text_id == "doc9"


def test_add_annotations_adopts_then_appends() -> None:
    doc = _annotated()
    more = Annotations(AnnotationsMetaData())
    more.add_dict(_row(9, 12, ann_type="verb"))

    doc.add_annotations(more)
    doc.add_annotations(Annotations(AnnotationsMetaData()))  # empty: ignored

    assert list(doc.annotations.df[KEY_TEXT_COL]) == ["beagle", "met", "collie"]


def test_add_annotations_copies_into_an_existing_empty_store() -> None:
    doc = AnnotatedText(TEXT, TextMetaData("doc1"))
    store = doc.annotations  # created, empty

    doc.add_annotations(_dogs())

    assert doc.annotations is store
    assert list(store.df["breed"]) == ["beagle", "collie"]


def test_get_text_series_pads_both_ends() -> None:
    series = AnnotatedText("ab", None).get_text_series(pad_len=2)

    assert "".join(series) == "  ab  "


def test_get_text_without_a_mask_is_the_text() -> None:
    assert _annotated().get_text() == TEXT


def test_get_text_masks_the_annotated_spans() -> None:
    assert _annotated().get_text({"breed": "#"}) == "a ###### met a ######"


def test_get_text_masks_by_the_frame_it_is_given() -> None:
    """``annot_df`` overrides the text's own annotations, as documented."""
    override = pd.DataFrame([_row(9, 12, ann_type="verb", verb="met")])

    assert _annotated().get_text({"verb": "_"}, annot_df=override) == "a beagle ___ a collie"


def test_get_text_masks_by_the_configured_position_columns() -> None:
    meta = AnnotationsMetaData(start_pos_col="s", end_pos_col="e")
    annots = Annotations(meta)
    annots.add_dict({"s": 2, "e": 8, "breed": "beagle"})
    doc = AnnotatedText(TEXT, None, annots=annots)

    assert doc.get_text({"breed": "#"}) == "a ###### met a collie"


def test_get_text_with_no_annotations_is_the_text() -> None:
    doc = AnnotatedText(TEXT, None)

    assert doc.get_text({"breed": "#"}) == TEXT


def test_get_annot_mask_marks_the_annotated_characters() -> None:
    mask = _annotated().get_annot_mask("breed")

    assert "".join("x" if m else "." for m in mask) == "..xxxxxx.......xxxxxx"


def test_get_annot_mask_pads_with_false_at_both_ends() -> None:
    mask = _annotated().get_annot_mask("breed", pad_len=1)

    assert len(mask) == len(TEXT) + 2
    assert "".join("x" if m else "." for m in mask) == "...xxxxxx.......xxxxxx."


def test_get_annot_mask_reads_the_configured_position_columns() -> None:
    meta = AnnotationsMetaData(start_pos_col="s", end_pos_col="e")
    annots = Annotations(meta)
    annots.add_dict({"s": 2, "e": 8, "breed": "beagle"})
    doc = AnnotatedText(TEXT, None, annots=annots)

    mask = doc.get_annot_mask("breed")

    assert "".join("x" if m else "." for m in mask) == "..xxxxxx" + "." * 13


def test_get_annot_mask_is_all_false_when_nothing_is_annotated() -> None:
    mask = AnnotatedText(TEXT, None).get_annot_mask("breed")

    assert len(mask) == len(TEXT)
    assert not mask.any()


def test_get_annot_mask_is_none_for_a_column_no_row_has() -> None:
    assert _annotated().get_annot_mask("colour") is None


# --- Annotator -----------------------------------------------------------------


def test_annotator_is_the_extension_point() -> None:
    class Breeds(Annotator):
        def annotate_input(self, text_obj, **kwargs):
            found = Annotations(text_obj.annotations.metadata)
            start = text_obj.text.find("collie")
            found.add_dict(_row(start, start + 6, breed="collie"))
            text_obj.add_annotations(found)
            return found

    doc = AnnotatedText(TEXT, None)
    added = Breeds("breeds").annotate_input(doc)

    assert list(added.df["breed"]) == ["collie"]
    assert doc.get_text({"breed": "#"}) == "a beagle met a ######"
    with pytest.raises(TypeError):
        Annotator("abstract")  # type: ignore[abstract]

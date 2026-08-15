"""``upload`` rejects unusable column labels with a diagnostic that helps.

Rejecting is not the finding — an unnamed SQL column is not something
``upload`` should invent a name for, and staying strict matches how the rest of
this module treats identifiers. The finding was the message.

``pd.DataFrame([[1, 2]])`` carries pandas' default *integer* labels, and the
caller used to see ``ValueError: Invalid SQL identifier: 0`` raised from inside
the schema builder. That names neither the subject (a DataFrame column label),
nor where it came from (pandas supplied it), nor the fix (one assignment to
``df.columns``). A caller reading it has to go and find all three.

No live server: the check runs before any SQL is built, which is also the point
— the frame is refused whole rather than half-created.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dataknobs_utils.sql_utils import PostgresDB


def _message(df: pd.DataFrame) -> str:
    with pytest.raises(ValueError) as caught:
        PostgresDB._require_usable_column_labels(df)
    return str(caught.value)


class TestTheDiagnosticNamesWhatIsWrong:
    def test_default_integer_labels_are_explained(self) -> None:
        """The case that produced the bare `Invalid SQL identifier: 0`."""
        message = _message(pd.DataFrame([[1, 2]]))

        assert "column label" in message
        assert "position 0" in message, "the offending column is not located"
        assert "int" in message, "the label's type is not named"
        assert "df.columns" in message, "the fix is not named"

    def test_every_offending_label_is_listed(self) -> None:
        """Up-front rather than per column: stopping at the first would make a
        three-bad-label frame a three-round trip.
        """
        message = _message(pd.DataFrame([[1, 2, 3]]))

        assert "3 of 3" in message
        for position in ("position 0", "position 1", "position 2"):
            assert position in message

    def test_only_the_offenders_are_listed(self) -> None:
        df = pd.DataFrame({"good": [1], "also_good": [2]})
        df.columns = ["good", 7]  # type: ignore[assignment]

        message = _message(df)

        assert "1 of 2" in message
        assert "position 1" in message
        assert "position 0" not in message

    def test_empty_string_label_is_refused(self) -> None:
        """``quote_ident`` rejects it too; the point is that it is refused here,
        where the message can say what it is.
        """
        df = pd.DataFrame({"a": [1]})
        df.columns = [""]

        assert "position 0" in _message(df)


class TestDuplicateLabelsAreRejectedHereToo:
    """A repeated label is unusable for the same reason, one step later.

    ``df[col]`` returns a *DataFrame* rather than a Series when the label
    appears twice, so the value conversion died on ``values.dtype`` with
    ``AttributeError: 'DataFrame' object has no attribute 'dtype'`` — a message
    naming an internal type, from a helper the caller never invoked, about a
    frame whose actual problem is one the guard right above it already exists
    to describe.

    It also cannot succeed on its own terms: the INSERT names each column once,
    so two columns of the same name have no distinguishable destination.
    """

    def test_a_repeated_label_is_named_with_its_positions(self) -> None:
        """Bug: this reached the value conversion and raised ``AttributeError``
        about a DataFrame having no ``dtype``.
        """
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])

        message = _message(df)

        assert "duplicate" in message.lower()
        assert "'a'" in message
        assert "0" in message and "2" in message, "both positions should be named"

    def test_the_message_distinguishes_it_from_the_unusable_label_case(self) -> None:
        """Two different problems, two different fixes — a caller must not have
        to guess which one they have.
        """
        duplicate = _message(pd.DataFrame([[1, 2]], columns=["a", "a"]))
        unusable = _message(pd.DataFrame([[1, 2]]))

        assert "non-empty strings" not in duplicate
        assert "duplicate" not in unusable.lower()

    def test_distinct_labels_are_still_accepted(self) -> None:
        PostgresDB._require_usable_column_labels(pd.DataFrame([[1, 2]], columns=["a", "b"]))


class TestUsableLabelsPass:
    def test_ordinary_names_are_accepted(self) -> None:
        PostgresDB._require_usable_column_labels(pd.DataFrame({"a": [1], "b": [2]}))

    def test_awkward_but_legal_names_are_accepted(self) -> None:
        """Quoting is what makes these work; the guard must not pre-empt it."""
        df = pd.DataFrame({"My Column": [1], "user": [2], 'we"ird': [3]})

        PostgresDB._require_usable_column_labels(df)

    def test_a_frame_with_no_columns_is_accepted(self) -> None:
        """Nothing to reject. Whether an empty frame is uploadable is a
        different question from whether its labels are usable.
        """
        PostgresDB._require_usable_column_labels(pd.DataFrame())

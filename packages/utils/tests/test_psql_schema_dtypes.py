"""Reproduce-first tests for the dtype ladder in PostgresDB._psql_schema_line.

The ladder named only integer and float, and both of its branches ended in
``df[col].str.len()`` -- which encodes "not integer and not float, therefore a
string". That is false for every dtype family the ladder does not name, and it
fails in two different ways:

* ``bool``, ``datetime64[ns]``, the nullable ``boolean`` extension type and
  tz-aware ``datetime64[ns, UTC]`` all reach ``.str`` and raise
  ``AttributeError`` -- a CREATE TABLE generator that crashes on a boolean
  column.
* ``timedelta64[ns]`` subclasses ``np.signedinteger``, so ``np.issubdtype(dtype,
  np.integer)`` reported it as an integer and the column was emitted as
  ``integer``. No crash, wrong schema.

The mapping below is the fix: one ladder over ``pd.api.types`` predicates, which
answer correctly for numpy dtypes and pandas ExtensionDtypes alike, and a
varchar fallback that measures the rendered width instead of assuming ``.str``
is available.
"""

import numpy as np
import pandas as pd

from dataknobs_utils.sql_utils import PostgresDB


class TestBooleanColumns:
    """bool is a type of its own, not a string and not an integer."""

    def test_numpy_bool_is_boolean(self):
        """Bug: bool is neither np.integer nor np.floating, so it fell through
        to df[col].str.len() and raised AttributeError.
        """
        df = pd.DataFrame({"flag": [True, False]})
        assert PostgresDB._psql_schema_line(df, "flag") == '"flag" boolean'

    def test_nullable_boolean_is_boolean(self):
        """The ExtensionDtype branch had the same gap as the numpy branch:
        is_integer_dtype and is_float_dtype are both False for BooleanDtype.
        """
        df = pd.DataFrame({"flag": pd.array([True, None], dtype="boolean")})
        assert PostgresDB._psql_schema_line(df, "flag") == '"flag" boolean'


class TestTemporalColumns:
    """Timestamps and durations have SQL types; neither is varchar."""

    def test_datetime_is_timestamp(self):
        """Bug: datetime64[ns] reached df[col].str.len() and raised."""
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-06-01"])})
        assert PostgresDB._psql_schema_line(df, "ts") == '"ts" timestamp'

    def test_tz_aware_datetime_is_timestamptz(self):
        """A tz-aware column is an ExtensionDtype, so it took the second branch
        and raised there. Emitting bare `timestamp` would silently discard the
        offset, so the two are distinguished.
        """
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"], utc=True)})
        assert PostgresDB._psql_schema_line(df, "ts") == '"ts" timestamptz'

    def test_timedelta_is_interval_not_integer(self):
        """Bug: timedelta64[ns] IS an np.signedinteger subtype, so
        np.issubdtype(dtype, np.integer) was True and the column was silently
        emitted as `integer`. This is the mistyping, not a crash.
        """
        df = pd.DataFrame({"dur": pd.to_timedelta(["1 days", "2 days"])})
        assert PostgresDB._psql_schema_line(df, "dur") == '"dur" interval'


class TestVarcharFallback:
    """The fallback must measure width, not assume the column holds strings."""

    def test_object_column_of_non_strings(self):
        """Bug: an object column holding ints raised AttributeError on .str.
        upload() sends str(value) for every cell, so the width that matters is
        the rendered width.
        """
        df = pd.DataFrame({"c": pd.array([1, 222], dtype=object)})
        assert PostgresDB._psql_schema_line(df, "c") == '"c" varchar(3)'

    def test_object_column_of_strings_unchanged(self):
        """The string case must produce exactly what it produced before."""
        df = pd.DataFrame({"c": ["a", "bbb"]})
        assert PostgresDB._psql_schema_line(df, "c") == '"c" varchar(3)'

    def test_nulls_do_not_widen_the_column(self):
        """.str.len() skipped nulls; the replacement must too, or every
        nullable text column silently grows to fit the string 'None'.
        """
        df = pd.DataFrame({"c": ["a", None]})
        assert PostgresDB._psql_schema_line(df, "c") == '"c" varchar(1)'

    def test_categorical_column(self):
        df = pd.DataFrame({"c": pd.Series(["a", "bb"], dtype="category")})
        assert PostgresDB._psql_schema_line(df, "c") == '"c" varchar(2)'


class TestNumericLadderUnchanged:
    """The two families the ladder already named must keep their mapping."""

    def test_int64_is_integer(self):
        df = pd.DataFrame({"count": pd.array([1, 2], dtype="int64")})
        assert PostgresDB._psql_schema_line(df, "count") == '"count" integer'

    def test_nullable_int_is_integer(self):
        df = pd.DataFrame({"count": pd.array([1, None], dtype="Int64")})
        assert PostgresDB._psql_schema_line(df, "count") == '"count" integer'

    def test_float32_is_real(self):
        df = pd.DataFrame({"score": np.array([1.0, 2.0], dtype=np.float32)})
        assert PostgresDB._psql_schema_line(df, "score") == '"score" real'

    def test_nullable_float_is_real(self):
        df = pd.DataFrame({"score": pd.array([1.0, None], dtype="Float64")})
        assert PostgresDB._psql_schema_line(df, "score") == '"score" real'


class TestEmptyFrames:
    """An empty column has no width to measure; it must not raise."""

    def test_empty_object_column(self):
        df = pd.DataFrame({"tag": np.array([], dtype=object)})
        assert PostgresDB._psql_schema_line(df, "tag") == '"tag" varchar(1)'

    def test_empty_string_column(self):
        df = pd.DataFrame({"tag": pd.Series([], dtype="string")})
        assert PostgresDB._psql_schema_line(df, "tag") == '"tag" varchar(1)'

    def test_empty_bool_column(self):
        df = pd.DataFrame({"flag": pd.Series([], dtype="bool")})
        assert PostgresDB._psql_schema_line(df, "flag") == '"flag" boolean'


class TestVarcharWidthFitsWhatUploadSends:
    """A declared ``varchar(n)`` has to be a width PostgreSQL takes and values fit.

    The empty-column guard above answers "no values to measure". Two cases sit
    beside it that it does not answer, and both produce a column the data
    cannot be written into:

    * a column whose values are all the empty string measures 0, and
      PostgreSQL rejects the declaration itself -- ``length for type varchar
      must be at least 1`` -- so the table cannot be created at all. It is the
      empty-column case with one value in it, and routine in CSV/ETL loads.
    * ``astype(str)`` is not the renderer ``upload`` uses. ``upload`` sends
      ``str(value)``, and for a ``bytes`` payload the two disagree: ``astype``
      yields ``abc`` where ``str`` yields ``b'abc'``. The column is then
      declared narrower than the text written into it, and the INSERT fails
      with ``value too long for type character varying``.

    The second is what the helper's own docstring already claims it does, so
    these pin the claim rather than adding one.
    """

    def test_all_empty_strings_do_not_produce_varchar_zero(self):
        df = pd.DataFrame({"tag": ["", ""]})
        assert PostgresDB._psql_schema_line(df, "tag") == '"tag" varchar(1)'

    def test_an_empty_string_beside_a_null_does_not_produce_varchar_zero(self):
        df = pd.DataFrame({"tag": ["", None]})
        assert PostgresDB._psql_schema_line(df, "tag") == '"tag" varchar(1)'

    def test_width_is_measured_with_the_renderer_upload_uses(self):
        df = pd.DataFrame({"payload": pd.Series([b"abc"], dtype=object)})
        # The width that has to fit is the one upload actually writes, so the
        # expectation is derived from upload's own rendering rather than restated.
        sent = [str(record["payload"]) for record in df.to_records()]
        assert sent == ["b'abc'"]
        assert PostgresDB._psql_schema_line(df, "payload") == f'"payload" varchar({len(sent[0])})'

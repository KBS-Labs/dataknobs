"""A flat file format keeps a field's metadata, or staleness cannot be detected.

``CSVFormat.save`` and ``ParquetFormat.save`` each reduced a serialized field
to its bare ``value``, dropping ``type`` and ``metadata``. A ``VectorField``
went in and a plain ``Field`` holding a list of numbers came back --- with no
``content_hash``.

That is not a cosmetic loss. A vector with no digest is one nothing can judge
stale, and ``VectorTextSynchronizer`` treats a field it cannot judge as
**current**. So on these formats an edited record was never re-embedded, and
the sweep reported success. Measured before the fix, same corpus, same edit,
one ``sync_all()``:

========  ============================
Format    ``sync_all()`` after an edit
========  ============================
``json``  ``updated=1``, one embedding
``csv``   ``updated=0``, no embeddings
========  ============================

Both formats had their own copy of the reduction, which is why fixing one
would have left the other. The rule now lives in one place: a field whose
whole content is a scalar value still reduces to a bare cell --- that is what
makes a CSV readable in a spreadsheet, and the reason to ask for one --- and a
field carrying anything more is written as its full JSON dict, which
``Record.from_dict`` reconstructs exactly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.vector.content import (
    CONTENT_HASH_KEY,
    FIELD_SEPARATOR_KEY,
    SOURCE_FIELDS_KEY,
    compute_content_hash,
    content_hash_metadata,
)
from dataknobs_data.vector.sync import VectorTextSynchronizer

# Formats whose handler flattens a record into one row. ``json`` is the control:
# it stores the whole field dict and has always round-tripped.
FLAT_FORMATS = ["csv", "tsv"]
ALL_FORMATS = ["json", *FLAT_FORMATS]


def _described_record() -> Record:
    """A record carrying a vector that describes its own assembly."""
    record = Record(data={"title": "hello", "content": "world"})
    record.fields["embedding"] = VectorField(
        name="embedding",
        value=[0.1, 0.2, 0.3],
        source_field="title,content",
        model_name="m",
        model_version="1",
        metadata=content_hash_metadata(
            ["title", "content"], " ", compute_content_hash("hello world")
        ),
    )
    return record


def _db(root: Path, fmt: str) -> Any:
    return SyncFileDatabase({"path": str(root / f"records.{fmt}")})


@pytest.fixture
def root() -> Any:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestTheVectorSurvivesTheRoundTrip:
    """What went in comes back, on every format the backend offers."""

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_it_is_still_a_vector_field(self, root: Path, fmt: str) -> None:
        db = _db(root, fmt)
        rid = db.create(_described_record())

        field = db.read(rid).fields["embedding"]

        assert isinstance(field, VectorField), f"came back as {type(field).__name__}"

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_the_assembly_description_survives(self, root: Path, fmt: str) -> None:
        db = _db(root, fmt)
        rid = db.create(_described_record())

        metadata = db.read(rid).fields["embedding"].metadata

        assert metadata[CONTENT_HASH_KEY] == compute_content_hash("hello world")
        assert metadata[SOURCE_FIELDS_KEY] == ["title", "content"]
        assert metadata[FIELD_SEPARATOR_KEY] == " "

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_the_numbers_survive(self, root: Path, fmt: str) -> None:
        db = _db(root, fmt)
        rid = db.create(_described_record())

        value = list(db.read(rid).fields["embedding"].value)

        assert value == pytest.approx([0.1, 0.2, 0.3])


class TestAScalarFieldIsStillAPlainCell:
    """The half that must NOT change: a flat format stays flat where it can.

    Companions --- they pass before and after. Writing every field as a JSON
    dict would satisfy the round-trip cells above and destroy the reason to
    choose CSV in the first place.
    """

    @pytest.mark.parametrize("fmt", FLAT_FORMATS)
    def test_the_file_holds_bare_values(self, root: Path, fmt: str) -> None:
        path = root / f"records.{fmt}"
        db = SyncFileDatabase({"path": str(path)})
        db.create(Record(data={"title": "hello", "count": 3}))

        text = path.read_text()

        assert "hello" in text
        assert '"value"' not in text, f"a scalar field was written as a field dict:\n{text}"

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_scalars_round_trip_unchanged(self, root: Path, fmt: str) -> None:
        db = _db(root, fmt)
        rid = db.create(Record(data={"title": "hello", "count": 3}))

        record = db.read(rid)

        assert record.get_value("title") == "hello"
        # CSV has no types: a number read back as its string form is this
        # format's long-standing behaviour and not what this module changes.
        assert str(record.get_value("count")) == "3"


class TestStalenessIsDetectableOnAFlatFormat:
    """The end-to-end consequence, which is the reason any of this matters."""

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    async def test_an_edited_record_is_re_embedded(self, root: Path, fmt: str) -> None:
        calls: list[str] = []

        def embed(text: str) -> np.ndarray:
            calls.append(text)
            return np.array([float(len(text)), 1.0, 2.0])

        db = AsyncFileDatabase({"path": str(root / f"records.{fmt}")})
        rid = await db.create(Record(data={"content": "the original text"}))
        synchronizer = VectorTextSynchronizer(
            database=db,
            embedding_fn=embed,
            text_fields=["content"],
            vector_field="embedding",
        )

        first = await synchronizer.sync_all()
        assert first["updated"] == 1, f"the initial embed did not happen: {first}"

        # An unchanged corpus costs nothing.
        assert (await synchronizer.sync_all())["updated"] == 0

        stored = await db.read(rid)
        stored.set_value("content", "an entirely different text")
        await db.update(rid, stored)

        after_edit = await synchronizer.sync_all()

        assert after_edit["updated"] == 1, (
            f"the edit went undetected on {fmt}: {after_edit}; embeddings so far: {calls}"
        )
        assert calls[-1] == "an entirely different text"


@pytest.mark.parametrize("fmt", ["parquet"])
class TestParquetCarriesItTooWhenAvailable:
    """Parquet shares the flattening, so it shares the fix.

    Gated because ``pyarrow`` is an optional extra (``dataknobs-data[parquet]``)
    and is not installed by default, so this is the one format below that a
    default run does not measure.
    """

    def test_the_assembly_description_survives(self, root: Path, fmt: str) -> None:
        pytest.importorskip("pyarrow", reason="parquet support is an optional extra")

        db = _db(root, fmt)
        rid = db.create(_described_record())

        metadata = db.read(rid).fields["embedding"].metadata

        assert metadata[CONTENT_HASH_KEY] == compute_content_hash("hello world")

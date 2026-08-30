"""A JSON file the backend cannot use answers as an empty store, not a crash.

``JSONFormat.load`` already answered ``{}`` for a missing file, an empty file,
whitespace, and content that does not parse. Content that *parses* into
something other than a mapping fell through all four and was returned as the
record store --- so the value reached callers, and every one of them failed on
``.items()``. Measured before the fix, on a file holding ``["not", "a",
"mapping"]``:

=============  ================================================
Call           Result
=============  ================================================
``read``       ``AttributeError: 'list' object has no ...``
``all``        the same
``exists``     the same
=============  ================================================

Answering ``{}`` here is the same answer the four neighbouring branches give
for the same reason, and it is logged rather than silent, because discarding a
file the caller pointed at is worth a line in the log.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.file import SyncFileDatabase

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Well-formed JSON whose top level is not a mapping of records. A list is the
#: shape a caller most plausibly arrives with -- it is what a hand-written
#: export or a different tool's dump looks like.
NOT_A_MAPPING = [["not", "a", "mapping"], ["a string"], [42], [None]]


@pytest.fixture
def root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _db_holding(root: Path, payload: object) -> SyncFileDatabase:
    path = root / "records.json"
    path.write_text(json.dumps(payload))
    return SyncFileDatabase({"path": str(path)})


class TestItReadsAsEmptyRatherThanRaising:
    @pytest.mark.parametrize("payload", NOT_A_MAPPING, ids=["list", "string", "number", "null"])
    def test_every_entry_point(self, root: Path, payload: object) -> None:
        db = _db_holding(root, payload)
        assert db.all() == []
        assert db.read("anything") is None
        assert db.exists("anything") is False

    def test_the_discarded_file_is_logged(
        self, root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silent would be the wrong kind of robust: the data is being dropped."""
        db = _db_holding(root, ["not", "a", "mapping"])
        with caplog.at_level(logging.WARNING):
            db.all()
        assert any("not a mapping of records" in r.getMessage() for r in caplog.records)


class TestAUsableFileIsUnaffected:
    """The guard: the new branch must not catch the shape that always worked."""

    def test_a_written_corpus_round_trips(self, root: Path) -> None:
        db = SyncFileDatabase({"path": str(root / "records.json")})
        rid = db.create(Record(data={"colour": "red"}))
        reopened = SyncFileDatabase({"path": str(root / "records.json")})
        assert reopened.read(rid).get_value("colour") == "red"

    def test_an_empty_mapping_is_a_valid_empty_store(self, root: Path) -> None:
        db = _db_holding(root, {})
        assert db.all() == []
        assert db.read("anything") is None

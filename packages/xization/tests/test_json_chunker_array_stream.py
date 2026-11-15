"""Behavioural tests for JSON-array streaming in ``JSONChunker.stream_chunks``.

One chunk per object element of the record array, each element exactly once,
with the same text and metadata ``JSONChunker.chunk`` gives that element.
"""

import gzip
import io
import json
from pathlib import Path


from dataknobs_xization.json.json_chunker import JSONChunker


def _stream(doc: str) -> list[tuple[str, str]]:
    return [
        (c.text, c.source_path)
        for c in JSONChunker().stream_chunks(io.StringIO(doc), is_jsonl=False)
    ]


class TestArrayStreaming:
    def test_top_level_array(self) -> None:
        assert _stream('[{"title":"A"},{"title":"B"},{"title":"C"}]') == [
            ("A", ".[0]"),
            ("B", ".[1]"),
            ("C", ".[2]"),
        ]

    def test_non_object_elements_skipped(self) -> None:
        assert _stream('[{"title":"A"}, 5, {"title":"C"}]') == [("A", ".[0]"), ("C", ".[2]")]

    def test_empty_object_element(self) -> None:
        assert _stream('[{}, {"t":"A"}]') == [("", ".[0]"), ("t: A", ".[1]")]

    def test_no_array_yields_nothing(self) -> None:
        assert _stream('{"title":"solo"}') == []

    def test_empty_array_yields_nothing(self) -> None:
        assert _stream("[]") == []

    def test_array_under_one_key(self) -> None:
        assert _stream('{"items":[{"title":"A"},{"title":"B"},{"title":"C"}]}') == [
            ("A", ".items[0]"),
            ("B", ".items[1]"),
            ("C", ".items[2]"),
        ]

    def test_element_with_nested_array(self) -> None:
        assert _stream('[{"title":"A","tags":["x","y"]},{"title":"B"}]') == [
            ("A\ntags: x, y", ".[0]"),
            ("B", ".[1]"),
        ]

    def test_array_beside_other_keys(self) -> None:
        assert _stream('{"meta":"m","items":[{"title":"A"},{"title":"B"}]}') == [
            ("A", ".items[0]"),
            ("B", ".items[1]"),
        ]

    def test_deeper_array(self) -> None:
        assert _stream('{"a":{"items":[{"title":"A"}]}}') == [("A", ".a.items[0]")]

    def test_sibling_arrays(self) -> None:
        assert _stream('{"x":[{"t":"A"}],"y":[{"t":"B"}]}') == [
            ("t: A", ".x[0]"),
            ("t: B", ".y[0]"),
        ]

    def test_matches_chunk_text_and_metadata(self) -> None:
        data = [
            {"title": "A", "sub": {"k": [1, {"z": 2}]}, "n": None, "b": True},
            {"title": "B", "tags": ["x"]},
        ]
        streamed = list(JSONChunker().stream_chunks(io.StringIO(json.dumps(data)), is_jsonl=False))
        whole = JSONChunker().chunk(data)
        assert [(c.text, c.metadata) for c in streamed] == [(c.text, c.metadata) for c in whole]
        assert [c.source_path for c in streamed] == [".[0]", ".[1]"]

    def test_chunk_index_counts_yielded_chunks(self) -> None:
        doc = '{"items":[{"title":"A","tags":["x","y"]},{"title":"B"},{"title":"C"}]}'
        chunks = list(JSONChunker().stream_chunks(io.StringIO(doc), is_jsonl=False))
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_source_file_label_for_file_like(self) -> None:
        chunks = list(
            JSONChunker().stream_chunks(
                io.StringIO('[{"title":"A"}]'), is_jsonl=False, source_file="label"
            )
        )
        assert [c.source_file for c in chunks] == ["label"]

    def test_path_source(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.json"
        f.write_text('[{"title":"A"},{"title":"B"}]', encoding="utf-8")
        chunks = list(JSONChunker().stream_chunks(str(f)))
        assert [(c.text, c.source_path, c.source_file) for c in chunks] == [
            ("A", ".[0]", str(f)),
            ("B", ".[1]", str(f)),
        ]

    def test_gzip_path_source(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.json.gz"
        with gzip.open(f, "wt", encoding="utf-8") as out:
            out.write('{"items":[{"title":"A","tags":["x"]},{"title":"B"}]}')
        chunks = list(JSONChunker().stream_chunks(str(f)))
        assert [(c.text, c.source_path) for c in chunks] == [
            ("A\ntags: x", ".items[0]"),
            ("B", ".items[1]"),
        ]


class _CountingReader(io.StringIO):
    """A text stream that records how many characters have been read."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.consumed = 0

    def read(self, size: int | None = -1) -> str:
        data = super().read(size)
        self.consumed += len(data)
        return data


class TestIncrementalStreaming:
    def test_first_chunk_arrives_before_the_source_is_read(self) -> None:
        elements = [{"title": f"T{i}", "body": "x" * 200} for i in range(2000)]
        text = json.dumps({"items": elements})
        reader = _CountingReader(text)
        chunks = JSONChunker().stream_chunks(reader, is_jsonl=False)
        try:
            first = next(chunks)
            assert first.source_path == ".items[0]"
            assert reader.consumed < len(text) // 2
        finally:
            chunks.close()  # type: ignore[attr-defined]

    def test_closing_early_stops_reading(self) -> None:
        elements = [{"title": f"T{i}", "body": "x" * 200} for i in range(2000)]
        text = json.dumps(elements)
        reader = _CountingReader(text)
        chunks = JSONChunker().stream_chunks(reader, is_jsonl=False)
        next(chunks)
        chunks.close()  # type: ignore[attr-defined]
        consumed_at_close = reader.consumed
        assert consumed_at_close < len(text)
        assert reader.consumed == consumed_at_close

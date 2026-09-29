r"""``Filter.matches`` reads a ``LIKE`` pattern the way SQL does.

The published contract is that ``LIKE``/``NOT_LIKE`` treat only ``%`` and
``_`` as wildcards, match case-insensitively, and match every other character
verbatim. ``Filter.matches`` turned the pattern into a regular expression
without escaping it, so the in-process backends (memory, file, S3), which all
filter through it, read ``.``, ``+``, ``[...]``, ``(`` and ``\`` as regex
syntax: ``'a.c'`` matched ``'abc'``, ``'a\b'`` did not match itself, and
``'(a%'`` raised ``re.error``. ``.`` also stopped at a newline, where SQL's
``%`` and ``_`` do not.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dataknobs_data import Query, Record, SyncDatabase
from dataknobs_data.query import Filter, Operator

# (pattern, value, matches) under SQL LIKE with only % and _ special.
CASES = [
    ("a%", "abc", True),
    ("A_C", "abc", True),
    ("50%", "50% off", True),
    ("a.c", "abc", False),
    ("a.c", "a.c", True),
    ("a+b", "aab", False),
    ("a[bc]", "ab", False),
    ("a[bc]", "a[bc]", True),
    ("a^b$", "a^b$", True),
    ("(a%", "(abc", True),
    ("a\\b", "a\\b", True),
    ("a\\%", "a\\b", True),
    ("a\\%", "a%", False),
    ("x%", "x\ny", True),
    ("x_y", "x\ny", True),
    ("abc", "abc\n", False),
    ("%", "", True),
    ("_", "", False),
]


@pytest.mark.parametrize(("pattern", "value", "expected"), CASES)
def test_like_treats_only_percent_and_underscore_as_wildcards(
    pattern: str, value: str, expected: bool
) -> None:
    assert Filter("t", Operator.LIKE, pattern).matches(value) is expected


@pytest.mark.parametrize(("pattern", "value", "expected"), CASES)
def test_not_like_is_the_complement_on_a_string(pattern: str, value: str, expected: bool) -> None:
    assert Filter("t", Operator.NOT_LIKE, pattern).matches(value) is not expected


@pytest.mark.parametrize("operator", [Operator.LIKE, Operator.NOT_LIKE])
@pytest.mark.parametrize("record_value", ["5", 5], ids=["str-record", "int-record"])
def test_a_pattern_that_is_not_a_string_is_refused_by_name(
    operator: Operator, record_value: object
) -> None:
    """Refused whatever the record holds, as the Elasticsearch translator refuses it."""
    with pytest.raises(ValueError, match="pattern must be a string"):
        Filter("t", operator, 5).matches(record_value)


@pytest.mark.parametrize("text", ["C++ (draft)", "[u]", "a.b?"])
def test_a_text_search_with_punctuation_finds_its_record_in_memory(text: str) -> None:
    """``Query.hybrid`` wraps free text in ``%...%``, so punctuation reached the regex."""
    db = SyncDatabase.from_backend("memory", config={})
    db.connect()
    db.create(Record({"content": f"notes on {text} here"}))
    db.create(Record({"content": "unrelated"}))
    found = db.search(Query().hybrid(text_query=text))
    assert [r["content"] for r in found] == [f"notes on {text} here"]


VALUES = sorted({value for _, value, _ in CASES})
PATTERNS = sorted({pattern for pattern, _, _ in CASES})

# DuckDB's LIKE is case-sensitive and the SQL emission does not fold case there,
# so the one pattern here whose case differs from its matches is a known
# disagreement. Named rather than derived, and strict, so the mark has to go
# when the emission is fixed.
DUCKDB_CASE_PATTERNS = {"A_C"}
DUCKDB_CASE = pytest.mark.xfail(
    strict=True, reason="DuckDB's LIKE is case-sensitive; the SQL emission does not fold case"
)


def _backend_pattern_params() -> list[object]:
    params: list[object] = []
    for backend in ("memory", "sqlite", "duckdb"):
        for pattern in PATTERNS:
            known = backend == "duckdb" and pattern in DUCKDB_CASE_PATTERNS
            marks = [DUCKDB_CASE] if known else []
            params.append(pytest.param(backend, pattern, marks=marks, id=f"{backend}-{pattern!r}"))
    return params


@pytest.fixture(scope="module")
def databases() -> Iterator[dict[str, SyncDatabase]]:
    opened: dict[str, SyncDatabase] = {}
    try:
        for backend in ("memory", "sqlite", "duckdb"):
            config = {} if backend == "memory" else {"path": ":memory:"}
            database = SyncDatabase.from_backend(backend, config=config)
            database.connect()
            opened[backend] = database
            for value in VALUES:
                database.create(Record({"t": value}))
        yield opened
    finally:
        for database in opened.values():
            database.close()


@pytest.mark.parametrize(("backend", "pattern"), _backend_pattern_params())
def test_every_in_process_backend_answers_like_alike(
    databases: dict[str, SyncDatabase], backend: str, pattern: str
) -> None:
    r"""Memory agrees with sqlite and DuckDB, whose LIKE is the engine's own.

    Postgres is left out on purpose: its ``LIKE`` takes ``\`` as an escape
    character by default, which the contract does not, and that is a defect in
    the SQL emission rather than in this oracle.
    """
    expected = [v for v in VALUES if Filter("t", Operator.LIKE, pattern).matches(v)]
    query = Query(filters=[Filter("t", Operator.LIKE, pattern)])
    assert sorted(r["t"] for r in databases[backend].search(query)) == expected

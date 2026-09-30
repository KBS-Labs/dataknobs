r"""``Filter.matches`` reads a ``LIKE`` pattern the way SQL does.

The published contract is that ``LIKE``/``NOT_LIKE`` treat only ``%`` and
``_`` as wildcards, match case-insensitively, and match every other character
verbatim. ``Filter.matches`` turned the pattern into a regular expression
without escaping it, so the in-process backends (memory, file, S3), which all
filter through it, read ``.``, ``+``, ``[...]``, ``(`` and ``\`` as regex
syntax: ``'a.c'`` matched ``'abc'``, ``'a\b'`` did not match itself, and
``'(a%'`` raised ``re.error``. ``.`` also stopped at a newline, where SQL's
``%`` and ``_`` do not.

Every backend is then held to that oracle. PostgreSQL and DuckDB pushed down a
bare ``LIKE``, which is case-sensitive on both, and PostgreSQL's reads ``\`` as
an escape character. Where an engine's case folding naturally stops short of
the oracle's, the difference is declared below rather than skipped.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import (
    requires_localstack,
    requires_postgres,
    requires_real_elasticsearch,
)

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data.query import Filter, Operator

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    ("beagle", "Beagle", True),
    ("BEAGLE", "beagle", True),
    ("é", "É", True),
    ("ς", "Σ", True),
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

BACKENDS = [
    "memory",
    "file",
    "sqlite",
    "duckdb",
    pytest.param("postgres", marks=requires_postgres),
    pytest.param("s3", marks=requires_localstack),
    pytest.param("elasticsearch", marks=requires_real_elasticsearch),
]

_SQLITE_ASCII = "sqlite's LIKE folds ASCII case only"
_ES_ASCII = "Elasticsearch's case-insensitive wildcard folds ASCII case only"
_NO_SIGMA = "the engine folds one character at a time; Python's IGNORECASE also pairs final sigma"

#: Where a backend's LIKE naturally answers differently from ``Filter.matches``,
#: and why. Each entry is asserted to still differ, so one that stops holding
#: fails rather than lingering as a stale excuse.
DECLARED_VARIATIONS: dict[tuple[str, str], str] = {
    ("sqlite", "é"): _SQLITE_ASCII,
    ("sqlite", "ς"): _SQLITE_ASCII,
    ("elasticsearch", "é"): _ES_ASCII,
    ("elasticsearch", "ς"): _ES_ASCII,
    ("duckdb", "ς"): _NO_SIGMA,
    ("postgres", "ς"): _NO_SIGMA,
}

#: PostgreSQL's ILIKE folds case as the database's ``LC_CTYPE`` does, and these
#: fold ASCII only. Read from the database under test rather than assumed.
_ASCII_ONLY_CTYPES = {"C", "POSIX"}


def _postgres_ctype(config: dict[str, Any]) -> str:
    import psycopg2

    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        dbname=config["database"],
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("SHOW lc_ctype")
            return str(cursor.fetchone()[0])
    finally:
        conn.close()


def _declared(kind: str, config: dict[str, Any]) -> set[str]:
    """The patterns whose answer on ``kind`` is a declared variation."""
    declared = {pattern for backend, pattern in DECLARED_VARIATIONS if backend == kind}
    if kind == "postgres" and _postgres_ctype(config) in _ASCII_ONLY_CTYPES:
        declared.add("é")
    return declared


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """One backend's kind and constructor config, resolved in a sync fixture."""
    kind = request.param
    if kind == "postgres":
        yield from (
            (kind, c) for c in request.getfixturevalue("make_postgres_test_db")("test_like_")
        )
    elif kind == "s3":
        for config in request.getfixturevalue("make_localstack_s3_bucket")("dataknobs-like"):
            yield kind, {**config, "prefix": f"like-{uuid.uuid4().hex[:10]}/"}
    elif kind == "elasticsearch":
        yield from (
            (kind, c)
            for c in request.getfixturevalue("make_elasticsearch_test_index")("test_like_")
        )
    else:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            yield (
                kind,
                {
                    "memory": {},
                    "file": {"path": str(root / "records.json")},
                    "sqlite": {"path": str(root / "records.db")},
                    "duckdb": {"path": str(root / "records.duckdb"), "table": "records"},
                }[kind],
            )


def _answers(found: list[Record]) -> list[str]:
    return sorted(r["t"] for r in found)


@pytest.fixture
def declared(backend: tuple[str, dict[str, Any]]) -> set[str]:
    """Declared variations for this backend, resolved before any loop runs."""
    return _declared(*backend)


def _disagreements(
    declared: set[str], answers: dict[tuple[Operator, str], list[str]]
) -> dict[str, object]:
    """Searches whose answer differs from the oracle's, less the declared ones.

    A declared variation that answers like the oracle is reported too.
    """
    wrong: dict[str, object] = {}
    for (operator, pattern), got in answers.items():
        expected = [v for v in VALUES if Filter("t", operator, pattern).matches(v)]
        is_declared = pattern in declared
        if (got != expected) != is_declared:
            key = f"{operator.value} {pattern!r}"
            wrong[key] = "declared, but agrees" if is_declared else {"got": got, "want": expected}
    return wrong


SEARCHES = [(op, p) for op in (Operator.LIKE, Operator.NOT_LIKE) for p in PATTERNS]


def _query(operator: Operator, pattern: str) -> Query:
    return Query(filters=[Filter("t", operator, pattern)])


def test_every_backend_answers_like_as_the_oracle_does_sync(
    backend: tuple[str, dict[str, Any]], declared: set[str]
) -> None:
    r"""Case-insensitive, and ``%`` and ``_`` the only wildcards, on every backend.

    Postgres and DuckDB matched case-sensitively, and Postgres read ``\`` as an
    escape character, while the published contract says otherwise.
    """
    kind, config = backend
    db = SyncDatabase.from_backend(kind, config=config)
    try:
        for value in VALUES:
            db.create(Record({"t": value}))
        answers = {s: _answers(db.search(_query(*s))) for s in SEARCHES}
    finally:
        if kind == "s3":
            db.clear()
        db.close()
    assert _disagreements(declared, answers) == {}


async def test_every_backend_answers_like_as_the_oracle_does_async(
    backend: tuple[str, dict[str, Any]], declared: set[str]
) -> None:
    kind, config = backend
    db = await AsyncDatabase.from_backend(kind, config=config)
    try:
        for value in VALUES:
            await db.create(Record({"t": value}))
        answers = {s: _answers(await db.search(_query(*s))) for s in SEARCHES}
    finally:
        if kind == "s3":
            await db.clear()
        await db.close()
    assert _disagreements(declared, answers) == {}

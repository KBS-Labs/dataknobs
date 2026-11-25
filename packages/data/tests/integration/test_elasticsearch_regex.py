"""``Operator.REGEX`` answers on Elasticsearch as ``Filter.matches`` answers.

Lucene's ``regexp`` matches the whole value, has no anchors and gives characters
Python matches literally a meaning of their own: ``eag`` did not find ``eagle``,
and ``@`` found every record.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest
from dataknobs_common.testing import requires_real_elasticsearch

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

VALUES = [
    "eagle", "Beagle", "Eagle", "a", "b", "ab", "abab", "aab", "ba", "a\n", "a\nb", "\n", "a\n\n",
    "a b", "a\tb", "a-b", "a.b", "a+b", "a*b", "a|b", "a\\b", 'say "hi"', "x@y", "~x", "a&b",
    "a#b", "<1-5>", "[x]", "{2}", "(a)", "^a", "a$", "3", "12", "555-1234", "a1", "_", ",x", "x",
    "é", "É", "日本", "😀", "a😀b", "a/b", "a]b", "a^b", "a\N{NO-BREAK SPACE}b", "a\N{LINE SEPARATOR}b", "", " ", "d",
    "w",
]  # fmt: skip

PATTERNS = [
    # Found anywhere in the value.
    "eag", "eagle", "[Bb]eagle", "", "a", "ab", "日", "😀", "a.b", "(ab)", "x|y", "(?:ab)+",
    # Anchored.
    "^eag", "gle$", "^eagle$", r"\Aa\Z", r"a\Z", "a$", "^$", "^", "$", "^a|b$", "^(a|b)$", "(^a|b)",
    "(^|,)x", "(a|b$)", "(^a)?b", "(a$)?", "^^a$$", r"(a$)\Z", r"(a\Z)$", "(^)a", "x($)", "^a+$",
    "(?:^|a)b(?:$|a)", r"(^|\n)b", "a$|b",
    # Characters Lucene reads as operators, which Python does not.
    "@", "~x", "a&b", "#", "<1-5>", '"', 'say "hi"', "-", "/", "&&",
    # Escapes.
    r"\.", r"\+", r"\*", r"\|", r"\\", r"\[x\]", r"\{2\}", r"\(a\)", r"\^a", r"a\$", r"\@", r"\~",
    r"\"", r"\#", r"\<", r"a\tb", r"a\nb", r"\x41", r"é", r"\U0001F600", r"\n", r"\-", r"\/",
    # Classes.
    "[0-9]+", "[^a-z]", "[a-]", "[]x]", "[^]]", r"[\]^]", r"[\\]", r"[.]", "[a^]", '["@~#&<]',
    r"\d", r"\D", r"^\d{3}-\d{4}$", r"\w", r"^\W$", r"\s", r"^\S+$", r"[\d,]", r"^[^\d\s]$",
    r"^[1\D]+$", r"^[\w-]+$", r"[\s\S]", "[é-ü]", "[^😀]", r"^[^\n]*$", r"[\W\d]", "[日-本]",
    # Repeats.
    "a*", "^a{2}", "^(ab){2}$", "a{1,}b", "^a{0,1}b", "a+?b", "a??b", "^a*?$", "a{,2}b$",
    "^(a|b)*$", "(?P<n>a)b", "()a", "(|a)b", "^(a|)$", "^.$", "^.*$", "^..$", "^(?:a|bc?)+$",
    "[ab]c?$", "(a*)*b", "(?#note)ab",
]  # fmt: skip


async def settle(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@pytest.fixture(params=[SyncElasticsearchDatabase, AsyncElasticsearchDatabase])
async def db(
    request: pytest.FixtureRequest, make_elasticsearch_test_index: Any
) -> AsyncIterator[Any]:
    for config in make_elasticsearch_test_index("test_regex_"):
        database = request.param(config)
        await settle(database.connect())
        await settle(database.create_batch([Record({"t": v}) for v in VALUES]))
        yield database
        await settle(database.close())


async def _found(db: Any, pattern: str) -> list[str]:
    query = Query(filters=[Filter("t", Operator.REGEX, pattern)], limit_value=1000)
    return sorted(r["t"] for r in await settle(db.search(query)))


async def test_a_pattern_finds_what_it_finds_in_memory(db: Any) -> None:
    wrong = {}
    for pattern in PATTERNS:
        spec = Filter("t", Operator.REGEX, pattern)
        expected = sorted(v for v in VALUES if spec.matches(v))
        found = await _found(db, pattern)
        if found != expected:
            wrong[pattern] = (expected, found)
    # ``\w`` is ASCII there, and these are the patterns that show it.
    assert wrong.keys() == {r"\w", r"^\W$", r"^[\w-]+$", r"[\W\d]"}, wrong


async def test_word_and_digit_classes_are_ascii(db: Any) -> None:
    assert await _found(db, r"^\w\Z") == sorted(["3", "_", "a", "b", "d", "w", "x"])
    assert await _found(db, r"^\W\Z") == sorted(["\n", " ", "É", "é", "😀"])


@pytest.mark.parametrize("pattern", ["(?=a)", r"\ba", "(?i)a", r"(a)\1", "("])
async def test_a_pattern_that_cannot_be_asked_raises(db: Any, pattern: str) -> None:
    with pytest.raises(ValueError, match="Elasticsearch cannot match"):
        await _found(db, pattern)

"""The Lucene expression a Python one becomes.

``integration/test_elasticsearch_regex.py`` asks Elasticsearch whether it finds
what ``re.search`` does.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from dataknobs_data.backends.elasticsearch_regex import _CATEGORIES, python_regex_to_lucene, sre


@pytest.mark.parametrize(
    ("pattern", "lucene"),
    [
        ("eag", ".*eag.*"),
        ("", ".*.*"),
        ("^eag", "eag.*"),
        ("gle$", ".*gle\n?"),
        (r"\Aa\Z", "a"),
        (r"a$\Z", ".*a"),
        ("^$", "\n?"),
        (r"\A\Z", "()"),
        ("^a|b$", "(a.*|.*b\n?)"),
        ("(^|,)x", "((.*\\,)?)x.*"),
        (r"(\A|\Z)", "((.*|.*))"),
        ("(^a)?b", "((a)|.*)b.*"),
        ("a|", ".*(a)?.*"),
        ("(|)", ".*(()).*"),
        ('@"', '.*\\@\\".*'),
        ("^.$", "[^\n]\n?"),
        (r"[\s\S]", ".*..*"),
        (r"[^\d]", ".*[^0-9].*"),
        (r"[a-c\-]", ".*[\\-a-c].*"),
        (r"(?:ab)+?", ".*(ab)+.*"),
        ("a{2,}", ".*a{2,}.*"),
        ("a{2}", ".*a{2,2}.*"),
        ("(?:^a){1}b", "(a)b.*"),
        ("[a-zc]", ".*[a-z].*"),
    ],
)
def test_translates(pattern: str, lucene: str) -> None:
    assert python_regex_to_lucene(pattern) == lucene


@pytest.mark.parametrize(
    ("pattern", "why"),
    [
        ("a(?=b)", "look-around"),
        ("a(?!b)", "look-around"),
        ("(?<=a)b", "look-around"),
        ("(?<!a)b", "look-around"),
        (r"(a)\1", "back-reference"),
        ("(a)?(?(1)b|c)", "conditional group"),
        ("(?>a)", "atomic group"),
        ("a++", "possessive repeat"),
        (r"\ba", "word boundary"),
        (r"a\B", "word boundary"),
        ("(?i)a", "inline flags"),
        ("(?i:a)", "inline flags"),
        ("(?a)a", "inline flags"),
        ("a^", "anchor"),
        ("a$b", "anchor"),
        ("(?:^a)*", "anchor"),
        (r"[^\s\S]", "no character"),
        ("(", "unterminated"),
        ("a{99999999999999}", "too large"),
    ],
)
def test_refuses_what_lucene_cannot_match(pattern: str, why: str) -> None:
    with pytest.raises(ValueError, match=rf"Elasticsearch cannot match.*{why}"):
        python_regex_to_lucene(pattern)


@pytest.mark.parametrize(
    "pattern",
    ["(" * 450 + "a" + ")" * 450, "(" * 2000 + "a" + ")" * 2000],
    ids=["by the translator", "by the parser"],
)
def test_refuses_what_is_nested_too_deeply_to_read(pattern: str) -> None:
    with pytest.raises(ValueError, match=r"cannot match.*nested too deeply"):
        python_regex_to_lucene(pattern)


@pytest.mark.parametrize("pattern", [None, 5, b"a", re.compile("a")])
def test_refuses_what_is_not_a_string(pattern: Any) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        python_regex_to_lucene(pattern)


def test_space_is_every_space_python_knows() -> None:
    listed = {c for lo, hi in _CATEGORIES[sre.CATEGORY_SPACE] for c in range(lo, hi + 1)}
    assert listed == {c for c in range(0x110000) if re.match(r"\s", chr(c))}

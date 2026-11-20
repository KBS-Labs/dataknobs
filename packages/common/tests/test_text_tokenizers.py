"""``words``: the default tokens, with an apostrophe inside a word kept.

:func:`~dataknobs_common.text.token_spans` makes every apostrophe a boundary,
so *don't* is ``don`` and ``t``, and a list that labels *don't* never sees it.
:func:`~dataknobs_common.text.word_spans` joins tokens an in-word apostrophe
separates, and is registered in :data:`~dataknobs_common.text.TOKENIZERS` as
``words`` so a configuration can name it.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from dataknobs_common.entity_resolution import ScanningSignal
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import Entity, MappingEntitySource
from dataknobs_common.text import (
    TOKENIZERS,
    token_spans,
    tokenizer_from_config,
    word_spans,
)


def _words(text: str, **options: str) -> list[str]:
    return [text[start:end] for start, end in word_spans(text, **options)]


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("don't stop", ["don't", "stop"]),
        # U+2019, the apostrophe most text arrives with, joins the same way --
        # and the token is a slice of the input, so it is not folded to U+0027.
        ("what\u2019s up", ["what\u2019s", "up"]),
        ("rock'n'roll", ["rock'n'roll"]),
        ("O'Neil said", ["O'Neil", "said"]),
        # An apostrophe joins only between two token characters.
        ("dogs' toys", ["dogs", "toys"]),
        ("'quoted'", ["quoted"]),
        ("don''t", ["don", "t"]),
        ("it ' s", ["it", "s"]),
        # Every other boundary is the default's.
        ("snake_case", ["snake", "case"]),
        ("Over-the-Counter, $3.50", ["Over", "the", "Counter", "3", "50"]),
        # Combining marks are token characters here as in the default.
        ("e\u0301te's", ["e\u0301te's"]),
        ("", []),
    ],
)
def test_keep_joins_tokens_an_in_word_apostrophe_separates(text: str, words: list[str]) -> None:
    assert _words(text) == words
    assert _words(text, apostrophes="keep") == words


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("don't stop", ["don", "'t", "stop"]),
        ("John\u2019s", ["John", "\u2019s"]),
        # Every in-word apostrophe, not a list of English clitics.
        ("rock'n'roll", ["rock", "'n", "'roll"]),
        ("dogs' toys", ["dogs", "toys"]),
        ("don''t", ["don", "t"]),
    ],
)
def test_clitic_starts_a_token_at_each_in_word_apostrophe(text: str, words: list[str]) -> None:
    assert _words(text, apostrophes="clitic") == words


def test_an_unknown_apostrophe_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="apostrophes"):
        word_spans("don't", apostrophes="split")


_ALPHABET = "ab '\u2019-_.\u0301"


@pytest.mark.parametrize("mode", ["keep", "clitic"])
def test_word_spans_meets_the_tokenizer_contract_and_only_joins(mode: str) -> None:
    """In order, non-overlapping, none empty; each a run of default tokens.

    ``keep`` covers a run of default tokens whose gaps are each one
    apostrophe; ``clitic`` covers one default token, possibly with the single
    apostrophe before it. Measured over random text rather than argued.
    """
    rng = random.Random(1604)
    apostrophes = {"'", "\u2019"}
    for _ in range(3000):
        text = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 20)))
        defaults = token_spans(text)
        spans = word_spans(text, apostrophes=mode)
        previous_end = 0
        for start, end in spans:
            assert previous_end <= start < end, (text, spans)
            previous_end = end
        if mode == "keep":
            covered = [t for s, e in spans for t in defaults if s <= t[0] and t[1] <= e]
            assert covered == list(defaults), (text, spans)
            for start, end in spans:
                inner = [t for t in defaults if start <= t[0] and t[1] <= end]
                assert (inner[0][0], inner[-1][1]) == (start, end), (text, spans)
                for left, right in pairwise(inner):
                    assert right[0] - left[1] == 1, (text, spans)
                    assert text[left[1]] in apostrophes, (text, spans)
        else:
            assert len(spans) == len(defaults), (text, spans)
            for (start, end), (t_start, t_end) in zip(spans, defaults, strict=True):
                assert end == t_end, (text, spans)
                assert start in (t_start, t_start - 1), (text, spans)
                if start != t_start:
                    assert text[start] in apostrophes, (text, spans)


def test_words_is_registered_and_built_by_name() -> None:
    assert TOKENIZERS.is_registered("words")
    assert tokenizer_from_config("words") is word_spans
    assert tokenizer_from_config({"name": "words", "apostrophes": "keep"}) is word_spans
    clitic = tokenizer_from_config({"name": "words", "apostrophes": "clitic"})
    assert clitic is not None
    assert list(clitic("don't")) == [(0, 3), (3, 5)]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"name": "words", "apostrophes": "split"}, "apostrophes"),
        ({"name": "words", "lowercase": True}, "lowercase"),
    ],
)
def test_words_refuses_what_it_does_not_take(spec: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        tokenizer_from_config(spec)


def test_a_rung_over_words_does_not_match_inside_a_contraction() -> None:
    """The difference a profile depends on, shown on the rung that already takes it.

    Over the default tokens ``t`` is a token of *don't*, so a vocabulary
    declaring ``T`` finds it there. Over ``words`` *don't* is one token, and
    a window cannot start inside it.
    """
    source = MappingEntitySource({"t": Entity(id="t", type="Letter", name="T")})
    query = "don't"
    assert [c.entity_id for c in ScanningSignal(source).candidates(query, k=5)] == ["t"]
    over_words = ScanningSignal(source, tokenizer=tokenizer_from_config("words"))
    assert over_words.candidates(query, k=5) == []

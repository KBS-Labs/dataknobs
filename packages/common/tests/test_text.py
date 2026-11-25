"""The boundary policy's second half: how far a form's punctuation reaches.

:func:`~dataknobs_common.text.token_spans` makes every token boundary a place a
probe may start or end, so a probe always starts and ends on an alphanumeric
character. A declared form that starts or ends on anything else -- ``(beagle)``,
``C.D.C.`` -- was therefore never probed. The three functions here are the
rule that reaches it: measure how much punctuation a form carries at its edges,
extend a span over the punctuation adjacent to it in the text, and enumerate
both in the order a scan publishes.
"""

from __future__ import annotations

import random
import re
from operator import itemgetter

import pytest

from dataknobs_common.text import (
    edge_extensions,
    narrowest,
    probe_spans,
    in_token,
    is_token_boundary,
    punctuation_edges,
    token_spans,
)
from dataknobs_common.text.scan import _core


@pytest.mark.parametrize(
    ("form", "edges"),
    [
        ("C.D.C.", (0, 1)),
        ("(beagle)", (1, 1)),
        ("K-9", (0, 0)),
        ("", (0, 0)),
        ("«x»", (1, 1)),
        ("((beagle)),", (2, 3)),
        # Whitespace is not punctuation, so a run stops at it.
        ("  beagle  ", (0, 0)),
        ("( beagle )", (1, 1)),
    ],
)
def test_punctuation_edges_measures_the_runs_at_each_end(form: str, edges: tuple[int, int]) -> None:
    """A run is characters that are neither alphanumeric nor whitespace.

    The same predicate as :func:`token_spans` reads as *not a token*, minus
    whitespace -- which is what a separator between words is, and what an
    extension must never cross.
    """
    assert punctuation_edges(form) == edges


def test_a_form_of_punctuation_alone_is_all_edge() -> None:
    """No token, so the whole form is both its leading and its trailing run."""
    assert punctuation_edges(":-)") == (3, 3)
    assert token_spans(":-)") == ()


def test_an_extension_is_every_prefix_of_the_adjacent_run() -> None:
    """``a (beagle), here`` -- the run on the right is ``),``.

    Only its prefix ``)`` reaches ``(beagle)``, which is why the rule takes
    every prefix rather than the maximal run: a maximal-run rule misses the
    most common shape a parenthesised form takes in prose.
    """
    text = "a (beagle), here"
    base = (3, 9)
    assert text[slice(*base)] == "beagle"

    extended = edge_extensions(text, base, None)

    assert {text[start:end] for start, end in extended} == {
        "(beagle",
        "beagle)",
        "beagle),",
        "(beagle)",
        "(beagle),",
    }
    assert base not in extended, "the base span is the caller's, not an extension"


def test_extensions_come_widest_first() -> None:
    """A form containing another is proposed first, as across token counts."""
    text = "a (beagle), here"
    widths = [end - start for start, end in edge_extensions(text, (3, 9), None)]
    assert widths == sorted(widths, reverse=True)


def test_the_limit_bounds_each_side() -> None:
    """``((beagle))`` with a limit of one never reaches the outer parentheses."""
    text = "a ((beagle)) here"
    base = (4, 10)
    assert text[slice(*base)] == "beagle"

    once = {text[start:end] for start, end in edge_extensions(text, base, 1)}
    assert once == {"(beagle", "beagle)", "(beagle)"}

    assert edge_extensions(text, base, 0) == ()
    assert len(edge_extensions(text, base, None)) == 3 * 3 - 1


def test_no_extension_where_no_punctuation_touches_the_span() -> None:
    """The cost is zero for a span with a space or a string end beside it."""
    assert edge_extensions("my beagle here", (3, 9), None) == ()
    assert edge_extensions("beagle", (0, 6), None) == ()


def test_an_extension_never_crosses_whitespace_or_a_token() -> None:
    """Punctuation stays a boundary; the extension only reaches over it.

    ``C.D.C.D`` with ``C.D.C`` as the base extends to ``C.D.C.`` -- the run
    to the right is one character and stops at ``D``. Nothing here asks what
    lies past the run: the base policy already treats punctuation as a
    boundary, and an extension does not change that.
    """
    text = "x C.D.C.D y"
    base = (2, 7)
    assert text[slice(*base)] == "C.D.C"
    assert [text[start:end] for start, end in edge_extensions(text, base, None)] == ["C.D.C."]

    assert edge_extensions("( beagle )", (2, 8), None) == ()


def test_a_negative_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="-1"):
        edge_extensions("(beagle)", (1, 7), -1)
    with pytest.raises(ValueError, match="-1"):
        list(probe_spans("(beagle)", None, edge=-1))


def test_probe_spans_with_no_edge_is_the_token_enumeration_longest_first() -> None:
    """``edge=0`` is the enumeration a scan ran before edges existed."""
    query = "my golden retriever has been limping"
    tokens = token_spans(query)

    expected = [
        (tokens[first][0], tokens[first + length - 1][1])
        for length in range(len(tokens), 0, -1)
        for first in range(len(tokens) - length + 1)
    ]
    assert list(probe_spans(query, None)) == expected
    assert len(expected) == 21
    assert list(probe_spans(query, 2)) == expected[-11:]


def test_probe_spans_yields_extensions_before_the_base_they_extend() -> None:
    """A form containing another before the form it contains.

    The same principle that orders the enumeration longest first across token
    counts, applied to the one span and its extensions.
    """
    query = "a (beagle) here"
    probes = [query[start:end] for start, end in probe_spans(query, 1, edge=1)]

    assert probes.index("(beagle)") < probes.index("beagle")
    assert probes.index("(beagle") < probes.index("beagle")
    assert probes.index("beagle)") < probes.index("beagle")
    assert probes == ["a", "(beagle)", "(beagle", "beagle)", "beagle", "here"]


def test_probe_spans_never_repeats_a_span() -> None:
    """Each extension starts or ends on punctuation, so none is another base.

    Nor is one base's extension another's: a shared start would put an
    alphanumeric character inside a run that holds none.
    """
    query = "The C.D.C. (and AT&T), said."
    probes = list(probe_spans(query, None, edge=None))
    assert len(probes) == len(set(probes))


# ---------------------------------------------------------------------------
# The caller's own tokens
# ---------------------------------------------------------------------------


def _units(text: str, pattern: str) -> tuple[tuple[int, int], ...]:
    """A caller's tokenization: every match of ``pattern``, as spans."""
    return tuple(match.span() for match in re.finditer(pattern, text))


#: Runs of alphanumerics, or any one character that is neither alphanumeric
#: nor a space -- so ``$`` is a token of its own.
_SYMBOLS = r"[^\W_]+|[^\w\s]|_"


def test_a_scan_over_tokens_that_include_a_symbol_reaches_it() -> None:
    """``$`` is never a window over the default tokens, and is one over these."""
    text = "Price: $5,000 firm."
    vocabulary = {"$"}

    def found(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [span for span in spans if text[span[0] : span[1]] in vocabulary]

    assert found(list(probe_spans(text, 1))) == []
    assert found(list(probe_spans(text, 1, tokens=_units(text, _SYMBOLS)))) == [(7, 8)]


_AGREEMENT_TEXTS = [
    "",
    "   ",
    "...",
    ":-)",
    "beagle",
    "a (beagle), here",
    "C.D.C. said so",
    "  leading and trailing  ",
    "straße und ß",
    "東京 タワー",
    "x--y__z",
    "golden retriever, (beagle) & C.D.C.!",
]


@pytest.mark.parametrize("text", _AGREEMENT_TEXTS)
@pytest.mark.parametrize("longest", [1, 3, None])
@pytest.mark.parametrize("edge", [0, 1, None])
def test_the_default_tokens_passed_explicitly_enumerate_exactly_as_none(
    text: str, longest: int | None, edge: int | None
) -> None:
    explicit = list(probe_spans(text, longest, edge=edge, tokens=token_spans(text)))
    assert explicit == list(probe_spans(text, longest, edge=edge))


def test_a_caller_unit_is_one_window_and_never_split() -> None:
    """A grouped number is one unit, so none of its pieces is probed alone."""
    text = "a $1,234.50 price"
    units = _units(text, r"\d[\d,.]*\d|[^\W_]+|[^\w\s]")
    windows = {text[start:end] for start, end in probe_spans(text, 1, tokens=units)}
    assert "1,234.50" in windows
    assert windows.isdisjoint({"1", "234", "50"})


@pytest.mark.parametrize(
    ("tokens", "index"),
    [
        ([(0, 3), (2, 5)], 1),  # overlapping
        ([(4, 6), (0, 3)], 1),  # out of order
        ([(0, 3), (4, 99)], 1),  # past the end
        ([(-1, 3)], 0),  # negative start
        ([(0, 3), (4, 4)], 1),  # empty
        ([(0, 3), (4,)], 1),  # not a pair
        ([(0, 3), ("4", 6)], 1),  # not ints
        ([(0, True)], 0),  # a bool is not a bound
    ],
)
def test_tokens_no_window_can_be_built_on_are_refused_by_index(
    tokens: list[tuple[int, ...]], index: int
) -> None:
    with pytest.raises(ValueError, match=rf"token {index}\b"):
        list(probe_spans("abc def", None, tokens=tokens))  # type: ignore[arg-type]


def test_touching_tokens_are_accepted() -> None:
    text = "$5"
    assert list(probe_spans(text, None, tokens=[(0, 1), (1, 2)])) == [(0, 2), (0, 1), (1, 2)]


# ---------------------------------------------------------------------------
# narrowest
# ---------------------------------------------------------------------------


def test_narrowest_keeps_one_hit_where_a_caller_s_tokens_carry_separators() -> None:
    """``TX``, ``, TX`` and ``, TX,`` share a core once whitespace is trimmed too."""
    text = "Austin, TX, USA"
    hits = [((6, 11), "tx"), ((6, 10), "tx"), ((8, 10), "tx")]
    assert text[6:11] == ", TX," and text[8:10] == "TX"
    assert narrowest(text, hits, span=itemgetter(0), key=itemgetter(1)) == [((8, 10), "tx")]


def test_narrowest_keeps_a_symbol_apart_from_its_neighbour() -> None:
    """``$`` has the empty core at its start, so ``$5`` does not absorb it."""
    text = "costs $5"
    hits = [((6, 8), "money"), ((6, 7), "money")]
    assert narrowest(text, hits, span=itemgetter(0), key=itemgetter(1)) == hits


def test_narrowest_is_generic_and_keeps_first_arrival_order() -> None:
    text = "a (beagle) and (pug)"
    hits = [
        ((15, 20), "pug"),
        ((2, 10), "beagle"),
        ((16, 19), "pug"),
        ((3, 9), "beagle"),
        ((3, 9), "dog"),
    ]
    kept = narrowest(text, hits, span=itemgetter(0), key=itemgetter(1))
    assert kept == [((16, 19), "pug"), ((3, 9), "beagle"), ((3, 9), "dog")]


def test_narrowest_keeps_the_first_of_a_tie() -> None:
    text = "(a)"
    first, second = ((0, 2), "a", 1), ((1, 3), "a", 2)
    assert narrowest(text, [first, second], span=itemgetter(0), key=itemgetter(1)) == [first]


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        ((4, 12), "ends past the text"),
        ((-3, 2), "starts before it"),
        ((6, 4), "ends before it starts"),
    ],
)
def test_narrowest_refuses_a_span_outside_the_text_naming_the_hit(
    bad: tuple[int, int], why: str
) -> None:
    """A second scanner's hits are a caller's spans, checked as tokens are.

    Unchecked, a span past the end raised ``IndexError`` from inside the
    trimming, and a negative start wrapped round to the end of the text and
    grouped the hit under a core it never covered.
    """
    text = "a (beagle)"
    hits = [((3, 9), "beagle"), (bad, "beagle")]
    with pytest.raises(ValueError, match=rf"hit 1 .*{re.escape(repr(bad))}.*length 10"):
        narrowest(text, hits, span=itemgetter(0), key=itemgetter(1))


def test_narrowest_accepts_an_empty_span_and_one_ending_at_the_text_s_end() -> None:
    text = "a (beagle)"
    hits = [((10, 10), "end"), ((0, 10), "all")]
    assert narrowest(text, hits, span=itemgetter(0), key=itemgetter(1)) == hits


_ALPHABET = "abcXYZ019 \n" + "!\"#$%&'()*+,-./:;<=>?@[]^_{"


def _punctuation_core(text: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    leading, trailing = punctuation_edges(text[start:end])
    return (start + leading, end - trailing)


def test_narrowest_trims_every_default_probe_as_the_extension_rule_did() -> None:
    """Over :func:`probe_spans`'s own spans, trimming whitespace too changes nothing.

    ``narrowest`` trims every non-alphanumeric character; the private rule it
    replaced trimmed punctuation only. Those agree on every span a default
    scan yields, because a window starts and ends on a token and an extension
    never crosses whitespace -- measured here rather than argued.
    """
    rng = random.Random(293)
    checked = 0
    for _ in range(2000):
        text = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 24)))
        for edge in (0, 1, 2, None):
            for span in probe_spans(text, 4, edge=edge):
                assert _core(text, span) == _punctuation_core(text, span), (text, span)
                checked += 1
    assert checked > 10_000


# A combining mark is part of the word it is written on. ``str.isalnum`` is
# False for every one (Unicode category M*), so a predicate built on it alone
# split such a word at the mark and left the mark in no token at all.

_DUNIYA = "\u0926\u0941\u0928\u093f\u092f\u093e"  # दुनिया: three letters, three vowel signs
_DECOMPOSED = "e\u0301te"  # e + COMBINING ACUTE ACCENT + te


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        (_DUNIYA, ((0, 6),)),
        (_DECOMPOSED, ((0, 4),)),
        (f"hello {_DUNIYA}!", ((0, 5), (6, 12))),
        # Precomposed and decomposed spellings tokenize alike.
        ("\u00e9te", ((0, 3),)),
        # A mark continues a token and never starts one: with no letter or
        # digit before it, it is a boundary like punctuation.
        (" \u0301x", ((2, 3),)),
        # Stacked marks all belong to the letter they follow.
        ("a\u0301\u0302b", ((0, 4),)),
        # U+FE0F (VARIATION SELECTOR-16) is a mark, and follows most emoji.
        ("I \u2708\ufe0fParis", ((0, 1), (4, 9))),
        ("I\u2764\ufe0fNY", ((0, 1), (3, 5))),
        ("\u2764\ufe0f", ()),
        # A keycap is marks written on its digit.
        ("1\ufe0f\u20e3", ((0, 3),)),
    ],
)
def test_a_combining_mark_stays_in_the_token_it_is_written_on(
    text: str, spans: tuple[tuple[int, int], ...]
) -> None:
    assert token_spans(text) == spans


@pytest.mark.parametrize("mark", ["\u0301", "\u093f", "\u094d", "\u20dd"])
def test_every_kind_of_mark_continues_a_token_and_none_starts_one(mark: str) -> None:
    """Nonspacing (Mn), spacing (Mc) and enclosing (Me) marks alike."""
    assert not mark.isalnum()
    assert in_token(f"a{mark}", 1)
    assert in_token(f"a{mark}{mark}", 2)
    assert not in_token(mark, 0)
    assert not in_token(f" {mark}", 1)
    assert not in_token(f"-{mark}", 1)


def test_in_token_is_alphanumeric_and_nothing_else_starts_one() -> None:
    for text in ("a", "7", "\u00df", "\u6771"):
        assert in_token(text, 0), repr(text)
    for character in (" ", "_", "-", "'", "\u2019", "$", "\u200c", "\u200d"):
        assert not in_token(f"a{character}b", 1), repr(character)


@pytest.mark.parametrize("index", [-1, 2])
def test_in_token_refuses_an_index_outside_the_string(index: int) -> None:
    with pytest.raises(IndexError):
        in_token("ab", index)


def test_a_boundary_is_any_gap_one_token_does_not_cross() -> None:
    text = "x-5 cafe\u0301!"
    boundaries = [i for i in range(len(text) + 1) if is_token_boundary(text, i)]
    # Both ends; around the hyphen; around the space; before "!" -- but not
    # between "e" and its accent, nor inside a word.
    assert boundaries == [0, 1, 2, 3, 4, 9, 10]
    assert is_token_boundary("", 0)


def test_the_boundaries_are_exactly_the_ends_of_the_default_tokens_and_the_gaps_between() -> None:
    """Measured over random text: no gap inside a token, every other gap free."""
    rng = random.Random(1605)
    alphabet = "ab -'\u0301\ufe0f\u2708."
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 16)))
        inside = {i for start, end in token_spans(text) for i in range(start + 1, end)}
        for index in range(len(text) + 1):
            assert is_token_boundary(text, index) == (index not in inside), (text, index)


@pytest.mark.parametrize("index", [-1, 3])
def test_is_token_boundary_refuses_a_gap_outside_the_string(index: int) -> None:
    with pytest.raises(IndexError):
        is_token_boundary("ab", index)


def test_a_mark_with_no_letter_before_it_is_punctuation_at_a_form_s_edge() -> None:
    assert punctuation_edges("\u2764\ufe0f") == (2, 2)
    assert punctuation_edges("(\u0301)") == (3, 3)


def test_a_combining_mark_is_not_punctuation_at_a_form_s_edge() -> None:
    """The edge rule measures the complement of the token predicate, so it moves too."""
    assert punctuation_edges(_DECOMPOSED) == (0, 0)
    assert punctuation_edges(f"({_DUNIYA})") == (1, 1)
    # A span ending on the base letter is not widened over its own accent.
    assert edge_extensions(_DECOMPOSED, (0, 1), None) == ()


def test_narrowest_does_not_trim_a_combining_mark() -> None:
    text = f"({_DECOMPOSED})"
    assert _core(text, (0, len(text))) == (1, 5)

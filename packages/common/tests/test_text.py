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

import pytest

from dataknobs_common.text import (
    edge_extensions,
    probe_spans,
    punctuation_edges,
    token_spans,
)


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

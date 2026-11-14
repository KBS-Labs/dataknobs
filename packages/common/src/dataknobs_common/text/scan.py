# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Folding a surface form for lookup, and locating one inside a string.

The scan half of :mod:`dataknobs_common.text`; its names are re-exported there.

One fold and two spans, and the module exists because of *who* needs them
rather than how much of it there is. An entity source folds the forms it
indexes; a match signal folds the query it is handed, and a **scanning** match
signal also has to say *where* in that query a form sat. Those live in
different subpackages, and while the fold shipped inside ``ontology/sources.py``
the second one could only reach it by importing the first -- an edge from the
resolution family back into ``ontology``, closing a cycle through that
package's ``__init__``. Hoisting it here is what makes the edge one-way.

**A fold is not a boundary policy**, which is why the two spans below are not
derivable from :func:`default_normalizer`. It strips and case-folds; it does
not collapse whitespace and it does not know what a token is. That serves a
whole-string lookup, where the caller has already decided the string is one
form, and it does not serve a scan unchanged: a scan that looked for
``beagle`` by substring alone would find it inside ``beagles`` and inside
``unbeagleable``, and no amount of folding tells it not to.

**The boundary policy has two halves.** :func:`token_spans` says where a span
may start and end: on a token, so never inside a word. That alone never
reaches a form that starts or ends on punctuation -- ``(beagle)``, ``C.D.C.``
-- because no span over the text starts on the parenthesis. The second half
is the **edge rule**: a span may also be extended over the punctuation
directly beside it, by every prefix of that run, as far as the vocabulary says
its own forms' punctuation reaches. :func:`punctuation_edges` measures that
reach on a form, :func:`edge_extensions` applies it to one span, and
:func:`probe_spans` enumerates both in the order a scan publishes. Punctuation
here is a character that is neither alphanumeric nor whitespace, so an
extension never crosses a word and never crosses a space.

Dependency-free on purpose: it imports nothing, so anything may import it.
The package's other half, :mod:`dataknobs_common.text.variants`, imports
:mod:`dataknobs_common.registry` and :mod:`dataknobs_common.exceptions`, which
import only the standard library -- so importing the package still opens no
edge back into the subpackages that import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "content_span",
    "default_normalizer",
    "edge_extensions",
    "probe_spans",
    "punctuation_edges",
    "token_spans",
]


def default_normalizer(form: str) -> str:
    """Fold a surface form for lookup: strip, then case-fold.

    ``casefold`` rather than ``lower`` because it folds more than ASCII --
    a vocabulary in German should match ``STRASSE`` against ``straße``, and
    ``lower`` does not.
    """
    return form.strip().casefold()


def content_span(form: str) -> tuple[int, int]:
    """The half-open extent of ``form`` that :func:`default_normalizer` keeps.

    What a rung matching the **whole** query reports as its span. The fold
    strips, so the characters that took part in the match are the stripped
    ones and the surrounding whitespace is not: a query of ``"  beagle  "``
    matched ``beagle`` at ``(2, 8)``, and reporting ``(0, 10)`` would claim
    two spaces as evidence.

    ``(0, 0)`` for a form that is empty or all whitespace -- there is nothing
    to point at, and it is the one input for which the arithmetic below would
    otherwise produce a start past its own end.

    Not a token boundary and not a claim about one: this says which characters
    survived the fold, which is a different question from :func:`token_spans`
    below and answers it for a string the caller has already decided is one
    form.
    """
    stripped = form.strip()
    if not stripped:
        return (0, 0)
    start = len(form) - len(form.lstrip())
    return (start, start + len(stripped))


def token_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Where ``text``'s tokens are -- **the boundary policy a scan needs**.

    A token is a maximal run of alphanumeric characters, by ``str.isalnum``,
    so the policy is Unicode-aware rather than ASCII-shaped: ``straße`` is one
    token and so is ``ß``. Everything else -- whitespace, punctuation, ``_``,
    a hyphen -- is a boundary.

    **This is what stops a scan matching inside a word.** A form is looked up
    at a *span*, and a span that begins and ends on a token boundary cannot
    start in the middle of ``unbeagleable``. It is also the enumeration an
    n-gram probe runs on: the spans this returns, taken pairwise, are the
    candidate forms in the string -- 21 of them for a six-token utterance --
    and each is looked up by slicing ``text`` between the first token's start
    and the last token's end.

    **The slice, not a join**, and the difference is the limitation to know
    about. ``text[start:end]`` for a span covering two tokens carries whatever
    separated them, so ``golden retriever`` and ``golden_retriever`` both
    reach the index as themselves and match whichever form the vocabulary
    declared. What does *not* match is a form separated differently from the
    way it was declared -- ``golden  retriever``, with two spaces -- because
    :func:`default_normalizer` does not collapse whitespace. Collapsing it
    here would mean handing the index a string that is not in the text, and
    then no offset into the text would be true of it.

    **Every span starts and ends on a token**, so a form whose first or last
    character is punctuation is out of reach of these spans alone.
    :func:`edge_extensions` is the half of the policy that reaches it, and it
    leaves this definition unchanged: punctuation stays a boundary, and an
    extension only reaches over the run beside a span.

    Returns:
        Half-open spans into ``text``, in order, non-overlapping and
        non-touching. Empty for a string with no alphanumeric character in it.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, character in enumerate(text):
        if character.isalnum():
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return tuple(spans)


def _is_edge_character(character: str) -> bool:
    """Neither part of a token nor a separator between words.

    The complement of :func:`token_spans`'s predicate, minus whitespace: what
    an extension may absorb, and so what a form's edge is measured in. One
    predicate for both, because an edge counted any other way would bound the
    wrong extension.
    """
    return not character.isalnum() and not character.isspace()


def _checked_limit(limit: int | None) -> int | None:
    if limit is not None and limit < 0:
        raise ValueError(f"an edge limit counts characters and cannot be negative, got {limit}")
    return limit


def punctuation_edges(form: str) -> tuple[int, int]:
    """How many punctuation characters ``form`` carries at its start and at its end.

    The leading and trailing runs of characters that are neither alphanumeric
    nor whitespace: ``(beagle)`` is ``(1, 1)``, ``C.D.C.`` is ``(0, 1)``, and
    ``K-9`` is ``(0, 0)`` -- interior punctuation is a boundary
    :func:`token_spans` already spans. A run stops at whitespace, so
    ``"( beagle )"`` is ``(1, 1)`` as well.

    What a vocabulary measures over its forms to say how far an extension has
    to reach. **An upper bound, not an exact reach**: ``"( beagle )"`` reports
    one on each side, though no extension can reach it across the spaces, and
    a form of punctuation alone -- ``":-)"`` is ``(3, 3)`` -- is all edge and
    has no token for a span to start on. Over-reporting costs probes; under-
    reporting would cost answers, so the measurement errs the cheap way.
    """
    leading = 0
    for character in form:
        if not _is_edge_character(character):
            break
        leading += 1
    trailing = 0
    for character in reversed(form):
        if not _is_edge_character(character):
            break
        trailing += 1
    return (leading, trailing)


def edge_extensions(
    text: str, span: tuple[int, int], limit: int | None
) -> tuple[tuple[int, int], ...]:
    """``span`` widened over the punctuation beside it -- every way, widest first.

    On each side, every non-empty prefix of the adjacent run of punctuation
    (characters neither alphanumeric nor whitespace), up to ``limit``
    characters, paired with every one on the other side including none. In
    ``a (beagle), here`` the run right of ``beagle`` is ``),``, and only its
    prefix ``)`` reaches the declared form ``(beagle)``: **prefixes, not the
    maximal run**, because the run a form sits in is usually longer than the
    form's own edge.

    Nothing is asked of what lies past a run. An extension stops at a letter,
    a digit, whitespace or the end of the text by construction, so it never
    reaches into another word; the declared form ``C.D.C.`` is reached inside
    ``C.D.C.D`` exactly as ``C.D.C`` already was.

    Args:
        text: The string ``span`` indexes.
        span: A half-open span into ``text``, normally one
            :func:`token_spans` produced or two of them joined.
        limit: The most characters to absorb on each side, or ``None`` for
            the whole adjacent run. ``0`` extends nothing.

    Returns:
        The extended spans, without ``span`` itself, widest first and, at one
        width, the one reaching further left first. Empty when no punctuation
        touches either end, which is what keeps the rule free for a span with
        a space beside it. At most ``(limit + 1) ** 2 - 1`` of them.

    Raises:
        ValueError: For a negative ``limit``.
    """
    limit = _checked_limit(limit)
    start, end = span
    left = 0
    while (
        (limit is None or left < limit)
        and start - left > 0
        and _is_edge_character(text[start - left - 1])
    ):
        left += 1
    right = 0
    while (
        (limit is None or right < limit)
        and end + right < len(text)
        and _is_edge_character(text[end + right])
    ):
        right += 1
    widened = sorted(
        ((before, after) for before in range(left + 1) for after in range(right + 1)),
        key=lambda reach: (-(reach[0] + reach[1]), -reach[0]),
    )
    return tuple((start - before, end + after) for before, after in widened if before or after)


def probe_spans(
    text: str, longest: int | None, *, edge: int | None = 0
) -> Iterator[tuple[int, int]]:
    """Every span a scan looks up: token windows, longest first, each with its extensions.

    A window is up to ``longest`` consecutive tokens of ``text``, sliced from
    the first token's start to the last token's end; ``None`` enumerates every
    window. Each window is preceded by its :func:`edge_extensions` under
    ``edge``, so a form containing another is proposed before the form it
    contains -- the principle that orders windows longest first, applied to
    one window and the punctuation around it.

    **Both bounds are the vocabulary's, and neither costs an answer.** A window
    wider than the longest form a vocabulary declares, or an extension wider
    than the widest punctuated edge it declares, is one no declared form
    could fill. ``edge=0`` -- a vocabulary none of whose forms has a
    punctuated edge -- is exactly the enumeration over tokens alone, *n x
    longest* spans for *n* tokens; each window touched by punctuation adds at
    most ``(edge + 1) ** 2 - 1``. ``None`` means the vocabulary cannot say
    and extends over the whole adjacent run: the same answers, at a cost that
    grows with the **square** of that run -- 90,600 extensions for one token
    between two 300-character rules. A caller scanning text it did not write
    passes a number rather than ``None`` where it can.

    Flavour-free on purpose: it looks nothing up, so a synchronous caller, an
    asynchronous one and any other scan over a vocabulary each drive it with
    their own lookup and cannot drift on what was enumerated.

    Raises:
        ValueError: For a negative ``edge``.
    """
    edge = _checked_limit(edge)
    tokens = token_spans(text)
    widest = len(tokens) if longest is None else min(longest, len(tokens))
    for length in range(widest, 0, -1):
        for first in range(len(tokens) - length + 1):
            window = (tokens[first][0], tokens[first + length - 1][1])
            if edge != 0:
                yield from edge_extensions(text, window, edge)
            yield window

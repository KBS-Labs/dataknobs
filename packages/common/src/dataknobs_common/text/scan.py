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
here is a character that is neither a token character nor whitespace, so an
extension never crosses a word and never crosses a space.

**A token character is alphanumeric or a combining mark**, by
:func:`is_token_character`, and every rule in this module asks that one
predicate. ``str.isalnum`` alone is False for every combining mark, so on its
own it split *दुनिया* at each vowel sign and kept the signs in no token.

**The token half is the caller's to replace.** :func:`probe_spans` takes the
tokens it enumerates over, so a caller whose units are not runs of
alphanumerics -- a currency sign, a grouped number, a script written without
spaces -- drives the same enumeration over their own. :func:`narrowest` is the
rule that keeps one piece of evidence per window once extensions or a fold
make several spans reach the same thing, and it is exact over any tokens.

Dependency-free on purpose: it imports only the standard library, so anything
may import it.
The package's other half, :mod:`dataknobs_common.text.variants`, imports
:mod:`dataknobs_common.registry` and :mod:`dataknobs_common.exceptions`, which
import only the standard library -- so importing the package still opens no
edge back into the subpackages that import it.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence

__all__ = [
    "content_span",
    "default_normalizer",
    "ApostropheMode",
    "edge_extensions",
    "is_token_character",
    "narrowest",
    "probe_spans",
    "punctuation_edges",
    "token_spans",
    "word_spans",
]

_Hit = TypeVar("_Hit")


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


def is_token_character(character: str) -> bool:
    """Whether ``character`` belongs to a token: alphanumeric, or a combining mark.

    Alphanumeric by ``str.isalnum``, which is Unicode-aware rather than
    ASCII-shaped, so ``ß`` and ``東`` qualify. A **combining mark** -- any
    character of Unicode category ``Mn``, ``Mc`` or ``Me``, such as the vowel
    signs of *दुनिया* or the U+0301 of a decomposed *é* -- is part of the word
    it is written on, and ``str.isalnum`` is False for every one of them. A
    mark counts wherever it stands, including with no letter before it, so no
    character of the text silently falls out of every token.

    Everything else is not: whitespace, punctuation, ``_``, a hyphen, an
    apostrophe, a joiner.

    Args:
        character: One character.

    Returns:
        ``True`` for a token character.
    """
    return character.isalnum() or unicodedata.category(character)[0] == "M"


def token_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Where ``text``'s tokens are -- **the boundary policy a scan needs**.

    A token is a maximal run of token characters, by
    :func:`is_token_character` -- alphanumerics and combining marks -- so the
    policy is Unicode-aware rather than ASCII-shaped: ``straße`` is one token,
    so is ``ß``, and so is *दुनिया* with its vowel signs. Everything else --
    whitespace, punctuation, ``_``, a hyphen, an apostrophe -- is a boundary.

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
        non-touching. Empty for a string with no token character in it.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, character in enumerate(text):
        if is_token_character(character):
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return tuple(spans)


#: How :func:`word_spans` treats an apostrophe between two token characters.
ApostropheMode: TypeAlias = Literal["keep", "clitic"]

_APOSTROPHE_MODES: tuple[ApostropheMode, ...] = ("keep", "clitic")

#: U+0027 and U+2019, the two characters text writes an apostrophe with.
#: U+02BC (MODIFIER LETTER APOSTROPHE) is a letter, so it is already a token
#: character and needs no joining.
_APOSTROPHES = frozenset("\u0027\u2019")


def word_spans(text: str, apostrophes: ApostropheMode = "keep") -> tuple[tuple[int, int], ...]:
    """:func:`token_spans`, with an apostrophe inside a word kept.

    The default tokens make every apostrophe a boundary, so *don't* is
    ``don`` and ``t`` -- right for a scan, which looks up the slice between
    tokens and so still reaches a declared *don't*, and wrong for anything
    that reads a token as a word, such as a list of function words that holds
    *don't*. These are the same tokens with the tokens an **in-word
    apostrophe** separates joined: one U+0027 or U+2019 with a token
    character directly on each side. Every other boundary is
    :func:`token_spans`' own, so the two never disagree about what a token
    character is.

    - ``"keep"`` (the default): *don't*, *what's* (with either apostrophe)
      and *rock'n'roll* are one token each.
    - ``"clitic"``: a token starts at each in-word apostrophe and carries it
      -- ``don`` ``'t``, ``John`` ``'s``, ``rock`` ``'n`` ``'roll``. Every
      in-word apostrophe splits, not a list of English clitics, so the rule
      holds in any language that writes one.

    In both modes an apostrophe with no token character after it -- the
    plural possessive *dogs'*, a closing quote -- and a doubled one stay
    boundaries and fall in no token, as every other punctuation character
    does. Telling a possessive from a closing quote needs quote balance,
    which no character window has.

    Tokens are slices of ``text``: U+2019 is not folded to U+0027 here. A
    lookup that wants the two spellings to meet folds the token it reads.

    Args:
        text: The string to tokenize.
        apostrophes: ``"keep"`` or ``"clitic"``.

    Returns:
        Half-open spans into ``text``, in order and non-overlapping. Under
        ``"clitic"`` a token carrying an apostrophe touches the one before it.

    Raises:
        ValueError: For an ``apostrophes`` value other than the two above.
    """
    if apostrophes not in _APOSTROPHE_MODES:
        raise ValueError(
            f"apostrophes must be one of {list(_APOSTROPHE_MODES)}; given {apostrophes!r}"
        )
    words: list[tuple[int, int]] = []
    for start, end in token_spans(text):
        joined = bool(words) and start - words[-1][1] == 1 and text[start - 1] in _APOSTROPHES
        if not joined:
            words.append((start, end))
        elif apostrophes == "keep":
            words[-1] = (words[-1][0], end)
        else:
            words.append((start - 1, end))
    return tuple(words)


def _is_edge_character(character: str) -> bool:
    """Neither part of a token nor a separator between words.

    The complement of :func:`token_spans`'s predicate, minus whitespace: what
    an extension may absorb, and so what a form's edge is measured in. One
    predicate for both, because an edge counted any other way would bound the
    wrong extension.
    """
    return not is_token_character(character) and not character.isspace()


def _checked_limit(limit: int | None) -> int | None:
    if limit is not None and limit < 0:
        raise ValueError(f"an edge limit counts characters and cannot be negative, got {limit}")
    return limit


def punctuation_edges(form: str) -> tuple[int, int]:
    """How many punctuation characters ``form`` carries at its start and at its end.

    The leading and trailing runs of characters that are neither token characters
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
    (characters neither token characters nor whitespace), up to ``limit``
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
        span: A half-open span into ``text``: one token, or a run of them
            joined, whether the tokens are :func:`token_spans`'s or the
            caller's own.
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


def _checked_tokens(text: str, tokens: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """``tokens`` as a tuple, having refused one no window could be built on.

    A window is sliced from its first token's start to its last token's end,
    so the enumeration is only true of ``text`` if every token is a non-empty
    span inside it and each one starts where the previous one has ended or
    later. Spans that **touch** are accepted, because a caller's units may
    abut: ``$`` and ``5`` in ``$5``. Spans that overlap or arrive out of order
    are refused rather than sorted, because a tokenizer producing them has a
    bug a sort would hide.

    Raises:
        ValueError: For a span that is not a pair of ints, an empty one, one
            outside ``text``, or one starting before the previous one ends.
            The message names the span's index.
    """
    checked: list[tuple[int, int]] = []
    previous_end = 0
    for index, token in enumerate(tokens):
        if (
            not isinstance(token, tuple | list)
            or len(token) != 2
            or not all(isinstance(bound, int) and not isinstance(bound, bool) for bound in token)
        ):
            raise ValueError(f"token {index} must be a (start, end) pair of ints, got {token!r}")
        start, end = token
        if start >= end:
            raise ValueError(
                f"token {index} is {token!r}, which is empty; a token is a "
                f"half-open span with start < end"
            )
        if start < 0 or end > len(text):
            raise ValueError(f"token {index} is {token!r}, outside a text of length {len(text)}")
        if start < previous_end:
            raise ValueError(
                f"token {index} is {token!r} and starts before token {index - 1} "
                f"ends at {previous_end}; tokens must be in order and must not "
                f"overlap"
            )
        checked.append((start, end))
        previous_end = end
    return tuple(checked)


def probe_spans(
    text: str,
    longest: int | None,
    *,
    edge: int | None = 0,
    tokens: Sequence[tuple[int, int]] | None = None,
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

    **The tokens may be the caller's.** ``tokens=None`` enumerates over
    :func:`token_spans`, which has no token for a character that is neither a
    letter nor a digit -- so a form like ``$`` is never a window, and a
    grouped number such as ``1,234.50`` is three tokens rather than one. A
    caller whose units differ passes their own spans and gets the same
    enumeration over them. Two things change with that:

    - ``longest`` counts **the caller's** tokens. A vocabulary's
      ``longest_form_tokens()`` and ``form_edge`` are measured in
      :func:`token_spans` terms, so they bound only a scan over the default
      tokens; a caller passing their own derives their own bounds or
      passes ``None``.
    - Tokens that already include punctuation need no extension over it, and
      such a caller passes ``edge=0``. Nothing checks this: an extension over
      tokens that cover the punctuation finds no run to extend over and costs
      nothing, but one that partly covers it can produce spans no token
      boundary justifies.

    Pair it with :func:`narrowest` wherever extensions or a fold can make
    several spans reach the same thing.

    Args:
        text: The string to enumerate spans of.
        longest: The most tokens a window may span, or ``None`` for every
            window.
        edge: The most punctuation characters an extension absorbs on each
            side, ``0`` for none, or ``None`` for the whole adjacent run.
        tokens: Half-open spans into ``text``, in order and not overlapping
            (touching is allowed), or ``None`` for :func:`token_spans`.

    Raises:
        ValueError: For a negative ``edge``, or for ``tokens`` that are not
            in-order, non-overlapping, non-empty spans inside ``text``.
    """
    edge = _checked_limit(edge)
    units = token_spans(text) if tokens is None else _checked_tokens(text, tokens)
    widest = len(units) if longest is None else min(longest, len(units))
    for length in range(widest, 0, -1):
        for first in range(len(units) - length + 1):
            window = (units[first][0], units[first + length - 1][1])
            if edge != 0:
                yield from edge_extensions(text, window, edge)
            yield window


def _core(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """``span`` with every character that is not a token character trimmed off both ends.

    Whitespace goes with the punctuation, so ``", TX"`` and ``"TX"`` share a
    core. A span with no token character in it has the empty core at
    its start, which keeps it distinct from a neighbour that has one and
    never produces a start past its own end.
    """
    start, end = span
    while start < end and not is_token_character(text[start]):
        start += 1
    if start == end:
        return (span[0], span[0])
    while not is_token_character(text[end - 1]):
        end -= 1
    return (start, end)


def narrowest(
    text: str,
    hits: Iterable[_Hit],
    *,
    span: Callable[[_Hit], tuple[int, int]],
    key: Callable[[_Hit], Hashable],
) -> list[_Hit]:
    """One hit per key per window: the narrowest span that reached it.

    **Why a scan needs it.** An extension widens a window over the
    punctuation beside it, and a fold that strips punctuation then makes every
    extension of ``beagle`` reach the key the bare window reaches -- in
    ``"(beagle)"`` one entity would be reported at four nested places. The
    punctuation took no part in that match, so the evidence is the narrowest
    span around it. Where the punctuation *is* the match -- ``(beagle)``
    declared -- the bare window misses and the extended span is the only one
    there is, so it is kept.

    **The core.** Hits are grouped by the span left after trimming every
    character that is not a letter or a digit off both ends, whitespace
    included -- so over a caller's tokens that include ``,`` and a space,
    ``TX``, ``, TX`` and ``, TX,`` in ``Austin, TX, USA`` share one core and
    one is kept, at ``TX``. Over :func:`token_spans` and :func:`probe_spans`'s
    extensions the trimming is exactly the extension undone, because a window
    starts and ends on a token and an extension adds only punctuation.

    **A hit with no letter or digit in it** -- ``$`` reached over a caller's
    tokens -- has the empty core at its own start, so it stays apart from a
    ``$5`` beside it that reached the same key.

    **Order.** One hit is kept per ``(core, key(hit))``: the one with the
    smallest ``end - start``, and the first on a tie. Each kept hit sits
    where the first hit for its pair arrived, so the result is in the order
    the hits were given -- a scan's enumeration order survives.

    ``ScanningSignal`` and its asynchronous twin call it over the
    ``FormHit`` objects they collect; a second scanner over :func:`probe_spans` calls it over
    its own hits, which need only expose a span and a key::

        >>> text = "a (beagle) here"
        >>> hits = [((2, 10), "beagle"), ((2, 9), "beagle"), ((3, 9), "beagle")]
        >>> narrowest(text, hits, span=lambda h: h[0], key=lambda h: h[1])
        [((3, 9), 'beagle')]

    Args:
        text: The string the spans index.
        hits: The hits, in the order to preserve.
        span: Reads a hit's half-open span into ``text``.
        key: Reads what a hit reached -- an entity id, say. Hits at one core
            with different keys are all kept.

    Returns:
        The kept hits, in first-arrival order.
    """
    kept: list[_Hit] = []
    at: dict[tuple[tuple[int, int], Hashable], int] = {}
    for hit in hits:
        start, end = span(hit)
        group = (_core(text, (start, end)), key(hit))
        index = at.get(group)
        if index is None:
            at[group] = len(kept)
            kept.append(hit)
        else:
            held_start, held_end = span(kept[index])
            if end - start < held_end - held_start:
                kept[index] = hit
    return kept

"""Folding a surface form for lookup, and locating one inside a string.

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

Dependency-free on purpose: it imports nothing, so anything may import it.
"""

from __future__ import annotations

__all__ = ["content_span", "default_normalizer", "token_spans"]


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

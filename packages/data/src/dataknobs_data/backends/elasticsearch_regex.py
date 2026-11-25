# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A Python regular expression as the Lucene one that finds the same values.

``Operator.REGEX`` is an unanchored :func:`re.search` everywhere else. Lucene
matches a whole value, has no anchors, and gives ``@ # ~ & < "`` meanings of
their own. So the pattern is read by Python's parser and written out again:
what it leaves unanchored is padded with ``.*``, and every character is escaped.

Lucene's answer is whether the value belongs to a set of strings, with nothing
said of where in it. That leaves out look-around, word boundaries, back-references,
atomic groups and possessive repeats, and an anchor anywhere but an end of the
pattern. Inline flags are refused too. A pattern using one raises ``ValueError``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from enum import Enum
from re import _constants as sre  # type: ignore[attr-defined]  # not in typeshed
from re import _parser  # type: ignore[attr-defined]  # not in typeshed
from typing import Any

#: One parsed element: its opcode and argument.
_Node = tuple[Any, Any]
#: Inclusive ranges of code points.
_Ranges = list[tuple[int, int]]

_LAST = 0x10FFFF
_NEWLINE = (10, 10)
#: The empty string.
_EMPTY = "()"
#: ASCII for ``\d`` and ``\w``; ``\s`` is all of Python's.
_CATEGORIES: dict[Any, _Ranges] = {
    sre.CATEGORY_DIGIT: [(48, 57)],
    sre.CATEGORY_WORD: [(48, 57), (65, 90), (95, 95), (97, 122)],
    sre.CATEGORY_SPACE: [
        (0x09, 0x0D),
        (0x1C, 0x20),
        (0x85, 0x85),
        (0xA0, 0xA0),
        (0x1680, 0x1680),
        (0x2000, 0x200A),
        (0x2028, 0x2029),
        (0x202F, 0x202F),
        (0x205F, 0x205F),
        (0x3000, 0x3000),
    ],
}
_NEGATED_CATEGORIES = {
    sre.CATEGORY_NOT_DIGIT: sre.CATEGORY_DIGIT,
    sre.CATEGORY_NOT_WORD: sre.CATEGORY_WORD,
    sre.CATEGORY_NOT_SPACE: sre.CATEGORY_SPACE,
}
_NO_EQUIVALENT = {
    sre.ASSERT: "look-around",
    sre.ASSERT_NOT: "look-around",
    sre.GROUPREF: "a back-reference",
    sre.GROUPREF_EXISTS: "a conditional group",
    sre.ATOMIC_GROUP: "an atomic group",
    sre.POSSESSIVE_REPEAT: "a possessive repeat",
}
_BOUNDARIES = frozenset({sre.AT_BOUNDARY, sre.AT_NON_BOUNDARY})


class _Edge(Enum):
    """What lies beyond one side of an element, and is written there."""

    INNER = None  # more of the pattern, so no anchor can stand here
    OPEN = ".*"  # any text
    LINE_END = "\n?"  # the value's end, or a newline that ends it
    CLOSED = ""  # the value's own start or end


_STARTS = frozenset({sre.AT_BEGINNING, sre.AT_BEGINNING_STRING})
_ENDS = {sre.AT_END: _Edge.LINE_END, sre.AT_END_STRING: _Edge.CLOSED}


class _UnmatchableError(Exception):
    """What in a pattern Lucene has no equivalent of."""


def python_regex_to_lucene(pattern: str) -> str:
    r"""The Lucene expression matching the values :func:`re.search` finds ``pattern`` in.

    For a ``regexp`` query whose ``flags`` are ``NONE``. ``\d`` and ``\w`` are
    ASCII.

    Raises:
        ValueError: ``pattern`` is not a string, is not a regular expression,
            or uses what Lucene cannot match.
    """
    if not isinstance(pattern, str):
        raise ValueError(f"REGEX pattern must be a string, got: {pattern!r}")
    try:
        parsed = _parser.parse(pattern)
        if parsed.state.flags != int(re.UNICODE):
            raise _UnmatchableError("inline flags")
        return _sequence(parsed, _Edge.OPEN, _Edge.OPEN)
    except (re.error, OverflowError, _UnmatchableError) as e:
        raise ValueError(f"Elasticsearch cannot match the pattern {pattern!r}: {e}") from e


def _sequence(nodes: Iterable[_Node], left: _Edge, right: _Edge) -> str:
    """Elements one after another, with what lies beyond each side."""
    items = list(nodes)
    while items and left is not _Edge.INNER and _anchor(items[0]) in _STARTS:
        left = _Edge.CLOSED
        del items[0]
    while items and right is not _Edge.INNER and _anchor(items[-1]) in _ENDS:
        if right is not _Edge.CLOSED:
            right = _ENDS[_anchor(items[-1])]
        del items[-1]

    # An anchor inside the element at an edge takes that edge for itself.
    last = len(items) - 1
    hand_left = bool(items) and left is not _Edge.INNER and _anchors(items[0])
    hand_right = bool(items) and right is not _Edge.INNER and _anchors(items[-1])
    written = "".join(
        _element(
            node,
            left if hand_left and i == 0 else _Edge.INNER,
            right if hand_right and i == last else _Edge.INNER,
        )
        for i, node in enumerate(items)
    )
    before = "" if hand_left else left.value or ""
    after = "" if hand_right else right.value or ""
    return f"{before}{written}{after}" or _EMPTY


def _anchor(node: _Node) -> Any:
    """Which anchor an element is; ``None`` for one that is not."""
    return node[1] if node[0] is sre.AT else None


def _anchors(node: _Node) -> bool:
    """Whether an element holds an anchor."""
    op, arg = node
    if op is sre.AT:
        return True
    if op is sre.SUBPATTERN:
        return any(_anchors(n) for n in arg[3])
    if op is sre.BRANCH:
        return any(_anchors(n) for branch in arg[1] for n in branch)
    if op in (sre.MAX_REPEAT, sre.MIN_REPEAT):
        return any(_anchors(n) for n in arg[2])
    return False


def _element(node: _Node, left: _Edge, right: _Edge) -> str:
    op, arg = node
    if op in (sre.LITERAL, sre.NOT_LITERAL, sre.ANY, sre.IN):
        return _character(op, arg)
    if op is sre.SUBPATTERN:
        _, added, removed, nodes = arg
        if added or removed:
            raise _UnmatchableError("inline flags")
        return f"({_sequence(nodes, left, right)})"
    if op is sre.BRANCH:
        return _alternatives(arg[1], left, right)
    # Lazy or not decides where a match ends, not whether there is one.
    if op in (sre.MAX_REPEAT, sre.MIN_REPEAT):
        low, high, nodes = arg
        return _repeat(low, high, nodes, left, right)
    if op is sre.AT:
        if arg in _BOUNDARIES:
            raise _UnmatchableError("a word boundary")
        raise _UnmatchableError("an anchor that is not at an end of the pattern")
    raise _UnmatchableError(_NO_EQUIVALENT.get(op, f"{op}"))


def _alternatives(branches: Iterable[Iterable[_Node]], left: _Edge, right: _Edge) -> str:
    written = [_sequence(branch, left, right) for branch in branches]
    # An empty alternative makes a ``wildcard`` field miss the others' values.
    some = [w for w in written if w != _EMPTY]
    if not some:
        return _EMPTY
    return f"({'|'.join(some)}){'?' if len(some) < len(written) else ''}"


def _repeat(low: int, high: int, nodes: Sequence[_Node], left: _Edge, right: _Edge) -> str:
    at_edge = left is not _Edge.INNER or right is not _Edge.INNER
    if at_edge and (low, high) == (0, 1):
        return _alternatives([nodes, []], left, right)
    if at_edge and (low, high) == (1, 1):
        return f"({_sequence(nodes, left, right)})"
    body = _sequence(nodes, _Edge.INNER, _Edge.INNER)
    if len(nodes) != 1 or nodes[0][0] not in (sre.LITERAL, sre.NOT_LITERAL, sre.ANY, sre.IN):
        body = f"({body})"
    if high is sre.MAXREPEAT:
        return body + {0: "*", 1: "+"}.get(low, f"{{{low},}}")
    if (low, high) == (0, 1):
        return f"{body}?"
    return f"{body}{{{low},{high}}}"


def _character(op: Any, arg: Any) -> str:
    """An element that matches one character."""
    if op is sre.LITERAL:
        return _escape(arg)
    if op is sre.NOT_LITERAL:
        ranges = _complement([(arg, arg)])
    elif op is sre.ANY:
        ranges = _complement([_NEWLINE])
    else:
        ranges = _class_ranges(arg)
    if not ranges:
        raise _UnmatchableError("a class no character is in")
    # Written from whichever side stops short of the last code point.
    negated = ranges[-1][1] == _LAST
    if negated:
        ranges = _complement(ranges)
        if not ranges:
            return "."
    members = "".join(
        _escape(lo) if lo == hi else f"{_escape(lo)}-{_escape(hi)}" for lo, hi in ranges
    )
    return f"[{'^' if negated else ''}{members}]"


def _class_ranges(members: Sequence[_Node]) -> _Ranges:
    ranges: _Ranges = []
    negated = False
    for op, arg in members:
        if op is sre.NEGATE:
            negated = True
        elif op is sre.LITERAL:
            ranges.append((arg, arg))
        elif op is sre.RANGE:
            ranges.append(arg)
        elif arg in _NEGATED_CATEGORIES:
            ranges += _complement(_CATEGORIES[_NEGATED_CATEGORIES[arg]])
        else:
            ranges += _CATEGORIES[arg]
    outside = _complement(ranges)
    return outside if negated else _complement(outside)


def _complement(ranges: _Ranges) -> _Ranges:
    """Every code point outside ``ranges``, in order and merged."""
    gaps: _Ranges = []
    start = 0
    for lo, hi in sorted(ranges):
        if lo > start:
            gaps.append((start, lo - 1))
        start = max(start, hi + 1)
    if start <= _LAST:
        gaps.append((start, _LAST))
    return gaps


def _escape(code: int) -> str:
    """A character as itself: Lucene gives punctuation, or an escaped letter, a meaning."""
    char = chr(code)
    special = char.isascii() and char.isprintable() and not char.isalnum()
    return f"\\{char}" if special else char

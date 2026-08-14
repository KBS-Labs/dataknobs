"""Guard: every declined ruff rule carries a category, and every unargued one a count.

The ``ignore`` list declines 83 rules standing in front of roughly 49,000
findings — twenty times the entire declared backlog. Nothing asked whether any
of those declines still held, and the one artifact that tried is prose: the
"Error Categories and Decisions" section of the linting page enumerates
decisions by hand, omits 48 of the 83, and documents a rule as declined that is
enforced. A classification nothing compares against the config is a
classification that drifts from it, and this one had.

So the classification moved to where the rules live, and this file is what
compares them. Each entry carries a ``[category]`` marker on its own comment:

* ``presentational`` — no finding of this rule can be a behaviour difference.
  Nothing further is required, because there is nothing to argue.
* ``covered-elsewhere`` — another tool enforces the property. The reason has to
  name which, so "covered" is checkable rather than asserted.
* ``behavioural`` — findings can be real, and the decline is argued. The
  argument is the continuation comment under the entry, and at least one line
  of it is required.
* ``provisional`` — findings can be real and the decline is **not** argued. The
  reason carries the measured count.

The fourth category is the one that earns its place. Three would force an
unargued decline to be written as though an argument existed, leaving it
indistinguishable from a real one without reading prose — which is the state
described above. Named, they are countable: the sum of the ``provisional``
counts is a number whose target is zero, and it is the only figure here that is
supposed to move.

**The parse is pinned.** ``parse_declines`` re-implements a read the toolchain
also performs, which is a liability rather than a convenience, so the check
below asserts that it and ``tomllib`` return the same code set in both
directions. Without that pin an entry the regex failed to match would be an
uncategorized rule slipping silently past the check written to catch
uncategorized rules — the guard would report green over exactly the shape it
exists for.

The checks are written as functions over parsed records rather than as
assertions over the real config, so that the last test in this file can drive
each of them over synthetic input and prove it fails on the shape it names. A
guard that has never been seen to fail is a guard whose passing means nothing.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable
from typing import Any

import pytest

from tests._workspace import ROOT, load_bin_module

#: A check: parsed declines in, human-readable faults out.
#:
#: Typed as ``list[Any]`` rather than ``list[Decline]`` because the record
#: classes come from a module loaded by path — ``bin/`` names are hyphenated,
#: so they cannot be imported normally — and a name bound from such a module is
#: a value to the type checker, not a type. The sibling contract guards resolve
#: it the same way.
Check = Callable[[list[Any]], list[str]]

_contract = load_bin_module("quality-contract")

DECLINE_CATEGORIES = _contract.DECLINE_CATEGORIES
DECLINE_COVERS = _contract.DECLINE_COVERS
Decline = _contract.Decline
PerFileWaiver = _contract.PerFileWaiver
parse_declines = _contract.parse_declines
parse_per_file_waivers = _contract.parse_per_file_waivers

CONFIG = ROOT / "pyproject.toml"

#: A ruff rule code, which a ``covered-elsewhere`` reason may name instead of a
#: tool: ``D211`` is a real answer to why ``D203`` is declined, and listing every
#: code here would be a second copy of the rule list.
_CODE_RE = re.compile(r"\b[A-Z]{1,6}\d{3,4}\b")

#: A measured count: a bare numeral, word-bounded so that the digits *inside* a
#: rule code do not pass for one. The distinction is not pedantry — it was
#: measured. A first draft matched any digit, and a ``provisional`` entry whose
#: reason had lost its figure still passed, on the ``12`` in ``RUF012``. The
#: contract module's ``_MEASUREMENT_IN_PROSE`` draws the same boundary for the
#: same reason, in the opposite direction.
_COUNT_RE = re.compile(r"\b\d+\b")

#: Floors under the scan. Real counts when written: 83 declines, 31 waivers. Set
#: below both so ordinary movement does not touch them, and far enough above
#: zero that a parse resolving to nothing fails instead of passing — an empty
#: scan and a fully categorized config otherwise produce the same report.
MINIMUM_DECLINES = 40
MINIMUM_WAIVERS = 15

#: Floors under the inline scan, whose real counts when written were 77 and 458.
#: Same purpose as the two above: a scan that resolved to nothing would satisfy
#: the totality check below, because zero equals zero.
MINIMUM_INLINE = {"ruff": 30, "mypy": 200}


def _config_text() -> str:
    return CONFIG.read_text(encoding="utf-8")


def uncategorized(declines: list[Any]) -> list[str]:
    """Entries carrying no category, or one outside the vocabulary."""
    return [
        f"{d.code}: {'no [category] marker' if d.category is None else f'unknown category [{d.category}]'}"
        for d in declines
        if d.category not in DECLINE_CATEGORIES
    ]


def unargued_behavioural(declines: list[Any]) -> list[str]:
    """``behavioural`` entries with no argument under them.

    The category's whole claim is that findings can be real and the decline was
    made anyway. Without the argument that is not a category, it is an assertion.
    """
    return [
        f"{d.code}: declared [behavioural] with no argument on the lines below it"
        for d in declines
        if d.category == "behavioural" and not d.argument
    ]


def uncovered_cover(declines: list[Any]) -> list[str]:
    """``covered-elsewhere`` entries naming no cover."""
    faults = []
    for d in declines:
        if d.category != "covered-elsewhere":
            continue
        if not any(cover in d.prose for cover in DECLINE_COVERS) and not _CODE_RE.search(d.prose):
            faults.append(
                f"{d.code}: declared [covered-elsewhere] but the reason names no cover — "
                f"say which tool or rule enforces the property"
            )
    return faults


def uncounted_provisional(declines: list[Any]) -> list[str]:
    """``provisional`` entries carrying no measured count.

    The count is the whole point of the category: it makes an unargued decline
    something a report can add up and a reader can watch fall.
    """
    return [
        f"{d.code}: declared [provisional] with no measured finding count in its reason"
        for d in declines
        if d.category == "provisional" and not _COUNT_RE.search(d.prose)
    ]


def reasonless_waivers(waivers: list[Any]) -> list[str]:
    """Per-file waivers with no trailing comment.

    A per-file entry also unflags every *future* finding of that code in that
    file. That price is payable, and it is paid per file with the file read
    first — but only a written reason shows it ever was.
    """
    return [
        f"{w.pattern}: waives {', '.join(w.codes)} with no reason on the line"
        for w in waivers
        if not w.reason
    ]


CHECKS = (uncategorized, unargued_behavioural, uncovered_cover, uncounted_provisional)


def test_the_parse_agrees_with_the_toml_reader() -> None:
    """The pin. A code this file cannot see is a code it cannot check."""
    text = _config_text()
    parsed = {d.code for d in parse_declines(text)}
    declared = set(tomllib.loads(text)["tool"]["ruff"]["lint"]["ignore"])

    assert len(parsed) >= MINIMUM_DECLINES, (
        f"the line parse found only {len(parsed)} declines, below the floor of "
        f"{MINIMUM_DECLINES} — the block's shape has changed and this parse no "
        "longer reads it"
    )
    assert parsed == declared, (
        "the line parse and tomllib disagree about which rules are declined.\n"
        f"  parsed but not declared: {sorted(parsed - declared)}\n"
        f"  declared but not parsed: {sorted(declared - parsed)}\n"
        "The second list is the dangerous one: those entries are invisible to "
        "every check in this file."
    )


def test_every_decline_carries_a_category() -> None:
    faults = uncategorized(parse_declines(_config_text()))
    assert not faults, "\n".join(
        [
            f"{len(faults)} declined rule(s) carry no usable category. Add a marker "
            f"from {sorted(DECLINE_CATEGORIES)} to the entry's own comment:",
            *faults,
        ]
    )


def test_every_behavioural_decline_carries_its_argument() -> None:
    faults = unargued_behavioural(parse_declines(_config_text()))
    assert not faults, "\n".join(
        [
            "[behavioural] claims findings can be real and the decline was made "
            "anyway. Write the reason as continuation comment lines under the entry:",
            *faults,
        ]
    )


def test_every_covered_decline_names_its_cover() -> None:
    faults = uncovered_cover(parse_declines(_config_text()))
    assert not faults, "\n".join(
        [
            f"[covered-elsewhere] has to say what the cover is — {' or '.join(DECLINE_COVERS)}, "
            "or the rule code that enforces the same property:",
            *faults,
        ]
    )


def test_every_provisional_decline_carries_its_count() -> None:
    faults = uncounted_provisional(parse_declines(_config_text()))
    assert not faults, "\n".join(
        [
            "[provisional] is the unargued category, and its count is what makes "
            "the backlog of unargued declines a number rather than a feeling:",
            *faults,
        ]
    )


def test_every_per_file_waiver_carries_a_reason() -> None:
    waivers = parse_per_file_waivers(_config_text())
    assert len(waivers) >= MINIMUM_WAIVERS, (
        f"found only {len(waivers)} per-file waivers, below the floor of "
        f"{MINIMUM_WAIVERS} — the block's shape has changed and this parse no "
        "longer reads it"
    )
    faults = reasonless_waivers(waivers)
    assert not faults, "\n".join(
        ["a per-file waiver unflags future findings too, so it states its reason:", *faults]
    )


def test_the_unargued_declines_are_countable() -> None:
    """The category's product: a number, not a posture.

    Asserted as *reportable* rather than as a target, because the target is zero
    and the whole point of naming these is that they are still above it. What
    would be a defect is the count becoming unobtainable — a provisional entry
    whose reason lost its figure, which the check above forbids, or the category
    quietly disappearing from the vocabulary while its members stayed declined.
    """
    provisional = [d for d in parse_declines(_config_text()) if d.category == "provisional"]
    assert "provisional" in DECLINE_CATEGORIES
    for entry in provisional:
        assert _COUNT_RE.search(entry.prose), entry.code


def test_the_checks_detect_the_shapes_they_exist_for() -> None:
    """Each check, driven over input built to break it.

    Every assertion above passes today, so nothing here has been seen to fail on
    a real config. That makes them indistinguishable from checks that cannot
    fail — the failure mode the parse pin exists for, one level up.
    """
    good = Decline("X100", "presentational", "a rendering property", ())
    assert not uncategorized([good])
    assert not unargued_behavioural([good])
    assert not uncovered_cover([good])
    assert not uncounted_provisional([good])

    assert uncategorized([Decline("X101", None, "no marker at all", ())])
    assert uncategorized([Decline("X102", "invented", "not in the vocabulary", ())])
    assert unargued_behavioural([Decline("X103", "behavioural", "asserted, not argued", ())])
    assert not unargued_behavioural(
        [Decline("X104", "behavioural", "argued", ("because of this",))]
    )
    assert uncovered_cover([Decline("X105", "covered-elsewhere", "something else does it", ())])
    assert not uncovered_cover([Decline("X106", "covered-elsewhere", "mypy does it", ())])
    assert not uncovered_cover([Decline("X107", "covered-elsewhere", "D211 does it", ())])
    assert uncounted_provisional([Decline("X108", "provisional", "not yet read", ())])
    assert uncounted_provisional(
        [Decline("X110", "provisional", "the same shape as RUF012, unread", ())]
    ), "a digit inside a rule code is not a measured count"
    assert not uncounted_provisional([Decline("X109", "provisional", "12 findings, unread", ())])
    assert reasonless_waivers([PerFileWaiver("a/b.py", ("F401",), "")])
    assert not reasonless_waivers([PerFileWaiver("a/b.py", ("F401",), "the import is the test")])


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_each_check_reads_something(check: Check) -> None:
    """No check may pass by looking at an empty list.

    A parse that silently returned nothing would satisfy every assertion in this
    file. The floor in the pin covers that for the real config; this covers the
    checks themselves, by proving each one is reached by real records rather than
    short-circuiting on an empty population.
    """
    declines = parse_declines(_config_text())
    assert declines, "the decline parse returned nothing"
    assert check(declines) == [], check.__name__


def _tracked_sources() -> list[tuple[str, str]]:
    return [
        (name, (ROOT / name).read_text(encoding="utf-8"))
        for name in _contract.tracked_python()
        if (ROOT / str(name)).is_file()
    ]


def test_every_inline_waiver_lands_in_a_cell() -> None:
    """The count is per cell, so a directive in no cell is a directive nobody counts.

    The same totality property the contract asserts over files, asked of the
    channel that sits inside them. A waiver outside every cell would be reported
    by neither the per-cell table nor a ceiling.
    """
    contract = json.loads((ROOT / ".dataknobs" / "quality-contract.json").read_text())
    for tool, scan in (("ruff", _contract.directives), ("mypy", _contract.type_ignores)):
        counted = sum(row.suppressions for row in _contract.inline_waivers(contract, tool))
        found = sum(len(scan(source)) for _name, source in _tracked_sources())
        assert found >= MINIMUM_INLINE[tool], (
            f"only {found} {tool} inline waivers found, below the floor — the scan "
            "has stopped reading what it claims to"
        )
        assert counted == found, (
            f"{found - counted} {tool} inline waiver(s) fall in no cell, so the "
            "per-cell table under-reports the channel it exists to size"
        )


def test_the_inline_count_reads_comments_rather_than_grepping() -> None:
    """Why the number is tokenized, stated as a measurement rather than a preference.

    A grep over the same tree returns half again as many `noqa` as this does,
    and nearly all of the excess is one guard's own test inputs — spellings
    inside strings, which are not directives to ruff and must not be to us. The
    figure that matches what the tool would act on is the tokenized one.

    Asserted as an inequality rather than a fixed gap: the excess moves whenever
    that guard gains a case, and a number pinned here would go stale silently,
    which is the defect this whole file exists to close.
    """
    grepped = sum(
        len(re.findall(r"#\s*noqa", source, re.IGNORECASE)) for _name, source in _tracked_sources()
    )
    tokenized = sum(len(_contract.directives(source)) for _name, source in _tracked_sources())
    assert tokenized < grepped, (
        "the tokenized and grepped counts now agree, so this no longer "
        "demonstrates anything — check the scan still skips string literals"
    )


def test_the_counter_does_not_count_its_own_documentation() -> None:
    """Regression: a module that scans the tree it lives in can report itself.

    Measured, not hypothetical. The comment introducing the type-directive
    pattern spelled that directive with its leading hash, so the tokenizer read
    it as a real one and the reported total was one too high — a module counting
    its own prose as a use of the thing it documents. Rewritten to keep the
    spelling out of a comment, the way the sibling guard keeps its examples in
    docstrings.
    """
    source = (ROOT / "bin" / "quality-contract.py").read_text(encoding="utf-8")
    assert not _contract.type_ignores(source), (
        "bin/quality-contract.py now contains a type-suppression directive in a "
        "comment. If it is documentation, keep the spelling out of a comment; if "
        "it is a real waiver, it needs its own reason and this test needs a note."
    )
    assert not _contract.directives(source), (
        "bin/quality-contract.py now contains a lint-suppression directive in a "
        "comment — same reasoning as above"
    )

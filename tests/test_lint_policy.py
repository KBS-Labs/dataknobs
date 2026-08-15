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
  Where ruff marks any of its fixes *unsafe* it is contradicting that claim, so
  the entry has to say why the unsafety is not a behaviour difference here.
* ``covered-elsewhere`` — another tool enforces the property. The reason names
  which, and a named rule has to be one ruff actually enforces: a cover this
  config also declines leaves the property enforced by nobody.
* ``behavioural`` — findings can be real, and the decline is argued. The
  argument is the continuation comment under the entry, and at least one line
  of it is required.
* ``provisional`` — findings can be real and the decline is **not** argued. The
  reason carries the count, compared against what ruff reports today.

``presentational`` had no check at all when this file was written, which is the
inverted gradient worth naming: it is the largest category and the one making
the strongest claim, so the cheapest way to satisfy every guard here was to
assert the most. Twelve of its 48 entries turned out to have fixes ruff will not
apply unasked; four were genuinely behavioural and were re-filed.

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
from collections import Counter
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
#:
#: The waiver floor was 15 against a real 31, which is not a floor: it is room
#: for sixteen waivers to stop parsing unnoticed. It is a backstop now rather
#: than the guard, because both blocks are pinned against ``tomllib`` below and
#: the pin catches a partial parse the floor was never going to see. What the
#: floor still covers is the case the pin cannot: both readers resolving to
#: nothing and agreeing about it.
MINIMUM_DECLINES = 40
MINIMUM_WAIVERS = 25

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
        if d.category == "behavioural" and not any(line.strip() for line in d.argument)
    ]


def uncovered_cover(declines: list[Any], enabled: frozenset[str]) -> list[str]:
    """``covered-elsewhere`` entries naming no cover that could be covering.

    Naming a rule was accepted on the spelling alone, and a rule this config
    *also* declines covers nothing — the property is then enforced by neither,
    which is the one arrangement this category exists to rule out. So a named
    code has to be one ruff actually enforces, asked of the same authority
    ``explain`` uses.
    """
    faults = []
    for d in declines:
        if d.category != "covered-elsewhere":
            continue
        if any(cover in d.prose for cover in DECLINE_COVERS):
            continue
        named = _CODE_RE.findall(d.prose)
        if any(code in enabled for code in named):
            continue
        detail = (
            f"names {named}, none of which ruff enforces here" if named else "names no cover at all"
        )
        faults.append(
            f"{d.code}: declared [covered-elsewhere] but {detail} — "
            f"say which tool, or which enforced rule, decides the property"
        )
    return faults


def uncounted_provisional(declines: list[Any], measured: dict[str, int]) -> list[str]:
    """``provisional`` entries whose stated count is not the one ruff reports.

    The count is the whole point of the category: it makes an unargued decline
    something a report can add up and a reader can watch fall. A figure nothing
    compares against the tree is a figure that goes stale silently, which is the
    same defect as the prose page this file replaced — and the reason it went
    unnoticed is that "carries a count" was checked by pattern rather than by
    measurement, so any numeral satisfied it. "PEP 484" would have passed.

    Compared as membership rather than equality because the reasons say more
    than one number — "36 findings, 17 of them in src/" — and which is the total
    is not something a regex should be deciding.
    """
    faults = []
    for d in declines:
        if d.category != "provisional":
            continue
        stated = {int(n) for n in _COUNT_RE.findall(d.prose)}
        if not stated:
            faults.append(
                f"{d.code}: declared [provisional] with no measured finding count in its reason"
            )
        elif measured[d.code] not in stated:
            faults.append(
                f"{d.code}: reason states {sorted(stated)} but ruff now reports "
                f"{measured[d.code]} findings — re-measure and rewrite the reason"
            )
    return faults


def unargued_presentational(declines: list[Any], unsafe: dict[str, int]) -> list[str]:
    """``presentational`` entries ruff contradicts, carrying no argument.

    The category with the strongest claim and, until this, the only one with no
    check behind it — so of the four the cheapest to satisfy was the one
    asserting the most, and 48 of the 83 entries sat under it.

    ruff already holds an opinion. A fix it marks ``unsafe`` is ruff declining to
    apply it unasked because doing so may change what the code does, which is the
    negation of this category's claim, made by the tool.

    Evidence rather than verdict: an unsafe marking is often about comment
    preservation or a preview flag, not semantics. So the requirement is an
    *argument*, exactly what ``behavioural`` is held to — say why ruff's
    unsafety is not a behaviour difference here, or file it as behavioural.
    """
    return [
        f"{d.code}: declared [presentational], but ruff marks {unsafe[d.code]} of its "
        f"fixes unsafe and the entry carries no argument"
        for d in declines
        if d.category == "presentational"
        and unsafe.get(d.code)
        and not any(line.strip() for line in d.argument)
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


#: The checks that decide from the config text alone. The other three consult a
#: tool — ruff's enforced set, its fix applicability, its finding counts — and
#: are driven by their own tests, because a check that shells out cannot be
#: called with a synthetic record list and still be asking the same question.
CHECKS = (uncategorized, unargued_behavioural)

#: How many entries each category holds, as a floor. Without this, driving a
#: category to zero silently retires its check: ``uncounted_provisional`` over
#: no provisional entries returns ``[]`` and passes, and emptying that category
#: is this file's own stated goal, so the guard would go quiet at exactly the
#: moment it was supposed to be confirming success. Real counts when written:
#: 39 presentational, 29 behavioural, 12 covered-elsewhere, 3 provisional.
MINIMUM_PER_CATEGORY = {"presentational": 20, "behavioural": 15, "covered-elsewhere": 6}


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


def test_a_waiver_the_line_regex_cannot_read_is_still_a_waiver() -> None:
    """The sibling parse that shipped with no pin, fixed rather than pinned.

    ``parse_per_file_waivers`` was a second line regex over a block ``tomllib``
    also reads, with the same failure mode as the decline parse and none of its
    comparison behind it. Its only guard was a floor of 15 against a real
    population of 31 — room for sixteen waivers to stop parsing with every
    assertion in this file still green.

    Both consequences point the unsafe way. A waiver nobody parses is never
    checked for a reason, and ``explain`` answers ``reported`` for a file ruff
    genuinely waives, which tells a worker a suppressed finding is live.

    The remedy is not a third comparison. The pattern and the codes are TOML
    *data* and are now read by the reader that cannot miss them; only the reason
    is a comment, which is the one thing ``tomllib`` cannot return. So this
    drives the shape that used to disappear — a waiver reformatted across lines,
    which the regex cannot match by construction — and asserts it survives.

    Written against a mutated copy of the real config rather than a minimal one,
    so what is exercised is this block with one entry rewrapped, not a synthetic
    file that shares nothing with it.
    """
    text = _config_text()
    single_line = '"packages/legacy/*" = ["D", "N", "UP"]'
    assert single_line in text, (
        "the waiver this rewraps is no longer in the config; pick another "
        "single-line entry, or this test mutates nothing and proves nothing"
    )
    rewrapped = text.replace(single_line, '"packages/legacy/*" = [\n    "D", "N", "UP",\n]', 1)

    parsed = {w.pattern: w for w in parse_per_file_waivers(rewrapped)}
    declared = tomllib.loads(rewrapped)["tool"]["ruff"]["lint"]["per-file-ignores"]

    assert set(parsed) == set(declared), (
        "a waiver written across lines was dropped by the parse.\n"
        f"  declared but not parsed: {sorted(set(declared) - set(parsed))}\n"
        "Those waivers are checked for no reason, and explain calls their files "
        "enforced — a worker is told a suppressed finding is live."
    )
    assert sorted(parsed["packages/legacy/*"].codes) == ["D", "N", "UP"], (
        "the rewrapped waiver parsed, but not with the codes it declares"
    )


def test_every_declared_waiver_is_one_this_file_checks() -> None:
    """Non-vacuity for the reason check, which is only as total as its population.

    ``reasonless_waivers`` reports entries carrying no reason. It says nothing
    about entries it never received, so the guard is worth exactly what the
    parse's coverage is worth — and that coverage was the thing that had no
    floor worth the name. Asserted here rather than assumed, and cheap now that
    the population comes from ``tomllib``.
    """
    text = _config_text()
    parsed = parse_per_file_waivers(text)
    declared = tomllib.loads(text)["tool"]["ruff"]["lint"]["per-file-ignores"]

    assert len(parsed) >= MINIMUM_WAIVERS, (
        f"found only {len(parsed)} per-file waivers, below the floor of "
        f"{MINIMUM_WAIVERS} — the block has moved and this no longer reads it"
    )
    assert {w.pattern for w in parsed} == set(declared)
    assert all(sorted(w.codes) == sorted(declared[w.pattern]) for w in parsed)


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
    """And the cover has to be something that could be covering.

    A named rule was accepted on its spelling. ``D213``'s reason names ``D203``
    as well as ``D212``, and ``D203`` is declined right beside it — so a reason
    that named only the declined one would have passed while describing a
    property nothing enforces. The enforced set comes from ruff.
    """
    faults = uncovered_cover(parse_declines(_config_text()), _contract.enabled_rules())
    assert not faults, "\n".join(
        [
            f"[covered-elsewhere] has to say what the cover is — {' or '.join(DECLINE_COVERS)}, "
            "or a rule code that ruff actually enforces here:",
            *faults,
        ]
    )


def test_every_provisional_decline_carries_its_count() -> None:
    """The count, compared against the tree rather than matched as a numeral.

    This is the category whose target is zero, so its figures are the only ones
    in the file that are supposed to move — which makes them the ones most able
    to go stale unnoticed. Measured, they cannot.
    """
    declines = parse_declines(_config_text())
    provisional = [d.code for d in declines if d.category == "provisional"]
    assert provisional, (
        "no provisional declines remain. That is this category's stated target, "
        "so it is good news — but delete this assertion deliberately rather than "
        "letting the check below pass over an empty list."
    )

    faults = uncounted_provisional(declines, _contract.measure_declines(provisional))
    assert not faults, "\n".join(
        [
            "[provisional] is the unargued category, and its count is what makes "
            "the backlog of unargued declines a number rather than a feeling:",
            *faults,
        ]
    )


def test_every_presentational_decline_ruff_contradicts_is_argued() -> None:
    """The check the largest category shipped without.

    Not hypothetical, and not a small correction. Run against the classification
    as first written, twelve of the 48 ``presentational`` entries had findings
    whose fixes ruff marks unsafe, and none of the twelve said a word about it.
    Four of those turned out to be genuinely behavioural on ruff's own stated
    reasoning — RUF005, SIM103, SIM118 and PLR1714, where the wording is "change
    program behaviour", "change the program's behavior", "not... known to be a
    dictionary" and "change behavior in the presence of side-effects" — and were
    re-filed. The other eight are unsafe over comments, a preview flag or
    docstring-parsing tools, and now say so.

    The measurement is one ruff invocation over the audit's roots and takes well
    under a second, which is the whole cost of the category having a check.
    """
    declines = parse_declines(_config_text())
    presentational = [d.code for d in declines if d.category == "presentational"]
    assert presentational, "no presentational declines, so this asserts nothing"

    unsafe = _contract.unsafely_fixable(presentational)
    faults = unargued_presentational(declines, unsafe)
    assert not faults, "\n".join(
        [
            "[presentational] claims no finding of the rule can be a behaviour "
            "difference. ruff disagrees about these, by declining to apply their "
            "fixes unasked. Write the argument for why its unsafety is not a "
            "behaviour difference here, or file the entry as [behavioural]:",
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
    enabled = frozenset({"D211", "D212"})
    good = Decline("X100", "presentational", "a rendering property", ())
    assert not uncategorized([good])
    assert not unargued_behavioural([good])
    assert not uncovered_cover([good], enabled)
    assert not uncounted_provisional([good], {})

    assert uncategorized([Decline("X101", None, "no marker at all", ())])
    assert uncategorized([Decline("X102", "invented", "not in the vocabulary", ())])
    assert unargued_behavioural([Decline("X103", "behavioural", "asserted, not argued", ())])
    assert not unargued_behavioural(
        [Decline("X104", "behavioural", "argued", ("because of this",))]
    )
    assert unargued_behavioural([Decline("X111", "behavioural", "blank lines below", ("", " "))]), (
        "an argument of empty comment lines is not an argument, and `not d.argument` "
        "is False for a tuple holding them"
    )
    assert uncovered_cover(
        [Decline("X105", "covered-elsewhere", "something else does it", ())], enabled
    )
    assert not uncovered_cover([Decline("X106", "covered-elsewhere", "mypy does it", ())], enabled)
    assert not uncovered_cover([Decline("X107", "covered-elsewhere", "D211 does it", ())], enabled)
    assert uncovered_cover([Decline("X112", "covered-elsewhere", "D203 does it", ())], enabled), (
        "a cover this config also declines enforces nothing, so naming it must not "
        "satisfy the check"
    )

    assert uncounted_provisional([Decline("X108", "provisional", "not yet read", ())], {"X108": 3})
    assert uncounted_provisional(
        [Decline("X110", "provisional", "the same shape as RUF012, unread", ())], {"X110": 3}
    ), "a digit inside a rule code is not a measured count"
    assert not uncounted_provisional(
        [Decline("X109", "provisional", "12 findings, 4 in src/, unread", ())], {"X109": 12}
    )
    assert uncounted_provisional(
        [Decline("X113", "provisional", "12 findings, unread", ())], {"X113": 400}
    ), "a count the tree no longer reports is a stale count, not a measured one"
    assert uncounted_provisional(
        [Decline("X114", "provisional", "PEP 484 prohibits implicit Optional", ())], {"X114": 7}
    ), "a numeral in unrelated prose is not a measurement"

    assert unargued_presentational(
        [Decline("X115", "presentational", "asserted", ())], {"X115": 4}
    ), "ruff calling the fix unsafe contradicts the category, so it needs an argument"
    assert not unargued_presentational(
        [Decline("X116", "presentational", "asserted", ("unsafe only over comments",))],
        {"X116": 4},
    )
    assert not unargued_presentational([Decline("X117", "presentational", "asserted", ())], {}), (
        "a rule with no unsafe fix is not contradicted, and must not be made to argue"
    )

    assert reasonless_waivers([PerFileWaiver("a/b.py", ("F401",), "")])
    assert not reasonless_waivers([PerFileWaiver("a/b.py", ("F401",), "the import is the test")])


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_each_check_reads_something(check: Check) -> None:
    """No check may pass by looking at an empty list.

    A parse that silently returned nothing would satisfy every assertion in this
    file. The floor in the pin covers that for the real config; this covers the
    checks themselves, by proving each one is reached by real records rather than
    short-circuiting on an empty population.

    Non-emptiness of the *whole* list is not enough, which is why the floors
    below are per category. A check reads one category, so it goes vacuous when
    that category empties while the list stays long — and for ``provisional``
    emptying it is the declared goal, so the guard would fall silent at the
    moment it was meant to confirm the win.
    """
    declines = parse_declines(_config_text())
    assert declines, "the decline parse returned nothing"
    held = Counter(d.category for d in declines)
    for category, floor in MINIMUM_PER_CATEGORY.items():
        assert held[category] >= floor, (
            f"[{category}] holds {held[category]} entries, below the floor of {floor}. "
            "Either the parse has stopped reading them, or the category emptied — "
            "and a check over an empty category passes without reading anything."
        )
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

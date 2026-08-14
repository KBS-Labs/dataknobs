"""Guard for ``# noqa`` directives that suppress nothing, or suppress everything.

A suppression is a claim that a finding was considered and waived. Four
spellings break that claim, and **not one of them fails anything today**:

* **A payload that is not a code list.** ``# noqa: not calling`` shipped in this
  repository. ruff prints ``warning: Invalid `# noqa` directive`` to *stderr*
  and **exits 0** -- measured on an otherwise-clean file -- so the run stays
  green and the line it was meant to cover is reported normally.
* **The keyword run into other text.** ``# noqafoo``, ``# noqa2``,
  ``# noqa_check: F401``. ruff wants end-of-comment, whitespace or ``:`` after
  the keyword; every other character makes the directive invalid, with a warning
  worded differently from the one above. Reads as a waiver, suppresses nothing.
* **A code ruff does not have.** ``# noqa: XYZ999`` is worse, because ruff says
  **nothing at all**: no warning, and ``RUF100`` does not fire either. It has
  the shape of a considered waiver and the effect of a comment.
* **A blanket ``# noqa``.** ``PGH004`` is exactly this rule and the workspace
  does not select the ``PGH`` family, so nothing enforces it. ``RUF100`` catches
  a blanket directive only once it is *unused*, which is the opposite case.

The first two are one outcome to ruff and two here, because the remedies differ:
one needs its payload rewritten, the other is not a directive at all. They and
the blanket case were a single pair of buckets until the parity test below was
given inputs of the second shape -- at which point it failed, which is the whole
reason that test drives the real binary instead of trusting this file's model of
it.

``RUF100`` covers the fourth spelling -- a well-formed directive for a rule that
no longer fires -- so that one is left to it rather than duplicated here.

**Scope is every tracked ``*.py``, deliberately wider than what ruff lints.**
The directive that prompted this sat in ``packages/*/tests``, which the gate did
not lint at all, so it had two independent reasons to survive: nothing would have
failed on it, and nothing was reading the file. Scoping this guard to the linted
set would rebuild the second reason inside the check written to close the first
-- and would do so *shrinkingly*, since those cells are being promoted into the
linted set one package at a time. A guard whose scope is "whatever ruff reaches
today" says less the more of them land, which is backwards.

The parser below is a re-implementation of ruff's, which is a liability rather
than a convenience -- so it is pinned. ``test_the_parser_agrees_with_ruff``
drives the real binary over every spelling this module claims to know and
compares verdicts; if ruff's rule ever changes, that test fails rather than this
file quietly reporting on a rule ruff no longer applies.

One consequence of matching ruff is worth stating, because it is what keeps this
file out of its own report: **only comment tokens are read.** A directive
spelling inside a string -- this docstring, or a test's probe input -- is not a
directive to ruff and is not one here either. So the examples above cost
nothing, while a ``#`` comment in this module genuinely cannot spell a bare
directive. That is a narrower constraint than the self-exemption the
internal-label guard needs, and it is enforced rather than trusted: this file is
in the scan like every other.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tokenize
from functools import cache

import pytest

from tests._workspace import ROOT, load_bin_module, tracked_python_files

#: The parser moved to ``bin/quality-contract.py`` when a second caller wanted
#: it — the per-cell inline-waiver count in ``explain --audit``. It is still
#: pinned from here, by ``test_the_parser_agrees_with_ruff`` below, because this
#: is where the real binary gets driven over every spelling it claims to know.
#: What changed is only that the thing being pinned is now the one thing both
#: callers use, instead of one of two copies.
_contract = load_bin_module("quality-contract")

BLANKET = _contract.BLANKET
UNREADABLE = _contract.UNREADABLE
RUN_ON = _contract.RUN_ON
NOQA_RE = _contract.NOQA_RE
CODE_RE = _contract.CODE_RE
directives = _contract.directives

#: How this repo resolves ruff; see ``test_ruff_config_single_source`` for why a bare
#: ``ruff`` would degrade the checks that use it into a silent skip.
RUFF = ("uv", "run", "ruff")

#: Floors under the scan. Real counts when written: 1,463 files, 72 directives.
#: Set well below both so ordinary growth does not move them, and far enough
#: above zero that an enumeration resolving to nothing fails instead of passing
#: -- an empty scan and a clean tree produce the same report otherwise.
MINIMUM_FILES_SCANNED = 800
MINIMUM_DIRECTIVES_SCANNED = 40

#: The two ruff refuses to read. Both warn on stderr, both suppress nothing, and
#: the parity test compares exactly this set against the binary's warnings.
RUFF_REJECTS = frozenset({UNREADABLE, RUN_ON})


@cache
def _scanned() -> tuple[tuple[str, str], ...]:
    """``(name, source)`` for every tracked Python file, read once."""
    return tuple(
        (name, (ROOT / name).read_text(encoding="utf-8"))
        for name in tracked_python_files()
        if (ROOT / name).is_file()
    )


@cache
def _all_directives() -> tuple[tuple[str, int, str, list[str]], ...]:
    found = []
    for name, source in _scanned():
        found.extend((name, lineno, kind, codes) for lineno, kind, codes in directives(source))
    return tuple(found)


@cache
def _untokenizable() -> tuple[str, ...]:
    """Files the scan could not read, which must never be silently dropped.

    All 1,463 tracked files tokenize today, so this is an exceptional state
    rather than a tolerated one -- and it is the shape that turns a guard vacuous
    without changing its report. Named and asserted rather than skipped past.
    """
    failed = []
    for name, source in _scanned():
        try:
            directives(source)
        except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    return tuple(failed)


def _ruff_missing() -> bool:
    return shutil.which("uv") is None


@cache
def _known_codes() -> frozenset[str]:
    """Every rule code ruff ships, asked of ruff rather than listed here."""
    result = subprocess.run(
        [*RUFF, "rule", "--all", "--output-format", "json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    codes = frozenset(rule["code"] for rule in json.loads(result.stdout))
    assert len(codes) > 100, f"ruff reported only {len(codes)} rules — did the format change?"
    return codes


def test_no_directive_carries_a_payload_ruff_cannot_read():
    """The spelling that shipped. ruff warns on stderr and exits 0."""
    violations = [
        f"{name}:{lineno}: `# noqa:` with no rule code — ruff rejects the "
        f"directive, warns on stderr and exits 0, so the finding it was meant "
        f"to waive is reported and nothing fails"
        for name, lineno, kind, _ in _all_directives()
        if kind == UNREADABLE
    ]
    assert not violations, "Unreadable `# noqa` directives:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_no_directive_runs_the_keyword_into_other_text():
    """``# noqafoo``: reads as a waiver, and ruff will not read it as one.

    Split from the blanket check rather than sharing it. Both arrive with no
    codes, so a parser that only asked "was there a colon" put them in one
    bucket and reported this one as blanket -- which is wrong twice over. It
    does not suppress every rule; it suppresses none. And the advice attached to
    the blanket message, *name the codes being waived*, is what the author of
    ``# noqa_check: F401`` was already trying to do.

    ruff's own warning is worded differently from the unreadable-payload one for
    the same reason, which is where the split came from.
    """
    violations = [
        f"{name}:{lineno}: the `noqa` keyword runs into other text — ruff wants "
        f"end-of-comment, whitespace or `:` after it, so it reads this as an "
        f"invalid directive, warns on stderr and suppresses nothing"
        for name, lineno, kind, _ in _all_directives()
        if kind == RUN_ON
    ]
    assert not violations, "Unread `# noqa` directives:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_no_directive_is_blanket():
    """``PGH004``'s rule, enforced here because the ``PGH`` family is not selected.

    A blanket directive is the inverse defect of the two above: they suppress
    nothing, this suppresses everything on the line -- including rules written
    years later that nobody considered when it was added.
    """
    violations = [
        f"{name}:{lineno}: bare `# noqa` suppresses every rule on the line — "
        f"name the codes being waived"
        for name, lineno, kind, _ in _all_directives()
        if kind == BLANKET
    ]
    assert not violations, "Blanket `# noqa` directives:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_no_directive_names_a_rule_ruff_does_not_have():
    """The quietest of the four: ruff reports nothing, and neither does RUF100.

    Separate from the payload check because the failure is invisible rather than
    merely unenforced — a typo'd code reads as a considered waiver forever. The
    other three all leave a trace somewhere; this one leaves none.
    """
    if _ruff_missing():
        pytest.skip("uv is not available to resolve ruff")

    known = _known_codes()
    violations = [
        f"{name}:{lineno}: `{code}` is not a ruff rule — the directive has the "
        f"shape of a waiver and the effect of a comment"
        for name, lineno, _, codes in _all_directives()
        for code in codes
        if code not in known
    ]
    assert not violations, "`# noqa` directives naming unknown rules:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_the_scan_reads_what_it_claims_to():
    """Two floors, because the checks above pass over an empty scan.

    A tree with no suppressions and an enumeration that returned nothing produce
    identical reports, and this is the only thing that tells them apart.
    """
    files = _scanned()
    assert len(files) >= MINIMUM_FILES_SCANNED, (
        f"only {len(files)} tracked Python files were scanned, below the floor "
        f"of {MINIMUM_FILES_SCANNED} — the enumeration has narrowed and the "
        f"checks above are passing over files they no longer read"
    )

    found = _all_directives()
    assert len(found) >= MINIMUM_DIRECTIVES_SCANNED, (
        f"only {len(found)} directives were found across {len(files)} files, "
        f"below the floor of {MINIMUM_DIRECTIVES_SCANNED} — the pattern has "
        f"stopped matching, so every check above holds vacuously"
    )

    assert not _untokenizable(), (
        "these tracked files could not be tokenized, so they left the scan "
        "without changing its verdict:\n" + "\n".join(f"  - {f}" for f in _untokenizable())
    )


def test_the_checks_detect_the_shapes_they_exist_for():
    """Mutation: each defect, and each near-miss that must stay legal.

    The near-misses matter more than the defects here. Every one of the 72
    directives in the tree is a code list followed by prose, so a parser that
    rejected trailing text would fail the whole repository and be "fixed" by
    deleting this file.
    """
    unreadable = {
        "empty payload": "x = 1  # noqa:",
        "prose payload": "x = 1  # noqa: not calling",
        "prose before any code": "x = 1  # noqa: see F401 above",
    }
    for label, line in unreadable.items():
        _, kind, codes = directives(line)[0]
        assert kind == UNREADABLE and not codes, f"not detected as unreadable: {label}"

    # The pair the blanket bucket used to swallow. Separated by exactly one
    # character, and on opposite sides of ruff's verdict.
    run_on = {
        "letter": "x = 1  # noqafoo",
        "digit": "x = 1  # noqa2",
        "underscore, with a code list": "x = 1  # noqa_check: F401",
        "hyphen": "x = 1  # noqa-foo",
        "dot": "x = 1  # noqa.foo",
        "paren": "x = 1  # noqa(x)",
    }
    for label, line in run_on.items():
        _, kind, _ = directives(line)[0]
        assert kind == RUN_ON, f"keyword run-on read as something else: {label}"

    blanket = {
        "bare": "x = 1  # noqa",
        "trailing space": "x = 1  # noqa ",
        "prose after whitespace": "x = 1  # noqa some prose here",
        "upper case": "x = 1  # NOQA",
    }
    for label, line in blanket.items():
        _, kind, _ = directives(line)[0]
        assert kind == BLANKET, f"not detected as blanket: {label}"

    accepted = {
        "code with em-dash prose": ("x = 1  # noqa: F401 — re-export", ["F401"]),
        "code with parenthetical": ("x = 1  # noqa: F401  (re-export)", ["F401"]),
        "code with hyphen prose": ("x = 1  # noqa: D401 - test fixture", ["D401"]),
        "comma separated": ("x = 1  # noqa:F401,E501", ["F401", "E501"]),
        "space separated": ("x = 1  # noqa: F401 F841", ["F401", "F841"]),
        "no space after hash": ("x = 1  #noqa:F401", ["F401"]),
        "upper case": ("x = 1  # NOQA: F401", ["F401"]),
    }
    for label, (line, expected) in accepted.items():
        _, kind, codes = directives(line)[0]
        assert kind == "codes" and codes == expected, f"wrongly rejected or misparsed: {label}"

    in_a_string = 'BAD = "x = 1  # noqa: not calling"\n'
    assert not directives(in_a_string), (
        "a directive spelling inside a string literal was read as a directive. "
        "ruff does not do that, and this module's own probe inputs are exactly "
        "such strings — so the guard would report itself and be answered with "
        "an exemption rather than a fix"
    )


def test_the_parser_agrees_with_ruff(tmp_path):
    """The re-implementation, checked against the thing it re-implements.

    Every spelling above is a claim about ruff's behaviour. This drives the real
    binary over all of them and compares verdicts, so a change in ruff's parser
    fails here instead of leaving the guard enforcing a rule ruff dropped.

    ``--isolated`` because the question is what ruff's *directive parser* does,
    not what this repo's configuration selects; the warning is emitted while
    reading the file, independent of whether any rule fires.
    """
    if _ruff_missing():
        pytest.skip("uv is not available to resolve ruff")

    lines = [
        "x = 1  # noqa:",
        "x = 2  # noqa: not calling",
        "x = 3  # noqa: see F401 above",
        "x = 4  # noqa",
        "x = 5  # noqa: F401 — re-export",
        "x = 6  # noqa: F401  (re-export)",
        "x = 7  # noqa: D401 - test fixture",
        "x = 8  # noqa:F401,E501",
        "x = 9  # noqa: F401 F841",
        "x = 10  #noqa:F401",
        "x = 11  # NOQA: F401",
        # The keyword running into other text. ruff reads `noqa` and then
        # requires end-of-comment, whitespace or `:`; every other character
        # makes the directive invalid, with a second warning wording. These
        # look blanket to a reader and are not blanket to ruff.
        "x = 12  # noqafoo",
        "x = 13  # noqa2",
        "x = 14  # noqa_check: F401",
        "x = 15  # noqa-foo",
        # Blanket, with prose after the whitespace. The counterpart to the
        # five above: whitespace terminates the keyword, so this one *is*
        # blanket and ruff says nothing about it.
        "x = 16  # noqa some prose here",
        'S = "x = 17  # noqa: not calling"',
    ]
    probe = tmp_path / "probe.py"
    probe.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = subprocess.run(
        [*RUFF, "check", "--isolated", "--no-cache", str(probe)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    warned = {
        int(m.group(1))
        for m in re.finditer(r"Invalid `# noqa` directive on \S+?:(\d+)", result.stderr)
    }
    assert warned, (
        "ruff warned about none of these, so either its parser changed or the "
        f"warning's wording did — this comparison is reading nothing:\n{result.stderr}"
    )

    ours = {
        lineno
        for lineno, kind, _ in directives(probe.read_text(encoding="utf-8"))
        if kind in RUFF_REJECTS
    }
    assert ours == warned, (
        f"this module's parser and ruff's disagree — ruff rejects lines "
        f"{sorted(warned)}, this module rejects {sorted(ours)}. The guard is "
        f"enforcing a rule ruff does not apply."
    )

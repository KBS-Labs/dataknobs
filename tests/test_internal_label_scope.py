"""What the internal-label guard reads, and whether its exemption still earns itself.

``bin/check-internal-labels.py`` had no tests. It is one of the checks that
decides whether a pull request passes, and its scope excluded the directory it
lives in -- so a tracker label in a gate script's comment, or in the docstring of
a guard asserting the toolchain is coherent, was caught by nothing. Widening the
scope closed that; these pin the two ways a widened scope goes quiet again.

The first is derivation. Both halves of the added scope are asked for at run
time -- ``package-discovery.sh workspace-targets`` and ``lint-shell.sh
--print-targets`` -- so a rename on either side narrows the scan without editing
this guard, and a scan that reads less still prints the same tick.

The second is the exemption. The file defining what a label looks like has to
write fourteen of them, so it is skipped; a skip nobody rechecks is how a real
label ends up in the one file exempt from noticing.

This file carries two allowlist entries of its own, in
``bin/internal-label-allowlist.txt``. They are not an exception to the rule --
they are the rule working: a test proving the guard catches a spelling has to
contain that spelling, which is a fixture value and not a reference to anything.
Widening the scope is what surfaced them, on the first run after this file
existed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests._workspace import ROOT, load_bin_module, tracked_shell_files, workspace_targets

GUARD = load_bin_module("check-internal-labels")


def _scanned() -> set[str]:
    """The default scope, as repo-relative posix names."""
    return {path.relative_to(ROOT).as_posix() for path in GUARD.iter_target_files([])}


def test_the_scope_covers_the_code_that_runs_the_gate() -> None:
    """Package code was never the gap: `bin/` and the workspace guards were.

    Asserted against the same declaration the guard reads rather than a list
    repeated here, so a directory added to ``workspace_targets`` is in scope the
    day it is added -- and one dropped from it fails here rather than silently
    leaving the scan.
    """
    scanned = _scanned()

    expected_roots = [t for t in workspace_targets() if (ROOT / t).is_dir()]
    missing_roots = [
        root for root in expected_roots if not any(name.startswith(f"{root}/") for name in scanned)
    ]
    assert not missing_roots, (
        f"declared workspace targets contribute no scanned file: {missing_roots}. "
        "The guard reports a clean scan over code it never opened."
    )

    expected_files = [t for t in workspace_targets() if (ROOT / t).is_file()]
    missing_files = sorted(set(expected_files) - scanned)
    assert not missing_files, f"declared workspace files not scanned: {missing_files}"


def test_the_scope_covers_every_shell_script_the_shell_lint_checks() -> None:
    """The row's own example was a gate script's comment, and those are shell.

    Compared against ``tracked_shell_files`` -- a different enumeration than the
    one the guard calls -- so the two have to agree. Asking ``lint-shell.sh`` and
    then checking against ``lint-shell.sh`` would pass for any answer it gave.
    """
    scanned = _scanned()
    missing = sorted(set(tracked_shell_files()) - scanned)
    assert not missing, "tracked shell scripts outside the label scan:\n" + "\n".join(
        f"  - {name}" for name in missing
    )


def test_the_scope_is_not_quietly_empty() -> None:
    """A floor under each half, so a narrowing is a failure and not a quiet pass."""
    scanned = _scanned()
    counts = {
        "bin/": sum(1 for n in scanned if n.startswith("bin/")),
        "tests/": sum(1 for n in scanned if n.startswith("tests/")),
        "packages/": sum(1 for n in scanned if n.startswith("packages/")),
    }
    thin = {where: n for where, n in counts.items() if n < 5}
    assert not thin, f"these halves of the scope resolved to almost nothing: {thin}"


def test_every_self_exemption_still_earns_itself() -> None:
    """The ratchet. An exemption that stopped being needed must be removed.

    A file is skipped here only because describing a label requires writing one.
    If its prose is reworded and no longer does, the entry is no longer a
    documented trade-off -- it is an unguarded file that reads like one, and the
    next real label written into it is reported by nothing.

    A listed file that does not exist is the same failure wearing a different
    shape, so the two are checked together rather than one being tolerated.
    """
    stale = []
    for name in sorted(GUARD.SELF_DESCRIBING):
        path = ROOT / name
        if not path.is_file():
            stale.append(f"{name}: listed but does not exist")
            continue
        text = path.read_text(encoding="utf-8")
        if not any(GUARD.LABEL_PATTERN.search(line) for line in text.splitlines()):
            stale.append(f"{name}: contains no label, so the exemption is dead")

    allowed_dead = {"bin/internal-label-allowlist.txt"}
    stale = [s for s in stale if s.split(":")[0] not in allowed_dead]
    assert not stale, (
        "SELF_DESCRIBING entries no longer justified:\n"
        + "\n".join(f"  - {s}" for s in stale)
        + "\nDrop the entry; the file is guarded again once it is gone."
    )


def test_the_exemption_is_what_keeps_the_scan_green() -> None:
    """Non-vacuity from the other side: without the skip, the guard fails.

    Otherwise the exemption could be removed with nothing to say so, and the two
    states -- exempt-and-needed, exempt-and-pointless -- are indistinguishable
    from a passing run.
    """
    guard_source = (ROOT / "bin" / "check-internal-labels.py").read_text(encoding="utf-8")
    hits = [line for line in guard_source.splitlines() if GUARD.LABEL_PATTERN.search(line)]
    assert len(hits) >= 5, (
        f"only {len(hits)} label-shaped lines in the guard's own source — the "
        "exemption is close to unnecessary, so check whether it can go"
    )


def test_both_separators_are_one_class() -> None:
    """``Item 116`` and ``Item-116`` are the same leak; only one was matched.

    Seven of these sat in the scope the guard already covered, one of them in
    shipped package source. Pinned because the separator is the single degree of
    freedom an author has, and the pattern reads as if it covered both.
    """
    for spelling in ("Item 116", "Item-116", "post-Item 116", "post-Item-116"):
        assert GUARD.LABEL_PATTERN.search(spelling), f"not matched: {spelling}"


def test_a_percent_escape_is_not_a_sub_item_id() -> None:
    r"""``sv%40c`` is a URL-encoded ``sv@c``, not the sub-item id it resembles.

    ``%`` is not a word character, so ``\b`` opens right after it and the
    sub-item branch reads the two hex digits plus the next letter as an
    id. ``%40`` is the encoding of ``@`` — the one character a userinfo
    field almost always has to escape — so this fires on any encoded
    username whose next character is ``a``-``g``, which is a standing
    collision in a repo whose Postgres DSNs are percent-encoded rather
    than a single unlucky fixture.

    The lookbehind already carries ``:`` and ``>`` for the same reason:
    a character that, immediately before digits, means the digits belong
    to an encoding rather than to an identifier.
    """
    for encoded in ("sv%40c", "u%40b", "x%40a", "p%40ss%2Fw0rd"):
        assert not GUARD.LABEL_PATTERN.search(encoded), f"false positive: {encoded}"


def test_narrowing_for_percent_escapes_still_catches_the_sub_item_ids() -> None:
    """The exclusion must be the escape, not the shape it collides with.

    Guarded in both directions on purpose: a lookbehind wide enough to
    silence ``%40c`` could also silence the ids the branch exists for,
    and a guard that stops catching its own subject still reports green.
    """
    for label in ("77a", "92b", "146b", "18a", "item 77a", "the 141 drift"):
        assert GUARD.LABEL_PATTERN.search(label), f"no longer matched: {label}"


def test_a_label_is_reported_and_sets_a_failing_status(tmp_path: Path) -> None:
    """End to end through the real script: the finding, and the exit code.

    A guard that finds a label and exits 0 is not a guard, and the caller in
    ``validate.sh`` reads nothing but the status.
    """
    leak = tmp_path / "leaky.py"
    leak.write_text('"""Docstring mentioning Item 210."""\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "check-internal-labels.py"), str(leak)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, (
        f"a file containing a tracker label exited {result.returncode}:\n"
        f"{result.stdout}{result.stderr}"
    )
    assert "Item 210" in result.stdout, result.stdout


def test_a_named_file_is_scanned_whatever_its_suffix(tmp_path: Path) -> None:
    """Naming a file is the statement that it should be read.

    The explicit-argument path filtered to ``*.py``, so pointing the guard at a
    shell script printed a clean result over a file it had silently declined to
    open -- which is the whole defect class this check belongs to.
    """
    leak = tmp_path / "leaky.sh"
    leak.write_text("#!/usr/bin/env bash\n# see Item 210 for why\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "check-internal-labels.py"), str(leak)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, (
        f"a shell script containing a tracker label exited {result.returncode}:\n"
        f"{result.stdout}{result.stderr}"
    )


def test_a_scope_helper_that_exits_non_zero_is_not_absorbed() -> None:
    """A probe that cannot run must not report a pass.

    Both halves of the added scope come from a subprocess. If one fails and the
    guard swallows it, the scan silently reverts to package code and still prints
    its tick -- a narrowing that reads from the output exactly like a clean run.

    Driven with a real failing command rather than by replacing
    ``subprocess.run``: a stub that raises would raise whatever ``check=`` said,
    so the check would pass against the very change it exists to catch. That is
    what the first draft of this test did, and the mutation found it.
    """
    with pytest.raises(subprocess.CalledProcessError):
        GUARD._declared([sys.executable, "-c", "raise SystemExit(1)"], "a failing probe")


def test_a_scope_helper_that_names_nothing_is_not_absorbed() -> None:
    """Exit zero and print nothing is the other way a scope silently empties.

    Distinct from the failure above and not covered by it: a helper whose
    directory list came back empty succeeds, so ``check=True`` says nothing, and
    an empty scope scans no files at all.
    """
    with pytest.raises(RuntimeError, match="named nothing"):
        GUARD._declared([sys.executable, "-c", "pass"], "a silent probe")


def test_a_suppression_that_covers_nothing_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The third way this guard goes quiet: the allowlist, not the scope.

    An entry is keyed by (path, substring) to survive line drift, and for a long
    time that was described as making it stable. It is not: adopting the
    formatter killed two entries in one commit -- one on quote style, one that
    had only ever matched because two tokens shared a line. Both happened to be
    loud, because the hit they stopped covering became a reported leak. The
    silent case is an entry whose target was deleted or reworded away, which
    goes on suppressing nothing while the run prints its tick.

    Run at full scope on purpose: that is the only mode where every entry is
    reachable, so it is the only mode where "unused" means "dead".

    Built from the *real* allowlist plus one planted entry so the run has no
    ordinary findings. A hand-written stub allowlist leaves every genuine
    fixture value unsuppressed, and the resulting leaks carry the failing
    status on their own -- so the test would still pass with the dead entry
    contributing nothing to it, which is one of the two things being checked.
    """
    planted = "- Item MOVED-BY-A-REFORMAT"
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text(
        GUARD.ALLOWLIST_FILE.read_text(encoding="utf-8")
        + f"packages/xization/tests/test_md_constructs.py\t{planted}\tplanted\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(GUARD, "ALLOWLIST_FILE", allowlist)

    status = GUARD.main([])
    out = capsys.readouterr().out

    assert "occurrence(s)" not in out, (
        "the repository has real label findings, so this test can no longer "
        f"attribute the failing status to the dead entry alone:\n{out}"
    )
    assert status == 1, f"a dead allowlist entry exited 0:\n{out}"
    assert planted in out, out
    assert "'- Item '" not in out, f"a live entry was reported as dead:\n{out}"


def test_a_targeted_run_does_not_call_every_other_entry_dead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-vacuity from the other side, and the reason the check is scoped.

    On a run over one named file, every entry pointing anywhere else is unused
    for a reason that says nothing about the entry. Reporting those would make
    the guard unusable for exactly the targeted invocation its argument list
    exists to support, so the fix would be to delete the check.
    """
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("packages/nowhere/absent.py\tnever matches\tdead\n", encoding="utf-8")
    monkeypatch.setattr(GUARD, "ALLOWLIST_FILE", allowlist)

    clean = tmp_path / "clean.py"
    clean.write_text('"""Nothing label-shaped here."""\n', encoding="utf-8")

    status = GUARD.main([str(clean)])
    out = capsys.readouterr().out

    assert status == 0, f"a targeted run reported unrelated entries as dead:\n{out}"
    assert "never matches" not in out, out


def test_a_criterion_number_is_caught_in_both_casings() -> None:
    """The one family that leaked, and the spelling its own census missed.

    A criterion marker above a test is a pointer into a document the reader
    cannot open. It is also the one planning object a test is genuinely
    *about*, which is why the association moved to the planning tree first --
    a criteria row there names the test function -- rather than being deleted
    and reconstructed from memory later.

    The census that found thirteen of them keyed on the capitalised form and
    missed three in the lowercase one, so both casings are pinned here. So is
    the plural: one test discharged two and its marker said so. A qualifier in
    front ("acceptance") changes nothing and needs no branch of its own.

    The spellings live in this one line rather than in the prose above,
    because a suppression is keyed to a substring and prose gets reworded --
    and a suppression whose target was reworded goes on suppressing nothing
    while the run prints its tick.
    """
    for spelling in ("Criterion 20", "criterion 19", "Criteria 19 and 16"):
        assert GUARD.LABEL_PATTERN.search(spelling), f"not matched: {spelling}"


def test_a_bare_single_digit_decision_code_is_caught() -> None:
    """A decision code resolves to nothing in this repository, source included.

    Nineteen lines carried one, two of them in shipped ``react.py`` -- the
    only half of this family a consumer could actually encounter. The
    separator varies the way the ``Item N`` branch's does, so the branch ends
    at a word boundary rather than at a space.
    """
    for spelling in ("D5", "D2", "D3-cap", "D1/D4", "(D3/D7)", "the D4 scoping"):
        assert GUARD.LABEL_PATTERN.search(spelling), f"not matched: {spelling}"


def test_the_decision_branch_does_not_catch_a_pydocstyle_code() -> None:
    """The sharper half, and the one that decides whether the branch is usable.

    ruff's pydocstyle codes are three digits, public, legitimate, and 27 of
    the 48 raw hits the census started from. A branch that is not single-digit
    and word-bounded catches every ``noqa`` naming one and the guard fails on
    its first run -- so this is checked from the other side, the way the
    percent-escape narrowing is.

    A hex literal is here for the leading boundary rather than the trailing
    one: ``0xD4`` has no word break before the ``D``, which is what keeps the
    branch off hash fragments and model ids.
    """
    for benign in (
        "x = 1  # noqa: D401 - test fixture",
        '"D400"',
        '"D100", # [presentational] Missing docstring in public module',
        "bin/quality-contract.py explain D203",
        "0xD4",
    ):
        assert not GUARD.LABEL_PATTERN.search(benign), f"false positive: {benign}"


def _prose_lines() -> list[tuple[str, str]]:
    """Every line of tracked prose, as (repo-relative name, line).

    Read from the repository rather than from literals here, which is what
    keeps the prose tests below free of new allowlist entries: a sample that
    comes out of the tree is a fixture nobody had to write down.
    """
    lines: list[tuple[str, str]] = []
    for path in GUARD._tracked_prose():
        rel = path.relative_to(ROOT).as_posix()
        lines.extend((rel, line) for line in path.read_text(encoding="utf-8").splitlines())
    return lines


def test_tracked_prose_is_in_the_default_scope() -> None:
    """The walk read ``*.py``, so the exposed prose in this repo was unread.

    Named individually rather than by a count, because the three that matter
    are the three a stranger can reach without cloning: the changelog the site
    publishes, and the two files the data package's sdist carries.
    """
    scanned = _scanned()
    exposed = [
        "docs/changelog.md",
        "packages/data/CHANGELOG.md",
        "packages/data/docs/record-serialization.md",
    ]
    missing = [name for name in exposed if name not in scanned]
    assert not missing, (
        f"published prose outside the label scan: {missing}. "
        "These reach a reader who has never cloned the repository."
    )
    prose = sum(1 for name in scanned if name.endswith(".md"))
    assert prose > 100, f"the prose half resolved to {prose} files, which is a narrowing"


def test_a_directory_walk_contributes_markdown(tmp_path: Path) -> None:
    """The suffix rule, checked where the prose half cannot stand in for it.

    ``_tracked_prose`` covers the default run, so a walk that silently went
    back to ``*.py`` would still pass every full-scope assertion. A directory
    named on the command line has no such second source -- pointing the guard
    at ``docs/`` has to read the docs.
    """
    (tmp_path / "notes.md").write_text("See Item 210 for why.\n", encoding="utf-8")

    scanned = {p.name for p in GUARD.iter_target_files([str(tmp_path)])}
    assert "notes.md" in scanned, f"a directory walk skipped its Markdown: {scanned}"


def test_the_prose_probe_that_names_nothing_is_not_absorbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The third scope half gets the same floor as the other two.

    A repository with no tracked Markdown is not a state this check should
    print a tick over -- it is the shape of a probe that stopped working.
    """

    class _Empty:
        stdout = ""

    monkeypatch.setattr(GUARD.subprocess, "run", lambda *a, **k: _Empty())
    with pytest.raises(RuntimeError, match="named nothing"):
        GUARD._tracked_prose()


def test_every_prose_exemption_has_a_real_subject_in_the_tree() -> None:
    """The ratchet on ``PROSE_EXEMPT``, in the shape the self-exemption uses.

    Each entry is a branch this guard declines to enforce over prose, argued
    from a measurement. An entry whose shape no longer occurs in any tracked
    Markdown file is not a documented trade-off any more -- it is a hole that
    reads exactly like a considered decision, and the next branch that starts
    matching through it is reported by nothing.
    """
    lines = _prose_lines()
    assert lines, "no tracked prose to measure against"

    barren = []
    for pattern, why in GUARD.PROSE_EXEMPT:
        subjects = [
            line
            for _, line in lines
            if (label := GUARD.first_label(line, prose=False)) and pattern.match(label)
        ]
        if not subjects:
            barren.append(why)
    assert not barren, (
        "PROSE_EXEMPT entries with no subject left in the tree:\n"
        + "\n".join(f"  - {why}" for why in barren)
        + "\nDrop the entry; prose is guarded against that branch again once it is gone."
    )


def test_an_exempt_shape_is_never_what_the_prose_half_reports() -> None:
    """The exemption applied, on the repository's own lines rather than fixtures.

    Checked over every tracked line, not a sample, because the failure this
    catches is a branch that fires on *one* document.
    """
    offenders = []
    for rel, line in _prose_lines():
        label = GUARD.first_label(line, prose=True)
        if label is not None and GUARD.prose_exempt(label):
            offenders.append(f"{rel}: {label!r}")
    assert not offenders, "exempt shapes reported by the prose half:\n" + "\n".join(
        f"  - {o}" for o in offenders[:10]
    )


def test_the_exemption_is_prose_only_and_the_same_line_reports_as_code() -> None:
    """The other half: these branches still catch, in the files they were written for.

    Driven from a line that really exists so the two readings differ by the
    argument alone. A phase tag in a package doc points at a plan document
    beside it in the same tree; the same token in a docstring points at
    nothing, and the guard caught this docstring writing one.
    """
    lines = _prose_lines()
    for pattern, why in GUARD.PROSE_EXEMPT:
        sample = next(
            (
                line
                for _, line in lines
                if (label := GUARD.first_label(line, prose=False)) and pattern.match(label)
            ),
            None,
        )
        assert sample is not None, f"no subject in the tree for: {why}"
        assert GUARD.first_label(sample, prose=False) is not None, (
            f"read as code this line reports nothing, so the branch is gone: {sample!r}"
        )


def test_the_prose_exemption_is_what_keeps_the_prose_half_green() -> None:
    """Non-vacuity: without it, widening the scope costs more than it buys.

    The measurement the exemption was taken on, kept live. If the tree ever
    drifts to where the whole pattern would be nearly clean over prose, the
    exemption is no longer paying for itself and should be re-argued.
    """
    unexempted = [
        f"{rel}: {label!r}"
        for rel, line in _prose_lines()
        if (label := GUARD.first_label(line, prose=False)) and GUARD.prose_exempt(label)
    ]
    assert len(unexempted) >= 50, (
        f"only {len(unexempted)} prose lines would be reported without the "
        "exemption — re-read whether the prose half still needs one"
    )


def test_the_bare_planning_pull_request_forms_are_caught() -> None:
    """Two spellings the prose half surfaced by standing next to them.

    Both were on the published site and neither was reachable by any branch:
    a bare ``PR`` + digits is a planning label rather than a pull request --
    the public form always carries the ``#`` -- and the slash-paired ``Items``
    spelling is the ``+`` branch's other separator.

    One spelling per line so each carries its own suppression. An entry keyed
    to a substring that only matches while two tokens share a line is the
    failure the allowlist's own header records twice.
    """
    for spelling in (
        "PR7",
        "PR5A",
        "Items 125/126",
        "Item 125/126",
    ):
        assert GUARD.LABEL_PATTERN.search(spelling), f"not matched: {spelling}"


def test_the_bare_form_does_not_reach_round_the_public_reference() -> None:
    """The other side: the hash spelling has its own branch and its own rules.

    A public pull-request reference is matched by that branch and exempted in
    prose. If the bare branch reached round it, every changelog line carrying
    one would report a second, un-exemptable label under a different name --
    so the reported label is asserted, not merely the fact that something
    matched.
    """
    line = "Address PR #317 review: fix TOMBSTONE additive-delta data loss"
    assert GUARD.first_label(line, prose=False) == "PR #317", (
        "read as code this line no longer reports the hash form"
    )
    assert GUARD.first_label(line, prose=True) is None, (
        "the bare branch reached round the exemption the hash form carries"
    )

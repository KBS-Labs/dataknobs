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


def test_the_scope_covers_the_code_that_runs_the_gate():
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


def test_the_scope_covers_every_shell_script_the_shell_lint_checks():
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


def test_the_scope_is_not_quietly_empty():
    """A floor under each half, so a narrowing is a failure and not a quiet pass."""
    scanned = _scanned()
    counts = {
        "bin/": sum(1 for n in scanned if n.startswith("bin/")),
        "tests/": sum(1 for n in scanned if n.startswith("tests/")),
        "packages/": sum(1 for n in scanned if n.startswith("packages/")),
    }
    thin = {where: n for where, n in counts.items() if n < 5}
    assert not thin, f"these halves of the scope resolved to almost nothing: {thin}"


def test_every_self_exemption_still_earns_itself():
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


def test_the_exemption_is_what_keeps_the_scan_green():
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


def test_both_separators_are_one_class():
    """``Item 116`` and ``Item-116`` are the same leak; only one was matched.

    Seven of these sat in the scope the guard already covered, one of them in
    shipped package source. Pinned because the separator is the single degree of
    freedom an author has, and the pattern reads as if it covered both.
    """
    for spelling in ("Item 116", "Item-116", "post-Item 116", "post-Item-116"):
        assert GUARD.LABEL_PATTERN.search(spelling), f"not matched: {spelling}"


def test_a_label_is_reported_and_sets_a_failing_status(tmp_path):
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


def test_a_named_file_is_scanned_whatever_its_suffix(tmp_path):
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


def test_a_scope_helper_that_exits_non_zero_is_not_absorbed():
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


def test_a_scope_helper_that_names_nothing_is_not_absorbed():
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

"""Guards the coverage-and-strictness contract itself.

``.dataknobs/quality-contract.json`` declares which files each tool covers and
how far from clean each part of the tree may be. It is the third artifact tier —
a *ceiling*, not evidence: no run produces it, CI never signs it, and moving a
number is a deliberate visible diff.

The declaration it replaces held one axis (files and ruff) with its counts in
comment prose, and prose is enforced in one direction only. An entry matching
nothing failed; ``241 findings`` stayed green at 400. So the two properties that
make this a ratchet are asserted here rather than described:

* every tracked first-party ``*.py`` lands in exactly one cell per tool, and
* a ceiling is compared against a measurement, and can only fall.

The expensive half — measuring — lives in ``bin/quality-contract.py`` and runs
where measurements belong. What is asserted here is cheap enough to run on every
pull request, plus the one behavioural property that decides whether this is a
ratchet at all: that re-running the baseline command cannot raise a ceiling.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from tests._workspace import (
    QUALITY_FIXTURE,
    QUALITY_FIXTURE_CELLS,
    QUALITY_FIXTURE_CONFIG,
    ROOT,
    load_bin_module,
    quality_fixture_contract,
    rel,
    write_quality_fixture_contract,
)
from tests._workspace import load_toml as _load_toml

CONTRACT = ROOT / ".dataknobs" / "quality-contract.json"
TOOL = ROOT / "bin" / "quality-contract.py"

contract_module = load_bin_module("quality-contract")


def _contract() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return loaded


def _cells(tool: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = _contract()["tools"][tool]["cells"]
    return cells


def _check_with(
    contract: dict[str, Any],
    tool: str,
    measurement: Any,
    only: set[str] | None = None,
) -> dict[str, Any]:
    """Run the real ``check`` over a supplied measurement instead of taking one.

    ``MEASURERS`` is a dispatch registry, so substituting an entry drives the
    whole of ``check`` — the cell walk, the tier reading, the ceiling
    comparison — over a measurement the test chose. What it replaces is the
    subprocess, not the logic under test, which is what makes it a seam rather
    than a stand-in for the thing being asserted.

    It exists because some measurements cannot be provoked from the tree: a
    finding attributed to a cell no tool is pointed at is reachable in principle
    (mypy follows imports) and not producible on demand.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(contract_module.MEASURERS, tool, lambda *_args: measurement)
        report: dict[str, Any] = contract_module.check(contract, [tool], only)
    return report


def _validate_targets() -> set[str]:
    """What ``bin/validate.sh`` reaches with no arguments.

    Asked rather than parsed, for the reason ``test_toolchain_consistency.py``
    asks it: the question is what the script checks, and reading its appends as
    text answers what it says.
    """
    listing = subprocess.run(
        [str(ROOT / "bin" / "validate.sh"), "--print-targets"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(listing.split())


def _covered_by_targets(cell_path: str, targets: set[str]) -> bool:
    """Whether every file a cell names is inside the default target set.

    A cell whose pattern contains a ``*`` is expanded, because
    ``packages/*/src`` is covered only if all ten expansions are.
    """
    if "*" in cell_path:
        expanded = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(cell_path))
        return bool(expanded) and all(name in targets for name in expanded)
    return cell_path in targets


def test_the_contract_is_total_and_well_formed() -> None:
    """Every tracked file decided about exactly once, per tool.

    A file in *no* cell is one nobody decided about, which is the state ``bin/``
    was in for as long as this repository has had a linter — outside every lint
    invocation with nothing saying so. A file in *two* is a decision that
    contradicts itself, and the winner would be whichever cell the matcher
    happened to try first.

    This also covers the *stale* direction — a cell matching no tracked file —
    which totality does not imply, since a cell can match zero files while every
    file still lands in some other cell. It is asserted through ``verify`` here
    rather than recomputed, and the fault itself is driven on a synthetic
    contract in ``test_verify_names_a_cell_that_matches_no_tracked_file``: a
    second implementation of the rule over the real contract would pass while
    the tool it is meant to be checking had stopped enforcing it.

    That second failure is not hypothetical: written with the obvious matcher,
    ``PurePosixPath.match``, the first run of this reported all 554 files under
    ``packages/*/src`` as belonging to two cells, because a relative pattern
    matches from the *right* and a cell named ``src`` therefore swallowed every
    package source.

    Driven through the tool rather than reimplemented here, so this asserts
    about the partition the measurement uses rather than a second copy of it.
    """
    result = subprocess.run(
        [sys.executable, str(TOOL), "verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{rel(CONTRACT)} does not describe the repository:\n{result.stdout}{result.stderr}"
    )


def test_the_contract_covers_every_tool_that_decides_a_verdict() -> None:
    """Non-vacuity: a tool with no cells is a tool this file says nothing about.

    Totality is checked per tool, so dropping a tool's entry entirely satisfies
    every remaining assertion — there are no cells to be inconsistent with. The
    set is pinned against the measurers rather than restated, so adding a fourth
    tool to the script without declaring its cells fails here.
    """
    declared = set(_contract()["tools"])
    measurable = set(contract_module.MEASURERS)
    assert declared == measurable, (
        f"{rel(CONTRACT)} declares {sorted(declared)} and {rel(TOOL)} can "
        f"measure {sorted(measurable)}. A tool on one side only is either an "
        "undeclared population or a ceiling nothing compares."
    )


def test_every_lint_cell_is_one_the_linter_actually_reaches() -> None:
    """Every ruff cell is inside ``bin/validate.sh``'s target set, whatever tier it names.

    This used to be two assertions over two tiers: a ``checked`` cell outside
    the target set is a ceiling of zero over files nothing lints, and a
    ``deferred`` cell inside it is a backlog declared for files already clean.
    The attack it was built for is the coordinated retreat — *"the cheapest way
    to satisfy a coverage rule is to drop a directory from the target set and
    re-file it as deferred"* — and the pair does not stop it. Each half fires on
    one edit arriving alone; performed together they agree with each other, and
    a directory leaves the local toolchain with both halves green.

    Ruff's ``deferred`` tier is gone, so the second assertion had no reachable
    input. **Deleting it and keeping the first would not have closed the
    retreat either.** ``verify`` fails a tier no cell holds, which stops the
    word being pre-staged, but a single commit that re-adds ``deferred`` to the
    tier map *and* re-files a cell into it passes ``verify`` — and then the
    first assertion does not fire, because the cell is no longer ``checked``.

    So the tier is not the subject any more. The invariant is the one a single
    tier makes sayable: **every cell the contract declares for ruff is one the
    linter reads.** No tier name appears below, which is what makes it hold
    against a vocabulary someone widens later — a cell filed under an invented
    tier fails ``verify`` for the tier and fails here for the coverage,
    independently.

    What is lost in a retreat is narrower than "coverage" and worth stating
    exactly, because it is what this guard is for. ``measure_ruff`` tallies one
    pass over the whole population regardless of tier, so ``check`` keeps
    measuring a retreated cell. The **local** half is what goes: bin/validate.sh
    stops reading the directory and bin/fix.sh stops offering a remedy, so a
    contributor's pre-push run reports clean over territory the gate measures.
    That is the shape 2c shipped.
    """
    targets = _validate_targets()
    assert targets, "bin/validate.sh --print-targets resolved nothing"

    unreached = sorted(
        cell["path"] for cell in _cells("ruff") if not _covered_by_targets(cell["path"], targets)
    )
    assert not unreached, (
        f"{rel(CONTRACT)} declares {unreached} for ruff, but bin/validate.sh does "
        "not lint them — so the ceiling beside each is a measurement of nothing "
        "locally, and `bin/fix.sh` offers no remedy for a finding the gate will "
        "still report. Restore the target. There is no tier to move the cell to: "
        "ruff declares only `checked`, and a tier nothing holds fails verify."
    )


def test_the_purpose_built_cell_is_dirty_to_itself_and_clean_to_the_gate() -> None:
    """The premise the guards over this tool now stand on, asserted in both halves.

    ``quality-fixture/`` exists because the repository stopped being able to
    serve as one. Every ruff cell here measures zero, so a guard that inflates a
    ceiling, pushes a cell under one, or checks that a census and a measurement
    agree was left comparing two empty tallies — which agree. The fixture gives
    those guards a backlog that is deliberate rather than borrowed.

    That only works while *both* halves hold, and each fails silently on its
    own. Dirty-to-itself failing leaves half a dozen guards passing over nothing,
    which is the shape they were rewritten to escape. Clean-to-the-gate failing
    puts findings in a cell declared at a ceiling of zero — the fixture would
    then be a backlog, in the one tree that is supposed to prove there is none.

    The third assertion is the one that would go first, and it names the reason
    rather than the symptom. The fixture is dirty only because it is read under
    rules this repository declines to select; adopt one of them repo-wide and
    the fixture becomes an ordinary lint failure whose message says nothing
    about why it is there. Asked of ruff both times, for the reason
    ``enabled_rules`` declines to re-implement selector resolution.
    """
    contract = quality_fixture_contract()
    measured = {
        cell["path"]: cell["ceiling"]
        for cell in contract["tools"]["ruff"]["cells"]
        if cell["path"] in QUALITY_FIXTURE_CELLS
    }
    assert sorted(measured) == sorted(QUALITY_FIXTURE_CELLS) and all(measured.values()), (
        f"{QUALITY_FIXTURE} measures {measured} under {QUALITY_FIXTURE_CONFIG}. A "
        "half that measures nothing is a guard that has stopped distinguishing "
        "anything, and it reports that by passing."
    )

    report = contract_module.check(_contract(), ["ruff"], {QUALITY_FIXTURE})
    assert not report["exceeded"], (
        f"{QUALITY_FIXTURE} is dirty to this repository's own linter: "
        f"{report['exceeded']}. It is declared at a ceiling of zero and it has "
        "to stay there — findings the gate can see make it a backlog, which is "
        "the one thing this tree exists to demonstrate the absence of."
    )

    fixture_rules = contract_module.enabled_rules(ROOT / QUALITY_FIXTURE_CONFIG)
    assert fixture_rules, f"{QUALITY_FIXTURE_CONFIG} enables no rule, so it reports nothing"
    adopted = sorted(fixture_rules & contract_module.enabled_rules())
    assert not adopted, (
        f"{QUALITY_FIXTURE_CONFIG} selects {adopted}, which this repository now "
        "enforces too. The fixture is clean to the gate only because the rules "
        "it trips are ones the gate declines — so either drop those from the "
        "fixture configuration and give it rules that are still declined, or "
        f"clear the findings they now report in {QUALITY_FIXTURE}/."
    )


def test_the_type_checker_reads_exactly_the_cells_it_is_measured_over() -> None:
    """``unchecked`` must mean the type checker never sees the file.

    The measurement runs mypy over the cells not marked ``unchecked``, so a file
    in that tier that mypy *does* read has its findings counted against whichever
    cell claimed it — or against none, and silently vanish. Both readings are
    wrong and neither is visible in the output.

    Asserted against what ``bin/validate.sh`` *does with* its targets rather than
    against the target list itself. The two used to be the same statement: the
    script type-checked whatever it linted, so comparing the tiers against
    ``--print-targets`` said something about mypy. It no longer does — that list
    is the linter's, and the script now asks ``scope`` which cells its targets
    name and hands those to the contract. The old comparison would still pass
    today, because the two lists are identical, which is exactly how a guard
    stops asserting what its name claims: unchanged, green, and about something
    else.
    """
    targets = sorted(_validate_targets())
    classified = contract_module.scope_paths(_contract(), "mypy", targets)
    reached = {cell for kind, _path, cell in classified if kind == contract_module.SCOPE_MEASURED}
    cells = _cells("mypy")

    read_anyway = sorted(
        cell["path"] for cell in cells if cell["tier"] == "unchecked" and cell["path"] in reached
    )
    assert not read_anyway, (
        f"{rel(CONTRACT)} calls {read_anyway} unchecked by mypy, but bin/validate.sh "
        "resolves its targets onto them, so mypy reads them. Give each a tier and "
        "a ceiling."
    )

    unread = sorted(
        cell["path"]
        for cell in cells
        if cell["tier"] != "unchecked" and cell["path"] not in reached
    )
    assert not unread, (
        f"{rel(CONTRACT)} gives {unread} a mypy ceiling, but bin/validate.sh "
        "does not type-check them, so the number is a measurement of nothing."
    )


def test_the_type_check_scope_comes_from_the_contract() -> None:
    """The structural property behind the tier comparison above.

    ``bin/validate.sh`` must *derive* its mypy scope from the contract rather
    than restate it. Restating it is not hypothetical — it is what the script
    did until this phase, and the two declarations then drifted in the way only
    duplicated declarations can: the script read a second configuration under
    which a transitional package was clean, so a new finding passed locally and
    failed in CI.

    Read as text, because what is being asserted is that the call exists at all.
    The behavioural half — that the derived scope is the one measured — is the
    tier comparison above, and neither substitutes for the other.
    """
    script = (ROOT / "bin" / "validate.sh").read_text(encoding="utf-8")
    assert "quality-contract.py" in script, (
        "bin/validate.sh no longer calls bin/quality-contract.py, so its mypy "
        "verdict is reached by something other than the ceilings the gate "
        "enforces. Two verdicts over one tree is the drift this replaced."
    )
    assert "scope --tool mypy" in script, (
        "bin/validate.sh no longer asks the contract which cells its targets "
        "name. Matching cell patterns in the script is a second copy of "
        "cell_matches, waiting to disagree with the one the ceilings were "
        "measured under."
    )


def test_a_baseline_update_lowers_a_ceiling_and_never_raises_one(tmp_path: Path) -> None:
    """The one property that makes this a ratchet rather than a record.

    A ceiling that rises when a command is re-run is not a ceiling; it is a
    transcript of whatever the tree happened to contain, and a backlog can grow
    all the way through the phase that is supposedly clearing it — with the
    tooling reporting green the whole way, because the number moved with it.

    Both directions in one test because it is one decision. Driven over a real
    measurement rather than a stubbed one: what is being pinned is what the
    command does to a ceiling over a tree it actually reads, and a fake
    measurement would pin it against a number that no tool produces.

    That tree is the purpose-built cell rather than this repository's own, which
    is not a weakening — the ceilings it carries are what ruff reports over it,
    taken at the same moment. This repository has no ruff backlog left to inflate
    or push under, and a ceiling of zero pushed five findings lower is still
    zero.
    """
    contract = quality_fixture_contract()
    ruff_cells = {cell["path"]: cell for cell in contract["tools"]["ruff"]["cells"]}

    inflated, deflated = QUALITY_FIXTURE_CELLS
    true_ceilings = {
        inflated: ruff_cells[inflated]["ceiling"],
        deflated: ruff_cells[deflated]["ceiling"],
    }

    ruff_cells[inflated]["ceiling"] = true_ceilings[inflated] + 500
    ruff_cells[deflated]["ceiling"] = max(true_ceilings[deflated] - 5, 0)

    destination = tmp_path / "quality-contract.json"
    lowered, exceeded = contract_module.update_baseline(
        contract, ["ruff"], destination, set(QUALITY_FIXTURE_CELLS)
    )

    assert ruff_cells[inflated]["ceiling"] == true_ceilings[inflated], (
        f"an inflated ceiling on {inflated} was not lowered to what the tree "
        f"measures; --update-baseline reported {lowered}"
    )
    assert ruff_cells[deflated]["ceiling"] == max(true_ceilings[deflated] - 5, 0), (
        f"{deflated} had a ceiling below its measurement and --update-baseline "
        "raised it. Re-running a command must never enlarge a backlog: that is "
        "an argument to have in a pull request, not a side effect of a rerun."
    )
    assert destination.is_file(), "a lowered ceiling was reported but never written"

    # Left alone *and* reported. Silence here tells a developer who has just
    # introduced a regression that there was nothing to do, which is the same
    # shape as a status field whose default is a verdict — and the function's
    # own docstring claimed it reported while the code named only what it
    # lowered.
    assert any(deflated in line for line in exceeded), (
        f"{deflated} measures above its ceiling and --update-baseline said "
        f"nothing about it; it reported only {exceeded}"
    )


def test_update_baseline_rewrites_only_the_cells_it_was_named(tmp_path: Path) -> None:
    """``--cell`` was validated and then dropped, so the narrowest command edited all.

    ``main`` resolved the named cells against the declaration — catching a
    misspelling, listing the known names — and then called ``update_baseline``
    without them. ``update-baseline --tool ruff --cell <one>`` therefore rewrote
    the ceiling of every ruff cell: the widest possible edit to the declaration,
    reached through the command that asks for the narrowest, and unrecoverable by
    re-running because a ceiling only falls.

    Both directions are asserted, and the second is the one that was broken. That
    the named cell is lowered is the feature; that an unnamed one is left exactly
    as declared is the fix.

    Written against ``tmp_path`` for the reason the ratchet test above is: the
    subject is what this command does to a ceiling, and it may not do it to the
    declaration this repository is measured against.
    """
    contract = quality_fixture_contract()
    ruff_cells = {cell["path"]: cell for cell in contract["tools"]["ruff"]["cells"]}

    named, untouched = QUALITY_FIXTURE_CELLS
    ruff_cells[named]["ceiling"] += 500
    ruff_cells[untouched]["ceiling"] += 500
    inflated = ruff_cells[untouched]["ceiling"]

    destination = tmp_path / "quality-contract.json"
    lowered, _exceeded = contract_module.update_baseline(contract, ["ruff"], destination, {named})

    assert lowered, (
        f"{named} was inflated by 500 and a scoped update-baseline lowered "
        "nothing, so the assertions below compare a run that did not happen"
    )
    assert all(named in line for line in lowered), (
        f"a scoped update-baseline reported lowering a cell it was not given: {lowered}"
    )
    assert ruff_cells[untouched]["ceiling"] == inflated, (
        f"{untouched} was not named and its ceiling moved anyway. A scoped "
        "update-baseline that rewrites every cell is the widest edit to the "
        "declaration, made by the command that asks for the narrowest."
    )


def test_a_baseline_update_leaves_the_prose_of_cells_it_did_not_touch(
    tmp_path: Path,
) -> None:
    r"""The test above asserts values; this asserts the bytes they are written as.

    ``update_baseline`` serialised with ``json.dumps``' default of
    ``ensure_ascii=True`` onto a file written without it. Every reason here is
    prose and several hold em-dashes, so lowering one ceiling re-encoded them as
    ``\u2014`` and arrived as a diff touching rows whose ceiling had not moved —
    with the single line that changed meaning buried among them. The declaration
    is meant to make moving a number a deliberate visible diff, and a diff nobody
    can read is the reviewable half of that property spent.

    It is asserted on the written text rather than on the reloaded object because
    ``json`` cannot see it: ``"\u2014"`` and ``"—"`` parse to the same string, so
    the round-trip this would seem to be tested by succeeds in both directions.
    The damage is only ever visible to a reader of the file.

    Not hypothetical — ``main`` carried one such row at the time this was
    written, from an earlier run of the same command.
    """
    contract = quality_fixture_contract()
    ruff_cells = {cell["path"]: cell for cell in contract["tools"]["ruff"]["cells"]}

    named, _ = QUALITY_FIXTURE_CELLS
    ruff_cells[named]["ceiling"] += 500

    unicode_reasons = [
        cell["reason"]
        for cells in (tool["cells"] for tool in contract["tools"].values())
        for cell in cells
        if not cell["reason"].isascii()
    ]
    assert unicode_reasons, (
        "no cell reason holds a non-ASCII character, so this test would pass "
        "against an encoder that mangles every one of them"
    )

    destination = tmp_path / "quality-contract.json"
    lowered, _exceeded = contract_module.update_baseline(contract, ["ruff"], destination, {named})

    assert lowered, (
        f"{named} was inflated by 500 and update-baseline lowered nothing, so "
        "no file was written and the assertion below reads a run that did not happen"
    )
    written = destination.read_text(encoding="utf-8")
    for reason in unicode_reasons:
        assert reason in written, (
            "a baseline update re-encoded the prose of a cell whose ceiling it "
            f"did not move. Expected to find {reason!r} written as it was declared"
        )


def test_a_baseline_update_rewrites_the_declaration_it_was_pointed_at(tmp_path: Path) -> None:
    """``--contract`` is read on the way in, so it has to be honoured on the way out.

    ``main`` loaded the declaration through an argument and wrote it back through
    a module constant, which is the one shape of this option that is worse than
    not having it: the command reports lowering a ceiling in the file it was
    given and lowers it in ``.dataknobs/quality-contract.json`` instead. A
    ceiling only ever falls, so re-running does not undo it, and the diff lands
    in the declaration this repository is measured against with nothing in the
    output naming it.

    Driven as a subprocess because the defect lives in ``main``'s wiring rather
    than in ``update_baseline``, which has taken the path as a parameter all
    along.
    """
    declaration = write_quality_fixture_contract(tmp_path)
    contract = json.loads(declaration.read_text(encoding="utf-8"))
    named = QUALITY_FIXTURE_CELLS[0]
    inflated = 0
    for cell in contract["tools"]["ruff"]["cells"]:
        if cell["path"] == named:
            cell["ceiling"] += 500
            inflated = cell["ceiling"]
    assert inflated, f"{named} is not a cell of the fixture declaration"
    declaration.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    untouched = CONTRACT.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "update-baseline",
            "--tool",
            "ruff",
            "--cell",
            named,
            "--contract",
            str(declaration),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"update-baseline exited {result.returncode}: {result.stderr}"
    lowered = {
        cell["path"]: cell["ceiling"]
        for cell in json.loads(declaration.read_text(encoding="utf-8"))["tools"]["ruff"]["cells"]
        if cell["path"] in QUALITY_FIXTURE_CELLS
    }
    assert lowered[named] < inflated, (
        f"{named} was inflated to {inflated} in the named declaration and the "
        f"command did not lower it there: {lowered}"
    )
    assert CONTRACT.read_bytes() == untouched, (
        f"update-baseline was pointed at {declaration} and rewrote "
        f"{rel(CONTRACT)} instead. A ceiling only falls, so this is not "
        "recoverable by re-running the command."
    )


def _declaration_with(tmp_path: Path, cell: str, delta: int) -> Path:
    """The fixture declaration written out with one ceiling moved by ``delta``."""
    declaration = write_quality_fixture_contract(tmp_path)
    contract = json.loads(declaration.read_text(encoding="utf-8"))
    moved = [c for c in contract["tools"]["ruff"]["cells"] if c["path"] == cell]
    assert len(moved) == 1, f"{cell} is not a single cell of the fixture declaration"
    moved[0]["ceiling"] += delta
    declaration.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return declaration


def _check(declaration: Path, cell: str) -> subprocess.CompletedProcess[str]:
    """``check`` over one cell of a named declaration, as the command line runs it."""
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "check",
            "--tool",
            "ruff",
            "--cell",
            cell,
            "--contract",
            str(declaration),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_cell_under_its_ceiling_ends_the_run(tmp_path: Path) -> None:
    """Headroom is a failure, not a note. Both signs of the comparison stop the run.

    A ceiling left standing above what the tree measures is capacity a later
    regression is absorbed into without any number moving and without anything
    reporting it: the cell rises back to its ceiling and the check stays green
    the whole way, because nothing it compares has changed. That is the same
    property the ratchet exists to deny in the other direction, and for as long
    as this reported ``is under its ceiling`` at INFO it was denied in one
    direction only.

    The zero-headroom condition it enforces is not new — it has been declared
    for as long as these ceilings have been moving, and honoured by hand every
    time one moved. What was missing was anything that noticed when it was not.

    Driven as a subprocess because what is asserted is the *exit status*.
    ``report["cleared"]`` was populated before this change too; the question is
    whether a run ends because of it, and only ``main`` answers that.
    """
    dense = QUALITY_FIXTURE_CELLS[0]
    result = _check(_declaration_with(tmp_path, dense, +500), dense)

    assert result.returncode == 1, (
        f"a cell 500 under its ceiling exited {result.returncode}. Headroom that "
        "reports success is a regression budget nobody voted for: the backlog "
        "can climb 500 findings back up and every run in between stays green."
    )
    assert dense in result.stderr, (
        f"the run failed without naming the cell that failed it: {result.stderr!r}"
    )


def test_the_advice_for_a_cleared_cell_is_the_command_that_clears_it(tmp_path: Path) -> None:
    """Reproduce-first: the remediation is executed, not read.

    ``tests/test_remediation_paths.py`` opens on advice that named a command
    which could not work, and its point is that wrong advice is caught by
    nothing — no test reads it, and the person who does is already dealing with
    a failure. That file guards shell scripts; this string is printed from
    Python, in the one place it cannot see.

    Two things are wrong with the obvious implementation and both are silent.
    The advice must carry ``--contract`` when the run was given one, or a check
    pointed at one declaration tells a developer to rewrite a different one —
    and since a ceiling only falls, following it is not undone by re-running
    anything. And it must name ``--cell``, which ``update-baseline`` honours
    only because a prior fix made it: before that, the command asking for the
    narrowest edit performed the widest one available.

    So the printed line is split and run verbatim rather than matched against a
    pattern. A pattern asserts the sentence was written; running it asserts the
    sentence is true.
    """
    dense, sparse = QUALITY_FIXTURE_CELLS
    declaration = _declaration_with(tmp_path, dense, +500)
    before = {
        cell["path"]: cell["ceiling"]
        for cell in json.loads(declaration.read_text(encoding="utf-8"))["tools"]["ruff"]["cells"]
        if cell["path"] in QUALITY_FIXTURE_CELLS
    }

    printed = _check(declaration, dense).stderr
    advice = next((line.strip() for line in printed.splitlines() if "update-baseline" in line), "")
    assert advice.startswith(contract_module._INVOCATION), (
        f"the remediation is spelled {advice!r}. It has to name the pinned "
        "toolchain: a bare `python bin/...` resolves against PATH, which is "
        "this repository's canonical instance of advice that cannot work."
    )

    argv = shlex.split(advice)[len(shlex.split(contract_module._INVOCATION)) :]
    assert "--contract" in argv, (
        f"the advice is {advice!r} — it does not name the declaration this run "
        f"was pointed at ({declaration}), so it tells a developer to lower a "
        "ceiling in the repository's own contract instead. A ceiling only "
        "falls, so that is not recoverable by re-running anything."
    )

    repair = subprocess.run(
        [sys.executable, str(TOOL), *argv], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert repair.returncode == 0, f"the advice exited {repair.returncode}: {repair.stderr}"

    after = {
        cell["path"]: cell["ceiling"]
        for cell in json.loads(declaration.read_text(encoding="utf-8"))["tools"]["ruff"]["cells"]
        if cell["path"] in QUALITY_FIXTURE_CELLS
    }
    assert after[dense] == before[dense] - 500, (
        f"running the printed advice moved {dense} from {before[dense]} to "
        f"{after[dense]}, not to what the tree measures. The command a failure "
        "recommends has to be the command that resolves it."
    )
    assert after[sparse] == before[sparse], (
        f"the advice named --cell {dense} and moved {sparse} as well: "
        f"{before} -> {after}. The narrowest command must not perform the "
        "widest edit available to it."
    )
    assert _check(declaration, dense).returncode == 0, (
        "the cell still fails after the command its own failure recommended"
    )


def test_a_charge_is_a_term_of_the_sum_the_ceiling_is_compared_against() -> None:
    """The per-file charge and the whole-cell verdict cannot disagree.

    ``charge`` exists because nothing else answers *"what does this file owe?"*
    — ``census`` decomposes a cell by rule, ``check --show-findings`` echoes a
    whole cell, and ``bin/validate.sh`` handed one filename measures the whole
    cell that file is in. The tempting implementation is to point the tool at
    the named file, and it is wrong in a way that would not show: mypy follows
    imports, so a finding's attribution depends on what else was in the pass,
    and a per-file measurement would drift from the cell total that the ceiling
    is actually compared against.

    So the cell is measured whole and only the display is filtered, and this is
    what pins that: every file's charge is a term of the cell's measured total,
    and the cell's total is reported beside it rather than replaced by it.

    Over the purpose-built cell, whose two files are what make "sums to" a
    claim rather than an identity.
    """
    contract = quality_fixture_contract()
    dense = QUALITY_FIXTURE_CELLS[0]

    owed = contract_module.charge(contract, "ruff", [dense])
    verdict = contract_module.check(contract, ["ruff"], only={dense})
    measured = verdict["cells"][f"ruff/{dense}"]["measured"]
    entry = owed["paths"][0]

    assert entry["total"] == measured, (
        f"charging {dense} reported {entry['total']} while check measures "
        f"{measured} over the same cell. A per-file answer that does not sum to "
        "the number the ceiling is compared against is a second measurement, "
        "and the convention it exists to serve would be denominated in it."
    )
    assert sum(held["count"] for held in entry["files"]) == entry["total"], (
        f"the per-file breakdown {entry['files']} does not sum to the total "
        f"{entry['total']} it is a breakdown of."
    )
    assert len(entry["files"]) > 1, (
        f"the fixture's dense half charged {len(entry['files'])} file(s), so "
        "this cannot tell a filtered display from an unfiltered one. Either the "
        "fixture was reduced to one file, or the filter is not filtering."
    )

    one = entry["files"][0]["file"]
    single = contract_module.charge(contract, "ruff", [one])["paths"][0]
    assert single["total"] == entry["files"][0]["count"], (
        f"charging {one} alone reported {single['total']}, but as part of its "
        f"cell it holds {entry['files'][0]['count']}. Naming one file must "
        "narrow the display, never the measurement."
    )
    assert single["cell_total"] == measured, (
        f"charging one file reported its cell as measuring {single['cell_total']} "
        f"rather than {measured}. The cell is still measured whole; a caller who "
        "cannot see that will read a file's charge as its cell's."
    )


def test_a_charge_reaches_the_same_answer_through_the_command_line(tmp_path: Path) -> None:
    """The wiring is asserted separately from the function, because it has failed there.

    ``update-baseline --cell`` validated the cell it was given and then called a
    function it did not pass it to, so the narrowest command performed the
    widest edit available — a defect entirely inside ``main``, with the function
    beneath it correct all along. Every option this command declares is one more
    place for that, and a caller of ``charge()`` exercises none of them.
    """
    declaration = write_quality_fixture_contract(tmp_path)
    dense = QUALITY_FIXTURE_CELLS[0]
    expected = contract_module.charge(quality_fixture_contract(), "ruff", [dense])

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "charge",
            "--tool",
            "ruff",
            dense,
            "--contract",
            str(declaration),
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"charge exited {result.returncode} over a cell with findings in it. It "
        "reports what is owed; the ceiling is what judges, and a command run "
        f"before the work must not look like the failure it precedes: {result.stderr}"
    )
    through_cli = json.loads(result.stdout)
    assert through_cli["total"] == expected["total"], (
        f"the command line reported {through_cli['total']} where the function "
        f"reports {expected['total']} over the same cell and declaration."
    )
    assert [entry["path"] for entry in through_cli["paths"]] == [dense], (
        f"the command line charged {[e['path'] for e in through_cli['paths']]} "
        f"rather than the one path it was given."
    )


@pytest.mark.parametrize(
    ("tool", "path", "because"),
    [
        ("ruff", "README.md", "falls in no cell, so it owes nothing to any ceiling"),
        ("mypy", "packages/bots/tests", "is in a tier the type checker is not pointed at"),
        ("ruff", f"{QUALITY_FIXTURE}/dense/nosuchfile.py", "names no tracked file"),
    ],
)
def test_a_charge_refuses_every_path_it_would_have_to_answer_with_a_bare_zero(
    tool: str, path: str, because: str
) -> None:
    """Three ways of naming a path, three different lies the same zero would tell.

    This command's whole value is that ``0`` means *paid*. Each of these would
    render as ``0`` and mean something else: a path outside every cell owes
    nothing to any ceiling and never will; a path in an unread tier was not
    measured at all, so its zero is a silence; and a mistyped path is the one a
    developer is most likely to produce, at the exact moment they are asking
    whether they are finished.

    The third is the one that decides whether the other two matter. A typo
    answered with "nothing outstanding" is a convention that reports itself
    satisfied by a misspelling — and unlike a wrong verdict, nothing downstream
    ever disagrees with it.
    """
    contract = quality_fixture_contract() if tool == "ruff" else _contract()
    with pytest.raises(SystemExit) as refused:
        contract_module.charge(contract, tool, [path])
    assert path in str(refused.value), (
        f"a charge for a path that {because} was refused without naming it: {refused.value}"
    )


def _with_cell(tool: str, **overrides: Any) -> dict[str, Any]:
    """The real contract with one extra cell, for exercising a single fault.

    Built from the real declaration rather than a minimal fake so the fault
    under test is the *only* thing wrong: a hand-built contract fails totality
    on all 1,470 tracked files at once, and every assertion below would then
    pass on a fault it did not cause.
    """
    contract = _contract()
    # ``checked`` because every caller taking the default is a ruff caller and
    # that is ruff's only tier; the one mypy caller names its own. It was
    # ``deferred``, which ruff no longer declares — so each default-tier caller
    # would have carried a second, uninvited tier fault, and every ``any(...)``
    # assertion over the result would have kept passing while measuring a
    # contract broken in a way its case did not ask for.
    cell = {"path": "packages/*/nowhere", "tier": "checked", "ceiling": 0, "reason": "x"}
    cell.update(overrides)
    contract["tools"][tool]["cells"].append(cell)
    return contract


def _faults_for(contract: dict[str, Any]) -> list[str]:
    faults: list[str] = contract_module.verify(contract)
    return faults


def test_verify_names_a_cell_that_matches_no_tracked_file() -> None:
    """The stale direction, which totality does not imply.

    A cell can match zero files while every file still lands in some other
    cell, so the partition stays valid and the entry stays wrong. This is half
    of the rule the previous declaration enforced — the half that fired when a
    directory was cleaned up and its deferral outlived it — and losing it in the
    move would leave the reader believing in a gap that closed.
    """
    faults = _faults_for(_with_cell("ruff"))
    assert any("packages/*/nowhere" in fault for fault in faults), (
        "verify() accepted a cell matching no tracked Python file. A gap that no "
        f"longer exists reads as one that does. Faults reported: {faults}"
    )


def test_verify_names_a_tool_whose_cells_are_missing_entirely() -> None:
    """A malformed declaration must produce a fault, not a traceback.

    ``verify`` promises 'a clear error naming the file rather than a traceback',
    and a tool entry with no ``cells`` key reached ``spec["cells"]`` directly.
    """
    contract = _contract()
    contract["tools"]["ruff"] = {"unit": "findings", "config": "pyproject.toml"}
    faults = _faults_for(contract)
    assert any("ruff" in fault for fault in faults), (
        f"a tool declaring no cells produced no fault: {faults}"
    )


def test_verify_names_each_malformed_cell() -> None:
    """Every fault branch, made to fail on purpose.

    None of these had been exercised. A guard is not done until it has been
    shown to go red, and this program has three times shipped one that could
    not — so the branches are driven here rather than trusted from reading.
    """
    cases: dict[str, tuple[dict[str, Any], str]] = {
        "tier": ({"tier": "invented"}, "invented"),
        "ceiling": ({"ceiling": "twelve"}, "twelve"),
        "boolean ceiling": ({"ceiling": True}, "True"),
        "reason": ({"reason": "   "}, "reason"),
        "reason stating a number": ({"reason": "20 autofixable"}, "states a number"),
    }
    for label, (override, expected) in cases.items():
        faults = _faults_for(_with_cell("ruff", **override))
        assert any(expected in fault for fault in faults), (
            f"a cell with a bad {label} produced no fault mentioning {expected!r}: {faults}"
        )

    duplicated = _contract()
    duplicated["tools"]["ruff"]["cells"].append(dict(duplicated["tools"]["ruff"]["cells"][0]))
    assert any("twice" in fault for fault in _faults_for(duplicated)), (
        "the same path declared in two cells produced no fault"
    )

    unmeasured = _with_cell("mypy", tier="unchecked", ceiling=7)
    assert any("claims a number nothing takes" in fault for fault in _faults_for(unmeasured)), (
        "an unmeasured tier with a positive ceiling produced no fault — the "
        "number would read as a backlog nothing is measuring"
    )


def test_verify_names_a_tier_no_cell_holds() -> None:
    """A word in the vocabulary describing no part of the tree.

    The other direction of the tier check, and it does not follow from it: every
    cell can name a declared tier while a declared tier names no cell. That was
    ruff's state for as long as it took to empty ``deferred`` — the tier had no
    cells and the contract said nothing, because the only question being asked
    was whether each cell's tier was spellable.

    It matters because vocabulary is what a retreat needs. Re-filing a cell into
    a tier that tolerates a backlog is the cheap half of un-covering a
    directory, and a tier already sitting in the map makes that a one-word edit
    to one cell. With the word gone, re-introducing it is a visible change to
    the tool's tier map — an argument someone has to make in review rather than
    a line that reads as bookkeeping.

    This is the pawl for the two mypy strikes still ahead: ``transitional`` the
    day M1 reaches zero and ``unchecked`` the day the last unmeasured cell is
    promoted. Neither needs remembering now, because the contract will fail and
    say so.
    """
    contract = _contract()
    contract["tools"]["ruff"]["tiers"]["deferred"] = "outside that target set"

    faults = _faults_for(contract)
    assert any("no cell holds that tier" in fault for fault in faults), (
        "verify() accepted a tier no cell holds. The word stays available to a "
        f"cell that wants out of being measured. Faults reported: {faults}"
    )

    # Every tier the contract really declares is held, so the fault above is the
    # only one — asserted rather than assumed, because a second fault here would
    # mean the check fires on the live declaration and this test proves nothing.
    assert not _faults_for(_contract()), (
        "the real contract already faults, so the assertion above cannot "
        "distinguish the tier it added from a pre-existing fault"
    )


def test_the_coordinated_retreat_fails_even_when_both_halves_arrive_together() -> None:
    """The attack the tier strike alone does not stop, driven end to end.

    Dropping a directory from the linter's target set and re-filing its cell
    into a backlog tier is two edits that are each a fault alone and agree with
    each other together. Striking ``deferred`` does not by itself close it: a
    commit that re-adds the word *and* uses it passes ``verify``, because the
    tier is declared and a cell holds it — the check above is satisfied by the
    very cell doing the retreating.

    What closes it is that
    ``test_every_lint_cell_is_one_the_linter_actually_reaches`` stopped reading
    tiers. Replayed here over a real target set rather than argued, because the
    two guards defend one property from opposite sides and the reasoning above
    is worth no more than the demonstration that they do.
    """
    contract = _contract()
    contract["tools"]["ruff"]["tiers"]["deferred"] = "outside that target set"
    retreating = "packages/*/examples"
    for cell in contract["tools"]["ruff"]["cells"]:
        if cell["path"] == retreating:
            cell["tier"] = "deferred"
            cell["ceiling"] = 12

    assert not _faults_for(contract), (
        "verify() was expected to accept the retreat — the tier is declared and "
        "a cell holds it. If it now rejects it, the guard below is no longer "
        "the thing standing between this repository and a quiet un-covering, "
        "and this test should say which check took over."
    )

    # The half that does catch it: the coverage guard, which never asks the tier.
    targets = _validate_targets() - {
        path.relative_to(ROOT).as_posix() for path in ROOT.glob(retreating)
    }
    assert not _covered_by_targets(retreating, targets), (
        f"{retreating} is still inside the target set after its directories were "
        "removed, so this replay does not reproduce a retreat"
    )


def test_verify_reads_a_rule_name_in_a_reason_as_prose_not_a_count() -> None:
    r"""The boundary the reason check turns on, pinned rather than left to a regex.

    A rule code carries digits and is not a measurement: ``NPY002`` names a
    *kind* of finding and stays accurate however many there are, while "130 of
    them" is the ceiling's job and only the ceiling is compared against
    anything. A check that rejected both would push every reason into vaguer
    prose in order to pass, which is the opposite of what it is for.

    Tightening the pattern to a bare ``\d`` is the plausible way to lose this,
    and it would fail here rather than in review.
    """
    named_rule = _with_cell("ruff", reason="the legacy NPY002 global RNG, which has no autofix")
    assert not any("states a number" in fault for fault in _faults_for(named_rule)), (
        f"a reason naming a rule was rejected as if it stated a count: {_faults_for(named_rule)}"
    )


def test_a_breached_ceiling_names_the_files_that_breached_it() -> None:
    """A count says *whether*; it cannot say *what*.

    The first ceiling this mechanism ever broke reported ``21 findings against
    20 allowed`` over a directory holding 21 files, and finding the
    twenty-first took a separate script. A failure nobody can act on gets
    suppressed as surely as one that fires spuriously — the same subject as G4,
    approached from the other side — so an exceeded cell carries the file names
    out with it.

    Driven over a real measurement with one ceiling pushed below what the tree
    holds, because what is being pinned is what a developer sees when a cell
    goes over. The purpose-built cell, since no cell of this repository's own
    can be pushed under a ceiling any more: they measure zero, and there is
    nothing below it.

    Its larger half holds two files, which is what makes the ranking below a
    ranking rather than a listing.
    """
    contract = quality_fixture_contract()
    cells = {cell["path"]: cell for cell in contract["tools"]["ruff"]["cells"]}
    breached = QUALITY_FIXTURE_CELLS[0]
    cells[breached]["ceiling"] = 0

    report = contract_module.check(contract, ["ruff"], {breached})

    entry = next((e for e in report["exceeded"] if e["cell"] == f"ruff/{breached}"), None)
    assert entry is not None, (
        f"a cell measuring above zero was not reported as exceeded: {report['exceeded']}"
    )
    assert entry["files"], (
        "the breach named no files, so the developer is told a number and left "
        "to find the offenders with a second tool"
    )
    counts = [offender["count"] for offender in entry["files"]]
    assert len(counts) > 1 and counts == sorted(counts, reverse=True), (
        "the offenders came back as one file, or out of order. Naming them "
        "largest first is what makes the list somewhere to start rather than "
        f"the same information the count already carried: {entry['files']}"
    )
    for offender in entry["files"]:
        # Asked of the same matcher the measurement used, rather than checked
        # against a path prefix. A prefix test passes for any file under
        # packages/ at all, so it would have accepted a breach reporting
        # another cell's offenders — which is the failure this test is for.
        assert contract_module.cell_matches(PurePosixPath(offender["file"]), breached), (
            f"{offender['file']} was named against {breached}, which it is not in"
        )
    assert entry["further_files"] >= 0


#: What mypy emits when a per-module override section matched no module, taken
#: verbatim from a run against this repository before the three dead sections
#: were removed. A *note*, so the exit status is untouched and nothing fails.
UNUSED_SECTION_NOTE = (
    "pyproject.toml: note: unused section(s): module = "
    "['dataknobs_legacy.*', 'python_nmap.*', 'sklearn.*']\n"
)


def test_an_override_section_that_matches_nothing_is_read_out_of_the_note() -> None:
    """The parse under the guard, driven on the tool's own wording.

    A section matching no module suppresses nothing, and is one of two things:
    a waiver whose spelling is wrong — so the findings it was written for are
    still being reported — or one whose subject is gone. Both read as "handled"
    to anyone looking at the config, and mypy files the observation as a note.
    """
    assert contract_module.unused_config_sections(UNUSED_SECTION_NOTE) == [
        "dataknobs_legacy.*",
        "python_nmap.*",
        "sklearn.*",
    ]
    assert contract_module.unused_config_sections("no such note here\n") == []


def test_a_section_may_match_nothing_when_it_says_why(tmp_path: Path) -> None:
    """The escape hatch, and its limit.

    An ``ignore_missing_imports`` override for a library imported only inside a
    ``try/except ImportError`` legitimately matches nothing in a run that does
    not take that branch. So the failure is an entry that suppresses nothing
    *and says nothing about why*, which is the shape the internal-label
    allowlist settled on: a reason on the line above.

    Two negative halves, and they are what the test is for — an escape hatch
    nobody has bounded excuses everything. The reason must be *adjacent*, or one
    comment covers every entry in the list beneath it; and it must be attached
    to a *module declaration*, or any quoted string under any comment becomes an
    excuse for a module that happens to share its spelling.
    """
    config = tmp_path / "pyproject.toml"
    config.write_text(
        "[tool.mypy]\n"
        "# Where a first-party module resolves from.\n"
        'mypy_path = "nltk.*"\n'
        "\n"
        "[[tool.mypy.overrides]]\n"
        "module = [\n"
        "    # Imported inside a try/except ImportError, so it may match nothing.\n"
        '    "psycopg2.*",\n'
        '    "sklearn.*",\n'
        "]\n"
        "ignore_missing_imports = true\n",
        encoding="utf-8",
    )

    excused = contract_module.excused_config_sections(config)
    assert "psycopg2.*" in excused
    assert "sklearn.*" not in excused, (
        "a section two lines below a comment was excused by it, so one reason "
        "would cover every entry in the list beneath it"
    )
    assert "nltk.*" not in excused, (
        "a commented setting that is not a module declaration supplied an "
        "excuse, so any quoted value under any comment waives the module that "
        "shares its spelling"
    )


def test_a_dead_override_section_fails_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detected is not enforced: the note has to reach the exit status.

    mypy has always reported this and nothing has ever failed on it — three
    sections were dead when this guard was written, two of them waivers for
    findings still being reported. That is this program's own defect class, in
    the configuration the program measures under.

    The note is injected rather than provoked: the repository is clean of them
    now, so the only way to drive the fault is to supply one. What is being
    pinned is the wiring from note to non-zero, not mypy's ability to emit it.
    """
    monkeypatch.setattr(
        contract_module,
        "_run",
        lambda _command: subprocess.CompletedProcess(
            args=_command, returncode=0, stdout=UNUSED_SECTION_NOTE, stderr=""
        ),
    )

    report = contract_module.check(_contract(), ["mypy"])
    dead = sorted(entry["section"] for entry in report["unused_config"])
    assert dead == ["dataknobs_legacy.*", "python_nmap.*", "sklearn.*"], (
        f"a section matching nothing was not reported: {report['unused_config']}"
    )

    scoped = contract_module.check(_contract(), ["mypy"], only={"bin"})
    assert not scoped["unused_config"], (
        "a scoped run reported unused sections. Scoped to part of the tree "
        "almost every section legitimately matches nothing, so treating that as "
        "a fault would make every single-package validation fail."
    )


def test_an_unmeasured_cell_does_not_report_as_a_cleared_one() -> None:
    """A cell no tool reads must not render as one that was read and found clean.

    ``_measured`` derives a cell's total from per-file counts rather than
    keeping a number beside them, and that shape is right — ``dddcb7ba`` moved
    to it because a total kept separately goes stale against the files it
    counts. What it cannot express is that a cell with no per-file counts is two
    different things. ``mypy/conftest.py`` has none because mypy read it and
    found nothing. ``mypy/packages/*/benchmarks`` has none because
    ``mypy_targets`` skips its tier, so nothing read it at all — and the M0
    census put 34 findings in it.

    Rendered as ``{"measured": 0, "ceiling": 0}`` the two are byte-identical.
    That is the defect class ``Measurement``'s own docstring names, one level
    up: an absence rendered as a pass. It becomes urgent rather than latent the
    moment ruff reaches zero, because that is when the artifact starts being
    read as nearly finished.
    """
    report = contract_module.check(
        _contract(), ["mypy"], only={"conftest.py", "packages/*/benchmarks"}
    )
    checked = report["cells"]["mypy/conftest.py"]
    unread = report["cells"]["mypy/packages/*/benchmarks"]

    assert checked != unread, (
        f"a cell nothing measured reports {unread}, identical to {checked} from a "
        "cell mypy read and found clean. One of those zeros is a count and the "
        "other is a silence, and the artifact cannot tell a reader which."
    )
    assert unread["measured"] is None, (
        f"an unmeasured cell reported measured={unread['measured']!r}. A number "
        "there is a measurement nothing took, and it sums into any total a "
        "consumer builds — silently, and low. Absent is the honest value."
    )
    assert checked["measured"] == 0, (
        f"a measured, genuinely clean cell reported {checked['measured']!r}. "
        "Distinguishing the two must not cost the ordinary case its count."
    )
    named = [entry["cell"] for entry in report["unmeasured"]]
    assert "mypy/packages/*/benchmarks" in named, (
        f"the unmeasured cell is distinguishable in its own entry but absent from "
        f"report['unmeasured']: {named}. A reader has to be able to find them "
        "without walking every cell and knowing which tiers mean what."
    )


def test_a_cell_the_contract_calls_unmeasured_still_reports_a_count_it_gets() -> None:
    """Nulling the silence must not null a finding that arrives anyway.

    The tier says the tool is not pointed at the cell; it does not guarantee the
    tool never reports against it. mypy follows imports, and a finding in
    ``packages/<pkg>/tests/`` lands in the population and matches an
    ``unchecked`` cell without ``mypy_targets`` ever having named it. Blanking
    that to ``None`` would suppress a real finding on the authority of a
    declaration the finding just contradicted — and because ``verify`` pins
    these ceilings at zero, the count is also a breach, so suppressing it
    silences a failure rather than merely a number.

    So ``None`` is what a *zero* on an unmeasured cell becomes, never a count.
    """
    contract = _contract()
    cells = contract["tools"]["mypy"]["cells"]
    unchecked = next(c for c in cells if c["tier"] in contract_module._UNMEASURED_TIERS)
    measurement = contract_module.Measurement(
        by_cell={unchecked["path"]: Counter({"packages/data/tests/test_x.py": 3})},
        unattributed=Counter(),
        output="",
    )

    report = _check_with(contract, "mypy", measurement, only={unchecked["path"]})
    entry = report["cells"][f"mypy/{unchecked['path']}"]

    assert entry["measured"] == 3, (
        f"a finding reported against an unmeasured cell came back as "
        f"{entry['measured']!r}. The declaration lost an argument with the tool "
        "and the tool's evidence is what got dropped."
    )
    assert not report["unmeasured"], (
        "a cell that produced findings was still listed as unmeasured, which is "
        f"the contradiction stated as both of its sides at once: {report['unmeasured']}"
    )
    assert [e["cell"] for e in report["exceeded"]] == [f"mypy/{unchecked['path']}"], (
        "a positive count against a ceiling of zero was not reported as a "
        f"breach: {report['exceeded']}. That ceiling is pinned at zero by "
        "verify precisely so this cannot pass quietly."
    )


def test_a_tier_only_silences_a_cell_for_a_tool_that_reads_tiers() -> None:
    """An unmeasured tier means "not looked at" for mypy alone, not for every tool.

    Whether a zero is a silence is a property of the *measurer*, and only
    ``mypy_targets`` consults the tier: it drops ``_UNMEASURED_TIERS`` from the
    directories one pass is pointed at, so a cell there is one mypy was never
    asked about. ``measure_ruff`` tallies a single pass over the whole
    population and ``measure_format`` reads every cell's files, so re-tiering a
    cell changes nothing about whether either looked at it. Its zero stays a
    count.

    Reading the tier alone would answer for all three, and be right about one.
    The wrong answer runs the defect this module just closed backwards — a
    presence rendered as an absence — and it lands in the artifact as a warning
    saying the tier put the cell outside ruff's target set, which ruff, having
    just measured it, contradicts.

    Unreachable today: ``verify`` faults a cell naming a tier its tool has not
    declared, and ruff declares only ``checked``. This pins the guard against
    the day that vocabulary widens, which is the day the guard is wrong and
    nothing else would say so.
    """
    contract = _contract()
    cell = contract["tools"]["ruff"]["cells"][0]
    contract["tools"]["ruff"]["tiers"]["unchecked"] = "measured by nothing"
    cell["tier"] = "unchecked"
    cell["ceiling"] = 0

    # An empty measurement is what measure_ruff actually returns for a clean
    # cell: it tallies per file, so a cell with no findings contributes no
    # entries. The seam supplies the same thing a real pass would.
    report = _check_with(
        contract,
        "ruff",
        contract_module.Measurement(by_cell={}, unattributed=Counter(), output=""),
        only={cell["path"]},
    )
    entry = report["cells"][f"ruff/{cell['path']}"]

    assert entry["measured"] == 0, (
        f"a ruff cell in an unmeasured tier reported measured={entry['measured']!r}. "
        "ruff measured it and found nothing, so that zero is a count. Reporting "
        "it as unknown discards a real measurement on the authority of a tier "
        "the measurer never read."
    )
    assert not report["unmeasured"], (
        f"a cell ruff measured was listed as unmeasured: {report['unmeasured']}. "
        "The warning that list drives says the tier puts the cell outside the "
        "tool's target set, and for ruff there is no such target set to be "
        "outside of."
    )


def test_the_tools_declared_tier_gated_are_the_ones_that_scope_by_tier() -> None:
    """``_TIER_GATED_TOOLS`` has to keep agreeing with what the measurers do.

    It is a one-bit fact per tool, written down because ``check`` cannot ask a
    measurer after the fact which cells it was pointed at. Written down is where
    drift starts, so it is asserted here against the scoping functions
    themselves rather than restated: widening the tier changes what mypy is
    pointed at, and changes nothing for the two measurers handed the whole
    population.

    The day a second measurer starts consulting the tier, this fails — which is
    the notice ``check``'s guard would otherwise not get.
    """
    contract = _contract()
    files = contract_module.tracked_python()

    assert sorted(contract_module._TIER_GATED_TOOLS) == ["mypy"], (
        "the declared set changed; the assertions below describe mypy, ruff and "
        "format specifically and need rewriting alongside it."
    )

    mypy_cells = contract["tools"]["mypy"]["cells"]
    narrow = set(contract_module.mypy_targets(mypy_cells, None, False, files))
    wide = set(contract_module.mypy_targets(mypy_cells, None, True, files))
    assert narrow < wide, (
        "mypy is declared tier-gated, but widening the run to unmeasured tiers "
        f"did not widen its target set ({len(narrow)} of {len(wide)}). Either it "
        "stopped consulting the tier, or the contract has no unmeasured cell "
        "left — and in the second case the declaration is what to revisit."
    )

    for tool in sorted(set(contract_module.MEASURERS) - contract_module._TIER_GATED_TOOLS):
        cells = json.loads(json.dumps(contract["tools"][tool]["cells"]))
        before = contract_module._files_in(cells, files, {cells[0]["path"]})
        tallied = contract_module._tally(cells, [str(path) for path in files])
        for cell in cells:
            cell["tier"] = "unchecked"
        assert contract_module._files_in(cells, files, {cells[0]["path"]}) == before, (
            f"{tool} is not declared tier-gated, but re-tiering its cells changed "
            "which files it reads. Its zeros over an unmeasured tier are no "
            "longer counts, and check's guard has to learn that."
        )
        assert contract_module._tally(cells, [str(path) for path in files]) == tallied, (
            f"{tool} is not declared tier-gated, but re-tiering its cells changed "
            "how findings attribute to them."
        )


def test_the_formatter_measurer_refuses_a_file_it_could_not_read() -> None:
    """A file the formatter cannot open must not be counted as a formatted one.

    ``ruff format --check`` reports an unreadable path as an ordinary result
    with ``code: "io"`` and exits 2. Under the text parse this measurer used to
    run, that message carries no ``--> path:line`` — so the file contributed
    nothing to the tally and the cell holding it measured *lower*, which is the
    direction a ratchet cannot survive. Zero findings and zero files read are
    the same report, and the second one is the one that reads as success.

    The same shape as ``measure_ruff``'s decode guard, which was already here:
    the formatter measurer just had no equivalent because its parse could not
    fail loudly, only quietly.
    """
    contract = _contract()
    faults = contract_module.measure_format

    try:
        faults(contract, [PurePosixPath("no/such/path.py")])
    except SystemExit as exit_:
        assert "no/such/path.py" in str(exit_), (
            f"the measurer refused, but did not name the file it could not read: {exit_}"
        )
    else:
        raise AssertionError(
            "measure_format read a file that does not exist and reported a "
            "measurement anyway. An unreadable file lowers the cell it belongs "
            "to, so a broken measurer reads as a cleaner tree."
        )


def test_the_formatter_measurer_refuses_output_it_could_not_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-JSON on stdout means nothing was measured, not that nothing was found.

    The floor under the check above. That one covers a fault ruff reports *in*
    JSON; this covers ruff not emitting JSON at all — a flag rename, a crash, a
    wrapper writing to stdout. Both collapse to an empty tally, and an empty
    tally is indistinguishable from a clean tree at the point it is compared
    against a ceiling of zero, which is what every ``format`` cell becomes.

    The invocation is replaced rather than the tool, because there is no input
    that makes a working ruff emit something other than JSON — the fault being
    driven is ruff not behaving like ruff, so it has to be injected.
    """
    completed = subprocess.CompletedProcess(
        args=["ruff"], returncode=1, stdout="not json", stderr=""
    )
    monkeypatch.setattr(contract_module, "_run", lambda _command: completed)

    try:
        contract_module.measure_format(_contract(), [PurePosixPath("conftest.py")])
    except SystemExit as exit_:
        assert "JSON" in str(exit_), f"refused without saying the parse failed: {exit_}"
    else:
        raise AssertionError(
            "measure_format accepted unparseable output and returned a "
            "measurement, so a broken formatter invocation reads as a clean tree."
        )


def test_the_linter_measurer_refuses_a_config_ruff_would_not_load(tmp_path: Path) -> None:
    """A configuration ruff rejects must not read as a tree with nothing wrong.

    ruff exits 2 on a config it cannot load — an unknown code in ``select``, a
    malformed table — having opened no file, and writes nothing to stdout. The
    measurer read ``result.stdout or "[]"``, so that emptiness parsed cleanly
    into an empty finding list and **every ruff cell measured zero**: a green
    ``check``, and an ``update-baseline`` that would write those zeroes down as
    the new ceilings.

    Driven through the real ruff rather than an injected result, because ruff
    genuinely does this and the assertion is worth no more than the claim that
    it does. The route is also the work: clearing a cell to the point where it
    can be linted by default means editing this config, which is precisely when
    a rejected one reaches the measurer.
    """
    rejected = tmp_path / "rejected.toml"
    rejected.write_text('[tool.ruff.lint]\nselect = ["NOSUCHRULE999"]\n', encoding="utf-8")
    contract = _contract()
    contract["tools"]["ruff"]["config"] = str(rejected)

    with pytest.raises(SystemExit) as refusal:
        contract_module.measure_ruff(contract, [PurePosixPath("conftest.py")])

    assert "exited 2" in str(refusal.value), (
        f"the measurer refused, but not for the reason it should have: {refusal.value}"
    )


def test_the_linter_measurer_refuses_a_file_it_could_not_read() -> None:
    """An unreadable file must not stand in for the findings it holds.

    ruff reports one as an ordinary JSON entry — ``code: "E902"``,
    ``name: "io-error"`` — and exits 1, exactly as it does with real findings.
    So the file contributes **one** finding instead of however many it holds.

    That is the formatter's ``io`` fault with the sign changed and it is the
    worse of the two. There, the entry vanishes and the cell measures lower;
    here it is replaced, and a file with twenty findings measuring one reads as
    an ordinary small backlog rather than as an absence — at a ceiling of 1,685,
    as nothing at all.
    """
    with pytest.raises(SystemExit) as refusal:
        contract_module.measure_ruff(_contract(), [PurePosixPath("no/such/path.py")])

    assert "no/such/path.py" in str(refusal.value), (
        f"the measurer refused, but did not name the file it could not read: {refusal.value}"
    )


def test_the_linter_measurer_refuses_a_status_its_output_contradicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit 0 with findings, or exit 1 without, means two runs are being read.

    Injected, because no input makes a working ruff answer this way — the fault
    being driven is ruff not behaving like ruff, or a wrapper interposing on one
    of the two channels. Which is the case worth holding: the count and the
    status are read from one invocation *by assumption*, and this is the only
    thing that checks the assumption.
    """
    completed = subprocess.CompletedProcess(
        args=["ruff"],
        returncode=0,
        stdout='[{"filename": "conftest.py", "code": "F401"}]',
        stderr="",
    )
    monkeypatch.setattr(contract_module, "_run", lambda _command: completed)

    with pytest.raises(SystemExit) as refusal:
        contract_module.measure_ruff(_contract(), [PurePosixPath("conftest.py")])

    assert "disagree" in str(refusal.value), (
        f"refused without saying the status and the output disagree: {refusal.value}"
    )


def test_the_linter_measurer_reports_a_path_less_finding_rather_than_dying_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finding with a null path belongs in ``unattributed``, which is reported.

    ``.get("filename", "")`` returns the default only when the key is *absent*;
    a key carrying ``null`` comes back as ``None`` and reaches ``Path(None)``,
    which raises inside the measurement rather than reporting anything about it.

    The remedy is not to drop the finding — that is the silent direction this
    file exists to refuse. An empty name tallies as unattributed, where
    ``check`` warns that something was reported and counted against no cell.
    """
    completed = subprocess.CompletedProcess(
        args=["ruff"],
        returncode=1,
        stdout='[{"filename": null, "code": "F401"}]',
        stderr="",
    )
    monkeypatch.setattr(contract_module, "_run", lambda _command: completed)

    measurement = contract_module.measure_ruff(_contract(), [PurePosixPath("conftest.py")])

    assert dict(measurement.unattributed) == {"<unnamed>": 1}, (
        "a path-less finding was not carried through to unattributed, where it "
        f"gets reported: {measurement.unattributed}"
    )


def test_the_contract_is_an_input_the_artifacts_are_hashed_over() -> None:
    """Editing a ceiling must invalidate the artifacts that were checked against it.

    The contract decides a recorded check's verdict without being code, which is
    exactly the shape that left ``.gitignore`` and ``bin/internal-label-allowlist.txt``
    outside every hash scope: the script was hashed and the data it consults was
    not, so editing one moved a recorded verdict with every stored hash intact.
    """
    changed_packages = load_bin_module("changed-packages")
    declared = {
        entry for entries in changed_packages.WORKSPACE_QUALITY_INPUTS.values() for entry in entries
    }
    relative = CONTRACT.relative_to(ROOT).as_posix()
    assert relative in declared, (
        f"{relative} is in no workspace hash scope, so raising a ceiling leaves "
        "every stored hash intact and CI accepts artifacts produced under the "
        "old one. Declare it in bin/changed-packages.py."
    )


#: One case per verdict ``explain`` can return, each pinned to a code whose real
#: disposition is known and stated. Written against the live configuration on
#: purpose: a synthetic config would prove the branch works and say nothing about
#: whether the branch is reachable, and "reachable" is the whole claim — the
#: command exists so a worker can ask instead of guessing.
EXPLAIN_CASES = (
    # Pinned to a decline that cannot be worked off rather than one that can.
    # This case used to name RUF012, an unargued `provisional` entry — and
    # `provisional` is the category whose stated target is zero, so the example
    # was guaranteed to stop being one. D203 is declined because D211 enforces
    # its exact opposite and is enabled: the two are mutually exclusive by
    # construction, so no amount of fixing makes this verdict change.
    ("D203", None, "declined globally", "declined repo-wide; D211 enforces the opposite"),
    ("PGH004", None, "not selected", "the PGH family is in no select entry"),
    ("F841", "packages/llm/examples/fsm_conversation.py", "reported", "F is selected, undeclined"),
    (
        "SIM115",
        "packages/utils/src/dataknobs_utils/xml_utils.py",
        "waived for this file",
        "one of the twelve per-file SIM115 waivers",
    ),
)


@pytest.mark.parametrize(("code", "path", "verdict", "why"), EXPLAIN_CASES, ids=str)
def test_explain_returns_the_right_verdict(
    code: str, path: str | None, verdict: str, why: str
) -> None:
    explanation = contract_module.explain_code(code, path)
    assert explanation.verdict == verdict, (
        f"explain {code} {path or ''} said {explanation.verdict!r}, expected {verdict!r} — {why}"
    )


def test_explain_reads_the_selected_set_from_ruff_rather_than_the_family_list() -> None:
    """The re-implementation this command deliberately does not contain.

    ``select`` lists the legacy selector ``TCH`` while the rules it enables are
    spelled ``TC00x``. A prefix match over the declared families — the obvious
    implementation, and the first one written here — reports those as unselected,
    inventing a reason the configuration does not hold.

    ``TC004`` is the case that distinguishes the two, and finding it took a
    second pass: this test first used ``TC002``, which is *declined*, so both
    implementations answered "declined globally" and the assertion passed
    without separating anything. The code has to be one the family enables and
    the ignore list does not decline, or the check is unfalsifiable — which is
    the shape this whole leg objects to.
    """
    lint = _load_toml(ROOT / "pyproject.toml")["tool"]["ruff"]["lint"]
    probe = "TC004"
    assert probe not in set(lint["ignore"]), (
        f"{probe} is now declined, so it can no longer separate asking ruff from "
        "deriving the answer — both would say 'declined globally'. Find another."
    )
    assert not any(probe.startswith(family) for family in lint["select"]), (
        f"{probe} now matches a declared family by prefix, so the derived "
        "implementation would get it right too; find another"
    )

    assert contract_module.explain_code(probe).verdict == "reported", (
        f"{probe} is enabled by the TCH selector and declined nowhere, so it is "
        "reported. Answering otherwise means the selected set is being derived "
        "from the family list rather than read from ruff."
    )


def test_explain_never_fails_the_caller() -> None:
    """A lookup, not a check. It must not become something a script can fail on.

    ``check`` owns the pass/fail role, and a second command that also exits
    non-zero on a condition is how a gate ends up with two verdicts that can
    disagree. Every verdict here exits 0, including the ones that mean "your
    finding is real".
    """
    for code, path, _verdict, _why in EXPLAIN_CASES:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bin" / "quality-contract.py"),
                "explain",
                code,
                *([path] if path else []),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"explain {code} exited {result.returncode}: {result.stderr}"
        assert result.stdout.strip(), f"explain {code} printed nothing"


def test_the_decline_measurer_refuses_a_config_ruff_would_not_load(tmp_path: Path) -> None:
    """The fourth site of a guard this file's own docstring says was forgotten once.

    ``_refuse_non_verdict`` exists because a ruff that exits 2 has opened no file
    and written nothing to stdout, and every parse downstream turns that
    emptiness into a measured zero. It was written for the formatter, copied to
    the linter, missing from the type checker — and missing again here, in a
    measurer added after the guard and beside three that call it.

    The zeroes are not inert. This measurer is what supplies the figure a
    ``provisional`` entry carries, and ``0 findings`` is indistinguishable from
    the entry whose backlog has actually been cleared — the one state the whole
    category exists to make visible. A broken invocation would read as the goal.

    Driven through the real ruff, like its sibling: the claim is that ruff does
    this, and an injected result would pin the claim to itself.
    """
    rejected = tmp_path / "rejected.toml"
    rejected.write_text('[tool.ruff.lint]\nselect = ["NOSUCHRULE999"]\n', encoding="utf-8")

    with pytest.raises(SystemExit) as refusal:
        contract_module.measure_declines(["RUF012"], config=rejected)

    assert "exited 2" in str(refusal.value), (
        f"the measurer refused, but not for the reason it should have: {refusal.value}"
    )


def test_a_waiver_naming_a_family_does_not_reach_another_linter() -> None:
    """The selector resolution ``explain`` deliberately does not re-implement.

    ``enabled_rules`` refuses to derive ruff's selector resolution, having
    measured what that costs. ``waiver_covers`` derived it anyway one scope down,
    as ``code.startswith(named)``, and was wrong in both directions.

    ``packages/legacy/*`` waives ``["D", "N", "UP"]``. ``N`` is pep8-naming;
    ``NPY`` is numpy, a different linter, and ``NPY`` is in ``select``. So a
    worker who hit ``NPY002`` in that package and asked was told the finding was
    waived — the one direction the function's own docstring argued it could not
    fail in, since it claimed to err toward *reported*.

    The other direction is ``PL``, which names ``PLC0415`` to ruff and to no
    letter-prefix rule, because the pylint selector spans four sub-families. No
    waiver here uses it today, which is why it is asserted on a synthetic one:
    the next waiver written with a compound family would silently under-report.
    """
    legacy = contract_module.PerFileWaiver(
        "packages/legacy/*", ("D", "N", "UP"), "Relax rules for legacy code"
    )
    target = "packages/legacy/src/dataknobs/__init__.py"

    assert not contract_module.waiver_covers(legacy, "NPY002", target), (
        "a waiver of N (pep8-naming) reported itself as covering NPY002 (numpy). "
        "NPY is selected, so this tells a worker to leave a reported finding alone."
    )
    assert contract_module.waiver_covers(legacy, "N802", target), (
        "the waiver stopped covering N802, which it does name — the fix has "
        "narrowed past the thing it was fixing"
    )

    compound = contract_module.PerFileWaiver(
        "packages/legacy/*", ("PL",), "synthetic: a compound family"
    )
    assert contract_module.waiver_covers(compound, "PLC0415", target), (
        "PL names PLC0415 to ruff. A letter-prefix rule cannot see that, so this "
        "is the direction a re-implementation under-reports in."
    )

    assert not contract_module.waiver_covers(legacy, "N802", "packages/data/src/x.py"), (
        "the pattern half stopped being consulted, so a waiver reaches files "
        "outside the directory it was written for"
    )


def test_the_waiver_lookup_asks_ruff_under_this_repository_s_settings() -> None:
    """Why the resolution keeps the config instead of running ``--isolated``.

    ``--isolated`` is the tempting way to ask "what does this family name",
    since the question looks like a property of ruff's registry. It is not:
    ``pydocstyle.convention`` and ``target-version`` both remove codes a family
    would otherwise name, and the two answers differ by ten rules for the one
    family-shaped waiver this config holds.

    The difference is not neutral. The isolated set is the *wider* one, and a
    waiver claiming to cover a code it does not is the direction that tells a
    worker not to look — so the config-free answer is wrong in exactly the way
    the fix above was written to stop.
    """
    codes = ("D", "N", "UP")
    with_settings = contract_module.selector_rules(codes)
    isolated = contract_module._run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "--isolated",
            "--select",
            ",".join(codes),
            "--show-settings",
            str(ROOT / "pyproject.toml"),
        ]
    )
    block = contract_module._ENABLED_BLOCK_RE.search(isolated.stdout)
    assert block is not None, "the isolated probe resolved no rule set, so it compares nothing"
    without_settings = frozenset(contract_module._RULE_CODE_RE.findall(block.group(1)))

    assert with_settings < without_settings, (
        "resolving under this repository's settings no longer narrows the family, "
        "so this test no longer demonstrates why the config is kept. Check "
        "whether pydocstyle.convention or target-version has been dropped."
    )


def test_the_audit_accounts_for_every_decline() -> None:
    """The table has to cover the list, or it is a curated subset like its predecessor.

    The prose page this replaces enumerated 35 of 83 declines and invented one.
    A summary whose rows do not add to the population is the same artifact in a
    new format.
    """
    audit = contract_module.decline_audit()
    rows = [entry for group in audit["by_category"].values() for entry in group]
    assert len(rows) == audit["total"]
    declared = _load_toml(ROOT / "pyproject.toml")["tool"]["ruff"]["lint"]["ignore"]
    assert {entry.code for entry in rows} == set(declared)
    assert "uncategorized" not in audit["by_category"], (
        "the audit grouped a decline under 'uncategorized' — test_lint_policy "
        "should have failed first; one of the two is not reading the config"
    )


# --------------------------------------------------------------------------
# The ledger — reading the declaration's own history
# --------------------------------------------------------------------------
#
# Every guard below drives a purpose-built history rather than this
# repository's, because the three events the ledger exists to distinguish have
# never happened here: no ceiling has ever been raised, no commit has ever
# carried a leg trailer, and no trailer has ever been misspelled. A reader that
# can only be pointed at its own history is tested against none of them.


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in ``repo`` with identity supplied, so it needs no global config."""
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Ledger Test",
            "-c",
            "user.email=ledger@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )


def _declaration(ceilings: dict[str, int]) -> str:
    """A minimal declaration holding one tool's cells at the given ceilings."""
    return json.dumps(
        {
            "version": 1,
            "about": "a purpose-built declaration",
            "tools": {
                "mypy": {
                    "cells": [
                        {"path": path, "tier": "transitional", "ceiling": ceiling}
                        for path, ceiling in sorted(ceilings.items())
                    ]
                }
            },
        },
        indent=2,
    )


def _commit(repo: Path, ceilings: dict[str, int], subject: str, leg: str | None = None) -> None:
    """Write the declaration at these ceilings and commit it."""
    written = repo / ".dataknobs" / "quality-contract.json"
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text(_declaration(ceilings), encoding="utf-8")
    message = subject if leg is None else f"{subject}\n\nQuality-Leg: {leg}\n"
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "history"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    return repo


def _ledger(repo: Path) -> dict[str, Any]:
    recorded: dict[str, Any] = contract_module.ledger("mypy", "HEAD", repo)
    return recorded


def test_a_redrawn_cell_set_is_not_counted_as_findings_cleared() -> None:
    """The defect the obvious implementation has, in the shape it really took.

    Summing every ceiling and subtracting the previous sum is the reading this
    command was sketched as, and it is wrong whenever the cell set moves. This
    is the movement that actually occurred here: a glob cell covering the
    per-package test trees was replaced by one cell per package.

    By sum the ceilings fell by 255. By cell — counting only what is present in
    both revisions, which is the only population where "the same thing measured
    twice" is even meaningful — 13 findings were cleared. The other 242 is the
    redraw, and a ledger that reports it as progress is reporting work nobody
    did, in the direction that flatters the programme it is evidence for.
    """
    before = {"packages/*/tests": 1685, "packages/data/src": 50}
    after = {"packages/bots/tests": 700, "packages/common/tests": 743, "packages/data/src": 37}

    movement = contract_module._movement(before, after)

    assert movement.cleared == 13, "only the cell present in both revisions moved"
    assert movement.added == 1443
    assert movement.removed == 1685
    assert movement.structural
    naive = sum(before.values()) - sum(after.values())
    assert naive == 255, "the sum-and-subtract reading, kept here as the thing being refused"
    assert movement.cleared != naive


def test_a_raised_ceiling_is_reported_apart_from_the_drain(tmp_path: Path) -> None:
    """A regression is never netted against progress.

    Summing signed deltas would let a cell that gained 40 findings and one that
    lost 40 report as a quiet zero, and "no cell ends higher than it started" is
    a safety criterion this is supposed to be able to answer.
    """
    repo = _repo(tmp_path)
    _commit(repo, {"packages/data/src": 100, "packages/fsm/src": 100}, "declare")
    _commit(repo, {"packages/data/src": 60, "packages/fsm/src": 100}, "clear forty")
    _commit(repo, {"packages/data/src": 60, "packages/fsm/src": 140}, "raise forty")

    recorded = _ledger(repo)

    assert recorded["cleared"] == 40
    assert recorded["raised"] == 40, "the raise is reported, not subtracted from the drain"
    assert recorded["standing"] == 200
    assert recorded["opened"]["total"] == 200


def test_a_leg_trailer_moves_its_merge_out_of_the_convention_population(tmp_path: Path) -> None:
    """The split the trailer exists to make, and it is read over the range.

    The trailer sits on the commit that did the clearing, which is inside the
    branch; the merge commit carries the branch's subject and none of its
    trailers. Reading trailers off the step itself finds nothing on every pull
    request ever merged, which is every leg there will ever be.
    """
    repo = _repo(tmp_path)
    _commit(repo, {"packages/data/src": 100, "packages/fsm/src": 100}, "declare")
    _commit(repo, {"packages/data/src": 90, "packages/fsm/src": 100}, "ordinary work")

    _git(repo, "checkout", "-b", "drain")
    _commit(
        repo, {"packages/data/src": 90, "packages/fsm/src": 40}, "drain fsm", leg="packages/fsm/src"
    )
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "drain", "-m", "Merge pull request #1 from drain")

    recorded = _ledger(repo)

    assert recorded["cleared"] == 70
    assert recorded["populations"]["leg"] == {"steps": 1, "cleared": 60}
    assert recorded["populations"]["convention"] == {"steps": 1, "cleared": 10}
    assert not recorded["faults"]


def test_a_trailer_naming_no_declared_cell_is_reported(tmp_path: Path) -> None:
    """A trailer naming no cell, and the narrower harm it actually does.

    This test was written asserting the wrong thing, and fixing it is the point
    worth recording. The check was proposed on the grounds that a typo would
    drop the commit out of the leg population and file it as incidental
    clearing. It does not: **presence is the discriminator**, so a misspelled
    trailer is still a leg and every total below is right.

    What the typo destroys is the attribution. ``packages/fsm`` for
    ``packages/fsm/src`` records nothing about which cell the scheduled work
    went to, while reading exactly like a record — and because the counts stay
    correct, nothing about the report invites anyone to look.
    """
    repo = _repo(tmp_path)
    _commit(repo, {"packages/fsm/src": 100}, "declare")
    _commit(repo, {"packages/fsm/src": 40}, "drain fsm", leg="packages/fsm")

    recorded = _ledger(repo)

    assert len(recorded["faults"]) == 1
    assert "packages/fsm" in recorded["faults"][0]
    assert "does not name as a cell" in recorded["faults"][0]
    # Still a leg, and still counted. The fault reports an unusable attribution;
    # it neither reclassifies the commit nor discards 60 findings really cleared.
    assert recorded["cleared"] == 60
    assert recorded["populations"]["leg"]["cleared"] == 60
    assert recorded["populations"]["convention"]["cleared"] == 0


def test_a_correctly_spelled_trailer_raises_no_fault(tmp_path: Path) -> None:
    """The other half of the check above, without which it could pass by always firing."""
    repo = _repo(tmp_path)
    _commit(repo, {"packages/fsm/src": 100}, "declare")
    _commit(repo, {"packages/fsm/src": 40}, "drain fsm", leg="packages/fsm/src")

    recorded = _ledger(repo)

    assert recorded["faults"] == []
    assert recorded["populations"]["leg"]["cleared"] == 60


def test_two_clearings_in_one_pull_request_are_one_paying_merge(tmp_path: Path) -> None:
    """The unit is a merge, and reading per commit inflates the paying rate.

    Against this repository's real history the two readings give 11 paying
    events out of 67 and 21 out of 66 — and the fraction paying is the figure
    the whole cost argument rests on. Counting per commit double-counts exactly
    the population whose rate the number describes.
    """
    repo = _repo(tmp_path)
    _commit(repo, {"packages/data/src": 100}, "declare")

    _git(repo, "checkout", "-b", "work")
    _commit(repo, {"packages/data/src": 90}, "clear ten")
    _commit(repo, {"packages/data/src": 70}, "clear twenty more")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "work", "-m", "Merge pull request #1 from work")

    recorded = _ledger(repo)

    assert recorded["cleared"] == 30, "both commits' clearing is counted"
    assert recorded["paying"] == 1, "but they arrived in one pull request"
    assert len(recorded["steps"]) == 1


def test_the_ledger_reproduces_this_repository_s_own_drain() -> None:
    """The integration case: the real history, read end to end.

    The purpose-built histories above pin each rule in isolation; this asserts
    the walk survives contact with merges, a declaration that arrives partway
    through a long history, and a cell set that was redrawn twice.
    """
    recorded = contract_module.ledger("mypy")

    assert recorded["opened"] is not None
    assert recorded["opened"]["total"] > recorded["standing"], "the backlog has fallen"
    assert recorded["cleared"] == recorded["opened"]["total"] - recorded["standing"], (
        "with no ceiling ever raised and no mypy cell added or removed, the drain "
        "must reconcile exactly against the opening and standing balances"
    )
    assert recorded["raised"] == 0
    assert recorded["paying"] <= recorded["window"]
    assert recorded["faults"] == [], (
        "a trailer in this repository names a cell that is not declared"
    )

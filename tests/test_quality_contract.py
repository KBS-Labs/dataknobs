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
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from tests._workspace import ROOT, biggest_ruff_cells, load_bin_module, rel

CONTRACT = ROOT / ".dataknobs" / "quality-contract.json"
TOOL = ROOT / "bin" / "quality-contract.py"

contract_module = load_bin_module("quality-contract")


def _contract() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return loaded


def _cells(tool: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = _contract()["tools"][tool]["cells"]
    return cells


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


def test_a_checked_cell_is_one_the_linter_actually_reaches() -> None:
    """The tier has to agree with what ``bin/validate.sh`` does, in both directions.

    ``checked`` says the linter reads these files and finds nothing; ``deferred``
    says it does not read them yet. Neither claim is worth anything on its own,
    because the cheapest way to satisfy a coverage rule is to drop a directory
    from the target set and re-file it as deferred — coverage gone, both halves
    of the declaration still internally consistent.

    So each tier is compared against the target set the script resolves. A
    ``checked`` cell outside it is a ceiling of zero over files nothing lints; a
    ``deferred`` cell inside it is a backlog declared for files that are already
    clean.
    """
    targets = _validate_targets()
    assert targets, "bin/validate.sh --print-targets resolved nothing"

    unreached = sorted(
        cell["path"]
        for cell in _cells("ruff")
        if cell["tier"] == "checked" and not _covered_by_targets(cell["path"], targets)
    )
    assert not unreached, (
        f"{rel(CONTRACT)} calls {unreached} checked with a ceiling of zero, but "
        "bin/validate.sh does not lint them — so the zero is a measurement of "
        "nothing. Restore the target, or move the cell to the deferred tier "
        "with the count that makes deferring honest."
    )

    contradicted = sorted(
        cell["path"]
        for cell in _cells("ruff")
        if cell["tier"] == "deferred" and _covered_by_targets(cell["path"], targets)
    )
    assert not contradicted, (
        f"{rel(CONTRACT)} defers {contradicted}, but bin/validate.sh lints them. "
        "Either the entry is obsolete and should be promoted to checked, or a "
        "default target was removed and should come back."
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

    Both directions in one test because it is one decision. Driven over the real
    measurement rather than a stubbed one: what is being pinned is what the
    command does to a ceiling on this repository, and a fake measurement would
    pin it against a number that no tool produces.
    """
    contract = _contract()
    ruff_cells = {cell["path"]: cell for cell in contract["tools"]["ruff"]["cells"]}

    inflated, deflated = biggest_ruff_cells(contract, 2)
    true_ceilings = {
        inflated: ruff_cells[inflated]["ceiling"],
        deflated: ruff_cells[deflated]["ceiling"],
    }

    ruff_cells[inflated]["ceiling"] = true_ceilings[inflated] + 500
    ruff_cells[deflated]["ceiling"] = max(true_ceilings[deflated] - 5, 0)

    destination = tmp_path / "quality-contract.json"
    lowered, exceeded = contract_module.update_baseline(contract, ["ruff"], destination)

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
    contract = _contract()
    ruff_cells = {cell["path"]: cell for cell in contract["tools"]["ruff"]["cells"]}

    named, untouched = biggest_ruff_cells(contract, 2)
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
    contract = _contract()
    ruff_cells = {cell["path"]: cell for cell in contract["tools"]["ruff"]["cells"]}

    named, _ = biggest_ruff_cells(contract, 2)
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


def _with_cell(tool: str, **overrides: Any) -> dict[str, Any]:
    """The real contract with one extra cell, for exercising a single fault.

    Built from the real declaration rather than a minimal fake so the fault
    under test is the *only* thing wrong: a hand-built contract fails totality
    on all 1,470 tracked files at once, and every assertion below would then
    pass on a fault it did not cause.
    """
    contract = _contract()
    cell = {"path": "packages/*/nowhere", "tier": "deferred", "ceiling": 0, "reason": "x"}
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


def test_a_breached_ceiling_names_the_files_that_breached_it() -> None:
    """A count says *whether*; it cannot say *what*.

    The first ceiling this mechanism ever broke reported ``21 findings against
    20 allowed`` over a directory holding 21 files, and finding the
    twenty-first took a separate script. A failure nobody can act on gets
    suppressed as surely as one that fires spuriously — the same subject as G4,
    approached from the other side — so an exceeded cell carries the file names
    out with it.

    Driven over the real measurement with one ceiling pushed below what the
    tree holds, because what is being pinned is what a developer sees when a
    real cell goes over.
    """
    contract = _contract()
    cells = {cell["path"]: cell for cell in contract["tools"]["ruff"]["cells"]}
    (breached,) = biggest_ruff_cells(contract, 1)
    cells[breached]["ceiling"] = 0

    report = contract_module.check(contract, ["ruff"])

    entry = next((e for e in report["exceeded"] if e["cell"] == f"ruff/{breached}"), None)
    assert entry is not None, (
        f"a cell measuring above zero was not reported as exceeded: {report['exceeded']}"
    )
    assert entry["files"], (
        "the breach named no files, so the developer is told a number and left "
        "to find the offenders with a second tool"
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
    it does. The route is also the work: promoting a cell out of the deferred
    tier means editing this config, which is precisely when a rejected one
    reaches the measurer.
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

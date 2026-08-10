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

from tests._workspace import ROOT, load_bin_module, rel, tracked_python_files

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
        f"{rel(CONTRACT)} does not describe the repository:\n"
        f"{result.stdout}{result.stderr}"
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


def test_no_cell_names_a_part_of_the_tree_that_is_gone() -> None:
    """A cell matching nothing leaves the reader believing in a gap that closed.

    The stale half of the rule the previous declaration enforced, kept because
    totality does not imply it: a cell can match zero files while every file
    still matches some other cell, so the partition stays valid and the entry
    stays wrong.
    """
    files = [PurePosixPath(name) for name in tracked_python_files()]
    assert files, "git tracks no Python at all — this guard would check nothing"

    stale = sorted(
        f"{tool}/{cell['path']}"
        for tool in _contract()["tools"]
        for cell in _cells(tool)
        if not any(contract_module.cell_matches(path, cell["path"]) for path in files)
    )
    assert not stale, (
        f"{rel(CONTRACT)} declares cells matching no tracked Python file: "
        f"{stale}. Drop them — a gap that no longer exists reads as one that does."
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
    """
    targets = _validate_targets()
    cells = _cells("mypy")

    read_anyway = sorted(
        cell["path"]
        for cell in cells
        if cell["tier"] == "unchecked" and _covered_by_targets(cell["path"], targets)
    )
    assert not read_anyway, (
        f"{rel(CONTRACT)} calls {read_anyway} unchecked by mypy, but they are "
        "inside bin/validate.sh's target set, so mypy reads them. Give each a "
        "tier and a ceiling."
    )

    unread = sorted(
        cell["path"]
        for cell in cells
        if cell["tier"] != "unchecked" and not _covered_by_targets(cell["path"], targets)
    )
    assert not unread, (
        f"{rel(CONTRACT)} gives {unread} a mypy ceiling, but bin/validate.sh "
        "does not type-check them, so the number is a measurement of nothing."
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

    inflated = "packages/*/tests"
    deflated = "packages/*/examples"
    true_ceilings = {inflated: ruff_cells[inflated]["ceiling"], deflated: ruff_cells[deflated]["ceiling"]}

    ruff_cells[inflated]["ceiling"] = true_ceilings[inflated] + 500
    ruff_cells[deflated]["ceiling"] = max(true_ceilings[deflated] - 5, 0)

    destination = tmp_path / "quality-contract.json"
    changed = contract_module.update_baseline(contract, ["ruff"], destination)

    assert ruff_cells[inflated]["ceiling"] == true_ceilings[inflated], (
        f"an inflated ceiling on {inflated} was not lowered to what the tree "
        f"measures; --update-baseline reported {changed}"
    )
    assert ruff_cells[deflated]["ceiling"] == max(true_ceilings[deflated] - 5, 0), (
        f"{deflated} had a ceiling below its measurement and --update-baseline "
        "raised it. Re-running a command must never enlarge a backlog: that is "
        "an argument to have in a pull request, not a side effect of a rerun."
    )
    assert destination.is_file(), "a lowered ceiling was reported but never written"


def test_the_contract_is_an_input_the_artifacts_are_hashed_over() -> None:
    """Editing a ceiling must invalidate the artifacts that were checked against it.

    The contract decides a recorded check's verdict without being code, which is
    exactly the shape that left ``.gitignore`` and ``bin/internal-label-allowlist.txt``
    outside every hash scope: the script was hashed and the data it consults was
    not, so editing one moved a recorded verdict with every stored hash intact.
    """
    changed_packages = load_bin_module("changed-packages")
    declared = {
        entry
        for entries in changed_packages.WORKSPACE_QUALITY_INPUTS.values()
        for entry in entries
    }
    relative = CONTRACT.relative_to(ROOT).as_posix()
    assert relative in declared, (
        f"{relative} is in no workspace hash scope, so raising a ceiling leaves "
        "every stored hash intact and CI accepts artifacts produced under the "
        "old one. Declare it in bin/changed-packages.py."
    )

#!/usr/bin/env python3
"""Check the repository against the coverage-and-strictness contract.

``.dataknobs/quality-contract.json`` declares, for each of three tools, which
files it covers and how far from clean each part of the tree is allowed to be.
This measures the tree and compares.

Two properties make that declaration a ratchet rather than a list of excuses,
and both are checked here rather than described:

**Totality.** Every tracked first-party ``*.py`` lands in exactly one cell per
tool. A file in no cell is one nobody decided about, and it is the state ``bin/``
was in for as long as this repository has had a linter — outside every lint
invocation, with nothing saying so. A file in two cells is a decision that
contradicts itself, and the resolution would be whichever cell the matcher
happened to try first.

**Ceilings are compared, not read.** The previous declaration recorded its
counts in comment prose, which is enforced in one direction only: an entry
matching nothing failed, while ``241 findings`` stayed green at 400. A number
nobody compares is a number that stops being true without anyone finding out.

The measurement is one invocation per tool over the whole population, bucketed
afterwards, rather than one invocation per cell: thirty subprocesses would make
the check cost more than the tools it runs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_CONTRACT = _ROOT / ".dataknobs" / "quality-contract.json"

#: Tiers whose files no tool reads. Their ceiling is not a backlog and must stay
#: zero — a positive one would claim a measurement that nothing takes.
_UNMEASURED_TIERS = frozenset({"unchecked"})


def load_contract(path: Path = _CONTRACT) -> dict[str, Any]:
    """The declaration, or a clear error naming the file rather than a traceback."""
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    except FileNotFoundError as exc:
        raise SystemExit(f"{path}: the quality contract is missing: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: the quality contract is not readable JSON: {exc}") from exc


def tracked_python() -> list[PurePosixPath]:
    """Every ``*.py`` git keeps, as repo-relative paths.

    Asking git rather than walking the tree keeps an editor backup, a stray
    ``.orig`` or a macOS ``.DS_Store`` from joining the population on one
    machine and not another — which for a set that decides a pass/fail verdict
    would mean a developer and CI checking different files.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [PurePosixPath(name) for name in listing.split("\0") if name]


def cell_matches(path: PurePosixPath, pattern: str) -> bool:
    """Whether a file is, or lives under, the directory a cell names.

    Segment by segment from the repository root, with the pattern's segments
    required to be a prefix of the path's. That covers a cell naming one file
    (``conftest.py``) and one naming a tree (``packages/*/src``) with the same
    rule, and it lets one cell stand for all ten packages without listing them
    — a list would leave the eleventh package silently in no cell at all.

    **Anchored at the root, which the obvious implementation is not.**
    ``PurePosixPath.match`` matches a relative pattern from the *right*, so
    ``PurePosixPath("packages/bots/src").match("src")`` is true and a cell named
    ``src`` swallows all ten package sources. Written that way, this reported
    every file under ``packages/*/src`` as belonging to two cells at once — on
    the first run, which is what the totality check is for. ``full_match`` would
    say it directly and arrives in 3.13; until then the comparison is explicit.
    """
    want = pattern.split("/")
    have = path.parts
    if len(want) > len(have):
        return False
    return all(fnmatch.fnmatchcase(part, glob) for part, glob in zip(have, want))


def partition(cells: list[dict[str, Any]], files: list[PurePosixPath]) -> dict[str, Any]:
    """Which cell each file lands in, and every file that lands in none or several.

    Takes the cells rather than the tool name so ``verify`` can pass only the
    ones it has already found well formed: a cell with no ``path`` would
    otherwise be indexed here and raise, turning a reportable fault into the
    traceback this module exists to avoid.
    """
    by_cell: dict[str, list[str]] = {cell["path"]: [] for cell in cells}
    orphans: list[str] = []
    overlaps: list[str] = []

    for path in files:
        hits = [cell["path"] for cell in cells if cell_matches(path, cell["path"])]
        if not hits:
            orphans.append(str(path))
        elif len(hits) > 1:
            overlaps.append(f"{path} -> {', '.join(hits)}")
        else:
            by_cell[hits[0]].append(str(path))

    return {"by_cell": by_cell, "orphans": sorted(orphans), "overlaps": sorted(overlaps)}


def _cell_for(cells: list[dict[str, Any]], relative: str) -> str | None:
    """The single cell a measured path belongs to, or None when it is outside them all."""
    path = PurePosixPath(relative)
    hits = [cell["path"] for cell in cells if cell_matches(path, cell["path"])]
    return hits[0] if len(hits) == 1 else None


#: How many offending files a breached ceiling names before it summarises.
_NAMED_OFFENDERS = 10


class Measurement(NamedTuple):
    """What one tool found, kept per file rather than per cell.

    A cell total answers *whether* a ceiling is breached; it cannot answer
    *what* breached it. The first ceiling this mechanism ever broke reported
    ``21 findings against 20 allowed`` over a directory of 21 files, and finding
    the twenty-first took a separate script — a failure nobody can act on gets
    suppressed just as surely as one that fires spuriously, which is G4's
    subject from the other side. So the file names are carried out of the
    measurement and the total is derived from them.

    ``unattributed`` holds findings that resolved to no cell at all. Totality
    makes that impossible for a tracked file, so it stays empty in the ordinary
    case — but mypy follows imports, and a finding reported against something
    outside the population would otherwise be dropped in silence, which is this
    repository's own defect class: an absence rendered as a pass.

    ``output`` is what the tool actually said, kept because two callers need the
    prose rather than the tally: a developer who has just been told a ceiling is
    breached needs the messages to act on, and the run reports on its own
    configuration in the same stream — a measurement taken under a section that
    has stopped applying is not the measurement it claims to be. Only mypy fills
    it; ruff and the formatter are read from JSON that *is* the tally, so
    echoing it would show the developer the parse rather than the findings.
    """

    by_cell: dict[str, Counter[str]]
    unattributed: Counter[str]
    output: str = ""


def _tally(cells: list[dict[str, Any]], names: list[str]) -> Measurement:
    """Bucket reported paths into the cell each belongs to."""
    by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    unattributed: Counter[str] = Counter()
    for name in names:
        cell = _cell_for(cells, name) if name else None
        if cell:
            by_cell[cell][name] += 1
        else:
            unattributed[name or "<unnamed>"] += 1
    return Measurement(dict(by_cell), unattributed)


def _restrict(cells: list[dict[str, Any]], only: set[str] | None) -> list[dict[str, Any]]:
    """The subset of cells a scoped call named, or all of them.

    ``only`` restricts what is *measured and compared*, never what findings are
    *attributed to*: a scoped run still tallies against the full cell list, so a
    finding outside the requested cells is reported as belonging to the cell it
    is in rather than silently becoming unattributed.
    """
    return cells if only is None else [cell for cell in cells if cell["path"] in only]


def _files_in(
    cells: list[dict[str, Any]], files: list[PurePosixPath], only: set[str] | None
) -> list[PurePosixPath]:
    """The population narrowed to the cells a scoped call named."""
    if only is None:
        return files
    return [path for path in files if _cell_for(cells, str(path)) in only]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a measuring tool from the repository root.

    ``check=False`` throughout: every one of these exits non-zero precisely when
    it has findings, which is the ordinary case here rather than a failure.
    """
    return subprocess.run(command, cwd=_ROOT, capture_output=True, text=True, check=False)


def measure_ruff(
    contract: dict[str, Any], files: list[PurePosixPath], only: set[str] | None = None
) -> Measurement:
    """Findings per cell, from one ruff pass over the whole population."""
    cells = contract["tools"]["ruff"]["cells"]
    config = contract["tools"]["ruff"]["config"]
    files = _files_in(cells, files, only)
    if not files:
        return Measurement({}, Counter())
    result = _run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "--config",
            config,
            "--output-format",
            "json",
            "--no-cache",
            *(str(f) for f in files),
        ]
    )
    try:
        findings = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"ruff did not emit JSON, so nothing was measured: {exc}\n{result.stderr[:800]}"
        ) from exc

    return _tally(cells, [_relative(finding.get("filename", "")) for finding in findings])


def measure_mypy(
    contract: dict[str, Any], _files: list[PurePosixPath], only: set[str] | None = None
) -> Measurement:
    """Findings per cell, from one mypy pass over the cells it actually covers.

    Takes the population and ignores it, so the three measurers share one
    signature and ``MEASURERS`` can name them rather than wrap them. mypy is
    given directories rather than a file list: handed individual files it still
    follows imports, and the same finding then arrives under whichever file
    reached it first.

    The target set is taken from the contract rather than from
    ``bin/validate.sh --print-targets``. That used to be a statement about two
    lists that had to agree; it is now the only list there is, since the script
    reaches its mypy verdict by calling this. Deriving it the other way round
    would make the ratchet move whenever that script's default changed.

    Scoping to ``only`` measures fewer cells, not fewer files within one: a
    ceiling is a whole-cell property, so a partial count compared against it is
    not a verdict. That the two agree is not an assumption — the per-cell
    numbers from a run over one package are identical to that package's numbers
    from a run over all fourteen.
    """
    cells = contract["tools"]["mypy"]["cells"]
    config = contract["tools"]["mypy"]["config"]
    targets = sorted(
        {
            str(path)
            for cell in _restrict(cells, only)
            if cell["tier"] not in _UNMEASURED_TIERS
            for path in _expand(cell["path"])
        }
    )
    if not targets:
        return Measurement({}, Counter())

    result = _run(["uv", "run", "mypy", *targets, "--config-file", config])
    reported = [
        _relative(match.group(1))
        for line in result.stdout.splitlines()
        if (match := re.match(r"([^:]+):\d+:(?:\d+:)?\s*error:", line))
    ]
    return _tally(cells, reported)._replace(output=result.stdout)


def measure_format(
    contract: dict[str, Any], files: list[PurePosixPath], only: set[str] | None = None
) -> Measurement:
    """Files the formatter would rewrite, per cell.

    Counted in files rather than findings because that is the unit the formatter
    reports and the unit the adoption diff is measured in. Names are de-duplicated
    first: ruff reports one block per rewritten region, so a file with four of
    them would otherwise count four times against a ceiling denominated in files.

    Read from JSON rather than from the diff text, which is not a tidying. The
    text parse regexed ``--> path:line`` out of the human output, and ruff
    reports a file it could not open with no such line — so an unreadable file
    contributed nothing and the cell holding it measured *lower*. Every failure
    of that parse pointed the same way, toward a cleaner tree than the one on
    disk, and once this tool's job is to hold a set of zeroes, "found nothing"
    and "read nothing" are the same report.

    So the result is checked three ways rather than parsed one way: the output
    must be JSON, every entry must be a formatting verdict rather than an I/O
    error, and the exit status must agree with what was counted.
    """
    cells = contract["tools"]["format"]["cells"]
    config = contract["tools"]["format"]["config"]
    files = _files_in(cells, files, only)
    if not files:
        return Measurement({}, Counter())
    result = _run(
        [
            "uv",
            "run",
            "ruff",
            "format",
            "--check",
            "--config",
            config,
            "--output-format",
            "json",
            *(str(f) for f in files),
        ]
    )
    try:
        reported = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"ruff format did not emit JSON, so nothing was measured: {exc}\n{result.stderr[:800]}"
        ) from exc

    # An `io` entry is a file ruff could not read. It is not a formatting
    # verdict and must never be absorbed into one: the tally would drop by one
    # and the cell would look cleaner for having become unreadable.
    unreadable = [entry for entry in reported if entry.get("code") != "unformatted"]
    if unreadable:
        detail = "\n".join(
            f"  {entry.get('filename', '?')}: {entry.get('message', entry.get('code', '?'))}"
            for entry in unreadable[:20]
        )
        raise SystemExit(
            f"ruff format reported {len(unreadable)} result(s) that are not "
            f"formatting verdicts, so the count below them is short by at least "
            f"that many:\n{detail}"
        )

    named = {_relative(entry.get("filename", "")) for entry in reported}

    # Exit 0 means every file is formatted and 1 means at least one is not.
    # Anything else is ruff declining to answer, and a disagreement between the
    # status and the count means the two halves of this measurement are reading
    # different runs.
    if result.returncode not in (0, 1):
        raise SystemExit(
            f"ruff format exited {result.returncode}, so its report is not a "
            f"measurement:\n{result.stderr[:800]}"
        )
    if bool(named) != bool(result.returncode):
        raise SystemExit(
            f"ruff format exited {result.returncode} but named {len(named)} "
            "unformatted file(s); status and output disagree, so one of them is "
            "not describing this run"
        )

    return _tally(cells, sorted(named))


def _relative(name: str) -> str:
    """A tool's reported path as a repo-relative POSIX one."""
    try:
        return str(Path(name).resolve().relative_to(_ROOT))
    except ValueError:
        return name


#: How mypy reports a per-module override section that matched nothing this run.
_UNUSED_SECTIONS_RE = re.compile(r": note: unused section\(s\): module = \[(.*)\]$", re.MULTILINE)

#: A quoted name, in either of TOML's two spellings.
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")

#: A line that declares module patterns: ``module = "x"``, ``module = [``, or a
#: bare list element. Narrow on purpose — every other quoted string in the
#: config would otherwise count as a declaration and be excusable by any comment
#: that happened to precede it.
_MODULE_LINE_RE = re.compile(r"^\s*(?:module\s*=|['\"][^'\"]+['\"]\s*,?\s*$)")


def unused_config_sections(output: str) -> list[str]:
    """Module patterns the type checker says its own config declared for nothing.

    A section matching no module suppresses nothing, which is only ever one of
    two things: a waiver whose spelling is wrong, so the findings it was written
    for are still being reported, or one whose subject is gone. Both read as
    "handled" to anyone looking at the config. mypy detects this and files it as
    a *note*, leaving the exit status untouched — so without this it is reported
    and nothing fails, which is the shape this contract exists to catch.

    Conclusive only over the whole population: scoped to one package, most
    sections legitimately match nothing. The caller enforces that.
    """
    names: list[str] = []
    for match in _UNUSED_SECTIONS_RE.finditer(output):
        names += _QUOTED_RE.findall(match.group(1))
    return sorted(set(names))


def excused_config_sections(config: Path) -> set[str]:
    """Module patterns whose declaration carries a comment saying why.

    An ``ignore_missing_imports`` section for a library imported only inside a
    ``try/except ImportError`` may legitimately match nothing in a given run —
    ``psycopg2.*`` is exactly that — so the failure is an entry that suppresses
    nothing *and says nothing about why*, the same shape the internal-label
    allowlist uses. A comment on the line above is the reason.

    A superset, deliberately: any commented quoted value that *looks* like a
    module declaration is collected, including a dependency specifier. Narrowing
    further would mean parsing TOML while tracking comments, and the excess is
    inert — this set is only ever consulted against names mypy has already
    reported as unused module sections, which a dependency specifier is not.
    """
    excused: set[str] = set()
    previous = ""
    for raw in config.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("#") and previous.startswith("#") and _MODULE_LINE_RE.match(line):
            excused.update(_QUOTED_RE.findall(line))
        previous = line
    return excused


def _expand(pattern: str) -> list[PurePosixPath]:
    """The directories or files one cell pattern names on disk."""
    if "*" in pattern:
        return [PurePosixPath(p.relative_to(_ROOT).as_posix()) for p in sorted(_ROOT.glob(pattern))]
    target = _ROOT / pattern
    return [PurePosixPath(pattern)] if target.exists() else []


#: One measurer per tool, each ``(contract, files, only) -> Measurement``.
MEASURERS: dict[
    str, Callable[[dict[str, Any], list[PurePosixPath], set[str] | None], Measurement]
] = {
    "ruff": measure_ruff,
    "mypy": measure_mypy,
    "format": measure_format,
}


def _measure(
    contract: dict[str, Any], tools: list[str], only: set[str] | None = None
) -> dict[str, Measurement]:
    """Run every requested tool once, over one shared file population.

    The single entry point for measuring, so ``check`` and ``update-baseline``
    cannot come to disagree about what a cell measures — which is the failure
    that would be hardest to see, since each would be internally consistent and
    the ratchet would move under one and not the other.

    ``only`` narrows the cells, and every measurer honours it. Implementing it
    for the one tool that currently asks would leave the other two accepting a
    restriction and quietly measuring everything — a scope silently wider than
    the one requested, reported as if it were the one requested.
    """
    files = tracked_python()
    return {tool: MEASURERS[tool](contract, files, only) for tool in tools}


def _measured(measurement: Measurement, cell: dict[str, Any]) -> int:
    """A cell's total, derived from its per-file counts rather than kept beside them."""
    return sum(measurement.by_cell.get(cell["path"], Counter()).values())


def verify(contract: dict[str, Any]) -> list[str]:
    """Structural faults in the declaration itself, as reportable sentences.

    Separate from ``check`` because these are pure comparisons over a small
    document while that shells out to three tools over the whole tree — a second
    or two with the type-checker's cache warm, and considerably more without it.
    A malformed contract should be reported in milliseconds rather than after a
    full measuring run.
    """
    faults: list[str] = []
    files = tracked_python()
    if not files:
        return [
            "git tracks no *.py at all — the population is empty and nothing below means anything"
        ]

    declared_tools = set(contract.get("tools", {}))
    missing = sorted(set(MEASURERS) - declared_tools)
    if missing:
        faults.append(f"no cells declared for {missing}, so those files are covered by nothing")

    for tool in sorted(declared_tools):
        spec = contract["tools"][tool]
        tiers = set(spec.get("tiers", {}))
        seen: set[str] = set()

        # Read rather than indexed, and checked before anything downstream uses
        # it. A tool entry with no cells is a population nothing decided about,
        # which is the same fault as a file in no cell — and reaching it through
        # ``spec["cells"]`` reported it as a traceback naming a line of this
        # script instead of a sentence naming the contract.
        cells = spec.get("cells")
        if not isinstance(cells, list) or not cells:
            faults.append(
                f"{tool}: declares no cells, so every file it would measure is covered by nothing"
            )
            continue

        # Split before the per-cell loop: a cell with no ``path`` cannot be
        # partitioned, and passing it on would raise where a fault belongs.
        well_formed: list[dict[str, Any]] = []
        for cell in cells:
            if isinstance(cell.get("path"), str):
                well_formed.append(cell)
            else:
                faults.append(f"{tool}: a cell has no path, so it names nothing")

        split = partition(well_formed, files)

        for cell in well_formed:
            where = f"{tool}/{cell['path']}"
            if cell["path"] in seen:
                faults.append(f"{where}: declared twice")
            seen.add(cell["path"])
            # The stale direction, which totality does not imply: a cell can
            # match zero files while every file still lands in some other cell,
            # so the partition stays valid and the entry stays wrong. The
            # declaration this replaced enforced exactly this, and it is the
            # half that fires when a directory is cleaned up and its deferral
            # outlives it — leaving the reader believing in a gap that closed.
            if not split["by_cell"][cell["path"]]:
                faults.append(
                    f"{where}: matches no tracked Python file, so it records a "
                    "gap that no longer exists"
                )
            if cell.get("tier") not in tiers:
                faults.append(f"{where}: tier {cell.get('tier')!r} is not one of {sorted(tiers)}")
            if not isinstance(cell.get("ceiling"), int) or isinstance(cell.get("ceiling"), bool):
                faults.append(f"{where}: ceiling {cell.get('ceiling')!r} is not a whole number")
            if not str(cell.get("reason", "")).strip():
                faults.append(f"{where}: no reason given, which is what makes deferring honest")
            if cell.get("tier") in _UNMEASURED_TIERS and cell.get("ceiling"):
                faults.append(
                    f"{where}: tier {cell['tier']!r} is not measured, so its ceiling "
                    f"of {cell['ceiling']} claims a number nothing takes"
                )

        for orphan in split["orphans"]:
            faults.append(f"{tool}: {orphan} is in no cell, so nobody decided about it")
        for overlap in split["overlaps"]:
            faults.append(
                f"{tool}: {overlap} is in several cells, so the decision contradicts itself"
            )

    return faults


def check(
    contract: dict[str, Any],
    tools: list[str],
    only: set[str] | None = None,
    show_findings: bool = False,
) -> dict[str, Any]:
    """Measure each tool, compare every cell against its ceiling, and report.

    Two things are reported besides the comparison, and both are there because
    a number produced under conditions that have changed is not the number it
    claims to be:

    * ``unused_config`` — sections of the measuring tool's own configuration
      that matched nothing. Collected only on an unrestricted run, since a
      scoped one makes almost every section look unused.
    * ``findings`` — what the tool said, when a caller asks for it. A ceiling
      breach reported as a count leaves a developer with nothing to open.
    """
    report: dict[str, Any] = {
        "exceeded": [],
        "cleared": [],
        "cells": {},
        "unattributed": [],
        "unused_config": [],
        "findings": {},
    }

    for tool, measurement in _measure(contract, tools, only).items():
        if show_findings and measurement.output:
            report["findings"][tool] = measurement.output
        if only is None and measurement.output:
            config = _ROOT / contract["tools"][tool]["config"]
            excused = excused_config_sections(config) if config.exists() else set()
            report["unused_config"] += [
                {"tool": tool, "config": contract["tools"][tool]["config"], "section": section}
                for section in unused_config_sections(measurement.output)
                if section not in excused
            ]

        for cell in _restrict(contract["tools"][tool]["cells"], only):
            name = f"{tool}/{cell['path']}"
            measured = _measured(measurement, cell)
            report["cells"][name] = {"measured": measured, "ceiling": cell["ceiling"]}
            entry = {"cell": name, "measured": measured, "ceiling": cell["ceiling"]}
            if measured > cell["ceiling"]:
                per_file = measurement.by_cell.get(cell["path"], Counter())
                ranked = sorted(per_file.items(), key=lambda item: (-item[1], item[0]))
                report["exceeded"].append(
                    {
                        **entry,
                        "files": [
                            {"file": name, "count": count}
                            for name, count in ranked[:_NAMED_OFFENDERS]
                        ],
                        "further_files": max(len(ranked) - _NAMED_OFFENDERS, 0),
                    }
                )
            elif measured < cell["ceiling"]:
                report["cleared"].append(entry)

        report["unattributed"] += [
            {"tool": tool, "file": name, "findings": count}
            for name, count in sorted(measurement.unattributed.items())
        ]

    return report


def update_baseline(
    contract: dict[str, Any], tools: list[str], path: Path
) -> tuple[list[str], list[str]]:
    """Lower every ceiling to what the tree currently measures.

    Lower only. Raising a ceiling is how a backlog grows while it is supposedly
    being worked, and doing it by re-running a command is how that happens
    without anyone deciding to — so an exceeded cell is left alone here and
    reported, which puts the argument for raising it in a pull request where it
    belongs.

    Returns ``(lowered, exceeded)``. The second half is why this returns a pair:
    leaving a breached ceiling alone *and silent* would tell a developer who has
    just introduced a regression that there was nothing to do — the docstring
    above said "reported" while the code only ever mentioned the cells it
    lowered, which is the same shape as a status field whose default is a
    verdict.
    """
    lowered: list[str] = []
    exceeded: list[str] = []

    for tool, measurement in _measure(contract, tools).items():
        for cell in contract["tools"][tool]["cells"]:
            measured = _measured(measurement, cell)
            if measured < cell["ceiling"]:
                lowered.append(f"{tool}/{cell['path']}: {cell['ceiling']} -> {measured}")
                cell["ceiling"] = measured
            elif measured > cell["ceiling"]:
                exceeded.append(
                    f"{tool}/{cell['path']}: {measured} findings against "
                    f"{cell['ceiling']} allowed, left alone"
                )

    if lowered:
        path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return lowered, exceeded


#: What one caller-named path is, to a tool: a path inside a measured cell, one
#: inside a cell the contract declares the tool does not read, or one outside
#: the population entirely.
SCOPE_MEASURED = "cell"
SCOPE_UNMEASURED = "unmeasured"
SCOPE_OUTSIDE = "outside"


def scope_paths(
    contract: dict[str, Any], tool: str, paths: list[str]
) -> list[tuple[str, str, str]]:
    """Which cell each caller-named path falls in, as ``(kind, path, cell)``.

    ``bin/validate.sh`` takes packages, directories and single files, and has to
    turn them into a mypy verdict. It cannot do that by matching cell patterns
    itself: a second copy of ``cell_matches`` is a second answer waiting to
    disagree with the one the ceilings are measured under. So it asks.

    The three kinds are three different verdicts, and the middle one is the
    reason this returns a classification rather than a filtered list. A path
    under a cell the contract marks unmeasured must not be type-checked *and
    must not be silently dropped either* — dropping it is how a caller comes to
    believe a directory was checked when nothing read it.
    """
    cells = contract["tools"][tool]["cells"]
    unmeasured = {cell["path"] for cell in cells if cell.get("tier") in _UNMEASURED_TIERS}

    resolved: list[tuple[str, str, str]] = []
    for given in paths:
        cell = _cell_for(cells, _relative(given))
        if cell is None:
            resolved.append((SCOPE_OUTSIDE, given, ""))
        elif cell in unmeasured:
            resolved.append((SCOPE_UNMEASURED, given, cell))
        else:
            resolved.append((SCOPE_MEASURED, given, cell))
    return resolved


def _selected(requested: str | None) -> list[str]:
    return [requested] if requested else sorted(MEASURERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=("check", "verify", "update-baseline", "partition", "scope")
    )
    parser.add_argument("paths", nargs="*", help="scope: the paths to classify")
    parser.add_argument("--tool", choices=sorted(MEASURERS), help="restrict to one tool")
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="PATH",
        help="check: compare only these cells, named exactly as the contract declares them",
    )
    parser.add_argument(
        "--show-findings",
        action="store_true",
        help="check: echo what the tool reported, not only the comparison",
    )
    parser.add_argument("--json", action="store_true", dest="use_json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    contract = load_contract()

    if args.command == "scope":
        if not args.tool:
            parser.error("scope classifies paths for one tool, so --tool is required")
        resolved = scope_paths(contract, args.tool, args.paths)
        if args.use_json:
            json.dump(
                [{"kind": kind, "path": path, "cell": cell} for kind, path, cell in resolved],
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
        else:
            for kind, path, cell in resolved:
                sys.stdout.write(f"{kind}\t{path}\t{cell}\n")
        sys.exit(0)

    # Named cells are resolved against the declaration rather than trusted. A
    # caller that misspells one would otherwise restrict the run to nothing and
    # be told every cell is within its ceiling.
    only: set[str] | None = None
    if args.cell:
        if not args.tool:
            parser.error("--cell names cells of one tool, so --tool is required with it")
        declared = {cell["path"] for cell in contract["tools"][args.tool]["cells"]}
        unknown = sorted(set(args.cell) - declared)
        if unknown:
            parser.error(
                f"{args.tool} declares no cell named {unknown} — known cells are {sorted(declared)}"
            )
        only = set(args.cell)

    if args.command == "verify":
        faults = verify(contract)
        if args.use_json:
            json.dump({"faults": faults}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            for fault in faults:
                logger.error("%s", fault)
            if not faults:
                logger.info("The contract is total and well formed.")
        sys.exit(1 if faults else 0)

    if args.command == "partition":
        files = tracked_python()
        split = {
            tool: partition(contract["tools"][tool]["cells"], files)
            for tool in _selected(args.tool)
        }
        json.dump(split, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.exit(0)

    # Both remaining commands measure, and measuring a malformed contract
    # reports cells that do not describe the tree. Fail on the cheap check first.
    faults = verify(contract)
    if faults:
        for fault in faults:
            logger.error("%s", fault)
        logger.error("The contract is not usable, so nothing was measured.")
        sys.exit(2)

    if args.command == "update-baseline":
        lowered, exceeded = update_baseline(contract, _selected(args.tool), _CONTRACT)
        for line in lowered:
            logger.info("%s", line)
        if not lowered:
            logger.info("No ceiling was above what the tree measures.")
        for line in exceeded:
            logger.warning("%s", line)
        if exceeded:
            logger.warning(
                "Raising a ceiling is a hand edit, so the argument for it lands "
                "in a pull request rather than in a rerun."
            )
        sys.exit(0)

    report = check(contract, _selected(args.tool), only, args.show_findings)
    failed = bool(report["exceeded"] or report["unused_config"])
    if args.use_json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.exit(1 if failed else 0)

    # Ahead of every verdict below: a developer told a ceiling is breached needs
    # the messages, and printing them after the ranked file list would separate
    # the count from the thing it counted.
    for tool in sorted(report["findings"]):
        sys.stdout.write(report["findings"][tool])
        sys.stdout.flush()

    for cleared in report["cleared"]:
        logger.info(
            "%s is under its ceiling (%d of %d) — lower it with --update-baseline",
            cleared["cell"],
            cleared["measured"],
            cleared["ceiling"],
        )
    for stray in report["unattributed"]:
        logger.warning(
            "%s reported %d in %s, which is in no cell — not counted anywhere",
            stray["tool"],
            stray["findings"],
            stray["file"],
        )
    for exceeded in report["exceeded"]:
        logger.error(
            "%s exceeds its ceiling: %d findings against %d allowed",
            exceeded["cell"],
            exceeded["measured"],
            exceeded["ceiling"],
        )
        # A count alone does not say which file to open, and a failure nobody
        # can act on gets suppressed as surely as one that fires spuriously.
        for offender in exceeded["files"]:
            logger.error("    %s (%d)", offender["file"], offender["count"])
        if exceeded["further_files"]:
            logger.error("    ... and %d more", exceeded["further_files"])
    for dead in report["unused_config"]:
        logger.error(
            "%s: the %s section for %r matched nothing, so it suppresses nothing "
            "— delete it, correct the module it names, or say on the line above "
            "why it may legitimately match nothing",
            dead["config"],
            dead["tool"],
            dead["section"],
        )
    if not failed:
        logger.info("Every cell is within its ceiling.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

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
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

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


def partition(contract: dict[str, Any], tool: str, files: list[PurePosixPath]) -> dict[str, Any]:
    """Which cell each file lands in, and every file that lands in none or several."""
    cells = contract["tools"][tool]["cells"]
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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a measuring tool from the repository root.

    ``check=False`` throughout: every one of these exits non-zero precisely when
    it has findings, which is the ordinary case here rather than a failure.
    """
    return subprocess.run(command, cwd=_ROOT, capture_output=True, text=True, check=False)


def measure_ruff(contract: dict[str, Any], files: list[PurePosixPath]) -> Counter[str]:
    """Findings per cell, from one ruff pass over the whole population."""
    cells = contract["tools"]["ruff"]["cells"]
    config = contract["tools"]["ruff"]["config"]
    result = _run(
        [
            "uv", "run", "ruff", "check",
            "--config", config,
            "--output-format", "json",
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

    counts: Counter[str] = Counter()
    for finding in findings:
        relative = _relative(finding.get("filename", ""))
        cell = _cell_for(cells, relative) if relative else None
        if cell:
            counts[cell] += 1
    return counts


def measure_mypy(contract: dict[str, Any], _files: list[PurePosixPath]) -> Counter[str]:
    """Findings per cell, from one mypy pass over the cells it actually covers.

    Takes the population and ignores it, so the three measurers share one
    signature and ``MEASURERS`` can name them rather than wrap them. mypy is
    given directories rather than a file list: handed individual files it still
    follows imports, and the same finding then arrives under whichever file
    reached it first.

    The target set is taken from the contract rather than from
    ``bin/validate.sh --print-targets``: the two must agree, and a guard asserts
    they do, but deriving it from the script would make this measurement move
    silently whenever that script's default changed.
    """
    cells = contract["tools"]["mypy"]["cells"]
    config = contract["tools"]["mypy"]["config"]
    targets = sorted(
        {
            str(path)
            for cell in cells
            if cell["tier"] not in _UNMEASURED_TIERS
            for path in _expand(cell["path"])
        }
    )
    if not targets:
        return Counter()

    result = _run(["uv", "run", "mypy", *targets, "--config-file", config])
    counts: Counter[str] = Counter()
    for line in result.stdout.splitlines():
        match = re.match(r"([^:]+):\d+:(?:\d+:)?\s*error:", line)
        if not match:
            continue
        cell = _cell_for(cells, _relative(match.group(1)))
        if cell:
            counts[cell] += 1
    return counts


def measure_format(contract: dict[str, Any], files: list[PurePosixPath]) -> Counter[str]:
    """Files the formatter would rewrite, per cell.

    Counted in files rather than findings because that is the unit the formatter
    reports and the unit the adoption diff is measured in. Names are de-duplicated
    first: ruff reports one block per rewritten region, so a file with four of
    them would otherwise count four times against a ceiling denominated in files.
    """
    cells = contract["tools"]["format"]["cells"]
    config = contract["tools"]["format"]["config"]
    result = _run(
        [
            "uv", "run", "ruff", "format", "--check",
            "--config", config,
            *(str(f) for f in files),
        ]
    )
    named = {
        _relative(match.group(1))
        for match in re.finditer(r"-->\s+(\S+?):\d+", result.stdout)
    }
    counts: Counter[str] = Counter()
    for relative in sorted(named):
        cell = _cell_for(cells, relative)
        if cell:
            counts[cell] += 1
    return counts


def _relative(name: str) -> str:
    """A tool's reported path as a repo-relative POSIX one."""
    try:
        return str(Path(name).resolve().relative_to(_ROOT))
    except ValueError:
        return name


def _expand(pattern: str) -> list[PurePosixPath]:
    """The directories or files one cell pattern names on disk."""
    if "*" in pattern:
        return [PurePosixPath(p.relative_to(_ROOT).as_posix()) for p in sorted(_ROOT.glob(pattern))]
    target = _ROOT / pattern
    return [PurePosixPath(pattern)] if target.exists() else []


#: One measurer per tool, each ``(contract, files) -> Counter[cell]``.
MEASURERS: dict[str, Callable[[dict[str, Any], list[PurePosixPath]], Counter[str]]] = {
    "ruff": measure_ruff,
    "mypy": measure_mypy,
    "format": measure_format,
}


def verify(contract: dict[str, Any]) -> list[str]:
    """Structural faults in the declaration itself, as reportable sentences.

    Separate from ``check`` because these cost nothing and those cost minutes:
    a malformed contract should be reported by the test suite in milliseconds
    rather than after a full measuring run.
    """
    faults: list[str] = []
    files = tracked_python()
    if not files:
        return ["git tracks no *.py at all — the population is empty and nothing below means anything"]

    declared_tools = set(contract.get("tools", {}))
    missing = sorted(set(MEASURERS) - declared_tools)
    if missing:
        faults.append(f"no cells declared for {missing}, so those files are covered by nothing")

    for tool in sorted(declared_tools):
        spec = contract["tools"][tool]
        tiers = set(spec.get("tiers", {}))
        seen: set[str] = set()

        for cell in spec["cells"]:
            where = f"{tool}/{cell.get('path', '?')}"
            if cell["path"] in seen:
                faults.append(f"{where}: declared twice")
            seen.add(cell["path"])
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

        split = partition(contract, tool, files)
        for orphan in split["orphans"]:
            faults.append(f"{tool}: {orphan} is in no cell, so nobody decided about it")
        for overlap in split["overlaps"]:
            faults.append(f"{tool}: {overlap} is in several cells, so the decision contradicts itself")

    return faults


def check(contract: dict[str, Any], tools: list[str]) -> dict[str, Any]:
    """Measure each tool and compare every cell against its ceiling."""
    files = tracked_python()
    report: dict[str, Any] = {"exceeded": [], "cleared": [], "cells": {}}

    for tool in tools:
        counts = MEASURERS[tool](contract, files)
        for cell in contract["tools"][tool]["cells"]:
            name = f"{tool}/{cell['path']}"
            measured = counts.get(cell["path"], 0)
            report["cells"][name] = {"measured": measured, "ceiling": cell["ceiling"]}
            if measured > cell["ceiling"]:
                report["exceeded"].append(
                    {"cell": name, "measured": measured, "ceiling": cell["ceiling"]}
                )
            elif measured < cell["ceiling"]:
                report["cleared"].append(
                    {"cell": name, "measured": measured, "ceiling": cell["ceiling"]}
                )

    return report


def update_baseline(contract: dict[str, Any], tools: list[str], path: Path) -> list[str]:
    """Lower every ceiling to what the tree currently measures.

    Lower only. Raising a ceiling is how a backlog grows while it is supposedly
    being worked, and doing it by re-running a command is how that happens
    without anyone deciding to — so an exceeded cell is left alone here and
    reported, which puts the argument for raising it in a pull request where it
    belongs.
    """
    files = tracked_python()
    changed: list[str] = []

    for tool in tools:
        counts = MEASURERS[tool](contract, files)
        for cell in contract["tools"][tool]["cells"]:
            measured = counts.get(cell["path"], 0)
            if measured < cell["ceiling"]:
                changed.append(f"{tool}/{cell['path']}: {cell['ceiling']} -> {measured}")
                cell["ceiling"] = measured

    if changed:
        path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return changed


def _selected(requested: str | None) -> list[str]:
    return [requested] if requested else sorted(MEASURERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=("check", "verify", "update-baseline", "partition")
    )
    parser.add_argument("--tool", choices=sorted(MEASURERS), help="restrict to one tool")
    parser.add_argument("--json", action="store_true", dest="use_json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    contract = load_contract()

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
        split = {
            tool: partition(contract, tool, tracked_python()) for tool in _selected(args.tool)
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
        changed = update_baseline(contract, _selected(args.tool), _CONTRACT)
        for line in changed:
            logger.info("%s", line)
        if not changed:
            logger.info("No ceiling was above what the tree measures.")
        sys.exit(0)

    report = check(contract, _selected(args.tool))
    if args.use_json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.exit(1 if report["exceeded"] else 0)

    for cleared in report["cleared"]:
        logger.info(
            "%s is under its ceiling (%d of %d) — lower it with --update-baseline",
            cleared["cell"], cleared["measured"], cleared["ceiling"],
        )
    for exceeded in report["exceeded"]:
        logger.error(
            "%s exceeds its ceiling: %d findings against %d allowed",
            exceeded["cell"], exceeded["measured"], exceeded["ceiling"],
        )
    if not report["exceeded"]:
        logger.info("Every cell is within its ceiling.")
    sys.exit(1 if report["exceeded"] else 0)


if __name__ == "__main__":
    main()

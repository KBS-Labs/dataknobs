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

**A ceiling says how much, never what.** It is denominated in findings, so the
comparison above can report that a cell holds 657 of them and nothing about
whether that is one mechanical omission repeated or six hundred separate
judgements — a distinction that decides whether the backlog has a plan. The
tools name a rule on every finding and the tally reads past it. ``census``
reads it, from the same run and through the same guards, so its per-rule
breakdown decomposes the number the ceiling is compared against rather than
standing beside it as a second opinion.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import logging
import re
import subprocess
import sys
import tokenize
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_CONTRACT = _ROOT / ".dataknobs" / "quality-contract.json"

#: How this script is spelled inside advice that names a command to run.
#:
#: Through ``uv run``, because the toolchain is pinned behind it and a bare
#: ``python bin/...`` resolves against ``PATH`` — which is
#: ``tests/test_remediation_paths.py``'s opening failure, arriving in the one
#: place that file cannot see, since it reads shell scripts and this is Python.
#: Written once so that every command this module recommends is spelled the same
#: way; the prose in ``docs/`` is not held to it by anything but review.
_INVOCATION = "uv run python bin/quality-contract.py"

#: Tiers whose files no tool reads. Their ceiling is not a backlog and must stay
#: zero — a positive one would claim a measurement that nothing takes.
_UNMEASURED_TIERS = frozenset({"unchecked"})

#: Tools whose measurer is pointed at a target set the contract's tiers decide.
#:
#: Only ``mypy_targets`` consults the tier, because only mypy is handed
#: directories rather than the file population: a cell in an unmeasured tier is
#: dropped from the pass, so it is one mypy was never asked about. ``measure_ruff``
#: tallies a single pass over the whole population and ``measure_format`` reads
#: every cell's files, so re-tiering a cell changes nothing about whether either
#: looked at it — and their zero over such a cell is a count, not a silence.
#:
#: So the tier alone cannot answer whether a zero was measured. It answers for
#: mypy and would be wrong about the other two, in the direction this module
#: exists to refuse: a presence rendered as an absence. Written down because
#: ``check`` has no way to ask a measurer afterwards which cells it was pointed
#: at, and pinned against the measurers themselves by
#: ``test_the_tools_declared_tier_gated_are_the_ones_that_scope_by_tier`` — if a
#: second measurer ever starts reading the tier, that is what says so.
#:
#: The alternative is for ``Measurement`` to carry the cells its pass skipped,
#: which is the shape to take if that day comes. It is more than this fact needs
#: today, and it has a trap: ``mypy_targets`` also drops a cell whose
#: directories hold no tracked ``*.py``, and *that* zero is a real one. A flat
#: "untargeted" set would conflate the two and start reporting ``None`` for
#: cells whose zero is true and knowable.
_TIER_GATED_TOOLS = frozenset({"mypy"})

#: A bare numeral in a cell's reason, which is a measurement written where
#: nothing compares it.
#:
#: The ``about`` block records that the declaration this replaced "carried counts
#: in comment text, which stayed green at any number", and the replacement fixed
#: that for the ceiling only. A count in the prose beside it has exactly the old
#: property: it is true when written, goes false the moment the cell ratchets,
#: and no run disagrees. Four reasons had gone false that way before this
#: existed, two of them describing a cell that had been deleted outright.
#:
#: Word boundaries keep rule names — ``NPY002``, ``PTH118`` — sayable, and the
#: distinction is the point rather than a concession to the regex: a rule names
#: a *kind* of finding and stays accurate however many there are, while "130 of
#: them" is the ceiling's job and only the ceiling is checked.
_MEASUREMENT_IN_PROSE = re.compile(r"\b\d+\b")


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
    rule, and it lets one cell stand for every package without listing them.

    **What the glob buys is brevity, not safety.** This once said a list
    "would leave the eleventh package silently in no cell at all", and the
    silence is the false part: ``verify`` reports an unmatched file as
    ``"<tool>: <path> is in no cell, so nobody decided about it"``, and
    ``test_the_contract_is_total_and_well_formed`` drives ``verify`` as a
    subprocess and asserts it exits zero. So an eleventh package arriving
    under a listed set of cells turns the suite red. Under a glob it is
    matched instead — and therefore declared, at whatever tier and ceiling
    the glob carries, over files nobody has opened. The list's default is
    *undecided*; the glob's is *decided already*.

    Which shape is right therefore follows the tier, not a preference. ``mypy``
    and the formatter give every package's tests the same answer — unchecked,
    and enforced at zero — and one ``packages/*/tests`` cell says it once.
    ``ruff`` lists them one per package, because each arrived carrying its own
    backlog and left carrying its own promotion reason, and the cell is where
    that reason lives.

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
    # strict=False deliberately: the guard above admits a path with MORE
    # segments than the pattern, and comparing the pattern against its leading
    # segments is the whole operation. strict=True here rejects every cell
    # whose pattern is shorter than the file it should match, which is all of
    # them.
    return all(fnmatch.fnmatchcase(part, glob) for part, glob in zip(have, want, strict=False))


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


def _bucket(
    cells: list[dict[str, Any]], counted: Iterable[tuple[str, str]]
) -> tuple[dict[str, Counter[str]], Counter[str]]:
    """Bucket ``(path, key)`` pairs into the cell each path belongs to.

    The attribution rule, written once. Both projections of a run need it — the
    ratchet's counts keyed by file, the census's keyed by rule — and it was
    written twice, the copies differing in nothing but which key they counted
    under. The whole claim the census rests on is that the two cannot disagree
    about which cell a finding is in, and two copies of the rule deciding that
    hold only until somebody edits one of them.

    A pair whose path resolves to no cell is counted as unattributed rather than
    dropped. Totality makes that impossible for a tracked file, but mypy follows
    imports, and a finding against something outside the population would
    otherwise vanish in silence.
    """
    by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    unattributed: Counter[str] = Counter()
    for path, key in counted:
        cell = _cell_for(cells, path) if path else None
        target = by_cell[cell] if cell else unattributed
        target[key] += 1
    return dict(by_cell), unattributed


def _tally(cells: list[dict[str, Any]], names: list[str]) -> Measurement:
    """Bucket reported paths into the cell each belongs to, counted per file."""
    by_cell, unattributed = _bucket(cells, ((name, name or "<unnamed>") for name in names))
    return Measurement(by_cell, unattributed)


class Finding(NamedTuple):
    """One thing a tool reported: the file it is in, and the rule it broke.

    The second half is what the ratchet throws away. A ceiling is denominated in
    findings per cell, so the measurement above keeps only the path — which
    leaves the declaration able to say *where* a backlog is, to the file, and
    nothing whatever about *what it is*. That is answerable from the same run,
    and it decides whether a backlog decomposes into a few mechanical rules or
    into hundreds of separate judgements. It is not a question a total can be
    asked.
    """

    path: str
    code: str


#: What a finding whose tool named no rule is counted as. mypy omits the
#: bracketed code on a few errors and ruff reports ``null`` for a syntax error,
#: and both must land somewhere a reader sees them: a parse that skips a line
#: shape measures a tidier tree than the one on disk, which is the direction
#: every guard in this module exists to refuse.
UNCODED = "<uncoded>"


class Census(NamedTuple):
    """What one tool found, per cell and per rule.

    The counterpart to ``Measurement`` over the same findings: that one is keyed
    by file and answers *where*, this one is keyed by rule and answers *what*.
    Both are projections of a single run, which is what lets the two be compared
    — a census taken from a second invocation would be a second answer to what
    the tool found, and the interesting case is precisely the one where it
    disagrees.

    ``unattributed`` is keyed by rule rather than by file, for the same reason
    the rest of it is. The count still has to match ``Measurement``'s.
    """

    by_cell: dict[str, Counter[str]]
    unattributed: Counter[str]


def _tally_codes(cells: list[dict[str, Any]], findings: list[Finding]) -> Census:
    """Bucket findings into the cell each is in, counted per rule.

    The same bucketing ``_tally`` does over the same findings, projected through
    a different key — which is what makes the two comparable rather than a
    second opinion.
    """
    by_cell, unattributed = _bucket(cells, ((f.path, f.code) for f in findings))
    return Census(by_cell, unattributed)


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


def _run(command: list[str], cwd: Path = _ROOT) -> subprocess.CompletedProcess[str]:
    """Run a measuring tool from the repository root.

    ``check=False`` throughout: every one of these exits non-zero precisely when
    it has findings, which is the ordinary case here rather than a failure.

    ``cwd`` is a parameter for the ledger's sake, and only the ledger passes it.
    That command reads git history rather than the tree, and the three events
    the trial it serves actually turns on — a raised ceiling, a leg trailer, a
    trailer naming a cell that does not exist — have never happened in this
    repository. A reader that can only be pointed at its own history cannot be
    tested against any of them.
    """
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def _refuse_non_verdict(tool: str, result: subprocess.CompletedProcess[str]) -> None:
    """Refuse a run whose exit status is not a verdict about the tree.

    All three of these tools exit 0 for clean and 1 for "found something".
    Anything else — a configuration it cannot load, a crash, a missing
    interpreter, a wrapper eating the invocation — means it read nothing, and
    what comes back on stdout is *empty*. Every parse below turns that emptiness
    into a valid and entirely fictional measurement of zero: the ratchet reports
    a tree it never opened as one with nothing wrong, and ``update-baseline``
    writes those zeroes down as the new ceilings, which the ratchet then refuses
    to raise again.

    Shared because it was written twice and missing a third time. The formatter
    earned the check, the linter was given a copy, and the type checker had
    neither — so a mypy that failed to start measured every type-checked cell at
    zero, ``check`` exited 0, and ``bin/validate.sh`` printed a green type-check
    verdict over a tree nothing had opened. Guarding two tools out of three is
    what a rule written twice looks like from the side where it was forgotten.
    """
    if result.returncode not in (0, 1):
        raise SystemExit(
            f"{tool} exited {result.returncode}, so its report is not a "
            f"measurement:\n{result.stderr[:800]}"
        )


def _refuse_disagreement(
    tool: str, result: subprocess.CompletedProcess[str], counted: int, unit: str
) -> None:
    """Refuse a run whose status and whose output describe different runs.

    The count and the status come from one invocation *by assumption*, and this
    is the only thing that checks the assumption.

    It also catches the half of a broken parse that the status check cannot. A
    tool that exits 1 having reported findings this module's parse does not
    recognise leaves the status saying "found something" and the tally saying
    the tree is clean — the same fictional zero as above, reached through the
    reader rather than through the tool.
    """
    if bool(counted) != bool(result.returncode):
        raise SystemExit(
            f"{tool} exited {result.returncode} but reported {counted} {unit}; "
            "status and output disagree, so one of them is not describing this run"
        )


#: How ruff reports a file it could not open — the rule code and the rule name
#: for the same entry. Either spelling identifies it, so renaming one does not
#: quietly reopen the hole below.
_RUFF_IO_ERROR = frozenset({"E902", "io-error"})


def _ruff_report(
    contract: dict[str, Any], files: list[PurePosixPath], only: set[str] | None = None
) -> list[dict[str, Any]]:
    """One guarded ruff pass, as the entries ruff reported.

    Split out of ``measure_ruff`` so that the census reads its rules from the
    same run, through the same guards, as the ratchet reads its counts from. A
    census with its own invocation and its own parse would be a second answer to
    what ruff found — and every guard below exists because a parse of this
    output can fail toward a tidier tree than the one on disk, which is exactly
    the disagreement that would go unnoticed.

    Guarded the way the other two measurers are — ``_refuse_non_verdict`` and
    ``_refuse_disagreement``, which every tool here needs — and against two
    further faults the decode check alone renders as a clean tree. The formatter
    earned those shared checks by having a parse fail toward a tidier tree than
    the one on disk; each of these is the same failure, reached by a route ruff's
    linter offers and its formatter does not.

    The status check matters more here than anywhere else: editing this
    repository's ruff config is how a cell gets promoted, and ruff exits 2 on a
    config it cannot load having read no file. So a rejected config arrives at
    this measurer as part of the work rather than as an accident beside it.

    * **An ``E902`` is not a lint verdict.** It is ruff reporting that it could
      not open the file, and it is worse here than the formatter's ``io`` entry
      is there: that one *vanishes* from the tally, this one **replaces** it. A
      file holding twenty findings that becomes unreadable measures one — which
      reads as an ordinary small backlog rather than as an absence, and at a
      ceiling of 1,685 is absorbed without trace.
    """
    cells = contract["tools"]["ruff"]["cells"]
    config = contract["tools"]["ruff"]["config"]
    files = _files_in(cells, files, only)
    if not files:
        return []
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

    # Before the parse, not after: the failure this catches produces *empty*
    # stdout, which the `or "[]"` below turns into a valid, and entirely
    # fictional, measurement of zero. Checked here, the developer gets ruff's
    # own complaint instead.
    _refuse_non_verdict("ruff", result)

    try:
        findings: list[dict[str, Any]] = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"ruff did not emit JSON, so nothing was measured: {exc}\n{result.stderr[:800]}"
        ) from exc

    unreadable = [
        entry for entry in findings if _RUFF_IO_ERROR & {entry.get("code"), entry.get("name")}
    ]
    if unreadable:
        detail = "\n".join(
            f"  {entry.get('filename', '?')}: {entry.get('message', entry.get('code', '?'))}"
            for entry in unreadable[:20]
        )
        raise SystemExit(
            f"ruff could not read {len(unreadable)} file(s) and reported that "
            f"instead of their findings, so the counts below are wrong by "
            f"however many those files hold:\n{detail}"
        )

    _refuse_disagreement("ruff", result, len(findings), "finding(s)")

    return findings


def ruff_findings(reported: list[dict[str, Any]]) -> list[Finding]:
    """Each entry ruff reported, as a path and the rule it names.

    ``or`` rather than a default on both halves: a path-less diagnostic carries
    the key with a null value, which a default never sees, and ruff reports a
    syntax error with a null ``code``. The empty name lands in ``unattributed``
    and the missing rule lands in ``UNCODED``, both of which are reported —
    rather than raising inside the tally, or vanishing from it.
    """
    return [
        Finding(_relative(entry.get("filename") or ""), entry.get("code") or UNCODED)
        for entry in reported
    ]


def measure_ruff(
    contract: dict[str, Any], files: list[PurePosixPath], only: set[str] | None = None
) -> Measurement:
    """Findings per cell, from one guarded ruff pass over the whole population."""
    cells = contract["tools"]["ruff"]["cells"]
    findings = ruff_findings(_ruff_report(contract, files, only))
    return _tally(cells, [finding.path for finding in findings])


#: How mypy reports one finding. The trailing group is the rest of the line,
#: which is where the rule name is — at the end, in brackets, and absent
#: altogether on a few errors. So *whether* a line is a finding and *which rule*
#: it names are two separate questions, and only the first may decide whether
#: the line is counted: a finding with no bracketed code is still a finding, and
#: dropping it would make the census total short of the measurement's.
_MYPY_FINDING_RE = re.compile(r"([^:]+):\d+:(?:\d+:)?\s*error:(.*)$")

#: The rule name mypy writes at the end of a finding, in brackets.
_MYPY_CODE_RE = re.compile(r"\[([a-zA-Z0-9_-]+)\]\s*$")


def mypy_lines(output: str) -> Iterator[tuple[str, str, str]]:
    """Each finding line in a mypy run's output, as ``(path, body, line)``.

    The single scan of that output, and the reason it is a function rather than
    a loop inside the one caller it used to have. Three readers now want
    something from these lines — the ratchet wants the path, the census wants
    the rule, and ``charge`` wants the message text to show a developer — and
    three loops over the same text is three chances to disagree about which
    lines are findings. mypy emits ``note:`` continuations, and a reader that
    treated one as a finding, or dropped an ``error:`` line for carrying no
    bracketed rule, would move them apart silently.

    The path is normalised here rather than by each caller, so a run that
    reported absolute paths cannot leave one reader matching against
    repo-relative names and finding nothing.
    """
    for line in output.splitlines():
        match = _MYPY_FINDING_RE.match(line)
        if match:
            yield _relative(match.group(1)), match.group(2), line


def mypy_findings(output: str) -> list[Finding]:
    """Every finding in a mypy run's output, as a path and the rule it names."""
    return [
        Finding(path, code.group(1) if (code := _MYPY_CODE_RE.search(body)) else UNCODED)
        for path, body, _line in mypy_lines(output)
    ]


def mypy_targets(
    cells: list[dict[str, Any]],
    only: set[str] | None,
    include_unmeasured: bool,
    files: list[PurePosixPath],
) -> list[str]:
    """The directories one mypy pass is pointed at.

    ``include_unmeasured`` is the whole of what stands between the contract's
    bottom tier and a number. Those cells are not measured as zero — they are
    not measured, and the ceiling ``verify`` insists they carry is zero
    precisely so that nothing reads their silence as a count. Widening the
    target set here is how the census asks what is actually in them.

    A directory holding no tracked ``*.py`` is dropped, because mypy exits 2 on
    one — reporting nothing, over every other target in the same pass. The other
    two measurers cannot reach that: both are handed the population directly, and
    a file list cannot hold a path that is not a Python file. Only this tool is
    pointed at directories, so only this tool can be pointed at an empty one.

    The drop is of directories, never of cells. ``packages/*/docs`` matches seven
    of them and one holds a single ``.py`` file; the cell exists so that file is
    in some cell rather than silently in none, and it is still measured, over the
    one directory that has it. So the zero such a cell reports is a count of what
    is there and not a run that failed to happen — which is the distinction the
    exit-status guard below exists to keep, and this keeps it without the pass
    having to die to do so.

    Filtered against the population git tracks rather than a walk of the tree,
    for the reason ``tracked_python`` gives: a directory holding one untracked
    ``.py`` would otherwise be measured, and the file in it belongs to no cell.
    """
    skipped = frozenset() if include_unmeasured else _UNMEASURED_TIERS
    return sorted(
        {
            str(path)
            for cell in _restrict(cells, only)
            if cell["tier"] not in skipped
            for path in _expand(cell["path"])
            if any(cell_matches(tracked, str(path)) for tracked in files)
        }
    )


def _mypy_report(
    contract: dict[str, Any],
    files: list[PurePosixPath],
    only: set[str] | None = None,
    *,
    include_unmeasured: bool = False,
    config: str | None = None,
    cache_dir: str | None = None,
) -> tuple[list[Finding], str]:
    """One mypy pass, as its findings and the output they were read from.

    mypy is given directories rather than a file list: handed individual files
    it still follows imports, and the same finding then arrives under whichever
    file reached it first.

    The target set is taken from the contract rather than from
    ``bin/validate.sh --print-targets``. That used to be a statement about two
    lists that had to agree; it is now the only list there is, since the script
    reaches its mypy verdict by calling this. Deriving it the other way round
    would make the ratchet move whenever that script's default changed.

    ``config`` and ``cache_dir`` are overridden only by the census, which asks
    what the tree measures under a configuration nobody has adopted yet. Both
    default to the contract's, so the ratchet cannot be measured under anything
    but the declared one.

    Guarded the way the other two measurers are, which for as long as this
    function existed it was not. mypy exits 2 on a config file it cannot find,
    on a usage error and on a blocking error, having written nothing to stdout —
    and an empty stdout parses to an empty finding list, so **every mypy cell
    measured zero**, ``check`` exited 0, and the type-check half of the gate
    reported green. The census widened the ways to reach that: a generated
    config deleted by a concurrent run is a missing ``--config-file``, and
    "0 findings without the relaxations" is the strongest claim this tool can
    make.
    """
    cells = contract["tools"]["mypy"]["cells"]
    targets = mypy_targets(cells, only, include_unmeasured, files)
    if not targets:
        return [], ""

    chosen = config or contract["tools"]["mypy"]["config"]
    command = ["uv", "run", "mypy", *targets, "--config-file", chosen]
    if cache_dir:
        command += ["--cache-dir", cache_dir]
    result = _run(command)

    # Both, and in this order. The status catches a mypy that never ran; the
    # agreement catches one that ran and reported in a shape the parse below
    # does not recognise — a blocking error carries no `path:line:`, so it exits
    # non-zero while `mypy_findings` returns nothing at all.
    _refuse_non_verdict("mypy", result)
    findings = mypy_findings(result.stdout)
    _refuse_disagreement("mypy", result, len(findings), "finding(s)")
    return findings, result.stdout


def measure_mypy(
    contract: dict[str, Any], files: list[PurePosixPath], only: set[str] | None = None
) -> Measurement:
    """Findings per cell, from one mypy pass over the cells it actually covers.

    Uses the population the other two measurers are given, though it passes
    directories to the tool rather than files: it is what decides whether a
    directory holds any Python to open. Taking it from the shared signature
    rather than asking git again keeps one answer to what this repository tracks
    — the ratchet and the census would otherwise each have their own, and a
    population that can differ is a cell that can measure two ways.

    Scoping to ``only`` measures fewer cells, not fewer files within one: a
    ceiling is a whole-cell property, so a partial count compared against it is
    not a verdict. That the two agree is not an assumption — the per-cell
    numbers from a run over one package are identical to that package's numbers
    from a run over all fourteen.
    """
    cells = contract["tools"]["mypy"]["cells"]
    findings, output = _mypy_report(contract, files, only)
    return _tally(cells, [finding.path for finding in findings])._replace(output=output)


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
    _refuse_non_verdict("ruff format", result)
    _refuse_disagreement("ruff format", result, len(named), "unformatted file(s)")

    return _tally(cells, sorted(named))


def _relative(name: str) -> str:
    """A tool's reported path as a repo-relative POSIX one.

    An empty name is returned as it came, because ``Path("")`` is ``Path(".")``
    and resolves to the working directory — so a diagnostic that named no file
    would arrive here nameless and leave holding the repository root. Down that
    route it is not reported as unattributed; it is attributed, to whichever
    cell the root happens to match.
    """
    if not name:
        return name
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


#: The four categories a decline may carry. Four rather than three because
#: "declined with an argument" and "declined without one" are the distinction
#: worth having, and with three categories they are indistinguishable without
#: reading prose — which is the state the linting page had drifted into.
DECLINE_CATEGORIES = frozenset(
    {"presentational", "covered-elsewhere", "behavioural", "provisional"}
)

#: The covers a ``covered-elsewhere`` reason may name. A rule code counts too,
#: which the caller checks separately: naming ``D211`` as the cover for ``D203``
#: is a real answer, and enumerating every code here would be a second copy of
#: the rule list.
DECLINE_COVERS = ("mypy", "ruff format")

#: ``"CODE", # [category] reason`` — the entry line. The marker is required by
#: the guard rather than by this parser, so an unmarked entry parses with
#: ``category=None`` and is *reported* instead of being silently skipped.
_DECLINE_RE = re.compile(r'^\s*"([A-Z]+[0-9]+)"\s*,\s*#\s*(?:\[([a-z-]+)\]\s*)?(.*)$')

#: A comment-only line, which under an entry is that entry's argument.
_CONTINUATION_RE = re.compile(r"^\s+#\s?(.*)$")

#: ``"pattern" = ["CODE", ...]  # reason`` — a per-file waiver.
_PER_FILE_RE = re.compile(r'^\s*"([^"]+)"\s*=\s*\[([^\]]*)\]\s*(?:#\s*(.*))?$')


class Decline(NamedTuple):
    """One entry in ``[tool.ruff.lint] ignore``, with its category and argument."""

    code: str
    category: str | None
    reason: str
    argument: tuple[str, ...]

    @property
    def prose(self) -> str:
        """Everything the entry says, first line and argument together."""
        return " ".join([self.reason, *self.argument]).strip()


class PerFileWaiver(NamedTuple):
    """One entry in ``[tool.ruff.lint.per-file-ignores]``."""

    pattern: str
    codes: tuple[str, ...]
    reason: str


def _block(text: str, opening: str) -> list[str]:
    """The lines between ``opening`` and the first line that is a bare ``]``."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(opening))
    except StopIteration:
        return []
    for offset, line in enumerate(lines[start + 1 :], start=start + 1):
        if line.rstrip() == "]":
            return lines[start + 1 : offset]
    return []


def parse_declines(text: str) -> list[Decline]:
    """Every globally declined rule, with the category and argument beside it.

    A line parse of a block this repository owns, rather than ``tomllib``, for
    the one reason ``tomllib`` cannot serve: TOML discards comments, and the
    category and the argument *are* comments. Keeping them in the config is what
    stops the classification drifting from the rules it classifies, which is the
    failure the prose page it replaces had already reached — 48 of the 83
    declines unmentioned there, and one rule documented as declined that is
    enforced.

    Being a re-implementation of a parse the toolchain also does, it is a
    liability rather than a convenience, and it is pinned: the guard asserts
    this function and ``tomllib`` return the same code set in both directions.
    Without that, an entry this regex failed to match would be an uncategorized
    rule slipping past the check written to catch uncategorized rules.
    """
    declines: list[Decline] = []
    lines = _block(text, "ignore = [")
    index = 0
    while index < len(lines):
        match = _DECLINE_RE.match(lines[index])
        if match is None:
            index += 1
            continue
        # An argument is aligned *under* its entry's comment. A section header
        # sits at the block's own indentation, to the left of every entry's `#`,
        # and without this column test the run swallows it and everything after
        # it: SIM113 was reported as arguing twelve lines about SIM115, because
        # the two blocks have no blank line between them. Worse than the wrong
        # display, an entry written directly above a header would inherit it and
        # satisfy the check that requires [behavioural] to be argued.
        comment_column = lines[index].index("#")
        argument: list[str] = []
        index += 1
        while index < len(lines):
            continuation = _CONTINUATION_RE.match(lines[index])
            if continuation is None or _DECLINE_RE.match(lines[index]):
                break
            if lines[index].index("#") < comment_column:
                break
            argument.append(continuation.group(1).strip())
            index += 1
        declines.append(
            Decline(match.group(1), match.group(2), match.group(3).strip(), tuple(argument))
        )
    return declines


def _waiver_reasons(text: str) -> dict[str, str]:
    """The trailing comment on each per-file waiver line, which TOML discards."""
    reasons: dict[str, str] = {}
    started = False
    for raw in text.splitlines():
        if raw.startswith("[tool.ruff.lint.per-file-ignores]"):
            started = True
            continue
        if not started:
            continue
        if raw.startswith("["):
            break
        match = _PER_FILE_RE.match(raw)
        if match is not None:
            reasons[match.group(1)] = (match.group(3) or "").strip()
    return reasons


def parse_per_file_waivers(text: str) -> list[PerFileWaiver]:
    """Every per-file waiver, with the reason its line carries.

    Separate from ``parse_declines`` because the two answer different questions
    about the same finding — whether a rule is declined everywhere, or waived
    here — and ``explain`` has to distinguish them to be worth asking.

    Split by what each reader can actually see, the way ``enabled_rules`` splits
    the question one scope up. The pattern and the code list are TOML *data*, so
    ``tomllib`` reads them and this module does not get a second opinion; only
    the reason is a comment, which is the one thing ``tomllib`` cannot return.

    It was a line regex over both until the second reading. That regex is
    anchored to a single line by construction, so a waiver reformatted across
    lines — ordinary, and what a formatter does to a long one — simply stopped
    being a waiver. Nothing compared the two readings, and the population had no
    floor worth the name: 15 declared against 31 real, room for sixteen to
    vanish. Both consequences point the unsafe way. A waiver nobody parses is
    never checked for a reason, and ``explain`` calls its file enforced — which
    tells a worker a suppressed finding is live.

    Reading the data from the reader that cannot miss it retires that whole
    class rather than guarding it. What is left is a reason that failed to
    parse, which arrives as an empty string and fails the guard that requires
    one, by name, loudly.
    """
    declared = (
        tomllib.loads(text).get("tool", {}).get("ruff", {}).get("lint", {}).get("per-file-ignores")
    )
    if not declared:
        return []
    reasons = _waiver_reasons(text)
    return [
        PerFileWaiver(pattern, tuple(codes), reasons.get(pattern, ""))
        for pattern, codes in declared.items()
    ]


#: The inline suppression channel, which no artifact reported until this leg.
#:
#: Its *spelling* is guarded — a payload ruff cannot read, a keyword run into
#: other text, a code ruff does not have, a blanket waiver — and its *deadness*
#: is guarded by RUF100 in the cells that enforce it. Its size was reported
#: nowhere, and a code is not a reason, so a channel of several hundred waivers
#: sat outside every count the contract takes.
#:
#: The parser below is a re-implementation of ruff's, which is a liability
#: rather than a convenience, so it is pinned: test_suppression_directives
#: drives the real binary over every spelling this claims to know and compares
#: verdicts. It lives here rather than in that test because two callers now
#: want it — the guard, and the per-cell count in ``inline_waivers`` — and a
#: second copy is how the two come to disagree about what a directive is while
#: both report a number.

#: ``noqa`` is case-insensitive, and everything after it is captured raw rather
#: than pre-split into colon-and-payload. What follows the keyword decides which
#: of four states the directive is in, and two of those differ by one character:
#: the keyword then a space is blanket, the keyword run straight into a letter is
#: not a directive ruff will read at all. A regex that made the colon optional
#: and skipped to the payload could not tell them apart, and called the second
#: one blanket. Verified against the binary rather than assumed.
#:
#: Spelled without a leading hash above, deliberately. This module is in its own
#: scan, and a comment here that spelled a directive would be reported by it --
#: which is the constraint the module docstring describes, met rather than
#: exempted. The examples live in docstrings and test inputs, which are strings.
NOQA_RE = re.compile(r"#\s*noqa(?P<trailer>[^#\n]*)", re.IGNORECASE)

#: A rule code as ruff spells one. Deliberately not anchored to the families
#: this repo selects: a directive naming a real rule from an unselected family
#: is a waiver that is merely inactive, which is a policy question rather than
#: a defect.
CODE_RE = re.compile(r"^[A-Z]+[0-9]+$")

#: The three unhealthy outcomes, named rather than spelled at each use site: the
#: parity test has to know which of them ruff warns about, and a literal in both
#: places is how that pair drifts.
BLANKET = "blanket"
UNREADABLE = "unreadable"
RUN_ON = "run_on"


def _leading_codes(payload: str) -> list[str]:
    """The codes ruff reads before it stops, which is how ruff itself parses.

    Measured against ruff 0.16.1: it takes comma- or space-separated codes from
    the front of the payload and stops at the first token that is not one, with
    no complaint about whatever follows. That is why ``# noqa: F401 - keeps the
    re-export`` is valid and ``# noqa: not calling`` is not -- the difference is
    whether *anything* was read, not whether prose is present.
    """
    codes = []
    for token in re.split(r"[,\s]+", payload.strip()):
        if not CODE_RE.match(token):
            break
        codes.append(token)
    return codes


def _classify(trailer: str) -> tuple[str, list[str]]:
    """Which of ruff's four outcomes the text after ``noqa`` produces.

    ruff reads the keyword and then requires end-of-comment, whitespace, or
    ``:``. Every other character -- letter, digit, underscore, hyphen, dot,
    paren -- makes the whole thing an invalid directive, with a warning worded
    differently from the unreadable-payload one. Measured across all six against
    ruff 0.16.1; the split is not inferred from the message text.
    """
    if trailer.startswith(":"):
        codes = _leading_codes(trailer[1:])
        return ("codes" if codes else UNREADABLE), codes
    if trailer == "" or trailer[0].isspace():
        return BLANKET, []
    return RUN_ON, []


def directives(source: str) -> list[tuple[int, str, list[str]]]:
    """Every directive in ``source`` as ``(lineno, kind, codes)``.

    ``kind`` is one of four, because ruff distinguishes four and collapsing any
    pair of them loses a defect:

    * ``"codes"`` -- a colon and at least one readable code. The only healthy one.
    * ``BLANKET`` -- the keyword alone, or followed by whitespace. Valid to ruff,
      rejected here, because it waives rules nobody has written yet.
    * ``UNREADABLE`` -- a colon whose payload is not codes (``# noqa: not calling``).
    * ``RUN_ON`` -- the keyword running into other text (``# noqafoo``).

    ``UNREADABLE`` and ``RUN_ON`` are one outcome to ruff -- both warn, both
    suppress nothing -- and two here, because the remedy differs: one directive
    needs its payload rewritten, the other is not a directive at all.

    ``RUN_ON`` is the one this parser did not have. It fell into ``BLANKET``,
    since the only question asked was whether a colon was present, and so
    ``# noqafoo`` was reported as suppressing every rule on the line when it
    suppresses none. It surfaced when the parity test below was given inputs of
    that shape and disagreed with the binary.

    Tokenized rather than scanned line by line, because a line scan reports a
    directive spelling inside a *string* and ruff does not. That divergence is
    not theoretical: this module's own test inputs are such strings, and the
    first draft of this function failed the file it lives in. A guard that
    disagrees with the tool on its own source is a guard whose first bug report
    is answered with an exemption.
    """
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = NOQA_RE.search(token.string)
        if match is None:
            continue
        kind, codes = _classify(match.group("trailer"))
        found.append((token.start[0], kind, codes))
    return found


#: The type checker's half of the same channel, and the larger half: the
#: ``type: ignore`` directive, with an optional bracketed code list.
#:
#: Spelled without its leading hash above, deliberately, and the reason is not
#: fastidiousness — it was measured. Written the natural way, this comment was
#: itself counted as a directive, because a comment token spelling one is one.
#: The count reported 459 and one of them was this line. The sibling guard
#: keeps its examples inside docstrings for the same reason and says so; the
#: rule generalises to anything that scans the tree it lives in.
_TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore(?P<codes>\[[^\]]*\])?")


class InlineCount(NamedTuple):
    """How many inline waivers one cell holds, by tool."""

    cell: str
    tier: str
    suppressions: int
    bare: int


def type_ignores(source: str) -> list[tuple[int, bool]]:
    """``(lineno, carries_a_code)`` for every ``# type: ignore`` comment."""
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = _TYPE_IGNORE_RE.search(token.string)
        if match is not None:
            found.append((token.start[0], match.group("codes") is not None))
    return found


def inline_waivers(contract: dict[str, Any], tool: str) -> list[InlineCount]:
    """Every inline waiver of ``tool``, attributed to the cell it sits in.

    Counted per cell rather than per repository because that is the unit every
    other number in the contract uses, and because the answer is lopsided in a
    way a total would hide: the directives concentrate in cells the tool does
    not read at all, where nothing could report them as unused either.

    The scan reads comment tokens, so a directive spelled inside a string is not
    counted — which is not a rounding difference. A grep over the same tree
    returns half again as many ``# noqa`` as this does, and nearly all of the
    excess is one guard's own test inputs. The tokenized figure is the one that
    matches what ruff would act on.
    """
    scanner = directives if tool == "ruff" else type_ignores
    cells = contract["tools"][tool]["cells"]
    per_cell: dict[str, list[int]] = {cell["path"]: [0, 0] for cell in cells}
    for name in tracked_python():
        path = _ROOT / name
        if not path.is_file():
            continue
        cell = _cell_for(cells, str(name))
        if cell is None:
            continue
        for record in scanner(path.read_text(encoding="utf-8")):
            per_cell[cell][0] += 1
            if (tool == "ruff" and record[1] != "codes") or (tool != "ruff" and not record[1]):
                per_cell[cell][1] += 1
    return [
        InlineCount(cell["path"], cell["tier"], *per_cell[cell["path"]])
        for cell in cells
        if per_cell[cell["path"]][0]
    ]


#: What ``explain`` can say about one code at one path. The first is the only
#: one that means a finding would be shown; the other three are three different
#: reasons it would not, and collapsing any pair of them loses the answer the
#: question was asked for.
EXPLAIN_REPORTED = "reported"
EXPLAIN_DECLINED = "declined globally"
EXPLAIN_WAIVED = "waived for this file"
EXPLAIN_UNSELECTED = "not selected"

#: ``ruff check --show-settings`` prints the fully resolved rule set, one
#: ``name (CODE),`` per line. Read rather than derived, for a reason measured
#: here: ``select`` lists the legacy selector ``TCH`` while the rules it enables
#: are spelled ``TC001``-``TC003``, so a prefix match over the declared families
#: reports three enforced rules as unselected. Every re-implementation of a
#: resolution the tool already performs has a case like that in it.
_ENABLED_BLOCK_RE = re.compile(r"^linter\.rules\.enabled = \[(.*?)^\]", re.DOTALL | re.MULTILINE)
_RULE_CODE_RE = re.compile(r"\(([A-Z]+\d+)\)")

#: What the audit measures over. Named once because two functions ask the same
#: question and a second copy is how they come to disagree about the population
#: while both report a number.
#:
#: These are *audit* reads — "how much does this decline stand in front of" —
#: not the ratchet, which derives its population from ``tracked_python()``
#: filtered through the contract's cells and is the authority on what is
#: measured. A cell added to the contract outside these roots is counted by the
#: ratchet and not by the audit, which is a gap in the audit's reach and not in
#: the ceilings.
_MEASURED_TARGETS = ("packages", "bin", "src", "tests", "conftest.py")


class Explanation(NamedTuple):
    """Whether a code is reported at a path, and if not, why not."""

    code: str
    path: str | None
    verdict: str
    category: str | None = None
    reason: str = ""
    argument: tuple[str, ...] = ()
    pattern: str | None = None

    @property
    def reported(self) -> bool:
        return self.verdict == EXPLAIN_REPORTED


def enabled_rules(config: Path = _ROOT / "pyproject.toml") -> frozenset[str]:
    """Every rule code the configuration actually enables, asked of ruff.

    The authority for *whether* a code is enforced is the tool; this module's
    business is *why* it is not, which the tool cannot say because the reasons
    are comments. Splitting the question that way is what keeps ``explain`` from
    becoming a second implementation of ruff's selector resolution — a thing that
    can disagree with ruff while looking authoritative, which is worse than not
    answering.
    """
    result = _run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "--config",
            str(config),
            "--show-settings",
            str(_ROOT / "pyproject.toml"),
        ]
    )
    block = _ENABLED_BLOCK_RE.search(result.stdout)
    if block is None:
        raise SystemExit(
            "ruff --show-settings reported no rule set; explain cannot answer "
            f"without it:\n{result.stdout}\n{result.stderr}"
        )
    return frozenset(_RULE_CODE_RE.findall(block.group(1)))


@cache
def selector_rules(
    selectors: tuple[str, ...], config: Path = _ROOT / "pyproject.toml"
) -> frozenset[str]:
    """Every rule code a set of selectors names, asked of ruff.

    ``enabled_rules`` above declines to re-implement ruff's selector resolution,
    for a reason it measured. This is the same question one scope down — what
    ``["D", "N", "UP"]`` on a per-file waiver reaches — and it was answered here
    by ``code.startswith(named)`` until the second reading of that argument.

    A prefix test is not ruff's rule, and it is wrong in *both* directions.
    ``"NPY002".startswith("N")`` is true and ``N`` is pep8-naming, a different
    linter from numpy — so a waiver written to relax naming reported itself as
    covering a rule it does not reach. In the other direction ``PL`` names
    ``PLC0415`` to ruff and no letter-prefix rule can see it, because the
    pylint selector spans ``PLC``/``PLE``/``PLR``/``PLW``.

    The repository's own settings are kept rather than resolved under
    ``--isolated``: ``pydocstyle.convention`` and ``target-version`` genuinely
    change which codes a family names here — 95 rules against 105 for these
    three — and the wider answer is the one that claims a waiver covers what it
    does not. Only ``ignore`` is neutralised, so the question stays "does this
    selector name this code" rather than "and is it declined elsewhere", which
    is the caller's own next question and already answered before this is asked.

    Cached because the ten distinct code sets in this config are asked about
    once per waiver per lookup, and each answer costs a ruff invocation.
    """
    result = _run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "--config",
            str(config),
            "--config",
            "lint.ignore = []",
            "--select",
            ",".join(selectors),
            "--show-settings",
            str(_ROOT / "pyproject.toml"),
        ]
    )
    block = _ENABLED_BLOCK_RE.search(result.stdout)
    if block is None:
        raise SystemExit(
            f"ruff --show-settings reported no rule set for {','.join(selectors)}; "
            f"explain cannot say what that waiver covers:\n{result.stdout}\n{result.stderr}"
        )
    return frozenset(_RULE_CODE_RE.findall(block.group(1)))


def waiver_covers(waiver: PerFileWaiver, code: str, path: str) -> bool:
    """Whether a per-file waiver reaches this code at this path.

    One approximation of ruff remains, and it is the deliberate one: a pattern
    with no separator matches a basename, which is how ``__init__.py`` reaches
    every package — with a separator it matches the repo-relative path. A path
    this gets wrong is reported as enforced, which is the direction that sends
    someone to look rather than the direction that tells them not to.

    The code half used to be an approximation too, and it failed in the opposite
    direction — see ``selector_rules``, which now answers it. The path is tested
    first because that half is local and this one costs a subprocess.
    """
    if "/" in waiver.pattern:
        if not fnmatch.fnmatchcase(path, waiver.pattern):
            return False
    elif not fnmatch.fnmatchcase(PurePosixPath(path).name, waiver.pattern):
        return False
    return code in selector_rules(waiver.codes)


def explain_code(
    code: str,
    path: str | None = None,
    config: Path = _ROOT / "pyproject.toml",
    enabled: frozenset[str] | None = None,
) -> Explanation:
    """Is this code reported at this path, and if not, why not.

    The lookup that replaces the argument. A finding a worker "fixed" because it
    looked wrong, or reported because a narrowed run surfaced it, is a finding
    nobody asked the configuration about — and until now the configuration could
    only be asked by reading 500 lines of TOML and knowing which of four
    mechanisms applied.

    Read-only and always exit 0. It is a lookup, not a check, so it must not
    become something a script can fail on: that role belongs to ``check``.
    """
    text = config.read_text(encoding="utf-8")
    if enabled is None:
        enabled = enabled_rules(config)

    if code not in enabled:
        declined = {d.code: d for d in parse_declines(text)}
        entry = declined.get(code)
        if entry is None:
            return Explanation(code, path, EXPLAIN_UNSELECTED)
        return Explanation(
            code,
            path,
            EXPLAIN_DECLINED,
            category=entry.category,
            reason=entry.reason,
            argument=entry.argument,
        )

    if path is not None:
        relative = _relative(path)
        for waiver in parse_per_file_waivers(text):
            if waiver_covers(waiver, code, relative):
                return Explanation(
                    code, relative, EXPLAIN_WAIVED, reason=waiver.reason, pattern=waiver.pattern
                )
        return Explanation(code, relative, EXPLAIN_REPORTED)

    return Explanation(code, path, EXPLAIN_REPORTED)


def explain_report(explanation: Explanation, out: Any) -> None:
    """One verdict, written the way a reader asked the question."""
    if explanation.reported:
        out.write(f"{explanation.verdict}\n")
        return
    out.write("not reported\n")
    if explanation.verdict == EXPLAIN_UNSELECTED:
        out.write(
            f"  {EXPLAIN_UNSELECTED} — no family in [tool.ruff.lint] select "
            f"matches {explanation.code}\n"
        )
        return
    if explanation.verdict == EXPLAIN_WAIVED:
        out.write(f'  {EXPLAIN_WAIVED} — "{explanation.pattern}"\n')
        out.write(f"    {explanation.reason}\n" if explanation.reason else "")
        return
    marker = f"  [{explanation.category}]" if explanation.category else "  [uncategorized]"
    out.write(f"  {EXPLAIN_DECLINED}{marker}  {explanation.reason}\n")
    for line in explanation.argument:
        out.write(f"    {line}\n")


def decline_audit(config: Path = _ROOT / "pyproject.toml") -> dict[str, Any]:
    """Every decline, by category, with the unargued ones totalled.

    The total is the product. Fifteen fixed sites is what this leg looks like
    from the diff; a population that was uncounted becoming counted, categorized
    and guarded is what it actually did, and ``provisional`` is the row whose
    target is zero.
    """
    declines = parse_declines(config.read_text(encoding="utf-8"))
    by_category: dict[str, list[Decline]] = defaultdict(list)
    for entry in declines:
        by_category[entry.category or "uncategorized"].append(entry)
    return {
        "total": len(declines),
        "by_category": {name: sorted(rows) for name, rows in sorted(by_category.items())},
    }


def unsafely_fixable(codes: list[str], config: Path = _ROOT / "pyproject.toml") -> dict[str, int]:
    """Which of these codes have findings ruff will not fix without ``--unsafe-fixes``.

    The authority behind the ``presentational`` category, which otherwise has
    none. That category claims no finding of a rule can be a behaviour
    difference, and it is the largest of the four and the only one nothing
    checked — so the cheapest way to satisfy every guard was to assert the most.

    ruff already publishes an opinion on this per finding. An ``unsafe``
    applicability is ruff saying it will not apply the fix unasked because doing
    so may change what the code does — which is the negation of the claim, made
    by the tool rather than by a reader of the rule's name.

    It is *evidence*, not a verdict, and the guard treats it that way: several of
    these are unsafe over comment preservation or a preview flag rather than
    semantics. What it establishes is that the entry needs an argument, which is
    the same standard ``behavioural`` is already held to.

    One invocation for the whole category, and the count per code so the argument
    can name it.
    """
    if not codes:
        return {}
    result = _run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            *_MEASURED_TARGETS,
            "--config",
            str(config),
            "--select",
            ",".join(sorted(codes)),
            "--no-cache",
            "--quiet",
            "--output-format",
            "json",
        ]
    )
    _refuse_non_verdict("ruff", result)
    try:
        findings: list[dict[str, Any]] = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"ruff did not return JSON, so nothing was measured: {exc}\n{result.stdout[:400]}"
        ) from exc

    unsafe: Counter[str] = Counter()
    for finding in findings:
        fix = finding.get("fix") or {}
        if fix.get("applicability") == "unsafe":
            unsafe[finding["code"]] += 1
    return dict(unsafe)


def measure_declines(codes: list[str], config: Path = _ROOT / "pyproject.toml") -> dict[str, int]:
    """How many findings each declined code stands in front of.

    ``--select`` on the command line overrides the config's ``ignore``, which is
    what makes this a read rather than a config edit. One run for all of them:
    83 invocations would take long enough that nobody would ask.
    """
    if not codes:
        return {}
    result = _run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            *_MEASURED_TARGETS,
            "--config",
            str(config),
            "--select",
            ",".join(sorted(codes)),
            "--no-cache",
            "--quiet",
            "--statistics",
        ]
    )

    # Before the parse, for the reason the canonical measurer states above: the
    # failure this catches produces *empty* stdout, and the parse below turns
    # that into a measured zero for every code asked about. Those zeroes are not
    # inert here — `provisional` entries carry a measured count, and a count of
    # zero is what an entry looks like when its findings have been cleared.
    _refuse_non_verdict("ruff", result)

    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) >= 2 and fields[0].isdigit():
            counts[fields[1]] = int(fields[0])
    return {code: counts.get(code, 0) for code in sorted(codes)}


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


#: Where a census configuration is written while it is being measured under.
#:
#: At the repository root, and not in a temporary directory, because
#: ``mypy_path`` is a colon-joined list of *relative* paths and the contract's
#: cells are relative too. The same file elsewhere resolves a different tree and
#: answers a different question — plausibly, and without saying so.
CENSUS_CONFIG = ".mypy-census.toml"

#: A cache of its own for a run under a configuration the tree is not otherwise
#: measured under. mypy is supposed to invalidate on a configuration change; a
#: measurement taken once, to decide the shape of everything after it, is not
#: the place to find out whether it always does.
CENSUS_CACHE = ".mypy_cache/census"


def first_party_modules(config: Path) -> set[str]:
    """Top-level module names this workspace's own sources provide.

    Read out of the type checker's ``mypy_path`` rather than listed here,
    because this set is what decides which override sections a census removes.
    A written-down list goes stale on the day a package is added — and it goes
    stale *quietly*, leaving that package's strictness relaxation in force
    through a run whose whole purpose was to measure without them.

    A module shipped as a single ``*.py`` counts, which reading only directories
    would miss. Every module this workspace ships today is a package, so that is
    a hole rather than a bug — and it is the same quiet one: the missed name
    reads as third-party, its overrides survive the removal, and the census
    reports a smaller number under a heading saying the relaxations are gone.
    """
    parsed = tomllib.loads(config.read_text(encoding="utf-8"))
    declared = parsed.get("tool", {}).get("mypy", {}).get("mypy_path", "")
    roots = declared if isinstance(declared, list) else str(declared).split(":")

    names: set[str] = set()
    for root in roots:
        source = _ROOT / str(root).strip()
        if not source.is_dir():
            continue
        for entry in source.iterdir():
            name = entry.name if entry.is_dir() else entry.stem
            if (entry.is_dir() or entry.suffix == ".py") and name.isidentifier():
                names.add(name)
    if not names:
        raise SystemExit(
            f"{_relative(str(config))} declares no mypy_path this workspace's own "
            "modules resolve from, so nothing can be identified as first-party "
            "and a census without relaxations would silently remove none of them"
        )
    return names


def _module_patterns(section: dict[str, Any]) -> list[str]:
    """The modules one override section names, in either of TOML's two spellings."""
    module = section.get("module", [])
    return [module] if isinstance(module, str) else [str(name) for name in module]


def _classified_patterns(
    section: dict[str, Any], first_party: set[str]
) -> tuple[list[str], list[str]]:
    """One override section's module patterns, split into ours and not-ours.

    Decided per pattern, because nothing requires a section to name only one
    kind and the classification is a property of each name it lists rather than
    of the section.

    A pattern not beginning with a module name — a leading ``*``, say — is
    refused rather than defaulted. Defaulting it either way is a silent answer
    to the question this whole mechanism exists to ask, and the direction it
    would default is the quiet one: unclassifiable reads as third-party, the
    section survives, and the census reports fewer findings than the heading
    over them claims.
    """
    ours: list[str] = []
    theirs: list[str] = []
    for pattern in _module_patterns(section):
        head = pattern.split(".")[0]
        if not head.isidentifier():
            raise SystemExit(
                f"the override section pattern {pattern!r} does not begin with a "
                "module name, so it cannot be told apart from a third-party one "
                "— a census without the relaxations would keep it and measure "
                "less than it says it did"
            )
        (ours if head in first_party else theirs).append(pattern)
    return ours, theirs


def _relaxes_first_party(section: dict[str, Any], first_party: set[str]) -> bool:
    """Whether an override section relaxes checking over code this repository ships.

    The distinction a census turns on. A section naming ``nltk.*`` waives the
    absence of type stubs in somebody else's library, which is not our backlog
    and not ours to fix. A section naming ``dataknobs_xization.*`` turns seven
    checks off over code we ship, and every finding it suppresses is one the
    declared configuration would otherwise report.

    A section naming both kinds is refused rather than resolved, because there
    is no resolution: removing it measures somebody else's missing stubs as our
    backlog, and keeping it leaves our own strictness relaxed through a run
    taken to remove exactly that. Answering it either way silently is how a
    census reports a number under a heading that does not describe it. Today's
    configuration has no mixed section — the one holding fifteen third-party
    patterns would flip all fifteen the day a first-party name joined it, which
    is precisely the edit nobody would think to check.
    """
    ours, theirs = _classified_patterns(section, first_party)
    if ours and theirs:
        raise SystemExit(
            f"the override section naming {sorted(ours + theirs)} covers both "
            f"code this repository ships ({sorted(ours)}) and code it does not "
            f"({sorted(theirs)}), so it can be neither removed nor kept without "
            "measuring the wrong tree — split it into one section per kind"
        )
    return bool(ours)


def config_without_relaxations(text: str, first_party: set[str]) -> tuple[str, list[str]]:
    """The configuration with every first-party override section removed.

    Text surgery rather than a re-serialisation, because the standard library
    reads TOML and does not write it, and adding a writer as a dependency to
    generate a file that is deleted seconds later fails the dependency bar. The
    sections are located by their headers and correlated with the parsed
    document by position, which array-of-tables order guarantees.

    Checked afterwards rather than trusted: the result is re-parsed, and both
    the number of surviving sections and their first-party-ness are compared
    against what the removal was supposed to do. A surgery that took out one
    section too many would otherwise produce a plausible number measured under a
    configuration nobody described.
    """
    parsed = tomllib.loads(text)
    sections = parsed.get("tool", {}).get("mypy", {}).get("overrides", [])
    doomed = {
        index
        for index, section in enumerate(sections)
        if _relaxes_first_party(section, first_party)
    }
    if not doomed:
        raise SystemExit(
            "no override section relaxes checking over first-party code, so a "
            "census without them would measure exactly what the ratchet already "
            "measures — run it without --without-overrides instead"
        )

    kept: list[str] = []
    index = -1
    dropping = False
    for line in text.splitlines(keepends=True):
        if line.startswith("["):
            # The comment comes off before the comparison. A header carrying one
            # is a thing a person writes, and in this repository's style a likely
            # one — and it was not the string being compared against, so the
            # block survived while the parse still counted it as doomed. The
            # correlation shifted and every later section was removed one place
            # out. The re-parse below caught that, which is what it is for; but a
            # feature that refuses whenever somebody annotates a header is not a
            # usable feature, and its complaint does not mention comments.
            #
            # Spellings TOML also allows but nobody writes — `[[ tool.mypy.overrides ]]`
            # with inner spaces — are left to that check rather than matched here.
            # Every spelling matched is one the re-parse no longer has to catch,
            # and a match loose enough to accept anything is one that no longer
            # locates sections.
            if line.split("#", 1)[0].strip() == "[[tool.mypy.overrides]]":
                index += 1
                dropping = index in doomed
            else:
                dropping = False
        if not dropping:
            kept.append(line)

    stripped = "".join(kept)
    surviving = tomllib.loads(stripped).get("tool", {}).get("mypy", {}).get("overrides", [])
    still_relaxed = [
        pattern
        for section in surviving
        if _relaxes_first_party(section, first_party)
        for pattern in _module_patterns(section)
    ]
    if len(surviving) != len(sections) - len(doomed) or still_relaxed:
        raise SystemExit(
            f"removing {len(doomed)} override section(s) left {len(surviving)} of "
            f"{len(sections)} behind, {len(still_relaxed)} of them still relaxing "
            "first-party code — the generated configuration does not describe "
            "what was asked for, so nothing was measured"
        )

    removed = [
        pattern
        for index, section in enumerate(sections)
        if index in doomed
        for pattern in _module_patterns(section)
    ]
    return stripped, removed


@contextmanager
def census_config(contract: dict[str, Any]) -> Iterator[tuple[str, list[str]]]:
    """A configuration with the first-party relaxations taken out, while it exists.

    Generated from the declared configuration on every run and deleted in a
    ``finally``, never hand-maintained. A second configuration that has to be
    kept in step with the first is the drift this whole mechanism exists to
    close, and one left behind by an interrupted run is a file a later commit
    could pick up.

    The path is a constant, so it is also shared: two censuses in one checkout
    write the same file, and the first to finish deletes it out from under the
    second — whose mypy then cannot find its config, exits 2 and, before the
    guard in ``_mypy_report``, reported zero findings under a heading claiming
    the relaxations had been removed. So an existing file is refused rather than
    overwritten. That covers the operator whose own file is at that path too,
    and it turns an interrupted run's leftovers into a sentence rather than into
    a second run that silently inherits them.
    """
    declared = _ROOT / contract["tools"]["mypy"]["config"]
    stripped, removed = config_without_relaxations(
        declared.read_text(encoding="utf-8"), first_party_modules(declared)
    )
    scratch = _ROOT / CENSUS_CONFIG
    if scratch.exists():
        raise SystemExit(
            f"{CENSUS_CONFIG} is already in the tree, so nothing was measured. "
            "Either a census is running in this checkout — they share this path, "
            "and the first to finish deletes it out from under the second — or "
            "one was interrupted before it could. Delete it once no census is "
            "running."
        )

    # Inside the `try`, so that a write which fails part way through — a full
    # disk, a read-only checkout — is cleaned up rather than left as a truncated
    # configuration at a path the next run refuses.
    try:
        scratch.write_text(stripped, encoding="utf-8")
        yield CENSUS_CONFIG, removed
    finally:
        scratch.unlink(missing_ok=True)


class CensusRun(NamedTuple):
    """One run, read twice: as counts per file, and as counts per rule.

    Both halves come from the same list of findings, so a disagreement between
    them is a defect in the bucketing rather than a difference between two runs.
    That is what makes them comparable at all, and comparing them is what
    ``tests/test_quality_census.py`` does.
    """

    measurement: Measurement
    census: Census
    config: str
    removed: list[str]


#: The tools a census can decompose. Every finding these two report names a
#: rule, so the number a ceiling is compared against breaks into rules. The
#: formatter's does not: its unit is files it would rewrite, and "this file is
#: not formatted" has no rule to attribute it to.
CENSUS_TOOLS = frozenset({"ruff", "mypy"})


def take_census(
    contract: dict[str, Any],
    tool: str,
    only: set[str] | None = None,
    *,
    include_unmeasured: bool = False,
    without_overrides: bool = False,
) -> CensusRun:
    """Measure one tool once, and bucket the findings by file and by rule.

    The request is refused rather than narrowed wherever part of it cannot be
    honoured. A flag a tool silently disregards — or a cell the run silently
    declines to visit — reports an answer to a narrower question than the one
    asked, under a heading that says otherwise, which is this repository's own
    defect class: an absence rendered as a result.

    All four refusals live here rather than in ``main``. The command line is one
    caller; the tests are another, and a future subcommand would be a third. A
    guard only the CLI applies is one the layer that actually dispatches on
    ``tool`` does not have.
    """
    if tool not in CENSUS_TOOLS:
        raise SystemExit(
            f"a census decomposes a count into the rules that produced it, and "
            f"{tool} reports files it would rewrite rather than rules broken — "
            "so this is a category error rather than a smaller version of the "
            "same question"
        )

    cells = contract["tools"][tool]["cells"]

    if without_overrides and tool != "mypy":
        raise SystemExit(
            f"--without-overrides removes per-module sections from the type "
            f"checker's configuration, which {tool} has none of"
        )
    if include_unmeasured and not any(cell.get("tier") in _UNMEASURED_TIERS for cell in cells):
        raise SystemExit(
            f"{tool} declares no cell in an unmeasured tier, so "
            "--include-unmeasured would widen the run by nothing"
        )

    # The inverse of the check above, and the one it was missing. Naming a cell
    # nothing reads asks for a row this run cannot produce: the target set skips
    # it, and the report filters it back out — so a scope consisting only of them
    # printed `0 finding(s)` over an empty table, which is exactly what a clean
    # tree prints. A mixed scope was worse: the unmeasured cells simply vanished
    # from a table the caller had named them in.
    if only is not None and not include_unmeasured:
        silent = sorted(
            cell["path"] for cell in _restrict(cells, only) if cell.get("tier") in _UNMEASURED_TIERS
        )
        if silent:
            raise SystemExit(
                f"the contract puts {silent} in a tier no tool reads, so this run "
                "would leave them out of the table rather than report them as "
                "zero — and a cell missing from a census cannot be told from one "
                "that measured nothing. Pass --include-unmeasured to read them."
            )

    config: str = contract["tools"][tool]["config"]
    removed: list[str] = []
    findings: list[Finding]
    # Read once and handed to whichever branch runs, so that a census and the
    # ratchet it decomposes cannot be looking at different populations.
    files = tracked_python()

    # Two branches for mypy rather than one, and no branch for anything else:
    # the guard above is what makes the final `else` the type checker rather
    # than whatever tool is not ruff.
    if tool == "ruff":
        findings = ruff_findings(_ruff_report(contract, files, only))
    elif without_overrides:
        with census_config(contract) as (generated, removed):
            config = generated
            findings, _output = _mypy_report(
                contract,
                files,
                only,
                include_unmeasured=include_unmeasured,
                config=config,
                cache_dir=CENSUS_CACHE,
            )
    else:
        findings, _output = _mypy_report(
            contract, files, only, include_unmeasured=include_unmeasured
        )

    return CensusRun(
        _tally(cells, [finding.path for finding in findings]),
        _tally_codes(cells, findings),
        config,
        removed,
    )


def _ranked(counts: Counter[str]) -> dict[str, int]:
    """Counts largest first, ties broken by name so a re-run reads identically."""
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def census_report(
    contract: dict[str, Any],
    tool: str,
    only: set[str] | None = None,
    *,
    include_unmeasured: bool = False,
    without_overrides: bool = False,
) -> dict[str, Any]:
    """What one tool found, per cell and per rule, as a reportable document.

    Every cell the run covered is listed, including the ones that measured
    nothing: a census whose table holds only the cells with findings cannot be
    told apart from one whose run never reached the rest.
    """
    run = take_census(
        contract,
        tool,
        only,
        include_unmeasured=include_unmeasured,
        without_overrides=without_overrides,
    )
    covered = [
        cell
        for cell in _restrict(contract["tools"][tool]["cells"], only)
        if include_unmeasured or cell.get("tier") not in _UNMEASURED_TIERS
    ]

    cells = []
    everywhere: Counter[str] = Counter()
    for cell in covered:
        counts = run.census.by_cell.get(cell["path"], Counter())
        everywhere += counts
        cells.append(
            {
                "cell": cell["path"],
                "tier": cell.get("tier"),
                "ceiling": cell.get("ceiling"),
                "total": sum(counts.values()),
                "codes": _ranked(counts),
            }
        )

    return {
        "tool": tool,
        "config": run.config,
        "include_unmeasured": include_unmeasured,
        "without_overrides": without_overrides,
        "removed_sections": run.removed,
        "total": sum(everywhere.values()),
        "codes": _ranked(everywhere),
        "cells": sorted(cells, key=lambda entry: (-entry["total"], entry["cell"])),
        "unattributed": _ranked(run.census.unattributed),
    }


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
        tiers_held: set[str] = set()

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
            else:
                tiers_held.add(str(cell["tier"]))
            if not isinstance(cell.get("ceiling"), int) or isinstance(cell.get("ceiling"), bool):
                faults.append(f"{where}: ceiling {cell.get('ceiling')!r} is not a whole number")
            if not str(cell.get("reason", "")).strip():
                faults.append(f"{where}: no reason given, which is what makes deferring honest")
            elif _MEASUREMENT_IN_PROSE.search(str(cell["reason"])):
                faults.append(
                    f"{where}: its reason states a number, and nothing compares that "
                    "number against anything. Say what kind of work the cell holds "
                    "and leave the counting to the ceiling, which is checked."
                )
            if cell.get("tier") in _UNMEASURED_TIERS and cell.get("ceiling"):
                faults.append(
                    f"{where}: tier {cell['tier']!r} is not measured, so its ceiling "
                    f"of {cell['ceiling']} claims a number nothing takes"
                )

        # A tier nothing holds. The other direction of the check above, and the
        # one that does not follow from it: every cell can name a declared tier
        # while a declared tier names no cell, which leaves a word in the
        # vocabulary that describes no part of the tree.
        #
        # It is vocabulary that a retreat needs. Un-covering a directory takes
        # two edits — drop it from the tool's target set, and re-file its cell
        # in a tier that tolerates a backlog — and neither half is a fault on
        # its own, because the pair agree with each other.
        #
        # What that costs differs by tool, and the milder case is ruff's.
        # measure_ruff tallies one pass over the whole population regardless of
        # tier, so `check` keeps measuring a retreated cell and only the *local*
        # toolchain goes: bin/validate.sh stops reading the directory and
        # bin/fix.sh stops offering a remedy, so a contributor's pre-push run
        # reports clean over territory the gate still checks. That is the shape
        # 2c shipped. mypy is worse — mypy_targets skips _UNMEASURED_TIERS
        # outright, so a cell retreating there stops being measured at all.
        #
        # A retreated mypy cell used to also report {"measured": 0, "ceiling": 0},
        # byte-identical to a cell that is genuinely clean, which made the retreat
        # invisible in the artifact as well as unopposed in the tree. That half is
        # closed: check() reports it as measured=None and lists it under
        # "unmeasured", so a retreat now shows up as a cell nobody reads. Only for
        # the tools in _TIER_GATED_TOOLS, which is mypy and is the point — ruff's
        # retreated cell is still measured, so a None there would be this module
        # inventing an absence rather than reporting one. What check cannot do is
        # object to it — the declaration is still the authority on what gets
        # measured, and a cell moved into an unmeasured tier is still a cell that
        # stopped being checked. Which is why the tier itself has to go.
        #
        # Re-filing is also the licensed first move toward a ceiling raise,
        # since a backlog tier is the one place a non-zero ceiling looks like
        # bookkeeping rather than a reversal.
        #
        # So the word goes back the day its last cell empties. Emptying a tier
        # is the work; striking it is that work's residue, and a pawl fitted
        # only after the last tooth has passed protects nothing on the way up.
        unheld = sorted(tiers - tiers_held)
        if unheld:
            faults.append(
                f"{tool}: declares {unheld} but no cell holds that tier, so it is a "
                "word available to a cell that wants out of being measured. Delete "
                "it — a tier is struck the day its last cell empties."
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

    The comparison has two failing signs, not one. ``exceeded`` is a cell above
    its ceiling; ``cleared`` is one below it, which the caller treats as a
    failure too — a ceiling left standing above what the tree measures is
    headroom a later regression can be absorbed into without any number moving.
    Both are reported here; which of them ends a run is ``main``'s to say, and
    it says so at the ``failed`` line.

    Two things are reported besides the comparison, and both are there because
    a number produced under conditions that have changed is not the number it
    claims to be:

    * ``unused_config`` — sections of the measuring tool's own configuration
      that matched nothing. Collected only on an unrestricted run, since a
      scoped one makes almost every section look unused.
    * ``findings`` — what the tool said, when a caller asks for it. A ceiling
      breach reported as a count leaves a developer with nothing to open.
    * ``unmeasured`` — cells whose tier puts them outside the tool's target set,
      so that a reader summing the report can find them without knowing which
      tiers mean what. Their ``measured`` is ``None``, not ``0``: the integer
      those cells used to carry was indistinguishable from a cell read and found
      clean, and it summed into any total a consumer built — silently, and low.
      Only a tool in ``_TIER_GATED_TOOLS`` can put a cell here, since only those
      measurers read the tier; for the rest a re-tiered cell is still measured
      and its zero is a count.
    """
    report: dict[str, Any] = {
        "exceeded": [],
        "cleared": [],
        "cells": {},
        "unattributed": [],
        "unused_config": [],
        "unmeasured": [],
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
            tier = cell.get("tier", "")
            measured = _measured(measurement, cell)

            # A zero from a cell nothing read is a silence, and a zero from a
            # cell that was read is a count. They are the same integer, so this is
            # where the difference has to be recorded — downstream is where it
            # stops being recoverable. _measured derives the total from per-file
            # counts (dddcb7ba, and that is the right shape), and a cell outside
            # the tool's target set has none of those for precisely the same
            # reason a clean cell has none.
            #
            # Both conjuncts, and the first is not decoration. Whether a tier
            # silences a cell is a property of the measurer rather than of the
            # tier: _TIER_GATED_TOOLS says which measurers read it, and for the
            # other two a re-tiered cell is still measured, so its zero is a
            # count. Reading the tier alone would answer for all three and be
            # right about one, and the wrong answer runs this module's own defect
            # backwards — a presence rendered as an absence, warning that a tier
            # put the cell outside a target set the tool does not have.
            #
            # Only a zero becomes None. A *count* arriving against such a cell is
            # the declaration losing an argument with the tool, and the tool is
            # the one holding evidence: mypy follows imports, so a finding in
            # packages/<pkg>/tests/ lands in the population and matches an
            # unchecked cell without mypy_targets ever having named it. Blanking
            # that would suppress a finding on the authority of the declaration it
            # just contradicted. Since verify pins these ceilings at zero, the
            # count is also a breach — so it falls through and is reported as one.
            if tool in _TIER_GATED_TOOLS and tier in _UNMEASURED_TIERS and not measured:
                report["cells"][name] = {
                    "measured": None,
                    "ceiling": cell["ceiling"],
                    "tier": tier,
                }
                report["unmeasured"].append({"cell": name, "tier": tier, "tool": tool})
                continue

            report["cells"][name] = {
                "measured": measured,
                "ceiling": cell["ceiling"],
                "tier": tier,
            }
            # ``tool`` and ``path`` beside the joined ``cell``, because the
            # remediation printed for a cleared cell names them as two separate
            # options. Splitting the joined name back apart at the printer would
            # re-derive from a string what the loop is holding, and a cell path
            # containing the separator would come apart at the wrong slash.
            entry = {
                "cell": name,
                "tool": tool,
                "path": cell["path"],
                "measured": measured,
                "ceiling": cell["ceiling"],
                "tier": tier,
            }
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
    contract: dict[str, Any], tools: list[str], path: Path, only: set[str] | None = None
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

    ``only`` narrows which ceilings are rewritten, and it is honoured rather
    than accepted. The command line validated a named cell and then passed none
    of it here, so ``update-baseline --tool mypy --cell packages/data/src``
    checked that the name existed and rewrote all fourteen — the widest possible
    edit to the declaration, reached by the command that asks for the narrowest,
    and unrecoverable by re-running because the ceilings only fall.
    """
    lowered: list[str] = []
    exceeded: list[str] = []

    for tool, measurement in _measure(contract, tools, only).items():
        for cell in _restrict(contract["tools"][tool]["cells"], only):
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
        # ensure_ascii=False, because the file it is rewriting was not written
        # with the default. Reasons are prose and contain em-dashes; escaping
        # them on the way out edits rows whose ceiling did not move, so a
        # ratchet of one cell arrives as a diff touching several and the one
        # line that changed meaning is the hardest one to find.
        path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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


def _tracked_under(given: str, files: list[PurePosixPath]) -> list[str]:
    """The tracked Python at or below a caller-named path.

    Against the population git tracks rather than the filesystem, for the reason
    ``tracked_python`` gives and one of its own: a misspelt path exists nowhere,
    matches nothing here, and is refused — where a filesystem walk would have
    found nothing either and reported it as a debt of zero, which is exactly
    what a file with nothing left to fix reports.
    """
    name = _relative(given)
    return [str(path) for path in files if str(path) == name or str(path).startswith(f"{name}/")]


def charge(contract: dict[str, Any], tool: str, paths: list[str]) -> dict[str, Any]:
    """What each caller-named path owes: its share of the count its cell carries.

    The gap this closes sits between two questions nothing else in the CLI can
    answer. ``census`` decomposes a cell by *rule*; ``check --show-findings``
    echoes a whole cell's output; ``bin/validate.sh`` handed a single filename
    deliberately measures the whole cell that file is in, because a ceiling is a
    whole-cell property and a partial count compared against one is not a
    verdict. All three are right, and none of them says:

        *"What does this file owe?"* — which a per-file convention has to ask
        before it can decide whether a file is worth clearing now.
        *"Have I paid it?"* — which a cell verdict cannot say, since the cell
        total moves for reasons that have nothing to do with this file.

    So the cell is still measured **whole**, exactly as ``check`` measures it,
    through the same ``_measure`` those two commands go through; only the
    *display* is filtered to the named paths. The number reported here is
    therefore a term of the sum the ceiling is compared against, and not a
    second measurement that could quietly disagree with the first.

    Every way of naming a path that cannot produce that number is refused rather
    than answered with a zero, and the three refusals are three different lies a
    zero would tell: a path in no cell owes nothing to any ceiling, a path in an
    unread tier was never measured at all, and a path naming no tracked file was
    almost certainly mistyped.
    """
    resolved = scope_paths(contract, tool, paths)

    outside = sorted(path for kind, path, _cell in resolved if kind == SCOPE_OUTSIDE)
    if outside:
        raise SystemExit(
            f"{outside} resolve to no single cell of {tool}'s — either outside "
            "every one, or naming a directory that spans several — so there is "
            "no ceiling their findings count toward and a total here would be a "
            f"sum over a scope nobody asked for. `{_INVOCATION} scope --tool "
            f"{tool} <path>` says which cell a path is in."
        )

    unread = sorted(path for kind, path, _cell in resolved if kind == SCOPE_UNMEASURED)
    if unread:
        raise SystemExit(
            f"{unread} are in cells the contract puts in a tier {tool} is not "
            "pointed at, so nothing measured them. A zero here would be a "
            "silence rendered as a debt already paid, which is the one thing "
            "this command must not be able to say."
        )

    files = tracked_python()
    holdings = {given: _tracked_under(given, files) for _kind, given, _cell in resolved}
    empty = sorted(given for given, held in holdings.items() if not held)
    if empty:
        raise SystemExit(
            f"{empty} name no tracked *.py. A path that is merely misspelt would "
            "otherwise be answered with a zero, which is what a file with "
            "nothing left to fix is also answered with."
        )

    only = {cell for _kind, _given, cell in resolved}
    measurement = _measure(contract, [tool], only)[tool]

    # Merged across the named cells only. Every file under a named path is in
    # exactly one of them, so nothing is double counted, and a finding in some
    # other cell that this run reached by following imports stays out of a total
    # the caller did not ask about.
    counted: Counter[str] = Counter()
    for cell in only:
        counted += measurement.by_cell.get(cell, Counter())

    # Keyed by the same normalised path the counts are, because that is the
    # whole reason ``mypy_lines`` normalises rather than each reader doing it.
    #
    # Gated on the tool rather than on whether ``output`` is empty. Which
    # measurers fill that field is decided in ``Measurement``, three hundred
    # lines away, and only mypy does today — so an emptiness test reads as a
    # test of *this* tool's output while actually testing that nobody upstream
    # has changed their mind. The day one of the others starts filling it, this
    # would parse a different tool's text with the type checker's regex and
    # report whatever survived.
    quotable = tool == "mypy"
    messages: dict[str, list[str]] = defaultdict(list)
    if quotable:
        for path, _body, line in mypy_lines(measurement.output):
            messages[path].append(line)

    declared = {cell["path"]: cell for cell in contract["tools"][tool]["cells"]}
    charged = []
    for _kind, given, cell in resolved:
        held = {name: counted[name] for name in holdings[given] if counted[name]}
        charged.append(
            {
                "path": given,
                "cell": cell,
                "total": sum(held.values()),
                "tracked": len(holdings[given]),
                "cell_total": sum(measurement.by_cell.get(cell, Counter()).values()),
                "cell_ceiling": declared[cell]["ceiling"],
                "files": [
                    {
                        "file": name,
                        "count": count,
                        "findings": messages.get(name, []),
                    }
                    for name, count in sorted(held.items(), key=lambda i: (-i[1], i[0]))
                ],
            }
        )

    return {
        "tool": tool,
        # Whether this tool's findings carry message text at all — the same
        # answer the parse above was gated on, so the report cannot claim prose
        # the loop was never allowed to collect. ruff and the formatter are read
        # from JSON that *is* the tally, so an empty ``findings`` list under them
        # means "this tool reports no prose", not "this file has no findings",
        # and those two must not render alike.
        "messages": quotable,
        "total": sum(entry["total"] for entry in charged),
        "paths": charged,
    }


def _write_charge(report: dict[str, Any]) -> None:
    """The charge as a table: one block per path named, its files under it."""
    out = sys.stdout
    named = len(report["paths"])
    out.write(
        f"{report['tool']} — {report['total']} finding(s) charged to "
        f"{named} path{'' if named == 1 else 's'}\n"
    )

    for entry in report["paths"]:
        out.write(
            f"\n{entry['path']}: {entry['total']} of {entry['cell_total']} in "
            f"{entry['cell']} (ceiling {entry['cell_ceiling']}), "
            f"over {entry['tracked']} tracked file(s)\n"
        )
        if not entry["total"]:
            # Said rather than left blank. "Nothing to show" and "nothing was
            # measured" print identically as an absent block, and this command
            # exists to be trusted on exactly that distinction.
            out.write("  nothing outstanding — this path is clean under this tool\n")
            continue
        for held in entry["files"]:
            out.write(f"  {held['file']}: {held['count']}\n")
            for line in held["findings"]:
                out.write(f"      {line}\n")
            # Unreachable while the count and the messages come from one parse
            # of one run, which they do: both are read from ``mypy_lines`` over
            # the same ``output``. Kept as the statement of that invariant —
            # were it ever broken, the alternative is a count printed with
            # nothing under it, which reads as a file whose findings are simply
            # not worth showing.
            if not held["findings"] and report["messages"]:
                out.write("      (reported against this file, but no message line named it)\n")

    if report["total"] and not report["messages"]:
        out.write(
            f"\n{report['tool']} is read from JSON that is already the tally, so "
            "there is no message text to quote — the counts above are the whole "
            "of what it said.\n"
        )


#: The declaration's path as git spells it. ``_CONTRACT`` is the copy on disk;
#: the ledger reads this one at many revisions and never touches the tree.
_CONTRACT_IN_GIT = ".dataknobs/quality-contract.json"

#: The trailer a commit carries when clearing backlog was its *purpose* rather
#: than something it did on the way past. Defined in
#: ``.claude/rules/touched-file-cleanup.md``, where its value is specified as a
#: cell path — which is the only reason ``_leg_faults`` below can exist. A
#: planning identifier could be checked against nothing.
_LEG_TRAILER = "Quality-Leg"


class Movement(NamedTuple):
    """What one revision did to a tool's ceilings, split four ways.

    The split is the entire correctness argument for this command, because the
    obvious implementation — sum every ceiling, subtract the previous sum — is
    wrong in a way that flatters the result and leaves no trace.

    A cell can be **added** or **removed** between two revisions, and a glob
    cell can be replaced by the several cells it used to cover. When that last
    thing happened here, one revision's ceilings fell by **255** by sum while
    the cells present in both revisions moved by **13**: a glob covering the
    per-package test trees was split into one cell per package, and the naive
    reading credits the split with clearing 242 findings nobody cleared.

    So ``cleared`` and ``raised`` are computed only over cells present in
    *both* revisions, and the two structural quantities are reported beside
    them rather than folded in. A ledger that cannot tell "somebody fixed 242
    findings" from "somebody re-drew the cells" is not measuring the thing it
    was built to measure.
    """

    cleared: int
    raised: int
    added: int
    removed: int

    @property
    def structural(self) -> bool:
        """Whether the cell set itself moved, which is not progress either way."""
        return bool(self.added or self.removed)


def _ceilings_at(sha: str, tool: str, repo: Path = _ROOT) -> dict[str, int] | None:
    """One tool's ceilings, by cell, as the declaration stood at ``sha``.

    ``None`` when the declaration did not exist there, which is emphatically
    not an empty contract: before it was created there is nothing for a later
    revision to be compared against, and reading that as "every cell at zero"
    would report the file's arrival as the largest regression in the history.
    """
    shown = _run(["git", "show", f"{sha}:{_CONTRACT_IN_GIT}"], cwd=repo)
    if shown.returncode != 0:
        return None
    declared = json.loads(shown.stdout)["tools"].get(tool, {}).get("cells", [])
    return {cell["path"]: cell["ceiling"] for cell in declared}


def _declared_at(sha: str, repo: Path = _ROOT) -> set[str]:
    """Every cell path the declaration names at ``sha``, across all tools.

    Across *all* tools deliberately. A leg draining a lint cell is still a leg
    when the ledger is denominated in type findings, so validating its trailer
    against one tool's cells would manufacture a fault out of a correct commit.
    """
    shown = _run(["git", "show", f"{sha}:{_CONTRACT_IN_GIT}"], cwd=repo)
    if shown.returncode != 0:
        return set()
    contract = json.loads(shown.stdout)
    return {cell["path"] for spec in contract["tools"].values() for cell in spec.get("cells", [])}


def _leg_faults(short: str, date: str, legs: list[str], declared: set[str]) -> list[str]:
    """Trailers on this step that name something the declaration does not.

    What this does *not* catch is worth stating first, because it is the thing
    the check was originally proposed to catch and it cannot happen: presence is
    the discriminator, so a commit carrying a misspelled trailer is still
    counted as a leg. The population split survives a typo.

    What a typo destroys is the *attribution*. The value is the only record of
    which cell a leg was aimed at, and one naming no cell records nothing while
    reading exactly like a record — so the leg column stays right while the
    question "which cell did the scheduled work go to?" quietly has no answer.

    Nothing else can catch it. The value is prose in a commit message: it is
    never parsed when written, the commit is merged before the ledger is next
    read, and a total that is still correct is not going to prompt anyone to
    look.
    """
    return [
        f"{short} ({date}) carries {_LEG_TRAILER}: {named!r}, which the "
        f"declaration at that revision does not name as a cell"
        for named in legs
        if named not in declared
    ]


def _movement(before: dict[str, int], after: dict[str, int]) -> Movement:
    """Compare two revisions of one tool's ceilings, per cell."""
    shared = before.keys() & after.keys()
    return Movement(
        cleared=sum(before[c] - after[c] for c in shared if after[c] < before[c]),
        raised=sum(after[c] - before[c] for c in shared if after[c] > before[c]),
        added=sum(after[c] for c in after.keys() - before.keys()),
        removed=sum(before[c] for c in before.keys() - after.keys()),
    )


def _legs_in(first_parent: str, sha: str, repo: Path = _ROOT) -> list[str]:
    """The cells named by ``Quality-Leg:`` trailers this step brought in.

    Over the *range*, not the commit. A merge commit carries the branch's
    subject and none of its trailers, and the commit that did the clearing is
    inside the branch — so reading trailers off the step itself would find
    nothing on every pull request, which is every leg there will ever be.

    Git parses the trailer block, so the awkward parts of the format — folded
    continuation lines, what counts as the last paragraph — are not reproduced
    here.
    """
    listed = _run(
        [
            "git",
            "log",
            f"--format=%(trailers:key={_LEG_TRAILER},valueonly)",
            f"{first_parent}..{sha}",
        ],
        cwd=repo,
    )
    return [line.strip() for line in listed.stdout.splitlines() if line.strip()]


#: A window boundary written as a calendar day. The only date form accepted;
#: everything else is resolved as a revision. See ``_opens_after``.
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def _birth(revision: str, repo: Path = _ROOT) -> str | None:
    """The first-parent commit where the declaration first appears."""
    found = _run(
        [
            "git",
            "log",
            "--first-parent",
            "--reverse",
            "--format=%H",
            revision,
            "--",
            _CONTRACT_IN_GIT,
        ],
        cwd=repo,
    )
    listed = found.stdout.split()
    return listed[0] if listed else None


def _stamped(sha: str, repo: Path = _ROOT) -> tuple[str, str]:
    """One commit's short name and day, for a report that has to cite it."""
    shown = _run(["git", "log", "-1", "--format=%h%x00%ad", "--date=short", sha], cwd=repo)
    short, _, date = shown.stdout.strip().partition("\0")
    return short, date


def _opens_after(since: str, revision: str, repo: Path = _ROOT) -> str | None:
    """Resolve a requested window to the commit it opens *after*.

    Two forms are accepted, both absolute, and excluding the third is the point
    of doing this here rather than handing the string to git's ``--since``.

    A ``YYYY-MM-DD`` day is pinned to midnight before git sees it, because git
    fills the fields an approxidate leaves out from **now**. Measured in this
    repository, at 17:20 on the day in question: ``--since=2026-08-22`` reported
    **0** merges and ``--since=2026-08-22T00:00:00`` reported **4**. A window
    that quietly shrinks by the hour you read it at is the precise failure this
    command exists to remove — and it fails silently, with a plausible number.

    Anything else is a git revision, which is the better form and the one a
    fixed window should use: it names the boundary event instead of
    approximating it, and a typo fails here instead of moving the window. Once
    resolved there are no dates left in the arithmetic, only commits.

    One revision form is refused with the dates: ``HEAD@{4.weeks.ago}`` and its
    relatives resolve against this machine's reflog, which is neither shared nor
    stable, so two readings of the *same* window disagree by where they were
    taken. It is the form somebody redirected away from a relative date reaches
    for next, which is exactly why it is named rather than left to chance.
    """
    if _ISO_DAY.match(since):
        found = _run(
            ["git", "rev-list", "--first-parent", "-1", f"--before={since} 00:00:00", revision],
            cwd=repo,
        )
        return found.stdout.strip() or None
    if "@{" in since:
        raise ValueError(
            f"--since {since!r} is a reflog expression, which resolves against this "
            "machine's reflog and against the clock; a window read on two machines "
            "would not be the same window. Name the boundary commit instead"
        )
    resolved = _run(["git", "rev-parse", "--verify", "--quiet", f"{since}^{{commit}}"], cwd=repo)
    if resolved.returncode != 0:
        raise ValueError(
            f"--since {since!r} is neither a YYYY-MM-DD day nor a revision in this "
            "repository. Relative expressions such as '4 weeks ago' are refused "
            "rather than passed through: git resolves them against the clock, so "
            "the same command answers a different question every time it is run"
        )
    return resolved.stdout.strip()


def _leg_merges(span: str, chain: list[str], repo: Path = _ROOT) -> int:
    """How many first-parent steps in ``span`` carry a leg trailer at all.

    A different population from the leg steps the walk below finds, and the
    difference is why this exists. Those are steps that carried a trailer *and*
    moved a ceiling. This is every step that carried one — which is the
    denominator the convention's cost has to be measured against, because a
    merge whose *purpose* was cleanup is not evidence about what ordinary work
    costs, whether or not the cleanup ended up moving anything. A leg that
    cleared nothing never touches the declaration, so no walk of the
    declaration's history can see it.

    Attribution costs a subprocess per step, so it runs only when one bulk walk
    says there is something to attribute. For a window with no leg in it — which
    is every window ending before the trailer was adopted — this answers 0 for
    the price of a single git invocation.
    """
    carried = _run(
        ["git", "log", f"--format=%(trailers:key={_LEG_TRAILER},valueonly)", span],
        cwd=repo,
    )
    if not carried.stdout.strip():
        return 0
    return sum(1 for sha in chain if _legs_in(f"{sha}^", sha, repo))


def ledger(
    tool: str,
    revision: str = "HEAD",
    since: str | None = None,
    repo: Path = _ROOT,
) -> dict[str, Any]:
    """What the declaration's own history records, per merge, for one tool.

    The contract is committed, so ``git log`` over it *is* the time series and
    no separate artifact is needed — one that changed on every run would
    conflict on every run, and a snapshot cannot answer a question about a
    series anyway.

    **The unit is a first-parent step, not a commit**, and the difference is
    not cosmetic. Measured over this repository's history on the day this was
    written: 21 paying events out of 66 read per commit, against 11 out of 67
    read per step, because a pull request that moves a ceiling twice is one
    pull request. The per-commit figure double-counts exactly the population
    whose *rate* the number is meant to describe, and it does so in the
    flattering direction.

    Only steps where the declaration changed are read. Between them it is
    constant by construction, so the arithmetic is identical either way while
    the walk pays its per-step subprocesses over the merges that moved
    something rather than over all of them — a small minority, and one that
    shrinks as a share of the history with every merge that moves nothing.

    ``since`` narrows the window to what happened after a boundary commit, which
    a fixed-length reading needs and hand-rolled arithmetic over the full series
    should not be asked to supply. The opening balance is then the declaration
    as it stood *at* that boundary — exact, not interpolated, because between
    two moving steps it does not move. A boundary older than the declaration is
    clamped to its first appearance and the report says so.
    """
    boundary = _opens_after(since, revision, repo) if since else _birth(revision, repo)
    opening = _ceilings_at(boundary, tool, repo) if boundary else None
    clamped = False
    if opening is None:
        # Either nothing was asked for that the declaration reaches, or the
        # boundary predates it. There is no state to compare against before the
        # declaration's first appearance — reading its arrival as a cell set
        # springing out of zero would report it as the largest regression in the
        # history — so the window opens there instead, and says which it did.
        birth = _birth(revision, repo)
        clamped = since is not None and birth is not None and birth != boundary
        boundary = birth
        opening = _ceilings_at(boundary, tool, repo) if boundary else None

    populations: dict[str, dict[str, int]] = {
        "leg": {"merges": 0, "paying": 0, "cleared": 0},
        "convention": {"merges": 0, "paying": 0, "cleared": 0},
    }
    report: dict[str, Any] = {
        "tool": tool,
        "since": since,
        "clamped": clamped,
        "opened": None,
        "standing": sum((_ceilings_at(revision, tool, repo) or {}).values()),
        "window": 0,
        "paying": 0,
        "cleared": 0,
        "raised": 0,
        "populations": populations,
        "months": {},
        "faults": [],
        "steps": [],
    }
    if boundary is None or opening is None:
        return report

    short, date = _stamped(boundary, repo)
    report["opened"] = {"sha": short, "date": date, "total": sum(opening.values())}
    span = f"{boundary}..{revision}"

    moved = _run(
        [
            "git",
            "log",
            "--first-parent",
            "--format=%H%x00%h%x00%ad%x00%s",
            "--date=short",
            "--reverse",
            span,
            "--",
            _CONTRACT_IN_GIT,
        ],
        cwd=repo,
    )
    steps: list[dict[str, Any]] = []
    faults: list[str] = []
    months: Counter[str] = Counter()

    for line in moved.stdout.splitlines():
        if not line:
            continue
        sha, short, date, subject = line.split("\0")
        after = _ceilings_at(sha, tool, repo)
        before = _ceilings_at(f"{sha}^", tool, repo)
        if after is None or before is None:
            # The declaration arriving or leaving. Neither end has a state to be
            # compared against, and the window's opening balance was read above
            # from the boundary rather than from whichever step this is.
            continue

        movement = _movement(before, after)
        legs = _legs_in(f"{sha}^", sha, repo)
        if legs:
            # Guarded because reading the declaration back is a subprocess, and
            # a step with no trailer has nothing for it to validate.
            faults += _leg_faults(short, date, legs, _declared_at(sha, repo))

        if movement.cleared or movement.raised or movement.structural:
            steps.append(
                {
                    "sha": short,
                    "date": date,
                    "subject": subject,
                    "cleared": movement.cleared,
                    "raised": movement.raised,
                    "added": movement.added,
                    "removed": movement.removed,
                    "legs": legs,
                }
            )
        if movement.cleared:
            months[date[:7]] += movement.cleared
            side = "leg" if legs else "convention"
            populations[side]["paying"] += 1
            populations[side]["cleared"] += movement.cleared

    chain = _run(["git", "rev-list", "--first-parent", span], cwd=repo).stdout.split()
    populations["leg"]["merges"] = _leg_merges(span, chain, repo)
    populations["convention"]["merges"] = max(len(chain) - populations["leg"]["merges"], 0)

    report["window"] = len(chain)
    report["paying"] = sum(side["paying"] for side in populations.values())
    report["cleared"] = sum(side["cleared"] for side in populations.values())
    report["raised"] = sum(step["raised"] for step in steps)
    report["months"] = dict(sorted(months.items()))
    report["faults"] = faults
    report["steps"] = steps
    return report


def _write_ledger(report: dict[str, Any]) -> None:
    """The ledger as a reading, with the two ambiguous quantities kept apart."""
    out = sys.stdout
    opened = report["opened"]
    if opened is None:
        out.write(f"{report['tool']}: the declaration has no history on this revision\n")
        return

    window = report["window"]
    out.write(
        f"{report['tool']} ledger — {opened['total']} at {opened['sha']} "
        f"({opened['date']}) to {report['standing']} now, over {window} merge(s)\n"
    )
    if report["since"]:
        # The resolution, not only the request. A boundary that moved without
        # saying so is the failure this flag was added to remove, so the commit
        # the window actually opened after is printed beside what was asked for.
        note = f"  --since {report['since']} opens the window after {opened['sha']}"
        if report["clamped"]:
            note += ", clamped: it is older than the declaration itself"
        out.write(note + "\n")

    paying = report["paying"]
    # Both halves, always. A rate is a fraction and a fraction printed without
    # its denominator is the shape of number that gets quoted back with a
    # different one attached.
    if window:
        out.write(
            f"\ncleared {report['cleared']} over {paying} of {window} merges "
            f"({100 * paying / window:.0f}% paying, "
            f"{100 * (window - paying) / window:.0f}% paying nothing), "
            f"mean {report['cleared'] / window:.1f} per merge\n"
        )

    # Never summed with the drain. A raised ceiling is the safety criterion
    # failing, not progress with a sign on it.
    out.write(f"raised {report['raised']}")
    out.write("\n" if report["raised"] else "  — no cell has ever ended higher than it started\n")

    # The second row is the one the convention is judged on, so it carries its
    # own rate rather than leaving it to be divided out of the total above. A
    # drain achieved entirely by scheduled cleanup would satisfy a threshold
    # read off that total while falsifying everything the total was quoted for.
    out.write("\nby population — the convention is the second row, and only the second row\n")
    for side in ("leg", "convention"):
        counts = report["populations"][side]
        line = (
            f"  {side:<12} {counts['cleared']:>6} over {counts['paying']} "
            f"of {counts['merges']} merge(s)"
        )
        if counts["merges"]:
            idle = counts["merges"] - counts["paying"]
            line += (
                f", mean {counts['cleared'] / counts['merges']:.1f} per merge, "
                f"{100 * idle / counts['merges']:.0f}% paying nothing"
            )
        out.write(line + "\n")
    if not report["populations"]["leg"]["merges"]:
        # Said, not left as a zero row. Before the trailer was adopted no commit
        # could carry one, so an empty leg column is indistinguishable from a
        # leg column that nothing has been filed into yet — and the whole reason
        # the trailer exists is that those two must not read alike.
        out.write(
            f"  no commit in this window carries a {_LEG_TRAILER}: trailer, so "
            "everything above is attributed to incidental clearing\n"
        )

    if report["months"]:
        out.write("\nper month\n")
        for month, count in report["months"].items():
            out.write(f"  {month}  {count}\n")

    structural = [step for step in report["steps"] if step["added"] or step["removed"]]
    if structural:
        out.write("\nstructural — the cell set moved, which is neither clearing nor regression\n")
        for step in structural:
            out.write(
                f"  {step['sha']} {step['date']}  +{step['added']} added, "
                f"-{step['removed']} removed  {step['subject'][:52]}\n"
            )

    for fault in report["faults"]:
        logger.error("%s", fault)


def _selected(requested: str | None) -> list[str]:
    return [requested] if requested else sorted(MEASURERS)


def _write_census(report: dict[str, Any]) -> None:
    """The census as a table, headed by the conditions it was taken under.

    The heading is not decoration. A census is taken once and then quoted for
    months, and every number in it depends on which configuration produced it
    and which cells the run reached. A table carrying its totals without those
    is a figure that goes stale without ever saying so — which this repository
    has already done once, in a documentation page that recorded counts nobody
    could date.
    """
    out = sys.stdout
    out.write(f"{report['tool']} census — {report['total']} finding(s) under {report['config']}\n")
    if report["include_unmeasured"]:
        out.write("  including the cells the contract declares no tool reads\n")
    if report["removed_sections"]:
        out.write("  with strictness relaxed for nothing first-party; removed:\n")
        for pattern in report["removed_sections"]:
            out.write(f"      {pattern}\n")

    out.write("\nper cell\n")
    for cell in report["cells"]:
        out.write(
            f"  {cell['cell']}: {cell['total']}  ({cell['tier']}, ceiling {cell['ceiling']})\n"
        )
        for code, count in cell["codes"].items():
            out.write(f"      {code}: {count}\n")

    # "the cells above", not "every cell read". The type checker follows imports,
    # so a scoped run reads well past the cells it was pointed at — those
    # findings are attributed to the cell they are in and left out of this total,
    # which is the right scoping and the wrong heading for it.
    out.write("\nper rule, across the cells above\n")
    for code, count in report["codes"].items():
        out.write(f"  {code}: {count}\n")

    if report["unattributed"]:
        out.write("\nreported against no cell, so counted toward no ceiling\n")
        for code, count in report["unattributed"].items():
            out.write(f"  {code}: {count}\n")


def _add_tool(command: argparse.ArgumentParser, help_text: str) -> None:
    command.add_argument("--tool", choices=sorted(MEASURERS), help=help_text)


def _add_cells(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="PATH",
        help="restrict to these cells, named exactly as the contract declares them",
    )


def _add_json(command: argparse.ArgumentParser) -> None:
    command.add_argument("--json", action="store_true", dest="use_json")


def _add_contract(command: argparse.ArgumentParser) -> None:
    """Let a command be pointed at a declaration other than this repository's.

    Which is not a convenience: it is what keeps these commands testable now
    that the tree they measure has no ruff backlog left in it. A guard over
    ``census`` has to watch it report findings and still exit zero, and one over
    the declaration has to watch a command decline to rewrite it — properties of
    the *command*, since the exit status is the only thing carrying a refusal
    out, and neither drivable over a tree that measures nothing. So the guards
    point this at ``quality-fixture/``, whose findings are deliberate and whose
    ceilings are its own, and what they exercise is the command that ships
    rather than an in-process approximation of it.

    One command writes a declaration back, so the path is remembered in ``main``
    rather than re-derived there. Reading through an argument and writing
    through the constant is the one shape of this option worse than not having
    it: ``update-baseline --contract`` would report lowering a ceiling in the
    file it was given and lower it in this repository's instead, unrecoverably,
    because a ceiling only ever falls.

    Declared per command like every other option here, and on the six that read
    a declaration rather than globally — ``explain`` answers a question about
    the linter's configuration, and the contract it happens to consult for the
    inline-waiver tally is this repository's by definition.
    """
    command.add_argument(
        "--contract",
        metavar="PATH",
        help="measure against this declaration instead of the repository's",
    )


def _build_parser() -> argparse.ArgumentParser:
    """One subparser per command, each declaring only the options it reads.

    Every option used to be global, which argparse accepts and no command was
    obliged to read. ``check --without-overrides`` ran an ordinary check under
    the declared configuration and reported nothing about the flag it discarded;
    ``update-baseline --cell packages/data/src`` validated the name and rewrote
    every ceiling the tool has. That is the defect ``take_census`` refuses per
    tool, one layer up and with nothing refusing it.

    Subparsers close it structurally rather than through a table somebody has to
    keep in step with the parser. A command that does not declare an option
    rejects it as a usage error, and a new option cannot be added without
    choosing which commands honour it — there is nowhere left to put one that
    means "everywhere".
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # Every namespace carries these whether or not the chosen command declares
    # them, so the resolution below reads one shape. Declaring the *option* is
    # still per command — a command that does not is the usage error; this only
    # decides what its namespace looks like once that has passed.
    parser.set_defaults(tool=None, cell=[], use_json=False, contract=None)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    measure = commands.add_parser("check", help="measure the tree and compare every cell")
    _add_tool(measure, "restrict to one tool")
    _add_cells(measure)
    _add_contract(measure)
    measure.add_argument(
        "--show-findings",
        action="store_true",
        help="echo what the tool reported, not only the comparison",
    )
    _add_json(measure)

    faults = commands.add_parser("verify", help="check the declaration itself, without measuring")
    _add_contract(faults)
    _add_json(faults)

    baseline = commands.add_parser(
        "update-baseline", help="lower every ceiling to what the tree measures"
    )
    _add_tool(baseline, "restrict to one tool")
    _add_cells(baseline)
    _add_contract(baseline)

    split = commands.add_parser("partition", help="which cell each tracked file lands in")
    _add_tool(split, "restrict to one tool")
    _add_contract(split)
    _add_json(split)

    where = commands.add_parser("scope", help="classify caller-named paths against the cells")
    _add_tool(where, "the tool whose cells the paths are classified against")
    _add_contract(where)
    where.add_argument("paths", nargs="*", help="the paths to classify")
    _add_json(where)

    owed = commands.add_parser("charge", help="what the named paths owe their cells' ceilings")
    _add_tool(owed, "the tool to read; a charge is denominated in one tool's findings")
    _add_contract(owed)
    owed.add_argument("paths", nargs="+", help="the files or directories to ask about")
    _add_json(owed)

    history = commands.add_parser(
        "ledger", help="what the declaration's history records, per merge"
    )
    _add_tool(history, "the tool to read; a ledger is denominated in one tool's findings")
    history.add_argument(
        "--revision",
        default="HEAD",
        help="read the first-parent history ending here (default: HEAD)",
    )
    history.add_argument(
        "--since",
        help=(
            "open the window after this boundary: a revision, or a YYYY-MM-DD day. "
            "Relative expressions are refused — git resolves them against the clock"
        ),
    )
    history.add_argument(
        "--repo",
        type=Path,
        default=_ROOT,
        help="read this repository's history instead of the one this script lives in",
    )
    _add_json(history)

    counted = commands.add_parser("census", help="break a cell's findings down by rule")
    _add_tool(counted, "the tool to read; a census reads one at a time")
    _add_cells(counted)
    _add_contract(counted)
    counted.add_argument(
        "--include-unmeasured",
        action="store_true",
        help="also read the cells whose tier the contract says no tool reads",
    )
    counted.add_argument(
        "--without-overrides",
        action="store_true",
        help=(
            "measure with the type checker's first-party strictness relaxations "
            "removed, which is the configuration a zeroed backlog would have to "
            "hold under"
        ),
    )
    _add_json(counted)

    why = commands.add_parser(
        "explain", help="is this rule reported in this file, and if not, why not"
    )
    why.add_argument("code", nargs="?", help="a ruff rule code, e.g. RUF012")
    why.add_argument("path", nargs="?", help="the file to ask about; omit to ask repo-wide")
    why.add_argument(
        "--audit",
        action="store_true",
        help="every declined rule by category, instead of one lookup",
    )
    why.add_argument(
        "--measure",
        action="store_true",
        help="with --audit, also count the findings each decline stands in front of",
    )
    _add_json(why)

    return parser


def _scoped_cells(
    parser: argparse.ArgumentParser, contract: dict[str, Any], args: argparse.Namespace
) -> set[str] | None:
    """The cells ``--cell`` named, resolved against the declaration rather than trusted.

    A caller that misspells one would otherwise restrict the run to nothing and
    be told every cell is within its ceiling.
    """
    if not args.cell:
        return None
    if not args.tool:
        parser.error("--cell names cells of one tool, so --tool is required with it")
    declared = {cell["path"] for cell in contract["tools"][args.tool]["cells"]}
    unknown = sorted(set(args.cell) - declared)
    if unknown:
        parser.error(
            f"{args.tool} declares no cell named {unknown} — known cells are {sorted(declared)}"
        )
    return set(args.cell)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Read once and remembered, because one command writes it back. A path that
    # is an argument on the way in and a constant on the way out is a command
    # that measures one declaration and rewrites another.
    contract_path = Path(args.contract) if args.contract else _CONTRACT
    contract = load_contract(contract_path)

    if args.command == "explain":
        if args.audit:
            audit = decline_audit()
            measured = (
                measure_declines([e.code for rows in audit["by_category"].values() for e in rows])
                if args.measure
                else {}
            )
            if args.use_json:
                json.dump(
                    {
                        "total": audit["total"],
                        "by_category": {
                            name: [
                                {
                                    "code": e.code,
                                    "reason": e.reason,
                                    **({"findings": measured[e.code]} if measured else {}),
                                }
                                for e in rows
                            ]
                            for name, rows in audit["by_category"].items()
                        },
                    },
                    sys.stdout,
                    indent=2,
                )
                sys.stdout.write("\n")
            else:
                for name, rows in audit["by_category"].items():
                    exposure = sum(measured[e.code] for e in rows) if measured else None
                    tail = f", {exposure} findings" if exposure is not None else ""
                    sys.stdout.write(f"\n{name}: {len(rows)} rules{tail}\n")
                    for entry in rows:
                        count = f"{measured[entry.code]:>7}  " if measured else ""
                        sys.stdout.write(f"  {count}{entry.code:<10} {entry.reason}\n")
                for tool, label in (("ruff", "# noqa"), ("mypy", "# type: ignore")):
                    counts = inline_waivers(contract, tool)
                    total = sum(row.suppressions for row in counts)
                    bare = sum(row.bare for row in counts)
                    sys.stdout.write(
                        f"\ninline {label}: {total} directives, {bare} without a code\n"
                    )
                    for row in sorted(counts, key=lambda r: -r.suppressions):
                        sys.stdout.write(f"  {row.suppressions:>7}  {row.cell:<28} {row.tier}\n")
            sys.exit(0)
        if not args.code:
            parser.error("explain takes a rule code, or --audit for the whole table")
        explanation = explain_code(args.code, args.path)
        if args.use_json:
            json.dump(explanation._asdict(), sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            explain_report(explanation, sys.stdout)
        sys.exit(0)

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

    if args.command == "ledger":
        if not args.tool:
            parser.error("a ledger is denominated in one tool's findings, so --tool is required")
        # Before the measuring gate below, and before `_scoped_cells`: this
        # command reads the declaration at past revisions and never looks at the
        # tree, so the current file being unusable is not a reason to refuse —
        # it is a reason somebody would want the history.
        try:
            recorded = ledger(args.tool, args.revision, args.since, args.repo)
        except ValueError as unresolved:
            # An unresolvable boundary is a bad argument, not a finding: exit 2
            # with the reason rather than reporting a window nobody asked for.
            parser.error(str(unresolved))
        if args.use_json:
            json.dump(recorded, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            _write_ledger(recorded)
        # Exits zero on a fault. A trailer naming an undeclared cell is a defect
        # in a commit that is already merged, so there is nothing for a non-zero
        # status to block; it is reported to be fixed in the ledger's reader,
        # not to fail a gate over history that cannot be edited.
        sys.exit(0)

    only = _scoped_cells(parser, contract, args)

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

    # Every remaining command measures, and measuring a malformed contract
    # reports cells that do not describe the tree. Fail on the cheap check first.
    faults = verify(contract)
    if faults:
        for fault in faults:
            logger.error("%s", fault)
        logger.error("The contract is not usable, so nothing was measured.")
        sys.exit(2)

    if args.command == "charge":
        if not args.tool:
            parser.error("a charge is denominated in one tool's findings, so --tool is required")
        owed = charge(contract, args.tool, args.paths)
        if args.use_json:
            json.dump(owed, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            _write_charge(owed)
        # A charge reports what is owed; the ceiling is what judges. Exiting
        # non-zero on a file that has findings would make the command a
        # developer runs *before* doing the work indistinguishable from the
        # failure they ran it to avoid, and a caller learns to ignore a status
        # that means both. Same reason the census exits zero over a tree with a
        # backlog in it.
        sys.exit(0)

    if args.command == "census":
        if not args.tool:
            parser.error("a census reads one tool at a time, so --tool is required")
        # Which tools can be censused is `take_census`'s to say, not this
        # branch's. Refused here, the guard covered the command line and left
        # the layer that dispatches on the tool routing `format` into the type
        # checker's measurer and filing its findings under formatting cells.
        counted = census_report(
            contract,
            args.tool,
            only,
            include_unmeasured=args.include_unmeasured,
            without_overrides=args.without_overrides,
        )
        if args.use_json:
            json.dump(counted, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            _write_census(counted)
        # A census reports; it does not judge. Exiting non-zero on a tree with
        # findings would make the one command whose whole purpose is to read a
        # backlog look like a failing check, and a caller would learn to ignore
        # its status.
        sys.exit(0)

    if args.command == "update-baseline":
        lowered, exceeded = update_baseline(contract, _selected(args.tool), contract_path, only)
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
    # A cell *under* its ceiling fails, in the same class as one over it. That is
    # not a new practice being imposed: the zero-headroom operating condition —
    # a cell that falls below its ceiling is re-baselined in the same pull
    # request that lowered it — is already declared, and until now was honoured
    # entirely by hand, across every commit that has moved one of these numbers.
    # This is that condition machine-checked. It lands on nothing the day it
    # arrives, because the tree measures exactly what the contract declares.
    #
    # It fails rather than lowering the ceiling itself, for two reasons, and the
    # second stands alone. A checker that rewrites the declaration it is
    # measuring against is the shape this harness refuses everywhere else. And
    # auto-lowering takes the one diff that records progress out of the pull
    # request where somebody reads it — ``update_baseline``'s own docstring
    # makes the argument for the mirror case: "doing it by re-running a command
    # is how that happens without anyone deciding to."
    #
    # Structurally this can only ever fire on a cell whose ceiling is above
    # zero, since nothing can measure below zero — which today is the six
    # transitional mypy cells and nothing else. No flag narrows it to them,
    # because arithmetic already does.
    failed = bool(report["exceeded"] or report["cleared"] or report["unused_config"])
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

    for unread in report["unmeasured"]:
        logger.warning(
            "%s was not measured: its tier (%s) is outside %s's target set, so it "
            "reports no count rather than a count of zero",
            unread["cell"],
            unread["tier"],
            unread["tool"],
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
    # Beside the breaches rather than above the warnings, where it used to sit
    # as a note. The two are one comparison with two signs, and reading them
    # apart is most of what made "under its ceiling" look like good news that
    # needed nothing doing about it.
    #
    # The advice carries `--contract` when this run was given one. Without it, a
    # check pointed at one declaration prints a command that rewrites a
    # different one — the defect
    # `test_a_baseline_update_rewrites_the_declaration_it_was_pointed_at` pins
    # for the write itself, arriving instead through the sentence that tells a
    # developer which write to perform. A ceiling only falls, so following it
    # would not be undone by re-running anything.
    declared = f" --contract {args.contract}" if args.contract else ""
    for cleared in report["cleared"]:
        logger.error(
            "%s is under its ceiling: %d findings against %d declared, so %d "
            "of headroom is left standing. Write the progress down with:\n"
            "    %s update-baseline --tool %s --cell %s%s",
            cleared["cell"],
            cleared["measured"],
            cleared["ceiling"],
            cleared["ceiling"] - cleared["measured"],
            _INVOCATION,
            cleared["tool"],
            cleared["path"],
            declared,
        )
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
        # "measures exactly", not "is within". Now that both signs of the
        # comparison end the run, a pass is the narrower fact: no cell is above
        # its ceiling *and none is below one either*, so every measured cell is
        # on its number. Saying "within" would understate a verdict this check
        # has just been given the power to make, and understating a guarantee is
        # how the next reader comes to believe there is slack that there is not.
        #
        # Not "every cell", when some of them were never read. The count is the
        # whole point: a reader who is told the tree is within its ceilings while
        # four cells went unmeasured has been told the thing this leg exists to
        # stop the artifact from saying.
        if report["unmeasured"]:
            unread_cells = len(report["unmeasured"])
            logger.info(
                "Every measured cell measures exactly what it declares. %d %s "
                "not measured, and %s contents are unknown rather than zero.",
                unread_cells,
                "cell was" if unread_cells == 1 else "cells were",
                "its" if unread_cells == 1 else "their",
            )
        else:
            logger.info("Every cell measures exactly what it declares.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

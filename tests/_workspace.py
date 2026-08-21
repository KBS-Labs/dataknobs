"""Shared helpers for the workspace-level guards.

These modules all answer questions about the repository rather than about any
package, so they all need the same three things: where the root is, how to name
a path relative to it, and what the declared Python floor is. Those were copied
into each module as they were written, which is how two of them ended up with
subtly different floor extraction.

Also the single entry point for reading ``bin/`` modules. Their names are
hyphenated, so they cannot be imported normally, and every guard that wants the
same declaration would otherwise carry its own copy of the loader.

One other copy exists and has to: ``bin/package-hashes.py`` reaches a sibling
the same way, in the gate's own path, where nothing under ``tests/`` is
importable. The two are kept in step by
``test_no_third_loader_appears_without_the_same_treatment``, which drives both
and fails on a third. That guard is the residue of this sentence having once
merely *named* the other copy while the correctness fix landed only here.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from functools import cache
from pathlib import Path, PurePosixPath
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def rel(path: Path) -> str:
    """Name a path the way a reader would type it: relative to the repo root."""
    return str(path.relative_to(ROOT))


#: Documents kept as records of a past design rather than as instructions.
#:
#: Rewriting their samples would falsify the record, so they are excluded here
#: and carry a "Historical record" banner telling the reader the same thing.
#: Listed as paths rather than inferred from a directory name, because
#: ``packages/fsm/docs/active/`` is a design archive whose name says the
#: opposite, and a convention that misleads is worse than an explicit list.
HISTORICAL = (
    "/docs/history/",
    "packages/fsm/docs/active/",
    "packages/data/docs/DESIGN_PLAN.md",
    "packages/llm/docs/LLM_ARCHITECTURE_EXPLORATION.md",
)


def documentation_files() -> list[Path]:
    """Every markdown document a reader can reach, minus the historical ones."""
    found: set[Path] = set(ROOT.joinpath("docs").rglob("*.md"))
    for package_docs in ROOT.joinpath("packages").glob("*/docs"):
        found |= set(package_docs.rglob("*.md"))
    found |= set(ROOT.joinpath("packages").glob("*/README.md"))
    if ROOT.joinpath("README.md").exists():
        found.add(ROOT / "README.md")
    return sorted(
        path for path in found if not any(marker in path.as_posix() for marker in HISTORICAL)
    )


def load_bin_module(stem: str) -> ModuleType:
    """Import a ``bin/<stem>.py`` script whose hyphenated name blocks ``import``."""
    return load_module_from_path(stem.replace("-", "_"), ROOT / "bin" / f"{stem}.py")


def load_module_from_path(name: str, script: Path) -> ModuleType:
    """Execute a script as a module, from its source rather than from bytecode.

    The compiled-cache step is skipped deliberately. CPython decides a
    ``__pycache__`` entry is current by comparing the source's size and its
    mtime **truncated to the second**, so two versions of a file that are the
    same length and are written inside the same second are indistinguishable to
    it: the second ``exec_module`` returns the *first* version's code while the
    file on disk holds the second's.

    That is not a contrived pair. It is what a red/green cycle looks like —
    disable a branch, run the guard, restore it, run again — and this repository
    requires that cycle: a guard is not done until it has been shown to go red.
    It cost a false red here, on a guard that was in fact correct, and the next
    step after a false red is to "fix" something that was never broken.

    It has the shape the surrounding program is about. A fresh checkout has no
    ``__pycache__``, so CI is never wrong; only the developer's local run is,
    and it is wrong in the direction that looks like a finished job. Removing
    the cache entry rather than merely declining to write one also repairs a
    stale entry left by an earlier run.
    """
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None, f"could not load {rel(script)}"
    cached = importlib.util.cache_from_source(str(script))
    Path(cached).unlink(missing_ok=True)
    module = importlib.util.module_from_spec(spec)
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = written
    return module


def version_pair(text: str) -> tuple[int, int] | None:
    """Extract the first ``major.minor`` pair from ``text``."""
    match = re.search(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


#: The purpose-built cell, and why the guards over ``bin/quality-contract.py``
#: need one.
#:
#: They used to be driven over whichever ruff cell carried the largest ceiling,
#: which worked for exactly as long as some part of this tree was dirty. Every
#: ruff cell now measures zero — that is the point of the work, not an accident
#: of it — so a guard that inflates a ceiling, pushes a cell under one, or asks
#: whether a census agrees with a measurement has nothing left to say. Two empty
#: tallies agree.
#:
#: So the backlog these guards need is built rather than borrowed.
#: ``quality-fixture/`` is clean under ``pyproject.toml`` and dirty under its own
#: ``ruff.toml``, which selects rules this repository does not. It therefore
#: carries a ceiling of zero in the real declaration, adds nothing to any tier,
#: and still measures something when read under its own configuration.
#: ``test_the_purpose_built_cell_is_dirty_to_itself_and_clean_to_the_gate``
#: asserts both halves, because either one failing quietly is a guard that has
#: stopped distinguishing anything.
QUALITY_FIXTURE = "quality-fixture"

#: The linter configuration it is dirty under.
QUALITY_FIXTURE_CONFIG = f"{QUALITY_FIXTURE}/ruff.toml"

#: Its cells, the one holding more findings first.
#:
#: Two rather than one because three guards watch a command touch the cell it was
#: named and leave the other exactly as declared, which a single cell cannot
#: express. The real declaration holds the parent instead — one cell, so that
#: the tier's claim can be compared against a target ``bin/validate.sh``
#: actually names.
QUALITY_FIXTURE_CELLS = (f"{QUALITY_FIXTURE}/dense", f"{QUALITY_FIXTURE}/sparse")

#: One reason per cell, and they differ — the guard on the writer's encoding
#: asserts that lowering one ceiling leaves the *other* cell's prose exactly as
#: declared, which two identical strings cannot distinguish. The em-dashes are
#: the subject: the default ``ensure_ascii`` escapes them, and that is the
#: defect the guard was written for.
_FIXTURE_REASONS = {
    QUALITY_FIXTURE_CELLS[0]: (
        "the larger half of the fixture — deliberate findings under the fixture "
        "linter configuration, spread over two files so a breach can rank them"
    ),
    QUALITY_FIXTURE_CELLS[1]: (
        "the smaller half — enough to stay a ceiling when a guard pushes it five "
        "findings under what the tree holds"
    ),
}


def _fixture_declaration(ceilings: dict[str, int]) -> dict:
    """The real declaration, read under the fixture's linter configuration.

    Derived rather than written, and derived from the file the repository is
    actually measured against. A hand-built declaration would have to be a total
    partition of every tracked ``*.py`` to get past ``verify`` — which is the
    right demand, since a command measuring a contract that does not describe
    the tree reports cells that mean nothing — and keeping a second copy of that
    partition in step with the first is a job nobody would remember to do.

    Two edits, and they are the whole difference. The ruff configuration becomes
    the fixture's, so the tree is read under rules this repository declines; and
    the single ``quality-fixture`` cell becomes its two halves, which prefix
    matching cannot express as an addition because the parent would then cover
    them as well. The parent is what the real declaration holds, because a
    ``checked`` tier is compared against ``bin/validate.sh``'s targets and the
    script names the directory rather than its halves.
    """
    declared = (ROOT / ".dataknobs" / "quality-contract.json").read_text("utf-8")
    contract: dict = json.loads(declared)
    ruff = contract["tools"]["ruff"]
    ruff["config"] = QUALITY_FIXTURE_CONFIG

    parent = [cell for cell in ruff["cells"] if cell["path"] == QUALITY_FIXTURE]
    assert len(parent) == 1, (
        f"the ruff declaration holds {len(parent)} cells named {QUALITY_FIXTURE!r}, "
        "so there is nothing single to split into its two halves"
    )
    ruff["cells"] = [cell for cell in ruff["cells"] if cell["path"] != QUALITY_FIXTURE] + [
        {
            "path": cell,
            "tier": parent[0]["tier"],
            "ceiling": ceilings[cell],
            "reason": _FIXTURE_REASONS[cell],
        }
        for cell in QUALITY_FIXTURE_CELLS
    ]
    return contract


@cache
def _fixture_ceilings() -> tuple[tuple[str, int], ...]:
    """What the fixture measures, taken rather than written down.

    A count in this file would be the defect the contract's own ``verify``
    rejects in a cell's reason: true when written, false the moment the fixture
    is edited, and disagreed with by nothing. So it is measured once per session
    and the declaration is built around the answer — which also means a guard
    that inflates a ceiling by 500 and asserts it comes back is comparing
    against the tree rather than against a number somebody typed.

    The floor is the anti-vacuity assertion the ceiling ranking used to carry.
    Both halves must measure above it, because the guard that pushes one five
    findings under its ceiling needs the result to still be a ceiling.
    """
    module = load_bin_module("quality-contract")
    contract = _fixture_declaration(dict.fromkeys(QUALITY_FIXTURE_CELLS, 0))
    measurement = module.measure_ruff(contract, module.tracked_python(), set(QUALITY_FIXTURE_CELLS))
    measured = {
        cell: sum(measurement.by_cell.get(cell, {}).values()) for cell in QUALITY_FIXTURE_CELLS
    }

    larger, smaller = QUALITY_FIXTURE_CELLS
    assert measured[smaller] > 5, (
        f"the purpose-built cell measures {measured}, and the smaller half is at "
        "or below 5 — too small for these guards to tell a real measurement from "
        f"a clean one. Either {QUALITY_FIXTURE}/ was edited, or a rule "
        f"{QUALITY_FIXTURE_CONFIG} selects has since been adopted repo-wide and "
        "no longer has anything left to report there."
    )
    assert measured[larger] > measured[smaller], (
        f"the purpose-built halves measure {measured}, so the pair is no longer "
        f"ordered. {larger} is named first because the guards that take both "
        "inflate one and push the other under its ceiling, and an order that "
        "silently reverses swaps which is which."
    )
    return tuple(measured.items())


def quality_fixture_contract() -> dict:
    """A fresh declaration over the purpose-built cell, at what it measures.

    Fresh on every call rather than shared: every guard here mutates the
    ceilings it is handed, so one cached object would carry a ``+500`` from the
    test that wrote it into the next one to ask.
    """
    return _fixture_declaration(dict(_fixture_ceilings()))


def write_quality_fixture_contract(directory: Path) -> Path:
    """Write the fixture declaration into ``directory`` and name the file.

    For the guards that drive the command rather than the function. The exit
    status is the only thing carrying a census's refusals out, and an in-process
    call has none to check.
    """
    destination = directory / "quality-contract.json"
    destination.write_text(
        json.dumps(quality_fixture_contract(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def python_floor() -> tuple[int, int]:
    """The workspace Python floor, taken from the root ``requires-python``.

    Every other declaration of a Python level is measured against this, so it
    is read rather than written down — a literal here would need an edit that
    nothing would prompt, which is the failure these guards exist to catch.
    """
    requires = load_toml(ROOT / "pyproject.toml")["project"]["requires-python"]
    pair = version_pair(requires)
    assert pair is not None, f"root requires-python is unparseable: {requires!r}"
    return pair


def pyprojects() -> list[Path]:
    """The root ``pyproject.toml`` and every package's, in a stable order."""
    return [ROOT / "pyproject.toml", *sorted(ROOT.glob("packages/*/pyproject.toml"))]


@cache
def tracked_files() -> tuple[str, ...]:
    """Every tracked path, root-relative, as ``git ls-files`` reports it.

    Tracked rather than walked: a filesystem walk cannot tell a directory the
    repository has from one a build left behind, and ``htmlcov/``, ``dist/``
    and ``site/`` all look like ordinary directories. A guard whose answer
    depends on whether coverage was last run is not a guard.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = tuple(name for name in listing.split("\0") if name)
    assert found, "no tracked files found — has the enumeration broken?"
    return found


@cache
def tracked_and_new_files() -> tuple[str, ...]:
    """``tracked_files`` plus files added but not yet committed.

    ``git ls-files`` cannot see a file that has never been committed, so a guard
    built on it alone is blind to exactly the file a developer is writing when
    they run it. It reports green on content it has not read, then goes red in
    CI once the file is committed — the answer changes with no edit between the
    two runs, which reads as a flaky guard rather than a real finding.

    This is not hypothetical. ``test_prose_cross_references`` was verified
    against a tree that excluded the file being verified, so it passed on a
    docstring naming three tests that do not exist, and said so only after the
    commit that made it visible to itself.

    ``--exclude-standard`` keeps the reason ``tracked_files`` is tracked rather
    than walked: ignored paths stay ignored, so ``htmlcov/`` and ``dist/`` are
    still absent whether or not coverage was last run. The difference is only
    that a new source file counts before it is staged.

    Additive on purpose. ``tracked_files`` decides the quality contract's
    totality, where an uncommitted scratch file becoming a coverage gap is a
    behaviour change and not obviously the wanted one.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    new = tuple(name for name in listing.split("\0") if name)
    return tuple(sorted({*tracked_files(), *new}))


@cache
def workspace_targets() -> tuple[str, ...]:
    """The first-party code belonging to no package, from the one declaration.

    Executed rather than parsed. The declaration is four filesystem tests, so
    reading it as text would report what it says while the question is what it
    returns.

    Lives here for the reason ``tracked_shell_files`` does: a third guard now
    asks it, and the answer decides what each of them scans. Three private
    copies of a subprocess call is how two of them end up disagreeing about
    which directories are in scope, with neither reporting that they do.
    """
    listing = subprocess.run(
        [str(ROOT / "bin" / "package-discovery.sh"), "workspace-targets"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = tuple(listing.split())
    assert found, "bin/package-discovery.sh workspace-targets named nothing"
    return found


@cache
def tracked_dirs() -> frozenset[str]:
    """Every directory holding at least one tracked file, root-relative."""
    dirs: set[str] = set()
    for name in tracked_files():
        parent = PurePosixPath(name).parent
        while str(parent) != ".":
            dirs.add(str(parent))
            parent = parent.parent
    return frozenset(dirs)


@cache
def tracked_python_files() -> tuple[str, ...]:
    """Every tracked ``*.py``, root-relative and sorted.

    The whole tree, not a scope: callers narrow it themselves, and the two that
    do want opposite widths. The handle guard reads only the directories the
    workspace owns, because package code has legitimate instances of what it
    rejects; the suppression guard reads everything precisely *because* some of
    it is unlinted, which is where a directive nothing enforces survives.

    Shared for the reason ``tracked_shell_files`` is: the answer to "which files
    are Python" now decides what two guards scan, and a second copy of the walk
    is how the Python-floor extraction ended up with two subtly different
    answers.
    """
    found = tuple(sorted(name for name in tracked_files() if name.endswith(".py")))
    assert found, "no tracked Python files found — has the enumeration broken?"
    return found


@cache
def tracked_shell_files() -> tuple[str, ...]:
    """Every tracked shell script, by ``git ls-files`` and then by shebang.

    Extension alone would miss ``bin/dk``, which is the entry point everything
    else is invoked through — and missing exactly that file is the shape of gap
    the guards reading this exist to close. Tracked files only: a filesystem
    walk picks up whatever an untracked scratch script left in ``bin/``.

    Lives here because two guards now ask the same question — which scripts are
    linted, and which of them run a linter — and a second copy of this walk is
    how the Python-floor extraction ended up with two subtly different answers.
    """
    found = []
    for name in tracked_files():
        path = ROOT / name
        if name.endswith(".sh"):
            found.append(name)
            continue
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                first = handle.readline()
        except OSError:  # pragma: no cover - unreadable tracked file
            continue
        if first.startswith(b"#!") and b"sh" in first.split(b"\n")[0]:
            found.append(name)
    assert found, "no tracked shell files found — has the enumeration broken?"
    return tuple(sorted(found))

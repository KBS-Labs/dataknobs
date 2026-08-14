"""Shared helpers for the workspace-level guards.

These modules all answer questions about the repository rather than about any
package, so they all need the same three things: where the root is, how to name
a path relative to it, and what the declared Python floor is. Those were copied
into each module as they were written, which is how two of them ended up with
subtly different floor extraction.

Also the single entry point for reading ``bin/`` modules. Their names are
hyphenated, so they cannot be imported normally — ``bin/package-hashes.py``
already carries a private copy of this loader, and every guard that wants the
same declaration would otherwise carry a third.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import tomllib
from functools import cache
from pathlib import Path, PurePosixPath
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def rel(path: Path) -> str:
    """Name a path the way a reader would type it: relative to the repo root."""
    return str(path.relative_to(ROOT))


def load_bin_module(stem: str) -> ModuleType:
    """Import a ``bin/<stem>.py`` script whose hyphenated name blocks ``import``."""
    script = ROOT / "bin" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace("-", "_"), script)
    assert spec is not None and spec.loader is not None, f"could not load {rel(script)}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def version_pair(text: str) -> tuple[int, int] | None:
    """Extract the first ``major.minor`` pair from ``text``."""
    match = re.search(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def biggest_ruff_cells(contract: dict, count: int = 1) -> list[str]:
    """The ``count`` ruff cells carrying the largest ceilings, largest first.

    Asked of the declaration rather than named. Deferred cells are promoted one
    at a time, and the tests backlog was a single glob until it was split into a
    cell per package so each could clear on its own — it collapses back to a glob
    when the last of them does. A literal name fails with a ``KeyError`` on the
    day of any of those moves: a guard going red over a change it holds no
    opinion about, saying nothing about the property it exists for.

    The floor is the anti-vacuity assertion, and it is why this returns cells
    rather than a name. Every caller needs one that measures well above zero — to
    inflate, to push under a ceiling, or to keep a census agreeing with a
    measurement from being two empty tallies agreeing because both are empty.
    Once the backlog clears past this the guards stop distinguishing anything, so
    they say so and stop, rather than passing quietly over nothing.

    Lives here because two modules ask it. They had a copy each, differing in
    sort direction and in whether the floor was checked against the smallest of
    the chosen cells or against the only one — the same divergence the Python
    floor extraction had before it moved here.
    """
    ranked = sorted(
        contract["tools"]["ruff"]["cells"], key=lambda cell: cell["ceiling"], reverse=True
    )
    assert len(ranked) >= count, f"the ruff declaration holds only {len(ranked)} cells"
    chosen = [cell["path"] for cell in ranked[:count]]
    assert ranked[count - 1]["ceiling"] > 5, (
        f"the {count} largest ruff ceilings are {chosen}, and the smallest of them "
        "is at or below 5 — too small for these guards to tell a real measurement "
        "from a clean cell. The backlog has been cleared past what they assume; "
        "drive them over a purpose-built cell instead."
    )
    return chosen


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

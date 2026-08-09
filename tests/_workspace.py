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

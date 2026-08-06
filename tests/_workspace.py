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
import tomllib
from pathlib import Path
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

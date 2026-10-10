"""Reproduce-first guard: the quality gate runs Python through ``uv``, never a bare ``python3``.

A bare ``python3`` is whichever interpreter the caller's ``PATH`` finds first,
and the gate does not choose the caller. On macOS a shell without the pyenv
shims finds ``/usr/bin/python3``, which is 3.9: ``docs-mirror-check.py`` calls
``zip(..., strict=True)``, which 3.9 rejects with ``TypeError: zip() takes no
keyword arguments``, so the doc-mirror check failed before comparing a single
document. Every other step of the same run, started through ``uv run``, used
the workspace's 3.12.

It need not fail loudly. ``manage-services.sh`` parsed the compose file with a
bare ``python3 -c "import yaml ..."`` and swallowed the error, so an
interpreter without PyYAML quietly yielded the default service list. The
change-detection parse in ``run-quality-checks.sh`` was the first instance:
there, a failed parse read as *no changes* and skipped the test suite while
reporting success.

The population is computed rather than listed: the scripts the gate starts
from, and every ``$SCRIPT_DIR/<name>.sh`` they reach, transitively. A guard over
the one script that failed is the shape that lets the same call live in the
script it calls.
"""

from __future__ import annotations

import re

from tests._workspace import ROOT

BIN = ROOT / "bin"

#: Where a gate run starts: ``bin/dk pr`` runs ``run-quality-checks.sh``, CI
#: verifies its artifacts with ``validate-quality-artifacts.sh``, and ``bin/dk``
#: reads a failed run back with ``diagnose-quality-failures.sh``.
ENTRY_POINTS = (
    "run-quality-checks.sh",
    "validate-quality-artifacts.sh",
    "diagnose-quality-failures.sh",
)

#: A sibling script one of them runs.
CALLED_RE = re.compile(r"\$\{?SCRIPT_DIR\}?/([\w.-]+\.sh)\b")

#: The word ``python`` or ``python3`` used as a command. Not part of a path
#: (``.venv/bin/python``) or a longer name (``python_version``).
PYTHON_RE = re.compile(r"(?<![\w/.-])python3?(?![\w.-])")

#: The one way the gate starts Python: the workspace's interpreter, via uv.
UV_RE = re.compile(r"\buv run(?: --no-project)? $")


def gate_scripts() -> list[str]:
    """Every script a gate run reaches, from its entry points."""
    seen: set[str] = set()
    queue = list(ENTRY_POINTS)
    while queue:
        name = queue.pop()
        if name in seen or not (BIN / name).is_file():
            continue
        seen.add(name)
        queue.extend(CALLED_RE.findall((BIN / name).read_text()))
    return sorted(seen)


def bare_python_calls(name: str) -> list[str]:
    """``name:line: text`` for each Python started other than through uv."""
    found = []
    for number, line in enumerate((BIN / name).read_text().splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        for match in PYTHON_RE.finditer(line):
            if not UV_RE.search(line[: match.start()]):
                found.append(f"bin/{name}:{number}: {line.strip()}")
    return found


def test_the_population_reaches_past_the_entry_points() -> None:
    scripts = gate_scripts()
    assert set(ENTRY_POINTS) <= set(scripts)
    # The two that failed are each one call away from an entry point.
    assert {"docs-checks.sh", "manage-services.sh"} <= set(scripts)


def test_every_python_the_gate_starts_is_the_workspace_interpreter() -> None:
    found = [call for name in gate_scripts() for call in bare_python_calls(name)]
    assert not found, (
        "Start Python with `uv run python`: a bare python3 is whatever PATH finds "
        "first, which on macOS can be the system 3.9.\n" + "\n".join(found)
    )

"""A run that names more than one package must still collect.

Every check in this file exists because of one defect: **which module-naming
rules apply is decided by how the command was invoked.** A single-package run
resolves its package's ``[tool.pytest.ini_options]``; a run naming two packages
— or naming none, which is how ``testpaths`` reaches all of them — resolves the
root ``pytest.ini`` instead. The two declared different import modes, so the
same file was imported by different rules depending on the argument list, and
neither configuration said so.

The visible cost was not in the gate, which runs one pytest process per target
and so never meets a multi-package invocation. It was on the two invocations a
person actually types: a bare ``pytest`` at the repo root, and this suite run
beside the packages it guards.

``test_a_whole_workspace_collection_reports_no_errors`` is the behavioural
backstop and cannot drift from pytest — it asks pytest. The two structural
checks below it exist for the error message: they name the colliding
directories, which a collection traceback does not.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from tests._workspace import ROOT, load_toml, pyprojects, rel

# The import mode every invocation must resolve. Read from the root config
# rather than written down, so the two cannot disagree silently — a literal
# here would keep passing after someone changed the root and would then be
# asserting the old answer.
_IMPORT_MODE_RE = re.compile(r"--import-mode=(\S+)")


def _root_import_mode() -> str:
    """The ``--import-mode`` the root ``pytest.ini`` declares."""
    text = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    match = _IMPORT_MODE_RE.search(text)
    assert match is not None, (
        "pytest.ini declares no --import-mode. Every package's "
        "[tool.pytest.ini_options] mirrors this value, so removing it here "
        "leaves them asserting nothing — delete the mirrors too, or restore it."
    )
    return match.group(1)


def _package_pytest_blocks() -> dict[str, dict]:
    """Each package ``pyproject.toml`` that declares a pytest block."""
    blocks = {}
    for path in pyprojects():
        if path.parent == ROOT:
            continue
        ini = load_toml(path).get("tool", {}).get("pytest", {}).get("ini_options")
        if ini is not None:
            blocks[rel(path)] = ini
    return blocks


def _root_namespace_claims() -> dict[str, list[str]]:
    """Top-level names the repo root supplies, derived from live imports.

    ``tests/`` carries no ``__init__.py``, so it supplies ``tests`` only
    because something imports through it — which the guards in this directory
    do, for ``tests._workspace``. Derived from those imports rather than
    written down, so deleting the last such import retires the claim instead
    of leaving a stale assertion behind.
    """
    claims: dict[str, list[str]] = {}
    pattern = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)\.", re.M)
    for source in sorted((ROOT / "tests").glob("*.py")):
        for name in pattern.findall(source.read_text(encoding="utf-8")):
            if (ROOT / name).is_dir():
                claims.setdefault(name, [])
                if name not in claims[name]:
                    claims[name].append(name)
    return claims


def _top_level_package_claims() -> dict[str, list[str]]:
    """Map each top-level importable name to every directory claiming it.

    A directory supplies a *top-level* name when it holds ``__init__.py`` and
    its parent does not: that is where pytest stops walking up, so the name it
    hands the import system is the directory's own, unqualified by anything
    above it. Two directories supplying the same name are indistinguishable to
    ``sys.modules``, and the second one imported loses.
    """
    claims = _root_namespace_claims()
    for init in sorted(ROOT.glob("packages/*/tests/**/__init__.py")):
        directory = init.parent
        if (directory.parent / "__init__.py").exists():
            continue
        claims.setdefault(directory.name, []).append(rel(directory))
    return claims


def _collect(*targets: str) -> subprocess.CompletedProcess[str]:
    """Run pytest's collection phase in a clean child process."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            *targets,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_a_whole_workspace_collection_reports_no_errors() -> None:
    """A bare ``pytest`` at the repo root must collect every test.

    This is the invocation with no arguments to get wrong, and ``testpaths``
    points it at every package plus this directory — so it is also the widest
    multi-package run there is. Asking pytest rather than re-deriving its
    naming rules is the point: a check that reimplements them can drift into
    agreeing with itself while the real run still aborts.
    """
    result = _collect()
    errors = sorted(
        line for line in result.stdout.splitlines() if line.startswith("ERROR ")
    )
    assert not errors, (
        "a bare `pytest` at the repo root does not collect:\n  "
        + "\n  ".join(errors)
        + "\n\nEach of these resolved a module name that another file in the "
        "run had already claimed. See the collision report from "
        "test_no_two_directories_claim_the_same_top_level_name."
    )
    assert result.returncode == 0, (
        f"collection exited {result.returncode} without naming an ERROR line:\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


def test_no_two_directories_claim_the_same_top_level_name() -> None:
    """No top-level importable name may be supplied by two directories.

    The failure this catches is silent in a single-package run and fatal in
    any run spanning both claimants — which is why it went unnoticed: every
    package passes alone.
    """
    claims = _top_level_package_claims()
    collisions = {name: dirs for name, dirs in claims.items() if len(dirs) > 1}
    assert not collisions, "top-level import names claimed by more than one directory:\n" + "\n".join(
        f"  {name!r} claimed by: {', '.join(dirs)}" for name, dirs in sorted(collisions.items())
    )


@pytest.mark.parametrize("pyproject", sorted(_package_pytest_blocks()))
def test_every_package_pytest_block_mirrors_the_root_import_mode(
    pyproject: str,
) -> None:
    """A package's pytest block must resolve the same import mode as the root.

    Only one of the two configurations applies to any given run, and which one
    is decided by the argument list. When they disagree, a file that imports
    cleanly under ``pytest packages/llm/tests`` fails under ``pytest
    packages/llm/tests packages/common/tests`` — with nothing in either
    configuration to suggest the argument list was load-bearing.

    ``asyncio_mode`` was already mirrored for exactly this reason, and the
    worked example in the comment beside it is a run that failed on import
    mode for years afterwards.
    """
    ini = _package_pytest_blocks()[pyproject]
    addopts = ini.get("addopts", [])
    if isinstance(addopts, str):
        addopts = addopts.split()
    declared = [_IMPORT_MODE_RE.match(opt) for opt in addopts]
    found = [m.group(1) for m in declared if m is not None]
    expected = _root_import_mode()
    assert found == [expected], (
        f"{pyproject} declares import mode {found or 'nothing'}; the root "
        f"pytest.ini declares {expected!r}. A run that resolves this block "
        f"would import the same files by different rules than a run that "
        f"resolves the root."
    )

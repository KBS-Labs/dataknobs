"""Reproduce-first guard for toolchain Python-level declarations.

Every project in the workspace declares ``requires-python = ">=3.12"``. Any
*other* declaration of a Python level — a type checker's target, a formatter's
target, an interpreter pin, a published classifier, or the scaffolding template
that seeds new packages — must agree with that floor.

A stale declaration fails silently, and in one direction only: telling a tool to
assume an older interpreter makes it reject or avoid what is actually
available, and it never reports the gap. The cost shows up as an *absence*, so
nothing goes red. These tests are the thing that goes red.

Each test collects **every** violation before asserting, so one run reports the
whole drift surface rather than the first item found.

``ruff``'s ``target-version`` is deliberately **excluded** — see
``test_ruff_target_version_is_deliberately_pinned``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: ``ruff``'s target is knowingly held below the floor. Raising it surfaces a
#: large modernization-lint surface across several rule families, most wanting a
#: different public API rather than different syntax, so it lands separately.
#: Pinned here so the hold stays a recorded decision rather than more drift.
RUFF_PINNED_TARGET = "py310"


def _pyprojects() -> list[Path]:
    return [ROOT / "pyproject.toml", *sorted(ROOT.glob("packages/*/pyproject.toml"))]


def _mypy_inis() -> list[Path]:
    return [p for p in [ROOT / "mypy.ini", *sorted(ROOT.glob("packages/*/mypy.ini"))] if p.exists()]


def _interpreter_pins() -> list[Path]:
    return [
        p
        for p in [ROOT / ".python-version", *sorted(ROOT.glob("packages/*/.python-version"))]
        if p.exists()
    ]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _version_pair(text: str) -> tuple[int, int] | None:
    """Extract the first ``major.minor`` pair from ``text``."""
    match = re.search(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


@pytest.fixture(scope="module")
def floor() -> tuple[int, int]:
    """The workspace Python floor, taken from the root ``requires-python``."""
    requires = _load(ROOT / "pyproject.toml")["project"]["requires-python"]
    pair = _version_pair(requires)
    assert pair is not None, f"root requires-python is unparseable: {requires!r}"
    return pair


def _fmt(violations: list[str], floor: tuple[int, int]) -> str:
    listed = "\n".join(f"  - {v}" for v in violations)
    return f"Declarations disagree with the >={floor[0]}.{floor[1]} floor:\n{listed}"


# --------------------------------------------------------------------------
# requires-python — the floor every other declaration is measured against
# --------------------------------------------------------------------------


def test_every_project_declares_the_same_floor(floor):
    violations = [
        f"{_rel(path)}: requires-python = {requires!r}"
        for path in _pyprojects()
        if (requires := _load(path).get("project", {}).get("requires-python")) is not None
        and _version_pair(requires) != floor
    ]

    assert not violations, _fmt(violations, floor)


# --------------------------------------------------------------------------
# Type checker / formatter targets
# --------------------------------------------------------------------------


def test_mypy_python_version_matches_floor(floor):
    """A stale mypy target makes it reject syntax the interpreter accepts.

    Concretely: with a 3.10 target, mypy treats a dependency's PEP 695 ``type``
    statements as a fatal syntax error and aborts on that dependency — so it
    silently type-checks nothing there.
    """
    violations = []

    for path in _pyprojects():
        version = _load(path).get("tool", {}).get("mypy", {}).get("python_version")
        if version is not None and _version_pair(str(version)) != floor:
            violations.append(f"{_rel(path)}: [tool.mypy] python_version = {version!r}")

    for path in _mypy_inis():
        for match in re.finditer(r"^\s*python_version\s*=\s*(\S+)", path.read_text(), re.MULTILINE):
            if _version_pair(match.group(1)) != floor:
                violations.append(f"{_rel(path)}: python_version = {match.group(1)!r}")

    assert not violations, _fmt(violations, floor)


def test_black_target_version_matches_floor(floor):
    expected = f"py{floor[0]}{floor[1]}"
    violations = [
        f"{_rel(path)}: [tool.black] target-version = {targets!r} (want [{expected!r}])"
        for path in _pyprojects()
        if (targets := _load(path).get("tool", {}).get("black", {}).get("target-version"))
        and list(targets) != [expected]
    ]

    assert not violations, _fmt(violations, floor)


def test_pylint_py_version_matches_floor(floor):
    """``.pylintrc`` is live via ``bin/dk lint`` and ``tox.ini``."""
    pylintrc = ROOT / ".pylintrc"
    if not pylintrc.exists():
        pytest.skip("no .pylintrc")

    violations = [
        f".pylintrc: py-version={match.group(1)}"
        for match in re.finditer(r"^\s*py-version\s*=\s*(\S+)", pylintrc.read_text(), re.MULTILINE)
        if _version_pair(match.group(1)) != floor
    ]

    assert not violations, _fmt(violations, floor)


def test_ruff_target_version_is_deliberately_pinned():
    """``ruff``'s target is knowingly held below the floor; pin that decision.

    This is the one declaration that may lag. It is asserted rather than
    skipped so that raising it is a deliberate edit here, not silent drift in
    either direction.
    """
    violations = [
        f"{_rel(path)}: [tool.ruff] target-version = {target!r}"
        for path in _pyprojects()
        if (target := _load(path).get("tool", {}).get("ruff", {}).get("target-version")) is not None
        and target != RUFF_PINNED_TARGET
    ]

    assert not violations, (
        f"ruff target-version is pinned at {RUFF_PINNED_TARGET!r} pending a separate "
        f"modernization pass; update RUFF_PINNED_TARGET when that lands:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# --------------------------------------------------------------------------
# Interpreter pins
# --------------------------------------------------------------------------


def test_interpreter_pins_satisfy_floor(floor):
    """``uv`` reads the *nearest* ``.python-version``.

    A package-level pin below the floor makes ``uv run`` from inside that
    package request an interpreter the package's own ``requires-python``
    declares unsupported.
    """
    violations = [
        f"{_rel(path)}: {content!r}"
        for path in _interpreter_pins()
        if (content := path.read_text().strip())
        and ((pair := _version_pair(content)) is None or pair < floor)
    ]

    assert not violations, _fmt(violations, floor)


# --------------------------------------------------------------------------
# Published metadata
# --------------------------------------------------------------------------


def test_no_classifier_below_floor(floor):
    """A classifier below ``requires-python`` advertises support pip refuses."""
    violations = []

    for path in _pyprojects():
        for classifier in _load(path).get("project", {}).get("classifiers", []):
            if not classifier.startswith("Programming Language :: Python :: "):
                continue
            pair = _version_pair(classifier.rsplit("::", 1)[-1])
            if pair is not None and pair < floor:
                violations.append(f"{_rel(path)}: {classifier!r}")

    assert not violations, _fmt(violations, floor)


# --------------------------------------------------------------------------
# The scaffolding template — where drift regenerates itself
# --------------------------------------------------------------------------


def test_new_package_template_declares_the_floor(floor):
    """Without this, every newly scaffolded package is born already stale."""
    script = ROOT / "bin" / "create-package.py"
    if not script.exists():
        pytest.skip("no bin/create-package.py")

    violations = [
        f"bin/create-package.py: requires-python = {match.group(1)!r}"
        for match in re.finditer(
            r'requires-python\s*=\s*"([^"]+)"', script.read_text(encoding="utf-8")
        )
        if _version_pair(match.group(1)) != floor
    ]

    assert not violations, _fmt(violations, floor)

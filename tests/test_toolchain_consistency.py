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

These are workspace-level guards: they read the root config, every package's
config, and ``bin/``. They belong to no package, so they live here — which puts
them outside the per-package discovery every test entry point uses. See
``test_workspace_tests_are_reachable``, which is the reason the rest of this
file is worth writing.
"""

from __future__ import annotations

import configparser
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Where the quality gate names the directory holding these tests. Extracted and
#: resolved rather than searched for, so the guard cannot pass on a coincidence.
GATE_TEST_DIR_RE = re.compile(r'^\s*WORKSPACE_TEST_DIR="\$PROJECT_ROOT/([^"]+)"', re.MULTILINE)


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
# Reachability — whether anything runs these guards at all
# --------------------------------------------------------------------------


def test_workspace_tests_are_reachable():
    """A guard nothing runs reports green in exactly the way a passing one does.

    Every test entry point is keyed by package — ``pytest.ini`` declares
    ``testpaths``, ``bin/test.sh`` takes a package name, and the quality gate
    loops ``packages/*``. This directory is in none of those by construction,
    so the guards here can go red and stay red without a single check turning
    red with them.

    Two mechanisms, because they cover different callers: ``testpaths`` covers
    a bare ``pytest`` at the root, and the gate covers CI. Both are read for a
    path and resolved, so neither can be satisfied by an unrelated mention of
    the word "tests" — and the gate is checked for *running* that path, not
    merely naming it, since a variable defined and never used would satisfy a
    weaker check while running nothing.
    """
    here = Path(__file__).resolve().parent
    violations = []

    parser = configparser.ConfigParser()
    parser.read(ROOT / "pytest.ini")
    testpaths = parser.get("pytest", "testpaths", fallback="").split()
    if not any((ROOT / p).resolve() == here for p in testpaths):
        violations.append(f"pytest.ini: testpaths = {' '.join(testpaths)!r} does not cover {_rel(here)}")

    gate = ROOT / "bin" / "run-quality-checks.sh"
    gate_text = gate.read_text()
    named = [(ROOT / m.group(1)).resolve() for m in GATE_TEST_DIR_RE.finditer(gate_text)]
    if here not in named:
        found = ", ".join(_rel(p) for p in named) or "nothing"
        violations.append(f"bin/run-quality-checks.sh: names {found}, not {_rel(here)}")
    else:
        # Naming the directory is not running it. Both modes are checked: the
        # gate defines the variable once and each branch has its own test path,
        # so dropping either leaves a mode that reports green without the
        # guards — the failure this whole file exists to make impossible.
        runs = len(re.findall(r'uv run pytest "\$WORKSPACE_TEST_DIR"', gate_text))
        if runs < 2:
            violations.append(
                f"bin/run-quality-checks.sh: names {_rel(here)} but runs pytest against it "
                f"{runs} time(s) — expected both the PR-mode and dev-mode paths"
            )

    assert not violations, "Workspace guards are unreachable:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


#: Variables the gate may interpolate into a direct ``pytest`` call. Everything
#: else in scope there holds ``bin/test.sh`` flags, which pytest rejects.
PYTEST_SAFE_VARS = frozenset({"WORKSPACE_TEST_DIR", "PYTEST_ARGS", "ARTIFACTS_DIR"})


def _gate_pytest_commands() -> list[str]:
    """Every direct ``uv run pytest`` command in the gate, continuations joined."""
    lines = (ROOT / "bin" / "run-quality-checks.sh").read_text().splitlines()
    commands = []
    for i, line in enumerate(lines):
        if "uv run pytest" not in line:
            continue
        parts, j = [line], i
        while j + 1 < len(lines) and lines[j].rstrip().endswith("\\"):
            j += 1
            parts.append(lines[j])
        commands.append(" ".join(p.strip().rstrip("\\").strip() for p in parts))
    return commands


def test_the_gate_passes_only_pytest_arguments_to_pytest():
    """``$TEST_FLAGS`` holds ``bin/test.sh`` flags, not pytest ones.

    Every other test path in the gate goes through ``bin/test.sh``, which
    translates ``--parallel`` / ``--quiet`` into pytest's own spelling. The
    workspace step cannot: ``test.sh`` takes a package name and scans
    ``packages/*``, so this one run calls pytest directly. Handing it a
    variable holding ``--parallel`` makes pytest exit on a *usage* error —
    which the gate counts as a failing test suite, while no test failed and
    nothing in the summary says why.
    """
    commands = _gate_pytest_commands()
    assert commands, "no direct pytest invocation found in bin/run-quality-checks.sh"

    violations = [
        f"bin/run-quality-checks.sh: pytest receives ${var}, not a pytest argument"
        for cmd in commands
        for var in re.findall(r"\$([A-Z_]+)", cmd)
        if var not in PYTEST_SAFE_VARS
    ]

    assert not violations, "Non-pytest arguments reach pytest:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


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


def test_ruff_target_version_matches_floor(floor):
    """A stale ruff target makes it decline modernizations that are available.

    This one used to be pinned below the floor while the modernization surface
    it gates was worked through. That is done, so it is now asserted the same
    way every other target is — against the floor rather than against a frozen
    literal, which is strictly the stronger check.
    """
    expected = f"py{floor[0]}{floor[1]}"
    violations = [
        f"{_rel(path)}: [tool.ruff] target-version = {target!r} (want {expected!r})"
        for path in _pyprojects()
        if (target := _load(path).get("tool", {}).get("ruff", {}).get("target-version")) is not None
        and target != expected
    ]

    assert not violations, _fmt(violations, floor)


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

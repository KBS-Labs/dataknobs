"""Reproduce-first guard for hand-maintained toolchain declarations.

Every project in the workspace declares ``requires-python = ">=3.12"``. Any
*other* declaration of a Python level — a type checker's target, a formatter's
target, an interpreter pin, a published classifier, or the scaffolding template
that seeds new packages — must agree with that floor. The same applies to a
toolchain declaration that names a *directory* rather than a version: a search
path pointing at something that does not exist.

Both fail silently, and in one direction only: telling a tool to assume an older
interpreter makes it reject or avoid what is actually available, and handing it
a search path that does not exist makes it look in one fewer place. Neither
reports the gap. The cost shows up as an *absence*, so nothing goes red. These
tests are the thing that goes red.

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
from pathlib import Path

import pytest

from tests._workspace import ROOT, load_bin_module, load_toml, pyprojects, python_floor
from tests._workspace import rel as _rel
from tests._workspace import version_pair as _version_pair

#: Where the quality gate names the directory holding these tests. Extracted and
#: resolved rather than searched for, so the guard cannot pass on a coincidence.
GATE_TEST_DIR_RE = re.compile(r'^\s*WORKSPACE_TEST_DIR="\$PROJECT_ROOT/([^"]+)"', re.MULTILINE)

#: Directories the gate excludes from the workspace run, read from the flags it
#: actually passes. Frozen literals here would stop tracking the gate the first
#: time a second exclusion is added — the same drift these guards exist to catch.
GATE_IGNORE_RE = re.compile(r'--ignore="\$WORKSPACE_TEST_DIR/([^"]+)"')

#: The single declaration of which files outside packages/ affect a quality
#: result. Read rather than restated so this guard cannot drift from the change
#: detection and artifact hashing that consume the same list.
_scopes = load_bin_module("changed-packages")
WORKSPACE_QUALITY_INPUTS: dict[str, list[str]] = _scopes.WORKSPACE_QUALITY_INPUTS

_pyprojects = pyprojects
_load = load_toml


def _mypy_inis() -> list[Path]:
    return [p for p in [ROOT / "mypy.ini", *sorted(ROOT.glob("packages/*/mypy.ini"))] if p.exists()]


def _interpreter_pins() -> list[Path]:
    return [
        p
        for p in [ROOT / ".python-version", *sorted(ROOT.glob("packages/*/.python-version"))]
        if p.exists()
    ]


@pytest.fixture(scope="module")
def floor() -> tuple[int, int]:
    """The workspace Python floor, taken from the root ``requires-python``."""
    return python_floor()


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


def test_a_change_to_these_guards_still_schedules_them():
    """A pull request that edits only this directory must still run it.

    Reachability is not only "does an entry point name the path" — the gate
    runs it inside a block change detection can switch off. These files
    belong to no package by construction, so a diff touching only ``tests/``
    maps to an empty package set, and an empty package set used to mean "run
    no tests at all". The suite that went unrun was the one the PR edited,
    and the PR reported green.

    Asserted against the decision rather than the file list, because the
    empty package set is *correct* here — there is genuinely no package to
    test. What was wrong was reading it as "nothing to test". ``test_scope``
    is the distinction: no-package-changed and nothing-changed are separate
    answers, and only the second one skips.
    """
    scope = _scopes.plan_for_files(["tests/test_toolchain_consistency.py"])

    assert scope["packages"] == [], (
        "a workspace guard belongs to no package — mapping one to a package "
        f"would re-run that package's suite for an unrelated edit, got {scope['packages']}"
    )
    assert scope["workspace_changed"] is True
    assert scope["test_scope"] == "workspace", (
        "a tests/-only diff must schedule the workspace guards; "
        f"got test_scope={scope['test_scope']!r}, which the gate reads as 'skip'"
    )

    # bin/ is the same case from the other direction — the guards read those
    # scripts, so a change to the gate moves their result and nothing else's.
    # Left out, the pull request that fixes the gate skips the gate.
    assert _scopes.plan_for_files(["bin/run-quality-checks.sh"])["test_scope"] == "workspace"

    # The other two answers, so the fix cannot be "always run everything".
    assert _scopes.plan_for_files(["README.md"])["test_scope"] == "none"
    assert _scopes.plan_for_files(["packages/common/src/x.py"])["test_scope"] == "packages"


def test_the_gate_reads_the_scope_change_detection_computes():
    """The decision above only helps if the gate acts on it.

    Text-matched, and deliberately narrow about what that proves: it pins
    that the gate reads ``test_scope`` and that its no-package branch turns
    off the *package* suites rather than the whole test block. It cannot
    prove the workspace run is reachable at runtime — the guard above owns
    the decision and ``test_workspace_tests_are_reachable`` owns the run;
    this is the wire between them, which is the part that was missing.
    """
    gate_text = (ROOT / "bin" / "run-quality-checks.sh").read_text()
    violations = []

    if "test_scope" not in gate_text:
        violations.append(
            "does not read test_scope from change detection, so it cannot tell "
            "'no package changed' from 'nothing changed'"
        )
    if "SKIP_PACKAGE_TESTS" not in gate_text:
        violations.append(
            "has no package-only skip, so the only way to skip the package "
            "suites is SKIP_TESTS, which also skips the workspace guards"
        )

    assert not violations, "bin/run-quality-checks.sh:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_no_workspace_test_is_filed_where_nothing_runs_it():
    """``tests/integration/`` is reached by no entry point, in either mode.

    The unit step skips it by name, on the reasonable assumption that
    "integration" means "needs a running service". The integration step cannot
    reach it either: that loop is ``packages/*/tests/integration``, so a
    workspace-level directory is outside it by construction. A file placed
    here therefore runs nowhere — which is how eight cross-package interop
    tests sat un-run, none of which needed a service in the first place.

    Asserting the directory stays empty is the cheap half of the fix. The
    expensive half — deciding where a workspace test that *does* need a
    service should run — is a real question, and this makes it get asked at
    the moment someone has one rather than after it has silently not run.
    """
    here = Path(__file__).resolve().parent
    gate_text = (ROOT / "bin" / "run-quality-checks.sh").read_text()
    excluded = sorted(set(GATE_IGNORE_RE.findall(gate_text)))
    assert excluded, (
        "bin/run-quality-checks.sh passes no --ignore for the workspace run — "
        "either the gate stopped excluding a directory or this guard stopped "
        "tracking how it spells the flag"
    )

    stranded = [
        path
        for name in excluded
        # rglob, not glob: a test one directory deeper is stranded in exactly
        # the same way and reads as covered under a non-recursive check.
        for path in sorted((here / name).rglob("test_*.py"))
    ]

    assert not stranded, (
        "These tests are collected by no entry point:\n"
        + "\n".join(f"  - {_rel(p)}" for p in stranded)
        + f"\n  The unit step passes --ignore for {', '.join(excluded)} and the "
        "integration step only loops packages/*/tests/integration.\n"
        "  Move them beside the other workspace guards if they need no "
        "service, or give the gate a step that runs them."
    )


#: The workflow whose path filter decides whether the quality gate runs at all.
CI_WORKFLOW = Path(".github/workflows/quality-validation.yml")


def _ci_code_filter_patterns() -> list[str]:
    """The ``code`` path-filter patterns, read without a YAML dependency.

    The filter is a block scalar handed to ``dorny/paths-filter``, so its
    entries are plain quoted strings one per line between ``code:`` and the
    next sibling key.
    """
    text = (ROOT / CI_WORKFLOW).read_text(encoding="utf-8")
    block = re.search(r"^(\s+)code:\n(.*?)(?=^\1\S)", text, re.DOTALL | re.MULTILINE)
    assert block is not None, f"{CI_WORKFLOW}: no 'code:' path filter found"
    return re.findall(r"^\s*-\s*'([^']+)'", block.group(2), re.MULTILINE)


def _glob_to_re(pattern: str) -> re.Pattern[str]:
    """A conservative subset of the matcher ``dorny/paths-filter`` uses.

    Brace expansion and character classes are not translated: they compile to
    literals, match nothing, and so report a file as *un*covered. That is the
    safe direction — a false alarm is read and fixed, a false all-clear is not.
    """
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + r"(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + r".*", i + 2
        elif pattern[i] == "*":
            out, i = out + r"[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + r"[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(f"^{out}$")


def _covers(name: str, patterns: list[str]) -> bool:
    """Whether ``dorny/paths-filter`` would report ``name`` as changed.

    A leading ``!`` is an *override*, not a pattern character: dorny applies
    negations over the positive set, so a file matched by both is excluded.
    Escaping the ``!`` instead — as a naive translation does — yields a regex
    that matches nothing while the positive pattern still matches, and the
    file reads as covered when the real filter would drop it. There are
    already three negations in the docs filter of the same workflow.
    """
    positive = [_glob_to_re(p) for p in patterns if not p.startswith("!")]
    negative = [_glob_to_re(p[1:]) for p in patterns if p.startswith("!")]
    return any(p.match(name) for p in positive) and not any(n.match(name) for n in negative)


def _workspace_input_probes() -> list[str]:
    """One concrete path per declared workspace input, for coverage checking.

    A directory entry is probed through a real file beneath it rather than by
    its own name: a filter reading ``tests/**`` covers ``tests/test_x.py`` and
    not the bare string ``tests``, so checking the directory name would fail
    against a filter that is in fact correct.
    """
    probes: list[str] = []
    for entries in WORKSPACE_QUALITY_INPUTS.values():
        for entry in entries:
            target = ROOT / entry.rstrip("/")
            if entry.endswith("/"):
                probes += [_rel(p) for p in sorted(target.rglob("*.py"))[:1]]
            elif target.exists():
                probes.append(entry)

    assert probes, "no workspace quality inputs resolved — the shared declaration is empty"
    return probes


def test_ci_runs_the_gate_when_a_guarded_file_changes():
    """A guard that CI never starts is the same as a guard that does not exist.

    Every check in this file reads a hand-maintained toolchain declaration,
    and the point of reading it is to catch the pull request that changes it.
    That pull request is exactly the one an over-narrow path filter drops: an
    unprefixed ``pyproject.toml`` pattern matches the root file and nothing
    under ``packages/``, so a change to a package's lint target, type-checker
    target, or published classifiers matched no pattern and the whole quality
    job was skipped. The declarations these tests assert on were the ones CI
    was least likely to look at.

    Files are checked for coverage rather than the pattern list being frozen,
    so adding a new guarded declaration and forgetting its trigger fails here.
    The workspace-level half of that set is read from the one declaration that
    change detection and artifact hashing also consume, so a file added there
    is covered here without anyone remembering to extend a second list.

    This workflow is in the set too. It is the only PR-time quality job, so a
    pull request whose sole change deletes these patterns matches nothing,
    skips the job, and reports green — leaving every declaration below
    untriggered from that merge forward.
    """
    guarded = [_rel(path) for path in (*_pyprojects(), *_mypy_inis(), *_interpreter_pins())]
    guarded += _workspace_input_probes()
    guarded.append(str(CI_WORKFLOW))

    patterns = _ci_code_filter_patterns()
    violations = [
        f"{name}: matched by no pattern, so a change to it skips the quality job"
        for name in sorted(set(guarded))
        if not _covers(name, patterns)
    ]

    assert not violations, (
        f"{CI_WORKFLOW} does not trigger on files these guards assert against:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_workspace_tests_are_linted_and_type_checked():
    """Being *run* is not the same as being *checked*.

    ``bin/validate.sh`` builds its default target list by looping ``packages/*``
    and appending each ``src`` directory, so this directory — which belongs to
    no package — was outside every lint and type-check invocation in the repo.
    The modules here are the ones asserting that the toolchain is coherent, and
    nothing was asserting anything about theirs.

    Read from the target computation rather than from a comment, because the
    comment cannot stop being true.
    """
    validate = ROOT / "bin" / "validate.sh"
    here = Path(__file__).resolve().parent

    default_block = re.search(
        r"if \[\[ \$\{#TARGETS\[@\]\} -eq 0 \]\]; then(.*?)\nelse", validate.read_text(), re.DOTALL
    )
    assert default_block is not None, (
        "bin/validate.sh: could not find the default-target branch — either it "
        "was restructured or this guard stopped recognising it"
    )

    named = re.findall(r'VALIDATE_TARGETS\+=\("([^"$]+)"\)', default_block.group(1))
    assert any((ROOT / name).resolve() == here for name in named), (
        f"bin/validate.sh validates {named or 'nothing'} by default, which does not "
        f"include {_rel(here)} — the workspace guards would go unlinted and untyped"
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
# Search paths — declarations that name a directory instead of a version
# --------------------------------------------------------------------------


#: The mkdocstrings handler's source roots, read without a YAML dependency.
#: A flow-sequence of bare paths, one per line, inside ``paths: [ ... ]``.
MKDOCS_PATHS_RE = re.compile(r"^\s*paths:\s*\[(.*?)\]", re.DOTALL | re.MULTILINE)


def _doc_search_path_entries() -> list[tuple[Path, str]]:
    """Every mkdocstrings source root, as ``(config, entry)``.

    mypy is not the only tool handed a list of directories to look in, and a
    stale entry fails the same way in each: mkdocstrings finds no modules
    under a path that does not exist, emits no API pages for them, and the
    build still succeeds. Reading both tools through one shape is what makes
    this a guard against the *class* rather than against one instance of it.
    """
    mkdocs = ROOT / "mkdocs.yml"
    if not mkdocs.exists():
        return []

    entries: list[tuple[Path, str]] = []
    for block in MKDOCS_PATHS_RE.finditer(mkdocs.read_text(encoding="utf-8")):
        entries += [
            (mkdocs, stripped)
            for raw in block.group(1).split(",")
            if (stripped := raw.strip().strip("'\""))
        ]
    return entries


def _mypy_path_entries() -> list[tuple[Path, str]]:
    """Every ``mypy_path`` entry declared anywhere, as ``(config, entry)``.

    Both spellings are read: the ``.ini`` files hold one colon-separated
    string, while a ``[tool.mypy]`` table may hold either that or a list.
    """
    entries: list[tuple[Path, str]] = []

    for path in _mypy_inis():
        for match in re.finditer(r"^\s*mypy_path\s*=\s*(.+)$", path.read_text(), re.MULTILINE):
            entries += [(path, part) for part in match.group(1).split(":")]

    for path in _pyprojects():
        declared = _load(path).get("tool", {}).get("mypy", {}).get("mypy_path")
        if isinstance(declared, str):
            entries += [(path, part) for part in declared.split(":")]
        elif isinstance(declared, list):
            entries += [(path, str(part)) for part in declared]

    return [(path, stripped) for path, entry in entries if (stripped := entry.strip())]


def test_mypy_path_entries_resolve():
    """A ``mypy_path`` entry that does not exist is skipped without a word.

    mypy does not validate its search path — a directory that was renamed,
    or a package that was planned and never created, simply contributes no
    modules. Every import that would have resolved through it then falls back
    to ``ignore_missing_imports``, so the symbols come back as ``Any`` and the
    run still reports success. The type checking is gone; the green is not.

    Entries are resolved against the repository root because that is where
    every gate script invokes mypy from — ``bin/validate.sh`` passes an
    absolute ``--config-file`` and runs at the root, so a relative entry here
    is root-relative in practice.
    """
    entries = _mypy_path_entries() + _doc_search_path_entries()
    assert entries, (
        "no search-path entry found in any config — either every one was "
        "dropped or this guard stopped recognising how they are spelled"
    )

    violations = [
        f"{_rel(path)}: search-path entry {entry!r} is not a directory"
        for path, entry in entries
        if not (ROOT / entry).is_dir()
    ]

    assert not violations, "Toolchain search paths point at nothing:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_mypy_configs_declare_the_same_search_path():
    """The two live configs may differ in strictness, but not in where source is.

    ``bin/validate.sh`` type-checks against ``mypy.ini`` by default and against
    the root ``[tool.mypy]`` under ``--all-errors``, so both are live and which
    one applies is decided by the flag. Strictness is *supposed* to differ
    between them — that is the point of having two. A search path is not a
    strictness knob: a package's source is in the same place either way, so a
    package listed in one and not the other means the same code type-checks
    differently depending on how the run was invoked, with the weaker side
    resolving those imports to ``Any`` and reporting success.

    Compared as sets, because order carries no meaning here.
    """
    declared: dict[str, set[str]] = {}
    for path, entry in _mypy_path_entries():
        declared.setdefault(_rel(path), set()).add(entry)

    if len(declared) < 2:
        pytest.skip("only one config declares a mypy_path — nothing to compare")

    names = sorted(declared)
    reference = declared[names[0]]
    violations = [
        f"{name}: {sorted(reference ^ declared[name])} declared in one config but not the other"
        for name in names[1:]
        if declared[name] != reference
    ]

    assert not violations, (
        f"mypy search paths disagree with {names[0]}:\n"
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

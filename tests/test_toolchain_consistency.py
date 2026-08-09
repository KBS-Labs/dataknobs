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
import subprocess
from pathlib import Path, PurePosixPath

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


#: A ``bin/`` script named inside a workflow ``run:`` block, i.e. one CI executes.
#: Prose mentions elsewhere in a workflow — a failure comment telling a developer
#: what to run — are not executions and carry no staleness.
_RUN_STEP_SCRIPT_RE = re.compile(r"(bin/[A-Za-z0-9_.-]+\.(?:sh|py))")


def _ci_executed_bin_scripts() -> set[str]:
    """Every ``bin/`` script a workflow actually runs, read from its run: blocks."""
    found: set[str] = set()
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        in_run = False
        run_indent = 0
        for raw in workflow.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            indent = len(raw) - len(raw.lstrip())
            if re.match(r"^-?\s*run:\s*\|?\s*$", stripped) or stripped.startswith("run: "):
                in_run = True
                run_indent = indent
                found |= set(_RUN_STEP_SCRIPT_RE.findall(stripped))
                continue
            # A run: block ends at the first line indented no further than the
            # key itself. Blank lines inside it are not the end.
            if in_run and stripped and indent <= run_indent:
                in_run = False
            if in_run:
                found |= set(_RUN_STEP_SCRIPT_RE.findall(raw))
    return found


def test_every_script_ci_executes_is_covered_by_a_hash_scope():
    """A gate script outside every hash scope lets its own edit go unvalidated.

    The artifact records a verdict, and these scripts are what computed it. Edit
    one and the packages are untouched — every suite that passed still passes —
    but the recorded verdict was produced under different rules, and the stored
    hashes have no way to say so. The pull request that changes the gate is
    exactly the one the gate cannot check, and it reports green.

    That is not hypothetical: ``bin/`` covers only ``*.py`` in the hash scope, so
    every shell script here sat outside it, including the two that write and
    verify the artifact.

    Coverage is asked through ``workspace_scope_files`` — the function the hash
    itself uses — rather than by re-deriving which paths an entry expands to. A
    second implementation of that rule could answer for a rule nothing follows.

    Scoped to what CI *executes*, so adding a developer convenience to ``bin/``
    costs nothing, while adding a script to a workflow forces the decision at
    review time.
    """
    # Both tiers are scopes — _GLOBAL_QUALITY_INPUTS *is* WORKSPACE_QUALITY_INPUTS
    # ["toolchain"] — so one pass over the scopes covers global and workspace-only
    # alike, and a script promoted between tiers stays covered without an edit here.
    hashes = load_bin_module("package-hashes")
    covered = {
        _rel(path)
        for scope in WORKSPACE_QUALITY_INPUTS
        for path in hashes.workspace_scope_files(scope)
    }

    executed = _ci_executed_bin_scripts()
    assert executed, (
        "no bin/ script was found in any workflow run: block — the extraction "
        "broke, and this guard would pass by checking nothing"
    )

    uncovered = sorted(name for name in executed if name not in covered)
    assert not uncovered, (
        "CI executes these scripts, but no hash scope covers them, so editing "
        "one leaves every stored hash intact and its own change unvalidated:\n"
        + "\n".join(f"  - {name}" for name in uncovered)
        + "\n\nAdd each to _GLOBAL_QUALITY_INPUTS (if it moves every package's "
        "result) or _WORKSPACE_ONLY_QUALITY_INPUTS (if it only decides what the "
        "gate checks or records) in bin/changed-packages.py."
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


#: Directories of first-party Python that ``bin/validate.sh`` does not reach
#: when run with no arguments, each with the size that makes it a project rather
#: than an edit. Deferring is a legitimate answer; deferring *silently* is not,
#: and silence is what kept ``bin/`` — which holds the checkers deciding whether
#: a pull request passes — outside every lint invocation in this repo for as
#: long as it has had one. Counts are from the root ruff config.
#:
#: Two rules keep this from becoming the excuse list it would otherwise decay
#: into, both enforced below: an entry matching no tracked file is an error, and
#: so is an entry matching a file that *is* linted — because the cheapest way to
#: silence the coverage test is to drop a directory from the targets and add it
#: here, and that must not be a passing move.
DEFERRED_FROM_DEFAULT_LINT = {
    "packages/*/tests": "~1,790 findings; wants each package's src cleared first",
    "packages/*/examples": "241 findings, ~90% of them under data/ and fsm/",
    "packages/*/scripts": "9 findings",
    "packages/*/benchmarks": "2 findings",
    "packages/*/docs": "7 findings, all in a single fsm/ module",
}


def _tracked_python() -> list[PurePosixPath]:
    """Every ``*.py`` git keeps, as repo-relative paths."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [PurePosixPath(name) for name in listing.split("\0") if name]


#: The scripts that build a default set of things to lint. Each used to answer
#: "which code do we check" for itself, and the copies agreed by accident while
#: there was one directory to name. ``workspace_targets`` in
#: bin/package-discovery.sh is the single answer now; this is who has to be
#: asking it.
WORKSPACE_TARGET_CONSUMERS = (
    "bin/validate.sh",
    "bin/fix.sh",
    "bin/dk",
    "bin/run-quality-checks.sh",
)

#: A *call*, not a mention. The earlier form asked whether the name appeared
#: anywhere in the file, which ``bin/dk`` satisfies twice over without using the
#: result: it wraps the helper in a one-line function, so deleting the call site
#: left the now-dead definition keeping this green while ``dk style`` reverted to
#: package sources. Both spellings are calls — the sourced function, and the CLI
#: verb that ``bin/dk`` and the gate use because sourcing would impose ``-u`` and
#: ``-o pipefail`` on files that set only ``-e``. A definition is neither.
WORKSPACE_TARGETS_CALL = re.compile(r"\$\([^)]*workspace[_-]targets\b[^)]*\)")

#: Three quoting forms, because bash accepts all three and the guard below is the
#: only thing standing between the gate and a hardcoded target list. Matching the
#: double-quoted form alone let ``VALIDATE_ARGS=tests`` — valid, and exactly the
#: bug — pass unseen.
VALIDATE_ARGS_ASSIGNMENT = re.compile(
    r"""^\s*VALIDATE_ARGS=(?:"([^"]*)"|'([^']*)'|([^\s;&|#]*))""", re.MULTILINE
)


def _workspace_targets() -> list[str]:
    """The first-party code belonging to no package, from the one declaration.

    Executed rather than parsed. The declaration is four filesystem tests, so
    reading it as text would report what it says while the question is what it
    returns.
    """
    listing = subprocess.run(
        [str(ROOT / "bin" / "package-discovery.sh"), "workspace-targets"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return listing.split()


def _validate_targets_for(*args: str) -> list[str]:
    """What ``bin/validate.sh`` resolves as its target list, for these arguments.

    Asked rather than parsed, for the reason ``_workspace_targets`` is: the
    question is what the script checks, and reading the appends as text answers
    what it says. The earlier form read them out of the default branch and
    treated each as unconditional, so wrapping the package loop in a condition
    that skipped it left this reporting full coverage — the append was still
    textually present. ``--print-targets`` resolves the list through the real
    code path and prints it before any check runs.
    """
    listing = subprocess.run(
        [str(ROOT / "bin" / "validate.sh"), *args, "--print-targets"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return listing.split()


def _default_validate_targets() -> list[str]:
    """Every path ``bin/validate.sh`` validates when given no arguments."""
    return _validate_targets_for()


#: What the default target set must contain, stated here rather than derived.
#:
#: This is the one duplication in this file that is load-bearing, and it is the
#: answer to a specific hole. Every other assertion about coverage reads the
#: target set and asks whether it reaches the tracked files — so dropping a
#: directory from ``workspace_targets`` and naming it in
#: ``DEFERRED_FROM_DEFAULT_LINT`` satisfied all of them at once: the files are no
#: longer uncovered because they are now deferred, and the deferral is not
#: contradicted because nothing lints them any more. Both guards green, ``bin/``
#: unlinted again. A check derived from the thing it checks moves when that thing
#: moves; this does not, so removing a member fails here and no entry elsewhere
#: can quiet it.
REQUIRED_DEFAULT_TARGETS = frozenset({"tests", "bin", "src", "conftest.py"})


def _linted_by(path: PurePosixPath, targets: list[str]) -> bool:
    """Whether some default target is this file, or a directory containing it."""
    name = str(path)
    return any(name == target or name.startswith(f"{target}/") for target in targets)


def _deferred_by(path: PurePosixPath, patterns: set[str]) -> bool:
    """Whether any ancestor directory of this file matches a deferral pattern."""
    return any(
        PurePosixPath(*path.parts[:depth]).match(pattern)
        for depth in range(1, len(path.parts))
        for pattern in patterns
    )


def test_every_first_party_python_file_is_linted_by_default():
    """Being *run* is not the same as being *checked*, and neither is being *shipped*.

    ``bin/validate.sh`` builds its default target list by looping ``packages/*``
    and appending each ``src`` directory. Everything else was therefore outside
    every lint and type-check invocation in the repo, and nothing said so: the
    workspace guards asserting the toolchain is coherent, and ``bin/`` itself —
    the scripts that decide whether a pull request passes, including the two
    that enforce documentation mirroring and internal-label hygiene.

    The first version of this guard asserted that one directory was in the
    target set, keyed to its own location. That is the shape that let the same
    omission survive one directory over, so it reads the whole tracked set now
    and takes the exceptions as declared data.
    """
    targets = _default_validate_targets()
    deferred = set(DEFERRED_FROM_DEFAULT_LINT)

    uncovered = sorted(
        str(path)
        for path in _tracked_python()
        if not _linted_by(path, targets) and not _deferred_by(path, deferred)
    )
    assert not uncovered, (
        f"bin/validate.sh validates {targets} by default, which reaches none of "
        f"these tracked files:\n"
        + "\n".join(f"  - {name}" for name in uncovered)
        + "\nAdd the directory to the default targets, or record it in "
        "DEFERRED_FROM_DEFAULT_LINT with the size that makes deferring honest."
    )


def test_the_default_target_set_still_contains_what_it_must():
    """The deferral list must not be able to buy its way out of a lost target.

    The guard above compares the target set against the tracked files and takes
    the deferrals as declared data, which makes it complete about *accidents* and
    silent about one deliberate move: drop a directory from ``workspace_targets``,
    add it to ``DEFERRED_FROM_DEFAULT_LINT``, and coverage is gone with both
    checks still green. Replayed over the real repository before this was
    written, the escape also worked for ``packages/*/src`` — all ten package
    sources could leave the target set without a single assertion failing.

    So this states the required members instead of deriving them. That is the
    duplication ``workspace_targets`` exists to remove, and here it is the point:
    an assertion computed from the declaration it guards cannot notice the
    declaration shrinking.
    """
    targets = set(_default_validate_targets())

    missing_workspace = sorted(REQUIRED_DEFAULT_TARGETS - targets)
    assert not missing_workspace, (
        f"bin/validate.sh no longer validates {missing_workspace} by default. "
        "These are not deferrable: tests/ holds the guards that check the "
        "toolchain, and bin/ holds the checkers that decide whether a pull "
        "request passes. Restore the target — recording it in "
        "DEFERRED_FROM_DEFAULT_LINT is not the fix, it is the failure this "
        "guard exists to catch."
    )

    packages = sorted(path.parent.name for path in pyprojects() if path.parent != ROOT)
    missing_sources = [
        f"packages/{name}/src"
        for name in packages
        if f"packages/{name}/src" not in targets and (ROOT / "packages" / name / "src").is_dir()
    ]
    assert not missing_sources, (
        f"bin/validate.sh no longer validates {missing_sources} by default, so "
        "the shipped source of those packages is linted by nothing."
    )


def test_the_gate_asks_for_the_workspace_target_set_rather_than_restating_it():
    """A second copy of the target list is a second thing to forget.

    When no package changed, the gate validates the workspace half alone — and
    it named that half literally, as ``tests``, back when ``tests/`` was all of
    it. Adding a directory to ``bin/validate.sh`` therefore left a pull request
    touching only that directory validating something else entirely, which is
    the failure this file already records one layer up.

    The fix is a flag: validate.sh owns the list, the gate says which list it
    wants. This asserts the gate keeps asking rather than answering — anything
    that is neither a variable nor an option is a hardcoded target set.
    """
    gate = (ROOT / "bin" / "run-quality-checks.sh").read_text(encoding="utf-8")
    assignments = [
        next(group for group in match.groups() if group is not None)
        for match in VALIDATE_ARGS_ASSIGNMENT.finditer(gate)
    ]
    assert assignments, (
        "bin/run-quality-checks.sh no longer assigns VALIDATE_ARGS — if the "
        "variable was renamed, update this guard rather than deleting it"
    )

    literal = sorted(
        value
        for value in assignments
        if value and "$" not in value and not value.startswith("-")
    )
    assert not literal, (
        f"bin/run-quality-checks.sh passes {literal} to validate.sh as a literal "
        "target list, which stops tracking validate.sh's own the moment either "
        "changes. Pass --workspace, or a variable holding the packages."
    )

    # Every branch that validates anything has to ask for the workspace half.
    # Requesting it on the no-package branch alone is what shipped: narrowing to
    # the changed packages dropped this set, so a pull request touching a package
    # validated packages/*/src and nothing else. The ruff config is a global
    # trigger, so it marked all ten packages changed and took that branch — which
    # means the change that started linting bin/ recorded a passing validation
    # without linting bin/.
    silent = sorted(
        value for value in assignments if value and "--workspace" not in value
    )
    assert not silent, (
        f"bin/run-quality-checks.sh assigns VALIDATE_ARGS={silent} without "
        "--workspace, so that branch validates package sources alone and the "
        "code belonging to no package — bin/, tests/, src/, conftest.py — goes "
        "unchecked while the run reports a passing validation. --workspace is "
        "additive; it does not displace the packages beside it."
    )

    unread = sorted(
        name
        for name in WORKSPACE_TARGET_CONSUMERS
        if not WORKSPACE_TARGETS_CALL.search((ROOT / name).read_text(encoding="utf-8"))
    )
    assert not unread, (
        f"{unread} build a default set of things to check without calling "
        "workspace_targets, so each carries its own idea of which code belongs "
        "to no package. That is how bin/ ended up in none of them."
    )


#: Paths the print check must judge, and the answer it must give. The first three
#: are shipped library code — ``dataknobs_common.testing`` and its siblings are
#: the constructs the house rules point at instead of mocks, and ``ab_testing``
#: is about A/B tests, not tests — which a substring match on "test" reads as
#: test files and skips.
PRINT_CHECK_TEST_FILE_CASES = {
    "packages/common/src/dataknobs_common/testing/threads.py": False,
    "packages/bots/src/dataknobs_bots/testing.py": False,
    "packages/llm/src/dataknobs_llm/prompts/versioning/ab_testing.py": False,
    "tests/_workspace.py": False,
    "conftest.py": False,
    "tests/test_toolchain_consistency.py": True,
    "packages/data/tests/something_test.py": True,
}


def test_the_print_check_recognises_test_files_by_name_not_by_substring():
    """Skipping "anything with test in it" skipped eleven shipped modules.

    The check had two spellings of one question — ``*test*`` for a named file and
    ``*/test*`` for a directory walk — and both were wider than the question. The
    directory form matched every path under a ``testing/`` package, so the print
    check silently exempted ``dataknobs_common.testing`` and its siblings: shipped
    library code, in scope on paper for as long as the check has existed and
    examined not once.

    Exercised rather than read. The predicate is lifted out of the script and run,
    so this asserts what it decides rather than what it looks like — a guard that
    only checked the loose glob was absent would pass against any third spelling
    of the same mistake.
    """
    source = (ROOT / "bin" / "validate.sh").read_text(encoding="utf-8")
    function = re.search(r"^is_test_file\(\) \{.*?^\}", source, re.MULTILINE | re.DOTALL)
    assert function is not None, (
        "bin/validate.sh no longer defines is_test_file, so the print check has "
        "gone back to deciding what a test file is inline — which is how the two "
        "call sites came to disagree. Restore the shared predicate."
    )

    script = "\n".join(
        [function.group(0)]
        + [
            f'if is_test_file "{path}"; then echo "{path} yes"; else echo "{path} no"; fi'
            for path in PRINT_CHECK_TEST_FILE_CASES
        ]
    )
    output = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout
    verdicts = dict(line.rsplit(" ", 1) for line in output.splitlines())

    wrong = sorted(
        f"{path}: expected {'a test file' if expected else 'checked'}, got "
        f"{'a test file' if verdicts[path] == 'yes' else 'checked'}"
        for path, expected in PRINT_CHECK_TEST_FILE_CASES.items()
        if (verdicts[path] == "yes") != expected
    )
    assert not wrong, "bin/validate.sh's print check judges these wrongly:\n" + "\n".join(
        f"  - {item}" for item in wrong
    )

    # Wiring, not just the predicate: both branches have to route through it. The
    # file branch and the directory walk each used to carry their own glob.
    for spelling in ('!= *test*', '! -path "*/test*"'):
        assert spelling not in source, (
            f"bin/validate.sh matches test files with {spelling!r} again. That is "
            "the substring form this predicate replaced, and it exempts every "
            "shipped module under a testing/ package from the print check."
        )


def test_the_lint_deferrals_still_describe_the_repository():
    """A deferral list nobody rechecks is how the omission it records becomes permanent.

    Both directions are wrong and only one of them is obvious. An entry matching
    nothing is stale, and leaves the reader believing a gap exists that closed.
    An entry matching something already linted is worse: it is the cheapest way
    to silence the test above — drop a directory from the targets, name it here,
    and coverage is lost with every check still green.
    """
    tracked = _tracked_python()
    targets = _default_validate_targets()

    stale = sorted(
        pattern
        for pattern in DEFERRED_FROM_DEFAULT_LINT
        if not any(_deferred_by(path, {pattern}) for path in tracked)
    )
    assert not stale, (
        f"DEFERRED_FROM_DEFAULT_LINT records {stale}, which matches no tracked "
        "Python file. Drop the entry — a gap that no longer exists reads as one "
        "that does."
    )

    contradicted = sorted(
        pattern
        for pattern in DEFERRED_FROM_DEFAULT_LINT
        if any(
            _deferred_by(path, {pattern}) and _linted_by(path, targets) for path in tracked
        )
    )
    assert not contradicted, (
        f"DEFERRED_FROM_DEFAULT_LINT records {contradicted} as unlinted, but "
        "bin/validate.sh does lint files there. Either the entry is obsolete and "
        "should go, or a default target was removed and should come back."
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

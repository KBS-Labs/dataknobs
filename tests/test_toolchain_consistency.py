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

import ast
import configparser
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from tests._workspace import (
    ROOT,
    load_bin_module,
    load_toml,
    pyprojects,
    python_floor,
    tracked_files,
)
from tests._workspace import rel as _rel
from tests._workspace import workspace_targets as _workspace_targets
from tests._workspace import version_pair as _version_pair

#: How the gate asks for the guards in this directory. A named target on
#: ``bin/test.sh`` rather than a directory the gate resolves itself: the gate
#: used to call pytest here directly — ``test.sh`` took a package name and
#: scanned ``packages/*``, so it could not reach a directory belonging to no
#: package — which left one check the gate performed by a route no developer
#: command shared, and so one place a gate pass and a local pass could differ.
GATE_WORKSPACE_TARGET_RE = re.compile(r"\$TEST_CMD\s+workspace\b")

#: Directories excluded from the workspace run, read from the flag ``bin/test.sh``
#: actually passes. Frozen literals here would stop tracking it the first time a
#: second exclusion is added — the same drift these guards exist to catch.
RUNNER_IGNORE_RE = re.compile(r"--ignore=\$test_path/(\S+)")

#: The single declaration of which files outside packages/ affect a quality
#: result. Read rather than restated so this guard cannot drift from the change
#: detection and artifact hashing that consume the same list.
_scopes = load_bin_module("changed-packages")
WORKSPACE_QUALITY_INPUTS: dict[str, list[str]] = _scopes.WORKSPACE_QUALITY_INPUTS

_pyprojects = pyprojects
_load = load_toml


def _mypy_inis() -> list[Path]:
    """Every ``mypy.ini`` there is, which is currently none.

    The root one is retired: two configurations meant two answers to "is this
    clean", and mypy reads a ``mypy.ini`` in preference to ``pyproject.toml``
    whenever one is present. This stays as a forward guard rather than being
    deleted — a new one, here or in a package, would silently take precedence
    over the strict configuration the ceilings are measured under, and the
    readers below would then have something to check it against.
    """
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


def test_workspace_tests_are_reachable() -> None:
    """A guard nothing runs reports green in exactly the way a passing one does.

    Every test entry point is keyed by package — ``pytest.ini`` declares
    ``testpaths``, ``bin/test.sh`` takes a package name, and the quality gate
    loops ``packages/*``. This directory is in none of those by construction,
    so the guards here can go red and stay red without a single check turning
    red with them.

    Three mechanisms, because they cover different callers: ``testpaths``
    covers a bare ``pytest`` at the root, ``bin/test.sh workspace`` covers a
    developer, and the gate covers CI.

    The runner half is *asked*, not parsed. The gate used to name this
    directory itself and the guard read that string back; the string is gone
    now — ``test.sh`` owns the target — and reading its replacement out of
    ``test.sh`` would only move the same weakness one file over, since a target
    that resolves to the wrong directory spells itself exactly like one that
    resolves to the right one. Collecting through the real target answers the
    question the string was standing in for: does asking for ``workspace``
    reach *this file*.
    """
    here = Path(__file__).resolve().parent
    violations = []

    parser = configparser.ConfigParser()
    parser.read(ROOT / "pytest.ini")
    testpaths = parser.get("pytest", "testpaths", fallback="").split()
    if not any((ROOT / p).resolve() == here for p in testpaths):
        violations.append(
            f"pytest.ini: testpaths = {' '.join(testpaths)!r} does not cover {_rel(here)}"
        )

    collected = subprocess.run(
        [str(ROOT / "bin" / "test.sh"), "-n", "workspace", "--", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        # A non-zero exit is a finding here, not an error to raise on: the
        # assertion below reports what was collected and what the run returned.
        check=False,
    )
    if Path(__file__).name not in collected.stdout:
        violations.append(
            "bin/test.sh workspace collects nothing from "
            f"{_rel(here)} (exit {collected.returncode})"
        )

    # Reaching it is not running it, and the gate has three test paths: the
    # per-package split, the quick loop over named packages, and the quick loop
    # over none. The first two name the target; the third hands test.sh no
    # target at all and gets the guards from the discovery fallback, which is
    # what the test below covers. Dropping any of the three leaves a mode that
    # reports green without the guards, which is the failure this whole file
    # exists to make impossible.
    gate_text = (ROOT / "bin" / "run-quality-checks.sh").read_text()
    runs = len(GATE_WORKSPACE_TARGET_RE.findall(gate_text))
    if runs < 2:
        violations.append(
            "bin/run-quality-checks.sh: asks for the workspace target "
            f"{runs} time(s) — expected the PR-mode path and the dev-mode "
            "named-package path"
        )

    assert not violations, "Workspace guards are unreachable:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def _runner_targets(*args: str) -> list[str]:
    """What ``bin/test.sh`` would run, for these arguments.

    Asked rather than parsed, for the reason ``_validate_targets_for`` is:
    reading the discovery block as text answers what it says, and a condition
    wrapped around it later would leave that reading unchanged.
    """
    listing = subprocess.run(
        [str(ROOT / "bin" / "test.sh"), *args, "--print-targets"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return listing.split()


def test_running_everything_includes_the_guards() -> None:
    """A bare ``bin/test.sh`` is the command that reads as "run the suite".

    It discovers by looping ``packages/*``, so this directory — which belongs
    to no package — sat outside it, and the entry point named for running
    everything ran every test except the ones checking the toolchain that runs
    them. The gate's no-package path hands test.sh no target and depends on
    this, so it is the third reachability mechanism, not a convenience.

    The negative case is asserted too: these need no service and are not
    integration tests, so an integration-only run must not pick them up — else
    the fix is "always run them", which reports on a suite the caller did not
    ask for.
    """
    assert "workspace" in _runner_targets(), (
        "a bare bin/test.sh does not run the workspace guards, so the gate's "
        "no-package path runs none of them either"
    )
    assert "workspace" not in _runner_targets("-t", "integration")
    assert "workspace" not in _runner_targets("--only-integration")

    # Naming a package is naming a target, not a filter over the default set.
    assert _runner_targets("common") == ["common"]


def test_a_change_to_these_guards_still_schedules_them() -> None:
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
    # LICENSE rather than README.md: the root README used to be the inert file
    # here, and stopped being one when the documented-import guard started
    # reading it. A negative control has to name something that feeds no check
    # *today*, or it silently becomes an assertion that a real input is ignored.
    assert _scopes.plan_for_files(["LICENSE"])["test_scope"] == "none"
    assert _scopes.plan_for_files(["packages/common/src/x.py"])["test_scope"] == "packages"


def test_a_release_bump_schedules_no_package_suite() -> None:
    """The two mechanisms agree about what a version bump is.

    The hasher strips a package's own version line and the cross-package
    constraints bumped beside it, precisely so a release does not dirty a
    package. Change detection matched on paths and stripped nothing, so the
    same bump that moved not one stored hash still scheduled all ten suites —
    through the package paths, and again through ``uv.lock`` as a global
    trigger. Both halves are asserted because either alone still schedules
    everything.
    """
    # A bump rewrites the version and the sibling constraints. Neither is a
    # change to how the package behaves, and both must compare equal.
    before = b'[project]\nversion = "1.6.3"\ndeps = [\n"dataknobs-config>=0.4.4",\n]\n'
    after = b'[project]\nversion = "2.0.0"\ndeps = [\n"dataknobs-config>=0.5.0",\n]\n'
    assert _scopes.strip_release_noise(before) == _scopes.strip_release_noise(after), (
        "a version bump must not read as a content change, or every release "
        "re-runs every suite to publish a version string"
    )

    # The exemption has to stay narrow: an edit beside the version is a change.
    edited = after.replace(b"[project]", b"[project]\nrequires-python = '>=3.13'")
    assert _scopes.strip_release_noise(after) != _scopes.strip_release_noise(edited)

    # The other file every bump rewrites. It is in no hash scope, so mapping it
    # to its package scheduled that package's whole suite for a release note.
    plan = _scopes.plan_for_files(["packages/common/CHANGELOG.md"])
    assert plan["packages"] == [], f"a changelog belongs to no suite, got {plan['packages']}"
    assert plan["test_scope"] == "none"
    assert plan["docs_changed"] is True


def test_an_unhashed_test_input_still_schedules_its_package() -> None:
    """Change detection shares the hasher's definition, not its blind spot.

    The hasher decides *membership* as well — ``_HASH_PATTERNS`` reaches the
    ``.py`` files under ``src/`` and ``tests/`` and nothing else. Deferring to
    that wholesale would be the unsafe half of the unification: these files
    decide whether a suite passes while moving no stored hash, so a golden
    file regenerated wrongly would stop scheduling the suite that would have
    caught it. Over-scheduling here is the deliberate asymmetry.
    """
    unhashed_inputs = (
        "packages/llm/tests/golden/anthropic_profile_golden.json",
        "packages/config/tests/fixtures/test_config.yaml",
    )

    for path in unhashed_inputs:
        assert (ROOT / path).exists(), (
            f"{path} was this test's real instance of an unhashed test input; "
            "if it moved, re-anchor on another rather than deleting the case"
        )
        assert _scopes.plan_for_files([path])["test_scope"] == "packages", (
            f"{path} feeds a test result — it must still schedule its package"
        )


def test_a_file_can_feed_more_than_one_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file declared in two tiers contributes to both, rather than the first.

    The tiers are not a partition and the declarations never claimed to be
    one: a file can be a package's test input *and* something a workspace
    guard reads. The mapping read them as a partition anyway — each branch
    ended in ``continue``, so the first tier that matched was the only one
    that spoke, and the rest of the file's blast radius was dropped in
    silence.

    ``packages/common/tests/conftest.py`` is the case that is not
    hypothetical. It is the common suite's own conftest, and a workspace
    guard reads it — ``test_worked_input_fences`` compares a vocabulary
    published in a package document against the constant this file carries.
    Declaring it in the workspace tier so the guard is scheduled used to
    stop the common suite being scheduled by its own conftest.

    Asserted with the declaration patched rather than by filing the entry,
    because what is being pinned is the mapping's arithmetic: two tiers, two
    contributions. Which files are *declared* in which tier is a separate
    decision that this guard must keep legal rather than make.
    """
    dual = "packages/common/tests/conftest.py"
    assert (ROOT / dual).exists(), (
        f"{dual} was this test's real instance of a file two tiers can claim; "
        "if it moved, re-anchor on another rather than deleting the case"
    )

    # Both halves run against a *patched* tier list, including the half that
    # leaves the entry out. Reading the real declaration for the baseline would
    # make this guard's control case depend on a decision it exists to keep
    # legal: file the entry for real and the baseline stops holding, so the
    # guard fails on the change it was written to permit.
    tiers = [t for t in _scopes.WORKSPACE_ONLY_TRIGGERS if t != dual]

    # Undeclared, it is exactly what it looks like: an input to one package.
    # Asserted on directly_changed rather than packages, which also carries
    # common's transitive dependents — nine more names that say nothing about
    # what this file was mapped to.
    monkeypatch.setattr(_scopes, "WORKSPACE_ONLY_TRIGGERS", tiers)
    before = _scopes.plan_for_files([dual])
    assert before["directly_changed"] == ["common"]
    assert before["workspace_changed"] is False

    monkeypatch.setattr(_scopes, "WORKSPACE_ONLY_TRIGGERS", [*tiers, dual])

    both = _scopes.plan_for_files([dual])
    assert both["workspace_changed"] is True, (
        "a file declared in the workspace tier must schedule the guards that "
        "read it — that is what the declaration is for"
    )
    assert both["directly_changed"] == ["common"], (
        "declaring a file in a second tier must not un-declare the first: "
        f"{dual} is the common suite's own conftest and still feeds it, "
        f"got directly_changed={both['directly_changed']}"
    )
    assert both["test_scope"] == "packages"


def test_a_test_only_change_stops_at_its_own_package() -> None:
    """A package's own tests decide that package's result and no other's.

    The transitive closure earns its cost on a change to a package's
    *exported* surface: edit ``common``'s source and the other nine packages
    are running against different code, so their suites have to re-run to say
    anything. A change under ``packages/common/tests/`` is not that. Nothing
    outside the common suite reads those files, and nothing outside it can:
    the gate runs each package's suite as its own pytest invocation, so no
    other package's run collects them at all. What holds that true rather
    than assuming it is ``test_no_package_suite_reads_another_packages_tests``
    below, which is this test's other half.

    Scheduled as an export anyway, ``common`` is both the worst case and the
    ordinary one — nine dependents — so a one-line edit to a common test ran
    every suite in the workspace to re-confirm nine results that could not
    have moved.

    ``pyproject.toml`` is the control in the other direction, and the reason
    this is about ``tests/`` rather than about "not source": a dependency
    constraint or a version is read by every dependent's resolution, so it
    exports and must keep dragging the closure.
    """
    local_only = "packages/common/tests/test_registry.py"
    exports_source = "packages/common/src/dataknobs_common/__init__.py"
    exports_metadata = "packages/common/pyproject.toml"

    for path in (local_only, exports_source, exports_metadata):
        assert (ROOT / path).exists(), (
            f"{path} anchors this guard in a real file; if it moved, "
            "re-anchor on another rather than deleting the case"
        )

    local = _scopes.plan_for_files([local_only])
    assert local["packages"] == ["common"], (
        "a change to a package's own tests is read by that package's suite "
        "and by nothing else, so it schedules that suite and nothing else; "
        f"got {local['packages']}"
    )
    assert local["directly_changed"] == ["common"]
    assert local["exporting"] == [], (
        "a test file is not part of what a package exports to its dependents"
    )
    assert local["test_scope"] == "packages"

    # Both halves of the control, because the closure is what is being
    # narrowed and a narrowing that took everything with it would pass the
    # assertion above. common has nine dependents; each of these must still
    # reach all ten.
    for path in (exports_source, exports_metadata):
        exporting = _scopes.plan_for_files([path])
        assert exporting["packages"] == sorted(_scopes.ALL_PACKAGES), (
            f"{path} changes what common's dependents build and run against, "
            f"so it must still schedule them; got {exporting['packages']}"
        )
        assert exporting["exporting"] == ["common"]


def test_no_package_suite_reads_another_packages_tests() -> None:
    """The premise the narrowing above rests on, checked against the tree.

    "A package's tests are local to that package" is true of this tree and is
    not true by construction: ``packages/*/tests`` directories go on
    ``sys.path`` under pytest's default import mode, so a bare
    ``from _vocabularies import ...`` resolves to whichever tests directory
    was inserted. Two packages sharing a helper that way would be a real
    coupling, invisible in the imports (nothing spells the other package's
    name) and silently un-scheduled by the change above.

    Measured when this was written: **zero** cross-package test imports, out
    of 995 test modules offering 935 bare-importable names.

    The count below is the positive control, and this guard is worth nothing
    without it. A reader that resolved no imports at all — a parse that threw,
    a name table built wrong — reports zero violations for the same reason a
    correct one does. Asserting that same-package bare imports are still
    *found* is what tells the two apart.
    """
    tests_dirs = {
        path.relative_to(ROOT).parts[1]: path
        for path in sorted(ROOT.glob("packages/*/tests"))
        if path.is_dir()
    }
    assert len(tests_dirs) > 1, "nothing to compare across"

    # Which packages' tests offer each bare-importable top-level name. A file
    # is importable as its own stem, and a subdirectory as its own name, once
    # the tests directory holding it is on sys.path.
    offered: dict[str, set[str]] = {}
    sources: list[tuple[str, Path]] = []
    for package, tests_dir in tests_dirs.items():
        for source in sorted(tests_dir.rglob("*.py")):
            sources.append((package, source))
            relative = source.relative_to(tests_dir)
            if relative.name != "__init__.py":
                offered.setdefault(source.stem, set()).add(package)
            if len(relative.parts) > 1:
                offered.setdefault(relative.parts[0], set()).add(package)

    foreign: list[str] = []
    local_hits = 0
    for package, source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                # A relative import cannot leave the package it is written in,
                # so only absolute ones can reach another package's tests.
                imported = [node.module.split(".")[0]]
            else:
                continue
            for name in imported:
                owners = offered.get(name)
                if owners is None:
                    continue
                if package in owners:
                    local_hits += 1
                else:
                    foreign.append(
                        f"{_rel(source)}:{node.lineno} imports {name!r}, which "
                        f"only {sorted(owners)} provide"
                    )

    assert local_hits, (
        "this guard resolved no bare test-module import anywhere in the "
        "workspace, which is what a broken reader and a clean tree both look "
        "like. There are such imports — _vocabularies, _anthropic_stubs — so "
        "finding none means the name table or the parse is wrong, not that "
        "nothing is coupled"
    )
    assert not foreign, (
        "a package's tests are scheduled by a change to that package alone "
        "(see test_a_test_only_change_stops_at_its_own_package), so a test "
        "reaching into another package's tests directory is a coupling the "
        "gate would stop scheduling:\n  " + "\n  ".join(foreign)
    )


def test_a_lock_change_maps_to_the_members_whose_resolution_moved() -> None:
    """``uv.lock`` says which packages it moved; the path alone does not.

    The entry is in the global tier because a resolution change really can
    reach every package -- a bumped third-party version is installed for all
    ten, and nothing in the path says which. But the file is not opaque. Each
    workspace member is its own ``[[package]]`` block, marked
    ``source = { editable = "packages/<name>" }``, so a diff confined to those
    blocks names the packages it moved and stops there.

    That is the common shape rather than a corner: measured over 150 merged
    pull requests, ``uv.lock`` was the single most frequent global trigger at
    12, and 7 of those 12 touched workspace-member blocks only -- release
    version bumps, and one package gaining a dependency.

    The closure still runs from whatever moved, which is what keeps a real
    dependency change reaching the packages that build against it.
    """
    moved = _scopes.plan_for_files(
        ["uv.lock"],
        {"uv.lock": _scopes.TriggerScan(steps=_scopes.BOTH_STEPS, packages=frozenset({"llm"}))},
    )
    assert moved["mode"] == "changed", (
        "a lock diff that named the members it moved has a blast radius the "
        f"file itself reported; got mode={moved['mode']}"
    )
    assert moved["packages"] == sorted(
        _scopes.get_transitive_dependents({"llm"}) & set(_scopes.DEPENDENCIES)
    ), (
        "a resolution change to one member reaches that member and the "
        f"packages that build against it; got {moved['packages']}"
    )
    assert moved["exporting"] == ["llm"], (
        "a member's resolution is exactly what its dependents build against, "
        "so a lock change to it exports"
    )


def test_an_unreadable_lock_change_keeps_the_global_blast_radius() -> None:
    """The fail-closed half: no scan means the old answer, not a smaller one.

    A lock diff that moved a third-party block, a lock the base ref does not
    carry, a parse that failed -- each is a case where the file did not say
    which packages it moved, and the only safe reading of that silence is the
    one the path has always carried. A narrowing that treated "I could not
    tell" as "nothing" would schedule fewer suites for exactly the changes
    that most need them.
    """
    # None is "nothing read the file"; the empty set is "a reader ran and
    # placed nothing", which is the shape a silently-broken parser produces for
    # every input. Both must keep the full blast radius.
    unattributed: tuple[frozenset[str] | None, ...] = (None, frozenset())
    for members in unattributed:
        scans = (
            {}
            if members is None
            else {"uv.lock": _scopes.TriggerScan(steps=_scopes.BOTH_STEPS, packages=members)}
        )
        plan = _scopes.plan_for_files(["uv.lock"], scans)
        assert plan["mode"] == "all", (
            f"a lock scan of {members!r} names no member, so nothing "
            f"narrowed the path's own blast radius; got mode={plan['mode']}"
        )
        assert plan["packages"] == sorted(_scopes.ALL_PACKAGES), plan["packages"]

    assert _scopes.plan_for_files(["uv.lock"])["packages"] == sorted(_scopes.ALL_PACKAGES), (
        "the default must be the global answer: a caller that does not supply "
        "a scan has not narrowed anything"
    )


def test_every_package_is_attributable_in_the_real_lock() -> None:
    """The premise the narrowing rests on, checked against the tree.

    ``lock_members_changed`` narrows a global trigger by finding workspace
    members inside ``uv.lock``. A reader that found *none* would return the
    empty set for every diff -- which the caller treats as unattributed, so
    the failure is safe, but it is also silent: the narrowing would simply
    stop paying while every test above still passed, because each supplies
    its own scan rather than reading the file.

    So this reads the real lock and asserts the marker accounts for every
    package, which is what makes a uv format change fail here rather than
    quietly switch the optimisation off.
    """
    lock = (ROOT / "uv.lock").read_bytes()
    text = lock.decode("utf-8", errors="surrogateescape")
    attributed = set(_scopes._LOCK_MEMBER_RE.findall(text))

    assert attributed == set(_scopes.ALL_PACKAGES), (
        "every workspace package is an editable block in uv.lock, and the "
        "narrowing can only attribute a change to a package it can find. "
        f"missing from the lock: {sorted(set(_scopes.ALL_PACKAGES) - attributed)}; "
        f"found but not a package: {sorted(attributed - set(_scopes.ALL_PACKAGES))}"
    )

    # The workspace root is an editable member too and owns no package's code,
    # so it must NOT be attributed -- a pattern loose enough to catch it would
    # map a root-only change onto some package.
    assert 'editable = "." }' in text, (
        "the workspace root block anchors the case below; if uv stopped "
        "emitting it, re-anchor rather than dropping the assertion"
    )
    assert _scopes._LOCK_MEMBER_RE.search('source = { editable = "." }\n') is None


def test_the_lock_reader_resolves_a_real_edit_and_refuses_what_it_cannot_place() -> None:
    """Both directions, against the real file rather than a fixture.

    The positive half is the one that matters: a reader that resolved nothing
    reports "no member moved" for the same reason a correct one reports it of
    an unchanged file, and this module's history is full of guards that
    reported green because they read nothing.
    """
    lock = (ROOT / "uv.lock").read_bytes()
    assert _scopes.lock_members_changed(lock, lock) == frozenset(), (
        "a file compared with itself moved no member"
    )

    marker = b'source = { editable = "packages/llm" }'
    assert lock.count(marker) == 1, "the llm block anchors this case"
    edited = lock.replace(marker, marker + b"\nbuild-constraint-dependencies = []")
    assert edited != lock
    assert _scopes.lock_members_changed(lock, edited) == frozenset({"llm"}), (
        "an edit inside one member's block is attributable to that member, "
        f"got {_scopes.lock_members_changed(lock, edited)}"
    )

    # A third-party block is what the workspace cannot account for: the bumped
    # version is installed for every package and the lock does not say which
    # of them care.
    third_party = b'\nname = "jinja2"\nversion = '
    assert lock.count(third_party) == 1, (
        "the jinja2 block anchors this case; spelled with its surrounding "
        "newlines because the bare name also appears in every dependency list"
    )
    bumped = lock.replace(third_party, b'\nname = "jinja2"\nx-probe = 1\nversion = ')
    assert _scopes.lock_members_changed(lock, bumped) is None, (
        "a change this reader cannot place must keep the global blast radius"
    )

    # And the preamble, which is every package's resolution at once.
    repython = b'requires-python = ">=3.12"'
    assert lock.count(repython) == 1
    assert (
        _scopes.lock_members_changed(lock, lock.replace(repython, b'requires-python = ">=3.13"'))
        is None
    )


def test_a_resolution_marker_reorder_is_not_a_change() -> None:
    """Uv reorders that list on its own; the set it denotes is what matters.

    Left literal, the same release-helper run behaved differently depending on
    whether uv happened to reorder -- measured across the release pull requests
    in the last 150, three were reordered and two were not, and only the
    unreordered two could ever narrow. One PR class, one answer.
    """
    lock = (ROOT / "uv.lock").read_bytes()
    text = lock.decode("utf-8", errors="surrogateescape")
    match = _scopes._LOCK_MARKERS_RE.search(text)
    assert match is not None, (
        "uv.lock carries a resolution-markers list; if the format changed, "
        "this canonicalisation needs re-reading rather than deleting"
    )
    entries = [line for line in match.group(2).splitlines() if line.strip()]
    assert len(entries) > 1, "a one-entry list cannot be reordered"

    shuffled = text.replace(
        match.group(0),
        match.group(1) + "".join(f"{line}\n" for line in reversed(entries)) + match.group(3),
    )
    assert shuffled != text, "the reorder must actually change the bytes"
    reordered = shuffled.encode("utf-8", errors="surrogateescape")

    assert _scopes.lock_members_changed(lock, reordered) == frozenset(), (
        "reordering a set changes nothing about the resolution it denotes"
    )

    # The control in the other direction: a marker whose *text* changed is a
    # real resolution change and must still be unattributable.
    # Altered mechanically rather than by editing a version inside it: every
    # entry is a quoted string, and nothing here should depend on which
    # markers uv happens to emit today.
    altered = text.replace(entries[0], entries[0].replace('"', '"x-probe and ', 1), 1)
    assert altered != text, "the control must actually alter a marker"
    assert (
        _scopes.lock_members_changed(lock, altered.encode("utf-8", errors="surrogateescape"))
        is None
    ), "a changed marker is a changed resolution, not a reordering"


def test_a_lint_only_input_does_not_schedule_package_test_suites() -> None:
    """``bin/validate.sh`` is the lint step, so it moves no package's test result.

    The entry is global for a reason ``09e1dbc5`` argued and demonstrated: the
    script *is* a step, so every package's recorded result for that step is
    whatever it produced, and the branch that introduced it fixed a runner
    exiting 0 on its second target. That argument is about the step the script
    runs. ``bin/test.sh`` produced the test results; ``bin/validate.sh``
    produced the validation ones, and neither can move the other's.

    So the blast radius keeps its full width on the side the script owns --
    ``packages`` is still all ten, and the validation step still re-runs over
    every one of them -- while the ten package test suites it cannot have
    moved stop being scheduled.
    """
    lint_only = sorted(
        name
        for name, steps in _scopes.GLOBAL_TRIGGER_STEPS.items()
        if steps == frozenset({_scopes.LINT_STEP})
    )
    # Driven off the table rather than naming one file, so every entry
    # classified lint-only is pinned by this. Reclassifying one back to both
    # steps -- the silent way to lose the narrowing -- then fails here instead
    # of passing because the guard happened to probe a different entry.
    assert lint_only == ["bin/package-discovery.sh", "bin/validate.sh"], (
        f"the lint-only classification changed to {lint_only}; that is a "
        f"scheduling decision, so it belongs in a diff that argues it"
    )

    for path in lint_only:
        plan = _scopes.plan_for_files([path])

        assert plan["packages"] == sorted(_scopes.ALL_PACKAGES), (
            f"the lint step's own blast radius is undiminished: a change to "
            f"{path} makes every package's recorded validation result stale, "
            f"exactly as before; got {plan['packages']}"
        )
        # Read with a fallback rather than by key, so this fails against a tree
        # without the capability by naming the packages it would have tested --
        # the defect -- rather than by raising KeyError, which only says "absent".
        assert plan.get("test_packages", plan["packages"]) == [], (
            f"{path} feeds the lint step; no package's *test* result can have "
            f"moved, so no package suite is scheduled. Got "
            f"{plan.get('test_packages', plan['packages'])}"
        )


def test_every_global_trigger_declares_which_step_it_moves() -> None:
    """The classification is exhaustive, so a new global input forces the call.

    ``09e1dbc5`` scoped its coverage guard for this reason exactly -- to make a
    new workflow step "force the tiering decision in review, where the blast
    radius is already being considered, instead of leaving it to whoever edits
    the script next". A global input added without a step is the same omission
    one layer in: the mapper falls back to both steps, which is safe and
    silent, and silent is what makes it permanent.
    """
    declared = set(_scopes.GLOBAL_TRIGGER_STEPS)
    triggers = set(_scopes.GLOBAL_TRIGGERS)

    assert declared == triggers, (
        "every global input declares the recorded step it can move:\n"
        f"  undeclared: {sorted(triggers - declared)}\n"
        f"  declared but not a trigger: {sorted(declared - triggers)}"
    )
    for name, steps in _scopes.GLOBAL_TRIGGER_STEPS.items():
        assert steps and steps <= _scopes.BOTH_STEPS, (
            f"{name} declares {sorted(steps)}; a global input moves at least "
            f"one of {sorted(_scopes.BOTH_STEPS)} or it is not global"
        )


def test_a_global_hash_scope_groups_inputs_that_move_the_same_step() -> None:
    """A scope's members agree about what they move, or its digest cannot say.

    ``validate_artifacts`` widens the dirty set to every package when a global
    scope moves, under a comment claiming such a scope "changes lint, type, or
    test results everywhere". That held while the global inputs were
    undifferentiated. ``GLOBAL_TRIGGER_STEPS`` then classified three of them
    test-only and two lint-only, and one digest was left standing for three
    different claims -- so a moved ``toolchain`` hash could report only that
    *something* global had changed, and the sentence above the widening became
    false for the lint-only pair.

    Partitioning by the declared step gives the name its meaning back: a moved
    ``toolchain_lint`` says every package's validation row is stale and its
    test rows are not, in the one field a reader of the artifact -- or of the
    gate's end-of-run re-check -- actually sees.

    The dirty set is unchanged and deliberately so. A lint-only input does move
    every package's recorded validation result, so all ten are still named;
    what the step split made wrong was the declaration, not the width.
    """
    for scope in sorted(_scopes.GLOBAL_SCOPES):
        entries = WORKSPACE_QUALITY_INPUTS[scope]
        assert entries, (
            f"the {scope} scope is global and declares no inputs, so nothing "
            f"it is supposed to catch can move its digest"
        )

        by_entry = {entry: _scopes.GLOBAL_TRIGGER_STEPS[entry] for entry in entries}
        assert len(set(by_entry.values())) == 1, (
            f"the {scope} scope holds inputs that move different recorded "
            f"steps, so a change to its digest cannot say which result went "
            f"stale: " + ", ".join(f"{name} -> {sorted(steps)}" for name, steps in by_entry.items())
        )


def test_the_global_hash_scopes_partition_the_global_triggers() -> None:
    """Splitting the scope must not drop an input out of the global tier.

    The widening in ``validate_artifacts`` is keyed on the scope rather than on
    the trigger list, so an entry that falls out of every global scope is still
    scheduled by change detection while its digest stops dirtying a single
    package. That is the same two-readers-one-question shape the split is
    fixing, reintroduced by the fix and pointing the unsafe way.

    Disjointness is asserted within the global tier for the opposite reason:
    one entry hashed into two *global* scopes moves both digests, which reads
    as two independent facts about one edit.

    Across tiers it is not a defect and is not asserted. ``bin/validate.sh`` is
    hashed in ``toolchain_lint`` and again under the ``bin/`` entry of the
    workspace-only tier, and both are true of it -- it is the lint step, and it
    is also a file the guards under ``tests/`` read. Two tiers saying so is the
    accumulation ``map_files_to_packages`` was fixed to allow, not a double
    count.
    """
    seen: dict[str, str] = {}
    for scope in sorted(_scopes.GLOBAL_SCOPES):
        for entry in WORKSPACE_QUALITY_INPUTS[scope]:
            assert entry not in seen, (
                f"{entry} is hashed in both {seen[entry]} and {scope}, so one "
                f"edit moves two global digests"
            )
            seen[entry] = scope

    triggers = set(_scopes.GLOBAL_TRIGGERS)
    assert sorted(seen) == sorted(triggers), (
        "the global hash scopes and the global trigger list disagree:\n"
        f"  scheduled, hashed in no global scope: {sorted(triggers - set(seen))}\n"
        f"  hashed globally, not a trigger: {sorted(set(seen) - triggers)}"
    )


def test_the_test_step_does_not_read_the_lint_only_inputs() -> None:
    """The premise under the split, checked against the script rather than assumed.

    ``bin/validate.sh`` and ``bin/package-discovery.sh`` are classified
    lint-only because the test step cannot reach them: ``bin/test.sh`` has its
    own ``discover_test_packages`` loop over ``packages/*`` and neither sources
    the discovery helper nor invokes the validator. That is a property of the
    file today, not a law -- one ``source`` line would make every lint-only
    classification wrong and every narrowed schedule unsound, with nothing else
    in the tree reporting it.

    Comments are stripped first, because ``bin/test.sh`` mentions
    ``bin/validate.sh`` twice in prose -- it mirrors its argument parsing -- and
    a guard that could not tell a mention from a use would have to be weakened
    to a substring nobody trusts.
    """
    source = (ROOT / "bin" / "test.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))

    lint_only = sorted(
        name
        for name, steps in _scopes.GLOBAL_TRIGGER_STEPS.items()
        if steps == frozenset({_scopes.LINT_STEP})
    )
    assert lint_only, "the classification declares no lint-only input to check"

    for path in lint_only:
        basename = path.rsplit("/", 1)[-1]
        assert basename not in code, (
            f"bin/test.sh reaches {basename}, so it is an input to the test "
            f"step and cannot be classified lint-only. Either the reference is "
            f"new -- in which case {path} belongs in BOTH_STEPS and every "
            f"narrowed schedule since is unsound -- or it is a comment this "
            f"strip did not catch."
        )

    # The control: a reader that stripped the whole file, or looked in the
    # wrong one, finds nothing for the same reason a correct one does. test.sh
    # does discover packages -- just not through the helper validate.sh uses.
    assert "discover_test_packages" in code, (
        "bin/test.sh discovers its own packages; finding no trace of that "
        "means this guard is reading the wrong file or stripping too much"
    )


def test_no_package_suite_reads_the_root_pyproject() -> None:
    """A package's tests read their own manifest, never the root one.

    The root ``pyproject.toml`` carries ``[tool.ruff]`` and ``[tool.mypy]``,
    and a diff confined to those schedules no package test suite. That is only
    true while no package suite *reads* them -- a guard asserting a ruff
    setting from inside ``packages/*/tests`` would be a test whose result the
    root manifest decides, and it would stop being scheduled.

    Checked by how far up each reader reaches rather than by the string, since
    two package suites legitimately read a manifest: from
    ``packages/<pkg>/tests/x.py``, ``parents[1]`` is the package and
    ``parents[2]`` is ``packages/``. Anything higher has left the package.
    """
    readers: dict[str, int] = {}
    for source in sorted(ROOT.glob("packages/*/tests/**/*.py")):
        text = source.read_text(encoding="utf-8")
        if "pyproject.toml" not in text:
            continue
        depths = [int(m) for m in re.findall(r"parents\[(\d+)\]", text)]
        readers[str(source.relative_to(ROOT))] = max(depths, default=0)

    # Positive control: there IS such a reader, so an empty result means the
    # glob or the pattern is wrong rather than that nothing reads a manifest.
    assert readers, (
        "no package test mentions pyproject.toml at all -- there is at least "
        "one (test_record_core_independence.py reads its own), so this guard "
        "is looking in the wrong place"
    )

    escaped = {path: depth for path, depth in readers.items() if depth > 2}
    assert not escaped, (
        "a package suite reaches above packages/ while reading a "
        "pyproject.toml, so the root manifest may decide its result and a "
        "lint-table change would stop scheduling it:\n  "
        + "\n  ".join(f"{path}: parents[{depth}]" for path, depth in sorted(escaped.items()))
    )


def test_a_pyproject_change_outside_the_linter_tables_stays_global() -> None:
    """The fail-closed half of the pyproject reader, against the real file.

    Three silences that must all read as "the path keeps its blast radius": a
    table this does not recognise as a linter's, the preamble, and -- the one
    that matters most -- a diff in which nothing moved at all. A reader that
    silently matched nothing would report "no table moved" for every input,
    and that must never become "no step moved", which is an empty schedule.
    """
    original = (ROOT / "pyproject.toml").read_bytes()
    text = original.decode("utf-8")

    assert _scopes.pyproject_steps_changed(original, original) is None, (
        "an unchanged file moved no table, which is exactly what a reader "
        "that matched nothing also reports -- so it must read as unattributed"
    )

    # A dependency change: both steps, every package, as the path has always said.
    marker = "[dependency-groups]"
    assert marker in text, "the real manifest no longer carries the table this probes"
    widened = text.replace(marker, marker + "\n# x-probe\n", 1)
    assert _scopes.pyproject_steps_changed(original, widened.encode("utf-8")) is None, (
        "a change outside the linter tables keeps the global blast radius"
    )


def test_the_pyproject_reader_resolves_a_real_edit_and_places_it_on_the_lint_step() -> None:
    """The positive half: an edit inside [tool.ruff] narrows, and to lint only.

    Without this, the guard above passes just as well against a reader that
    returns ``None`` unconditionally -- which is a reader that has quietly
    switched the optimisation off while every fail-closed test still agrees
    with it.
    """
    original = (ROOT / "pyproject.toml").read_bytes()
    text = original.decode("utf-8")

    for table in sorted(_scopes._LINT_ONLY_TABLES):
        header = f"[{table}]"
        assert header in text, (
            f"the real manifest no longer carries {header}; the reader is "
            f"declared to narrow on it and nothing would exercise that"
        )
        edited = text.replace(header, header + "\n# x-probe\n", 1)
        scan = _scopes.pyproject_steps_changed(original, edited.encode("utf-8"))

        assert scan is not None, f"an edit confined to {header} is attributable"
        assert scan.steps == frozenset({_scopes.LINT_STEP}), (
            f"{header} decides what the validation step reports and nothing a "
            f"test can observe; got {sorted(scan.steps)}"
        )
        assert scan.packages is None, (
            "a rule change is re-validated across every package -- the reader "
            "narrows the step, not the package set"
        )

    # A *subtable*, which is what the measured cases actually edit: eight of
    # the nine lint-only pull requests in the window touched [tool.ruff.lint]
    # or deeper, never the bare [tool.ruff] header. Keying a subtable on its
    # own header instead of its tool leaves every one of them unattributable,
    # which fails closed -- so the optimisation switches itself off and every
    # other test in this file still agrees with it.
    subtable = "[tool.ruff.lint.per-file-ignores]"
    assert subtable in text, (
        f"the real manifest no longer carries {subtable}; the grouping this "
        f"probes is what makes a per-file-ignore edit read as the ruff change "
        f"it is"
    )
    edited = text.replace(subtable, subtable + "\n# x-probe\n", 1)
    scan = _scopes.pyproject_steps_changed(original, edited.encode("utf-8"))
    assert scan is not None and scan.steps == frozenset({_scopes.LINT_STEP}), (
        "a tool's configuration is one document however deeply it is spelled; "
        f"an edit inside {subtable} is a ruff change, not a table of its own"
    )

    # And the plan that answer produces: lint everything, test nothing.
    plan = _scopes.plan_for_files(
        ["pyproject.toml"],
        {
            "pyproject.toml": _scopes.TriggerScan(
                steps=frozenset({_scopes.LINT_STEP}), packages=None
            )
        },
    )
    assert plan["packages"] == sorted(_scopes.ALL_PACKAGES), plan["packages"]
    assert plan["test_packages"] == [], plan["test_packages"]


def test_the_test_list_never_exceeds_the_package_list() -> None:
    """``test_packages`` is a subset of ``packages``, on every shape.

    The compatibility claim in one assertion: a consumer that knows only
    ``packages`` schedules a superset of what the test stage runs, so reading
    the old key alone can over-test and can never skip a suite the new key
    would have run. Spelled the other way round, that consumer under-tests in
    silence.
    """
    shapes = [
        ["bin/validate.sh"],
        ["bin/test.sh"],
        ["pyproject.toml"],
        ["uv.lock"],
        ["packages/common/src/x.py"],
        ["packages/common/tests/test_x.py"],
        ["tests/test_toolchain_consistency.py"],
        ["bin/validate.sh", "packages/llm/src/x.py"],
        ["LICENSE"],
        [],
    ]
    for files in shapes:
        plan = _scopes.plan_for_files(files)
        assert set(plan["test_packages"]) <= set(plan["packages"]), (
            f"{files}: test_packages {plan['test_packages']} is not a subset "
            f"of packages {plan['packages']}"
        )


def _documents_a_package_suite_reads() -> dict[str, str]:
    """Every package document read by a test in that package's own suite.

    Found rather than listed, and found structurally: a ``Path(__file__)``
    expression divided by ``"docs"``. The naive search — a ``"docs"`` string
    literal anywhere under ``packages/*/tests`` — returns 192 hits here, almost
    all of them a knowledge source or a RAG adapter that happens to be named
    ``docs``, so a guard built on it would be re-tuned rather than read.

    Returns ``{document path: the package whose suite reads it}``.
    """
    found: dict[str, str] = {}
    for source in sorted(ROOT.glob("packages/*/tests/**/*.py")):
        text = source.read_text(encoding="utf-8")
        if "__file__" not in text:
            continue
        package = source.relative_to(ROOT).parts[1]
        tree = ast.parse(text)

        # Only the outermost link of a `a / "b" / "c"` chain. Every prefix of
        # one is itself a division node, so walking them all reconstructs
        # `packages/bots/docs` alongside the document beneath it — a directory
        # that is not a reader, reported as a reader whose file is missing.
        inner = {
            id(node.left)
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        }

        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                continue
            if id(node) in inner:
                continue
            segment = ast.get_source_segment(text, node) or ""
            if "__file__" not in segment:
                continue
            parts = _divided_constants(node)
            if "docs" not in parts:
                continue
            # Everything from "docs" onward names the document; whatever
            # precedes it is the expression that walked up to the package
            # root, and the assertion below is what checks it arrived there.
            tail = parts[parts.index("docs") :]
            document = f"packages/{package}/{'/'.join(tail)}"
            assert (ROOT / document).is_file(), (
                f"{_rel(source)}:{node.lineno} reads a package document this "
                f"guard reconstructs as {document}, which does not exist. "
                "Either the document is missing — in which case the test that "
                "reads it is failing — or it is reached by an expression the "
                "reconstruction above does not model, and dropping it silently "
                "is how a reader stops being declared while still deciding a "
                "suite's result."
            )
            found[document] = package
    return found


def _divided_constants(node: ast.BinOp) -> list[str]:
    """The string constants of a ``a / "b" / "c"`` chain, in order."""
    parts: list[str] = []
    stack: list[ast.expr] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.BinOp) and isinstance(current.op, ast.Div):
            stack.extend([current.right, current.left])
        elif isinstance(current, ast.Constant) and isinstance(current.value, str):
            parts.append(current.value)
    return parts


def test_every_package_document_a_package_suite_reads_is_declared() -> None:
    """The list that decides scheduling is checked against the tree, not trusted.

    A package document belongs to no package's suite by default: 141 of the
    148 here are read only by the workspace guards, and scheduling their
    package for one is what ran two full suites for a link repair. The other
    seven are read by a test *in* that package, so they do decide whether it
    passes, and they have to keep scheduling and dirtying it.

    Which of the two a document is cannot be inferred from its path, so it is
    declared. This is the guard that stops the declaration from being a list
    somebody remembered to update: a test reading an undeclared document fails
    here on arrival, naming itself, rather than being scheduled by nothing
    until it goes stale.
    """
    declared = _scopes.PACKAGE_TEST_DOC_INPUTS
    found = _documents_a_package_suite_reads()

    assert found == declared, (
        "the declaration and the tree disagree about which package documents "
        "feed a package suite.\n"
        f"  read but undeclared: {sorted(set(found) - set(declared))}\n"
        f"  declared but unread: {sorted(set(declared) - set(found))}\n"
        "An undeclared one is scheduled by nothing and hashed by nothing, so "
        "editing it can neither run the test that reads it nor invalidate that "
        "test's recorded verdict."
    )
    assert found, (
        "no package suite reads a package document, so this guard is now "
        "asserting that two empty sets agree. If that is genuinely the end "
        "state, delete the declaration and the branch that reads it rather "
        "than leaving a guard that cannot distinguish anything."
    )


def test_a_documentation_change_schedules_the_guards_that_read_it() -> None:
    """Documentation feeds the workspace guards, so it must schedule them.

    Four guards under ``tests/`` read every document in the repository — 370
    of them, the site tree and the package tree alike — and check the imports,
    the configuration keys, the tool catalogue and the fenced samples they
    contain against the code. None of that ran on a documentation change:
    ``docs/`` mapped to no package, an empty package set classified as
    ``none``, and the gate reads ``none`` as "skip the tests". The guards that
    would have read the edited file were the ones switched off.

    The package tree failed the same way for a different reason and did not
    look like it: a package document mapped to *its package*, so the gate ran
    that package's whole suite and its dependents — while still running none
    of the four guards that actually read the file.
    """
    site = _scopes.plan_for_files(["docs/packages/bots/guides/tools.md"])
    assert site["docs_changed"] is True
    assert site["test_scope"] == "workspace", (
        "a site-tree edit must run the workspace guards that read it; "
        f"got {site['test_scope']!r}, which the gate reads as 'skip'"
    )

    # A package document read by no package suite: the workspace guards read
    # it, and nothing else does.
    package_doc = _scopes.plan_for_files(["packages/bots/docs/architecture.md"])
    assert package_doc["packages"] == [], (
        "a package document that no package test reads belongs to no suite — "
        f"scheduling one runs a whole package for a prose edit, got {package_doc['packages']}"
    )
    assert package_doc["test_scope"] == "workspace"

    # And the declared exception, which does decide whether its suite passes.
    for document, package in _scopes.PACKAGE_TEST_DOC_INPUTS.items():
        plan = _scopes.plan_for_files([document])
        assert package in plan["packages"], (
            f"{document} is read by a test in {package} — it must still "
            f"schedule that suite, got {plan['packages']}"
        )
        assert plan["docs_changed"] is True, (
            f"{document} is still documentation, so the documentation checks "
            "must still re-run for it"
        )


def test_every_package_document_a_package_suite_reads_is_in_that_package_hash() -> None:
    """Scheduling alone is not enough — the recorded verdict has to move too.

    The two mechanisms answer different questions and both have to say yes.
    Change detection decides what *this* run tests; the package hash decides
    whether a *stored* ``pass`` still describes the tree. A document that
    schedules its suite but enters no hash lets an artifact recorded before
    the edit validate after it, which is the whole defect the hashes exist to
    catch — and ``_HASH_PATTERNS`` reaches ``**/*.py`` and ``pyproject.toml``,
    so no package document was in any package's hash by default.
    """
    hashes = load_bin_module("package-hashes")

    for document, package in _scopes.PACKAGE_TEST_DOC_INPUTS.items():
        covered = [_rel(p) for p in hashes.package_hash_files(package)]
        assert document in covered, (
            f"{document} decides whether the {package} suite passes but is in "
            f"no hash scope, so a {package} verdict recorded before an edit to "
            "it still validates after one"
        )


def test_the_release_noise_definition_has_one_home() -> None:
    """Declared once, in the module both readers already share.

    Identity is not the assertion: ``load_bin_module`` execs a fresh module
    per call, so the two would compare unequal while agreeing perfectly. What
    a re-introduced copy looks like is a *second definition* in the hasher.
    """
    hashes = load_bin_module("package-hashes")

    for name in ("_VERSION_LINE_RE", "_DEP_CONSTRAINT_LINE_RE"):
        assert not hasattr(hashes, name), (
            f"package-hashes.py defines its own {name} again — that is the "
            "duplication whose two copies disagreed about a version bump"
        )

    sample = b'version = "1.0.0"\n"dataknobs-common>=1.0.0",\nkept = 1\n'
    assert hashes.strip_release_noise(sample) == _scopes.strip_release_noise(sample)


def test_the_gate_reads_the_scope_change_detection_computes() -> None:
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


def test_every_script_ci_executes_exists() -> None:
    """A workflow step naming a script that is not there fails only when it runs.

    Which is later than it sounds, and on someone else's branch: the quality job
    is conditional on a path filter, the release job runs at release time, and
    ``actionlint`` checks a ``run:`` block's shell without ever asking whether
    the file it names is in the repository. So a rename that misses one caller
    sits green until the job it broke happens to be the one that starts.

    This test used to ask a second question too — whether a hash scope covered
    each of these scripts — and that half **could not fail**. The regex is
    anchored to ``bin/``, ``bin/`` is a directory entry, and the entry admits
    both suffixes the regex can match, so every name it finds was covered by
    construction and the only reachable failure was a name that does not exist.
    A guard whose sole live failure is one its message does not describe is not
    a weaker guard; it is a second instance of the defect it was written for,
    since it reports a scope problem for what is a typo. Proved before removing
    it, by injecting a real script into a ``run:`` block: passed.

    Coverage is not lost with it. ``bin/`` is a scope entry, and what makes that
    entry keep reaching these files is
    ``test_every_linted_shell_script_is_covered_by_a_hash_scope``, which asks
    about a strictly larger set and *can* fail — it did, against a ``.bash``
    file the lint reports on and the suffix predicate rejects.
    """
    executed = _ci_executed_bin_scripts()
    assert executed, (
        "no bin/ script was found in any workflow run: block — the extraction "
        "broke, and this guard would pass by checking nothing"
    )

    missing = sorted(name for name in executed if not (ROOT / name).is_file())
    assert not missing, (
        "CI runs these scripts, and they are not in the repository:\n"
        + "\n".join(f"  - {name}" for name in missing)
        + "\n\nThe step fails at the moment that job runs, which for a "
        "conditional or release-time job is not the pull request that broke it. "
        "Fix the name in the workflow, or restore the script."
    )


def _documentation_inputs() -> list[str]:
    """Every tracked file a recorded documentation check reads.

    Three of the gate's checks are about documentation — ``mkdocs build
    --strict``, the version-table sync, and the dual-docs mirror — and between
    them they read the two documentation trees plus three individual files. The
    trees are taken from git rather than walked, so an untracked scratch file
    cannot join the set on one machine and not another.

    Of the three files, the manifest and the version registry are read from the
    scripts that consume them, for the reason every declaration in this file is:
    a restatement here would keep passing after a rename, having asserted about
    a path nothing uses. ``mkdocs.yml`` is named directly because nothing names
    it — it is mkdocs' own default, passed to no command.
    """
    tracked = _tracked("docs", "packages")
    inputs = {
        str(path)
        for path in tracked
        if path.parts[0] == "docs" or (len(path.parts) > 2 and path.parts[2] == "docs")
    }

    inputs.add("mkdocs.yml")
    inputs.add(_rel(load_bin_module("docs-mirror-check").MANIFEST))

    versions = (ROOT / "bin" / "docs-update-versions.sh").read_text()
    registry = re.search(r'^PACKAGES_JSON="([^"]+)"', versions, re.MULTILINE)
    assert registry, (
        "bin/docs-update-versions.sh no longer names its registry as "
        'PACKAGES_JSON="..." — this guard stopped tracking what that check reads'
    )
    inputs.add(registry.group(1))

    return sorted(inputs)


def test_every_documentation_input_is_covered_by_a_hash_scope() -> None:
    """A documentation input outside every hash scope lets its own edit go unchecked.

    This is ``test_every_linted_shell_script_is_covered_by_a_hash_scope`` one
    domain over, and it was found the same way: an edit to ``mkdocs.yml``
    dirtied nothing, so the artifacts recording ``documentation: pass`` stayed
    valid over a tree they no longer described.

    The consequence is larger here than staleness alone, because CI's docs job
    gates its build on this same hash check and skips when nothing is dirty. So
    a documentation-only pull request left every hash intact, the gate's stored
    verdict was accepted unexamined, *and* the job that would have rebuilt the
    site declined to run — three mechanisms agreeing to check nothing. Verified
    against a broken intra-doc link, which ``--strict`` rejects and both paths
    passed.

    Coverage is asked through ``workspace_scope_files``, the function the hash
    itself uses, rather than by re-deriving which paths an entry expands to.

    The universe asked about is every tracked file in those trees, not every
    ``*.md``, and that is deliberate: the hasher decides what to include by
    suffix, and a theme override, an included snippet, or a stylesheet all
    change what the site build does. Asserting over the whole tree is what makes
    the suffix list keep up — adding a file of a kind nothing hashes fails here,
    at the moment there is someone to decide, rather than silently later.
    """
    hashes = load_bin_module("package-hashes")
    covered = {
        _rel(path)
        for scope in WORKSPACE_QUALITY_INPUTS
        for path in hashes.workspace_scope_files(scope)
    }

    inputs = _documentation_inputs()
    assert len(inputs) > 100, (
        f"only {len(inputs)} documentation inputs resolved — the git listing "
        "broke, and this guard would pass by checking almost nothing"
    )

    uncovered = [name for name in inputs if name not in covered]
    assert not uncovered, (
        f"{len(uncovered)} of {len(inputs)} documentation inputs are in no hash "
        "scope, so editing one leaves every stored hash intact, keeps the "
        "recorded documentation verdict valid over a tree it no longer "
        "describes, and skips CI's docs build:\n"
        + "\n".join(f"  - {name}" for name in uncovered[:15])
        + (f"\n  ... and {len(uncovered) - 15} more" if len(uncovered) > 15 else "")
        + "\n\nAdd the tree or file to _DOCS_QUALITY_INPUTS in "
        "bin/changed-packages.py."
    )


def _workflow_lint_inputs() -> list[str]:
    """Every tracked file the recorded ``workflow_lint`` check reads.

    Both the directory and the extensions are read from ``bin/lint-workflows.sh``
    rather than written here, for the reason every declaration in this file is:
    a restatement keeps passing after the script changes, having asserted about
    a set nothing lints. The script globs ``.yml`` *and* ``.yaml`` — GitHub
    accepts either, and it says so where it does it.
    """
    script = (ROOT / "bin" / "lint-workflows.sh").read_text(encoding="utf-8")

    directory = re.search(r'^WORKFLOW_DIR="\$PROJECT_ROOT/([^"]+)"', script, re.MULTILINE)
    assert directory, (
        "bin/lint-workflows.sh no longer names its directory as "
        'WORKFLOW_DIR="$PROJECT_ROOT/..." — this guard stopped tracking what '
        "the workflow lint reads"
    )

    suffixes = set(re.findall(r'"\$WORKFLOW_DIR"/\*(\.[a-z]+)', script))
    assert suffixes, (
        "bin/lint-workflows.sh no longer globs its workflow files by extension "
        "— re-point this guard rather than leaving it asking about none"
    )

    prefix = directory.group(1).rstrip("/") + "/"
    return sorted(
        str(path)
        for path in _tracked(directory.group(1))
        if str(path).startswith(prefix) and path.suffix in suffixes
    )


def test_every_workflow_lint_input_is_covered_by_a_hash_scope() -> None:
    """A workflow outside every hash scope lets its own edit go unvalidated.

    ``workflow_lint`` is a recorded check and these files are its entire input,
    so this is ``test_every_documentation_input_is_covered_by_a_hash_scope`` a
    third domain over, found the same way and by the same question: what does a
    recorded check read, and does anything notice when it changes? Editing a
    workflow moved the recorded verdict while leaving every stored hash intact,
    and CI — which validates the artifact rather than re-running the gate —
    accepted the ``workflow_lint: pass`` the edit had just invalidated.

    Sharper here than elsewhere, because these files are also what CI *is*: the
    path filter deciding which jobs start is one of them, so the pull request
    that narrows the filter is one no filter would have started a check for.

    Asked through ``workspace_scope_files``, the function the hash itself uses,
    rather than by re-deriving what an entry expands to. Note what that costs a
    reader to check: a directory entry expands through a suffix predicate, so
    declaring ``.github/workflows/`` while ``.yml`` is not a quality-input
    suffix would expand to nothing at all — a declared scope covering none of
    its files, which reads exactly like coverage. This guard is what tells the
    two apart.
    """
    hashes = load_bin_module("package-hashes")
    covered = {
        _rel(path)
        for scope in WORKSPACE_QUALITY_INPUTS
        for path in hashes.workspace_scope_files(scope)
    }

    inputs = _workflow_lint_inputs()
    assert inputs, (
        "no workflow files resolved — the extraction broke, and this guard "
        "would pass by checking nothing"
    )

    uncovered = [name for name in inputs if name not in covered]
    assert not uncovered, (
        f"{len(uncovered)} of {len(inputs)} workflow-lint inputs are in no hash "
        "scope, so editing one leaves every stored hash intact and keeps the "
        "recorded workflow_lint verdict valid over files it no longer "
        "describes:\n"
        + "\n".join(f"  - {name}" for name in uncovered)
        + "\n\nAdd the directory to _WORKSPACE_ONLY_QUALITY_INPUTS in "
        "bin/changed-packages.py, and check that _QUALITY_INPUT_SUFFIXES in "
        "bin/package-hashes.py admits the extensions it holds — a directory "
        "entry whose files the predicate rejects expands to nothing."
    )


#: A ``ROOT / "literal"`` chain in a workspace guard, i.e. a file it reads by
#: name. Interpolated names are out of reach and out of scope: the population
#: this asks about is the hand-written literals, which is where the omissions
#: have been.
_ROOT_RELATIVE_RE = re.compile(r'ROOT / "([^"]+)"((?: / "[^"]+")*)')


def _files_the_workspace_guards_read() -> list[str]:
    """Every root-relative file the guards under ``tests/`` name.

    Derived from their own source rather than listed, which is the whole point:
    a list would be a fourth hand-maintained registration set beside the three
    this slice is about, and it would go stale the first time a guard started
    reading something new — silently, since a guard reading an unhashed file is
    not a guard that fails.

    Directories are skipped rather than probed. What a guard does with a
    directory varies — walk it, glob it, check it exists — so "covered" has no
    single meaning for one, while for a named file it has exactly one.

    Walked recursively even though this tree is flat today, because the
    alternative fails the way everything else here does: a guard filed one
    directory down would be outside the population and nothing would say so.

    One read, one expression. A chain split across statements — ``area = ROOT /
    "bin"`` and then ``area / "thing.txt"`` — is invisible here, and invisible
    in the direction that passes: the non-vacuity floor in the test below is
    met by every other read, so the one that went missing is reported by
    nothing. Write a root-relative read as a single chained expression and it
    stays in the population.
    """
    named: set[str] = set()
    sources = (p for p in ROOT.glob("tests/**/*.py") if "__pycache__" not in p.parts)
    for source in sorted(sources):
        for head, tail in _ROOT_RELATIVE_RE.findall(source.read_text(encoding="utf-8")):
            parts = [head, *re.findall(r'"([^"]+)"', tail)]
            candidate = ROOT.joinpath(*parts)
            if candidate.is_file():
                named.add(_rel(candidate))
    return sorted(named)


def test_every_file_the_workspace_guards_read_is_covered_by_a_hash_scope() -> None:
    """A guard's own input outside every hash scope lets its verdict go stale.

    The guards under ``tests/`` are themselves hashed — ``tests/`` is a scope
    entry — but what they *read* is not, and the two are different sets. So a
    file like ``.gitignore``, which decides the answer of three guards here and
    is named by none of the scopes, could be edited to flip one of them from
    pass to fail while every stored hash stayed intact and the recorded
    ``unit_tests`` verdict stayed valid over it.

    That is the same sentence as the documentation and workflow guards above,
    turned on the guards themselves — which is the case most likely to be
    missed, because the scope entry covering the *code* looks like coverage of
    the check.

    The population is derived from the guards' own source, so a guard that
    starts reading a fourth root file is covered by this the day it is written
    rather than the day someone remembers.
    """
    hashes = load_bin_module("package-hashes")
    covered = {
        _rel(path)
        for scope in WORKSPACE_QUALITY_INPUTS
        for path in hashes.workspace_scope_files(scope)
    }

    named = _files_the_workspace_guards_read()
    assert len(named) > 10, (
        f"only {len(named)} named files resolved from the workspace guards — "
        "the extraction broke, and this guard would pass by checking almost "
        "nothing"
    )

    uncovered = [name for name in named if name not in covered]
    assert not uncovered, (
        f"{len(uncovered)} of {len(named)} files the workspace guards read are "
        "in no hash scope, so editing one changes what a guard reports while "
        "leaving every stored hash intact:\n"
        + "\n".join(f"  - {name}" for name in uncovered)
        + "\n\nAdd each to _WORKSPACE_ONLY_QUALITY_INPUTS in "
        "bin/changed-packages.py — a guard's input moves that guard's result "
        "and no package's, which is what that tier is for."
    )


def test_every_docs_hash_input_also_reruns_the_docs_checks() -> None:
    """Hashing an input the docs checks are not re-run for recomputes nothing.

    The two halves have to agree in both directions. The hash decides whether a
    stored verdict still describes the tree; ``docs_changed`` decides whether
    the gate recomputes that verdict. An input in the first but not the second
    produces the worst of the three possible states: the artifact goes stale, the
    author is told to re-run the gate, the gate skips the documentation checks
    because it sees no documentation change, and a fresh artifact is stamped
    carrying the *old* verdict and the *new* hash. That is not a missing check —
    it is a check that reports having run.

    Found while adding the hash scope, against ``.dataknobs/packages.json``:
    hashed as the input the version-table check compares against, but matched by
    no docs pattern, so it invalidated the artifact and then let it be
    regenerated without anything reading it.

    The other direction is deliberately not asserted. A path may set
    ``docs_changed`` without being hashed — package sources do, since
    ``mkdocstrings`` renders them — and those are already covered by the package
    hashes.
    """
    hashes = load_bin_module("package-hashes")
    plan_for_files = _scopes.plan_for_files

    untriggered = [
        _rel(path)
        for path in hashes.workspace_scope_files("docs")
        if not plan_for_files([_rel(path)])["docs_changed"]
    ]

    assert not untriggered, (
        "these files are hashed into the docs scope but match no docs pattern, "
        "so changing one invalidates the artifact without making the gate "
        "recompute the verdict it invalidated:\n"
        + "\n".join(f"  - {name}" for name in untriggered[:15])
        + (f"\n  ... and {len(untriggered) - 15} more" if len(untriggered) > 15 else "")
        + "\n\nAdd each to DOCS_PATTERNS in bin/changed-packages.py."
    )


def test_no_workspace_test_is_filed_where_nothing_runs_it() -> None:
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
    runner_text = (ROOT / "bin" / "test.sh").read_text()
    excluded = sorted(set(RUNNER_IGNORE_RE.findall(runner_text)))
    assert excluded, (
        "bin/test.sh passes no --ignore for the workspace run — either the "
        "runner stopped excluding a directory or this guard stopped tracking "
        "how it spells the flag"
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
        + f"\n  The workspace target passes --ignore for {', '.join(excluded)} "
        "and the integration step only loops packages/*/tests/integration.\n"
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

    Which file, though, is the same question ``scope_entry_files`` answers when
    it decides what an entry hashes — so it is asked there rather than restated
    here. It used to be restated, as ``rglob("*.py")``, and that was right only
    while ruff and mypy were the only readers of these directories. Once the
    gate gained a shell lint, a shell-only directory entry produced *no probe at
    all*: the entry was declared, its files moved a recorded verdict, and this
    guard silently asked nothing about whether CI would run on a change to them.
    Reproduced before fixing, with a shell-only directory declared and no
    matching CI pattern — the guard passed.

    The restatement it was replaced with had the same shape of hole waiting in
    it: a ``*`` in a directory entry resolved to no directory here, so
    ``packages/*/docs/`` would again have been declared and silently unprobed.
    Calling the hasher's own expansion is what stops that recurring per entry
    kind.
    """
    scope_entry_files = load_bin_module("package-hashes").scope_entry_files

    probes: list[str] = []
    for entries in WORKSPACE_QUALITY_INPUTS.values():
        for entry in entries:
            probes += [_rel(p) for p in sorted(scope_entry_files(entry))[:1]]

    assert probes, "no workspace quality inputs resolved — the shared declaration is empty"
    return probes


def test_ci_runs_the_gate_when_a_guarded_file_changes() -> None:
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


# ``DEFERRED_FROM_DEFAULT_LINT`` used to live here: five directory classes of
# first-party Python that ``bin/validate.sh`` does not reach, each with a count
# in its comment. It has moved to ``.dataknobs/quality-contract.json`` as that
# file's ruff axis, and the two guards over it — every tracked file lands in
# exactly one cell, and no cell is stale or unreached — are in
# ``tests/test_quality_contract.py``. The "or deferred" half of that sentence
# has gone with the tier: ruff declares only ``checked``, so there is no longer
# a way to be decided about and unread at the same time.
#
# Two things changed in the move rather than merely relocating.
#
# The counts became **ceilings that are compared**. In prose they were enforced
# in one direction only: an entry matching nothing failed, while "241 findings"
# stayed green at 400. A number nobody checks is one that stops being true
# without anyone finding out, and this program has now found that shape in a
# guard, a reader, an ignore rule and a status field.
#
# And the population became **partitioned rather than filtered**. The old pair
# asked "is this file linted, or excused?", which is satisfiable by a file that
# is neither — it just has to be excused. The contract asks which cell each file
# is in and fails when the answer is none or several, so a directory added to
# the repository and to no declaration fails immediately rather than the first
# time somebody wonders.


def _tracked(*pathspecs: str) -> list[PurePosixPath]:
    """Every file git keeps under the given pathspecs, as repo-relative paths.

    Asking git rather than walking the tree is what keeps an editor backup, a
    stray ``.orig``, or a macOS ``.DS_Store`` from joining the answer on one
    machine and not another — which for a set feeding a content hash would mean
    a developer and CI computing different digests over the same commit.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", *pathspecs],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [PurePosixPath(name) for name in listing.split("\0") if name]


def _tracked_python() -> list[PurePosixPath]:
    """Every ``*.py`` git keeps, as repo-relative paths."""
    return _tracked("*.py")


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

#: The other way to not keep your own copy: ask the consumer that already has
#: one. ``run-quality-checks.sh`` resolves the scope it writes the diagnostics
#: artifact over by calling ``validate.sh --print-targets`` with the very
#: argument string it is about to validate with, so the artifact's scope and the
#: validated scope are one value rather than two that have to agree. That is
#: stronger than calling ``workspace-targets`` directly, not weaker -- a direct
#: call would restore the second answer this removed -- so the guard below has
#: to accept it or it fails the fix for the defect it exists to catch.
#:
#: Sound only because the delegate is itself in WORKSPACE_TARGET_CONSUMERS and so
#: is held to the same rule; the assertion below checks that rather than assuming
#: it, since a delegation to something unguarded is just a copy one file over.
WORKSPACE_TARGETS_DELEGATION = re.compile(
    r"\$\(\s*\"?\$\{?\w+\}?/(?P<delegate>[\w.-]+)\"?\s+--print-targets\b[^)]*\)"
)

#: Which consumer each delegation resolves to, by basename under bin/.
DELEGATE_ROOT = "bin"

#: The variable a consumer captures the set into, when it captures rather than
#: expanding inline. Feeds the "is it ever read" half of the guard below.
CAPTURED_WORKSPACE_TARGETS = re.compile(
    r"^\s*(\w+)=\$\([^)]*workspace[_-]targets\b[^)]*\)", re.MULTILINE
)

#: Three quoting forms, because bash accepts all three and the guard below is the
#: only thing standing between the gate and a hardcoded target list. Matching the
#: double-quoted form alone let ``VALIDATE_ARGS=tests`` — valid, and exactly the
#: bug — pass unseen.
VALIDATE_ARGS_ASSIGNMENT = re.compile(
    r"""^\s*VALIDATE_ARGS=(?:"([^"]*)"|'([^']*)'|([^\s;&|#]*))""", re.MULTILINE
)


def _lint_targets_for(script: str, *args: str) -> list[str]:
    """What ``script`` resolves as the linter's population, for these arguments.

    Asked rather than parsed, for the reason ``_workspace_targets`` is: the
    question is what the script reads, and reading the appends as text answers
    what it says. The earlier form read them out of the default branch and
    treated each as unconditional, so wrapping the package loop in a condition
    that skipped it left this reporting full coverage — the append was still
    textually present. ``--print-targets`` resolves the list through the real
    code path and prints it before anything is checked or rewritten.

    Two scripts answer it, and their answers are not the same list: the check's
    is what the contract holds to a lint ceiling, the fix's is that plus the
    cells it may rewrite anyway.
    """
    listing = subprocess.run(
        [str(ROOT / "bin" / script), *args, "--print-targets"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return listing.split()


def _validate_targets_for(*args: str) -> list[str]:
    """What ``bin/validate.sh`` resolves as its target list, for these arguments."""
    return _lint_targets_for("validate.sh", *args)


def _default_validate_targets() -> list[str]:
    """Every path ``bin/validate.sh`` validates when given no arguments."""
    return _validate_targets_for()


def _format_targets_for(script: str, *args: str) -> list[str]:
    """What ``script`` resolves as the formatter's population, for these arguments.

    Asked rather than parsed, for the reason ``_validate_targets_for`` is. The
    formatter has its own list because its declared coverage is not the
    linter's, so a caller cannot substitute one for the other.
    """
    listing = subprocess.run(
        [str(ROOT / "bin" / script), *args, "--print-format-targets"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return listing.split()


def _enforced_format_cells() -> list[str]:
    """Every path the quality contract holds to a formatting ceiling."""
    contract = json.loads((ROOT / ".dataknobs" / "quality-contract.json").read_text("utf-8"))
    return [
        cell["path"] for cell in contract["tools"]["format"]["cells"] if cell["tier"] == "enforced"
    ]


def _checked_lint_cells() -> list[str]:
    """Every path the quality contract holds to a lint ceiling of zero.

    Read from the tier rather than from what ``bin/validate.sh`` reaches. The
    two agree — ``test_every_lint_cell_is_one_the_linter_actually_reaches`` in
    tests/test_quality_contract.py is what makes them — and deriving this from
    the target set would make the guard below move whenever that set does,
    which is the shape ``REQUIRED_DEFAULT_TARGETS`` exists to refuse.
    """
    contract = json.loads((ROOT / ".dataknobs" / "quality-contract.json").read_text("utf-8"))
    return [
        cell["path"] for cell in contract["tools"]["ruff"]["cells"] if cell["tier"] == "checked"
    ]


def _cell_is_covered(cell: str, targets: set[str]) -> bool:
    """Whether some target reaches the files ``cell`` names.

    A cell is a glob over directories; a target is one of those directories, or
    an ancestor of it. Compared by expansion rather than by string, because
    ``packages/*/tests`` and ``packages/bots/tests`` are the same claim written
    two ways and only one of them is what a script emits.

    Named at length rather than ``_covers``, which is taken. The first draft of
    this reused that name and, being defined later, silently replaced it — so
    the CI-scheduling guard above began asking a question about formatter
    targets and reported every guarded file untriggered. Ruff does not flag it
    (F811 wants the earlier binding unused, and that one is called), so the
    only thing that caught it was the suite this file belongs to.
    """
    expanded = {_rel(path) for path in ROOT.glob(cell)} if "*" in cell else {cell}
    if not expanded:
        return True  # A cell naming nothing on disk cannot be uncovered.
    return all(
        any(match == target or match.startswith(f"{target}/") for target in targets)
        for match in expanded
    )


#: What the default target set must contain, stated here rather than derived.
#:
#: This is the one duplication in this file that is load-bearing, and it is the
#: answer to a specific hole. Every other assertion about coverage reads the
#: target set and asks whether it reaches the tracked files — so dropping a
#: directory from ``workspace_targets`` and naming it in
#: the quality contract's deferred tier satisfied all of them at once: the files
#: were no longer uncovered because they were now deferred, and the deferral was
#: not contradicted because nothing linted them any more. Both guards green,
#: ``bin/`` unlinted again. A check derived from the thing it checks moves when
#: that thing moves; this does not, so removing a member fails here and no entry
#: elsewhere can quiet it.
#:
#: That escape has since been closed from the other end as well — ruff declares
#: no tier a cell can retreat into, and
#: ``test_every_lint_cell_is_one_the_linter_actually_reaches`` no longer reads
#: tiers at all. This stays because the two are not the same claim: that one is
#: about cells the contract declares, and a target dropped from a set the
#: contract never mentions is still a target dropped.
REQUIRED_DEFAULT_TARGETS = frozenset({"tests", "bin", "src", "conftest.py", "quality-fixture"})


# ``test_every_first_party_python_file_is_linted_by_default`` used to sit here,
# with ``_linted_by`` and ``_deferred_by`` beneath it. It asked whether each
# tracked file was reached by ``bin/validate.sh`` or excused by name, which the
# quality contract now answers more completely: totality places every file in
# exactly one cell, and the coverage guard compares every ruff cell against the
# target set the script resolves.
#
# The helpers went with it, and one of them was subtly wrong in a way the
# replacement is not. ``_deferred_by`` matched with ``PurePosixPath.match``,
# which anchors a *relative* pattern at the right-hand end — so a single-segment
# entry would have matched any directory of that name at any depth, while its
# sibling ``_linted_by`` compared string prefixes from the left. Two rules over
# one question. Nothing had exercised the difference because every entry in the
# old list began with ``packages/``; ``cell_matches`` in bin/quality-contract.py
# compares segment by segment from the root, for every cell whatever its tier.


def test_the_default_target_set_still_contains_what_it_must() -> None:
    """The contract must not be able to buy its way out of a lost target.

    The guard above compares the target set against the tracked files and takes
    the contract as declared data, which makes it complete about *accidents* and
    silent about one deliberate move: drop a directory from ``workspace_targets``,
    re-file its cell in a tier that tolerates a backlog, and coverage was gone
    with both checks still green. Replayed over the real repository before this was
    written, the escape also worked for ``packages/*/src`` — all ten package
    sources could leave the target set without a single assertion failing.

    Ruff no longer declares such a tier, and the cell guard no longer reads
    tiers, so that exact route is shut. This still states the members, because
    the property it holds is not about cells: ``workspace_targets`` can lose an
    entry the contract never named, and nothing derived from the contract would
    notice.

    So this states the required members instead of deriving them. That is the
    duplication ``workspace_targets`` exists to remove, and here it is the point:
    an assertion computed from the declaration it guards cannot notice the
    declaration shrinking.

    Probed with a package named as well as with nothing, because those are two
    different questions and only one of them was asked. The gate runs
    ``validate.sh $PACKAGES --workspace`` on every pull request that touches a
    package; it never runs it bare. Reverting the append to fire only when no
    target was named leaves the *bare* answer byte-identical, so a check that
    asks only that one reports full coverage while every real gate invocation
    validates ``packages/*/src`` alone — the defect this whole file exists to
    catch, restored with the suite green.
    """
    packages = sorted(path.parent.name for path in pyprojects() if path.parent != ROOT)
    probes: list[tuple[str, ...]] = [()]
    if packages:
        probes.append((packages[0], "--workspace"))

    for probe in probes:
        targets = set(_validate_targets_for(*probe))
        missing_workspace = sorted(REQUIRED_DEFAULT_TARGETS - targets)
        shown = " ".join(probe) or "(no arguments)"
        assert not missing_workspace, (
            f"bin/validate.sh {shown} no longer validates {missing_workspace}. "
            "These are not deferrable: tests/ holds the guards that check the "
            "toolchain, bin/ holds the checkers that decide whether a pull "
            "request passes, and quality-fixture/ is the one tree whose whole "
            "claim is that the gate reads it and finds nothing. Restore the "
            "target — recording it in "
            "deferring it in .dataknobs/quality-contract.json is not the fix, "
            "it is the failure this guard exists to catch."
        )

    #: The pin must also grow when the declaration does, or a directory added to
    #: workspace_targets and later dropped from it would be missing from both —
    #: the same omission one directory over, which is the shape above. Asserting
    #: containment fails when the declaration grows unpinned; the pin above fails
    #: when it shrinks. Neither is derived from the other, so both directions
    #: hold without the circularity that would make either vacuous.
    unpinned = sorted(set(_workspace_targets()) - REQUIRED_DEFAULT_TARGETS)
    assert not unpinned, (
        f"workspace_targets now declares {unpinned}, which REQUIRED_DEFAULT_TARGETS "
        "does not name. Add it there: until it is pinned, dropping it again and "
        "deferring it in .dataknobs/quality-contract.json passes every check here."
    )

    targets = set(_default_validate_targets())
    missing_sources = [
        f"packages/{name}/src"
        for name in packages
        if f"packages/{name}/src" not in targets and (ROOT / "packages" / name / "src").is_dir()
    ]
    assert not missing_sources, (
        f"bin/validate.sh no longer validates {missing_sources} by default, so "
        "the shipped source of those packages is linted by nothing."
    )


def test_every_formatting_ceiling_is_reachable_by_the_check_and_by_the_fix() -> None:
    """The formatter's two entry points must span what the contract enforces.

    ``test_remediation_paths`` pins *which* scripts may run the formatter. This
    is the other half — whether the ones that may, reach far enough — and it is
    the half that was missing while a docstring there said it existed.

    The gap it was written against: the format check iterated the *linter's*
    target set. That set reached only the cells ruff was pointed at, while the
    contract enforced ``format`` at ceiling 0 on all ten of its cells. So
    ``validate.sh`` opened well under half the files it reported on and printed
    "Formatting is clean" over the rest, while ``fix.sh`` could not repair some
    of them at all — a finding the gate reports and no local command can clear.

    Both directions matter and neither implies the other. A check that reads
    less than the contract enforces is a green verdict over unexamined files;
    a fix that reaches less than the check flags is a red gate with no remedy.
    """
    cells = _enforced_format_cells()
    assert cells, "the contract enforces no formatting ceiling, so this proves nothing"

    for script, role in (
        ("validate.sh", "checks formatting"),
        ("fix.sh", "repairs what the check reports"),
    ):
        targets = set(_format_targets_for(script))
        uncovered = [cell for cell in cells if not _cell_is_covered(cell, targets)]
        assert not uncovered, (
            f"bin/{script} {role} over {sorted(targets)}, which does not reach "
            f"{uncovered}. The quality contract holds those cells to a "
            "formatting ceiling of 0, so a file arriving unformatted there "
            f"fails `dk pr` while bin/{script} says nothing about it."
        )


def test_every_lint_ceiling_is_reachable_by_the_fix() -> None:
    """The linter's half of the guard above — the half that was still missing.

    The check side is already held, from the other end:
    ``test_every_lint_cell_is_one_the_linter_actually_reaches`` compares every
    ruff cell against what ``bin/validate.sh`` resolves. Nothing compared the
    *fix* side, and that is the direction with no symptom — a cell the check
    reads and the fix cannot reach is a red gate with no local remedy, which
    looks exactly like a developer who has not run ``bin/fix.sh`` yet.

    Reachability, not repair. Half of ruff's rules have no autofix, so the
    property here is that the pass runs over the cell at all; whether a given
    finding is fixable is the rule's business. That is the whole claim, and it
    is enough — a cell the pass never opens has *nothing* fixable in it.

    Load-bearing now rather than later because promoting a cell is two
    declarations, its tier and each script's reach, and shipping one without
    the other is what 2c did: a check that passed over the territory the
    contract had started enforcing. Every promotion still to come has the same
    shape, and this is what makes the second declaration compulsory.

    Nothing is excepted from this any more, and the paragraph that stood here is
    worth recording as closed rather than deleted. It excepted the deferred cells
    and noted that a bare ``bin/fix.sh`` reached none of ``examples``,
    ``scripts``, ``benchmarks`` or ``docs``, while their ceilings *were*
    compared — a red gate whose stated remedy did not reach the finding unless
    the developer knew to name the directory. The first three arrived in the fix
    pass with their promotion. The fourth has no ruff cell at all, so there is no
    lint ceiling over it to be unreachable. The policy call that paragraph
    deferred — whether to widen the default over files nobody had asked to be
    clean — was answered by making them clean.
    """
    cells = _checked_lint_cells()
    assert cells, "the contract holds no cell to a lint ceiling, so this proves nothing"

    targets = set(_lint_targets_for("fix.sh"))
    uncovered = [cell for cell in cells if not _cell_is_covered(cell, targets)]
    assert not uncovered, (
        f"bin/fix.sh repairs lint findings over {sorted(targets)}, which does not "
        f"reach {uncovered}. The quality contract holds those cells to a lint "
        "ceiling of 0, so a finding arriving there fails `dk pr` while the "
        "command the failure tells you to run cannot touch it."
    )


def test_the_named_opt_in_asks_for_the_format_target_set_rather_than_restating_it() -> None:
    """The third owner, which cannot be asked and so is read instead.

    ``bin/dk format`` runs the formatter directly and prints no target list, so
    the check above cannot interrogate it. Its whole history is this defect:
    it formatted ``packages/*/src`` alone, was widened once to add the workspace
    set, and was still narrower than the check both times. A third answer to
    *which code do we format* is what the single declaration exists to prevent.
    """
    source = (ROOT / "bin" / "dk").read_text("utf-8")
    branch = source[source.index("format|fmt)") :]
    branch = branch[: branch.index("\n            ;;")]

    # Reading the branch alone proved it *named* the helper, not that the name
    # resolves. bin/dk invokes package-discovery.sh rather than sourcing it, so
    # each helper needs a wrapper here; the first draft of this called
    # format_targets without one and would have died with command-not-found on
    # every run, with this test green. Asked of the real script for that reason.
    resolved = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/bin/dk" 2>/dev/null; format_targets'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resolved.returncode == 0 and resolved.stdout.split(), (
        "bin/dk names format_targets but cannot run it: "
        f"exit {resolved.returncode}, stdout {resolved.stdout!r}, "
        f"stderr {resolved.stderr.strip()[:200]!r}. It invokes "
        "package-discovery.sh rather than sourcing it, so the helper needs a "
        "wrapper in bin/dk."
    )

    assert "$(format_targets)" in branch, (
        "bin/dk's format branch no longer asks package-discovery.sh for the "
        "target set. Restating it here is how this command came to format less "
        "than bin/validate.sh checks, twice."
    )
    restated = [
        line
        for line in branch.splitlines()
        if "packages/" in line and not line.lstrip().startswith("#")
    ]
    assert not restated, (
        f"bin/dk's format branch names package paths directly: {restated}. "
        "Those are format_targets' answer to give."
    )


def test_the_formatter_population_is_not_the_linters() -> None:
    """Non-vacuity: the two sets must actually differ, or the guard above is free.

    If the formatter's list were ever made equal to the linter's again, every
    assertion above would still pass on the day the linter's list happened to
    be wide enough — and would start failing later for a reason with nothing to
    do with the formatter. Stating the difference keeps the guard anchored to
    why the two lists exist separately.

    This expires when the two populations legitimately converge, and the
    difference is now down to its last member. Promoting ``packages/<pkg>/tests``
    did not do it, and neither did promoting ``examples``, ``scripts`` and
    ``benchmarks``: what survives is ``packages/<pkg>/docs``, which the formatter
    reaches and the linter does not.

    That one is a standing target rather than a live one — no tracked ``*.py``
    lives under any of them today, and the contract declares no cell for them
    because totality is a statement about tracked files. It is in the
    formatter's list so that a Python file landing there is formatted from the
    day it arrives rather than from the day somebody notices. When it is
    promoted too, delete this — the check above is what carries the property.
    """
    linted = set(_validate_targets_for())
    formatted = set(_format_targets_for("validate.sh"))
    assert formatted - linted, (
        "the formatter now checks exactly what the linter does. If "
        "packages/*/docs was promoted into the linter's target set, that is "
        "correct and this guard has expired — delete it. Otherwise the format "
        "step has been pointed back at VALIDATE_TARGETS, and every directory "
        "the two lists used to differ by is unchecked again."
    )


def test_the_gate_asks_for_the_workspace_target_set_rather_than_restating_it() -> None:
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
        value for value in assignments if value and "$" not in value and not value.startswith("-")
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
    silent = sorted(value for value in assignments if value and "--workspace" not in value)
    assert not silent, (
        f"bin/run-quality-checks.sh assigns VALIDATE_ARGS={silent} without "
        "--workspace, so that branch validates package sources alone and the "
        "code belonging to no package — bin/, tests/, src/, conftest.py — goes "
        "unchecked while the run reports a passing validation. --workspace is "
        "additive; it does not displace the packages beside it."
    )

    sources = {
        name: (ROOT / name).read_text(encoding="utf-8") for name in WORKSPACE_TARGET_CONSUMERS
    }

    #: Delegation is only "not keeping your own copy" if what it delegates to is
    #: held to this same rule. Checked before it is honoured, so a delegation to
    #: an unguarded script fails here rather than silently satisfying the guard.
    unguarded_delegates = sorted(
        f"{name} -> {match.group('delegate')}"
        for name, text in sources.items()
        for match in WORKSPACE_TARGETS_DELEGATION.finditer(text)
        if f"{DELEGATE_ROOT}/{match.group('delegate')}" not in WORKSPACE_TARGET_CONSUMERS
    )
    assert not unguarded_delegates, (
        f"{unguarded_delegates} resolve their target set from a script this guard "
        "does not hold to the same rule, so the copy moved rather than went away"
    )

    unread = sorted(
        name
        for name, text in sources.items()
        if not WORKSPACE_TARGETS_CALL.search(text) and not WORKSPACE_TARGETS_DELEGATION.search(text)
    )
    assert not unread, (
        f"{unread} build a default set of things to check without calling "
        "workspace_targets or asking a consumer that does, so each carries its "
        "own idea of which code belongs to no package. That is how bin/ ended "
        "up in none of them."
    )

    # A call whose result nothing reads is the same dead end as a definition
    # nobody calls, one step later. Both consumers that capture into a variable
    # do so because errexit applies to a bare assignment but not to a
    # substitution inside an argument list — which means dropping the expansion
    # from the command leaves a well-formed script, a satisfied call check, and
    # the narrowed target set the call existed to widen.
    discarded = sorted(
        f"{name}: ${{{variable}}}"
        for name in WORKSPACE_TARGET_CONSUMERS
        for variable in CAPTURED_WORKSPACE_TARGETS.findall(
            (ROOT / name).read_text(encoding="utf-8")
        )
        if not re.search(
            rf"\$\{{{re.escape(variable)}\b|\${re.escape(variable)}\b",
            (ROOT / name).read_text(encoding="utf-8").split(f"{variable}=", 1)[1],
        )
    )
    assert not discarded, (
        f"{discarded} capture the workspace target set and never expand it, so "
        "the set is computed and thrown away. The command that was supposed to "
        "receive it checks package sources alone."
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


def test_the_print_check_recognises_test_files_by_name_not_by_substring() -> None:
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

    # Both branches have to *call* it. The predicate being correct says nothing
    # about who consults it, and the outcome guard below only catches a
    # re-introduced exemption — deleting the call outright widens the check
    # instead of narrowing it, so that guard stays green while every test file
    # in the repository starts being scanned for print statements.
    call_sites = len(re.findall(r'is_test_file\s+"', source))
    assert call_sites == 2, (
        f"bin/validate.sh calls is_test_file at {call_sites} site(s), expected 2 — "
        "the named-file branch and the directory walk. They each carried their own "
        "glob before, which is how they came to disagree; a branch that stops "
        "consulting the shared predicate has silently grown its own answer again."
    )


def test_the_print_check_examines_shipped_modules_under_a_testing_package(tmp_path: Path) -> None:
    """The predicate being right does not mean the directory walk uses it.

    Asserted as an outcome rather than a spelling. The first version of this
    listed the two globs it had replaced and checked they were absent, which is a
    blacklist of two strings: ``! -path '*/test*'`` in single quotes is the same
    exemption, matches no entry, and restores it with the suite green. The same
    commit widened another guard in this file to three quoting forms for exactly
    that reason, and this one was written beside it without the lesson.

    So this runs the real walk over a real directory and asserts the finding
    comes back. Any spelling that re-exempts a ``testing/`` package fails here,
    including ones nobody has thought of.
    """
    package = tmp_path / "shipped"
    (package / "testing").mkdir(parents=True)
    (package / "testing" / "helpers.py").write_text(
        'def emit() -> None:\n    print("not a test file")\n', encoding="utf-8"
    )

    result = subprocess.run(
        [str(ROOT / "bin" / "validate.sh"), str(package)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr

    assert "helpers.py" in combined and "print" in combined.lower(), (
        "bin/validate.sh's print check did not examine a shipped module under a "
        "testing/ package. That is the exemption the is_test_file predicate "
        "replaced — it hid eleven shipped files, including the in-memory "
        "constructs the house rules point at instead of mocks.\n\n" + combined
    )


def test_the_print_check_counts_files_rather_than_words(tmp_path: Path) -> None:
    """One offending file must not be reported as a dozen.

    The findings are ``path:line:col:content`` and content is a line of source,
    so the tally that joined the array with ``echo "${PRINT_RESULTS[@]}"`` and
    split it on spaces was counting the *vocabulary* of the offending lines:
    ``cut -d: -f1`` turned ``print(f"Error`` into a token and ``sort -u`` called
    it a filename. A promotion that surfaced four prints in ONE file printed
    "... and 12 more files", and the reader went looking for twelve files that
    do not exist.

    It only ever over-counted, never under-counted, which is why a wrong number
    stayed plausible for as long as it did. The single file below carries more
    distinct words than the ten-file display threshold, so the old tally reaches
    that threshold on its own and the "more files" line appears; the fixed one
    counts one.
    """
    package = tmp_path / "wordy"
    package.mkdir()
    (package / "chatty.py").write_text(
        "def emit() -> None:\n"
        '    print("the quick brown fox jumps over the lazy dog again")\n'
        '    print(f"another line with plenty of separate words in it {1}")\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(ROOT / "bin" / "validate.sh"), str(package)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr

    # The finding itself must still be reported — a tally that counts nothing
    # would also pass the assertion below.
    assert "chatty.py" in combined, (
        "bin/validate.sh's print check did not report the offending file at "
        "all, so the count below is being asserted over an empty result.\n\n" + combined
    )
    assert "more files" not in combined, (
        "bin/validate.sh reported additional offending files for a run with "
        "exactly one. The tally is splitting finding text on spaces and "
        "counting words as filenames.\n\n" + combined
    )


# ``test_the_lint_deferrals_still_describe_the_repository`` was here: a deferral
# entry matching nothing is stale, and one matching something already linted is
# the cheapest way to silence a coverage check. Both directions survive, over
# every tool rather than just ruff, as
# ``test_verify_names_a_cell_that_matches_no_tracked_file`` and
# ``test_every_lint_cell_is_one_the_linter_actually_reaches`` in
# tests/test_quality_contract.py.


def _gate_pytest_commands() -> list[str]:
    """Every direct ``pytest`` command in the gate, continuations joined."""
    lines = (ROOT / "bin" / "run-quality-checks.sh").read_text().splitlines()
    commands = []
    for i, line in enumerate(lines):
        if "run pytest" not in line:
            continue
        parts, j = [line], i
        while j + 1 < len(lines) and lines[j].rstrip().endswith("\\"):
            j += 1
            parts.append(lines[j])
        commands.append(" ".join(p.strip().rstrip("\\").strip() for p in parts))
    return commands


def test_the_gate_runs_no_checker_itself() -> None:
    """Every suite the gate records is run by the command a developer runs.

    This started as a narrower guard: the gate's one direct pytest call had to
    receive only pytest arguments, because ``$TEST_FLAGS`` holds ``bin/test.sh``
    spellings (``--parallel``, ``--quiet``) and handing one to pytest produces a
    *usage* error — which the gate counts as a failing suite while no test
    failed and nothing in the summary says why.

    That hazard existed because the call was there at all. The workspace guards
    were the one suite ``test.sh`` could not reach, so the gate ran them itself,
    and a gate result and a local result had a place to differ that no assertion
    covered. ``test.sh workspace`` closed it, and this became the emptiness
    check it should always have been: a set that must stay empty cannot quietly
    narrow to the one case someone remembered, and the flag-translation hazard
    cannot come back without the call coming back first.
    """
    commands = _gate_pytest_commands()

    assert not commands, (
        "bin/run-quality-checks.sh runs pytest directly:\n"
        + "\n".join(f"  - {c}" for c in commands)
        + "\n  Every suite must go through bin/test.sh, so a gate run and a "
        "developer run execute the same invocation. Add a target there if the "
        "suite has no home."
    )


# --------------------------------------------------------------------------
# requires-python — the floor every other declaration is measured against
# --------------------------------------------------------------------------


def test_every_project_declares_the_same_floor(floor: tuple[int, int]) -> None:
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


def test_mypy_python_version_matches_floor(floor: tuple[int, int]) -> None:
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


def test_black_target_version_matches_floor(floor: tuple[int, int]) -> None:
    expected = f"py{floor[0]}{floor[1]}"
    violations = [
        f"{_rel(path)}: [tool.black] target-version = {targets!r} (want [{expected!r}])"
        for path in _pyprojects()
        if (targets := _load(path).get("tool", {}).get("black", {}).get("target-version"))
        and list(targets) != [expected]
    ]

    assert not violations, _fmt(violations, floor)


def test_pylint_py_version_matches_floor(floor: tuple[int, int]) -> None:
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


def test_ruff_target_version_matches_floor(floor: tuple[int, int]) -> None:
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


def test_mypy_path_entries_resolve() -> None:
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


#: A cell naming one package's source root, as opposed to a glob over several
#: (``packages/*/examples``) or a directory that is not an importable root
#: (``bin``, ``tests``). Only these carry the property below: they are the
#: directories whose modules another package imports *by name*.
PACKAGE_SOURCE_CELL = "packages/*/src"


def _type_checked_package_sources() -> list[str]:
    """Every package source root the mypy contract measures.

    Read from the contract rather than from ``packages/*/src`` on disk. The
    contract is what decides a directory is type-checked at all, so a package
    the contract has deliberately left ``unchecked`` must not be demanded here
    — and a package that appears on disk without a cell is already a failure,
    of the totality rule in ``test_quality_contract.py``, reported there.
    """
    contract = json.loads((ROOT / ".dataknobs" / "quality-contract.json").read_text("utf-8"))
    return [
        cell["path"]
        for cell in contract["tools"]["mypy"]["cells"]
        if cell["tier"] != "unchecked" and PurePosixPath(cell["path"]).match(PACKAGE_SOURCE_CELL)
    ]


def test_type_checked_packages_are_on_the_search_path() -> None:
    """A type-checked package missing from ``mypy_path`` is checked against ``Any``.

    mypy resolves an import by name against its search path. A package under
    check whose own root is *absent* from that path does not fail to resolve —
    it falls back to ``ignore_missing_imports``, so every symbol crossing that
    boundary comes back as ``Any`` and the errors that would have been reported
    are never computed. The package still measures a number, and the number is
    lower than the truth.

    ``test_mypy_configs_declare_the_same_search_path`` used to sit here and
    asserted that every config declared the *same* ``mypy_path``. Two failings,
    and the second is why the replacement is shaped differently. It opened with
    ``if len(declared) < 2: pytest.skip(...)``, so retiring ``mypy.ini`` would
    have left it skipping forever — a check reporting green because it cannot
    report anything else. And comparing two declarations to each other can only
    find a *disagreement*: ``packages/legacy/src`` was missing from both, so the
    guard read agreement and passed over the one real search-path fault in the
    tree, which is the defect this test now names.

    Stated as completeness against the contract, a shared omission has nothing
    to hide behind: there is one declaration to check and one population to
    check it against.

    **Only this direction.** The converse — every ``mypy_path`` entry is a
    measured cell — is not asserted, because an entry that is *not* a package
    source root is legitimate (a stubs directory is the obvious one), so the
    converse would forbid a correct future declaration to catch a fault that
    ``test_mypy_path_entries_resolve`` already reports as a path pointing at
    nothing.
    """
    declared = {entry for _, entry in _mypy_path_entries()}
    cells = _type_checked_package_sources()

    assert cells, (
        "no package source cell found in the mypy contract — either every "
        "package became unchecked or this guard stopped recognising the shape"
    )

    violations = [
        f"{cell}: type-checked by the contract, absent from mypy_path, so its "
        f"modules resolve to Any when imported by name"
        for cell in sorted(cells)
        if cell not in declared
    ]

    assert not violations, "Type-checked packages missing from the search path:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


#: The rule a package's suppressions become readable under, and the one the
#: adoption series turns on per package rather than tree-wide at once.
ADOPTION_ERROR_CODE = "ignore-without-code"


def _top_level_modules(cell_path: str) -> list[str]:
    """The importable names a package source root contributes.

    Read from disk rather than guessed from the package directory name: the two
    agree today (``packages/utils/src`` holds ``dataknobs_utils``) and the one
    place they do not is exactly where a guessed name would be wrong — the
    legacy package's source root holds ``dataknobs``.

    Single-file modules count too. Every package here ships a directory, so the
    branch is unreached today — but a guard that stops seeing a package the day
    it changes shape is a guard that reports green over it.
    """
    root = ROOT / cell_path
    if not root.is_dir():
        return []
    return sorted(
        child.stem if child.suffix == ".py" else child.name
        for child in root.iterdir()
        if not child.name.startswith((".", "_")) and (child.is_dir() or child.suffix == ".py")
    )


def _modules_with_code_disabled(code: str) -> set[str]:
    """Top-level module names for which ``code`` is switched off by an override.

    A pattern's leading segment is what identifies the package: ``dataknobs_fsm.*``
    and a hypothetical ``dataknobs_fsm.vector.*`` both name fsm, and either one
    leaves some of fsm's suppressions unreadable.
    """
    overrides = _load(ROOT / "pyproject.toml")["tool"]["mypy"].get("overrides", [])
    disabled: set[str] = set()
    for section in overrides:
        if code not in section.get("disable_error_code", []):
            continue
        module = section.get("module", [])
        patterns = [module] if isinstance(module, str) else module
        disabled.update(pattern.split(".", 1)[0] for pattern in patterns)
    return disabled


def test_ignore_without_code_tracks_the_adopted_set() -> None:
    """A package cannot be at tier ``strict`` with this rule switched off for it.

    ``ignore-without-code`` is enabled tree-wide and paused per package, because
    four of them hold a backlog of bare directives their ceilings already account
    for. Each pause is meant to end when its package is adopted: the override
    comes out, the directives get their codes, and the ceiling falls.

    Nothing in mypy notices when one does not. ``warn_unused_configs`` reports a
    section matching *no module*, and ``dataknobs_fsm.*`` goes on matching every
    fsm module for as long as fsm exists — so an override that has outlived its
    reason is indistinguishable, to mypy, from one still earning its place. The
    result would be a package the contract calls ``strict`` whose suppressions
    are still unreadable: a tier that reports clean because the rule that would
    dirty it is off, which is the failure this whole series is about.

    **Only this direction.** An override naming something that is not a module
    at all — a typo, or a package that has been deleted — is the case mypy *does*
    detect, as a note the contract's dead-override check reads and fails on. This
    guard covers the case that check cannot see, and asserting the converse here
    would duplicate it while forbidding a future pause on a cell that is not a
    package source root.
    """
    contract = json.loads((ROOT / ".dataknobs" / "quality-contract.json").read_text("utf-8"))
    cells = [
        cell
        for cell in contract["tools"]["mypy"]["cells"]
        if PurePosixPath(cell["path"]).match(PACKAGE_SOURCE_CELL)
    ]
    disabled = _modules_with_code_disabled(ADOPTION_ERROR_CODE)

    assert cells, (
        "no package source cell found in the mypy contract — either every "
        "package became unchecked or this guard stopped recognising the shape"
    )

    violations = [
        f"{cell['path']}: tier {cell['tier']!r}, but '{module}.*' still disables "
        f"{ADOPTION_ERROR_CODE} in pyproject.toml, so its bare directives are "
        f"unreachable and its ceiling of {cell['ceiling']} does not count them"
        for cell in sorted(cells, key=lambda c: str(c["path"]))
        if cell["tier"] == "strict"
        for module in _top_level_modules(cell["path"])
        if module in disabled
    ]

    assert not violations, (
        f"Adopted packages with {ADOPTION_ERROR_CODE} still switched off:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# --------------------------------------------------------------------------
# Interpreter pins
# --------------------------------------------------------------------------


def test_interpreter_pins_satisfy_floor(floor: tuple[int, int]) -> None:
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


def test_no_classifier_below_floor(floor: tuple[int, int]) -> None:
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


def test_new_package_template_declares_the_floor(floor: tuple[int, int]) -> None:
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


# --------------------------------------------------------------------------
# The verdict a checker reports, versus the one it reached
# --------------------------------------------------------------------------


def test_the_type_check_fails_when_mypy_does(tmp_path: Path) -> None:
    """A checker whose verdict ignores its own exit status reports only success.

    ``bin/validate.sh`` decided this by piping mypy into ``grep`` and testing the
    pipeline. The script sets ``pipefail``, and mypy exits non-zero exactly when
    it has findings — so on every real type error the pipeline status was
    non-zero, the ``if`` took the *else* branch, and the run printed "Type checks
    passed" directly beneath the errors grep had just echoed. ``FAILED`` was
    never set. The check could not fail, which is the shape the rest of this file
    exists to catch, in the script that does the checking.

    The probe is written outside the repository on purpose, and that is now a
    second property rather than an accident of using ``tmp_path``. A path in no
    cell has no ceiling to be within, so any finding there is a breach — while
    inside a cell the verdict is ``measured <= ceiling`` and a lone type error
    is absorbed by a backlog that already allows thousands. Both are correct;
    this pins the half where "any error fails" still holds.

    What is matched is the verdict line rather than a diagnosis of it. The
    contract's check now fails a cell measuring *below* its ceiling as well —
    progress nobody wrote down, not a type error — so a summary line naming the
    cause would be false for one of its two reasons. It names neither, and each
    reason prints its own sentence above it.
    """
    probe = tmp_path / "type_error_probe.py"
    probe.write_text("def broken() -> int:\n    return undefined_symbol_xyz\n", encoding="utf-8")

    result = subprocess.run(
        [str(ROOT / "bin" / "validate.sh"), str(probe)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr

    assert "Type check failed" in combined, (
        "bin/validate.sh reported no type-check failure for a file mypy rejects. "
        "Its mypy verdict must follow mypy's exit status, not a grep over output "
        "that `pipefail` then inverts.\n\n" + combined
    )


#: The states the gate's validation-scope decision can be in when it reaches the
#: VALIDATE_ARGS chain, and whether that state must validate something. Only the
#: explicit "nothing changed" case may validate nothing.
VALIDATION_SCOPE_STATES = (
    ("packages changed", "data llm", "no", "no", True),
    ("workspace-only change", "", "yes", "no", True),
    ("docs only", "", "no", "yes", False),
    ("change detection failed", "", "no", "no", True),
)


def test_every_state_that_should_validate_something_does() -> None:
    """A run that cannot tell what changed must not validate nothing and pass.

    When change detection fails the gate prints "testing all packages", and then
    fell through the whole VALIDATE_ARGS chain: PACKAGES is empty and neither
    skip flag is set, so no branch matched, VALIDATE_ARGS stayed empty, and the
    empty string is *also* how "nothing to validate" is spelled. The run
    validated no code — and reported PASS rather than PASS_WITH_SKIPS, because
    that needs SKIP_TESTS=yes, which this path never sets.

    Executed, not read. The decision is lifted out of the script and run under
    each state, so this asserts what the chain decides rather than which branches
    it appears to have.
    """
    gate = (ROOT / "bin" / "run-quality-checks.sh").read_text(encoding="utf-8")

    chain = re.search(
        r"^\s*VALIDATE_ARGS=\"\"\n(.*?)^\s*# Skip if no packages to validate",
        gate,
        re.MULTILINE | re.DOTALL,
    )
    assert chain is not None, (
        "cannot find the VALIDATE_ARGS decision in bin/run-quality-checks.sh; "
        "this guard reads it out of the script so it cannot drift, and it has."
    )
    condition = re.search(r"^\s*if \[ -n \"\$VALIDATE_ARGS\" \].*?; then$", gate, re.MULTILINE)
    assert condition is not None, "cannot find the gate's run-validation condition"

    wrong = []
    for label, packages, skip_package_tests, skip_tests, must_validate in VALIDATION_SCOPE_STATES:
        script = "\n".join(
            [
                f'PACKAGES="{packages}"',
                f'SKIP_PACKAGE_TESTS="{skip_package_tests}"',
                f'SKIP_TESTS="{skip_tests}"',
                'RUN_MODE="pr"',
                'VALIDATE_ARGS=""',
                chain.group(1),
                condition.group(0),
                '    echo "VALIDATES"',
                "else",
                '    echo "NOTHING"',
                "fi",
            ]
        )
        verdict = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        if (verdict == "VALIDATES") != must_validate:
            wrong.append(
                f"{label}: expected {'to validate' if must_validate else 'a skip'}, got {verdict}"
            )

    assert not wrong, "bin/run-quality-checks.sh decides the wrong validation scope:\n" + "\n".join(
        f"  - {item}" for item in wrong
    )


#: Where the size of ``PACKAGE_TEST_DOC_INPUTS`` is stated in prose, and the
#: shape each statement takes. Two files say it four times between them, in
#: sentences whose whole job is to convince a reader the exception is rare --
#: so a stale number does not merely age, it argues for the opposite of what
#: the tree contains.
_COUNT_CLAIMS = (
    ("bin/changed-packages.py", r"(?P<rest>\d+) of the (?P<total>\d+) are read only by"),
    ("bin/changed-packages.py", r"The (?P<declared>\w+) below are the exception"),
    (
        "tests/test_toolchain_consistency.py",
        r"default: (?P<rest>\d+) of the\n\s*(?P<total>\d+) here",
    ),
    ("tests/test_toolchain_consistency.py", r"The other\n\s*(?P<declared>\w+) are read by a test"),
)

#: Spelled out in the prose, so the comparison has to cross the same gap a
#: reader does. Only as far as the list can plausibly grow.
_WORDS = {
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _declared_door_names(package: str) -> int:
    """How many names ``packages/<package>``'s door declares in ``__all__``."""
    door = ROOT / "packages" / package / "src" / f"dataknobs_{package}" / "__init__.py"
    tree = ast.parse(door.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return len(node.value.elts)  # type: ignore[attr-defined]
    raise AssertionError(f"{door} declares no __all__")


def test_the_changelog_door_count_matches_the_door() -> None:
    """A release note counting the door's names is checked against the door.

    The same shape as the guard above and a different denominator, which is
    why it is a second test rather than a row in ``_COUNT_CLAIMS``. The claim
    is in ``packages/common/CHANGELOG.md``: an entry announcing that a family
    reached the package door, and saying how many names that left there.

    **Stale twice over before this existed.** The sentence said 315 while the
    tree said 319, and the change that found it added two more without
    touching the sentence. A count in a release note ages exactly like a count
    in a comment -- every later change to the same unreleased section moves it
    -- and this one is worse than a comment, because a consumer reads it as a
    description of what they will get.

    Scheduling runs the right way round. Editing the door maps to the common
    package and runs its suite; this guard lives here, where the workspace
    tier reaches it, so the case that matters -- a name added to ``__all__``
    without the note being corrected -- is the case that fails.
    """
    declared = _declared_door_names("common")
    changelog = (ROOT / "packages" / "common" / "CHANGELOG.md").read_text(encoding="utf-8")

    match = re.search(
        r"(?P<added>\d+) names, taking the package's `__all__` to (?P<total>\d+)", changelog
    )
    assert match is not None, (
        "packages/common/CHANGELOG.md no longer carries the sentence counting "
        "the door's names. Either it was rewritten -- in which case update "
        "this pattern -- or it was deleted, and this guard now checks nothing."
    )

    assert int(match.group("total")) == declared, (
        f"the CHANGELOG says the door carries {match.group('total')} names and "
        f"it carries {declared}. The entry describes what a consumer gets, so "
        f"a stale total there is a promise about a surface that is not the "
        f"one shipping."
    )


def _is_package_document(path: str) -> bool:
    """``packages/<pkg>/docs/**/*.md``, spelled against a root-relative path.

    The segment test rather than :func:`fnmatch.fnmatch`, whose ``*`` crosses
    ``/`` and would count every nested guide a second time under the
    top-level pattern.
    """
    parts = path.split("/")
    return len(parts) > 3 and parts[0] == "packages" and parts[2] == "docs" and path.endswith(".md")


def test_the_declarations_prose_counts_match_the_tree() -> None:
    """A count in a comment is a claim, and this is what makes it checkable.

    ``PACKAGE_TEST_DOC_INPUTS`` went from five entries to seven in the change
    that added the two guide rows, and all four sentences describing its size
    stayed behind -- in the same pull request that corrected an identical
    staleness two files away and wrote down the rule it was breaking: *"A count
    in a comment is a claim about the rest of the file, so it is corrected in
    the change that falsifies it rather than left to be noticed."*

    Noticing is what this replaces. The rule is right and it is not
    self-enforcing, and a claim about a number the tree already knows is the
    kind a test can hold.
    """
    declared = len(_scopes.PACKAGE_TEST_DOC_INPUTS)
    # Tracked rather than globbed. The claim is about the documents the
    # repository has, and a glob also counts an untracked scratch file left
    # under a package's ``docs/`` -- which would fail this guard with a
    # message about stale prose for a reason that has nothing to do with the
    # prose. ``tracked_files`` is what the guards either side of this one use.
    total = sum(1 for path in tracked_files() if _is_package_document(path))
    expected = {"declared": declared, "total": total, "rest": total - declared}

    for name, pattern in _COUNT_CLAIMS:
        text = (ROOT / name).read_text(encoding="utf-8")
        match = re.search(pattern, text)
        assert match is not None, (
            f"{name} no longer carries a sentence matching {pattern!r}. Either "
            f"the prose was rewritten -- in which case update this pattern -- "
            f"or the claim was deleted, and this guard is now checking one "
            f"file where it used to check two."
        )
        for group, claimed in match.groupdict().items():
            actual = _WORDS.get(claimed) if not claimed.isdigit() else int(claimed)
            assert actual == expected[group], (
                f"{name} says {claimed!r} where the tree says {expected[group]}: "
                f"PACKAGE_TEST_DOC_INPUTS declares {declared} of {total} package "
                f"documents. The sentence argues the exception is rare, so a "
                f"stale number there argues from the wrong figure."
            )

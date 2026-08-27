#!/usr/bin/env python3
"""Compute per-package content hashes for quality artifact validation.

Produces deterministic SHA-256 hashes of each package's quality-relevant
files (source, tests, pyproject.toml). Used by run-quality-checks.sh to
stamp artifacts and by validate-quality-artifacts.sh to detect staleness.

Reuses the dependency graph from changed-packages.py to compute the
transitive "dirty set" of packages needing re-validation.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES_DIR = _ROOT / "packages"
_ARTIFACTS_DIR = _ROOT / ".quality-artifacts"

# Quality-relevant file patterns per package (relative to package dir)
_HASH_PATTERNS = [
    ("src", "**/*.py"),
    ("tests", "**/*.py"),
    (".", "pyproject.toml"),
]

# The lines stripped before hashing because a release rewrites them without
# changing behaviour are declared in changed-packages.py and imported below,
# beside the workspace-input declaration this module already takes from there.
# They were defined here, where only this module could see them, and change
# detection reached the opposite answer about the same version bump as a result.

# Increment when the hashing algorithm changes (e.g., adding version stripping).
# A mismatch between stored and current version means hashes are incomparable
# and the artifacts should be treated as needing a fresh baseline.
_HASH_ALGORITHM_VERSION = 3


def load_module_from_path(name: str, script: Path) -> ModuleType:
    """Execute a script as a module, from its source rather than from bytecode.

    The compiled-cache step is skipped deliberately. CPython decides a
    ``__pycache__`` entry is current by comparing the source's size and its
    mtime **truncated to the second**, and compares nothing else, so two
    versions of a file that are the same length and are written inside the same
    second are indistinguishable to it: the second ``exec_module`` returns the
    *first* version's code while the file on disk holds the second's.

    What that costs here is not a testing inconvenience. This is how the hasher
    reaches ``changed-packages.py``, so a stale read is the dependency graph and
    the release-noise stripper answering from a version that has been replaced —
    and the gate deciding which packages need re-validating on that basis, while
    reporting nothing unusual.

    ``tests/_workspace.py`` carries the same function for the same reason. The
    duplication is not an oversight and cannot be removed by moving it: a
    ``bin/`` script's name is hyphenated, so reaching a sibling needs a loader
    already in hand, and a shared one could not itself be reached that way.
    ``test_no_third_loader_appears_without_the_same_treatment`` drives both and
    fails on a third, because the hazard in an irreducible twin is a fix that
    lands on one half — which is what happened to this half.
    """
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script}"
        raise ImportError(msg)
    # Removing a stale entry repairs this read; declining to write one stops the
    # next process inheriting the trap. Both halves are needed, and the second
    # is the one whose absence still passes a test that only reads.
    Path(importlib.util.cache_from_source(str(script))).unlink(missing_ok=True)
    module = importlib.util.module_from_spec(spec)
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = written
    return module


def _load_changed_packages() -> Any:
    """Import changed-packages.py (hyphenated name requires importlib)."""
    return load_module_from_path(
        "changed_packages", Path(__file__).resolve().parent / "changed-packages.py"
    )


# Load dependency graph utilities and the shared workspace-input declaration
_changed_packages = _load_changed_packages()
ALL_PACKAGES: list[str] = _changed_packages.ALL_PACKAGES
get_transitive_dependents = _changed_packages.get_transitive_dependents
WORKSPACE_QUALITY_INPUTS: dict[str, list[str]] = _changed_packages.WORKSPACE_QUALITY_INPUTS
GLOBAL_SCOPES: frozenset[str] = _changed_packages.GLOBAL_SCOPES
PACKAGE_TEST_DOC_INPUTS: dict[str, str] = _changed_packages.PACKAGE_TEST_DOC_INPUTS
strip_release_noise = _changed_packages.strip_release_noise


def _hash_files(files: list[Path], base: Path) -> str:
    """Hash relative paths and contents, so renames and deletions register too.

    Shared by the package and workspace scopes: both answer "did any input to
    the recorded result change", and a scope that hashed differently from its
    sibling would make the two incomparable for no reason.
    """
    hasher = hashlib.sha256()

    # Sort by relative path for cross-platform determinism
    for filepath in sorted(files, key=lambda f: str(f.relative_to(base))):
        rel_path = str(filepath.relative_to(base))
        content = filepath.read_bytes()

        # Strip release-time noise so version bumps don't dirty packages:
        #   - own version lines in pyproject.toml / __init__.py
        #   - cross-package dataknobs-* dep constraint lines in pyproject.toml
        # Both change at release time but don't reflect this package's behavior.
        # Shared with change detection, which decides whether that same bump
        # schedules a suite — one definition, so the two cannot disagree.
        filtered_content = strip_release_noise(content)

        # Hash path + content with null-byte separators to avoid ambiguity
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(filtered_content)
        hasher.update(b"\x00")

    return hasher.hexdigest()


def package_hash_files(package_name: str) -> list[Path]:
    """Every file one package's hash is computed over.

    Split out from compute_package_hash so a guard can ask "is this file
    covered?" through the same rule the hash uses, rather than restating it —
    the treatment scope_entry_files already gets on the workspace side, and for
    the same reason: a restatement answers about a rule the hasher does not
    follow.
    """
    pkg_dir = _PACKAGES_DIR / package_name
    if not pkg_dir.is_dir():
        return []

    found: list[Path] = []
    for subdir, glob_pattern in _HASH_PATTERNS:
        target = pkg_dir / subdir if subdir != "." else pkg_dir
        if target.exists():
            found.extend(target.glob(glob_pattern))

    # The documents a test in this package reads. _HASH_PATTERNS reaches the
    # .py files under src/ and tests/ and pyproject.toml, so a document was in
    # no package's hash at all — and these four decide whether their package's
    # suite passes. Left out, a verdict recorded before an edit to one still
    # validates after it, which is the staleness the hashes exist to catch.
    # Only the declared ones: folding in packages/*/docs wholesale would dirty
    # a package for a prose edit no test reads, which is the over-scheduling
    # this scope was narrowed to avoid.
    found.extend(
        _ROOT / document
        for document, owner in PACKAGE_TEST_DOC_INPUTS.items()
        if owner == package_name and (_ROOT / document).is_file()
    )
    return found


def compute_package_hash(package_name: str) -> str:
    """Compute a deterministic SHA-256 hash of a package's quality-relevant files."""
    pkg_dir = _PACKAGES_DIR / package_name
    if not pkg_dir.is_dir():
        return hashlib.sha256(b"missing").hexdigest()

    return _hash_files(package_hash_files(package_name), pkg_dir)


def scope_entry_files(entry: str) -> list[Path]:
    """Every file one declared entry expands to.

    The unit of expansion is the entry rather than the scope because two
    readers need different granularities: the hash wants a whole scope, while
    the guard that checks CI triggers on these files wants one probe per entry.
    That guard used to restate this rule instead of calling it, and a
    restatement answers for a rule the hasher does not follow.

    A directory entry may name several directories through a "*", which is how
    ``packages/*/docs/`` reaches all seven without listing them — a list would
    leave the eighth package's documentation silently unhashed.
    """
    if not entry.endswith("/"):
        target = _ROOT / entry
        return [target] if target.is_file() else []

    pattern = entry.rstrip("/")
    roots = (
        sorted(p for p in _ROOT.glob(pattern) if p.is_dir())
        if "*" in pattern
        else [_ROOT / pattern]
    )
    return [p for root in roots for p in root.rglob("*") if _is_quality_input(p)]


def workspace_scope_files(scope: str) -> list[Path]:
    """Every file one workspace scope actually hashes.

    Split out from compute_workspace_hash so a guard can ask "is this script
    covered?" through the same entry semantics the hash uses, rather than
    restating them. A restatement would answer the question about a rule the
    hasher does not follow, which is how a coverage check ends up passing for
    a file nothing hashes.
    """
    return [p for entry in WORKSPACE_QUALITY_INPUTS[scope] for p in scope_entry_files(entry)]


#: Suffixes carried by a file that feeds a recorded check. Code for the lint and
#: test verdicts; markup, stylesheet and script for the three documentation ones,
#: which read whole trees rather than named files; workflow YAML for workflow_lint,
#: whose entire input is one directory of them.
#:
#: This set is what a *directory* entry expands through, so it and the entries in
#: WORKSPACE_QUALITY_INPUTS are one decision written in two files. Declaring a
#: directory whose files this rejects yields a scope covering nothing, which reads
#: from the declaration exactly like a scope covering everything — the reason each
#: family of inputs has a guard asserting its own coverage rather than trusting
#: that the entry is there.
#:
#: Slightly over-inclusive in one direction: ``bin/README.md`` feeds no check but
#: sits under a hashed directory, so editing it invalidates the artifacts. Over-
#: inclusion costs a gate run that was not needed; under-inclusion costs a verdict
#: recorded over a tree that no longer produced it, which is the defect this whole
#: mechanism exists to prevent. A per-scope suffix set would be more machinery
#: than the single file it saves.
#:
#: Over-inclusion has a limit, though, and ``.txt`` is past it: one file under a
#: hashed directory feeds a check (bin/internal-label-allowlist.txt, which the
#: lint step honours) and the other feeds nothing, so admitting the suffix would
#: take a documentation file to reach a data one. It is named in the scope
#: instead. Naming what counts is also what keeps a developer's hash equal to
#: CI's, and that argument runs the same way here: a suffix admitted for one file
#: admits every stray of that kind an editor leaves behind.
_QUALITY_INPUT_SUFFIXES = frozenset({".py", ".sh", ".md", ".css", ".js", ".yml", ".yaml"})


def _is_quality_input(path: Path) -> bool:
    """Whether a file beneath a directory entry feeds a recorded check.

    A directory entry used to expand to ``*.py`` alone, which was right while
    the only checkers reading these directories were ruff and mypy. It stopped
    being right when the gate gained a shell lint: of the 46 shell scripts it
    reports on, only the seven named individually across the two scopes were
    covered. The other 39 — every remaining script in ``bin/``, ``bin/dk`` among
    them, plus both at the repository root — sat outside every scope, so editing
    one moved the recorded ``shell_lint`` verdict while leaving every stored hash
    intact. CI would then accept the artifact that the edit had just invalidated.

    It stopped being right a second time, the same way, when the documentation
    trees were declared: they are almost entirely ``*.md``.

    Extension is not sufficient, for the same reason it is not sufficient in
    lint-shell.sh: ``bin/dk`` carries none, and it is the entry point the rest
    are invoked through. So a shebang is read when the suffix does not answer.

    But only then, and that is where this stops agreeing with ``shell_targets``
    in lint-shell.sh — a similarity worth naming precisely, because the sentence
    above reads as if the two rules were the same. They are not. That one asks
    *is this a shell script* and reads a shebang under **any** suffix, so a file
    named ``.bash`` is a lint target; this asks *does this feed a recorded
    check* and reads one **only when there is no suffix**, so the same file is
    not a hash input. The asymmetry is deliberate: a suffix here is a positive
    statement that a kind of file feeds a check, and an unlisted suffix is an
    unhandled case rather than an excluded one. Neither rule is the bug when
    they disagree, and the shell lint's coverage guard —
    ``test_every_linted_shell_script_is_covered_by_a_hash_scope`` — is what
    reports the disagreement to someone who can decide which to change.

    Naming what counts, rather than taking everything not obviously junk, is
    also what keeps the answer identical on a developer's machine and on a CI
    checkout: ``.DS_Store`` reaches the shebang test and fails it, while
    ``user-guide.md.orig`` and ``user-guide.md~`` carry suffixes of their own and
    are rejected outright. A file git does not track cannot move the hash.
    """
    if not path.is_file():
        return False
    if path.suffix in _QUALITY_INPUT_SUFFIXES:
        return True
    if path.suffix:
        return False
    try:
        with path.open("rb") as handle:
            first = handle.readline()
    except OSError:
        return False
    return first.startswith(b"#!") and b"sh" in first


def compute_workspace_hash(scope: str) -> str:
    """Hash one workspace-level scope declared in WORKSPACE_QUALITY_INPUTS.

    These live outside packages/ and so were hashed by nothing, which meant a
    change to pytest.ini, .python-version, .pylintrc, the root pyproject.toml,
    or a workspace guard itself left every stored hash intact. CI would start
    the job, find nothing dirty, and pass on a stale artifact.

    A missing entry contributes nothing rather than raising: these are optional
    by nature (.pylintrc is absent in a fresh checkout of some branches), and
    its later appearance changes the hash on its own.
    """
    return _hash_files(workspace_scope_files(scope), _ROOT)


def compute_all_hashes() -> dict[str, str]:
    """Compute content hashes for all packages."""
    return {pkg: compute_package_hash(pkg) for pkg in ALL_PACKAGES}


def compute_all_workspace_hashes() -> dict[str, str]:
    """Compute content hashes for every workspace-level scope."""
    return {scope: compute_workspace_hash(scope) for scope in WORKSPACE_QUALITY_INPUTS}


def validate_artifacts() -> dict[str, Any]:
    """Compare current content hashes against stored artifact hashes.

    Uses the dependency graph to compute the transitive dirty set:
    any package whose content changed, plus all packages that depend
    on a changed package.

    Returns a structured result dict with validity status and details.
    """
    summary_path = _ARTIFACTS_DIR / "quality-summary.json"

    if not summary_path.exists():
        return {
            "valid": False,
            "error": "quality-summary.json not found",
            "changed_packages": [],
            "dirty_packages": [],
            "changed_scopes": [],
        }

    summary = json.loads(summary_path.read_text())
    stored_hashes = summary.get("package_hashes", {})

    if not stored_hashes:
        return {
            "valid": True,
            "warning": "No package_hashes in quality-summary.json — skipping hash validation",
            "changed_packages": [],
            "dirty_packages": [],
            "changed_scopes": [],
        }

    stored_algorithm = stored_hashes.pop("_algorithm_version", 1)
    if stored_algorithm != _HASH_ALGORITHM_VERSION:
        return {
            "valid": True,
            "warning": (
                f"Hash algorithm changed (stored: v{stored_algorithm}, "
                f"current: v{_HASH_ALGORITHM_VERSION}) — skipping hash validation"
            ),
            "changed_packages": [],
            "dirty_packages": [],
            "changed_scopes": [],
        }

    current_hashes = compute_all_hashes()

    # Find packages whose content has changed
    changed: set[str] = set()
    for pkg in current_hashes:
        if current_hashes[pkg] != stored_hashes.get(pkg):
            changed.add(pkg)

    # Workspace-level scopes. Absent on artifacts generated before these were
    # hashed at all — reported rather than treated as changed, because "this
    # run predates the check" and "this file was edited" are different facts
    # and only the second should fail a pull request.
    stored_workspace = summary.get("workspace_hashes")
    changed_scopes: set[str] = set()
    workspace_warning: str | None = None

    if stored_workspace is None:
        workspace_warning = (
            "No workspace_hashes in quality-summary.json — toolchain and workspace-test "
            "changes are unvalidated until the next full quality run"
        )
    else:
        current_workspace = compute_all_workspace_hashes()
        changed_scopes = {
            scope
            for scope, digest in current_workspace.items()
            if digest != stored_workspace.get(scope)
        }

    # A global scope changes lint, type, or test results everywhere, so every
    # package needs re-validation. A workspace-only scope moves no package's
    # result, so it invalidates the artifacts without dirtying a single suite —
    # that asymmetry is the whole reason the scopes are declared separately.
    if changed_scopes & GLOBAL_SCOPES:
        changed |= set(ALL_PACKAGES)

    # Compute transitive dirty set (changed + all downstream dependents)
    dirty = get_transitive_dependents(changed) if changed else set()

    overall_status = summary.get("overall_status", "")
    status_ok = overall_status in ("PASS", "PASS_WITH_SKIPS")

    result: dict[str, Any] = {
        "valid": len(dirty) == 0 and len(changed_scopes) == 0 and status_ok,
        "changed_packages": sorted(changed),
        "dirty_packages": sorted(dirty),
        "changed_scopes": sorted(changed_scopes),
        "status_ok": status_ok,
        "overall_status": overall_status,
    }
    if workspace_warning:
        result["warning"] = workspace_warning
    return result


def cmd_compute() -> None:
    """Print per-package content hashes as JSON (with algorithm version).

    The payload is widened rather than the version stringified: validate_artifacts
    pops this key and compares it to the int constant, so emitting "3" would make
    every stored artifact read as a different algorithm and silently skip hash
    validation entirely.
    """
    result: dict[str, Any] = dict(compute_all_hashes())
    result["_algorithm_version"] = _HASH_ALGORITHM_VERSION
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


def cmd_compute_workspace() -> None:
    """Print per-scope workspace content hashes as JSON.

    Emitted separately from the package hashes so each keeps its own shape:
    package hashes feed the dependency graph, workspace scopes carry their
    own blast radius and never enter it.
    """
    json.dump(compute_all_workspace_hashes(), sys.stdout, indent=2)
    sys.stdout.write("\n")


def changed_since(
    packages: dict[str, Any] | None, workspace: dict[str, Any] | None
) -> tuple[list[str], list[str]]:
    """Which recorded digests no longer describe the tree, and which were absent.

    Both arguments are documents this script printed earlier in the same run.
    Recomputing and comparing is how the gate tells "the tree I checked" from
    "the tree that happens to be here now": equal means every check read the
    content the artifact is about to attest; unequal means it did not, for the
    named scopes.

    An equality of two digests rather than an ordering of two clocks. Comparing
    file mtimes against the ``.run-in-progress`` timestamp would be cheaper and
    is available for free, but mtime is settable, checkouts and rebases rewrite
    it, and editors rewrite it on saves that change nothing — more *sensitive*
    and less *reliable*, and a check that fails spuriously ends up suppressed.

    Returns ``(moved, unchecked)``. The second list is what keeps an empty
    document from reading as a clean comparison: the gate falls back to ``{}``
    when a hash computation fails, and a half with no recorded digests has
    nothing to compare rather than nothing wrong. Silently returning "no
    movement" for it would be this repository's own defect class — an absence
    rendered as a pass.

    Names are qualified because the two namespaces overlap in kind but not in
    meaning, and a bare name would leave the reader guessing which moved.
    """
    moved: list[str] = []
    unchecked: list[str] = []

    # Dropped rather than compared: the algorithm version is a property of this
    # script, not of the tree, so it cannot move during a run and naming it in a
    # diff would only mislead.
    recorded = {k: v for k, v in packages.items() if k != "_algorithm_version"} if packages else {}
    if recorded:
        current = compute_all_hashes()
        moved += sorted(
            f"packages/{name}"
            for name in set(recorded) | set(current)
            if recorded.get(name) != current.get(name)
        )
    else:
        unchecked.append("packages")

    if workspace:
        current_workspace = compute_all_workspace_hashes()
        moved += sorted(
            f"workspace/{scope}"
            for scope in set(workspace) | set(current_workspace)
            if workspace.get(scope) != current_workspace.get(scope)
        )
    else:
        unchecked.append("workspace")

    return moved, unchecked


def cmd_changed_since(packages_json: str | None, workspace_json: str | None) -> None:
    """Print every recorded scope the tree moved out from under; exit 1 if any.

    Three outcomes, three exit codes, because the caller has to respond to them
    differently: 0 the digests still hold, 1 the tree moved and the run must be
    repeated, 2 the comparison could not be made at all. Collapsing the last two
    would send a developer to re-run the gate over a bug in the gate.

    A half with no recorded digests is reported rather than skipped in silence.
    """
    try:
        packages = json.loads(packages_json) if packages_json else None
        workspace = json.loads(workspace_json) if workspace_json else None
    except json.JSONDecodeError as exc:
        logger.error("Recorded hashes are not readable JSON: %s", exc)
        sys.exit(2)

    if packages is None and workspace is None:
        logger.error("Nothing to compare — pass --packages, --workspace, or both")
        sys.exit(2)

    moved, unchecked = changed_since(packages, workspace)
    for half in unchecked:
        logger.warning("No %s hashes were recorded, so nothing about them was re-checked", half)
    for name in moved:
        print(name)
    sys.exit(1 if moved else 0)


def cmd_validate(*, use_json: bool = False) -> None:
    """Validate that artifacts match current source content."""
    result = validate_artifacts()

    if use_json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        # A warning is additive, not an alternative outcome. Reporting it in
        # place of a failure would hide the failure while still exiting 1 —
        # a red check that says nothing about why, which this repo has had.
        if result.get("warning"):
            logger.warning("%s", result["warning"])

        if result.get("error"):
            logger.error("Validation error: %s", result["error"])
        elif result["valid"]:
            if result["dirty_packages"]:
                logger.info("All dirty packages have been tested")
            else:
                logger.info("All packages unchanged since last quality run")
        else:
            logger.error("Quality artifacts are stale")
            if result["changed_scopes"]:
                logger.error("Changed workspace scopes: %s", ", ".join(result["changed_scopes"]))
            if result["changed_packages"]:
                logger.error("Changed packages: %s", ", ".join(result["changed_packages"]))
            if result["dirty_packages"]:
                logger.error(
                    "Packages needing re-validation: %s",
                    ", ".join(result["dirty_packages"]),
                )

    sys.exit(0 if result["valid"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and validate per-package content hashes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("compute", help="Print per-package content hashes as JSON")

    subparsers.add_parser(
        "compute-workspace", help="Print per-scope workspace content hashes as JSON"
    )

    changed_parser = subparsers.add_parser(
        "changed-since",
        help="Report which recorded hashes no longer match the tree",
    )
    changed_parser.add_argument(
        "--packages", help="the package-hash document recorded earlier this run"
    )
    changed_parser.add_argument(
        "--workspace", help="the workspace-hash document recorded earlier this run"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate artifacts against current source content"
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        dest="use_json",
        help="Output structured JSON result",
    )

    args = parser.parse_args()

    if args.command == "compute":
        cmd_compute()
    elif args.command == "compute-workspace":
        cmd_compute_workspace()
    elif args.command == "changed-since":
        cmd_changed_since(args.packages, args.workspace)
    elif args.command == "validate":
        cmd_validate(use_json=args.use_json)


if __name__ == "__main__":
    main()

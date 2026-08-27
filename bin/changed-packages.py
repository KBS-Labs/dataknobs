#!/usr/bin/env python3
"""Detect changed packages and their dependents for targeted testing.

Analyzes git changes to determine which packages need testing,
computing the transitive closure of dependents via the dependency graph.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Root of the repository
_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES_DIR = _ROOT / "packages"

# Regex to extract dataknobs-<name> from dependency strings like:
#   "dataknobs-common>=1.0.1",
_DK_DEP_RE = re.compile(r'"dataknobs-([a-z]+)')


def discover_dependencies() -> dict[str, list[str]]:
    """Build the dependency graph by parsing each package's pyproject.toml.

    Returns a dict mapping package short name to the list of internal
    dataknobs package short names it depends on.
    """
    deps: dict[str, list[str]] = {}
    for pyproject in sorted(_PACKAGES_DIR.glob("*/pyproject.toml")):
        pkg_name = pyproject.parent.name
        internal_deps: list[str] = []
        in_deps_section = False
        for line in pyproject.read_text().splitlines():
            stripped = line.strip()
            if stripped == "dependencies = [":
                in_deps_section = True
                continue
            if in_deps_section:
                if stripped == "]":
                    break
                m = _DK_DEP_RE.search(stripped)
                if m:
                    dep_name = m.group(1)
                    if dep_name != pkg_name:  # skip self-references
                        internal_deps.append(dep_name)
        deps[pkg_name] = sorted(internal_deps)
    return deps


# Discover at import time — this script is short-lived (CLI tool)
DEPENDENCIES = discover_dependencies()

# All valid package names
ALL_PACKAGES = sorted(DEPENDENCIES.keys())

# ---------------------------------------------------------------------------
# Workspace-level quality inputs
# ---------------------------------------------------------------------------
#
# Every file outside packages/ that can change a quality result, declared once
# with the blast radius it carries. Four things read this: change detection
# below, artifact freshness (bin/package-hashes.py), the CI path filter
# (.github/workflows/quality-validation.yml — bridged by a guard, since Actions
# cannot import Python), and tests/test_toolchain_consistency.py.
#
# They used to be four hand-maintained lists, and they disagreed. That is how a
# change to a type-checker config could match no CI pattern, leave every
# artifact hash untouched, and report green through both mechanisms meant to
# catch it.
#
# Splitting by blast radius is what keeps the fix from overcorrecting. Marking
# everything global would re-run ten package suites because someone fixed a
# typo in a guard's docstring; marking nothing global is the hole above.
_GLOBAL_QUALITY_INPUTS = [
    "pyproject.toml",  # root ruff + mypy config, and the dependency set
    "uv.lock",  # the resolved versions every package is tested against
    "conftest.py",  # root fixtures, on the path of every test run
    "pytest.ini",  # testpaths, addopts, and asyncio_mode for every run
    ".python-version",  # the interpreter itself
    # The two scripts that *are* the lint and test steps. Every package's
    # recorded result is whatever these produced, so a change to either makes
    # all ten stale — the same blast radius as the config they read, which is
    # already listed above. They sit here rather than under bin/ below because
    # that tier is for inputs no package result depends on, and these are the
    # inputs every package result depends on.
    "bin/validate.sh",  # the validation step: ruff, mypy, import checks
    "bin/test.sh",  # the test step: selection, markers, coverage flags
    # Sourced by both of the above, and it answers the two questions that decide
    # what they act on: which packages exist, and which code belongs to none of
    # them. It sits in this tier rather than the workspace one for the same
    # reason validate.sh does — it moves every package's recorded result, not
    # just the artifact, and the "bin/" entry in the workspace tier would put it
    # in the wrong tier rather than in none.
    "bin/package-discovery.sh",  # which packages exist, and what else to check
]

# Reachable only by the workspace guards, so a change here cannot move any
# package's result. .pylintrc qualifies because no gate step runs pylint —
# it is read by `bin/dk lint`, tox, and the guard that asserts its py-version.
#
# bin/ qualifies for the same reason from the other direction: the guards under
# tests/ read these scripts — the gate, change detection, the doc-mirror check —
# so a change here moves their result and nothing else's. Without the entry a
# change to the gate itself matched no pattern, and the run it edited skipped
# every test while reporting success.
#
# The two readers used to disagree about what "beneath" covers: change
# *detection* below matches by path prefix, so bin/*.sh counted, while the
# artifact *hash* scope in package-hashes.py globbed "*.py", so a shell-only
# change triggered the guards without dirtying the stored hash. The scripts that
# produce and verify the artifact are shell and landed in exactly that gap —
# editing the gate ran the guards but left the hash intact, and the artifact
# written under the old rules still validated under the new ones.
#
# That gap is closed: workspace_scope_files now expands a directory entry
# through _is_quality_input, which reaches shell as well as Python. The bare "*"
# this comment used to argue against would have swept __pycache__ and moved
# every stored hash on a stray import; a predicate does not.
#
# So the named shell entries below are no longer what makes those files hashed.
# They are still load-bearing, for a different reason worth stating because it
# is not visible from here: they are the only non-.py probes in
# _workspace_input_probes, which is what proves CI's path filter starts the
# quality job for a change to them. Delete them as redundant and the filter can
# lose "bin/**" with nothing reporting it.
#
# A file entry is matched exactly by both readers, so listing one is unambiguous
# in a way a directory entry is not.
_WORKSPACE_ONLY_QUALITY_INPUTS = [
    ".pylintrc",
    "bin/",
    "tests/",
    # The whole input to the recorded workflow_lint check, and a directory entry
    # rather than six names so a seventh workflow is covered on arrival. Note
    # what makes that work: a directory entry expands through the suffix
    # predicate in package-hashes.py, so this entry and ".yml"/".yaml" being
    # quality-input suffixes are one decision written in two files — declared
    # here while the predicate rejected them, it would have expanded to nothing
    # and read exactly like coverage.
    ".github/workflows/",
    # Named individually, per the note above: these are the probes that prove
    # CI's path filter covers non-Python files. Each decides what the gate checks
    # or records without moving any package's own result — a suite that passed
    # still passes, but the verdict about it was computed by different rules, so
    # the artifact has to be regenerated under the new ones.
    "bin/run-quality-checks.sh",  # writes the artifact CI validates
    "bin/validate-quality-artifacts.sh",  # the checks CI actually runs
    "bin/docs-update-versions.sh",  # the documentation_versions check it records
    "bin/lint-workflows.sh",  # the workflow_lint check it records
    "bin/lint-shell.sh",  # the shell_lint check it records
    # Shell scripts at the repository root, which no directory entry above
    # reaches. Both are reported on by the shell lint, so editing one moves the
    # recorded shell_lint verdict; setup-dk.sh is also the installer the
    # contributing docs tell a new developer to run.
    "run_api.sh",
    "setup-dk.sh",
    # Data files that decide a recorded check's answer without being code, and
    # so reached by no directory entry's suffix predicate. Each sits beside
    # something already hashed, which is what kept them out of view: the guard
    # is hashed but the file it reads was not, and the script is hashed but the
    # list it consults was not. Editing one moves a recorded verdict with every
    # stored hash intact — the same sentence as a shell script outside every
    # scope, one layer in.
    ".gitignore",  # what three artifact-contract guards are a verdict about
    ".gitattributes",  # ditto, for the merge-driver guard
    "bin/internal-label-allowlist.txt",  # suppressions the lint step honours
    ".dataknobs/quality-contract.json",  # the ceilings the contract check compares against
    # The root README, read by the documented-import guard along with every
    # package README and the site tree. The per-package copies ride their own
    # package scope and docs/ rides the docs scope; this one is reached by no
    # directory entry at all. It belongs to this tier rather than the docs one
    # despite being documentation: DOCS_PATTERNS does not list it, so the
    # shadowing hazard the note below describes does not apply, and the check it
    # actually moves is a workspace guard rather than one of the three recorded
    # documentation checks.
    "README.md",
]

# The inputs to the three documentation checks the gate records: the site build,
# the version-table sync, and the dual-docs mirror. Like the tier above these
# move no package's result, so they invalidate the artifacts without dirtying a
# suite.
#
# They were in no scope at all, and the consequence went past staleness. CI's
# docs job asks package-hashes.py whether anything is dirty and skips the build
# when nothing is, so a documentation-only pull request left every hash intact,
# had its stored "documentation: pass" accepted over a tree that no longer
# produced it, and skipped the job that would have rebuilt the site. Three
# mechanisms agreed to check nothing. Reproduced with a broken intra-doc link,
# which `mkdocs build --strict` rejects and both paths passed.
#
# Deliberately NOT added to _WORKSPACE_ONLY_QUALITY_INPUTS, though they are
# workspace-only in blast radius. That list is also what change detection
# matches, and map_files_to_packages tests it *before* DOCS_PATTERNS and stops
# at the first hit — so filing them there would stop setting docs_changed,
# which is what makes the gate re-run the very checks this scope exists to keep
# honest.
#
# Note what the hazard is and is not. A docs edit *is* a workspace-guard change
# and the mapping now says so: the guards under tests/ read every document.
# What must not happen is that becoming the *only* thing it says, because the
# two facts have different populations — a changelog re-runs the documentation
# checks and is read by no guard. Setting both flags in the DOCS_PATTERNS
# branch keeps them independent; moving the entry would collapse them.
#
# Two file entries rather than ".dataknobs/", which also holds notes and an
# example workflow that feed no check.
_DOCS_QUALITY_INPUTS = [
    "mkdocs.yml",  # the site build's own configuration
    "docs/",  # the site tree
    "packages/*/docs/",  # symlinked and transcluded into the tree above
    ".dataknobs/docs-mirror-manifest.json",  # what documentation_mirrors reads
    ".dataknobs/packages.json",  # what documentation_versions compares against
]

# Files that trigger testing all packages. Only the global tier: a workspace-only
# input still invalidates the artifacts, but through the workspace hash scope
# rather than by dirtying every package. See bin/package-hashes.py.
GLOBAL_TRIGGERS = list(_GLOBAL_QUALITY_INPUTS)

# The workspace-only tier, matched rather than merely declared. Three readers
# consulted the list above and a fourth — the mapping below — did not, which is
# how a diff touching only tests/ came out as "no quality input changed" and
# skipped the very guards it edited.
WORKSPACE_ONLY_TRIGGERS = list(_WORKSPACE_ONLY_QUALITY_INPUTS)

#: Every workspace-level input, by scope name. Consumed by package-hashes.py to
#: hash each scope separately and by the toolchain guards to assert that CI
#: triggers on all of them. Directory entries end in "/" and may name several
#: directories through a "*"; what "beneath" covers differs by reader — hashing
#: takes the files that feed a check, change detection takes every path under
#: the prefix. See the caveat on _WORKSPACE_ONLY_QUALITY_INPUTS.
WORKSPACE_QUALITY_INPUTS: dict[str, list[str]] = {
    "toolchain": _GLOBAL_QUALITY_INPUTS,
    "workspace_tests": _WORKSPACE_ONLY_QUALITY_INPUTS,
    "docs": _DOCS_QUALITY_INPUTS,
}

#: Scopes whose change invalidates every package's result rather than only the
#: workspace guard suite. package-hashes.py reads this to size the dirty set.
GLOBAL_SCOPES = frozenset({"toolchain"})

# ---------------------------------------------------------------------------
# Release-time noise
# ---------------------------------------------------------------------------
#
# The lines release-helper.sh rewrites when it bumps a version: a package's own
# version, and the cross-package dataknobs-* constraints bumped alongside it.
# Neither says anything about how the code behaves — the depended-on package is
# hashed independently and reaches its dependents through the transitive-dirty
# graph, so the constraint string adds no signal the graph does not already
# carry.
#
# These lived in package-hashes.py, which strips them precisely so a version
# bump does not dirty a package. Change detection does not import that module
# and so did not strip them, and the two readers ended up disagreeing about what
# a version bump *is*: the hasher proved not one package input had moved while
# change detection, matching on paths alone, scheduled all ten suites for the
# same diff. Declared here, in the module both readers already share, so the
# question has one answer instead of two that drift.
#
# The regexes match a stripped line, so leading indentation is already gone by
# the time they are applied.
_VERSION_LINE_RE = re.compile(r'^(?:version\s*=\s*"[^"]*"|__version__\s*=\s*"[^"]*")\s*$')
_DEP_CONSTRAINT_LINE_RE = re.compile(r'^"dataknobs-[a-z]+(?:>=|==)[^"]+",?$')


def strip_release_noise(content: bytes) -> bytes:
    """Drop the release-rewritten lines from a file's content.

    Returns a comparison key, not a rendering: two files agree on behaviour
    when their stripped forms are equal.

    Decoded *and re-encoded* with ``surrogateescape`` so content that is not
    valid UTF-8 round-trips instead of raising. The decode always tolerated it
    and the encode did not, which left the pair able to raise on a file the
    workspace scopes reach through their suffix predicate. Bytes are unchanged
    for any input that encoded cleanly before — which is every input that
    hashes today, since the alternative was a crash — so no stored hash moves.
    """
    decoded = content.decode("utf-8", errors="surrogateescape")
    kept = [
        line
        for line in decoded.splitlines(keepends=True)
        if not _VERSION_LINE_RE.match(line.strip())
        and not _DEP_CONSTRAINT_LINE_RE.match(line.strip())
    ]
    return "".join(kept).encode("utf-8", errors="surrogateescape")


#: Documentation that sits at a package root rather than under its ``docs/``.
#: Reached by no hash scope — not ``_HASH_PATTERNS`` in package-hashes.py (the
#: ``.py`` files under ``src/`` and ``tests/``, plus ``pyproject.toml``) and not
#: the ``packages/*/docs/`` entry in the docs scope — so a change here moves no
#: recorded verdict about the package, and no test reads one. Every CHANGELOG
#: mention under ``tests/`` is a citation in a docstring.
#:
#: Deliberately not extended to ``README.md``: a package README is a candidate
#: for transclusion into the site in a way a changelog is not, and the entry
#: that would make that safe is a docs-scope one, not this.
_PACKAGE_DOC_FILES = frozenset({"CHANGELOG.md"})

#: Package documentation that a test in that package's own suite reads, mapped
#: to the package whose result it decides.
#:
#: Almost no package document is one of these. 138 of the 142 are read only by
#: the workspace guards — which check every document's imports, configuration
#: keys, tool names and fenced samples against the code — and by the three
#: documentation checks the gate records. None of that is a package's suite, so
#: mapping a document to its package scheduled that whole suite, and its
#: dependents, for a prose edit that could not move any of their results. A link
#: repair touching two packages' docs ran two full test suites and no guard that
#: reads a link.
#:
#: The four below are the exception and they are a real one: each is read by a
#: test *in* the package, comparing a published table against the code it
#: describes, so the document genuinely decides whether that suite passes. They
#: keep scheduling their package, and package-hashes.py folds them into that
#: package's hash for the same reason — a verdict recorded before an edit to one
#: must not validate after it.
#:
#: Declared rather than detected because the two cases are indistinguishable
#: from a path. What keeps the list honest is
#: ``test_every_package_document_a_package_suite_reads_is_declared``, which finds
#: them structurally — a ``Path(__file__)`` expression divided by ``"docs"`` —
#: and fails on a fifth. A naive search for the string is not available: 192
#: lines under ``packages/*/tests`` mention ``"docs"``, and all but these name a
#: knowledge source or a RAG adapter.
PACKAGE_TEST_DOC_INPUTS: dict[str, str] = {
    "packages/bots/docs/multi-tenant.md": "bots",
    "packages/bots/docs/behavior-packs.md": "bots",
    "packages/common/docs/guides/packs.md": "common",
    "packages/data/docs/batch-processing-guide.md": "data",
}


# Paths whose change means the gate should re-run the documentation checks.
# Matched by prefix, so a full path names exactly one file.
#
# Distinct from the "docs" hash scope above, and easy to conflate: that one
# decides whether a *stored* verdict still describes the tree, this one decides
# whether *this run* recomputes it. Neither subsumes the other — package sources
# belong here in effect (mkdocstrings renders them) while being hashed per
# package, and the reverse omission is worse: an input hashed but not matched
# here goes stale, prompts a re-run, and is then re-stamped with the verdict
# nothing recomputed. That is what the two .dataknobs entries were.
#
# packages/*/docs is absent because map_files_to_packages recognises it
# separately — it has to, since these are prefixes and that shape needs a glob.
DOCS_PATTERNS = [
    "docs/",
    "mkdocs.yml",
    ".dataknobs/docs-mirror-manifest.json",
    ".dataknobs/packages.json",
]


def _run_git(*args: str) -> list[str]:
    """Run a git command and return non-empty output lines."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def _resolve_base_ref(base_ref: str) -> str:
    """Resolve the base ref, preferring the remote-tracking branch.

    When the user passes "main", we want "origin/main" so that change
    detection works even when the local branch is behind the remote.
    Falls back to the original ref if the remote variant doesn't exist.
    """
    # If already a remote ref or explicit path, use as-is
    if "/" in base_ref:
        return base_ref

    # Try origin/<ref> first
    remote_ref = f"origin/{base_ref}"
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", remote_ref],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return remote_ref

    return base_ref


def _blob_at(ref: str, path: str) -> bytes | None:
    """One path's content at a git ref, or ``None`` when it is absent there.

    Run from the repository root so the repo-relative paths ``git diff
    --name-only`` produces resolve the same way whatever directory the caller
    invoked the script from.
    """
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        check=False,
        cwd=_ROOT,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _worktree_bytes(path: str) -> bytes | None:
    """One path's content in the working tree, or ``None`` when unreadable."""
    try:
        return (_ROOT / path).read_bytes()
    except OSError:
        return None


def drop_release_noise_only(files: list[str], resolved_ref: str) -> list[str]:
    """Drop files whose only difference from *resolved_ref* is release noise.

    Applied to the raw git answer so every later reader — the global-trigger
    check, the workspace-input check, the package mapping — sees one list of
    files that *materially* changed. Filtering here rather than at each of the
    three is what keeps them from disagreeing again: a version bump rewrites
    ``uv.lock`` (a global trigger) as well as every package's ``pyproject.toml``
    and ``__init__.py``, so a fix applied only to the package mapping would
    still have scheduled all ten suites through the global trigger.

    A file that was added or removed is kept without comparison: absence on one
    side is a real change, and the noise patterns describe an edit to a line
    that exists on both.

    Note the asymmetry with the hasher this shares its definition with. The
    hasher decides *membership* too — it hashes only the ``.py`` files under
    ``src/`` and ``tests/`` — and deferring to that would be unsafe here,
    because a test input it does not hash (a golden JSON file, a YAML fixture)
    still decides whether a suite passes. So this shares the "what counts as a
    change" half and not the "what counts as an input" half: a fixture edit
    keeps scheduling its package's suite even though it moves no stored hash.
    """
    material: list[str] = []

    for path in files:
        before = _blob_at(resolved_ref, path)
        after = _worktree_bytes(path)

        if before is None or after is None:
            material.append(path)
            continue

        if strip_release_noise(before) != strip_release_noise(after):
            material.append(path)

    return material


def get_changed_files(base_ref: str) -> list[str]:
    """Get all changed files: committed on branch, staged, and unstaged.

    Release noise is removed before the list is returned, so a caller asking
    "what changed" is told what changed in a sense every reader of the answer
    agrees on. See :func:`drop_release_noise_only`.
    """
    files: set[str] = set()

    resolved_ref = _resolve_base_ref(base_ref)

    # Changes committed on branch vs base
    files.update(_run_git("diff", "--name-only", f"{resolved_ref}...HEAD"))

    # Staged changes
    files.update(_run_git("diff", "--name-only", "--cached"))

    # Unstaged changes
    files.update(_run_git("diff", "--name-only"))

    # Untracked files (new files not yet staged)
    files.update(_run_git("ls-files", "--others", "--exclude-standard"))

    return drop_release_noise_only(sorted(files), resolved_ref)


def build_reverse_graph() -> dict[str, list[str]]:
    """Build reverse dependency graph: package -> packages that depend on it."""
    reverse: dict[str, list[str]] = {pkg: [] for pkg in DEPENDENCIES}
    for pkg, deps in DEPENDENCIES.items():
        for dep in deps:
            reverse[dep].append(pkg)
    return reverse


def get_transitive_dependents(packages: set[str]) -> set[str]:
    """Compute transitive closure of all packages that depend on the given set."""
    reverse = build_reverse_graph()
    result = set(packages)
    queue = list(packages)

    while queue:
        pkg = queue.pop()
        for dependent in reverse.get(pkg, []):
            if dependent not in result:
                result.add(dependent)
                queue.append(dependent)

    return result


def _is_workspace_only_input(filepath: str) -> bool:
    """Whether a path is a workspace-only quality input.

    Directory entries end in "/" and cover everything beneath them; file
    entries match exactly. Spelled to the same convention
    WORKSPACE_QUALITY_INPUTS documents, so the list stays the declaration.
    """
    return any(
        filepath.startswith(entry) if entry.endswith("/") else filepath == entry
        for entry in WORKSPACE_ONLY_TRIGGERS
    )


def map_files_to_packages(files: list[str]) -> tuple[set[str], bool, bool, bool]:
    """Map changed files to affected packages.

    Returns:
        (directly_changed_packages, docs_changed, all_packages_triggered,
         workspace_only_changed)
    """
    changed_packages: set[str] = set()
    docs_changed = False
    all_triggered = False
    workspace_changed = False

    for filepath in files:
        # Check for global triggers
        if filepath in GLOBAL_TRIGGERS:
            all_triggered = True
            continue

        # Check for workspace-only inputs. These belong to no package, so they
        # move no package's result — but they are still a quality input, and
        # the guards under tests/ are the ones that check the toolchain.
        if _is_workspace_only_input(filepath):
            workspace_changed = True
            continue

        # Documentation. Two flags, because two different things read it and
        # they disagree about a changelog: docs_changed re-runs the three
        # recorded documentation checks, while workspace_changed schedules the
        # guards under tests/ — four of which read every document in the
        # repository and check its imports, configuration keys, tool names and
        # fenced samples against the code. Only the second was missing, and its
        # absence is what let a documentation-only change classify as "none"
        # and switch off the very guards that would have read the edited file.
        if any(filepath.startswith(pattern) for pattern in DOCS_PATTERNS):
            docs_changed = True
            workspace_changed = True
            continue

        # Package documentation. It belongs to no package's suite unless a test
        # in that suite reads it — see PACKAGE_TEST_DOC_INPUTS, which is the
        # whole of the exception and is checked against the tree rather than
        # trusted. Without the `continue` every package document mapped to its
        # package below, which ran that suite and its dependents for a prose
        # edit while running none of the four workspace guards that read it.
        if "/docs/" in filepath and filepath.startswith("packages/"):
            docs_changed = True
            workspace_changed = True
            owner = PACKAGE_TEST_DOC_INPUTS.get(filepath)
            if owner is not None:
                changed_packages.add(owner)
            continue

        # Package-root documentation, which the mapping below would otherwise
        # read as a change to the package itself — scheduling its whole suite
        # to publish a release note. See _PACKAGE_DOC_FILES for why nothing
        # about a package's recorded verdict depends on one.
        if filepath.startswith("packages/") and filepath.rsplit("/", 1)[-1] in _PACKAGE_DOC_FILES:
            docs_changed = True
            continue

        # Map to package
        if filepath.startswith("packages/"):
            parts = filepath.split("/")
            if len(parts) >= 2:
                pkg_name = parts[1]
                if pkg_name in DEPENDENCIES:
                    changed_packages.add(pkg_name)

    return changed_packages, docs_changed, all_triggered, workspace_changed


def classify_test_scope(packages: list[str], workspace_changed: bool) -> str:
    """Which suites a change set needs run: "packages", "workspace" or "none".

    An empty package list used to be read as "run nothing". That is right for
    a change touching no quality input and wrong for one touching only the
    workspace guards, which belong to no package by construction and so map
    to an empty list exactly like a no-op diff does. Naming the two cases
    apart is what lets the gate skip the per-package suites without also
    skipping the suite the change edited.

    Documentation reaches the same answer through the same flag, and the
    mapping above is where it is set: four guards under ``tests/`` read every
    document in the repository, so a document is a workspace-guard input in
    exactly the way ``tests/`` and ``bin/`` are. It is *also* an input to the
    three recorded documentation checks, which is a separate fact carried by a
    separate flag — the two questions have different answers for a changelog,
    which re-runs the documentation checks and is read by no guard.
    """
    if packages:
        return "packages"
    if workspace_changed:
        return "workspace"
    return "none"


def plan_for_files(files: list[str]) -> dict[str, Any]:
    """Decide what a change set needs tested, without consulting git.

    Split out from detect_changes so the decision is reachable from a test
    with a literal file list. The git half is what made the previous
    behaviour awkward to pin, and it is the decision that was wrong.

    Returns dict with:
        packages: sorted list of package names that need testing
        docs_changed: whether docs-related files changed
        directly_changed: packages with direct file changes
        mode: "all" if global trigger hit, "changed" otherwise
        workspace_changed: whether a workspace-only quality input changed
        test_scope: "packages", "workspace" or "none" (see classify_test_scope)
    """
    if not files:
        return {
            "packages": [],
            "docs_changed": False,
            "directly_changed": [],
            "mode": "none",
            "workspace_changed": False,
            "test_scope": "none",
        }

    (
        directly_changed,
        docs_changed,
        all_triggered,
        workspace_changed,
    ) = map_files_to_packages(files)

    if all_triggered:
        packages = list(ALL_PACKAGES)
        mode = "all"
    else:
        # Compute transitive dependents, filtered to packages that exist
        all_affected = get_transitive_dependents(directly_changed)
        packages = sorted(pkg for pkg in all_affected if pkg in DEPENDENCIES)
        mode = "changed"

    return {
        "packages": packages,
        "docs_changed": docs_changed,
        "directly_changed": sorted(directly_changed),
        "mode": mode,
        "workspace_changed": workspace_changed,
        "test_scope": classify_test_scope(packages, workspace_changed),
    }


def detect_changes(base_ref: str = "main") -> dict[str, Any]:
    """Detect changed packages and docs status. See plan_for_files."""
    return plan_for_files(get_changed_files(base_ref))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect changed packages for targeted testing")
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Git ref to compare against (default: main)",
    )
    args = parser.parse_args()

    result = detect_changes(args.base_ref)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

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
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, NamedTuple

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
    # Sourced by validate.sh, and it answers the two questions that decide what
    # that script acts on: which packages exist, and which code belongs to none
    # of them. It sits in this tier rather than the workspace one for the same
    # reason validate.sh does — it moves every package's recorded result, not
    # just the artifact, and the "bin/" entry in the workspace tier would put it
    # in the wrong tier rather than in none.
    #
    # This said "sourced by both of the above" until the step classification
    # below went looking. bin/test.sh does not source it — it has its own
    # discover_test_packages loop over packages/* — and the difference is not a
    # nicety: it is the whole reason this entry and validate.sh are classified
    # lint-only, so the sentence that was wrong is the one a reader would have
    # used to argue they are not.
    # test_the_test_step_does_not_read_the_lint_only_inputs holds it.
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
    ".dataknobs/release-readiness.json",  # the pointers release-helper.sh reads and verifies
    # The licensing surface, read by test_licensing.py: the root LICENSE and
    # NOTICE, each package's copy of the pair pinned byte-for-byte against
    # them, and the historical MIT text whose per-package version list that
    # guard checks against packages.json. Editing one moves that guard's verdict and
    # no package's, which is what this tier is for.
    #
    # File entries rather than a "LICENSES/" directory entry, for the reason the
    # workflows note above gives in the other direction: a directory entry
    # expands through the suffix predicate in package-hashes.py, and ".txt" is
    # not a quality-input suffix, so it would expand to nothing and read exactly
    # like coverage. A file entry is tested with is_file() and never consults
    # that predicate, which is also what lets the extensionless names here be
    # declared at all.
    #
    # The copies are globbed rather than listed one per package, for the
    # reason scope_entry_files gives about "packages/*/docs/": a list leaves
    # the next package's copy unhashed the day it is created, while still
    # reading like coverage. They are the guard's actual subject, so leaving
    # them out is the whole of what it asserts about.
    "LICENSE",
    "NOTICE",
    "LICENSES/MIT-historical.txt",
    "packages/*/LICENSE",
    "packages/*/NOTICE",
    # The two module docstrings that pay for one family being split across two
    # packages: the four identity keys in common, the fifth embedder key in
    # data, and a cross-reference in each direction so a reader who reaches
    # either module finds the other. A workspace guard asserts both directions,
    # because a guard inside packages/common/tests asserting anything about
    # dataknobs_data would import the package two guards there exist to keep
    # out of common's import graph.
    #
    # The first entries in this tier that DO move a package's result, so the
    # sentence every note above ends with does not apply to them and is not
    # repeated. They are declared anyway, and the reason is the one this whole
    # mechanism is about: a package scope covering the source looks like
    # coverage of the workspace guard that reads it, and those are different
    # sets. The cost of the redundancy is a gate run that was not needed, which
    # is the direction _QUALITY_INPUT_SUFFIXES already says to err in.
    "packages/common/src/dataknobs_common/ontology/tags.py",
    "packages/data/src/dataknobs_data/vector/content.py",
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
# Kept out of _WORKSPACE_ONLY_QUALITY_INPUTS, though they are workspace-only in
# blast radius. The reason used to be an ordering: that list is also what change
# detection matches, every branch ended in `continue`, and the workspace tier was
# tested first — so filing them there stopped them setting docs_changed, which is
# what makes the gate re-run the very checks this scope exists to keep honest.
#
# That ordering is gone: map_files_to_packages accumulates, so a file declared in
# two tiers now contributes to both and filing an entry here is no longer a way
# to un-declare it there. What remains is a plain scoping question — these feed
# the three recorded documentation checks, so they are hashed under "docs" where
# a docs-only change invalidates them without dirtying the guard suite's scope.
#
# Note what the hazard is and is not. A docs edit *is* a workspace-guard change
# and the mapping now says so: the guards under tests/ read every document.
# What must not happen is that becoming the *only* thing it says, because the
# two facts have different populations — a changelog re-runs the documentation
# checks and does not dirty a package. Setting both flags in the DOCS_PATTERNS
# branch keeps them independent; moving the entry would collapse them.
#
# A changelog *is* read by a guard now, which is a hashing question rather than
# a tier one: `packages/*/CHANGELOG.md` is in the docs hash scope below, so
# editing one invalidates the artifact, while the tier it matches is unchanged
# and it still re-runs the documentation checks.
#
# Two file entries rather than ".dataknobs/", which also holds notes and an
# example workflow that feed no check.
_DOCS_QUALITY_INPUTS = [
    "mkdocs.yml",  # the site build's own configuration
    "docs/",  # the site tree
    "packages/*/docs/",  # symlinked and transcluded into the tree above
    ".dataknobs/docs-mirror-manifest.json",  # what documentation_mirrors reads
    ".dataknobs/packages.json",  # what documentation_versions compares against
    # Read by a workspace guard, which is why it is hashed: a release note
    # counting the names on a package door is checked against that door by
    # test_the_changelog_door_count_matches_the_door. Hashed *here* rather
    # than in the workspace-only tier for the reason the note above gives --
    # which is now a scoping one rather than an ordering one, since the mapping
    # accumulates and a second tier no longer silences the first.
    #
    # Named individually rather than as "packages/*/CHANGELOG.md", because a
    # "*" expands only in a *directory* entry: scope_entry_files tests a file
    # entry with is_file(), so the glob would resolve to nothing and read
    # exactly like coverage -- the hazard the workflows note above describes.
    # The guard that found this derives its population from the guards' own
    # source, so a guard reading a second package's changelog fails until that
    # one is named too.
    "packages/common/CHANGELOG.md",
]

# Files that trigger testing all packages. Only the global tier: a workspace-only
# input still invalidates the artifacts, but through the workspace hash scope
# rather than by dirtying every package. See bin/package-hashes.py.
GLOBAL_TRIGGERS = list(_GLOBAL_QUALITY_INPUTS)

# ---------------------------------------------------------------------------
# Which recorded step a global input can move
# ---------------------------------------------------------------------------
#
# The gate records two per-package results, produced by two different scripts:
# the validation row comes from bin/validate.sh, the unit and integration rows
# from bin/test.sh. "Global" says a change can invalidate a result no package's
# own content explains. It does not say *which* result, and most of these
# inputs feed exactly one of the two.
#
# 09e1dbc5 is the commit that put the step scripts in this tier, and it argued
# the point rather than assuming it: "bin/validate.sh and bin/test.sh ARE the
# lint and test steps, so every package's recorded result is whatever they
# produced", demonstrated on a branch fixing a runner that exited 0 on its
# second target. Read precisely, that sentence is per step — each script owns
# the results *it* produced — and the tier had no way to say so, so both landed
# on everything. This table is that sentence with the step named.
#
# The split is checkable rather than asserted, which is what makes it safe to
# act on: bin/test.sh has its own discover_test_packages loop over packages/*
# and neither sources bin/package-discovery.sh nor invokes bin/validate.sh, so
# no edit to either can reach a test result.
# test_the_test_step_does_not_read_the_lint_only_inputs fails if that stops
# being true, and the gate never passes -f, so validate.sh cannot rewrite the
# tree the tests then run against either.
LINT_STEP = "lint"
TEST_STEP = "test"
BOTH_STEPS = frozenset({LINT_STEP, TEST_STEP})

#: Every entry in GLOBAL_TRIGGERS, by the step it can move. Checked for
#: exhaustiveness against that list by the toolchain guards, so a new global
#: input fails the build until the decision is made — the same shape 09e1dbc5
#: chose when it scoped its coverage guard to force the tiering decision in
#: review "instead of leaving it to whoever edits the script next".
GLOBAL_TRIGGER_STEPS: dict[str, frozenset[str]] = {
    # The linters' configuration and the resolved versions both steps run
    # against. pyproject.toml is read further than the path: see
    # pyproject_steps_changed for the sections that move only the lint half.
    "pyproject.toml": BOTH_STEPS,
    "uv.lock": BOTH_STEPS,
    # The interpreter: mypy's target version and the runtime under test.
    ".python-version": BOTH_STEPS,
    # Test-step inputs. None of the three has fired in the last 150 merged
    # pull requests, so this classification buys nothing measurable today; it
    # is here because the table is a declaration of what is true, and a table
    # that records only the profitable half is one nobody can check.
    "conftest.py": frozenset({TEST_STEP}),
    "pytest.ini": frozenset({TEST_STEP}),
    "bin/test.sh": frozenset({TEST_STEP}),
    # The lint step, and the discovery it sources. This is where the measured
    # return is: 39 of the 61 test suite-runs this change removes.
    "bin/validate.sh": frozenset({LINT_STEP}),
    "bin/package-discovery.sh": frozenset({LINT_STEP}),
}

#: The hash-scope name for each step set a global input can declare. The widest
#: one keeps the name "toolchain": the scope predates the step classification,
#: and renaming it would move a stored digest for every global input rather
#: than only for the ones leaving it.
_GLOBAL_SCOPE_NAMES: dict[frozenset[str], str] = {
    BOTH_STEPS: "toolchain",
    frozenset({LINT_STEP}): "toolchain_lint",
    frozenset({TEST_STEP}): "toolchain_test",
}


def _global_scopes_by_step() -> dict[str, list[str]]:
    """Partition the global inputs into one hash scope per step they move.

    The scope is what the artifact stores and package-hashes.py compares, so
    while there was one of them a moved digest could say only that *something*
    global had changed -- over inputs whose recorded consequences the table
    above had already been made to tell apart. An edited bin/validate.sh and an
    edited conftest.py produced identical evidence, under a comment claiming
    both moved "lint, type, or test results everywhere", which stopped being
    true of the lint-only pair on the day that table was written.

    Derived rather than declared again. A hand-kept list here would be a fifth
    reader of the blast-radius question, and the four that already existed
    disagreeing is the defect this declaration was built to end. A new global
    input reaches the right scope by declaring its step, and declaring one is
    not optional -- the toolchain guards fail on a trigger missing from the
    table.

    Fails closed at both lookups. An undeclared input takes BOTH_STEPS, the
    same default map_files_to_packages applies, and an unrecognised step set
    takes the widest scope: a gap here over-dirties and the guard names it,
    rather than raising at import time in a module four programs load.
    """
    grouped: dict[str, list[str]] = {}
    for entry in GLOBAL_TRIGGERS:
        steps = GLOBAL_TRIGGER_STEPS.get(entry, BOTH_STEPS)
        grouped.setdefault(_GLOBAL_SCOPE_NAMES.get(steps, "toolchain"), []).append(entry)
    return grouped


#: The global tier, split by the recorded step its members move. A step nothing
#: declares contributes no scope rather than an empty one: an empty scope hashes
#: to the same digest every run and compares equal forever, which is what a
#: scope being checked also looks like.
_GLOBAL_SCOPE_INPUTS: dict[str, list[str]] = _global_scopes_by_step()

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
#:
#: The global tier arrives already split by step, so the key that moved names
#: which recorded result went stale rather than only that one did.
WORKSPACE_QUALITY_INPUTS: dict[str, list[str]] = {
    **_GLOBAL_SCOPE_INPUTS,
    "workspace_tests": _WORKSPACE_ONLY_QUALITY_INPUTS,
    "docs": _DOCS_QUALITY_INPUTS,
}

#: Scopes whose change invalidates every package's result rather than only the
#: workspace guard suite. package-hashes.py reads this to size the dirty set.
#:
#: Every member of the split global tier is one, and the width is unchanged by
#: the split: a lint-only input moves every package's recorded validation
#: result, so all ten are still dirty. What the split changes is which claim a
#: moved digest carries, not how many packages it names.
GLOBAL_SCOPES = frozenset(_GLOBAL_SCOPE_INPUTS)

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

#: The licensing files every package carries a copy of. Not documentation —
#: no recorded documentation check reads either — but like the set above they
#: sit at a package root and belong to no package's suite, so the generic
#: path-to-package rule must not claim them. The guard that does read them is
#: tests/test_licensing.py, which the workspace-only tier schedules.
_PACKAGE_LICENSE_FILES = frozenset({"LICENSE", "NOTICE"})

#: Directories directly under a package whose contents that package's own suite
#: reads and no other package can. A change to one schedules that package and
#: stops there, rather than dragging the dependents the transitive closure
#: exists to reach.
#:
#: The closure earns its cost on a change to what a package *exports*: edit
#: common's source and the other nine are running against different code, so
#: their recorded results say nothing until they re-run. A test file is not
#: that. The gate runs each package's suite as its own pytest invocation, so no
#: other package's run collects these files at all — and that is checked rather
#: than assumed, by test_no_package_suite_reads_another_packages_tests, because
#: it is not true by construction: a tests directory goes on sys.path under
#: pytest's default import mode, so two packages could share a helper through a
#: bare import that spells neither package's name. Measured at zero, out of 995
#: test modules offering 935 importable names.
#:
#: ``pyproject.toml`` is the case that shows this is about ``tests/`` rather
#: than about "not src": it sits at the package root, no suite reads it as a
#: test input, and a dependency constraint or a version in it is read by every
#: dependent's resolution. It exports, so it is not here.
_LOCAL_ONLY_PACKAGE_DIRS = frozenset({"tests"})

#: Package documentation that a test in that package's own suite reads, mapped
#: to the package whose result it decides.
#:
#: Almost no package document is one of these. 141 of the 152 are read only by
#: the workspace guards — which check every document's imports, configuration
#: keys, tool names and fenced samples against the code — and by the three
#: documentation checks the gate records. None of that is a package's suite, so
#: mapping a document to its package scheduled that whole suite, and its
#: dependents, for a prose edit that could not move any of their results. A link
#: repair touching two packages' docs ran two full test suites and no guard that
#: reads a link.
#:
#: The eleven below are the exception and they are a real one: each is read by a
#: test *in* the package, comparing a published table against the code it
#: describes --- or, for the last of them, checking that no such comparison is
#: owed --- so the document genuinely decides whether that suite passes. They
#: keep scheduling their package, and package-hashes.py folds them into that
#: package's hash for the same reason — a verdict recorded before an edit to one
#: must not validate after it.
#:
#: Declared rather than detected because the two cases are indistinguishable
#: from a path. What keeps the list honest is
#: ``test_every_package_document_a_package_suite_reads_is_declared``, which finds
#: them structurally — a ``Path(__file__)`` expression divided by ``"docs"`` —
#: and fails on one this list does not carry. A naive search is not available: 192
#: lines under ``packages/*/tests`` mention ``"docs"``, and all but these name a
#: knowledge source or a RAG adapter.
PACKAGE_TEST_DOC_INPUTS: dict[str, str] = {
    "packages/bots/docs/multi-tenant.md": "bots",
    "packages/bots/docs/behavior-packs.md": "bots",
    "packages/common/docs/guides/packs.md": "common",
    # Both publish a `worked-input` fence holding a vocabulary the common suite
    # also carries as a conftest constant, and test_worked_input_fences.py is
    # what stops the two copies parting. Read from the package side rather than
    # from tests/ deliberately: the comparison needs the conftest constant, and
    # a workspace guard reading a package's test file sits in no workspace hash
    # scope. The second half of that argument has expired — filing the conftest
    # in the workspace-only tier used to stop it scheduling its own package,
    # because the mapping tested that tier first and stopped, and it no longer
    # does. Reading from this side still costs nothing and covers both halves
    # with rules that already exist, so the placement stands on the first half
    # alone; see test_a_file_can_feed_more_than_one_tier for what is now legal.
    "packages/common/docs/guides/ontology.md": "common",
    "packages/common/docs/guides/entity-resolution.md": "common",
    # A third `worked-input` fence, and half of a pair: `MAMMALS_GUIDE_DOCUMENT`
    # holds this vocabulary, and content-tags.md below publishes it too, so the
    # common suite reads this one for the comparison. It said the opposite until
    # recently -- that no constant held it -- and predicted its own expiry in
    # the saying: the day a constant does hold this document, a comparison row
    # is owed and `UNPAIRED` in test_worked_input_fences.py has gone stale. That
    # day was the change that added content-tags.md, which added the row and
    # corrected `UNPAIRED` and left this sentence claiming what it had just
    # falsified. That read is what earns the entry; an unread document would
    # not want one.
    "packages/common/docs/guides/anchored-view.md": "common",
    # The fourth `worked-input` fence, and the only one that is not half of a
    # pair: it publishes the smallest of the four vocabularies and no constant
    # holds it. Read by the common suite to check that claim against the tree
    # rather than take the declaration's word for it -- which is the read
    # anchored-view.md above earned until a constant came to hold its
    # vocabulary, and the reason `UNPAIRED` now names this document alone.
    "packages/common/docs/guides/hierarchy.md": "common",
    # The fifth `worked-input` fence, and the second half of a pair whose first
    # half used to be a single copy: it publishes the same vocabulary
    # anchored-view.md does, deliberately, so a second service-free acceptance
    # runs against one substrate rather than two. Read by the common suite for
    # the comparison, and by the workspace runner that executes the call site.
    "packages/common/docs/guides/content-tags.md": "common",
    # The sixth `worked-input` fence, and the third row naming one constant:
    # this page publishes the same vocabulary anchored-view.md and
    # content-tags.md do, so a third service-free acceptance runs against one
    # substrate rather than three. Read by the common suite for the comparison,
    # and by the workspace runner that executes the roll-up call site.
    "packages/common/docs/guides/roll-up.md": "common",
    "packages/data/docs/batch-processing-guide.md": "data",
    "packages/data/docs/vector-store-capabilities.md": "data",
}


#: Directories beneath a package's ``tests/`` holding what its suite reads
#: rather than what pytest collects: a golden answer, a YAML configuration, the
#: markdown a knowledge source ingests. ``_HASH_PATTERNS`` in package-hashes.py
#: reaches ``tests/**/*.py``, which is the suite and not the suite's inputs, so
#: editing one of these changed what the tests assert while every stored hash
#: stayed intact — a recorded ``pass`` surviving an edit to the thing it was a
#: verdict about. The golden file is the case that names itself: its entire
#: purpose is to be the expected answer.
#:
#: Unlike the declaration above, this one buys no scheduling and needs none. A
#: file under ``packages/<p>/tests/`` already maps to ``<p>`` by the generic
#: rule in map_files_to_packages, so the edit was already running that suite;
#: what was missing is only the stored verdict moving with it. Hashing these
#: therefore costs no gate run that the edit does not already cause, which is
#: the whole of why they may be taken a directory at a time where a document
#: had to be argued for one at a time. There, the cost of being wrong was a
#: suite scheduled for prose; here there is no such cost.
#:
#: Directories for a second reason too: the files are not all nameable. config
#: and llm do name theirs — ``Path(__file__).parent / "fixtures" /
#: "test_config.yaml"``, and the golden answer beside it — but bots hands the
#: whole ``packages/bots/tests/test_docs/`` directory to a knowledge source and
#: never names a document inside it, and utils names three of its four through a
#: conftest helper composed with a module constant. The fourth, a gzipped
#: ``.json``, appears in no test at all: it is read by the one that walks the
#: directory asking which of its files are gzipped. So a list of files would be
#: hand-maintained for two of the four packages and would have missed that one
#: outright. The directory is what every reader does name, which is also what
#: lets one be checked against the tree rather than believed.
#:
#: Deliberately absent: a README beside a test package. Two exist under
#: packages/data/tests/ and no test reads either, so neither moves a verdict
#: and hashing one would dirty a package for prose. They are named one at a
#: time rather than left to a rule about the word README, and
#: ``test_every_test_input_a_suite_reads_is_in_that_packages_hash`` is what
#: holds the exception to exactly those two: a third file feeding nothing
#: fails there on arrival instead of joining a pattern.
PACKAGE_TEST_FIXTURE_DIRS: tuple[str, ...] = (
    "packages/bots/tests/test_docs",
    "packages/config/tests/fixtures",
    "packages/llm/tests/golden",
    "packages/utils/tests/resources",
)


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


# ---------------------------------------------------------------------------
# What a lock change actually moved
# ---------------------------------------------------------------------------
#
# uv.lock is a global trigger because a resolution change really can reach
# every package: a bumped third-party version is installed for all ten, and
# the path says nothing about which. But the file is not opaque, and the
# blast radius it carries is written inside it. Each workspace member is its
# own ``[[package]]`` block marked ``source = { editable = "packages/<name>" }``,
# so a diff confined to those blocks names the packages it moved.
#
# This is the same move drop_release_noise_only above already makes — read the
# content rather than trusting the path — one step further. That filter drops
# uv.lock when the *whole* diff is release noise; measured over 150 merged
# pull requests it did so twice, while uv.lock still reached the global
# trigger ten times and was the sole trigger in eight of them. The remainder
# is not noise, so dropping the file would be wrong; it is a real change that
# names the packages it reaches.
#
# Kept here rather than widened into strip_release_noise deliberately. That
# function is shared with the hasher (package-hashes.py imports it by name, so
# the two cannot disagree about what a change *is*), and uv.lock is hashed in
# the "toolchain" scope — so teaching it the lock's spellings would move every
# stored workspace hash for a scheduling fix. Scheduling is the question here,
# so the reading stays on the scheduling side.
class TriggerScan(NamedTuple):
    """What reading a global input said it actually moved.

    Two axes, because the two readable inputs narrow different ones. ``uv.lock``
    names *packages* and leaves both steps in range; the root ``pyproject.toml``
    names *steps* and leaves every package in range. One type carrying both
    rather than one parameter each, so a third reader is a row in
    :data:`_TRIGGER_READERS` instead of a fourth keyword argument threaded
    through three functions — the shape this module would otherwise grow one
    special case at a time.

    ``packages`` is ``None`` when the content said nothing about which packages
    are affected, which is not the same as saying none are. An *empty* set is
    the reading a silently-broken parser produces for every input, so the
    caller treats it as unattributed too — see map_files_to_packages, which is
    the one place that rule is written.
    """

    #: The recorded steps this change can have moved.
    steps: frozenset[str]
    #: The packages it moved, or None for "the content did not say".
    packages: frozenset[str] | None


_LOCK_FILE = "uv.lock"

#: The marker that makes a ``[[package]]`` block a workspace member, and names
#: the directory it is. Read from the block rather than derived from the
#: distribution name, because the two do not match: ``name = "dataknobs"`` is
#: ``packages/legacy``. ``source = { editable = "." }`` is the workspace root,
#: which owns no package's code and so matches this deliberately-narrow pattern
#: not at all — it is unattributable, and handled as such below.
_LOCK_MEMBER_RE = re.compile(r'^source = \{ editable = "packages/([^"/]+)" \}\s*$', re.M)

#: The block header of one locked distribution, and the name line that follows.
_LOCK_BLOCK = "[[package]]"
_LOCK_NAME_RE = re.compile(r'^name = "([^"]+)"\s*$')

#: ``resolution-markers`` is a set that the format writes as a list, and uv
#: reorders it on its own. Three of the release pull requests measured differed
#: from their base in nothing else at the top of the file — the same reordering
#: each time, an identical set of markers — which is a diff that means nothing
#: and, read literally, made those releases behave differently from the two
#: that happened not to get reordered. Compared as the set it is, so one PR
#: class gets one answer.
_LOCK_MARKERS_RE = re.compile(r"^(resolution-markers = \[\n)(.*?)(^\]$)", re.M | re.S)


def _canonical_lock(text: str) -> str:
    """Sort the one list in a lock file whose order carries no meaning."""

    def _sorted_markers(match: re.Match[str]) -> str:
        entries = sorted(line for line in match.group(2).splitlines() if line.strip())
        return match.group(1) + "".join(f"{line}\n" for line in entries) + match.group(3)

    return _LOCK_MARKERS_RE.sub(_sorted_markers, text)


def _toml_blocks(text: str, opens: Callable[[str], bool]) -> list[list[str]]:
    """Split a TOML file into blocks, each opened by a line ``opens`` accepts.

    The line walk both readers below need, and the only thing they share: what
    a block *means* is the part that differs, so the callers key the blocks and
    this returns them in file order. Block 0 is whatever precedes the first
    opener, which both callers treat as the preamble and neither lets a package
    or a section claim.

    ``opens`` sees the line with its newline stripped and nothing else, so a
    line's indentation is what keeps a bracketed value inside a multi-line
    array from reading as a table header — a top-level header is at column
    zero, and TOML does not require the contents of an array to be.
    """
    blocks: list[list[str]] = [[]]
    for line in text.splitlines(keepends=True):
        if opens(line.rstrip("\n")):
            blocks.append([])
        blocks[-1].append(line)
    return blocks


def _opens_lock_region(stripped: str) -> bool:
    """A ``[[package]]`` block, or any top-level table that is not its subtable."""
    if stripped == _LOCK_BLOCK:
        return True
    return stripped.startswith("[") and not stripped.startswith("[package.")


def _lock_regions(text: str) -> dict[str, str]:
    """Every top-level region of a lock file, keyed by what identifies it.

    A ``[[package]]`` block is keyed by its ``name``; every other top-level
    table by its header; everything before the first of either by "". Tables
    spelled ``[package.*]`` are sub-tables and stay with the block they
    qualify, which is what keeps a dependency edit inside its own package
    rather than reading as a region of its own.

    Two blocks may share a name — uv emits one per resolution fork — so a
    repeated key accumulates rather than overwriting. A change in either half
    then still shows up as a difference in the whole, which is the answer that
    keeps this safe rather than the one that makes it precise.

    A block whose name cannot be read keys on its position. It is then its own
    region either way, and position is the stabler of the two spellings: keying
    on a running count made an unnamed block's key depend on how many regions
    happened to precede it.
    """
    regions: dict[str, str] = {}
    for index, block in enumerate(_toml_blocks(text, _opens_lock_region)):
        if not block:
            continue
        head = block[0].rstrip("\n")
        if index == 0:
            key = ""
        elif head == _LOCK_BLOCK:
            key = next(
                (
                    match.group(1)
                    for line in block
                    if (match := _LOCK_NAME_RE.match(line.rstrip("\n")))
                ),
                f"\x00unnamed:{index}",
            )
        else:
            key = head
        regions[key] = regions.get(key, "") + "".join(block)
    return regions


def lock_members_changed(before: bytes, after: bytes) -> frozenset[str] | None:
    """Which workspace members' resolutions moved between two lock files.

    Returns ``None`` when the answer is "more than the workspace can account
    for" — a third-party block moved, the preamble moved, the manifest moved,
    a block could not be attributed to a package directory. That is the honest
    reading of a bumped shared dependency: it is installed for every package
    and the lock does not say which of them care.

    **The empty set is also ``None``'s answer at the call site, and
    deliberately so.** A lock that survived drop_release_noise_only and then
    named no member at all is a file this reader did not account for, and the
    shape of that mistake is the dangerous one: a parser that silently matched
    nothing would return an empty set for every input, which reads as "no
    package is affected" and — when the lock is the only material file —
    becomes an empty schedule and a skipped test suite reporting success. This
    module's own history has that failure twice over. So the caller treats
    empty as unattributed; see map_files_to_packages.
    """
    try:
        before_text = _canonical_lock(before.decode("utf-8", errors="surrogateescape"))
        after_text = _canonical_lock(after.decode("utf-8", errors="surrogateescape"))
    except (UnicodeDecodeError, ValueError):  # pragma: no cover - defensive
        return None

    before_regions = _lock_regions(before_text)
    after_regions = _lock_regions(after_text)

    moved: set[str] = set()
    for name in set(before_regions) | set(after_regions):
        old = before_regions.get(name)
        new = after_regions.get(name)
        if old == new:
            continue
        member = _LOCK_MEMBER_RE.search(new if new is not None else old or "")
        if member is None:
            return None
        package = member.group(1)
        if package not in DEPENDENCIES:
            return None
        moved.add(package)

    return frozenset(moved)


def lock_scan_for(resolved_ref: str) -> TriggerScan | None:
    """Read the lock at both ends and say which members it moved.

    Separated from the decision so plan_for_files stays a function of its
    arguments: this is the half that touches git, and it is the half a test
    cannot pin without a repository.

    An empty answer is passed along rather than converted to ``None`` here.
    Both mean "unattributed" and the caller is where that is decided, so this
    stays a reader and the rule keeps one home.
    """
    before = _blob_at(resolved_ref, _LOCK_FILE)
    after = _worktree_bytes(_LOCK_FILE)
    if before is None or after is None:
        return None
    members = lock_members_changed(before, after)
    if members is None:
        return None
    # Both steps: a member's resolution decides what its dependents lint
    # against and what they run against, and the lock says nothing that
    # separates the two.
    return TriggerScan(steps=BOTH_STEPS, packages=members)


# ---------------------------------------------------------------------------
# What a pyproject change actually moved
# ---------------------------------------------------------------------------
#
# The root pyproject.toml is a global trigger because it carries the dependency
# set and the uv workspace declaration, and a change to either really does
# reach every package through both steps. It also carries the ruff and mypy
# configuration, which is the whole of what most changes to it touch: measured
# over 150 merged pull requests it moved ten times, and nine of those ten were
# confined to [tool.ruff] or [tool.mypy].
#
# A linter's configuration decides what the *validation* step reports and
# nothing a test can observe. So a diff confined to those tables keeps its full
# width on the lint side — every package is re-validated under the new rules —
# and schedules no package test suite.
#
# Read by region rather than by diff hunk, the same way the lock is: comparing
# whole top-level tables needs no hunk parser and no line arithmetic, and it
# cannot mis-attribute a line to the table above it.
_PYPROJECT_FILE = "pyproject.toml"

#: The tables whose content only the lint step reads. Deliberately the two
#: tool configurations and nothing else — [project], [dependency-groups] and
#: [tool.uv] decide what is installed, which both steps run against, and
#: [tool.pytest.ini_options] would be the test step's alone if this file
#: carried one (pytest.ini does).
_LINT_ONLY_TABLES = frozenset({"tool.ruff", "tool.mypy"})


def _opens_toml_table(stripped: str) -> bool:
    """A top-level table header, at column zero."""
    return stripped.startswith("[")


def _pyproject_table(header: str) -> str:
    """The region a table header belongs to.

    ``[tool.ruff.lint.per-file-ignores]`` is part of ``tool.ruff``: a tool's
    configuration is one document however deeply it is spelled, and grouping
    by the first two components is what makes a per-file-ignore edit read as
    the ruff change it is rather than as a table of its own.
    """
    parts = header.strip().strip("[]").split(".")
    return ".".join(parts[:2]) if parts[0] == "tool" else parts[0]


def _pyproject_regions(text: str) -> dict[str, str]:
    """Every top-level region of a pyproject file, keyed by the table it is.

    Everything before the first header keys on "", like the lock reader: the
    preamble is a region nobody's table may claim, so an edit to it is a
    difference the caller cannot attribute — which is the answer that keeps
    this fail-closed.
    """
    regions: dict[str, str] = {}
    for index, block in enumerate(_toml_blocks(text, _opens_toml_table)):
        if not block:
            continue
        key = "" if index == 0 else _pyproject_table(block[0].rstrip("\n"))
        regions[key] = regions.get(key, "") + "".join(block)
    return regions


def pyproject_steps_changed(before: bytes, after: bytes) -> TriggerScan | None:
    """Which recorded steps a root pyproject diff can have moved.

    Returns ``None`` for anything this cannot place in a linter's table — a
    dependency bump, a workspace member added, the preamble, a table it does
    not recognise. ``None`` means the entry keeps the blast radius its path has
    always carried, which is both steps over every package.

    **An empty answer is ``None``'s answer too**, for the reason
    lock_members_changed states at length: a reader that silently matched
    nothing returns "no table moved" for every input, and "no table moved"
    would otherwise read as "no step moved" and schedule nothing at all.
    """
    try:
        before_regions = _pyproject_regions(before.decode("utf-8", errors="surrogateescape"))
        after_regions = _pyproject_regions(after.decode("utf-8", errors="surrogateescape"))
    except (UnicodeDecodeError, ValueError):  # pragma: no cover - defensive
        return None

    moved = {
        table
        for table in set(before_regions) | set(after_regions)
        if before_regions.get(table) != after_regions.get(table)
    }
    if not moved or not moved <= _LINT_ONLY_TABLES:
        return None
    # Every package, still: a ruff or mypy rule change is re-validated across
    # the whole tree. What it cannot move is any package's test result.
    return TriggerScan(steps=frozenset({LINT_STEP}), packages=None)


def pyproject_scan_for(resolved_ref: str) -> TriggerScan | None:
    """Read the root pyproject at both ends and say which steps it moved."""
    before = _blob_at(resolved_ref, _PYPROJECT_FILE)
    after = _worktree_bytes(_PYPROJECT_FILE)
    if before is None or after is None:
        return None
    return pyproject_steps_changed(before, after)


#: The global inputs that can report on themselves, and what reads each. A
#: third readable input is a row here; nothing else in the module changes.
_TRIGGER_READERS: dict[str, Callable[[str], TriggerScan | None]] = {
    _LOCK_FILE: lock_scan_for,
    _PYPROJECT_FILE: pyproject_scan_for,
}


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
    entries match exactly, or segment-wise when they carry a "*". Spelled to
    the same convention WORKSPACE_QUALITY_INPUTS documents, so the list stays
    the declaration.

    The "*" is matched per segment rather than with a bare fnmatch over the
    whole path, because fnmatch's "*" crosses a separator: it would read
    ``packages/a/b/LICENSE`` as a match for ``packages/*/LICENSE``, which is
    neither what scope_entry_files hashes nor what the CI filter triggers on.
    Three readers of one entry disagreeing about which files it names is the
    defect this whole declaration exists to prevent.
    """
    return any(_entry_matches(filepath, entry) for entry in WORKSPACE_ONLY_TRIGGERS)


def _entry_matches(filepath: str, entry: str) -> bool:
    """Whether one declared entry names ``filepath``."""
    if entry.endswith("/"):
        return filepath.startswith(entry)
    if "*" not in entry:
        return filepath == entry
    parts, pattern = filepath.split("/"), entry.split("/")
    return len(parts) == len(pattern) and all(
        fnmatchcase(part, glob) for part, glob in zip(parts, pattern, strict=True)
    )


def _is_local_only_package_input(filepath: str) -> bool:
    """Whether a package file is read only by its own package's suite.

    Asked of a path already known to map to a package, so the question is
    narrow: which directory under the package holds it. See
    _LOCAL_ONLY_PACKAGE_DIRS for what makes a directory answer yes and why
    pyproject.toml, which is also not source, answers no.
    """
    parts = filepath.split("/")
    return len(parts) > 3 and parts[2] in _LOCAL_ONLY_PACKAGE_DIRS


class FileScan(NamedTuple):
    """What one change set's files were mapped to.

    Named rather than positional because two of the five members are sets of
    package names that differ by a subset relation, and the whole decision
    below turns on which of them reaches the closure. Swapped, the mistake is
    silent in both directions: the closure over *changed* schedules exactly
    what it used to, and reporting only *exporting* under-reports what changed.
    """

    #: Every package a file in the change set belongs to. Each runs its suite.
    changed_packages: set[str]
    #: The subset whose change is observable from outside the package, so its
    #: dependents' recorded results no longer describe what they run against.
    #: Always a subset of changed_packages; only this reaches the closure.
    exporting_packages: set[str]
    #: Whether to recompute the three recorded documentation checks.
    docs_changed: bool
    #: Which recorded steps a global quality input moved, which dirties every
    #: package for that step. Empty when no global input fired. A set rather
    #: than a flag because the gate records two per-package results from two
    #: different scripts, and most global inputs feed exactly one of them —
    #: see GLOBAL_TRIGGER_STEPS.
    triggered_steps: frozenset[str]
    #: Whether an input the guards under tests/ read moved.
    workspace_changed: bool


def map_files_to_packages(
    files: list[str], scans: Mapping[str, TriggerScan] | None = None
) -> FileScan:
    """Map changed files to affected packages. See FileScan.

    ``scans`` is what the readable global inputs reported about themselves,
    keyed by path — see :data:`_TRIGGER_READERS`. A path absent from it is one
    nothing read, so it keeps the blast radius it has always carried.
    """
    changed_packages: set[str] = set()
    exporting_packages: set[str] = set()
    docs_changed = False
    triggered_steps: set[str] = set()
    workspace_changed = False

    for filepath in files:
        # Every tier that matches contributes. The tiers are not a partition
        # and the declarations never said they were: a file can be a package's
        # test input *and* something a workspace guard reads, and three pairs
        # genuinely overlap on the tree today. Each branch used to end in
        # `continue`, so the first match was the only one that spoke and the
        # rest of the file's blast radius was dropped in silence — which made
        # declaring a file in a second tier a way to *un*-declare it from the
        # first. Four comments in this module warned about that ordering; they
        # describe the shape below instead.
        if filepath in GLOBAL_TRIGGERS:
            # One of these can say what it moved. A lock diff confined to
            # workspace-member blocks names the packages whose resolution
            # changed, and their dependents follow through the closure like
            # any other exporting change — a member's resolution is exactly
            # what its dependents build against. Every other global input,
            # and a lock this reader could not account for, keeps the blast
            # radius the path carries. An *empty* scan counts as unaccounted
            # for: see lock_members_changed for why that direction is the
            # safe one.
            scan = scans.get(filepath) if scans else None
            if scan is not None and scan.packages:
                changed_packages.update(scan.packages)
                exporting_packages.update(scan.packages)
            else:
                # What the content said, or — for an input nothing read, or one
                # whose reader could not place the change — what the table
                # declares. BOTH_STEPS is the fallback for a global input the
                # table does not carry, so a new entry behaves exactly as it
                # would have before the table existed until someone classifies
                # it. The guards fail first, but not at the moment it runs.
                triggered_steps |= (
                    scan.steps
                    if scan is not None
                    else GLOBAL_TRIGGER_STEPS.get(filepath, BOTH_STEPS)
                )

        # Workspace-only inputs belong to no package, so they move no package's
        # result — but they are still a quality input, and the guards under
        # tests/ are the ones that check the toolchain. "Belongs to no package"
        # is a property of the entries filed here, not of the branch: a file
        # that does belong to one and is also read by a guard reaches the
        # package mapping below as well.
        if _is_workspace_only_input(filepath):
            workspace_changed = True

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

        if filepath.startswith("packages/"):
            # What the generic rule below must not map to its package. A
            # package *document* is the bulk of it: 159 files match both — 149
            # under docs/ and ten changelogs — and mapping them ran a whole
            # suite and its dependents for a prose edit while running none of
            # the four guards that actually read the file. A document reaches
            # its package only by being declared in PACKAGE_TEST_DOC_INPUTS.
            #
            # The flag is named for what it gates rather than for that majority
            # case, because the licensing copies below are the member that is
            # not a document at all — and under the narrower name they read like
            # a category error rather than like the third thing the rule covers.
            belongs_to_no_suite = False

            # Package documentation. It belongs to no package's suite unless a
            # test in that suite reads it — see PACKAGE_TEST_DOC_INPUTS, which
            # is the whole of the exception and is checked against the tree
            # rather than trusted.
            if "/docs/" in filepath:
                docs_changed = True
                workspace_changed = True
                belongs_to_no_suite = True
                # Declared because a test *in* that package reads it, which
                # is also why it does not export: the document decides one
                # suite's result, and no dependent's run opens it.
                owner = PACKAGE_TEST_DOC_INPUTS.get(filepath)
                if owner is not None:
                    changed_packages.add(owner)

            # Package-root documentation, which the mapping below would
            # otherwise read as a change to the package itself — scheduling its
            # whole suite to publish a release note. See _PACKAGE_DOC_FILES for
            # why nothing about a package's recorded verdict depends on one.
            if filepath.rsplit("/", 1)[-1] in _PACKAGE_DOC_FILES:
                docs_changed = True
                belongs_to_no_suite = True

            # A package's LICENSE and NOTICE copy. Read by one workspace guard
            # and by no package's suite, so mapping one to its package is wrong
            # in both directions at once: it schedules that suite and — the
            # path being three segments, so _is_local_only_package_input says
            # no — the whole dependent closure behind it, while making
            # classify_test_scope answer "packages" and skip the only suite
            # that opens the file.
            #
            # No docs_changed here, unlike the two cases above. The three
            # recorded documentation checks read neither file: docs/license.md
            # links to them on GitHub rather than including them.
            if filepath.rsplit("/", 1)[-1] in _PACKAGE_LICENSE_FILES:
                belongs_to_no_suite = True

            # Map to package, and decide separately whether the change is
            # one a dependent's run can see. Only the second drives the
            # closure — see _LOCAL_ONLY_PACKAGE_DIRS for which it is.
            if not belongs_to_no_suite:
                parts = filepath.split("/")
                if len(parts) >= 2:
                    pkg_name = parts[1]
                    if pkg_name in DEPENDENCIES:
                        changed_packages.add(pkg_name)
                        if not _is_local_only_package_input(filepath):
                            exporting_packages.add(pkg_name)

    return FileScan(
        changed_packages=changed_packages,
        exporting_packages=exporting_packages,
        docs_changed=docs_changed,
        triggered_steps=frozenset(triggered_steps),
        workspace_changed=workspace_changed,
    )


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


def plan_for_files(
    files: list[str], scans: Mapping[str, TriggerScan] | None = None
) -> dict[str, Any]:
    """Decide what a change set needs tested, without consulting git.

    Split out from detect_changes so the decision is reachable from a test
    with a literal file list. The git half is what made the previous
    behaviour awkward to pin, and it is the decision that was wrong.

    ``scans`` keeps that split intact now that some inputs have to be *read* to
    be sized: detect_changes does the reading and passes the answers in, so
    this stays a function of its arguments. See map_files_to_packages.

    Returns dict with:
        packages: sorted list of packages affected in any way — unchanged in
            meaning, so a reader that knows only this key over-tests at worst
        test_packages: the subset whose *test* result can have moved
        docs_changed: whether docs-related files changed
        directly_changed: packages with direct file changes
        exporting: the subset of those whose change their dependents can see
        mode: "all" if global trigger hit, "changed" otherwise
        workspace_changed: whether a workspace-only quality input changed
        test_scope: "packages", "workspace" or "none" (see classify_test_scope)

    **``packages`` deliberately keeps its old meaning rather than becoming the
    lint list.** The two differ only when a global input moves one step and not
    the other, and the direction of that difference is what matters: a consumer
    reading ``packages`` alone schedules everything it would have scheduled
    before, so the narrowing is something a reader opts into rather than
    something it can miss. Spelled the other way — ``packages`` narrowed, a
    wider list beside it — the same consumer silently under-tests.
    """
    if not files:
        return {
            "packages": [],
            "test_packages": [],
            "docs_changed": False,
            "directly_changed": [],
            "exporting": [],
            "mode": "none",
            "workspace_changed": False,
            "test_scope": "none",
        }

    scan = map_files_to_packages(files, scans)

    # Only what a package *exports* reaches its dependents. Their recorded
    # results describe running against this package, so a change they can
    # see is what makes those results stop describing anything — and a
    # change they cannot see is one their suites would re-confirm
    # unchanged. Every directly changed package still runs its own suite;
    # the union is what keeps a local-only change scheduling one.
    all_affected = get_transitive_dependents(scan.exporting_packages) | scan.changed_packages
    closure = sorted(pkg for pkg in all_affected if pkg in DEPENDENCIES)

    # One closure, two overrides. A global input widens the list for the steps
    # it moves and leaves the other at whatever the files themselves said,
    # which is the whole of the difference between these two lines.
    packages = list(ALL_PACKAGES) if scan.triggered_steps else closure
    test_packages = list(ALL_PACKAGES) if TEST_STEP in scan.triggered_steps else closure
    mode = "all" if scan.triggered_steps else "changed"

    return {
        "packages": packages,
        "test_packages": test_packages,
        "docs_changed": scan.docs_changed,
        "directly_changed": sorted(scan.changed_packages),
        "exporting": sorted(scan.exporting_packages),
        "mode": mode,
        "workspace_changed": scan.workspace_changed,
        "test_scope": classify_test_scope(packages, scan.workspace_changed),
    }


def detect_changes(base_ref: str = "main") -> dict[str, Any]:
    """Detect changed packages and docs status. See plan_for_files.

    Reads a global input only when it is in the material change set, so the two
    extra git object reads per reader are paid on the change sets that can
    benefit and on no others.
    """
    files = get_changed_files(base_ref)
    resolved = _resolve_base_ref(base_ref)
    scans: dict[str, TriggerScan] = {}
    for path, read in _TRIGGER_READERS.items():
        if path not in files:
            continue
        scan = read(resolved)
        if scan is not None:
            scans[path] = scan
    return plan_for_files(files, scans)


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

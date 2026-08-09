"""Reproduce-first guard: the shell half of ``bin/`` is linted by something.

Every gate defect this batch tripped over was in a shell script — the change
detection that decided which packages to check, a ``show_usage`` ending in
``exit 0``, a ``print_fail`` reaching no failure path, a lint-target enumeration
that named one directory. Four for four, each found by hand-reading the script,
because 46 shell files totalling ~9,900 lines were linted by nothing at all
while every ``*.py`` beside them went through ruff and mypy.

``shellcheck`` detects the two classes those defects fall into. ``SC2086``: an
unquoted expansion, where an empty variable becomes *zero* arguments rather than
one empty one — so a target set silently comes out empty and the run reports
success over nothing. ``SC2155``: ``local x=$(cmd)``, where ``local``'s exit
status wins and the command's is discarded — so a failing command reads as a
succeeding one.

The tiers below are a **ratchet**, and that is the whole design. A script that
is clean must stay clean; a script that is not is held to errors only, and is
promoted the moment it reaches zero. So the deferred set can only shrink, and
nobody has to remember to revisit it — :func:`test_no_baseline_script_is_already_clean`
fails when one becomes eligible.

These guards read every tracked shell file rather than the check's own
declaration, because a guard derived from the thing it guards moves when that
thing moves. The enumeration here is ``git ls-files``; the check's is its own.
They have to agree, and that disagreement is the failure this file reports.
"""

from __future__ import annotations

import os
import re
import subprocess
from functools import cache

import pytest

from tests._workspace import ROOT, load_bin_module

#: ``source``/``.`` whose operand is fixed at authoring time — no ``$``, no
#: backtick — so the path shellcheck resolves under ``-x`` is the same on every
#: machine. A variable-bearing operand is a different question and a different
#: diagnostic (SC1090), so it is deliberately not matched here.
_LITERAL_SOURCE_RE = re.compile(r"^\s*(?:source|\.)\s+(?P<path>[^\s;&|)]+)")

LINT_SHELL = ROOT / "bin" / "lint-shell.sh"

#: The scripts that decide whether a quality run passes, pinned literally.
#:
#: Not derived from ``lint-shell.sh``'s own strict list, for the same reason
#: ``REQUIRED_DEFAULT_TARGETS`` is not derived from ``workspace_targets``: an
#: assertion computed from the declaration it guards cannot notice that
#: declaration shrinking. Demoting one of these to the baseline tier would
#: otherwise be a passing move, and it is precisely the move that would let a
#: defect back into the code that decides the verdict.
REQUIRED_STRICT = frozenset(
    {
        "bin/run-quality-checks.sh",  # owns the verdict and writes the artifact
        "bin/validate.sh",  # the ruff / mypy / print step
        "bin/test.sh",  # the test step
        "bin/validate-quality-artifacts.sh",  # what CI runs in place of the gate
        "bin/package-discovery.sh",  # which packages exist, and what else to check
        "bin/lint-workflows.sh",  # a recorded check in its own right
        "bin/docs-checks.sh",  # produces the documentation status
        "bin/dk",  # the entry point every other invocation goes through
        "bin/fix.sh",  # rewrites the code the above then judges
        "bin/lint-shell.sh",  # this file's subject, and a recorded check itself
    }
)


def _tracked_shell_files() -> list[str]:
    """Every tracked shell script, by ``git ls-files`` and then by shebang.

    Extension alone would miss ``bin/dk``, which is the entry point and carries
    no suffix — and missing exactly the file everything else is invoked through
    is the shape of gap this file exists to close.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = []
    for name in listing.split("\0"):
        if not name:
            continue
        path = ROOT / name
        if name.endswith(".sh"):
            found.append(name)
            continue
        if not path.is_file():
            continue
        try:
            first = path.open("rb").readline()
        except OSError:  # pragma: no cover - unreadable tracked file
            continue
        if first.startswith(b"#!") and b"sh" in first.split(b"\n")[0]:
            found.append(name)
    assert found, "no tracked shell files found — has the enumeration broken?"
    return sorted(found)


def _lint_shell(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LINT_SHELL), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@cache
def _declared(mode: str) -> frozenset[str]:
    """Ask the check which files it covers, rather than parsing what it says.

    Same reasoning as ``validate.sh --print-targets``: the declaration is a
    filesystem walk plus a tier split, so reading it as text reports what it
    says while the question is what it returns.

    Cached because nine tests here ask this across four modes, and the answer is
    a function of tracked content — which no test in this file changes. A
    ``frozenset`` rather than a ``set``: every caller now receives the *same*
    object, so a mutation in one test would silently change what the others see.

    A test that adds or removes a tracked shell file would need to bypass this.
    None does, and one that did would be asserting about a tree that is not the
    one under test.
    """
    result = _lint_shell(mode)
    assert result.returncode == 0, (
        f"bin/lint-shell.sh {mode} failed ({result.returncode}):\n{result.stderr}"
    )
    return frozenset(line.strip() for line in result.stdout.split() if line.strip())


def test_the_shell_lint_exists_and_is_executable():
    """The seam every other test here trusts. Without it they assert nothing."""
    assert LINT_SHELL.is_file(), (
        f"{LINT_SHELL.relative_to(ROOT)} does not exist — the shell half of bin/ "
        "is linted by nothing, which is the gap this file is written for."
    )
    import os

    assert os.access(LINT_SHELL, os.X_OK), f"{LINT_SHELL.relative_to(ROOT)} is not executable"


def test_every_tracked_shell_file_is_checked():
    """No shell script sits outside the check.

    A file in neither tier is not deferred, it is invisible: nothing reports it
    and no entry anywhere records that it was skipped.
    """
    covered = _declared("--print-targets")
    uncovered = sorted(set(_tracked_shell_files()) - covered)

    assert not uncovered, (
        "Tracked shell files that bin/lint-shell.sh does not check:\n"
        + "\n".join(f"  - {path}" for path in uncovered)
        + "\nEvery shell file belongs to a tier. If one genuinely should not be "
        "checked, exclude it explicitly in lint-shell.sh so the exclusion is "
        "readable — silence is what this guard exists to prevent."
    )


def test_the_check_claims_no_file_that_does_not_exist():
    """The other direction: a stale entry naming a deleted or renamed script.

    Cheap to check and the failure is otherwise invisible, since a tier listing
    a path that matches nothing simply does less work and still reports pass.
    """
    tracked = set(_tracked_shell_files())
    phantom = sorted(_declared("--print-targets") - tracked)

    assert not phantom, (
        "bin/lint-shell.sh names files that are not tracked shell scripts:\n"
        + "\n".join(f"  - {path}" for path in phantom)
    )


def test_the_scripts_that_decide_the_verdict_are_held_to_zero():
    """A literal pin, so demotion to the baseline tier cannot be a passing move."""
    strict = _declared("--print-strict")
    missing = sorted(REQUIRED_STRICT - strict)

    assert not missing, (
        f"These scripts are no longer held to zero shellcheck findings: {missing}. "
        "They decide whether a quality run passes — moving one to the baseline "
        "tier is not a fix, it is the regression this guard exists to catch."
    )


def test_the_strict_pin_has_not_drifted_from_the_declaration():
    """Containment in the other direction, so the pin cannot quietly go stale.

    A script promoted into the strict tier and later dropped from both the tier
    and this list would be missing from each without either noticing. This does
    not require the two to be equal — the strict tier grows as the ratchet turns
    — only that every pinned name is a real one.
    """
    tracked = set(_tracked_shell_files())
    phantom = sorted(REQUIRED_STRICT - tracked)

    assert not phantom, (
        f"REQUIRED_STRICT names scripts that no longer exist: {phantom}. "
        "Rename or remove the entry; a pin matching nothing pins nothing."
    )


def _tracked_paths() -> set[str]:
    """Every tracked path, for deciding whether a sourced file is in the repo."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {name for name in listing.split("\0") if name}


def test_no_checked_script_follows_a_path_that_is_not_in_the_repository():
    """The recorded verdict must be a function of tracked content, nothing else.

    ``lint-shell.sh`` runs shellcheck with ``-x``, which follows ``source``. A
    literal operand naming a path that is not in the repository therefore makes
    the verdict depend on the machine: present, and shellcheck reads it; absent,
    and it reports SC1091 *at ``info``* — inside the strict tier's floor. The
    whole gate then fails, on a file nobody edited, naming a path that is not in
    git.

    That is not hypothetical. ``bin/debug-test.sh`` sources ``.venv/bin/activate``
    twice and sits in the strict tier, so before the directives below the gate
    went red on any tree without a ``.venv`` — a fresh clone before ``uv sync``,
    or any tree after ``bin/dk cleanall``. It is worse than an ordinary flake
    because CI validates the committed artifact rather than re-running the gate:
    a verdict that depends on untracked state is a verdict CI cannot reproduce.

    Scoped to every checked script rather than only the strict tier, because the
    baseline tier is not a resting place — the ratchet *promotes* a script the
    moment it comes back clean, and it comes back clean on the machine of whoever
    happens to have the untracked file. Deferring this to promotion time means
    planting the failure for someone else to trip over.

    The remedy is ``# shellcheck source=/dev/null`` above the line: it states
    that the file is not available at lint time, which is true and worth saying,
    and leaves SC1091 live everywhere else to catch a genuinely broken path.
    """
    tracked = _tracked_paths()
    violations: list[str] = []

    for name in sorted(_declared("--print-targets")):
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            match = _LITERAL_SOURCE_RE.match(line)
            if not match:
                continue
            operand = match.group("path").strip("\"'")
            if "$" in operand or "`" in operand:
                continue
            if operand in tracked:
                continue
            preceding = next(
                (prior for prior in reversed(lines[: number - 1]) if prior.strip()),
                "",
            )
            if "shellcheck" in preceding and "source=" in preceding:
                continue
            violations.append(f"{name}:{number}: sources {operand!r}, which is not tracked")

    assert not violations, (
        "These scripts follow a path that is not in the repository, so the "
        "shell-lint verdict depends on whether that path happens to exist:\n"
        + "\n".join(f"  - {entry}" for entry in violations)
        + "\nAdd '# shellcheck source=/dev/null' above the line. A verdict CI "
        "cannot reproduce from tracked content is not a verdict."
    )


def test_every_linted_shell_script_is_covered_by_a_hash_scope():
    """A checked script outside every hash scope lets its own edit go unvalidated.

    The gate records a ``shell_lint`` verdict and these scripts are what it is a
    verdict *about*. Edit one and no package is touched — every suite that passed
    still passes — but the recorded verdict was computed over different code, and
    the stored hashes have no way to say so. CI validates the committed artifact
    instead of re-running the gate, so it accepts the artifact the edit has just
    invalidated. The pull request that changes a checked script is exactly the one
    the gate cannot check, and it reports green.

    Not hypothetical, and not narrowly avoided: when the shell lint was first
    wired in, 38 of the 46 scripts it reports on were outside every scope,
    because a directory entry expanded to ``*.py`` — right while ruff and mypy
    were the only readers, wrong the moment a shell checker joined them.

    Coverage is asked through ``workspace_scope_files``, the function the hash
    itself uses, rather than by re-deriving which paths an entry expands to. A
    second implementation of that rule could answer for a rule nothing follows.
    """
    hashes = load_bin_module("package-hashes")
    covered = {
        str(path.relative_to(ROOT))
        for scope in hashes.WORKSPACE_QUALITY_INPUTS
        for path in hashes.workspace_scope_files(scope)
    }

    targets = _declared("--print-targets")
    assert targets, "the shell lint reports no targets — this guard would check nothing"

    uncovered = sorted(targets - covered)
    assert not uncovered, (
        "The shell lint reports on these scripts, but no hash scope covers them, "
        "so editing one leaves every stored hash intact and its own change "
        "unvalidated:\n"
        + "\n".join(f"  - {path}" for path in uncovered)
        + "\nAdd them to a scope in bin/changed-packages.py, or widen the "
        "directory-entry rule in bin/package-hashes.py so it reaches them."
    )


def test_the_check_holds_the_two_tiers_to_different_floors(tmp_path):
    """The tiers are enforced, not merely declared.

    Every other guard here reads a ``--print-*`` mode, which reports the tier
    *split* — who is in which list. None of them reaches the code that turns a
    tier into a severity floor and a finding into a verdict. So deleting the
    strict floor from the run loop, dropping all 46 scripts to ``error``, left
    the entire suite green: ``--print-strict`` still named the right 18 files
    while the tier they name was enforced on nothing.

    ``--check-file`` exists so this can be asserted about the real executor
    rather than a second implementation of it — the run loop calls the same
    function. The probe carries SC2086, which is reported at ``info``: inside
    the strict floor, outside the baseline one. That single finding therefore
    has to produce two different verdicts, and it is the same finding class the
    check was written for.
    """
    probe = tmp_path / "probe.sh"
    probe.write_text("#!/bin/bash\nls $HOME\n", encoding="utf-8")

    baseline = _lint_shell("--check-file", str(probe), "baseline")
    assert baseline.returncode == 0, (
        "an info-level finding failed the baseline tier, which is held to errors "
        f"only:\n{baseline.stdout}{baseline.stderr}"
    )

    strict = _lint_shell("--check-file", str(probe), "strict")
    assert strict.returncode != 0, (
        "an info-level finding did not fail the strict tier. The strict floor is "
        "what the tier means; without it the two tiers are one tier.\n"
        f"{strict.stdout}{strict.stderr}"
    )
    assert "SC2086" in strict.stdout, (
        f"the finding itself was not reported, only the verdict:\n{strict.stdout}"
    )


def test_a_files_own_tier_decides_its_floor_when_none_is_named(tmp_path):
    """The default path, over real repository files rather than a fixture.

    Holds because of the ratchet, not by luck: no baseline script is clean at
    the strict floor — :func:`test_no_baseline_script_is_already_clean` is what
    makes that true — so every one of them must pass at its own tier and fail
    when held to the other. A tier lookup wired to a constant passes the first
    assertion and fails the second.
    """
    baseline_scripts = sorted(_declared("--print-baseline"))
    assert baseline_scripts, "no baseline scripts — this guard would check nothing"
    subject = baseline_scripts[0]

    own_tier = _lint_shell("--check-file", subject)
    assert own_tier.returncode == 0, (
        f"{subject} is in the baseline tier but failed at its own floor:\n"
        f"{own_tier.stdout}{own_tier.stderr}"
    )

    as_strict = _lint_shell("--check-file", subject, "strict")
    assert as_strict.returncode != 0, (
        f"{subject} passed at the strict floor, so either the floors are the "
        "same or the tier argument is ignored. If it is genuinely clean, the "
        "ratchet should have promoted it."
    )


def test_the_run_holds_each_script_to_the_tier_it_declares():
    """The run loop's own tier decision, which the single-file mode cannot reach.

    ``--check-file`` with a tier named bypasses ``tier_for`` by construction —
    that is what naming a tier means — so a lookup wired to a constant satisfies
    every assertion in the two tests above: the fixture still fails at ``strict``
    because ``strict`` was passed explicitly, and a baseline script still passes
    at its own floor because the constant *is* baseline. Verified by mutation;
    the suite went green with all 46 scripts held to one floor.

    The counts the run prints are that decision made 46 times, so comparing them
    against the declaration reaches it. This is why they are printed at all.
    """
    result = _lint_shell()
    assert result.returncode == 0, (
        f"the shell lint does not pass on its own tree:\n{result.stdout}{result.stderr}"
    )

    counts = {
        label: re.search(pattern, result.stdout)
        for label, pattern in (
            ("strict", r"held to zero findings:\s*(\d+)"),
            ("baseline", r"held to errors only:\s*(\d+)"),
        )
    }
    missing = sorted(label for label, match in counts.items() if match is None)
    assert not missing, (
        f"the run does not report how many scripts it held to each tier ({missing}). "
        "Those counts are the only observable form of the run loop's tier decision — "
        f"restore them or re-point this guard.\n{result.stdout}"
    )

    for label, mode in (("strict", "--print-strict"), ("baseline", "--print-baseline")):
        reported = int(counts[label].group(1))  # type: ignore[union-attr]
        declared = len(_declared(mode))
        assert reported == declared, (
            f"the run held {reported} scripts to the {label} tier but {mode} "
            f"declares {declared}. The run loop and the declaration disagree, "
            "which means one of them is not consulting the tier at all."
        )


def test_a_run_with_findings_exits_non_zero():
    """The aggregate verdict, asserted structurally — the weaker half, named.

    ``check_one`` is now reachable and tested, but the step from "some file
    failed" to the process's own exit status is not: it needs a run over a tree
    that has findings, and this one does not have any. So this reads the branch
    instead of running it, which reports what the script says rather than what
    it returns — the weaker form, and the reason it is written narrowly.
    """
    source = LINT_SHELL.read_text(encoding="utf-8")
    branch = re.search(
        r'if \[\[ "\$FAILED" == true \]\]; then(?P<body>.*?)\nfi', source, re.DOTALL
    )
    assert branch, (
        "the aggregate failure branch was not found in lint-shell.sh — if it was "
        "restructured, re-point this guard rather than leaving it passing vacuously"
    )
    assert re.search(r"^\s*exit [1-9]", branch.group("body"), re.MULTILINE), (
        "the branch that reports a failed shell lint does not exit non-zero, so "
        "the gate would record a pass over a run that found something."
    )


# --- a defect class shellcheck does not report ---

#: ``=~`` whose right-hand side is quoted, capturing the pattern.
#:
#: Quoting is not itself the defect and must not be flagged as one: it forces
#: *literal* matching, which is exactly right for the array-contains idiom
#: (``[[ " ${arr[*]} " =~ " $needle " ]]``), where unquoting would let the
#: needle's own characters act as a regex. Two live sites depend on that.
_QUOTED_REGEX_RE = re.compile(r"""=~\s*("(?P<d>[^"]*)"|'(?P<s>[^']*)')""")

#: Characters that mean something to a regex and nothing to a literal match, so
#: quoting them is a contradiction: the author wrote a pattern and got a string.
#: ``$`` is deliberately absent — inside double quotes it is shell expansion,
#: not an anchor, and treating it as evidence would flag the idiom above.
_REGEX_METACHARACTERS = set("^*+?[]()|")


def test_no_quoted_regex_hides_a_pattern_shellcheck_will_not_report():
    """A regex quoted into a literal string, which no severity of shellcheck sees.

    ``[[ "$1" =~ "^-" ]]`` does not ask whether the argument *starts with* a
    dash. The quotes make the right-hand side literal, so it asks whether the
    argument *contains the two characters* ``^-`` — which nothing does. The test
    is therefore always false, and ``! [[ ... ]]`` around it always true.

    Verified against shellcheck 0.11 at every severity, and with SC2076 forced
    on by name: it reports nothing. So this class is invisible to the check the
    rest of this file is about, and the guard has to be written by hand.

    Scoped to patterns carrying regex metacharacters rather than to quoting as
    such, because quoting is correct wherever the pattern is genuinely literal —
    see the note on the constant above. That distinction is the whole guard: a
    blanket rule would demand a change that breaks two working call sites.
    """
    offenders: list[str] = []

    for name in sorted(_declared("--print-targets")):
        for number, line in enumerate(
            (ROOT / name).read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = _QUOTED_REGEX_RE.search(line)
            if not match:
                continue
            pattern = match.group("d") if match.group("d") is not None else match.group("s")
            used = sorted(set(pattern) & _REGEX_METACHARACTERS)
            if used:
                offenders.append(f"{name}:{number}: =~ {pattern!r} — {''.join(used)} is literal here")

    assert not offenders, (
        "These comparisons quote a regex into a literal string, so the pattern's "
        "metacharacters match themselves and the test asks a different question "
        "than it reads as:\n"
        + "\n".join(f"  - {entry}" for entry in offenders)
        + "\nDrop the quotes to match as a regex, or use a glob comparison "
        "([[ \"$x\" != -* ]]) where that says it better. Quoting is right when "
        "the pattern is genuinely literal — this only flags patterns that are not."
    )


def test_attach_does_not_consume_a_following_flag_as_the_deploy_root(tmp_path):
    """The reproduction for the guard above, over the real argument parser.

    ``--deploy`` takes an optional server-root, and decides whether one was
    supplied by testing that the next argument is not a flag. With the pattern
    quoted, that test never fired: every following argument read as a root, so
    ``attach.sh -d --ssh`` silently set the deploy root to ``--ssh`` and dropped
    the ssh request entirely.

    ``--dryrun`` is passed first and deliberately: the loop parses it before the
    ``-d`` arm swallows anything, so the script echoes its parsed state and
    exits without running rsync or ssh. Ordering the arguments the other way is
    the same command with real side effects — which is itself a demonstration
    of the defect, and not something to run from a test.
    """
    script = ROOT / "bin" / "attach.sh"
    result = subprocess.run(
        [str(script), "--dryrun", "-d", "--ssh"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
    )

    parsed = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line and not line.startswith(" ")
    )

    assert parsed.get("server_root") != "--ssh", (
        "attach.sh consumed the following flag as the optional deploy root:\n"
        f"  server_root={parsed.get('server_root')!r}\n{result.stdout}"
    )
    assert "ssh " in result.stdout, (
        "the --ssh request was dropped, so the flag was consumed by --deploy "
        f"rather than parsed:\n{result.stdout}"
    )


@pytest.mark.skipif(
    not LINT_SHELL.is_file(), reason="lint-shell.sh does not exist yet"
)
def test_no_baseline_script_is_already_clean():
    """The ratchet. A deferred script that reaches zero must be promoted.

    Without this the baseline is a list nobody revisits: a script cleaned as a
    side effect of unrelated work stays deferred forever, and the next change to
    it can reintroduce findings with nothing to say so. Turning the ratchet is
    the fix — move the name into the strict tier.
    """
    result = _lint_shell("--print-promotable")
    assert result.returncode == 0, f"--print-promotable failed:\n{result.stderr}"
    promotable = sorted(line.strip() for line in result.stdout.split() if line.strip())

    assert not promotable, (
        "These scripts are in the baseline tier but have no shellcheck findings "
        "at all:\n"
        + "\n".join(f"  - {path}" for path in promotable)
        + "\nMove them to the strict tier in bin/lint-shell.sh. A clean script "
        "left deferred can regress without anything reporting it."
    )

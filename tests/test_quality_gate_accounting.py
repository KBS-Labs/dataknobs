"""Guards that every check the quality gate runs can actually fail it.

A check has to survive two hops to mean anything. It has to reach the local
exit code, so the developer's run goes red; and it has to reach
``quality-summary.json``, because CI validates the committed artifacts rather
than re-running the gate — that is the documented design, not an oversight, and
it makes the artifact the *only* channel by which a result reaches CI. A check
missing from the summary is therefore invisible to CI by construction.

The workflow lint made neither hop. It printed ``✗ Workflow lint failed`` and
the gate went on to report ``PASS``, with nothing in the artifact to say the
check had ever run. The comment above ``compute_overall_status`` names that
exact failure and says a new check must be wired in there — and this one was
wired into zero sites, so the note meant to prevent it documented it instead.

The same defect was already caught once, in ``bin/validate-quality-artifacts.sh``,
by a guard keyed to that script's ``print_fail`` / ``VALIDATION_FAILED``
vocabulary. ``bin/run-quality-checks.sh`` says ``print_error`` and ``*_STATUS``,
so the guard that was supposed to close the class closed it in one file. Hence
the table below: the vocabulary is data, and both scripts are checked by one
implementation. Adding a third gate script is a row, not a rewrite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tests._workspace import ROOT, rel, tracked_shell_files

GATE = ROOT / "bin" / "run-quality-checks.sh"
VALIDATOR = ROOT / "bin" / "validate-quality-artifacts.sh"


@dataclass(frozen=True)
class GateScript:
    """One script that runs checks, described in its own vocabulary.

    ``end`` matters as much as the rest: without it the final section runs to
    EOF and absorbs the script's own epilogue — which both of these end with,
    and which sets statuses and exits. Every section would then look accounted.
    """

    path: Path
    section: str  # regex: the helper that announces a check
    failure: str  # literal: the helper that announces a failure
    verdict: str  # regex: names of the variables carrying the verdict
    end: str  # literal: the line after the last check section


GATE_SCRIPTS = (
    GateScript(
        path=GATE,
        section=r"^\s*print_status\s",
        failure="print_error",
        verdict=r"[A-Z][A-Z0-9_]*_STATUS",
        # Where the checking stops and the reporting begins. It was the line
        # opening the summary heredoc until the summary stopped being one; the
        # writer call now occupies the same position, for the same reason.
        end='quality-summary.py" build',
    ),
    GateScript(
        path=VALIDATOR,
        section=r"^\s*print_check\s",
        failure="print_fail",
        verdict=r"VALIDATION_FAILED",
        end="# Final summary",
    ),
)


def _accounts(line: str, verdict: str) -> bool:
    """Whether this line carries a check's outcome somewhere it can be seen.

    The gate has five ways of doing that and they are all legitimate, so all
    five are recognised here — a guard that knew only about plain assignment
    flagged the docs check and the unit-test helper, both of which account
    correctly:

    * ``X=1`` — plain assignment
    * ``cmd || X=$?`` — assignment on the failure path
    * ``read -r X Y Z <<< "$codes"`` — three statuses from one sub-run
    * ``exit`` — abort the whole gate
    * ``return $rc`` — a helper handing its outcome to its caller

    The first two share a pattern below, so five idioms need four branches.

    ``return 0`` is deliberately not accounting: a helper that prints a failure
    and returns success has thrown the outcome away just as thoroughly as one
    that assigns nothing.

    The ``return`` branch is the weak one, and it is not self-sufficient: it
    accepts the hand-off without checking that anyone catches it, so a helper
    whose caller drops the status would satisfy it while discarding the outcome
    just as completely as ``return 0``. That second half is enforced separately,
    by :func:`test_a_helper_that_returns_its_outcome_has_it_caught`. Neither
    test is sufficient alone; do not delete one and leave the other reading as
    though the case were closed.
    """
    return bool(
        re.search(rf"\b{verdict}\b\s*=", line)
        or re.search(rf"^\s*read\b.*\b{verdict}\b", line)
        or re.match(r"^\s*exit\b", line)
        or re.match(r"^\s*return\s+(?!0\s*(?:#.*)?$)", line)
    )


def _sections(script: GateScript) -> list[tuple[int, str, list[str]]]:
    """The script's checks as ``(line_number, title, body)``, one-indexed."""
    lines = script.path.read_text(encoding="utf-8").splitlines()
    end = next((i for i, ln in enumerate(lines) if script.end in ln), None)
    assert end is not None, (
        f"{rel(script.path)} no longer contains {script.end!r}, which is where "
        "its check sections stop. Without it the last section swallows the "
        "epilogue and every assertion below passes vacuously — so re-point this "
        "marker rather than deleting it."
    )
    starts = [i for i, ln in enumerate(lines[:end]) if re.match(script.section, ln)]
    bounds = zip(starts, starts[1:] + [end], strict=True)
    return [(a + 1, lines[a].strip(), lines[a:b]) for a, b in bounds]


def test_the_gate_scripts_still_look_the_way_this_guard_reads_them():
    """Non-vacuity, per script: sections exist and at least one can fail.

    A rename of either helper would leave the extraction empty, and every check
    below would then pass by comparing nothing — which is the failure mode these
    guards exist to catch, so it must not be one they can have.
    """
    for script in GATE_SCRIPTS:
        name = rel(script.path)
        sections = _sections(script)
        assert sections, (
            f"no check sections found in {name} — the announcing helper was "
            "renamed or the script restructured. Fix this table, do not delete "
            "the guard."
        )
        assert any(script.failure in "\n".join(body) for _, _, body in sections), (
            f"no section in {name} calls {script.failure} — the helper was renamed"
        )
        assert any(any(_accounts(ln, script.verdict) for ln in body) for _, _, body in sections), (
            f"no section in {name} accounts for its outcome — {script.verdict} moved"
        )


def test_every_check_that_reports_a_failure_can_fail_the_gate():
    """A section that announces a failure must carry it somewhere it counts.

    Printing is not accounting. The failure helpers write a red line and nothing
    else, so a check can report ``✗`` and let the gate exit 0 — which is what
    the workflow lint did, and what the artifact-signature check did before it.

    The distinction being enforced is failing versus advisory. A check that
    cannot fail the gate must say so with the warning helper instead, the way
    coverage does, rather than announcing a failure it has no intention of
    causing.

    Granularity is the section, not the branch. Shell arms are not worth parsing
    here, and every instance of this defect so far has been a whole check that
    could not fail rather than one arm of a check that could.
    """
    silent = [
        f"{rel(script.path)}:{line} — {title}"
        for script in GATE_SCRIPTS
        for line, title, body in _sections(script)
        if any(script.failure in ln for ln in body)
        and not any(_accounts(ln, script.verdict) for ln in body)
    ]
    assert not silent, (
        "these checks print a failure the gate never sees, so they report '✗' "
        f"and pass: {silent}. Capture the exit status into a verdict variable, "
        "or use the warning helper if the check is genuinely advisory."
    )


def _shell_functions(path: Path) -> dict[str, list[str]]:
    """Shell function definitions as ``name -> body``.

    Matched at any indentation on purpose: the one function this applies to is
    defined inside a block, and a guard that only saw column-zero definitions
    would find nothing and pass while checking it.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    bodies: dict[str, list[str]] = {}
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\(\)\s*\{", line)
        if not match:
            continue
        indent = len(match.group(1))
        body = []
        for following in lines[i + 1 :]:
            closing = following.strip() == "}"
            if closing and len(following) - len(following.lstrip()) == indent:
                break
            body.append(following)
        bodies[match.group(2)] = body
    return bodies


def _drops_the_status(call: str, name: str) -> bool:
    """Whether this call line throws the callee's exit status away."""
    return not (
        re.search(rf"\b{name}\b[^&|]*(\|\||&&)", call)  # handled on failure
        or re.match(rf"^\s*(if|elif|while|until)\s+!?\s*{name}\b", call)  # the condition
        or re.search(rf"=\s*\$\(\s*{name}\b", call)  # captured by substitution
        or re.search(rf"\b{name}\b.*&\s*$", call)  # backgrounded; the wait carries it
    )


def test_a_helper_that_returns_its_outcome_has_it_caught():
    """``return $rc`` only accounts for a check if the caller catches it.

    ``_accounts`` takes the ``return`` on trust — it sees one line and cannot
    see the call sites — so on its own it would accept a helper that reports a
    failure, returns it, and is then called bare, which discards the status as
    thoroughly as the ``return 0`` that branch is careful to reject. That is the
    same shape as the defect this whole file exists for: an outcome produced and
    then dropped on the way to somewhere it counts.

    Backgrounded calls are accepted here and checked below instead: ``cmd &``
    cannot capture anything by itself, and the status arrives through ``wait``.
    """
    offenders = []
    checked = []
    for script in GATE_SCRIPTS:
        lines = script.path.read_text(encoding="utf-8").splitlines()
        for name, body in _shell_functions(script.path).items():
            if not any(re.match(r"^\s*return\s+(?!0\s*(?:#.*)?$)", ln) for ln in body):
                continue
            checked.append(f"{rel(script.path)}:{name}")
            definition = rf"^\s*{name}\(\)"
            calls = [
                (i + 1, ln)
                for i, ln in enumerate(lines)
                if re.search(rf"\b{name}\b", ln)
                and not re.match(definition, ln)
                and not ln.strip().startswith("#")
            ]
            assert calls, (
                f"{name} in {rel(script.path)} returns a status and is never "
                "called — either it is dead, or this guard stopped finding its "
                "call sites and is now checking nothing"
            )
            offenders += [
                f"{rel(script.path)}:{line} — {name}"
                for line, ln in calls
                if _drops_the_status(ln, name)
            ]
            if any(re.search(rf"\b{name}\b.*&\s*$", ln) for _, ln in calls):
                assert any(re.match(r"^\s*wait\b", ln) and "||" in ln for ln in lines), (
                    f"{name} is backgrounded in {rel(script.path)}, but no "
                    "'wait' there catches a failing job — so the status of "
                    "every concurrent run is discarded"
                )
    assert checked, (
        "no function in the gate scripts returns a non-zero status, so this "
        "guard compared nothing. Either the ``return`` branch of _accounts is "
        "now dead and both should go, or the extraction stopped finding shell "
        "function definitions — re-point it rather than leaving it passing."
    )
    assert not offenders, (
        "these calls discard the status of a helper that returns one, so a "
        f"failure it reported goes nowhere: {offenders}. Capture it with "
        "'|| STATUS=$?', or test it as a condition."
    )


# --- the second hop: from the verdict variables into the committed artifact ---

#: Statuses that legitimately do not gate the verdict directly, mapped to the
#: aggregate that does it for them. The two test statuses are OR-ed into
#: TEST_STATUS, which is what `compute_overall_status` reads.
AGGREGATED_INTO = {
    "UNIT_TEST_STATUS": "TEST_STATUS",
    "INTEGRATION_TEST_STATUS": "TEST_STATUS",
}

#: Statuses that gate the verdict but have no `checks` entry of their own,
#: mapped to the entries that report them. TEST_STATUS is exactly the OR of the
#: two below; a third entry repeating it would give a future reader three places
#: to disagree about one fact.
REPORTED_VIA_COMPONENTS = {
    "TEST_STATUS": ("UNIT_TEST_STATUS", "INTEGRATION_TEST_STATUS"),
}

#: The computed verdict, not an input to it. It is assigned from
#: `compute_overall_status` and written to the summary as `overall_status`.
VERDICT = "OVERALL_STATUS"

_STATUS_NAME = r"[A-Z][A-Z0-9_]*_STATUS"
_STATUS = re.compile(rf"\b({_STATUS_NAME})\b")


def _gate_lines() -> list[str]:
    return GATE.read_text(encoding="utf-8").splitlines()


def _assigned() -> set[str]:
    """Every status the gate writes, by either assignment or ``read``."""
    found: set[str] = set()
    for line in _gate_lines():
        found |= set(re.findall(rf"\b({_STATUS_NAME})\b\s*=", line))
        if re.match(r"^\s*read\b", line):
            found |= set(_STATUS.findall(line))
    return found - {VERDICT}


def _block(opening: str, closing: str) -> str:
    """The text from the line containing ``opening`` to the one closing it.

    ``closing`` is matched against the stripped line so a brace or heredoc
    terminator inside the block cannot end it early.
    """
    lines = _gate_lines()
    start = next((i for i, ln in enumerate(lines) if opening in ln), None)
    assert start is not None, (
        f"{rel(GATE)} no longer contains {opening!r} — re-point this guard "
        "rather than letting it read an empty block"
    )
    stop = next(
        (i for i, ln in enumerate(lines[start + 1 :], start + 1) if re.match(closing, ln.strip())),
        None,
    )
    assert stop is not None, f"{opening!r} in {rel(GATE)} is never closed by {closing!r}"
    return "\n".join(lines[start : stop + 1])


def _gates_the_verdict() -> set[str]:
    """Statuses ``compute_overall_status`` actually reads."""
    body = _block("compute_overall_status() {", r"\}$")
    return set(_STATUS.findall(body)) - {VERDICT}


def _reported_in_summary() -> set[str]:
    """Statuses that reach ``quality-summary.json``'s ``checks`` object.

    The population is every ``record_check`` call, wherever it sits. It used to
    be one block — the heredoc that serialized all eight checks in one place —
    and reading a block is what a positional guard can do; now each check states
    its own outcome at the site that produced it, so the calls are scattered by
    design and gathering them by name is the only reading that stays true.

    Continuation lines are not scanned, and do not need to be: the status is the
    call's second argument, so it is always on the line that opens it.
    """
    calls = [line for line in _gate_lines() if re.match(r"\s*record_check\b", line)]
    assert calls, (
        f"no record_check calls found in {rel(GATE)} — the checks no longer "
        "report their outcomes by that name, so this guard is comparing against "
        "nothing. Re-point it rather than leaving it passing."
    )
    return set(_STATUS.findall("\n".join(calls))) - {VERDICT}


def test_the_status_exception_tables_still_describe_the_gate():
    """An exception matching nothing is the failure mode this batch just paid for.

    Both tables below name specific variables, and a rename or a removal would
    turn an entry into a permanent free pass over a variable that no longer
    exists — silently widening the guard rather than failing it.
    """
    assigned, gating, reported = _assigned(), _gates_the_verdict(), _reported_in_summary()
    assert assigned and gating and reported, (
        "one of the three status sets came out empty, so the comparisons below "
        f"check nothing: assigned={sorted(assigned)}, gating={sorted(gating)}, "
        f"reported={sorted(reported)}"
    )

    for name, aggregate in AGGREGATED_INTO.items():
        assert name in assigned, f"AGGREGATED_INTO names {name}, which nothing assigns"
        assert aggregate in gating, (
            f"{name} is recorded as aggregated into {aggregate}, but "
            f"compute_overall_status does not read {aggregate}"
        )
    for name, components in REPORTED_VIA_COMPONENTS.items():
        assert name in gating, (
            f"REPORTED_VIA_COMPONENTS names {name}, which no longer gates the verdict"
        )
        absent = sorted(set(components) - reported)
        assert not absent, (
            f"{name} is recorded as reported through {absent}, which the summary does not contain"
        )


def test_every_status_the_gate_computes_gates_the_verdict():
    """A status nothing reads is a check whose result is discarded on arrival."""
    orphans = sorted(_assigned() - _gates_the_verdict() - set(AGGREGATED_INTO))
    assert not orphans, (
        f"{rel(GATE)} computes {orphans} and compute_overall_status reads none of "
        "them, so those checks cannot fail the gate. Add them to that function, "
        "or record the aggregate that carries them in AGGREGATED_INTO."
    )


def test_every_status_that_gates_the_verdict_reaches_the_artifact():
    """CI sees the artifact and nothing else, so an unreported check is invisible.

    This is the half that is easy to mistake for bookkeeping. It is not: the
    workflow lint had no ``checks`` entry, so even once it can fail a developer's
    run, a stale or hand-edited artifact would leave CI with no field to notice
    it by. The exit code and the summary are two independent hops and a check
    has to make both.
    """
    unreported = sorted(
        _gates_the_verdict() - _reported_in_summary() - set(REPORTED_VIA_COMPONENTS)
    )
    assert not unreported, (
        f"{unreported} decide whether the gate passes but appear nowhere in "
        "quality-summary.json, so CI — which validates the artifact rather than "
        "re-running the gate — cannot see them. Add a 'checks' entry, or record "
        "the entries that report them in REPORTED_VIA_COMPONENTS."
    )


def _init_lines(lines: list[str]) -> frozenset[int]:
    """The block that sets every status before any check has run.

    An assignment here records nothing: it is the default a path that never
    touches the variable falls through to.
    """
    start = next((i for i, ln in enumerate(lines) if "# Initialize status tracking" in ln), None)
    stop = next((i for i, ln in enumerate(lines) if "# How long each check took" in ln), None)
    assert start is not None and stop is not None, (
        f"{rel(GATE)} no longer has the initialization block this guard reads "
        "between those two comments — re-point it rather than letting it read "
        "an empty range."
    )
    return frozenset(range(start, stop))


def _pr_only_lines(lines: list[str]) -> frozenset[int]:
    """Line numbers reachable only when ``PR_MODE`` is yes.

    Every such block, not just the test one. A block's ``else`` ends the
    PR-only region: that arm is the dev path.
    """
    gated: set[int] = set()
    for number, line in enumerate(lines):
        opened = re.match(r'(\s*)if \[ "\$PR_MODE" = "yes" \]; then\s*$', line)
        if not opened:
            continue
        closing = {f"{opened.group(1)}fi", f"{opened.group(1)}else"}
        for end in range(number + 1, len(lines)):
            if lines[end].rstrip() in closing:
                gated.update(range(number, end))
                break
        else:  # pragma: no cover - a block that never closes is a syntax error
            raise AssertionError(f"unterminated PR_MODE block at {rel(GATE)}:{number + 1}")
    return frozenset(gated)


def test_every_status_the_summary_records_is_measured_on_a_dev_run():
    """A recorded status must not be able to reach the writer as its default.

    Writing one record per check closed the "absent reads as passing" half of
    this by construction: a check that records nothing has no entry, and no
    entry cannot be mistaken for a passing one. Six of the eight checks are
    finished by that argument — they run unconditionally, or their skip arm
    passes a literal ``0`` beside ``--skipped true``.

    The two test checks are not, and the difference is worth stating because it
    is the reason this guard outlived the heredoc. ``UNIT_TEST_STATUS`` and
    ``INTEGRATION_TEST_STATUS`` are **accumulators**: a package loop folds
    failures into them with ``|| UNIT_TEST_STATUS=$?``, so ``0`` means "nothing
    failed" and initialising to ``0`` is correct there. But the same variables
    are what the dev arm records, and the dev arm runs no such loop — it assigns
    them from ``TEST_STATUS``. Delete those two assignments and the record is
    still written, still well-formed, and reports ``pass`` for a run whose tests
    failed. That is the original defect, and the writer cannot see it: a stale
    accumulator and a measured zero are the same byte by the time it arrives.

    So the record is only as honest as the variable behind it, and a variable
    assigned nowhere on the dev path is a verdict the run never measured. This
    retires when the two test checks record at the site that produced their
    outcome, the way the other six already do.
    """
    lines = _gate_lines()
    init = _init_lines(lines)
    pr_only = _pr_only_lines(lines)
    assert pr_only, (
        f"found no PR-gated region in {rel(GATE)} — the mode split has moved, "
        "so this guard is exempting nothing and would pass on a gate that "
        "measured nothing. Re-point it."
    )

    def assigned_on_dev(name: str) -> bool:
        for number, line in enumerate(lines):
            if number in init or number in pr_only:
                continue
            if re.match(rf"\s*{name}=", line) or re.search(rf"\bread -r\b[^;|#]*\b{name}\b", line):
                return True
        return False

    reported = _reported_in_summary()
    assert reported, (
        f"no status variable reaches a record_check call in {rel(GATE)} — this "
        "guard is comparing against nothing. Re-point it rather than leaving it "
        "passing."
    )

    unmeasured = sorted(name for name in reported if not assigned_on_dev(name))
    assert not unmeasured, (
        f"{unmeasured} reach quality-summary.json but are assigned only inside "
        "PR-gated regions, so a dev run records whatever they were initialised "
        "to — and they are initialised to 0, which the writer reports as "
        '"pass". Assign the measured outcome on the dev path too, or record '
        "that check from the site that produced it."
    )


# --- a check whose tool is missing must fail, not skip ---


def _required_tool_probes(path: Path) -> list[tuple[int, str, list[str]]]:
    """``if ! command -v X`` probes, as ``(line, tool, branch body)``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    probes = []
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)(?:el)?if\s+!\s+command\s+-v\s+(\S+)", line)
        if not match:
            continue
        indent = len(match.group(1))
        body = []
        for following in lines[i + 1 :]:
            stripped = following.strip()
            same_level = len(following) - len(following.lstrip()) == indent
            if same_level and re.match(r"^(fi\b|else\b|elif\b)", stripped):
                break
            body.append(following)
        probes.append((i + 1, match.group(2), body))
    return probes


def _acts_on_the_missing_tool(tool: str, body: list[str]) -> bool:
    """Whether a not-found branch does something about it rather than continuing.

    Two dispositions are correct and the difference is real, which is why this
    is not simply "must exit": a hard requirement exits, and install-on-demand
    installs and carries on. Both leave the script running with the tool present
    or not running at all. The third possibility — printing a warning and going
    on — is the defect.

    Told apart without a vocabulary of installer commands, which would be a new
    hand-maintained set standing in for the question actually being asked. The
    question is whether the branch *acts*, so output lines are set aside and
    what remains has to either exit or name the tool it just failed to find.
    Every install here reads ``uv pip install <tool>``; every requirement reads
    ``exit N`` under an echoed message that mentions ``brew install`` — which is
    why a keyword search for "install" gets both backwards, and why exit is
    tested first regardless of what the branch printed.

    Fails closed. An install this cannot recognise is reported rather than
    assumed, and the remedy is to make the branch say what it does; a branch
    that silently continues is reported either way.
    """
    actions = [
        stripped
        for line in body
        if (stripped := line.strip())
        and not stripped.startswith("#")
        and not re.match(r"^(echo|printf)\b", stripped)
    ]
    if any(re.match(r"^exit\b", action) for action in actions):
        return True
    return any(tool in action for action in actions)


def test_no_shell_script_continues_when_a_tool_it_probed_for_is_missing():
    """Skipping on a missing tool reports green while testing nothing.

    Which is worse than having no check, because it also reports success — and
    under an artifact-validating CI it is undetectable after the fact: a
    developer without shellcheck produces an artifact byte-identical to one from
    a run where the linter ran and found nothing. ``environment.json`` records
    the Python and uv versions and no per-check tool availability, so there is
    nothing downstream to notice the difference.

    Failing loudly is also what the dependency rules require of any tool we
    invoke as a subprocess, which is what shellcheck is. Two independent
    arguments, one behaviour.

    The population is **every tracked shell file**, derived rather than listed.
    It was three scripts named by hand, guarded by a comment that deferred the
    widening because "telling install-on-demand from a genuine requirement needs
    something better than a keyword search for 'install'" — which is true, and
    was solved by not searching for a keyword. Classifying all twelve negative
    probes in the repository comes out six exit, six install, and **none**
    ambiguous, so the rule the comment was waiting for is just the two
    dispositions written down.

    Two things the hand-maintained list also hid, both visible only once the set
    was derived: one of its three members contains no probe at all, so its
    contribution was empty while the non-empty check passed on the other two;
    and the two scripts the comment named as the obstacle use the *positive*
    form (``if command -v docker-compose``), which is alternative selection and
    was never in this population to begin with. Its stated failure mode — a
    script added to the gate and not to the list — now cannot happen rather than
    being tracked.
    """
    probes = [
        (name, line, tool, body)
        for name in tracked_shell_files()
        for line, tool, body in _required_tool_probes(ROOT / name)
    ]
    assert probes, (
        "no 'if ! command -v' probes found in any tracked shell file — if the "
        "requirement checks were restructured, re-point this guard rather than "
        "leaving it passing vacuously"
    )

    continuing = [
        f"{name}:{line} — {tool}"
        for name, line, tool, body in probes
        if not _acts_on_the_missing_tool(tool, body)
    ]
    assert not continuing, (
        f"these branches continue when a probed-for tool is absent: {continuing}. "
        "The run then reports success having verified nothing. Either print an "
        "error naming the tool and exit non-zero, or install it — and if the "
        "branch does install it, name the tool in the command that does so, "
        "which is how this tells the two apart."
    )

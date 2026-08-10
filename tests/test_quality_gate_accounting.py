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
        end='quality-summary.json" <<EOF',
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
    bounds = zip(starts, starts[1:] + [end])
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
        assert any(
            script.failure in "\n".join(body) for _, _, body in sections
        ), f"no section in {name} calls {script.failure} — the helper was renamed"
        assert any(
            any(_accounts(ln, script.verdict) for ln in body)
            for _, _, body in sections
        ), f"no section in {name} accounts for its outcome — {script.verdict} moved"


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
                assert any(
                    re.match(r"^\s*wait\b", ln) and "||" in ln for ln in lines
                ), (
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
        (
            i
            for i, ln in enumerate(lines[start + 1 :], start + 1)
            if re.match(closing, ln.strip())
        ),
        None,
    )
    assert stop is not None, f"{opening!r} in {rel(GATE)} is never closed by {closing!r}"
    return "\n".join(lines[start : stop + 1])


def _gates_the_verdict() -> set[str]:
    """Statuses ``compute_overall_status`` actually reads."""
    body = _block("compute_overall_status() {", r"\}$")
    return set(_STATUS.findall(body)) - {VERDICT}


def _reported_in_summary() -> set[str]:
    """Statuses that reach ``quality-summary.json``'s ``checks`` object."""
    return set(_STATUS.findall(_block('"checks": {', r"EOF$"))) - {VERDICT}


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
            f"{name} is recorded as reported through {absent}, which the summary "
            "does not contain"
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


# --------------------------------------------------------------------------
# A verdict the summary reports must be one some path actually recorded.
# --------------------------------------------------------------------------

#: Where the summary's status fields come from: ``$([ $X -eq 0 ] && echo '"pass"'``.
_REPORTED_STATUS_RE = re.compile(r"""\$\(\s*\[\s*"?\$(\w+)"?\s+-eq\s+0\s*\]\s*&&\s*echo\s*'"pass"'""")


def _anchor(lines: list[str], needle: str) -> int:
    """Index of the one line containing ``needle``.

    Asserted rather than searched leniently: this guard slices the script by
    position, so an anchor that has moved must fail loudly. A structural guard
    that quietly finds nothing to check is the failure mode this whole file
    exists to catch.
    """
    hits = [i for i, line in enumerate(lines) if needle in line]
    assert len(hits) == 1, f"expected exactly one {needle!r} in {rel(GATE)}, found {len(hits)}"
    return hits[0]


def _summary_entries(lines: list[str]) -> list[tuple[str, str, str | None]]:
    """Every check the summary writes, as ``(name, status_var, skipped_var)``.

    Read out of the heredoc rather than listed here, so a check added to the
    summary is covered the day it is added — which is the property the check
    below is about in the first place.

    A ``skipped`` field spelled as an inline expression rather than a variable
    yields ``None``: deciding "did this run" in the heredoc is a second answer to
    a question the block that ran it already answered, and the comment above
    ``UNIT_SKIPPED`` records what that cost the last time.
    """
    start = _anchor(lines, 'quality-summary.json" <<EOF')
    entries: list[tuple[str, str, str | None]] = []
    name: str | None = None
    status: str | None = None
    skipped: str | None = None

    for line in lines[start:]:
        opened = re.match(r'\s*"(\w+)":\s*\{\s*$', line)
        if opened:
            name, status, skipped = opened.group(1), None, None
            continue
        if name is None:
            continue
        found = _REPORTED_STATUS_RE.search(line)
        if found:
            status = found.group(1)
        flag = re.match(r'\s*"skipped":\s*\$(\w+)\s*,?\s*$', line)
        if flag:
            skipped = flag.group(1)
        if re.match(r"\s*\}", line):
            if status is not None:
                entries.append((name, status, skipped))
            name = None
    return entries


def _pr_only_lines(lines: list[str]) -> frozenset[int]:
    """Line numbers reachable only when ``PR_MODE`` is yes.

    Every such block, not just the test one. The docs checks are gated the same
    way and report three statuses, so a guard that knew about one block would
    have cleared the other three — the same "scoped to where you already looked"
    shape this file keeps finding.

    A block's ``else`` ends the PR-only region: that arm is the dev path.
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


def test_every_status_the_summary_reports_is_recorded_on_both_run_modes():
    """A status field must never fall through to its initial value.

    The statuses are initialised to ``0``, and ``0`` renders as ``"pass"``. So a
    variable no path assigns does not read as absent or unknown — it reads as a
    check that ran and passed.

    ``UNIT_TEST_STATUS`` and ``INTEGRATION_TEST_STATUS`` were assigned only on
    the PR arm; the dev arm tracks ``TEST_STATUS`` alone. That was harmless while
    the summary was written by the gate only, and stopped being harmless when the
    summary became a record both tiers write: a dev run with failing tests then
    wrote ``overall_status: FAIL`` beside ``unit_tests: pass``, and the
    diagnostics tool gates its whole test-failure section on those two fields —
    so it reported the failure and named nothing, for the most common developer
    command and the most common kind of failure.

    Note the duration fields got this right by accident of their default: they
    initialise to ``null``, which is the absence of a measurement rather than a
    passing one. ``0`` is a verdict; ``null`` is not.

    **Delete this guard rather than porting it** when the summary stops being
    written by a shell heredoc. It is a compensating control, not a property
    worth keeping: it parses the producer's source to check an invariant the
    producer cannot express, and it exists only because a status variable has a
    default and that default is a verdict. A writer that emits one record per
    check as the check runs has no defaults to fall through to — a check that did
    not run has no record, and an absent record cannot be mistaken for a passing
    one. At that point every regex above is describing a shape that is gone, and
    keeping them would leave the next author maintaining a reader for a producer
    that no longer exists.
    """
    lines = GATE.read_text(encoding="utf-8").splitlines()
    entries = _summary_entries(lines)
    assert entries, "no check entries found in the summary — has its shape changed?"

    # Assignments made before any check runs record nothing; they are the
    # defaults the bug fell through to.
    init = range(_anchor(lines, "# Initialize status tracking"), _anchor(lines, "# How long each check took"))
    pr_only = _pr_only_lines(lines)
    assert pr_only, "found no PR-gated region — has the mode split moved?"

    def records_on_dev(name: str) -> bool:
        for number, line in enumerate(lines):
            if number in init or number in pr_only:
                continue
            if re.match(rf"\s*{name}=", line) or re.search(rf"\bread -r\b[^;|#]*\b{name}\b", line):
                return True
        return False

    def defaults_to_skipped(name: str | None) -> bool:
        """True when the flag reads "skipped" unless a run says otherwise.

        The default is what a path that never touches the variable reports, so
        it is the only value that matters to a check the run did not perform.
        """
        return name is not None and any(
            re.match(rf'\s*{name}="true"', lines[number]) for number in init
        )

    unaccounted = {}
    for check, status_var, skipped_var in entries:
        if records_on_dev(status_var):
            continue
        if defaults_to_skipped(skipped_var):
            continue
        unaccounted[check] = (
            f"${status_var} is assigned only inside a PR-gated block"
            if skipped_var is None
            else f"${status_var} is PR-gated and ${skipped_var} does not default to \"true\""
        )

    assert not unaccounted, (
        "these checks are reported in quality-summary.json but neither run nor\n"
        'declared skipped on a dev run, so the summary reports "pass" for work\n'
        "that was never done:\n  "
        + "\n  ".join(f"{check}: {why}" for check, why in sorted(unaccounted.items()))
        + "\nEither record the verdict on the dev path, or carry a skipped flag "
        'that defaults to "true".'
    )

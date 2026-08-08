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

from tests._workspace import ROOT, rel

GATE = ROOT / "bin" / "run-quality-checks.sh"
VALIDATOR = ROOT / "bin" / "validate-quality-artifacts.sh"
WORKFLOW_LINT = ROOT / "bin" / "lint-workflows.sh"


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

#: Scoped to the gate's own scripts rather than all of ``bin/``, because a
#: negative ``command -v`` probe does not mean the same thing everywhere. Of the
#: nine outside this scope, six are install-on-demand (``not found → uv pip
#: install → carry on``) and the other three already exit — audit-floor.sh for
#: uv and osv-scanner, publish-test.sh for uv. So none of the nine is a defect
#: today, and widening the scope would find nothing; the positive form used for
#: sha256sum/shasum is alternative selection, also fine.
#:
#: Which is the actual reason to leave the scope alone: the cost of widening is
#: not the three that would pass anyway but the six that would not, and telling
#: install-on-demand from a genuine requirement needs something better than a
#: keyword search for "install". Worth revisiting if a third gate script arrives
#: with its own tool requirement, since that is the case this narrow scope would
#: quietly fail to cover.
TOOL_PROBE_SCRIPTS = (GATE, WORKFLOW_LINT)


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


def test_a_gate_check_never_skips_because_its_tool_is_missing():
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
    """
    probes = [
        (path, line, tool, body)
        for path in TOOL_PROBE_SCRIPTS
        for line, tool, body in _required_tool_probes(path)
    ]
    assert probes, (
        "no 'if ! command -v' probes found in the gate scripts — if the "
        "requirement checks were restructured, re-point this guard rather than "
        "leaving it passing vacuously"
    )

    skipping = [
        f"{rel(path)}:{line} — {tool}"
        for path, line, tool, body in probes
        if not any(re.match(r"^\s*exit\b", ln) for ln in body)
    ]
    assert not skipping, (
        f"these checks continue when a required tool is absent: {skipping}. The "
        "run then reports success having verified nothing. Print an error naming "
        "the tool and exit non-zero instead."
    )

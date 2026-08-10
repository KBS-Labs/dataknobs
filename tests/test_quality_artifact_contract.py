"""Guards the JSON contract between the artifact validator and its CI consumer.

``bin/package-hashes.py validate --json`` produces a verdict;
``bin/validate-quality-artifacts.sh`` consumes it and is what the quality
workflow actually runs. Nothing connects the two — the producer can add a field,
or a new pairing of fields, and the consumer keeps parsing the shape it was
written against. Both failures in this file's history are that shape:

* ``changed_scopes`` was added to the producer and never read, so a
  workspace-only change failed the gate while naming nothing that changed.
* ``warning`` was chained ahead of the verdict, so a result carrying both a
  warning and a failure reported the warning and skipped the failure. That is
  a green gate on stale artifacts, and it appeared independently in the Python
  CLI and in the shell — the same mistake twice, which is what makes it a class
  rather than an incident.

Both are checked here against the producer's *actual* emitted keys, read from
its source, so a field added later is covered without an edit to this file.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

from tests._workspace import ROOT

PRODUCER = ROOT / "bin" / "package-hashes.py"
CONSUMER = ROOT / "bin" / "validate-quality-artifacts.sh"
ARTIFACTS = ROOT / ".quality-artifacts"
GITATTRIBUTES = ROOT / ".gitattributes"
GIT_SETUP = ROOT / "bin" / "setup-git-config.sh"

#: Emitted for completeness but deliberately not read from this payload. The
#: consumer greps ``overall_status`` out of quality-summary.json directly, a few
#: checks further down, and ``status_ok`` is that same value already reduced to
#: a boolean — both are folded into ``valid`` before the consumer sees them, so
#: reading them here would report the same fact twice.
NOT_CONSUMED_BY_DESIGN = frozenset({"status_ok", "overall_status"})


def _emitted_keys() -> set[str]:
    """Every key ``validate_artifacts`` can put in its result.

    Taken from the source rather than by calling it: the early returns are the
    cases that matter most here, and reaching them means standing up missing or
    corrupt artifacts on disk.
    """
    tree = ast.parse(PRODUCER.read_text(encoding="utf-8"))
    func = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "validate_artifacts"
        ),
        None,
    )
    assert func is not None, f"validate_artifacts not found in {PRODUCER.name}"

    keys: set[str] = set()
    for node in ast.walk(func):
        # Dict literals: the early returns and the main result.
        if isinstance(node, ast.Dict):
            keys |= {k.value for k in node.keys if isinstance(k, ast.Constant)}
        # Later additions: result["warning"] = ...
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.value, ast.Name) and isinstance(node.slice.value, str):
                keys.add(node.slice.value)
    return keys


def test_producer_emits_the_keys_this_guard_expects():
    """Non-vacuity: the AST walk must actually find the payload.

    A renamed function or a restructured return would leave the extraction
    empty, and every check below would then pass by comparing nothing.
    """
    keys = _emitted_keys()
    assert keys, "no result keys extracted — the producer's shape changed"
    missing = {"valid", "warning", "error", "changed_packages"} - keys
    assert not missing, f"expected result keys not found: {sorted(missing)}"


def test_consumer_reads_every_field_the_producer_emits():
    """A field the consumer never reads is a diagnostic that reaches no one.

    This is the ``changed_scopes`` failure exactly: the producer knew which
    workspace scope had changed, the gate failed, and the report named nothing,
    because the one script that prints the reason had never been taught the
    field existed.
    """
    consumer = CONSUMER.read_text(encoding="utf-8")
    unread = sorted(
        key
        for key in _emitted_keys() - NOT_CONSUMED_BY_DESIGN
        if f"'{key}'" not in consumer and f'"{key}"' not in consumer
    )
    assert not unread, (
        f"{CONSUMER.name} never reads {unread}, so that detail is computed and "
        "then dropped. Print it in the failure branch, or add it to "
        "NOT_CONSUMED_BY_DESIGN with the reason."
    )


def test_consumer_does_not_let_a_warning_shadow_the_verdict():
    """``warning`` must be printed additively, never as a branch before the verdict.

    A warning says what could *not* be checked. It carries no claim about what
    was checked, so a chain that tests it first will, on any result carrying
    both, print the warning and never reach the failure. The producer emits
    exactly that pairing whenever artifacts predate workspace hashing and a
    package is also dirty.
    """
    lines = CONSUMER.read_text(encoding="utf-8").splitlines()
    branches = [ln.strip() for ln in lines if "HASH_WARNING" in ln and re.match(r"^\s*(el)?if ", ln)]
    assert branches, (
        f"{CONSUMER.name} no longer branches on HASH_WARNING at all — if the "
        "variable was renamed, update this guard rather than deleting it"
    )
    chained = [b for b in branches if b.startswith("elif")]
    assert not chained, (
        "The warning is tested as 'elif', so a result carrying both a warning "
        f"and a failure reports only the warning: {chained}. Print it in its "
        "own 'if' ahead of the verdict instead."
    )


def _required_files() -> list[str]:
    """The REQUIRED_FILES array the consumer checks for existence."""
    body = CONSUMER.read_text(encoding="utf-8")
    match = re.search(r"^REQUIRED_FILES=\((.*?)^\)", body, re.MULTILINE | re.DOTALL)
    assert match, f"REQUIRED_FILES array not found in {CONSUMER.name}"
    return re.findall(r'"([^"]+)"', match.group(1))


def _is_committable(relative_path: str) -> bool:
    """Whether git would keep this artifact path, i.e. .gitignore does not drop it.

    Deliberately *without* ``--no-index``, unlike ``_excluded_by_gitignore``
    below, because the two ask different questions and git answers each one
    correctly for its own. Here the question is whether a path this repository
    requires can be committed at all, and a path already tracked can be — git
    does not apply ignore rules to tracked files — which is what the plain form
    reports. Below the question is what the *rules* say, for which a tracked
    path returning "not ignored" is the wrong answer and the trap that made the
    obvious implementation unable to fail.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", f".quality-artifacts/{relative_path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    # 0 means the path IS ignored, 1 means it is not, and anything else is an
    # error — check-ignore reports 128 outside a work tree. Treating that as
    # "not ignored" would pass this guard on every path without consulting a
    # single rule, which is the failure it exists to catch.
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"git check-ignore failed ({result.returncode}) for {relative_path}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.returncode == 1


def test_every_required_artifact_is_one_git_actually_keeps():
    """A required file that .gitignore drops fails every pull request at once.

    The two declarations sit in different files with nothing between them, and
    the failure mode is maximally unhelpful: CI reports a missing artifact for
    every branch, including ones that changed nothing, and the developer's
    fix — re-running the checks — regenerates a file git then discards again.

    This is a live risk rather than a hypothetical: coverage.xml was required
    here and committed for months before being untracked as generated churn,
    and the only thing that made that safe was removing it from the array in
    the same change.
    """
    required = _required_files()
    assert required, "no REQUIRED_FILES entries extracted — the array's shape changed"

    dropped = sorted(name for name in required if not _is_committable(name))
    assert not dropped, (
        f"{CONSUMER.name} requires {dropped}, which .gitignore excludes from the "
        "repository. CI would fail every pull request on a file no developer can "
        "commit. Either add a '!' un-ignore rule, or drop it from REQUIRED_FILES."
    )


def _tracked_artifacts(root: Path) -> list[str]:
    """Every path git tracks under ``.quality-artifacts/``, repo-relative."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", ".quality-artifacts"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [name for name in listing.split("\0") if name]


def _excluded_by_gitignore(root: Path, paths: list[str], *, no_index: bool = True) -> list[str]:
    """Which of ``paths`` the ignore *rules* exclude, whether tracked or not.

    ``--no-index`` is load-bearing and is a parameter only so a test can pin
    that. Without it, ``git check-ignore`` declines to report a path that is
    already in the index — it answers "not ignored" for exactly the files this
    is looking for, and the check reports every violation as compliant.
    """
    if not paths:
        return []
    command = ["git", "check-ignore", "-z", "--stdin"]
    if no_index:
        command.insert(3, "--no-index")
    result = subprocess.run(
        command,
        cwd=root,
        input="\0".join(paths),
        capture_output=True,
        text=True,
        check=False,
    )
    # 0: at least one path is ignored. 1: none are. Anything else is a real
    # failure, and reading it as "none are" would make this pass without
    # consulting a rule.
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"git check-ignore failed ({result.returncode}): {result.stderr.strip()}"
        )
    return [name for name in result.stdout.split("\0") if name]


def test_no_committed_artifact_is_one_gitignore_excludes():
    """A file both tracked and ignored is in the repository by accident.

    ``.gitignore`` declares the committed artifact set as an allowlist —
    ``.quality-artifacts/*`` and a handful of ``!`` un-ignores — which makes
    "what is committed" a decision written down in one place. A file that is
    tracked *and* matched by the ignore rule contradicts that declaration
    without contradicting anything git enforces: ignore rules do not apply to
    tracked files, so it keeps being committed, keeps conflicting on every run,
    and the allowlist keeps reading as though it were the whole story.

    Which is how ``coverage.xml`` accumulated over a gigabyte of object history
    while the rule excluding it sat right there. Reproduced with ``git add -f``:
    all twenty artifact guards passed over it.

    The other direction is asserted separately, below.
    """
    tracked = _tracked_artifacts(ROOT)
    assert tracked, (
        "no tracked files under .quality-artifacts/ — the listing broke, and "
        "this guard would pass by checking nothing"
    )

    excluded = _excluded_by_gitignore(ROOT, tracked)
    assert not excluded, (
        "these files are committed and .gitignore excludes them, so the "
        "allowlist no longer describes what is in the repository:\n"
        + "\n".join(f"  - {name}" for name in excluded)
        + "\n\nEither add a '!' un-ignore rule for each, if it belongs in the "
        "repository, or 'git rm --cached' it, if it does not. Leaving it is the "
        "state that costs the most: it stays committed regardless of the rule."
    )


def _un_ignored_artifact_names() -> list[str]:
    """Every ``.quality-artifacts/`` path ``.gitignore`` un-ignores, in order."""
    prefix = "!.quality-artifacts/"
    return [
        line.strip().removeprefix("!")
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(prefix)
    ]


def test_every_un_ignored_artifact_is_one_something_writes():
    """An allowance with no producer is a slot, and the slot is signed.

    The reverse of the guard above, and it used to be legitimately false, which
    is how it went unasserted: ``lint-report.json`` was un-ignored for a writer
    that never existed. A reader in ``bin/diagnose-quality-failures.sh``, an
    allowance here, and nothing in between — so the lint half of that tool
    returned at its first line on every run it ever made, reporting nothing and
    looking exactly like a run with no lint findings.

    The cost is not the dead read. It is that the signature enumerates this
    directory by glob — ``git ls-files --cached --others --exclude-standard --
    '*.json' '*.xml'`` — so an un-ignored name is not merely permitted to be
    committed, it joins *what CI attests* the moment anything writes it. An
    allowance nobody is maintaining is a place a future write lands in the
    signed set without a decision.

    Tracked is the test for "something writes it", because the gate writes all
    six on every run and commits them. A file the gate wrote only sometimes
    would fail here — correctly: it would be a name that changes the attested
    set depending on the run, which is the condition the signature exists to
    detect and the last thing to encode in an ignore rule.
    """
    allowed = _un_ignored_artifact_names()
    assert allowed, (
        "no '!' un-ignore rules found under .quality-artifacts/ — the parse "
        "broke, and this guard would pass by checking nothing"
    )

    tracked = set(_tracked_artifacts(ROOT))
    orphans = [name for name in allowed if name not in tracked]

    assert not orphans, (
        ".gitignore un-ignores these, and git tracks none of them:\n"
        + "\n".join(f"  - {name}" for name in orphans)
        + "\n\nEither something should be writing and committing it — in which "
        "case the producer is missing — or the allowance is dead and should be "
        "removed. Leaving it is the state that costs the most: the signature "
        "enumerates this directory by glob, so the name is a slot a future "
        "write joins the attested set through, without anyone deciding to."
    )


def _summary_check_names() -> set[str]:
    """The ``checks`` keys ``run-quality-checks.sh`` writes into the summary.

    Gathered from the ``record_check`` calls, because that is where each name is
    stated now: the summary is assembled from records the checks write as they
    run, so there is no longer one block that names all eight. The name is the
    call's first argument, so the definition line — ``record_check() {`` — does
    not match, having no space after the name.
    """
    source = (ROOT / "bin" / "run-quality-checks.sh").read_text(encoding="utf-8")
    return set(re.findall(r"^\s*record_check\s+([a-z_]+)\b", source, re.MULTILINE))


def test_no_reader_names_a_check_the_summary_does_not_record():
    """A reader asking for a key nobody writes gets a value, and it is wrong.

    ``jq -r '.checks.lint.status'`` on a summary with no ``lint`` check prints
    ``null`` and exits 0 — so the read succeeds, the value is not ``"pass"``,
    and the caller renders a warning. ``bin/diagnose-quality-failures.sh`` did
    that for ``lint`` and ``style``, neither of which the producer has ever
    emitted, and showed two amber rows on every run it ever made. A row that is
    always amber carries no information, and it costs more than a missing row:
    it is indistinguishable from one that is amber for a reason.

    The same shape as the ``lint-report.json`` allowance and the
    permanently-mismatching signature before it — a reader, no writer, and a
    result too plausible to look wrong.

    Comment lines are skipped. This scans source text, so without that it also
    matches the note left behind at a site where the defect was *removed*,
    making the guard fire on the sentence explaining why it no longer can. A
    guard that punishes describing the bug it guards against gets the
    description deleted, which is the opposite of what it is for.
    """
    emitted = _summary_check_names()
    assert emitted, "no check names extracted from the gate's record_check calls"

    readers = {"diagnose-quality-failures.sh", "validate-quality-artifacts.sh"}
    unknown: list[str] = []
    for name in sorted(readers):
        code = "\n".join(
            line
            for line in (ROOT / "bin" / name).read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        for key in sorted(set(re.findall(r"\.checks\.([a-z_]+)", code))):
            if key not in emitted:
                unknown.append(f"{name}: .checks.{key}")

    assert not unknown, (
        "these read a check the summary does not record, so they read null and "
        "render it as a verdict:\n"
        + "\n".join(f"  - {u}" for u in unknown)
        + f"\n\n  Recorded checks: {sorted(emitted)}"
    )


def test_the_allowlist_check_reads_rules_rather_than_the_index(tmp_path):
    """Pins ``--no-index``, which the obvious implementation omits.

    This is worth a test rather than a comment because the omission produces a
    guard that **cannot fail**: ``git check-ignore`` skips a path that is in the
    index, so on precisely the tracked-and-ignored file the guard above exists
    to find, the plain form answers "not ignored" and the violation reads as
    compliant. There is a helper in this very file that omits it — correctly,
    for its own question — so reusing that one is the natural first move.

    Asserted in a throwaway repository rather than by staging a file in this
    one. Reaching into the developer's index to prove a point about git leaves
    a mess behind if the assertion fails, and the property being pinned belongs
    to git rather than to this repository's current contents.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(
        ".quality-artifacts/*\n!.quality-artifacts/kept.json\n", encoding="utf-8"
    )
    artifacts = tmp_path / ".quality-artifacts"
    artifacts.mkdir()
    (artifacts / "kept.json").write_text("{}", encoding="utf-8")
    (artifacts / "dropped.xml").write_text("<x/>", encoding="utf-8")

    subprocess.run(
        ["git", "add", "-f", ".gitignore", ".quality-artifacts"],
        cwd=tmp_path,
        check=True,
    )

    probe = [".quality-artifacts/kept.json", ".quality-artifacts/dropped.xml"]

    assert _excluded_by_gitignore(tmp_path, probe) == [".quality-artifacts/dropped.xml"], (
        "with --no-index, check-ignore must report the tracked-and-ignored file "
        "and only that one"
    )
    assert _excluded_by_gitignore(tmp_path, probe, no_index=False) == [], (
        "without --no-index, check-ignore reports nothing for tracked paths — "
        "if this ever stops being true the parameter can go, but until then it "
        "is the whole reason the guard above can fail at all"
    )


def test_the_committed_style_artifact_is_a_result_and_not_an_accident():
    """``style-check.json`` is committed, signed — and read by nothing at all.

    Not by ``validate-quality-artifacts.sh``, not by any other guard here. So
    whatever it holds is what CI accepts, and there are two ways for it to hold
    something other than a style result:

    * ``ruff`` exited 2 (bad flag, unreadable config) and the redirect captured
      an error message where JSON belongs.
    * ``ruff`` could not read a target. An unmatched glob is passed through
      literally and comes back as one ``E902`` io-error — **exit 1, valid
      JSON**, identical in shape to "found one style issue". That is what the
      gate produced when run from a subdirectory, before it started ``cd``-ing
      to the root, and no exit-status check can tell the two apart.

    Both mean the style check did not run over the code it claims to cover.
    The gate now refuses to write either; this is the guard for one already
    committed, which nothing else would notice.
    """
    artifact = ARTIFACTS / "style-check.json"
    assert artifact.is_file(), f"{artifact.name} is missing from {ARTIFACTS.name}/"

    raw = artifact.read_text(encoding="utf-8")
    try:
        findings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{artifact.name} is not JSON — ruff failed and its error was "
            f"captured as the result:\n{raw[:400]}"
        ) from exc

    assert isinstance(findings, list), f"{artifact.name} holds {type(findings).__name__}, not a list"

    unreadable = sorted(
        {
            entry.get("filename", "?")
            for entry in findings
            if isinstance(entry, dict) and entry.get("code") == "E902"
        }
    )
    assert not unreadable, (
        f"{artifact.name} reports targets ruff could not read: {unreadable}. "
        "This is not a style finding — it is the whole check not having run. "
        "Re-run ./bin/run-quality-checks.sh from the repository root."
    )


def test_no_coverage_report_is_committed():
    """Coverage XML is generated, multi-megabyte, and cannot fail the gate.

    It was committed on nearly every artifact run, conflicting on each one and
    accumulating over a gigabyte of object history, to carry a line-rate the
    validator reports as a warning and never fails on. That number lives in
    quality-summary.json now. Re-adding an un-ignore rule for a coverage report
    would quietly restore all of that.
    """
    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    unignored = [
        line.strip()
        for line in ignore_rules
        if line.strip().startswith("!.quality-artifacts/") and "coverage" in line
    ]
    assert not unignored, (
        f"coverage reports are un-ignored again: {unignored}. The gate reads "
        "coverage_percent from quality-summary.json and needs no XML committed."
    )


def test_the_merge_driver_gitattributes_names_is_actually_defined():
    """`merge=<name>` is a reference; an undefined driver silently does nothing.

    Git does not warn when a named merge driver has no `merge.<name>.driver`
    config — it falls back to the default text merge and conflicts exactly as if
    the attribute were absent. So .gitattributes on its own is indistinguishable
    from protection that works, right up until the merge that needed it.

    Asserting the setup script defines every driver .gitattributes names keeps
    the two from drifting: renaming the driver in one file and not the other
    fails here rather than at someone's next rebase.
    """
    attributes = GITATTRIBUTES.read_text(encoding="utf-8")
    named = set(re.findall(r"\bmerge=([A-Za-z0-9_.-]+)", attributes))
    assert named, (
        f"{GITATTRIBUTES.name} names no merge driver — if the merge=ours line "
        "was removed, delete this guard rather than leaving it passing vacuously"
    )

    setup = GIT_SETUP.read_text(encoding="utf-8")
    undefined = sorted(
        driver for driver in named if f'"merge.{driver}.driver"' not in setup
    )
    assert not undefined, (
        f"{GITATTRIBUTES.name} uses merge drivers {undefined} that "
        f"{GIT_SETUP.name} never configures. Git falls back to a text merge "
        "without warning, so the attribute would do nothing at all."
    )


def test_the_signature_covers_the_committed_set_on_both_sides():
    """Producer and verifier must enumerate the signed files the same way.

    They did not: the producer signed every ``*.json``/``*.xml`` on disk, which
    includes per-package coverage output that is gitignored, so the stored
    signature named files a CI checkout never contains and the comparison
    mismatched on every run. A check that always reports the same thing carries
    no information, which is how it ended up explicitly non-failing.

    Both sides enumerate through ``git ls-files`` now. If one reverts to a
    filesystem walk they diverge again, and silently.
    """
    producer = (ROOT / "bin" / "run-quality-checks.sh").read_text(encoding="utf-8")
    consumer = CONSUMER.read_text(encoding="utf-8")

    for name, body in (("run-quality-checks.sh", producer), (CONSUMER.name, consumer)):
        assert "git ls-files" in body, (
            f"{name} no longer enumerates signed artifacts with 'git ls-files', "
            "so it signs or verifies a different set than its counterpart."
        )


# The "a check that prints ✗ but cannot fail the build" guard used to live here,
# scoped to this one consumer and keyed to its print_fail / VALIDATION_FAILED
# vocabulary. The identical defect then turned up one script over, in
# run-quality-checks.sh, where the words are print_error and *_STATUS — so the
# guard that was meant to close the class had closed it in a single file. It now
# lives in tests/test_quality_gate_accounting.py, which takes the vocabulary as
# data and checks both scripts with one implementation.

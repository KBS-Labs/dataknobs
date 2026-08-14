"""Guards the per-rule census the ratchet's per-cell counts cannot give.

``bin/quality-contract.py`` compares each cell against a ceiling, and a ceiling
is denominated in findings. That answers *whether* a cell is over budget, and —
since the file names are carried out of the measurement — *where*. It has never
been able to answer *what*. The type checker writes its rule name at the end of
every finding, and the measurement's parse captured the path and read past the
rest, so the declaration could say ``packages/data/src: 657`` while saying
nothing about whether those 657 are one mechanical omission repeated or six
hundred separate judgements. The two have entirely different plans.

The census answers that from the same run, and that is where its risk is: two
readings of one output, where the second can quietly disagree with the first. In
this module the disagreement would run the same direction every other guard here
exists to refuse — toward a tidier tree than the one on disk, since a parse that
skips a line shape reports fewer findings than the measurement it is supposed to
decompose.

So what is asserted below is agreement, twice over: that the two bucketings of
one findings list total the same per cell, and that the census's findings are
the measurement's findings rather than a second reading of them. Plus the
self-tests those need in order to be worth anything — each parse driven over the
line shapes it has to survive, and each refusal driven to the point of refusing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from tests._workspace import ROOT, load_bin_module

CONTRACT = ROOT / ".dataknobs" / "quality-contract.json"
TOOL = ROOT / "bin" / "quality-contract.py"

contract_module = load_bin_module("quality-contract")


def _contract() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return loaded


def _biggest_ruff_cell(contract: dict[str, Any]) -> str:
    """The ruff cell carrying the largest ceiling.

    Asked rather than named, for the reason ``test_quality_contract.py`` asks
    it: the deferred cells are promoted one at a time, so a literal name fails
    with a ``KeyError`` on the day of a promotion — a guard going red over a
    change it holds no opinion about, and saying nothing about the property it
    is for.
    """
    ranked = sorted(contract["tools"]["ruff"]["cells"], key=lambda cell: cell["ceiling"])
    biggest = ranked[-1]
    assert biggest["ceiling"] > 5, (
        f"the largest ruff ceiling is {biggest['path']} at {biggest['ceiling']}, "
        "which is too small for a census over it to distinguish agreement from "
        "two empty tallies. The backlog has been cleared past what this assumes."
    )
    path: str = biggest["path"]
    return path


#: One mypy run, in the shapes the parse has to survive. Taken from real output
#: rather than invented: no column numbers (this repository does not enable
#: them), the rule name last and in brackets, ``note:`` continuations
#: interleaved, and a summary line at the end. The uncoded error is the one
#: shape not present in any current run and the one that would silently shorten
#: the census if it were dropped.
MYPY_OUTPUT = """\
packages/xization/src/dataknobs_xization/normalize.py:85: error: Incompatible \
default for parameter "subs"  [assignment]
packages/xization/src/dataknobs_xization/normalize.py:150: error: Need type \
annotation for "seen"  [var-annotated]
packages/xization/src/dataknobs_xization/normalize.py:150: note: Perhaps you \
need to add a type annotation
bin/quality-contract.py:12:9: error: Argument 1 has incompatible type "str"  [arg-type]
bin/quality-contract.py:20: error: Cannot determine consistent method resolution order
.venv/lib/python3.12/site-packages/somelib/core.py:9: error: Missing return  [return]
Found 5 errors in 4 files (checked 300 source files)
"""


def test_the_parse_keeps_every_error_line_and_no_other_line() -> None:
    """The one reading of mypy's output, driven over the shapes it has to survive.

    Three of them decide whether the census can be trusted against the
    measurement. A ``note:`` counted as a finding inflates the census above the
    tally; an ``error:`` dropped for carrying no bracketed rule deflates it
    below; and the summary line is a sentence containing a number, which is what
    a loose parse latches onto.

    The uncoded error is the interesting one, because no run in this repository
    currently produces one — so the shape is here rather than in the tree, and
    without it the first mypy release that emits one would shorten every census
    taken afterwards with nothing reporting that it had.
    """
    findings = contract_module.mypy_findings(MYPY_OUTPUT)

    assert len(findings) == 5, (
        "the parse did not find exactly the five error lines. A note counted as "
        f"a finding or an error missed both break parity with the tally: {findings}"
    )
    assert [finding.code for finding in findings] == [
        "assignment",
        "var-annotated",
        "arg-type",
        contract_module.UNCODED,
        "return",
    ], (
        "the rules came back wrong. The fourth line carries no bracketed code "
        "and must still be counted, under UNCODED rather than dropped: "
        f"{[finding.code for finding in findings]}"
    )


def test_a_finding_with_no_position_is_not_a_finding() -> None:
    """The parse decides what a finding *is*, and the tally trusts it.

    mypy writes several things that are not per-file findings — a fatal
    configuration complaint, the unused-section note the contract reads
    elsewhere, its own summary. None carries the ``path:line: error:`` shape,
    and none may be counted: every one of them would land in ``unattributed``,
    where it is reported as a finding against no cell, which reads as a hole in
    the contract's totality rather than as a parse fault.
    """
    noise = (
        "pyproject.toml: note: unused section(s): module = ['nltk.*']\n"
        "mypy: error: Cannot find config file 'nope.toml'\n"
        "Success: no issues found in 300 source files\n"
    )
    assert contract_module.mypy_findings(noise) == [], (
        "output with no per-file findings produced some anyway, which would be "
        "counted against a cell or reported as unattributable"
    )


def test_the_census_and_the_measurement_bucket_one_run_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the census rests on: two views, one run, one total per cell.

    A census is only comparable with the ceilings if it decomposes the same
    findings the ceilings are compared against. That holds structurally — both
    projections are built from one ``mypy_findings`` call — and this is what
    keeps it holding: an edit that gives either side its own parse, or its own
    cell attribution, breaks here rather than in a document six months later
    that reports per-rule counts summing to something other than the backlog.

    The run is injected rather than real. The property is about two bucketings
    of one output, so a genuine invocation would add minutes and test the type
    checker rather than this.
    """
    monkeypatch.chdir(ROOT)
    completed = subprocess.CompletedProcess(
        args=["mypy"], returncode=1, stdout=MYPY_OUTPUT, stderr=""
    )
    monkeypatch.setattr(contract_module, "_run", lambda _command: completed)
    contract = _contract()

    measurement = contract_module.measure_mypy(contract, [])
    run = contract_module.take_census(contract, "mypy")

    per_cell = {
        cell: sum(counts.values()) for cell, counts in measurement.by_cell.items() if counts
    }
    censused = {cell: sum(counts.values()) for cell, counts in run.census.by_cell.items() if counts}
    assert per_cell == censused, (
        "the measurement and the census disagree about how many findings each "
        f"cell holds: {per_cell} against {censused}. One of them is reading a "
        "different set of lines, and the census is the one nothing else checks."
    )
    assert sum(measurement.unattributed.values()) == sum(run.census.unattributed.values()), (
        "the two disagree about how many findings belong to no cell. The "
        "measurement keys those by file and the census by rule, so the counts "
        "are the only thing comparable — and they have to be."
    )
    assert per_cell.get("packages/xization/src") == 2, (
        f"the fixture's two xization findings did not land in that cell: {per_cell}"
    )
    assert sum(run.census.unattributed.values()) == 1, (
        "the finding reported against a path inside .venv is outside every cell "
        "and must be carried as unattributed, where check() warns about it — "
        f"mypy follows imports, so this is the ordinary case: {run.census.unattributed}"
    )


def test_the_census_agrees_with_the_linter_over_a_real_cell() -> None:
    """The same property against ruff, over a cell with a real backlog in it.

    Injected output proves the two bucketings agree; it cannot prove that the
    census reads ruff the way the measurement does, because both sides of that
    comparison were handed the same fabricated bytes. This runs the tool.

    Cheap enough to be worth it: ruff over one cell is a second or so, and it
    is the only assertion here that would catch the census and the measurement
    applying different guards to the same invocation — an unreadable file
    refused on one path and counted on the other, say.
    """
    contract = _contract()
    cell = _biggest_ruff_cell(contract)

    measurement = contract_module.measure_ruff(contract, contract_module.tracked_python(), {cell})
    run = contract_module.take_census(contract, "ruff", {cell})

    measured = sum(measurement.by_cell.get(cell, Counter()).values())
    censused = sum(run.census.by_cell.get(cell, Counter()).values())
    assert measured == censused, (
        f"{cell} measures {measured} findings and censuses {censused}. Two runs "
        "of ruff over one cell disagreeing means the census applies different "
        "guards to the same output, and its per-rule table is not a "
        "decomposition of the number the ceiling is compared against."
    )
    assert measured > 0, (
        f"{cell} carries the largest ruff ceiling in the contract and measured "
        "nothing, so this compared two empty tallies and asserted nothing"
    )


def test_a_finding_ruff_names_no_rule_for_is_still_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A syntax error carries a null ``code``, and must not vanish into it.

    ``.get("code", ...)`` returns its default only when the key is *absent*; ruff
    sends the key holding ``null``. Down that route the rule is ``None``, which
    is a perfectly good dictionary key — so the finding is counted, under a rule
    named ``None``, and the census grows a row that means nothing while still
    totalling correctly. The failure is legible only because the total is
    checked; the row is not.
    """
    completed = subprocess.CompletedProcess(
        args=["ruff"],
        returncode=1,
        stdout='[{"filename": "conftest.py", "code": null}]',
        stderr="",
    )
    monkeypatch.setattr(contract_module, "_run", lambda _command: completed)
    monkeypatch.chdir(ROOT)

    run = contract_module.take_census(_contract(), "ruff", {"conftest.py"})

    assert dict(run.census.by_cell["conftest.py"]) == {contract_module.UNCODED: 1}, (
        "a finding ruff named no rule for was not carried under UNCODED: "
        f"{run.census.by_cell.get('conftest.py')}"
    )


def test_an_unmeasured_cell_is_read_only_when_the_census_asks() -> None:
    """The one line standing between the bottom tier and a number.

    ``_UNMEASURED_TIERS`` is why ``packages/*/tests`` has no count: not that the
    run finds nothing there, but that the run is never pointed there, and the
    ceiling of zero ``verify`` insists those cells carry exists precisely so
    nothing reads their silence as a measurement.

    Both directions are asserted. That the flag widens the target set is the
    feature; that its absence leaves the set exactly as the ratchet measures it
    is what keeps the flag from quietly becoming the default and moving every
    number the contract holds.
    """
    cells = _contract()["tools"]["mypy"]["cells"]
    # Named as the declaration spells it, the way the other contract guards do,
    # rather than read from the module's own constant — the tier vocabulary is
    # what this leg's successors narrow, and a guard reading the code's copy of
    # it would follow a rename without anyone deciding to.
    unmeasured = {cell["path"] for cell in cells if cell["tier"] == "unchecked"}
    assert unmeasured, "the contract declares no unmeasured mypy cell, so this asserts nothing"

    narrow = contract_module.mypy_targets(cells, None, False)
    wide = contract_module.mypy_targets(cells, None, True)

    assert set(narrow) < set(wide), (
        "asking for the unmeasured cells did not widen the target set, so the "
        f"census would report the same numbers the ratchet already has: {narrow}"
    )
    added = set(wide) - set(narrow)
    assert all(
        any(contract_module.cell_matches(PurePosixPath(target), cell) for cell in unmeasured)
        for target in added
    ), (
        f"the widened run reaches {sorted(added)}, which includes a target "
        "outside every unmeasured cell — the flag is widening more than it says"
    )


def test_the_census_reports_a_cell_that_measured_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table holding only the cells with findings is one nobody can read.

    A census is quoted for months, and the difference between "this cell is
    clean" and "the run never reached this cell" is the difference between a
    finished leg and an unmeasured one. Listing only what was found renders the
    two identically — which is this repository's own defect class, an absence
    that reads as a result.
    """
    monkeypatch.chdir(ROOT)
    completed = subprocess.CompletedProcess(
        args=["mypy"], returncode=1, stdout=MYPY_OUTPUT, stderr=""
    )
    monkeypatch.setattr(contract_module, "_run", lambda _command: completed)

    report = contract_module.census_report(
        _contract(), "mypy", {"packages/xization/src", "packages/utils/src"}
    )

    listed = {entry["cell"]: entry["total"] for entry in report["cells"]}
    assert listed == {"packages/xization/src": 2, "packages/utils/src": 0}, (
        "a cell the run covered and found nothing in was left out of the table, "
        f"so it cannot be told from one the run never reached: {listed}"
    )


def test_the_first_party_set_is_read_from_the_configuration() -> None:
    """Which modules are ours decides which relaxations a census removes.

    Written down, that list goes stale on the day a package is added — and
    quietly, leaving the new package's strictness relaxation in force through a
    run whose entire purpose was to measure without them. Read from
    ``mypy_path``, it cannot.
    """
    modules = contract_module.first_party_modules(ROOT / "pyproject.toml")

    packaged = {path.name for path in ROOT.glob("packages/*/src/*") if path.is_dir()}
    assert modules >= {name for name in packaged if name.isidentifier()}, (
        f"a module this workspace ships is not in the first-party set: "
        f"{sorted(packaged - modules)}. Its relaxations would survive a census "
        "taken without them."
    )
    assert not modules & {"nltk", "numpy", "pandas", "asyncpg"}, (
        f"a third-party module was classified as first-party: {sorted(modules)}. "
        "Removing its ignore_missing_imports section would measure the absence "
        "of stubs in somebody else's library as our backlog."
    )


def test_removing_the_relaxations_removes_every_first_party_section_and_no_other() -> None:
    """The surgery, over the configuration it will actually be run against.

    The distinction it turns on is not cosmetic. A section naming ``nltk.*``
    waives the absence of type stubs in a library we do not own and cannot fix;
    one naming ``dataknobs_xization.*`` turns seven checks off over code we
    ship, and every finding it suppresses is one the declared configuration
    would otherwise report. A census that removed the first kind would measure
    somebody else's backlog as ours.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    first_party = contract_module.first_party_modules(ROOT / "pyproject.toml")

    stripped, removed = contract_module.config_without_relaxations(text, first_party)

    assert removed, "the real configuration carries first-party relaxations and none were removed"
    assert all(pattern.split(".")[0] in first_party for pattern in removed), (
        f"a section covering third-party modules was removed: {removed}"
    )
    assert "ignore_missing_imports" in stripped, (
        "the sections waiving missing stubs for third-party libraries were "
        "removed too, so the census would report their absence as our backlog"
    )
    assert "[tool.ruff]" in stripped, (
        "the surgery ran past the end of the mypy overrides and took another "
        "tool's configuration with it"
    )


def test_the_surgery_refuses_a_result_that_does_not_describe_what_was_asked(
    tmp_path: Path,
) -> None:
    """Located by header text, checked by re-parsing — because the first can miss.

    The blocks are found in the text and correlated with the parsed document by
    position. A header carrying a trailing comment is not the string being
    compared against, so its block survives while the parse still counts it as
    doomed. That is the whole failure mode of text surgery, and it fails toward
    a *lower* number: the relaxation stays in force and the census reports fewer
    findings under a heading claiming it was measured without them.

    Driven over a small configuration rather than the real one, because the
    fault has to be introduced and introducing it in ``pyproject.toml`` is not
    something a test may do.
    """
    config = tmp_path / "sample.toml"
    config.write_text(
        "[tool.mypy]\n"
        'mypy_path = "src"\n'
        "\n"
        "[[tool.mypy.overrides]]  # a comment the header match does not expect\n"
        'module = "ourpkg.*"\n'
        "warn_return_any = false\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as refusal:
        contract_module.config_without_relaxations(config.read_text(encoding="utf-8"), {"ourpkg"})

    assert "does not describe" in str(refusal.value), (
        f"the surgery left a first-party section in place and reported a "
        f"measurement anyway: {refusal.value}"
    )


def test_a_configuration_with_nothing_to_remove_is_refused() -> None:
    """Measuring "without the relaxations" when there are none is the same run.

    Returning the ratchet's own numbers under a heading that says otherwise is
    worse than refusing: the two censuses would be compared, found identical,
    and read as evidence that the relaxations cost nothing.
    """
    text = "[tool.mypy]\nwarn_return_any = true\n"

    with pytest.raises(SystemExit) as refusal:
        contract_module.config_without_relaxations(text, {"ourpkg"})

    assert "already measures" in str(refusal.value), (
        f"refused, but not for the reason it should have: {refusal.value}"
    )


def test_the_generated_configuration_does_not_outlive_the_run() -> None:
    """A second configuration left in the tree is the state this repository ended.

    ``mypy.ini`` and ``[tool.mypy]`` disagreed for as long as both existed, and
    the weaker one was what a developer saw. A scratch config abandoned by an
    interrupted run is that again, with the additional property that nobody
    wrote it deliberately.

    Both exits are driven: the ordinary one, and the exception — which is the
    one that leaves the file behind if the cleanup is not in a ``finally``.
    """
    scratch = ROOT / contract_module.CENSUS_CONFIG
    assert not scratch.exists(), f"{contract_module.CENSUS_CONFIG} is in the tree before any run"

    with contract_module.census_config(_contract()) as (config, removed):
        assert scratch.exists(), "the generated configuration was not written"
        assert config == contract_module.CENSUS_CONFIG
        assert removed, "nothing was reported as removed, so the run is measuring the same thing"
    assert not scratch.exists(), "the generated configuration outlived a clean exit"

    with pytest.raises(RuntimeError), contract_module.census_config(_contract()):
        raise RuntimeError("interrupted")
    assert not scratch.exists(), (
        "the generated configuration outlived a failed run, so the next commit "
        "picks up a second type-checker configuration nobody wrote"
    )


def test_what_a_census_writes_is_ignored_by_git() -> None:
    """The second layer under the ``finally`` above.

    A cleanup that runs is not a cleanup that always runs — a killed process
    skips it. Both the configuration and the cache it writes are ignored, so the
    worst case of an interrupted census is a stray file rather than a committed
    one.
    """
    for path in (contract_module.CENSUS_CONFIG, contract_module.CENSUS_CACHE):
        ignored = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False)
        assert ignored.returncode == 0, (
            f"{path} is not ignored, so an interrupted census leaves something a "
            "later commit can pick up"
        )


def test_the_census_refuses_a_tool_whose_unit_is_not_a_rule() -> None:
    """The formatter counts files it would rewrite, which does not decompose.

    Refused with a sentence rather than returning an empty table, because an
    empty table is what a clean tree looks like — and ``--tool`` accepting all
    three everywhere else is exactly why someone will try it.
    """
    result = subprocess.run(
        [sys.executable, str(TOOL), "census", "--tool", "format"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, "a census of the formatter was accepted"
    assert "category error" in result.stderr, (
        f"refused without saying why the request does not make sense: {result.stderr}"
    )


@pytest.mark.parametrize(
    ("option", "expected"),
    [("without_overrides", "has none of"), ("include_unmeasured", "widen the run by nothing")],
)
def test_an_option_a_tool_has_no_use_for_is_refused_rather_than_ignored(
    option: str, expected: str
) -> None:
    """A flag silently disregarded reports a narrower run under a wider heading.

    ruff has no per-module override sections and no cell in an unmeasured tier,
    so both of these would do nothing for it. Doing nothing quietly is how a
    census ends up filed as "ruff, without the relaxations" — a claim nothing in
    the output contradicts, because there was nothing to remove and the header
    would say so either way.
    """
    with pytest.raises(SystemExit) as refusal:
        contract_module.take_census(_contract(), "ruff", **{option: True})

    assert expected in str(refusal.value), (
        f"--{option.replace('_', '-')} was refused for the wrong reason: {refusal.value}"
    )

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
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from tests._workspace import ROOT, biggest_ruff_cells, load_bin_module

CONTRACT = ROOT / ".dataknobs" / "quality-contract.json"
TOOL = ROOT / "bin" / "quality-contract.py"

contract_module = load_bin_module("quality-contract")


def _contract() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return loaded


def _tracked() -> list[PurePosixPath]:
    """The population the measurers are handed.

    Passed explicitly by every caller below, including the ones that inject the
    run and never look at a file. It used to be sound to hand ``measure_mypy`` an
    empty list, because it took the population and ignored it; it now decides
    which directories hold Python worth opening, so an empty one is a run with no
    targets — which returns without calling the tool at all and quietly measures
    nothing. That is the shape every guard in this module is against, and it
    would have arrived here as three tests passing for the wrong reason.
    """
    tracked: list[PurePosixPath] = contract_module.tracked_python()
    assert tracked, "git tracks no *.py here, so nothing below measures anything"
    return tracked


def _unmeasured_mypy_cell() -> str:
    """One mypy cell the contract puts in a tier no tool reads.

    Named from the declaration for the reason ``biggest_ruff_cells`` is: the
    bottom tier is what this whole leg exists to empty, so a literal name here
    fails on the day one of them is promoted, over a change these guards hold no
    opinion about.
    """
    cells = _contract()["tools"]["mypy"]["cells"]
    unmeasured: list[str] = sorted(cell["path"] for cell in cells if cell["tier"] == "unchecked")
    assert unmeasured, "the contract declares no unmeasured mypy cell, so this asserts nothing"
    return unmeasured[0]


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

    measurement = contract_module.measure_mypy(contract, _tracked())
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
    (cell,) = biggest_ruff_cells(contract)

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

    tracked = contract_module.tracked_python()
    narrow = contract_module.mypy_targets(cells, None, False, tracked)
    wide = contract_module.mypy_targets(cells, None, True, tracked)

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


def test_every_target_is_a_directory_the_checker_can_be_pointed_at() -> None:
    """A cell can name a directory holding no Python at all.

    The other two measurers cannot reach this: both are handed the tracked
    ``*.py`` population, and a file list cannot contain a path that is not a
    Python file. Only mypy is pointed at directories, and mypy exits 2 on a
    directory with no ``.py[i]`` in it — reporting nothing, over every other
    target in the same pass.

    Which is not a small thing to leave to the exit-status guard. Before that
    guard existed the empty stdout would have parsed to an empty finding list and
    the widened census would have reported that the unmeasured cells hold
    nothing, which is both the strongest claim this flag can make and the exact
    inversion of the truth. The guard turns it into a refusal, so what is left is
    a flag that cannot run rather than a number that is wrong — better, and still
    not the feature.

    ``packages/*/docs`` is why: seven directories match it, one holds a single
    ``.py`` file, and the cell exists so that file is in some cell rather than
    silently in none. Totality is a claim about files, and this asserts that the
    targets derived from it are a claim about directories worth opening.
    """
    cells = _contract()["tools"]["mypy"]["cells"]
    tracked = contract_module.tracked_python()

    for target in contract_module.mypy_targets(cells, None, True, tracked):
        assert any(contract_module.cell_matches(path, target) for path in tracked), (
            f"{target} is handed to the type checker and holds no tracked *.py "
            "file, so the pass exits 2 and measures nothing anywhere"
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


def test_a_header_carrying_a_comment_is_still_the_section_it_names(tmp_path: Path) -> None:
    """The likely spelling, which the header match did not accept.

    ``[[tool.mypy.overrides]]  # why this exists`` is valid TOML and, in a
    configuration where nearly every decision carries its reason on the line
    beside it, the natural thing to write. It was not the string being compared
    against, so the block survived while the parsed document still counted it as
    doomed — the correlation shifted and every later section came out one place
    wrong.

    The re-parse below caught that and refused, which is what it is for. But a
    feature that stops working whenever somebody annotates a header is not a
    usable feature, and its complaint mentions nothing about comments. So the
    comment comes off before the comparison, and what is pinned here is that the
    annotated section is the one found.
    """
    config = tmp_path / "commented.toml"
    config.write_text(
        "[tool.mypy]\n"
        'mypy_path = "src"\n'
        "\n"
        "[[tool.mypy.overrides]]  # ours, relaxed while the backlog clears\n"
        'module = "ourpkg.*"\n'
        "warn_return_any = false\n"
        "\n"
        "[[tool.mypy.overrides]]\n"
        'module = "somelib.*"\n'
        "ignore_missing_imports = true\n",
        encoding="utf-8",
    )

    stripped, removed = contract_module.config_without_relaxations(
        config.read_text(encoding="utf-8"), {"ourpkg"}
    )

    assert removed == ["ourpkg.*"], f"the annotated section was not the one removed: {removed}"
    assert "somelib" in stripped, (
        "the third-party section went instead of, or as well as, the annotated one"
    )
    assert "ourpkg" not in stripped, "the first-party section survived its own removal"


def test_the_surgery_refuses_a_result_that_does_not_describe_what_was_asked(
    tmp_path: Path,
) -> None:
    """Located by header text, checked by re-parsing — because the first can miss.

    The blocks are found in the text and correlated with the parsed document by
    position, and any header spelling the match does not accept breaks that
    correlation. The common one is a trailing comment, which the test above now
    pins as accepted. What remains are spellings TOML allows and nobody writes:
    ``[[ tool.mypy.overrides ]]``, with spaces inside the brackets, is one.

    Deliberately unmatched rather than not yet matched. Every spelling the match
    accepts is one this check no longer has to catch, and a match loose enough to
    accept anything has stopped locating sections at all. So what is asserted
    here is the floor under that decision: whatever the match misses comes back
    as a refusal rather than as a number.

    It needs a floor because it fails toward a *lower* one. The relaxation stays
    in force, and the census reports fewer findings under a heading claiming it
    measured without them.
    """
    config = tmp_path / "sample.toml"
    config.write_text(
        "[tool.mypy]\n"
        'mypy_path = "src"\n'
        "\n"
        "[[ tool.mypy.overrides ]]\n"
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


def test_an_override_section_naming_both_kinds_is_refused_rather_than_resolved(
    tmp_path: Path,
) -> None:
    """A section covering ours and theirs has no right answer, so it gets none.

    The classification was per *section* and decided by ``any``: one first-party
    name anywhere in a section removed the whole section. Today's configuration
    carries one holding fifteen third-party patterns, and adding a single
    ``dataknobs_*`` name to it would have taken all fifteen with it — the census
    then reporting the absence of type stubs in somebody else's library as our
    own backlog, under a heading saying the relaxations had been removed.

    Removing it and keeping it are both wrong, so neither is chosen silently.
    """
    config = tmp_path / "mixed.toml"
    config.write_text(
        "[tool.mypy]\n"
        'mypy_path = "src"\n'
        "\n"
        "[[tool.mypy.overrides]]\n"
        'module = ["ourpkg.*", "somelib.*"]\n'
        "ignore_missing_imports = true\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as refusal:
        contract_module.config_without_relaxations(config.read_text(encoding="utf-8"), {"ourpkg"})

    message = str(refusal.value)
    assert "ourpkg.*" in message and "somelib.*" in message, (
        f"refused without naming both halves of the section it cannot split: {message}"
    )


def test_an_override_pattern_that_names_no_module_is_refused(tmp_path: Path) -> None:
    """A leading wildcard is unclassifiable, and defaulting it is the quiet answer.

    ``"*.tests.*"`` is a pattern mypy accepts. Split on the first dot it yields
    ``"*"``, which is in no first-party set — so it classified as third-party,
    the section survived, and a relaxation over our own code stayed in force
    through a run taken to remove exactly those. The direction is the one this
    module refuses everywhere else: fewer findings, under a heading claiming
    more were looked for.
    """
    config = tmp_path / "wild.toml"
    config.write_text(
        "[tool.mypy]\n"
        'mypy_path = "src"\n'
        "\n"
        "[[tool.mypy.overrides]]\n"
        'module = "ourpkg.*"\n'
        "warn_return_any = false\n"
        "\n"
        "[[tool.mypy.overrides]]\n"
        'module = "*.tests.*"\n'
        "warn_return_any = false\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as refusal:
        contract_module.config_without_relaxations(config.read_text(encoding="utf-8"), {"ourpkg"})

    assert "*.tests.*" in str(refusal.value), (
        f"refused without naming the pattern it could not classify: {refusal.value}"
    )


def test_a_module_shipped_as_one_file_counts_as_first_party(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reading only directories misses it, and the miss runs the quiet direction.

    Every module this workspace ships today is a package, so this is a hole
    rather than a live fault — and it is the hole the rest of this file exists to
    refuse. A first-party name read as third-party keeps its override sections
    through a run whose whole purpose was to remove them, and the census then
    reports a smaller number under a heading saying they are gone.

    Driven against a relocated root, because the set is read from ``mypy_path``
    relative to the repository and the shape being tested is not in it.
    """
    (tmp_path / "src" / "packaged").mkdir(parents=True)
    (tmp_path / "src" / "solo.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "notes.txt").write_text("", encoding="utf-8")
    config = tmp_path / "sample.toml"
    config.write_text('[tool.mypy]\nmypy_path = "src"\n', encoding="utf-8")
    monkeypatch.setattr(contract_module, "_ROOT", tmp_path)

    assert contract_module.first_party_modules(config) == {"packaged", "solo"}, (
        "a module shipped as a single file was not read as first-party, or "
        "something that is not a module was"
    )


def test_the_sections_a_census_removes_turn_real_checks_off() -> None:
    """Removing them has to change the number, or the flag reports the same run.

    The surgery test pins *which* sections go. This pins that going changes
    something: a section naming first-party modules and setting nothing would be
    removed, reported in the header as removed, and leave the census identical to
    the one the ratchet already takes — the two compared, found equal, and read
    as evidence that the relaxations cost nothing.

    What it does not prove is the size of the difference. That needs two full
    type-checker runs over the workspace, which is minutes, and belongs to the
    command rather than to a guard on it.
    """
    parsed = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sections = parsed["tool"]["mypy"]["overrides"]
    first_party = contract_module.first_party_modules(ROOT / "pyproject.toml")

    doomed = [
        section
        for section in sections
        if contract_module._relaxes_first_party(section, first_party)
    ]
    assert doomed, "no section relaxes first-party checking, so this asserts nothing"

    toothless = [
        section["module"] for section in doomed if not [key for key in section if key != "module"]
    ]
    assert not toothless, (
        f"the sections naming {toothless} would be removed by a census without "
        "the relaxations, but they relax nothing — so removing them changes no "
        "finding while the header reports them as removed"
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


def test_the_type_checker_measurer_refuses_a_status_that_is_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mypy that never ran must not measure every type-checked cell at zero.

    mypy exits 2 on a config file it cannot find, on a usage error and on a
    blocking error, having written nothing to stdout — and an empty stdout parses
    to an empty finding list. So every mypy cell measured 0, ``check`` found
    nothing above a ceiling and exited 0, and ``bin/validate.sh`` printed a green
    type-check verdict over a tree nothing had opened. The way back is worse:
    ``update-baseline`` sees every cell under its ceiling and writes the zeroes
    down, and a ceiling only ever falls.

    The linter and the formatter both refused this already, each with its own
    copy of the check; the type checker had neither. The census then added two
    fresh routes to a mypy that cannot start — a generated config deleted by a
    concurrent run, and a target set pointed outside ``mypy_path`` — of which the
    first would have reported "0 findings without the relaxations", the strongest
    claim this tool can make.

    Injected, because the fault is a tool that did not run and no input to a
    working mypy produces it.
    """
    completed = subprocess.CompletedProcess(
        args=["mypy"],
        returncode=2,
        stdout="",
        stderr="mypy: error: Cannot find config file '.mypy-census.toml'",
    )
    monkeypatch.setattr(contract_module, "_run", lambda _command: completed)

    with pytest.raises(SystemExit) as refusal:
        contract_module.measure_mypy(_contract(), _tracked())

    assert "exited 2" in str(refusal.value), (
        f"the measurer refused, but not for the reason it should have: {refusal.value}"
    )


def test_the_type_checker_measurer_refuses_a_status_its_output_contradicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit 1 with nothing the parse recognises is a short count, not a clean tree.

    The half the status check cannot reach. A blocking error — two modules
    sharing a basename, which ``packages/*/tests`` holds nine of — is reported as
    ``path: error: ...`` with no line number, a shape ``_MYPY_FINDING_RE`` does
    not match and must not: a parse loose enough to catch it catches the summary
    line too. So the status says "found something" while the tally says the tree
    is clean, and the gap between them is the whole finding.

    Which makes this the guard that keeps ``--include-unmeasured`` honest. That
    flag points mypy at cells outside ``mypy_path`` with no package structure,
    and a blocking error there would otherwise report the bottom tier as empty —
    the one claim that tier is declared specifically not to support.
    """
    blocking = 'packages/a/tests/test_registry.py: error: Duplicate module named "test_registry"\n'
    completed = subprocess.CompletedProcess(args=["mypy"], returncode=1, stdout=blocking, stderr="")
    monkeypatch.setattr(contract_module, "_run", lambda _command: completed)

    with pytest.raises(SystemExit) as refusal:
        contract_module.measure_mypy(_contract(), _tracked())

    assert "disagree" in str(refusal.value), (
        f"refused without saying the status and the output disagree: {refusal.value}"
    )


def test_a_census_of_a_tool_whose_unit_is_not_a_rule_never_reaches_a_measurer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused at the layer that dispatches, not only at the command line.

    The refusal lived in ``main``. So ``take_census(contract, "format")`` — which
    this module already calls directly, and a second subcommand would — fell
    through a branch meaning "not ruff" into the *type checker's* measurer, and
    filed mypy's findings under the formatter's cells. The two cell lists
    overlap, so what came back was a plausible table of type errors under a
    ``format census`` heading, exit 0, with nothing in it saying the wrong tool
    had run.

    ``_run`` is replaced with something that fails if it is called at all: what
    is asserted is not only that the request is refused, but that nothing was
    measured before refusing it.
    """

    def _never(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"a census of the formatter invoked {command[:3]}")

    monkeypatch.setattr(contract_module, "_run", _never)

    with pytest.raises(SystemExit) as refusal:
        contract_module.take_census(_contract(), "format")

    assert "category error" in str(refusal.value), (
        f"refused, but not for the reason it should have: {refusal.value}"
    )


def test_a_scope_of_cells_nothing_reads_is_refused_rather_than_left_blank() -> None:
    """``0 finding(s)`` over an empty table is also what a clean tree prints.

    Naming an unmeasured cell asks for a row this run cannot produce: the target
    set skips the cell and the report filters it back out. What came back was a
    header reading ``0 finding(s)``, an empty ``per cell`` and an empty ``per
    rule`` — indistinguishable from a cell measured and found clean, which is the
    distinction this feature's own documentation says it exists to preserve. A
    mixed scope was quieter still: the unmeasured cells simply vanished from a
    table the caller had named them in.

    The inverse guard was already here — the flag refused when no cell needs it.
    This is the direction it was missing.
    """
    with pytest.raises(SystemExit) as refusal:
        contract_module.census_report(_contract(), "mypy", {_unmeasured_mypy_cell()})

    assert "--include-unmeasured" in str(refusal.value), (
        f"refused without naming the flag that answers the request: {refusal.value}"
    )


def test_a_generated_configuration_already_in_the_tree_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two censuses share one path, and the first to finish deletes it.

    ``CENSUS_CONFIG`` is a module constant, so concurrent runs in one checkout
    write the same file: A writes, B writes, A's mypy finishes, A unlinks, and
    B's mypy starts against a configuration that is no longer there. Before the
    status guard above, that was ``0 findings under .mypy-census.toml``.

    Refusing also covers the narrower harm, which needs no concurrency at all: an
    operator with a file of their own at that path used to lose it silently.

    The second assertion is the point of refusing rather than overwriting. A
    guard that then deleted the file it declined to overwrite would have done the
    damage it exists to prevent.
    """
    (tmp_path / "src" / "ourpkg").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        "[tool.mypy]\n"
        'mypy_path = "src"\n'
        "\n"
        "[[tool.mypy.overrides]]\n"
        'module = "ourpkg.*"\n'
        "warn_return_any = false\n",
        encoding="utf-8",
    )
    squatter = tmp_path / contract_module.CENSUS_CONFIG
    squatter.write_text("someone else's file\n", encoding="utf-8")
    monkeypatch.setattr(contract_module, "_ROOT", tmp_path)

    contract = {"tools": {"mypy": {"config": "pyproject.toml"}}}
    with pytest.raises(SystemExit) as refusal, contract_module.census_config(contract):
        raise AssertionError("a census ran against a configuration it did not write")

    assert "already in the tree" in str(refusal.value), (
        f"refused, but not for the reason it should have: {refusal.value}"
    )
    assert squatter.read_text(encoding="utf-8") == "someone else's file\n", (
        "the refusal deleted or rewrote the file it declined to overwrite"
    )


def test_a_census_that_ran_reports_a_backlog_without_failing() -> None:
    """Not a verdict — the one command whose whole purpose is to read a backlog.

    Exiting non-zero over a tree with findings would make it look like a failing
    check, and a caller would learn to ignore its status. At which point the
    refusals above stop being heard, since the status is the only thing carrying
    them.

    Driven over a real cell with a real backlog, so the zero is a status reported
    *despite* findings rather than in the absence of any.
    """
    (cell,) = biggest_ruff_cells(_contract())
    result = subprocess.run(
        [sys.executable, str(TOOL), "census", "--tool", "ruff", "--cell", cell, "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"a census over a cell carrying a backlog exited {result.returncode}: {result.stderr}"
    )
    assert json.loads(result.stdout)["total"] > 0, (
        f"{cell} carries the largest ruff ceiling and censused nothing, so the "
        "exit status above was reported over an empty run"
    )


def test_a_census_does_not_move_the_thing_it_measures() -> None:
    """Not a ratchet move. Only ``update_baseline`` writes the declaration.

    True by inspection today, and guarded by nothing else — which is the
    combination worth a test rather than a reading. The census and the baseline
    share the measuring path, so a helper down there that learned to write would
    be caught here and nowhere else. A measurement that also moved the ceiling it
    was taken against leaves nobody able to say what the tree looked like
    beforehand.
    """
    before = CONTRACT.read_bytes()
    (cell,) = biggest_ruff_cells(_contract())

    subprocess.run(
        [sys.executable, str(TOOL), "census", "--tool", "ruff", "--cell", cell],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert CONTRACT.read_bytes() == before, (
        "a census rewrote .dataknobs/quality-contract.json, so the measurement "
        "moved the ceiling it was being compared against"
    )


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("verify", ["--cell", "tests"]),
        ("verify", ["--show-findings"]),
        ("partition", ["--without-overrides"]),
        ("partition", ["--include-unmeasured"]),
    ],
)
def test_an_option_a_command_does_not_read_is_a_usage_error(command: str, argv: list[str]) -> None:
    """Every option used to be global, and no command was obliged to read one.

    ``check --without-overrides`` ran an ordinary check under the declared
    configuration and said nothing about the flag it discarded;
    ``update-baseline --cell <one>`` validated the name and rewrote every ceiling
    the tool has. Both are runs answering a narrower question than the one asked,
    under a heading saying otherwise — the defect ``take_census`` refuses per
    tool, one layer up with nothing refusing it.

    Subparsers make it structural rather than a table somebody keeps in step: a
    command that does not declare an option rejects it, and a new option cannot
    be added without choosing which commands honour it.

    The pairs driven here are the ones that stay cheap if the guard regresses.
    ``check`` and ``update-baseline`` would measure the whole tree — and the
    second would rewrite the declaration — so they are argued from the same
    parser rather than executed against it.
    """
    result = subprocess.run(
        [sys.executable, str(TOOL), command, *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, (
        f"{command} accepted {argv}, which it never reads — so it ran and "
        f"reported under a heading that does not describe it:\n{result.stdout}"
    )
    assert "unrecognized arguments" in result.stderr, (
        f"{command} {argv} failed, but for some other reason: {result.stderr}"
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

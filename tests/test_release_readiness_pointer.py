"""The release-readiness pointer must never go quiet.

``bin/release-helper.sh`` had no step naming what a cut makes permanent, so a
deliberate refusal or a published guarantee behind an unreleased door was free
to change right up to the release and a migration afterwards -- with nothing at
the place the release is run to say so. ``release_readiness`` is that step.

**What these pin is not the wording.** It is that the reminder cannot vanish.
A pointer whose manifest is deleted, whose ``jq`` is missing, or whose target
section has been renamed under it is exactly the failure the pointer exists to
prevent, arriving one level up -- so every degraded state has to *say* something,
and silence is the only failing outcome.

The manifest holds pointers rather than lists on purpose: the reasoning behind a
readiness item belongs with the plan that produced it, and a second copy here
would give a cut two sources that can disagree.

**An entry names the section, not the file.** There is no path in the manifest;
the heading is the whole pointer and the reminder searches the tree for it. Two
of the tests below exist because the obvious ways to run that search are both
wrong -- a substring match is satisfied by prose that merely QUOTES a heading,
and a whole-line match is satisfied by nothing at all, because a ruled heading
carries a trailing marker the manifest does not. What is pinned here is the
predicate that works: a line that BEGINS with the heading.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests._workspace import ROOT

SCRIPT = ROOT / "bin" / "release-helper.sh"
#: Resolved once, and invoked by absolute path: one case below runs with a PATH
#: that carries nothing, and a relative "bash" is not findable from there.
BASH = shutil.which("bash") or "/bin/bash"
MANIFEST = ROOT / ".dataknobs" / "release-readiness.json"

#: Keys an entry must carry for the reminder to be able to print and verify it.
REQUIRED = ("tree", "section", "heading", "why")

#: Keys an entry must NOT carry. ``document`` was the path into the planning
#: tree; it is the citation this manifest stopped making.
FORBIDDEN = ("document", "path", "file")


def _manifest() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(MANIFEST.read_text())
    return loaded


def _run(
    manifest: Path | None = None, tree: Path | None = None, path: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the reminder alone, with the seams a test needs.

    Returns the whole result rather than its text: one state below is a
    *refusal*, and the exit code is the half of that promise a message cannot
    carry. A helper that returns only the text cannot be asked about it, which
    is how the refusal came to be documented and unasserted.
    """
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin" if path is None else path,
        "HOME": str(Path.home()),
    }
    if manifest is not None:
        env["DK_RELEASE_READINESS_MANIFEST"] = str(manifest)
    if tree is not None:
        env["DK_XDOCS_DIR"] = str(tree)
    return subprocess.run(
        [BASH, str(SCRIPT), "readiness"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _readiness(
    manifest: Path | None = None, tree: Path | None = None, path: str | None = None
) -> str:
    """Everything the reminder said, on either stream."""
    done = _run(manifest=manifest, tree=tree, path=path)
    return done.stdout + done.stderr


def test_the_manifest_is_well_formed() -> None:
    """Every entry names a declared tree and carries what the printer reads.

    A missing key would print the string ``null`` beside a package name, which
    reads like a pointer and is not one.
    """
    data = _manifest()
    trees = data["trees"]
    assert trees, "a manifest with no trees can resolve no pointer"

    entries = [e for entries in data["packages"].values() for e in entries]
    assert entries, "a manifest with no entries makes every assertion below vacuous"

    for entry in entries:
        for key in REQUIRED:
            assert entry.get(key), f"{entry} is missing {key}"
        for key in FORBIDDEN:
            assert key not in entry, (
                f"{entry} carries {key!r} -- an entry names the SECTION, not the "
                f"file that holds it, so a path here is the citation this "
                f"manifest stopped making"
            )
        assert entry["tree"] in trees, f"{entry['tree']} is not a declared tree"
        assert entry["heading"].lstrip().startswith("#"), (
            f"{entry['heading']!r} is not a markdown heading -- the search "
            f"anchors at the line start, so a heading that does not look like "
            f"one would match prose"
        )

    for name, tree in trees.items():
        assert tree.get("path"), f"tree {name} has no default path"
        assert tree.get("env"), f"tree {name} has no environment override"


def test_the_check_command_reaches_the_reminder() -> None:
    """The trigger is at the site that fires it, not in somebody's memory.

    This is the whole point of the change: ``check`` is what a person runs
    before cutting, so the reminder has to be reachable from ``check`` rather
    than from a command they would have to already know about.
    """
    text = SCRIPT.read_text()
    start = text.index("check_changes() {")
    end = text.index("\n}\n", start)
    assert "release_readiness" in text[start:end], (
        "check_changes no longer calls release_readiness -- the reminder is "
        "reachable only by somebody who already knows it exists"
    )


def test_an_absent_manifest_is_reported_rather_than_assumed_empty(tmp_path: Path) -> None:
    """Deleting the manifest must not read as 'nothing outstanding'."""
    out = _readiness(manifest=tmp_path / "gone.json")
    assert "not being checked" in out


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq reads the manifest")
def test_a_present_pointer_is_confirmed_against_its_target(tmp_path: Path) -> None:
    """The happy path, built here rather than depending on a checkout.

    Without this the drift assertions below are satisfied by a printer that
    reports drift unconditionally.

    **The document is at a path the manifest never named**, which is the whole
    change: the section is found because the tree is searched for its heading,
    so a document that merely moved still resolves instead of reporting a drift
    that is not one. The path is printed, because the manifest no longer
    carries one and this line is where a reader learns which file to open.

    **The heading carries a trailing ruling marker**, as a ruled heading in
    that tree does. The manifest's text is therefore a PREFIX of the line and
    never the whole of it, which is why the search cannot be a whole-line match.
    """
    entry = next(iter(_manifest()["packages"].values()))[0]
    document = tmp_path / "tree" / "somewhere" / "else" / "renamed.md"
    document.parent.mkdir(parents=True)
    document.write_text(f"{entry['heading']} -- ruled `D216` 2026-09-10\n\nbody\n")

    out = _readiness(tree=tmp_path / "tree")
    assert "found" in out
    assert "somewhere/else/renamed.md" in out, (
        "the manifest holds no path, so the reminder must print the one it "
        "resolved to or nobody can open the document"
    )


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq reads the manifest")
def test_an_absent_tree_is_distinguished_from_a_drifted_pointer(tmp_path: Path) -> None:
    """A clone without the planning checkout is normal and is not a warning."""
    out = _readiness(tree=tmp_path / "nowhere")
    assert "not checked out here" in out
    assert "DRIFTED" not in out


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq reads the manifest")
def test_a_heading_in_no_document_is_reported_as_drift(tmp_path: Path) -> None:
    """The tree is there and nothing in it is headed this.

    **One state where there used to be two.** A path-based pointer could tell
    "the file moved" from "the section was renamed"; a search cannot, and does
    not need to -- the first of those is no longer a failure, because a moved
    document still resolves. What is left is the case that always mattered: the
    section is gone.

    Both shapes are exercised, an empty tree and a tree of documents that are
    headed something else, because they are one verdict now and a test that
    covered only the empty one would pass over a printer that never searched.

    Drift is reported and *survived*, which is the control for the refusal at
    the bottom of this file: a non-zero exit there means "the tool is missing"
    rather than "this function exits non-zero whenever it has something to
    complain about".
    """
    (tmp_path / "empty").mkdir()
    done = _run(tree=tmp_path / "empty")
    assert "DRIFTED" in done.stdout + done.stderr
    assert done.returncode == 0, "drift is a warning, not a gate"

    populated = tmp_path / "tree"
    (populated / "plans").mkdir(parents=True)
    (populated / "plans" / "other.md").write_text("## Some other section\n\nbody\n")
    assert "DRIFTED" in _readiness(tree=populated)


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq reads the manifest")
def test_a_heading_in_two_documents_is_reported_as_ambiguous(tmp_path: Path) -> None:
    """A heading that identifies two sections has stopped being a pointer.

    The state a path could not reach, and so the one the search brings with it.
    Reported rather than resolved by picking the first match: which of the two
    the entry meant is not something this function can know, and choosing
    silently would leave a reader reading the wrong list.

    A warning like drift, not a refusal -- the release is not blocked on it.
    """
    entry = next(iter(_manifest()["packages"].values()))[0]
    tree = tmp_path / "tree"
    for name in ("first.md", "second.md"):
        document = tree / "plans" / name
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(f"{entry['heading']}\n\nbody\n")

    done = _run(tree=tree)
    out = done.stdout + done.stderr

    assert "AMBIGUOUS" in out
    assert "plans/first.md" in out and "plans/second.md" in out, (
        "an ambiguity the reader cannot see the members of is not actionable"
    )
    assert done.returncode == 0, "ambiguity is a warning, not a gate"


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq reads the manifest")
def test_prose_that_quotes_the_heading_is_not_a_match(tmp_path: Path) -> None:
    """A document that TALKS ABOUT the section is not the document that has it.

    **This is the defect the search shipped with if the match is a substring,
    and it is not hypothetical.** The ruling that removed the path from this
    manifest quotes the heading four times in its own record, in the very tree
    the pointer searches -- so on the day it landed, a substring search returned
    five documents where one is headed the section and four merely mention it.
    A planning tree discusses its own headings; that is what a planning tree is.

    Anchoring at the line start separates them, and both halves are asserted:
    a mention alone is drift, and a mention beside a real heading still resolves
    to exactly one rather than reporting an ambiguity that does not exist.
    """
    entry = next(iter(_manifest()["packages"].values()))[0]
    tree = tmp_path / "tree"
    (tree / "plans").mkdir(parents=True)
    mention = tree / "plans" / "discussion.md"
    mention.write_text(f"The list lives at `{entry['heading']}` and nobody has read it.\n")

    assert "DRIFTED" in _readiness(tree=tree), (
        "a quoted heading is prose about a section, not a section"
    )

    real = tree / "plans" / "actual.md"
    real.write_text(f"{entry['heading']} -- ruled `D216` 2026-09-10\n\nbody\n")

    out = _readiness(tree=tree)
    assert "found" in out and "AMBIGUOUS" not in out
    assert "plans/actual.md" in out


def test_the_reminder_fails_loudly_without_jq(tmp_path: Path) -> None:
    """``jq`` reads the manifest, and its absence must not read as silence.

    The one degraded state that arrives without anybody editing anything -- a
    runner whose image simply does not carry ``jq``. It names the tool, gives
    the two install lines, and leaves behind the instruction that still works:
    read the file yourself.

    **It exits non-zero rather than continuing.** A tool we invoke as a
    subprocess must fail loudly when absent -- a check that skips because its
    tool is missing reports success having verified nothing, which is this
    reminder's own subject one level up. ``tests/test_quality_gate_accounting``
    holds every tracked shell file to that, and this is the same rule read from
    the other side: what the branch must SAY, where that guard fixes what it
    must DO.

    ``jq`` is hidden by building a PATH that carries the script's other tools
    and not that one, rather than by emptying PATH: an empty PATH breaks the
    script's own preamble, which would pass this assertion for the wrong reason.
    """
    stand_in = tmp_path / "bin"
    stand_in.mkdir()
    for tool in ("dirname", "basename", "sed", "grep", "sort", "uniq", "cat", "tr"):
        found = shutil.which(tool)
        if found:
            (stand_in / tool).symlink_to(found)
    assert shutil.which("dirname", path=str(stand_in)), "the stand-in PATH is unusable"
    assert not shutil.which("jq", path=str(stand_in)), "jq is still reachable"

    done = _run(path=str(stand_in))
    out = done.stdout + done.stderr

    assert done.returncode != 0, "a missing tool must not read as success"
    assert "jq is not installed" in out
    assert ".dataknobs/release-readiness.json" in out

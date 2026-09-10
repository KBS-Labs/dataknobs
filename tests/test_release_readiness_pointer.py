"""The release-readiness pointer must never go quiet.

``bin/release-helper.sh`` had no step naming what a cut makes permanent, so a
deliberate refusal or a published guarantee behind an unreleased door was free
to change right up to the release and a migration afterwards -- with nothing at
the place the release is run to say so. ``release_readiness`` is that step.

**What these pin is not the wording.** It is that the reminder cannot vanish.
A pointer whose manifest is deleted, whose ``jq`` is missing, or whose target
document has been renamed under it is exactly the failure the pointer exists to
prevent, arriving one level up -- so every degraded state has to *say* something,
and silence is the only failing outcome.

The manifest holds pointers rather than lists on purpose: the reasoning behind a
readiness item belongs with the plan that produced it, and a second copy here
would give a cut two sources that can disagree.
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
REQUIRED = ("tree", "document", "section", "heading", "why")


def _manifest() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(MANIFEST.read_text())
    return loaded


def _readiness(
    manifest: Path | None = None, tree: Path | None = None, path: str | None = None
) -> str:
    """Run the reminder alone, with the seams a test needs."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin" if path is None else path,
        "HOME": str(Path.home()),
    }
    if manifest is not None:
        env["DK_RELEASE_READINESS_MANIFEST"] = str(manifest)
    if tree is not None:
        env["DK_XDOCS_DIR"] = str(tree)
    done = subprocess.run(
        [BASH, str(SCRIPT), "readiness"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
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
        assert entry["tree"] in trees, f"{entry['tree']} is not a declared tree"

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

    Without this the three drift assertions below are satisfied by a printer
    that reports drift unconditionally.
    """
    entry = next(iter(_manifest()["packages"].values()))[0]
    document = tmp_path / "tree" / entry["document"]
    document.parent.mkdir(parents=True)
    document.write_text(f"{entry['heading']}\n\nbody\n")

    assert "found" in _readiness(tree=tmp_path / "tree")


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq reads the manifest")
def test_an_absent_tree_is_distinguished_from_a_drifted_pointer(tmp_path: Path) -> None:
    """A clone without the planning checkout is normal and is not a warning."""
    out = _readiness(tree=tmp_path / "nowhere")
    assert "not checked out here" in out
    assert "DRIFTED" not in out


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq reads the manifest")
def test_a_moved_document_is_reported_as_drift(tmp_path: Path) -> None:
    """The tree is there and the document is not -- somebody moved it."""
    (tmp_path / "empty").mkdir()
    assert "DRIFTED" in _readiness(tree=tmp_path / "empty")


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq reads the manifest")
def test_a_renamed_section_is_reported_as_drift(tmp_path: Path) -> None:
    """The document is there and the section is not.

    The weaker check -- does the file exist -- passes over a document whose
    readiness section was renamed or removed, which is the likelier of the two
    and the one that leaves the pointer aiming at nothing in particular.
    """
    entry = next(iter(_manifest()["packages"].values()))[0]
    document = tmp_path / "tree" / entry["document"]
    document.parent.mkdir(parents=True)
    document.write_text("## Some other section\n\nbody\n")

    assert "no section headed" in _readiness(tree=tmp_path / "tree")


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

    out = _readiness(path=str(stand_in))

    assert "jq is not installed" in out
    assert ".dataknobs/release-readiness.json" in out

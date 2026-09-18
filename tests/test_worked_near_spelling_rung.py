"""The near-spelling section of the common guide runs, and prints what it says it prints.

``dataknobs-common``'s ``entity-resolution.md`` argues for one number -- the
``0.85`` threshold -- by showing what a lower one returns, and it makes the
argument in **hits and spans**. Those are the whole reason the section exists,
and a number in prose is guarded by nobody.

**It runs from the workspace root**, for :mod:`tests.test_worked_entity_resolution_call_site`'s
reason, restated because it is the reason and not a filing preference: a test
inside ``packages/common/tests`` can reach anything the package defines whether
or not the door exports it, and the section's first line is
``from dataknobs_common.entity_resolution import ... LexicalSignal``.

**It executes the fences themselves** rather than a character-identical ``.py``
copy, as :mod:`tests.test_worked_authority_rung` does and for that file's
reason: what is being pinned is not only that each block *runs* but that its
**declared output** is what running it prints, and a copy in a second file
could not carry that. The cost is that these fences are not linted; what is
bought is that the page cannot state a span the code does not produce.

**Only the marked fences**, and in the order the page carries them, in one
namespace -- the later blocks use ``breeds`` and ``typo``, which the first one
binds. The page has a dozen other python fences belonging to other guards, so
this one selects by marker rather than by language.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Any

import pytest

from tests._workspace import ROOT, code_fences, rel

if TYPE_CHECKING:
    from tests._workspace import Fence

GUIDE = ROOT / "packages" / "common" / "docs" / "guides" / "entity-resolution.md"

#: The section's fences, in the order they must run.
MARKERS = ("worked-near-spelling", "worked-threshold", "worked-cutoff")


#: Every fence, **including the untagged ones**, which is why this is read here
#: rather than through ``tests._workspace.code_fences``. That reader's
#: ``FENCE_OPEN`` requires a language, so an output block written as a bare
#: ```` ``` ```` -- which is how this repository's documents write one -- is
#: invisible to it. It is still the reader for the python half below, so the
#: two agree on what a python fence is.
_FENCE = re.compile(r"^```(?P<lang>[\w+-]*)[^\n]*\n(?P<body>.*?)^```\s*$", re.M | re.S)


def _pairs() -> list[tuple[Fence, str | None]]:
    """Each marked fence with the text of the block declaring its output.

    The python halves come from ``code_fences`` -- so the marker and the line
    number are the shared reader's -- and are matched to the untagged blocks
    positionally. The assertion that the two readings agree on the bodies is
    what makes that pairing sound rather than hopeful.
    """
    raw = [
        (match.group("lang").lower(), match.group("body"))
        for match in _FENCE.finditer(GUIDE.read_text(encoding="utf-8"))
    ]
    marked = {fence.marker: fence for fence in code_fences(GUIDE) if fence.marker in MARKERS}

    found: list[tuple[Fence, str | None]] = []
    for marker in MARKERS:
        fence = marked.get(marker)
        assert fence is not None, f"{rel(GUIDE)} no longer carries a <!-- {marker} --> fence"
        at = next(
            (index for index, (lang, body) in enumerate(raw) if body == fence.body + "\n"),
            None,
        )
        assert at is not None, (
            f"the two fence readings disagree about the `{marker}` block in {rel(GUIDE)}"
        )
        following = raw[at + 1] if at + 1 < len(raw) else None
        declared = following[1] if following is not None and following[0] != "python" else None
        found.append((fence, declared))
    return found


@pytest.fixture(scope="module")
def ran() -> dict[str, str]:
    """What each marked fence printed, keyed by its marker.

    Module-scoped and executed once: the namespace is cumulative by design, so
    re-running per assertion would either rebuild it each time or -- worse --
    leave the tests below depending on the order pytest happens to run them in.
    ``pytest-randomly`` is configured for this repository, so that is not a
    hypothetical.
    """
    namespace: dict[str, Any] = {}
    printed: dict[str, str] = {}
    for fence, _declared in _pairs():
        out = io.StringIO()
        with redirect_stdout(out):
            exec(compile(fence.body, f"{rel(GUIDE)}:{fence.line}", "exec"), namespace)
        printed[str(fence.marker)] = out.getvalue()
    return printed


def test_every_marked_fence_declares_an_output() -> None:
    """Three blocks, each followed by the block saying what it prints.

    A count and a list of markers, because the failure this guards against is
    silent: a fence deleted from the page takes its assertion with it, and
    every remaining assertion keeps passing.
    """
    pairs = _pairs()
    assert [fence.marker for fence, _ in pairs] == list(MARKERS)

    undeclared = [fence.marker for fence, declared in pairs if declared is None]
    assert undeclared == [], (
        f"{undeclared} no longer declare an output, so nothing checks what they print"
    )


@pytest.mark.parametrize("marker", MARKERS)
def test_each_declared_output_is_what_the_fence_prints(ran: dict[str, str], marker: str) -> None:
    """The page's numbers, against the ones the code produces.

    Parametrized per fence rather than looped, so a failure names the block
    that drifted instead of the first one that did.
    """
    declared = next(text for fence, text in _pairs() if fence.marker == marker)
    assert declared is not None

    assert ran[marker].strip() == declared.strip(), (
        f"the `{marker}` fence in {rel(GUIDE)} prints something other than the "
        f"block below it declares. The section argues for a threshold in hits "
        f"and spans, so a declared output that is no longer produced is the "
        f"page teaching a behaviour the code does not have."
    )

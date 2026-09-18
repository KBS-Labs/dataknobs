"""The authority-rung guide runs, and the outputs it prints are the ones on the page.

``dataknobs-xization``'s ``entity-resolution.md`` makes the comparison between
the injected authority rung and ``dataknobs-common``'s default, and it makes it
in **numbers** -- which spans come back per axis and per form, which id a
pattern resolves to. Those numbers are the whole reason the page exists, and a
number in prose is guarded by nobody.

**It runs from the workspace root**, for :mod:`tests.test_worked_ontology_call_site`'s
reason, restated because it is the reason and not a filing preference: a test
inside ``packages/xization/tests`` can reach anything the package defines
whether or not the door exports it, and the page's first line is
``from dataknobs_xization import AuthoritySignal``.

**It differs from its three siblings in one way, deliberately.** They hold a
published fence and an executed ``.py`` copy character-identical, so the code a
reader copies is also code that is linted. This one executes the fences
themselves, because what it is pinning is not only that the block *runs* but
that each block's **declared output** is what running it prints -- and a copy
in a second file could not carry that. The cost is that these fences are not
linted; what is bought is that the page cannot state a span the code does not
produce.

Every fence runs **in order, in one namespace**, because that is how a reader
reads the page: the later blocks use ``breeds`` and ``re``, which the first one
binds. Running one fence alone would pass a page whose blocks had stopped
composing.
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

GUIDE = ROOT / "packages" / "xization" / "docs" / "entity-resolution.md"


#: Every fence, **including the untagged ones**, which is why this is read
#: here rather than through ``tests._workspace.code_fences``. That reader's
#: ``FENCE_OPEN`` requires a language, so an output block written as a bare
#: ```` ``` ```` -- which is how this repository's documents write one, by a
#: margin of hundreds -- is invisible to it. It is still the reader for the
#: python half below, so the two agree on what a python fence is.
_FENCE = re.compile(r"^```(?P<lang>[\w+-]*)[^\n]*\n(?P<body>.*?)^```\s*$", re.M | re.S)


def _pairs() -> list[tuple[int, Fence, str | None]]:
    """Each python fence with the text of the block declaring its output, if any.

    The python halves come from ``code_fences`` -- so the marker and the line
    number are the shared reader's -- and are matched to the untagged blocks
    positionally. The assertion below that the two readings agree on the
    python fences is what makes that pairing sound rather than hopeful.
    """
    raw = [
        (match.group("lang").lower(), match.group("body"))
        for match in _FENCE.finditer(GUIDE.read_text(encoding="utf-8"))
    ]
    shared = [fence for fence in code_fences(GUIDE) if fence.lang == "python"]
    bodies = [body for lang, body in raw if lang == "python"]
    assert [fence.body.strip() for fence in shared] == [body.strip() for body in bodies], (
        f"the two fence readings disagree about {rel(GUIDE)}'s python blocks"
    )

    found: list[tuple[int, Fence, str | None]] = []
    at = 0
    for index, (lang, _body) in enumerate(raw):
        if lang != "python":
            continue
        following = raw[index + 1] if index + 1 < len(raw) else None
        declared = following[1] if following is not None and following[0] != "python" else None
        found.append((shared[at].line, shared[at], declared))
        at += 1
    return found


@pytest.fixture(scope="module")
def ran() -> dict[int, str]:
    """What each python fence printed, keyed by the line it starts on.

    Module-scoped and executed once: the namespace is cumulative by design, so
    re-running per assertion would either rebuild it each time or -- worse --
    leave the tests below depending on the order pytest happens to run them in.
    ``pytest-randomly`` is configured for this repository, so that is not a
    hypothetical.
    """
    namespace: dict[str, Any] = {}
    printed: dict[int, str] = {}
    for line, fence, _declared in _pairs():
        out = io.StringIO()
        with redirect_stdout(out):
            exec(compile(fence.body, f"{rel(GUIDE)}:{line}", "exec"), namespace)
        printed[line] = out.getvalue()
    return printed


def test_the_page_carries_the_fences_this_file_thinks_it_does() -> None:
    """Five python blocks, four of which declare an output.

    A count, because the failure this guards against is silent: a fence
    deleted from the page takes its assertion with it, and every remaining
    assertion keeps passing. The markers are named too, so a block renamed
    rather than removed is caught by the same failure.
    """
    pairs = _pairs()
    assert len(pairs) == 5, f"{rel(GUIDE)} has {len(pairs)} python fences, not 5"

    markers = [fence.marker for _line, fence, _declared in pairs]
    assert markers == [
        None,
        "worked-authority-rung",
        "worked-overlap",
        "worked-registry",
        "worked-factory",
    ], f"the page's fences are no longer the ones this file pins: {markers}"

    declared = [fence.marker for _line, fence, output in pairs if output is not None]
    assert declared == [
        "worked-authority-rung",
        "worked-overlap",
        "worked-registry",
        "worked-factory",
    ], "a fence that declared an output no longer does, so nothing checks it"


@pytest.mark.parametrize(
    ("line", "marker"),
    [(line, fence.marker) for line, fence, output in _pairs() if output is not None],
)
def test_each_declared_output_is_what_the_fence_prints(
    ran: dict[int, str], line: int, marker: str
) -> None:
    """The page's numbers, against the ones the code produces.

    Parametrized per fence rather than looped, so a failure names the block
    that drifted instead of the first one that did.
    """
    declared = next(output for at, _f, output in _pairs() if at == line)
    assert declared is not None

    assert ran[line].strip() == declared.strip(), (
        f"the `{marker}` fence at {rel(GUIDE)}:{line} prints something other "
        f"than the block below it declares. The page is the documentation for "
        f"a difference between two rungs, stated in spans and ids -- so a "
        f"declared output that is no longer produced is the page teaching a "
        f"behaviour the code does not have."
    )

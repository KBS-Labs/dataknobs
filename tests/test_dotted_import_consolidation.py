"""Dynamic imports live in one module, workspace-wide.

The companion to ``test_dotted_path_agreement.py``. That one checks the sites
we know about still behave alike; this one checks no new site appears. Neither
substitutes for the other — an agreement table cannot notice a copy nobody
adds to it, and a source scan cannot notice an adopted site drifting.

Run from the workspace root over every package because the copies this
consolidation removed were spread across four of them, and a per-package
version of this file would be four files that can each be deleted by a
refactor of the package that owns it.

What is left in ``DEFERRED`` is not a backlog — see the comments on each.
"""

from __future__ import annotations

from dataknobs_common.testing import assert_no_ad_hoc_dotted_import

from tests._workspace import ROOT

#: Sites deliberately left resolving their own dotted paths, each with the
#: reason. What remains is not a backlog: the three same-shape copies in
#: ``config``, ``fsm`` and ``xization`` have been adopted, and neither of these
#: two resolves a dotted path from configuration at all. They import
#: dynamically, which is all the scan can see.
#:
#: Recorded here rather than remembered, because ``allow=`` fails on an entry
#: that matches nothing. That cuts both ways and both are wanted: adopting a
#: site breaks this test until its entry is dropped, and an entry whose site
#: *moved* breaks it too — a suppression that silently stops covering its line
#: is a hole that reads as a clean scan. Expect to re-run and re-read the line
#: numbers after any edit above one of them.
DEFERRED = {
    # Takes a pre-split `FunctionRef`, so it parses no path — but it does
    # import dynamically, which is what the scan sees.
    "fsm/src/dataknobs_fsm/config/builder.py:848",
    # Inside `_cli_main`, `# pragma: no cover`: parses a CLI argument and
    # exits. Not config-driven resolution at all.
    "llm/src/dataknobs_llm/prompts/syntax.py:491",
}


def test_dynamic_imports_live_only_in_the_canonical_resolver() -> None:
    assert_no_ad_hoc_dotted_import(
        *sorted((ROOT / "packages").glob("*/src")),
        allow=DEFERRED,
    )

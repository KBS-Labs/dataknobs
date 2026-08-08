"""Dynamic imports live in one module, workspace-wide.

The companion to ``test_dotted_path_agreement.py``. That one checks the sites
we know about still behave alike; this one checks no new site appears. Neither
substitutes for the other — an agreement table cannot notice a copy nobody
adds to it, and a source scan cannot notice an adopted site drifting.

Run from the workspace root over every package because the copies this
consolidation removed were spread across four of them, and a per-package
version of this file would be four files that can each be deleted by a
refactor of the package that owns it.
"""

from __future__ import annotations

from dataknobs_common.testing import assert_no_ad_hoc_dotted_import

from tests._workspace import ROOT

#: Sites deliberately left resolving their own dotted paths, each with the
#: reason. The consolidation stopped at the nine in one package to keep the
#: change reviewable; these three are the same shape in three other packages
#: and are a mechanical follow-up.
#:
#: Recorded here rather than remembered, because ``allow=`` fails on an entry
#: that matches nothing — so adopting one of these breaks this test until the
#: entry is dropped, which is what closes the follow-up rather than leaving it
#: to be noticed.
DEFERRED = {
    # Resolves a builder callable from an object-graph config. Same shape,
    # different package; adoption is mechanical.
    "config/src/dataknobs_config/builders.py:303",
    # Resolves a *resource class* and instantiates it with no shape check —
    # the instantiate-then-check pattern this item removed from two sites,
    # surviving in a third package. Adoption is a real behaviour change.
    "fsm/src/dataknobs_fsm/config/builder.py:357",
    # Takes a pre-split `FunctionRef`, so it parses no path — but it does
    # import dynamically, which is what the scan sees.
    "fsm/src/dataknobs_fsm/config/builder.py:859",
    # Inside `_cli_main`, `# pragma: no cover`: parses a CLI argument and
    # exits. Not config-driven resolution at all.
    "llm/src/dataknobs_llm/prompts/syntax.py:486",
    # `_resolve_dotted_import(dotted_path, base_type)` — independently
    # written, and almost exactly `resolve_class`. The single best evidence
    # that the shape is right, in the package furthest from the one that had
    # nine copies.
    "xization/src/dataknobs_xization/chunking/registry.py:85",
}


def test_dynamic_imports_live_only_in_the_canonical_resolver() -> None:
    assert_no_ad_hoc_dotted_import(
        *sorted((ROOT / "packages").glob("*/src")),
        allow=DEFERRED,
    )

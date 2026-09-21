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
#: ``config``, ``fsm`` and ``xization`` have been adopted, and none of these
#: three resolves a dotted path from configuration at all. They import
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
    "fsm/src/dataknobs_fsm/config/builder.py:851",
    # Inside `_cli_main`, `# pragma: no cover`: parses a CLI argument and
    # exits. Not config-driven resolution at all.
    "llm/src/dataknobs_llm/prompts/syntax.py:494",
    # `DataclassSweep._walk`: imports each module `pkgutil.walk_packages`
    # enumerated over a package's own `__path__`. Nothing here is a dotted
    # path a consumer wrote, so none of the four decisions the canonical
    # resolver settles arises -- there is no separator to choose, no
    # attribute to look up, and a typo is impossible because nothing typed
    # the name. The exception handling is the construct's subject rather
    # than an oversight: a module that will not import is recorded as a
    # hole in the sweep, because one silently skipped reads as a clean tree.
    "common/src/dataknobs_common/testing/dataclass_sweep.py:184",
    # The `dataknobs_utils` package door's PEP 562 `__getattr__`: imports one
    # of its **own** submodules by a name that has already been checked
    # against a frozen literal set three lines above, so an unknown name
    # raises `AttributeError` before this line is reached. None of the four
    # decisions the canonical resolver settles arises -- there is no
    # separator (the name is a bare identifier), no attribute to look up, no
    # shape to check, and a typo cannot get here. Reaching for
    # `dataknobs_common.imports` would also import `dataknobs_common` at this
    # door, which is the cost the lazy door exists to remove.
    "utils/src/dataknobs_utils/__init__.py:100",
}


def test_dynamic_imports_live_only_in_the_canonical_resolver() -> None:
    assert_no_ad_hoc_dotted_import(
        *sorted((ROOT / "packages").glob("*/src")),
        allow=DEFERRED,
    )

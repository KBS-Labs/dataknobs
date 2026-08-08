"""Source guard against ad-hoc dotted-path resolution.

Turning a dotted string from configuration into a live Python object is one
operation with several decisions in it — which separator to accept, what to
raise, whether to check the target's shape before or after constructing it,
whether a typo is fatal. Written once it is a policy; written nine times it is
nine policies that agree until they do not.

That is not hypothetical. One package accumulated nine copies, and they
disagreed on all four: three accepted only ``:``, four only ``.``, and two
either; four raised four different exception types and two did not raise at
all; two validated the target's shape before constructing it and two after, so
a mistyped path ran an unrelated class's ``__init__``. Nobody set out to write
nine — each was written because the existing one was not findable, being
exported from no ``__init__`` and reached through a function-local deep import.

The companion to this scan is an *agreement* test, and neither substitutes for
the other:

============ ================================== ==========================
Guard        Catches                            Cannot catch
============ ================================== ==========================
Agreement    drift between the known sites      a tenth copy nobody adds
Source scan  a tenth copy, the day it is        drift within an adopted
             written                            site
============ ================================== ==========================

Usage::

    from dataknobs_common.testing import assert_no_ad_hoc_dotted_import

    def test_no_ad_hoc_dotted_path_resolution():
        assert_no_ad_hoc_dotted_import(
            *(root / "packages").glob("*/src"),
            allow={
                # Resolves a builder callable; adoption tracked separately.
                "config/builders.py:303",
            },
        )

It is a source scan, not a runtime check: the defect is a shape in the code,
and a copy can sit on a path that needs a live service to reach.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "CANONICAL_MODULE",
    "AdHocImportFinding",
    "assert_no_ad_hoc_dotted_import",
]

#: The module allowed to call ``importlib.import_module``. Everything else
#: should be calling it *through* this one.
CANONICAL_MODULE = "dataknobs_common/imports.py"

#: Call targets that resolve a name to a module at runtime, matched on the
#: bare name. ``import_module`` is the one every copy used. Both names are
#: distinctive enough that a same-named method on an unrelated object would
#: be a surprise.
_DYNAMIC_IMPORT_CALLS = frozenset({"import_module", "__import__"})

#: The same operation spelled differently, matched **only** when qualified by
#: its module. Listed so that "avoid the scan" and "write a tenth copy" are
#: not the same edit — but qualified, because these names are generic enough
#: to collide with methods that do something else entirely. ``resolve_name``
#: is the live example: a config loader in this workspace has a
#: ``resolve_name`` hook that maps a config *name* to a *path*, and matching
#: it on the bare name flagged a public API of a neighbouring subsystem as a
#: tenth copy of this one.
_QUALIFIED_DYNAMIC_IMPORT_CALLS = frozenset(
    {("pkgutil", "resolve_name"), ("pydoc", "locate")}
)


class AdHocImportFinding(NamedTuple):
    """One dynamic-import call outside the canonical module."""

    path: Path
    lineno: int
    call: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: calls {self.call}()"


def _dynamic_import_call(node: ast.Call) -> str | None:
    """The name of a dynamic-import call target, or ``None`` if it is not one.

    ``a.b.import_module(...)`` matches on the bare attribute;
    ``pkgutil.resolve_name(...)`` matches only with its module, so a
    same-named method elsewhere is not flagged.
    """
    func = node.func

    if isinstance(func, ast.Name):
        return func.id if func.id in _DYNAMIC_IMPORT_CALLS else None

    if isinstance(func, ast.Attribute):
        if func.attr in _DYNAMIC_IMPORT_CALLS:
            return func.attr
        receiver = func.value
        qualifier = (
            receiver.id
            if isinstance(receiver, ast.Name)
            else receiver.attr
            if isinstance(receiver, ast.Attribute)
            else None
        )
        if (qualifier, func.attr) in _QUALIFIED_DYNAMIC_IMPORT_CALLS:
            return f"{qualifier}.{func.attr}"

    return None


def _scan_file(path: Path) -> list[AdHocImportFinding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        # A file the scan cannot parse is not a file that can hold a copy.
        return []

    return [
        AdHocImportFinding(path, node.lineno, name)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _dynamic_import_call(node))
    ]


def assert_no_ad_hoc_dotted_import(
    *roots: Path,
    allow: Iterable[str] = (),
    canonical: str = CANONICAL_MODULE,
) -> None:
    """Assert dynamic imports happen only in the canonical resolver.

    Args:
        *roots: Directories (or files) to scan for ``*.py``.
        allow: ``"<path-suffix>:<lineno>"`` entries to exempt, for a site
            reviewed and deliberately left alone. Each needs a comment saying
            why. Matched on a path-component boundary, so ``"builder.py:357"``
            exempts that line in *every* ``builder.py`` under the roots while
            ``"fsm/config/builder.py:357"`` exempts one — give as much path as
            you mean.

            **An entry matching nothing is an error.** A suppression whose
            site moved is a hole, and a silent one reads as a clean scan.
            This is the same hard-won rule the error-text guard carries, for
            the same reason.
        canonical: Path suffix of the module allowed to import dynamically.

    Raises:
        AssertionError: Listing every finding, so one run reports the whole
            surface rather than the first offender.

    Note:
        ``importlib.util.find_spec`` is not flagged — it answers whether a
        module *could* be imported without importing it, which is the probe an
        optional-dependency guard makes and not this operation at all.
    """
    exempt = frozenset(allow)
    used: set[str] = set()
    findings: list[AdHocImportFinding] = []

    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            posix = path.as_posix()
            if posix.endswith(canonical):
                continue
            for finding in _scan_file(path):
                key = f"{posix}:{finding.lineno}"
                matched = {
                    entry
                    for entry in exempt
                    if key == entry or key.endswith(f"/{entry}")
                }
                used |= matched
                if not matched:
                    findings.append(finding)

    if stale := sorted(exempt - used):
        listed = "\n".join(f"  {entry}" for entry in stale)
        raise AssertionError(
            f"{len(stale)} allow entr(ies) matched no flagged site:\n{listed}\n\n"
            "Either the site was adopted — drop the entry — or it moved and "
            "the suppression silently stopped covering it."
        )

    if findings:
        listed = "\n".join(f"  {f}" for f in findings)
        raise AssertionError(
            f"{len(findings)} dynamic import(s) outside {canonical}:\n{listed}\n\n"
            "Resolving a dotted path from config is one operation with four "
            "decisions in it — separator, exception type, shape-check order, "
            "and whether a typo is fatal. Use "
            "`dataknobs_common.imports.resolve_dotted` / `resolve_callable` / "
            "`resolve_class` so those decisions are made once. If this site "
            "genuinely differs, add it to `allow=` with a reason."
        )

"""The two heights of one measure name each other, and nothing else says so.

:func:`~dataknobs_common.ontology.ontology_support` counts every vocabulary a
corpus's tags name and holds no vocabulary at all;
``OntologyRegistry.ontologies_in_play`` narrows that answer to the ones one
deployment holds. They ship in two distributions, and the second is a filter
over the first rather than a second count --- which is the whole reason
:class:`~dataknobs_common.ontology.OntologySupport` is published from
``dataknobs-common`` rather than from where the narrowing lives.

What pays for that split is a cross-reference in both directions, so a reader
who reaches either half finds the other. It is prose, so nothing enforced it,
and prose is exactly what a later edit tidies away without noticing what it was
for. This pair is worse than most: the ``common`` side names the narrowing **by
symbol**, from a distribution that cannot import it, so a rename on the ``data``
side leaves two correct-looking sentences naming a member that no longer exists
and no test anywhere reporting it.

**Read as text, from the workspace root, and deliberately.** ``common`` does not
depend on ``data``, and the guards in ``packages/common/tests`` exist in part to
keep it that way --- one of them runs an import in a subprocess and fails on any
``dataknobs_data`` module in ``sys.modules``. A test asserting this
cross-reference from inside that suite would have to import the package those
guards exist to exclude. :mod:`tests.test_ontology_tag_family_cross_reference`
is the sibling this copies, and states the same reason for the same location
over a different pair.

**The dependency is asymmetric and this guard is not.** ``data`` depends on
``common``, so its half of the cross-reference *could* be asserted from
``packages/data/tests``. It is asserted here instead because the two directions
are one fact, and splitting them across two suites is how one goes quiet while
the other keeps passing.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from dataknobs_common.ontology import OntologySupport, ontology_support

from dataknobs_data.ontology import OntologyRegistry

from tests._workspace import ROOT

#: Spelled once each, and read through these rather than beside them.
#:
#: ``ascent.py`` is declared in ``_WORKSPACE_ONLY_QUALITY_INPUTS``
#: (``bin/changed-packages.py``) for the reason the two entries beside it give:
#: the package scope covering the source looks like coverage of the workspace
#: guard that reads it, and those are different sets. Both guides already ride
#: the docs scope.
COUNTS = ROOT / "packages/common/src/dataknobs_common/ontology/ascent.py"
COUNTS_GUIDE = ROOT / "packages/common/docs/guides/roll-up.md"
NARROWS_GUIDE = ROOT / "packages/data/docs/guides/ontology-registry.md"

#: The two names the prose on either side spells out.
NARROWING = "ontologies_in_play"
MEASURE = "ontology_support"


def _class_docstring(path: Path, name: str) -> str:
    """The docstring of one class in a module read as text, or ``""``."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_docstring(node) or ""
    return ""


def test_the_member_the_counting_side_names_by_symbol_exists() -> None:
    """The half of this that a text comparison cannot reach.

    Both assertions below would keep passing over a ``common`` docstring naming
    a member ``data`` had renamed or removed: they compare one string against
    another string, and neither string is the code. This resolves the name.

    It also pins the flavour, because the sentence being guarded says the
    member awaits nothing --- a narrowing that became a coroutine would leave
    every reference to it correct and every published call site broken.
    """
    narrowing = getattr(OntologyRegistry, NARROWING, None)

    assert narrowing is not None, (
        f"OntologyRegistry has no {NARROWING!r}, and two places in "
        f"dataknobs-common name it by symbol. Either restore the member or "
        f"correct {COUNTS.name} and {COUNTS_GUIDE.name} in the same change"
    )
    assert callable(narrowing)
    assert not inspect.iscoroutinefunction(narrowing), (
        f"{NARROWING} has become a coroutine, so the published call sites that "
        "read it without an await are now wrong"
    )


def test_the_counting_half_names_the_narrowing_half() -> None:
    """``OntologySupport``'s own docstring says who else answers this type.

    The direction that is easier to lose. ``ascent.py`` is about counting and
    knows nothing of a registry, so a reader editing it has no reason to know
    the sentence is load-bearing --- and no test in its own package could tell
    them, because asserting it there means importing ``dataknobs_data``.
    """
    doc = _class_docstring(COUNTS, OntologySupport.__name__)

    assert doc, f"{OntologySupport.__name__} has no docstring in {COUNTS.name}"
    assert NARROWING in doc, (
        f"{OntologySupport.__name__} no longer names {NARROWING}, so the type "
        "reads as having one caller when the reason it is published from here "
        "is that it has two"
    )


def test_the_counting_guide_names_the_narrowing_half() -> None:
    """The roll-up guide points at the member that narrows its answer."""
    text = COUNTS_GUIDE.read_text(encoding="utf-8")

    assert NARROWING in text, (
        f"{COUNTS_GUIDE.name} no longer names {NARROWING}, so a reader holding "
        "a page of results and a registry is left to find the narrowing"
    )


def test_the_narrowing_half_names_the_counting_half() -> None:
    """And the registry guide points back at the unnarrowed measure.

    This is the recovery door for everything the narrowing drops, so a page
    that stops naming it turns a silent drop into an unrecoverable one.
    """
    text = NARROWS_GUIDE.read_text(encoding="utf-8")

    assert MEASURE in text, (
        f"{NARROWS_GUIDE.name} no longer names {MEASURE}, which is the only "
        "way back to the vocabularies the narrowing dropped"
    )
    assert ontology_support.__name__ == MEASURE, (
        "the measure has been renamed, so the guide names a function that no "
        "longer exists under that name"
    )


def test_the_three_files_this_guard_names_all_exist() -> None:
    """A path guard whose paths have moved passes by reading nothing.

    Each assertion above parses or reads one of these three. If any were
    renamed, ``read_text`` would raise --- but a guard that fails only by
    raising ``FileNotFoundError`` says nothing useful about which half moved.
    """
    assert COUNTS.is_file()
    assert COUNTS_GUIDE.is_file()
    assert NARROWS_GUIDE.is_file()

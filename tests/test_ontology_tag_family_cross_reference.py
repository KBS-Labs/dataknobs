"""The two halves of the tag family name each other, and nothing else says so.

Five keys describe one indexed row. Four of them --- which vocabulary, which
axis, which node, which surface forms --- are about **identity** and live in
``dataknobs_common.ontology.tags``. The fifth, the model that produced the
vector, is about an **embedder** and lives in ``dataknobs_data.vector.content``,
where it already shipped and where five files import it.

Splitting a family across two packages is a cost, and the thing that pays for
it is a cross-reference in both directions: a reader who reaches either module
finds the other. That is prose, so nothing enforced it, and prose is exactly
what a later edit tidies away without noticing what it was for.

**Read as text, from the workspace root, and deliberately.** ``common`` does
not depend on ``data``, and the guards in ``packages/common/tests`` exist in
part to keep it that way --- one of them runs an import in a subprocess and
fails on any ``dataknobs_data`` module in ``sys.modules``. A test asserting
this cross-reference from inside that suite would have to import the package
those guards exist to exclude. Reading both files as text is indifferent to
import direction, which is the property that makes the guard placeable at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._workspace import ROOT

#: Spelled once each, and read through these rather than beside them.
#:
#: Both were written twice --- once here, once as a bare string argument
#: below --- and the two spellings answer different questions: the existence
#: guard checks these, ``_files_the_workspace_guards_read`` populates its
#: hash scope from these, and the assertions read the strings. A rename would
#: leave the first two agreeing about a file the third had stopped reading.
IDENTITY_KEYS = ROOT / "packages/common/src/dataknobs_common/ontology/tags.py"
EMBEDDER_KEY = ROOT / "packages/data/src/dataknobs_data/vector/content.py"


def _module_docstring(path: Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""


def test_the_identity_half_names_the_embedder_half() -> None:
    """``tags.py`` says where the fifth key of the same row lives."""
    doc = _module_docstring(IDENTITY_KEYS)

    assert doc, "tags.py has no module docstring"
    assert "dataknobs_data.vector.content" in doc or "vector/content.py" in doc


def test_the_embedder_half_names_the_identity_half() -> None:
    """``content.py`` says where the other four live.

    The direction that is easier to lose: ``content.py`` shipped first and is
    about hashing rather than about the family, so a reader editing it has no
    reason to know the sentence is load-bearing.
    """
    doc = _module_docstring(EMBEDDER_KEY)

    assert doc, "vector/content.py has no module docstring"
    assert "dataknobs_common.ontology.tags" in doc or "ontology/tags.py" in doc


def test_the_two_files_this_guard_names_both_exist() -> None:
    """A path guard whose paths have moved passes by reading nothing.

    Both assertions above parse one of these two paths. If either file were
    renamed, ``read_text`` would raise --- but a guard that only fails by
    raising ``FileNotFoundError`` says nothing useful about which half moved.
    """
    assert IDENTITY_KEYS.is_file()
    assert EMBEDDER_KEY.is_file()

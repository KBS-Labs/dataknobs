"""A refusal is documented where it escapes, not where it is written.

Five members of this vocabulary raise, and a caller meets the class rather
than the function that constructed it -- so the ``Raises:`` section belongs on
the member a caller holds. The list is **fixed by the rule** rather than
discovered here: this file names the five, which is what makes it fail when one
of them loses its section rather than passing over whatever set it found.

The unit is the contract, not the flavour. A twin that restates its sync
sibling restates the clauses too; a twin whose docstring is a pointer at that
sibling owes nothing, because the contract is written once at the place the
pointer leads. Both halves are asserted.
"""

from __future__ import annotations

import pytest

from dataknobs_common.ontology.taxonomy import AsyncTaxonomy, Taxonomy
from dataknobs_common.ontology.values import AsyncOntology, Ontology

#: The five, with every class that escapes each of them.
#:
#: ``Ontology.taxonomy`` and its twin raise two: ``NotFoundError`` for an axis
#: the vocabulary does not declare, and ``ValidationError`` for one whose
#: ``materialization`` asks for a copy of every entity on it. The second is the
#: half a reader would not predict from the member's name, which is why it is
#: spelled out rather than covered by "the docstring has a Raises section".
DOCUMENTED = [
    (Ontology, "taxonomy", ("NotFoundError", "ValidationError")),
    (AsyncOntology, "taxonomy", ("NotFoundError", "ValidationError")),
    (Taxonomy, "walk", ("NotFoundError",)),
    (Taxonomy, "subtree_keys", ("NotFoundError",)),
    (Taxonomy, "inherited_attributes", ("NotFoundError",)),
]


@pytest.mark.parametrize(
    ("owner", "member", "classes"),
    DOCUMENTED,
    ids=[f"{owner.__name__}.{member}" for owner, member, _ in DOCUMENTED],
)
def test_a_member_that_raises_documents_every_class_that_escapes_it(
    owner: type, member: str, classes: tuple[str, ...]
) -> None:
    """Named one by one, because the set is a ruling rather than a discovery."""
    doc = getattr(owner, member).__doc__ or ""

    assert "Raises:" in doc, f"{owner.__name__}.{member} raises and says nothing"
    for name in classes:
        assert name in doc, f"{owner.__name__}.{member} does not name {name}"


def test_a_member_that_raises_nothing_keeps_saying_nothing() -> None:
    """The negative control, and the half that can rot.

    ``Ontology.entity`` answers ``None`` for an id this vocabulary does not
    hold -- a lookup, not a refusal -- and a suite that only checked for the
    presence of sections would pass a module that had grown one everywhere.
    """
    doc = Ontology.entity.__doc__ or ""

    assert "Raises:" not in doc
    assert "NotFoundError" not in doc


#: The four asynchronous twins whose docstrings point rather than restate.
POINTERS = [
    (AsyncOntology, "localize"),
    (AsyncTaxonomy, "walk"),
    (AsyncTaxonomy, "subtree_keys"),
    (AsyncTaxonomy, "inherited_attributes"),
]


@pytest.mark.parametrize(
    ("owner", "member"),
    POINTERS,
    ids=[f"{owner.__name__}.{member}" for owner, member in POINTERS],
)
def test_a_twin_that_points_at_its_sibling_owes_no_clause(owner: type, member: str) -> None:
    """A pointer is not an abridgement, and the test is mechanical about which.

    Each of these opens with ``:meth:`` on its first line, which is what makes
    "it points" checkable rather than a reading. A twin that starts restating
    would stop matching here and would then owe its own clauses -- which is
    exactly what ``AsyncOntology.taxonomy`` does, and why that member is in
    ``DOCUMENTED`` above rather than in this list.
    """
    doc = (getattr(owner, member).__doc__ or "").strip()

    assert doc.startswith(":meth:"), f"{owner.__name__}.{member} no longer points"

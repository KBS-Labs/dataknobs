"""``RelationRef`` is two spellings of one name, and a holder must keep one.

:data:`~dataknobs_common.ontology.model.RelationRef` is ``str | RelationType``
-- a relation id to resolve, or the definition itself -- and
:func:`~dataknobs_common.ontology.model.relation_id` is where this package
decides which it was handed. Its own docstring says *every comparison goes
through here rather than each site deciding what it was handed*.

The members ``@dataclass`` generates are comparisons, and they go through
nothing. A value named by the definition therefore reads identically to one
named by the id and compares **unequal** to it, which is a defect in every
holder of the field rather than in any one of them:

    compare by the identity of what you hold, and the value of what you name

That rule is ``8313b8f2``'s, stated when the two assertion-backed axes were
canonicalised. It is a rule about the *field shape*, so it reaches every
dataclass carrying one, and the three types below were left behind by an
audit that stopped at the axes.

**Why a table rather than five separate constructions.** The population is the
point: a sixth holder added later inherits the same defect, and
:func:`test_the_holder_table_is_every_dataclass_that_names_a_relation`
measures the package independently -- through the same walk
``test_hashable_contract.py`` uses, in ``_dataclass_sweep`` -- and fails when
the table and the tree disagree. Without it the guard covers whatever somebody
remembered to list.

**Hash is deliberately not asserted here.** Canonicalising removes
``RelationType`` as *a* reason a frozen holder's hash raises; it does not
remove the others. ``ProjectionContext`` still holds two ``Mapping`` fields
and still raises at ``hash()``, which is a separate open question recorded in
``test_hashable_contract.py``. The two axes' hashes are asserted where their
own tests live, in ``test_hierarchy_view.py``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from _dataclass_sweep import every_dataclass
from dataknobs_common.ontology import (
    Assertion,
    AsyncMappingAssertionSource,
    EntityRef,
    MappingAssertionSource,
    ProjectionContext,
    RelationType,
    TaxonomyDefinition,
)
from dataknobs_common.ontology.hierarchy import AssertionHierarchy, AsyncAssertionHierarchy

if TYPE_CHECKING:
    from collections.abc import Callable

    from dataknobs_common.ontology import RelationRef

#: One source instance per flavour, shared by both spellings of every axis.
#:
#: The axes compare their source by *identity* -- that is the other half of the
#: rule -- so two sources built from the same rows would make the comparison
#: below fail for a reason that has nothing to do with the relation.
_ASSERTIONS = MappingAssertionSource(())
_ASYNC_ASSERTIONS = AsyncMappingAssertionSource(())

#: Every dataclass in this package with a ``relation: RelationRef`` field,
#: each with a way to build one that differs only in how the relation is spelt.
HOLDERS: dict[str, Callable[[RelationRef], object]] = {
    "ontology.hierarchy.AssertionHierarchy": lambda r: AssertionHierarchy(_ASSERTIONS, r),
    "ontology.hierarchy.AsyncAssertionHierarchy": (
        lambda r: AsyncAssertionHierarchy(_ASYNC_ASSERTIONS, r)
    ),
    "ontology.model.Assertion": lambda r: Assertion(
        id="dog-isa-mammal", subject="dog", relation=r, object=EntityRef("mammal")
    ),
    "ontology.model.ProjectionContext": lambda r: ProjectionContext(
        taxonomy_id="species",
        relation=r,
        roots=frozenset({"mammal"}),
        depths={"dog": 1},
        types={"dog": "Species"},
    ),
    "ontology.model.TaxonomyDefinition": lambda r: TaxonomyDefinition(id="species", relation=r),
}


def _fields_declaring(matches: Callable[[str], bool]) -> set[str]:
    """Names of the package's dataclasses with a field whose declared type matches.

    Matched against the **declared** annotation string rather than a resolved
    one. Under ``from __future__ import annotations`` that string is what the
    source says and it needs no resolution -- which matters here, because
    :func:`typing.get_type_hints` raises ``NameError`` on several of these
    types, whose collaborators are imported only under ``if TYPE_CHECKING:``.
    Resolution would drop them silently, and a discovery pass that cannot see a
    holder reports the table complete.
    """
    return {
        name
        for name, cls in every_dataclass().items()
        if any(
            isinstance(field.type, str) and matches(field.type) for field in dataclasses.fields(cls)
        )
    }


def _spellings(name: str) -> tuple[object, object]:
    """The same value named by the id and by the definition."""
    build = HOLDERS[name]
    return build("isa"), build(RelationType(id="isa"))


def test_an_assertion_compares_by_the_relation_it_names() -> None:
    """Two assertions differing only in how their relation is spelt are one fact.

    ``Assertion`` is the type the field was declared for, and it is unfrozen
    with equality on -- so its ``__hash__`` is None and nothing about
    hashability is at stake. What is at stake is that *the same stated fact*,
    written by a caller holding the definition, compares unequal to one written
    by a caller holding the id.

    The document path never produced this: the loader coerces with ``str()``
    before constructing. It is reachable from code that builds an assertion
    itself, holding a ``RelationType`` it just looked up, which is the ordinary
    way to build one.
    """
    by_id, by_definition = _spellings("ontology.model.Assertion")

    assert by_definition.relation == "isa"  # the door still takes either
    assert by_id == by_definition


def test_a_taxonomy_definition_compares_by_the_relation_it_names() -> None:
    """A definition is *a value*, and its own docstring says so.

    ``TaxonomyDefinition`` already had a ``__post_init__`` -- it defaults
    ``name`` to ``id`` -- so the canonicalisation joins a body that was already
    there rather than adding one. That the method existed is why a static audit
    for "does this type canonicalise" answered yes about it and was wrong: the
    body never touched ``relation``.
    """
    by_id, by_definition = _spellings("ontology.model.TaxonomyDefinition")

    assert by_definition.relation == "isa"
    assert by_id == by_definition


def test_a_projection_context_compares_by_the_relation_it_names() -> None:
    """Everything a ``ParentChoice`` may consult, and nothing in this package builds one.

    ``ProjectionContext`` is declared surface: the projection core would hand
    one to a policy, and no code here constructs it. So the only caller who can
    reach this defect is a consumer writing a policy and a context to test it
    against -- which is exactly the population least able to tell that two
    contexts naming one relation compared unequal.

    Frozen with equality on, so ``__hash__`` is generated; it still raises,
    because ``depths`` and ``types`` are ``Mapping``. Canonicalising the
    relation removes one reason and not the others, and the remainder is
    recorded in ``test_hashable_contract.py``.
    """
    by_id, by_definition = _spellings("ontology.model.ProjectionContext")

    assert by_definition.relation == "isa"
    assert by_id == by_definition


@pytest.mark.parametrize("name", sorted(HOLDERS))
def test_every_relation_holder_canonicalises(name: str) -> None:
    """The rule, asserted over the whole population rather than type by type.

    The three tests above say what is specific to each type. This says the one
    thing that is true of all of them, including the two axes ``8313b8f2``
    already fixed -- so the guard covers the population a sixth holder would
    join rather than the three that happened to be found.
    """
    by_id, by_definition = _spellings(name)

    assert by_definition.relation == "isa", (
        f"{name} keeps the definition it was handed; a holder canonicalises to the id"
    )
    assert by_id == by_definition, (
        f"{name} compares two spellings of one relation as different values"
    )


def test_the_holder_table_is_every_dataclass_that_names_a_relation() -> None:
    """:data:`HOLDERS` and the package must agree about the population.

    Fails in both directions. A new holder is a new instance of the defect and
    must be built into the table before it can be asserted about; a name in the
    table that no longer declares the field is a factory testing nothing.
    """
    declared = _fields_declaring(lambda annotation: annotation == "RelationRef")

    assert declared - set(HOLDERS) == set(), (
        f"dataclass(es) declare a relation and are untested: {sorted(declared - set(HOLDERS))}"
    )
    assert set(HOLDERS) - declared == set(), (
        f"HOLDERS names type(s) that no longer declare a relation: "
        f"{sorted(set(HOLDERS) - declared)}"
    )


def test_the_alias_is_the_only_spelling_of_the_union() -> None:
    """The discovery above matches ``RelationRef`` literally, so nothing may evade it.

    A field declared ``str | RelationType`` longhand is the same field and
    would be invisible to :func:`_declared_relation_holders`. Keeping the alias
    the one spelling is what makes matching on it complete.
    """
    longhand = _fields_declaring(
        lambda annotation: (
            "RelationType" in annotation
            and annotation != "RelationRef"
            # A registry of definitions keyed by id, which is not a *reference* to
            # one and carries no second spelling to reconcile.
            and not annotation.startswith("Mapping[")
        )
    )

    assert longhand == set(), (
        f"type(s) spell the relation union longhand and evade the discovery above: "
        f"{sorted(longhand)}"
    )

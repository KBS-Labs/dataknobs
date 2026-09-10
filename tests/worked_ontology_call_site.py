"""The worked call site from the ontology guide, executed as written.

This module is not collected by pytest. ``test_worked_ontology_call_site.py``
runs it with :func:`runpy.run_path`, in a directory holding the vocabulary the
guide publishes beside it, and asserts on what it leaves bound.

**Everything below the blank line after this docstring is the guide's fence,
character for character.** The test asserts that, in both directions, so
editing either copy alone turns the suite red rather than letting the page and
the code drift.

So this file is the one place where a formatter, a linter and a type checker
read the published page. That is the point of executing a copy rather than
transcribing one, and it is not free: the fence is written the way
``ruff format`` writes it, because a formatting finding here is a finding
against a documentation page that no ``# noqa`` can answer -- the directive
would render on the page. The one bare-attribute rule that has no such
spelling, ``B018``, is waived for this file in the root config with its reason.

Do not add to it. An assertion that is not in the fence is an assertion the
reader of the guide never sees; the ``assert`` that *is* in the fence is there
because ``get`` returns ``Entity | None`` and the page should say so.
"""

from pathlib import Path

from dataknobs_common import ancestors
from dataknobs_common.ontology import (
    AssertionHierarchy,
    build_resolver,
    load_ontology,
)

onto = load_ontology(Path("mammals.yaml"))  # -> Ontology
# No database. No embedder. No event loop.

# (1) the reader typed "beagles"
onto.entities.by_surface_form("beagles")  # -> frozenset({"beagle"})
beagle = onto.entities.get("beagle")  # -> Entity | None
assert beagle is not None  # an id that misses is a typo, not a result
beagle.name  # "Beagle"

# (2) what is it a kind of?
structure = AssertionHierarchy(onto.assertions, "isa")  # a Hierarchy over the isa edges
ancestors(structure, "beagle")  # ("dog", "mammal") -- module-level, over any Hierarchy

# (3) what does the vocabulary say about it?
onto.assertions.find(subject="beagle", relation="isa")  # -> [Assertion(...)]

# (4) leave with something spendable on your own data
beagle.source  # SourceRef(clinic_db, ...)
onto.entities.describe().capabilities  # frozenset() -- no ORIGIN_FETCH, so the
# reference is yours to spend and not ours to dereference

# (5) the same placement, ranked and with its reasons
resolver = build_resolver(Path("mammals.yaml"), onto)
result = resolver.resolve("beagles", k=5)
result.candidates[0].entity_id  # "beagle"
result.candidates[0].evidence[0].kind  # EvidenceKind.DECLARED

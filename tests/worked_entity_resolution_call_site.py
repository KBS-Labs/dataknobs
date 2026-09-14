"""The worked call site from the entity-resolution guide, executed as written.

This module is not collected by pytest.
``test_worked_entity_resolution_call_site.py`` runs it with
:func:`runpy.run_path`, in a directory holding the vocabulary the guide
publishes beside it, and asserts on what it leaves bound.

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
reader of the guide never sees -- and the assertions this file's runner makes
are exactly the ones the fence's comments claim, re-called after the fact.

:mod:`worked_ontology_call_site` is the sibling this copies, one guide over.
"""

from pathlib import Path

from dataknobs_common.entity_resolution import CascadingResolver, ScanningSignal
from dataknobs_common.ontology import load_ontology

onto = load_ontology(Path("mammals.yaml"))  # -> Ontology
# No database. No embedder. No event loop.

resolver = CascadingResolver([ScanningSignal(onto.entities)], onto.entities)
result = resolver.resolve("my golden retriever has been limping", k=5)

# (1) which declared forms the sentence carries, longest first
[c.entity_id for c in result.candidates]  # ["golden_retriever", "retriever"]

# (2) where each one sat -- half-open offsets into `result.query`
result.explain("golden_retriever")[0].span  # (3, 19)
result.explain("retriever")[0].span  # (10, 19)
result.explain("golden_retriever")[0].matched_text  # "golden retriever"

# (3) what the vocabulary did not account for
result.coverage.matched  # ((3, 19),) -- the union: (10, 19) is inside it
result.coverage.unmatched  # ((0, 2), (20, 36))
result.unmatched_text()  # ("my", "has been limping")

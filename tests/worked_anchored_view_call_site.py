"""The worked call site from the anchored-view guide, executed as written.

This module is not collected by pytest. ``test_worked_anchored_view_call_site.py``
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
reader of the guide never sees; the two ``assert`` lines that *are* in the
fence are there because ``entity()`` returns ``Entity | None`` and the page
should say so rather than let a reader assume otherwise.
"""

from pathlib import Path

from dataknobs_common.ontology import build_resolver, load_ontology

onto = load_ontology(Path("mammals.yaml"))
axis = onto.taxonomy("species")  # no store, no embedder, no event loop

# (1) the triage flow places a message
resolver = build_resolver(Path("mammals.yaml"), onto)
hit = resolver.resolve("golden retriever", k=5).ranked()[0]
hit.entity_id  # "golden_retriever"

# (2) anchor a cursor there, and widen to the context above it
here = axis.at(hit.entity_id)  # the anchored view
here.node  # "golden_retriever"
here.parents()  # (view("retriever"),) -- PLURAL, always
here.ancestors()  # retriever, dog, mammal

for above in here.ancestors():
    entity = above.entity()  # -> Entity | None
    assert entity is not None  # an id that misses is a typo, not a result
    entity.name  # "Retriever", "Dog", "Mammal"
    entity.description  # what folds into the prompt
    onto.assertions.find(subject=above.node)  # what is TRUE of it -- and it is
    # NOT on the view: a taxonomy holds no assertion axis

# (3) what does this type inherit? -- the OTHER isa lattice, on the axis
placed = here.entity()
assert placed is not None
axis.inherited_attributes(placed.type)  # akc_group, latin_name, lifespan_years

# (4) see what is still unspecified
there = here.at("dog")  # re-anchor: the message stopped here
there.is_leaf()  # False -- the CONSUMER concludes
there.children()  # retriever, beagle -- ask which

# (5) leave with keys, in your own id space
axis.subtree_keys(there.node)  # ["dog", "retriever", "golden_retriever", "beagle"]

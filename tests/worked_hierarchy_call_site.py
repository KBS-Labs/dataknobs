"""The worked call site from the hierarchy guide, executed as written.

This module is not collected by pytest. ``test_worked_hierarchy_call_site.py``
runs it with :func:`runpy.run_path`, in a directory holding the vocabulary the
guide publishes above it, and asserts on what it leaves bound.

**Everything below the blank line after this docstring is the guide's fence,
character for character.** The test asserts that, in both directions, so
editing either copy alone turns the suite red rather than letting the page and
the code drift.

Do not add to it. An assertion that is not in the fence is an assertion the
reader of the guide never sees; the ``assert`` that *is* in the fence is the
page's own claim about what the axis walks.
"""

from pathlib import Path

from dataknobs_common.ontology import load_ontology

onto = load_ontology(Path("mammals.yaml"))
species = onto.taxonomy("species")

assert tuple(species.walk()) == ("mammal", "dog", "retriever", "beagle")

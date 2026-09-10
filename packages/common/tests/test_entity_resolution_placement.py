"""Where this family sits in the import graph, and what that buys.

Two properties, both measured in a **fresh interpreter** because neither is
observable in a process that has already imported everything. The first is
that the family does not import the vocabulary package; the second is that it
does not import the data package. They are different claims with different
reasons, and each is asserted per submodule rather than once on the package.

**They are measured through two different probes, and the difference is the
point.** ``dataknobs_common``'s own door publishes the vocabulary surface, so
it imports ``dataknobs_common.ontology`` -- and importing any submodule of a
package runs that package's ``__init__`` first. A probe that lets the door run
therefore reports the vocabulary as reached no matter what this family does,
which is a probe that has stopped being able to see its own subject.

So the vocabulary claim is measured with the door **stubbed**: the parent is
placed in ``sys.modules`` as a bare module carrying only the real
``__path__``, which is enough for the submodule machinery and leaves
``__init__`` unrun. What that measures is this family's own import closure,
which is what the claim was always about. The database claim keeps the
door-running probe, because *that* claim is about what a consumer pays and the
consumer pays for the door.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from dataknobs_common import entity_resolution

#: Every submodule of the family, asserted individually.
#:
#: Not just the package. A package ``__init__`` imports its submodules in some
#: order, and an edge reachable only from one of them can be masked by that
#: order -- so the package-level assertion can pass while the module carrying
#: the edge would fail on its own. Asserting per submodule is what makes this
#: a claim about the **absence of an edge** rather than about the survival of
#: an ordering, which is the one form no ``__init__`` line order can satisfy
#: by luck.
SUBMODULES = [
    f"dataknobs_common.entity_resolution.{name}"
    for name in ("cascade", "protocols", "registry", "signals", "values")
]


#: Put the package door in ``sys.modules`` without running it.
#:
#: A module object carrying the real ``__path__`` is all the import machinery
#: needs to resolve ``dataknobs_common.<name>``; because the parent is already
#: in ``sys.modules``, its ``__init__`` is never executed. Everything the
#: probe then reports was reached by the module under test, not by the door.
STUB_THE_DOOR = (
    "import importlib, importlib.util, sys, types; "
    "_spec = importlib.util.find_spec('dataknobs_common'); "
    "_stub = types.ModuleType('dataknobs_common'); "
    "_stub.__path__ = list(_spec.submodule_search_locations); "
    "_stub.__spec__ = _spec; "
    "sys.modules['dataknobs_common'] = _stub; "
)


def imported_modules(statement: str) -> set[str]:
    """The modules a fresh interpreter has after running ``statement``."""
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; {statement}; print('\\n'.join(sys.modules))"],
        capture_output=True,
        text=True,
        check=True,
    )
    reached = set(result.stdout.split())

    # A negative asserted over an empty measurement is not a negative. Every
    # caller below asks whether some module is *absent*, and an empty set
    # answers yes to all of them -- so a probe that ran but imported nothing
    # would report the very thing these tests exist to establish. The positive
    # control belongs here rather than in each caller, because it is a property
    # of the measurement rather than of any one question asked of it.
    assert "dataknobs_common" in reached, (
        f"the probe for {statement!r} imported nothing; an absence asserted "
        f"over this measurement would be vacuous"
    )
    return reached


def imported_by_the_module_alone(module: str) -> set[str]:
    """The modules ``module`` reaches, with the package door stubbed out.

    The positive control is different here, and has to be. Its sibling checks
    that ``dataknobs_common`` was reached, which under a stub is true before
    the probe imports anything at all -- so this one checks that the module
    under test really loaded, and that the door really did not run.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"{STUB_THE_DOOR}"
            f"importlib.import_module({module!r}); "
            "print(hasattr(sys.modules['dataknobs_common'], '__all__')); "
            "print('\\n'.join(sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    door_ran, _, rest = result.stdout.partition("\n")
    reached = set(rest.split())

    # Belt to the stub's braces. Installing the stub before any import means
    # ``__init__`` cannot run, so this cannot fail as written -- it is here to
    # name the invariant at the point that depends on it, not as coverage.
    assert door_ran.strip() == "False", (
        f"the door ran during the probe for {module!r}; everything it imports "
        f"would be attributed to the module under test"
    )
    # This one is load-bearing: it is what makes an absence below a measurement.
    assert module in reached, (
        f"the probe for {module!r} did not load it; an absence asserted over "
        f"this measurement would be vacuous"
    )
    return reached


@pytest.mark.parametrize("module", ["dataknobs_common.entity_resolution", *SUBMODULES])
def test_the_family_does_not_import_the_vocabulary_package(module: str) -> None:
    """The edge whose absence keeps this graph acyclic.

    ``ontology/__init__`` imports the loader, the loader builds a cascade, and
    the cascade constructs value types. An edge back from here would close
    that loop -- and it would fail on import **order**, so a suite that
    happens to import the vocabulary first stays green while the other order
    is broken. This is the assertion that does not care about order.

    Measured with the door stubbed, for the reason in this module's docstring:
    the door imports the vocabulary, so a probe that runs it answers a
    question about ``dataknobs_common/__init__.py`` and not about this family.
    """
    reached = imported_by_the_module_alone(module)

    assert not {name for name in reached if name.startswith("dataknobs_common.ontology")}


def test_the_stubbed_probe_still_sees_a_vocabulary_import() -> None:
    """The control for the assertion above, on a module that must fail it.

    A negative measured through a broken probe reads exactly like a negative
    measured through a working one. So the control has to exercise the thing
    the negative rests on -- that the probe reports what the module under test
    reaches **transitively** -- and asserting that importing a module loads
    that module would not: it restates ``module in reached`` above and holds
    even for a probe that can see nothing else.

    ``ontology.taxonomy`` is therefore probed and the vocabulary modules it was
    **not** handed are what is asserted. Every name below arrives through an
    import chain, and ``hierarchy`` crosses out of the vocabulary package
    entirely, so a stub that disabled the measurement rather than the door
    shows up here as an empty set.
    """
    reached = imported_by_the_module_alone("dataknobs_common.ontology.taxonomy")

    assert {"dataknobs_common.ontology.values", "dataknobs_common.ontology.model"} <= reached
    assert "dataknobs_common.hierarchy" in reached


def test_the_door_reaches_the_vocabulary_and_that_is_what_the_stub_hides() -> None:
    """What the stub costs, stated rather than left as an absence.

    Publishing the vocabulary on ``dataknobs_common``'s door means importing
    anything from this package imports the vocabulary too. The family's
    independence is therefore a property of the **module graph** and not of
    the process, and this is the test that says so out loud -- so that the
    stubbed probe above reads as a narrowed question rather than a weakened
    one.
    """
    reached = imported_modules("import dataknobs_common.entity_resolution")

    assert "dataknobs_common.ontology" in reached


@pytest.mark.parametrize("module", ["dataknobs_common.entity_resolution", *SUBMODULES])
def test_a_rung_needs_no_database_package(module: str) -> None:
    """The criterion the placement exists for.

    A rung over a vocabulary someone typed is dictionary work, and the point
    of putting this family in the dependency-free package is that building one
    costs nothing. Trivial to satisfy in this leg and **not** trivial to keep,
    which is why it is a test rather than an observation.

    **Door-running probe, deliberately.** Its sibling above stubs the door
    because the door imports the vocabulary; nothing about that argument
    applies here, and this claim is about what a consumer pays -- who pays for
    the door. Measured this way it covers the whole package, which is strictly
    the stronger reading of *costs nothing*.
    """
    reached = imported_modules(f"import {module}")

    assert not {name for name in reached if name.startswith("dataknobs_data")}


@pytest.mark.parametrize(
    "order",
    [
        "import dataknobs_common.ontology, dataknobs_common.entity_resolution",
        "import dataknobs_common.entity_resolution, dataknobs_common.ontology",
    ],
    ids=["vocabulary-first", "family-first"],
)
def test_both_import_orders_succeed(order: str) -> None:
    """Both, because exactly one of them passed against the broken graph.

    The arrangement this leg replaced imported cleanly one way round and
    raised ``ImportError`` the other. A test written in the passing order
    would have reported green on it.

    **Stubbed, or the parametrization measures one thing twice.** The door
    imports both packages, so whichever name is written first runs
    ``dataknobs_common/__init__`` to completion and settles the order before
    either statement's own imports resolve -- leaving the second name a
    ``sys.modules`` hit under both ids. With the door stubbed the written
    order is the executed one: ``vocabulary-first`` starts ``ontology`` and
    reaches the family mid-initialization, ``family-first`` completes the
    family and then starts ``ontology``. Those are the two shapes the
    replaced arrangement told apart, which is what this test is for.
    """
    imported_modules(f"{STUB_THE_DOOR}{order}")


def test_the_moved_types_are_one_object_in_a_fresh_interpreter() -> None:
    """Identity across the re-export, measured where it could actually differ.

    In-process the two names are bound in one interpreter and identity is
    nearly guaranteed. A subprocess is where a second definition -- the
    failure mode a move has -- would actually show up.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dataknobs_common.ontology as o, dataknobs_common.entity_resolution as e; "
            "print(o.Scoring is e.Scoring, o.CompatibilityVerdict is e.CompatibilityVerdict, "
            "o.ResolutionRef is e.ResolutionRef)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.split() == ["True", "True", "True"]


def test_the_module_path_the_moved_types_shipped_under_still_resolves() -> None:
    """Both spellings kept working, because both were reachable before the move.

    A caller who wrote ``from dataknobs_common.ontology.model import Scoring``
    named a real module, and a re-export on the package door alone leaves that
    import raising ``ImportError`` while the door's own claim -- every existing
    import still works -- reads as though it did not. The module-path spelling
    is the one nothing in this repository uses, which is exactly why it needs
    an assertion rather than a reader's confidence.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dataknobs_common.ontology.model import Scoring, CompatibilityVerdict, "
            "ResolutionRef; import dataknobs_common.entity_resolution as e; "
            "print(Scoring is e.Scoring, CompatibilityVerdict is e.CompatibilityVerdict, "
            "ResolutionRef is e.ResolutionRef)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.split() == ["True", "True", "True"]


def test_a_rung_built_here_satisfies_the_protocol_and_produces_a_candidate() -> None:
    """The other half of the criterion: not merely importable, but usable.

    ``isinstance`` against a runtime-checkable protocol compares member
    *names* and nothing else, so this constructs a candidate as well -- the
    part a member scan cannot tell you.
    """
    from dataknobs_common.ontology.model import Entity
    from dataknobs_common.ontology.sources import MappingEntitySource

    source = MappingEntitySource({"dog": Entity(id="dog", type="Species", name="Dog")})
    rung = entity_resolution.ExactNormalizedSignal(source)

    assert isinstance(rung, entity_resolution.MatchSignal)

    produced = rung.candidates("dog", 5)

    assert [c.entity_id for c in produced] == ["dog"]
    assert isinstance(produced[0], entity_resolution.EntityCandidate)
    assert produced[0].evidence[0].kind is entity_resolution.EvidenceKind.DECLARED

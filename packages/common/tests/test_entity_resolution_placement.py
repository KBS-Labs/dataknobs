"""Where this family sits in the import graph, and what that buys.

Two properties, both measured in a **fresh interpreter** because neither is
observable in a process that has already imported everything. The first is
that the family does not import the vocabulary package; the second is that it
does not import the data package. They are different claims with different
reasons, and each is asserted per submodule rather than once on the package.
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


@pytest.mark.parametrize("module", ["dataknobs_common.entity_resolution", *SUBMODULES])
def test_the_family_does_not_import_the_vocabulary_package(module: str) -> None:
    """The edge whose absence keeps this graph acyclic.

    ``ontology/__init__`` imports the loader, the loader builds a cascade, and
    the cascade constructs value types. An edge back from here would close
    that loop -- and it would fail on import **order**, so a suite that
    happens to import the vocabulary first stays green while the other order
    is broken. This is the assertion that does not care about order.
    """
    reached = imported_modules(f"import {module}")

    assert not {name for name in reached if name.startswith("dataknobs_common.ontology")}


@pytest.mark.parametrize("module", ["dataknobs_common.entity_resolution", *SUBMODULES])
def test_a_rung_needs_no_database_package(module: str) -> None:
    """The criterion the placement exists for.

    A rung over a vocabulary someone typed is dictionary work, and the point
    of putting this family in the dependency-free package is that building one
    costs nothing. Trivial to satisfy in this leg and **not** trivial to keep,
    which is why it is a test rather than an observation.
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
    """
    imported_modules(order)


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

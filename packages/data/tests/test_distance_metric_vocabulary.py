# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""One vocabulary for distance metrics, read the same way everywhere.

``DistanceMetric`` has six members naming four metrics: ``INNER_PRODUCT`` is
``DOT_PRODUCT`` and ``L2`` is ``EUCLIDEAN``, both stated in a trailing comment
on the member and nowhere a program could read. Every site that mapped a
metric to something therefore had to restate the aliasing by hand. There were
eight, and the four that branched on the metric each missed a member:

* ``get_vector_operator`` knew ``inner_product`` and not ``dot_product``, knew
  ``l2`` and ``euclidean``, and knew no ``l1`` --- so two of the six members
  fell through its ``.get(metric, "<=>")`` default and a caller asking for
  ``DOT_PRODUCT`` or ``L1`` was answered in **cosine** distances, silently.
* the operator class in ``build_vector_index_sql`` knew ``dot_product`` and
  still no ``l1``, so the two tables disagreed about the same member.
* ``PgVectorStore`` chose an operator class in one ``if``/``elif`` and a
  distance operator in another, both ending in a cosine default, so ``L1``
  was cosine there too --- in the class most specifically about pgvector.
* the distance-to-score conversion, written once per Postgres twin plus once
  in ``hybrid_search`` and once in ``PgVectorStore``, knew ``dot_product`` in
  some copies --- which is the contradiction that makes the first bullet
  visible from inside a single method: one half of the sync twin's
  ``_vector_search`` said ``dot_product`` is the inner-product metric and the
  other half gave it the cosine operator.

Meanwhile ``get_aliases`` published a fourth vocabulary --- ``cos``,
``manhattan``, ``euclidean_distance`` and five more --- that nothing in the
library resolved. Six of its eight aliases raised ``ValueError`` from
``DistanceMetric(...)``; the two that worked did so only because they were
also member values.

So the guard is not "the operator table is right". It is that **the enum owns
the vocabulary** and every table is derived from it, which is what makes a
seventh member or a ninth alias impossible to half-add.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dataknobs_data.backends.postgres_vector import (
    distance_to_score,
    get_vector_operator,
    get_vector_opclass,
)
from dataknobs_data.vector.types import DistanceMetric

# The four metrics the six members name. Written out rather than derived from
# ``canonical()`` so that the test states the expectation instead of asking
# the code under test what it expects.
FAMILIES = {
    DistanceMetric.COSINE: DistanceMetric.COSINE,
    DistanceMetric.EUCLIDEAN: DistanceMetric.EUCLIDEAN,
    DistanceMetric.L2: DistanceMetric.EUCLIDEAN,
    DistanceMetric.DOT_PRODUCT: DistanceMetric.DOT_PRODUCT,
    DistanceMetric.INNER_PRODUCT: DistanceMetric.DOT_PRODUCT,
    DistanceMetric.L1: DistanceMetric.L1,
}


class TestEveryMemberIsNamedOnce:
    def test_the_sweep_covers_every_member(self):
        """A seventh member fails here first, before it reaches a lookup table."""
        assert set(FAMILIES) == set(DistanceMetric)

    @pytest.mark.parametrize("member", list(DistanceMetric))
    def test_a_member_canonicalises_to_its_family(self, member):
        assert member.canonical() is FAMILIES[member]

    def test_the_two_spellings_of_one_metric_agree(self):
        """``DOT_PRODUCT`` and ``INNER_PRODUCT`` are one metric, and so are
        ``L2`` and ``EUCLIDEAN``. Both pairs answered differently at the
        operator table: ``<=>`` for one spelling and ``<#>`` for the other.
        """
        assert get_vector_operator(DistanceMetric.DOT_PRODUCT) == get_vector_operator(
            DistanceMetric.INNER_PRODUCT
        )
        assert get_vector_operator(DistanceMetric.L2) == get_vector_operator(
            DistanceMetric.EUCLIDEAN
        )
        assert get_vector_opclass(DistanceMetric.DOT_PRODUCT) == get_vector_opclass(
            DistanceMetric.INNER_PRODUCT
        )
        assert get_vector_opclass(DistanceMetric.L2) == get_vector_opclass(DistanceMetric.EUCLIDEAN)


class TestNoMemberFallsThroughToCosine:
    """The defect, stated as the property that forbids it.

    Asserting the four operators by name would pass a table that reached them
    through a default, which is exactly what the broken one did for four of
    the six members. What forbids the defect is that no *non-cosine* member
    may answer with the cosine operator.
    """

    @pytest.mark.parametrize(
        "member", [m for m in DistanceMetric if m.canonical() is not DistanceMetric.COSINE]
    )
    def test_a_non_cosine_member_gets_a_non_cosine_operator(self, member):
        assert get_vector_operator(member) != get_vector_operator(DistanceMetric.COSINE)

    @pytest.mark.parametrize(
        "member", [m for m in DistanceMetric if m.canonical() is not DistanceMetric.COSINE]
    )
    def test_a_non_cosine_member_gets_a_non_cosine_opclass(self, member):
        assert get_vector_opclass(member) != get_vector_opclass(DistanceMetric.COSINE)

    def test_the_operators_are_the_pgvector_ones(self):
        """Pinned by value too, since the property above admits any bijection."""
        assert get_vector_operator(DistanceMetric.COSINE) == "<=>"
        assert get_vector_operator(DistanceMetric.EUCLIDEAN) == "<->"
        assert get_vector_operator(DistanceMetric.DOT_PRODUCT) == "<#>"
        assert get_vector_operator(DistanceMetric.L1) == "<+>"

    def test_the_opclasses_are_the_pgvector_ones(self):
        assert get_vector_opclass(DistanceMetric.COSINE) == "vector_cosine_ops"
        assert get_vector_opclass(DistanceMetric.EUCLIDEAN) == "vector_l2_ops"
        assert get_vector_opclass(DistanceMetric.DOT_PRODUCT) == "vector_ip_ops"
        assert get_vector_opclass(DistanceMetric.L1) == "vector_l1_ops"

    def test_an_unknown_metric_is_refused_rather_than_answered(self):
        """``.get(metric, "<=>")`` answered every typo with cosine distances."""
        with pytest.raises(ValueError, match="quagmire"):
            get_vector_operator("quagmire")
        with pytest.raises(ValueError, match="quagmire"):
            get_vector_opclass("quagmire")


class TestAnAliasResolvesToTheMemberThatDeclaresIt:
    """``get_aliases`` described a vocabulary nothing resolved.

    The published surface said ``cos`` and ``manhattan`` were names for
    metrics; ``DistanceMetric("cos")`` raised. This binds the two together so
    that the table cannot describe a name the resolver refuses.
    """

    def test_every_declared_alias_resolves_back(self):
        declared = [(member, alias) for member in DistanceMetric for alias in member.get_aliases()]
        assert declared, "the sweep found no aliases, so it would pass vacuously"
        for member, alias in declared:
            assert DistanceMetric.resolve(alias).canonical() is member.canonical(), (
                f"{alias!r} is published as an alias of {member} and resolves elsewhere"
            )

    @pytest.mark.parametrize("member", list(DistanceMetric))
    def test_a_member_resolves_to_itself(self, member):
        assert DistanceMetric.resolve(member) is member
        assert DistanceMetric.resolve(member.value) is member

    def test_case_is_not_part_of_the_name(self):
        assert DistanceMetric.resolve("COSINE") is DistanceMetric.COSINE
        assert DistanceMetric.resolve("Cos").canonical() is DistanceMetric.COSINE

    def test_an_unknown_name_is_refused_and_the_message_lists_what_is_accepted(self):
        with pytest.raises(ValueError) as excinfo:
            DistanceMetric.resolve("quagmire")
        message = str(excinfo.value)
        assert "quagmire" in message
        for name in ("cosine", "euclidean", "dot_product", "l1"):
            assert name in message, f"{name} is accepted and the refusal does not say so"


class TestOneScoreConversion:
    """The two Postgres twins converted distance to score differently.

    For cosine the sync twin returned ``1 - d`` (cosine similarity, the same
    number the ten Python-path backends return) and the async twin returned
    ``1 - min(d, 2)/2``, which is a different number for the same corpus and
    the same query. That difference is invisible until something compares a
    score against a constant, which ``score_threshold`` now does.
    """

    def test_cosine_distance_becomes_cosine_similarity(self):
        assert distance_to_score(DistanceMetric.COSINE, 0.0) == pytest.approx(1.0)
        assert distance_to_score(DistanceMetric.COSINE, 1.0) == pytest.approx(0.0)
        assert distance_to_score(DistanceMetric.COSINE, 2.0) == pytest.approx(-1.0)

    def test_both_spellings_convert_alike(self):
        for a, b in (
            (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT),
            (DistanceMetric.L2, DistanceMetric.EUCLIDEAN),
        ):
            assert distance_to_score(a, 0.5) == distance_to_score(b, 0.5)

    def test_a_nearer_neighbour_never_scores_lower(self):
        """The property every metric's conversion must have, whatever the formula."""
        for member in DistanceMetric:
            near = distance_to_score(member, 0.25)
            far = distance_to_score(member, 0.75)
            assert near > far, f"{member} ranks a farther neighbour at least as high"


class TestTheTablesAreNotRestatedElsewhere:
    """Nothing outside ``postgres_vector`` may name a pgvector operator.

    The defect was never one wrong table --- it was eight sites. Two chose a
    distance operator (a dict in ``postgres_vector`` and an ``if``/``elif`` in
    ``vector/stores/pgvector.py``), two chose an index operator class (one in
    each of the same two files), and four converted a distance to a score
    (one per Postgres twin, disagreeing; one hardcoded in ``hybrid_search``;
    one in ``PgVectorStore``). Each was written by someone reasonably
    declining to import from the others, and the four that branched on the
    metric each independently missed at least one member.

    So the property worth guarding is not any table's contents --- the cells
    above cover that --- but that there is only one place to get them wrong.
    A seventh copy fails here on the day it is written, which is the only
    moment it is cheap to remove.
    """

    HOME = "packages/data/src/dataknobs_data/backends/postgres_vector.py"
    LITERALS = frozenset(
        {
            "<=>",
            "<->",
            "<#>",
            "<+>",
            "vector_cosine_ops",
            "vector_l2_ops",
            "vector_ip_ops",
            "vector_l1_ops",
        }
    )

    @staticmethod
    def _docstring_nodes(tree: ast.Module) -> set[int]:
        """The string constants that are docstrings, which may name anything."""
        found = set()
        for node in ast.walk(tree):
            if not isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                continue
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
        return found

    def test_the_operators_are_named_in_one_file(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src" / "dataknobs_data"
        home = Path(__file__).resolve().parents[3] / self.HOME

        offenders: list[str] = []
        for path in sorted(source_root.rglob("*.py")):
            if path == home:
                continue
            tree = ast.parse(path.read_text())
            docstrings = self._docstring_nodes(tree)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in self.LITERALS
                    and id(node) not in docstrings
                ):
                    offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")

        assert not offenders, (
            "a pgvector operator or operator class is spelled outside "
            f"{Path(self.HOME).name}, which is how the tables diverged:\n  "
            + "\n  ".join(offenders)
        )

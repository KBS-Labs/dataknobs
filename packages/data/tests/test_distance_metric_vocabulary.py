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
from typing import ClassVar

import numpy as np
import pytest

from dataknobs_common.testing import requires_chromadb

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

    @classmethod
    def _spellings_in(cls, source: str) -> list[tuple[int, str]]:
        """Every pgvector operator named by a non-docstring string in *source*.

        **Containment, not equality.** An earlier version compared
        ``node.value`` against the literal set, which sees only an operator
        that is a string all by itself. The code this change removed from
        ``PgVectorStore`` had both spellings --- ``distance_op = "<=>"``, which
        equality catches, and ``f"1 - ({col} <=> $1::vector)"``, whose pieces
        are ``"1 - ("`` and ``" <=> $1::vector)"`` and match nothing. So the
        guard would have found one half of the very defect it is written to
        forbid, and the half it missed is the more common way to write it.
        """
        tree = ast.parse(source)
        docstrings = cls._docstring_nodes(tree)
        found: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in docstrings:
                continue
            found.extend(
                (node.lineno, literal) for literal in cls.LITERALS if literal in node.value
            )
        return found

    def test_the_operators_are_named_in_one_file(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src" / "dataknobs_data"
        home = Path(__file__).resolve().parents[3] / self.HOME

        offenders: list[str] = []
        for path in sorted(source_root.rglob("*.py")):
            if path == home:
                continue
            for lineno, literal in self._spellings_in(path.read_text(encoding="utf-8")):
                offenders.append(f"{path.name}:{lineno}: {literal!r}")

        assert not offenders, (
            "a pgvector operator or operator class is spelled outside "
            f"{Path(self.HOME).name}, which is how the tables diverged:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_sweep_sees_an_operator_embedded_in_a_larger_string(self) -> None:
        """The spelling the removed ``PgVectorStore`` code actually used.

        A guard that only catches a bare literal is a guard against the
        tidier half of the defect.
        """
        embedded = 'score = f"1 - ({col} <=> $1::vector)"\n'
        assert self._spellings_in(embedded) == [(1, "<=>")]

        interpolated_opclass = 'sql = f"USING ivfflat ({col} vector_cosine_ops)"\n'
        assert self._spellings_in(interpolated_opclass) == [(1, "vector_cosine_ops")]

    def test_the_sweep_still_ignores_prose(self) -> None:
        """Containment must not start failing the docstrings that explain this."""
        assert self._spellings_in('"""Cosine is <=>, and l2 is <->."""\n') == []


class TestEveryTableIsKeyedOnTheFamily:
    """The three tables the pgvector unification did not reach.

    The four sites above --- two operator tables, two score conversions ---
    are the ones the pass that introduced :meth:`DistanceMetric.canonical`
    counted. They are not all of them. A metric is also read by the table
    that scores a Python-path search, by the table that builds an
    Elasticsearch ``dense_vector`` mapping, and by the four parsers that turn
    a configured name into a member; none of those was derived from the enum,
    and two of them ended in the same cosine default the pgvector tables were
    fixed for.

    ``_compute_similarity`` is the one that matters most, because
    ``PythonVectorSearchMixin._score_and_rank`` calls it for all eight
    backends with no native k-NN, on every search. It branched on the member
    rather than the family, so ``DistanceMetric.L2`` --- a legitimate member
    value that ``_apply_vector_config`` accepts without a warning --- reached
    its ``else`` and raised ``Unsupported metric``. The published
    documentation names ``l1`` as an accepted setting for the same knob.
    """

    NON_COSINE: ClassVar[list[DistanceMetric]] = [
        m for m in DistanceMetric if m.canonical() is not DistanceMetric.COSINE
    ]

    @pytest.mark.parametrize("member", list(DistanceMetric))
    def test_the_python_path_scores_every_member(self, member):
        """Not "returns the right number" --- returns a number at all."""
        from dataknobs_data.backends.sqlite_mixins import SQLiteVectorSupport

        support = SQLiteVectorSupport()
        score = support._compute_similarity(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            member,
        )
        assert isinstance(score, float)

    def test_the_python_path_reads_both_spellings_alike(self):
        from dataknobs_data.backends.sqlite_mixins import SQLiteVectorSupport

        support = SQLiteVectorSupport()
        one = np.array([1.0, 0.25], dtype=np.float32)
        two = np.array([0.5, 1.0], dtype=np.float32)
        for a, b in (
            (DistanceMetric.L2, DistanceMetric.EUCLIDEAN),
            (DistanceMetric.INNER_PRODUCT, DistanceMetric.DOT_PRODUCT),
        ):
            assert support._compute_similarity(one, two, a) == support._compute_similarity(
                one, two, b
            ), f"{a} and {b} name one metric and score differently"

    def test_the_python_path_ranks_a_nearer_neighbour_higher_under_every_member(self):
        """A table that answers is not yet a table that answers correctly."""
        from dataknobs_data.backends.sqlite_mixins import SQLiteVectorSupport

        support = SQLiteVectorSupport()
        query = np.array([1.0, 0.0], dtype=np.float32)
        near = np.array([0.9, 0.1], dtype=np.float32)
        far = np.array([-1.0, 0.2], dtype=np.float32)
        for member in DistanceMetric:
            assert support._compute_similarity(query, near, member) > support._compute_similarity(
                query, far, member
            ), f"{member} ranks the farther vector at least as high"

    @pytest.mark.parametrize("member", NON_COSINE)
    def test_elasticsearch_never_answers_a_non_cosine_member_with_cosine(self, member):
        """``mapping.get(metric, "cosine")`` is the pgvector defect, in another file.

        An explicit ``metric="l2"`` on ``create_vector_index`` built a
        ``dense_vector`` mapping with ``similarity: cosine`` and said nothing.
        Elasticsearch has no L1 similarity, so the honest answer there is a
        refusal --- which is still not cosine.
        """
        from dataknobs_data.vector.elasticsearch_utils import get_similarity_for_metric

        try:
            similarity = get_similarity_for_metric(member)
        except ValueError:
            return
        assert similarity != "cosine", f"{member} silently became a cosine dense_vector mapping"

    def test_elasticsearch_reads_both_spellings_alike(self):
        from dataknobs_data.vector.elasticsearch_utils import get_similarity_for_metric

        assert get_similarity_for_metric(DistanceMetric.L2) == get_similarity_for_metric(
            DistanceMetric.EUCLIDEAN
        )
        assert get_similarity_for_metric(DistanceMetric.INNER_PRODUCT) == (
            get_similarity_for_metric(DistanceMetric.DOT_PRODUCT)
        )

    def test_elasticsearch_refuses_a_metric_it_cannot_serve(self):
        """L1 has no ``dense_vector`` similarity, and saying so beats ranking by cosine."""
        from dataknobs_data.vector.elasticsearch_utils import get_similarity_for_metric

        with pytest.raises(ValueError, match="l1"):
            get_similarity_for_metric(DistanceMetric.L1)

    @pytest.mark.parametrize("spelling", ["manhattan", "cos", "ip", "euclidean_distance", "L2"])
    def test_a_configured_metric_accepts_every_published_spelling(self, spelling):
        """``_apply_vector_config`` used ``DistanceMetric(name.lower())``.

        So six of the eight names ``get_aliases`` publishes fell into its
        ``except ValueError`` and were configured as cosine with a warning
        that called the name invalid. The enum resolves them; the parser that
        reads a consumer's configuration has to ask it.
        """
        from dataknobs_data.backends.memory import SyncMemoryDatabase

        database = SyncMemoryDatabase(config={"vector_enabled": True, "vector_metric": spelling})
        assert database.vector_metric is DistanceMetric.resolve(spelling).canonical()


class TestResolveMetricReturnsTheFamily:
    """``resolve_metric`` is what makes the tables below it safe --- or not.

    It returned ``DistanceMetric.resolve(metric)``, so ``"l2"`` arrived at a
    backend as ``L2`` rather than ``EUCLIDEAN``. Every table keyed on the
    member then had to restate the aliasing again, which is the divergence
    ``canonical()`` exists to end. Canonicalising here is one line and covers
    all twelve backends and every table any of them reaches.
    """

    @pytest.mark.parametrize("member", list(DistanceMetric))
    def test_a_member_arrives_canonical(self, member):
        from dataknobs_data.vector.mixins import resolve_metric

        assert resolve_metric(object(), member) is member.canonical()

    @pytest.mark.parametrize("spelling", ["l2", "inner_product", "manhattan", "cos"])
    def test_a_spelling_arrives_canonical(self, spelling):
        from dataknobs_data.vector.mixins import resolve_metric

        resolved = resolve_metric(object(), spelling)
        assert resolved is DistanceMetric.resolve(spelling).canonical()

    def test_a_configured_alias_arrives_canonical(self):
        from dataknobs_data.backends.memory import SyncMemoryDatabase
        from dataknobs_data.vector.mixins import resolve_metric

        database = SyncMemoryDatabase(config={"vector_enabled": True, "vector_metric": "l2"})
        assert resolve_metric(database, None) is DistanceMetric.EUCLIDEAN


class TestTheConfiguredMetricReachesEverySurface:
    """``metric`` is settled above all twelve --- on one method of the three.

    ``vector_search`` took ``metric=None`` and resolved it against the
    database's configuration. ``hybrid_search`` and ``create_vector_index``,
    declared on the same mixin and describing the same database, kept the
    hardcoded ``DistanceMetric.COSINE`` default that this pass calls a bug
    everywhere else. So a database configured for euclidean searched under
    euclidean, built a **cosine** index by default, and ran its hybrid
    search's vector arm under cosine --- three answers from one object.
    """

    METHODS = ("vector_search", "hybrid_search", "create_vector_index")

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("lane", ["Sync", "Async"])
    def test_the_metric_default_defers_to_the_database(self, lane, method):
        import inspect

        from dataknobs_data.vector import mixins

        mixin = getattr(mixins, f"{lane}VectorOperationsMixin")
        default = inspect.signature(getattr(mixin, method)).parameters["metric"].default
        assert default is None, (
            f"{lane}VectorOperationsMixin.{method} defaults metric to {default!r}, "
            f"so a database's configured metric never reaches it"
        )

    def test_a_euclidean_database_runs_its_hybrid_vector_arm_under_euclidean(self):
        """The difference is visible: two records one cosine cannot tell apart.

        ``[1, 0]`` and ``[2, 0]`` are the same direction, so cosine scores
        both 1.0 against the query. Euclidean separates them --- 1.0 and 0.5
        --- so the vector arm's own score says which metric ran.
        """
        from dataknobs_data import Record
        from dataknobs_data.backends.memory import SyncMemoryDatabase

        database = SyncMemoryDatabase(config={"vector_enabled": True, "vector_metric": "euclidean"})
        database.connect()
        try:
            database.create(Record(data={"id": "at", "text": "widget", "embedding": [1.0, 0.0]}))
            database.create(Record(data={"id": "far", "text": "widget", "embedding": [2.0, 0.0]}))

            scores = {
                result.record.id: result.vector_score
                for result in database.hybrid_search(
                    query_text="widget",
                    query_vector=np.array([1.0, 0.0]),
                    text_fields=["text"],
                    vector_field="embedding",
                    k=10,
                )
            }
            assert scores["at"] == pytest.approx(1.0)
            assert scores["far"] == pytest.approx(0.5), (
                "the vector arm ranked under cosine, which scores both 1.0"
            )
        finally:
            database.close()


class TestTheStoreLaneReadsTheSameVocabulary:
    """``vector/stores/`` is the lane the first audit did not reach.

    That audit's ``code_paths`` name ``backends/sqlite_mixins.py``,
    ``backends/vector_config_mixin.py``, ``vector/elasticsearch_utils.py``
    and ``vector/mixins.py`` --- the **database** vector lane and the shared
    mixin. The ``VectorStore`` family is the other half, and both of the
    shapes that audit fixed were still present in it one directory over:

    * ``_setup`` parsed a configured metric with ``DistanceMetric(cfg.metric)``,
      which knows member values only --- so six of the twelve published
      spellings were refused at this door and accepted at every other one;
    * ``ChromaVectorStore`` ended its metric map in ``.get(self.metric,
      "cosine")`` over a five-key dict with no ``L1`` entry, so a store
      configured for Manhattan distance built and queried a **cosine**
      ``hnsw:space`` with no error and no log line.

    The second contradicted a decision the first audit took deliberately:
    ``get_similarity_for_metric`` refuses ``L1`` rather than answering cosine,
    *"a BREAKING change in the direction of an error"*. Chroma answered
    cosine for the same member, so the two doors disagreed about what an
    unservable metric does --- and the disagreement was introduced by the fix
    rather than surviving it. **They stop disagreeing because both refuse.**

    Which is also why this class imports ``dataknobs_data.vector.stores`` at
    all: the first guard's whole import surface is the audited lane, so it
    reported green over the one that still held the defect.
    """

    PUBLISHED: ClassVar[list[str]] = sorted(
        {member.value for member in DistanceMetric}
        | {alias for member in DistanceMetric for alias in member.get_aliases()}
    )

    def test_the_published_vocabulary_is_twelve_spellings(self):
        """A count, so a thirteenth arriving is a visible change here."""
        assert len(self.PUBLISHED) == 12

    @pytest.mark.parametrize("spelling", PUBLISHED)
    def test_a_configured_store_accepts_every_published_spelling(self, spelling):
        """Six of these raised ``ValueError`` at this door and nowhere else.

        The six that worked were member values rather than anything the alias
        table achieved --- which is ``resolve``'s own docstring about the
        table it replaced, reproduced one directory over.
        """
        from dataknobs_data.vector.stores.memory import MemoryVectorStore

        store = MemoryVectorStore({"dimensions": 4, "metric": spelling})
        assert store.metric is DistanceMetric.resolve(spelling).canonical()

    def test_an_unknown_metric_is_refused_and_the_message_lists_what_is_accepted(self):
        """Refusing is the point: the table this replaces answered cosine."""
        from dataknobs_data.vector.stores.memory import MemoryVectorStore

        with pytest.raises(ValueError, match="Unknown distance metric"):
            MemoryVectorStore({"dimensions": 4, "metric": "nearest-ish"})

    @requires_chromadb
    @pytest.mark.parametrize(
        ("spelling", "space"),
        [
            ("cosine", "cosine"),
            ("cos", "cosine"),
            ("cosine_similarity", "cosine"),
            ("euclidean", "l2"),
            ("l2", "l2"),
            ("euclidean_distance", "l2"),
        ],
    )
    def test_chroma_maps_each_servable_family_to_its_own_space(self, spelling, space):
        """Keyed on the family, so no spelling can be missed --- there are four keys."""
        from dataknobs_data.vector.stores.chroma import ChromaVectorStore

        store = ChromaVectorStore({"dimensions": 4, "metric": spelling})
        assert store.chroma_metric == space

    @requires_chromadb
    @pytest.mark.parametrize("spelling", ["dot_product", "inner_product", "ip"])
    def test_chroma_maps_the_inner_product_family_to_ip(self, spelling):
        from dataknobs_data.vector.stores.chroma import ChromaVectorStore

        assert ChromaVectorStore({"dimensions": 4, "metric": spelling}).chroma_metric == "ip"

    @requires_chromadb
    @pytest.mark.parametrize("spelling", ["l1", "manhattan", "l1_distance"])
    def test_chroma_refuses_a_metric_it_cannot_serve(self, spelling):
        """Chromadb validates ``hnsw:space`` against ``^(l2|cosine|ip)$``.

        So there is no ``L1`` arm to add and the fix at this site is a
        refusal, which is what the sibling door already does. A store
        configured for Manhattan distance must fail at construction rather
        than build a cosine index.
        """
        from dataknobs_data.vector.stores.chroma import ChromaVectorStore

        with pytest.raises(ValueError, match="l1"):
            ChromaVectorStore({"dimensions": 4, "metric": spelling})

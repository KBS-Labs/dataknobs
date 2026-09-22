# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""One question --- *is this the same model?* --- asked in one voice.

``MODEL_NAME_KEY`` was published so that the **key** would be spelled once,
because it had been spelled at each of its sites and that is how a reader
came to read a key nothing wrote. The rule for reading it was left at each
site, and the same thing happened again: two copies of *absent is unknown,
otherwise exact equality* drifted apart in two cases, and both drifts are
false alarms on the older copy.

What is deliberately **not** unified is the registry's comparison. That one
has a human-written document on one side and an embedder's ``model_id`` on
the other, so a provider prefix and a version tag are things the document
may omit; both sides of the comparisons here are ``model_id`` values, where
an omission is a disagreement. Two questions, two rules --- and a test below
pins the difference so it stays a decision rather than an accident.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dataknobs_data
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.dedup import DedupChecker, DedupConfig
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector.content import MODEL_NAME_KEY
from dataknobs_data.vector.semantic_index import SemanticIndex
from dataknobs_data.vector.stores.memory import MemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dataknobs_common.index import IndexItem

DIMENSIONS = 8

#: The one text every row here holds, so a search always returns the row and
#: the model comparison is the only thing under test.
TEXT = "alpha beta"


class _NoSource:
    """A source that streams nothing.

    Every test here writes its rows directly, because what is under test is
    how a *stored* row is read --- so the source exists only to satisfy the
    constructor, and a source that yields would put rows in the store that
    the test did not choose the metadata of.
    """

    source_field = "text"

    async def stream_items(self) -> AsyncIterator[IndexItem]:
        return
        yield  # pragma: no cover - unreachable, declares the generator

    async def describe(self) -> dict[str, Any]:
        return {}


async def _store_holding(metadata: dict[str, Any]) -> MemoryVectorStore:
    """A store holding one row for :data:`TEXT`, carrying *metadata*."""
    store = MemoryVectorStore(dimensions=DIMENSIONS)
    await store.initialize()
    writer = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="writer")
    await store.add_vectors(
        vectors=[(await writer.embed([TEXT]))[0]],
        ids=["r-1"],
        metadata=[{"text": TEXT, **metadata}],
    )
    return store


async def _index_reports(metadata: dict[str, Any], mine: str) -> list[str]:
    """What ``SemanticIndex`` says about that row, searched under *mine*."""
    store = await _store_holding(metadata)
    index = SemanticIndex(_NoSource(), DeterministicEmbedder(DIMENSIONS, model_id=mine), store)
    await index.search(TEXT, k=5)
    return index.mismatched_model_ids


async def _dedup_reports(metadata: dict[str, Any], mine: str) -> list[str]:
    """What ``DedupChecker`` says about that row, checked under *mine*."""
    store = await _store_holding(metadata)
    db = AsyncMemoryDatabase()
    await db.connect()
    checker = DedupChecker(
        db=db,
        config=DedupConfig(semantic_check=True),
        vector_store=store,
        embedder=DeterministicEmbedder(DIMENSIONS, model_id=mine),
    )
    # A *near* duplicate, so it reaches the semantic pass rather than being
    # answered by the exact-hash lane before any vector is compared.
    result = await checker.check({"body": f"{TEXT} gamma"})
    return result.mismatched_model_ids


#: Every case the two store-row readers can be asked, and the one answer
#: each has. Written as a table because the point is that *one* column of
#: answers is correct for both readers --- two columns would be the defect.
CASES: list[tuple[str, dict[str, Any], str, list[str]]] = [
    (
        "a foreign name is reported",
        {MODEL_NAME_KEY: "writer"},
        "reader",
        ["writer"],
    ),
    (
        "the same name is not",
        {MODEL_NAME_KEY: "writer"},
        "writer",
        [],
    ),
    (
        "a row that recorded no name is unknown, not foreign",
        {},
        "reader",
        [],
    ),
    (
        "a row that recorded an empty name is unknown too",
        {MODEL_NAME_KEY: ""},
        "reader",
        [],
    ),
    (
        "an embedder publishing no identity accuses nothing",
        {MODEL_NAME_KEY: "writer"},
        "",
        [],
    ),
]


class TestOneRuleOneAnswer:
    """The two store-row readers agree, case by case."""

    async def test_the_semantic_index_answers_the_table(self) -> None:
        for label, metadata, mine, expected in CASES:
            assert await _index_reports(metadata, mine) == expected, label

    async def test_the_dedup_checker_answers_the_table(self) -> None:
        for label, metadata, mine, expected in CASES:
            assert await _dedup_reports(metadata, mine) == expected, label

    async def test_they_answer_it_alike(self) -> None:
        """The assertion that survives the table being wrong.

        Both tests above could be satisfied by a table written to whichever
        answers the code happens to give. This one cannot: it compares the
        two readers to each other, so it fails on any case where they differ
        whatever the table says.
        """
        for label, metadata, mine, _expected in CASES:
            index = await _index_reports(metadata, mine)
            dedup = await _dedup_reports(metadata, mine)
            assert index == dedup, f"{label}: index said {index}, dedup said {dedup}"


class TestAnEmptyNameIsSilenceNotDisagreement:
    """``""`` records no model, exactly as an absent key does.

    A name is written from an embedder's ``model_id``, and an embedder that
    publishes an empty one has published nothing. Reading that as a foreign
    model names the empty string in ``mismatched_model_ids`` --- a model
    identity a caller cannot look up, act on, or re-embed against.
    """

    async def test_a_row_recording_an_empty_name_is_not_foreign(self) -> None:
        assert await _dedup_reports({MODEL_NAME_KEY: ""}, "reader") == []

    async def test_an_embedder_publishing_no_identity_accuses_no_row(self) -> None:
        """The worse half: *every* row in the store becomes a mismatch.

        With no identity of its own there is nothing to compare against, so
        the comparison has one side. Running it anyway makes every stored
        name differ from ``""`` and reports the whole corpus stale --- the
        case ``_check_declared_model`` handles explicitly one module over,
        because *"refusing would make a legitimate embedder unusable"*.
        """
        assert await _dedup_reports({MODEL_NAME_KEY: "writer"}, "") == []


class TestTheKeyIsReachableWhereItsSiblingIs:
    """``MODEL_NAME_KEY`` is published, and then not re-exported.

    ``content.py`` publishes it *"because it was spelled at each site, and a
    reader reaching for the wrong one gets silence"*. Its declared sibling
    ``CONTENT_HASH_KEY`` reaches the package door; this one stops at the
    module, so a consumer writing the import the module docstring invites
    gets an ``ImportError`` for half of it.
    """

    def test_both_staleness_keys_reach_the_package_door(self) -> None:
        from dataknobs_data import vector

        assert vector.CONTENT_HASH_KEY == "content_hash"
        assert vector.MODEL_NAME_KEY == "model_name"
        assert vector.MODEL_VERSION_KEY == "model_version"
        # Reachable *and* declared: an attribute that resolves because the
        # module happens to import it is a different promise from one the
        # door says it exports, and the digest key makes both.
        assert {"CONTENT_HASH_KEY", "MODEL_NAME_KEY", "MODEL_VERSION_KEY"} <= set(vector.__all__)

    def test_the_rule_and_its_readers_reach_it_too(self) -> None:
        """A key nothing can read back is half a publication.

        The reason the key is published --- so a reader does not spell it
        and get silence --- is only served if the reading is reachable from
        the same place.
        """
        from dataknobs_data import vector

        assert {
            "foreign_model_names",
            "is_foreign_model",
            "row_model_name",
            "sidecar_model_name",
            "sidecar_model_version",
        } <= set(vector.__all__)


class TestWhatAStoredRowIsNotAskedToMean:
    """The boundary, pinned --- both halves of it.

    These pass against the tree they were written on and must keep passing.
    They are here because the two changes that would break them are exactly
    the two plausible ways to over-unify this: read a *different* container's
    shape out of a store row, and apply the *registry's* omission rule to
    two ``model_id`` values.
    """

    async def test_a_row_whose_own_data_names_a_model_is_not_an_accusation(self) -> None:
        """A store row's metadata is the caller's namespace, not a schema.

        The store documents five keys of its own and passes everything else
        through untouched --- ``search_similar_records`` even promotes them
        to fields of the record it synthesises. A corpus of vehicles carries
        ``model``; reading that as the embedding model it was written under
        would warn a correctly built index that its own rankings are
        meaningless.

        This is why the nested ``{"model": {"name": ...}}`` shape is not read
        here. That shape belongs to the ``{field}_metadata`` sidecar, where
        it is a declared key of ``VectorMetadata``; in a store row the same
        spelling is data.
        """
        assert await _index_reports({"model": {"name": "Civic", "year": 2019}}, "reader") == []
        assert await _index_reports({"model": "Civic"}, "reader") == []

    async def test_a_document_may_omit_a_tag_and_a_stored_row_may_not(self) -> None:
        """Two questions, two rules, and the difference is deliberate.

        ``_same_model`` lets a configuration document declare
        ``nomic-embed-text`` against an embedder publishing
        ``nomic-embed-text:latest``, because the tag is a default nobody
        typed and requiring it would refuse a correctly configured
        deployment. Both sides of a stored-row comparison are ``model_id``
        values instead, produced by the same mechanism --- so an omission
        there is a disagreement between two embedders, and softening it
        would make the staleness guard miss a version bump, which is
        precisely the change a calibrated threshold cannot survive.
        """
        from dataknobs_data.ontology.registry import _model_id_readings, _same_model

        assert any(
            _same_model("nomic-embed-text", model)
            for _, model in _model_id_readings("ollama:nomic-embed-text:latest")
        )

        assert await _index_reports(
            {MODEL_NAME_KEY: "ollama:nomic-embed-text:latest"},
            "ollama:nomic-embed-text",
        ) == ["ollama:nomic-embed-text:latest"]

    async def test_the_whole_published_identity_is_always_one_of_its_readings(self) -> None:
        """The document may always name the identity it was shown, verbatim.

        The registry's side of the boundary reads a ``model_id`` under every
        ``(provider, model)`` pair it could be, because the protocol promises
        no format. This pins the reading that assumes nothing: whatever the
        string is, *the whole of it* is one of the models offered for
        comparison.

        It is the invariant a determinate split breaks. Splitting on the
        first colon dropped this reading, and with it a document naming the
        published identity exactly --- while the refusal that followed
        printed that identity as the repair.
        """
        from dataknobs_data.ontology.registry import _model_id_readings

        for published in (
            "deterministic",
            "nomic-embed-text:latest",
            "kb:my-model",
            "ollama:nomic-embed-text",
            "ollama:nomic-embed-text:latest",
            "",
            ":leading",
            "trailing:",
        ):
            assert (None, published) in _model_id_readings(published), published


# --------------------------------------------------------------------------
# The recurrence guard
# --------------------------------------------------------------------------
#
# The rule was re-spelled at five sites over the life of this key, and two of
# the copies drifted before anyone compared them. A test that pins today's five
# does not stop a seventh, so this detects the *shape* --- a stored model name
# compared by hand --- across the package.
#
# It is written against the AST rather than a token, because every one of the
# five read the name into a local first and compared the local. A census keyed
# to `MODEL_NAME_KEY` near a `!=` would have found none of them, which is the
# lesson the same package's async-dispatch guard was rewritten for.
#
# Scoped to `dataknobs-data`, which owns the key and the rule, and the other
# packages were measured rather than assumed: bots, common, utils and config
# hold none of this shape, and llm's one hit --- `session.model_name !=
# self.model` in the FSM resource seam --- is a chat session's requested
# model against a resource's configured one, which shares a spelling and
# nothing else. So widening the walk today would add 385 files and find
# nothing; it becomes worth doing when a second package reads a stored
# vector's identity.

#: Where the rule itself lives, and the one file allowed to compare by hand.
_RULE_MODULE = "vector/content.py"

#: What a compare-by-hand looks like once the read is followed one hop.
#: Every shape this replaced is reproduced in the positive controls below,
#: so the guard proves it can still fire rather than asserting zero into
#: a scope expression that quietly stopped matching.
_MODEL_NAME_READERS = frozenset({"row_model_name", "sidecar_model_name"})

#: A floor on the scope, for the reason a guard with no declarations needs
#: one: nothing else would notice this walk matching fewer files than the
#: package has.
MINIMUM_FILES_SCANNED = 90


def _is_model_name_read(node: ast.expr) -> bool:
    """Whether this expression reads a vector's recorded model name."""
    if isinstance(node, ast.Attribute) and node.attr == MODEL_NAME_KEY:
        return True
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and key.value == MODEL_NAME_KEY:
            return True
        if isinstance(key, ast.Name) and key.id == "MODEL_NAME_KEY":
            return True
    if isinstance(node, ast.Call):
        function = node.func
        if isinstance(function, ast.Name) and function.id in _MODEL_NAME_READERS:
            return True
        if isinstance(function, ast.Name) and function.id == "getattr" and len(node.args) >= 2:
            named = node.args[1]
            if isinstance(named, ast.Constant) and named.value == MODEL_NAME_KEY:
                return True
        if isinstance(function, ast.Attribute) and function.attr == "get" and node.args:
            key_arg = node.args[0]
            if isinstance(key_arg, ast.Constant) and key_arg.value == MODEL_NAME_KEY:
                return True
            if isinstance(key_arg, ast.Name) and key_arg.id == "MODEL_NAME_KEY":
                return True
    return False


def _names_bound_to_a_read(function: ast.AST) -> set[str]:
    """Locals in *function* assigned from a model-name read, transitively.

    One hop is what every site here used --- read into a local, compare the
    local --- and the loop keeps going so that a second hop does not escape.
    """
    bound: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                continue
            value = node.value
            if value is None:
                continue
            # A conditional read is still a read: `x.get(K) if meta else None`
            # is how one of the five spelled it.
            reads = _is_model_name_read(value) or (
                isinstance(value, ast.IfExp)
                and (_is_model_name_read(value.body) or _is_model_name_read(value.orelse))
            )
            if not reads:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in bound:
                    bound.add(target.id)
                    changed = True
    return bound


def _hand_written_comparisons(source: str, label: str) -> list[str]:
    """Every ``==``/``!=`` in *source* with a model name on one side."""
    found: list[str] = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound = _names_bound_to_a_read(function)
        for node in ast.walk(function):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                continue
            for operand in (node.left, *node.comparators):
                if _is_model_name_read(operand) or (
                    isinstance(operand, ast.Name) and operand.id in bound
                ):
                    found.append(f"{label}:{node.lineno}  {ast.unparse(node)}")
                    break
    return found


#: The shapes this change replaced, **one per site**, reproduced rather than
#: referenced. A guard asserting only "nothing found" passes just as well
#: when it has stopped being able to find anything, and a guard that can find
#: four of five shapes reports green over the fifth.
_POSITIVE_CONTROLS = {
    "dedup": """
def f(results, current_model):
    for rid, score, meta in results:
        stored_model = meta.get(MODEL_NAME_KEY) if meta else None
        if stored_model is not None and stored_model != current_model:
            pass
""",
    "semantic index": """
def f(hits, mine):
    for hit in hits:
        theirs = hit.metadata.get(MODEL_NAME_KEY)
        if not theirs or theirs == mine:
            continue
""",
    "synchronizer, VectorField lane": """
def f(field_obj, self):
    stored_name = field_obj.model_name
    if stored_name is not None and stored_name != self.model_name:
        return False
""",
    "synchronizer, plain-value lane": """
def f(record, vector_field, self):
    metadata = record.get_value(f"{vector_field}_metadata")
    stored_name = _stored_model_name(metadata) if isinstance(metadata, dict) else None
    if stored_name is not None and stored_name != self.model_name:
        return False
""",
    "sync mixin": """
def f(record, vector_field, model_name):
    stored_name = getattr(record.fields[vector_field], "model_name", None)
    if model_name is not None and stored_name is not None and stored_name != model_name:
        return True
""",
    "a new site, reading through the published reader": """
def f(sidecar, mine):
    if sidecar_model_name(sidecar) != mine:
        return False
""",
}


class TestTheRuleIsNotRespelled:
    """No site in ``dataknobs-data`` compares a stored model name by hand."""

    def test_the_guard_finds_every_shape_it_replaced(self) -> None:
        for label, source in _POSITIVE_CONTROLS.items():
            assert _hand_written_comparisons(source, label), f"{label} went undetected"

    def test_nothing_in_the_package_compares_one_by_hand(self) -> None:
        root = Path(dataknobs_data.__file__).parent
        scanned = 0
        found: list[str] = []
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative == _RULE_MODULE:
                continue
            scanned += 1
            found += _hand_written_comparisons(path.read_text(), relative)

        assert scanned >= MINIMUM_FILES_SCANNED, (
            f"the walk reached {scanned} files; it used to reach more, so the guard "
            f"is reporting green over a package it has stopped reading"
        )
        assert not found, (
            "a stored model name is compared by hand here; call "
            "`dataknobs_data.vector.is_foreign_model`, which is the rule these "
            "sites kept restating and disagreeing about:\n  " + "\n  ".join(found)
        )

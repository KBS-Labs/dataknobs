"""Behavioral tests for identity/metadata configurability in
``VectorKnowledgeSource``.

Covers Item 97: three optional callables (``dedup_key``,
``source_id_fn``, ``metadata_fn``) on both the main
``VectorKnowledgeSource.query`` path and the topic-index path
(``_build_vector_query_fn`` closure), plus factory wiring via
``GroundedSourceConfig.options``.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.sources.base import RetrievalIntent

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.knowledge.sources.factory import (
    _build_vector_query_fn,
    _create_vector_kb_source,
    build_topic_index,
)
from dataknobs_bots.knowledge.sources.vector import (
    VectorKnowledgeSource,
    default_dedup_key,
    default_metadata,
    default_source_id,
)
from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig


# -------------------------------------------------------------------
# Module-level helpers resolvable via dotted-import paths (factory tests)
# -------------------------------------------------------------------


def _dedup_by_entity_ref(r: dict[str, Any]) -> Any:
    return r.get("metadata", {}).get("entity_ref") or id(r)


def _id_by_entity_ref(r: dict[str, Any]) -> str:
    return r.get("metadata", {}).get("entity_ref", "")


def _md_by_entity_ref(r: dict[str, Any]) -> dict[str, Any]:
    m = r.get("metadata", {})
    return {"entity_ref": m.get("entity_ref")}


# -------------------------------------------------------------------
# Test doubles
# -------------------------------------------------------------------


class ScriptedKnowledgeBase(KnowledgeBase):
    """Test KB that returns a fixed record list for every query."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.query_count = 0
        self.last_filter: dict[str, Any] | None = None

    async def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.query_count += 1
        self.last_filter = filter_metadata
        return self._records[:k]

    async def close(self) -> None:
        pass


def _record(
    text: str,
    source: str = "doc.md",
    chunk_index: int | None = 0,
    heading_path: str = "",
    metadata: dict[str, Any] | None = None,
    similarity: float = 0.9,
) -> dict[str, Any]:
    md: dict[str, Any] = {}
    if chunk_index is not None:
        md["chunk_index"] = chunk_index
    if metadata:
        md.update(metadata)
    return {
        "text": text,
        "source": source,
        "heading_path": heading_path,
        "similarity": similarity,
        "metadata": md,
    }


# -------------------------------------------------------------------
# Main-path tests (Change 1 in the plan)
# -------------------------------------------------------------------


class TestDefaultBehaviorBackwardCompat:
    """The three defaults must preserve historical behavior exactly."""

    async def test_default_dedup_and_source_id_backward_compat(self) -> None:
        kb = ScriptedKnowledgeBase([
            _record("a", "doc.md", 0),
            _record("b", "doc.md", 1),
        ])
        source = VectorKnowledgeSource(kb, name="docs")

        intent = RetrievalIntent(text_queries=["q"])
        results = await source.query(intent)

        assert len(results) == 2
        ids = sorted(r.source_id for r in results)
        assert ids == ["doc.md:chunk_0", "doc.md:chunk_1"]

    async def test_default_dedup_across_queries(self) -> None:
        kb = ScriptedKnowledgeBase([_record("same", "doc.md", 0)])
        source = VectorKnowledgeSource(kb, name="docs")

        # Two text queries each match the same record — dedup should
        # collapse to one emitted SourceResult.
        intent = RetrievalIntent(text_queries=["q1", "q2"])
        results = await source.query(intent)

        assert len(results) == 1
        assert kb.query_count == 2  # KB called once per query...
        # ...but the dedup collapsed the duplicate.

    async def test_default_id_fallback_when_chunk_index_missing(
        self,
    ) -> None:
        # Two distinct record dicts with no chunk_index → id(r) makes
        # the dedup keys distinct, preserving pre-change behavior.
        r1 = _record("a", "doc.md", chunk_index=None)
        r2 = _record("b", "doc.md", chunk_index=None)
        kb = ScriptedKnowledgeBase([r1, r2])
        source = VectorKnowledgeSource(kb, name="docs")

        intent = RetrievalIntent(text_queries=["q"])
        results = await source.query(intent)

        assert len(results) == 2

    async def test_default_metadata_surface_backward_compat(self) -> None:
        kb = ScriptedKnowledgeBase([
            _record(
                "x",
                source="doc.md",
                chunk_index=0,
                heading_path="A > B",
                metadata={"extra": 1},
            ),
        ])
        source = VectorKnowledgeSource(kb, name="docs")

        intent = RetrievalIntent(text_queries=["q"])
        results = await source.query(intent)

        assert len(results) == 1
        assert results[0].metadata == {
            "heading_path": "A > B",
            "source": "doc.md",
            "chunk_index": 0,
            "extra": 1,
        }


class TestCustomCallables:
    """Opt-in callables must override the corresponding defaults."""

    async def test_custom_dedup_key_collapses_by_entity_ref(self) -> None:
        # Two records with same entity_ref but distinct source/chunk_index
        # — default dedup would keep both, custom dedup collapses.
        kb = ScriptedKnowledgeBase([
            _record(
                "first",
                source="fd-02.md",
                chunk_index=0,
                metadata={"entity_ref": "FD-02"},
            ),
            _record(
                "second",
                source="other.md",
                chunk_index=5,
                metadata={"entity_ref": "FD-02"},
            ),
        ])
        source = VectorKnowledgeSource(
            kb,
            name="claims",
            dedup_key=_dedup_by_entity_ref,
        )

        intent = RetrievalIntent(text_queries=["q"])
        results = await source.query(intent)

        assert len(results) == 1

    async def test_custom_source_id_fn(self) -> None:
        kb = ScriptedKnowledgeBase([
            _record(
                "claim",
                source="whatever.md",
                chunk_index=0,
                metadata={"entity_ref": "FD-02"},
            ),
        ])
        source = VectorKnowledgeSource(
            kb,
            name="claims",
            source_id_fn=_id_by_entity_ref,
        )

        results = await source.query(RetrievalIntent(text_queries=["q"]))

        assert results[0].source_id == "FD-02"

    async def test_custom_metadata_fn_replaces_surface(self) -> None:
        kb = ScriptedKnowledgeBase([
            _record(
                "claim",
                source="whatever.md",
                chunk_index=0,
                heading_path="Sec > Sub",
                metadata={"entity_ref": "FD-02", "noise": "ignore"},
            ),
        ])
        source = VectorKnowledgeSource(
            kb,
            name="claims",
            metadata_fn=_md_by_entity_ref,
        )

        results = await source.query(RetrievalIntent(text_queries=["q"]))

        assert results[0].metadata == {"entity_ref": "FD-02"}
        assert "heading_path" not in results[0].metadata
        assert "source" not in results[0].metadata

    async def test_custom_callables_together(self) -> None:
        kb = ScriptedKnowledgeBase([
            _record(
                "a",
                source="file1.md",
                chunk_index=0,
                metadata={"entity_ref": "FD-02"},
            ),
            _record(
                "b",
                source="file2.md",
                chunk_index=7,
                metadata={"entity_ref": "FD-02"},  # duplicate by entity_ref
            ),
            _record(
                "c",
                source="file3.md",
                chunk_index=2,
                metadata={"entity_ref": "FD-03"},
            ),
        ])
        source = VectorKnowledgeSource(
            kb,
            name="claims",
            dedup_key=_dedup_by_entity_ref,
            source_id_fn=_id_by_entity_ref,
            metadata_fn=_md_by_entity_ref,
        )

        results = await source.query(RetrievalIntent(text_queries=["q"]))

        assert len(results) == 2
        ids = sorted(r.source_id for r in results)
        assert ids == ["FD-02", "FD-03"]
        for r in results:
            assert set(r.metadata.keys()) == {"entity_ref"}

    async def test_missing_identity_graceful_defaults(self) -> None:
        # Record missing both source and chunk_index — defaults should
        # fall back to id(r) for dedup (same pre-change behavior).
        r1 = {"text": "a", "similarity": 0.9, "metadata": {}}
        r2 = {"text": "b", "similarity": 0.9, "metadata": {}}
        kb = ScriptedKnowledgeBase([r1, r2])
        source = VectorKnowledgeSource(kb, name="docs")

        results = await source.query(
            RetrievalIntent(text_queries=["q1", "q2"]),
        )

        # Two distinct dicts → two distinct id(r) → two emitted results
        # per query, but same objects repeated across queries collapse.
        assert len(results) == 2


class TestRaisingCallables:
    """A misbehaving identity callable must not crash the retrieval turn.

    Regression: before the exception-safety fix, a callable that raised
    propagated uncaught from ``VectorKnowledgeSource.query``, discarding
    all results from every source in the turn. The fix isolates each
    record: the offending record is skipped with a warning; sibling
    records on the same query — and on later queries — still flow.
    """

    async def test_raising_dedup_key_skips_record(self) -> None:
        def boom(r: dict[str, Any]) -> Any:
            raise RuntimeError("dedup_key blew up")

        kb = ScriptedKnowledgeBase([
            _record("a", "doc.md", 0),
            _record("b", "doc.md", 1),
        ])
        source = VectorKnowledgeSource(kb, name="docs", dedup_key=boom)

        results = await source.query(RetrievalIntent(text_queries=["q"]))

        # Every record trips the raising callable, so we end up with
        # zero results — but the call returned cleanly (no exception).
        assert results == []

    async def test_raising_source_id_fn_preserves_other_records(
        self,
    ) -> None:
        # Raise only for the "bad" record, let the good one through.
        def partial_boom(r: dict[str, Any]) -> str:
            if r.get("source") == "bad.md":
                raise ValueError("nope")
            return r.get("source", "")

        kb = ScriptedKnowledgeBase([
            _record("good", "good.md", 0),
            _record("bad", "bad.md", 0),
        ])
        source = VectorKnowledgeSource(
            kb, name="docs", source_id_fn=partial_boom,
        )

        results = await source.query(RetrievalIntent(text_queries=["q"]))

        assert len(results) == 1
        assert results[0].source_id == "good.md"

    async def test_raising_metadata_fn_does_not_pollute_dedup(
        self,
    ) -> None:
        # Raising metadata_fn must happen BEFORE the dedup key is added
        # to the seen set — otherwise a second text query would see a
        # duplicate key and silently drop the record even though the
        # first record never emitted.
        call_count = {"n": 0}

        def raise_first_then_pass(r: dict[str, Any]) -> dict[str, Any]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first call always fails")
            return {"ok": True}

        kb = ScriptedKnowledgeBase([_record("a", "doc.md", 0)])
        source = VectorKnowledgeSource(
            kb, name="docs", metadata_fn=raise_first_then_pass,
        )

        # Two identical queries — if the dedup key from the first
        # (raising) attempt had leaked into ``seen``, the second
        # would be skipped. The record must emit on retry.
        results = await source.query(
            RetrievalIntent(text_queries=["q1", "q2"]),
        )

        assert len(results) == 1
        assert results[0].metadata == {"ok": True}

    async def test_raising_callable_in_topic_index_closure(self) -> None:
        # Mirror the contract on the topic-index path: raising callables
        # skip the offending record without aborting the closure.
        def boom_once(r: dict[str, Any]) -> str:
            if r.get("source") == "bad.md":
                raise ValueError("nope")
            return f"{r.get('source', '')}:chunk_{r.get('metadata', {}).get('chunk_index', '')}"

        kb = ScriptedKnowledgeBase([
            _record("good", "good.md", 0),
            _record("bad", "bad.md", 0),
        ])
        fn = _build_vector_query_fn(kb, "docs", source_id_fn=boom_once)

        results = await fn("q", top_k=5)

        assert len(results) == 1
        assert results[0].source_id == "good.md:chunk_0"


class TestTopicIndexFilterPassthrough:
    """``intent.filters[source_name]`` must reach the KB through both
    the main :class:`VectorKnowledgeSource` path AND the topic-index
    seeding path.
    """

    async def test_topic_index_forwards_filter_to_kb(self) -> None:
        # KB carries the filter it receives on ``last_filter`` so we
        # can assert the filter slice arrived on the topic-index seed
        # fetch, not the no-filter fallback. The filter assertion is
        # independent of whether resolve() emits results — we only
        # need the seed fetch to run.
        kb = ScriptedKnowledgeBase([
            _record(
                "a",
                source="whatever.md",
                chunk_index=0,
                heading_path="Security",
                metadata={
                    "entity_ref": "FD-02",
                    "headings": ["Security"],
                    "heading_levels": [1],
                },
                similarity=0.95,
            ),
        ])
        cfg = GroundedSourceConfig(
            source_type="vector_kb",
            name="claims",
            topic_index={
                "type": "heading_tree",
                "seed_score_threshold": 0.0,
                "min_heading_depth": 1,
            },
        )
        source = _create_vector_kb_source(cfg, kb)

        assert source.topic_index is not None
        await source.topic_index.resolve(
            "security",
            top_k=5,
            intent=RetrievalIntent(
                text_queries=["security"],
                filters={"claims": {"entity_ref": "FD-02"}},
            ),
        )

        assert kb.last_filter == {"entity_ref": "FD-02"}

    async def test_topic_index_no_filter_preserves_legacy_behavior(
        self,
    ) -> None:
        # Without intent.filters the KB must see ``None``, not ``{}``.
        kb = ScriptedKnowledgeBase([
            _record(
                "a",
                source="whatever.md",
                chunk_index=0,
                heading_path="Security",
                metadata={
                    "headings": ["Security"],
                    "heading_levels": [1],
                },
                similarity=0.95,
            ),
        ])
        cfg = GroundedSourceConfig(
            source_type="vector_kb",
            name="claims",
            topic_index={
                "type": "heading_tree",
                "seed_score_threshold": 0.0,
                "min_heading_depth": 1,
            },
        )
        source = _create_vector_kb_source(cfg, kb)

        assert source.topic_index is not None
        await source.topic_index.resolve(
            "security",
            top_k=5,
            intent=RetrievalIntent(text_queries=["security"]),
        )

        assert kb.last_filter is None


# -------------------------------------------------------------------
# Module-level default sanity
# -------------------------------------------------------------------


class TestModuleDefaults:
    def test_default_dedup_key_tuple(self) -> None:
        r = {
            "source": "doc.md",
            "metadata": {"chunk_index": 3},
        }
        assert default_dedup_key(r) == ("doc.md", 3)

    def test_default_source_id_template(self) -> None:
        r = {
            "source": "doc.md",
            "metadata": {"chunk_index": 3},
        }
        assert default_source_id(r) == "doc.md:chunk_3"

    def test_default_metadata_surface(self) -> None:
        r = {
            "source": "doc.md",
            "heading_path": "A > B",
            "metadata": {"extra": 1},
        }
        assert default_metadata(r) == {
            "heading_path": "A > B",
            "source": "doc.md",
            "extra": 1,
        }


# -------------------------------------------------------------------
# Topic-index path tests (Change 3 in the plan)
# -------------------------------------------------------------------


class TestTopicIndexPath:
    """The topic-index vector_query_fn closure must respect the same
    callables when configured."""

    async def test_topic_index_default_source_id(self) -> None:
        kb = ScriptedKnowledgeBase([
            _record("a", "doc.md", 0),
        ])
        fn = _build_vector_query_fn(kb, "docs")

        results = await fn("q", top_k=5)

        assert len(results) == 1
        assert results[0].source_id == "doc.md:chunk_0"
        assert results[0].source_name == "docs"
        assert results[0].source_type == "vector_kb"

    async def test_topic_index_custom_source_id_fn(self) -> None:
        kb = ScriptedKnowledgeBase([
            _record(
                "a",
                source="whatever.md",
                chunk_index=0,
                metadata={"entity_ref": "FD-02"},
            ),
        ])
        fn = _build_vector_query_fn(
            kb, "claims", source_id_fn=_id_by_entity_ref,
        )

        results = await fn("q", top_k=5)

        assert results[0].source_id == "FD-02"

    async def test_topic_index_custom_metadata_fn(self) -> None:
        kb = ScriptedKnowledgeBase([
            _record(
                "a",
                source="whatever.md",
                chunk_index=0,
                heading_path="Sec",
                metadata={"entity_ref": "FD-02", "noise": "x"},
            ),
        ])
        fn = _build_vector_query_fn(
            kb, "claims", metadata_fn=_md_by_entity_ref,
        )

        results = await fn("q", top_k=5)

        assert results[0].metadata == {"entity_ref": "FD-02"}


# -------------------------------------------------------------------
# Factory wiring tests (Change 2 in the plan)
# -------------------------------------------------------------------


class TestFactoryResolution:
    """``_create_vector_kb_source`` must resolve dotted-import callable
    strings from ``config.options`` and pass them through."""

    def _config(self, **options: Any) -> GroundedSourceConfig:
        return GroundedSourceConfig(
            source_type="vector_kb",
            name="test_kb",
            options=dict(options),
        )

    def test_factory_resolves_dotted_dedup_key(self) -> None:
        kb = ScriptedKnowledgeBase([])
        cfg = self._config(
            dedup_key=(
                "tests.knowledge.sources.test_vector_identity:"
                "_dedup_by_entity_ref"
            ),
        )
        source = _create_vector_kb_source(cfg, kb)

        assert isinstance(source, VectorKnowledgeSource)
        assert source._dedup_key is _dedup_by_entity_ref

    def test_factory_resolves_dotted_source_id_fn(self) -> None:
        kb = ScriptedKnowledgeBase([])
        cfg = self._config(
            source_id_fn=(
                "tests.knowledge.sources.test_vector_identity:"
                "_id_by_entity_ref"
            ),
        )
        source = _create_vector_kb_source(cfg, kb)

        assert isinstance(source, VectorKnowledgeSource)
        assert source._source_id_fn is _id_by_entity_ref

    def test_factory_resolves_dotted_metadata_fn(self) -> None:
        kb = ScriptedKnowledgeBase([])
        cfg = self._config(
            metadata_fn=(
                "tests.knowledge.sources.test_vector_identity:"
                "_md_by_entity_ref"
            ),
        )
        source = _create_vector_kb_source(cfg, kb)

        assert isinstance(source, VectorKnowledgeSource)
        assert source._metadata_fn is _md_by_entity_ref

    def test_factory_raises_on_invalid_callable_ref(self) -> None:
        kb = ScriptedKnowledgeBase([])
        cfg = self._config(dedup_key="nonexistent.module:missing_fn")

        with pytest.raises(ValueError) as exc_info:
            _create_vector_kb_source(cfg, kb)

        msg = str(exc_info.value)
        assert "'test_kb'" in msg
        assert "dedup_key" in msg

    def test_factory_omitted_callables_use_defaults(self) -> None:
        kb = ScriptedKnowledgeBase([])
        cfg = self._config()
        source = _create_vector_kb_source(cfg, kb)

        assert isinstance(source, VectorKnowledgeSource)
        assert source._dedup_key is default_dedup_key
        assert source._source_id_fn is default_source_id
        assert source._metadata_fn is default_metadata

    async def test_build_topic_index_threads_callables_to_closure(
        self,
    ) -> None:
        # ``build_topic_index`` is the shared factory helper used by
        # both ``_create_vector_kb_source`` and the legacy
        # ``GroundedReasoning.set_knowledge_base`` path. Verify that
        # custom identity callables passed here end up in the closure
        # used by the resulting topic index — exercised through the
        # public ``_fetch_vector_seeds`` path the index uses at query
        # time (no reliance on a specific resolve()-level heading
        # configuration).
        kb = ScriptedKnowledgeBase([
            _record(
                "a",
                source="whatever.md",
                chunk_index=0,
                heading_path="Sec",
                metadata={"entity_ref": "FD-02"},
            ),
        ])
        topic_index = build_topic_index(
            {"type": "heading_tree", "seed_score_threshold": 0.0},
            kb,
            source_name="claims",
            source_id_fn=_id_by_entity_ref,
            metadata_fn=_md_by_entity_ref,
        )

        assert topic_index is not None

        from dataknobs_bots.knowledge.sources.heading_tree import (
            _resolve_params,
        )

        params = _resolve_params(topic_index._config, None)
        seeds = await topic_index._fetch_vector_seeds(
            "anything", params,
        )

        assert len(seeds) == 1
        assert seeds[0].source_id == "FD-02"
        assert seeds[0].metadata == {"entity_ref": "FD-02"}

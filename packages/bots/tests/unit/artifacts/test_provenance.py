"""Tests for artifact provenance models."""

from __future__ import annotations

from dataknobs_bots.artifacts.provenance import (
    LLMInvocation,
    ProvenanceRecord,
    RevisionRecord,
    SourceReference,
    ToolInvocation,
    create_provenance,
)


class TestSourceReference:
    def test_creation(self) -> None:
        ref = SourceReference(
            source_id="doc_123",
            source_type="document",
            source_location="/docs/math.pdf",
            relevance="Primary content source",
            excerpt="Chapter 3: Algebra",
            confidence=0.95,
        )
        assert ref.source_id == "doc_123"
        assert ref.source_type == "document"
        assert ref.confidence == 0.95

    def test_defaults(self) -> None:
        ref = SourceReference(source_id="x", source_type="artifact")
        assert ref.source_location is None
        assert ref.relevance == ""
        assert ref.excerpt is None
        assert ref.confidence == 1.0

    def test_serialization_round_trip(self) -> None:
        ref = SourceReference(
            source_id="art_abc",
            source_type="artifact",
            relevance="Derived from",
            confidence=0.8,
        )
        restored = SourceReference.from_dict(ref.to_dict())
        assert restored.source_id == ref.source_id
        assert restored.source_type == ref.source_type
        assert restored.confidence == ref.confidence

    def test_to_dict_omits_none_fields(self) -> None:
        ref = SourceReference(source_id="x", source_type="artifact")
        d = ref.to_dict()
        assert "source_location" not in d
        assert "excerpt" not in d
        assert "relevance" not in d  # Empty string omitted


class TestToolInvocation:
    def test_creation(self) -> None:
        inv = ToolInvocation(
            tool_name="quiz_generator",
            tool_version="1.2.0",
            parameters={"topic": "algebra", "count": 5},
        )
        assert inv.tool_name == "quiz_generator"
        assert inv.tool_version == "1.2.0"
        assert inv.parameters == {"topic": "algebra", "count": 5}
        assert inv.timestamp != ""

    def test_serialization_round_trip(self) -> None:
        inv = ToolInvocation(
            tool_name="formatter",
            parameters={"format": "markdown"},
        )
        restored = ToolInvocation.from_dict(inv.to_dict())
        assert restored.tool_name == inv.tool_name
        assert restored.parameters == inv.parameters


class TestLLMInvocation:
    def test_creation(self) -> None:
        inv = LLMInvocation(
            purpose="decode_intent",
            model="llama3.2",
            prompt_hash="abc123",
        )
        assert inv.purpose == "decode_intent"
        assert inv.model == "llama3.2"

    def test_serialization_round_trip(self) -> None:
        inv = LLMInvocation(
            purpose="encode_feedback",
            model="ollama:llama3.2",
        )
        restored = LLMInvocation.from_dict(inv.to_dict())
        assert restored.purpose == inv.purpose
        assert restored.model == inv.model


class TestRevisionRecord:
    def test_creation(self) -> None:
        rev = RevisionRecord(
            previous_version="1.0.0",
            reason="Failed rubric evaluation",
            changes_summary="Updated question wording",
            triggered_by="rubric_evaluation:eval_123",
        )
        assert rev.revision_id.startswith("rev_")
        assert rev.previous_version == "1.0.0"
        assert rev.triggered_by == "rubric_evaluation:eval_123"

    def test_serialization_round_trip(self) -> None:
        rev = RevisionRecord(
            revision_id="rev_custom",
            previous_version="1.0.0",
            reason="Feedback",
            changes_summary="Fixed content",
            triggered_by="user:jane",
        )
        restored = RevisionRecord.from_dict(rev.to_dict())
        assert restored.revision_id == rev.revision_id
        assert restored.previous_version == rev.previous_version
        assert restored.triggered_by == rev.triggered_by


class TestProvenanceRecord:
    def test_creation_with_defaults(self) -> None:
        prov = ProvenanceRecord()
        assert prov.created_by == ""
        assert prov.created_at != ""
        assert prov.sources == []
        assert prov.tool_chain == []
        assert prov.llm_invocations == []
        assert prov.review_history == []
        assert prov.revision_history == []

    def test_full_provenance(self) -> None:
        prov = ProvenanceRecord(
            created_by="system:generator:quiz_gen_v1",
            creation_method="generator",
            creation_context={"topic": "algebra"},
            sources=[
                SourceReference(
                    source_id="doc_1",
                    source_type="document",
                ),
            ],
            tool_chain=[
                ToolInvocation(tool_name="quiz_generator"),
            ],
            llm_invocations=[
                LLMInvocation(purpose="decode_intent"),
            ],
            review_history=["eval_001"],
            revision_history=[
                RevisionRecord(
                    previous_version="1.0.0",
                    reason="Revision",
                ),
            ],
        )
        assert prov.created_by == "system:generator:quiz_gen_v1"
        assert len(prov.sources) == 1
        assert len(prov.tool_chain) == 1
        assert len(prov.llm_invocations) == 1
        assert prov.review_history == ["eval_001"]
        assert len(prov.revision_history) == 1

    def test_serialization_round_trip(self) -> None:
        prov = ProvenanceRecord(
            created_by="user:jane",
            creation_method="manual",
            sources=[
                SourceReference(
                    source_id="ref_1",
                    source_type="external",
                    confidence=0.9,
                ),
            ],
            tool_chain=[
                ToolInvocation(
                    tool_name="editor",
                    parameters={"mode": "advanced"},
                ),
            ],
            llm_invocations=[
                LLMInvocation(purpose="encode_feedback", model="llama3.2"),
            ],
            review_history=["eval_a", "eval_b"],
            revision_history=[
                RevisionRecord(
                    previous_version="1.0.0",
                    reason="User edit",
                    triggered_by="user:jane",
                ),
            ],
        )
        restored = ProvenanceRecord.from_dict(prov.to_dict())
        assert restored.created_by == prov.created_by
        assert restored.creation_method == prov.creation_method
        assert len(restored.sources) == 1
        assert restored.sources[0].source_id == "ref_1"
        assert len(restored.tool_chain) == 1
        assert restored.tool_chain[0].tool_name == "editor"
        assert len(restored.llm_invocations) == 1
        assert restored.llm_invocations[0].model == "llama3.2"
        assert restored.review_history == ["eval_a", "eval_b"]
        assert len(restored.revision_history) == 1
        assert restored.revision_history[0].triggered_by == "user:jane"


class TestCreateProvenance:
    def test_factory(self) -> None:
        prov = create_provenance(
            created_by="bot:configbot",
            creation_method="wizard",
        )
        assert prov.created_by == "bot:configbot"
        assert prov.creation_method == "wizard"
        assert prov.created_at != ""

    def test_factory_with_extras(self) -> None:
        prov = create_provenance(
            created_by="system:gen",
            creation_method="generator",
            creation_context={"key": "value"},
            sources=[SourceReference(source_id="s1", source_type="artifact")],
        )
        assert prov.creation_context == {"key": "value"}
        assert len(prov.sources) == 1

"""Provenance models for tracking artifact creation and revision history.

This module provides data structures for comprehensive provenance tracking:
- Source references: Where content came from
- Tool invocations: Which tools were used during creation
- LLM invocations: When and why LLMs were consulted
- Revision records: History of changes
- Provenance records: Complete creation context

Example:
    >>> provenance = create_provenance(
    ...     created_by="system:generator:quiz_gen_v1",
    ...     creation_method="generator",
    ... )
    >>> provenance.sources.append(SourceReference(
    ...     source_id="doc_123",
    ...     source_type="document",
    ... ))
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class SourceReference:
    """A reference to a source used in creating an artifact.

    Attributes:
        source_id: ID of source artifact, document, or external reference.
        source_type: Category of source (artifact, document, vector_result,
            user_input, external).
        source_location: URI, file path, or vector store reference.
        relevance: Why this source was used.
        excerpt: Relevant portion of the source.
        confidence: Relevance confidence score (0.0 to 1.0).
    """

    source_id: str
    source_type: str
    source_location: str | None = None
    relevance: str = ""
    excerpt: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        result: dict[str, Any] = {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "confidence": self.confidence,
        }
        if self.source_location is not None:
            result["source_location"] = self.source_location
        if self.relevance:
            result["relevance"] = self.relevance
        if self.excerpt is not None:
            result["excerpt"] = self.excerpt
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceReference:
        """Deserialize from a dictionary."""
        return cls(
            source_id=data["source_id"],
            source_type=data["source_type"],
            source_location=data.get("source_location"),
            relevance=data.get("relevance", ""),
            excerpt=data.get("excerpt"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ToolInvocation:
    """Record of a tool used during artifact creation.

    Attributes:
        tool_name: Function or tool identifier.
        tool_version: Version of the tool if applicable.
        parameters: Input parameters passed to the tool.
        timestamp: ISO 8601 timestamp of invocation.
    """

    tool_name: str
    tool_version: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        result: dict[str, Any] = {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
        }
        if self.tool_version is not None:
            result["tool_version"] = self.tool_version
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolInvocation:
        """Deserialize from a dictionary."""
        return cls(
            tool_name=data["tool_name"],
            tool_version=data.get("tool_version"),
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class LLMInvocation:
    """Record of an LLM call during artifact creation.

    Attributes:
        purpose: Why the LLM was called (decode_intent, encode_feedback,
            generate_context).
        model: Model identifier used.
        prompt_hash: Hash of the prompt for reproducibility (not the
            full prompt, for privacy).
        timestamp: ISO 8601 timestamp of invocation.
    """

    purpose: str
    model: str = ""
    prompt_hash: str = ""
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "purpose": self.purpose,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMInvocation:
        """Deserialize from a dictionary."""
        return cls(
            purpose=data["purpose"],
            model=data.get("model", ""),
            prompt_hash=data.get("prompt_hash", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class RevisionRecord:
    """Record of a single revision to an artifact.

    Attributes:
        revision_id: Unique identifier for this revision.
        previous_version: The version being revised.
        reason: Why the revision was made.
        changes_summary: What changed in this revision.
        triggered_by: Who or what triggered the revision
            (e.g., "rubric_evaluation:eval_123", "user:jane", "system").
        timestamp: ISO 8601 timestamp.
    """

    revision_id: str = field(default_factory=lambda: _generate_id("rev"))
    previous_version: str = ""
    reason: str = ""
    changes_summary: str = ""
    triggered_by: str = ""
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "revision_id": self.revision_id,
            "previous_version": self.previous_version,
            "reason": self.reason,
            "changes_summary": self.changes_summary,
            "triggered_by": self.triggered_by,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RevisionRecord:
        """Deserialize from a dictionary."""
        return cls(
            revision_id=data.get("revision_id", _generate_id("rev")),
            previous_version=data.get("previous_version", ""),
            reason=data.get("reason", ""),
            changes_summary=data.get("changes_summary", ""),
            triggered_by=data.get("triggered_by", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class ProvenanceRecord:
    """Complete provenance for an artifact's creation and history.

    Attributes:
        created_by: Who or what created the artifact
            (e.g., "system:generator:quiz_gen_v1", "user:jane", "bot:configbot").
        created_at: ISO 8601 timestamp of creation.
        creation_method: How the artifact was created
            (generator, wizard, manual, derived, llm_assisted).
        creation_context: Parameters, configuration, and other context.
        sources: References to source materials used.
        tool_chain: Tools used during creation.
        llm_invocations: LLM calls made during creation.
        review_history: IDs of RubricEvaluation results.
        revision_history: Records of revisions made.
    """

    created_by: str = ""
    created_at: str = field(default_factory=now_iso)
    creation_method: str = ""
    creation_context: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceReference] = field(default_factory=list)
    tool_chain: list[ToolInvocation] = field(default_factory=list)
    llm_invocations: list[LLMInvocation] = field(default_factory=list)
    review_history: list[str] = field(default_factory=list)
    revision_history: list[RevisionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "created_by": self.created_by,
            "created_at": self.created_at,
            "creation_method": self.creation_method,
            "creation_context": self.creation_context,
            "sources": [s.to_dict() for s in self.sources],
            "tool_chain": [t.to_dict() for t in self.tool_chain],
            "llm_invocations": [inv.to_dict() for inv in self.llm_invocations],
            "review_history": self.review_history,
            "revision_history": [r.to_dict() for r in self.revision_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRecord:
        """Deserialize from a dictionary."""
        return cls(
            created_by=data.get("created_by", ""),
            created_at=data.get("created_at", ""),
            creation_method=data.get("creation_method", ""),
            creation_context=data.get("creation_context", {}),
            sources=[
                SourceReference.from_dict(s) for s in data.get("sources", [])
            ],
            tool_chain=[
                ToolInvocation.from_dict(t) for t in data.get("tool_chain", [])
            ],
            llm_invocations=[
                LLMInvocation.from_dict(inv)
                for inv in data.get("llm_invocations", [])
            ],
            review_history=data.get("review_history", []),
            revision_history=[
                RevisionRecord.from_dict(r)
                for r in data.get("revision_history", [])
            ],
        )


def create_provenance(
    created_by: str,
    creation_method: str,
    **kwargs: Any,
) -> ProvenanceRecord:
    """Convenience factory for creating a ProvenanceRecord.

    Args:
        created_by: Who or what created the artifact.
        creation_method: How it was created (generator, wizard, manual, etc.).
        **kwargs: Additional fields passed to ProvenanceRecord.

    Returns:
        A new ProvenanceRecord with the given values.
    """
    return ProvenanceRecord(
        created_by=created_by,
        creation_method=creation_method,
        **kwargs,
    )

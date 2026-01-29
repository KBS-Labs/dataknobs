"""Context accumulator for managing conversation context.

This module provides data structures for:
- ConversationContext: Unified context from wizard state, artifacts, assumptions
- Assumption: Tracked assumptions about user intent or requirements
- ContextSection: Prioritized sections for prompt injection

Example:
    >>> context = ConversationContext(conversation_id="conv_123")
    >>> context.add_assumption(
    ...     content="User wants a math tutor",
    ...     source="inferred",
    ...     confidence=0.7,
    ... )
    >>> prompt_context = context.to_prompt_injection(max_tokens=2000)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

AssumptionSource = Literal["inferred", "user_stated", "default", "extracted"]


def _generate_assumption_id() -> str:
    """Generate a unique assumption ID."""
    return f"asn_{uuid.uuid4().hex[:8]}"


@dataclass
class Assumption:
    """A tracked assumption about user intent or requirements.

    Assumptions can be:
    - inferred: Bot reasoned about this based on context
    - user_stated: User explicitly said this
    - default: Applied a default value
    - extracted: Extracted from user input via schema extractor

    Attributes:
        id: Unique identifier
        content: What is being assumed
        source: How this assumption was made
        confidence: How confident we are (0.0-1.0)
        confirmed: Whether user has confirmed this
        confirmed_at: When confirmation happened
        related_to: What this assumption is about (field, stage, etc.)
        created_at: When assumption was created
    """

    id: str = field(default_factory=_generate_assumption_id)
    content: str = ""
    source: AssumptionSource = "inferred"
    confidence: float = 0.5
    confirmed: bool = False
    confirmed_at: float | None = None
    related_to: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "confirmed": self.confirmed,
            "confirmed_at": self.confirmed_at,
            "related_to": self.related_to,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assumption:
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", _generate_assumption_id()),
            content=data.get("content", ""),
            source=data.get("source", "inferred"),
            confidence=data.get("confidence", 0.5),
            confirmed=data.get("confirmed", False),
            confirmed_at=data.get("confirmed_at"),
            related_to=data.get("related_to"),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class ContextSection:
    """A section of context with priority and token budget.

    Sections allow organizing context by importance and
    controlling what gets included when tokens are limited.

    Attributes:
        name: Section identifier
        content: The content of this section
        priority: Higher = more important (included first, 0-100)
        max_tokens: Maximum tokens for this section (None = no limit)
        include_always: Always include regardless of budget
        formatter: How to format this section for prompts
            - "default": String representation
            - "json": JSON formatted
            - "list": Bulleted list
            - "summary": Key-value summary
    """

    name: str
    content: Any
    priority: int = 50  # 0-100
    max_tokens: int | None = None
    include_always: bool = False
    formatter: str = "default"  # "default", "json", "list", "summary"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "content": self.content,
            "priority": self.priority,
            "max_tokens": self.max_tokens,
            "include_always": self.include_always,
            "formatter": self.formatter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextSection:
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", ""),
            content=data.get("content"),
            priority=data.get("priority", 50),
            max_tokens=data.get("max_tokens"),
            include_always=data.get("include_always", False),
            formatter=data.get("formatter", "default"),
        )


@dataclass
class ConversationContext:
    """Accumulated context for a conversation.

    Provides a comprehensive view of everything known about
    the conversation, including:
    - Wizard state and progress
    - Artifacts produced
    - Assumptions made
    - Reviews completed
    - Tool execution history

    Example:
        >>> context = ConversationContext.from_manager(manager)
        >>> unconfirmed = context.get_unconfirmed_assumptions()
        >>> artifacts = context.get_artifacts(status="approved")
        >>> prompt_context = context.to_prompt_injection(max_tokens=2000)
    """

    # Core identification
    conversation_id: str | None = None

    # Wizard state
    wizard_stage: str | None = None
    wizard_data: dict[str, Any] = field(default_factory=dict)
    wizard_progress: float = 0.0
    wizard_tasks: list[dict[str, Any]] = field(default_factory=list)

    # Artifacts
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    # Assumptions and confirmations
    assumptions: list[Assumption] = field(default_factory=list)

    # Tool execution history
    tool_history: list[dict[str, Any]] = field(default_factory=list)

    # Transitions and navigation
    transitions: list[dict[str, Any]] = field(default_factory=list)

    # Custom sections
    sections: list[ContextSection] = field(default_factory=list)

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # =========================================================================
    # Assumption Management
    # =========================================================================

    def add_assumption(
        self,
        content: str,
        source: AssumptionSource = "inferred",
        confidence: float = 0.5,
        related_to: str | None = None,
    ) -> Assumption:
        """Add a new assumption to the context.

        Args:
            content: What is being assumed
            source: How this assumption was made
            confidence: Confidence level (0.0-1.0)
            related_to: What this relates to

        Returns:
            Created Assumption
        """
        assumption = Assumption(
            content=content,
            source=source,
            confidence=confidence,
            related_to=related_to,
        )
        self.assumptions.append(assumption)
        self.updated_at = time.time()
        return assumption

    def confirm_assumption(self, assumption_id: str) -> bool:
        """Mark an assumption as confirmed by the user.

        Args:
            assumption_id: ID of assumption to confirm

        Returns:
            True if found and confirmed
        """
        for assumption in self.assumptions:
            if assumption.id == assumption_id:
                assumption.confirmed = True
                assumption.confirmed_at = time.time()
                self.updated_at = time.time()
                return True
        return False

    def reject_assumption(self, assumption_id: str) -> bool:
        """Remove an assumption (user rejected it).

        Args:
            assumption_id: ID of assumption to remove

        Returns:
            True if found and removed
        """
        for i, assumption in enumerate(self.assumptions):
            if assumption.id == assumption_id:
                self.assumptions.pop(i)
                self.updated_at = time.time()
                return True
        return False

    def get_unconfirmed_assumptions(self) -> list[Assumption]:
        """Get assumptions that haven't been confirmed."""
        return [a for a in self.assumptions if not a.confirmed]

    def get_assumptions_for(self, related_to: str) -> list[Assumption]:
        """Get assumptions related to a specific field/topic.

        Args:
            related_to: Field or topic to filter by

        Returns:
            List of matching assumptions
        """
        return [a for a in self.assumptions if a.related_to == related_to]

    def get_low_confidence_assumptions(
        self,
        threshold: float = 0.6,
    ) -> list[Assumption]:
        """Get assumptions with confidence below threshold.

        Args:
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            List of low-confidence assumptions
        """
        return [
            a for a in self.assumptions
            if not a.confirmed and a.confidence < threshold
        ]

    # =========================================================================
    # Artifact Access
    # =========================================================================

    def get_artifacts(
        self,
        status: str | None = None,
        artifact_type: str | None = None,
        definition_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get artifacts with optional filters.

        Args:
            status: Filter by status (draft, approved, etc.)
            artifact_type: Filter by artifact type
            definition_id: Filter by definition

        Returns:
            List of matching artifact dicts
        """
        results = self.artifacts
        if status:
            results = [a for a in results if a.get("status") == status]
        if artifact_type:
            results = [a for a in results if a.get("type") == artifact_type]
        if definition_id:
            results = [a for a in results if a.get("definition_id") == definition_id]
        return results

    def get_artifact_reviews(self, artifact_id: str) -> list[dict[str, Any]]:
        """Get reviews for a specific artifact.

        Args:
            artifact_id: ID of artifact

        Returns:
            List of review dicts
        """
        for artifact in self.artifacts:
            if artifact.get("id") == artifact_id:
                return artifact.get("reviews", [])
        return []

    # =========================================================================
    # Context Sections
    # =========================================================================

    def add_section(
        self,
        name: str,
        content: Any,
        priority: int = 50,
        max_tokens: int | None = None,
        include_always: bool = False,
        formatter: str = "default",
    ) -> None:
        """Add a custom context section.

        If a section with the same name exists, it is replaced.

        Args:
            name: Section identifier
            content: Section content
            priority: Importance (higher = more important, 0-100)
            max_tokens: Token limit for this section
            include_always: Always include in prompts
            formatter: How to format content
        """
        # Remove existing section with same name
        self.sections = [s for s in self.sections if s.name != name]
        self.sections.append(ContextSection(
            name=name,
            content=content,
            priority=priority,
            max_tokens=max_tokens,
            include_always=include_always,
            formatter=formatter,
        ))
        self.updated_at = time.time()

    def get_section(self, name: str) -> ContextSection | None:
        """Get a section by name.

        Args:
            name: Section name to find

        Returns:
            ContextSection if found, None otherwise
        """
        for section in self.sections:
            if section.name == name:
                return section
        return None

    def remove_section(self, name: str) -> bool:
        """Remove a section by name.

        Args:
            name: Section name to remove

        Returns:
            True if found and removed
        """
        original_count = len(self.sections)
        self.sections = [s for s in self.sections if s.name != name]
        if len(self.sections) < original_count:
            self.updated_at = time.time()
            return True
        return False

    # =========================================================================
    # Prompt Injection
    # =========================================================================

    def to_prompt_injection(
        self,
        max_tokens: int = 2000,
        include_sections: list[str] | None = None,
        exclude_sections: list[str] | None = None,
    ) -> str:
        """Generate context string for prompt injection.

        Builds a formatted context string within token budget,
        prioritizing more important sections.

        Args:
            max_tokens: Maximum tokens for entire context
            include_sections: Only include these sections (None = all)
            exclude_sections: Exclude these sections

        Returns:
            Formatted context string
        """
        lines = ["## Conversation Context\n"]

        # Build sections in priority order
        all_sections = self._build_standard_sections() + self.sections
        all_sections.sort(key=lambda s: -s.priority)  # High priority first

        # Filter sections
        if include_sections:
            all_sections = [s for s in all_sections if s.name in include_sections]
        if exclude_sections:
            all_sections = [s for s in all_sections if s.name not in exclude_sections]

        # Track token usage (rough estimate: 4 chars per token)
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        current_chars = len(lines[0])

        for section in all_sections:
            section_text = self._format_section(section)
            section_chars = len(section_text)

            # Check if we have room
            if section.include_always or current_chars + section_chars <= max_chars:
                lines.append(section_text)
                current_chars += section_chars
            elif section.max_tokens:
                # Try to include truncated version
                max_section_chars = section.max_tokens * chars_per_token
                if current_chars + max_section_chars <= max_chars:
                    truncated = section_text[:max_section_chars] + "..."
                    lines.append(truncated)
                    current_chars += len(truncated)

        return "\n".join(lines)

    def _build_standard_sections(self) -> list[ContextSection]:
        """Build standard context sections from accumulated data."""
        sections: list[ContextSection] = []

        # Wizard progress (high priority)
        if self.wizard_stage:
            sections.append(ContextSection(
                name="wizard_progress",
                content={
                    "stage": self.wizard_stage,
                    "progress": f"{self.wizard_progress * 100:.0f}%",
                    "data_collected": list(self.wizard_data.keys()),
                },
                priority=90,
                include_always=True,
                formatter="summary",
            ))

        # Unconfirmed assumptions (high priority)
        unconfirmed = self.get_unconfirmed_assumptions()
        if unconfirmed:
            sections.append(ContextSection(
                name="unconfirmed_assumptions",
                content=[
                    {"assumption": a.content, "confidence": a.confidence}
                    for a in unconfirmed
                ],
                priority=85,
                formatter="list",
            ))

        # Pending artifacts
        pending = [
            a for a in self.artifacts
            if a.get("status") in ("draft", "pending_review", "needs_revision")
        ]
        if pending:
            sections.append(ContextSection(
                name="pending_artifacts",
                content=[
                    {"name": a.get("name"), "status": a.get("status")}
                    for a in pending
                ],
                priority=70,
                formatter="list",
            ))

        # Approved artifacts (lower priority)
        approved = self.get_artifacts(status="approved")
        if approved:
            sections.append(ContextSection(
                name="approved_artifacts",
                content=[a.get("name") for a in approved],
                priority=50,
                formatter="list",
            ))

        # Recent tool executions
        if self.tool_history:
            recent = self.tool_history[-5:]  # Last 5
            sections.append(ContextSection(
                name="recent_tools",
                content=[
                    {"tool": t.get("tool_name"), "success": t.get("success")}
                    for t in recent
                ],
                priority=40,
                formatter="list",
            ))

        return sections

    def _format_section(self, section: ContextSection) -> str:
        """Format a section for prompt inclusion.

        Args:
            section: Section to format

        Returns:
            Formatted section string
        """
        title = section.name.replace("_", " ").title()
        content = section.content

        if section.formatter == "json":
            content_str = json.dumps(content, indent=2)
        elif section.formatter == "list":
            if isinstance(content, list):
                items = []
                for item in content:
                    if isinstance(item, dict):
                        items.append(", ".join(f"{k}: {v}" for k, v in item.items()))
                    else:
                        items.append(str(item))
                content_str = "\n".join(f"- {item}" for item in items)
            else:
                content_str = str(content)
        elif section.formatter == "summary":
            if isinstance(content, dict):
                content_str = ", ".join(f"{k}: {v}" for k, v in content.items())
            else:
                content_str = str(content)
        else:  # default
            content_str = str(content) if not isinstance(content, str) else content

        return f"### {title}\n{content_str}\n"

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "conversation_id": self.conversation_id,
            "wizard_stage": self.wizard_stage,
            "wizard_data": self.wizard_data,
            "wizard_progress": self.wizard_progress,
            "wizard_tasks": self.wizard_tasks,
            "artifacts": self.artifacts,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "tool_history": self.tool_history,
            "transitions": self.transitions,
            "sections": [s.to_dict() for s in self.sections],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationContext:
        """Restore from dictionary.

        Args:
            data: Serialized context data

        Returns:
            ConversationContext instance
        """
        sections = [
            ContextSection.from_dict(s) for s in data.get("sections", [])
        ]
        assumptions = [
            Assumption.from_dict(a) for a in data.get("assumptions", [])
        ]
        return cls(
            conversation_id=data.get("conversation_id"),
            wizard_stage=data.get("wizard_stage"),
            wizard_data=data.get("wizard_data", {}),
            wizard_progress=data.get("wizard_progress", 0.0),
            wizard_tasks=data.get("wizard_tasks", []),
            artifacts=data.get("artifacts", []),
            assumptions=assumptions,
            tool_history=data.get("tool_history", []),
            transitions=data.get("transitions", []),
            sections=sections,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

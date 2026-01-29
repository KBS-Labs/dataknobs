"""Tests for context accumulator."""

import pytest

from dataknobs_bots.context.accumulator import (
    Assumption,
    ContextSection,
    ConversationContext,
)


class TestAssumption:
    """Tests for Assumption dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set."""
        assumption = Assumption()
        assert assumption.id.startswith("asn_")
        assert assumption.content == ""
        assert assumption.source == "inferred"
        assert assumption.confidence == 0.5
        assert assumption.confirmed is False
        assert assumption.confirmed_at is None
        assert assumption.created_at > 0

    def test_with_values(self) -> None:
        """Test assumption with values."""
        assumption = Assumption(
            content="User wants a math tutor",
            source="extracted",
            confidence=0.8,
            related_to="subject",
        )
        assert assumption.content == "User wants a math tutor"
        assert assumption.source == "extracted"
        assert assumption.confidence == 0.8
        assert assumption.related_to == "subject"

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = Assumption(
            content="Test assumption",
            source="user_stated",
            confidence=0.9,
            related_to="field",
        )
        data = original.to_dict()
        restored = Assumption.from_dict(data)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.source == original.source
        assert restored.confidence == original.confidence
        assert restored.related_to == original.related_to


class TestContextSection:
    """Tests for ContextSection dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set."""
        section = ContextSection(name="test", content="content")
        assert section.name == "test"
        assert section.content == "content"
        assert section.priority == 50
        assert section.max_tokens is None
        assert section.include_always is False
        assert section.formatter == "default"

    def test_with_values(self) -> None:
        """Test section with custom values."""
        section = ContextSection(
            name="high_priority",
            content={"key": "value"},
            priority=90,
            max_tokens=500,
            include_always=True,
            formatter="json",
        )
        assert section.priority == 90
        assert section.max_tokens == 500
        assert section.include_always is True
        assert section.formatter == "json"

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = ContextSection(
            name="test",
            content=["item1", "item2"],
            priority=75,
            formatter="list",
        )
        data = original.to_dict()
        restored = ContextSection.from_dict(data)

        assert restored.name == original.name
        assert restored.content == original.content
        assert restored.priority == original.priority
        assert restored.formatter == original.formatter


class TestConversationContextBasic:
    """Basic tests for ConversationContext."""

    def test_default_values(self) -> None:
        """Test default values are set."""
        context = ConversationContext()
        assert context.conversation_id is None
        assert context.wizard_stage is None
        assert context.wizard_data == {}
        assert context.wizard_progress == 0.0
        assert context.artifacts == []
        assert context.assumptions == []
        assert context.tool_history == []
        assert context.transitions == []
        assert context.sections == []

    def test_with_conversation_id(self) -> None:
        """Test context with conversation ID."""
        context = ConversationContext(conversation_id="conv_123")
        assert context.conversation_id == "conv_123"


class TestConversationContextAssumptions:
    """Tests for assumption management."""

    def test_add_assumption(self) -> None:
        """Test adding an assumption."""
        context = ConversationContext()
        assumption = context.add_assumption(
            content="User wants help with math",
            source="inferred",
            confidence=0.7,
            related_to="subject",
        )

        assert assumption.content == "User wants help with math"
        assert assumption.source == "inferred"
        assert assumption.confidence == 0.7
        assert len(context.assumptions) == 1

    def test_confirm_assumption(self) -> None:
        """Test confirming an assumption."""
        context = ConversationContext()
        assumption = context.add_assumption(
            content="Test assumption",
            confidence=0.6,
        )

        assert assumption.confirmed is False
        result = context.confirm_assumption(assumption.id)

        assert result is True
        assert assumption.confirmed is True
        assert assumption.confirmed_at is not None

    def test_confirm_assumption_not_found(self) -> None:
        """Test confirming non-existent assumption."""
        context = ConversationContext()
        result = context.confirm_assumption("nonexistent")
        assert result is False

    def test_reject_assumption(self) -> None:
        """Test rejecting an assumption."""
        context = ConversationContext()
        assumption = context.add_assumption(content="Test")

        result = context.reject_assumption(assumption.id)
        assert result is True
        assert len(context.assumptions) == 0

    def test_get_unconfirmed_assumptions(self) -> None:
        """Test getting unconfirmed assumptions."""
        context = ConversationContext()
        assumption1 = context.add_assumption(content="Unconfirmed 1")
        assumption2 = context.add_assumption(content="Confirmed")
        context.add_assumption(content="Unconfirmed 2")

        context.confirm_assumption(assumption2.id)

        unconfirmed = context.get_unconfirmed_assumptions()
        assert len(unconfirmed) == 2

    def test_get_assumptions_for(self) -> None:
        """Test getting assumptions for a specific field."""
        context = ConversationContext()
        context.add_assumption(content="Subject assumption", related_to="subject")
        context.add_assumption(content="Grade assumption", related_to="grade")
        context.add_assumption(content="Another subject", related_to="subject")

        subject_assumptions = context.get_assumptions_for("subject")
        assert len(subject_assumptions) == 2

        grade_assumptions = context.get_assumptions_for("grade")
        assert len(grade_assumptions) == 1

    def test_get_low_confidence_assumptions(self) -> None:
        """Test getting low confidence assumptions."""
        context = ConversationContext()
        context.add_assumption(content="High confidence", confidence=0.9)
        context.add_assumption(content="Low confidence", confidence=0.3)
        context.add_assumption(content="Medium confidence", confidence=0.5)

        low_confidence = context.get_low_confidence_assumptions(threshold=0.6)
        assert len(low_confidence) == 2


class TestConversationContextArtifacts:
    """Tests for artifact access."""

    def test_get_artifacts_no_filter(self) -> None:
        """Test getting all artifacts."""
        context = ConversationContext()
        context.artifacts = [
            {"id": "a1", "status": "draft", "type": "content"},
            {"id": "a2", "status": "approved", "type": "config"},
        ]

        artifacts = context.get_artifacts()
        assert len(artifacts) == 2

    def test_get_artifacts_by_status(self) -> None:
        """Test filtering artifacts by status."""
        context = ConversationContext()
        context.artifacts = [
            {"id": "a1", "status": "draft", "type": "content"},
            {"id": "a2", "status": "approved", "type": "content"},
            {"id": "a3", "status": "draft", "type": "config"},
        ]

        drafts = context.get_artifacts(status="draft")
        assert len(drafts) == 2

        approved = context.get_artifacts(status="approved")
        assert len(approved) == 1

    def test_get_artifacts_by_type(self) -> None:
        """Test filtering artifacts by type."""
        context = ConversationContext()
        context.artifacts = [
            {"id": "a1", "status": "draft", "type": "content"},
            {"id": "a2", "status": "approved", "type": "config"},
        ]

        content_artifacts = context.get_artifacts(artifact_type="content")
        assert len(content_artifacts) == 1
        assert content_artifacts[0]["id"] == "a1"

    def test_get_artifact_reviews(self) -> None:
        """Test getting reviews for an artifact."""
        context = ConversationContext()
        context.artifacts = [
            {
                "id": "a1",
                "reviews": [
                    {"reviewer": "adversarial", "passed": True},
                    {"reviewer": "skeptical", "passed": False},
                ]
            }
        ]

        reviews = context.get_artifact_reviews("a1")
        assert len(reviews) == 2

        no_reviews = context.get_artifact_reviews("nonexistent")
        assert len(no_reviews) == 0


class TestConversationContextSections:
    """Tests for custom sections."""

    def test_add_section(self) -> None:
        """Test adding a section."""
        context = ConversationContext()
        context.add_section(
            name="custom",
            content={"key": "value"},
            priority=80,
            formatter="json",
        )

        assert len(context.sections) == 1
        assert context.sections[0].name == "custom"
        assert context.sections[0].priority == 80

    def test_add_section_replaces_existing(self) -> None:
        """Test that adding section with same name replaces it."""
        context = ConversationContext()
        context.add_section(name="test", content="old")
        context.add_section(name="test", content="new")

        assert len(context.sections) == 1
        assert context.sections[0].content == "new"

    def test_get_section(self) -> None:
        """Test getting a section by name."""
        context = ConversationContext()
        context.add_section(name="test", content="content")

        section = context.get_section("test")
        assert section is not None
        assert section.content == "content"

        not_found = context.get_section("nonexistent")
        assert not_found is None

    def test_remove_section(self) -> None:
        """Test removing a section."""
        context = ConversationContext()
        context.add_section(name="test", content="content")

        result = context.remove_section("test")
        assert result is True
        assert len(context.sections) == 0

        result = context.remove_section("nonexistent")
        assert result is False


class TestConversationContextPromptInjection:
    """Tests for prompt injection generation."""

    def test_basic_prompt_injection(self) -> None:
        """Test basic prompt injection generation."""
        context = ConversationContext(
            wizard_stage="collect_info",
            wizard_progress=0.5,
        )
        context.wizard_data = {"name": "test"}

        prompt = context.to_prompt_injection(max_tokens=500)

        assert "## Conversation Context" in prompt
        assert "Wizard Progress" in prompt
        assert "collect_info" in prompt

    def test_prompt_injection_with_assumptions(self) -> None:
        """Test prompt injection includes unconfirmed assumptions."""
        context = ConversationContext()
        context.add_assumption(content="Test assumption", confidence=0.6)

        prompt = context.to_prompt_injection(max_tokens=500)

        assert "Unconfirmed Assumptions" in prompt
        assert "Test assumption" in prompt

    def test_prompt_injection_with_artifacts(self) -> None:
        """Test prompt injection includes pending artifacts."""
        context = ConversationContext()
        context.artifacts = [
            {"name": "Questions", "status": "pending_review"},
        ]

        prompt = context.to_prompt_injection(max_tokens=500)

        assert "Pending Artifacts" in prompt
        assert "Questions" in prompt

    def test_prompt_injection_token_limit(self) -> None:
        """Test that prompt injection respects token limit."""
        context = ConversationContext()
        # Add lots of assumptions
        for i in range(50):
            context.add_assumption(content=f"Assumption number {i}" * 10)

        # Very low token limit
        prompt = context.to_prompt_injection(max_tokens=100)

        # Should be truncated
        assert len(prompt) < 500  # Rough check (100 tokens * 4 chars)

    def test_prompt_injection_include_sections(self) -> None:
        """Test filtering sections by include list."""
        context = ConversationContext()
        context.wizard_stage = "test"
        context.add_assumption(content="Test")

        # Only include wizard progress
        prompt = context.to_prompt_injection(
            include_sections=["wizard_progress"]
        )

        assert "Wizard Progress" in prompt
        assert "Unconfirmed Assumptions" not in prompt

    def test_prompt_injection_exclude_sections(self) -> None:
        """Test filtering sections by exclude list."""
        context = ConversationContext()
        context.wizard_stage = "test"
        context.add_assumption(content="Test")

        # Exclude wizard progress
        prompt = context.to_prompt_injection(
            exclude_sections=["wizard_progress"]
        )

        assert "Wizard Progress" not in prompt
        assert "Unconfirmed Assumptions" in prompt

    def test_format_section_json(self) -> None:
        """Test JSON formatting of sections."""
        context = ConversationContext()
        context.add_section(
            name="test",
            content={"key": "value"},
            formatter="json",
        )

        prompt = context.to_prompt_injection(max_tokens=500)
        assert '"key"' in prompt
        assert '"value"' in prompt

    def test_format_section_list(self) -> None:
        """Test list formatting of sections."""
        context = ConversationContext()
        context.add_section(
            name="test",
            content=["item1", "item2"],
            formatter="list",
        )

        prompt = context.to_prompt_injection(max_tokens=500)
        assert "- item1" in prompt
        assert "- item2" in prompt


class TestConversationContextSerialization:
    """Tests for context serialization."""

    def test_to_dict(self) -> None:
        """Test serializing context to dict."""
        context = ConversationContext(
            conversation_id="conv_123",
            wizard_stage="test_stage",
            wizard_progress=0.5,
        )
        context.wizard_data = {"key": "value"}
        context.add_assumption(content="Test assumption")
        context.add_section(name="custom", content="data")

        data = context.to_dict()

        assert data["conversation_id"] == "conv_123"
        assert data["wizard_stage"] == "test_stage"
        assert data["wizard_progress"] == 0.5
        assert len(data["assumptions"]) == 1
        assert len(data["sections"]) == 1

    def test_from_dict(self) -> None:
        """Test restoring context from dict."""
        data = {
            "conversation_id": "conv_123",
            "wizard_stage": "test_stage",
            "wizard_data": {"key": "value"},
            "wizard_progress": 0.75,
            "wizard_tasks": [{"id": "t1", "status": "pending"}],
            "artifacts": [{"id": "a1", "status": "draft"}],
            "assumptions": [
                {"id": "asn_1", "content": "Test", "source": "inferred", "confidence": 0.8}
            ],
            "tool_history": [{"tool_name": "test", "success": True}],
            "transitions": [{"from": "a", "to": "b"}],
            "sections": [{"name": "custom", "content": "data", "priority": 60}],
        }

        context = ConversationContext.from_dict(data)

        assert context.conversation_id == "conv_123"
        assert context.wizard_stage == "test_stage"
        assert context.wizard_progress == 0.75
        assert len(context.artifacts) == 1
        assert len(context.assumptions) == 1
        assert context.assumptions[0].content == "Test"
        assert len(context.sections) == 1
        assert context.sections[0].name == "custom"

    def test_round_trip_serialization(self) -> None:
        """Test full serialization round trip."""
        original = ConversationContext(
            conversation_id="conv_123",
            wizard_stage="test",
        )
        original.add_assumption(content="Test", confidence=0.8)
        original.add_section(name="custom", content="data", priority=80)

        data = original.to_dict()
        restored = ConversationContext.from_dict(data)

        assert restored.conversation_id == original.conversation_id
        assert restored.wizard_stage == original.wizard_stage
        assert len(restored.assumptions) == len(original.assumptions)
        assert len(restored.sections) == len(original.sections)

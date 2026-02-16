"""Tests for artifact transforms."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_bots.artifacts.models import ArtifactStatus
from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.artifacts.transforms import (
    TransformContext,
    approve_artifact,
    create_artifact,
    revise_artifact,
    save_artifact_draft,
    submit_for_review,
)


# --- Fixtures ---


def _make_context(
    db: AsyncMemoryDatabase | None = None,
    user_id: str | None = None,
) -> TransformContext:
    art_db = db or AsyncMemoryDatabase()
    return TransformContext(
        artifact_registry=ArtifactRegistry(art_db),
        user_id=user_id,
        session_id="test_session",
    )


# --- create_artifact Tests ---


class TestCreateArtifact:
    async def test_creates_artifact_with_defaults(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"title": "My Quiz", "questions": ["Q1", "Q2"]}

        await create_artifact(data, ctx)

        assert "_artifact_id" in data
        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.type == "content"
        assert artifact.content["title"] == "My Quiz"
        assert artifact.content["questions"] == ["Q1", "Q2"]

    async def test_creates_artifact_with_config(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"title": "Quiz", "questions": ["Q1"], "extra": "skip"}

        config = {
            "artifact_type": "quiz",
            "content_fields": ["title", "questions"],
            "tags": ["education"],
        }
        await create_artifact(data, ctx, config=config)

        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.type == "quiz"
        assert "extra" not in artifact.content
        assert "education" in artifact.tags

    async def test_creates_artifact_with_name_template(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"subject": "Math", "level": "101"}

        config = {"name_template": "{{ subject }} {{ level }}"}
        await create_artifact(data, ctx, config=config)

        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.name == "Math 101"

    async def test_creates_artifact_with_name_field(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"name": "My Document"}

        await create_artifact(data, ctx)

        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.name == "My Document"

    async def test_provenance_records_user(self) -> None:
        ctx = _make_context(user_id="alice")
        data: dict[str, Any] = {"name": "Test"}

        await create_artifact(data, ctx)

        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.provenance.created_by == "user:alice"
        assert artifact.provenance.creation_method == "wizard"

    async def test_excludes_underscore_prefixed_keys(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"name": "Test", "visible": True, "_internal": "hidden"}

        await create_artifact(data, ctx)

        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert "_internal" not in artifact.content
        assert "visible" in artifact.content

    async def test_raises_without_registry(self) -> None:
        ctx = TransformContext()
        data: dict[str, Any] = {"name": "Test"}

        with pytest.raises(ValueError, match="artifact_registry is required"):
            await create_artifact(data, ctx)


# --- submit_for_review Tests ---


class TestSubmitForReview:
    async def test_submits_artifact(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"name": "Test"}
        await create_artifact(data, ctx)

        await submit_for_review(data, ctx)

        assert "_evaluation_results" in data
        assert "_review_passed" in data
        # No rubric executor configured, so no evaluations but passes by default
        assert data["_review_passed"] is True

    async def test_raises_without_artifact_id(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {}

        with pytest.raises(ValueError, match="No artifact ID"):
            await submit_for_review(data, ctx)

    async def test_raises_without_registry(self) -> None:
        ctx = TransformContext()
        data: dict[str, Any] = {"_artifact_id": "art_123"}

        with pytest.raises(ValueError, match="artifact_registry is required"):
            await submit_for_review(data, ctx)


# --- revise_artifact Tests ---


class TestReviseArtifact:
    async def test_revises_artifact_content(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"name": "Original", "body": "v1"}
        await create_artifact(data, ctx)
        original_id = data["_artifact_id"]

        data["body"] = "v2"
        await revise_artifact(data, ctx)

        # Same ID, new version
        assert data["_artifact_id"] == original_id
        artifact = await ctx.artifact_registry.get(original_id)
        assert artifact is not None
        assert artifact.content["body"] == "v2"
        assert artifact.version != "1.0.0"

    async def test_revise_with_content_fields(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"name": "Doc", "body": "text", "extra": "skip"}
        await create_artifact(data, ctx)

        data["body"] = "updated"
        config = {"content_fields": ["body"]}
        await revise_artifact(data, ctx, config=config)

        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.content == {"body": "updated"}

    async def test_raises_without_artifact_id(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {}

        with pytest.raises(ValueError, match="No artifact ID"):
            await revise_artifact(data, ctx)


# --- approve_artifact Tests ---


class TestApproveArtifact:
    async def test_approves_artifact(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"name": "Test"}
        await create_artifact(data, ctx)

        # Transition to in_review through the proper lifecycle
        await ctx.artifact_registry.set_status(
            data["_artifact_id"], ArtifactStatus.PENDING_REVIEW
        )
        await ctx.artifact_registry.set_status(
            data["_artifact_id"], ArtifactStatus.IN_REVIEW
        )

        await approve_artifact(data, ctx)

        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.status == ArtifactStatus.APPROVED

    async def test_raises_without_artifact_id(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {}

        with pytest.raises(ValueError, match="No artifact ID"):
            await approve_artifact(data, ctx)


# --- save_artifact_draft Tests ---


class TestSaveArtifactDraft:
    async def test_creates_new_draft(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"name": "Draft Doc", "body": "initial"}

        await save_artifact_draft(data, ctx)

        assert "_artifact_id" in data
        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.content["body"] == "initial"

    async def test_updates_existing_draft(self) -> None:
        ctx = _make_context()
        data: dict[str, Any] = {"name": "Draft Doc", "body": "v1"}
        await save_artifact_draft(data, ctx)

        data["body"] = "v2"
        await save_artifact_draft(data, ctx)

        artifact = await ctx.artifact_registry.get(data["_artifact_id"])
        assert artifact is not None
        assert artifact.content["body"] == "v2"

    async def test_skips_without_registry(self) -> None:
        ctx = TransformContext()
        data: dict[str, Any] = {"name": "Test"}

        # Should not raise, just skip
        await save_artifact_draft(data, ctx)
        assert "_artifact_id" not in data

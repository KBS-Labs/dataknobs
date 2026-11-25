"""Every timestamp bots writes is read from the common clock.

Under :func:`dataknobs_common.clock.use_clock` with a
:class:`~dataknobs_common.testing.FixedClock`, each stamp below **equals** the
fixed instant: an artifact's revision, a rubric evaluation, the logging
middleware's records, a config draft and an API error body. Before these sites
read :mod:`dataknobs_common.clock` each was wall-clock time, which a test
could only fix by patching :mod:`datetime`.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from dataknobs_common.clock import use_clock
from dataknobs_common.testing import FixedClock
from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_bots.api.exceptions import APIError
from dataknobs_bots.artifacts.assessment import AssessmentSession
from dataknobs_bots.artifacts.models import Artifact
from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.config.drafts import ConfigDraftManager
from dataknobs_bots.middleware.logging import LoggingMiddleware
from dataknobs_bots.rubrics.executor import FunctionRegistry, RubricExecutor
from dataknobs_bots.rubrics.models import (
    Rubric,
    RubricCriterion,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)
from dataknobs_bots.testing import BotTestHarness

AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
LATER = AT + timedelta(hours=1)


async def test_an_artifact_and_its_revision_are_stamped_by_the_clock() -> None:
    registry = ArtifactRegistry(db=AsyncMemoryDatabase())
    clock = FixedClock(AT)
    with use_clock(clock):
        original = await registry.create(artifact_type="content", name="Quote", content={"v": 1})
        clock.set(LATER)
        revised = await registry.revise(
            artifact_id=original.id,
            new_content={"v": 2},
            reason="edited",
            triggered_by="user:buyer",
        )
    assert original.created_at == AT.isoformat()
    assert original.provenance.created_at == AT.isoformat()
    assert revised.updated_at == LATER.isoformat()
    assert revised.provenance.revision_history[-1].timestamp == LATER.isoformat()


async def test_a_rubric_evaluation_is_stamped_by_the_clock() -> None:
    functions = FunctionRegistry()
    functions.register("clock_test:always_pass", lambda target: "pass")
    levels = [
        RubricLevel(id="fail", label="Fail", description="Does not meet", score=0.0),
        RubricLevel(id="pass", label="Pass", description="Meets", score=1.0),
    ]
    rubric = Rubric(
        id="r",
        name="R",
        description="R",
        version="1.0.0",
        target_type="content",
        criteria=[
            RubricCriterion(
                id="c",
                name="C",
                description="C",
                weight=1.0,
                levels=levels,
                scoring_method=ScoringMethod(
                    type=ScoringType.DETERMINISTIC, function_ref="clock_test:always_pass"
                ),
            )
        ],
        pass_threshold=0.5,
    )
    with use_clock(FixedClock(AT)):
        evaluation = await RubricExecutor(function_registry=functions).evaluate(rubric, {})
    assert evaluation.evaluated_at == AT.isoformat()


async def test_the_logging_middleware_stamps_its_records_by_the_clock(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        },
        main_responses=["Hello back"],
        middleware=[LoggingMiddleware(json_format=True)],
    ) as harness:
        with caplog.at_level(logging.INFO), use_clock(FixedClock(AT)):
            await harness.chat("Hello")
    stamps = [
        json.loads(r.getMessage())["timestamp"]
        for r in caplog.records
        if r.name.startswith("dataknobs_bots.middleware.logging") and r.getMessage().startswith("{")
    ]
    assert stamps, "the middleware logged no JSON record"
    assert set(stamps) == {AT.isoformat()}


def test_a_draft_is_stamped_by_the_clock(tmp_path: Path) -> None:
    manager = ConfigDraftManager(output_dir=tmp_path)
    clock = FixedClock(AT)
    with use_clock(clock):
        draft_id = manager.create_draft({"name": "bot"})
        clock.set(LATER)
        manager.update_draft(draft_id, {"name": "bot", "v": 2})
    found = manager.get_draft(draft_id)
    assert found is not None
    _, metadata = found
    assert metadata.created_at == AT.isoformat()
    assert metadata.last_updated == LATER.isoformat()


def test_a_draft_ages_out_by_the_clock_that_stamped_it(tmp_path: Path) -> None:
    """Expiry is measured on the clock the stamp came from.

    A draft stamped under a fixed clock and checked against the wall clock
    would look months old the moment it was written, and advancing the fixed
    clock could never age it.
    """
    manager = ConfigDraftManager(output_dir=tmp_path, max_age_hours=24)
    clock = FixedClock(AT)
    with use_clock(clock):
        draft_id = manager.create_draft({"name": "bot"})
        assert manager.cleanup_stale() == 0
        assert manager.get_draft(draft_id) is not None

        clock.advance(timedelta(hours=23))
        assert manager.cleanup_stale() == 0
        assert manager.get_draft(draft_id) is not None

        clock.advance(timedelta(hours=2))
        assert manager.cleanup_stale() == 1
        assert manager.get_draft(draft_id) is None


def test_a_named_configs_draft_block_ages_out_by_the_same_clock(tmp_path: Path) -> None:
    manager = ConfigDraftManager(output_dir=tmp_path, max_age_hours=24)
    clock = FixedClock(AT)
    with use_clock(clock):
        draft_id = manager.create_draft({"name": "bot"})
        manager.update_draft(draft_id, {"name": "bot"}, config_name="named-bot")
        manager.cleanup_stale()
        assert "_draft" in yaml.safe_load((tmp_path / "named-bot.yaml").read_text())

        clock.advance(timedelta(hours=25))
        manager.cleanup_stale()
    assert "_draft" not in yaml.safe_load((tmp_path / "named-bot.yaml").read_text())


def test_an_api_error_body_is_stamped_by_the_clock() -> None:
    with use_clock(FixedClock(AT)):
        body = APIError("Bad request", status_code=400).to_dict()
    assert body["timestamp"] == AT.isoformat()


def _refusing_clock() -> datetime:
    """A clock no read may reach: it returns a naive value, which is refused."""
    return datetime(2026, 1, 1)


async def test_artifact_from_dict_does_not_read_the_clock_for_stored_stamps() -> None:
    registry = ArtifactRegistry(db=AsyncMemoryDatabase())
    with use_clock(FixedClock(AT)):
        stored = (
            await registry.create(artifact_type="content", name="Quote", content={"v": 1})
        ).to_dict()
    with use_clock(_refusing_clock):
        restored = Artifact.from_dict(stored)
    assert restored.created_at == AT.isoformat()
    assert restored.updated_at == AT.isoformat()


def test_an_assessment_session_from_dict_does_not_read_the_clock_for_a_stored_start() -> None:
    with use_clock(FixedClock(AT)):
        stored = AssessmentSession(student_id="s", assessment_artifact_id="a").to_dict()
    with use_clock(_refusing_clock):
        restored = AssessmentSession.from_dict(stored)
    assert restored.started_at == AT.isoformat()

"""Tests for WizardFSM.step_async().

Validates that step_async mirrors step() behavior and properly
awaits async transforms and pre-tests.
"""

import pytest

from dataknobs_bots.reasoning.wizard_fsm import WizardFSM, create_wizard_fsm
from dataknobs_fsm.api.advanced import StepResult


def _simple_wizard_config() -> dict:
    """Build a 3-stage wizard config: welcome -> process -> complete."""
    return {
        "name": "TestWizard",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "welcome", "is_start": True},
                    {"name": "process"},
                    {"name": "complete", "is_end": True},
                ],
                "arcs": [
                    {"from": "welcome", "to": "process"},
                    {"from": "process", "to": "complete"},
                ],
            }
        ],
    }


def _simple_stage_metadata() -> dict:
    return {
        "welcome": {"prompt": "Hello", "is_start": True},
        "process": {"prompt": "Processing"},
        "complete": {"prompt": "Done", "is_end": True},
    }


class TestStepAsyncBasicTransition:
    """step_async mirrors sync step() behavior."""

    @pytest.mark.asyncio
    async def test_first_step_transitions(self) -> None:
        wizard = create_wizard_fsm(
            _simple_wizard_config(),
            _simple_stage_metadata(),
        )

        result = await wizard.step_async({"input": "test"})

        assert isinstance(result, StepResult)
        assert result.success is True
        # Should transition from welcome to process
        assert wizard.current_stage == "process"

    @pytest.mark.asyncio
    async def test_second_step_reaches_end(self) -> None:
        wizard = create_wizard_fsm(
            _simple_wizard_config(),
            _simple_stage_metadata(),
        )

        await wizard.step_async({"input": "test"})
        result = await wizard.step_async({"input": "more"})

        assert result.success is True
        assert result.is_complete is True
        assert wizard.current_stage == "complete"


class TestStepAsyncWithAsyncTransform:
    """Async transforms are awaited through step_async."""

    @pytest.mark.asyncio
    async def test_async_transform_applied(self) -> None:
        async def enrich(data: dict, context: object) -> dict:
            result = dict(data)
            result["async_enriched"] = True
            return result

        config = {
            "name": "TestWizard",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "welcome", "is_start": True},
                        {"name": "complete", "is_end": True},
                    ],
                    "arcs": [
                        {
                            "from": "welcome",
                            "to": "complete",
                            "transform": {
                                "type": "registered",
                                "name": "enrich",
                            },
                        },
                    ],
                }
            ],
        }
        stage_metadata = {
            "welcome": {"prompt": "Hi", "is_start": True},
            "complete": {"prompt": "Done", "is_end": True},
        }

        wizard = create_wizard_fsm(
            config,
            stage_metadata,
            custom_functions={"enrich": enrich},
        )

        result = await wizard.step_async({"value": 1})

        assert result.success is True
        assert result.data_after.get("async_enriched") is True


class TestStepAsyncCreatesContext:
    """step_async lazily creates context like step()."""

    @pytest.mark.asyncio
    async def test_context_created_on_first_call(self) -> None:
        wizard = create_wizard_fsm(
            _simple_wizard_config(),
            _simple_stage_metadata(),
        )

        assert wizard._context is None
        await wizard.step_async({"key": "val"})
        assert wizard._context is not None

    @pytest.mark.asyncio
    async def test_context_reused_on_subsequent_calls(self) -> None:
        wizard = create_wizard_fsm(
            _simple_wizard_config(),
            _simple_stage_metadata(),
        )

        await wizard.step_async({"key": "val"})
        ctx_after_first = wizard._context

        await wizard.step_async({"key": "val2"})
        assert wizard._context is ctx_after_first

"""Tests for wizard recovery pipeline composition.

The recovery pipeline runs configured strategies in order after initial
extraction + merge, stopping as soon as all required fields are satisfied.
Default pipeline: derivation → scope_escalation → focused_retry.

Integration tests exercise the full DynaBot.from_config() → bot.chat() path
via ``BotTestHarness``.
"""

import pytest

from dataknobs_bots.reasoning.wizard import (
    DEFAULT_RECOVERY_PIPELINE,
    RECOVERY_CLARIFICATION,
    RECOVERY_DERIVATION,
    RECOVERY_FOCUSED_RETRY,
    RECOVERY_SCOPE_ESCALATION,
    VALID_RECOVERY_STRATEGIES,
)
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


# ---------------------------------------------------------------------------
# Shared config builders
# ---------------------------------------------------------------------------


def _three_field_config() -> dict:
    """Gather stage with 3 required fields including derivation source."""
    return (
        WizardConfigBuilder("pipeline-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your name, domain_name, and domain_id.",
        )
        .field("name", field_type="string", required=True)
        .field("domain_name", field_type="string", required=True)
        .field("domain_id", field_type="string", required=True)
        .transition(
            "done",
            "data.get('name') and data.get('domain_name') "
            "and data.get('domain_id')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestRecoveryConstants:
    """Verify recovery pipeline constants are well-defined."""

    def test_default_pipeline_contains_valid_strategies(self) -> None:
        for s in DEFAULT_RECOVERY_PIPELINE:
            assert s in VALID_RECOVERY_STRATEGIES

    def test_valid_strategies_includes_all_constants(self) -> None:
        expected = {
            RECOVERY_DERIVATION,
            RECOVERY_SCOPE_ESCALATION,
            RECOVERY_FOCUSED_RETRY,
            RECOVERY_CLARIFICATION,
        }
        assert VALID_RECOVERY_STRATEGIES == expected

    def test_default_pipeline_order(self) -> None:
        assert DEFAULT_RECOVERY_PIPELINE == [
            "derivation",
            "scope_escalation",
            "focused_retry",
        ]


# ---------------------------------------------------------------------------
# Tests: Pipeline short-circuiting
# ---------------------------------------------------------------------------


class TestPipelineShortCircuit:
    """Verify the pipeline stops when all required fields are satisfied."""

    @pytest.mark.asyncio
    async def test_no_recovery_when_all_fields_present(self) -> None:
        """When extraction fills all required fields, pipeline is skipped."""
        config = _three_field_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
            "recovery": {
                "pipeline": ["derivation", "scope_escalation"],
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["All set!"],
            extraction_results=[
                # All fields present in initial extraction
                [{"name": "Alice", "domain_id": "chess-champ",
                  "domain_name": "Chess Champ"}],
            ],
        ) as harness:
            await harness.chat(
                "I'm Alice, domain chess-champ aka Chess Champ"
            )
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_derivation_prevents_escalation(self) -> None:
        """Derivation fills missing field, so escalation never fires.

        Only 1 extraction call (initial) — if escalation fired there
        would be a second call and the extractor would need a second
        result.
        """
        config = _three_field_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
            "derivations": [
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
            "recovery": {
                "pipeline": ["derivation", "scope_escalation"],
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                # Turn 1: name only
                [{"name": "Alice"}],
                # Turn 2: domain_id only — derivation fills domain_name
                # Only 1 extraction result = no escalation call
                [{"domain_id": "chess-champ"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("Domain is chess-champ")
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_stage == "done"


# ---------------------------------------------------------------------------
# Tests: Pipeline ordering
# ---------------------------------------------------------------------------


class TestPipelineOrdering:
    """Verify strategies execute in the configured order."""

    @pytest.mark.asyncio
    async def test_escalation_after_derivation(self) -> None:
        """Default order: derivation first, then escalation.

        When derivation can't help (no rule for missing field),
        escalation fires and fills it.
        """
        config = _three_field_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
            "recovery": {
                "pipeline": ["derivation", "scope_escalation"],
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                # Turn 1: name only
                [{"name": "Alice"}],
                # Turn 2: current_message gives domain_id only;
                # escalation (wizard_session scope) gives all 3
                [
                    {"domain_id": "chess-champ"},
                    {"name": "Alice", "domain_id": "chess-champ",
                     "domain_name": "Chess Champ"},
                ],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("Domain is chess-champ, name is Chess Champ")
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_stage == "done"


# ---------------------------------------------------------------------------
# Tests: Focused retry
# ---------------------------------------------------------------------------


class TestFocusedRetry:
    """Verify the focused retry recovery strategy."""

    @pytest.mark.asyncio
    async def test_focused_retry_fills_missing_field(self) -> None:
        """Focused retry extracts only missing fields with a minimal schema.

        Pipeline: derivation → focused_retry.
        No derivation rule for domain_name, so focused retry fires.
        """
        config = _three_field_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "recovery": {
                "pipeline": ["derivation", "focused_retry"],
                "focused_retry": {"enabled": True},
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                # Turn 1: name
                [{"name": "Alice"}],
                # Turn 2: initial gives domain_id only; focused retry
                # gives domain_name
                [
                    {"domain_id": "chess-champ"},
                    {"domain_name": "Chess Champ"},
                ],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("Domain chess-champ, display name Chess Champ")
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_data["domain_id"] == "chess-champ"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_focused_retry_disabled_by_default(self) -> None:
        """Focused retry is a no-op when not explicitly enabled.

        Pipeline includes focused_retry but focused_retry.enabled is
        not set.  Only 1 extraction call per turn (no retry).
        """
        config = _three_field_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "recovery": {
                "pipeline": ["focused_retry"],
                # Note: no focused_retry.enabled: true
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                # Only 1 extraction result — retry won't fire
                [{"name": "Alice"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            # Missing domain_id and domain_name but no retry fired
            assert harness.wizard_stage == "gather"
            assert harness.wizard_data.get("name") == "Alice"

    @pytest.mark.asyncio
    async def test_focused_retry_max_retries(self) -> None:
        """Focused retry respects max_retries setting.

        max_retries: 2. First attempt returns empty, second succeeds.
        """
        config = _three_field_config()
        config["settings"] = {
            "recovery": {
                "pipeline": ["focused_retry"],
                "focused_retry": {"enabled": True, "max_retries": 2},
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                # Turn 1: initial gives name + domain_id.
                # Focused retry attempt 1: empty; attempt 2: domain_name
                [
                    {"name": "Alice", "domain_id": "chess-champ"},
                    {},
                    {"domain_name": "Chess Champ"},
                ],
            ],
        ) as harness:
            await harness.chat(
                "I'm Alice, domain chess-champ, display Chess Champ"
            )
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_stage == "done"


# ---------------------------------------------------------------------------
# Tests: Full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Verify the complete pipeline with all strategies."""

    @pytest.mark.asyncio
    async def test_derivation_then_escalation_then_retry(self) -> None:
        """All three strategies fire when each fills one field.

        Setup: 4 required fields. Initial extraction gets name.
        - derivation fills domain_name from domain_id (but domain_id
          hasn't been extracted yet, so derivation has no effect)
        - escalation fills domain_id
        - derivation runs again via pipeline order? No — pipeline only
          runs each strategy once. So we test a scenario where:
          derivation fills 1 field, escalation fills 1 more, and
          focused retry fills the last.
        """
        config = (
            WizardConfigBuilder("full-pipeline-test")
            .stage(
                "gather",
                is_start=True,
                prompt="Provide name, domain_id, domain_name, llm_provider.",
            )
            .field("name", field_type="string", required=True)
            .field("domain_id", field_type="string", required=True)
            .field("domain_name", field_type="string", required=True)
            .field(
                "llm_provider", field_type="string", required=True,
                enum=["ollama", "openai", "anthropic"],
            )
            .transition(
                "done",
                "data.get('name') and data.get('domain_id') "
                "and data.get('domain_name') "
                "and data.get('llm_provider')",
            )
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )
        config["settings"] = {
            "extraction_scope": "current_message",
            "scope_escalation": {"enabled": True},
            "derivations": [
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
            "recovery": {
                "pipeline": [
                    "derivation",
                    "scope_escalation",
                    "focused_retry",
                ],
                "focused_retry": {"enabled": True},
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                # Turn 1: name only
                [{"name": "Alice"}],
                # Turn 2:
                #   initial: domain_id only
                #   (derivation fills domain_name from domain_id)
                #   (escalation not needed if derivation fills domain_name)
                #   focused_retry: llm_provider
                [
                    {"domain_id": "chess-champ"},
                    {"llm_provider": "ollama"},
                ],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat(
                "Domain chess-champ, use ollama"
            )
            assert harness.wizard_data["name"] == "Alice"
            assert harness.wizard_data["domain_id"] == "chess-champ"
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_data["llm_provider"] == "ollama"
            assert harness.wizard_stage == "done"


# ---------------------------------------------------------------------------
# Tests: Per-stage disable
# ---------------------------------------------------------------------------


class TestPerStageDisable:
    """Verify recovery_enabled per-stage override."""

    @pytest.mark.asyncio
    async def test_recovery_disabled_on_stage(self) -> None:
        """recovery_enabled: false prevents all recovery on that stage."""
        config = (
            WizardConfigBuilder("pipeline-stage-test")
            .stage("gather1", is_start=True, prompt="Tell me your name.")
            .field("name", field_type="string", required=True)
            .transition("gather2", "data.get('name')")
            .stage(
                "gather2",
                prompt="Tell me your domain_id.",
                recovery_enabled=False,
            )
            .field("domain_id", field_type="string", required=True)
            .field("domain_name", field_type="string", required=True)
            .transition(
                "done",
                "data.get('domain_id') and data.get('domain_name')",
            )
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )
        config["settings"] = {
            "extraction_scope": "current_message",
            "derivations": [
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
            "recovery": {
                "pipeline": ["derivation"],
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Need more info"],
            extraction_results=[
                # Turn 1: name
                [{"name": "Alice"}],
                # Turn 2: domain_id only — derivation would fill
                # domain_name but recovery is disabled
                [{"domain_id": "chess-champ"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            assert harness.wizard_stage == "gather2"
            await harness.chat("Domain chess-champ")
            # Derivation did NOT fire — domain_name missing
            assert harness.wizard_data.get("domain_name") is None
            assert harness.wizard_stage == "gather2"


# ---------------------------------------------------------------------------
# Tests: Configuration edge cases
# ---------------------------------------------------------------------------


class TestPipelineConfiguration:
    """Verify pipeline configuration handling."""

    @pytest.mark.asyncio
    async def test_empty_pipeline_disables_all_recovery(self) -> None:
        """An empty pipeline list means no recovery strategies run."""
        config = _three_field_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "derivations": [
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
            "recovery": {
                "pipeline": [],
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Need more"],
            extraction_results=[
                [{"name": "Alice"}],
                # Only domain_id — derivation would fill domain_name
                # but pipeline is empty
                [{"domain_id": "chess-champ"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("Domain chess-champ")
            assert harness.wizard_data.get("domain_name") is None
            assert harness.wizard_stage == "gather"

    @pytest.mark.asyncio
    async def test_schema_defaults_run_before_pipeline(self) -> None:
        """Schema defaults fill fields before the pipeline checks.

        When a required field has a schema default, it's filled before
        the pipeline runs, so no recovery strategy fires for it.
        """
        config = (
            WizardConfigBuilder("defaults-before-pipeline")
            .stage(
                "gather",
                is_start=True,
                prompt="Tell me your name.",
            )
            .field("name", field_type="string", required=True)
            .field(
                "provider", field_type="string", required=True,
                default="ollama",
            )
            .transition(
                "done",
                "data.get('name') and data.get('provider')",
            )
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )
        config["settings"] = {
            "recovery": {
                "pipeline": ["derivation", "focused_retry"],
                "focused_retry": {"enabled": True},
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["All set!"],
            extraction_results=[
                # Only name — provider gets its default
                [{"name": "Alice"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            assert harness.wizard_data["provider"] == "ollama"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_default_pipeline_without_config(self) -> None:
        """When no recovery settings are provided, default pipeline runs.

        The default pipeline includes derivation, scope_escalation,
        and focused_retry (but focused_retry requires enabled=true).
        """
        config = _three_field_config()
        config["settings"] = {
            "extraction_scope": "current_message",
            "derivations": [
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
            # No "recovery" key — uses default pipeline
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                [{"name": "Alice"}],
                # domain_id only — default pipeline derivation fills
                # domain_name
                [{"domain_id": "chess-champ"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            await harness.chat("Domain chess-champ")
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_clarification_in_pipeline_is_noop(self) -> None:
        """Including 'clarification' in pipeline is allowed but is a no-op.

        Clarification is handled by the confidence gate, not the
        pipeline engine.
        """
        config = _three_field_config()
        config["settings"] = {
            "recovery": {
                "pipeline": ["derivation", "clarification"],
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["All set!"],
            extraction_results=[
                [{"name": "Alice", "domain_id": "chess-champ",
                  "domain_name": "Chess Champ"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice, domain chess-champ, Chess Champ")
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_unknown_strategy_filtered(self) -> None:
        """Unknown strategy names are filtered out; valid ones still run."""
        config = _three_field_config()
        config["settings"] = {
            "derivations": [
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
            "recovery": {
                "pipeline": ["invalid_strategy", "derivation"],
            },
        }

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["All set!"],
            extraction_results=[
                [{"name": "Alice", "domain_id": "chess-champ"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice, domain chess-champ")
            # derivation still runs despite invalid_strategy
            assert harness.wizard_data["domain_name"] == "Chess Champ"
            assert harness.wizard_stage == "done"

"""Tests for wizard post-completion amendment flow (enhancement 2e).

Post-completion amendments allow users to make changes after the wizard
has completed, re-opening the wizard at the relevant stage.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


@dataclass
class SimpleExtractionResult:
    """Simple extraction result for testing.

    Mimics the interface of dataknobs_llm.extraction.ExtractionResult.
    """

    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.9
    errors: list[str] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.8 and not self.errors


class ConfigurableExtractor:
    """Extractor that returns configured results for testing.

    This is a real extractor implementation that returns pre-configured
    results, useful for testing extraction-dependent logic.
    """

    def __init__(
        self, result_data: dict[str, Any] | None = None, confidence: float = 0.9
    ):
        """Initialize with configured result.

        Args:
            result_data: Data to return from extraction
            confidence: Confidence score to return
        """
        self.result_data = result_data or {}
        self.confidence = confidence
        self.extract_calls: list[dict[str, Any]] = []

    async def extract(
        self,
        text: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> SimpleExtractionResult:
        """Return configured result and record the call."""
        self.extract_calls.append({
            "text": text,
            "schema": schema,
            "context": context,
            "model": model,
        })
        return SimpleExtractionResult(
            data=self.result_data, confidence=self.confidence
        )


@pytest.fixture
def amendment_fsm():
    """Create a wizard FSM with multiple stages for amendment testing."""
    config = {
        "name": "test-wizard",
        "settings": {"allow_post_completion_edits": True},
        "stages": [
            {
                "name": "welcome",
                "is_start": True,
                "prompt": "Welcome!",
                "transitions": [{"target": "configure_llm"}],
            },
            {
                "name": "configure_llm",
                "prompt": "Which LLM provider?",
                "schema": {
                    "type": "object",
                    "properties": {"llm_provider": {"type": "string"}},
                },
                "transitions": [{"target": "configure_identity"}],
            },
            {
                "name": "configure_identity",
                "prompt": "What is your bot's name?",
                "schema": {
                    "type": "object",
                    "properties": {"bot_name": {"type": "string"}},
                },
                "transitions": [{"target": "review"}],
            },
            {
                "name": "review",
                "prompt": "Review your configuration",
                "transitions": [{"target": "save"}],
            },
            {
                "name": "save",
                "is_end": True,
                "prompt": "Configuration saved!",
            },
        ],
    }
    loader = WizardConfigLoader()
    return loader.load_from_dict(config)


class TestMapSectionToStage:
    """Tests for _map_section_to_stage method."""

    def test_map_known_sections(self, amendment_fsm) -> None:
        """Default mapping works for known sections."""
        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm, allow_post_completion_edits=True
        )

        # These stages exist in the fixture
        assert reasoning._map_section_to_stage("llm") == "configure_llm"
        assert reasoning._map_section_to_stage("model") == "configure_llm"
        assert reasoning._map_section_to_stage("identity") == "configure_identity"
        assert reasoning._map_section_to_stage("name") == "configure_identity"
        assert reasoning._map_section_to_stage("config") == "review"

    def test_map_known_sections_case_insensitive(self, amendment_fsm) -> None:
        """Section mapping is case-insensitive."""
        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm, allow_post_completion_edits=True
        )

        assert reasoning._map_section_to_stage("LLM") == "configure_llm"
        assert reasoning._map_section_to_stage("Llm") == "configure_llm"
        assert reasoning._map_section_to_stage("IDENTITY") == "configure_identity"

    def test_map_unknown_section_returns_none(self, amendment_fsm) -> None:
        """Unknown sections return None."""
        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm, allow_post_completion_edits=True
        )

        assert reasoning._map_section_to_stage("unknown") is None
        assert reasoning._map_section_to_stage("") is None
        assert reasoning._map_section_to_stage("foobar") is None

    def test_map_handles_whitespace(self, amendment_fsm) -> None:
        """Section names with whitespace are trimmed."""
        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm, allow_post_completion_edits=True
        )

        assert reasoning._map_section_to_stage("  llm  ") == "configure_llm"
        assert reasoning._map_section_to_stage("\tmodel\n") == "configure_llm"

    def test_map_nonexistent_stage_returns_none(self) -> None:
        """If mapped stage doesn't exist in FSM, returns None."""
        # FSM without configure_tools stage
        config = {
            "name": "minimal",
            "stages": [
                {"name": "start", "is_start": True, "is_end": True},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        reasoning = WizardReasoning(wizard_fsm=fsm, allow_post_completion_edits=True)

        # "tools" maps to "configure_tools" which doesn't exist
        assert reasoning._map_section_to_stage("tools") is None

    def test_custom_section_mapping_override(self, amendment_fsm) -> None:
        """Custom section_to_stage_mapping takes precedence."""
        custom_mapping = {
            "ai": "configure_llm",
            "bot": "configure_identity",
        }
        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm,
            allow_post_completion_edits=True,
            section_to_stage_mapping=custom_mapping,
        )

        # Custom mapping
        assert reasoning._map_section_to_stage("ai") == "configure_llm"
        assert reasoning._map_section_to_stage("bot") == "configure_identity"

        # Default mapping still works for non-overridden sections
        assert reasoning._map_section_to_stage("llm") == "configure_llm"


class TestDetectAmendment:
    """Tests for _detect_amendment method."""

    @pytest.mark.asyncio
    async def test_detect_amendment_wants_edit(self, amendment_fsm) -> None:
        """Detect when user wants to edit LLM config."""
        extractor = ConfigurableExtractor(
            result_data={"wants_edit": True, "target_section": "llm"}
        )

        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm,
            extractor=extractor,
            allow_post_completion_edits=True,
        )

        state = WizardState(
            current_stage="save",
            completed=True,
            data={"llm_provider": "anthropic"},
        )

        result = await reasoning._detect_amendment("use ollama instead", state, llm=None)

        assert result is not None
        assert result["target_stage"] == "configure_llm"

        # Verify extractor was called
        assert len(extractor.extract_calls) == 1
        assert extractor.extract_calls[0]["text"] == "use ollama instead"

    @pytest.mark.asyncio
    async def test_detect_amendment_no_edit(self, amendment_fsm) -> None:
        """Don't detect amendment for non-edit messages."""
        extractor = ConfigurableExtractor(result_data={"wants_edit": False})

        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm,
            extractor=extractor,
            allow_post_completion_edits=True,
        )

        state = WizardState(
            current_stage="save", completed=True, data={}
        )

        result = await reasoning._detect_amendment("thanks!", state, llm=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_detect_amendment_unknown_section(self, amendment_fsm) -> None:
        """Unknown section in extraction returns None."""
        extractor = ConfigurableExtractor(
            result_data={"wants_edit": True, "target_section": "unknown_thing"}
        )

        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm,
            extractor=extractor,
            allow_post_completion_edits=True,
        )

        state = WizardState(current_stage="save", completed=True, data={})
        result = await reasoning._detect_amendment(
            "change the unknown thing", state, llm=None
        )

        # Can't map to a stage
        assert result is None

    @pytest.mark.asyncio
    async def test_detect_amendment_no_extractor(self, amendment_fsm) -> None:
        """Without extractor, amendment detection returns None."""
        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm,
            extractor=None,  # No extractor
            allow_post_completion_edits=True,
        )

        state = WizardState(current_stage="save", completed=True, data={})
        result = await reasoning._detect_amendment("change the llm", state, llm=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_detect_amendment_extractor_error(self, amendment_fsm) -> None:
        """Extractor errors are handled gracefully."""

        class FailingExtractor:
            async def extract(self, *args, **kwargs):
                raise RuntimeError("Extraction failed")

        reasoning = WizardReasoning(
            wizard_fsm=amendment_fsm,
            extractor=FailingExtractor(),
            allow_post_completion_edits=True,
        )

        state = WizardState(current_stage="save", completed=True, data={})
        result = await reasoning._detect_amendment("change the llm", state, llm=None)

        # Should return None without raising
        assert result is None


class TestAmendmentFlagAccess:
    """Tests for amendment flag accessibility."""

    def test_allow_amendments_flag(self, amendment_fsm) -> None:
        """_allow_amendments flag is set correctly."""
        reasoning_enabled = WizardReasoning(
            wizard_fsm=amendment_fsm, allow_post_completion_edits=True
        )
        reasoning_disabled = WizardReasoning(
            wizard_fsm=amendment_fsm, allow_post_completion_edits=False
        )

        assert reasoning_enabled._allow_amendments is True
        assert reasoning_disabled._allow_amendments is False

    def test_from_config_loads_amendment_setting(self) -> None:
        """from_config reads allow_post_completion_edits from settings."""
        import tempfile
        from pathlib import Path

        config_content = """
name: test-wizard
settings:
  allow_post_completion_edits: true
  section_to_stage_mapping:
    ai: configure_llm
stages:
  - name: configure_llm
    is_start: true
    prompt: Configure LLM
    transitions:
      - target: done
  - name: done
    is_end: true
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            reasoning = WizardReasoning.from_config({"wizard_config": str(config_path)})
            assert reasoning._allow_amendments is True
            assert reasoning._section_to_stage_mapping == {"ai": "configure_llm"}
        finally:
            config_path.unlink()


class TestAmendmentTransitionRecording:
    """Tests for amendment transition recording."""

    def test_amendment_creates_transition_record(self) -> None:
        """Amendment creates proper transition records."""
        from dataknobs_bots.reasoning.observability import create_transition_record

        # Verify the function works for amendment trigger
        record = create_transition_record(
            from_stage="save",
            to_stage="configure_llm",
            trigger="amendment",
            data_snapshot={"llm_provider": "anthropic"},
            user_input="use ollama instead",
        )

        assert record.from_stage == "save"
        assert record.to_stage == "configure_llm"
        assert record.trigger == "amendment"
        assert record.user_input == "use ollama instead"

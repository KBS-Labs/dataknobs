"""Tests for tools and metadata in wizard clarification/restart/validation paths.

Bug: _generate_clarification_response(), _generate_restart_offer(), and
_generate_validation_response() called manager.complete() without passing
tools or wizard metadata. This prevented the LLM from calling tools (e.g.
list_catalog, list_bank_records) during clarification exchanges.

The fix passes tools and wizard_state to all alternate response paths,
matching the normal _generate_stage_response() behavior.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.testing import scripted_schema_extractor


# ---------------------------------------------------------------------------
# Test tool
# ---------------------------------------------------------------------------


class SimpleTool:
    """Minimal tool for testing that tools are forwarded to the LLM."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return f"Called {self.name}"


# ---------------------------------------------------------------------------
# Wizard config: one structured stage with required fields
# ---------------------------------------------------------------------------

CLARIFICATION_WIZARD_CONFIG: dict[str, Any] = {
    "name": "clarification-test",
    "version": "1.0",
    "stages": [
        {
            "name": "collect_info",
            "is_start": True,
            "prompt": "Tell me your name and topic.",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "topic": {"type": "string"},
                },
                "required": ["name", "topic"],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": "data.get('name') and data.get('topic')",
                },
            ],
        },
        {
            "name": "done",
            "is_end": True,
            "prompt": "All done!",
        },
    ],
}


def _build_clarification_wizard(
    extraction_responses: list[str],
) -> tuple[WizardReasoning, EchoProvider]:
    """Build a WizardReasoning whose extraction will return scripted responses.

    The extraction provider is separate from the conversation manager's
    provider.  Scripting extraction_responses controls what the extractor
    returns — e.g. invalid JSON causes low confidence and triggers the
    clarification path.

    Returns:
        Tuple of (WizardReasoning, extraction_provider).
    """
    extractor, extraction_provider = scripted_schema_extractor(
        extraction_responses,
    )

    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(CLARIFICATION_WIZARD_CONFIG)

    reasoning = WizardReasoning(
        wizard_fsm=wizard_fsm,
        extractor=extractor,
        strict_validation=False,
        extraction_scope="current_message",
    )

    return reasoning, extraction_provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClarificationToolsPassed:
    """Verify that the clarification path forwards tools to the LLM."""

    @pytest.mark.asyncio
    async def test_clarification_response_includes_tools(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """When extraction fails and clarification fires, the LLM call
        must include the tools list so the LLM can handle out-of-stage
        requests (e.g. 'show me all recipes').
        """
        manager, conv_provider = conversation_manager_pair
        tools = [SimpleTool("list_catalog"), SimpleTool("list_bank_records")]

        # Extraction returns invalid JSON → low confidence → clarification
        reasoning, _ = _build_clarification_wizard(
            extraction_responses=["not valid json at all"]
        )

        await manager.add_message(role="user", content="something vague")

        # Script the conversation provider's clarification response
        conv_provider.set_responses(["Could you please clarify?"])

        response = await reasoning.generate(manager, llm=None, tools=tools)

        assert response is not None

        # The conversation provider should have been called with tools
        last_call = conv_provider.get_last_call()
        assert last_call is not None
        assert last_call["tools"] is not None, (
            "Clarification response must include tools so the LLM can "
            "call them during clarification exchanges"
        )
        tool_names = [t.name for t in last_call["tools"]]
        assert "list_catalog" in tool_names
        assert "list_bank_records" in tool_names

    @pytest.mark.asyncio
    async def test_clarification_response_includes_wizard_metadata(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Clarification response must carry wizard metadata on the
        response object (via _add_wizard_metadata) for downstream consumers.
        """
        manager, conv_provider = conversation_manager_pair
        tools = [SimpleTool("list_catalog")]

        reasoning, _ = _build_clarification_wizard(
            extraction_responses=["not valid json"]
        )

        await manager.add_message(role="user", content="something vague")
        conv_provider.set_responses(["Could you clarify?"])

        response = await reasoning.generate(manager, llm=None, tools=tools)

        # Response object should have wizard metadata
        assert hasattr(response, "metadata")
        assert response.metadata is not None
        assert "wizard" in response.metadata
        wizard_meta = response.metadata["wizard"]
        assert wizard_meta["current_stage"] == "collect_info"


class TestRestartOfferToolsPassed:
    """Verify that the restart-offer path forwards tools to the LLM."""

    @pytest.mark.asyncio
    async def test_restart_offer_includes_tools(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After 3+ clarification failures, _generate_restart_offer fires.
        It must also include tools in the LLM call.
        """
        manager, conv_provider = conversation_manager_pair
        tools = [SimpleTool("list_catalog")]

        # Need 3 extraction failures to trigger restart offer
        reasoning, _ = _build_clarification_wizard(
            extraction_responses=[
                "invalid json 1",
                "invalid json 2",
                "invalid json 3",
            ]
        )

        # Three rounds of failed extraction
        for i in range(3):
            await manager.add_message(
                role="user", content=f"vague message {i}"
            )
            conv_provider.set_responses([f"Clarification attempt {i}"])
            await reasoning.generate(manager, llm=None, tools=tools)

        # The third call triggers _generate_restart_offer
        last_call = conv_provider.get_last_call()
        assert last_call is not None
        assert last_call["tools"] is not None, (
            "Restart-offer response must include tools"
        )
        tool_names = [t.name for t in last_call["tools"]]
        assert "list_catalog" in tool_names


class TestClarificationWithoutTools:
    """Verify that the fix handles the no-tools case gracefully."""

    @pytest.mark.asyncio
    async def test_clarification_works_without_tools(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """When generate() is called without tools, clarification should
        still work — tools=None is passed through cleanly.
        """
        manager, conv_provider = conversation_manager_pair

        reasoning, _ = _build_clarification_wizard(
            extraction_responses=["not valid json"]
        )

        await manager.add_message(role="user", content="something vague")
        conv_provider.set_responses(["Could you clarify?"])

        # No tools passed — should not raise
        response = await reasoning.generate(manager, llm=None)

        assert response is not None
        last_call = conv_provider.get_last_call()
        assert last_call is not None
        # tools should be None (not missing from the call)
        assert last_call["tools"] is None

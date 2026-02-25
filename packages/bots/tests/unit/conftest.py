"""Test configuration and fixtures for wizard reasoning tests."""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.llm import LLMConfig
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary


@pytest.fixture
def simple_wizard_config() -> dict[str, Any]:
    """Create a simple 3-stage wizard configuration."""
    return {
        "name": "test-wizard",
        "version": "1.0",
        "description": "A test wizard",
        "stages": [
            {
                "name": "welcome",
                "is_start": True,
                "prompt": "What would you like to do?",
                "schema": {
                    "type": "object",
                    "properties": {"intent": {"type": "string"}},
                    "required": ["intent"],
                },
                "suggestions": ["Create something", "Edit something"],
                "transitions": [
                    {"target": "configure", "condition": "data.get('intent')"}
                ],
            },
            {
                "name": "configure",
                "prompt": "How would you like to configure it?",
                "can_skip": True,
                "transitions": [{"target": "complete"}],
            },
            {
                "name": "complete",
                "is_end": True,
                "prompt": "All done!",
            },
        ],
    }


@pytest.fixture
def wizard_config_with_branches() -> dict[str, Any]:
    """Create a wizard configuration with branching transitions."""
    return {
        "name": "branching-wizard",
        "version": "1.0",
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "prompt": "Choose your path",
                "transitions": [
                    {
                        "target": "path_a",
                        "condition": "data.get('choice') == 'a'",
                        "priority": 0,
                    },
                    {
                        "target": "path_b",
                        "condition": "data.get('choice') == 'b'",
                        "priority": 1,
                    },
                    {"target": "default", "priority": 2},
                ],
            },
            {"name": "path_a", "is_end": True, "prompt": "Path A chosen"},
            {"name": "path_b", "is_end": True, "prompt": "Path B chosen"},
            {"name": "default", "is_end": True, "prompt": "Default path"},
        ],
    }


@pytest.fixture
def wizard_fsm(simple_wizard_config: dict[str, Any]):
    """Create a WizardFSM from simple config."""
    loader = WizardConfigLoader()
    return loader.load_from_dict(simple_wizard_config)


@pytest.fixture
def wizard_reasoning(simple_wizard_config: dict[str, Any]) -> WizardReasoning:
    """Create a WizardReasoning instance with no strict validation."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(simple_wizard_config)
    return WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)


@pytest.fixture
async def conversation_manager_pair():
    """Create a real ConversationManager + EchoProvider pair.

    Uses ConfigPromptLibrary (no filesystem) with a simple system prompt.
    Provides the standard conversation manager for all wizard tests.

    Yields:
        Tuple of (ConversationManager, EchoProvider)
    """
    config = LLMConfig(
        provider="echo",
        model="echo-test",
        options={"echo_prefix": ""},
    )
    provider = EchoProvider(config)
    library = ConfigPromptLibrary({
        "system": {
            "assistant": {
                "template": "You are a helpful assistant.",
            },
        },
    })
    builder = AsyncPromptBuilder(library=library)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())

    manager = await ConversationManager.create(
        llm=provider,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="assistant",
    )

    yield manager, provider

    await provider.close()


@pytest.fixture
async def conversation_manager(
    conversation_manager_pair: tuple[ConversationManager, EchoProvider],
) -> ConversationManager:
    """Convenience fixture: just the ConversationManager."""
    return conversation_manager_pair[0]


@pytest.fixture
async def echo_provider(
    conversation_manager_pair: tuple[ConversationManager, EchoProvider],
) -> EchoProvider:
    """Convenience fixture: just the EchoProvider."""
    return conversation_manager_pair[1]


def set_wizard_state(
    manager: ConversationManager,
    fsm_state: dict[str, Any],
) -> None:
    """Set wizard FSM state on a ConversationManager's metadata.

    Helper for tests that need to pre-populate wizard state.

    Args:
        manager: The ConversationManager instance.
        fsm_state: The FSM state dict to set.
    """
    manager.metadata["wizard"] = {"fsm_state": fsm_state}

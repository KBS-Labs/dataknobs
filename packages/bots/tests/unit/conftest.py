"""Test configuration and fixtures for wizard reasoning tests."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.llm import LLMMessage, LLMResponse
from dataknobs_llm.llm.providers.echo import EchoProvider


@dataclass
class TestLLMResponse:
    """Lightweight response object for testing.

    Can be used with real LLM providers or mock managers.
    """

    content: str = "Test response"
    metadata: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
    usage: dict[str, Any] | None = None


class WizardTestManager:
    """Lightweight conversation manager for wizard reasoning tests.

    This is a minimal implementation that provides the interface
    WizardReasoning expects. It uses EchoProvider under the hood for
    generating controlled responses in tests.

    Features:
    - Wraps EchoProvider for scripted responses
    - Provides metadata storage for wizard state
    - Tracks messages as simple dicts (not LLMMessage objects)
    - Records all complete() calls for test verification

    Example:
        ```python
        manager = WizardTestManager()

        # Script responses using the underlying EchoProvider
        manager.echo_provider.set_responses(["First response", "Second response"])

        # Or use pattern matching
        manager.echo_provider.add_pattern_response(r"hello", "Hi there!")

        # Verify what was called
        assert manager.echo_provider.call_count == 2
        ```

    For full integration tests, use the real ConversationManager with
    EchoProvider via the bot configuration fixtures.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        initial_metadata: dict[str, Any] | None = None,
    ):
        self.metadata: dict[str, Any] = initial_metadata or {}
        self.messages: list[dict[str, Any]] = []
        self.system_prompt = system_prompt
        self.complete_calls: list[dict[str, Any]] = []

        # Create EchoProvider for generating responses
        self.echo_provider = EchoProvider(
            {"provider": "echo", "model": "echo-test", "options": {"echo_prefix": ""}}
        )

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append({"role": "assistant", "content": content})

    async def add_message(self, content: str, role: str = "user") -> None:
        """Add a message to the conversation (async interface for ReAct compatibility).

        This method supports the interface used by ReAct-style loops where
        tool observations are added as system messages.

        Args:
            content: Message content
            role: Message role (user, assistant, system)
        """
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the conversation."""
        return self.messages

    async def complete(
        self,
        system_prompt_override: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using EchoProvider.

        Records the call parameters for test verification, then uses
        EchoProvider to generate the actual response.

        Args:
            system_prompt_override: Override system prompt (recorded but not used)
            tools: Available tools (recorded but not used)
            **kwargs: Additional parameters

        Returns:
            LLMResponse from EchoProvider
        """
        self.complete_calls.append(
            {
                "system_prompt_override": system_prompt_override,
                "tools": tools,
                **kwargs,
            }
        )

        # Convert messages to LLMMessage format for EchoProvider
        llm_messages = [
            LLMMessage(role=m["role"], content=m.get("content", ""))
            for m in self.messages
        ]

        # Use EchoProvider to generate response
        response = await self.echo_provider.complete(llm_messages)

        # Ensure metadata dict exists (some responses may not have it)
        if not hasattr(response, "metadata") or response.metadata is None:
            response.metadata = {}

        return response


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
def test_manager() -> WizardTestManager:
    """Create a test conversation manager."""
    return WizardTestManager()


@pytest.fixture
def test_manager_with_state(simple_wizard_config: dict[str, Any]) -> WizardTestManager:
    """Create a test manager with pre-populated wizard state."""
    return WizardTestManager(
        initial_metadata={
            "wizard": {
                "fsm_state": {
                    "current_stage": "welcome",
                    "history": ["welcome"],
                    "data": {},
                    "completed": False,
                    "clarification_attempts": 0,
                }
            }
        }
    )

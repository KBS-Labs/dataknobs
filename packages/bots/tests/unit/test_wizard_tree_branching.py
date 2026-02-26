"""Tests for conversation tree branching on wizard stage revisits.

When a wizard stage is revisited (via restart, go_back, or FSM loop), the
conversation tree should create a sibling branch from the point where the
stage was previously entered, rather than chaining deeper as a child.

Tests cover:
- Restart creates a sibling branch for the greeting stage
- Back navigation creates a sibling branch for the previous stage
- First visit does not branch (normal linear chain)
- Multiple restarts create multiple sibling branches
"""

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.conversations.storage import ConversationNode
from dataknobs_llm.llm.providers.echo import EchoProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_stage_assistant_nodes(
    manager: ConversationManager, stage_name: str
) -> list:
    """Find all assistant nodes in the tree for a given wizard stage."""
    tree = manager.state.message_tree
    return tree.find_nodes(
        lambda n: (
            isinstance(n.data, ConversationNode)
            and n.data.message.role == "assistant"
            and n.data.metadata.get("wizard", {}).get("current_stage")
            == stage_name
        ),
    )


def _get_assistant_node_ids(manager: ConversationManager) -> list[str]:
    """Collect node_ids for assistant messages in the tree (BFS order)."""
    tree = manager.state.message_tree
    nodes = tree.find_nodes(
        lambda n: (
            isinstance(n.data, ConversationNode)
            and n.data.message.role == "assistant"
        ),
        traversal="bfs",
    )
    return [
        n.data.node_id
        for n in nodes
        if isinstance(n.data, ConversationNode)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def template_wizard_config() -> dict:
    """Wizard with response_template stages and message-based transitions.

    Uses response_template so no LLM calls are needed for responses.
    Transitions use _message conditions so no schema extractor is needed.
    """
    return {
        "name": "tree-branch-test",
        "version": "1.0",
        "stages": [
            {
                "name": "greeting",
                "is_start": True,
                "prompt": "What is your name?",
                "response_template": "Hello! What is your name?",
                "transitions": [
                    {
                        "target": "topic",
                        "condition": (
                            "'advance' in data.get('_message', '').lower()"
                        ),
                    },
                ],
            },
            {
                "name": "topic",
                "prompt": "What topic?",
                "response_template": "Great! What topic?",
                "transitions": [
                    {
                        "target": "summary",
                        "condition": (
                            "'advance' in data.get('_message', '').lower()"
                        ),
                    },
                ],
            },
            {
                "name": "summary",
                "is_end": True,
                "prompt": "All done!",
                "response_template": "All done!",
            },
        ],
    }


@pytest.fixture
def template_reasoning(template_wizard_config: dict) -> WizardReasoning:
    """WizardReasoning with template-only stages."""
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(template_wizard_config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFirstVisitNoBranching:
    """On first visit, the tree should be a normal linear chain."""

    @pytest.mark.asyncio
    async def test_first_visit_no_branching(
        self,
        template_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Forward-only wizard produces a linear chain with no siblings."""
        manager, _provider = conversation_manager_pair

        # Greet → greeting stage
        await template_reasoning.greet(manager, llm=None)

        # Advance to topic stage
        await manager.add_message(role="user", content="advance please")
        await template_reasoning.generate(manager, llm=None)

        # Advance to summary stage
        await manager.add_message(role="user", content="advance please")
        await template_reasoning.generate(manager, llm=None)

        # Verify: 3 assistant nodes (greeting, topic, summary)
        assistant_ids = _get_assistant_node_ids(manager)
        assert len(assistant_ids) == 3

        # Each stage visited exactly once
        for stage_name in ("greeting", "topic", "summary"):
            nodes = _find_stage_assistant_nodes(manager, stage_name)
            assert len(nodes) == 1, (
                f"Expected 1 node for stage '{stage_name}', got {len(nodes)}"
            )


class TestRestartCreatesSiblingBranch:
    """Restart should create a sibling branch at the greeting level."""

    @pytest.mark.asyncio
    async def test_restart_creates_sibling_branch(
        self,
        template_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After restart, the new greeting is a sibling of the original."""
        manager, _provider = conversation_manager_pair

        # Greet → greeting stage (creates assistant node)
        await template_reasoning.greet(manager, llm=None)

        # Restart immediately
        await manager.add_message(role="user", content="restart")
        await template_reasoning.generate(manager, llm=None)

        # Should have 2 greeting nodes (original + restart)
        greeting_nodes = _find_stage_assistant_nodes(manager, "greeting")
        assert len(greeting_nodes) == 2

        # The two greeting nodes should share the same parent
        assert greeting_nodes[0].parent is greeting_nodes[1].parent


class TestBackCreatesSiblingBranch:
    """Back navigation should create a sibling branch."""

    @pytest.mark.asyncio
    async def test_back_creates_sibling_branch(
        self,
        template_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Going back to greeting creates a sibling of the original greeting."""
        manager, _provider = conversation_manager_pair

        # Greet → greeting stage
        await template_reasoning.greet(manager, llm=None)

        # Advance to topic stage (using _message-based transition)
        await manager.add_message(role="user", content="advance please")
        await template_reasoning.generate(manager, llm=None)

        # Go back to greeting
        await manager.add_message(role="user", content="back")
        await template_reasoning.generate(manager, llm=None)

        # Should have 2 greeting nodes (original + back)
        greeting_nodes = _find_stage_assistant_nodes(manager, "greeting")
        assert len(greeting_nodes) == 2

        # They should share the same parent
        assert greeting_nodes[0].parent is greeting_nodes[1].parent


class TestMultipleRestartsCreateSiblingBranches:
    """Multiple restarts should create multiple sibling branches."""

    @pytest.mark.asyncio
    async def test_multiple_restarts_create_sibling_branches(
        self,
        template_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Restarting twice produces 3 sibling greeting nodes."""
        manager, _provider = conversation_manager_pair

        # First greeting
        await template_reasoning.greet(manager, llm=None)

        # First restart
        await manager.add_message(role="user", content="restart")
        await template_reasoning.generate(manager, llm=None)

        # Second restart
        await manager.add_message(role="user", content="restart")
        await template_reasoning.generate(manager, llm=None)

        # Should have 3 greeting nodes (original + 2 restarts)
        greeting_nodes = _find_stage_assistant_nodes(manager, "greeting")
        assert len(greeting_nodes) == 3

        # All three should share the same parent
        parents = {id(n.parent) for n in greeting_nodes}
        assert len(parents) == 1

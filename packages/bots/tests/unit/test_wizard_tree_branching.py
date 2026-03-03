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


def _get_node_path(
    manager: ConversationManager,
) -> list[tuple[str, str, str | None]]:
    """Return linearized path from root to current node.

    Returns:
        List of ``(node_id, role, stage_name_or_None)`` tuples.
    """
    nodes = manager.state.get_current_nodes()
    result: list[tuple[str, str, str | None]] = []
    for node in nodes:
        stage = node.metadata.get("wizard", {}).get("current_stage")
        result.append((node.node_id, node.message.role, stage))
    return result


def _is_descendant_of(node_id: str, ancestor_id: str) -> bool:
    """True if *node_id* is a strict descendant of *ancestor_id*.

    Both IDs use dot-delimited segment notation (e.g. ``"0.1.2"``).
    """
    if not ancestor_id:
        # Root is ancestor of everything
        return True
    return node_id.startswith(ancestor_id + ".")


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


class TestPostRestartStagesIsolatedFromOldBranch:
    """After restart, post-restart stage nodes must live under the new branch.

    Bug: ``_find_stage_node_id`` searched the entire tree, so advancing
    from the new (Run-2) greeting to topic found the OLD (Run-1) topic
    node and branched from it — grafting Run-2 nodes onto Run-1's branch.
    """

    @pytest.mark.asyncio
    async def test_post_restart_stages_isolated_from_old_branch(
        self,
        template_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Full restart scenario: Run-2 topic/summary are descendants of
        Run-2 greeting, not siblings of Run-1 nodes."""
        manager, _provider = conversation_manager_pair

        # ── Run 1 ──────────────────────────────────────────────
        await template_reasoning.greet(manager, llm=None)

        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        # ── Restart ────────────────────────────────────────────
        await manager.add_message(role="user", content="restart")
        await template_reasoning.generate(manager, llm=None)

        # ── Run 2 ──────────────────────────────────────────────
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        # ── Assertions ─────────────────────────────────────────
        # 2 greeting nodes (Run-1, Run-2) sharing the same parent
        greeting_nodes = _find_stage_assistant_nodes(manager, "greeting")
        assert len(greeting_nodes) == 2
        assert greeting_nodes[0].parent is greeting_nodes[1].parent

        # Identify which greeting belongs to which run (DFS order)
        run1_greeting = greeting_nodes[0]
        run2_greeting = greeting_nodes[1]
        assert isinstance(run1_greeting.data, ConversationNode)
        assert isinstance(run2_greeting.data, ConversationNode)
        run1_greeting_id = run1_greeting.data.node_id
        run2_greeting_id = run2_greeting.data.node_id

        # 2 topic nodes
        topic_nodes = _find_stage_assistant_nodes(manager, "topic")
        assert len(topic_nodes) == 2

        # Classify topic nodes by which greeting branch they descend from
        run1_topics = [
            n for n in topic_nodes
            if isinstance(n.data, ConversationNode)
            and _is_descendant_of(n.data.node_id, run1_greeting_id)
        ]
        run2_topics = [
            n for n in topic_nodes
            if isinstance(n.data, ConversationNode)
            and _is_descendant_of(n.data.node_id, run2_greeting_id)
        ]
        assert len(run1_topics) == 1, (
            f"Expected 1 topic under Run-1 greeting ({run1_greeting_id}), "
            f"got {len(run1_topics)}"
        )
        assert len(run2_topics) == 1, (
            f"Expected 1 topic under Run-2 greeting ({run2_greeting_id}), "
            f"got {len(run2_topics)}"
        )

        # Run-1 and Run-2 topic nodes must NOT share a parent
        assert run1_topics[0].parent is not run2_topics[0].parent

        # 2 summary nodes, each under the correct topic
        summary_nodes = _find_stage_assistant_nodes(manager, "summary")
        assert len(summary_nodes) == 2

        run2_topic_id = run2_topics[0].data.node_id
        assert isinstance(run2_topic_id, str)
        run2_summaries = [
            n for n in summary_nodes
            if isinstance(n.data, ConversationNode)
            and _is_descendant_of(n.data.node_id, run2_topic_id)
        ]
        assert len(run2_summaries) == 1, (
            f"Expected 1 summary under Run-2 topic ({run2_topic_id}), "
            f"got {len(run2_summaries)}"
        )


class TestWithinRunRevisitStillBranchesAfterRestart:
    """Back navigation within Run 2 should still branch correctly."""

    @pytest.mark.asyncio
    async def test_within_run_revisit_still_branches_after_restart(
        self,
        template_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After restart + back within Run 2, new nodes stay on Run-2's
        branch and do not leak into Run-1."""
        manager, _provider = conversation_manager_pair

        # ── Run 1 ──────────────────────────────────────────────
        await template_reasoning.greet(manager, llm=None)
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        # ── Restart ────────────────────────────────────────────
        await manager.add_message(role="user", content="restart")
        await template_reasoning.generate(manager, llm=None)

        # ── Run 2: advance to topic ────────────────────────────
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        # ── Run 2: back to greeting ────────────────────────────
        await manager.add_message(role="user", content="back")
        await template_reasoning.generate(manager, llm=None)

        # ── Run 2: advance to topic again ──────────────────────
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        # ── Assertions ─────────────────────────────────────────
        # 3+ greeting nodes: Run-1, Run-2 initial, Run-2 after back
        greeting_nodes = _find_stage_assistant_nodes(manager, "greeting")
        assert len(greeting_nodes) >= 3

        # All greeting nodes share the same parent (sibling branches)
        parents = {id(n.parent) for n in greeting_nodes}
        assert len(parents) == 1

        # Run-2 topic nodes must NOT be descendants of Run-1 greeting
        run1_greeting_id = greeting_nodes[0].data.node_id
        assert isinstance(run1_greeting_id, str)

        topic_nodes = _find_stage_assistant_nodes(manager, "topic")
        run1_descendant_topics = [
            n for n in topic_nodes
            if isinstance(n.data, ConversationNode)
            and _is_descendant_of(n.data.node_id, run1_greeting_id)
        ]
        # Only Run-1's own topic should be under Run-1's greeting
        assert len(run1_descendant_topics) == 1


class TestLinearizedContextExcludesOldBranch:
    """The linearized message path (what the LLM sees) must not contain
    messages from the old branch after a restart."""

    @pytest.mark.asyncio
    async def test_linearized_context_excludes_old_branch(
        self,
        template_reasoning: WizardReasoning,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After restart + advance, the linearized path must descend
        entirely through Run-2's greeting branch, not through Run-1's."""
        manager, _provider = conversation_manager_pair

        # ── Run 1 ──────────────────────────────────────────────
        await template_reasoning.greet(manager, llm=None)
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        # ── Restart ────────────────────────────────────────────
        await manager.add_message(role="user", content="restart")
        await template_reasoning.generate(manager, llm=None)

        # Identify Run-2 greeting (the most recent greeting node)
        greeting_nodes = _find_stage_assistant_nodes(manager, "greeting")
        assert len(greeting_nodes) == 2
        run2_greeting = greeting_nodes[-1]  # DFS order: last = newest
        assert isinstance(run2_greeting.data, ConversationNode)
        run2_greeting_id = run2_greeting.data.node_id

        # ── Run 2: advance through all stages ──────────────────
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)
        await manager.add_message(role="user", content="advance")
        await template_reasoning.generate(manager, llm=None)

        # ── Assertions ─────────────────────────────────────────
        path = _get_node_path(manager)

        # Every assistant node with wizard metadata in the path must be
        # either the Run-2 greeting itself OR a descendant of it.
        stage_nodes_in_path = [
            (nid, stage)
            for (nid, role, stage) in path
            if stage is not None and role == "assistant"
        ]

        for nid, stage in stage_nodes_in_path:
            on_run2_branch = (
                nid == run2_greeting_id
                or _is_descendant_of(nid, run2_greeting_id)
            )
            assert on_run2_branch, (
                f"Linearized path contains assistant node '{nid}' "
                f"(stage={stage}) which is NOT on Run-2's branch "
                f"(Run-2 greeting is '{run2_greeting_id}')"
            )

        # Exactly 3 assistant stage nodes in the path
        assert len(stage_nodes_in_path) == 3, (
            f"Expected 3 assistant stage nodes in path, "
            f"got {len(stage_nodes_in_path)}: {stage_nodes_in_path}"
        )

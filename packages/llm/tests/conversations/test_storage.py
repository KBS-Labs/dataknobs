"""Tests for conversation storage with tree-based branching."""

import pytest
from datetime import datetime
from dataknobs_structures.tree import Tree
from dataknobs_llm.llm.base import LLMMessage
from dataknobs_llm.conversations import (
    ConversationNode,
    ConversationState,
    calculate_node_id,
    get_node_by_id,
    get_messages_for_llm,
)


class TestConversationNode:
    """Tests for ConversationNode."""

    def test_create_node(self):
        """Test creating a conversation node."""
        msg = LLMMessage(role="user", content="Hello")
        node = ConversationNode(
            message=msg,
            node_id="0",
            prompt_name="greeting"
        )

        assert node.message == msg
        assert node.node_id == "0"
        assert node.prompt_name == "greeting"
        assert node.branch_name is None
        assert isinstance(node.timestamp, datetime)

    def test_node_with_metadata(self):
        """Test node with metadata."""
        msg = LLMMessage(role="assistant", content="Hi there")
        node = ConversationNode(
            message=msg,
            node_id="0.0",
            prompt_name="response",
            branch_name="friendly",
            metadata={"model": "gpt-4", "tokens": 10}
        )

        assert node.branch_name == "friendly"
        assert node.metadata["model"] == "gpt-4"
        assert node.metadata["tokens"] == 10

    def test_node_serialization(self):
        """Test node to_dict/from_dict."""
        msg = LLMMessage(role="user", content="Test message")
        node = ConversationNode(
            message=msg,
            node_id="0.1.2",
            prompt_name="test_prompt",
            branch_name="variant-a",
            metadata={"custom": "data"}
        )

        # Serialize
        data = node.to_dict()
        assert data["node_id"] == "0.1.2"
        assert data["prompt_name"] == "test_prompt"
        assert data["branch_name"] == "variant-a"
        assert data["metadata"]["custom"] == "data"
        assert data["message"]["role"] == "user"
        assert data["message"]["content"] == "Test message"

        # Deserialize
        restored = ConversationNode.from_dict(data)
        assert restored.node_id == node.node_id
        assert restored.prompt_name == node.prompt_name
        assert restored.branch_name == node.branch_name
        assert restored.message.role == node.message.role
        assert restored.message.content == node.message.content
        assert restored.metadata == node.metadata


class TestNodeIdentification:
    """Tests for node ID calculation and navigation."""

    def test_calculate_node_id_root(self):
        """Test calculating ID for root node."""
        root = Tree("root")
        assert calculate_node_id(root) == ""

    def test_calculate_node_id_single_level(self):
        """Test calculating ID for first-level children."""
        root = Tree("root")
        child0 = root.add_child("child0")
        child1 = root.add_child("child1")
        child2 = root.add_child("child2")

        assert calculate_node_id(child0) == "0"
        assert calculate_node_id(child1) == "1"
        assert calculate_node_id(child2) == "2"

    def test_calculate_node_id_nested(self):
        """Test calculating ID for nested children."""
        root = Tree("root")
        child0 = root.add_child("child0")
        child1 = root.add_child("child1")

        grandchild0_0 = child0.add_child("grandchild0_0")
        grandchild0_1 = child0.add_child("grandchild0_1")
        grandchild1_0 = child1.add_child("grandchild1_0")

        assert calculate_node_id(grandchild0_0) == "0.0"
        assert calculate_node_id(grandchild0_1) == "0.1"
        assert calculate_node_id(grandchild1_0) == "1.0"

    def test_calculate_node_id_deep(self):
        """Test calculating ID for deeply nested nodes."""
        root = Tree("root")
        level1 = root.add_child("level1")
        level2 = level1.add_child("level2")
        level3 = level2.add_child("level3")
        level4 = level3.add_child("level4")

        assert calculate_node_id(level4) == "0.0.0.0"

    def test_get_node_by_id_root(self):
        """Test getting root node by ID."""
        root = Tree("root")
        assert get_node_by_id(root, "") == root
        assert get_node_by_id(root, "").data == "root"

    def test_get_node_by_id_single_level(self):
        """Test getting first-level children by ID."""
        root = Tree("root")
        child0 = root.add_child("child0")
        child1 = root.add_child("child1")

        assert get_node_by_id(root, "0") == child0
        assert get_node_by_id(root, "1") == child1
        assert get_node_by_id(root, "0").data == "child0"

    def test_get_node_by_id_nested(self):
        """Test getting nested children by ID."""
        root = Tree("root")
        child0 = root.add_child("child0")
        child1 = root.add_child("child1")
        grandchild0_0 = child0.add_child("grandchild0_0")  # First grandchild
        grandchild0_1 = child0.add_child("grandchild0_1")  # Second grandchild

        node = get_node_by_id(root, "0.1")
        assert node == grandchild0_1
        assert node.data == "grandchild0_1"

    def test_get_node_by_id_invalid(self):
        """Test getting node with invalid ID."""
        root = Tree("root")
        root.add_child("child0")

        assert get_node_by_id(root, "5") is None
        assert get_node_by_id(root, "0.0") is None
        assert get_node_by_id(root, "0.5.2") is None

    def test_get_messages_for_llm(self):
        """Test extracting message path for LLM."""
        # Build conversation tree
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="You are helpful"),
            node_id=""
        )
        root = Tree(root_node)

        user_node = ConversationNode(
            message=LLMMessage(role="user", content="Hello"),
            node_id="0"
        )
        user_tree = root.add_child(Tree(user_node))

        assistant_node = ConversationNode(
            message=LLMMessage(role="assistant", content="Hi there!"),
            node_id="0.0"
        )
        user_tree.add_child(Tree(assistant_node))

        # Get messages for LLM
        messages = get_messages_for_llm(root, "0.0")

        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful"
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"
        assert messages[2].role == "assistant"
        assert messages[2].content == "Hi there!"

    def test_get_messages_for_llm_partial_path(self):
        """Test getting messages for intermediate node."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        root = Tree(root_node)

        user1_node = ConversationNode(
            message=LLMMessage(role="user", content="Question 1"),
            node_id="0"
        )
        user1_tree = root.add_child(Tree(user1_node))

        assistant1_node = ConversationNode(
            message=LLMMessage(role="assistant", content="Answer 1"),
            node_id="0.0"
        )
        assistant1_tree = user1_tree.add_child(Tree(assistant1_node))

        user2_node = ConversationNode(
            message=LLMMessage(role="user", content="Question 2"),
            node_id="0.0.0"
        )
        assistant1_tree.add_child(Tree(user2_node))

        # Get messages up to first user message
        messages = get_messages_for_llm(root, "0")
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "Question 1"


class TestConversationState:
    """Tests for ConversationState."""

    def test_create_state(self):
        """Test creating conversation state."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        state = ConversationState(
            conversation_id="conv-123",
            message_tree=tree,
            current_node_id="",
            metadata={"user_id": "alice"}
        )

        assert state.conversation_id == "conv-123"
        assert state.current_node_id == ""
        assert state.metadata["user_id"] == "alice"
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)

    def test_get_current_node(self):
        """Test getting current node."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        user_node = ConversationNode(
            message=LLMMessage(role="user", content="Hello"),
            node_id="0"
        )
        tree.add_child(Tree(user_node))

        state = ConversationState(
            conversation_id="conv-123",
            message_tree=tree,
            current_node_id="0"
        )

        current = state.get_current_node()
        assert current is not None
        assert current.data.message.content == "Hello"

    def test_get_current_messages(self):
        """Test getting messages for current position."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        user_node = ConversationNode(
            message=LLMMessage(role="user", content="Hello"),
            node_id="0"
        )
        tree.add_child(Tree(user_node))

        state = ConversationState(
            conversation_id="conv-123",
            message_tree=tree,
            current_node_id="0"
        )

        messages = state.get_current_messages()
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    def test_state_serialization_simple(self):
        """Test serializing simple state."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        state = ConversationState(
            conversation_id="conv-123",
            message_tree=tree,
            current_node_id="",
            metadata={"test": "data"}
        )

        # Serialize
        data = state.to_dict()
        assert data["conversation_id"] == "conv-123"
        assert data["current_node_id"] == ""
        assert data["metadata"]["test"] == "data"
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 0

        # Deserialize
        restored = ConversationState.from_dict(data)
        assert restored.conversation_id == state.conversation_id
        assert restored.current_node_id == state.current_node_id
        assert restored.metadata == state.metadata

    def test_state_serialization_with_history(self):
        """Test serializing state with message history."""
        # Build tree
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        user_node = ConversationNode(
            message=LLMMessage(role="user", content="Hello"),
            node_id="0"
        )
        user_tree = tree.add_child(Tree(user_node))

        assistant_node = ConversationNode(
            message=LLMMessage(role="assistant", content="Hi!"),
            node_id="0.0"
        )
        user_tree.add_child(Tree(assistant_node))

        state = ConversationState(
            conversation_id="conv-123",
            message_tree=tree,
            current_node_id="0.0"
        )

        # Serialize
        data = state.to_dict()
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2

        # Deserialize
        restored = ConversationState.from_dict(data)
        messages = restored.get_current_messages()
        assert len(messages) == 3
        assert messages[0].content == "System"
        assert messages[1].content == "Hello"
        assert messages[2].content == "Hi!"

    def test_state_serialization_with_branching(self):
        """Test serializing state with branched conversations."""
        # Build tree with branches
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        user_node = ConversationNode(
            message=LLMMessage(role="user", content="Hello"),
            node_id="0"
        )
        user_tree = tree.add_child(Tree(user_node))

        # Two alternative assistant responses
        assistant1_node = ConversationNode(
            message=LLMMessage(role="assistant", content="Response A"),
            node_id="0.0",
            branch_name="variant-a"
        )
        user_tree.add_child(Tree(assistant1_node))

        assistant2_node = ConversationNode(
            message=LLMMessage(role="assistant", content="Response B"),
            node_id="0.1",
            branch_name="variant-b"
        )
        user_tree.add_child(Tree(assistant2_node))

        state = ConversationState(
            conversation_id="conv-123",
            message_tree=tree,
            current_node_id="0.0"
        )

        # Serialize
        data = state.to_dict()
        assert len(data["nodes"]) == 4  # system, user, 2 assistants
        assert len(data["edges"]) == 3

        # Deserialize
        restored = ConversationState.from_dict(data)

        # Check current path (variant-a)
        messages_a = get_messages_for_llm(restored.message_tree, "0.0")
        assert len(messages_a) == 3
        assert messages_a[2].content == "Response A"

        # Check alternative path (variant-b)
        messages_b = get_messages_for_llm(restored.message_tree, "0.1")
        assert len(messages_b) == 3
        assert messages_b[2].content == "Response B"

    def test_state_serialization_deep_branching(self):
        """Test serializing state with deep branching."""
        # Build complex tree
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        # First exchange
        user1_tree = tree.add_child(Tree(ConversationNode(
            message=LLMMessage(role="user", content="Q1"),
            node_id="0"
        )))
        asst1_tree = user1_tree.add_child(Tree(ConversationNode(
            message=LLMMessage(role="assistant", content="A1"),
            node_id="0.0"
        )))

        # Second exchange with branching
        user2a_tree = asst1_tree.add_child(Tree(ConversationNode(
            message=LLMMessage(role="user", content="Q2a"),
            node_id="0.0.0"
        )))
        user2a_tree.add_child(Tree(ConversationNode(
            message=LLMMessage(role="assistant", content="A2a"),
            node_id="0.0.0.0"
        )))

        user2b_tree = asst1_tree.add_child(Tree(ConversationNode(
            message=LLMMessage(role="user", content="Q2b"),
            node_id="0.0.1"
        )))
        user2b_tree.add_child(Tree(ConversationNode(
            message=LLMMessage(role="assistant", content="A2b"),
            node_id="0.0.1.0"
        )))

        state = ConversationState(
            conversation_id="conv-123",
            message_tree=tree,
            current_node_id="0.0.0.0"
        )

        # Serialize and deserialize
        data = state.to_dict()
        restored = ConversationState.from_dict(data)

        # Verify both branches exist
        path_a = get_messages_for_llm(restored.message_tree, "0.0.0.0")
        assert len(path_a) == 5
        assert path_a[3].content == "Q2a"
        assert path_a[4].content == "A2a"

        path_b = get_messages_for_llm(restored.message_tree, "0.0.1.0")
        assert len(path_b) == 5
        assert path_b[3].content == "Q2b"
        assert path_b[4].content == "A2b"


@pytest.mark.asyncio
class TestDataknobsConversationStorage:
    """Tests for DataknobsConversationStorage with real backends."""

    async def test_save_and_load_conversation(self):
        """Test saving and loading a conversation."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        # Create storage with memory backend
        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        # Create conversation state
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)
        state = ConversationState(
            conversation_id="conv-123",
            message_tree=tree,
            current_node_id="",
            metadata={"user_id": "alice"}
        )

        # Save conversation
        await storage.save_conversation(state)

        # Load conversation
        loaded = await storage.load_conversation("conv-123")
        assert loaded is not None
        assert loaded.conversation_id == "conv-123"
        assert loaded.metadata["user_id"] == "alice"

    async def test_save_and_load_with_history(self):
        """Test saving and loading conversation with message history."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        # Build conversation tree
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        user_node = ConversationNode(
            message=LLMMessage(role="user", content="Hello"),
            node_id="0"
        )
        user_tree = tree.add_child(Tree(user_node))

        assistant_node = ConversationNode(
            message=LLMMessage(role="assistant", content="Hi!"),
            node_id="0.0"
        )
        user_tree.add_child(Tree(assistant_node))

        state = ConversationState(
            conversation_id="conv-456",
            message_tree=tree,
            current_node_id="0.0"
        )

        # Save and load
        await storage.save_conversation(state)
        loaded = await storage.load_conversation("conv-456")

        assert loaded is not None
        messages = loaded.get_current_messages()
        assert len(messages) == 3
        assert messages[0].content == "System"
        assert messages[1].content == "Hello"
        assert messages[2].content == "Hi!"

    async def test_save_and_load_with_branching(self):
        """Test conversation with branches."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        # Build branched tree
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)

        user_node = ConversationNode(
            message=LLMMessage(role="user", content="Hello"),
            node_id="0"
        )
        user_tree = tree.add_child(Tree(user_node))

        # Two alternative responses
        assistant1_node = ConversationNode(
            message=LLMMessage(role="assistant", content="Response A"),
            node_id="0.0",
            branch_name="variant-a"
        )
        user_tree.add_child(Tree(assistant1_node))

        assistant2_node = ConversationNode(
            message=LLMMessage(role="assistant", content="Response B"),
            node_id="0.1",
            branch_name="variant-b"
        )
        user_tree.add_child(Tree(assistant2_node))

        state = ConversationState(
            conversation_id="conv-789",
            message_tree=tree,
            current_node_id="0.0"
        )

        # Save and load
        await storage.save_conversation(state)
        loaded = await storage.load_conversation("conv-789")

        assert loaded is not None

        # Verify both branches exist
        messages_a = get_messages_for_llm(loaded.message_tree, "0.0")
        assert len(messages_a) == 3
        assert messages_a[2].content == "Response A"

        messages_b = get_messages_for_llm(loaded.message_tree, "0.1")
        assert len(messages_b) == 3
        assert messages_b[2].content == "Response B"

    async def test_update_conversation(self):
        """Test updating an existing conversation."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        # Create initial state
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)
        state = ConversationState(
            conversation_id="conv-update",
            message_tree=tree,
            current_node_id="",
            metadata={"version": 1}
        )

        # Save
        await storage.save_conversation(state)

        # Add message and update
        user_node = ConversationNode(
            message=LLMMessage(role="user", content="New message"),
            node_id="0"
        )
        tree.add_child(Tree(user_node))
        state.current_node_id = "0"
        state.metadata["version"] = 2

        # Save again (upsert)
        await storage.save_conversation(state)

        # Load and verify
        loaded = await storage.load_conversation("conv-update")
        assert loaded is not None
        assert loaded.metadata["version"] == 2
        messages = loaded.get_current_messages()
        assert len(messages) == 2

    async def test_delete_conversation(self):
        """Test deleting a conversation."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        # Create and save
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="System"),
            node_id=""
        )
        tree = Tree(root_node)
        state = ConversationState(
            conversation_id="conv-delete",
            message_tree=tree
        )
        await storage.save_conversation(state)

        # Verify it exists
        loaded = await storage.load_conversation("conv-delete")
        assert loaded is not None

        # Delete
        result = await storage.delete_conversation("conv-delete")
        assert result is True

        # Verify it's gone
        loaded = await storage.load_conversation("conv-delete")
        assert loaded is None

    async def test_delete_nonexistent_conversation(self):
        """Test deleting a conversation that doesn't exist."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        result = await storage.delete_conversation("nonexistent")
        assert result is False

    async def test_list_conversations(self):
        """Test listing all conversations."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        # Create multiple conversations
        for i in range(3):
            root_node = ConversationNode(
                message=LLMMessage(role="system", content=f"System {i}"),
                node_id=""
            )
            tree = Tree(root_node)
            state = ConversationState(
                conversation_id=f"conv-list-{i}",
                message_tree=tree,
                metadata={"index": i}
            )
            await storage.save_conversation(state)

        # List all
        conversations = await storage.list_conversations()
        assert len(conversations) == 3
        assert all(c.conversation_id.startswith("conv-list-") for c in conversations)

    async def test_list_conversations_with_filters(self):
        """Test listing conversations with metadata filters."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage
        import uuid

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        # Create conversations with different users
        for idx, user in enumerate(["alice", "bob", "alice"]):
            root_node = ConversationNode(
                message=LLMMessage(role="system", content="System"),
                node_id=""
            )
            tree = Tree(root_node)
            state = ConversationState(
                conversation_id=f"conv-{user}-{uuid.uuid4()}",  # Use UUID for uniqueness
                message_tree=tree,
                metadata={"user_id": user}
            )
            await storage.save_conversation(state)

        # Filter for alice
        alice_convs = await storage.list_conversations(
            filter_metadata={"user_id": "alice"}
        )
        assert len(alice_convs) == 2
        assert all(c.metadata["user_id"] == "alice" for c in alice_convs)

    async def test_list_conversations_with_limit(self):
        """Test listing conversations with limit."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        # Create 5 conversations
        for i in range(5):
            root_node = ConversationNode(
                message=LLMMessage(role="system", content="System"),
                node_id=""
            )
            tree = Tree(root_node)
            state = ConversationState(
                conversation_id=f"conv-limit-{i}",
                message_tree=tree
            )
            await storage.save_conversation(state)

        # List with limit
        conversations = await storage.list_conversations(limit=3)
        assert len(conversations) == 3

    async def test_load_nonexistent_conversation(self):
        """Test loading a conversation that doesn't exist."""
        from dataknobs_data.backends import AsyncMemoryDatabase
        from dataknobs_llm.conversations import DataknobsConversationStorage

        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        loaded = await storage.load_conversation("nonexistent")
        assert loaded is None

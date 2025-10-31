"""Test ConversationManager functionality."""

import pytest
from pathlib import Path
import tempfile
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import LLMConfig, EchoProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_data.backends.memory import AsyncMemoryDatabase


def create_test_prompts(prompt_dir: Path):
    """Create test prompt files."""
    import yaml

    # System prompts
    system_dir = prompt_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    (system_dir / "helpful.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
    )
    (system_dir / "assistant.yaml").write_text(
        yaml.dump({"template": "You are an AI assistant"})
    )

    # User prompts
    user_dir = prompt_dir / "user"
    user_dir.mkdir(parents=True, exist_ok=True)

    (user_dir / "question.yaml").write_text(
        yaml.dump({"template": "What is {{topic}}?"})
    )
    (user_dir / "followup_question.yaml").write_text(
        yaml.dump({"template": "Tell me more about {{topic}}."})
    )
    (user_dir / "greeting.yaml").write_text(
        yaml.dump({"template": "Hello!"})
    )


@pytest.fixture
async def test_components():
    """Create test LLM, builder, and storage."""
    # Create temporary directory for prompts
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        create_test_prompts(prompt_dir)

        # Create LLM provider
        config = LLMConfig(
            provider="echo",
            model="echo-model",
            options={"echo_prefix": ""}  # No prefix for cleaner test output
        )
        llm = EchoProvider(config)

        # Create prompt library and builder
        library = FileSystemPromptLibrary(prompt_dir)
        builder = AsyncPromptBuilder(library=library)

        # Create storage
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        yield {
            "llm": llm,
            "builder": builder,
            "storage": storage
        }

        # Cleanup
        await llm.close()


class TestConversationManager:
    """Test ConversationManager functionality."""

    @pytest.mark.asyncio
    async def test_create_with_system_prompt(self, test_components):
        """Test creating conversation with system prompt."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            system_prompt_name="helpful",
        )

        assert manager.conversation_id is not None
        assert manager.current_node_id == ""

        # Verify system message was added
        history = await manager.get_history()
        assert len(history) == 1
        assert history[0].role == "system"
        assert history[0].content == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_basic_conversation_flow(self, test_components):
        """Test basic conversation: system -> user -> assistant -> user -> assistant."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            system_prompt_name="assistant",
        )

        # Add user message
        await manager.add_message(role="user", content="What is Python?")

        # Get LLM response (EchoProvider will echo back the user message)
        response = await manager.complete()
        assert "What is Python?" in response.content

        # Check history
        history = await manager.get_history()
        assert len(history) == 3  # system, user, assistant
        assert history[0].role == "system"
        assert history[1].role == "user"
        assert history[1].content == "What is Python?"
        assert history[2].role == "assistant"

        # Continue conversation
        await manager.add_message(role="user", content="Tell me about decorators")
        response = await manager.complete()
        assert "Tell me about decorators" in response.content

        # Check final history
        history = await manager.get_history()
        assert len(history) == 5  # system, user, assistant, user, assistant

    @pytest.mark.asyncio
    async def test_add_message_with_prompt(self, test_components):
        """Test adding message using prompt template."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        # Add message using prompt
        node = await manager.add_message(
            role="user", prompt_name="question", params={"topic": "Python"}
        )

        assert "Python" in node.message.content
        assert node.prompt_name == "question"

        history = await manager.get_history()
        assert len(history) == 1
        assert "Python" in history[0].content

    @pytest.mark.asyncio
    async def test_branching_conversation(self, test_components):
        """Test creating alternative branches in conversation."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        # Add user message
        await manager.add_message(role="user", content="Hello")
        user_node_id = manager.current_node_id  # Should be "0"

        # Get first response
        await manager.complete(branch_name="branch-a")
        branch_a_node_id = manager.current_node_id  # Should be "0.0"

        # Go back to user message
        await manager.switch_to_node(user_node_id)
        assert manager.current_node_id == user_node_id

        # Get alternative response
        await manager.complete(branch_name="branch-b")
        branch_b_node_id = manager.current_node_id  # Should be "0.1"

        # Verify we have two branches
        branches = await manager.get_branches(user_node_id)
        assert len(branches) == 2
        assert branches[0]["branch_name"] == "branch-a"
        assert branches[0]["node_id"] == branch_a_node_id
        assert branches[1]["branch_name"] == "branch-b"
        assert branches[1]["node_id"] == branch_b_node_id

        # Check history for each branch
        await manager.switch_to_node(branch_a_node_id)
        history_a = await manager.get_history()
        assert len(history_a) == 2

        await manager.switch_to_node(branch_b_node_id)
        history_b = await manager.get_history()
        assert len(history_b) == 2

    @pytest.mark.asyncio
    async def test_resume_conversation(self, test_components):
        """Test resuming an existing conversation."""
        # Create initial conversation
        manager1 = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )
        await manager1.add_message(role="system", content="You are helpful")
        await manager1.add_message(role="user", content="Hello")
        await manager1.complete()

        conversation_id = manager1.conversation_id

        # Resume conversation
        manager2 = await ConversationManager.resume(
            conversation_id=conversation_id,
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        # Verify state was restored
        assert manager2.conversation_id == conversation_id
        history = await manager2.get_history()
        assert len(history) == 3
        assert history[0].content == "You are helpful"
        assert history[1].content == "Hello"

        # Continue conversation
        await manager2.add_message(role="user", content="How are you?")
        await manager2.complete()

        history = await manager2.get_history()
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_resume_nonexistent_conversation(self, test_components):
        """Test that resuming nonexistent conversation raises error."""
        with pytest.raises(ValueError, match="not found"):
            await ConversationManager.resume(
                conversation_id="nonexistent",
                llm=test_components["llm"],
                prompt_builder=test_components["builder"],
                storage=test_components["storage"],
            )

    @pytest.mark.asyncio
    async def test_stream_complete(self, test_components):
        """Test streaming LLM completion."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )
        await manager.add_message(role="user", content="Hello")

        # Stream response
        chunks = []
        async for chunk in manager.stream_complete():
            chunks.append(chunk.delta)

        # Verify chunks were received
        full_content = "".join(chunks)
        assert len(full_content) > 0
        assert "Hello" in full_content

        # Verify message was added to history
        history = await manager.get_history()
        assert len(history) == 2
        assert history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_switch_to_invalid_node(self, test_components):
        """Test switching to invalid node raises error."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )
        # Add a message to initialize the state properly
        await manager.add_message(role="user", content="Hello")

        with pytest.raises(ValueError, match="not found"):
            await manager.switch_to_node("99.99")

    @pytest.mark.asyncio
    async def test_complete_without_messages(self, test_components):
        """Test that complete() without messages raises error."""
        # Create manager without initializing state
        manager = ConversationManager(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            state=None,
        )

        with pytest.raises(ValueError, match="no messages"):
            await manager.complete()

    @pytest.mark.asyncio
    async def test_get_branches(self, test_components):
        """Test getting branch information."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )
        await manager.add_message(role="user", content="Question")
        user_node = manager.current_node_id

        # Create three branches
        await manager.complete(branch_name="first")
        first_node = manager.current_node_id

        await manager.switch_to_node(user_node)
        await manager.complete(branch_name="second")
        second_node = manager.current_node_id

        await manager.switch_to_node(user_node)
        await manager.complete(branch_name="third")
        third_node = manager.current_node_id

        # Get branches from user message
        branches = await manager.get_branches(user_node)
        assert len(branches) == 3

        # Verify branch info
        assert branches[0]["node_id"] == first_node
        assert branches[0]["branch_name"] == "first"
        assert branches[0]["role"] == "assistant"

        assert branches[1]["node_id"] == second_node
        assert branches[1]["branch_name"] == "second"

        assert branches[2]["node_id"] == third_node
        assert branches[2]["branch_name"] == "third"

    @pytest.mark.asyncio
    async def test_metadata_management(self, test_components):
        """Test adding and persisting metadata."""
        # Create with initial metadata
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            metadata={"user_id": "alice", "session": "abc123"},
        )
        await manager.add_message(role="user", content="Hello")

        # Add more metadata
        await manager.add_metadata("rating", 5)
        await manager.add_metadata("language", "en")

        # Resume and verify metadata persisted
        conversation_id = manager.conversation_id
        manager2 = await ConversationManager.resume(
            conversation_id=conversation_id,
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        assert manager2.state.metadata["user_id"] == "alice"
        assert manager2.state.metadata["session"] == "abc123"
        assert manager2.state.metadata["rating"] == 5
        assert manager2.state.metadata["language"] == "en"

    @pytest.mark.asyncio
    async def test_node_metadata(self, test_components):
        """Test adding metadata to individual message nodes."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        # Add message with metadata
        node = await manager.add_message(
            role="user",
            content="Hello",
            metadata={"source": "web", "confidence": 0.95},
        )

        assert node.metadata["source"] == "web"
        assert node.metadata["confidence"] == 0.95

        # Complete with metadata
        response = await manager.complete(metadata={"temperature": 0.7})

        # Verify assistant message has metadata (including usage)
        current_node = manager.state.get_current_node()
        assert "usage" in current_node.data.metadata
        assert current_node.data.metadata["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_complex_branching_tree(self, test_components):
        """Test complex tree with multiple levels of branching."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        # Build tree:
        #          root
        #           |
        #          user1 (0)
        #         /  |  \
        #       a1  a2  a3 (0.0, 0.1, 0.2)
        #       |
        #      user2 (0.0.0)
        #       |
        #      a4 (0.0.0.0)

        await manager.add_message(role="user", content="Q1")
        q1_node = manager.current_node_id

        # First branch
        await manager.complete(branch_name="answer-1")
        a1_node = manager.current_node_id
        await manager.add_message(role="user", content="Q2")
        await manager.complete()

        # Second branch from Q1
        await manager.switch_to_node(q1_node)
        await manager.complete(branch_name="answer-2")

        # Third branch from Q1
        await manager.switch_to_node(q1_node)
        await manager.complete(branch_name="answer-3")

        # Verify structure
        branches_from_q1 = await manager.get_branches(q1_node)
        assert len(branches_from_q1) == 3

        # Navigate to deepest node and verify full path
        await manager.switch_to_node(a1_node)
        await manager.add_message(role="user", content="Q2")
        await manager.complete()

        history = await manager.get_history()
        assert len(history) == 4  # user1, answer-1, user2, answer-4

    @pytest.mark.asyncio
    async def test_add_message_requires_content_or_prompt(self, test_components):
        """Test that add_message requires either content or prompt_name."""
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
        )

        with pytest.raises(ValueError, match="Either content or prompt_name"):
            await manager.add_message(role="user")


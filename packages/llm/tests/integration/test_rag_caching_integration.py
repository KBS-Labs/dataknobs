"""Integration tests for end-to-end RAG caching in conversations."""

import pytest

from dataknobs_llm.conversations.manager import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.llm.providers import EchoProvider
from dataknobs_llm.prompts.implementations import ConfigPromptLibrary
from dataknobs_llm.prompts.builders import AsyncPromptBuilder
from dataknobs_llm.prompts.adapters import InMemoryAsyncAdapter

try:
    from dataknobs_data.backends import AsyncMemoryDatabase
    DATAKNOBS_DATA_AVAILABLE = True
except ImportError:
    DATAKNOBS_DATA_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not DATAKNOBS_DATA_AVAILABLE,
    reason="dataknobs-data not available"
)


class TestEndToEndRAGCaching:
    """Test complete RAG caching flow in conversations."""

    @pytest.mark.asyncio
    async def test_complete_rag_caching_workflow(self):
        """Test complete RAG caching workflow with branching."""
        # Setup: Create conversation with RAG caching enabled
        config = {
            "system": {
                "assistant": {
                    "template": "You are a helpful coding assistant."
                }
            },
            "user": {
                "code_question": {
                    "template": """Context from documentation:
{{DOCS}}

User question: {{question}}""",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "{{language}} {{topic}}",
                            "k": 5,
                            "placeholder": "DOCS",
                            "header": "# Relevant Documentation\n\n",
                            "item_template": "- {{content}}\n"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)

        # InMemory documentation adapter
        docs_adapter = InMemoryAsyncAdapter(
            search_results=[
                {"content": "Python decorators are functions that modify other functions", "score": 0.95},
                {"content": "Use @decorator syntax to apply decorators", "score": 0.90},
                {"content": "Decorators can take arguments", "score": 0.85},
            ],
            name="docs"
        )

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": docs_adapter}
        )

        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="assistant",
            cache_rag_results=True,
            reuse_rag_on_branch=True
        )

        # Step 1: Add first user question with RAG
        await manager.add_message(
            role="user",
            prompt_name="code_question",
            params={
                "question": "How do I use decorators?",
                "language": "python",
                "topic": "decorators"
            }
        )

        # Verify RAG search was executed
        assert docs_adapter.search_count == 1

        # Verify RAG metadata was stored
        user_node_1 = manager.state.get_current_node().data
        assert "rag_metadata" in user_node_1.metadata
        rag_meta_1 = user_node_1.metadata["rag_metadata"]
        assert "DOCS" in rag_meta_1
        assert rag_meta_1["DOCS"]["query"] == "python decorators"
        assert rag_meta_1["DOCS"]["k"] == 5
        assert len(rag_meta_1["DOCS"]["results"]) == 3
        assert "query_hash" in rag_meta_1["DOCS"]

        # Step 2: Get assistant response
        await manager.complete()

        # Step 3: Add follow-up question
        await manager.add_message(
            role="user",
            prompt_name="code_question",
            params={
                "question": "Can you show an example?",
                "language": "python",
                "topic": "decorator examples"
            }
        )

        # Verify new RAG search was executed (different query)
        assert docs_adapter.search_count == 2

        await manager.complete()

        # Step 4: Branch back to first question
        first_user_node_id = "0"  # First user message
        await manager.switch_to_node(first_user_node_id)

        # Step 5: Add alternative question with SAME parameters as first question
        docs_adapter.reset()  # Reset call count
        await manager.add_message(
            role="user",
            prompt_name="code_question",
            params={
                "question": "How do I use decorators?",  # Same query parameters
                "language": "python",
                "topic": "decorators"
            }
        )

        # Verify RAG search was NOT executed (cache was reused!)
        assert docs_adapter.search_count == 0

        # Verify the new node has RAG metadata (passed through from cache)
        branched_node = manager.state.get_current_node().data
        assert "rag_metadata" in branched_node.metadata
        rag_meta_branched = branched_node.metadata["rag_metadata"]
        assert "DOCS" in rag_meta_branched

        # Verify the cached RAG metadata matches the original
        assert rag_meta_branched["DOCS"]["query"] == "python decorators"
        assert rag_meta_branched["DOCS"]["query_hash"] == rag_meta_1["DOCS"]["query_hash"]

    @pytest.mark.asyncio
    async def test_rag_caching_with_multiple_adapters(self):
        """Test RAG caching with multiple adapters and placeholders."""
        config = {
            "system": {"sys": {"template": "System"}},
            "user": {
                "research": {
                    "template": """Documentation:
{{DOCS}}

Examples:
{{EXAMPLES}}

Question: {{question}}""",
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "{{topic}} documentation",
                            "placeholder": "DOCS"
                        },
                        {
                            "adapter_name": "examples",
                            "query": "{{topic}} code examples",
                            "placeholder": "EXAMPLES"
                        }
                    ]
                }
            }
        }

        library = ConfigPromptLibrary(config)

        # InMemory adapters
        docs_adapter = InMemoryAsyncAdapter(
            search_results=[{"content": "Doc 1"}],
            name="docs"
        )

        examples_adapter = InMemoryAsyncAdapter(
            search_results=[{"content": "Example 1"}],
            name="examples"
        )

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": docs_adapter, "examples": examples_adapter}
        )

        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="sys",
            cache_rag_results=True,
            reuse_rag_on_branch=True
        )

        # Add message - both adapters should be called
        await manager.add_message(
            role="user",
            prompt_name="research",
            params={"question": "How?", "topic": "async"}
        )

        assert docs_adapter.search_count == 1
        assert examples_adapter.search_count == 1

        # Verify both RAG configs captured
        node = manager.state.get_current_node().data
        rag_meta = node.metadata["rag_metadata"]
        assert "DOCS" in rag_meta
        assert "EXAMPLES" in rag_meta

        # Branch and reuse
        await manager.complete()
        await manager.switch_to_node("0")

        docs_adapter.reset()
        examples_adapter.reset()

        await manager.add_message(
            role="user",
            prompt_name="research",
            params={"question": "How?", "topic": "async"}
        )

        # Neither adapter should be called (both cached)
        assert docs_adapter.search_count == 0
        assert examples_adapter.search_count == 0

    @pytest.mark.asyncio
    async def test_rag_metadata_inspection(self):
        """Test inspecting RAG metadata from conversation history."""
        config = {
            "system": {"sys": {"template": "System"}},
            "user": {
                "q": {
                    "template": "{{RAG}}",
                    "rag_configs": [{
                        "adapter_name": "docs",
                        "query": "test query",
                        "placeholder": "RAG"
                    }]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAsyncAdapter(
            search_results=[
                {"content": "Result 1", "score": 0.9, "metadata": {"source": "doc1.md"}},
                {"content": "Result 2", "score": 0.8, "metadata": {"source": "doc2.md"}}
            ],
            name="docs"
        )

        builder = AsyncPromptBuilder(library=library, adapters={"docs": adapter})
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="sys",
            cache_rag_results=True
        )

        # Add message
        await manager.add_message(role="user", prompt_name="q", params={})

        # Inspect RAG metadata
        metadata = manager.get_rag_metadata()

        assert metadata is not None
        assert "RAG" in metadata

        rag_data = metadata["RAG"]
        assert rag_data["adapter_name"] == "docs"
        assert rag_data["query"] == "test query"
        assert len(rag_data["results"]) == 2
        assert rag_data["results"][0]["content"] == "Result 1"
        assert rag_data["results"][0]["metadata"]["source"] == "doc1.md"
        assert "formatted_content" in rag_data
        assert "timestamp" in rag_data

    @pytest.mark.asyncio
    async def test_rag_caching_disabled(self):
        """Test that RAG searches happen every time when caching is disabled."""
        config = {
            "system": {"sys": {"template": "System"}},
            "user": {
                "q": {
                    "template": "{{RAG}}",
                    "rag_configs": [{
                        "adapter_name": "docs",
                        "query": "test",
                        "placeholder": "RAG"
                    }]
                }
            }
        }

        library = ConfigPromptLibrary(config)
        adapter = InMemoryAsyncAdapter(
            search_results=[{"content": "Result"}],
            name="docs"
        )

        builder = AsyncPromptBuilder(library=library, adapters={"docs": adapter})
        llm = EchoProvider(config={"provider": "echo", "model": "echo"})
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        # Create manager WITHOUT caching
        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="sys",
            cache_rag_results=False,
            reuse_rag_on_branch=False
        )

        # Add message
        await manager.add_message(role="user", prompt_name="q", params={})
        assert adapter.search_count ==1

        # Branch and add again
        await manager.complete()
        await manager.switch_to_node("0")
        await manager.add_message(role="user", prompt_name="q", params={})

        # Search should be called again (no caching)
        assert adapter.search_count ==2

        # Verify no RAG metadata stored
        metadata = manager.get_rag_metadata()
        assert metadata is None

"""RAG (Retrieval Augmented Generation) chatbot example.

This example demonstrates:
- Knowledge base integration
- Document chunking and indexing
- Vector search and retrieval
- Context-aware responses using RAG
- Swapping storage backends (memory vs postgres)

Required Ollama models:
    ollama pull gemma3:1b           # For chat
    ollama pull nomic-embed-text    # For embeddings

Usage:
    # With in-memory storage (default)
    python examples/03_rag_chatbot.py

    # With PostgreSQL storage
    STORAGE_BACKEND=postgres python examples/03_rag_chatbot.py
"""

import asyncio
import os

from dataknobs_bots import BotContext, DynaBot


# Sample documents for the knowledge base
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "text": """
        Python is a high-level, interpreted programming language known for its
        simplicity and readability. Created by Guido van Rossum and first released
        in 1991, Python emphasizes code readability with its use of significant
        indentation. It supports multiple programming paradigms including
        procedural, object-oriented, and functional programming.
        """,
        "metadata": {"source": "python_intro", "category": "programming"},
    },
    {
        "id": "doc2",
        "text": """
        The DataKnobs ecosystem is a collection of Python packages designed for
        building data-intensive applications. It includes modules for database
        abstraction, LLM integration, configuration management, and state machines.
        The ecosystem prioritizes modularity, type safety, and ease of use.
        """,
        "metadata": {"source": "dataknobs_intro", "category": "framework"},
    },
    {
        "id": "doc3",
        "text": """
        Retrieval Augmented Generation (RAG) is a technique that combines information
        retrieval with large language models. It works by first retrieving relevant
        documents from a knowledge base, then using those documents as context for
        generating responses. This approach helps reduce hallucinations and provides
        more accurate, grounded answers.
        """,
        "metadata": {"source": "rag_explained", "category": "ai"},
    },
    {
        "id": "doc4",
        "text": """
        Vector databases store data as high-dimensional vectors (embeddings) that
        represent the semantic meaning of text. They enable similarity search,
        allowing you to find documents that are semantically similar to a query
        even if they don't share exact keywords. This is fundamental to modern
        information retrieval systems.
        """,
        "metadata": {"source": "vector_db_intro", "category": "database"},
    },
]


async def main():
    """Run a RAG chatbot conversation."""
    print("=" * 60)
    print("RAG Chatbot Example")
    print("=" * 60)
    print()
    print("This example shows a chatbot with knowledge base integration.")
    print("Required: ollama pull gemma3:1b nomic-embed-text")
    print()

    # Determine storage backend from environment
    storage_backend = os.getenv("STORAGE_BACKEND", "memory")
    print(f"Storage backend: {storage_backend}")
    print()

    # Base configuration
    storage_config = {"backend": storage_backend}

    # Add postgres-specific config if needed
    if storage_backend == "postgres":
        storage_config.update(
            {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                "database": os.getenv("POSTGRES_DB", "dynabot_examples"),
                "table": "rag_conversations",
                "schema": "public",
            }
        )

    # Configuration with knowledge base
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:1b",
            "temperature": 0.7,
            "max_tokens": 500,
        },
        "conversation_storage": storage_config,
        "knowledge_base": {
            "enabled": True,
            "provider": "vector",  # Use vector-based knowledge base
            "embedding_model": "nomic-embed-text",
            "embedding_provider": "ollama",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "top_k": 3,  # Retrieve top 3 relevant documents
        },
        "prompts": {
            "rag_assistant": "You are a knowledgeable AI assistant. "
            "When answering questions, use the provided knowledge context to give "
            "accurate, detailed responses. If the context doesn't contain relevant "
            "information, say so honestly."
        },
        "system_prompt": {
            "name": "rag_assistant",
        },
    }

    print("Creating bot with knowledge base...")
    bot = await DynaBot.from_config(config)
    print("✓ Bot created successfully")
    print("✓ Knowledge base enabled (vector-based)")
    print()

    # Index documents into knowledge base
    print("Indexing sample documents...")
    for doc in SAMPLE_DOCUMENTS:
        await bot.knowledge_base.add_document(
            doc_id=doc["id"],
            text=doc["text"],
            metadata=doc["metadata"],
        )
    print(f"✓ Indexed {len(SAMPLE_DOCUMENTS)} documents")
    print()

    # Create context for this conversation
    context = BotContext(
        conversation_id="rag-chat-001",
        client_id="example-client",
        user_id="demo-user",
    )

    # Questions that should be answered using the knowledge base
    questions = [
        "What is Python and who created it?",
        "Can you explain what RAG is and how it works?",
        "What is the DataKnobs ecosystem?",
        "How do vector databases work?",
        "What's the weather like today?",  # Not in knowledge base
    ]

    for i, question in enumerate(questions, 1):
        print(f"[{i}] User: {question}")

        response = await bot.chat(
            message=question,
            context=context,
        )

        print(f"[{i}] Bot: {response}")
        print()

        # Add a small delay between messages
        if i < len(questions):
            await asyncio.sleep(1)

    print("=" * 60)
    print("RAG demonstration complete!")
    print()
    print("Notice how the bot:")
    print("- Used knowledge base to answer questions 1-4 accurately")
    print("- Admitted when information wasn't in the knowledge base (question 5)")
    print("- Retrieved relevant context before generating responses")
    print()
    print(f"Storage backend used: {storage_backend}")
    if storage_backend == "memory":
        print("To use PostgreSQL: STORAGE_BACKEND=postgres python examples/03_rag_chatbot.py")


if __name__ == "__main__":
    asyncio.run(main())

"""Simple chatbot example.

This example demonstrates:
- Basic DynaBot configuration
- Using Ollama as the LLM provider
- In-memory conversation storage
- Simple message exchange

Required Ollama model:
    ollama pull gemma3:1b
"""

import asyncio

from dataknobs_bots import BotContext, DynaBot


async def main():
    """Run a simple chatbot conversation."""
    print("=" * 60)
    print("Simple Chatbot Example")
    print("=" * 60)
    print()
    print("This example shows a basic chatbot with no memory.")
    print("Required: ollama pull gemma3:1b")
    print()

    # Configuration for a simple chatbot
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:1b",
            "temperature": 0.7,
            "max_tokens": 500,
        },
        "conversation_storage": {
            "backend": "memory",
        },
        "prompts": {
            "friendly_assistant": "You are a friendly and helpful AI assistant. "
            "Keep your responses concise and clear."
        },
        "system_prompt": {
            "name": "friendly_assistant",
        },
    }

    print("Creating bot from configuration...")
    bot = await DynaBot.from_config(config)
    print("✓ Bot created successfully")
    print()

    # Create context for this conversation
    context = BotContext(
        conversation_id="simple-chat-001",
        client_id="example-client",
        user_id="demo-user",
    )

    # Example conversation
    messages = [
        "Hello! What can you help me with?",
        "Tell me a fun fact about Python programming.",
        "That's interesting! What makes Python so popular?",
    ]

    for i, user_message in enumerate(messages, 1):
        print(f"User: {user_message}")

        response = await bot.chat(
            message=user_message,
            context=context,
        )

        print(f"Bot: {response}")
        print()

        # Add a small delay between messages
        if i < len(messages):
            await asyncio.sleep(1)

    print("=" * 60)
    print("Conversation complete!")
    print()
    print("Note: This bot has no memory between conversations.")
    print("Each new conversation starts fresh.")


if __name__ == "__main__":
    asyncio.run(main())

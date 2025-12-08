"""Chatbot with memory example.

This example demonstrates:
- Buffer memory configuration
- Context retention across messages
- Memory limits and management
- How the bot remembers previous conversation

Required Ollama model:
    ollama pull gemma3:1b
"""

import asyncio

from dataknobs_bots import BotContext, DynaBot


async def main():
    """Run a chatbot with memory."""
    print("=" * 60)
    print("Chatbot with Memory Example")
    print("=" * 60)
    print()
    print("This example shows a chatbot that remembers context.")
    print("Required: ollama pull gemma3:1b")
    print()

    # Configuration with buffer memory
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
        "memory": {
            "type": "buffer",
            "max_messages": 10,  # Remember last 10 messages
        },
        "prompts": {
            "helpful_assistant": "You are a helpful AI assistant with excellent memory. "
            "You remember details from earlier in the conversation and can reference them."
        },
        "system_prompt": {
            "name": "helpful_assistant",
        },
    }

    print("Creating bot with buffer memory...")
    bot = await DynaBot.from_config(config)
    print("✓ Bot created successfully")
    print(f"✓ Memory: Buffer (max {config['memory']['max_messages']} messages)")
    print()

    # Create context for this conversation
    context = BotContext(
        conversation_id="memory-chat-001",
        client_id="example-client",
        user_id="demo-user",
    )

    # Conversation demonstrating memory
    messages = [
        "Hello! My name is Alice and I love reading science fiction.",
        "What's your favorite sci-fi book?",
        "Do you remember my name?",
        "What did I tell you I love to read?",
        "Can you recommend a sci-fi book for me based on what you know about my interests?",
    ]

    for i, user_message in enumerate(messages, 1):
        print(f"[{i}] User: {user_message}")

        response = await bot.chat(
            message=user_message,
            context=context,
        )

        print(f"[{i}] Bot: {response}")
        print()

        # Add a small delay between messages
        if i < len(messages):
            await asyncio.sleep(1)

    print("=" * 60)
    print("Conversation complete!")
    print()
    print("Memory demonstration:")
    print("- The bot remembered the user's name (Alice)")
    print("- The bot remembered the user's interest (science fiction)")
    print("- The bot used this context to make relevant recommendations")
    print()
    print(f"Memory buffer stores last {config['memory']['max_messages']} messages")


if __name__ == "__main__":
    asyncio.run(main())

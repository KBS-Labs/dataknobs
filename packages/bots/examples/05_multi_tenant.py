"""Multi-tenant bot example.

This example demonstrates:
- Single bot instance serving multiple clients
- Conversation isolation per client
- Concurrent conversation handling
- Shared bot configuration
- Client-specific contexts

Required Ollama model:
    ollama pull gemma3:1b
"""

import asyncio

from dataknobs_bots import BotContext, DynaBot


async def customer_conversation(bot: DynaBot, client_id: str, customer_name: str):
    """Simulate a customer conversation.

    Args:
        bot: Shared bot instance
        client_id: Unique client identifier
        customer_name: Customer name for display
    """
    # Create unique context for this customer
    context = BotContext(
        conversation_id=f"conv-{client_id}",
        client_id=client_id,
        user_id=f"user-{customer_name.lower().replace(' ', '-')}",
        session_metadata={"customer_name": customer_name},
    )

    print(f"\n[{customer_name}] Starting conversation...")

    # Each customer asks different questions
    if client_id == "client-1":
        messages = [
            "Hello! I'm interested in your product.",
            "What are the main features?",
            "That sounds great, thank you!",
        ]
    elif client_id == "client-2":
        messages = [
            "Hi, I have a question about pricing.",
            "Do you offer enterprise discounts?",
            "Perfect, I'll be in touch!",
        ]
    else:
        messages = [
            "Good morning! I need technical support.",
            "How do I integrate your API?",
            "Thanks for the information!",
        ]

    for i, message in enumerate(messages, 1):
        print(f"[{customer_name}] User: {message}")

        # Each customer uses the same bot instance with their own context
        response = await bot.chat(
            message=message,
            context=context,
        )

        print(f"[{customer_name}] Bot: {response}")

        # Small delay between messages
        if i < len(messages):
            await asyncio.sleep(0.5)

    print(f"[{customer_name}] Conversation complete")


async def main():
    """Run multi-tenant bot simulation."""
    print("=" * 60)
    print("Multi-Tenant Bot Example")
    print("=" * 60)
    print()
    print("This example shows one bot instance serving multiple clients.")
    print("Required: ollama pull gemma3:1b")
    print()

    # Configuration for a multi-tenant bot
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
            "max_messages": 10,
        },
        "prompts": {
            "customer_support": "You are a helpful customer support AI assistant. "
            "Provide friendly, professional responses to customer inquiries. "
            "Keep responses concise and helpful."
        },
        "system_prompt": {
            "name": "customer_support",
        },
    }

    print("Creating shared bot instance...")
    bot = await DynaBot.from_config(config)
    print("✓ Bot created successfully")
    print()

    print("Simulating 3 concurrent customer conversations...")
    print("(Each customer has isolated conversation state)")
    print()

    # Create tasks for concurrent conversations
    tasks = [
        customer_conversation(bot, "client-1", "Alice Johnson"),
        customer_conversation(bot, "client-2", "Bob Smith"),
        customer_conversation(bot, "client-3", "Charlie Davis"),
    ]

    # Run all conversations concurrently
    await asyncio.gather(*tasks)

    print()
    print("=" * 60)
    print("Multi-tenant demonstration complete!")
    print()
    print("Key observations:")
    print("- Single bot instance served 3 different clients concurrently")
    print("- Each client had isolated conversation state")
    print("- No conversation data leaked between clients")
    print("- The bot maintained separate context for each customer")
    print()
    print("In production:")
    print("- This pattern enables horizontal scaling")
    print("- Each container can serve thousands of concurrent clients")
    print("- Conversation storage can use PostgreSQL for persistence")
    print("- Client isolation is guaranteed by conversation_id and client_id")


if __name__ == "__main__":
    asyncio.run(main())

# Simple Chatbot Example

A basic conversational bot demonstrating core dataknobs-bots functionality.

## Overview

This example shows how to:

- Create a basic DynaBot from configuration
- Use Ollama as the LLM provider
- Set up in-memory conversation storage
- Have a simple conversation with the bot

## Prerequisites

```bash
# Install Ollama: https://ollama.ai/

# Pull the required model
ollama pull gemma3:1b

# Install dataknobs-bots
pip install dataknobs-bots
```

## Key Concepts

### Configuration

The bot is configured entirely through a dictionary:

```python
config = {
    "llm": {
        "provider": "ollama",  # Use Ollama for local inference
        "model": "gemma3:1b",  # Lightweight model
        "temperature": 0.7,
        "max_tokens": 500
    },
    "conversation_storage": {
        "backend": "memory"  # Store conversations in memory
    },
    "prompts": {
        "friendly_assistant": "You are a friendly and helpful AI assistant."
    },
    "system_prompt": {
        "name": "friendly_assistant"
    }
}
```

### Bot Context

Each conversation needs a context that identifies:

- `conversation_id` - Unique ID for this conversation
- `client_id` - Tenant/application identifier
- `user_id` - User identifier (optional)

```python
context = BotContext(
    conversation_id="simple-chat-001",
    client_id="example-client",
    user_id="demo-user"
)
```

### Chatting

The `chat()` method sends a message and returns a response:

```python
response = await bot.chat(
    message="Hello! What can you help me with?",
    context=context
)
```

## Complete Code

```python title="01_simple_chatbot.py"
--8<-- "packages/bots/examples/01_simple_chatbot.py"
```

## Running the Example

```bash
# Navigate to the bots package
cd packages/bots

# Run the example
python examples/01_simple_chatbot.py
```

## Expected Output

```
============================================================
Simple Chatbot Example
============================================================

This example shows a basic chatbot with no memory.
Required: ollama pull gemma3:1b

Creating bot from configuration...
✓ Bot created successfully

User: Hello! What can you help me with?
Bot: Hi! I'm here to assist you with various tasks...

User: Tell me a fun fact about Python programming.
Bot: Python was named after Monty Python's Flying Circus...

User: That's interesting! What makes Python so popular?
Bot: Python's popularity comes from its simplicity...

============================================================
Conversation complete!

Note: This bot has no memory between conversations.
Each new conversation starts fresh.
```

## What's Next?

This example has **no memory** - the bot doesn't remember previous messages in the conversation.

To add memory, see the [Memory Chatbot Example](memory-chatbot.md).

## Key Takeaways

1. ✅ **Configuration-First** - Bot behavior defined entirely through configuration
2. ✅ **Async/Await** - All operations are asynchronous
3. ✅ **Context Isolation** - Each conversation has its own context
4. ✅ **Local LLMs** - No API keys needed with Ollama
5. ⚠️ **No Memory** - Bot doesn't remember conversation history

## Customization

### Change the Model

```python
"llm": {
    "provider": "ollama",
    "model": "llama3.1:8b",  # Use a different model
}
```

### Adjust Response Length

```python
"llm": {
    "provider": "ollama",
    "model": "gemma3:1b",
    "max_tokens": 1000,  # Longer responses
}
```

### Change System Prompt

**Using a template name:**

```python
"prompts": {
    "custom_assistant": "You are an expert in Python programming."
},
"system_prompt": {
    "name": "custom_assistant"
}
```

**Using inline content directly:**

```python
# Multi-line prompts can be specified directly without a prompts library
"system_prompt": """You are an expert Python programming assistant.

Key responsibilities:
- Help users write clean, idiomatic Python code
- Explain Python concepts clearly
- Suggest best practices and design patterns
"""
```

**Or as a dict with content:**

```python
"system_prompt": {
    "content": "You are an expert in Python programming. Help users write clean code."
}
```

## Related Examples

- [Memory Chatbot](memory-chatbot.md) - Add conversation memory
- [RAG Chatbot](rag-chatbot.md) - Add knowledge base
- [ReAct Agent](react-agent.md) - Add tools and reasoning

## Related Documentation

- [Quick Start Guide](../quickstart.md)
- [User Guide](../guides/user-guide.md)
- [Configuration Reference](../guides/configuration.md)

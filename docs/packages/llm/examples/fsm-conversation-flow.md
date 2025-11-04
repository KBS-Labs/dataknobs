# FSM-Based Conversation Flow Example

This example demonstrates how to build sophisticated conversation systems using FSM-based conversation flows with the LLM package.

## Overview

This example showcases:

- **Multi-stage conversation flow** (analyze, respond, refine)
- **Intent recognition** and dynamic state transitions
- **Context management** and conversation history
- **Error recovery** with fallback mechanisms
- **Template selection** based on user input

## Key Components

### Conversation Flow Definition

```python
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    KeywordCondition,
    LLMClassifierCondition
)
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_data.backends import AsyncMemoryDatabase
from dataknobs_llm.conversations import DataknobsConversationStorage
from pathlib import Path

# Configure LLM
config = LLMConfig(provider="openai", api_key="your-key", model="gpt-4")
llm = create_llm_provider(config)

# Create conversation flow
conversation_flow = ConversationFlow(
    name="intelligent_conversation",
    initial_state="analyze_input",
    states={
        "analyze_input": FlowState(
            prompt="analyze_user_intent",
            transitions={
                "question": KeywordCondition(["what", "how", "why", "when", "where"]),
                "command": KeywordCondition(["do", "create", "make", "build", "generate"]),
                "greeting": KeywordCondition(["hello", "hi", "hey", "greetings"]),
                "feedback": KeywordCondition(["thanks", "good", "bad", "helpful"])
            },
            next_states={
                "question": "handle_question",
                "command": "handle_command",
                "greeting": "handle_greeting",
                "feedback": "handle_feedback"
            }
        ),
        "handle_question": FlowState(
            prompt="answer_question",
            transitions={
                "needs_clarification": KeywordCondition(["unclear", "don't understand"]),
                "complete": KeywordCondition([".*"])  # Default
            },
            next_states={
                "needs_clarification": "request_clarification",
                "complete": "refine_response"
            }
        ),
        "handle_command": FlowState(
            prompt="process_command",
            transitions={
                "complete": KeywordCondition([".*"])
            },
            next_states={"complete": "refine_response"}
        ),
        "handle_greeting": FlowState(
            prompt="greeting_response",
            transitions={
                "complete": KeywordCondition([".*"])
            },
            next_states={"complete": "end"}
        ),
        "handle_feedback": FlowState(
            prompt="acknowledge_feedback",
            transitions={
                "complete": KeywordCondition([".*"])
            },
            next_states={"complete": "end"}
        ),
        "request_clarification": FlowState(
            prompt="ask_clarification",
            transitions={},
            next_states={}
        ),
        "refine_response": FlowState(
            prompt="refine_and_validate",
            transitions={},
            next_states={}
        ),
        "end": FlowState(
            prompt=None,
            transitions={},
            next_states={}
        )
    }
)
```

### Prompt Templates

Create prompt templates in your prompt directory (`prompts/user/`):

**analyze_user_intent.yaml**:
```yaml
template: |
  Analyze the user's intent in this message: "{{user_input}}"

  Determine the primary intent and extract key information.

  User message: {{user_input}}

defaults:
  user_input: ""
```

**answer_question.yaml**:
```yaml
template: |
  Answer this question clearly and concisely:

  Question: {{user_input}}

  {% if context %}
  Previous conversation context:
  {{context}}
  {% endif %}

  Provide a helpful, accurate answer.

defaults:
  user_input: ""
  context: ""
```

**process_command.yaml**:
```yaml
template: |
  Process this user command:

  Command: {{user_input}}

  Explain what you will do and provide the result.

defaults:
  user_input: ""
```

**refine_and_validate.yaml**:
```yaml
template: |
  Review and refine this response for clarity and accuracy:

  {{draft_response}}

  Make it more helpful and precise.

defaults:
  draft_response: ""
```

## Complete Usage Example

```python
import asyncio
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_llm.conversations.flow import ConversationFlow, FlowState, KeywordCondition
from dataknobs_data.backends import AsyncMemoryDatabase
from pathlib import Path

async def create_conversation_system():
    """Create an intelligent conversation system."""

    # Setup LLM
    config = LLMConfig(
        provider="openai",
        api_key="your-key",
        model="gpt-4",
        temperature=0.7
    )
    llm = create_llm_provider(config)

    # Setup prompt builder
    library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
    builder = AsyncPromptBuilder(library=library)

    # Setup storage
    db = AsyncMemoryDatabase()
    storage = DataknobsConversationStorage(db)

    # Define conversation flow (as shown above)
    flow = conversation_flow

    # Create conversation manager
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        flow=flow,
        system_prompt_name="conversation_assistant"
    )

    return manager

async def interactive_conversation():
    """Run an interactive conversation."""

    manager = await create_conversation_system()

    print("Conversation System Ready!")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            break

        # Add user message
        await manager.add_message(
            role="user",
            content=user_input
        )

        # Execute flow to get response
        response = await manager.execute_flow()

        print(f"Assistant: {response.content}\n")

        # View conversation tree (optional)
        tree = await manager.get_tree_structure()
        print(f"[Flow state: {manager.current_state}]")

# Run the conversation system
asyncio.run(interactive_conversation())
```

## Advanced Features

### Intent Classification with LLM

For more sophisticated intent detection, use `LLMClassifierCondition`:

```python
from dataknobs_llm.conversations.flow import LLMClassifierCondition

# Define flow with LLM-based classification
sophisticated_flow = ConversationFlow(
    name="advanced_conversation",
    initial_state="classify_intent",
    states={
        "classify_intent": FlowState(
            prompt="initial_prompt",
            transitions={
                "technical_question": LLMClassifierCondition(
                    llm=llm,
                    classification_prompt="""
                    Is this a technical question requiring code or technical explanation?
                    Input: {{user_input}}
                    Answer with 'yes' or 'no'.
                    """,
                    target_class="yes"
                ),
                "general_question": LLMClassifierCondition(
                    llm=llm,
                    classification_prompt="""
                    Is this a general knowledge question?
                    Input: {{user_input}}
                    Answer with 'yes' or 'no'.
                    """,
                    target_class="yes"
                )
            },
            next_states={
                "technical_question": "handle_technical",
                "general_question": "handle_general"
            }
        ),
        "handle_technical": FlowState(
            prompt="technical_response",
            transitions={},
            next_states={}
        ),
        "handle_general": FlowState(
            prompt="general_response",
            transitions={},
            next_states={}
        )
    }
)
```

### Context Preservation

The conversation manager automatically maintains context:

```python
# Context is preserved across flow executions
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    flow=flow,
    conversation_id="user123-session1"  # Persistent across restarts
)

# First exchange
await manager.add_message(role="user", content="What is Python?")
response1 = await manager.execute_flow()

# Second exchange - context is automatically included
await manager.add_message(role="user", content="Show me an example")
response2 = await manager.execute_flow()  # Knows we're talking about Python

# View conversation history
history = manager.get_history()
for msg in history:
    print(f"{msg.role}: {msg.content}")
```

### Error Recovery

Add error handling with fallback states:

```python
error_handling_flow = ConversationFlow(
    name="resilient_conversation",
    initial_state="start",
    states={
        "start": FlowState(
            prompt="initial_prompt",
            transitions={
                "success": KeywordCondition([".*"]),
                "error": KeywordCondition([])  # Fallback
            },
            next_states={
                "success": "process",
                "error": "handle_error"
            }
        ),
        "handle_error": FlowState(
            prompt="error_recovery",
            transitions={},
            next_states={}
        ),
        "process": FlowState(
            prompt="main_processing",
            transitions={},
            next_states={}
        )
    }
)
```

### Branching Conversations

Explore alternative responses:

```python
# Save checkpoint before trying different approaches
checkpoint = manager.current_node

# Try approach A
await manager.add_message(role="user", content="Explain in detail")
detail_response = await manager.execute_flow()

# Go back and try approach B
await manager.switch_to_node(checkpoint)
await manager.add_message(role="user", content="Give me a brief summary")
brief_response = await manager.execute_flow()

# Compare results
print(f"Detailed: {detail_response.content[:100]}...")
print(f"Brief: {brief_response.content[:100]}...")
```

## Benefits of FSM-Based Conversation Flow

1. **Structured Flow**: Explicit state transitions ensure consistent conversation patterns
2. **Intent-Driven**: Automatically route to appropriate handlers based on user intent
3. **Context Management**: Automatic history tracking and context preservation
4. **Error Resilience**: Graceful handling of unexpected inputs
5. **Branching Support**: Explore alternative conversation paths
6. **Debuggable**: Clear visibility into conversation state and transitions

## Comparison with Simple Conversations

### Simple Conversation (No Flow)

```python
# Simple back-and-forth without explicit flow
manager = await ConversationManager.create(llm=llm, prompt_builder=builder, storage=storage)

await manager.add_message(role="user", content="Hello")
response = await manager.complete()
```

### FSM-Based Conversation (With Flow)

```python
# Structured conversation with explicit states and transitions
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    flow=conversation_flow
)

await manager.add_message(role="user", content="Hello")
response = await manager.execute_flow()  # Follows defined flow states
```

The FSM approach provides more control over conversation logic, making it ideal for:
- Customer support chatbots
- Multi-step wizards
- Interview/survey systems
- Complex domain-specific assistants

## See Also

- [Conversation Flow Examples](conversation-flows.md) - More flow patterns
- [Conversation Management Guide](../guides/conversations.md) - Core concepts
- [FSM-Based Flows Guide](../guides/flows.md) - Flow orchestration details
- [API Reference](../api/conversations.md) - Complete API documentation

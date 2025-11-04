# LLM Conversation System Example

> **Note**: LLM functionality has moved to the dedicated [dataknobs-llm package](../../llm/index.md).

This example has been migrated to the LLM package documentation. For FSM-based conversation management using LLM, see:

## New Location

**[FSM-Based Conversation Flow Example](../../llm/examples/fsm-conversation-flow.md)** in the LLM package documentation.

The new example demonstrates:

- Multi-stage conversation flows using `ConversationFlow`
- Intent recognition with state transitions
- Context management and history
- Error recovery mechanisms
- Integration with the LLM package

## Quick Example

```python
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.conversations.flow import ConversationFlow, FlowState, KeywordCondition
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_data.backends import AsyncMemoryDatabase
from pathlib import Path

# Setup
config = LLMConfig(provider="openai", api_key="your-key")
llm = create_llm_provider(config)

library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
builder = AsyncPromptBuilder(library=library)

db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

# Define conversation flow
flow = ConversationFlow(
    name="conversation",
    initial_state="greet",
    states={
        "greet": FlowState(
            prompt="greeting",
            transitions={"continue": KeywordCondition([".*"])},
            next_states={"continue": "respond"}
        ),
        "respond": FlowState(
            prompt="response",
            transitions={},
            next_states={}
        )
    }
)

# Create conversation manager
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    flow=flow
)

# Use the conversation system
await manager.add_message(role="user", content="Hello!")
response = await manager.execute_flow()
print(response.content)
```

## Related Documentation

- **[LLM Package Overview](../../llm/index.md)** - Introduction to the LLM package
- **[Conversation Management](../../llm/guides/conversations.md)** - Conversation concepts
- **[Conversation Flows](../../llm/guides/flows.md)** - FSM-based flow orchestration
- **[Conversation Flow Examples](../../llm/examples/conversation-flows.md)** - More flow patterns
- **[FSM Package](../index.md)** - FSM package documentation

## Migration Notes

The LLM functionality previously in the FSM package has been extracted into a dedicated package with improved APIs:

### Old API (FSM Package)
```python
from dataknobs_fsm.llm.providers import create_llm_provider
from dataknobs_fsm.llm.base import LLMConfig
```

### New API (LLM Package)
```python
from dataknobs_llm import create_llm_provider, LLMConfig
```

The new LLM package provides:
- Dedicated conversation management with `ConversationManager`
- FSM-based flows with `ConversationFlow`
- Prompt templating and versioning
- A/B testing and metrics
- RAG integration and caching

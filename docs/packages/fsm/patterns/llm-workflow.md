# LLM Workflow Pattern

> **Note**: LLM functionality has moved to the dedicated [dataknobs-llm package](../../llm/index.md).

The LLM workflow pattern has been migrated to the LLM package with improved APIs and better integration.

## New Location

For LLM workflow orchestration, see:

- **[Conversation Flows Guide](../../llm/guides/flows.md)** - FSM-based conversation orchestration
- **[Conversation Flow Examples](../../llm/examples/conversation-flows.md)** - Flow pattern examples
- **[FSM-Based Conversation Flow](../../llm/examples/fsm-conversation-flow.md)** - Detailed FSM example
- **[LLM Package Overview](../../llm/index.md)** - Complete LLM package documentation

## Migration Guide

### Old API (FSM Package)

The FSM package previously provided LLM workflow patterns with classes like:

```python
# OLD - No longer available
from dataknobs_fsm.patterns.llm_workflow import (
    LLMWorkflow,
    LLMWorkflowConfig,
    LLMStep,
    WorkflowType
)
```

### New API (LLM Package)

The LLM package now provides equivalent functionality with improved APIs:

```python
# NEW - Use these instead
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.conversations.flow import ConversationFlow, FlowState, KeywordCondition
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
```

## Workflow Type Mapping

### Simple LLM Workflow

**Old approach:**
```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.SIMPLE,
    steps=[LLMStep(name="generate", prompt_template=template)]
)
workflow = LLMWorkflow(config)
```

**New approach:**
```python
# Simple conversation without flow
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage
)

await manager.add_message(role="user", prompt_name="generate", params=params)
response = await manager.complete()
```

### Sequential Chain

**Old approach:**
```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.CHAIN,
    steps=[step1, step2, step3]
)
```

**New approach:**
```python
flow = ConversationFlow(
    name="chain",
    initial_state="step1",
    states={
        "step1": FlowState(
            prompt="prompt1",
            transitions={"next": KeywordCondition([".*"])},
            next_states={"next": "step2"}
        ),
        "step2": FlowState(
            prompt="prompt2",
            transitions={"next": KeywordCondition([".*"])},
            next_states={"next": "step3"}
        ),
        "step3": FlowState(
            prompt="prompt3",
            transitions={},
            next_states={}
        )
    }
)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    flow=flow
)
```

### RAG (Retrieval-Augmented Generation)

**Old approach:**
```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.RAG,
    rag_config=RAGConfig(retriever_type="vector", ...)
)
```

**New approach:**

Configure RAG in your prompt templates:

```yaml
# prompts/user/rag_query.yaml
template: |
  Answer this question: {{question}}

  Relevant context:
  {{RAG_DOCS}}

rag_configs:
  - adapter_name: docs
    query_template: "{{question}}"
    k: 5
    placeholder: "RAG_DOCS"
```

Then use with conversation manager:

```python
# Setup resource adapter
from dataknobs_llm.prompts import InMemoryAdapter

adapter = InMemoryAdapter(documents=[...])
builder = AsyncPromptBuilder(library=library, adapters={"docs": adapter})

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    cache_rag_results=True  # Enable RAG caching
)

await manager.add_message(role="user", prompt_name="rag_query", params={"question": "..."})
response = await manager.complete()
```

### Chain-of-Thought Reasoning

**Old approach:**
```python
config = LLMWorkflowConfig(
    workflow_type=WorkflowType.COT,
    steps=[decompose, reason, synthesize]
)
```

**New approach:**
```python
cot_flow = ConversationFlow(
    name="chain_of_thought",
    initial_state="decompose",
    states={
        "decompose": FlowState(
            prompt="break_down_problem",
            transitions={"next": KeywordCondition([".*"])},
            next_states={"next": "reason"}
        ),
        "reason": FlowState(
            prompt="solve_steps",
            transitions={"next": KeywordCondition([".*"])},
            next_states={"next": "synthesize"}
        ),
        "synthesize": FlowState(
            prompt="combine_solutions",
            transitions={},
            next_states={}
        )
    }
)
```

## Complete Migration Example

### Before (FSM Package)

```python
from dataknobs_fsm.patterns.llm_workflow import (
    LLMWorkflow,
    LLMWorkflowConfig,
    LLMStep,
    WorkflowType
)
from dataknobs_fsm.llm.base import LLMConfig

# Define workflow
step1 = LLMStep(
    name="summarize",
    prompt_template=PromptTemplate("Summarize: {text}"),
    output_key="summary"
)

step2 = LLMStep(
    name="analyze",
    prompt_template=PromptTemplate("Analyze: {summary}"),
    depends_on=["summarize"],
    output_key="analysis"
)

config = LLMWorkflowConfig(
    workflow_type=WorkflowType.CHAIN,
    steps=[step1, step2],
    default_model_config=LLMConfig(provider="openai", model="gpt-4")
)

workflow = LLMWorkflow(config)
result = await workflow.execute({"text": "..."})
```

### After (LLM Package)

```python
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.conversations.flow import ConversationFlow, FlowState, KeywordCondition
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_data.backends import AsyncMemoryDatabase
from pathlib import Path

# Setup
config = LLMConfig(provider="openai", api_key="your-key", model="gpt-4")
llm = create_llm_provider(config)

library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
builder = AsyncPromptBuilder(library=library)

db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

# Define flow
flow = ConversationFlow(
    name="analysis_chain",
    initial_state="summarize",
    states={
        "summarize": FlowState(
            prompt="summarize_text",  # prompts/user/summarize_text.yaml
            transitions={"next": KeywordCondition([".*"])},
            next_states={"next": "analyze"}
        ),
        "analyze": FlowState(
            prompt="analyze_summary",  # prompts/user/analyze_summary.yaml
            transitions={},
            next_states={}
        )
    }
)

# Create manager and execute
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    flow=flow
)

await manager.add_message(role="user", prompt_name="summarize_text", params={"text": "..."})
summary = await manager.execute_flow()

# Analysis happens automatically in next flow state
```

## Key Improvements in LLM Package

1. **Separation of Concerns**: Conversation management is separate from FSM execution
2. **Better Prompt Management**: File-based prompt library with Jinja2 templating
3. **Conversation History**: Built-in conversation tree with branching support
4. **RAG Integration**: YAML-based RAG configuration in prompts
5. **Versioning & A/B Testing**: Track and test prompts systematically
6. **Caching**: RAG metadata caching at conversation level
7. **Persistence**: Save and restore conversation state

## Additional Resources

### Guides
- [Conversation Management](../../llm/guides/conversations.md)
- [FSM-Based Flows](../../llm/guides/flows.md)
- [Prompt Engineering](../../llm/guides/prompts.md)
- [Performance & Caching](../../llm/guides/performance.md)

### Examples
- [Basic Usage](../../llm/examples/basic-usage.md)
- [Advanced Prompting](../../llm/examples/advanced-prompting.md)
- [Conversation Flows](../../llm/examples/conversation-flows.md)
- [FSM Conversation Flow](../../llm/examples/fsm-conversation-flow.md)

### API Reference
- [LLM API](../../llm/api/llm.md)
- [Conversations API](../../llm/api/conversations.md)
- [Prompts API](../../llm/api/prompts.md)

## Questions or Issues?

If you have questions about migrating from the old FSM LLM workflow pattern to the new LLM package:

1. Check the [LLM Package Quick Start](../../llm/quickstart.md)
2. Review the [migration examples above](#complete-migration-example)
3. See [conversation flow examples](../../llm/examples/conversation-flows.md) for common patterns
4. Open an issue on [GitHub](https://github.com/kbs-labs/dataknobs/issues)

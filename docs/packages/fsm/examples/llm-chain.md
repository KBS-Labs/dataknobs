# LLM Chain Processing Example

> **Note**: LLM functionality has moved to the dedicated [dataknobs-llm package](../../llm/index.md).

This example has been superseded by comprehensive conversation flow examples in the LLM package.

## New Location

For LLM chain processing and multi-step workflows, see:

- **[Conversation Flow Examples](../../llm/examples/conversation-flows.md)** - Multi-step conversation patterns
- **[Advanced Prompting Examples](../../llm/examples/advanced-prompting.md)** - Prompt chaining and RAG
- **[FSM-Based Conversation Flow](../../llm/examples/fsm-conversation-flow.md)** - FSM orchestration

## Supported Patterns

The LLM package now provides these patterns directly:

### Sequential LLM Chains

Use conversation flows for sequential processing:

```python
from dataknobs_llm.conversations.flow import ConversationFlow, FlowState, KeywordCondition

chain_flow = ConversationFlow(
    name="sequential_chain",
    initial_state="summarize",
    states={
        "summarize": FlowState(
            prompt="summarize_text",
            transitions={"next": KeywordCondition([".*"])},
            next_states={"next": "analyze"}
        ),
        "analyze": FlowState(
            prompt="analyze_summary",
            transitions={"next": KeywordCondition([".*"])},
            next_states={"next": "conclude"}
        ),
        "conclude": FlowState(
            prompt="draw_conclusions",
            transitions={},
            next_states={}
        )
    }
)
```

### Chain-of-Thought Reasoning

Implement step-by-step reasoning flows:

```python
cot_flow = ConversationFlow(
    name="chain_of_thought",
    initial_state="decompose",
    states={
        "decompose": FlowState(
            prompt="break_down_problem",
            transitions={"decomposed": KeywordCondition([".*"])},
            next_states={"decomposed": "solve_steps"}
        ),
        "solve_steps": FlowState(
            prompt="solve_each_step",
            transitions={"solved": KeywordCondition([".*"])},
            next_states={"solved": "synthesize"}
        ),
        "synthesize": FlowState(
            prompt="combine_solutions",
            transitions={},
            next_states={}
        )
    }
)
```

### RAG (Retrieval-Augmented Generation)

Configure RAG in prompt templates:

```yaml
# prompts/user/rag_query.yaml
template: |
  Answer this question using the provided context:

  Question: {{question}}

  Context:
  {{RAG_DOCS}}

rag_configs:
  - adapter_name: knowledge_base
    query_template: "{{question}}"
    k: 5
    placeholder: "RAG_DOCS"
```

### Prompt Chaining

Chain prompts with context preservation:

```python
# First prompt
await manager.add_message(
    role="user",
    prompt_name="initial_analysis",
    params={"data": raw_data}
)
result1 = await manager.complete()

# Second prompt - context automatically included
await manager.add_message(
    role="user",
    prompt_name="deep_dive",
    params={"aspect": "key_findings"}
)
result2 = await manager.complete()  # Has context from result1
```

## See Also

- **[LLM Package](../../llm/index.md)** - Full LLM package documentation
- **[Conversation Flows Guide](../../llm/guides/flows.md)** - Flow orchestration
- **[RAG Integration](../../llm/examples/advanced-prompting.md)** - RAG examples
- **[Prompt Engineering](../../llm/guides/prompts.md)** - Prompt best practices

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
from dataknobs_llm.conversations.flow import ConversationFlow, FlowState, always

chain_flow = ConversationFlow(
    name="sequential_chain",
    initial_state="summarize",
    states={
        "summarize": FlowState(
            prompt_name="summarize_text",
            transitions={"next": "analyze"},
            transition_conditions={"next": always()},
        ),
        "analyze": FlowState(
            prompt_name="analyze_summary",
            transitions={"next": "conclude"},
            transition_conditions={"next": always()},
        ),
        "conclude": FlowState(prompt_name="draw_conclusions"),   # terminal
    }
)
```

`transitions` names the target state and `transition_conditions` holds the
condition that decides the arc, under the same key. `always()` is the
unconditional step. Run the flow with
`async for node in manager.execute_flow(chain_flow)`.

### Chain-of-Thought Reasoning

Implement step-by-step reasoning flows:

```python
cot_flow = ConversationFlow(
    name="chain_of_thought",
    initial_state="decompose",
    states={
        "decompose": FlowState(
            prompt_name="break_down_problem",
            transitions={"decomposed": "solve_steps"},
            transition_conditions={"decomposed": always()},
        ),
        "solve_steps": FlowState(
            prompt_name="solve_each_step",
            transitions={"solved": "synthesize"},
            transition_conditions={"solved": always()},
        ),
        "synthesize": FlowState(prompt_name="combine_solutions"),
    }
)
```

A flow state renders its prompt; it does not call the LLM. This sequences the
*prompts* of a chain-of-thought — drive the reasoning turns themselves with
`manager.complete()`, and see
[FSM-Based Flows](../../llm/guides/flows.md) for the full model.

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
    query: "{{question}}"
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

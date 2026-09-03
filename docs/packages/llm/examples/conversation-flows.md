# Conversation Flow Examples

Runnable patterns for FSM-based conversation flows. Read
[FSM-Based Flows](../guides/flows.md) first — in particular, that a flow state
renders a prompt rather than calling the LLM, and that conditions test the text
a state rendered.

Every example on this page uses `EchoProvider` and an in-memory prompt library,
so each one runs as written with no API key and no files on disk.

## Setup

All the snippets below assume this preamble.

```python
import asyncio

from dataknobs_common.exceptions import OperationError
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.conversations.flow import (
    CompositeCondition,
    ConversationFlow,
    ConversationFlowAdapter,
    FlowState,
    LLMClassifierCondition,
    always,
    context_condition,
    keyword_condition,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary

PROMPTS = {
    "system": {"support_agent": {"template": "You are a support agent."}},
    "user": {
        "greeting": {"template": "Hello! What can I help you with today?"},
        "collect_issue": {"template": "Tell me more about the {{category}} problem."},
        "tech_support": {"template": "Let's troubleshoot. Have you restarted?"},
        "billing_support": {"template": "I'll pull up your billing records."},
        "farewell": {"template": "Glad I could help. Goodbye!"},
        "with_state": {"template": "state={{state}} loop={{loop_count}}"},
    },
}


def make_builder() -> AsyncPromptBuilder:
    return AsyncPromptBuilder(library=ConfigPromptLibrary(PROMPTS))


def make_llm() -> EchoProvider:
    return EchoProvider(
        LLMConfig(provider="echo", model="echo", options={"echo_prefix": ""})
    )
```

## A linear flow

Two states, one unconditional arc. `always()` is the condition for a step that
should simply happen next.

```python
flow = ConversationFlow(
    name="short",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt_name="greeting",
            transitions={"go": "farewell"},
            transition_conditions={"go": always()},
        ),
        "farewell": FlowState(prompt_name="farewell"),
    },
)

adapter = ConversationFlowAdapter(flow=flow, prompt_builder=make_builder())
final = await adapter.execute()

assert adapter.execution_state.history == [
    ("greeting", "Hello! What can I help you with today?"),
    ("farewell", "Glad I could help. Goodbye!"),
]
assert final["response"] == "Glad I could help. Goodbye!"
assert final["state"] == "farewell"
```

## Routing on what a state rendered

The `collect_issue` template interpolates `category`, and the arcs out of it
match keywords in the *rendered* text. Setting `prompt_params={"category":
"billing"}` therefore chooses the branch.

```python
def routing_flow(category: str) -> ConversationFlow:
    return ConversationFlow(
        name="customer_support",
        initial_state="greeting",
        states={
            "greeting": FlowState(
                prompt_name="greeting",
                transitions={"has_issue": "collect_issue"},
                transition_conditions={"has_issue": always()},
            ),
            "collect_issue": FlowState(
                prompt_name="collect_issue",
                prompt_params={"category": category},
                transitions={
                    "technical": "tech_support",
                    "billing": "billing_support",
                },
                transition_conditions={
                    "technical": keyword_condition(["bug", "error", "crash"]),
                    "billing": keyword_condition(["billing", "charge", "refund"]),
                },
            ),
            "tech_support": FlowState(prompt_name="tech_support"),
            "billing_support": FlowState(prompt_name="billing_support"),
        },
    )


adapter = ConversationFlowAdapter(flow=routing_flow("billing"), prompt_builder=make_builder())
await adapter.execute({"customer_id": "c-42"})

assert [state for state, _ in adapter.execution_state.history] == [
    "greeting",
    "collect_issue",
    "billing_support",
]
```

Routing on the *user's* words instead means putting them where a template can
render them — pass them as initial data and interpolate them into the state's
prompt.

## Routing on data, not text

`context_condition` ignores the response and applies a predicate to the run
context. `initial_context` seeds that context, and is merged into the flow's
data as well.

```python
flow = ConversationFlow(
    name="tiered",
    initial_state="greeting",
    initial_context={"customer_tier": "gold"},
    states={
        "greeting": FlowState(
            prompt_name="greeting",
            transitions={"priority": "tech_support", "standard": "farewell"},
            transition_conditions={
                "priority": context_condition(lambda ctx: ctx.get("customer_tier") == "gold"),
                "standard": context_condition(lambda ctx: ctx.get("customer_tier") != "gold"),
            },
        ),
        "tech_support": FlowState(prompt_name="tech_support"),
        "farewell": FlowState(prompt_name="farewell"),
    },
)

adapter = ConversationFlowAdapter(flow=flow, prompt_builder=make_builder())
final = await adapter.execute()

assert [state for state, _ in adapter.execution_state.history] == ["greeting", "tech_support"]
assert final["customer_tier"] == "gold"
```

## Combining conditions

`CompositeCondition` evaluates every member — there is no short-circuit — and
combines with `"and"` or `"or"`.

```python
gold_refund = CompositeCondition(
    conditions=[
        keyword_condition(["billing", "refund"]),
        context_condition(lambda ctx: ctx.get("customer_tier") == "gold"),
    ],
    operator="and",
)

assert await gold_refund.evaluate("I'll pull up your billing records.", {"customer_tier": "gold"})
assert not await gold_refund.evaluate("I'll pull up your billing records.", {"customer_tier": "free"})
```

## Stopping a loop

A state that transitions to itself needs a ceiling. `max_loops` bounds one
state; `max_total_loops` bounds the whole run. Either one tripping fails the
run, and `stop_reason` says which.

```python
flow = ConversationFlow(
    name="looping",
    initial_state="ask",
    states={
        "ask": FlowState(
            prompt_name="with_state",
            max_loops=2,
            transitions={"again": "ask"},
            transition_conditions={"again": always()},
        )
    },
)

adapter = ConversationFlowAdapter(flow=flow, prompt_builder=make_builder())

try:
    await adapter.execute()
except OperationError as exc:
    print(exc)   # Conversation flow 'looping' failed: Max loops exceeded for state ask

assert adapter.get_execution_summary()["stop_reason"] == "Max loops exceeded for state ask"
# Two renders happened; the third entry tripped the guard before rendering.
assert len(adapter.execution_state.history) == 2
assert adapter.execution_state.loop_counts["ask"] == 3
```

## Hooks

`on_enter` runs before the state's prompt is rendered, `on_exit` after. Both are
awaited with `(state_name, data, context)`, so both must be `async def`, and
both are wrapped: an exception in a hook is logged and the flow continues.

```python
visited: list[tuple[str, str]] = []


async def record_enter(state_name, data, context):
    visited.append(("enter", state_name))


async def record_exit(state_name, data, context):
    visited.append(("exit", state_name))


flow = ConversationFlow(
    name="hooked",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt_name="greeting",
            on_enter=record_enter,
            on_exit=record_exit,
            transitions={"go": "farewell"},
            transition_conditions={"go": always()},
        ),
        "farewell": FlowState(
            prompt_name="farewell", on_enter=record_enter, on_exit=record_exit
        ),
    },
)

await ConversationFlowAdapter(flow=flow, prompt_builder=make_builder()).execute()

assert visited == [
    ("enter", "greeting"),
    ("exit", "greeting"),
    ("enter", "farewell"),
    ("exit", "farewell"),
]
```

Mutating the `context` dictionary a hook receives is how a flow accumulates
state across its own steps — it is the same dictionary the next state's
conditions and render will see.

## Branching with an LLM

`LLMClassifierCondition` is the one condition that calls a provider. Pass the
provider to the adapter as `llm=`; the condition finds it in the run context.

```python
llm = make_llm()

flow = ConversationFlow(
    name="classified",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt_name="greeting",
            transitions={"escalate": "tech_support", "close": "farewell"},
            transition_conditions={
                "escalate": LLMClassifierCondition(
                    classifier_prompt=(
                        "Does this message describe a technical problem? "
                        "Answer with one word, yes or no.\n\n{{response}}"
                    ),
                    expected_value="yes",
                ),
                "close": always(),
            },
        ),
        "tech_support": FlowState(prompt_name="tech_support"),
        "farewell": FlowState(prompt_name="farewell"),
    },
)

adapter = ConversationFlowAdapter(flow=flow, prompt_builder=make_builder(), llm=llm)
await adapter.execute()
await llm.close()
```

The completion is stripped, lower-cased, and compared to `expected_value` for
exact equality — so the prompt has to ask for a single word. Without a provider
in context and without `llm_config`, `evaluate` raises `ValueError`.

## Through a conversation

`ConversationManager.execute_flow()` writes one assistant node per state into
the conversation tree and yields them. The conversation must already have
state, so create the manager with a system prompt or add a message first.

```python
async def run_through_a_conversation():
    llm = make_llm()
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=make_builder(),
        storage=DataknobsConversationStorage(AsyncMemoryDatabase()),
        system_prompt_name="support_agent",
    )

    async for node in manager.execute_flow(routing_flow("billing")):
        print(node.metadata["state"], "->", node.message.content)

    history = await manager.get_history()
    assert [m.role for m in history] == ["system", "assistant", "assistant", "assistant"]
    await llm.close()


asyncio.run(run_through_a_conversation())
```

Output:

```
greeting -> Hello! What can I help you with today?
collect_issue -> Tell me more about the billing problem.
billing_support -> I'll pull up your billing records.
```

## Handling a failed flow

`execute_flow` converts the adapter's `OperationError` into a `ValueError`,
keeping the original as `__cause__`. The most common cause is a state with no
matching arc — a dead end is a failure, not a graceful stop.

```python
async def handle_a_failure():
    llm = make_llm()
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=make_builder(),
        storage=DataknobsConversationStorage(AsyncMemoryDatabase()),
        system_prompt_name="support_agent",
    )

    dead_end = ConversationFlow(
        name="dead_end",
        initial_state="greeting",
        states={
            "greeting": FlowState(
                prompt_name="greeting",
                transitions={"never": "farewell"},
                transition_conditions={"never": keyword_condition(["no-such-word"])},
            ),
            "farewell": FlowState(prompt_name="farewell"),
        },
    )

    try:
        async for _node in manager.execute_flow(dead_end):
            pass
    except ValueError as exc:
        print(exc)                       # Flow execution failed: ... No valid transitions from state: greeting
        print(type(exc.__cause__))       # <class 'dataknobs_common.exceptions.OperationError'>
        print(exc.__cause__.context)     # {'flow': 'dead_end', 'state': 'greeting'}

    await llm.close()


asyncio.run(handle_a_failure())
```

To let a state end the conversation instead, make it terminal — give it no
`transitions` — or add a final `always()` arc to a terminal state.

## Persisting and resuming

Flow nodes are written to storage as they are yielded, so a flow run survives a
restart like any other conversation. Resume with
`ConversationManager.resume()`, not `create()` — `create()` starts a new
conversation whatever id you give it.

```python
async def persist_and_resume():
    db = AsyncMemoryDatabase()          # swap for any dataknobs-data backend
    storage = DataknobsConversationStorage(db)
    llm = make_llm()
    builder = make_builder()

    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="support_agent",
        conversation_id="support-session-1",
    )
    async for _node in manager.execute_flow(routing_flow("billing")):
        pass

    resumed = await ConversationManager.resume(
        conversation_id="support-session-1",
        llm=llm,
        prompt_builder=builder,
        storage=storage,
    )
    assert len(await resumed.get_history()) == 4     # system + three flow states

    async for _node in resumed.execute_flow(routing_flow("billing")):
        pass
    assert len(await resumed.get_history()) == 7

    await llm.close()


asyncio.run(persist_and_resume())
```

Each run appends to the same branch, so the second run's nodes are children of
the first run's last node.

## Validating before you run

```python
flow = ConversationFlow(
    name="warned",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt_name="greeting",
            transitions={"go": "farewell"},
            transition_conditions={"go": always()},
        ),
        "farewell": FlowState(prompt_name="farewell"),
        "orphan": FlowState(prompt_name="tech_support"),
    },
)

assert flow.validate_flow() == [
    "State 'orphan' is unreachable",
    "State 'farewell' has no exit transitions (potential dead end)",
    "State 'orphan' has no exit transitions (potential dead end)",
]
```

Every terminal state produces a "potential dead end" warning; that one is about
shape, not correctness.

The structural errors are raised at construction instead:

```python
FlowState(prompt_name="greeting", transitions={"go": "farewell"})
# ValueError: Transition 'go' has no corresponding condition

FlowState(prompt_name="")
# ValueError: prompt_name is required

ConversationFlow(
    name="bad",
    initial_state="a",
    states={
        "a": FlowState(
            prompt_name="greeting",
            transitions={"go": "nowhere"},
            transition_conditions={"go": always()},
        )
    },
)
# ValueError: State 'a' transitions to unknown state 'nowhere'
```

## See Also

- [FSM-Based Flows](../guides/flows.md) — the reference guide
- [FSM-Based Conversation Flow](fsm-conversation-flow.md) — one complete program
- [Conversation Management](../guides/conversations.md) — branching and persistence
- [Basic Usage](basic-usage.md) — conversations without a flow

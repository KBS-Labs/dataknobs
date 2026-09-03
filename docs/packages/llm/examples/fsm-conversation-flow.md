# FSM-Based Conversation Flow

One complete program: a support-triage bot that restates an incoming request,
routes it to the right queue, and closes with a ticket number — as a
`ConversationFlow` executed through a `ConversationManager`.

It runs as written. `EchoProvider` and an in-memory prompt library stand in for
a real provider and a prompt directory, so there is no API key and nothing on
disk.

For the reference material behind it, see
[FSM-Based Flows](../guides/flows.md); for smaller single-purpose snippets, see
[Conversation Flow Examples](conversation-flows.md).

## What it demonstrates

- Routing on **the user's own words**, by interpolating them into the first
  state's template so the arcs out of that state can match them
- A **catch-all arc** declared last, since the first passing condition wins
- **Static parameters** per state (`prompt_params`) alongside run-wide ones
  (`initial_context`)
- Reading the resulting **conversation transcript**, which mixes the real user
  turn with the flow's generated ones

## The prompts

A flow state names a prompt in the library and renders it. These five are the
whole vocabulary of the bot.

```python
PROMPTS = {
    "system": {
        "triage_agent": {"template": "You triage incoming support requests."},
    },
    "user": {
        "restate": {
            "template": "Let me make sure I have this right: {{user_message}}",
        },
        "technical": {
            "template": (
                "That sounds like a technical fault. I'm opening ticket "
                "{{ticket_id}} and routing it to engineering."
            ),
        },
        "billing": {
            "template": (
                "That's a billing matter. I'm opening ticket {{ticket_id}} "
                "and routing it to accounts."
            ),
        },
        "general": {
            "template": "I'll pass this to a support agent under ticket {{ticket_id}}.",
        },
        "close": {"template": "Ticket {{ticket_id}} is open. Anything else?"},
    },
}
```

In production these live in a prompt directory and are loaded with
`FileSystemPromptLibrary(prompt_dir=Path("prompts/"))`; nothing else changes.

## The flow

Five states. `restate` fans out to three queues and each queue converges on
`close`, which is terminal.

```python
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    always,
    keyword_condition,
)


def triage_flow(ticket_id: str) -> ConversationFlow:
    routed = {"prompt_params": {"ticket_id": ticket_id}}
    return ConversationFlow(
        name="support_triage",
        initial_state="restate",
        description="Restate the request, route it, then close.",
        max_total_loops=4,
        initial_context={"ticket_id": ticket_id},
        states={
            "restate": FlowState(
                prompt_name="restate",
                transitions={
                    "technical": "technical",
                    "billing": "billing",
                    "general": "general",
                },
                transition_conditions={
                    "technical": keyword_condition(["error", "crash", "broken", "bug"]),
                    "billing": keyword_condition(["invoice", "charge", "refund", "billing"]),
                    "general": always(),
                },
            ),
            "technical": FlowState(
                prompt_name="technical",
                transitions={"done": "close"},
                transition_conditions={"done": always()},
                **routed,
            ),
            "billing": FlowState(
                prompt_name="billing",
                transitions={"done": "close"},
                transition_conditions={"done": always()},
                **routed,
            ),
            "general": FlowState(
                prompt_name="general",
                transitions={"done": "close"},
                transition_conditions={"done": always()},
                **routed,
            ),
            "close": FlowState(prompt_name="close", **routed),
        },
    )
```

Three things in there are load-bearing:

**`general: always()` is declared last.** Arcs are tried in declaration order
and the first passing one wins, so a catch-all placed earlier would swallow
every request before the keyword arcs were reached.

**The keyword arcs match the text `restate` rendered**, which is the user's
message wrapped in one sentence. That is why the request has to be interpolated
into the template — a condition never sees the raw input, only what a state
produced.

**`close` has no transitions**, which is what makes it terminal. A state that
should end the run needs no arcs; a state with arcs and no passing condition
fails the run instead.

## Running it

```python
import asyncio

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary


async def triage(user_message: str, ticket_id: str) -> None:
    llm = EchoProvider(LLMConfig(provider="echo", model="echo", options={"echo_prefix": ""}))
    builder = AsyncPromptBuilder(library=ConfigPromptLibrary(PROMPTS))
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())

    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="triage_agent",
        conversation_id=ticket_id,
    )
    await manager.add_message(role="user", content=user_message)

    print(f"\n--- {ticket_id}: {user_message!r}")
    async for node in manager.execute_flow(
        triage_flow(ticket_id), initial_params={"user_message": user_message}
    ):
        print(f"  [{node.metadata['state']}] {node.message.content}")

    print("  transcript:")
    for message in await manager.get_history():
        print(f"    {message.role}: {message.content}")
    await llm.close()


async def main() -> None:
    await triage("The export button throws an error every time", "T-1001")
    await triage("I was charged twice on my last invoice", "T-1002")
    await triage("How do I add a teammate?", "T-1003")


asyncio.run(main())
```

`initial_params` becomes the flow's starting data, which every state's render
can read — that is how `{{user_message}}` gets filled. The manager must already
have a conversation before `execute_flow` is called, which the
`system_prompt_name` and the `add_message()` between them guarantee.

## The output

```
--- T-1001: 'The export button throws an error every time'
  [restate] Let me make sure I have this right: The export button throws an error every time
  [technical] That sounds like a technical fault. I'm opening ticket T-1001 and routing it to engineering.
  [close] Ticket T-1001 is open. Anything else?
  transcript:
    system: You triage incoming support requests.
    user: The export button throws an error every time
    assistant: Let me make sure I have this right: The export button throws an error every time
    assistant: That sounds like a technical fault. I'm opening ticket T-1001 and routing it to engineering.
    assistant: Ticket T-1001 is open. Anything else?

--- T-1002: 'I was charged twice on my last invoice'
  [restate] Let me make sure I have this right: I was charged twice on my last invoice
  [billing] That's a billing matter. I'm opening ticket T-1002 and routing it to accounts.
  [close] Ticket T-1002 is open. Anything else?
  transcript:
    system: You triage incoming support requests.
    user: I was charged twice on my last invoice
    assistant: Let me make sure I have this right: I was charged twice on my last invoice
    assistant: That's a billing matter. I'm opening ticket T-1002 and routing it to accounts.
    assistant: Ticket T-1002 is open. Anything else?

--- T-1003: 'How do I add a teammate?'
  [restate] Let me make sure I have this right: How do I add a teammate?
  [general] I'll pass this to a support agent under ticket T-1003.
  [close] Ticket T-1003 is open. Anything else?
  transcript:
    system: You triage incoming support requests.
    user: How do I add a teammate?
    assistant: Let me make sure I have this right: How do I add a teammate?
    assistant: I'll pass this to a support agent under ticket T-1003.
    assistant: Ticket T-1003 is open. Anything else?
```

Every `assistant:` line is a rendered template, not a completion. The flow
never called the provider — a flow's LLM use is confined to
`LLMClassifierCondition`. What the flow gives you instead is a guaranteed
shape: the same three turns, in the same order, for every request that routes
the same way.

## Where to take it next

- **Route with a model rather than keywords.** Swap the keyword arcs for
  [`LLMClassifierCondition`](../guides/flows.md#llmclassifiercondition) and pass
  a real provider; the condition finds it in the run context.
- **Branch on account data.** `context_condition` reads the run context instead
  of the text, so a `customer_tier` seeded through `initial_context` can pick
  the queue.
- **Persist across restarts.** Swap `AsyncMemoryDatabase` for any
  `dataknobs-data` backend and reopen with `ConversationManager.resume()` — see
  [Persisting and resuming](conversation-flows.md#persisting-and-resuming).
- **Let a real model answer.** A flow scripts the shape of a conversation; use
  `manager.complete()` for the turns that need a model to speak. The two share
  one conversation tree.

## See Also

- [FSM-Based Flows](../guides/flows.md) — the reference guide
- [Conversation Flow Examples](conversation-flows.md) — smaller patterns
- [Conversation Management](../guides/conversations.md) — branching, metadata, persistence
- [Conversations API](../api/conversations.md) — generated reference

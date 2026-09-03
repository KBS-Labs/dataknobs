# FSM-Based Conversation Flows

A conversation flow is a state machine over your prompt library. Each state
names a prompt; each transition names a condition that decides whether to take
it. `dataknobs-fsm` runs the machine.

## What a flow does, exactly

Read this before writing one — the mental model is not the one the name
suggests.

**A state renders a prompt. It does not call the LLM.** When the flow enters a
state, the adapter renders that state's prompt template and records the
rendered text as the state's *response*. Nothing is sent to a provider. The
`ConversationNode` that reaches you carries `role="assistant"` and that
rendered text as its content.

**Conditions test the text the state just produced.** A condition's `response`
argument is the rendered prompt of the state being left, not user input. A
`KeywordCondition(["billing"])` on an arc out of a state whose template says
"Tell me about the billing problem" matches on the *template's* wording.

The one place a provider is used is
[`LLMClassifierCondition`](#llmclassifiercondition), which asks an LLM to
classify that text.

So a flow is a deterministic, prompt-driven walk that produces a scripted
sequence of assistant turns, with optional LLM-backed branching. It is not an
agent loop.

## Installation

Flows need a state-machine engine, which the base install leaves out:

```bash
pip install dataknobs-llm[fsm]
```

`[all]` includes it. `ConversationFlow`, `FlowState`, `TransitionCondition` and
the conditions need no engine and import on a base install;
`ConversationFlowAdapter` and `FlowExecutionState` are resolved lazily, so
importing the package still works and asking for either name without the extra
raises `ModuleNotFoundError` naming `dataknobs_fsm`.

`ConversationManager` needs no engine either — only `execute_flow()` does, and
only when you call it.

## Defining a flow

```python
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    keyword_condition,
)

flow = ConversationFlow(
    name="customer_support",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt_name="greeting",
            transitions={"has_issue": "collect_issue"},
            transition_conditions={"has_issue": keyword_condition(["help"])},
        ),
        "collect_issue": FlowState(
            prompt_name="collect_issue",
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
```

Two dictionaries, keyed the same way, do the work of one:

- `transitions` maps a **condition name** to a **target state name**.
- `transition_conditions` maps that same condition name to the
  `TransitionCondition` object that decides it.

Every key in `transitions` must appear in `transition_conditions`, or
`FlowState.__post_init__` raises `ValueError: Transition 'has_issue' has no
corresponding condition`.

A state with no `transitions` is terminal.

### Order decides the branch

Arcs are evaluated in the order the `transitions` dictionary declares them, and
the **first** condition that passes wins. Nothing scores or ranks them, so a
catch-all is a real pattern — put `always()` last:

```python
transitions={"technical": "tech_support", "billing": "billing_support", "other": "general"},
transition_conditions={
    "technical": keyword_condition(["bug", "error"]),
    "billing": keyword_condition(["charge", "refund"]),
    "other": always(),          # last, or it takes every request
},
```

Declaration order is dictionary insertion order, so it is whatever your source
says — which also means a dictionary built by merging or comprehension orders
the arcs however that construction happened to.

### `FlowState` fields

| Field | Type | Meaning |
|---|---|---|
| `prompt_name` | `str` | Required. The user-prompt key to render for this state. |
| `transitions` | `dict[str, str]` | Condition name → target state name. |
| `transition_conditions` | `dict[str, TransitionCondition]` | Condition name → condition object. |
| `max_loops` | `int \| None` | How many times this state may be entered before the run stops. `None` = unlimited. |
| `prompt_params` | `dict[str, Any]` | Static parameters merged into the render. |
| `on_enter` | `Callable \| None` | Awaited as `on_enter(state_name, data, context)` before the render. |
| `on_exit` | `Callable \| None` | Awaited as `on_exit(state_name, data, context)` after it. |
| `metadata` | `dict[str, Any]` | Yours; the adapter does not read it. |

Both hooks must be `async def`. A plain function is still *called* — and its
side effects still happen — but awaiting its `None` then raises `TypeError`,
which the adapter logs before continuing. Hook failures never fail the run.

### `ConversationFlow` fields

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Required. Names the flow in errors and node metadata. |
| `initial_state` | `str` | Required. Must be a key of `states`. |
| `states` | `dict[str, FlowState]` | Required, non-empty. |
| `max_total_loops` | `int` | Total state entries across the whole run. Default `10`. |
| `timeout_seconds` | `float \| None` | Declared, and **read by nothing** — see [Current limitations](#current-limitations). |
| `initial_context` | `dict[str, Any]` | Seeds the run's context, and is merged into the initial data. |
| `description` | `str \| None` | Names the machine in the FSM config; defaults to `"Conversation flow: <name>"`. |
| `version` | `str` | Carried into the FSM config. Defaults to `"1.0.0"`. |
| `metadata` | `dict[str, Any]` | Yours; neither the adapter nor the engine reads it. |

`__post_init__` rejects a missing name, a missing `initial_state`, an empty
`states`, an `initial_state` that is not in `states`, and any transition whose
target is neither a declared state nor the literal `"end"`.

## Running a flow

### Through a conversation

`ConversationManager.execute_flow()` runs the flow and appends one assistant
node per executed state to the conversation tree.

```python
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="support_agent",
)

async for node in manager.execute_flow(flow):
    print(node.metadata["state"], "->", node.message.content)
```

Three things to know about that loop:

- **The conversation must already have state.** `execute_flow` raises
  `ValueError: No conversation state` if no message has been added yet. Create
  the manager with a `system_prompt_name`, or `add_message()` first.
- **Nothing is yielded until the run is over.** The adapter executes the whole
  flow, then the manager walks its history and yields. The iterator is a
  reporting surface, not a progress feed.
- **Each node is persisted as it is yielded** — appended to the tree, stamped
  with a `node_id`, and saved through the storage backend.

Node metadata carries `state`, `flow_name`, and `flow_execution: True`, and
`node.prompt_name` is the state's `prompt_name`.

### Directly, through the adapter

Use the adapter when you want the flow's data and statistics without writing
turns into a conversation.

```python
from dataknobs_llm.conversations.flow import ConversationFlowAdapter

adapter = ConversationFlowAdapter(flow=flow, prompt_builder=builder, llm=llm)

final_data = await adapter.execute({"customer_id": "c-42"})
```

`execute()` returns the flow's final data dictionary: your initial data, plus
`response` (the last state's rendered text), `state`, `loop_count`, and
`history`.

`llm=` is optional and is used for one thing: it is seeded into the run context
as `_llm_provider`, where `LLMClassifierCondition` finds it.

**One adapter drives one run.** `execute()` resets the execution state and the
function registry on `self`, and the closures the engine calls read that state,
so two concurrent `execute()` calls on one adapter interleave into each other's
history and loop counts. Construct one per run — `execute_flow` does.

## What a state's prompt can see

The adapter builds the render parameters from three sources, in increasing
priority: the flow's current data, the state's `prompt_params`, and the run
context. It then adds two of its own:

| Parameter | Value |
|---|---|
| `state` | The current state's name |
| `loop_count` | How many times this state has been entered, starting at 1 |

Internal markers — `_llm_provider`, `_force_end`, `_error` — are withheld. The
provider is a live object that can hold a credential, and a nested prompt
reference propagates the parent render's variables into the child's, so
anything passed here travels further than the state it was rendered for.

Templates are rendered in the prompt system's default mixed mode, where
`{{name}}` and `{{ name }}` are not the same: the second preserves the spaces
inside the braces in the output. Prefer `{{name}}`. See
[Prompt Engineering](prompts.md) for the full syntax.

## Transition conditions

All conditions implement `TransitionCondition`:

```python
async def evaluate(self, response: str, context: dict[str, Any]) -> bool: ...
```

`response` is the rendered text of the state being left. `context` is the run
context merged with the flow's current data, the data winning on a key
collision.

### `KeywordCondition`

```python
from dataknobs_llm.conversations.flow import KeywordCondition, keyword_condition

KeywordCondition(keywords=["yes", "sure"], case_sensitive=False, match_whole_word=False)
keyword_condition(["yes", "sure"])                      # same thing
keyword_condition(["bill"], match_whole_word=True)      # "billing" does not match
```

True if any keyword appears in the response. Substring matching by default;
`match_whole_word=True` wraps each keyword in word boundaries.

### `RegexCondition`

```python
import re

from dataknobs_llm.conversations.flow import RegexCondition, regex_condition

RegexCondition(pattern=r"\b\d{3}-\d{4}\b", flags=re.IGNORECASE)
regex_condition(r"\b\d{3}-\d{4}\b")
```

True if the pattern is found anywhere in the response (`re.search`).

### `ContextCondition`

```python
from dataknobs_llm.conversations.flow import context_condition

context_condition(lambda ctx: ctx.get("tier") == "gold")
```

Ignores the response and applies a **synchronous** predicate to the context.
This is how you branch on data rather than on text.

### `AlwaysCondition`

```python
from dataknobs_llm.conversations.flow import always

always()
```

Unconditional. Use it for a linear step, or as the last arc out of a state.

### `SentimentCondition`

```python
from dataknobs_llm.conversations.flow import SentimentCondition

SentimentCondition(expected_sentiment="positive", threshold=0.5)
```

`expected_sentiment` must be `"positive"`, `"negative"` or `"neutral"`;
anything else raises at construction. The classifier is a small built-in
word list, not a model — treat it as a placeholder for a real one, and
remember it is scoring the rendered prompt.

### `CompositeCondition`

```python
from dataknobs_llm.conversations.flow import (
    CompositeCondition,
    context_condition,
    keyword_condition,
)

CompositeCondition(
    conditions=[keyword_condition(["refund"]), context_condition(lambda c: c["tier"] == "gold")],
    operator="and",   # or "or"
)
```

Evaluates every member — there is no short-circuit — then combines. An
`operator` other than `"and"`/`"or"` raises at construction.

### `LLMClassifierCondition`

```python
from dataknobs_llm.conversations.flow import LLMClassifierCondition

LLMClassifierCondition(
    classifier_prompt="Is this about billing? Answer yes or no.\n{{response}}",
    expected_value="yes",
    llm_config=None,
)
```

The only condition that calls a provider. `{{response}}` in
`classifier_prompt` is replaced with the response text; the completion is
stripped, lower-cased, and compared to `expected_value.lower()` for exact
equality — so ask for a single word.

The provider comes from the run context (`_llm_provider`, seeded by the
adapter's `llm=` argument). If there is none and no `llm_config` was given,
`evaluate` raises `ValueError`.

### Writing your own

Subclass `TransitionCondition` and implement both abstract methods:

```python
from dataknobs_llm.conversations.flow import TransitionCondition

class LengthCondition(TransitionCondition):
    def __init__(self, minimum: int) -> None:
        self.minimum = minimum

    async def evaluate(self, response: str, context: dict) -> bool:
        return len(response) >= self.minimum

    def to_fsm_function(self) -> str:
        return f"length_{id(self)}"
```

`to_fsm_function` is part of the abstract interface. The adapter does not call
it — it registers its own wrapper under a name derived from the state and the
condition — but it must be implemented for the class to instantiate.

## Loop guards

Two independent ceilings stop a run that will not terminate.

```python
FlowState(prompt_name="ask", max_loops=2)          # per state
ConversationFlow(name="f", initial_state="ask", states=states, max_total_loops=3)
```

Both are checked when a state is *entered*, before its prompt is rendered.
Tripping either stops the run and fails it:

```
OperationError: Conversation flow 'looping' failed: Max loops exceeded for state ask
```

The counter reads one higher than the ceiling afterwards — the entry that
tripped the guard is counted before it is refused — and no history entry is
recorded for it. `get_execution_summary()["stop_reason"]` carries the cause;
the engine on its own would only report that the state had no arc left.

## When a flow fails

`adapter.execute()` raises `OperationError`. `manager.execute_flow()` wraps it
in `ValueError("Flow execution failed: ...")`, preserving the `OperationError`
as `__cause__`.

Three things fail a run:

| Cause | Message |
|---|---|
| No condition matched, in a non-terminal state | `No valid transitions from state: <name>` |
| A state's prompt could not be rendered | the renderer's error, naming the state |
| A condition *raised* | the condition's own exception |

That last row is deliberate. A condition that raises is not a condition that
answered "no": nothing is caught, so an outage in whatever the condition
consults surfaces as an error instead of being recorded as a data-quality
outcome.

A dead end is a failure, not a graceful stop. If a state should be allowed to
end the conversation, make it terminal — no `transitions` — or give it a final
`always()` arc to a terminal state.

## Inspecting a run

```python
adapter.execution_state.history        # [(state_name, rendered_response), ...]
adapter.execution_state.loop_counts    # {state_name: entries}
adapter.execution_state.context        # the run context
adapter.get_execution_summary()
```

`get_execution_summary()` returns `total_transitions`, `loop_counts`,
`current_state`, `history_length`, `states_visited` and `stop_reason`.

## Validating a flow

Construction catches the structural errors; `validate_flow()` reports the
softer ones as warnings, and returns an empty list when there are none.

```python
warnings = flow.validate_flow()
# ["State 'orphan' is unreachable",
#  "State 'farewell' has no exit transitions (potential dead end)"]
```

Note that a terminal state is reported as a "potential dead end" — that
warning is about shape, not correctness, and a well-formed flow will produce
one per terminal state.

Also available: `flow.get_state(name)` (raises `KeyError`) and
`flow.get_reachable_states(name)`.

## Current limitations

- **A flow needs at least two states.** A single state is both the start and
  the end of its machine, and that combination is not handled: the engine finds
  no start state, falls back to the literal name `start`, and the run fails with
  `No valid transitions from state: start` — naming a state your configuration
  never declared. Naming your one state `start` makes that fallback coincide
  with it and the run succeeds, but only by coincidence; write two states.
- **`timeout_seconds` is not enforced.** It is declared on `ConversationFlow`
  and read by nothing — neither the adapter nor the engine. Bound a run with
  `max_total_loops`, or impose your own deadline with `asyncio.timeout()`.
- **`execute_flow` yields only after the run completes**, so a long flow
  produces no output until it is finished.
- **An adapter instance is single-use per run.** See
  [Directly, through the adapter](#directly-through-the-adapter).

## See Also

- [Conversation Flow Examples](../examples/conversation-flows.md) — runnable patterns
- [FSM-Based Conversation Flow](../examples/fsm-conversation-flow.md) — one complete program
- [Conversation Management](conversations.md) — the manager, branching, persistence
- [Prompt Engineering](prompts.md) — template syntax and the prompt library
- [FSM Package](../../fsm/index.md) — the underlying engine
- [Conversations API](../api/conversations.md) — generated reference

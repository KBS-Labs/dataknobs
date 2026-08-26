# Tool Execution Context

The tool system has two halves. [Tools, Cost Tracking, and Advanced
Features](https://github.com/KBS-Labs/dataknobs/blob/main/docs/packages/llm/guides/tools-and-enhancements.md) covers the
first: `Tool`, `ToolRegistry`, and the function-calling formats a provider
expects. This page covers the second — the tools that need to know *something
about the conversation they are running inside*, which the LLM's tool call does
not tell them.

A `Tool` sees exactly the arguments the model sent. A `ContextAwareTool` also
sees a `ToolExecutionContext`: who the user is, which conversation this is, and
what a wizard has collected so far.

## When you need it

Reach for `ContextAwareTool` when the tool's answer depends on conversation
state rather than on its arguments alone:

- a preview tool that renders the config a wizard has been collecting;
- a save tool that needs the conversation ID to file its output under;
- a search tool that filters by the calling user.

Static dependencies — a database handle, a template registry, a service client
— still go through the constructor. The context is for the things that change
per request.

## Writing one

Implement `execute_with_context` instead of `execute`. The base class's
`execute` finds the context, checks the arguments your schema declares
`required`, and forwards.

```python
from dataknobs_llm.tools import ContextAwareTool, ToolExecutionContext


class SummarizeProgressTool(ContextAwareTool):
    def __init__(self) -> None:
        super().__init__(
            name="summarize_progress",
            description="Summarize what the wizard has collected so far.",
        )

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs,
    ) -> dict:
        data = context.wizard_data()
        if data is None:
            return {"error": "This tool requires a wizard conversation."}
        return {"collected": sorted(data), "stage": context.wizard_state.current_stage}
```

`wizard_data()` returns `None` — not `{}` — when there is no wizard state at
all. That distinction is deliberate: a tool handed an empty dict would write
into a throwaway and report success, which is indistinguishable from working.
Treat `None` as an error condition and say so in the result, as above.

Prefer `wizard_data()` over reaching into `context.wizard_state.collected_data`
directly, for the same reason: the accessor is the one that can tell you there
is nothing there.

## How the context is built

Reasoning strategies call `ToolExecutionContext.from_manager(manager)` and pass
the result as `_context`:

```python
context = ToolExecutionContext.from_manager(manager)
result = await tool.execute(**params, _context=context)
```

Outside that framework — in tests, scripts, or direct invocation — two
shorthands exist:

```python
# Build a context around a data dict
context = ToolExecutionContext.from_wizard_data({"name": "Alice"})

# Or let execute() build one for you
result = await tool.execute(wizard_data={"name": "Alice"})
```

With neither, `execute()` supplies `ToolExecutionContext.empty()`, and
`wizard_data()` returns `None`.

## `ToolWizardState`: what a tool may and may not do

`context.wizard_state` is a `ToolWizardState` — five fields, and the tool-facing
projection of wizard state:

| Field | Meaning |
|---|---|
| `current_stage` | the stage the wizard is on |
| `collected_data` | data collected across all stages |
| `history` | visited stage names |
| `completed` | whether the wizard has finished |
| `stage_metadata` | the current stage's declared config — prompt, schema, `can_skip` |

**Two suppliers build this object, and which one ran decides what your writes
do.** This is the part that catches people out.

- **Published (live).** A reasoning strategy publishes an instance for the
  duration of a turn. `collected_data` is the strategy's own dict, held by
  reference — so a tool's writes land in wizard state, and its reads see values
  extracted earlier in the *same* turn.
- **Metadata fallback (the last save).** When no strategy published,
  `ToolWizardState.from_manager_metadata` reads the *persisted* wizard
  metadata. Writes are visible to the rest of the turn, but the component that
  owns the wizard rewrites that dict from its own state when the turn is saved,
  so they do not survive it. Reads are as old as the last save, for the same
  reason.

A tool cannot tell the two apart, and should not try to. Write what you mean;
whether it persists is the wizard's decision, not the tool's.

`stage_metadata` is empty on the fallback route: the stage's declared
configuration is not part of the persisted state, so there is nothing there to
read. An empty dict therefore means either "this came off the fallback" or "the
stage declares nothing", and the two are deliberately not distinguished.

## Converting from an observability snapshot

`dataknobs_bots` holds a much larger `WizardStateSnapshot` for observability —
transitions, task tracking, main-flow progress. To hand a tool the state it
expects, convert:

```python
snapshot = strategy.get_state_snapshot(manager)
view = snapshot.to_tool_view()          # -> ToolWizardState
context = ToolExecutionContext(wizard_state=view)
```

The conversion runs in this direction only. A snapshot carries fields the tool
view has no room for, so the inverse would have to invent them; a tool that
needs progress or transitions reads the snapshot.

**The converted view's payloads are copies.** A *published* `ToolWizardState`
holds `collected_data` by reference on purpose — that is the live channel. A
snapshot is not that channel: it is already a copy taken at a point in time, so
writes to a converted view go nowhere, and copying makes that structural rather
than a matter of documentation.

`stage_metadata` on a converted view is populated when the snapshot came from
`get_state_snapshot()` and empty when it came from
`WizardReasoning.snapshot_from_metadata()` — the same asymmetry, reached by a
different path, because the field has the same single supplier either way.

## Wrapping a tool you cannot change

`ContextEnhancedTool` adds context awareness to an existing `Tool` without
modifying it. An injector turns the context into kwargs the inner tool already
understands:

```python
from dataknobs_llm.tools import ContextEnhancedTool, default_wizard_data_injector

enhanced = ContextEnhancedTool(legacy_tool, context_injector=default_wizard_data_injector)
```

`default_wizard_data_injector` supplies `wizard_data`, by reference, and
injects nothing when the context carries no wizard state — so the inner tool
falls back to whatever default it declares. Injected values never override
kwargs the caller passed explicitly.

## A note on the old name

`ToolWizardState` was called `WizardStateSnapshot` until `dataknobs-llm` 0.8.0,
which deprecated the old spelling. That name also belongs to the unrelated and
much larger observability dataclass in `dataknobs_bots.reasoning.observability`,
and the two were routinely confused — shipped prose had already documented a
field of the `bots` class under an import of this one.

The old spelling still resolves, from both `dataknobs_llm.tools` and
`dataknobs_llm.tools.context`, and emits a `DeprecationWarning` on access. Type
checkers resolve it to the class directly, so an unmigrated call site keeps full
type precision while it lasts. The removal version is named in the warning
itself.

## See also

- [Tools API](https://github.com/KBS-Labs/dataknobs/blob/main/docs/packages/llm/api/tools.md) — `Tool`, `ToolRegistry`,
  and the provider function-calling formats
- [Tools, Cost Tracking, and Advanced Features](https://github.com/KBS-Labs/dataknobs/blob/main/docs/packages/llm/guides/tools-and-enhancements.md)
  — the non-context-aware half, plus cost tracking and rate limiting

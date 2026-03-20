# Testing DynaBot

This guide covers the testing utilities provided by `dataknobs-bots` for writing
reliable, maintainable tests for wizard bots and DynaBot interactions.

## Overview

Testing DynaBot wizard flows requires wiring together an LLM provider,
conversation storage, extraction, and wizard configuration. The
`dataknobs_bots.testing` module provides high-level helpers that reduce this
from ~50 lines of boilerplate to ~5 lines.

**Key principle:** Test wizard behavior through the public API
(`DynaBot.from_config()` + `bot.chat()`) rather than by constructing
`WizardReasoning` directly and poking internal state.

## BotTestHarness

`BotTestHarness` is the preferred way to test wizard bots. It wraps
`DynaBot.from_config()`, `EchoProvider`, and `ConfigurableExtractor` into a
single object with `chat()`/`greet()` methods and automatic wizard state
capture.

### Basic Usage

```python
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

config = (WizardConfigBuilder("my-wizard")
    .stage("gather", is_start=True, prompt="Tell me your name and topic.")
        .field("name", field_type="string", required=True)
        .field("topic", field_type="string", required=True)
        .transition("done", "data.get('name') and data.get('topic')")
    .stage("done", is_end=True, prompt="All done!")
    .build())

async with await BotTestHarness.create(
    wizard_config=config,
    main_responses=["Got it!", "All done!"],
    extraction_results=[
        [{"name": "Alice", "topic": "math"}],
    ],
) as harness:
    result = await harness.chat("I'm Alice and I like math")
    assert harness.wizard_data["name"] == "Alice"
    assert harness.wizard_data["topic"] == "math"
    assert harness.wizard_stage == "done"
```

### What `create()` Does

1. Builds a bot config dict with `EchoProvider` as main LLM
2. Flattens `extraction_results` into a `ConfigurableExtractor` sequence
3. Calls `DynaBot.from_config()` to create a real bot
4. Injects the `ConfigurableExtractor` via `inject_providers()`
5. Queues `main_responses` on the EchoProvider

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `harness.wizard_stage` | `str \| None` | Current wizard stage after last turn |
| `harness.wizard_data` | `dict` | Wizard state data after last turn |
| `harness.wizard_state` | `dict \| None` | Full normalized wizard state |
| `harness.last_response` | `str` | Bot response from last turn |
| `harness.turn_count` | `int` | Number of turns executed |
| `harness.bot` | `DynaBot` | The underlying bot (for advanced assertions) |
| `harness.provider` | `EchoProvider` | Main provider (for call history) |
| `harness.extractor` | `ConfigurableExtractor \| None` | Extractor (for call verification) |

### TurnResult

Each `chat()` and `greet()` call returns a `TurnResult`:

```python
result = await harness.chat("hello")
assert result.response == "Got it!"
assert result.wizard_stage == "gather"
assert result.wizard_data == {"name": "Alice"}
assert result.turn_index == 1  # 1-based
```

### Per-Turn Extraction Results

`extraction_results` groups extraction calls by turn. The harness flattens them
into a `ConfigurableExtractor` sequence internally, but you think in turns:

```python
extraction_results=[
    # Turn 1: one extraction call
    [{"name": "Alice"}],
    # Turn 2: two calls (initial + escalated scope)
    [{"domain_id": "chess"}, {"name": "Alice", "domain_id": "chess"}],
]
```

### Full Bot Config

For complete control, pass `bot_config` instead of `wizard_config`:

```python
async with await BotTestHarness.create(
    bot_config={
        "llm": {"provider": "echo", "model": "test"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": {
            "strategy": "wizard",
            "wizard_config": {...},
            "extraction_config": {"provider": "echo", "model": "ext"},
        },
    },
    main_responses=["Hello!"],
) as harness:
    await harness.chat("hi")
```

## WizardConfigBuilder

Fluent builder for wizard config dicts. Replaces verbose 40-line inline dicts
with a readable chained API, and validates at build time.

```python
from dataknobs_bots.testing import WizardConfigBuilder

config = (WizardConfigBuilder("quiz-wizard")
    .stage("gather", is_start=True, prompt="What topic?")
        .field("topic", field_type="string", required=True)
        .field("level", field_type="string", required=True, default="beginner")
        .transition("quiz", "data.get('topic') and data.get('level')")
    .stage("quiz", prompt="Answer the question.")
        .field("answer", field_type="string", required=True)
        .transition("done", "data.get('answer')")
    .stage("done", is_end=True, prompt="All done!")
    .settings(extraction_scope="current_message")
    .build())
```

### Methods

| Method | Description |
|--------|-------------|
| `.stage(name, *, is_start, is_end, prompt, mode, extraction_scope, auto_advance, skip_extraction)` | Add a stage |
| `.field(name, *, field_type, required, description, enum, default, x_extraction)` | Add a field to the current stage |
| `.transition(target, condition, priority)` | Add a transition from the current stage |
| `.settings(**kwargs)` | Set wizard-level settings |
| `.build()` | Validate and return the config dict |

### Build-Time Validation

`build()` raises `ValueError` on:

- No start stage defined
- No end stage defined
- Transition to a nonexistent stage name

## inject_providers

Injects LLM providers and extractors into a DynaBot instance for testing.

```python
from dataknobs_bots.testing import inject_providers
from dataknobs_llm import EchoProvider
from dataknobs_llm.testing import ConfigurableExtractor

bot = await DynaBot.from_config(config)

# Replace main LLM
inject_providers(bot, main_provider=EchoProvider({...}))

# Replace extraction provider (swaps provider inside existing SchemaExtractor)
inject_providers(bot, extraction_provider=EchoProvider({...}))

# Replace entire extractor (e.g., with ConfigurableExtractor)
inject_providers(bot, extractor=ConfigurableExtractor(results=[...]))
```

`extractor` and `extraction_provider` are mutually exclusive.

## Extraction Testing Utilities

Two approaches for testing extraction, depending on what you need to verify:

### ConfigurableExtractor (Bypass Extraction Pipeline)

Returns pre-configured results without calling any LLM. Use this when testing
wizard flow behavior, not extraction quality:

```python
from dataknobs_llm.testing import ConfigurableExtractor, SimpleExtractionResult

extractor = ConfigurableExtractor(results=[
    SimpleExtractionResult(data={"name": "Alice"}, confidence=0.9),
    SimpleExtractionResult(data={"topic": "math"}, confidence=0.5),
])

# Track calls
assert len(extractor.extract_calls) == 0
await extractor.extract("text", schema={})
assert len(extractor.extract_calls) == 1
```

### scripted_schema_extractor (Real Extraction Pipeline)

Creates a real `SchemaExtractor` backed by scripted `EchoProvider` responses.
Exercises the full extraction pipeline (prompt building, JSON parsing,
confidence scoring):

```python
from dataknobs_llm.testing import scripted_schema_extractor

extractor, ext_provider = scripted_schema_extractor([
    '{"name": "Alice", "topic": "math"}',
])

# Use with WizardReasoning directly
reasoning = WizardReasoning(wizard_fsm=fsm, extractor=extractor)

# Or inject into a bot
inject_providers(bot, extractor=extractor)
```

## Anti-Patterns

Avoid these patterns in bot tests:

| Anti-Pattern | Why It's Wrong | Use Instead |
|---|---|---|
| `WizardReasoning()` + `reasoning.generate(manager)` | Bypasses `from_config()`, middleware, context pipeline | `BotTestHarness.create()` + `harness.chat()` |
| `bot._conversation_managers` access | Couples to internal cache implementation | `bot.get_wizard_state()` or `harness.wizard_data` |
| `strategy._extractor = extractor` | Private attribute injection | `strategy.set_extractor(ext)` or `inject_providers(bot, extractor=ext)` |
| Per-file `_get_wizard_state()` helpers | Duplicated internal metadata access | `harness.wizard_stage` / `harness.wizard_data` |
| `MagicMock(spec=ConversationManager)` | Mocks hide integration bugs | `BotTestHarness` creates real bots via `from_config()` |
| Inline 40-line wizard config dicts | Verbose, error-prone, copy-pasted | `WizardConfigBuilder` fluent API |

### Exception: Internal-Method Unit Tests

Tests that verify WizardReasoning internal logic (`_evaluate_condition`,
`_can_auto_advance`, transform flows) are legitimate unit tests. These may
use `WizardReasoning` directly with the `conversation_manager_pair` conftest
fixture. They test specific internal methods, not wizard flow behavior.

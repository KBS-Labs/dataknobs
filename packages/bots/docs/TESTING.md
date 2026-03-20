# Testing DynaBot

Testing utilities for wizard bots and DynaBot interactions. These live in
`dataknobs_bots.testing` and `dataknobs_llm.testing`.

## Preferred Pattern: BotTestHarness

Use `BotTestHarness` for all wizard flow tests. It wires `DynaBot.from_config()`,
`EchoProvider`, and `ConfigurableExtractor` into one object:

```python
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

config = (WizardConfigBuilder("test")
    .stage("gather", is_start=True, prompt="Tell me your name.")
        .field("name", field_type="string", required=True)
        .field("topic", field_type="string", required=True)
        .transition("done", "data.get('name') and data.get('topic')")
    .stage("done", is_end=True, prompt="All done!")
    .build())

async with await BotTestHarness.create(
    wizard_config=config,
    main_responses=["Got it!"],
    extraction_results=[[{"name": "Alice", "topic": "math"}]],
) as harness:
    await harness.chat("Alice and math")
    assert harness.wizard_data["name"] == "Alice"
    assert harness.wizard_stage == "done"
```

## Testing Constructs

| Construct | Import | Use For |
|-----------|--------|---------|
| `BotTestHarness` | `from dataknobs_bots.testing import BotTestHarness` | Wizard flow integration tests |
| `WizardConfigBuilder` | `from dataknobs_bots.testing import WizardConfigBuilder` | Building wizard configs fluently |
| `TurnResult` | `from dataknobs_bots.testing import TurnResult` | Per-turn response + state snapshot |
| `inject_providers()` | `from dataknobs_bots.testing import inject_providers` | Provider/extractor injection |
| `ConfigurableExtractor` | `from dataknobs_llm.testing import ConfigurableExtractor` | Scripted extraction (bypasses LLM) |
| `scripted_schema_extractor()` | `from dataknobs_llm.testing import scripted_schema_extractor` | Real extraction pipeline with scripted responses |
| `CaptureReplay` | `from dataknobs_bots.testing import CaptureReplay` | Replay recorded LLM conversations |

## Anti-Patterns

- Do not construct `WizardReasoning` directly for flow tests — use `BotTestHarness`
- Do not access `bot._conversation_managers` — use `bot.get_wizard_state()` or `harness.wizard_data`
- Do not set `strategy._extractor` — use `strategy.set_extractor()` or `inject_providers(bot, extractor=...)`
- Do not use `MagicMock` for bot dependencies — use real objects via `BotTestHarness`
- Do not copy-paste wizard config dicts — use `WizardConfigBuilder`

See the [MkDocs testing guide](../../../docs/packages/bots/guides/testing.md) for
full documentation including `TurnResult` fields, per-turn extraction sequencing,
and the exception for internal-method unit tests.

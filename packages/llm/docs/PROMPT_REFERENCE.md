# Extraction Prompt Key Reference

Configurable prompt keys for the `dataknobs-llm` extraction system.
These keys are used by `SchemaExtractor` and can be overridden via
the prompt library configuration.

## Extraction Prompts (`extraction.*`)

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `extraction.default` | meta | jinja2 | `schema`, `context`, `text` |
| `extraction.default.schema_section` | fragment | format | `schema` |
| `extraction.default.context_section` | fragment | format | `context` |
| `extraction.default.instructions` | fragment | format | -- |
| `extraction.default.message_section` | fragment | format | `text` |
| `extraction.with_assumptions` | meta | jinja2 | `schema`, `context`, `text` |
| `extraction.with_assumptions.instructions` | fragment | format | -- |
| `extraction.with_assumptions.example` | fragment | format | -- |
| `extraction.with_assumptions.message_section` | fragment | format | `text` |

## Usage

```python
from dataknobs_llm.extraction.prompts import get_extraction_prompt_library

library = get_extraction_prompt_library()
template = library.get_system_prompt("extraction.default")
```

See also: [dataknobs-bots Prompt Key Reference](../../bots/docs/PROMPT_REFERENCE.md)
for the full catalog including wizard, review, grounded, and focus prompts.

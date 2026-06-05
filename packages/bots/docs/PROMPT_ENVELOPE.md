# Prompt envelope

`DynaBot` wraps the user prompt's context sections — "Knowledge base",
"Conversation history", "Question" — in a consistent envelope style.
The same style also wraps the knowledge-base block in the
grounded-reasoning synthesis system prompt, so a single configuration
knob covers the user-side and system-side of every turn.

## Why this exists

Small instruction-tuned models can complete an XML-wrapped input shape
by emitting a matching wrapper element around their reply (for example
`<response>...</response>`). The assistant's wrapped output then
persists in conversation history and seeds the next turn —
self-reinforcing. Switching the default envelope to markdown removes
the mirroring cue. Consumers that depend on the legacy XML shape can
pin it back with a one-line config change.

## Styles

| Style | Example output | When to use |
|---|---|---|
| `markdown` (default) | `## Knowledge base\n\nbody` sections separated by `\n\n---\n\n` | The default. Non-mirroring and clearly bounded for the model. |
| `xml` | `<knowledge_base>\nbody\n</knowledge_base>` blocks separated by `\n\n` | Back-compat. Reproduces the pre-fix byte shape exactly. |
| `prose` | `Knowledge base:\n\nbody` blocks separated by `\n\n` | Conservative fallback for models that over-formalize on `##` headings. |

## Selecting a style

Set `prompt_envelope` on the bot config:

```yaml
llm:
  provider: ollama
  model: llama3.2
conversation_storage:
  backend: memory
prompt_envelope: markdown   # default — non-mirroring
```

```python
from dataknobs_bots import DynaBot

# Default behavior — markdown envelope.
bot = await DynaBot.from_config({
    "llm": {"provider": "ollama", "model": "llama3.2"},
    "conversation_storage": {"backend": "memory"},
})

# Opt back into the legacy XML envelope — byte-identical to pre-fix
# output, useful while migrating prompts or fixtures.
bot = await DynaBot.from_config({
    "llm": {"provider": "ollama", "model": "llama3.2"},
    "conversation_storage": {"backend": "memory"},
    "prompt_envelope": "xml",
})
```

An unknown value raises a clear `ValueError` from
`DynaBotConfig.__post_init__` listing the accepted values.

## Direct callers of `KnowledgeBase.format_context` / `ContextFormatter.wrap_for_prompt`

The KB-layer helpers preserve their pre-envelope behavior when called
without an envelope: `wrap_in_tags=True` still produces the legacy
`<knowledge_base>...</knowledge_base>` shape, byte-for-byte identical
to the pre-envelope output. To opt into the envelope shape from a
direct caller, pass `envelope=` keyword:

```python
from dataknobs_bots.prompts import PromptEnvelope, PromptEnvelopeStyle

env = PromptEnvelope(PromptEnvelopeStyle.MARKDOWN)
wrapped = kb.format_context(results, envelope=env)
# Produces "## Knowledge base\n\n..." instead of "<knowledge_base>\n...".
```

When a bot is the caller, the bot wires its own envelope through this
path automatically — direct callers only need this when they bypass
`DynaBot._build_message_with_context`.

## Constructing an envelope explicitly

```python
from dataknobs_bots.prompts import PromptEnvelope, PromptEnvelopeStyle

env = PromptEnvelope(PromptEnvelopeStyle.MARKDOWN)
section = env.section("Knowledge base", "[1] body", tag="knowledge_base")
# → "## Knowledge base\n\n[1] body"

joiner = env.joiner()
# → "\n\n---\n\n"
```

Empty bodies render to the empty string regardless of style, so a
missing section doesn't introduce a stray heading.

## Migration notes

When upgrading, expect a behavior change in model output bytes: the
default user-prompt and synthesis-system-prompt shapes shift from XML
tags to markdown headings. Tests that pin the byte shape of the
augmented user message (assertions like `assert "<question>" in
message`) need to be updated, or the consumer can pin
`prompt_envelope: "xml"` to defer the change.

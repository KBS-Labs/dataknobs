# Prompt Customization Guide

How to override, extend, and compose prompts in DynaBot without modifying
source code.

## Architecture

DynaBot uses a three-layer prompt architecture with clear precedence:

```
Layer 3: Consumer Config (highest priority)
  - Inline prompts in bot config
  - prompt_libraries config entries
         |
Layer 2: Default Prompt Library
  - Built-in prompt fragments and meta-prompts
  - All 77 prompt keys registered with syntax annotations
         |
Layer 1: Prompt Infrastructure
  - TemplateSyntax enum + conversion utilities
  - prompt_ref() for meta-prompt composition
  - Syntax-aware rendering pipeline
```

**Precedence:** Inline `prompts` > `prompt_libraries` > built-in defaults.

## Quick Start: Override a Single Fragment

The simplest customization: override one fragment while keeping the rest
of the composed prompt intact.

```yaml
bot:
  llm:
    provider: ollama
    model: llama3.2
  reasoning:
    strategy: wizard
    wizard_config: my_wizard.yaml
  prompts:
    wizard.clarification.instructions: >
      Please ask a clarifying question in formal, professional language.
      Do not use casual tone or contractions.
```

This changes only the instructions fragment in clarification responses.
The header, preamble, issues, and goal fragments remain unchanged.

## Override Patterns

### 1. Single Fragment Override

Override one piece of a composed prompt:

```yaml
prompts:
  # Change the clarification header
  wizard.clarification.header: "## Please Clarify"

  # Change the validation tone
  wizard.validation.instructions: >
    Please politely request the corrected information.
    Use formal language appropriate for a business setting.
```

### 2. Structured Override (with syntax annotation)

For templates that need Jinja2 features:

```yaml
prompts:
  focus.guidance.instructions:
    template: >
      Stay focused on {{ primary_goal }}. If the user goes off-topic,
      redirect to the current task.
    template_syntax: jinja2
```

### 3. Meta-Prompt Restructure

Override the entire composition to change how fragments are assembled:

```yaml
prompts:
  wizard.clarification:
    template: |
      {{ prompt_ref("wizard.clarification.header") }}

      {{ prompt_ref("wizard.clarification.issues", issue_list=issue_list) }}

      {{ prompt_ref("wizard.clarification.instructions") }}
    template_syntax: jinja2
```

This removes the preamble and goal fragments from clarification
responses while keeping the header, issues, and instructions.

### 4. Custom Review Persona

Add a new review persona by registering its keys:

```yaml
prompts:
  review.persona.security.role:
    template: >
      You are a security reviewer. Your job is to find
      vulnerabilities, data exposure risks, and authentication gaps.
    template_syntax: format

  review.persona.security.focus:
    template: |
      ## Your Focus
      - Input validation and sanitization
      - Authentication and authorization gaps
      - Data exposure or leakage risks
      - Injection vulnerabilities (SQL, XSS, command)
      - Cryptographic weaknesses
    template_syntax: format

  review.persona.security.instructions:
    template: |
      ## Instructions
      1. Examine the artifact for security issues
      2. Classify each issue by severity (critical/high/medium/low)
      3. Provide remediation guidance
      4. Assign a score from 0.0 (critical vulns) to 1.0 (secure)
    template_syntax: format

  review.persona.security:
    template: |
      {{ prompt_ref("review.persona.security.role") }}

      {{ prompt_ref("review.persona.security.focus") }}

      {{ prompt_ref("review.artifact_section",
         artifact_type=artifact_type, artifact_name=artifact_name,
         artifact_purpose=artifact_purpose, artifact_content=artifact_content) }}

      {{ prompt_ref("review.persona.security.instructions") }}
      {{ prompt_ref("review.format.response") }}
    template_syntax: jinja2
```

## Using prompt_libraries

For reusable overrides across multiple bots, use `prompt_libraries`:

```yaml
bot:
  prompt_libraries:
    # Load overrides from a directory
    - type: filesystem
      path: /shared/prompts/
      priority: 10      # Lower = higher precedence

    # Inline config library
    - type: config
      priority: 20
      prompts:
        wizard.clarification.instructions: "Be concise and formal."
```

### Filesystem Library Layout

Filesystem libraries follow the standard `dataknobs-llm` prompt
library layout. Create YAML files in the directory:

```
/shared/prompts/
  system/
    wizard.clarification.instructions.yaml
    review.persona.custom.yaml
```

Each YAML file contains a `PromptTemplateDict`:

```yaml
# wizard.clarification.instructions.yaml
template: "Please ask for clarification in a formal tone."
template_syntax: format
```

## Template Syntax

### Format Syntax (default for fragments)

Simple variable substitution using `{var}`:

```
Hello {name}, you have {count} items.
```

For literal braces (e.g., JSON examples), double them: `{{` and `}}`.

### Jinja2 Syntax (for meta-prompts and overrides)

Full Jinja2 features: variables, conditionals, loops, filters:

```jinja2
{{ prompt_ref("wizard.clarification.header") }}

{% if current_task %}
**Current Task**: {{ current_task }}
{% endif %}

{{ prompt_ref("wizard.clarification.instructions") }}
```

### prompt_ref() Function

Available in Jinja2 templates. Looks up a prompt key from the library
and renders it with the provided variables:

```jinja2
{{ prompt_ref("key.name", var1=value1, var2=value2) }}
```

Resolution is recursive -- a `prompt_ref`'d prompt can itself contain
`prompt_ref()` calls. Cycle detection prevents infinite loops.

## CLI Utilities

The syntax module provides command-line tools for prompt authoring:

```bash
# Convert format syntax to jinja2
echo "Hello {name}" | uv run python -m dataknobs_llm.prompts.syntax \
  convert --from format --to jinja2
# Output: Hello {{ name }}

# Detect syntax of a prompt file
uv run python -m dataknobs_llm.prompts.syntax detect prompt.txt
# Output: format

# Validate a prompt against a key's placeholder requirements
uv run python -m dataknobs_llm.prompts.syntax validate \
  --key wizard.clarification.issues prompt.txt
```

## Programmatic Access

```python
from dataknobs_bots.prompts.defaults import get_default_prompt_library
from dataknobs_bots.prompts.resolver import PromptResolver

# Create a resolver with default prompts
library = get_default_prompt_library()
resolver = PromptResolver(library)

# Resolve and render a prompt
text = resolver.resolve(
    "wizard.clarification",
    issue_list="- missing name",
    stage_prompt="Enter your name",
    suggestions_text="",
)

# Access the full library (bots + extraction)
from dataknobs_bots.prompts.defaults import get_full_prompt_library
full_library = get_full_prompt_library()
```

## See Also

- [Prompt Key Reference](PROMPT_REFERENCE.md) -- complete catalog of all keys
- Source modules: `dataknobs_bots.prompts.wizard`, `.memory`, `.rubric`,
  `.review`, `.grounded`, `.focus`, `.defaults`

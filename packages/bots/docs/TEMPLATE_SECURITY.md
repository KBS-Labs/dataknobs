# Template Security Guide

Security reference for wizard template rendering in `dataknobs-bots`.

## Security Model

All wizard template rendering uses Jinja2
[`SandboxedEnvironment`](https://jinja.palletsprojects.com/en/3.1.x/sandbox/).
This blocks attribute traversal attacks (`__class__`, `__globals__`,
`__subclasses__`, etc.) that would otherwise allow server-side template
injection (SSTI) leading to remote code execution.

User-entered wizard state data is treated as **data** (Jinja2 context
variables), never as **code** (template text). Even if a user enters
`{{ malicious }}` as a field value, it appears as literal text in the
rendered output because it is never parsed as template syntax.

All rendering routes through a single `WizardRenderer` class that
enforces these properties consistently across every template site.

## Two Template Syntaxes

### Jinja2 (Primary)

Standard Jinja2 syntax is available in all template fields:

- `{{ variable }}` — variable substitution
- `{{ variable | default('fallback') }}` — filters
- `{% if condition %}...{% endif %}` — conditionals
- `{% for item in list %}...{% endfor %}` — loops

### `(( ))` Conditional Syntax (Mixed Mode)

The `context_template` field supports an additional preprocessing
syntax from `dataknobs-llm`:

- `((content with {{variable}}))` — the section is **removed** if all
  `{{variables}}` inside are empty, missing, or None
- `((content with {{variable}}))` — the section is **rendered** (without
  the outer parentheses) if any variable has a value

This preprocessing runs **before** Jinja2 rendering.

## Context Partitioning

The `(( ))` preprocessor works by regex-substituting `{{variable}}`
values directly into the template text via `str(value)`. The result is
then parsed as Jinja2. This is a **double-interpretation pattern** that
creates an injection risk if user-controlled values enter the
substitution step.

`WizardRenderer` mitigates this by partitioning the rendering context
into two sets:

| Context Set | Contents | Used By |
|---|---|---|
| **Template params** | Author-controlled values only: stage metadata (`stage_name`, `stage_label`, `stage_prompt`, `help_text`, `suggestions`), navigation flags (`can_skip`, `can_go_back`), wizard progress (`completed`, `history`) | `(( ))` preprocessor — values substituted into template text |
| **Full context** | Template params + user-entered state data (`collected_data`, `all_data`, top-level field variables) | Jinja2 rendering — values available as context variables (data, not code) |

User-data variables referenced in the template (e.g., `{{topic}}`) are
**not** in the template params, so they survive the `(( ))` preprocessing
step unchanged. They resolve safely in the Jinja2 phase as context
variables.

## Implications for Config Authors

### Use Jinja2 `{% if %}` for conditionals on user data

The `(( ))` preprocessor only has access to author-controlled values.
User data variables are not substituted during preprocessing, so a
`(( ))` section referencing only user data will be removed (the
preprocessor sees the variables as missing).

```yaml
# CORRECT: Jinja2 conditional on user data
context_template: |
  {% if topic %}Your topic is {{ topic }}.{% endif %}

# WRONG: (( )) on user data — section removed because topic
# is not in the author-controlled template params
context_template: |
  ((Your topic is {{topic}}.))
```

### Use `(( ))` for author-controlled structure

```yaml
# CORRECT: (( )) with author-controlled field
context_template: |
  Stage: {{stage_name}}((, help: {{help_text}}))
```

### User data in templates is safe by default

`{{ topic }}` in `response_template`, `prompt`, or `suggestions`
resolves from the Jinja2 context. Even if a user enters
`{{ malicious }}` as a field value, it appears as literal text — it is
data, not code.

### Do not construct template strings from user input

The security model assumes templates are author-defined (YAML config).
If application code dynamically builds template strings incorporating
user input, the sandbox protections may be insufficient. Templates must
be static author content.

## Variable Availability

All template sites use the canonical context from `WizardRenderer.build_context()`:

| Variable | Type | Source | Description |
|---|---|---|---|
| `stage_name` | `str` | Author | Current stage name |
| `stage_label` | `str` | Author | Stage display label |
| `stage_prompt` | `str` | Author | Stage prompt text |
| `help_text` | `str` | Author | Additional help (empty string if none) |
| `suggestions` | `list[str]` | Author | Quick-reply suggestions |
| `completed` | `bool` | Author | Whether wizard is complete |
| `history` | `list[str]` | Author | Visited stage names |
| `can_skip` | `bool` | Author | Whether stage can be skipped. Accurate in `context_template`, `stage_prompt`, and `metadata`; defaults to `False` in other template sites unless the caller passes `extra_context`. |
| `can_go_back` | `bool` | Author | Whether back navigation is allowed. Same accuracy note as `can_skip`. |
| `collected_data` | `dict` | User | Non-internal state data (excludes `_` keys) |
| `all_data` | `dict` | User | All state data including internal and transient |
| `raw_data` | `dict` | User | Persistent state data including internal keys |
| *(top-level keys)* | varies | User | Each key in `state.data` and `state.transient` is available as a top-level variable |
| `bank` | `MemoryBank \| None` | System | MemoryBank accessor. Non-`None` only in `response_template`; `None` in all other template sites. Use `{% if bank %}` guards. |
| `artifact` | `ArtifactBank \| None` | System | ArtifactBank. Non-`None` only in `response_template`; `None` in all other template sites. Use `{% if artifact %}` guards. |

**Author** = author-controlled, safe for `(( ))` preprocessing.
**User** = user-entered data, Jinja2 context only.
**System** = framework-provided, populated only in `response_template` (via `extra_context`).

## Error Behavior

| Situation | Behavior |
|---|---|
| Undefined variable | Renders as empty string (default mode) |
| Template syntax error in `stage_prompt` | Falls back to raw prompt text |
| Template syntax error in suggestions | Individual broken item returned as-is |
| Template syntax error in `response_template` | Exception propagates |
| Sandbox blocks an operation | `SecurityError` raised |
| Template syntax error in `context_template` | Exception propagates |
| Template syntax error in clarification template | Falls back to default list format |

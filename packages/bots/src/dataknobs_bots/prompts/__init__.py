"""Default prompt definitions for the dataknobs-bots package.

This package contains all default LLM prompts used by DynaBot components,
organized into modules by functional area. Each module defines prompt
fragments as ``PromptTemplateDict`` constants and provides a factory
function to create a ``ConfigPromptLibrary`` for that area.

Modules:
- ``wizard`` — Wizard response prompts (clarification, validation, etc.)
- ``memory`` — Summary memory prompt
- ``rubric`` — Rubric feedback and classification prompts
- ``defaults`` — Aggregated default prompt library (all modules combined)

Prompt key namespaces:
- ``wizard.*`` — Wizard response fragments and meta-prompts
- ``memory.*`` — Memory-related prompts
- ``rubric.*`` — Rubric evaluation prompts
"""

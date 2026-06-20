"""Default prompt definitions for the dataknobs-bots package.

This package contains all default LLM prompts used by DynaBot components,
organized into modules by functional area. Each module defines prompt
fragments as ``PromptTemplateDict`` constants and registers them in a
prompt key dict that the ``defaults`` module aggregates.

Modules:
- ``wizard`` — Wizard response prompts (clarification, validation, etc.)
- ``memory`` — Summary memory prompt
- ``rubric`` — Rubric feedback and classification prompts
- ``review`` — Review persona fragments and meta-prompts
- ``grounded`` — Grounded synthesis fragments and meta-prompts
- ``focus`` — Focus guidance and drift correction prompts
- ``envelope`` — Prompt envelope helper for labeled context sections
- ``defaults`` — Aggregated default prompt library (all modules combined)

Prompt key namespaces:
- ``wizard.*`` — Wizard response fragments and meta-prompts
- ``memory.*`` — Memory-related prompts
- ``rubric.*`` — Rubric evaluation prompts
- ``review.*`` — Review persona fragments and meta-prompts
- ``grounded.*`` — Grounded synthesis fragments and meta-prompts
- ``focus.*`` — Focus guidance and drift correction prompts
"""

from dataknobs_bots.prompts.envelope import (
    SECTION_LABELS,
    PromptEnvelope,
    PromptEnvelopeStyle,
)
from dataknobs_bots.prompts.scope import JinjaInputsProjector

__all__ = [
    "SECTION_LABELS",
    "JinjaInputsProjector",
    "PromptEnvelope",
    "PromptEnvelopeStyle",
]

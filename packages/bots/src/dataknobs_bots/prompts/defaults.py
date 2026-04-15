"""Aggregated default prompt library for the dataknobs-bots package.

Combines all prompt modules into a single ``ConfigPromptLibrary`` that serves
as the default (Layer 2) in the three-layer prompt architecture:

    Consumer Config > Default Prompt Library > Prompt Infrastructure

The :func:`get_default_prompt_library` factory creates a library with all
bots-level prompt keys registered. The :func:`get_full_prompt_library` factory
additionally includes the extraction prompt keys from ``dataknobs-llm``,
returning a ``CompositePromptLibrary`` with both layers.

Prompt key namespaces included:
- ``wizard.*`` — Wizard response fragments and meta-prompts
- ``memory.*`` — Memory-related prompts
- ``rubric.*`` — Rubric evaluation prompts
- ``review.*`` — Review persona fragments and meta-prompts
- ``grounded.*`` — Grounded synthesis fragments and meta-prompts
- ``focus.*`` — Focus guidance and drift correction prompts
- ``extraction.*`` — Extraction prompt fragments and meta-prompts (full library only)
"""

import logging

from dataknobs_llm.prompts import ConfigPromptLibrary, CompositePromptLibrary
from dataknobs_llm.prompts.base.types import PromptTemplateDict

from dataknobs_bots.prompts.wizard import WIZARD_PROMPT_KEYS
from dataknobs_bots.prompts.memory import MEMORY_PROMPT_KEYS
from dataknobs_bots.prompts.rubric import RUBRIC_PROMPT_KEYS
from dataknobs_bots.prompts.review import REVIEW_PROMPT_KEYS
from dataknobs_bots.prompts.grounded import GROUNDED_PROMPT_KEYS
from dataknobs_bots.prompts.focus import FOCUS_PROMPT_KEYS

logger = logging.getLogger(__name__)


_MODULES: list[tuple[str, dict[str, PromptTemplateDict]]] = [
    ("wizard", WIZARD_PROMPT_KEYS),
    ("memory", MEMORY_PROMPT_KEYS),
    ("rubric", RUBRIC_PROMPT_KEYS),
    ("review", REVIEW_PROMPT_KEYS),
    ("grounded", GROUNDED_PROMPT_KEYS),
    ("focus", FOCUS_PROMPT_KEYS),
]

# Lazily collected on first access — avoids hard ImportError on key collision.
_all_bots_keys: dict[str, PromptTemplateDict] | None = None


def get_all_bots_keys() -> dict[str, PromptTemplateDict]:
    """Merge all bots prompt key registries into a single dict.

    Returns a cached dict of all prompt keys across all bots prompt modules.
    Useful for testing and introspection.

    Raises ``ValueError`` if any key appears in more than one module
    (namespacing should prevent this, but we verify defensively).
    """
    global _all_bots_keys  # noqa: PLW0603
    if _all_bots_keys is not None:
        return _all_bots_keys

    all_keys: dict[str, PromptTemplateDict] = {}
    for module_name, keys in _MODULES:
        for key, template in keys.items():
            if key in all_keys:
                raise ValueError(
                    f"Duplicate prompt key {key!r} found in module "
                    f"{module_name!r}. Each key must be unique across "
                    f"all prompt modules."
                )
            all_keys[key] = template

    logger.debug(
        "Collected %d bots prompt keys from %d modules",
        len(all_keys),
        len(_MODULES),
    )
    _all_bots_keys = all_keys
    return _all_bots_keys


def get_default_prompt_library() -> ConfigPromptLibrary:
    """Create a ``ConfigPromptLibrary`` with all default bots prompt keys.

    This is the Layer 2 default library containing wizard, memory, rubric,
    review, grounded, and focus prompt keys. It does **not** include
    extraction prompts (which live in ``dataknobs-llm``).

    Returns:
        A ``ConfigPromptLibrary`` with all bots prompt keys registered
        as system prompts.
    """
    return ConfigPromptLibrary(config={"system": get_all_bots_keys()})


def get_full_prompt_library() -> CompositePromptLibrary:
    """Create a ``CompositePromptLibrary`` combining bots and extraction defaults.

    Returns a composite library that searches the bots default library
    first, then falls back to the extraction prompt library from
    ``dataknobs-llm``. This provides a single library with all default
    prompt keys across both packages.

    Returns:
        A ``CompositePromptLibrary`` with bots defaults (priority) and
        extraction defaults (fallback).
    """
    from dataknobs_llm.extraction.prompts import get_extraction_prompt_library

    bots_library = get_default_prompt_library()
    extraction_library = get_extraction_prompt_library()

    return CompositePromptLibrary(
        libraries=[bots_library, extraction_library],
        names=["bots_defaults", "extraction_defaults"],
    )

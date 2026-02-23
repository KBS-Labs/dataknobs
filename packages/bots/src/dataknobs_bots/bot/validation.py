"""Bot capability validation utilities.

Infers required LLM capabilities from bot configuration structure and
validates them against environment resources. Provides defense-in-depth:

- **Pre-deployment** (Layer 1): Warn during registration
- **Startup** (Layer 2): Reject at bot creation time

Usage:
    ```python
    from dataknobs_bots.bot.validation import (
        infer_capability_requirements,
        validate_bot_capabilities,
    )

    # Infer from config structure
    requirements = infer_capability_requirements(bot_config)
    # => ["function_calling"] for react + tools

    # Validate against environment
    warnings = validate_bot_capabilities(bot_config, environment)
    for w in warnings:
        logger.warning(w)
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataknobs_config import EnvironmentConfig

logger = logging.getLogger(__name__)

# Models known to support tool/function calling (shared heuristic).
# Keep in sync with OllamaProvider.get_capabilities().
TOOL_CAPABLE_MODEL_FRAGMENTS: list[str] = [
    "llama3",
    "mistral",
    "mixtral",
    "qwen",
    "command-r",
    "phi3",
    "phi4",
    "nemotron",
    "firefunction",
    "hermes",
]


def infer_capability_requirements(bot_config: dict[str, Any]) -> list[str]:
    """Infer required LLM capabilities from bot config structure.

    Examines the reasoning strategy and tool declarations to determine
    what the bot needs from its LLM provider, even without explicit
    ``$requires`` declarations.

    Args:
        bot_config: Bot configuration dictionary (the "bot" section).

    Returns:
        Deduplicated list of required capability names.
    """
    requirements: list[str] = []

    strategy = bot_config.get("reasoning", {}).get("strategy")
    has_tools = bool(bot_config.get("tools"))

    if strategy == "react" and has_tools:
        requirements.append("function_calling")
    if strategy == "wizard":
        requirements.append("json_mode")

    # Merge with explicit $requires if present
    llm_config = bot_config.get("llm", {})
    explicit = llm_config.get("$requires", [])
    if isinstance(explicit, list):
        requirements.extend(explicit)

    return list(set(requirements))


def _model_supports_capability(
    model_name: str,
    capability: str,
) -> bool | None:
    """Heuristic check whether a model supports a capability.

    Returns:
        True/False if the heuristic is confident, None if unknown.
    """
    model_lower = model_name.lower()

    if capability == "function_calling":
        if any(frag in model_lower for frag in TOOL_CAPABLE_MODEL_FRAGMENTS):
            return True
        # Models known NOT to support tools
        if any(frag in model_lower for frag in ["gemma", "tinyllama"]):
            return False
        return None  # Unknown model — can't tell

    # For other capabilities we don't have heuristics yet
    return None


def validate_bot_capabilities(
    bot_config: dict[str, Any],
    environment: EnvironmentConfig,
) -> list[str]:
    """Validate bot capability requirements against environment resources.

    Checks both inferred and explicit requirements against the resolved
    LLM resource. Uses capability metadata on the resource if available,
    falling back to model-name heuristics.

    Args:
        bot_config: Bot configuration dictionary (the "bot" section).
        environment: Loaded EnvironmentConfig for resource lookup.

    Returns:
        List of warning/error messages. Empty list means valid.
    """
    requirements = infer_capability_requirements(bot_config)
    if not requirements:
        return []

    warnings: list[str] = []

    # Resolve the LLM resource
    llm_config = bot_config.get("llm", {})
    resource_name = llm_config.get("$resource")
    resource_type = llm_config.get("type", "llm_providers")

    if not resource_name:
        # No resource reference — can't validate statically
        return []

    try:
        resolved = environment.get_resource(resource_type, resource_name)
    except KeyError:
        warnings.append(
            f"LLM resource '{resource_name}' not found in environment "
            f"'{environment.name}' — cannot validate capabilities"
        )
        return warnings

    # Check against explicit capabilities metadata on the resource
    declared_capabilities = resolved.get("capabilities")
    model_name = resolved.get("model", "")

    for req in requirements:
        if declared_capabilities is not None:
            # Resource declares its capabilities — authoritative check
            if req not in declared_capabilities:
                warnings.append(
                    f"Requires '{req}' but resource '{resource_name}' "
                    f"declares capabilities: {declared_capabilities}"
                )
        else:
            # Fall back to model-name heuristic
            supports = _model_supports_capability(model_name, req)
            if supports is False:
                warnings.append(
                    f"Requires '{req}' but model '{model_name}' on resource "
                    f"'{resource_name}' is not known to support it"
                )
            elif supports is None:
                warnings.append(
                    f"Requires '{req}' — unable to verify for model "
                    f"'{model_name}' on resource '{resource_name}'"
                )

    return warnings

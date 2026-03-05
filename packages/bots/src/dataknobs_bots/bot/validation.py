"""Bot capability validation utilities.

Infers required LLM capabilities from bot configuration structure and
validates them against environment resources. Provides defense-in-depth:

- **Pre-deployment** (Layer 1): Warn during registration
- **Startup** (Layer 2): Reject at bot creation time

Capabilities are assigned to **roles** so multi-LLM bots (e.g. wizard with
a separate extraction LLM) validate each requirement against the correct
provider.

Usage:
    ```python
    from dataknobs_bots.bot.validation import (
        infer_capability_requirements,
        infer_main_capability_requirements,
        validate_bot_capabilities,
    )

    # Per-role requirements
    role_reqs = infer_capability_requirements(bot_config)
    # => {"main": ["function_calling"], "extraction": ["json_mode"]}

    # Main-only (for startup check against instantiated LLM)
    main_reqs = infer_main_capability_requirements(bot_config)
    # => ["function_calling"]

    # Full validation against environment resources
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
# Keep in sync with OllamaProvider._detect_capabilities().
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


def infer_capability_requirements(
    bot_config: dict[str, Any],
) -> dict[str, list[str]]:
    """Infer required LLM capabilities per role from bot config structure.

    Examines the reasoning strategy and tool declarations to determine
    what each LLM role needs from its provider, even without explicit
    ``$requires`` declarations.

    Roles:
        - ``"main"``: The primary LLM (natural-language generation).
        - ``"extraction"``: The extraction LLM used by wizard strategy
          for JSON schema extraction.

    Args:
        bot_config: Bot configuration dictionary (the "bot" section).

    Returns:
        Dict mapping role name to deduplicated list of required capability
        names. Only roles with requirements are included.
    """
    main_reqs: list[str] = []
    extraction_reqs: list[str] = []

    strategy = bot_config.get("reasoning", {}).get("strategy")
    has_tools = bool(bot_config.get("tools"))

    if strategy == "react" and has_tools:
        main_reqs.append("function_calling")
    if strategy == "wizard":
        # json_mode is needed by the extraction LLM, not the main LLM
        extraction_reqs.append("json_mode")

    # Merge with explicit $requires if present (applies to main LLM)
    llm_config = bot_config.get("llm", {})
    explicit = llm_config.get("$requires", [])
    if isinstance(explicit, list):
        main_reqs.extend(explicit)

    result: dict[str, list[str]] = {}
    if main_reqs:
        result["main"] = list(set(main_reqs))
    if extraction_reqs:
        result["extraction"] = list(set(extraction_reqs))
    return result


def infer_main_capability_requirements(
    bot_config: dict[str, Any],
) -> list[str]:
    """Infer capability requirements for the main LLM only.

    Convenience wrapper for the startup check in ``_build_from_config()``,
    which has the main LLM instance available but not the extraction LLM.

    Args:
        bot_config: Bot configuration dictionary (the "bot" section).

    Returns:
        Deduplicated list of required capability names for the main LLM.
    """
    role_reqs = infer_capability_requirements(bot_config)
    return role_reqs.get("main", [])


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


def _validate_resource_capabilities(
    requirements: list[str],
    resource_name: str,
    resolved: dict[str, Any],
    role_label: str,
) -> list[str]:
    """Check requirements against a resolved resource.

    Args:
        requirements: Capability names required.
        resource_name: Name of the resource for messages.
        resolved: Resolved resource config dict.
        role_label: Human-readable role label for messages (e.g. "main LLM").

    Returns:
        List of warning messages.
    """
    warnings: list[str] = []
    declared_capabilities = resolved.get("capabilities")
    model_name = resolved.get("model", "")

    for req in requirements:
        if declared_capabilities is not None:
            if req not in declared_capabilities:
                warnings.append(
                    f"{role_label} requires '{req}' but resource "
                    f"'{resource_name}' declares capabilities: "
                    f"{declared_capabilities}"
                )
        else:
            supports = _model_supports_capability(model_name, req)
            if supports is False:
                warnings.append(
                    f"{role_label} requires '{req}' but model '{model_name}' "
                    f"on resource '{resource_name}' is not known to support it"
                )
            elif supports is None:
                warnings.append(
                    f"{role_label} requires '{req}' — unable to verify for "
                    f"model '{model_name}' on resource '{resource_name}'"
                )

    return warnings


def validate_bot_capabilities(
    bot_config: dict[str, Any],
    environment: EnvironmentConfig,
) -> list[str]:
    """Validate bot capability requirements against environment resources.

    Checks per-role requirements against the correct LLM resource:
    - ``"main"`` requirements are checked against ``bot_config["llm"]``
    - ``"extraction"`` requirements are checked against
      ``bot_config["reasoning"]["config"]["extraction_config"]``

    Uses capability metadata on the resource if available, falling back
    to model-name heuristics.

    Args:
        bot_config: Bot configuration dictionary (the "bot" section).
        environment: Loaded EnvironmentConfig for resource lookup.

    Returns:
        List of warning/error messages. Empty list means valid.
    """
    role_reqs = infer_capability_requirements(bot_config)
    if not role_reqs:
        return []

    warnings: list[str] = []

    # --- Main LLM validation ---
    main_reqs = role_reqs.get("main", [])
    if main_reqs:
        llm_config = bot_config.get("llm", {})
        resource_name = llm_config.get("$resource")
        resource_type = llm_config.get("type", "llm_providers")

        if resource_name:
            try:
                resolved = environment.get_resource(resource_type, resource_name)
                warnings.extend(_validate_resource_capabilities(
                    main_reqs, resource_name, resolved, "Main LLM",
                ))
            except KeyError:
                warnings.append(
                    f"LLM resource '{resource_name}' not found in environment "
                    f"'{environment.name}' — cannot validate main LLM capabilities"
                )

    # --- Extraction LLM validation ---
    extraction_reqs = role_reqs.get("extraction", [])
    if extraction_reqs:
        reasoning_config = bot_config.get("reasoning", {}).get("config", {})
        extraction_config = reasoning_config.get("extraction_config", {})

        # extraction_config may use $resource or inline provider/model
        resource_name = extraction_config.get("$resource")
        if resource_name:
            resource_type = extraction_config.get("type", "llm_providers")
            try:
                resolved = environment.get_resource(resource_type, resource_name)
                warnings.extend(_validate_resource_capabilities(
                    extraction_reqs, resource_name, resolved, "Extraction LLM",
                ))
            except KeyError:
                warnings.append(
                    f"Extraction LLM resource '{resource_name}' not found in "
                    f"environment '{environment.name}' — cannot validate "
                    f"extraction capabilities"
                )
        elif extraction_config.get("model"):
            # Inline config — validate against model name directly
            inline_resolved = {
                "model": extraction_config.get("model", ""),
                "capabilities": extraction_config.get("capabilities"),
            }
            warnings.extend(_validate_resource_capabilities(
                extraction_reqs,
                f"inline:{extraction_config.get('model', 'unknown')}",
                inline_resolved,
                "Extraction LLM",
            ))
        # If no extraction_config at all, skip — the wizard strategy will
        # handle extraction with its own defaults

    return warnings

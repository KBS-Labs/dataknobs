"""Claude-family request-shape knowledge shared across providers.

Claude runs behind **two** dataknobs providers — the native Anthropic Messages
API (:mod:`~dataknobs_llm.llm.providers.anthropic`) and the Amazon Bedrock
Converse API (:mod:`~dataknobs_llm.llm.providers.bedrock`, which runs Claude on
Bedrock). Both serve the *same model families* with the *same* per-model
output-token (``max_tokens``) ceilings and the *same* Claude-5 ``temperature``
rejection, so that family knowledge lives here — imported by both — rather than
duplicated in each provider (where it would inevitably drift).

Two kinds of Claude constraint live here:

- **``max_tokens`` ceilings** — the bundled fallback resource
  (``data/anthropic_model_limits.yaml``) and the family-matching resolver
  (:func:`match_ceiling` / :func:`resource_ceiling`). The native Anthropic
  provider layers a *live Models-API* source on top of this fallback (a generic
  :class:`~dataknobs_llm.llm.model_profile.LiveApiSource` bound to the
  Anthropic ``list_models`` walker + ceiling extractor in ``anthropic.py``); the
  resource resolution itself is shared because a Claude model's output ceiling
  is a property of the *model*, not the endpoint.
- **Claude-5 ``temperature`` rejection** (:func:`claude_rejects_temperature`) —
  the Claude 5 generation does not support the ``temperature`` sampling
  parameter; this is a model-family property, true regardless of which endpoint
  serves it.

The resource is read once at import into :data:`RESOURCE_MODEL_LIMITS` via
:mod:`importlib.resources`, so the disk read never lands on the event loop or a
per-request path (the async-transport rule is satisfied — no I/O in any
``async def``).
"""

from __future__ import annotations

import importlib.resources
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from dataknobs_common.config_loading import load_yaml_or_json

from ..base import ModelCapability
from ..model_profile import CallableModelMetadataSource, ModelProfile

logger = logging.getLogger(__name__)

#: Claude model families that reject the ``temperature`` sampling parameter at
#: the request boundary. Matched as lowercase substrings of the model id,
#: distinguishing the Claude 5 generation (``claude-sonnet-5``,
#: ``claude-opus-5``, ...) from the Claude 4.x generation (``claude-opus-4-8``,
#: ``claude-haiku-4-5-...``), which still accepts ``temperature``. The Claude 5
#: models do not support ``temperature`` at all — a model-family property, so it
#: holds on the native Anthropic endpoint *and* on Bedrock Converse. Consumers
#: can declare or withdraw the rule at runtime via ``LLMConfig.constraints``
#: without a dataknobs release (see
#: :class:`~dataknobs_llm.llm.base.ModelConstraints`).
CLAUDE_5_TEMPERATURE_REJECTORS: tuple[str, ...] = (
    "claude-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def _project_field(entry: Any, field: str) -> int | None:
    """Project one ``models:`` entry to an int ceiling for *field*, or ``None``.

    Tolerant of **both** resource shapes so a stale on-disk file always loads:

    - **Nested** (current): ``{max_tokens: N, max_input_tokens: M}`` — the named
      field is read (missing → ``None``).
    - **Legacy flat int** (pre-input-ceiling): a bare ``int`` is the *output*
      ``max_tokens`` ceiling only; there is no input value (``max_input_tokens``
      → ``None``).
    """
    if isinstance(entry, dict):
        value = entry.get(field)
    elif field == "max_tokens":
        value = entry  # legacy flat form: the bare int is the output ceiling
    else:
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def project_model_limits(
    section: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    """Project a raw ``models:`` mapping into (output, input) ceiling maps.

    The single tolerant projection shared by the packaged-resource loaders
    (below) and the maintainer tooling
    (:mod:`dataknobs_llm.tooling.model_limits`), so the two can never drift on
    how the nested / legacy-flat resource shapes are read. Each returned map is
    ``{lowercased-model-id: ceiling}`` and omits models with no value for that
    field.
    """
    output: dict[str, int] = {}
    input_ceilings: dict[str, int] = {}
    for key, entry in section.items():
        lower = str(key).lower()
        out = _project_field(entry, "max_tokens")
        if out is not None:
            output[lower] = out
        inp = _project_field(entry, "max_input_tokens")
        if inp is not None:
            input_ceilings[lower] = inp
    return output, input_ceilings


def _load_models_section() -> Mapping[str, Any]:
    """Read the bundled resource's raw ``models:`` mapping (id → int | dict)."""
    ref = (
        importlib.resources.files("dataknobs_llm.llm.providers")
        / "data"
        / "anthropic_model_limits.yaml"
    )
    with importlib.resources.as_file(ref) as path:
        data = load_yaml_or_json(path, require_dict=True)
    return data.get("models") or {}


def load_model_limits_resource() -> dict[str, int]:
    """Load the bundled Claude ``max_tokens`` (output) fallback resource.

    Read once at import (below) into :data:`RESOURCE_MODEL_LIMITS`. The resource
    is fallback-only: the primary source is the live Models API ``max_tokens``
    on the native Anthropic endpoint (the per-provider
    :class:`~dataknobs_llm.llm.model_profile.LiveApiSource`).
    """
    output, _ = project_model_limits(_load_models_section())
    return output


def load_model_input_limits_resource() -> dict[str, int]:
    """Load the bundled Claude ``max_input_tokens`` (context) fallback resource.

    The input-ceiling sibling of :func:`load_model_limits_resource`, read once at
    import into :data:`RESOURCE_MODEL_INPUT_LIMITS`. Fallback-only: the primary
    source is the live Models API ``max_input_tokens`` column.
    """
    _, input_ceilings = project_model_limits(_load_models_section())
    return input_ceilings


try:
    #: Bundled fallback maps ``{lowercased-model-id: ceiling}`` for the output
    #: (``max_tokens``) and input (``max_input_tokens``) ceilings, read once at
    #: import. Consulted only when the dynamic Models-API path has produced no
    #: value for a model (see :func:`claude_resource_profile`, the
    #: bundled-resource half of the layered ceiling resolution).
    #: Degrade to ``{}`` if the resource is unreadable so a data-file issue
    #: never breaks import — the packaging regression is caught instead by an
    #: ``importlib.resources`` test.
    _section = _load_models_section()
    RESOURCE_MODEL_LIMITS: dict[str, int]
    RESOURCE_MODEL_INPUT_LIMITS: dict[str, int]
    RESOURCE_MODEL_LIMITS, RESOURCE_MODEL_INPUT_LIMITS = project_model_limits(
        _section
    )
    del _section
except Exception:  # pragma: no cover - resource ships + is guarded by a test
    logger.exception(
        "Failed to load bundled Claude model-limits resource; "
        "max_tokens/max_input_tokens ceilings will resolve dynamically "
        "or not at all"
    )
    RESOURCE_MODEL_LIMITS = {}
    RESOURCE_MODEL_INPUT_LIMITS = {}


def match_ceiling(
    model_lower: str, items: Iterable[tuple[str, int]]
) -> int | None:
    """Resolve a ``max_tokens`` ceiling for *model_lower* against ``items``.

    A single family-matching rule shared by both ceiling sources — the live
    dynamic cache (keys are full dated ids from the Models API) and the bundled
    resource (keys are short family aliases) — so the two can never disagree on
    how a given model id resolves. In precedence order:

    1. **Exact** id match.
    2. **Family-alias** (resource-style): the longest key that is a substring
       of the request — a short family key (``claude-sonnet-5``) covers a longer
       dated request (``claude-sonnet-5-20260514``).
    3. **Bare-alias** (dynamic-style): the longest key of which the request is a
       substring — a bare request (``claude-sonnet-5``) resolves against a
       longer dated cache key (``claude-sonnet-5-20260514``) fetched from the
       API.

    Exact wins over any substring (it returns immediately); among substring
    matches the longest overlap wins, so a longer, more-specific key is never
    shadowed by a shorter prefix. ``None`` when nothing matches → permissive
    (no clamp). For two distinct strings only one substring direction can hold,
    so cases 2 and 3 never both fire for one key.

    Args:
        model_lower: The lowercased requested model id.
        items: ``(key, ceiling)`` pairs to match against — either
            ``RESOURCE_MODEL_LIMITS.items()`` or a view over the dynamic cache.

    Returns:
        The resolved ceiling, or ``None`` when no family matches.
    """
    best: int | None = None
    best_len = -1
    for key, ceiling in items:
        if key == model_lower:
            return ceiling
        if key in model_lower:
            if len(key) > best_len:
                best, best_len = ceiling, len(key)
        elif model_lower in key and len(model_lower) > best_len:
            best, best_len = ceiling, len(model_lower)
    return best


def resource_ceiling(model_lower: str) -> int | None:
    """Resolve *model_lower* against the bundled fallback resource, or ``None``.

    Thin wrapper over :func:`match_ceiling` for the resource map. ``None`` when
    nothing matches → permissive (no clamp).
    """
    return match_ceiling(model_lower, RESOURCE_MODEL_LIMITS.items())


def resource_input_ceiling(model_lower: str) -> int | None:
    """Resolve *model_lower* against the bundled input-ceiling resource.

    Input-ceiling sibling of :func:`resource_ceiling` over
    :data:`RESOURCE_MODEL_INPUT_LIMITS`, using the *same* family-matching rule
    (:func:`match_ceiling`). ``None`` when nothing matches → the input ceiling is
    simply unknown (the consumer's proactive budget falls back to a configured
    absolute or is disabled).
    """
    return match_ceiling(model_lower, RESOURCE_MODEL_INPUT_LIMITS.items())


def claude_rejects_temperature(model_lower: str) -> bool:
    """Whether *model_lower* is a Claude family that rejects ``temperature``.

    ``True`` for the Claude 5 generation (which does not support the
    ``temperature`` parameter), ``False`` otherwise. Endpoint-agnostic — holds
    for the native Anthropic API and Bedrock Converse alike.
    """
    return any(m in model_lower for m in CLAUDE_5_TEMPERATURE_REJECTORS)


# ---------------------------------------------------------------------------
# Claude family model-metadata sources (shared by the Anthropic and Bedrock
# resolvers). A Claude model's capabilities, output/context ceilings, and
# ``temperature`` rejection are properties of the *model family*, not the
# endpoint — identical whether served by the native Anthropic Messages API or
# Amazon Bedrock Converse — so the two non-live profile sources live here and
# are composed by both providers' resolvers. Pricing and availability, by
# contrast, ARE endpoint properties (AWS bills Bedrock separately; the account
# catalog is an AWS question), so those two facets are provider-owned and NOT
# sourced here.
# ---------------------------------------------------------------------------

#: Claude family-name substrings that carry the modern (Claude 3+) capability
#: set — vision, function calling, JSON mode. Matched as lowercased substrings
#: of the model id. The Claude 5 generation adds names (``fable`` / ``mythos``)
#: that carry no opus/sonnet/haiku marker, so they are listed explicitly or they
#: would be mis-detected as lacking these. Consumed by
#: :func:`claude_heuristic_profile`.
MODERN_CAPABILITY_FAMILIES: tuple[str, ...] = (
    "claude-3", "claude-3.5", "claude-4", "claude-5",
    "claude-sonnet", "claude-opus", "claude-haiku",
    "claude-fable", "claude-mythos",
)


def claude_resource_profile(model: str) -> ModelProfile:
    """Bundled-resource source: the Claude ceiling facets (output + context).

    Resolves the two ceiling facets against the bundled Claude resource
    (:func:`resource_ceiling` / :func:`resource_input_ceiling`), using the
    family-alias matching rule. Contributes only the ceiling facets (``None``
    elsewhere), so it fills a ceiling only when a higher-precedence source (a
    live Models-API cache on the native Anthropic endpoint, or a config
    override) has no value for it — per facet. Endpoint-agnostic: a Claude
    model's ceilings are a property of the model, so both the Anthropic and
    Bedrock (Claude-on-Bedrock) resolvers compose this source. Bedrock has no
    live Models API, so on Bedrock this resource is the sole ceiling source.
    """
    key = model.lower()
    return ModelProfile(
        max_output_tokens=resource_ceiling(key),
        context_window=resource_input_ceiling(key),
    )


def claude_heuristic_profile(model: str) -> ModelProfile:
    """Heuristic source: Claude capabilities + family ``rejected_params`` by name.

    The last-resort family-substring rules for the two non-ceiling Claude facets:

    - ``capabilities`` — the base set (text/chat/streaming/code) always, plus the
      modern set (vision/function-calling/JSON) for a
      :data:`MODERN_CAPABILITY_FAMILIES` match.
    - ``rejected_params`` — ``{"temperature"}`` for the Claude 5 family (which
      rejects it with a 400), else an authoritative empty set.

    Contributes only these two facets (``None`` ceilings). This is the
    *unconditional* form the native Anthropic resolver uses — it returns the
    base capability set for *any* id (an unknown model is a Claude family on the
    Anthropic endpoint, so a permissive base fallback is correct there). The
    Bedrock resolver, which serves non-Claude families too, composes the
    Claude-*only* variant (:func:`claude_only_heuristic_profile`) so this
    heuristic never clobbers a non-Claude model's capabilities.
    """
    model_lower = model.lower()
    capabilities = {
        ModelCapability.TEXT_GENERATION,
        ModelCapability.CHAT,
        ModelCapability.STREAMING,
        ModelCapability.CODE,
    }
    if any(m in model_lower for m in MODERN_CAPABILITY_FAMILIES):
        capabilities |= {
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
        }
    rejected = (
        frozenset({"temperature"})
        if claude_rejects_temperature(model_lower)
        else frozenset()
    )
    return ModelProfile(
        capabilities=frozenset(capabilities),
        rejected_params=rejected,
    )


def claude_only_heuristic_profile(model: str) -> ModelProfile:
    """Claude-only variant of :func:`claude_heuristic_profile`.

    Abstains (returns an all-``None`` partial) for a non-Claude id, so it can sit
    *above* a provider's own capability heuristic in a multi-family resolver
    without clobbering a non-Claude model's capabilities. Used by the Bedrock
    resolver (which serves non-Claude families); the native Anthropic resolver
    uses the unconditional :func:`claude_heuristic_profile` (an unknown id there
    is a Claude family, so the permissive base fallback is correct).
    """
    if "claude" not in model.lower():
        return ModelProfile()
    return claude_heuristic_profile(model)


#: The stateless lower-precedence Claude family sources as module singletons
#: (they read only the bundled resource / apply a pure family-substring rule — no
#: per-instance state). The ceiling resource + the unconditional heuristic are
#: composed by the native Anthropic resolver
#: (:meth:`~dataknobs_llm.llm.providers.anthropic.AnthropicProvider._profile_resolver`);
#: the ceiling resource + the *Claude-only* heuristic are composed by the Bedrock
#: resolver
#: (:meth:`~dataknobs_llm.llm.providers.bedrock.BedrockProvider._profile_resolver`).
#: Living here — rather than in either provider — is what keeps the two from
#: drifting on the Claude ceilings / capabilities they share.
CLAUDE_RESOURCE_PROFILE_SOURCE = CallableModelMetadataSource(
    "bundled_resource", claude_resource_profile
)
CLAUDE_HEURISTIC_PROFILE_SOURCE = CallableModelMetadataSource(
    "heuristic", claude_heuristic_profile
)
CLAUDE_ONLY_HEURISTIC_PROFILE_SOURCE = CallableModelMetadataSource(
    "claude_heuristic", claude_only_heuristic_profile
)

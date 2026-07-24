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
  provider layers a *live Models-API* cache on top of this fallback (that
  dynamic sourcing is Anthropic-endpoint-specific and stays in
  ``anthropic.py``); the resource resolution itself is shared because a Claude
  model's output ceiling is a property of the *model*, not the endpoint.
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
from collections.abc import Iterable

from dataknobs_common.config_loading import load_yaml_or_json

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
)


def load_model_limits_resource() -> dict[str, int]:
    """Load the bundled Claude ``max_tokens`` fallback resource.

    Read once at import (below) into :data:`RESOURCE_MODEL_LIMITS`. The resource
    is fallback-only: the primary source is the live Models API ``max_tokens``
    on the native Anthropic endpoint (see
    :meth:`~dataknobs_llm.llm.providers.anthropic.AnthropicProvider._refresh_model_limits`).
    """
    ref = (
        importlib.resources.files("dataknobs_llm.llm.providers")
        / "data"
        / "anthropic_model_limits.yaml"
    )
    with importlib.resources.as_file(ref) as path:
        data = load_yaml_or_json(path, require_dict=True)
    models = data.get("models") or {}
    return {str(k).lower(): int(v) for k, v in models.items()}


try:
    #: Bundled fallback map ``{lowercased-model-id: max_tokens}``, read once at
    #: import. Consulted only when the dynamic Models-API path has produced no
    #: value for a model (see
    #: :func:`~dataknobs_llm.llm.providers.anthropic._resolve_ceiling`).
    #: Degrades to ``{}`` if the resource is unreadable so a data-file issue
    #: never breaks import — the packaging regression is caught instead by an
    #: ``importlib.resources`` test.
    RESOURCE_MODEL_LIMITS: dict[str, int] = load_model_limits_resource()
except Exception:  # pragma: no cover - resource ships + is guarded by a test
    logger.exception(
        "Failed to load bundled Claude model-limits resource; "
        "max_tokens ceilings will resolve dynamically or not at all"
    )
    RESOURCE_MODEL_LIMITS = {}


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


def claude_rejects_temperature(model_lower: str) -> bool:
    """Whether *model_lower* is a Claude family that rejects ``temperature``.

    ``True`` for the Claude 5 generation (which does not support the
    ``temperature`` parameter), ``False`` otherwise. Endpoint-agnostic — holds
    for the native Anthropic API and Bedrock Converse alike.
    """
    return any(m in model_lower for m in CLAUDE_5_TEMPERATURE_REJECTORS)

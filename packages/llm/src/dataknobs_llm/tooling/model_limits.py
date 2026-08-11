"""Reconcile a provider's bundled model resource against its live catalog.

A bundled fallback resource is only as useful as it is current, so this
maintainer tool keeps it honest per provider (``--provider``, default
``anthropic``). The drift *semantic* differs per provider because the two live
catalogs serve different facts:

- **anthropic** — the live Models API serves per-model ``max_tokens`` /
  ``max_input_tokens`` **ceilings**, so ``--check`` diffs the output ceiling and
  ``--update`` rewrites ``anthropic_model_limits.yaml`` from live values. Gated
  on ``ANTHROPIC_API_KEY`` (unset → clean no-op, exit 0).
- **bedrock** — ``ListFoundationModels`` serves **availability + modalities**
  but NOT token ceilings, so ``--check`` diffs the available-model set and the
  vision/streaming modalities against ``bedrock_models.yaml`` (a model AWS added,
  or one that gained vision). ``--update`` is unsupported (ceilings/pricing are
  not live-sourced — maintain them by hand). Gated on control-plane
  availability (no credentials / ``AccessDenied`` → clean no-op, exit 0), so an
  inference-only role never fails the check.

Usage::

    uv run python -m dataknobs_llm.tooling.model_limits --check
    uv run python -m dataknobs_llm.tooling.model_limits --update
    uv run python -m dataknobs_llm.tooling.model_limits --provider bedrock --check

Or via the wrapper: ``bin/update-model-limits.sh --check``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.resources
import inspect
import os
from dataclasses import dataclass
from datetime import datetime as _datetime
from datetime import UTC
from pathlib import Path
from typing import Any, Awaitable, Callable

from dataknobs_common.config_loading import load_yaml_or_json

from dataknobs_llm.llm.base import ModelCapability
from dataknobs_llm.llm.model_profile import match_family_key

# Reuse the provider's extraction + the shared tolerant projection so the
# checker compares exactly what the runtime caches (no parallel, drift-prone
# logic).
from dataknobs_llm.llm.providers._claude_shared import project_model_limits
from dataknobs_llm.llm.providers.anthropic import (
    _extract_max_input_tokens,
    _extract_max_tokens,
)

#: One model's live/resource ceilings: output ``max_tokens`` + input
#: ``max_input_tokens`` (either may be ``None`` when the column is absent).
_ModelLimits = dict[str, int | None]

#: The bundled resource, regenerated verbatim by ``--update``. The ``models:``
#: block is rewritten nested + sorted; the header carries provenance.
_HEADER_TEMPLATE = """\
# Anthropic model token ceilings, synchronous Messages API.
# FALLBACK ONLY -- the primary source is the live Models API `max_tokens`
#   (output) and `max_input_tokens` (input/context window) fields.
# Source: https://platform.claude.com/docs/en/docs/about-claude/models/overview
# Last verified: {verified_date}
# NOTE: the Batches API supports higher output (up to 300k) via the
#   output-300k-2026-03-24 beta header; not represented here (sync-API values).
#   Some models support a 1M-token input window via a beta header; the values
#   here are the documented default context windows (exactness comes from the
#   dynamic Models-API path).
# Shape: each model maps to {{max_tokens (output), max_input_tokens (context)}}.
#   The loader is tolerant of the legacy flat `model: <int>` (output-only) form.
# Maintained by:
#   uv run python -m dataknobs_llm.tooling.model_limits --update
models:
"""


def _packaged_resource_path() -> Path:
    """Filesystem path of the bundled resource (real path in a source install)."""
    ref = (
        importlib.resources.files("dataknobs_llm.llm.providers")
        / "data"
        / "anthropic_model_limits.yaml"
    )
    return Path(str(ref))


def load_resource_limits(path: Path) -> dict[str, _ModelLimits]:
    """Load ``{lowercased-model-id: {max_tokens, max_input_tokens}}`` from *path*.

    Routes through the shared :func:`~._claude_shared.project_model_limits` so
    the tooling reads the nested (and legacy flat, output-only) resource shapes
    exactly as the runtime does — no parallel, drift-prone parser.
    """
    data = load_yaml_or_json(path, require_dict=True)
    section = data.get("models") or {}
    output, input_ceilings = project_model_limits(section)
    return {
        model_id: {
            "max_tokens": output.get(model_id),
            "max_input_tokens": input_ceilings.get(model_id),
        }
        for model_id in sorted(set(output) | set(input_ceilings))
    }


async def fetch_live_limits(client: Any) -> dict[str, _ModelLimits]:
    """Fetch ``{model-id: {max_tokens, max_input_tokens}}`` from the live API.

    ``client.models.list()`` returns an auto-paging ``AsyncPaginator``; a model
    carrying **neither** ceiling is skipped (it resolves to the resource fallback
    at runtime). Reuses the provider's extraction so the tool captures exactly
    what the runtime caches.
    """
    limits: dict[str, _ModelLimits] = {}
    async for model_obj in client.models.list(limit=1000):
        max_tokens = _extract_max_tokens(model_obj)
        max_input_tokens = _extract_max_input_tokens(model_obj)
        model_id = getattr(model_obj, "id", None)
        if model_id and (max_tokens is not None or max_input_tokens is not None):
            limits[str(model_id).lower()] = {
                "max_tokens": max_tokens,
                "max_input_tokens": max_input_tokens,
            }
    return limits


def _output_map(limits: dict[str, _ModelLimits]) -> dict[str, int]:
    """Project a nested limits map to its output-ceiling-only ``{id: int}`` view."""
    return {
        model_id: entry["max_tokens"]
        for model_id, entry in limits.items()
        if entry.get("max_tokens") is not None
    }  # type: ignore[misc]  # filtered to non-None above


def diff_limits(
    live: dict[str, int], resource: dict[str, int]
) -> list[tuple[str, int | None, int | None]]:
    """Return ``(model_id, live, resource)`` for every divergent model, sorted.

    Compares the **output** (``max_tokens``) ceilings — the drift alarm's
    historical scope (the input/context window is stable and live-sourced at
    runtime). Callers project the nested maps via :func:`_output_map` first.
    """
    drift: list[tuple[str, int | None, int | None]] = []
    for model_id in sorted(set(live) | set(resource)):
        live_val = live.get(model_id)
        resource_val = resource.get(model_id)
        if live_val != resource_val:
            drift.append((model_id, live_val, resource_val))
    return drift


def render_resource(limits: dict[str, _ModelLimits], *, verified_date: str) -> str:
    """Render the full resource file text from ``limits`` (nested, sorted).

    Each model renders its present ceilings (``max_tokens`` / ``max_input_tokens``);
    a ``None`` column is omitted rather than written as a literal ``None``.
    """
    parts = [_HEADER_TEMPLATE.format(verified_date=verified_date)]
    for model_id in sorted(limits):
        entry = limits[model_id]
        parts.append(f"  {model_id}:\n")
        if entry.get("max_tokens") is not None:
            parts.append(f"    max_tokens: {entry['max_tokens']}\n")
        if entry.get("max_input_tokens") is not None:
            parts.append(f"    max_input_tokens: {entry['max_input_tokens']}\n")
    return "".join(parts)


def _build_client_from_env() -> Any:
    """Build a live ``AsyncAnthropic`` client from ``ANTHROPIC_API_KEY``."""
    import anthropic

    return anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _anthropic_drift_lines(
    live: dict[str, _ModelLimits], resource: dict[str, _ModelLimits]
) -> list[str]:
    """Printable output-ceiling drift lines (the Anthropic drift semantic)."""
    rows = diff_limits(_output_map(live), _output_map(resource))
    return [f"{mid}: live={lv} resource={rv}" for mid, lv, rv in rows]


# ---------------------------------------------------------------------------
# Bedrock binding — ``ListFoundationModels`` serves availability + modalities,
# NOT token ceilings, so the drift check compares the available-model set and
# the vision/streaming modalities (not ceilings). Live ids are full dated ids;
# the resource is keyed by family alias, so each live id is resolved against the
# resource keys via ``match_family_key`` (the same rule the runtime uses).
# ---------------------------------------------------------------------------

#: One model's modality facts (``vision`` / ``streaming``) from catalog or resource.
_ModelModalities = dict[str, bool]


def _bedrock_resource_path() -> Path:
    """Filesystem path of the bundled Bedrock resource."""
    ref = importlib.resources.files("dataknobs_llm.llm.providers") / "data" / "bedrock_models.yaml"
    return Path(str(ref))


async def fetch_bedrock_facts(client: Any) -> dict[str, _ModelModalities]:
    """Fetch ``{model-id: {vision, streaming}}`` from ``ListFoundationModels``.

    The control-plane API returns the account's catalog in one call. It serves
    availability + modalities but **not** token ceilings, so this captures
    ``vision`` (``IMAGE`` in ``inputModalities``) and ``streaming``
    (``responseStreamingSupported``) — the facts the resource can drift on. The
    call is sync on a boto3 client and awaitable on aioboto3 / the test stub;
    both are handled.
    """
    result = client.list_foundation_models()
    if inspect.isawaitable(result):
        result = await result
    facts: dict[str, _ModelModalities] = {}
    for summary in result.get("modelSummaries", []):
        model_id = summary.get("modelId")
        if not model_id:
            continue
        facts[str(model_id).lower()] = {
            "vision": "IMAGE" in (summary.get("inputModalities") or []),
            "streaming": bool(summary.get("responseStreamingSupported", False)),
        }
    return facts


def load_bedrock_resource_facts(path: Path) -> dict[str, _ModelModalities]:
    """Load ``{family-key: {vision, streaming}}`` from ``bedrock_models.yaml``.

    Reads each capability-bearing entry into the same modality facts as the live
    catalog, so the two are directly comparable. A Claude-on-Bedrock entry (no
    ``capabilities`` — pricing/availability only) is skipped: its modalities come
    from the shared Claude capability source, not this resource.
    """
    data = load_yaml_or_json(path, require_dict=True)
    section = data.get("models") or {}
    facts: dict[str, _ModelModalities] = {}
    for key, entry in section.items():
        if not isinstance(entry, dict):
            continue
        caps = entry.get("capabilities")
        if not caps:
            continue
        cap_set = {str(c) for c in caps}
        facts[str(key).lower()] = {
            "vision": ModelCapability.VISION.value in cap_set,
            "streaming": ModelCapability.STREAMING.value in cap_set,
        }
    return facts


def diff_bedrock_facts(
    live: dict[str, _ModelModalities],
    resource: dict[str, _ModelModalities],
) -> list[str]:
    """Drift line per live model uncovered by / modality-mismatched with the resource.

    For each live catalog model id, resolve it to a resource family key
    (:func:`~dataknobs_llm.llm.model_profile.match_family_key`, the runtime's
    substring rule). Flags an **uncovered** live model (AWS added a model the
    resource does not describe) and a **modality drift** (a resolved model whose
    live ``vision`` / ``streaming`` disagrees with the resource — a model gained a
    modality). Claude-on-Bedrock ids are skipped — their capabilities come from
    the shared Claude source, so a modality diff against the modality-less Claude
    entries here would be vacuous.
    """
    drift: list[str] = []
    keys = list(resource.keys())
    for model_id in sorted(live):
        if model_id.startswith("anthropic.") or ".anthropic." in model_id:
            continue
        matched = match_family_key(model_id, keys)
        if matched is None:
            drift.append(f"{model_id}: in catalog, not covered by resource")
            continue
        for facet in ("vision", "streaming"):
            if live[model_id][facet] != resource[matched][facet]:
                drift.append(
                    f"{model_id} [{matched}]: {facet} "
                    f"live={live[model_id][facet]} "
                    f"resource={resource[matched][facet]}"
                )
    return drift


def _build_bedrock_client_from_env() -> Any:
    """Build a sync boto3 ``bedrock`` control-plane client from the AWS chain."""
    import boto3

    return boto3.client("bedrock")


def _bedrock_skip_errors() -> tuple[type[BaseException], ...]:
    """Fetch-time errors treated as a clean no-op (no creds / ``AccessDenied``).

    Bedrock has no API-key env var to gate on up front, so the gate is deferred:
    a missing-credentials / permission error at ``ListFoundationModels`` time is
    a clean skip (exit 0), mirroring the keyless Anthropic no-op — an
    inference-only role never fails the check.
    """
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:  # pragma: no cover - botocore ships with boto3
        return ()
    return (BotoCoreError, ClientError)


@dataclass(frozen=True)
class _ProviderBinding:
    """Per-provider reconcile hooks — the ``--provider`` dispatch surface.

    Each provider's live catalog serves different facts, so the drift *semantic*
    differs: anthropic diffs output ceilings; bedrock diffs the available-model
    set + vision/streaming modalities. A binding bundles the
    resource path, the credential gate, the client factory, the live-fact
    fetcher, the resource loader, the drift renderer, and (when the live catalog
    carries enough to regenerate the resource) the ``--update`` renderer.
    """

    name: str
    supports_update: bool
    resource_path: Callable[[], Path]
    gate: Callable[[], str | None]
    build_client: Callable[[], Any]
    skip_errors: Callable[[], tuple[type[BaseException], ...]]
    fetch: Callable[[Any], Awaitable[dict[str, Any]]]
    load_resource: Callable[[Path], dict[str, Any]]
    drift_lines: Callable[[dict[str, Any], dict[str, Any]], list[str]]
    drift_header: str
    ok_noun: str
    render: Callable[[dict[str, Any], str], str] | None


_ANTHROPIC_BINDING = _ProviderBinding(
    name="anthropic",
    supports_update=True,
    resource_path=_packaged_resource_path,
    gate=lambda: (
        None
        if os.environ.get("ANTHROPIC_API_KEY")
        else "model_limits: skipped -- ANTHROPIC_API_KEY not set (no-op)."
    ),
    build_client=_build_client_from_env,
    skip_errors=lambda: (),
    fetch=fetch_live_limits,
    load_resource=load_resource_limits,
    drift_lines=_anthropic_drift_lines,
    drift_header="DRIFT vs the live Models API (model: live vs resource)",
    ok_noun="models match the live API",
    render=lambda limits, date: render_resource(limits, verified_date=date),
)

_BEDROCK_BINDING = _ProviderBinding(
    name="bedrock",
    supports_update=False,
    resource_path=_bedrock_resource_path,
    gate=lambda: None,  # deferred to fetch (no API-key env var); see skip_errors
    build_client=_build_bedrock_client_from_env,
    skip_errors=_bedrock_skip_errors,
    fetch=fetch_bedrock_facts,
    load_resource=load_bedrock_resource_facts,
    drift_lines=diff_bedrock_facts,
    drift_header="DRIFT vs the live catalog (ListFoundationModels)",
    ok_noun="model families match the catalog modalities",
    render=None,
)

_PROVIDER_BINDINGS: dict[str, _ProviderBinding] = {
    "anthropic": _ANTHROPIC_BINDING,
    "bedrock": _BEDROCK_BINDING,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dataknobs_llm.tooling.model_limits",
        description="Reconcile a provider's bundled model resource against its live catalog.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_BINDINGS),
        default="anthropic",
        help="Which provider's resource to reconcile (default: anthropic).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Diff the resource against live values; exit non-zero on drift (default).",
    )
    group.add_argument(
        "--update",
        action="store_true",
        help="Rewrite the resource from live values.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    client: Any = None,
    resource_path: Path | str | None = None,
    verified_date: str | None = None,
) -> int:
    """CLI entry point. Returns a process exit code.

    Dispatches on ``--provider`` (default ``anthropic``) to a per-provider
    binding (:class:`_ProviderBinding`). ``client`` / ``resource_path`` /
    ``verified_date`` are injection seams for tests (drive with a stand-in
    client, write to a temp path, pin the date) — unset, the tool builds a live
    client from the environment and targets the provider's bundled resource.
    """
    args = _parse_args(argv)
    binding = _PROVIDER_BINDINGS[args.provider]
    path = Path(resource_path) if resource_path is not None else binding.resource_path()

    if client is None:
        skip = binding.gate()
        if skip is not None:
            print(skip)
            return 0
        try:
            client = binding.build_client()
        except ImportError as exc:
            print(f"model_limits: skipped -- {exc}")
            return 0

    if args.update and not binding.supports_update:
        print(
            f"model_limits: --update is not supported for {binding.name} "
            "(ceilings/pricing are not live-sourced); maintain the resource by "
            "hand and use --check."
        )
        return 2

    try:
        live = asyncio.run(binding.fetch(client))
    except binding.skip_errors() as exc:  # credentials / permission → clean no-op
        print(f"model_limits: skipped -- {binding.name} catalog unavailable ({exc}).")
        return 0

    if args.update:
        assert binding.render is not None  # supports_update gate above
        stamp = verified_date or _datetime.now(UTC).date().isoformat()
        path.write_text(binding.render(live, stamp), encoding="utf-8")
        print(f"model_limits: updated {path} ({len(live)} models).")
        return 0

    resource = binding.load_resource(path)
    drift = binding.drift_lines(live, resource)
    if not drift:
        print(f"model_limits: OK -- {len(resource)} {binding.ok_noun}.")
        return 0
    print(f"model_limits: {binding.drift_header}:")
    for line in drift:
        print(f"  {line}")
    return 1


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())

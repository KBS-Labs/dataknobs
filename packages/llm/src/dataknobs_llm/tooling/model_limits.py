"""Reconcile the bundled Anthropic ``max_tokens`` fallback against the live API.

The provider resolves per-model output-token ceilings dynamically from the live
Anthropic Models API and falls back to a bundled resource
(``llm/providers/data/anthropic_model_limits.yaml``) only when the API is
unavailable. That fallback is only as useful as it is current, so this
maintainer tool keeps it honest:

- ``--check`` fetches live ``max_tokens`` for every model, diffs against the
  resource, and exits non-zero on drift (a keyed CI / nightly signal).
- ``--update`` rewrites the resource from live values, stamping today's date.

**Key-gated:** with ``ANTHROPIC_API_KEY`` unset the tool is a clean no-op
(exit 0), so a keyless CI invocation never fails.

Usage::

    uv run python -m dataknobs_llm.tooling.model_limits --check
    uv run python -m dataknobs_llm.tooling.model_limits --update

Or via the wrapper: ``bin/update-model-limits.sh --check``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.resources
import os
from datetime import datetime as _datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from dataknobs_common.config_loading import load_yaml_or_json

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


def render_resource(
    limits: dict[str, _ModelLimits], *, verified_date: str
) -> str:
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dataknobs_llm.tooling.model_limits",
        description="Reconcile the bundled Anthropic max_tokens fallback "
        "against the live Models API.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Diff the resource against live values; exit non-zero on drift "
        "(default).",
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

    ``client`` / ``resource_path`` / ``verified_date`` are injection seams for
    tests (drive with a stand-in client, write to a temp path, pin the date) —
    unset, the tool builds a live client from the environment and targets the
    bundled resource.
    """
    args = _parse_args(argv)
    path = Path(resource_path) if resource_path is not None else _packaged_resource_path()

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("model_limits: skipped -- ANTHROPIC_API_KEY not set (no-op).")
            return 0
        try:
            client = _build_client_from_env()
        except ImportError:
            print("model_limits: skipped -- anthropic package not installed.")
            return 0

    live = asyncio.run(fetch_live_limits(client))

    if args.update:
        stamp = verified_date or _datetime.now(timezone.utc).date().isoformat()
        path.write_text(render_resource(live, verified_date=stamp), encoding="utf-8")
        print(f"model_limits: updated {path} ({len(live)} models).")
        return 0

    # Default action is --check. Drift is compared on the output ceiling only
    # (the input/context window is stable and live-sourced at runtime).
    resource = load_resource_limits(path)
    drift = diff_limits(_output_map(live), _output_map(resource))
    if not drift:
        print(f"model_limits: OK -- {len(resource)} models match the live API.")
        return 0
    print("model_limits: DRIFT vs the live Models API (model: live vs resource):")
    for model_id, live_val, resource_val in drift:
        print(f"  {model_id}: live={live_val} resource={resource_val}")
    return 1


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())

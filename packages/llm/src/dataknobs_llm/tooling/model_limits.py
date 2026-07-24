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

# Reuse the provider's extraction so the checker compares exactly what the
# runtime caches (no parallel, drift-prone extraction logic).
from dataknobs_llm.llm.providers.anthropic import _extract_max_tokens

#: The bundled resource, regenerated verbatim by ``--update``. The ``models:``
#: block is rewritten flat + sorted; the header carries provenance.
_HEADER_TEMPLATE = """\
# Anthropic model output-token ceilings (max_tokens), synchronous Messages API.
# FALLBACK ONLY -- the primary source is the live Models API `max_tokens` field.
# Source: https://platform.claude.com/docs/en/docs/about-claude/models/overview
# Last verified: {verified_date}
# NOTE: the Batches API supports higher output (up to 300k) via the
#   output-300k-2026-03-24 beta header; not represented here (sync-API values).
# Values are the documented (rounded) maxima; exactness comes from the dynamic
# Models-API path. Maintained by:
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


def load_resource_limits(path: Path) -> dict[str, int]:
    """Load ``{lowercased-model-id: max_tokens}`` from the resource file."""
    data = load_yaml_or_json(path, require_dict=True)
    models = data.get("models") or {}
    return {str(k).lower(): int(v) for k, v in models.items()}


async def fetch_live_limits(client: Any) -> dict[str, int]:
    """Fetch ``{lowercased-model-id: max_tokens}`` from the live Models API.

    ``client.models.list()`` returns an auto-paging ``AsyncPaginator``; models
    with no ``max_tokens`` are skipped (they resolve to the resource fallback at
    runtime).
    """
    limits: dict[str, int] = {}
    async for model_obj in client.models.list(limit=1000):
        max_tokens = _extract_max_tokens(model_obj)
        model_id = getattr(model_obj, "id", None)
        if max_tokens is not None and model_id:
            limits[str(model_id).lower()] = max_tokens
    return limits


def diff_limits(
    live: dict[str, int], resource: dict[str, int]
) -> list[tuple[str, int | None, int | None]]:
    """Return ``(model_id, live, resource)`` for every divergent model, sorted."""
    drift: list[tuple[str, int | None, int | None]] = []
    for model_id in sorted(set(live) | set(resource)):
        live_val = live.get(model_id)
        resource_val = resource.get(model_id)
        if live_val != resource_val:
            drift.append((model_id, live_val, resource_val))
    return drift


def render_resource(limits: dict[str, int], *, verified_date: str) -> str:
    """Render the full resource file text from ``limits`` (flat, sorted)."""
    parts = [_HEADER_TEMPLATE.format(verified_date=verified_date)]
    parts.extend(
        f"  {model_id}: {limits[model_id]}\n" for model_id in sorted(limits)
    )
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

    # Default action is --check.
    resource = load_resource_limits(path)
    drift = diff_limits(live, resource)
    if not drift:
        print(f"model_limits: OK -- {len(resource)} models match the live API.")
        return 0
    print("model_limits: DRIFT vs the live Models API (model: live vs resource):")
    for model_id, live_val, resource_val in drift:
        print(f"  {model_id}: live={live_val} resource={resource_val}")
    return 1


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())

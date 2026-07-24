#!/usr/bin/env bash
# Reconcile the bundled Anthropic max_tokens fallback resource against the live
# Models API. Thin wrapper over the maintainer tool; pass --check or --update.
#
#   bin/update-model-limits.sh --check     # drift check (exit non-zero on drift)
#   bin/update-model-limits.sh --update    # rewrite the resource from live values
#
# Key-gated: a clean no-op when ANTHROPIC_API_KEY is unset.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m dataknobs_llm.tooling.model_limits "$@"

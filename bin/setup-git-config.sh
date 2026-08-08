#!/bin/bash
set -e

# Configure the repository-local git settings that .gitattributes depends on.
#
# Git attributes name a merge driver; they do not define one. `.gitattributes`
# marks .quality-artifacts/** as merge=ours, but with `merge.ours.driver` unset
# git falls back to the default text merge and produces conflict markers exactly
# as if the attribute were absent. The failure is silent — there is no warning,
# and the file looks like it is protecting something.
#
# `bin/dk` runs this on every invocation (it is a no-op once set), so anyone
# using the standard tooling is configured without thinking about it. Run it
# directly if you drive git some other way.
#
# Idempotent and repository-local: writes to .git/config only, never --global.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The `true` driver exits 0 without touching the working-tree file, which git
# has already populated with our side. That is the whole behavior: keep one
# version intact rather than interleaving two.
#
# Safe here because quality artifacts are regenerated, not merged. A branch that
# takes main must re-run `bin/dk pr` regardless — the gate rehashes the checkout
# and rejects an artifact that does not describe it — so the surviving side is
# overwritten before it is ever validated. Keeping either one loses nothing.
ensure_config() {
    local key="$1" value="$2"
    local current
    current=$(git -C "$PROJECT_ROOT" config --local --get "$key" 2>/dev/null || true)
    if [ "$current" = "$value" ]; then
        return 1
    fi
    git -C "$PROJECT_ROOT" config --local "$key" "$value"
    return 0
}

CHANGED=0
ensure_config "merge.ours.driver" "true" && CHANGED=1

if [ "${1:-}" = "--quiet" ]; then
    exit 0
fi

if [ "$CHANGED" -eq 1 ]; then
    echo "Configured merge.ours.driver — .quality-artifacts/ will no longer conflict."
else
    echo "Git config already up to date."
fi

#!/usr/bin/env bash
# bin/audit-floor.sh — Run a floor-level CVE audit on the workspace.
#
# Generates a lowest-direct uv lockfile (every direct dep pinned to its
# declared lower bound, transitive deps resolve normally) and scans it
# with osv-scanner. CI runs the same audit weekly via
# .github/workflows/dependency-update.yml; this lets developers run it
# locally before pushing dependency-floor changes.
#
# `uv lock` always writes to uv.lock; we back up the upgraded resolve,
# regenerate as lowest-direct, copy aside for the scan, and restore
# the original uv.lock state.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not on PATH." >&2
  exit 2
fi

if ! command -v osv-scanner >/dev/null 2>&1; then
  echo "ERROR: osv-scanner not on PATH." >&2
  echo "Install: see .github/workflows/dependency-update.yml (currently v2.3.3)" >&2
  echo "macOS:   brew install osv-scanner" >&2
  exit 2
fi

cleanup() {
  # Always restore uv.lock if we backed it up, even on error.
  if [ -f uv.lock.upgraded ]; then
    mv uv.lock.upgraded uv.lock
  fi
  rm -f uv-lowest.lock
}
trap cleanup EXIT

echo "==> Generating lowest-direct resolve..."
cp uv.lock uv.lock.upgraded
uv lock --resolution lowest-direct >/dev/null
cp uv.lock uv-lowest.lock
mv uv.lock.upgraded uv.lock

echo "==> Running osv-scanner against floor resolve..."
# osv-scanner needs the type:path syntax for non-standard filenames.
exit_code=0
osv-scanner scan --lockfile uv.lock:uv-lowest.lock --format table || exit_code=$?

case "$exit_code" in
  0)
    echo "==> No floor-level vulnerabilities found."
    ;;
  1)
    echo "==> Floor-level vulnerabilities detected." >&2
    echo "    Bump the affected pyproject.toml floors with inline" >&2
    echo "    CVE-rationale comments (see packages/llm/pyproject.toml:34)." >&2
    ;;
  *)
    echo "==> osv-scanner exited with code $exit_code (scanner error)." >&2
    ;;
esac

exit "$exit_code"

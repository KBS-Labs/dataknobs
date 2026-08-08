#!/bin/bash
# Lint GitHub Actions workflow files
# Checks for:
#   1. Mutable action refs (must use SHA pinning)
#   2. ShellCheck issues in run: blocks
#
# Requires: uv (for python/pyyaml) and shellcheck. Both are required, not
# optional — see the note above the probes below.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_DIR="$PROJECT_ROOT/.github/workflows"

# Colors
if [ -t 1 ] && [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' NC=''
fi

errors=0

# Report whatever has been counted so far. Called from the paths that abort
# before the summary at the bottom, so an aborted run still says how many issues
# it had found; silent when there are none, so an abort with a clean slate is
# not dressed up as a finding.
report_pending_errors() {
    if [ $errors -gt 0 ]; then
        echo ""
        echo -e "${RED}Found $errors workflow issue(s) before stopping${NC}"
    fi
}

# Check workflow directory exists
if [ ! -d "$WORKFLOW_DIR" ]; then
    echo -e "${YELLOW}No .github/workflows/ directory found — skipping${NC}"
    exit 0
fi

# Both extensions: GitHub Actions accepts .yaml as readily as .yml, so globbing
# only .yml would leave such a workflow unlinted without saying so — the same
# silent non-execution the tool probes below exist to prevent. Built by filtering
# rather than assigned directly because a glob that matches nothing expands to
# itself, which the old "does element zero exist" test only caught for .yml.
workflow_files=()
for candidate in "$WORKFLOW_DIR"/*.yml "$WORKFLOW_DIR"/*.yaml; do
    if [ -e "$candidate" ]; then
        workflow_files+=("$candidate")
    fi
done
if [ ${#workflow_files[@]} -eq 0 ]; then
    echo -e "${YELLOW}No workflow files found — skipping${NC}"
    exit 0
fi

# 1. Check for mutable action refs (uses: foo/bar@v1 instead of SHA)
echo "Checking for mutable action refs..."
for file in "${workflow_files[@]}"; do
    basename_file=$(basename "$file")
    while IFS= read -r line; do
        lineno=$(echo "$line" | cut -d: -f1)
        content=$(echo "$line" | cut -d: -f2-)
        echo -e "  ${RED}✗${NC} $basename_file:$lineno:$content"
        errors=$((errors + 1))
    done < <(grep -n 'uses:.*@v[0-9]' "$file" | grep -v '^\s*#' || true)
done

if [ $errors -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} All action refs are SHA-pinned"
fi

# 2. ShellCheck on run: blocks
#
# Both tools are required, and their absence is fatal rather than a skip. A skip
# leaves the run reporting success having analysed nothing — and since CI
# validates the committed artifacts instead of re-running the gate, the artifact
# from a machine without shellcheck is indistinguishable from one where the
# linter ran and found no issues. Nothing downstream can tell them apart.
#
# This is also what the dependency rules require of a tool we invoke as a
# subprocess: gate its presence explicitly and fail, never warn and continue.
# Check 1 has already run and may have counted errors, and these branches exit
# before reaching the summary at the bottom, so they report the running total
# themselves — otherwise a run that found mutable refs and then hit a missing
# tool would print its ✗ lines and stop without ever saying how many. The exit
# stays in the branch rather than falling through to a shared one: continuing
# past a missing tool is the defect being prevented, and the guard in
# tests/test_quality_gate_accounting.py reads each branch for it.
if ! command -v shellcheck >/dev/null 2>&1; then
    echo -e "  ${RED}✗${NC} shellcheck is required but not installed"
    echo "    macOS:  brew install shellcheck"
    echo "    Debian: apt-get install shellcheck"
    report_pending_errors
    exit 1
elif ! command -v uv >/dev/null 2>&1; then
    echo -e "  ${RED}✗${NC} uv is required but not installed"
    echo "    Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    report_pending_errors
    exit 1
else
    echo "Running shellcheck on workflow run: blocks..."
    sc_errors=0
    tmpdir=$(mktemp -d /tmp/workflow-lint-XXXXXX)
    trap 'rm -rf "$tmpdir"' EXIT

    # Extract all run: blocks from all workflow files using pyyaml
    uv run python << 'PYEOF' - "$tmpdir" "${workflow_files[@]}"
import yaml
import sys
import os

tmpdir = sys.argv[1]
files = sys.argv[2:]

def extract_run_blocks(node):
    """Recursively find all 'run' string values in a YAML structure."""
    results = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                results.append(value)
            else:
                results.extend(extract_run_blocks(value))
    elif isinstance(node, list):
        for item in node:
            results.extend(extract_run_blocks(item))
    return results

for filepath in files:
    basename = os.path.basename(filepath)
    with open(filepath) as f:
        data = yaml.safe_load(f)
    if not data:
        continue
    for idx, script in enumerate(extract_run_blocks(data)):
        stripped = script.strip()
        # Skip trivial one-liners unlikely to have shell issues
        if "\n" not in stripped and len(stripped) < 60:
            continue
        outpath = os.path.join(tmpdir, f"{basename}.{idx}.sh")
        with open(outpath, "w") as out:
            out.write("#!/bin/bash\n")
            out.write(script)
            out.write("\n")
PYEOF

    # ShellCheck each extracted block
    for block_file in "$tmpdir"/*.sh; do
        [ -f "$block_file" ] || continue
        block_name=$(basename "$block_file" .sh)
        # block_name is "workflow.yml.N" — split to get workflow name and block index
        wf_name="${block_name%.*}"
        block_idx="${block_name##*.}"
        # Exclude:
            # Exclude rules that produce false positives on GitHub Actions syntax:
            #   SC1083 - literal } from ${{ }} expressions
            #   SC2086 - word splitting (too noisy for workflow scripts)
            #   SC2193 - false positive on ${{ }} comparisons
            #   SC2296 - false positive on ${{ }} parameter expansions
            #   SC2034 - "unused" variables consumed by ${{ }} expressions
            #   SC2164 - cd without || exit (acceptable in workflow steps)
            sc_output=$(shellcheck -S warning -s bash -e SC1083,SC2086,SC2193,SC2296,SC2034,SC2164 "$block_file" 2>&1 || true)
        if [ -n "$sc_output" ]; then
            cleaned=$(echo "$sc_output" | sed "s|$block_file|$wf_name (run block #$block_idx)|g")
            while IFS= read -r sc_line; do
                if echo "$sc_line" | grep -qE "SC[0-9]+"; then
                    echo -e "  ${RED}✗${NC} $sc_line"
                    sc_errors=$((sc_errors + 1))
                fi
            done <<< "$cleaned"
        fi
    done

    if [ $sc_errors -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} No shellcheck issues found"
    else
        errors=$((errors + sc_errors))
    fi

    rm -rf "$tmpdir"
fi

# Summary
echo ""
if [ $errors -gt 0 ]; then
    echo -e "${RED}Found $errors workflow issue(s)${NC}"
    exit 1
else
    echo -e "${GREEN}All workflow checks passed${NC}"
    exit 0
fi

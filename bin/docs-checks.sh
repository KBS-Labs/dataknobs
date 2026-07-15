#!/usr/bin/env bash
# Canonical documentation-quality checks.
#
# This is the SINGLE source of truth for "what are the doc checks". Every entry
# point that runs doc checks delegates here instead of re-enumerating the list,
# so adding a check is a one-line edit in this file and every caller inherits it:
#
#   * bin/dk docs-check          -> bin/docs-checks.sh
#   * bin/run-quality-checks.sh  -> bin/docs-checks.sh --artifacts <dir>  (dk pr)
#
# (The three CI doc workflows remain thin single-command callers of the
# underlying per-check scripts.)
#
# Checks run, in order:
#   docs-build     `mkdocs build --strict`            (rendered site is clean)
#   docs-versions  `docs-update-versions.sh --check`  (versions match packages.json)
#   docs-mirror    `docs-mirror-check.py --check`     (package<->site mirrors in sync)
#
# Usage:
#   bin/docs-checks.sh [--artifacts DIR]
#
#   --artifacts DIR  Write each check's output to DIR/<name>.log and a
#                    machine-readable DIR/docs-checks-status.json mapping each
#                    check name to its exit code. Without it, output streams to
#                    the terminal (local `dk docs-check` use).
#
# Exit status: 0 if all checks pass, 1 if any fail (all checks always run).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR" || exit 1

# Colors (disabled when stdout is not a TTY)
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
else
    RED=''; GREEN=''; CYAN=''; NC=''
fi

ARTIFACTS_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --artifacts)
            ARTIFACTS_DIR="${2:-}"
            if [ -z "$ARTIFACTS_DIR" ]; then
                echo -e "${RED}Error: --artifacts requires a directory argument${NC}" >&2
                exit 2
            fi
            mkdir -p "$ARTIFACTS_DIR"
            shift 2
            ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo -e "${RED}Error: unknown argument '$1'${NC}" >&2
            exit 2
            ;;
    esac
done

STATUS_NAMES=()
STATUS_CODES=()
OVERALL=0

# run_doc_check <name> <label> <logfile> -- <command...>
run_doc_check() {
    local name="$1" label="$2" log="$3"
    shift 3
    [ "${1:-}" = "--" ] && shift

    echo -e "${CYAN}Checking ${label}...${NC}"
    local rc=0
    if [ -n "$ARTIFACTS_DIR" ]; then
        "$@" > "$ARTIFACTS_DIR/$log" 2>&1 || rc=$?
    else
        "$@" || rc=$?
    fi

    STATUS_NAMES+=("$name")
    STATUS_CODES+=("$rc")
    if [ "$rc" -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} ${label} passed"
    else
        OVERALL=1
        echo -e "  ${RED}✗${NC} ${label} FAILED"
        if [ -n "$ARTIFACTS_DIR" ]; then
            echo -e "    see ${CYAN}$ARTIFACTS_DIR/$log${NC}"
        fi
    fi
}

run_doc_check "docs-build" "documentation build (mkdocs --strict)" "docs-build.log" \
    -- env NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict

run_doc_check "docs-versions" "documentation versions" "docs-versions.log" \
    -- "$SCRIPT_DIR/docs-update-versions.sh" --check

run_doc_check "docs-mirror" "doc mirrors (package <-> site)" "docs-mirror.log" \
    -- python3 "$SCRIPT_DIR/docs-mirror-check.py" --check

# Emit machine-readable per-check status for callers (run-quality-checks.sh).
if [ -n "$ARTIFACTS_DIR" ]; then
    {
        echo "{"
        for i in "${!STATUS_NAMES[@]}"; do
            sep=","
            [ "$i" -eq $(( ${#STATUS_NAMES[@]} - 1 )) ] && sep=""
            echo "  \"${STATUS_NAMES[$i]}\": ${STATUS_CODES[$i]}${sep}"
        done
        echo "}"
    } > "$ARTIFACTS_DIR/docs-checks-status.json"
fi

echo ""
if [ "$OVERALL" -eq 0 ]; then
    echo -e "${GREEN}✓ All documentation checks passed${NC}"
else
    echo -e "${RED}✗ One or more documentation checks failed${NC}"
    echo -e "${CYAN}  Fixes: mkdocs errors -> bin/dk docs; version drift -> bin/docs-update-versions.sh;${NC}"
    echo -e "${CYAN}         mirror drift -> bin/docs-mirror-check.py --fix${NC}"
fi
exit "$OVERALL"

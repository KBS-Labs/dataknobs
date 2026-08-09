#!/usr/bin/env bash
# ShellCheck over the repository's own shell scripts.
#
# Every gate defect this batch tripped over was in a shell script, and the shell
# half of bin/ was linted by nothing while every *.py beside it went through
# ruff and mypy. The two dominant finding classes are the same defect stated
# twice: SC2086, where an unquoted empty variable becomes *zero* arguments so a
# target set silently comes out empty; and SC2155, where `local x=$(cmd)`
# discards the command's exit status because `local`'s status wins.
#
# TWO TIERS, AND THEY ARE A RATCHET.
#
#   strict   — zero findings at `info` and above. The scripts that decide whether
#              a quality run passes, plus every script that already happened to
#              be clean when this check was written.
#   baseline — errors only. Everything else, held to shellcheck's `error`
#              severity while its warnings and notes are worked down.
#
# THE STRICT FLOOR IS `info`, NOT `style`, AND THE LINE IS SHELLCHECK'S OWN.
# Its `style` severity is stylistic preference with no defect content — SC2001
# (`sed` where a parameter expansion would do), SC2129 (consecutive redirects),
# SC2006 (backticks). `info` and above is "this may be a bug", and that is where
# SC2086 lives: it is reported as `info`, so a floor of `warning` — the obvious
# choice, and the one bin/lint-workflows.sh uses on generated blocks — would
# exclude the single class most of this check exists to catch. Raising the floor
# to `style` is the next turn of the ratchet, not a separate decision; it is 13
# findings across the strict tier today, all mechanical.
#
# A baseline script that reaches zero must be promoted, and
# tests/test_shell_lint_coverage.py fails when one becomes eligible. So the
# deferred set can only shrink and nobody has to remember to revisit it. That
# guard is also what stops the tiers being used the wrong way round: demoting a
# script instead of fixing it is a passing move here and a failing one there.
#
# Requires: shellcheck. Its absence is fatal, never a skip — see the guard below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

#: The severity floor the strict tier is held to. See the header: `info` and
#: above is shellcheck's own line between "may be a bug" and "stylistic
#: preference", and SC2086 sits on the `info` side of it.
STRICT_SEVERITY=info

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Scripts held to zero findings.
#
# The first ten decide whether a quality run passes; the rest were already
# clean and are listed so they stay that way. Names only — the enumeration below
# decides what exists, and a name here matching no tracked file is caught by
# test_the_strict_pin_has_not_drifted_from_the_declaration rather than silently
# checking nothing.
STRICT_SCRIPTS=(
    # The verdict path.
    "bin/run-quality-checks.sh"
    "bin/validate.sh"
    "bin/test.sh"
    "bin/validate-quality-artifacts.sh"
    "bin/package-discovery.sh"
    "bin/lint-workflows.sh"
    "bin/docs-checks.sh"
    "bin/dk"
    "bin/fix.sh"
    "bin/lint-shell.sh"  # this file: a check exempt from itself checks nothing
    # Already clean at the strict floor. Kept clean.
    "bin/check-ollama.sh"
    "bin/debug-test.sh"
    "bin/docs-build.sh"
    "bin/docs-serve.sh"
    "bin/ensure-network.sh"
    "bin/setup-git-config.sh"
    "bin/update-model-limits.sh"
    "setup-dk.sh"  # the installer four documentation pages tell you to run
)

# Enumerate the shell scripts to check: every tracked shell file, repo-wide.
#
# DELIBERATELY NOT DERIVED FROM `workspace_targets`. That helper is the single
# declaration of which first-party code belongs to no package, and reusing it
# here was the obvious move — it already names bin/, which is where 44 of the 46
# shell files live. It is the wrong shape all the same, and trying it is how
# that was found: it silently omitted setup-dk.sh, the installer four separate
# documentation pages tell a new contributor to run, along with run_api.sh.
# Both sit at the repository root, which belongs to no target in that list
# because ruff and mypy have no reason to name it.
#
# The two questions only look alike. "Which non-package Python do we lint" is
# answered by a list of directories; "which shell scripts exist" is answered by
# the files themselves, and no package ships one — verified, and the guard in
# tests/test_shell_lint_coverage.py re-verifies it on every run by comparing
# this output against its own independent walk.
#
# Tracked files only, via git: a filesystem walk would pick up whatever an
# untracked scratch script happened to leave in bin/. Extension alone is not
# enough — bin/dk carries no suffix, and missing exactly the entry point every
# other invocation goes through is the shape of gap this check exists to close.
# So: *.sh by name, everything else by shebang.
shell_targets() {
    local found=() candidate first

    while IFS= read -r candidate; do
        [[ -z "$candidate" ]] && continue
        if [[ "$candidate" == *.sh ]]; then
            found+=("$candidate")
            continue
        fi
        [[ -f "$ROOT_DIR/$candidate" ]] || continue
        # A builtin read rather than `head -c 128 | head -n 1`. This loop visits
        # every tracked file, ~2000 of which are not *.sh, and two forks apiece
        # cost close to four seconds — paid by every mode: the check the gate
        # records, and each guard that asks what the tiers contain.
        #
        # `-n 128` keeps the original bound for a file with no newline. Both the
        # redirect (unreadable file) and `read` itself (EOF before a delimiter)
        # can report failure, and neither is an error here, so both are absorbed
        # explicitly — `set -e` is in force.
        first=""
        IFS= read -r -n 128 first < "$ROOT_DIR/$candidate" 2>/dev/null || true
        if [[ "$first" == '#!'* && "$first" == *sh* ]]; then
            found+=("$candidate")
        fi
    done < <(cd "$ROOT_DIR" && git ls-files 2>/dev/null | sort)

    [[ ${#found[@]} -eq 0 ]] || printf '%s\n' "${found[@]}"
}

is_strict() {
    local needle="$1" entry
    for entry in "${STRICT_SCRIPTS[@]}"; do
        [[ "$entry" == "$needle" ]] && return 0
    done
    return 1
}

# Which tier a file is held to. The single place that decision is made, so the
# run loop and --check-file cannot answer it differently.
tier_for() {
    if is_strict "$1"; then echo strict; else echo baseline; fi
}

severity_for() {
    case "$1" in
        strict) echo "$STRICT_SEVERITY" ;;
        baseline) echo error ;;
        *) echo "Unknown tier: $1" >&2; return 2 ;;
    esac
}

# Check one file at its tier, printing any findings. Non-zero if it has some.
#
# THE RUN LOOP CALLS THIS, WHICH IS THE ENTIRE POINT OF ITS BEING A FUNCTION.
# The tier split used to live inline in that loop, where no test could reach it:
# every guard read a --print-* mode, so all of them kept passing with the strict
# floor deleted and all 46 scripts silently dropped to `error`. A guard that
# cannot fail against the regression it is written for is worse than none,
# because it also reports green. Sharing the executor is what makes
# test_the_check_holds_the_two_tiers_to_different_floors an assertion about the
# real run rather than about a second implementation of it.
check_one() {
    local file="$1" tier="${2:-}"
    local severity label output rc
    [[ -n "$tier" ]] || tier="$(tier_for "$file")"
    severity="$(severity_for "$tier")" || return 2
    if [[ "$tier" == strict ]]; then label="must be clean"; else label="errors only"; fi

    output=$( cd "$ROOT_DIR" && shellcheck -x -f gcc -S "$severity" "$file" 2>&1 ) && rc=0 || rc=$?
    if [[ "$rc" -ne 0 || -n "$output" ]]; then
        echo -e "  ${RED}✗${NC} $file ($label)"
        printf '%s\n' "$output" | sed 's/^/      /'
        return 1
    fi
    return 0
}

# Findings for one file at the given severity floor, as a count.
count_findings() {
    local file="$1" severity="$2"
    ( cd "$ROOT_DIR" && shellcheck -x -f gcc -S "$severity" "$file" 2>/dev/null || true ) | grep -c . || true
}

require_shellcheck() {
    # Fatal, not a skip. A skip leaves the run reporting success having analysed
    # nothing — and CI validates the committed artifact instead of re-running the
    # gate, so an artifact from a machine without shellcheck is indistinguishable
    # from one where the linter ran and found nothing. This is also what the
    # dependency rules require of a tool we invoke as a subprocess: gate its
    # presence explicitly and fail, never warn and continue.
    if ! command -v shellcheck >/dev/null 2>&1; then
        echo -e "  ${RED}✗${NC} shellcheck is required but not installed" >&2
        echo "    macOS:  brew install shellcheck" >&2
        echo "    Debian: apt-get install shellcheck" >&2
        exit 1
    fi
}

usage() {
    cat <<'EOF'
Usage: lint-shell.sh [OPTION]

Run shellcheck over the repository's own shell scripts.

  (no option)          Check every script and report. Non-zero on findings.
  --print-targets      List the scripts checked, one per line.
  --print-strict       List the scripts held to zero findings.
  --print-baseline     List the scripts held to errors only.
  --print-promotable   List baseline scripts that are already clean.
  --check-file PATH [TIER]
                       Check one file and exit non-zero on findings. TIER is
                       strict or baseline; without it the file's own tier is
                       used. Naming it answers "would this pass promotion",
                       which is --print-promotable for a single file, with the
                       findings shown rather than just the verdict.
  -h, --help           Show this message.

The print modes do not require shellcheck; --print-promotable and --check-file do.
EOF
}

MODE="${1:-check}"

case "$MODE" in
    --print-targets)
        shell_targets
        exit 0
        ;;
    --print-strict)
        # Intersected with what exists rather than echoed back, so this answers
        # "which files are held to zero" and not "what does the list say".
        while IFS= read -r file; do
            is_strict "$file" && echo "$file"
        done < <(shell_targets)
        exit 0
        ;;
    --print-baseline)
        while IFS= read -r file; do
            is_strict "$file" || echo "$file"
        done < <(shell_targets)
        exit 0
        ;;
    --print-promotable)
        require_shellcheck
        while IFS= read -r file; do
            is_strict "$file" && continue
            [[ "$(count_findings "$file" "$STRICT_SEVERITY")" -eq 0 ]] && echo "$file"
        done < <(shell_targets)
        exit 0
        ;;
    --check-file)
        require_shellcheck
        if [[ $# -lt 2 ]]; then
            echo "--check-file needs a path" >&2
            usage >&2
            exit 2
        fi
        if [[ ! -f "$2" ]]; then
            echo "--check-file: no such file: $2" >&2
            exit 2
        fi
        # Guarded by `if`, not left to propagate: under `set -e` a non-zero
        # return here would exit before the status could be turned into a
        # verdict, which is the same shape of mistake this file exists to find.
        if check_one "$2" "${3:-}"; then
            exit 0
        fi
        exit 1
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    check)
        ;;
    *)
        echo "Unknown option: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac

require_shellcheck

# A strict entry naming no real file is the one failure this check cannot report
# by running: the tiers are decided by membership, so a typo'd or renamed entry
# simply stops matching, and the script it named drops into the baseline tier
# and is quietly held to a lower bar. Nothing goes red — the run gets *easier*.
# So the list is checked against the enumeration before it is used.
# Collected into an array first, deliberately. The obvious spelling —
# `shell_targets | grep -qxF "$entry"` — is wrong under this script's own
# `pipefail`: grep -q exits at the first match, shell_targets takes SIGPIPE for
# writing to a closed pipe, and pipefail reports the pipeline as failed. So an
# entry that matches EARLY reads as absent while one that matches last reads as
# present, which is a membership test whose answer depends on sort order. It is
# the same defect as deciding a verdict with `mypy | grep`: the exit status
# consulted is not the one that answers the question.
ALL_TARGETS=()
while IFS= read -r target_line; do
    ALL_TARGETS+=("$target_line")
done < <(shell_targets)

STRICT_PHANTOMS=()
for entry in "${STRICT_SCRIPTS[@]}"; do
    # Not named `found` — shell_targets() has a local array by that name, and
    # reusing it here reads as reassigning an array to a string.
    entry_matched=false
    for target_line in "${ALL_TARGETS[@]}"; do
        [[ "$target_line" == "$entry" ]] && { entry_matched=true; break; }
    done
    [[ "$entry_matched" == false ]] && STRICT_PHANTOMS+=("$entry")
done
if [[ ${#STRICT_PHANTOMS[@]} -gt 0 ]]; then
    echo -e "  ${RED}✗${NC} STRICT_SCRIPTS names files that are not tracked shell scripts:" >&2
    printf '      %s\n' "${STRICT_PHANTOMS[@]}" >&2
    echo "    Renamed or deleted? Fix the entry — leaving it stale silently" >&2
    echo "    demotes that script to the baseline tier." >&2
    exit 1
fi

echo -e "${BLUE}Linting shell scripts...${NC}"

FAILED=false
STRICT_CHECKED=0
BASELINE_CHECKED=0

while IFS= read -r file; do
    tier="$(tier_for "$file")"
    if [[ "$tier" == strict ]]; then
        STRICT_CHECKED=$((STRICT_CHECKED + 1))
    else
        BASELINE_CHECKED=$((BASELINE_CHECKED + 1))
    fi

    check_one "$file" "$tier" || FAILED=true
done < <(shell_targets)

echo ""
echo "  Scripts held to zero findings: $STRICT_CHECKED"
echo "  Scripts held to errors only:   $BASELINE_CHECKED"

if [[ "$FAILED" == true ]]; then
    echo -e "\n${RED}✗ Shell lint failed${NC}"
    echo -e "${YELLOW}Note:${NC} a finding in a strict script is fixed, not deferred."
    echo "      Moving it to the baseline tier fails tests/test_shell_lint_coverage.py."
    exit 1
fi

echo -e "\n${GREEN}✓ Shell lint passed${NC}"

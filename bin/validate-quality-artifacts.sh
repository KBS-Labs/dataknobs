#!/bin/bash
set -e

# Validate Quality Artifacts Script
# This script validates that quality check artifacts exist, are recent,
# and show passing results. Used by CI to ensure developers ran checks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/.quality-artifacts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REQUIRED_COVERAGE=${REQUIRED_COVERAGE:-70}  # Minimum coverage percentage

# --read-summary is dispatched below, before the banner, so its output is the
# projection and nothing else. A mode whose stdout carries a decorative header
# is one every caller has to strip, and the first caller to forget produces a
# parse failure that looks like a data problem.
READ_SUMMARY_MODE=""
if [ "${1:-}" = "--read-summary" ]; then
    if [ -z "${2:-}" ]; then
        echo "--read-summary needs a path" >&2
        exit 2
    fi
    READ_SUMMARY_MODE="$2"
fi

if [ -z "$READ_SUMMARY_MODE" ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}         Validating Quality Check Artifacts                       ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
fi

# Function to print status
print_check() {
    echo -e "${BLUE}▶${NC} Checking: $1"
}

print_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

# For a check that reports a concern it will not fail the build over. Every
# print_fail must reach VALIDATION_FAILED or exit — a red ✗ on a run that then
# reports success is indistinguishable from a passing run, and both times that
# has happened here it went unnoticed for months. Guarded by
# tests/test_quality_gate_accounting.py, which applies the same rule to
# run-quality-checks.sh.
print_warn() {
    echo -e "  ${YELLOW}!${NC} $1"
}

print_info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

# Read quality-summary.json in one JSON parse, projected into tab-delimited
# lines the shell can iterate:
#
#   OVERALL<TAB><overall_status>
#   CHECK<TAB><name><TAB><status><TAB><skipped><TAB><label>
#   ERROR<TAB><message>              (unreadable file; nothing else is emitted)
#
# This replaces seven line-offset greps — `grep -A2 '"unit_tests"'` for a status,
# `grep -A3` for a skipped flag. Those read the file by POSITION: any field added
# above the one they wanted pushed it out of the window and they returned
# nothing, so the validator rejected artifacts it had simply failed to read and
# gave "Unit tests: " with an empty status as the reason. Reordering the keys of
# a valid, passing summary was enough to fail a pull request, and adding a field
# in the wrong place was enough to do it repository-wide.
#
# Field order in the summary is now irrelevant, which is the point: JSON objects
# are unordered by definition, and the producer should not have to know where a
# reader's window falls.
# The parser lives in bin/read-quality-summary.py rather than in a string here,
# so that ruff and mypy check it and a test can drive it directly. A program
# embedded in a shell string is checked by nothing until it runs.
read_summary() {
    python3 "$SCRIPT_DIR/read-quality-summary.py" "$1" 2>/dev/null \
        || printf 'ERROR\tcould not run read-quality-summary.py\n'
}

# Print the projection for one summary file and exit. The main path below calls
# the same function, so a test that drives this is asserting about the reader the
# validator actually uses rather than a second copy of it — the reason this
# script accumulated seven positional greps is that nothing ever executed it.
if [ -n "$READ_SUMMARY_MODE" ]; then
    read_summary "$READ_SUMMARY_MODE"
    exit 0
fi

VALIDATION_FAILED=0

# Check if artifacts directory exists
print_check "Artifacts directory exists"
if [ ! -d "$ARTIFACTS_DIR" ]; then
    print_fail "Directory .quality-artifacts/ not found"
    print_fail "Run: ./bin/run-quality-checks.sh before creating PR"
    exit 1
fi
print_pass "Found .quality-artifacts/"

# Check for required files
print_check "Required artifact files"
# Every entry here must be a committed file, or this loop fails every pull
# request rather than the ones with a real problem. A guard in
# tests/test_quality_artifact_contract.py asserts that against .gitignore.
REQUIRED_FILES=(
    "quality-summary.json"
    "environment.json"
    "signature.sha256"
    "unit-test-results.xml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$ARTIFACTS_DIR/$file" ]; then
        print_fail "Missing: $file"
        VALIDATION_FAILED=1
    else
        print_pass "Found: $file"
    fi
done

if [ $VALIDATION_FAILED -eq 1 ]; then
    echo ""
    print_fail "Missing required artifacts. Run: ./bin/run-quality-checks.sh"
    exit 1
fi

# Check package content hashes
print_check "Package content hashes"
HASH_RESULT=$(uv run python "$SCRIPT_DIR/package-hashes.py" validate --json 2>/dev/null) || true

if [ -n "$HASH_RESULT" ]; then
    HASH_VALID=$(echo "$HASH_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('valid', False))")
    HASH_ERROR=$(echo "$HASH_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', ''))")

    HASH_WARNING=$(echo "$HASH_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('warning', ''))")

    # Printed additively, not as a branch. A warning describes what could not be
    # checked; it says nothing about whether what *was* checked passed. Chaining
    # it ahead of the verdict means any result carrying both reports the warning
    # and silently skips the failure — which is a green gate on stale artifacts.
    if [ -n "$HASH_WARNING" ]; then
        print_info "$HASH_WARNING"
    fi

    if [ -n "$HASH_ERROR" ]; then
        print_fail "Hash validation error: $HASH_ERROR"
        VALIDATION_FAILED=1
    elif [ "$HASH_VALID" = "True" ]; then
        DIRTY_COUNT=$(echo "$HASH_RESULT" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('dirty_packages', [])))")
        if [ "$DIRTY_COUNT" = "0" ]; then
            print_pass "All packages unchanged since last quality run"
        else
            print_pass "All dirty packages have been tested"
        fi
    else
        CHANGED=$(echo "$HASH_RESULT" | python3 -c "import sys, json; print(', '.join(json.load(sys.stdin).get('changed_packages', [])))")
        DIRTY=$(echo "$HASH_RESULT" | python3 -c "import sys, json; print(', '.join(json.load(sys.stdin).get('dirty_packages', [])))")
        SCOPES=$(echo "$HASH_RESULT" | python3 -c "import sys, json; print(', '.join(json.load(sys.stdin).get('changed_scopes', [])))")
        print_fail "Package content has changed since quality checks were run"
        if [ -n "$CHANGED" ]; then
            print_info "Changed packages: $CHANGED"
        fi
        if [ -n "$DIRTY" ]; then
            print_fail "Packages needing re-validation: $DIRTY"
        fi
        # A workspace-only scope dirties no package by design, so without this
        # line that case fails with every other field empty — the artifacts are
        # stale and the report names nothing that changed.
        if [ -n "$SCOPES" ]; then
            print_info "Changed workspace scopes: $SCOPES"
        fi
        print_fail "Please run: ./bin/run-quality-checks.sh"
        VALIDATION_FAILED=1
    fi
else
    print_fail "Could not validate package content hashes"
    print_info "Ensure uv and Python are available"
    VALIDATION_FAILED=1
fi

# Validate test results
print_check "Test results"
if [ -f "$ARTIFACTS_DIR/quality-summary.json" ]; then
    SUMMARY_PROJECTION=$(read_summary "$ARTIFACTS_DIR/quality-summary.json")
    SUMMARY_ERROR=$(printf '%s\n' "$SUMMARY_PROJECTION" | sed -n 's/^ERROR\t//p')

    if [ -n "$SUMMARY_ERROR" ]; then
        # An unreadable summary is a failure, not a skip. It is the file the
        # whole gate attests through, so "could not parse it" and "it says the
        # run passed" must never reach the same verdict.
        print_fail "Could not read quality-summary.json: $SUMMARY_ERROR"
        VALIDATION_FAILED=1
    else
        OVERALL_STATUS=$(printf '%s\n' "$SUMMARY_PROJECTION" | sed -n 's/^OVERALL\t//p')

        if [ "$OVERALL_STATUS" = "PASS" ]; then
            print_pass "Overall status: PASS"
        elif [ "$OVERALL_STATUS" = "PASS_WITH_SKIPS" ]; then
            print_pass "Overall status: PASS_WITH_SKIPS (some checks were skipped)"
        else
            print_fail "Overall status: $OVERALL_STATUS (expected: PASS or PASS_WITH_SKIPS)"
            VALIDATION_FAILED=1
        fi

        # Every check the summary records, rather than a hand-kept list of three.
        # The old list named unit_tests, integration_tests and workflow_lint and
        # said nothing about the other five, so a shell-lint or docs failure
        # reached CI as a bare "Overall status: FAIL" with no indication of what
        # broke. Naming them cannot change any verdict — all eight already feed
        # overall_status, which is asserted by the guards in
        # tests/test_quality_gate_accounting.py — so this is strictly a better
        # message for a run that was failing either way.
        #
        # A herestring, not a pipe: a `while` loop on the right of a pipe runs in
        # a subshell, where every VALIDATION_FAILED set below would be discarded
        # when it exits, and the gate would report success over failing checks.
        CHECKS_SEEN=""
        while IFS="$(printf '\t')" read -r kind name status _skipped label; do
            [ "$kind" = "CHECK" ] || continue
            CHECKS_SEEN="$CHECKS_SEEN $name"

            if [ "$status" = "pass" ]; then
                print_pass "$label: PASS"
            else
                print_fail "$label: ${status:-no status recorded}"
                VALIDATION_FAILED=1
            fi
        done <<< "$SUMMARY_PROJECTION"

        # Absence is tolerated for every check except this one, because an
        # artifact predating a check carries no entry for it and must still
        # validate. Unit tests are the exception: a summary with no unit-test
        # entry is not an old artifact, it is one that never ran them.
        case " $CHECKS_SEEN " in
            *" unit_tests "*) ;;
            *)
                print_fail "Unit tests: no entry in quality-summary.json"
                VALIDATION_FAILED=1
                ;;
        esac

        if [ "$OVERALL_STATUS" = "PASS_WITH_SKIPS" ]; then
            while IFS="$(printf '\t')" read -r kind name _status skipped _label; do
                [ "$kind" = "CHECK" ] && [ "$skipped" = "true" ] || continue
                print_info "  Skipped: $name"
            done <<< "$SUMMARY_PROJECTION"
        fi
    fi
fi

# Validate coverage
#
# Read from the summary rather than from coverage.xml, which is no longer
# committed. The number this reports has always been advisory — no branch below
# sets VALIDATION_FAILED — so it never justified carrying a multi-megabyte
# generated report through every merge. It says so with print_warn: announcing
# a failure it will not act on is how the signature check below spent its
# entire life reporting a defect nobody could see.
print_check "Code coverage"
COVERAGE=$(python3 -c "
import json
try:
    with open('$ARTIFACTS_DIR/quality-summary.json') as fh:
        value = json.load(fh).get('coverage_percent')
    print('' if value is None else f'{float(value):.1f}')
except Exception:
    print('')
" 2>/dev/null || echo "")

if [ -z "$COVERAGE" ]; then
    # Absent on artifacts written before the field existed, and null when the
    # run produced no coverage data at all. Neither is a failing condition for a
    # measure that cannot fail, so this reports what it knows and moves on.
    print_info "Coverage: not recorded in quality-summary.json"
elif (( $(echo "$COVERAGE >= $REQUIRED_COVERAGE" | bc -l 2>/dev/null || echo 0) )); then
    print_pass "Coverage: ${COVERAGE}% (minimum: ${REQUIRED_COVERAGE}%)"
else
    print_warn "Coverage: ${COVERAGE}% (below minimum: ${REQUIRED_COVERAGE}%)"
    print_info "Low coverage is a warning, not a failure"
fi

# Verify artifact signature
print_check "Artifact integrity"
if [ -f "$ARTIFACTS_DIR/signature.sha256" ]; then
    cd "$ARTIFACTS_DIR"
    
    # Must enumerate the same way the producer does — the committed set, not
    # everything on disk — or the two sides compare different file lists and
    # mismatch unconditionally.
    CURRENT_SIG=$(git ls-files --cached --others --exclude-standard -- '*.json' '*.xml' \
        | sed 's|^|./|' | sort | xargs sha256sum 2>/dev/null | sort)
    
    # Read stored signature (excluding the signature file itself)
    STORED_SIG=$(grep -v "signature.sha256" signature.sha256 2>/dev/null | sort)
    
    if [ "$CURRENT_SIG" = "$STORED_SIG" ]; then
        print_pass "Artifact signature valid"
    else
        print_fail "Artifact signature mismatch - artifacts do not match their signature"
        print_info "Usually a merge or rebase that spliced two runs together,"
        print_info "or a file added to or dropped from the committed set."
        print_info "Re-run: ./bin/dk pr"
        VALIDATION_FAILED=1
    fi

    cd "$PROJECT_ROOT"
else
    # Unreachable while signature.sha256 is in REQUIRED_FILES above, which exits
    # first. Accounted for anyway: the day it leaves that list, a missing
    # signature must not become the one artifact whose absence is tolerated.
    print_fail "Signature file not found"
    VALIDATION_FAILED=1
fi

# Final summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                        Validation Summary                        ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $VALIDATION_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All validations passed!${NC}"
    echo -e "${GREEN}  Quality checks have been run and passed.${NC}"
    echo -e "${GREEN}  PR is ready for review.${NC}"
    echo ""
    
    # Output for GitHub Actions
    if [ -n "$GITHUB_ACTIONS" ]; then
        echo "::notice::Quality artifacts validated successfully"
    fi
    
    exit 0
else
    echo -e "${RED}✗ Validation failed!${NC}"
    echo -e "${RED}  Please run: ./bin/run-quality-checks.sh${NC}"
    echo -e "${RED}  Ensure all tests pass before creating a PR.${NC}"
    echo ""
    
    # Output for GitHub Actions
    if [ -n "$GITHUB_ACTIONS" ]; then
        echo "::error::Quality artifacts validation failed. Run ./bin/run-quality-checks.sh locally."
    fi
    
    exit 1
fi
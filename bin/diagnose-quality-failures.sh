#!/bin/bash
set -e

# Quality Failure Diagnostic Script for DataKnobs
# Analyzes the output of a prior quality run to pinpoint failures.
#
# Two tiers can hold that output: .quality-artifacts/ when the gate produced it,
# and the diagnostics tier when a checker did. This tool reads whichever holds
# the newer summary, so it diagnoses the last run either way — which is the
# point of the diagnostics tier existing at all. A checker writes nothing under
# .quality-artifacts/, so before it there was nothing here to read unless the
# developer had run the gate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/.quality-artifacts"

# Which run to diagnose. Resolved after the arguments are parsed, because
# --from names the answer outright and the default has to go ask for it.
resolve_source_dir() {
    if [ -n "$FROM_DIR" ]; then
        printf '%s\n' "$FROM_DIR"
        return 0
    fi

    # Asked for by name rather than spelled here, so this tool cannot disagree
    # with run-quality-checks.sh about where that script's own output goes. The
    # literal is the fallback for the case where asking fails.
    local reports_dir
    reports_dir=$("$SCRIPT_DIR/run-quality-checks.sh" --print-output-dir 2>/dev/null) \
        || reports_dir="$PROJECT_ROOT/.quality-reports"

    # Newer summary wins. Bash's -nt is also true when the left exists and the
    # right does not, which is the "only a checker has run" case; both-absent
    # leaves the artifacts directory selected and is reported below.
    if [ "$reports_dir/quality-summary.json" -nt "$ARTIFACTS_DIR/quality-summary.json" ]; then
        printf '%s\n' "$reports_dir"
    else
        printf '%s\n' "$ARTIFACTS_DIR"
    fi
}

# Colors for output
if [ -t 1 ] && [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    MAGENTA='\033[0;35m'
    BOLD='\033[1m'
    DIM='\033[2m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    MAGENTA=''
    BOLD=''
    DIM=''
    NC=''
fi

# Function to show usage
show_usage() {
    cat << EOF
${CYAN}DataKnobs Quality Failure Diagnostics${NC}

Usage: $0 [OPTIONS]

Analyze the most recent quality run to pinpoint specific failures and
provide actionable fixes. Reads .quality-reports/ (a checker run) or
.quality-artifacts/ (bin/dk pr), whichever ran more recently.

${YELLOW}Options:${NC}
    -v, --verbose       Show detailed output for all issues
    -s, --summary       Show only the summary (default)
    -t, --tests         Focus on test failures
    -l, --lint          Focus on linting issues
    -c, --coverage      Show coverage details
    -f, --fix           Show auto-fix commands where available
        --from DIR      Diagnose the run whose output is in DIR
    -h, --help          Show this help message

${YELLOW}Examples:${NC}
    $0                  # Show summary of all failures
    $0 -v               # Show detailed failure information
    $0 -t               # Focus on test failures only
    $0 -f               # Show commands to auto-fix issues

EOF
    exit 0
}

# Default options
VERBOSE=false
SHOW_TESTS=true
SHOW_LINT=true
SHOW_STYLE=true
SHOW_COVERAGE=false
SHOW_FIXES=false
FOCUS_MODE=""
FROM_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--summary)
            VERBOSE=false
            shift
            ;;
        -t|--tests)
            FOCUS_MODE="tests"
            SHOW_TESTS=true
            SHOW_LINT=false
            SHOW_STYLE=false
            shift
            ;;
        -l|--lint)
            FOCUS_MODE="lint"
            SHOW_TESTS=false
            SHOW_LINT=true
            SHOW_STYLE=true
            shift
            ;;
        -c|--coverage)
            SHOW_COVERAGE=true
            shift
            ;;
        -f|--fix)
            SHOW_FIXES=true
            shift
            ;;
        --from)
            if [ -z "${2:-}" ]; then
                echo "--from needs a directory"
                exit 2
            fi
            FROM_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

SOURCE_DIR="$(resolve_source_dir)"

# A run that started here and never recorded a verdict. run-quality-checks.sh
# writes this before its first abortable check and removes it when the summary
# lands, so its presence means the newest thing in this directory is a run whose
# logs are here and whose verdict is not.
#
# Which matters because the summary beside those logs may well be an older run's.
# The gate cannot clear its directory on entry — the directory holds committed
# files — so an abort leaves two runs interleaved, and the older half is the half
# holding the verdict. Without this the tool reads that verdict out as the
# current one, which for a green previous run means answering "what broke?" with
# a full set of passing checks.
ABORTED_AT=""
if [ -f "$SOURCE_DIR/.run-in-progress" ]; then
    ABORTED_AT=$(head -n 1 "$SOURCE_DIR/.run-in-progress" 2>/dev/null)
fi

# Neither tier holds a run. Naming the checker first: it is the cheaper of the
# two and produces everything this tool reads.
if [ ! -f "$SOURCE_DIR/quality-summary.json" ]; then
    if [ -n "$ABORTED_AT" ]; then
        echo -e "${RED}✗ The last run (started ${ABORTED_AT}) stopped before recording anything.${NC}"
        echo -e "${YELLOW}  It aborted ahead of the first check — services, package resolution${NC}"
        echo -e "${YELLOW}  or the style scope. That run printed the reason; any logs it did${NC}"
        echo -e "${YELLOW}  write are in ${SOURCE_DIR}.${NC}"
        exit 1
    fi
    echo -e "${RED}✗ No quality run found to diagnose!${NC}"
    echo -e "${YELLOW}  Run './bin/run-quality-checks.sh' (checks only), or${NC}"
    echo -e "${YELLOW}  './bin/dk pr' if you also need the artifacts CI verifies.${NC}"
    exit 1
fi

# Function to print section headers
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

# Function to extract test failures
analyze_test_failures() {
    local test_type=$1
    local output_file=$2
    
    if [ ! -f "$output_file" ]; then
        return
    fi
    
    # Count failures - use echo to ensure we get a valid number
    local failure_count=$(grep -c "FAILED" "$output_file" 2>/dev/null || true)
    failure_count=${failure_count:-0}
    local error_count=$(grep -c "ERROR" "$output_file" 2>/dev/null || true)
    error_count=${error_count:-0}
    
    if [ "$failure_count" -eq 0 ] && [ "$error_count" -eq 0 ]; then
        return
    fi
    
    echo -e "\n${YELLOW}$test_type Test Failures:${NC} $failure_count failed, $error_count errors"
    
    if [ "$VERBOSE" = true ]; then
        # Extract detailed failure information
        echo -e "${DIM}────────────────────────────────────────${NC}"
        
        # Find FAILED lines and show context
        grep -A 5 "FAILED" "$output_file" 2>/dev/null | while IFS= read -r line; do
            if [[ "$line" =~ ^FAILED ]]; then
                # Extract test name and file
                test_info=$(echo "$line" | sed 's/FAILED //')
                echo -e "${RED}  ✗ $test_info${NC}"
            elif [[ "$line" =~ AssertionError ]] || [[ "$line" =~ Error ]]; then
                echo -e "${DIM}    $line${NC}"
            fi
        done
        
        # Show short test summary if available
        if grep -q "= short test summary info =" "$output_file"; then
            echo -e "\n${CYAN}Summary from pytest:${NC}"
            sed -n '/= short test summary info =/,/^=/p' "$output_file" | grep -E "^(FAILED|ERROR)" | head -10
        fi
    else
        # Just show first few failures
        echo -e "${DIM}  First few failures:${NC}"
        grep "FAILED" "$output_file" 2>/dev/null | head -3 | sed 's/^/    /'
    fi
    
    if [ "$SHOW_FIXES" = true ]; then
        echo -e "\n${GREEN}To re-run failed tests:${NC}"
        echo "    ./bin/test.sh -- --lf              # Run last failed"
        echo "    ./bin/test.sh -- -k 'test_name'    # Run specific test"
        echo "    ./bin/test.sh -- -x -vvs           # Stop on first failure with verbose output"
    fi
}

# Function to analyze linting issues
#
# Reads validation.log — what the lint check writes. This read used to be
# lint-report.json, which nothing has ever written: the file had a reader here,
# an un-ignore in .gitignore, and no producer, so the lint half of this tool
# returned at its first line on every run since it was written. The shape it
# parsed says where it came from — `.message-id` and `.path` are pylint's JSON
# schema, not ruff's, and pylint has never run in the gate.
#
# validation.log is the output of bin/validate.sh: syntax, ruff, imports, mypy,
# per target, as text. Tailed rather than parsed, because a line-oriented log
# has no schema to parse and inventing one here is how the last version of this
# function came to describe a file that did not exist.
analyze_lint_issues() {
    local log="$SOURCE_DIR/validation.log"
    if [ ! -f "$log" ]; then
        echo -e "\n${DIM}  No validation.log in $SOURCE_DIR — nothing to show.${NC}"
        return
    fi

    local lines
    if [ "$VERBOSE" = true ]; then
        lines=200
    else
        lines=40
    fi

    echo -e "\n${YELLOW}Validation output${NC} ${DIM}(last $lines lines of validation.log)${NC}"
    echo -e "${DIM}────────────────────────────────────────${NC}"
    tail -n "$lines" "$log"
    echo -e "${DIM}────────────────────────────────────────${NC}"
    echo -e "${DIM}  Full log: $log${NC}"

    if [ "$SHOW_FIXES" = true ]; then
        echo -e "\n${GREEN}To re-run validation, with auto-fix where available:${NC}"
        echo "    ./bin/validate.sh -f"
    fi
}

# Function to analyze style issues
analyze_style_issues() {
    if [ ! -f "$SOURCE_DIR/style-check.json" ]; then
        return
    fi
    
    # Try to parse JSON
    if command -v jq &> /dev/null; then
        local issue_count=$(jq 'length' "$SOURCE_DIR/style-check.json" 2>/dev/null || echo "0")
        
        if [ "$issue_count" -gt 0 ]; then
            echo -e "\n${YELLOW}Style Issues:${NC} $issue_count violations found"
            
            if [ "$VERBOSE" = true ]; then
                echo -e "${DIM}────────────────────────────────────────${NC}"
                jq -r '.[] | "  \(.filename):\(.location.row): \(.message) [\(.code)]"' \
                    "$SOURCE_DIR/style-check.json" 2>/dev/null | head -20
            else
                # Show summary by violation code
                echo -e "${DIM}  Violation types:${NC}"
                jq -r '[.[].code] | group_by(.) | map({code: .[0], count: length}) | .[] | "    \(.code): \(.count)"' \
                    "$SOURCE_DIR/style-check.json" 2>/dev/null | sort -t: -k2 -rn | head -10
            fi
        fi
    else
        # Fallback without jq
        echo -e "\n${YELLOW}Style Report:${NC} (install 'jq' for better formatting)"
        python3 -c "
import json
with open('$SOURCE_DIR/style-check.json') as f:
    data = json.load(f)
    if data:
        print(f'  {len(data)} violations found')
        for item in data[:5]:
            loc = item.get('location', {})
            print(f\"  {item.get('filename', 'unknown')}:{loc.get('row', '?')}: {item.get('message', 'no message')}\")
        if len(data) > 5:
            print(f'  ... and {len(data)-5} more')
" 2>/dev/null || echo "  Could not parse style report"
    fi
    
    if [ "$SHOW_FIXES" = true ]; then
        # Entry points, not tool invocations. The commands spelled out here
        # named packages/*/src and passed no --config, so neither could
        # reproduce a bin/ finding this same script had just listed, and both
        # resolved the per-package [tool.ruff] sections rather than the root one
        # the gate uses. Naming the entry point means the advice cannot drift
        # from what the gate does, because it is what the gate does.
        echo -e "\n${GREEN}To auto-fix style issues:${NC}"
        echo "    ./bin/fix.sh"
        echo -e "${GREEN}To see all style issues:${NC}"
        echo "    ./bin/validate.sh --workspace"
    fi
}

# Function to show coverage details
analyze_coverage() {
    if [ ! -f "$SOURCE_DIR/coverage-by-package.json" ]; then
        return
    fi
    
    echo -e "\n${CYAN}Test Coverage by Package:${NC}"
    echo -e "${DIM}────────────────────────────────────────${NC}"
    
    if command -v jq &> /dev/null; then
        jq -r '.packages | to_entries | .[] | "  \(.key): \(.value.coverage) (\(.value.statements - .value.missing)/\(.value.statements) statements)"' \
            "$SOURCE_DIR/coverage-by-package.json" 2>/dev/null
    else
        python3 -c "
import json
with open('$SOURCE_DIR/coverage-by-package.json') as f:
    data = json.load(f)
    for pkg, info in data.get('packages', {}).items():
        covered = info['statements'] - info['missing']
        print(f\"  {pkg}: {info['coverage']} ({covered}/{info['statements']} statements)\")
" 2>/dev/null || echo "  Could not parse coverage report"
    fi
    
    if [ -f "$SOURCE_DIR/htmlcov/index.html" ]; then
        echo -e "\n${GREEN}View detailed coverage report:${NC}"
        echo "    open $SOURCE_DIR/htmlcov/index.html"
    fi
}

# Main diagnostic flow
echo -e "${BOLD}${CYAN}DataKnobs Quality Diagnostics${NC}"
echo -e "${DIM}Analyzing artifacts from $(date -r "$SOURCE_DIR/quality-summary.json" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'recent run')${NC}"

# Read the summary once, through the parser CI reads it with.
#
# This used to branch on `command -v jq`. The jq path asked for `.checks.lint`
# and `.checks.style`, neither of which the producer has ever emitted: jq prints
# null for a missing path, null is not "pass", so it showed two amber rows on
# every run it ever made, including runs where everything passed. The other path
# set no status variables at all, so a machine without jq diagnosed every run as
# failing in four places. Both are the same defect — a second reader of a file
# that already has one — and jq being unpinned here is what made the second
# reader necessary. Python is not optional in a repository whose toolchain is
# Python, so there is one reader now and no branch.
SUMMARY_PROJECTION=$(python3 "$SCRIPT_DIR/read-quality-summary.py" \
    "$SOURCE_DIR/quality-summary.json" 2>/dev/null \
    || printf 'ERROR\037could not run read-quality-summary.py\n')

# One field of one record, by name. Callers below name the field they want, so
# no reader here holds a column number.
summary_meta() {
    local want="$1" kind key value
    while IFS="$(printf '\037')" read -r kind key value; do
        [ "$kind" = "META" ] && [ "$key" = "$want" ] || continue
        printf '%s\n' "$value"
        return 0
    done <<< "$SUMMARY_PROJECTION"
}

check_field() {
    local want="$1" field="$2"
    local kind name status skipped exit_code tool label
    while IFS="$(printf '\037')" read -r kind name status skipped exit_code tool label; do
        [ "$kind" = "CHECK" ] && [ "$name" = "$want" ] || continue
        case "$field" in
            status)    printf '%s\n' "$status" ;;
            skipped)   printf '%s\n' "$skipped" ;;
            exit_code) printf '%s\n' "$exit_code" ;;
            tool)      printf '%s\n' "$tool" ;;
            label)     printf '%s\n' "$label" ;;
        esac
        return 0
    done <<< "$SUMMARY_PROJECTION"
}

OVERALL_STATUS=$(printf '%s\n' "$SUMMARY_PROJECTION" \
    | while IFS="$(printf '\037')" read -r kind value; do
        [ "$kind" = "OVERALL" ] && printf '%s\n' "$value" && break
    done)
TIMESTAMP=$(summary_meta timestamp)
ENVIRONMENT=$(summary_meta environment)
PACKAGES=$(summary_meta packages)

# The four the sections below analyse. Every other recorded check is displayed
# and reported by enumeration; these are named only because there is specific
# analysis to run for them.
#
# `validation`, not `lint`: the producer has never emitted a check by either
# that name or `style`. Style has no verdict of its own to read — ruff runs
# inside validate.sh and its result is part of `validation` — so it is derived
# from style-check.json, where an empty findings array is a pass.
LINT_STATUS=$(check_field validation status)
LINT_CODE=$(check_field validation exit_code)
UNIT_STATUS=$(check_field unit_tests status)
INT_STATUS=$(check_field integration_tests status)
STYLE_COUNT=$(python3 -c "
import json,sys
try:
    with open(sys.argv[1]) as fh:
        print(len(json.load(fh)))
except Exception:
    print(0)
" "$SOURCE_DIR/style-check.json" 2>/dev/null || echo 0)
if [ "$STYLE_COUNT" = "0" ]; then
    STYLE_STATUS="pass"
else
    STYLE_STATUS="fail"
fi

# Said before the summary rather than after it, and in the words that describe
# what the reader is about to see. A warning printed under a heading that says
# "Quality Check Summary" is read as commentary on the summary; this has to be
# read as a statement about which run it belongs to.
if [ -n "$ABORTED_AT" ]; then
    echo ""
    echo -e "${RED}${BOLD}⚠ The run that started ${ABORTED_AT} never finished.${NC}"
    echo -e "${YELLOW}  Everything below is from the run before it. Some of the logs in${NC}"
    echo -e "${YELLOW}  ${SOURCE_DIR}${NC}"
    echo -e "${YELLOW}  have been overwritten since, so they and this verdict are not the${NC}"
    echo -e "${YELLOW}  same run. Re-run to get a diagnosis of the current tree.${NC}"
fi

# Show summary header
print_header "Quality Check Summary"
echo -e "  Timestamp:    ${TIMESTAMP}"
echo -e "  Environment:  ${ENVIRONMENT}"
echo -e "  Packages:     ${PACKAGES}"
if [ "$OVERALL_STATUS" = "PASS" ]; then
    echo -e "  Overall:      ${GREEN}✓ PASSED${NC}"
elif [ "$OVERALL_STATUS" = "PASS_WITH_SKIPS" ]; then
    echo -e "  Overall:      ${YELLOW}✓ PASSED (some checks skipped)${NC}"
else
    echo -e "  Overall:      ${RED}✗ FAILED${NC}"
fi

# Every check the summary records, rather than the four this script used to name.
#
# The gate records eight. Naming them meant that a run red on shell lint,
# workflow lint or any of the three documentation checks showed four passing
# rows and nothing else — an all-clear, from the tool whose entire job is to say
# what broke. All eight are blocking (compute_overall_status fails on any of
# them), so there is no such thing here as a check not worth displaying, and a
# check added to the gate tomorrow appears without this file being edited.
echo -e "\n${BOLD}Check Results:${NC}"
while IFS="$(printf '\037')" read -r kind name status skipped exit_code tool label; do
    [ "$kind" = "CHECK" ] || continue
    : "$name" "$tool"
    if [ "$skipped" = "true" ]; then
        printf '  %-24s' "${label}:"
        echo -e "${DIM}– skipped${NC}"
    elif [ "$status" = "pass" ]; then
        printf '  %-24s' "${label}:"
        echo -e "${GREEN}✓${NC}"
    else
        printf '  %-24s' "${label}:"
        echo -e "${RED}✗ (exit: ${exit_code:-?})${NC}"
    fi
done <<< "$SUMMARY_PROJECTION"
[ "$STYLE_STATUS" = "pass" ] \
    && echo -e "  ${DIM}Style findings:         none${NC}" \
    || echo -e "  ${DIM}Style findings:         ${STYLE_COUNT} (in style-check.json)${NC}"

# If everything passed, exit early
if ([ "$OVERALL_STATUS" = "PASS" ] || [ "$OVERALL_STATUS" = "PASS_WITH_SKIPS" ]) && [ "$SHOW_COVERAGE" = false ]; then
    echo -e "\n${GREEN}✓ All checks passed! No failures to diagnose.${NC}"
    exit 0
fi

# Analyze failures based on focus mode
if [ "$FOCUS_MODE" = "" ] || [ "$FOCUS_MODE" = "tests" ]; then
    if [ "$UNIT_STATUS" != "pass" ] || [ "$INT_STATUS" != "pass" ]; then
        print_header "Test Failures"
        
        # Check for unit test failures
        if [ "$UNIT_STATUS" != "pass" ] && [ "$SHOW_TESTS" = true ]; then
            if [ -f "$SOURCE_DIR/unit-test-output.txt" ]; then
                analyze_test_failures "Unit" "$SOURCE_DIR/unit-test-output.txt"
            fi
            
            # Check individual package outputs
            for output_file in "$SOURCE_DIR"/unit-test-output-*.txt; do
                if [ -f "$output_file" ]; then
                    pkg_name=$(basename "$output_file" | sed 's/unit-test-output-\(.*\)\.txt/\1/')
                    analyze_test_failures "Unit ($pkg_name)" "$output_file"
                fi
            done
        fi
        
        # Check for integration test failures
        if [ "$INT_STATUS" != "pass" ] && [ "$SHOW_TESTS" = true ]; then
            if [ -f "$SOURCE_DIR/integration-test-output.txt" ]; then
                analyze_test_failures "Integration" "$SOURCE_DIR/integration-test-output.txt"
            fi
            
            # Check individual package outputs
            for output_file in "$SOURCE_DIR"/integration-test-output-*.txt; do
                if [ -f "$output_file" ]; then
                    pkg_name=$(basename "$output_file" | sed 's/integration-test-output-\(.*\)\.txt/\1/')
                    analyze_test_failures "Integration ($pkg_name)" "$output_file"
                fi
            done
        fi
    fi
fi

if [ "$FOCUS_MODE" = "" ] || [ "$FOCUS_MODE" = "lint" ]; then
    if [ "$LINT_STATUS" != "pass" ] && [ "$SHOW_LINT" = true ]; then
        print_header "Code Quality Issues"
        analyze_lint_issues
    fi
    
    if [ "$STYLE_STATUS" != "pass" ] && [ "$SHOW_STYLE" = true ]; then
        if [ "$LINT_STATUS" = "pass" ] || [ "$SHOW_LINT" = false ]; then
            print_header "Code Quality Issues"
        fi
        analyze_style_issues
    fi
fi

# Show coverage if requested
if [ "$SHOW_COVERAGE" = true ]; then
    print_header "Coverage Report"
    analyze_coverage
fi

# Show actionable summary
print_header "Next Steps"

if [ "$OVERALL_STATUS" != "PASS" ] && [ "$OVERALL_STATUS" != "PASS_WITH_SKIPS" ]; then
    echo -e "${YELLOW}To fix these issues:${NC}"
    
    priority=1
    
    # Prioritize test failures
    if [ "$UNIT_STATUS" != "pass" ] || [ "$INT_STATUS" != "pass" ]; then
        echo -e "  ${BOLD}$priority.${NC} Fix failing tests:"
        echo "       ./bin/test.sh -- --lf     # Re-run only failed tests"
        ((priority++))
    fi
    
    # Style issues can be auto-fixed
    if [ "$STYLE_STATUS" != "pass" ]; then
        echo -e "  ${BOLD}$priority.${NC} Auto-fix style issues:"
        echo "       ./bin/fix.sh"
        ((priority++))
    fi
    
    # Linting requires manual fixes
    if [ -n "$LINT_STATUS" ] && [ "$LINT_STATUS" != "pass" ]; then
        echo -e "  ${BOLD}$priority.${NC} Fix linting issues:"
        echo "       ./bin/dk lint"
        ((priority++))
    fi

    # Everything else that failed, named by the check that failed and the tool
    # that produced it.
    #
    # The three lines above are the whole of what this section used to print, so
    # a run red on shell lint, workflow lint or a documentation check reached
    # here and printed a heading with nothing under it — and then told the
    # developer to re-run the gate, which would fail the same way for the same
    # unnamed reason. The `-n` guard on the linting branch is the other half of
    # that: an absent `validation` entry read as empty, which is not "pass", so
    # this recommended fixing lint findings for a run that had not linted.
    while IFS="$(printf '\037')" read -r kind name status skipped exit_code tool label; do
        [ "$kind" = "CHECK" ] || continue
        if [ "$status" = "pass" ] || [ "$skipped" = "true" ]; then
            continue
        fi
        case "$name" in
            unit_tests|integration_tests|validation) continue ;;
        esac
        : "$exit_code"
        echo -e "  ${BOLD}$priority.${NC} ${label} failed:"
        if [ -n "$tool" ] && [ -x "$PROJECT_ROOT/bin/$tool" ]; then
            echo "       ./bin/$tool"
        else
            echo "       see the ${label} output in $SOURCE_DIR"
        fi
        ((priority++))
    done <<< "$SUMMARY_PROJECTION"

    echo -e "\n${CYAN}After fixing, re-run quality checks:${NC}"
    echo "    ./bin/dk pr"
else
    echo -e "${GREEN}✓ All recorded checks passed.${NC}"

    # Style findings do not fail the gate on their own — ruff's verdict reaches
    # it through `validation` — so they are worth mentioning on a green run and
    # are not worth blocking one. The branch this replaces tested both statuses
    # against "warning", a value the producer has never written, so it could not
    # run: two dead lines whose subject was a real thing worth saying.
    if [ "$STYLE_STATUS" != "pass" ]; then
        echo -e "\n${YELLOW}Minor issues to consider:${NC}"
        echo "  - ${STYLE_COUNT} style findings (non-blocking; ./bin/fix.sh)"
    fi
fi

echo ""
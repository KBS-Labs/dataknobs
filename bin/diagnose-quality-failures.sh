#!/bin/bash
set -e

# Quality Failure Diagnostic Script for DataKnobs
# Analyzes .quality-artifacts from prior PR execution to pinpoint failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/.quality-artifacts"

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

Analyze .quality-artifacts from the most recent PR quality check run
to pinpoint specific failures and provide actionable fixes.

${YELLOW}Options:${NC}
    -v, --verbose       Show detailed output for all issues
    -s, --summary       Show only the summary (default)
    -t, --tests         Focus on test failures
    -l, --lint          Focus on linting issues
    -c, --coverage      Show coverage details
    -f, --fix           Show auto-fix commands where available
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
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Check if artifacts directory exists
if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo -e "${RED}✗ No quality artifacts found!${NC}"
    echo -e "${YELLOW}  Run './bin/run-quality-checks.sh' first to generate artifacts.${NC}"
    exit 1
fi

# Check if summary exists
if [ ! -f "$ARTIFACTS_DIR/quality-summary.json" ]; then
    echo -e "${RED}✗ No quality summary found!${NC}"
    echo -e "${YELLOW}  The quality check may not have completed. Re-run './bin/run-quality-checks.sh'.${NC}"
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
analyze_lint_issues() {
    if [ ! -f "$ARTIFACTS_DIR/lint-report.json" ]; then
        return
    fi
    
    # Try to parse JSON - check if jq is available
    if command -v jq &> /dev/null; then
        local issue_count=$(jq 'length' "$ARTIFACTS_DIR/lint-report.json" 2>/dev/null || echo "0")
        
        if [ "$issue_count" -gt 0 ]; then
            echo -e "\n${YELLOW}Linting Issues:${NC} $issue_count issues found"
            
            if [ "$VERBOSE" = true ]; then
                echo -e "${DIM}────────────────────────────────────────${NC}"
                # Group by file
                jq -r 'group_by(.path) | .[] | "\n\(.[]|.path):\n" + (map("  Line \(.line): \(.message) [\(.message-id)]") | join("\n"))' \
                    "$ARTIFACTS_DIR/lint-report.json" 2>/dev/null | head -50
            else
                # Show summary by message type
                echo -e "${DIM}  Issue types:${NC}"
                jq -r '[.[].message-id] | group_by(.) | map({type: .[0], count: length}) | .[] | "    \(.type): \(.count)"' \
                    "$ARTIFACTS_DIR/lint-report.json" 2>/dev/null | sort -t: -k2 -rn | head -10
                
                echo -e "${DIM}  Most affected files:${NC}"
                jq -r '[.[].path] | group_by(.) | map({file: .[0], count: length}) | sort_by(.count) | reverse | .[] | "    \(.file): \(.count) issues"' \
                    "$ARTIFACTS_DIR/lint-report.json" 2>/dev/null | head -5
            fi
        fi
    else
        # Fallback without jq
        echo -e "\n${YELLOW}Linting Report:${NC} (install 'jq' for better formatting)"
        python3 -c "
import json
with open('$ARTIFACTS_DIR/lint-report.json') as f:
    data = json.load(f)
    if data:
        print(f'  {len(data)} issues found')
        for item in data[:5]:
            print(f\"  {item.get('path', 'unknown')}:{item.get('line', '?')}: {item.get('message', 'no message')}\")
        if len(data) > 5:
            print(f'  ... and {len(data)-5} more')
" 2>/dev/null || echo "  Could not parse lint report"
    fi
    
    if [ "$SHOW_FIXES" = true ]; then
        echo -e "\n${GREEN}To see all linting issues:${NC}"
        echo "    uv run pylint packages/*/src --rcfile=.pylintrc"
    fi
}

# Function to analyze style issues
analyze_style_issues() {
    if [ ! -f "$ARTIFACTS_DIR/style-check.json" ]; then
        return
    fi
    
    # Try to parse JSON
    if command -v jq &> /dev/null; then
        local issue_count=$(jq 'length' "$ARTIFACTS_DIR/style-check.json" 2>/dev/null || echo "0")
        
        if [ "$issue_count" -gt 0 ]; then
            echo -e "\n${YELLOW}Style Issues:${NC} $issue_count violations found"
            
            if [ "$VERBOSE" = true ]; then
                echo -e "${DIM}────────────────────────────────────────${NC}"
                jq -r '.[] | "  \(.filename):\(.location.row): \(.message) [\(.code)]"' \
                    "$ARTIFACTS_DIR/style-check.json" 2>/dev/null | head -20
            else
                # Show summary by violation code
                echo -e "${DIM}  Violation types:${NC}"
                jq -r '[.[].code] | group_by(.) | map({code: .[0], count: length}) | .[] | "    \(.code): \(.count)"' \
                    "$ARTIFACTS_DIR/style-check.json" 2>/dev/null | sort -t: -k2 -rn | head -10
            fi
        fi
    else
        # Fallback without jq
        echo -e "\n${YELLOW}Style Report:${NC} (install 'jq' for better formatting)"
        python3 -c "
import json
with open('$ARTIFACTS_DIR/style-check.json') as f:
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
        echo -e "\n${GREEN}To auto-fix style issues:${NC}"
        echo "    uv run ruff check --fix packages/*/src"
        echo -e "${GREEN}To see all style issues:${NC}"
        echo "    uv run ruff check packages/*/src"
    fi
}

# Function to show coverage details
analyze_coverage() {
    if [ ! -f "$ARTIFACTS_DIR/coverage-by-package.json" ]; then
        return
    fi
    
    echo -e "\n${CYAN}Test Coverage by Package:${NC}"
    echo -e "${DIM}────────────────────────────────────────${NC}"
    
    if command -v jq &> /dev/null; then
        jq -r '.packages | to_entries | .[] | "  \(.key): \(.value.coverage) (\(.value.statements - .value.missing)/\(.value.statements) statements)"' \
            "$ARTIFACTS_DIR/coverage-by-package.json" 2>/dev/null
    else
        python3 -c "
import json
with open('$ARTIFACTS_DIR/coverage-by-package.json') as f:
    data = json.load(f)
    for pkg, info in data.get('packages', {}).items():
        covered = info['statements'] - info['missing']
        print(f\"  {pkg}: {info['coverage']} ({covered}/{info['statements']} statements)\")
" 2>/dev/null || echo "  Could not parse coverage report"
    fi
    
    if [ -f "$ARTIFACTS_DIR/htmlcov/index.html" ]; then
        echo -e "\n${GREEN}View detailed coverage report:${NC}"
        echo "    open $ARTIFACTS_DIR/htmlcov/index.html"
    fi
}

# Main diagnostic flow
echo -e "${BOLD}${CYAN}DataKnobs Quality Diagnostics${NC}"
echo -e "${DIM}Analyzing artifacts from $(date -r "$ARTIFACTS_DIR/quality-summary.json" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'recent run')${NC}"

# Read the summary
if command -v jq &> /dev/null; then
    OVERALL_STATUS=$(jq -r '.overall_status' "$ARTIFACTS_DIR/quality-summary.json")
    TIMESTAMP=$(jq -r '.timestamp' "$ARTIFACTS_DIR/quality-summary.json")
    ENVIRONMENT=$(jq -r '.environment' "$ARTIFACTS_DIR/quality-summary.json")
    PACKAGES=$(jq -r '.packages' "$ARTIFACTS_DIR/quality-summary.json")
    
    # Get individual check statuses
    LINT_STATUS=$(jq -r '.checks.lint.status' "$ARTIFACTS_DIR/quality-summary.json")
    LINT_CODE=$(jq -r '.checks.lint.exit_code' "$ARTIFACTS_DIR/quality-summary.json")
    STYLE_STATUS=$(jq -r '.checks.style.status' "$ARTIFACTS_DIR/quality-summary.json")
    UNIT_STATUS=$(jq -r '.checks.unit_tests.status' "$ARTIFACTS_DIR/quality-summary.json")
    UNIT_CODE=$(jq -r '.checks.unit_tests.exit_code' "$ARTIFACTS_DIR/quality-summary.json")
    INT_STATUS=$(jq -r '.checks.integration_tests.status' "$ARTIFACTS_DIR/quality-summary.json")
    INT_CODE=$(jq -r '.checks.integration_tests.exit_code' "$ARTIFACTS_DIR/quality-summary.json")
else
    # Fallback parsing without jq
    OVERALL_STATUS=$(grep '"overall_status"' "$ARTIFACTS_DIR/quality-summary.json" | cut -d'"' -f4)
    TIMESTAMP=$(grep '"timestamp"' "$ARTIFACTS_DIR/quality-summary.json" | cut -d'"' -f4)
    echo -e "${YELLOW}Note: Install 'jq' for better JSON parsing${NC}"
fi

# Show summary header
print_header "Quality Check Summary"
echo -e "  Timestamp:    ${TIMESTAMP}"
echo -e "  Environment:  ${ENVIRONMENT}"
echo -e "  Packages:     ${PACKAGES}"
echo -e "  Overall:      $([ "$OVERALL_STATUS" = "PASS" ] && echo -e "${GREEN}✓ PASSED${NC}" || echo -e "${RED}✗ FAILED${NC}")"

# Show individual check results
echo -e "\n${BOLD}Check Results:${NC}"
[ "$LINT_STATUS" = "pass" ] && echo -e "  Linting:      ${GREEN}✓${NC}" || echo -e "  Linting:      ${YELLOW}⚠ (exit: $LINT_CODE)${NC}"
[ "$STYLE_STATUS" = "pass" ] && echo -e "  Style:        ${GREEN}✓${NC}" || echo -e "  Style:        ${YELLOW}⚠${NC}"
[ "$UNIT_STATUS" = "pass" ] && echo -e "  Unit Tests:   ${GREEN}✓${NC}" || echo -e "  Unit Tests:   ${RED}✗ (exit: $UNIT_CODE)${NC}"
[ "$INT_STATUS" = "pass" ] && echo -e "  Integration:  ${GREEN}✓${NC}" || echo -e "  Integration:  ${RED}✗ (exit: $INT_CODE)${NC}"

# If everything passed, exit early
if [ "$OVERALL_STATUS" = "PASS" ] && [ "$SHOW_COVERAGE" = false ]; then
    echo -e "\n${GREEN}✓ All checks passed! No failures to diagnose.${NC}"
    exit 0
fi

# Analyze failures based on focus mode
if [ "$FOCUS_MODE" = "" ] || [ "$FOCUS_MODE" = "tests" ]; then
    if [ "$UNIT_STATUS" != "pass" ] || [ "$INT_STATUS" != "pass" ]; then
        print_header "Test Failures"
        
        # Check for unit test failures
        if [ "$UNIT_STATUS" != "pass" ] && [ "$SHOW_TESTS" = true ]; then
            if [ -f "$ARTIFACTS_DIR/unit-test-output.txt" ]; then
                analyze_test_failures "Unit" "$ARTIFACTS_DIR/unit-test-output.txt"
            fi
            
            # Check individual package outputs
            for output_file in "$ARTIFACTS_DIR"/unit-test-output-*.txt; do
                if [ -f "$output_file" ]; then
                    pkg_name=$(basename "$output_file" | sed 's/unit-test-output-\(.*\)\.txt/\1/')
                    analyze_test_failures "Unit ($pkg_name)" "$output_file"
                fi
            done
        fi
        
        # Check for integration test failures
        if [ "$INT_STATUS" != "pass" ] && [ "$SHOW_TESTS" = true ]; then
            if [ -f "$ARTIFACTS_DIR/integration-test-output.txt" ]; then
                analyze_test_failures "Integration" "$ARTIFACTS_DIR/integration-test-output.txt"
            fi
            
            # Check individual package outputs
            for output_file in "$ARTIFACTS_DIR"/integration-test-output-*.txt; do
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

if [ "$OVERALL_STATUS" != "PASS" ]; then
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
        echo "       uv run ruff check --fix packages/*/src"
        ((priority++))
    fi
    
    # Linting requires manual fixes
    if [ "$LINT_STATUS" != "pass" ] && [ "$LINT_STATUS" != "warning" ]; then
        echo -e "  ${BOLD}$priority.${NC} Fix linting issues:"
        echo "       uv run pylint packages/*/src --rcfile=.pylintrc"
        ((priority++))
    fi
    
    echo -e "\n${CYAN}After fixing, re-run quality checks:${NC}"
    echo "    ./bin/run-quality-checks.sh"
else
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    
    if [ "$LINT_STATUS" = "warning" ] || [ "$STYLE_STATUS" = "warning" ]; then
        echo -e "\n${YELLOW}Minor issues to consider:${NC}"
        [ "$LINT_STATUS" = "warning" ] && echo "  - Some linting warnings (non-blocking)"
        [ "$STYLE_STATUS" = "warning" ] && echo "  - Some style issues (non-blocking)"
    fi
fi

echo ""
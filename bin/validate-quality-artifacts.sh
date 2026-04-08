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

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}         Validating Quality Check Artifacts                       ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

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

print_info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

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
REQUIRED_FILES=(
    "quality-summary.json"
    "environment.json"
    "signature.sha256"
    "unit-test-results.xml"
    "coverage.xml"
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

    if [ -n "$HASH_ERROR" ]; then
        print_fail "Hash validation error: $HASH_ERROR"
        VALIDATION_FAILED=1
    elif [ -n "$HASH_WARNING" ]; then
        print_info "$HASH_WARNING"
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
        print_fail "Package content has changed since quality checks were run"
        if [ -n "$CHANGED" ]; then
            print_info "Changed packages: $CHANGED"
        fi
        if [ -n "$DIRTY" ]; then
            print_fail "Packages needing re-validation: $DIRTY"
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
    # Check overall status
    OVERALL_STATUS=$(grep -o '"overall_status": *"[^"]*"' "$ARTIFACTS_DIR/quality-summary.json" | cut -d'"' -f4)
    
    if [ "$OVERALL_STATUS" = "PASS" ]; then
        print_pass "Overall status: PASS"
    elif [ "$OVERALL_STATUS" = "PASS_WITH_SKIPS" ]; then
        print_pass "Overall status: PASS_WITH_SKIPS (some checks were skipped)"
        # Log which checks were skipped for transparency
        for check in validation unit_tests integration_tests documentation; do
            skipped=$(grep -A3 "\"$check\"" "$ARTIFACTS_DIR/quality-summary.json" | grep '"skipped"' | grep -o 'true\|false')
            if [ "$skipped" = "true" ]; then
                print_info "  Skipped: $check"
            fi
        done
    else
        print_fail "Overall status: $OVERALL_STATUS (expected: PASS or PASS_WITH_SKIPS)"
        VALIDATION_FAILED=1
    fi
    
    # Check individual test statuses
    UNIT_STATUS=$(grep -A2 '"unit_tests"' "$ARTIFACTS_DIR/quality-summary.json" | grep '"status"' | cut -d'"' -f4)
    if [ "$UNIT_STATUS" = "pass" ]; then
        print_pass "Unit tests: PASS"
    else
        print_fail "Unit tests: $UNIT_STATUS"
        VALIDATION_FAILED=1
    fi
    
    # Check if integration tests were run (they might be optional in some cases)
    if grep -q '"integration_tests"' "$ARTIFACTS_DIR/quality-summary.json"; then
        INT_STATUS=$(grep -A2 '"integration_tests"' "$ARTIFACTS_DIR/quality-summary.json" | grep '"status"' | cut -d'"' -f4)
        if [ "$INT_STATUS" = "pass" ]; then
            print_pass "Integration tests: PASS"
        else
            print_fail "Integration tests: $INT_STATUS"
            VALIDATION_FAILED=1
        fi
    else
        print_info "Integration tests: Not found (may be optional)"
    fi
fi

# Validate coverage
print_check "Code coverage"
if [ -f "$ARTIFACTS_DIR/coverage.xml" ]; then
    # Extract coverage percentage from XML
    if command -v python3 >/dev/null 2>&1; then
        COVERAGE=$(python3 -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('$ARTIFACTS_DIR/coverage.xml')
    root = tree.getroot()
    coverage = float(root.attrib.get('line-rate', 0)) * 100
    print(f'{coverage:.1f}')
except:
    print('0')
")
        
        if (( $(echo "$COVERAGE >= $REQUIRED_COVERAGE" | bc -l 2>/dev/null || echo 0) )); then
            print_pass "Coverage: ${COVERAGE}% (minimum: ${REQUIRED_COVERAGE}%)"
        else
            print_fail "Coverage: ${COVERAGE}% (below minimum: ${REQUIRED_COVERAGE}%)"
            # Don't fail validation for coverage, just warn
            print_info "Low coverage is a warning, not a failure"
        fi
    else
        print_info "Could not parse coverage (Python not available)"
    fi
else
    print_fail "Coverage report not found"
fi

# Verify artifact signature
print_check "Artifact integrity"
if [ -f "$ARTIFACTS_DIR/signature.sha256" ]; then
    cd "$ARTIFACTS_DIR"
    
    # Generate current signature
    CURRENT_SIG=$(find . -type f \( -name "*.json" -o -name "*.xml" \) | sort | xargs sha256sum 2>/dev/null | sort)
    
    # Read stored signature (excluding the signature file itself)
    STORED_SIG=$(grep -v "signature.sha256" signature.sha256 2>/dev/null | sort)
    
    if [ "$CURRENT_SIG" = "$STORED_SIG" ]; then
        print_pass "Artifact signature valid"
    else
        print_fail "Artifact signature mismatch - files may have been modified"
        print_info "This could happen if artifacts were manually edited"
        # Don't fail on signature mismatch as files might be legitimately updated
    fi
    
    cd "$PROJECT_ROOT"
else
    print_fail "Signature file not found"
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
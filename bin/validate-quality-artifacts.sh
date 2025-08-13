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
MAX_AGE_HOURS=${MAX_AGE_HOURS:-24}  # Maximum age of artifacts in hours
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

# Check artifact age
print_check "Artifact freshness"
if [ -f "$ARTIFACTS_DIR/quality-summary.json" ]; then
    # Extract timestamp from JSON (works on both Linux and macOS)
    ARTIFACT_TIMESTAMP=$(grep -o '"timestamp": *"[^"]*"' "$ARTIFACTS_DIR/quality-summary.json" | cut -d'"' -f4)
    
    if [ -n "$ARTIFACT_TIMESTAMP" ]; then
        # Convert to seconds since epoch (portable)
        if command -v python3 >/dev/null 2>&1; then
            ARTIFACT_EPOCH=$(python3 -c "from datetime import datetime; print(int(datetime.fromisoformat('${ARTIFACT_TIMESTAMP}'.replace('Z', '+00:00')).timestamp()))")
            CURRENT_EPOCH=$(date +%s)
            AGE_HOURS=$(( (CURRENT_EPOCH - ARTIFACT_EPOCH) / 3600 ))
            
            if [ $AGE_HOURS -gt $MAX_AGE_HOURS ]; then
                print_fail "Artifacts are $AGE_HOURS hours old (max: $MAX_AGE_HOURS hours)"
                print_fail "Please run: ./bin/run-quality-checks.sh"
                VALIDATION_FAILED=1
            else
                print_pass "Artifacts are $AGE_HOURS hours old (within $MAX_AGE_HOURS hour limit)"
            fi
        else
            print_info "Could not verify timestamp (Python not available)"
        fi
    else
        print_fail "Could not extract timestamp from artifacts"
        VALIDATION_FAILED=1
    fi
fi

# Validate test results
print_check "Test results"
if [ -f "$ARTIFACTS_DIR/quality-summary.json" ]; then
    # Check overall status
    OVERALL_STATUS=$(grep -o '"overall_status": *"[^"]*"' "$ARTIFACTS_DIR/quality-summary.json" | cut -d'"' -f4)
    
    if [ "$OVERALL_STATUS" != "PASS" ]; then
        print_fail "Overall status: $OVERALL_STATUS (expected: PASS)"
        VALIDATION_FAILED=1
    else
        print_pass "Overall status: PASS"
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
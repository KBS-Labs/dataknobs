#!/bin/bash
set -e

# Quality Checks Script for DataKnobs
# This script runs all quality checks including unit tests, integration tests,
# linting, and code coverage. Results are saved as artifacts for CI validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/.quality-artifacts"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Default values
PACKAGES=""
SKIP_STYLE="no"
SKIP_TESTS="no"
# Skips the per-package suites while leaving the workspace guards running.
# Distinct from SKIP_TESTS because "no package changed" and "nothing that
# affects a quality result changed" are different answers: the guards under
# tests/ belong to no package, so a change to them produces the first and used
# to be read as the second — skipping the suite the change edited.
SKIP_PACKAGE_TESTS="no"
PYTEST_ARGS=""
KEEP_SERVICES="false"
PR_MODE="auto"  # auto, yes, no
RUN_MODE=""     # pr, all, full (set after argument parsing)
BASE_REF="main" # Git ref for changed-package detection

# Check if we're inside a Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER:-}" ]; then
    IN_DOCKER=true
fi

# Colors for output (check if terminal supports colors)
if [ -t 1 ] && [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# Function to show usage
show_usage() {
    cat << EOF
${CYAN}DataKnobs Quality Checks Runner${NC}

Usage: $0 [OPTIONS] [PACKAGE...] [-- PYTEST_ARGS]

Run quality checks (linting, style, tests) for DataKnobs packages.

${YELLOW}Options:${NC}
    -p, --package PACKAGE    Package to check (can be specified multiple times)
                            If not specified, checks all packages
    --pr                    PR mode (default): Only test changed packages + dependents.
                            Uses parallel execution, quiet output, XML-only coverage.
                            Skips docs build if no docs changed.
    --all                   All mode: Test all packages with parallel execution
                            and optimized coverage (no HTML reports)
    --full                  Full mode: Legacy behavior — all packages, sequential,
                            verbose output, all coverage reports (HTML + XML + term)
    --dev                   Dev mode: Run quick checks without artifacts
                            (combined tests, no artifact pollution)
    --base-ref REF          Git ref for change detection (default: main)
    --skip-style            Skip code validation (syntax, ruff, imports, mypy)
    --skip-tests            Skip test execution
    --keep-services         Keep services running after completion
    -h, --help              Show this help message

${YELLOW}Advanced Usage:${NC}
    Any arguments after -- are passed directly to pytest:
    $0 data -- -xvs --tb=short

${YELLOW}Examples:${NC}
    $0 --pr                 # PR mode: Only changed packages (default)
    $0 --all                # All packages, parallel, optimized
    $0 --full               # Legacy: all packages, sequential, verbose
    $0 --dev data           # Dev mode: Quick checks for data package
    $0 data config          # Dev mode: Check specific packages (no artifacts)
    $0 --pr data            # PR mode for data package only
    $0 --skip-style         # Run all checks except style checks
    $0 data -- -x           # Run data package with pytest -x flag

${YELLOW}Environment:${NC}
    Running in: $([ "$IN_DOCKER" = true ] && echo "Docker container" || echo "Host system")
    
${YELLOW}Output:${NC}
    All artifacts are saved to: .quality-artifacts/
    - environment.json: System information
    - validation.log: Code validation results (syntax, ruff, imports, mypy)
    - style-check.json: Ruff findings (JSON format)
    - *-test-results.xml: Test results in JUnit format
    - coverage*.xml: Coverage reports
    - quality-summary.json: Overall summary

EOF
    exit 0
}

# Function to print status
print_status() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--package)
            PACKAGES="$PACKAGES $2"
            shift 2
            ;;
        --pr)
            PR_MODE="yes"
            RUN_MODE="pr"
            shift
            ;;
        --all)
            PR_MODE="yes"
            RUN_MODE="all"
            shift
            ;;
        --full)
            PR_MODE="yes"
            RUN_MODE="full"
            shift
            ;;
        --base-ref)
            BASE_REF="$2"
            shift 2
            ;;
        --dev)
            PR_MODE="no"
            shift
            ;;
        --skip-style)
            SKIP_STYLE="yes"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="yes"
            shift
            ;;
        --keep-services)
            KEEP_SERVICES="true"
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        --)
            shift
            # $* not $@: the target is a string. Both join the same way in a
            # scalar assignment — "$@" does not truncate, which an earlier note
            # here claimed — so this is about saying what is meant rather than
            # about behaviour. It is re-split at the pytest call sites, which is
            # why they carry SC2086 waivers.
            PYTEST_ARGS="$*"
            break
            ;;
        *)
            # Assume it's a package name
            PACKAGES="$PACKAGES $1"
            shift
            ;;
    esac
done

# Auto-detect mode if not explicitly set
if [ "$PR_MODE" = "auto" ]; then
    if [ -n "$PACKAGES" ]; then
        # If specific packages are selected, assume dev mode
        PR_MODE="no"
    else
        # If running all packages, assume PR mode
        PR_MODE="yes"
    fi
fi

# Default RUN_MODE based on PR_MODE
if [ -z "$RUN_MODE" ]; then
    if [ "$PR_MODE" = "yes" ]; then
        RUN_MODE="pr"
    else
        RUN_MODE="dev"
    fi
fi

# Changed-package detection (pr mode only, when no explicit packages given)
DOCS_CHANGED="true"
TESTED_PACKAGES_JSON="[]"
if [ "$RUN_MODE" = "pr" ] && [ -z "$PACKAGES" ]; then
    print_status "Detecting changed packages..."
    CHANGED_INFO=$(uv run python "$SCRIPT_DIR/changed-packages.py" --base-ref "$BASE_REF" 2>/dev/null) || {
        print_warning "Change detection failed — testing all packages"
        CHANGED_INFO=""
    }

    if [ -n "$CHANGED_INFO" ]; then
        # Parse every field with the SAME interpreter that produced the JSON
        # (uv-managed Python), NOT a bare `python3` that may resolve to a
        # broken pyenv shim. A parse failure here is a real error — the JSON
        # was just emitted by changed-packages.py and is well formed by
        # construction — so abort loudly. The previous behavior swallowed the
        # error (`2>/dev/null || echo ""`) and left CHANGED_PACKAGES empty,
        # which silently set SKIP_TESTS=yes and skipped the ENTIRE test suite
        # while still reporting success.
        if ! CHANGED_PARSED=$(printf '%s' "$CHANGED_INFO" | uv run python -c '
import json
import sys

data = json.load(sys.stdin)
print(" ".join(data["packages"]))
print(str(data["docs_changed"]).lower())
print(data.get("mode", "all"))
print(json.dumps(data["packages"]))
# Default "packages", not "none": an older detector that does not emit this
# field must fall back to running something, never to running nothing.
print(data.get("test_scope", "packages"))
'); then
            print_error "Failed to parse change-detection output; aborting so"
            print_error "the test suite is not silently skipped. Raw output:"
            printf '%s\n' "$CHANGED_INFO" >&2
            exit 1
        fi
        _changed_fields=()
        while IFS= read -r _line; do
            _changed_fields+=("$_line")
        done <<< "$CHANGED_PARSED"
        CHANGED_PACKAGES="${_changed_fields[0]}"
        DOCS_CHANGED="${_changed_fields[1]}"
        CHANGE_MODE="${_changed_fields[2]}"
        TESTED_PACKAGES_JSON="${_changed_fields[3]}"
        TEST_SCOPE="${_changed_fields[4]}"

        # Three answers, not two. An empty package list is correct for a change
        # to the workspace guards — they belong to no package — but reading it
        # as "nothing to test" skipped the suite that change edited, and the
        # run reported success. changed-packages.py names the cases apart.
        case "$TEST_SCOPE" in
            packages)
                PACKAGES="$CHANGED_PACKAGES"
                print_success "Changed packages: $PACKAGES"
                if [ "$CHANGE_MODE" = "all" ]; then
                    print_status "Global files changed — testing all packages"
                fi
                ;;
            workspace)
                print_success "No package changes detected — running workspace guards only"
                SKIP_PACKAGE_TESTS="yes"
                ;;
            *)
                print_success "No package or workspace changes detected — skipping tests"
                SKIP_TESTS="yes"
                ;;
        esac
    fi

    if [ "$DOCS_CHANGED" = "true" ]; then
        print_status "Documentation changes detected"
    else
        print_status "No documentation changes detected — skipping docs build"
    fi
fi

# Determine test runner flags based on mode
TEST_PARALLEL_FLAG=""
TEST_VERBOSITY_FLAG=""
TEST_COV_REPORT="xml"
case "$RUN_MODE" in
    pr)
        TEST_PARALLEL_FLAG="--parallel"
        TEST_VERBOSITY_FLAG="--quiet"
        TEST_COV_REPORT="xml"
        ;;
    all)
        TEST_PARALLEL_FLAG="--parallel"
        TEST_VERBOSITY_FLAG=""
        TEST_COV_REPORT="xml"
        ;;
    full)
        TEST_PARALLEL_FLAG="--no-parallel"
        TEST_VERBOSITY_FLAG="--verbose"
        TEST_COV_REPORT="term-missing,html,xml"
        ;;
    dev)
        TEST_PARALLEL_FLAG=""
        TEST_VERBOSITY_FLAG=""
        TEST_COV_REPORT="term-missing"
        ;;
esac

# Function to set environment variables based on context
set_environment_vars() {
    if [ "$IN_DOCKER" = true ]; then
        # Inside Docker container - use service names
        export DATABASE_URL="postgresql://postgres:postgres@postgres:5432/dataknobs"
        export ELASTICSEARCH_URL="http://elasticsearch:9200"
        export AWS_ENDPOINT_URL="http://localstack:4566"
        export LOCALSTACK_ENDPOINT="http://localstack:4566"
        
        # Individual host/port variables for test fixtures
        export POSTGRES_HOST="postgres"
        export POSTGRES_PORT=5432
        export POSTGRES_USER="postgres"
        export POSTGRES_PASSWORD="postgres"
        export POSTGRES_DB="dataknobs_test"
        
        export ELASTICSEARCH_HOST="elasticsearch"
        export ELASTICSEARCH_PORT=9200
    else
        # On host system - use localhost
        export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/dataknobs"
        export ELASTICSEARCH_URL="http://localhost:9200"
        export AWS_ENDPOINT_URL="http://localhost:4566"
        export LOCALSTACK_ENDPOINT="http://localhost:4566"
        
        # Individual host/port variables for test fixtures
        export POSTGRES_HOST="localhost"
        export POSTGRES_PORT=5432
        export POSTGRES_USER="postgres"
        export POSTGRES_PASSWORD="postgres"
        export POSTGRES_DB="dataknobs_test"
        
        export ELASTICSEARCH_HOST="localhost"
        export ELASTICSEARCH_PORT=9200
    fi
    
    # AWS credentials for LocalStack (same regardless of environment)
    export AWS_ACCESS_KEY_ID="test"
    export AWS_SECRET_ACCESS_KEY="test"
    export AWS_DEFAULT_REGION="us-east-1"
    export S3_BUCKET="dataknobs-local"
    
    # Enable test flags for integration tests
    export TEST_S3="true"
    export TEST_ELASTICSEARCH="true"
    export TEST_POSTGRES="true"
    export TEST_OLLAMA="true"
}

# Function to cleanup resources
#
# Reached only through `trap cleanup EXIT INT TERM` below, which shellcheck does
# not read as a call site. Deleting it on the strength of the finding would
# leave the service teardown unreferenced *and* the trap naming nothing.
# shellcheck disable=SC2329
cleanup() {
    if [ "$IN_DOCKER" = false ]; then
        # Only cleanup if manage-services.sh indicates we started them
        if [ -f "/tmp/.dataknobs_services_started_$$" ]; then
            if [ "${KEEP_SERVICES}" != "true" ]; then
                echo ""
                print_status "Cleaning up services..."
                "$SCRIPT_DIR/manage-services.sh" stop
            else
                echo ""
                print_status "Services are still running. To stop them, run:"
                echo "$SCRIPT_DIR/manage-services.sh stop"
            fi
        fi
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
case "$RUN_MODE" in
    pr)   echo -e "${BLUE}       DataKnobs Quality Checks - PR Mode (changed packages)     ${NC}" ;;
    all)  echo -e "${BLUE}       DataKnobs Quality Checks - All Packages                   ${NC}" ;;
    full) echo -e "${BLUE}       DataKnobs Quality Checks - Full Mode (legacy)             ${NC}" ;;
    *)    echo -e "${BLUE}       DataKnobs Quality Checks - Developer Mode                 ${NC}" ;;
esac
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Display environment
if [ "$IN_DOCKER" = true ]; then
    print_status "Environment: Docker Container"
else
    print_status "Environment: Host System"
fi

# Display mode and packages
case "$RUN_MODE" in
    pr)   print_status "Mode: PR (changed packages, parallel, quiet)" ;;
    all)  print_status "Mode: All (all packages, parallel)" ;;
    full) print_status "Mode: Full (all packages, sequential, verbose)" ;;
    *)    print_status "Mode: Developer (quick checks, no artifacts)" ;;
esac

# Display packages to check
if [ -n "$PACKAGES" ]; then
    print_status "Packages: $PACKAGES"
else
    print_status "Packages: All"
fi

if [ -n "$PYTEST_ARGS" ]; then
    print_status "Pytest args: $PYTEST_ARGS"
fi
echo ""

# Create artifacts directory only in PR mode
if [ "$PR_MODE" = "yes" ]; then
    mkdir -p "$ARTIFACTS_DIR"
fi

# Ensure all packages are installed
print_status "Ensuring all packages are installed..."
if [ "$IN_DOCKER" = true ]; then
    # Use sync-packages.sh if available in Docker
    if [ -f "$SCRIPT_DIR/sync-packages.sh" ]; then
        "$SCRIPT_DIR/sync-packages.sh" >/dev/null 2>&1 || print_warning "Package sync had issues"
    else
        uv sync --all-packages >/dev/null 2>&1 || print_warning "Package sync had issues"
    fi
else
    uv sync --all-packages >/dev/null 2>&1 || print_warning "Package sync had issues"
fi
print_success "Packages synced"

# Start services if needed (only on host).
# Not needed for a workspace-only run: the guards under tests/ read files, and
# the one directory there that would need a service is excluded from the run
# and asserted empty (tests/test_toolchain_consistency.py).
if [ "$SKIP_PACKAGE_TESTS" = "yes" ]; then
    print_status "Skipping service startup (workspace guards need no services)"
elif [ "$IN_DOCKER" = false ] && [ "$SKIP_TESTS" != "yes" ]; then
    print_status "Ensuring test services are running..."
    if ! "$SCRIPT_DIR/manage-services.sh" ensure; then
        print_error "Failed to start services"
        exit 1
    fi
    print_success "Services are ready"
elif [ "$IN_DOCKER" = true ] && [ "$SKIP_TESTS" != "yes" ]; then
    print_status "Testing service connectivity..."
    
    # Set environment for connectivity tests
    set_environment_vars
    
    # Test PostgreSQL. Use the uv-managed interpreter (where psycopg2 is
    # installed) rather than a bare `python3` that may resolve to a pyenv
    # shim lacking the dependency — otherwise this always warns even when
    # PostgreSQL is up.
    if uv run python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')" 2>/dev/null; then
        print_success "PostgreSQL is accessible"
    else
        print_warning "PostgreSQL not accessible - integration tests may fail"
    fi
    
    # Test Elasticsearch
    if curl -s "$ELASTICSEARCH_URL/_cluster/health" >/dev/null 2>&1; then
        print_success "Elasticsearch is accessible"
    else
        print_warning "Elasticsearch not accessible - integration tests may fail"
    fi
    
    # Test LocalStack
    if curl -s "$AWS_ENDPOINT_URL/_localstack/health" >/dev/null 2>&1; then
        print_success "LocalStack is accessible"
    else
        print_warning "LocalStack not accessible - integration tests may fail"
    fi
fi

# Set environment variables for all checks
set_environment_vars

# Capture environment information (PR mode only)
if [ "$PR_MODE" = "yes" ]; then
    print_status "Capturing environment information..."
    cat > "$ARTIFACTS_DIR/environment.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "python_version": "$(uv run python --version 2>&1)",
  "uv_version": "$(uv --version 2>&1)",
  "os": "$(uname -s)",
  "os_version": "$(uname -r)",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "docker_container": "$IN_DOCKER",
  "docker_compose_services": "postgres, elasticsearch, localstack"
}
EOF
    print_success "Environment captured"
fi

# Initialize status tracking
VALIDATION_STATUS=0
VALIDATION_SKIPPED="false"
DOCS_STATUS=0
DOCS_VERSIONS_STATUS=0
DOCS_MIRROR_STATUS=0
TEST_STATUS=0
UNIT_TEST_STATUS=0
INTEGRATION_TEST_STATUS=0
WORKFLOW_LINT_STATUS=0
SHELL_LINT_STATUS=0

# Compute the overall PASS/FAIL/PASS_WITH_SKIPS verdict from the individual check
# statuses. Called from two sites that MUST agree — the quality-summary.json
# generation (the value CI validates) and the terminal banner that drives the
# exit code — so the logic lives here once. A new check added to the gate must
# be added to this one function AND to the checks object in quality-summary.json,
# which is the only thing CI reads. Wiring it into one site and not the other
# silently desyncs the exit code from the reported summary (the doc-mirror check)
# or leaves it invisible to CI entirely (the workflow lint). Both halves are
# guarded by tests/test_quality_gate_accounting.py.
compute_overall_status() {
    if [ "$VALIDATION_STATUS" -ne 0 ] || [ "$DOCS_STATUS" -ne 0 ] || [ "$DOCS_VERSIONS_STATUS" -ne 0 ] || [ "$DOCS_MIRROR_STATUS" -ne 0 ] || [ "$TEST_STATUS" -ne 0 ] || [ "$WORKFLOW_LINT_STATUS" -ne 0 ] || [ "$SHELL_LINT_STATUS" -ne 0 ]; then
        echo "FAIL"
    elif [ "$VALIDATION_SKIPPED" = "true" ] && [ "$SKIP_TESTS" = "yes" ]; then
        echo "PASS_WITH_SKIPS"
    else
        echo "PASS"
    fi
}

# Determine package glob pattern
if [ -n "$PACKAGES" ]; then
    # Build glob pattern for specific packages
    PACKAGE_PATTERN=""
    for pkg in $PACKAGES; do
        if [ -d "packages/$pkg" ]; then
            if [ -z "$PACKAGE_PATTERN" ]; then
                PACKAGE_PATTERN="packages/$pkg/src"
            else
                PACKAGE_PATTERN="$PACKAGE_PATTERN packages/$pkg/src"
            fi
        else
            print_warning "Package not found: $pkg"
        fi
    done
    
    if [ -z "$PACKAGE_PATTERN" ]; then
        print_error "No valid packages specified"
        exit 1
    fi
else
    PACKAGE_PATTERN="packages/*/src"
fi

# The code belonging to no package, from the one declaration. This was a fifth
# answer to "which code do we check", and it feeds the ruff JSON a developer
# reads after a failure — so without it a bin/ finding reaches validation.log and
# appears nowhere in the diagnostics artifact, which is the shape that hides a
# finding rather than reporting it.
#
# Invoked rather than sourced: package-discovery.sh sets -u and -o pipefail, and
# this file sets only -e. Captured into a variable first because errexit applies
# to a bare assignment but not to a substitution inside an argument list — a
# failure there would yield an empty string and silently narrow the pattern back.
_workspace_pattern=$("$SCRIPT_DIR/package-discovery.sh" workspace-targets)
PACKAGE_PATTERN="$PACKAGE_PATTERN $_workspace_pattern"

# Validate package references
print_status "Validating package references across codebase..."
if uv run python "$SCRIPT_DIR/validate-package-references.py" > "$ARTIFACTS_DIR/package-validation.log" 2>&1; then
    print_success "Package references are consistent"
else
    print_error "Package validation failed - see $ARTIFACTS_DIR/package-validation.log"
    cat "$ARTIFACTS_DIR/package-validation.log"
    exit 1
fi

# Lint GitHub Actions workflows. The status is captured rather than discarded:
# this check printed ✗ and let the gate go on to report PASS, which is worse
# than not running it, because it also reports success. Every check has to reach
# compute_overall_status (the exit code) and the checks object in
# quality-summary.json (the only thing CI looks at) — see both, below.
print_status "Linting GitHub Actions workflow files..."
if "$SCRIPT_DIR/lint-workflows.sh"; then
    print_success "Workflow files are valid"
else
    WORKFLOW_LINT_STATUS=$?
    print_error "Workflow lint failed"
fi

# Lint the repository's own shell scripts. Same three-place wiring as above, for
# the same reason: 46 shell files — including every script on this verdict path,
# this one among them — went through no linter at all, while every *.py beside
# them went through ruff and mypy.
print_status "Linting shell scripts..."
if "$SCRIPT_DIR/lint-shell.sh"; then
    print_success "Shell scripts are clean"
else
    SHELL_LINT_STATUS=$?
    print_error "Shell lint failed"
fi

# Run code validation (syntax, ruff, imports, mypy, print statements)
if [ "$SKIP_STYLE" != "yes" ]; then
    print_status "Running code validation (syntax, ruff, imports, mypy, print statements)..."

    # Build package args for validate.sh
    VALIDATE_ARGS=""
    # Set when the run must validate everything and so passes no arguments at
    # all. Distinct from an empty VALIDATE_ARGS, which is also how "nothing to
    # validate" is spelled — the two were the same string, and the ambiguity is
    # the bug below.
    VALIDATE_EVERYTHING="no"
    if [ -n "$PACKAGES" ]; then
        # --workspace on this branch too, and unconditionally. Narrowing to the
        # changed packages dropped the workspace half entirely, so a pull request
        # touching any package validated packages/*/src alone — and the ruff
        # config is a global trigger, so editing it marks all ten packages
        # changed and takes this branch. The change that started linting bin/
        # therefore did not lint bin/, and recorded a passing validation for it.
        #
        # Not conditioned on whether the workspace half changed. It is four
        # targets holding the checkers that decide whether this run passes, the
        # marginal cost is seconds, and a condition here is what came out short
        # twice: once reading an empty package list as nothing to validate, and
        # again narrowing by package name. A branch that always asks cannot come
        # out short — though "always" is a claim about the branches that validate
        # anything, not about every path through this script: see the
        # change-detection fallback below, which used to validate nothing.
        VALIDATE_ARGS="$PACKAGES --workspace"
    elif [ "$SKIP_PACKAGE_TESTS" = "yes" ]; then
        # Same hole as the test block, one step earlier: this branch is keyed
        # by package too, so a change to the workspace guards produced no
        # arguments and was read as "nothing to validate" — leaving the edited
        # files unlinted and un-type-checked.
        #
        # Ask for the set by name rather than naming a directory. It was "tests"
        # when tests/ was the only code belonging to no package; bin/ and the
        # root conftest are in it now, so a literal here would have left a
        # bin-only pull request — the shape this very script is edited by — with
        # its own changes unlinted, which is the defect one line up, restated.
        # validate.sh owns the list; this asks which list.
        VALIDATE_ARGS="--workspace"
    elif [ "$SKIP_TESTS" != "yes" ]; then
        # Change detection failed. Nothing above matched, because PACKAGES is
        # empty and neither skip flag was set — the case where the run does not
        # know what changed, announces "testing all packages", and then took the
        # empty-VALIDATE_ARGS path to validating nothing at all. Worse than the
        # skip it resembled: compute_overall_status returns PASS rather than
        # PASS_WITH_SKIPS, because that needs SKIP_TESTS=yes, which this path
        # never sets. A run that could not tell what changed reported a clean
        # validation over no code.
        #
        # No arguments, deliberately: that is how validate.sh spells "everything"
        # — all package sources plus the workspace half — which is what the
        # warning already claims is happening.
        VALIDATE_EVERYTHING="yes"
    fi

    # Skip if no packages to validate in PR mode (e.g., only docs changed)
    if [ -n "$VALIDATE_ARGS" ] || [ "$VALIDATE_EVERYTHING" = "yes" ] || [ "$RUN_MODE" != "pr" ]; then
        if [ "$PR_MODE" = "yes" ]; then
            # Also generate ruff JSON artifact for diagnostics
            # The word splitting is the point. These variables hold argument *lists* —
            # VALIDATE_ARGS is "$PACKAGES --workspace", TEST_FLAGS and PYTEST_ARGS are
            # flag strings, PACKAGE_PATTERN is several globs. Quoting one passes the whole
            # string as a single argument: validate.sh would look for a target literally
            # named "bots --workspace", find nothing, warn, and validate nothing while
            # still reporting success. That is this track's defect exactly, so the waiver
            # is here to stop a future editor "fixing" the finding into a silent failure.
            # shellcheck disable=SC2086
            uv run ruff check $PACKAGE_PATTERN --output-format=json --config "$PROJECT_ROOT/pyproject.toml" > "$ARTIFACTS_DIR/style-check.json" 2>&1 || true

            # Deliberate word splitting — see the waiver above the ruff call.
            # shellcheck disable=SC2086
            if "$SCRIPT_DIR/validate.sh" $VALIDATE_ARGS > "$ARTIFACTS_DIR/validation.log" 2>&1; then
                print_success "Code validation passed"
            else
                VALIDATION_STATUS=$?
                print_error "Code validation failed - see $ARTIFACTS_DIR/validation.log"
                cat "$ARTIFACTS_DIR/validation.log"
            fi
        else
            # Dev mode: show output directly
            # Deliberate word splitting — see the waiver above the ruff call.
            # shellcheck disable=SC2086
            if "$SCRIPT_DIR/validate.sh" $VALIDATE_ARGS; then
                print_success "Code validation passed"
            else
                VALIDATION_STATUS=$?
                print_error "Code validation failed"
            fi
        fi
    else
        VALIDATION_SKIPPED="true"
        print_status "Skipping code validation (no package changes)"
    fi
else
    VALIDATION_SKIPPED="true"
    print_status "Skipping code validation"
fi

# Documentation checks (PR mode only, skip if no docs changes in pr mode)
if [ "$PR_MODE" = "yes" ]; then
    if [ "$DOCS_CHANGED" = "true" ] || [ "$RUN_MODE" != "pr" ]; then
        print_status "Running documentation checks (build, versions, mirrors)..."
        # bin/docs-checks.sh is the single source of truth for the doc-check set.
        # It writes per-check logs (docs-build.log / docs-versions.log /
        # docs-mirror.log) plus docs-checks-status.json into ARTIFACTS_DIR.
        "$SCRIPT_DIR/docs-checks.sh" --artifacts "$ARTIFACTS_DIR" || true
        DOCS_CHECK_CODES=$(python3 -c "
import json
try:
    d = json.load(open('$ARTIFACTS_DIR/docs-checks-status.json'))
    print(d.get('docs-build', 1), d.get('docs-versions', 1), d.get('docs-mirror', 1))
except Exception:
    print(1, 1, 1)
" 2>/dev/null || echo "1 1 1")
        read -r DOCS_STATUS DOCS_VERSIONS_STATUS DOCS_MIRROR_STATUS <<< "$DOCS_CHECK_CODES"

        if [ "$DOCS_STATUS" -eq 0 ] && [ "$DOCS_VERSIONS_STATUS" -eq 0 ] && [ "$DOCS_MIRROR_STATUS" -eq 0 ]; then
            print_success "Documentation checks passed (build, versions, mirrors)"
        else
            print_error "Documentation checks failed - see .quality-artifacts/docs-*.log (details below)"
        fi
    else
        print_status "Skipping docs checks (no documentation changes detected)"
    fi
fi

# Run tests using the test.sh script
if [ "$SKIP_TESTS" != "yes" ]; then
    print_status "Running tests..."
    
    # Build test command (skip service management — already handled above)
    TEST_CMD="$SCRIPT_DIR/test.sh -n"

    # Workspace-level guards check the root config, every package's config, and
    # bin/ — so they belong to no package, and every test path below is keyed by
    # package. Named once here because both modes have to run them: a guard the
    # quick loop skips is one a developer only sees go red at PR time.
    WORKSPACE_TEST_DIR="$PROJECT_ROOT/tests"

    # The useful part of a pytest output file: the FAILED lines when there are
    # any, otherwise the tail. A run can exit non-zero without producing a
    # single FAILED line — a usage error, a collection error, an import error
    # and an internal error all do — and a summary that reports a failure while
    # naming nothing sends the reader to the artifacts to find out what broke.
    # Echoes nothing when the file records no failure, so callers can test for
    # emptiness before printing a header.
    print_test_output_detail() {
        local clean failed
        clean=$(sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null)
        failed=$(echo "$clean" | grep -E '^FAILED ' || true)
        if [ -n "$failed" ]; then
            echo "$failed" | sed 's/^/    /'
        elif echo "$clean" | grep -qE '^ERROR: usage:|^INTERNALERROR|^ERROR |errors during collection'; then
            echo "$clean" | tail -12 | sed 's/^/    /'
        fi
    }

    if [ "$PR_MODE" = "yes" ]; then
        # PR/All/Full mode: Run unit and integration tests separately with artifacts

        # Build common test flags
        TEST_FLAGS="$TEST_PARALLEL_FLAG $TEST_VERBOSITY_FLAG"

        # Always capture failure and skip reasons in output files for the summary
        # -rf = show FAILED lines, -rs = show SKIPPED reasons
        if [ -z "$PYTEST_ARGS" ]; then
            PYTEST_ARGS="-rfs"
        else
            PYTEST_ARGS="-rfs $PYTEST_ARGS"
        fi

        # Helper: run unit tests for a single package, saving artifacts.
        # Uses per-package coverage filenames to avoid race conditions in concurrent mode.
        # Args: $1=package name, $2=cov_report_type (optional override)
        run_unit_for_pkg() {
            local pkg=$1
            local cov_report="${2:-$TEST_COV_REPORT}"
            print_status "Running unit tests for $pkg..."
            local test_exit=0

            # Set unique coverage file to avoid collisions in concurrent mode
            export COVERAGE_FILE="$PROJECT_ROOT/.coverage.unit.$pkg"

            if [ -n "$PYTEST_ARGS" ]; then
                # Deliberate word splitting — see the waiver above the ruff call.
                # shellcheck disable=SC2086
                $TEST_CMD "$pkg" -t unit --cov-report "$cov_report" $TEST_FLAGS -- $PYTEST_ARGS > "$ARTIFACTS_DIR/unit-test-output-$pkg.txt" 2>&1 || test_exit=$?
            else
                # Deliberate word splitting — see the waiver above the ruff call.
                # shellcheck disable=SC2086
                $TEST_CMD "$pkg" -t unit --cov-report "$cov_report" $TEST_FLAGS > "$ARTIFACTS_DIR/unit-test-output-$pkg.txt" 2>&1 || test_exit=$?
            fi

            # Save coverage artifacts
            if [ -f "coverage.xml" ]; then
                mv coverage.xml "$ARTIFACTS_DIR/coverage-unit-$pkg.xml"
            fi
            if [ -f "$PROJECT_ROOT/.coverage.unit.$pkg" ]; then
                mv "$PROJECT_ROOT/.coverage.unit.$pkg" "$ARTIFACTS_DIR/.coverage.unit.$pkg"
            elif [ -f ".coverage" ]; then
                mv .coverage "$ARTIFACTS_DIR/.coverage.unit.$pkg"
            fi

            # Unset to avoid leaking
            unset COVERAGE_FILE

            if [ $test_exit -ne 0 ] && [ $test_exit -ne 5 ]; then
                print_error "Unit tests failed for $pkg"
                # Show failed test names inline (strip ANSI codes first)
                if [ -f "$ARTIFACTS_DIR/unit-test-output-$pkg.txt" ]; then
                    sed 's/\x1b\[[0-9;]*m//g' "$ARTIFACTS_DIR/unit-test-output-$pkg.txt" 2>/dev/null | \
                        grep -E '^FAILED ' | sed 's/^/    /' || true
                fi
                return $test_exit
            else
                print_success "Unit tests passed for $pkg"
                return 0
            fi
        }

        # Determine packages to test. The empty case is load-bearing: it is
        # what makes the loops below iterate zero times while the workspace
        # block that follows them still runs. Gated here rather than around
        # each loop so a new per-package step cannot miss the distinction.
        PACKAGES_TO_TEST=""
        if [ "$SKIP_PACKAGE_TESTS" = "yes" ]; then
            print_status "No package changed — running workspace guards only"
        elif [ -n "$PACKAGES" ]; then
            PACKAGES_TO_TEST="$PACKAGES"
        else
            for pkg_dir in "$PROJECT_ROOT"/packages/*/; do
                if [ -d "$pkg_dir" ]; then
                    pkg_name=$(basename "$pkg_dir")
                    if [ -d "$pkg_dir/tests" ]; then
                        PACKAGES_TO_TEST="$PACKAGES_TO_TEST $pkg_name"
                    fi
                fi
            done
        fi

        # Run unit tests
        print_status "Running unit tests..."
        if [ "$RUN_MODE" = "full" ]; then
            # Full mode: sequential execution
            for pkg in $PACKAGES_TO_TEST; do
                if [ -d "packages/$pkg" ]; then
                    run_unit_for_pkg "$pkg" || UNIT_TEST_STATUS=$?
                fi
            done
        else
            # PR/All mode: concurrent execution of independent packages
            # Each package gets its own COVERAGE_FILE and output file to avoid races.
            # Skip per-package XML reports to avoid write collisions — XML is generated
            # from combined .coverage data in the coverage combining phase.
            # Limit concurrency to avoid CPU contention breaking timing-sensitive tests.
            MAX_CONCURRENT=3
            unit_pids=()
            unit_pkgs=()
            for pkg in $PACKAGES_TO_TEST; do
                if [ -d "packages/$pkg" ]; then
                    run_unit_for_pkg "$pkg" "none" &
                    unit_pids+=($!)
                    unit_pkgs+=("$pkg")

                    # Throttle: when we hit MAX_CONCURRENT, wait for one to finish
                    if [ ${#unit_pids[@]} -ge $MAX_CONCURRENT ]; then
                        # Wait for the oldest job
                        wait "${unit_pids[0]}" 2>/dev/null || {
                            UNIT_TEST_STATUS=1
                        }
                        unit_pids=("${unit_pids[@]:1}")
                        unit_pkgs=("${unit_pkgs[@]:1}")
                    fi
                fi
            done

            # Wait for remaining jobs
            for idx in "${!unit_pids[@]}"; do
                wait "${unit_pids[$idx]}" 2>/dev/null || {
                    UNIT_TEST_STATUS=1
                }
            done
        fi

        # Workspace guards (see WORKSPACE_TEST_DIR above). Folded into the unit
        # status rather than given their own so a failure surfaces through the
        # existing summary. Exit 5 is "no tests collected", not a failure.
        if [ -d "$WORKSPACE_TEST_DIR" ]; then
            print_status "Running workspace tests..."
            workspace_exit=0
            # No $TEST_FLAGS here: those are bin/test.sh flags (--parallel,
            # --quiet), which test.sh translates into pytest's spelling. This
            # run calls pytest directly — test.sh takes a package name and
            # scans packages/*, so it cannot reach this directory — and pytest
            # exits on an unrecognized argument, which the gate would count as
            # a failing suite with no failing test. $PYTEST_ARGS is passed
            # because it is already pytest's own.
            # Deliberate word splitting — see the waiver above the ruff call.
            # shellcheck disable=SC2086
            uv run pytest "$WORKSPACE_TEST_DIR" --ignore="$WORKSPACE_TEST_DIR/integration" \
                -p no:cacheprovider $PYTEST_ARGS --color=yes \
                > "$ARTIFACTS_DIR/unit-test-output-workspace.txt" 2>&1 || workspace_exit=$?

            if [ $workspace_exit -ne 0 ] && [ $workspace_exit -ne 5 ]; then
                UNIT_TEST_STATUS=$workspace_exit
                print_error "Workspace tests failed"
                print_test_output_detail "$ARTIFACTS_DIR/unit-test-output-workspace.txt"
            else
                print_success "Workspace tests passed"
            fi
        fi

        # Run integration tests (always sequential — shared external services)
        print_status "Running integration tests..."
        for pkg in $PACKAGES_TO_TEST; do
            if [ -d "packages/$pkg" ] && [ -d "packages/$pkg/tests/integration" ]; then
                print_status "Running integration tests for $pkg..."
                local_exit=0

                # Use per-package coverage file
                export COVERAGE_FILE="$PROJECT_ROOT/.coverage.integration.$pkg"

                if [ -n "$PYTEST_ARGS" ]; then
                    # Deliberate word splitting — see the waiver above the ruff call.
                    # shellcheck disable=SC2086
                    $TEST_CMD "$pkg" -t integration --cov-report "$TEST_COV_REPORT" $TEST_FLAGS -- $PYTEST_ARGS > "$ARTIFACTS_DIR/integration-test-output-$pkg.txt" 2>&1 || local_exit=$?
                else
                    # Deliberate word splitting — see the waiver above the ruff call.
                    # shellcheck disable=SC2086
                    $TEST_CMD "$pkg" -t integration --cov-report "$TEST_COV_REPORT" $TEST_FLAGS > "$ARTIFACTS_DIR/integration-test-output-$pkg.txt" 2>&1 || local_exit=$?
                fi

                unset COVERAGE_FILE

                if [ $local_exit -ne 0 ] && [ $local_exit -ne 5 ]; then
                    INTEGRATION_TEST_STATUS=$local_exit
                    print_error "Integration tests failed for $pkg"
                    # Show failed test names inline (strip ANSI codes first)
                    if [ -f "$ARTIFACTS_DIR/integration-test-output-$pkg.txt" ]; then
                        sed 's/\x1b\[[0-9;]*m//g' "$ARTIFACTS_DIR/integration-test-output-$pkg.txt" 2>/dev/null | \
                            grep -E '^FAILED ' | sed 's/^/    /' || true
                    fi
                else
                    print_success "Integration tests passed for $pkg"
                fi

                if [ -f "coverage.xml" ]; then
                    mv coverage.xml "$ARTIFACTS_DIR/coverage-integration-$pkg.xml"
                fi
                if [ -f "$PROJECT_ROOT/.coverage.integration.$pkg" ]; then
                    mv "$PROJECT_ROOT/.coverage.integration.$pkg" "$ARTIFACTS_DIR/.coverage.integration.$pkg"
                elif [ -f ".coverage" ]; then
                    mv .coverage "$ARTIFACTS_DIR/.coverage.integration.$pkg"
                fi
            fi
        done

        # Set overall test status
        if [ $UNIT_TEST_STATUS -ne 0 ] || [ $INTEGRATION_TEST_STATUS -ne 0 ]; then
            TEST_STATUS=1
        else
            TEST_STATUS=0
        fi

        if [ $UNIT_TEST_STATUS -eq 0 ]; then
            print_success "Unit tests passed"
        else
            print_error "Unit tests failed"
        fi

        if [ $INTEGRATION_TEST_STATUS -eq 0 ]; then
            print_success "Integration tests passed"
        else
            print_error "Integration tests failed"
        fi

        # Surface test failure details from output files
        if [ $TEST_STATUS -ne 0 ]; then
            echo ""
            echo -e "${RED}── Test Failure Details ──${NC}"

            # Collect all FAILED lines from unit and integration output files
            # Strip ANSI codes before matching since pytest uses colored output
            for output_file in "$ARTIFACTS_DIR"/unit-test-output-*.txt "$ARTIFACTS_DIR"/integration-test-output-*.txt; do
                if [ -f "$output_file" ]; then
                    detail=$(print_test_output_detail "$output_file")
                    if [ -n "$detail" ]; then
                        pkg_label=$(basename "$output_file" .txt | sed 's/.*-output-//')
                        test_type=$(basename "$output_file" .txt | sed 's/-test-output-.*//')
                        echo -e "  ${YELLOW}$test_type ($pkg_label):${NC}"
                        echo "$detail"
                    fi
                fi
            done

            echo -e "${RED}──────────────────────────${NC}"
            echo ""
        fi

        # Surface skip summary from output files (unique reasons with counts)
        skip_summary=""
        for output_file in "$ARTIFACTS_DIR"/unit-test-output-*.txt "$ARTIFACTS_DIR"/integration-test-output-*.txt; do
            if [ -f "$output_file" ]; then
                skips=$(sed 's/\x1b\[[0-9;]*m//g' "$output_file" 2>/dev/null | grep -E '^SKIPPED ' || true)
                if [ -n "$skips" ]; then
                    pkg_label=$(basename "$output_file" .txt | sed 's/.*-output-//')
                    test_type=$(basename "$output_file" .txt | sed 's/-test-output-.*//')
                    # Extract unique skip reasons (part after the last ": ")
                    unique_reasons=$(echo "$skips" | sed 's/.*: //' | sort | uniq -c | sort -rn)
                    if [ -n "$unique_reasons" ]; then
                        skip_summary="${skip_summary}\n  ${YELLOW}${test_type} (${pkg_label}):${NC}\n$(echo "$unique_reasons" | sed 's/^ */    /')\n"
                    fi
                fi
            fi
        done

        if [ -n "$skip_summary" ]; then
            echo ""
            echo -e "${YELLOW}── Skipped Tests ──${NC}"
            echo -e "$skip_summary"
            echo -e "${YELLOW}───────────────────${NC}"
        fi

        # Surface warnings the same way. A test that reports a known divergence
        # rather than failing on it — lint policy drift, an unprobed config —
        # writes to an output file that is only printed when the run fails, so
        # on a green run the number it exists to publish reached nobody.
        warning_summary=""
        for output_file in "$ARTIFACTS_DIR"/unit-test-output-*.txt "$ARTIFACTS_DIR"/integration-test-output-*.txt; do
            if [ -f "$output_file" ]; then
                # pytest's warnings summary spells each entry
                #   "  <file>:<line>: <Category>: <message>"
                # and wraps the message across the lines beneath it. The header
                # line is what carries the category and the headline number.
                warns=$(sed 's/\x1b\[[0-9;]*m//g' "$output_file" 2>/dev/null \
                    | grep -E '^[[:space:]]+.*:[0-9]+: [A-Za-z_.]*Warning: ' || true)
                if [ -n "$warns" ]; then
                    pkg_label=$(basename "$output_file" .txt | sed 's/.*-output-//')
                    warning_summary="${warning_summary}\n  ${YELLOW}${pkg_label}:${NC}\n$(echo "$warns" | sed 's/^ */    /')\n"
                fi
            fi
        done

        if [ -n "$warning_summary" ]; then
            echo ""
            echo -e "${YELLOW}── Warnings ──${NC}"
            echo -e "$warning_summary"
            echo -e "${YELLOW}──────────────${NC}"
        fi
    else
        # Dev mode: Run combined tests without polluting artifacts
        if [ -n "$PACKAGES" ]; then
            # Run tests for each package
            for pkg in $PACKAGES; do
                if [ -d "packages/$pkg" ]; then
                    print_status "Testing package: $pkg"
                    
                    if [ -n "$PYTEST_ARGS" ]; then
                        # Deliberate word splitting — see the waiver above the ruff call.
                        # shellcheck disable=SC2086
                        $TEST_CMD "$pkg" -- $PYTEST_ARGS
                    else
                        $TEST_CMD "$pkg"
                    fi
                    
                    pkg_status=$?
                    if [ $pkg_status -ne 0 ]; then
                        TEST_STATUS=$pkg_status
                        print_error "Tests failed for package: $pkg"
                    else
                        print_success "Tests passed for package: $pkg"
                    fi
                fi
            done
        else
            # This shouldn't happen in dev mode (auto-detection), but handle it
            if [ -n "$PYTEST_ARGS" ]; then
                # Deliberate word splitting — see the waiver above the ruff call.
                # shellcheck disable=SC2086
                $TEST_CMD -- $PYTEST_ARGS
            else
                $TEST_CMD
            fi
            
            TEST_STATUS=$?
            if [ $TEST_STATUS -eq 0 ]; then
                print_success "All tests passed"
            else
                print_error "Some tests failed"
            fi
        fi

        # Workspace guards (see WORKSPACE_TEST_DIR above). The loop over
        # $PACKAGES cannot reach them, and they run in under a second, so the
        # quick loop carries them too. No artifacts in dev mode.
        if [ -d "$WORKSPACE_TEST_DIR" ]; then
            print_status "Running workspace tests..."
            workspace_exit=0
            uv run pytest "$WORKSPACE_TEST_DIR" --ignore="$WORKSPACE_TEST_DIR/integration" \
                -p no:cacheprovider -q || workspace_exit=$?

            if [ $workspace_exit -ne 0 ] && [ $workspace_exit -ne 5 ]; then
                TEST_STATUS=$workspace_exit
                print_error "Workspace tests failed"
            else
                print_success "Workspace tests passed"
            fi
        fi
    fi
    
    # Create test results XML files for CI systems (PR mode only)
    if [ "$PR_MODE" = "yes" ]; then
        if [ ! -f "$ARTIFACTS_DIR/unit-test-results.xml" ]; then
            echo '<?xml version="1.0" encoding="utf-8"?><testsuites></testsuites>' > "$ARTIFACTS_DIR/unit-test-results.xml"
        fi
        if [ ! -f "$ARTIFACTS_DIR/integration-test-results.xml" ]; then
            echo '<?xml version="1.0" encoding="utf-8"?><testsuites></testsuites>' > "$ARTIFACTS_DIR/integration-test-results.xml"
        fi
    fi
else
    print_status "Skipping tests"
fi

# Generate coverage.xml (PR mode only)
# This must happen before the summary/signature so coverage.xml is included in the signature.
# The slower per-package coverage reports run afterward.
COVERAGE_COMBINED=false
if [ "$PR_MODE" = "yes" ]; then
    if ls "$ARTIFACTS_DIR"/.coverage.* >/dev/null 2>&1; then
        print_status "Combining coverage data..."
        cd "$ARTIFACTS_DIR"

        if uv run coverage combine .coverage.* 2>/dev/null; then
            COVERAGE_COMBINED=true
            print_success "Coverage data combined"

            if uv run coverage xml -o coverage.xml 2>/dev/null; then
                print_success "Combined coverage XML generated"
            else
                print_warning "Could not generate combined XML"
                if [ -f "coverage-unit.xml" ]; then
                    cp coverage-unit.xml coverage.xml
                elif [ -f "coverage-integration.xml" ]; then
                    cp coverage-integration.xml coverage.xml
                fi
            fi
        else
            print_warning "Could not combine coverage data, using individual reports"
            if [ -f "coverage-unit.xml" ] && [ -f "coverage-integration.xml" ]; then
                cp coverage-unit.xml coverage.xml
                print_warning "Using unit test coverage as primary report"
            elif [ -f "coverage-unit.xml" ]; then
                cp coverage-unit.xml coverage.xml
            elif [ -f "coverage-integration.xml" ]; then
                cp coverage-integration.xml coverage.xml
            fi
        fi

        cd "$PROJECT_ROOT"
    elif ls "$ARTIFACTS_DIR"/coverage*.xml >/dev/null 2>&1; then
        print_status "Processing coverage XML files..."
        if [ ! -f "$ARTIFACTS_DIR/coverage.xml" ]; then
            if [ -f "$ARTIFACTS_DIR/coverage-unit.xml" ]; then
                cp "$ARTIFACTS_DIR/coverage-unit.xml" "$ARTIFACTS_DIR/coverage.xml"
            elif [ -f "$ARTIFACTS_DIR/coverage-integration.xml" ]; then
                cp "$ARTIFACTS_DIR/coverage-integration.xml" "$ARTIFACTS_DIR/coverage.xml"
            fi
        fi
    else
        echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$ARTIFACTS_DIR/coverage.xml"
    fi

    # Read the line-rate out of coverage.xml now and record it in the summary.
    # coverage.xml itself is no longer committed: it is a multi-megabyte
    # generated file that changed on nearly every run, and the single thing the
    # gate ever took from it was this number — which it reports as a warning and
    # never fails on. A float belongs in the summary; the report belongs on disk.
    COVERAGE_PERCENT=$(python3 -c "
import xml.etree.ElementTree as ET
try:
    root = ET.parse('$ARTIFACTS_DIR/coverage.xml').getroot()
    print(f\"{float(root.attrib.get('line-rate', 0)) * 100:.1f}\")
except Exception:
    print('null')
" 2>/dev/null || echo "null")
    # A blank would emit invalid JSON, which would fail the gate for a reason
    # that has nothing to do with the code under test.
    COVERAGE_PERCENT=${COVERAGE_PERCENT:-null}

    # Generate quality summary and signature immediately after coverage.xml is ready.
    # These are the files CI validates and that must be committed — they must not be
    # delayed by the slower per-package coverage reporting that follows.
    print_status "Computing per-package content hashes..."
    PACKAGE_HASHES_JSON=$(uv run python "$SCRIPT_DIR/package-hashes.py" compute 2>/dev/null || echo "{}")
    # Workspace-level inputs (toolchain config, workspace guards) are hashed
    # separately: they carry their own blast radius and never enter the
    # package dependency graph. Without them a change to mypy.ini or a guard
    # left every stored hash intact and CI validated a stale artifact.
    WORKSPACE_HASHES_JSON=$(uv run python "$SCRIPT_DIR/package-hashes.py" compute-workspace 2>/dev/null || echo "{}")

    print_status "Generating quality summary..."
    OVERALL_STATUS=$(compute_overall_status)

    cat > "$ARTIFACTS_DIR/quality-summary.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "overall_status": "$OVERALL_STATUS",
  "run_mode": "$RUN_MODE",
  "environment": "$([ "$IN_DOCKER" = true ] && echo "docker" || echo "host")",
  "packages": "$([ -n "$PACKAGES" ] && echo "$PACKAGES" || echo "all")",
  "tested_packages": $TESTED_PACKAGES_JSON,
  "coverage_percent": $COVERAGE_PERCENT,
  "package_hashes": $PACKAGE_HASHES_JSON,
  "workspace_hashes": $WORKSPACE_HASHES_JSON,
  "checks": {
    "documentation": {
      "status": $([ "$DOCS_STATUS" -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $DOCS_STATUS,
      "skipped": $([ "$DOCS_CHANGED" = "true" ] || [ "$RUN_MODE" != "pr" ] && echo "false" || echo "true"),
      "tool": "mkdocs"
    },
    "documentation_versions": {
      "status": $([ "$DOCS_VERSIONS_STATUS" -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $DOCS_VERSIONS_STATUS,
      "tool": "docs-update-versions.sh"
    },
    "documentation_mirrors": {
      "status": $([ "$DOCS_MIRROR_STATUS" -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $DOCS_MIRROR_STATUS,
      "tool": "docs-mirror-check.py"
    },
    "validation": {
      "status": $([ $VALIDATION_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $VALIDATION_STATUS,
      "skipped": $VALIDATION_SKIPPED,
      "tool": "validate.sh"
    },
    "shell_lint": {
      "status": $([ "$SHELL_LINT_STATUS" -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $SHELL_LINT_STATUS,
      "tool": "lint-shell.sh"
    },
    "workflow_lint": {
      "status": $([ "$WORKFLOW_LINT_STATUS" -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $WORKFLOW_LINT_STATUS,
      "tool": "lint-workflows.sh"
    },
    "unit_tests": {
      "status": $([ $UNIT_TEST_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $UNIT_TEST_STATUS,
      "skipped": $([ "$SKIP_TESTS" = "yes" ] && echo "true" || echo "false")
    },
    "integration_tests": {
      "status": $([ $INTEGRATION_TEST_STATUS -eq 0 ] && echo '"pass"' || echo '"fail"'),
      "exit_code": $INTEGRATION_TEST_STATUS,
      "skipped": $(if [ "$SKIP_TESTS" = "yes" ] || [ "$SKIP_PACKAGE_TESTS" = "yes" ]; then echo "true"; else echo "false"; fi)
    }
  }
}
EOF

    print_status "Generating artifact signature..."
    cd "$ARTIFACTS_DIR"
    # Sign exactly the files that get committed, not everything the run left on
    # disk. A `find` over the directory also picked up per-package coverage
    # reports and docs status that are gitignored, so the stored signature named
    # a dozen files a CI checkout never contains and the integrity check
    # mismatched on every single pull request — a permanently-red diagnostic,
    # which is how it came to be non-failing instead of correct.
    SIGNED_FILES=$(git ls-files --cached --others --exclude-standard -- '*.json' '*.xml' | sed 's|^|./|' | sort)
    if [ -z "$SIGNED_FILES" ]; then
        # Empty input would leave xargs to run the hasher against stdin and write
        # a checksum of nothing, which reads as a valid signature file.
        print_warning "No committable artifacts found — signature not generated"
        : > signature.sha256
    elif command -v sha256sum >/dev/null 2>&1; then
        echo "$SIGNED_FILES" | xargs sha256sum > signature.sha256
    else
        echo "$SIGNED_FILES" | xargs shasum -a 256 > signature.sha256
    fi
    cd "$PROJECT_ROOT"
    print_success "Quality summary and signature generated"

    # Generate supplementary per-package coverage reports (not needed by CI).
    # This section is slow due to multiple `uv run coverage report` calls but
    # runs after the summary/signature are already written.
    if [ "$COVERAGE_COMBINED" = true ]; then
        print_status "Generating per-package coverage reports..."
        cd "$ARTIFACTS_DIR"

        # Generate combined HTML report (full mode only — slow)
        if [ "$RUN_MODE" = "full" ]; then
            if uv run coverage html -d htmlcov 2>/dev/null; then
                print_success "Combined coverage HTML generated in .quality-artifacts/htmlcov/"
            fi
        fi

        # Generate terminal report
        echo "" >> test-coverage-summary.txt
        echo "Combined Coverage Report:" >> test-coverage-summary.txt
        echo "=========================" >> test-coverage-summary.txt
        uv run coverage report >> test-coverage-summary.txt 2>&1 || true

        # Generate per-package text and JSON summaries in a single loop
        echo "" >> test-coverage-summary.txt
        echo "Coverage by Package:" >> test-coverage-summary.txt
        echo "====================" >> test-coverage-summary.txt

        echo "{" > coverage-by-package.json
        echo '  "generated": "'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'",' >> coverage-by-package.json
        echo '  "packages": {' >> coverage-by-package.json

        first_pkg=true
        for pkg_dir in "$PROJECT_ROOT"/packages/*/; do
            if [ -d "$pkg_dir" ]; then
                pkg_name=$(basename "$pkg_dir")
                if [ "$pkg_name" = "legacy" ]; then
                    src_name="dataknobs"
                else
                    src_name="dataknobs_${pkg_name}"
                fi
                if [ -d "$pkg_dir/src/${src_name}" ]; then
                    # Single coverage report call per package — used for both text and JSON
                    echo -ne "  ${DIM}$pkg_name...${NC} "
                    coverage_output=$(uv run coverage report --data-file="$ARTIFACTS_DIR/.coverage" --include="*/${src_name}/*" 2>/dev/null || true)
                    echo -e "${GREEN}done${NC}"

                    # Text summary
                    echo "" >> test-coverage-summary.txt
                    echo "Package: $pkg_name" >> test-coverage-summary.txt
                    echo "--------" >> test-coverage-summary.txt
                    if [ -n "$coverage_output" ]; then
                        echo "$coverage_output" >> test-coverage-summary.txt
                    else
                        echo "  No coverage data for $pkg_name" >> test-coverage-summary.txt
                    fi

                    # JSON summary
                    total_line=$(echo "$coverage_output" | tail -1)
                    if echo "$total_line" | grep -q "TOTAL"; then
                        coverage_pct=$(echo "$total_line" | awk '{print $(NF)}' | sed 's/%//')
                        statements=$(echo "$total_line" | awk '{print $2}')
                        missing=$(echo "$total_line" | awk '{print $3}')

                        if [ "$first_pkg" = false ]; then
                            echo "," >> coverage-by-package.json
                        fi
                        first_pkg=false

                        echo -n '    "'"$pkg_name"'": {' >> coverage-by-package.json
                        echo -n '"statements": '"$statements"', ' >> coverage-by-package.json
                        echo -n '"missing": '"$missing"', ' >> coverage-by-package.json
                        echo -n '"coverage": "'"$coverage_pct"'%"}' >> coverage-by-package.json
                    fi
                fi
            fi
        done

        echo "" >> coverage-by-package.json
        echo "  }" >> coverage-by-package.json
        echo "}" >> coverage-by-package.json

        cd "$PROJECT_ROOT"
        print_success "Per-package coverage reports generated"
    fi
fi

# Print summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                        Quality Check Summary                     ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ "$PR_MODE" = "yes" ]; then
    # Show documentation build status (only in PR mode)
    if [ "$DOCS_STATUS" -eq 0 ]; then
        echo -e "  Documentation:      ${GREEN}✓ PASSED${NC}"
    else
        echo -e "  Documentation:      ${RED}✗ FAILED${NC}"
    fi

    # Show documentation versions status
    if [ "$DOCS_VERSIONS_STATUS" -eq 0 ]; then
        echo -e "  Doc Versions:       ${GREEN}✓ PASSED${NC}"
    else
        echo -e "  Doc Versions:       ${RED}✗ FAILED${NC}"
    fi

    # Show documentation mirror status
    if [ "$DOCS_MIRROR_STATUS" -eq 0 ]; then
        echo -e "  Doc Mirrors:        ${GREEN}✓ PASSED${NC}"
    else
        echo -e "  Doc Mirrors:        ${RED}✗ FAILED${NC}"
    fi
fi

if [ "$SKIP_STYLE" = "yes" ]; then
    echo -e "  Code Validation:    ${CYAN}⊘ SKIPPED${NC}"
elif [ $VALIDATION_STATUS -eq 0 ]; then
    echo -e "  Code Validation:    ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Code Validation:    ${RED}✗ FAILED${NC}"
fi

# Reported unconditionally because it runs unconditionally — it is not gated by
# PR mode, changed packages, or any skip flag.
if [ "$WORKFLOW_LINT_STATUS" -eq 0 ]; then
    echo -e "  Workflow Lint:      ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Workflow Lint:      ${RED}✗ FAILED${NC}"
fi

# Unconditional for the same reason as the block above it.
if [ "$SHELL_LINT_STATUS" -eq 0 ]; then
    echo -e "  Shell Lint:         ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Shell Lint:         ${RED}✗ FAILED${NC}"
fi

if [ "$PR_MODE" = "yes" ]; then
    # PR mode: Show unit and integration tests separately
    if [ "$SKIP_TESTS" = "yes" ]; then
        echo -e "  Unit Tests:        ${CYAN}⊘ SKIPPED${NC}"
        echo -e "  Integration Tests: ${CYAN}⊘ SKIPPED${NC}"
    elif [ "$SKIP_PACKAGE_TESTS" = "yes" ]; then
        # Named apart rather than reported as two passing suites: no package
        # suite ran, and "Unit Tests: PASSED" for a run that collected none is
        # the same green-for-work-not-done this branch exists to end. The
        # workspace guards did run, and their status is folded into the unit one.
        if [ $UNIT_TEST_STATUS -eq 0 ]; then
            echo -e "  Workspace Guards:  ${GREEN}✓ PASSED${NC}"
        else
            echo -e "  Workspace Guards:  ${RED}✗ FAILED${NC}"
        fi
        echo -e "  Package Tests:     ${CYAN}⊘ SKIPPED (no package changed)${NC}"
    else
        if [ $UNIT_TEST_STATUS -eq 0 ]; then
            echo -e "  Unit Tests:        ${GREEN}✓ PASSED${NC}"
        else
            echo -e "  Unit Tests:        ${RED}✗ FAILED${NC}"
        fi
        
        if [ $INTEGRATION_TEST_STATUS -eq 0 ]; then
            echo -e "  Integration Tests: ${GREEN}✓ PASSED${NC}"
        else
            echo -e "  Integration Tests: ${RED}✗ FAILED${NC}"
        fi
    fi
else
    # Dev mode: Show combined test status
    if [ "$SKIP_TESTS" = "yes" ]; then
        echo -e "  Tests:             ${CYAN}⊘ SKIPPED${NC}"
    elif [ $TEST_STATUS -eq 0 ]; then
        echo -e "  Tests:             ${GREEN}✓ PASSED${NC}"
    else
        echo -e "  Tests:             ${RED}✗ FAILED${NC}"
    fi
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Determine overall status (same helper as the quality-summary.json computation,
# so the exit code below can never disagree with the reported summary).
OVERALL_STATUS=$(compute_overall_status)

if [ "$OVERALL_STATUS" = "PASS" ] || [ "$OVERALL_STATUS" = "PASS_WITH_SKIPS" ]; then
    echo ""
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    if [ "$PR_MODE" = "yes" ]; then
        echo -e "${GREEN}  Artifacts saved to: .quality-artifacts/${NC}"
        echo -e "${GREEN}  You can now create your pull request.${NC}"
    fi
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some checks failed!${NC}"
    echo -e "${RED}  Please fix the issues and run this script again.${NC}"
    
    # Show quick diagnostic info for failures
    echo ""
    echo -e "${YELLOW}Quick Diagnostics:${NC}"
    
    if [ "$PR_MODE" = "yes" ]; then
        # In PR mode, show specific commands to investigate failures
        if [ "$DOCS_STATUS" -ne 0 ] && [ -f "$ARTIFACTS_DIR/docs-build.log" ]; then
            echo -e "  ${CYAN}Documentation Build Failures:${NC}"
            echo "    View documentation errors:"
            echo "      cat $ARTIFACTS_DIR/docs-build.log"
            echo ""
        fi

        if [ "$DOCS_VERSIONS_STATUS" -ne 0 ] && [ -f "$ARTIFACTS_DIR/docs-versions.log" ]; then
            echo -e "  ${CYAN}Documentation Version Mismatch:${NC}"
            echo "    View version differences:"
            echo "      cat $ARTIFACTS_DIR/docs-versions.log"
            echo "    To fix:"
            echo "      bin/docs-update-versions.sh"
            echo ""
        fi

        if [ "$DOCS_MIRROR_STATUS" -ne 0 ] && [ -f "$ARTIFACTS_DIR/docs-mirror.log" ]; then
            echo -e "  ${CYAN}Documentation Mirror Drift:${NC}"
            echo "    View mirror differences:"
            echo "      cat $ARTIFACTS_DIR/docs-mirror.log"
            echo "    To fix:"
            echo "      bin/docs-mirror-check.py --fix   # or reclassify in .dataknobs/docs-mirror-manifest.json"
            echo ""
        fi

        if [ $UNIT_TEST_STATUS -ne 0 ] || [ $INTEGRATION_TEST_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}Test Failures:${NC}"

            for output_file in "$ARTIFACTS_DIR"/*-test-output-*.txt; do
                if [ -f "$output_file" ]; then
                    if grep -q "FAILED" "$output_file" 2>/dev/null; then
                        pkg_name=$(basename "$output_file" | sed 's/.*-test-output-\(.*\)\.txt/\1/')
                        echo "    Package $pkg_name has failures:"
                        echo "      grep -E '(FAILED|ERROR)' $output_file"
                    fi
                fi
            done
        fi
        
        if [ $VALIDATION_STATUS -ne 0 ] && [ -f "$ARTIFACTS_DIR/validation.log" ]; then
            echo -e "  ${CYAN}Code Validation Failures:${NC}"
            echo "    View full validation output:"
            echo "      cat $ARTIFACTS_DIR/validation.log"
            echo "    To auto-fix what's possible:"
            echo "      bin/validate.sh -f"
        fi
        
        echo ""
        echo -e "  ${CYAN}Full artifacts in:${NC} .quality-artifacts/"
        echo -e "  ${CYAN}View summary:${NC} cat .quality-artifacts/quality-summary.json | python -m json.tool"
    else
        # In dev mode, suggest re-running with specific focus
        if [ $TEST_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}To re-run only failed tests:${NC}"
            echo "    $0 $PACKAGES -- --lf    # Run last failed tests"
            echo "    $0 $PACKAGES -- -x      # Stop on first failure"
            echo "    $0 $PACKAGES -- -vvs    # Verbose output with stdout"
        fi
        
        if [ $VALIDATION_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}To auto-fix validation issues:${NC}"
            echo "    bin/validate.sh -f"
        fi
    fi
    
    echo ""
    exit 1
fi


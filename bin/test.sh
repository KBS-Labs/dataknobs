#!/bin/bash
# Enhanced test runner with flexible pytest options

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output (check if we're in a terminal that supports colors)
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

# Default values
TEST_TYPE="both"
# Both accumulate, mirroring bin/validate.sh: a positional is classified as a
# path or a package name and appended. They used to be single-valued, so a
# second positional fell through to the unknown-option arm — which printed the
# usage text and exited 0, reporting success for a run that never started.
PACKAGE_NAMES=()
TEST_PATHS=()  # For direct file/directory paths
# The guards under tests/ belong to no package, so no package name reaches them
# and the path arm would treat them as a package's suite — coverage target and
# all. Named as a target of its own so the quality gate can ask for them through
# this script instead of calling pytest itself, which was the last place the
# gate ran a check the developer's command could not.
RUN_WORKSPACE="no"
# Resolve what would run, print it, run nothing.
PRINT_TARGETS="no"
START_SERVICES="auto"
COVERAGE="yes"
PYTEST_ARGS=""
COV_REPORT="term-missing"
SKIP_INTEGRATION="false"  # If true, sets TEST_*=false
ONLY_INTEGRATION="false"  # If true, only runs integration tests
PARALLEL="no"  # If yes, use pytest-xdist for parallel test execution
VERBOSITY=""   # Empty = default, "quiet" = -q, "verbose" = -v

# Check if we're running in a Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER:-}" ]; then
    IN_DOCKER=true
fi

# Build pytest coverage args for a package
# Args: $1=package name (or path fallback), $2="append" to add --cov-append
build_cov_args() {
    if [ "$COVERAGE" != "yes" ]; then
        echo ""
        return
    fi

    local pkg="$1"
    local append="${2:-}"
    local args=""

    # Determine --cov target
    if [ -z "$pkg" ]; then
        # No package identified — skip coverage
        echo ""
        return
    elif [ "$pkg" = "legacy" ]; then
        args="--cov=packages/$pkg/src/dataknobs"
    else
        args="--cov=packages/$pkg/src/dataknobs_${pkg}"
    fi

    if [ "$append" = "append" ]; then
        args="$args --cov-append"
    fi

    # Add coverage report types
    IFS=',' read -ra REPORT_TYPES <<< "$COV_REPORT"
    for report_type in "${REPORT_TYPES[@]}"; do
        case $report_type in
            term|term-missing)
                args="$args --cov-report=$report_type"
                ;;
            html)
                args="$args --cov-report=html:htmlcov"
                ;;
            xml)
                args="$args --cov-report=xml:coverage.xml"
                ;;
        esac
    done

    echo "$args"
}

# Build extra pytest args for parallel execution and verbosity
build_extra_args() {
    local parts=()
    if [ "$PARALLEL" = "yes" ]; then
        # PYTEST_WORKERS controls xdist worker count (default: 4)
        # Using a fixed count rather than "auto" to avoid oversubscription
        # when multiple packages run concurrently
        local workers="${PYTEST_WORKERS:-4}"
        parts+=("-n" "$workers" "--dist" "loadscope")
    fi
    case "$VERBOSITY" in
        quiet)   parts+=("-q") ;;
        verbose) parts+=("-v") ;;
    esac
    echo "${parts[*]}"
}

# Consolidated function to execute pytest with proper error handling
execute_pytest() {
    # Collapse multiple spaces in command string for clean display
    local cmd
    cmd=$(echo "$1" | tr -s ' ')
    local context="${2:-tests}"  # Optional context for warning message

    echo -e "${CYAN}Command: $cmd${NC}"
    
    local test_result
    if command -v uv &> /dev/null; then
        eval "uv run $cmd"
        test_result=$?
    else
        eval "$cmd"
        test_result=$?
    fi
    
    # Exit code 5 means no tests were collected - treat as success with warning
    if [ $test_result -eq 5 ]; then
        echo -e "${YELLOW}Warning: No $context found${NC}"
        return 0
    fi
    
    return $test_result
}

# Function to set environment variables for integration tests
set_integration_env_vars() {
    if [ "$IN_DOCKER" = true ]; then
        # Use Docker network hostnames when inside container
        export POSTGRES_HOST=postgres
        export ELASTICSEARCH_HOST=elasticsearch
        export AWS_ENDPOINT_URL=http://localstack:4566
        export LOCALSTACK_ENDPOINT=http://localstack:4566
    else
        # Use localhost when running on host
        export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
        export ELASTICSEARCH_HOST="${ELASTICSEARCH_HOST:-localhost}"
        export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"
        export LOCALSTACK_ENDPOINT="${LOCALSTACK_ENDPOINT:-http://localhost:4566}"
    fi
    export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
    export POSTGRES_USER="${POSTGRES_USER:-postgres}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
    export POSTGRES_DB="${POSTGRES_DB:-dataknobs_test}"
    export ELASTICSEARCH_PORT="${ELASTICSEARCH_PORT:-9200}"
    export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
    export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
    
    # Enable test flags based on environment or defaults
    # These control whether integration tests are skipped or run
    # Can be overridden by setting these before calling the script
    if [ "$SKIP_INTEGRATION" = "true" ]; then
        # Explicitly skip integration tests
        export TEST_S3="false"
        export TEST_ELASTICSEARCH="false"
        export TEST_POSTGRES="false"
        export TEST_REDIS="false"
        export TEST_OLLAMA="false"
    else
        # Use environment values or default to true (run tests if services available)
        export TEST_S3="${TEST_S3:-true}"
        export TEST_ELASTICSEARCH="${TEST_ELASTICSEARCH:-true}"
        export TEST_POSTGRES="${TEST_POSTGRES:-true}"
        export TEST_REDIS="${TEST_REDIS:-true}"
        export TEST_OLLAMA="${TEST_OLLAMA:-true}"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
${CYAN}DataKnobs Test Runner${NC}

Usage: $0 [OPTIONS] [PACKAGE|PATH]... [-- PYTEST_ARGS]

Run unit and/or integration tests for DataKnobs packages with flexible pytest options.

PACKAGE|PATH can be given more than once, and can be:
  - A package name (e.g., 'data', 'config')
  - 'workspace' — the guards under tests/, which belong to no package
  - A test file path (e.g., 'packages/data/tests/test_backends/test_s3.py')
  - A test directory (e.g., 'packages/data/tests/integration/')

${YELLOW}Options:${NC}
    -t, --type TYPE          Test type: unit, integration, or both (default: both)
    -p, --package PACKAGE    Package or path to test (repeatable)
                            Can be a package name (e.g., data, config)
                            or a file/directory path (e.g., packages/data/tests/test_s3.py)
                            If not specified, tests all packages
    -s, --services          Start services for integration tests (auto by default)
    -n, --no-services       Don't start services (assume they're already running)
    --skip-integration      Skip integration tests (sets TEST_*=false)
    --only-integration      Only run integration tests (requires services)
    --no-cov                Disable coverage reporting
    --cov-report TYPE       Coverage report type: term, term-missing, html, xml, or combinations
                           (default: term-missing, use comma to combine: term-missing,html,xml)
    --print-targets         Print what this invocation would run and exit, running nothing
    -h, --help              Show this help message

${YELLOW}Advanced Usage:${NC}
    Any arguments after -- are passed directly to pytest:
    $0 data -- -xvs --tb=short --pdb
    
    Common pytest options you can pass after --:
    -v, -vv, -vvv           Verbosity level
    -x                      Exit on first failure
    -s                      No capture, show print statements
    --tb=STYLE              Traceback style (auto, short, line, no, native, long)
    -k EXPRESSION           Only run tests matching expression
    -m MARKERS              Only run tests with specified markers
    --lf                    Rerun only failures from last run
    --ff                    Run failures first, then other tests
    --pdb                   Drop into debugger on failures
    --maxfail=N             Stop after N failures

${YELLOW}Randomized test order (pytest-randomly):${NC}
    Test order is randomized each run. pytest prints the seed in its
    header (e.g. "Using --randomly-seed=123456789"); pass it back to
    reproduce an order-dependent flake from the log:
    $0 data -- -p no:randomly             # Disable randomization
    $0 data -- --randomly-seed=last       # Replay the previous run's order
    $0 data -- --randomly-seed=123456789  # Replay a specific logged seed

${YELLOW}Examples:${NC}
    $0                                    # Run all tests with default settings
    $0 data                               # Test data package
    $0 workspace                          # Run the workspace guards under tests/
    $0 data config                        # Test several packages (one run each)
    $0 -t unit data                       # Unit tests only for data package
    $0 packages/data/tests/test_s3.py    # Run specific test file
    $0 packages/data/tests/integration/  # Run all integration tests for data
    $0 data -- -xvs                       # Exit on first failure, verbose, no capture
    $0 data -- -vv --tb=short             # Very verbose with short tracebacks
    $0 data -- -k "test_s3"               # Run only tests matching "test_s3"
    $0 data -- -m "slow"                  # Run only tests marked as slow
    $0 data -- --lf                       # Rerun only last failures
    $0 data -- --pdb --maxfail=3          # Custom pytest args
    $0 -n data                            # Run without starting services (Docker)
    $0 --skip-integration data            # Skip integration tests, only run unit tests
    $0 --only-integration data            # Only run integration tests

${YELLOW}Docker/Container Notes:${NC}
    - Services are automatically detected when running in Docker
    - Use -n/--no-services if services are already running
    - Coverage reports are saved to project root for persistence

EOF
}

# Classify a positional as a path or a package name and record it.
# Shared by -p/--package and the bare positional arm so the two cannot
# disagree about what counts as a path.
add_target() {
    if [ "$1" = "workspace" ]; then
        RUN_WORKSPACE="yes"
    elif [[ "$1" == *"/"* ]] || [ -f "$1" ] || [ -d "$1" ]; then
        TEST_PATHS+=("$1")
    else
        PACKAGE_NAMES+=("$1")
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -p|--package)
            add_target "$2"
            shift 2
            ;;
        -s|--services)
            START_SERVICES="yes"
            shift
            ;;
        -n|--no-services)
            START_SERVICES="no"
            shift
            ;;
        --skip-integration)
            SKIP_INTEGRATION="true"
            shift
            ;;
        --only-integration)
            ONLY_INTEGRATION="true"
            TEST_TYPE="integration"
            shift
            ;;
        --no-cov)
            COVERAGE="no"
            shift
            ;;
        --cov-report)
            COV_REPORT="$2"
            shift 2
            ;;
        --cov-report=*)
            COV_REPORT="${1#--cov-report=}"
            shift
            ;;
        -j|--parallel)
            PARALLEL="yes"
            shift
            ;;
        --no-parallel)
            PARALLEL="no"
            shift
            ;;
        --print-targets)
            PRINT_TARGETS="yes"
            shift
            ;;
        --quiet|-q)
            VERBOSITY="quiet"
            shift
            ;;
        --verbose|-v)
            VERBOSITY="verbose"
            shift
            ;;
        # Help
        -h|--help)
            show_usage
            exit 0
            ;;
        # Separator for custom pytest args - everything after this goes to pytest
        --)
            shift
            # $* not $@: the target is a string, and in a scalar assignment both
            # join identically — the array form is misleading here, not wrong.
            # Re-split at the pytest call site.
            PYTEST_ARGS="$*"
            break
            ;;
        # An unrecognized flag. Matched before the positional arm below so it is
        # diagnosed as a bad option rather than looked up as a package that does
        # not exist — and so it exits non-zero, which the shared show_usage no
        # longer decides for its callers.
        -*)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            echo "Use -- to pass arguments to pytest" >&2
            show_usage >&2
            exit 2
            ;;
        *)
            add_target "$1"
            shift
            ;;
    esac
done

# Validate test type
if [[ "$TEST_TYPE" != "unit" && "$TEST_TYPE" != "integration" && "$TEST_TYPE" != "both" ]]; then
    echo -e "${RED}Invalid test type: $TEST_TYPE${NC}"
    echo "Must be one of: unit, integration, both"
    exit 1
fi

# Handle conflicting flags
if [ "$SKIP_INTEGRATION" = "true" ] && [ "$ONLY_INTEGRATION" = "true" ]; then
    echo -e "${RED}Error: Cannot use --skip-integration and --only-integration together${NC}"
    exit 1
fi

# Adjust test type based on flags
if [ "$SKIP_INTEGRATION" = "true" ]; then
    if [ "$TEST_TYPE" = "integration" ]; then
        echo -e "${RED}Error: Cannot skip integration tests when test type is 'integration'${NC}"
        echo "Remove --skip-integration or change test type"
        exit 1
    elif [ "$TEST_TYPE" = "both" ]; then
        # Silently change to unit tests only
        TEST_TYPE="unit"
    fi
fi

# Function to extract package name from a test path
extract_package_from_path() {
    local path=$1
    # Convert to absolute path if relative
    if [[ "$path" != /* ]]; then
        path="$ROOT_DIR/$path"
    fi
    
    # Extract package name if path is under packages/
    if [[ "$path" == *"/packages/"* ]]; then
        local package_part="${path#*/packages/}"
        echo "${package_part%%/*}"
    else
        echo ""
    fi
}

# Function to run tests for a specific file or directory
run_path_tests() {
    local path=$1
    echo -e "${YELLOW}Running tests from path: $path${NC}"
    
    # Convert to absolute path if relative
    if [[ "$path" != /* ]]; then
        # First check if the path exists relative to current directory
        if [ -e "$path" ]; then
            path="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
        # Otherwise try relative to ROOT_DIR
        elif [ -e "$ROOT_DIR/$path" ]; then
            path="$ROOT_DIR/$path"
        else
            echo -e "${RED}Path not found: $path${NC}"
            return 1
        fi
    elif [ ! -e "$path" ]; then
        echo -e "${RED}Path not found: $path${NC}"
        return 1
    fi
    
    # Always set environment variables for tests (they control which tests are skipped)
    set_integration_env_vars
    
    # Check if it's an integration test path to determine if services are needed
    # Skip service startup if we're explicitly skipping integration tests
    if [ "$SKIP_INTEGRATION" != "true" ] && ([[ "$path" == *"/integration"* ]] || [[ "$path" == *"/integration/"* ]] || [[ "$path" == *"/tests"* ]]); then
        # Start services if needed using manage-services.sh
        if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
            if [ "$IN_DOCKER" = true ]; then
                echo -e "${BLUE}Running in Docker container, checking service connectivity...${NC}"
                "$SCRIPT_DIR/manage-services.sh" ensure >/dev/null 2>&1 || {
                    echo -e "${YELLOW}Warning: Some services may not be reachable from container${NC}"
                }
            else
                if [ "$START_SERVICES" = "yes" ] || [ "$START_SERVICES" = "auto" ]; then
                    echo -e "${YELLOW}Ensuring services are running for integration tests...${NC}"
                    "$SCRIPT_DIR/manage-services.sh" ensure || {
                        echo -e "${RED}Failed to ensure services are running${NC}"
                        return 1
                    }
                    SERVICES_STARTED=true
                fi
            fi
        fi
    fi
    
    # Build coverage args
    local pkg_for_cov
    pkg_for_cov=$(extract_package_from_path "$path")
    local cov_args
    cov_args=$(build_cov_args "$pkg_for_cov")
    # Fall back to covering the test path itself if no package identified
    if [ "$COVERAGE" = "yes" ] && [ -z "$cov_args" ]; then
        cov_args="--cov=$path"
    fi

    # Run tests
    local extra_args
    extra_args=$(build_extra_args)
    local cmd="pytest $path $cov_args $extra_args $PYTEST_ARGS --color=yes"
    execute_pytest "$cmd" "tests in $path"
}

# Function to discover packages with tests
discover_test_packages() {
    local test_type=$1
    local packages=()
    
    for pkg_dir in "$ROOT_DIR"/packages/*/; do
        if [ -d "$pkg_dir" ]; then
            pkg_name=$(basename "$pkg_dir")
            
            # Skip legacy package for tests
            if [ "$pkg_name" = "legacy" ]; then
                continue
            fi
            
            # Check for test directories based on type
            if [ "$test_type" = "unit" ] || [ "$test_type" = "both" ]; then
                if [ -d "$pkg_dir/tests" ] && [ "$(find "$pkg_dir/tests" -name "test_*.py" -type f 2>/dev/null | head -1)" ]; then
                    packages+=("$pkg_name")
                    continue
                fi
            fi
            
            if [ "$test_type" = "integration" ] || [ "$test_type" = "both" ]; then
                if [ -d "$pkg_dir/tests/integration" ]; then
                    packages+=("$pkg_name")
                    continue
                fi
            fi
        fi
    done
    
    echo "${packages[@]}"
}

# Function to run unit tests
run_unit_tests() {
    local package=$1
    echo -e "${YELLOW}Running unit tests for package: $package${NC}"
    
    local test_path="packages/$package/tests"
    
    # Exclude integration tests if they're in a subdirectory
    local exclude_args=""
    if [ -d "packages/$package/tests/integration" ]; then
        exclude_args="--ignore=packages/$package/tests/integration"
    fi

    local cov_args
    cov_args=$(build_cov_args "$package")

    # Run tests
    local extra_args
    extra_args=$(build_extra_args)
    local cmd="pytest $test_path $exclude_args $cov_args $extra_args $PYTEST_ARGS --color=yes"
    execute_pytest "$cmd" "unit tests for $package"
}

# Function to run integration tests
run_integration_tests() {
    local package=$1
    echo -e "${YELLOW}Running integration tests for package: $package${NC}"
    
    local test_path="packages/$package/tests/integration"
    
    # Check if integration tests exist
    if [ ! -d "$test_path" ]; then
        echo -e "${BLUE}No integration tests found for package: $package${NC}"
        return 0
    fi
    
    # Start services if needed using manage-services.sh
    if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
        if [ "$IN_DOCKER" = true ]; then
            echo -e "${BLUE}Running in Docker container, checking service connectivity...${NC}"
            # Use manage-services.sh to check service connectivity from within container
            "$SCRIPT_DIR/manage-services.sh" ensure >/dev/null 2>&1 || {
                echo -e "${YELLOW}Warning: Some services may not be reachable from container${NC}"
            }
            START_SERVICES="no"  # Don't try to start services from within container
        else
            if [ "$START_SERVICES" = "auto" ]; then
                START_SERVICES="yes"  # Let manage-services.sh decide if they're already running
            fi
            
            if [ "$START_SERVICES" = "yes" ]; then
                echo -e "${YELLOW}Ensuring services are running for integration tests...${NC}"
                "$SCRIPT_DIR/manage-services.sh" ensure || {
                    echo -e "${RED}Failed to ensure services are running${NC}"
                    return 1
                }
                SERVICES_STARTED=true
            fi
        fi
    fi
    
    # Set environment variables for tests
    set_integration_env_vars

    local cov_args
    cov_args=$(build_cov_args "$package" "append")

    # Run tests
    local extra_args
    extra_args=$(build_extra_args)
    local cmd="pytest $test_path $cov_args $extra_args $PYTEST_ARGS --color=yes"
    execute_pytest "$cmd" "integration tests for $package"
}

# Function to run the workspace guards — the suite under tests/, which belongs
# to no package.
#
# No coverage: the guards read configuration and shell scripts, so a --cov
# target would measure package source none of them import. No services either;
# the one directory here that would need them is tests/integration, excluded
# below and asserted empty by tests/test_toolchain_consistency.py.
#
# -p no:cacheprovider because these run in a gate that must not leave state
# behind, and --durations=10 because one number for the whole suite says it is
# expensive without saying which part is: the first run with the flag attributed
# 45 of 54 seconds to a single file re-invoking the shell lint per test.
run_workspace_tests() {
    echo -e "${YELLOW}Running workspace guards${NC}"

    local test_path="$ROOT_DIR/tests"
    if [ ! -d "$test_path" ]; then
        echo -e "${BLUE}No workspace guards found${NC}"
        return 0
    fi

    local extra_args
    extra_args=$(build_extra_args)
    local cmd="pytest $test_path --ignore=$test_path/integration -p no:cacheprovider --durations=10 $extra_args $PYTEST_ARGS --color=yes"
    execute_pytest "$cmd" "workspace guards"
}

# Function to run combined tests with coverage
run_combined_tests() {
    local package=$1
    echo -e "${YELLOW}Running all tests for package: $package${NC}"
    
    local test_path="packages/$package/tests"
    
    # Check if we need to start services for integration tests
    # Skip service startup if we're explicitly skipping integration tests
    if [ "$SKIP_INTEGRATION" != "true" ] && [ -d "$test_path/integration" ]; then
        # Start services if needed using manage-services.sh
        if [ "$START_SERVICES" = "auto" ] || [ "$START_SERVICES" = "yes" ]; then
            if [ "$IN_DOCKER" = true ]; then
                echo -e "${BLUE}Running in Docker container, checking service connectivity...${NC}"
                # Use manage-services.sh to check service connectivity from within container
                "$SCRIPT_DIR/manage-services.sh" ensure >/dev/null 2>&1 || {
                    echo -e "${YELLOW}Warning: Some services may not be reachable from container${NC}"
                }
                START_SERVICES="no"  # Don't try to start services from within container
            else
                if [ "$START_SERVICES" = "auto" ]; then
                    START_SERVICES="yes"  # Let manage-services.sh decide if they're already running
                fi
                
                if [ "$START_SERVICES" = "yes" ]; then
                    echo -e "${YELLOW}Ensuring services are running for integration tests...${NC}"
                    "$SCRIPT_DIR/manage-services.sh" ensure || {
                        echo -e "${RED}Failed to ensure services are running${NC}"
                        echo -e "${YELLOW}Integration tests may fail without services${NC}"
                    }
                    SERVICES_STARTED=true
                fi
            fi
        fi
        
        # Set environment variables for integration tests
        set_integration_env_vars
    fi

    local cov_args
    cov_args=$(build_cov_args "$package")

    # Run all tests together for combined coverage
    local extra_args
    extra_args=$(build_extra_args)
    local cmd="pytest $test_path $cov_args $extra_args $PYTEST_ARGS --color=yes"
    execute_pytest "$cmd" "tests for $package"
}

# Main execution

# Determine what to test. Paths and packages are no longer exclusive: each
# named target runs, in the order it was classified. Only a run that names
# neither falls back to discovering every package.
#
# Decided before anything is announced, so --print-targets can answer without
# a banner on the same stream a caller is parsing.
PACKAGES=()

if [ ${#PACKAGE_NAMES[@]} -gt 0 ]; then
    # Every name is checked before anything runs, so an unknown package in the
    # list fails immediately rather than after the earlier ones have run.
    for pkg in "${PACKAGE_NAMES[@]}"; do
        if [ ! -d "$ROOT_DIR/packages/$pkg" ]; then
            echo -e "${RED}Package not found: $pkg${NC}" >&2
            exit 1
        fi
    done
    PACKAGES=("${PACKAGE_NAMES[@]}")
elif [ ${#TEST_PATHS[@]} -eq 0 ] && [ "$RUN_WORKSPACE" = "no" ]; then
    # Discover packages based on test type
    # Word splitting is intended — discover_test_packages emits a space-
    # separated list — but a read loop states it, keeps a name containing a
    # glob character from expanding against the filesystem, and unlike the
    # mapfile shellcheck suggests it works on bash 3.2 (stock macOS).
    PACKAGES=()
    while IFS= read -r _pkg; do
        [[ -n "$_pkg" ]] && PACKAGES+=("$_pkg")
    done < <(discover_test_packages "$TEST_TYPE" | tr ' ' '\n')

    # "Everything" has to mean everything. Discovery loops packages/*, so the
    # guards under tests/ — which belong to no package — were outside it, and a
    # bare `bin/test.sh`, the command that reads as "run the whole suite", was
    # the one entry point that ran every test except the ones checking the
    # toolchain running them.
    #
    # Not for an integration-only run: these need no service and are not
    # integration tests, so folding them in would report on a suite that run
    # did not ask for.
    if [ "$TEST_TYPE" != "integration" ]; then
        RUN_WORKSPACE="yes"
    fi
fi

# Resolved, before any check runs — the shape bin/validate.sh uses, for the same
# reason: a caller that needs to know what this script runs can ask it rather
# than re-derive the answer from its source.
if [ "$PRINT_TARGETS" = "yes" ]; then
    if [ "$RUN_WORKSPACE" = "yes" ]; then
        echo "workspace"
    fi
    if [ ${#TEST_PATHS[@]} -gt 0 ]; then
        printf '%s\n' "${TEST_PATHS[@]}"
    fi
    if [ ${#PACKAGES[@]} -gt 0 ]; then
        printf '%s\n' "${PACKAGES[@]}"
    fi
    exit 0
fi

echo -e "${GREEN}DataKnobs Test Runner${NC}"
echo "======================================"
echo -e "Test type: ${BLUE}$TEST_TYPE${NC}"
if [ "$IN_DOCKER" = true ]; then
    echo -e "Environment: ${CYAN}Docker Container${NC}"
fi

if [ "$RUN_WORKSPACE" = "yes" ]; then
    echo -e "Workspace guards: ${BLUE}tests/${NC}"
fi
if [ ${#TEST_PATHS[@]} -gt 0 ]; then
    echo -e "Test paths: ${BLUE}${TEST_PATHS[*]}${NC}"
fi
if [ ${#PACKAGES[@]} -gt 0 ]; then
    echo -e "Packages: ${BLUE}${PACKAGES[*]}${NC}"
fi

if [ -n "$PYTEST_ARGS" ]; then
    echo -e "Pytest args: ${CYAN}$PYTEST_ARGS${NC}"
fi
echo -e "${YELLOW}Test order is randomized (pytest-randomly).${NC} The seed is" \
        "printed in the pytest header below; replay an order-dependent" \
        "flake with ${CYAN}-- --randomly-seed=last${NC} (or a logged seed)," \
        "or disable with ${CYAN}-- -p no:randomly${NC}."
echo ""

# Track overall test result
OVERALL_RESULT=0
SERVICES_STARTED=false

# Function to cleanup services if we started them
#
# Reached only through `trap cleanup_services EXIT INT TERM` below, which the
# linter does not read as a call site. (Do not begin a comment line with the
# tool's name — it is parsed as a malformed directive.)
# shellcheck disable=SC2329
cleanup_services() {
    if [ "$SERVICES_STARTED" = true ] && [ "$IN_DOCKER" = false ]; then
        if [ -f "/tmp/.dataknobs_services_started_$$" ]; then
            echo -e "\n${YELLOW}Cleaning up services...${NC}"
            "$SCRIPT_DIR/manage-services.sh" stop
        fi
    fi
}

# Set trap for cleanup on exit
trap cleanup_services EXIT INT TERM

# Run every named target. Each gets its own pytest process — packages have
# always been run one at a time by the discovery path, and naming several
# explicitly takes the same route rather than handing pytest one invocation
# spanning multiple package trees.
if [ "$RUN_WORKSPACE" = "yes" ]; then
    run_workspace_tests || OVERALL_RESULT=$?
fi

for path in "${TEST_PATHS[@]}"; do
    run_path_tests "$path" || OVERALL_RESULT=$?
done

for pkg in "${PACKAGES[@]}"; do
    echo -e "\n${GREEN}Testing package: $pkg${NC}"
    echo "----------------------------------------"

    case "$TEST_TYPE" in
        unit)
            run_unit_tests "$pkg" || OVERALL_RESULT=$?
            ;;
        integration)
            run_integration_tests "$pkg" || OVERALL_RESULT=$?
            ;;
        both)
            run_combined_tests "$pkg" || OVERALL_RESULT=$?
            ;;
    esac
done

# Summary
echo ""
echo "======================================"
if [ $OVERALL_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

exit $OVERALL_RESULT
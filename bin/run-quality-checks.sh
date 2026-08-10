#!/bin/bash
set -e

# Quality Checks Script for DataKnobs
# This script runs all quality checks including unit tests, integration tests,
# linting, and code coverage. Results are saved as artifacts for CI validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run from the root, always. The script already assumed this — three places
# `cd` into the artifacts directory and `cd "$PROJECT_ROOT"` back — but nothing
# established it, and `bin/dk` dispatches the gate without changing directory.
# So from a subdirectory the relative target globs below resolved against the
# wrong place: ruff received `packages/*/src` unexpanded, reported it as a
# single E902 io-error *finding* in valid JSON, and the committed
# style-check.json read as one style issue instead of the whole check not
# having run. Anchoring here rather than at the ruff call because the same
# working directory decides `per-file-ignores` matching, and a `cd` in a
# subshell at the call site would come too late for the glob.
cd "$PROJECT_ROOT"

# Where the committed artifacts live. Read in exactly one place — the OUTPUT_DIR
# resolution below — and never written to directly. Every write in this script
# targets OUTPUT_DIR, which is this directory only when --emit-artifacts is
# passed. That is what makes "this script checks, it does not produce evidence"
# a property of the code rather than a claim: a run without the flag cannot
# reach this path at all, whatever it writes.
# tests/test_quality_check_roles.py holds both halves.
ARTIFACTS_DIR="$PROJECT_ROOT/.quality-artifacts"
# Where a check-only run's output lands: per-package test output, coverage
# reports, the validation and docs logs. Git-ignored wholesale, read by no CI
# step, and read by bin/diagnose-quality-failures.sh — which is why this is a
# stable path and not a temporary directory. Sending check-only output to
# `mktemp -d` and deleting it at exit satisfies the rule above and throws the
# evidence away with the attestation: the diagnose tool then has input only
# after a gate run, and a failing check-only run leaves an uncleaned directory
# under /tmp that nothing knows how to find.
#
# Separate from ARTIFACTS_DIR rather than a subdirectory of it, because the rule
# above is enforceable only as "the checker cannot name that directory". Which
# files CI attests is not a list a writer could consult and avoid: the signature
# enumerates by glob (git ls-files ... -- '*.json' '*.xml'), so it is a property
# of .gitignore at the time of the run.
REPORTS_DIR="$PROJECT_ROOT/.quality-reports"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Wall-clock start, for the durations recorded in quality-summary.json. Whole
# seconds via `date +%s`: bash 3.2 has no EPOCHREALTIME and macOS `date` has no
# %N, so finer resolution would cost a subprocess per stamp and change no
# decision anyone makes from these numbers.
RUN_START=$(date +%s)

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
# Whether this run produces the evidence CI verifies. Off unless asked, in every
# mode, and asked for by `bin/dk pr` alone — the gate. Separated from PR_MODE
# because that flag answers a different question (how thorough is the run: the
# per-package unit/integration split, or the quick combined loop), and the two
# were one value. The consequence was that three of this script's four modes
# rewrote .quality-artifacts/, including the no-argument default, and the CI
# comment for a failed gate told a developer to run the artifact-writing form —
# so the documented remedy for a red gate was the command that rewrites the
# evidence the gate reads.
EMIT_ARTIFACTS="no"
# Resolve the output directory, print it, run nothing. Same shape as
# `validate.sh --print-targets`: it lets a guard ask this script where its
# writes land without running a check, through the real resolution rather than a
# reimplementation of it.
PRINT_OUTPUT_DIR="no"

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

${YELLOW}This script checks; it does not produce evidence.${NC} Every mode writes its
working output to a temporary directory, removed on exit and kept on failure so
the logs stay readable. Only --emit-artifacts writes .quality-artifacts/, and
only ${CYAN}bin/dk pr${NC} passes it. To regenerate the artifacts CI verifies,
run ${CYAN}bin/dk pr${NC}.

${YELLOW}Options:${NC}
    -p, --package PACKAGE    Package to check (can be specified multiple times)
                            If not specified, checks all packages
    --pr                    PR scope (default): Only test changed packages + dependents.
                            Uses parallel execution, quiet output, XML-only coverage.
                            Skips docs build if no docs changed.
    --all                   All scope: Test all packages with parallel execution
                            and optimized coverage (no HTML reports)
    --full                  Full scope: Legacy behavior — all packages, sequential,
                            verbose output, all coverage reports (HTML + XML + term)
    --dev                   Dev mode: Run quick checks (combined tests)
    --emit-artifacts        Write .quality-artifacts/ (the gate; bin/dk pr passes this).
                            Not valid with --dev, which runs no step that produces them.
    --print-output-dir      Print where this invocation's output would go, run nothing
    --base-ref REF          Git ref for change detection (default: main)
    --skip-style            Skip code validation (syntax, ruff, imports, mypy)
    --skip-tests            Skip test execution
    --keep-services         Keep services running after completion
    -h, --help              Show this help message

${YELLOW}Advanced Usage:${NC}
    Any arguments after -- are passed directly to pytest:
    $0 data -- -xvs --tb=short

${YELLOW}Examples:${NC}
    $0                      # Check changed packages (default scope; no artifacts)
    $0 --all                # All packages, parallel, optimized
    $0 --full               # Legacy: all packages, sequential, verbose
    $0 --dev data           # Dev mode: Quick checks for data package
    $0 data config          # Dev mode: Check specific packages
    $0 --pr data            # PR scope for data package only
    $0 --skip-style         # Run all checks except style checks
    $0 data -- -x           # Run data package with pytest -x flag
    bin/dk pr               # The gate: same checks, plus .quality-artifacts/

${YELLOW}Environment:${NC}
    Running in: $([ "$IN_DOCKER" = true ] && echo "Docker container" || echo "Host system")
    
${YELLOW}Output:${NC}
    Without --emit-artifacts, everything below goes to .quality-reports/ — the
    diagnostics tier. Git-ignored, read by no CI step, cleared at the start of
    each run, and where bin/diagnose-quality-failures.sh looks.

    With --emit-artifacts (bin/dk pr), the same files go to .quality-artifacts/
    instead, six of which are committed and verified by CI.

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
        --emit-artifacts)
            EMIT_ARTIFACTS="yes"
            shift
            ;;
        --print-output-dir)
            PRINT_OUTPUT_DIR="yes"
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

# Dev mode runs the quick combined loop, which executes none of the steps that
# produce the committed artifacts — no per-package output capture, no coverage
# combine, no summary. Accepting the flag here would write a directory holding
# whichever files happened to fall out, which is the shape of a stale artifact
# set that still passes the signature check. Refuse instead of half-emitting.
if [ "$EMIT_ARTIFACTS" = "yes" ] && [ "$PR_MODE" != "yes" ]; then
    print_error "--emit-artifacts is not valid with --dev (or with named packages,"
    print_error "which imply dev mode). Use: bin/dk pr"
    exit 2
fi

# Everything this run writes goes here. The artifacts directory only when the
# gate asked for it; otherwise the diagnostics tier. Every later write names
# OUTPUT_DIR, so check-only is enforced by there being no other path to name
# rather than by remembering to test a flag at each site.
if [ "$EMIT_ARTIFACTS" = "yes" ]; then
    OUTPUT_DIR="$ARTIFACTS_DIR"
else
    OUTPUT_DIR="$REPORTS_DIR"
fi

if [ "$PRINT_OUTPUT_DIR" = "yes" ]; then
    printf '%s\n' "$OUTPUT_DIR"
    # Resolution only — nothing is created, so there is nothing to undo.
    exit 0
fi

# The diagnostics tier is cleared on entry, not at exit. On entry, so that what
# is in it is one run's and cannot be read as this run's the way a month-old
# coverage report sitting beside today's can be. Not at exit, so a failing run
# leaves its logs where the paths it just printed say they are — the reason the
# temporary-directory shape needed a keep-on-failure flag to be usable at all.
if [ "$EMIT_ARTIFACTS" = "no" ]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# A run is in progress from here until it records a verdict. Written before the
# first thing that can abort one, removed by the summary write at the far end.
#
# Seven checks exit ahead of that write, most because the run could not proceed
# at all — services down, ruff unable to read its targets. The clause above
# cannot cover it: on the gate path the directory holds committed files, so
# clearing on entry would delete evidence on behalf of a run that then aborts.
# What is left instead is the *previous* run's summary sitting beside logs this
# run has already overwritten, and nothing in the directory says the two are
# from different runs. A reader takes it for one, and the older half is the
# half that carries the verdict.
#
# So the marker is not a lock and does not protect against concurrent runs. It
# records one fact — this directory has not been closed out — for the reader
# that would otherwise have to compare timestamps across files to notice.
printf '%s\n' "$TIMESTAMP" > "$OUTPUT_DIR/.run-in-progress"

# One line per check, appended by the site that ran it, read by the writer at
# the far end. Truncated here rather than appended to: the diagnostics tier is
# cleared on entry but the artifacts tier is not, so without this a gate run
# would build its summary from its own records plus the previous run's.
RECORDS_FILE="$OUTPUT_DIR/check-records.jsonl"
: > "$RECORDS_FILE"

# record_check <name> <exit_code> [--tool T] [--skipped true|false]
#              --duration <seconds|null> [--field key=<json>]
#
# Called from the site that ran the check, or from the arm that deliberately
# skipped it — which is the whole point of the file. The summary used to be a
# heredoc over variables initialised to 0, and 0 renders as "pass", so a check
# no path assigned reported as one that ran and passed. That shipped twice. A
# record written where the outcome is produced has no default to fall through
# to: a check nothing reaches writes nothing, and nothing cannot be read as a
# pass.
#
# Not backgrounded and not batched. errexit is on, so a failure here stops the
# run rather than leaving the summary quietly short of a check.
record_check() {
    local name="$1" code="$2"
    shift 2
    python3 "$SCRIPT_DIR/quality-summary.py" record \
        --records "$RECORDS_FILE" --name "$name" --exit-code "$code" "$@"
}

# Content hashes, taken before the first check reads anything.
#
# These were computed at the far end, after every check had finished, which
# meant the recorded digest described the tree at the *end* of the run rather
# than the tree the checks read. A file checked at minute 1 holding content A
# and edited to B at minute 2 was hashed as B, so the validator later compared
# disk (B) against recorded (B), found them equal, and accepted an artifact
# attesting content that no check had seen.
#
# Taken here, that same edit surfaces as the ordinary "Package content has
# changed since quality checks were run" the validator already prints. No new
# mechanism, no new failure mode, and it fails closed — which the end-of-run
# version structurally cannot.
#
# The far end now re-computes and compares instead; see the re-check below.
#
# The `|| echo "{}"` fallback below is the checker role's, and only the
# checker's. An empty document is a hash computation that *failed*, and it
# compares clean against anything — so in the gate it would put a signed
# artifact on the far side of a comparison that never happened, which is this
# repository's own defect class rather than a tolerable degradation. The
# diagnostics tier attests nothing, so there it is exactly a tolerable
# degradation: the run is still worth finishing.
#
# Failing here rather than at the re-check is the difference between a minute
# and the whole run: there is nothing the intervening checks could produce that
# would make an artifact writable without these.
print_status "Computing per-package content hashes..."
PACKAGE_HASHES_JSON=$(uv run python "$SCRIPT_DIR/package-hashes.py" compute 2>/dev/null || echo "{}")
# Workspace-level inputs (toolchain config, workspace guards) are hashed
# separately: they carry their own blast radius and never enter the
# package dependency graph. Without them a change to mypy.ini or a guard
# left every stored hash intact and CI validated a stale artifact.
WORKSPACE_HASHES_JSON=$(uv run python "$SCRIPT_DIR/package-hashes.py" compute-workspace 2>/dev/null || echo "{}")

if [ "$EMIT_ARTIFACTS" = "yes" ]; then
    for _half in "packages:$PACKAGE_HASHES_JSON" "workspace:$WORKSPACE_HASHES_JSON"; do
        if [ "${_half#*:}" = "{}" ]; then
            print_error "Could not compute ${_half%%:*} content hashes."
            print_error "An artifact signed over an empty digest set attests nothing,"
            print_error "so no checks were run. Fix the hash computation and re-run."
            exit 1
        fi
    done
    unset _half
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
    # Nothing to remove: the output directory is either the artifacts the gate
    # was asked to produce, or the diagnostics tier, which is cleared at the
    # start of the next run and is exactly what someone reads after a failure.
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

# Capture environment information (an artifact, so gate only)
if [ "$EMIT_ARTIFACTS" = "yes" ]; then
    print_status "Capturing environment information..."
    cat > "$OUTPUT_DIR/environment.json" <<EOF
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
CONTRACT_STATUS=0

# How long each check took, in whole seconds, for the durations recorded
# alongside its status in quality-summary.json. These answer "where does a gate
# run spend its time" from the artifact rather than from a stopwatch and a
# guess, which is the only way to tell an expensive check from one that merely
# feels slow.
#
# `null` rather than 0 for a check that did not run: 0 is a measurement, and a
# skipped check has none. Two of the entries below carry no "skipped" field, so
# a 0 there would be indistinguishable from a check that ran instantly.
DOCS_SECONDS=null
DOCS_VERSIONS_SECONDS=null
DOCS_MIRROR_SECONDS=null
VALIDATION_SECONDS=null
SHELL_LINT_SECONDS=null
WORKFLOW_LINT_SECONDS=null
CONTRACT_SECONDS=null
UNIT_TEST_SECONDS=null
INTEGRATION_TEST_SECONDS=null
WORKSPACE_GUARD_SECONDS=null

# Whether each test stage ran, decided once and used for BOTH the duration
# above and the "skipped" field in the summary. Deriving it in two places is
# how integration_tests came to report a duration of 0 for a stage that never
# ran: the heredoc knew the stage was skipped, while the timing span wrapped
# around a zero-iteration loop did not. VALIDATION_SKIPPED above is the same
# shape for the same reason.
UNIT_SKIPPED="false"
INTEGRATION_SKIPPED="false"
if [ "$SKIP_TESTS" = "yes" ]; then
    UNIT_SKIPPED="true"
    INTEGRATION_SKIPPED="true"
fi

# Seconds elapsed since a `date +%s` stamp.
elapsed_since() { echo $(( $(date +%s) - $1 )); }

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
    if [ "$VALIDATION_STATUS" -ne 0 ] || [ "$DOCS_STATUS" -ne 0 ] || [ "$DOCS_VERSIONS_STATUS" -ne 0 ] || [ "$DOCS_MIRROR_STATUS" -ne 0 ] || [ "$TEST_STATUS" -ne 0 ] || [ "$WORKFLOW_LINT_STATUS" -ne 0 ] || [ "$SHELL_LINT_STATUS" -ne 0 ] || [ "$CONTRACT_STATUS" -ne 0 ]; then
        echo "FAIL"
    elif [ "$VALIDATION_SKIPPED" = "true" ] && [ "$SKIP_TESTS" = "yes" ]; then
        echo "PASS_WITH_SKIPS"
    else
        echo "PASS"
    fi
}

# Reject a package name that names nothing, before any check runs on it.
#
# This block used to also build the glob the ruff diagnostics artifact is
# written over, and that was a second, independent answer to "what does this run
# check" — decided here, eighty lines before the one that decides what actually
# gets validated. They disagreed on the workspace-only branch: nothing had
# changed under packages/, so validate.sh was called with --workspace and
# checked no package source at all, while this fell through to packages/*/src
# and pointed ruff at all ten. style-check.json — the artifact a developer opens
# after a failure — then carried findings from code the run never validated and
# could not have failed on, presented beside a verdict they did not produce.
#
# The scope is derived from the validation decision now (see _ruff_scope at the
# invocation), so the two cannot disagree. What is left here is input validation,
# which genuinely belongs before anything runs.
if [ -n "$PACKAGES" ]; then
    _named_a_real_package=""
    for pkg in $PACKAGES; do
        if [ -d "packages/$pkg" ]; then
            _named_a_real_package="yes"
        else
            print_warning "Package not found: $pkg"
        fi
    done

    if [ -z "$_named_a_real_package" ]; then
        print_error "No valid packages specified"
        exit 1
    fi
fi

# Validate package references
print_status "Validating package references across codebase..."
if uv run python "$SCRIPT_DIR/validate-package-references.py" > "$OUTPUT_DIR/package-validation.log" 2>&1; then
    print_success "Package references are consistent"
else
    print_error "Package validation failed - see $OUTPUT_DIR/package-validation.log"
    cat "$OUTPUT_DIR/package-validation.log"
    exit 1
fi

# Lint GitHub Actions workflows. The status is captured rather than discarded:
# this check printed ✗ and let the gate go on to report PASS, which is worse
# than not running it, because it also reports success. Every check has to reach
# compute_overall_status (the exit code) and the checks object in
# quality-summary.json (the only thing CI looks at) — see both, below.
print_status "Linting GitHub Actions workflow files..."
_check_start=$(date +%s)
if "$SCRIPT_DIR/lint-workflows.sh"; then
    print_success "Workflow files are valid"
else
    WORKFLOW_LINT_STATUS=$?
    print_error "Workflow lint failed"
fi
WORKFLOW_LINT_SECONDS=$(elapsed_since "$_check_start")
# No "skipped" field: this check is not gated by anything, so there is no state
# in which it did not run. A field that can only ever hold one value is one more
# thing for a reader to check and one more thing for an author to get wrong.
record_check workflow_lint "$WORKFLOW_LINT_STATUS" \
    --tool lint-workflows.sh --duration "$WORKFLOW_LINT_SECONDS"

# Lint the repository's own shell scripts. Same three-place wiring as above, for
# the same reason: 46 shell files — including every script on this verdict path,
# this one among them — went through no linter at all, while every *.py beside
# them went through ruff and mypy.
print_status "Linting shell scripts..."
_check_start=$(date +%s)
if "$SCRIPT_DIR/lint-shell.sh"; then
    print_success "Shell scripts are clean"
else
    SHELL_LINT_STATUS=$?
    print_error "Shell lint failed"
fi
SHELL_LINT_SECONDS=$(elapsed_since "$_check_start")
# Unconditional for the same reason as the block above it, so no "skipped".
record_check shell_lint "$SHELL_LINT_STATUS" \
    --tool lint-shell.sh --duration "$SHELL_LINT_SECONDS"

# Compare the tree against the coverage-and-strictness contract. Same three-place
# wiring as the two above, and unconditional for the same reason.
#
# This is what makes a ceiling a ceiling. .dataknobs/quality-contract.json says
# which files each tool covers and how far from clean each part of the tree may
# be, and a number nobody compares is one that stops being true without anyone
# finding out — the declaration it replaces carried its counts in comment prose,
# where an entry matching nothing failed while "241 findings" stayed green at
# 400. Enforced in one direction is how a backlog grows during the phase that is
# supposed to be clearing it.
#
# It re-runs ruff and mypy rather than reading the verdicts validate.sh already
# produced, because they are different questions. validate.sh asks "is the code
# this run touched clean", over the packages in scope; this asks "has any cell
# of the whole tree moved past what it declared", which needs every cell
# measured whether the run touched it or not. Reusing one answer for the other
# would make the ratchet depend on which packages happened to change.
#
# For ruff the difference is scope alone — both read the root config. For mypy
# it is scope *and* config, since validate.sh runs under mypy.ini and the
# contract measures under the root one; that second half disappears when mypy.ini
# is retired, and the two invocations then differ only in what they cover.
print_status "Checking the quality contract..."
_check_start=$(date +%s)
if uv run python "$SCRIPT_DIR/quality-contract.py" check; then
    print_success "Every cell is within its ceiling"
else
    CONTRACT_STATUS=$?
    print_error "The quality contract is not satisfied"
fi
CONTRACT_SECONDS=$(elapsed_since "$_check_start")
record_check contract "$CONTRACT_STATUS" \
    --tool quality-contract.py --duration "$CONTRACT_SECONDS"

# Validation as a check that did not run. Two arms below reach it — asked to
# skip style, and a PR whose diff gave validate.sh nothing to look at — and both
# need the same record, so it is written once here.
#
# The zero is a literal, not an unset variable. A skipped check has always been
# recorded with exit_code 0 and status "pass", and the field it is read by is
# "skipped"; what this phase removes is not that convention but the *default*
# behind it, where the same values arrived because nothing had assigned them.
record_skipped_validation() {
    record_check validation 0 --skipped true --tool validate.sh --duration null
}

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
        # Spans both branches below and the ruff JSON generation, which is part
        # of what a validation phase costs rather than a separate step.
        _check_start=$(date +%s)
        if [ "$PR_MODE" = "yes" ]; then
            # What the diagnostics artifact is written over: the targets this run
            # is about to validate, asked for rather than derived a second time.
            # --print-targets runs no checks; it resolves the same argument string
            # passed to validate.sh below and prints the list, so the artifact
            # cannot report on code the run did not check, and cannot omit code it
            # did. A second construction here is what let the two drift.
            #
            # Captured into a variable first: errexit applies to a bare assignment
            # but not to a substitution inside an argument list, so a failure there
            # would silently pass ruff an empty target set — which reports no
            # findings and exits 0, the one result indistinguishable from clean.
            #
            # shellcheck disable=SC2086  # VALIDATE_ARGS is an argument list; see below
            _ruff_scope=$("$SCRIPT_DIR/validate.sh" --print-targets $VALIDATE_ARGS)
            if [ -z "$_ruff_scope" ]; then
                print_error "Could not resolve a scope for the style artifact"
                exit 1
            fi

            # Also generate ruff JSON artifact for diagnostics
            # The word splitting is the point. These variables hold argument *lists* —
            # VALIDATE_ARGS is "$PACKAGES --workspace", TEST_FLAGS and PYTEST_ARGS are
            # flag strings, _ruff_scope is several paths. Quoting one passes the whole
            # string as a single argument: validate.sh would look for a target literally
            # named "bots --workspace", find nothing, warn, and validate nothing while
            # still reporting success. That is this track's defect exactly, so the waiver
            # is here to stop a future editor "fixing" the finding into a silent failure.
            _ruff_json_rc=0
            # shellcheck disable=SC2086
            uv run ruff check $_ruff_scope --output-format=json --config "$PROJECT_ROOT/pyproject.toml" > "$OUTPUT_DIR/style-check.json" 2>&1 || _ruff_json_rc=$?

            # `|| true` used to cover both of the ways this goes wrong, and they
            # are not the same way.
            #
            # Exit 2+ is "ruff could not run" — a bad flag, an unreadable
            # config. The artifact then holds an error message where JSON
            # belongs, and every reader of it reports zero findings.
            #
            # Exit 1 is *both* "found violations" and "could not read a target":
            # an unmatched glob is passed through literally and comes back as a
            # single E902 io-error in valid JSON. Identical shape to a real
            # result, so the exit status cannot tell them apart and the content
            # has to. The artifact is then committed, and no consumer inspects
            # it — so a check that never ran ships as a check that found one
            # thing. tests/test_quality_artifact_contract.py guards the
            # committed copy; this guards writing one in the first place.
            if [ "$_ruff_json_rc" -gt 1 ]; then
                print_error "ruff could not produce the style artifact (exit $_ruff_json_rc)"
                cat "$OUTPUT_DIR/style-check.json"
                exit 1
            fi
            if grep -q '"code": *"E902"' "$OUTPUT_DIR/style-check.json" 2>/dev/null; then
                print_error "ruff could not read its targets — the style check did not run"
                grep -o '"filename": *"[^"]*"' "$OUTPUT_DIR/style-check.json" | head -5
                exit 1
            fi

            # Deliberate word splitting — see the waiver above the ruff call.
            # shellcheck disable=SC2086
            if "$SCRIPT_DIR/validate.sh" $VALIDATE_ARGS > "$OUTPUT_DIR/validation.log" 2>&1; then
                print_success "Code validation passed"
            else
                VALIDATION_STATUS=$?
                print_error "Code validation failed - see $OUTPUT_DIR/validation.log"
                cat "$OUTPUT_DIR/validation.log"
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
        VALIDATION_SECONDS=$(elapsed_since "$_check_start")
        record_check validation "$VALIDATION_STATUS" \
            --skipped false --tool validate.sh --duration "$VALIDATION_SECONDS"
    else
        VALIDATION_SKIPPED="true"
        print_status "Skipping code validation (no package changes)"
        record_skipped_validation
    fi
else
    VALIDATION_SKIPPED="true"
    print_status "Skipping code validation"
    record_skipped_validation
fi

# Documentation checks (PR mode only, skip if no docs changes in pr mode)
#
# One condition rather than nested ones, so there is one arm that runs the three
# checks and one that records them as not run. Three arms reached the second
# state before — a PR with no doc changes, and every dev run, which does not
# perform them at all — and each would have needed its own copy of the records.
if [ "$PR_MODE" = "yes" ] && { [ "$DOCS_CHANGED" = "true" ] || [ "$RUN_MODE" != "pr" ]; }; then
    print_status "Running documentation checks (build, versions, mirrors)..."
    # bin/docs-checks.sh is the single source of truth for the doc-check set.
    # It writes per-check logs (docs-build.log / docs-versions.log /
    # docs-mirror.log) plus docs-checks-status.json into ARTIFACTS_DIR.
    "$SCRIPT_DIR/docs-checks.sh" --artifacts "$OUTPUT_DIR" || true
    # Six fields, in a fixed order: three exit codes then three durations.
    # The durations are measured inside docs-checks.sh because this script
    # invokes it once for all three checks, so timing it here could only
    # ever have produced one number to spread across three entries.
    #
    # Still fails closed on any unreadable shape — a missing or malformed
    # status file reports three failures, not three passes — and an
    # unmeasured duration reports null rather than 0.
    DOCS_CHECK_CODES=$(python3 -c "
import json
try:
    d = json.load(open('$OUTPUT_DIR/docs-checks-status.json'))
    names = ('docs-build', 'docs-versions', 'docs-mirror')
    entries = [d.get(n) or {} for n in names]
    print(*[e.get('exit_code', 1) for e in entries],
          *[e.get('duration_seconds', 'null') for e in entries])
except Exception:
    print(1, 1, 1, 'null', 'null', 'null')
" 2>/dev/null || echo "1 1 1 null null null")
    read -r DOCS_STATUS DOCS_VERSIONS_STATUS DOCS_MIRROR_STATUS \
        DOCS_SECONDS DOCS_VERSIONS_SECONDS DOCS_MIRROR_SECONDS <<< "$DOCS_CHECK_CODES"

    if [ "$DOCS_STATUS" -eq 0 ] && [ "$DOCS_VERSIONS_STATUS" -eq 0 ] && [ "$DOCS_MIRROR_STATUS" -eq 0 ]; then
        print_success "Documentation checks passed (build, versions, mirrors)"
    else
        print_error "Documentation checks failed - see $OUTPUT_DIR/docs-*.log (details below)"
    fi

    record_check documentation "$DOCS_STATUS" \
        --skipped false --tool mkdocs --duration "$DOCS_SECONDS"
    record_check documentation_versions "$DOCS_VERSIONS_STATUS" \
        --skipped false --tool docs-update-versions.sh --duration "$DOCS_VERSIONS_SECONDS"
    record_check documentation_mirrors "$DOCS_MIRROR_STATUS" \
        --skipped false --tool docs-mirror-check.py --duration "$DOCS_MIRROR_SECONDS"
else
    # Only PR mode announces the skip, because only PR mode was ever going to
    # run them. A dev run reaches here having never offered.
    if [ "$PR_MODE" = "yes" ]; then
        print_status "Skipping docs checks (no documentation changes detected)"
    fi
    record_check documentation 0 \
        --skipped true --tool mkdocs --duration null
    record_check documentation_versions 0 \
        --skipped true --tool docs-update-versions.sh --duration null
    record_check documentation_mirrors 0 \
        --skipped true --tool docs-mirror-check.py --duration null
fi

# Run tests using the test.sh script
if [ "$SKIP_TESTS" != "yes" ]; then
    print_status "Running tests..."
    
    # Build test command (skip service management — already handled above)
    TEST_CMD="$SCRIPT_DIR/test.sh -n"

    # Workspace-level guards check the root config, every package's config, and
    # bin/ — so they belong to no package, and every test path below is keyed by
    # package. Both modes run them: a guard the quick loop skips is one a
    # developer only sees go red at PR time. Asked for by name (`test.sh
    # workspace`) rather than by directory, so this script holds no second
    # answer to where they live.

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
                $TEST_CMD "$pkg" -t unit --cov-report "$cov_report" $TEST_FLAGS -- $PYTEST_ARGS > "$OUTPUT_DIR/unit-test-output-$pkg.txt" 2>&1 || test_exit=$?
            else
                # Deliberate word splitting — see the waiver above the ruff call.
                # shellcheck disable=SC2086
                $TEST_CMD "$pkg" -t unit --cov-report "$cov_report" $TEST_FLAGS > "$OUTPUT_DIR/unit-test-output-$pkg.txt" 2>&1 || test_exit=$?
            fi

            # Save coverage artifacts
            if [ -f "coverage.xml" ]; then
                mv coverage.xml "$OUTPUT_DIR/coverage-unit-$pkg.xml"
            fi
            if [ -f "$PROJECT_ROOT/.coverage.unit.$pkg" ]; then
                mv "$PROJECT_ROOT/.coverage.unit.$pkg" "$OUTPUT_DIR/.coverage.unit.$pkg"
            elif [ -f ".coverage" ]; then
                mv .coverage "$OUTPUT_DIR/.coverage.unit.$pkg"
            fi

            # Unset to avoid leaking
            unset COVERAGE_FILE

            if [ $test_exit -ne 0 ] && [ $test_exit -ne 5 ]; then
                print_error "Unit tests failed for $pkg"
                # Show failed test names inline (strip ANSI codes first)
                if [ -f "$OUTPUT_DIR/unit-test-output-$pkg.txt" ]; then
                    sed 's/\x1b\[[0-9;]*m//g' "$OUTPUT_DIR/unit-test-output-$pkg.txt" 2>/dev/null | \
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
        _unit_start=$(date +%s)
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

        # Workspace guards (see the note by TEST_CMD above). Folded into the
        # unit status rather than given their own so a failure surfaces through
        # the existing summary. Exit 5 is "no tests collected", not a failure.
        {
            print_status "Running workspace tests..."
            _guards_start=$(date +%s)
            workspace_exit=0
            # Through test.sh, like every other suite the gate runs. This called
            # pytest directly — test.sh took a package name and scanned
            # packages/*, so it could not reach a directory belonging to no
            # package — which left one check the gate performed by a route no
            # developer command shares, and so one place the two could drift.
            # test.sh owns the invocation now (see run_workspace_tests there),
            # including the --durations=10 that names the slowest guards.
            #
            # No $TEST_FLAGS: --parallel would hand a guard suite that shells
            # out to the linters to xdist for no gain, and --quiet would drop
            # the durations line this captures the output for.
            # Deliberate word splitting — see the waiver above the ruff call.
            # shellcheck disable=SC2086
            $TEST_CMD workspace -- $PYTEST_ARGS \
                > "$OUTPUT_DIR/unit-test-output-workspace.txt" 2>&1 || workspace_exit=$?

            if [ $workspace_exit -ne 0 ] && [ $workspace_exit -ne 5 ]; then
                UNIT_TEST_STATUS=$workspace_exit
                print_error "Workspace tests failed"
                print_test_output_detail "$OUTPUT_DIR/unit-test-output-workspace.txt"
            else
                print_success "Workspace tests passed"
            fi
            WORKSPACE_GUARD_SECONDS=$(elapsed_since "$_guards_start")
        }
        UNIT_TEST_SECONDS=$(elapsed_since "$_unit_start")

        # Run integration tests (always sequential — shared external services).
        # An empty PACKAGES_TO_TEST makes the loop below iterate zero times, so
        # the stage did not run rather than running instantly. Recorded here so
        # the duration and the summary's "skipped" cannot disagree about it.
        if [ -z "$PACKAGES_TO_TEST" ]; then
            INTEGRATION_SKIPPED="true"
            print_status "Skipping integration tests (no package selected)"
        else
            print_status "Running integration tests..."
        fi
        _integration_start=$(date +%s)
        for pkg in $PACKAGES_TO_TEST; do
            if [ -d "packages/$pkg" ] && [ -d "packages/$pkg/tests/integration" ]; then
                print_status "Running integration tests for $pkg..."
                local_exit=0

                # Use per-package coverage file
                export COVERAGE_FILE="$PROJECT_ROOT/.coverage.integration.$pkg"

                if [ -n "$PYTEST_ARGS" ]; then
                    # Deliberate word splitting — see the waiver above the ruff call.
                    # shellcheck disable=SC2086
                    $TEST_CMD "$pkg" -t integration --cov-report "$TEST_COV_REPORT" $TEST_FLAGS -- $PYTEST_ARGS > "$OUTPUT_DIR/integration-test-output-$pkg.txt" 2>&1 || local_exit=$?
                else
                    # Deliberate word splitting — see the waiver above the ruff call.
                    # shellcheck disable=SC2086
                    $TEST_CMD "$pkg" -t integration --cov-report "$TEST_COV_REPORT" $TEST_FLAGS > "$OUTPUT_DIR/integration-test-output-$pkg.txt" 2>&1 || local_exit=$?
                fi

                unset COVERAGE_FILE

                if [ $local_exit -ne 0 ] && [ $local_exit -ne 5 ]; then
                    INTEGRATION_TEST_STATUS=$local_exit
                    print_error "Integration tests failed for $pkg"
                    # Show failed test names inline (strip ANSI codes first)
                    if [ -f "$OUTPUT_DIR/integration-test-output-$pkg.txt" ]; then
                        sed 's/\x1b\[[0-9;]*m//g' "$OUTPUT_DIR/integration-test-output-$pkg.txt" 2>/dev/null | \
                            grep -E '^FAILED ' | sed 's/^/    /' || true
                    fi
                else
                    print_success "Integration tests passed for $pkg"
                fi

                if [ -f "coverage.xml" ]; then
                    mv coverage.xml "$OUTPUT_DIR/coverage-integration-$pkg.xml"
                fi
                if [ -f "$PROJECT_ROOT/.coverage.integration.$pkg" ]; then
                    mv "$PROJECT_ROOT/.coverage.integration.$pkg" "$OUTPUT_DIR/.coverage.integration.$pkg"
                elif [ -f ".coverage" ]; then
                    mv .coverage "$OUTPUT_DIR/.coverage.integration.$pkg"
                fi
            fi
        done
        # Only a stage that ran has a duration. A zero-iteration loop measures
        # 0, and 0 is a measurement rather than the absence of one.
        if [ "$INTEGRATION_SKIPPED" != "true" ]; then
            INTEGRATION_TEST_SECONDS=$(elapsed_since "$_integration_start")
        fi

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
            for output_file in "$OUTPUT_DIR"/unit-test-output-*.txt "$OUTPUT_DIR"/integration-test-output-*.txt; do
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
        for output_file in "$OUTPUT_DIR"/unit-test-output-*.txt "$OUTPUT_DIR"/integration-test-output-*.txt; do
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
        for output_file in "$OUTPUT_DIR"/unit-test-output-*.txt "$OUTPUT_DIR"/integration-test-output-*.txt; do
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

            # Workspace guards (see the note by TEST_CMD above). Named here
            # because the loop above is keyed by package and cannot reach them,
            # by the same target the PR path uses — one invocation, not two
            # that have to agree.
            #
            # Only on this arm. The other one hands test.sh no target, and a
            # bare test.sh runs the guards alongside every discovered package,
            # so naming them again there would run them twice.
            print_status "Running workspace tests..."
            workspace_exit=0
            $TEST_CMD workspace --quiet || workspace_exit=$?

            if [ $workspace_exit -ne 0 ] && [ $workspace_exit -ne 5 ]; then
                TEST_STATUS=$workspace_exit
                print_error "Workspace tests failed"
            else
                print_success "Workspace tests passed"
            fi
        else
            # No package named, so test.sh discovers every package — and the
            # workspace guards with them.
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

        # The verdict the summary reports for both suites, recorded once here
        # rather than at each assignment above, so neither sub-branch can be the
        # one that forgets.
        #
        # Dev mode runs `test.sh -n` with no --type, so unit and integration go
        # through one invocation and come back as one exit code. That coarseness
        # is the truth available, and it is why the two are set from the same
        # value; the per-suite durations stay null, which is the honest record of
        # a measurement that was never taken separately.
        #
        # Without this they kept their initial 0 while TEST_STATUS carried the
        # real answer, and 0 renders as "pass". A failing dev run wrote
        # overall_status FAIL beside unit_tests pass — and the diagnostics tool
        # gates its entire test-failure section on exactly those two fields, so
        # it announced a failure and then named nothing that had failed.
        UNIT_TEST_STATUS=$TEST_STATUS
        INTEGRATION_TEST_STATUS=$TEST_STATUS
    fi

    record_check unit_tests "$UNIT_TEST_STATUS" \
        --skipped "$UNIT_SKIPPED" --duration "$UNIT_TEST_SECONDS" \
        --field "workspace_guards_seconds=$WORKSPACE_GUARD_SECONDS"
    record_check integration_tests "$INTEGRATION_TEST_STATUS" \
        --skipped "$INTEGRATION_SKIPPED" --duration "$INTEGRATION_TEST_SECONDS"

    # Create test results XML files for CI systems (an artifact, so gate only)
    if [ "$EMIT_ARTIFACTS" = "yes" ]; then
        if [ ! -f "$OUTPUT_DIR/unit-test-results.xml" ]; then
            echo '<?xml version="1.0" encoding="utf-8"?><testsuites></testsuites>' > "$OUTPUT_DIR/unit-test-results.xml"
        fi
        if [ ! -f "$OUTPUT_DIR/integration-test-results.xml" ]; then
            echo '<?xml version="1.0" encoding="utf-8"?><testsuites></testsuites>' > "$OUTPUT_DIR/integration-test-results.xml"
        fi
    fi
else
    print_status "Skipping tests"
    # Neither suite ran, so neither has a duration and the workspace guards have
    # none either. UNIT_SKIPPED and INTEGRATION_SKIPPED are already "true" on
    # this path — they are set beside SKIP_TESTS — and are read rather than
    # re-asserted so the two cannot come to disagree.
    record_check unit_tests 0 \
        --skipped "$UNIT_SKIPPED" --duration null \
        --field "workspace_guards_seconds=null"
    record_check integration_tests 0 \
        --skipped "$INTEGRATION_SKIPPED" --duration null
fi

# Coverage, the summary and — for the gate — the signature.
#
# Only the signature is gate-only, and it is the one thing here that *attests*
# rather than records: it is what makes a committed set verifiable, and it is
# meaningless over a directory git ignores wholesale. Everything else is a
# record of verdicts the run has already produced, and a record is exactly what
# the diagnostics tier is for. Writing quality-summary.json in both tiers is
# what lets bin/diagnose-quality-failures.sh read a check-only run at all — it
# opens that file first and exits when it is absent, so gating it here left the
# tier holding logs that the tool built to read them could not find.
#
# A summary under .quality-reports/ cannot be mistaken for evidence: it is
# git-ignored, outside the attested directory, and both readers that matter —
# validate-quality-artifacts.sh and package-hashes.py — resolve
# .quality-artifacts by literal path.
#
# Coverage runs before the summary because the summary reads a line-rate out of
# coverage.xml. The slower per-package reports run afterward.
COVERAGE_COMBINED=false
if ls "$OUTPUT_DIR"/.coverage.* >/dev/null 2>&1; then
    print_status "Combining coverage data..."
    cd "$OUTPUT_DIR"

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
elif ls "$OUTPUT_DIR"/coverage*.xml >/dev/null 2>&1; then
    print_status "Processing coverage XML files..."
    if [ ! -f "$OUTPUT_DIR/coverage.xml" ]; then
        if [ -f "$OUTPUT_DIR/coverage-unit.xml" ]; then
            cp "$OUTPUT_DIR/coverage-unit.xml" "$OUTPUT_DIR/coverage.xml"
        elif [ -f "$OUTPUT_DIR/coverage-integration.xml" ]; then
            cp "$OUTPUT_DIR/coverage-integration.xml" "$OUTPUT_DIR/coverage.xml"
        fi
    fi
else
    echo '<?xml version="1.0" encoding="utf-8"?><coverage version="1" line-rate="0"><packages></packages></coverage>' > "$OUTPUT_DIR/coverage.xml"
fi

# Read the line-rate out of coverage.xml now and record it in the summary.
# coverage.xml itself is no longer committed: it is a multi-megabyte
# generated file that changed on nearly every run, and the single thing the
# gate ever took from it was this number — which it reports as a warning and
# never fails on. A float belongs in the summary; the report belongs on disk.
COVERAGE_PERCENT=$(python3 -c "
import xml.etree.ElementTree as ET
try:
root = ET.parse('$OUTPUT_DIR/coverage.xml').getroot()
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
# The hashes were taken before the first check. Re-take them now: equal means
# every check read the content this artifact is about to attest, and unequal
# means the tree moved while the run was reading it, so no digest describes
# what was actually checked.
#
# Refusing is the only honest answer. Recording the start digest would attest a
# tree half the checks never saw; recording the end digest would attest one the
# other half never saw; and writing a summary with overall_status=FAIL would
# say the *checks* failed, which is a different and untrue claim. So the run
# stops here without an artifact, and .run-in-progress stays behind to say so.
#
# One case is open and stays open: an A→B→A edit inside a single run leaves both
# digests equal while some check may have read B. Only a timestamp catches that,
# and mtime is settable, rewritten by checkouts and rebases, and touched by
# editors on saves that change nothing — the rarest case at the highest
# flakiness cost, which is how a check ends up suppressed.
print_status "Re-checking content hashes..."
MOVED_SCOPES=$(uv run python "$SCRIPT_DIR/package-hashes.py" changed-since \
    --packages "$PACKAGE_HASHES_JSON" \
    --workspace "$WORKSPACE_HASHES_JSON") && HASH_RECHECK=0 || HASH_RECHECK=$?

# Three outcomes, kept apart on purpose. "The tree moved" sends the developer
# to re-run; "the comparison could not be made" sends them to the gate itself,
# and telling them to re-run would be advice about the wrong program.
if [ "$HASH_RECHECK" -eq 1 ]; then
    print_error "The tree changed while the checks were running:"
    # Unquoted on purpose: one name per line from the comparison, re-split here
    # into one argument per name so printf's format cycles over them.
    # shellcheck disable=SC2086
    printf '  - %s\n' $MOVED_SCOPES >&2
    print_error "No digest describes what was checked, so no artifact was written."
    print_error "Re-run once the tree is settled."
    exit 1
elif [ "$HASH_RECHECK" -ne 0 ]; then
    print_error "Could not re-check the content hashes (exit $HASH_RECHECK)."
    print_error "The artifact would attest a tree nothing verified, so none was written."
    exit 1
fi

print_status "Generating quality summary..."
OVERALL_STATUS=$(compute_overall_status)

# The checks half of the document comes from the records each check wrote as it
# ran; this call supplies the run's own metadata around them. Splitting the two
# is the point: a check's outcome is stated once, where it is produced, and
# there is no per-check stanza here to keep in step with anything.
#
# Every top-level field below is required by the writer, which refuses a name it
# does not know and refuses to run without one it does. A field dropped from
# this call fails the run rather than disappearing from the artifact.
#
# --str for the ones that are strings, --json for the ones that are not: the
# hashes are objects, tested_packages an array, coverage_percent a float or
# null. A quoted number reaching CI as a string is the kind of thing a shell
# serializer got wrong by omission.
if ! python3 "$SCRIPT_DIR/quality-summary.py" build \
    --records "$RECORDS_FILE" \
    --output "$OUTPUT_DIR/quality-summary.json" \
    --str "timestamp=$TIMESTAMP" \
    --str "overall_status=$OVERALL_STATUS" \
    --str "run_mode=$RUN_MODE" \
    --str "environment=$([ "$IN_DOCKER" = true ] && echo "docker" || echo "host")" \
    --str "packages=$([ -n "$PACKAGES" ] && echo "$PACKAGES" || echo "all")" \
    --json "tested_packages=$TESTED_PACKAGES_JSON" \
    --json "coverage_percent=$COVERAGE_PERCENT" \
    --json "package_hashes=$PACKAGE_HASHES_JSON" \
    --json "workspace_hashes=$WORKSPACE_HASHES_JSON" \
    --json "total_seconds=$(elapsed_since "$RUN_START")"; then
    # No summary means no verdict for CI to read and nothing for the signature
    # to attest, so there is nothing useful left to do. The .run-in-progress
    # marker below is deliberately left in place: this is exactly the state it
    # exists to announce.
    print_error "Could not write the quality summary — see the error above"
    exit 1
fi

# The run is now readable: a verdict for every check sits beside the logs that
# produced it. Whatever fails after this point, the summary is this run's, so
# the marker has nothing left to warn anyone about.
rm -f "$OUTPUT_DIR/.run-in-progress"

# The signature, and only the signature, is gate-only. It attests rather than
# records: it is what makes a committed set verifiable, and it enumerates its
# input with `git ls-files --exclude-standard`, so over a directory git ignores
# wholesale it finds nothing and writes a checksum of an empty list — which
# reads as a valid signature file. There is no useful thing for it to do in the
# diagnostics tier, and one harmful one.
if [ "$EMIT_ARTIFACTS" = "yes" ]; then
    print_status "Generating artifact signature..."
    cd "$OUTPUT_DIR"
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
else
    print_success "Quality summary generated"
fi

# Generate supplementary per-package coverage reports (not needed by CI).
# This section is slow due to multiple `uv run coverage report` calls but
# runs after the summary/signature are already written.
if [ "$COVERAGE_COMBINED" = true ]; then
    print_status "Generating per-package coverage reports..."
    cd "$OUTPUT_DIR"

    # Generate combined HTML report (full mode only — slow)
    if [ "$RUN_MODE" = "full" ]; then
        if uv run coverage html -d htmlcov 2>/dev/null; then
            print_success "Combined coverage HTML generated in .quality-artifacts/htmlcov/"
        fi
    fi

    # Generate terminal report.
    #
    # Truncated first. Every write below appends, and nothing ever reset the
    # file, so each run added a full report to the end of every previous
    # one: 95.7 MB by the time anyone looked, for a file whose whole purpose
    # is to show the current run's coverage. Untracked, so it never reached
    # CI — it just grew.
    : > test-coverage-summary.txt
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
                coverage_output=$(uv run coverage report --data-file="$OUTPUT_DIR/.coverage" --include="*/${src_name}/*" 2>/dev/null || true)
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

# Print summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                        Quality Check Summary                     ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# The check lines, rendered from the document rather than from the variables
# that produced it. They used to be a second derivation of the same statuses,
# and the two disagreed: on any pull request that changed no documentation the
# banner printed three ✓ PASSED rows beside a summary recording skipped: true
# for all three — the artifact half of that defect was fixed one phase earlier,
# and this half was not, because it reads its own variables.
#
# Both arguments are presentation and neither can change what a row says. --mode
# decides whether the documentation rows appear and whether the two test suites
# are shown apart; --package-tests-skipped labels the unit row as the workspace
# guards it is reduced to when no package changed. Every verdict comes from the
# file.
_render_args=""
if [ "$SKIP_PACKAGE_TESTS" = "yes" ]; then
    _render_args="--package-tests-skipped"
fi
# shellcheck disable=SC2086  # _render_args is an argument list, empty or one flag
python3 "$SCRIPT_DIR/quality-summary.py" render \
    --summary "$OUTPUT_DIR/quality-summary.json" \
    --mode "$([ "$PR_MODE" = "yes" ] && echo pr || echo dev)" $_render_args
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Determine overall status (same helper as the quality-summary.json computation,
# so the exit code below can never disagree with the reported summary).
OVERALL_STATUS=$(compute_overall_status)

if [ "$OVERALL_STATUS" = "PASS" ] || [ "$OVERALL_STATUS" = "PASS_WITH_SKIPS" ]; then
    echo ""
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    if [ "$EMIT_ARTIFACTS" = "yes" ]; then
        echo -e "${GREEN}  Artifacts saved to: .quality-artifacts/${NC}"
        echo -e "${GREEN}  You can now create your pull request.${NC}"
    else
        echo -e "${GREEN}  Checks only — no artifacts written. Reports in ${CYAN}${OUTPUT_DIR#"$PROJECT_ROOT"/}/${GREEN}.${NC}"
        echo -e "${GREEN}  Run ${CYAN}bin/dk pr${GREEN} to produce the artifacts CI verifies.${NC}"
    fi
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some checks failed!${NC}"
    echo -e "${RED}  Please fix the issues and run this script again.${NC}"

    # The diagnostics below name log files under OUTPUT_DIR. They survive the
    # run either way — the artifacts directory because the gate produced it, the
    # reports directory because it is cleared on entry rather than at exit — so
    # the paths printed here still resolve when the prompt comes back.

    # Show quick diagnostic info for failures
    echo ""
    echo -e "${YELLOW}Quick Diagnostics:${NC}"
    
    if [ "$PR_MODE" = "yes" ]; then
        # In PR mode, show specific commands to investigate failures
        if [ "$DOCS_STATUS" -ne 0 ] && [ -f "$OUTPUT_DIR/docs-build.log" ]; then
            echo -e "  ${CYAN}Documentation Build Failures:${NC}"
            echo "    View documentation errors:"
            echo "      cat $OUTPUT_DIR/docs-build.log"
            echo ""
        fi

        if [ "$DOCS_VERSIONS_STATUS" -ne 0 ] && [ -f "$OUTPUT_DIR/docs-versions.log" ]; then
            echo -e "  ${CYAN}Documentation Version Mismatch:${NC}"
            echo "    View version differences:"
            echo "      cat $OUTPUT_DIR/docs-versions.log"
            echo "    To fix:"
            echo "      bin/docs-update-versions.sh"
            echo ""
        fi

        if [ "$DOCS_MIRROR_STATUS" -ne 0 ] && [ -f "$OUTPUT_DIR/docs-mirror.log" ]; then
            echo -e "  ${CYAN}Documentation Mirror Drift:${NC}"
            echo "    View mirror differences:"
            echo "      cat $OUTPUT_DIR/docs-mirror.log"
            echo "    To fix:"
            echo "      bin/docs-mirror-check.py --fix   # or reclassify in .dataknobs/docs-mirror-manifest.json"
            echo ""
        fi

        if [ $UNIT_TEST_STATUS -ne 0 ] || [ $INTEGRATION_TEST_STATUS -ne 0 ]; then
            echo -e "  ${CYAN}Test Failures:${NC}"

            for output_file in "$OUTPUT_DIR"/*-test-output-*.txt; do
                if [ -f "$output_file" ]; then
                    if grep -q "FAILED" "$output_file" 2>/dev/null; then
                        pkg_name=$(basename "$output_file" | sed 's/.*-test-output-\(.*\)\.txt/\1/')
                        echo "    Package $pkg_name has failures:"
                        echo "      grep -E '(FAILED|ERROR)' $output_file"
                    fi
                fi
            done
        fi
        
        if [ $VALIDATION_STATUS -ne 0 ] && [ -f "$OUTPUT_DIR/validation.log" ]; then
            echo -e "  ${CYAN}Code Validation Failures:${NC}"
            echo "    View full validation output:"
            echo "      cat $OUTPUT_DIR/validation.log"
            echo "    To auto-fix what's possible:"
            echo "      bin/validate.sh -f"
        fi
        
        echo ""
        if [ "$EMIT_ARTIFACTS" = "yes" ]; then
            echo -e "  ${CYAN}Full artifacts in:${NC} .quality-artifacts/"
            echo -e "  ${CYAN}View summary:${NC} cat .quality-artifacts/quality-summary.json | python -m json.tool"
        else
            echo -e "  ${CYAN}Full output kept in:${NC} $OUTPUT_DIR"
        fi
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

